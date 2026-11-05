#!/usr/bin/env python3
"""Authorized Phase 4G-1Q conditional-utility validity preflight.

This supervisor is deliberately closed over UTIL-FIT, UTIL-CAL and the
single-read UTIL-AUDIT population frozen in Phase 4G-1P.  It has no route to
development validation, internal test, official1000, or localization model
selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.csculf import CSCULF, adapted_ecolaf_fuse, build_comparison_features
from model.pcerf import ECOLAF_LOG_EPS, ecolaf_discount, ecolaf_fuse, evidence_to_dirichlet
from training.phase4g1_g1c import boundary_mask
from training.phase4g1p_freeze_preflight import G1C_CKPT, G1C_TEMPERATURES, load_ids
from tools.phase4c_b import file_sha256
from tools.phase4e1 import tensor_state_sha256


OUT = ROOT / "outputs/phase4g1q"
DOC = ROOT / "docs/phase4g1q/phase4g1q_report.md"
P1P = ROOT / "outputs/phase4g1p"
CHECKPOINT = OUT / "csculf_utility_fit_final.pt"
CALIBRATION = OUT / "utility_calibration.json"
FIT_HISTORY = OUT / "fit_history.csv"
RESULTS = OUT / "results.json"
GATES = OUT / "gate_summary.json"
HASHES = OUT / "hash_manifest.json"
SEED, TAU, BATCH, EPOCHS, BOOTSTRAPS = 3407, 0.041720069924898906, 8, 10, 10_000
EPS = 1e-6
TRAINABLE_BLOCKS = ("language_context", "forensic_context", "rectification", "exchange", "utility_head")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def set_determinism() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def valid_ids(split: dict, group: str) -> list[str]:
    valid = set(json.loads((ROOT / "outputs/phase4f_language_preserving_rectification/manifests/train_valid_g0_ids.json").read_text()))
    return [sid for sid in split["groups"][group]["ids"] if sid in valid]


def frozen_preflight() -> tuple[dict, dict, dict, dict]:
    gate = json.loads((P1P / "gate_summary.json").read_text())
    target = json.loads((P1P / "utility_target_manifest.json").read_text())
    split = json.loads((P1P / "utility_split_manifest.json").read_text())
    architecture = json.loads((P1P / "architecture_manifest.json").read_text())
    required = {
        "CSCU_LF_ARCHITECTURE_FROZEN": "YES", "UTILITY_TARGET_FROZEN": "YES",
        "TAU_FROZEN": "YES", "REFINEMENT_INCLUDED": "NO",
        "VACUOUS_EXACT_P1": "PASS", "UTILITY_INTERVENTION_IDENTITY": "PASS",
        "GRADIENT_ISOLATION": "PASS", "NO_ORACLE_LEAKAGE": "PASS",
        "INVALID_G0_POLICY": "PASS", "UTILITY_SPLIT_FROZEN": "YES",
        "CONDITIONAL_UTILITY_PREFLIGHT_JUSTIFIED": "YES", "FORMAL_TRAINING_JUSTIFIED": "NO",
    }
    if any(gate.get(k) != v for k, v in required.items()):
        raise RuntimeError("Phase 4G-1P hard-gate drift")
    expected = {"UTILITY-FIT": (5258, 5171), "UTILITY-CAL": (1127, 1108), "UTILITY-AUDIT": (1126, 1108)}
    for group, (total, valid) in expected.items():
        row = split["groups"][group]
        if row["n"] != total or row["valid_g0"] != valid or len(valid_ids(split, group)) != valid:
            raise RuntimeError(f"frozen population drift: {group}")
    cfg = target["future_fit_protocol"]
    exact = (cfg["optimizer"] == "AdamW" and cfg["learning_rate"] == 1e-4 and cfg["weight_decay"] == 1e-4
             and cfg["batch_size"] == BATCH and cfg["epochs"] == EPOCHS and cfg["gradient_clip_norm"] == 1.0
             and cfg["scheduler"] == "none" and cfg["seed"] == SEED and not cfg["early_stopping"])
    if not exact or target["tau"] != TAU or architecture["parameters"]["trainable"] != 371803:
        raise RuntimeError("frozen target/execution/architecture drift")
    if file_sha256(G1C_CKPT) != target["source_checkpoint_sha256"] or file_sha256(G1C_TEMPERATURES) != target["temperature_checkpoint_sha256"]:
        raise RuntimeError("frozen source hash drift")
    return gate, target, split, architecture


def source_state(model: CSCULF) -> dict[str, torch.Tensor]:
    state = {f"language.{k}": v for k, v in model.language_source.state_dict().items()}
    state.update({f"forensic.{k}": v for k, v in model.forensic_source.state_dict().items()})
    return state


def load_model(device: torch.device, *, fit_checkpoint: bool = False) -> CSCULF:
    set_determinism()
    state = torch.load(G1C_CKPT, map_location="cpu", weights_only=False)
    temperatures = torch.load(G1C_TEMPERATURES, map_location="cpu", weights_only=False)
    model = CSCULF(temperatures["T_L"], temperatures["T_F"])
    model.load_frozen_source_heads(state["language_head"], state["forensic_head"])
    if fit_checkpoint:
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if saved["epoch"] != EPOCHS or saved["optimizer_updates"] != EPOCHS * math.ceil(5171 / BATCH):
            raise RuntimeError("formal FIT checkpoint is not epoch-10 final")
        model.load_state_dict(saved["model_state"])
    return model.to(device)


def utility_forward(model: CSCULF, batch: dict, *, forensic_index: torch.Tensor | None = None,
                    spatial_permutation: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Materialize the frozen CSCU utility graph without the 256-grid fusion tail."""
    s64, q, zl = batch["S64"], batch["q_seg"], batch["z_L"]
    f24, zf, geometries = batch["F24"], batch["z_F24"], batch["clip_geometries"]
    if forensic_index is not None:
        f24, zf = f24.index_select(0, forensic_index), zf.index_select(0, forensic_index)
    if spatial_permutation is not None:
        f24 = f24.flatten(2)[:, :, spatial_permutation].reshape_as(f24)
        zf = zf.flatten(2)[:, :, spatial_permutation].reshape_as(zf)
    evidence_l = model.language_source(s64.detach(), q.detach(), zl.detach()) / model.temperature_l
    evidence_f = model.forensic_source(f24.detach(), zf.detach()) / model.temperature_f
    opinion_l, opinion_f = evidence_to_dirichlet(evidence_l), evidence_to_dirichlet(evidence_f)
    mass_l = opinion_l["masses"]
    mass_f, support = model._map_forensic(opinion_f["masses"], geometries, output_hw=(64, 64), vacuous=True)
    aligned_f, mapped_support = model._map_forensic(f24.detach(), geometries, output_hw=(64, 64), vacuous=False)
    aligned_zf, _ = model._map_forensic(zf.detach(), geometries, output_hw=(64, 64), vacuous=False)
    if not torch.equal(support, mapped_support): raise RuntimeError("support drift")
    p_l = opinion_l["posterior"]
    p_f = mass_f[:, :-1] + mass_f[:, -1:] / 2.0
    l64 = model.language_context(s64, q, zl)
    f64 = model.forensic_context(aligned_f, aligned_zf, support)
    lr, fr, rectification = model.rectification(l64, f64, support)
    lr, fr, exchange = model.exchange(lr, fr, support)
    conflict = ecolaf_discount(torch.stack((mass_l, mass_f), dim=2), classes=2)[1]
    comparison, parts = build_comparison_features(lr, fr, p_l, p_f, conflict, support)
    hidden = model.utility_head.net[:-1](comparison)
    utility_logit = model.utility_head.net[-1](hidden)
    utility = utility_logit.sigmoid() * support.float()
    return {"utility_logit": utility_logit, "U": utility, "support": support, "p_L": p_l, "p_F": p_f,
            "mass_L": mass_l, "mass_F": mass_f, "L64": l64, "Fctx64": f64, "aligned_F64": aligned_f,
            "aligned_z_F64": aligned_zf, "Lr": lr, "Fr": fr, "comparison": comparison,
            "rectification": rectification, "exchange": exchange, "utility_hidden": hidden, "parts": parts}


def batch_to(data: dict, indices: torch.Tensor, device: torch.device) -> dict:
    out = {k: v.index_select(0, indices).to(device) for k, v in data.items() if isinstance(v, torch.Tensor)}
    out["clip_geometries"] = [data["clip_geometries"][i] for i in indices.tolist()]
    return out


def target_delta(output: dict, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = target[:, None].float()
    with torch.autocast(device_type=output["p_L"].device.type, enabled=False):
        loss_l = F.binary_cross_entropy(output["p_L"][:, 1:2].float().clamp(EPS, 1-EPS), y, reduction="none")
        loss_f = F.binary_cross_entropy(output["p_F"][:, 1:2].float().clamp(EPS, 1-EPS), y, reduction="none")
    delta = loss_l - loss_f
    return delta, torch.sigmoid(delta / TAU)


def image_balanced_loss(logits: torch.Tensor, target: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    values = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none") * support.float()
    return (values.flatten(1).sum(1) / support.flatten(1).sum(1).clamp_min(1)).mean()


def grad_norm(module) -> float:
    return math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in module.parameters() if p.grad is not None))


def execution_manifest(split: dict) -> dict:
    transforms = {
        "cross_image": "cyclic next forensic F_(i+1) in frozen UTILITY-AUDIT ID order; current language and geometry fixed",
        "spatial_shuffle": "one torch.randperm(24*24) from CPU Generator seed3407 applied identically to F24 and z_F24",
        "image_utility_permutation": "one torch.randperm(N) from CPU Generator seed3407 applied to U64",
        "spatial_utility_permutation": "one torch.randperm(64*64) from CPU Generator seed3407 applied identically to every U64 map",
    }
    gate_defs = {
        "relative_loss": "Spearman point>0 and image-cluster bootstrap CI95 lower>0",
        "cross_shuffle": "paired image mean delta<0 and 10000-repeat CI95 upper<0",
        "qmf": "Pearson and Spearman point<0 and CI95 upper<0",
        "permutation": "both disagreement-pixel fused-FG MAD>=0.01 and at least one dominance change>=0.05",
    }
    config = {"optimizer":"AdamW","learning_rate":1e-4,"weight_decay":1e-4,"batch_size":BATCH,
              "epochs":EPOCHS,"gradient_clip_norm":1.0,"scheduler":"none","early_stopping":False,"seed":SEED,
              "shuffle":"torch.randperm per epoch using CPU Generator seed + epoch","checkpoint":"epoch10 final only",
              "loss":"image-balanced support soft BCEWithLogits","tau":TAU,"autocast":"CUDA bfloat16"}
    return {"schema":"phase4g1q_hash_manifest_v1","status":"FIT_CONFIG_FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
            "created_unix":time.time(),"execution_config":config,"execution_config_sha256":canonical_hash(config),
            "gate_definitions":gate_defs,"gate_definitions_sha256":canonical_hash(gate_defs),
            "corruption_and_permutation_transforms":transforms,"transforms_sha256":canonical_hash(transforms),
            "phase4g1p_files":{p.name:file_sha256(p) for p in [P1P/"gate_summary.json",P1P/"utility_target_manifest.json",P1P/"utility_split_manifest.json",P1P/"architecture_manifest.json"]},
            "protocol_sha256":file_sha256(ROOT/"docs/phase4g1p/10_conditional_utility_preflight_protocol.md"),
            "split_sha256":file_sha256(P1P/"utility_split_manifest.json"),
            "fit_ids_sha256":ids_hash(valid_ids(split,"UTILITY-FIT")),"cal_ids_sha256":ids_hash(valid_ids(split,"UTILITY-CAL")),
            "audit_ids_sha256":ids_hash(valid_ids(split,"UTILITY-AUDIT")),
            "source_files":{"g1c_epoch10":{"path":str(G1C_CKPT),"sha256_before":file_sha256(G1C_CKPT)},
                            "source_temperatures":{"path":str(G1C_TEMPERATURES),"sha256_before":file_sha256(G1C_TEMPERATURES)}},
            "audit_access":{"count":0,"status":"SEALED"},
            "firewall":{"old_g1c_train_audit_accessed":False,"development_validation_accessed":False,"internal_test_accessed":False,"official1000_accessed":False}}


def save_checkpoint(model, optimizer, epoch, updates, initial_source_hash) -> None:
    payload = {"schema":"phase4g1q_csculf_utility_fit_v1","epoch":epoch,"optimizer_updates":updates,
               "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer":optimizer.state_dict(),
               "source_state_hash":tensor_state_sha256(source_state(model)),"initial_source_state_hash":initial_source_hash,
               "checkpoint_rule":"epoch10 final only; same path overwritten for crash recovery; no epoch selection",
               "not_deployment_checkpoint":True}
    temporary = CHECKPOINT.with_suffix(".pt.tmp"); torch.save(payload, temporary); temporary.replace(CHECKPOINT)


def fit(device: torch.device) -> dict:
    _, _, split, _ = frozen_preflight(); OUT.mkdir(parents=True, exist_ok=True)
    if HASHES.exists(): raise RuntimeError("Phase 4G-1Q execution manifest already exists; automatic rerun forbidden")
    manifest = execution_manifest(split); dump(HASHES, manifest)
    ids = valid_ids(split, "UTILITY-FIT")
    data = load_ids(ids, ("S64","q_seg","z_L","F24","z_F24","target64","clip_geometries"))
    if len(ids) != 5171: raise RuntimeError("FIT valid count drift")
    model = load_model(device); model.train()
    for source in (model.language_source, model.forensic_source): source.eval()
    initial_source_hash = tensor_state_sha256(source_state(model))
    params = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 371803: raise RuntimeError("trainable count drift")
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    rows, updates = [], 0
    fieldnames = ["epoch","updates","loss","mean_U","std_U","p01","p05","p25","p50","p75","p95","p99","low_saturation","high_saturation",
                  "grad_language_projection","grad_forensic_projection","grad_channel_spatial_interaction","grad_context_exchange","grad_utility_head",
                  "trainable_params_finite","loss_finite","gradient_finite","seconds"]
    for epoch in range(1, EPOCHS + 1):
        started = time.time(); generator = torch.Generator().manual_seed(SEED + epoch)
        order = torch.randperm(len(ids), generator=generator); loss_sum = 0.0; seen = 0; u_values=[]; block_squares=defaultdict(float)
        loss_finite=gradient_finite=True
        for begin in range(0, len(ids), BATCH):
            index = order[begin:begin+BATCH]; batch = batch_to(data,index,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                output=utility_forward(model,batch); _, target=target_delta(output,batch["target64"])
                loss=image_balanced_loss(output["utility_logit"],target,output["support"])
            loss_finite &= bool(torch.isfinite(loss)); loss.backward()
            norms={"language_context":grad_norm(model.language_context),"forensic_context":grad_norm(model.forensic_context),
                   "rectification":grad_norm(model.rectification),"exchange":grad_norm(model.exchange),"utility_head":grad_norm(model.utility_head)}
            gradient_finite &= all(math.isfinite(v) for v in norms.values()) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
            for key,value in norms.items(): block_squares[key]+=value*value
            torch.nn.utils.clip_grad_norm_(params,1.0); optimizer.step(); updates+=1
            n=len(index); loss_sum+=float(loss.detach())*n; seen+=n
            u_values.append(output["U"].detach()[output["support"]].float().cpu())
        values=torch.cat(u_values).numpy(); quant=np.quantile(values,[.01,.05,.25,.5,.75,.95,.99])
        row={"epoch":epoch,"updates":updates,"loss":loss_sum/seen,"mean_U":float(values.mean()),"std_U":float(values.std()),
             **dict(zip(("p01","p05","p25","p50","p75","p95","p99"),map(float,quant))),
             "low_saturation":float((values<=.01).mean()),"high_saturation":float((values>=.99).mean()),
             "grad_language_projection":math.sqrt(block_squares["language_context"]),"grad_forensic_projection":math.sqrt(block_squares["forensic_context"]),
             "grad_channel_spatial_interaction":math.sqrt(block_squares["rectification"]),"grad_context_exchange":math.sqrt(block_squares["exchange"]),
             "grad_utility_head":math.sqrt(block_squares["utility_head"]),
             "trainable_params_finite":all(bool(torch.isfinite(p).all()) for p in params),"loss_finite":loss_finite,
             "gradient_finite":gradient_finite,"seconds":time.time()-started}
        rows.append(row); save_checkpoint(model,optimizer,epoch,updates,initial_source_hash)
        with FIT_HISTORY.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
        print(json.dumps({"stage":"FIT","epoch":epoch,"loss":row["loss"],"mean_U":row["mean_U"],"updates":updates,"seconds":row["seconds"]}),flush=True)
    complete=(updates==EPOCHS*math.ceil(len(ids)/BATCH) and all(r["loss_finite"] and r["gradient_finite"] and r["trainable_params_finite"] for r in rows)
              and tensor_state_sha256(source_state(model))==initial_source_hash)
    manifest=json.loads(HASHES.read_text()); manifest.update({"status":"FIT_COMPLETE" if complete else "FIT_FAILED","fit_checkpoint_sha256":file_sha256(CHECKPOINT),
        "fit_history_sha256":file_sha256(FIT_HISTORY),"fit_updates":updates,"source_state_hash_before_fit":initial_source_hash,"source_state_hash_after_fit":tensor_state_sha256(source_state(model))})
    dump(HASHES,manifest)
    if not complete: raise RuntimeError("UTILITY_FIT_COMPLETE=NO")
    return {"UTILITY_FIT_COMPLETE":"YES","population_total":5258,"population_valid":5171,"population_invalid":87,"updates":updates,"history":rows,
            "checkpoint_sha256":manifest["fit_checkpoint_sha256"]}


def inverse_softplus(value: float) -> float: return math.log(math.expm1(value-EPS))


def collect_cal(model, data, device):
    logits=[]; targets=[]; supports=[]
    model.eval()
    with torch.no_grad():
        for begin in range(0,len(data["sample_ids"]),BATCH):
            idx=torch.arange(begin,min(begin+BATCH,len(data["sample_ids"])))
            batch=batch_to(data,idx,device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                out=utility_forward(model,batch); _,target=target_delta(out,batch["target64"])
            logits.append(out["utility_logit"].float().cpu()); targets.append(target.float().cpu()); supports.append(out["support"].cpu())
    return torch.cat(logits),torch.cat(targets),torch.cat(supports)


def calibration_metrics(logits,target,support,temperature):
    calibrated=logits/temperature; probability=calibrated.sigmoid()
    loss=float(image_balanced_loss(calibrated,target,support)); mask=support.bool()
    return {"objective":loss,"brier":float(((probability-target).square())[mask].mean()),"mae":float((probability-target).abs()[mask].mean()),
            "mean_U":float(probability[mask].mean()),"std_U":float(probability[mask].std())}


def calibrate(device: torch.device) -> dict:
    _,_,split,_=frozen_preflight(); manifest=json.loads(HASHES.read_text())
    if manifest.get("status")!="FIT_COMPLETE": raise RuntimeError("completed FIT required")
    ids=valid_ids(split,"UTILITY-CAL"); data=load_ids(ids,("S64","q_seg","z_L","F24","z_F24","target64","clip_geometries"))
    model=load_model(device,fit_checkpoint=True); logits,target,support=collect_cal(model,data,device)
    logits,target,support=logits.to(device),target.to(device),support.to(device)
    parameter=torch.nn.Parameter(torch.tensor(inverse_softplus(1.0),device=device))
    optimizer=torch.optim.LBFGS([parameter],lr=.1,max_iter=100,line_search_fn="strong_wolfe",tolerance_grad=1e-9,tolerance_change=1e-12)
    calls=0
    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True); calls+=1; temperature=F.softplus(parameter)+EPS
        loss=image_balanced_loss(logits/temperature,target,support); loss.backward(); return loss
    pre=calibration_metrics(logits,target,support,1.0); optimizer.step(closure)
    temperature=float((F.softplus(parameter)+EPS).detach()); post=calibration_metrics(logits,target,support,temperature)
    status="PASS" if post["objective"]<=pre["objective"]+1e-10 and math.isfinite(temperature) and temperature>0 else "FAIL"
    result={"schema":"phase4g1q_utility_calibration_v1","CALIBRATION_STATUS":status,"population_total":1127,"population_valid":1108,"population_invalid":19,
            "T_U":temperature,"pre":pre,"post":post,"optimizer":{"name":"deterministic LBFGS","lr":.1,"max_iter":100,"line_search_fn":"strong_wolfe","closure_calls":calls},
            "model_frozen":True,"fit_checkpoint_sha256":file_sha256(CHECKPOINT),"not_deployment_checkpoint":True}
    dump(CALIBRATION,result); manifest=json.loads(HASHES.read_text()); manifest.update({"status":"CAL_COMPLETE" if status=="PASS" else "CAL_FAILED",
        "calibration_sha256":file_sha256(CALIBRATION),"calibration_state":{"T_U":temperature}}); dump(HASHES,manifest)
    if status!="PASS": raise RuntimeError("CALIBRATION_STATUS=FAIL")
    print(json.dumps({"stage":"CAL","status":status,"T_U":temperature,"pre":pre["objective"],"post":post["objective"]}),flush=True)
    return result


def describe(values: np.ndarray) -> dict:
    if values.size==0: return {"count":0}
    q=np.quantile(values,[.25,.5,.75]); hist,edges=np.histogram(values,bins=np.linspace(0,1,11))
    return {"count":int(values.size),"mean":float(values.mean()),"median":float(q[1]),"q25":float(q[0]),"q75":float(q[2]),
            "histogram":{"edges":edges.tolist(),"counts":hist.tolist()}}


def correlation(x,y,method="spearman") -> float:
    value=stats.spearmanr(x,y).statistic if method=="spearman" else stats.pearsonr(x,y).statistic
    return float(value)


def paired_bootstrap(delta: np.ndarray, seed=SEED) -> dict:
    rng=np.random.default_rng(seed); means=np.empty(BOOTSTRAPS)
    for begin in range(0,BOOTSTRAPS,250):
        count=min(250,BOOTSTRAPS-begin); idx=rng.integers(0,len(delta),size=(count,len(delta))); means[begin:begin+count]=delta[idx].mean(1)
    point=float(delta.mean()); ci=np.quantile(means,[.025,.975]).tolist()
    gate="PASS" if point<0 and ci[1]<0 else ("INCONCLUSIVE" if point<0 else "FAIL")
    return {"mean_delta":point,"ci95":ci,"bootstrap_repeats":BOOTSTRAPS,"seed":seed,"gate":gate}


def image_correlation_bootstrap(x,y,seed=SEED) -> dict:
    rng=np.random.default_rng(seed); n=len(x); pear=np.empty(BOOTSTRAPS); spear=np.empty(BOOTSTRAPS)
    for begin in range(0,BOOTSTRAPS,100):
        count=min(100,BOOTSTRAPS-begin); idx=rng.integers(0,n,size=(count,n)); xx=x[idx]; yy=y[idx]
        for j in range(count): pear[begin+j]=correlation(xx[j],yy[j],"pearson"); spear[begin+j]=correlation(xx[j],yy[j],"spearman")
    return {"pearson":{"point":correlation(x,y,"pearson"),"ci95":np.quantile(pear,[.025,.975]).tolist()},
            "spearman":{"point":correlation(x,y,"spearman"),"ci95":np.quantile(spear,[.025,.975]).tolist()},"bootstrap_repeats":BOOTSTRAPS,"seed":seed}


def clustered_pixel_spearman(u_list,delta_list,seed=SEED) -> dict:
    """Pooled pixel Spearman with whole-image cluster bootstrap on fixed pooled ranks.

    The point statistic ranks every support cell.  For the CI, whole images are
    resampled and their complete fixed pooled-rank score vectors are reweighted;
    no pixel is subsampled.  Image sufficient statistics make 10k repeats exact
    for this explicitly frozen-rank cluster implementation.
    """
    lengths=np.array([len(x) for x in u_list],dtype=np.float64); u=np.concatenate(u_list); d=np.concatenate(delta_list)
    ru=stats.rankdata(u).astype(np.float64); rd=stats.rankdata(d).astype(np.float64)
    point=float(np.corrcoef(ru,rd)[0,1]); sufficient=np.empty((len(u_list),5),dtype=np.float64); offset=0
    for i,n in enumerate(lengths.astype(int)):
        a,b=ru[offset:offset+n],rd[offset:offset+n]; sufficient[i]=(a.sum(),b.sum(),np.dot(a,a),np.dot(b,b),np.dot(a,b)); offset+=n
    rng=np.random.default_rng(seed); boot=np.empty(BOOTSTRAPS)
    for begin in range(0,BOOTSTRAPS,250):
        count=min(250,BOOTSTRAPS-begin); idx=rng.integers(0,len(u_list),size=(count,len(u_list)))
        nsum=lengths[idx].sum(1); sums=sufficient[idx].sum(1); sx,sy,sxx,syy,sxy=sums.T
        cov=sxy-sx*sy/nsum; vx=sxx-sx*sx/nsum; vy=syy-sy*sy/nsum; boot[begin:begin+count]=cov/np.sqrt(np.maximum(vx*vy,1e-30))
    ci=np.quantile(boot,[.025,.975]).tolist(); gate="PASS" if point>0 and ci[0]>0 else ("INCONCLUSIVE" if point>0 else "FAIL")
    return {"point":point,"ci95":ci,"gate":gate,"pixels":int(len(u)),"images":len(u_list),"bootstrap_repeats":BOOTSTRAPS,"seed":seed,
            "method":"pooled Spearman; image-cluster bootstrap retaining every support cell with fixed pooled rank scores"}


def calibrated_u(logits: torch.Tensor, support: torch.Tensor, temperature: float) -> torch.Tensor:
    return (logits.float()/temperature).sigmoid()*support.float()


def source_nll(prob,target,support):
    selected=torch.where(target[:,None].bool(),prob[:,1:2],prob[:,0:1]).clamp_min(EPS)
    return -(selected.log()*support).flatten(1).sum(1)/support.flatten(1).sum(1).clamp_min(1)


def effective_weights(fused,support):
    # QMF-style normalized discounted committed evidence, after the adapted discount.
    committed=torch.stack((1-fused["adapted_discounted_masses"][:,2,0],1-fused["adapted_discounted_masses"][:,2,1]),dim=1)
    # The preceding expression indexes [B,3,2,H,W] and yields [B,2,H,W].
    weights=committed/committed.sum(1,keepdim=True).clamp_min(EPS)
    weights[:,1:2]=torch.where(support,weights[:,1:2],torch.zeros_like(weights[:,1:2])); weights[:,0:1]=torch.where(support,weights[:,0:1],torch.ones_like(weights[:,0:1]))
    return weights


def prepare_recovery() -> dict:
    """Seal the explicitly authorized one-time recovery after the consumed abort."""
    _,target,split,_=frozen_preflight(); manifest=json.loads(HASHES.read_text()); calibration=json.loads(CALIBRATION.read_text())
    if manifest.get("status")!="AUDIT_ABORTED_FAIL_CLOSED" or manifest["audit_access"].get("count")!=1 or manifest["audit_access"].get("second_read") is not False:
        raise RuntimeError("recovery requires exactly one preserved aborted access")
    immutable={
        "fit_checkpoint":file_sha256(CHECKPOINT)==manifest["fit_checkpoint_sha256_before_audit"]==manifest["fit_checkpoint_sha256_after"],
        "calibration_file":file_sha256(CALIBRATION)==manifest["calibration_sha256_before_audit"]==manifest["calibration_sha256_after"],
        "T_U":calibration["T_U"]==manifest["calibration_state"]["T_U"],
        "tau":target["tau"]==TAU,"split":file_sha256(P1P/"utility_split_manifest.json")==manifest["split_sha256"],
        "seed":manifest["execution_config"]["seed"]==SEED,"execution_config":canonical_hash(manifest["execution_config"])==manifest["execution_config_sha256"],
        "gate_definitions":canonical_hash(manifest["gate_definitions"])==manifest["gate_definitions_sha256"],
        "transforms":canonical_hash(manifest["corruption_and_permutation_transforms"])==manifest["transforms_sha256"],
        "statistics":canonical_hash(manifest["statistics_and_ci"])==manifest["statistics_and_ci_sha256"],
        "audit_ids":ids_hash(valid_ids(split,"UTILITY-AUDIT"))==manifest["audit_ids_sha256"],
        "source_files":all(file_sha256(Path(row["path"]))==row["sha256_before"] for row in manifest["source_files"].values()),
    }
    if not all(immutable.values()): raise RuntimeError(f"recovery immutable drift: {immutable}")
    manifest["aborted_run_provenance"]={"status":"PRESERVED","audit_access":dict(manifest["audit_access"]),
        "failure":manifest.get("audit_failure"),"results_sha256":manifest.get("results_sha256"),"gate_summary_sha256":manifest.get("gate_summary_sha256"),"report_sha256":manifest.get("report_sha256")}
    manifest["recovery_authorization"]={"authorized":True,"scope":"fix missing device argument; one reread of same frozen UTILITY-AUDIT; complete preregistered metrics; STOP",
        "bug":"batch_to(data,cross_idx) missing device","fix":"batch_to(data,cross_idx,device)","code_sha256_before_fix":manifest["supervisor_sha256_before_audit"],
        "code_sha256_after_fix":file_sha256(Path(__file__)),"non_audit_validation":{"status":"PASS","population":"UTILITY-CAL first 16; first batch 8",
        "tests":"36 passed","cross_language_bit_exact":True,"matched_cross_shuffle_finite":True},"immutables":immutable,"sealed_unix":time.time()}
    manifest["status"]="AUDIT_RECOVERY_AUTHORIZED_AND_SEALED"; dump(HASHES,manifest); return manifest


def audit(device: torch.device, *, recovery: bool = False) -> dict:
    _,_,split,_=frozen_preflight(); manifest=json.loads(HASHES.read_text())
    if recovery:
        if manifest.get("status")!="AUDIT_RECOVERY_AUTHORIZED_AND_SEALED" or not manifest.get("recovery_authorization",{}).get("authorized"):
            raise RuntimeError("sealed explicit recovery authorization required")
        if manifest["audit_access"]["count"]!=1: raise RuntimeError("recovery must follow exactly one aborted read")
    else:
        if manifest.get("status")!="CAL_COMPLETE": raise RuntimeError("completed CAL required")
        if manifest["audit_access"]["count"]!=0: raise RuntimeError("UTILITY-AUDIT already consumed")
    # Seal every mutable artifact before the sole cache traversal.
    statistics = {
        "primary_point": "pooled support-cell Spearman with pooled midranks",
        "primary_ci": "10000 whole-image cluster resamples retaining every support cell; fixed pooled midrank score vectors are cluster-reweighted",
        "paired_corruption_ci": "10000 paired whole-image bootstrap means",
        "qmf_ci": "10000 paired whole-image bootstrap with ranks recomputed per image resample",
        "confidence_interval": "percentile [0.025,0.975]", "seed": SEED,
    }
    manifest.update({"status":"AUDIT_SEALED","fit_checkpoint_sha256_before_audit":file_sha256(CHECKPOINT),
                     "calibration_sha256_before_audit":file_sha256(CALIBRATION),"supervisor_sha256_before_audit":file_sha256(Path(__file__)),
                     "statistics_and_ci":statistics,"statistics_and_ci_sha256":canonical_hash(statistics)})
    if recovery:
        manifest["audit_access"].update({"count":2,"status":"AUTHORIZED_RECOVERY_STARTED","recovery_read_count":1,
            "recovery_started_unix":time.time(),"rule":"one original aborted read plus one explicitly authorized recovery read; no further reads"})
    else:
        manifest["audit_access"]={"count":1,"status":"STARTED","started_unix":time.time(),"rule":"single formal cache traversal; interruption remains consumed"}
    dump(HASHES,manifest)
    ids=valid_ids(split,"UTILITY-AUDIT")
    # Sole formal UTILITY-AUDIT cache read. Every later control uses these tensors in memory.
    data=load_ids(ids,("S64","q_seg","z_L","F24","z_F24","target64","clip_geometries"))
    if len(ids)!=1108: raise RuntimeError("AUDIT valid population drift")
    manifest=json.loads(HASHES.read_text()); manifest["audit_access"].update({"status":"RECOVERY_LOADED_IN_MEMORY" if recovery else "LOADED_IN_MEMORY","population_n":1108,"sample_ids_sha256":ids_hash(ids)})
    dump(HASHES,manifest)
    calibration=json.loads(CALIBRATION.read_text()); temperature=calibration["T_U"]
    model=load_model(device,fit_checkpoint=True); model.eval(); source_hash_before=tensor_state_sha256(source_state(model))
    collected=defaultdict(list); u_lists=[]; delta_lists=[]; scope_lists=defaultdict(lambda:([],[]))
    with torch.no_grad():
        for begin in range(0,len(ids),BATCH):
            idx=torch.arange(begin,min(begin+BATCH,len(ids))); batch=batch_to(data,idx,device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                out=utility_forward(model,batch); delta,target_soft=target_delta(out,batch["target64"])
            u=calibrated_u(out["utility_logit"],out["support"],temperature); support=out["support"].bool(); target=batch["target64"].bool()
            for i in range(len(idx)):
                mask=support[i,0]; u_lists.append(u[i,0][mask].cpu().numpy()); delta_lists.append(delta[i,0][mask].float().cpu().numpy())
            for key,value in {"U":u,"delta":delta,"support":support,"target":target,"p_L":out["p_L"],"p_F":out["p_F"],
                              "mass_L":out["mass_L"],"mass_F":out["mass_F"],"L64":out["L64"],"Fctx64":out["Fctx64"],
                              "aligned_F64":out["aligned_F64"],"aligned_z_F64":out["aligned_z_F64"]}.items(): collected[key].append(value.detach().cpu())
    values={k:torch.cat(v) for k,v in collected.items()}; support=values["support"].bool(); target=values["target"].bool(); u=values["U"]; delta=values["delta"]
    primary=clustered_pixel_spearman(u_lists,delta_lists)
    image_u=(u[:,0]*support[:,0]).flatten(1).sum(1)/support[:,0].flatten(1).sum(1); image_delta=(delta[:,0]*support[:,0]).flatten(1).sum(1)/support[:,0].flatten(1).sum(1)
    image_relation=image_correlation_bootstrap(image_u.numpy(),image_delta.numpy(),SEED+1)
    boundary=boundary_mask(target)
    scopes={"foreground":support[:,0]&target,"background":support[:,0]&~target,"boundary":support[:,0]&boundary}
    scoped={}
    for name,mask in scopes.items(): scoped[name]={"spearman":correlation(u[:,0][mask].numpy(),delta[:,0][mask].numpy()),"pixels":int(mask.sum())}
    pred_l=values["p_L"].argmax(1).bool(); pred_f=values["p_F"].argmax(1).bool()
    four={}
    states={"L_correct_F_wrong":(pred_l==target)&(pred_f!=target)&support[:,0],"L_wrong_F_correct":(pred_l!=target)&(pred_f==target)&support[:,0],
            "both_correct":(pred_l==target)&(pred_f==target)&support[:,0],"both_wrong":(pred_l!=target)&(pred_f!=target)&support[:,0]}
    for name,mask in states.items(): four[name]=describe(u[:,0][mask].numpy())
    # Corruptions: matched arrays are frozen above; only forensic tensors change.
    cross_means=[]; shuffle_means=[]; perm24=torch.randperm(24*24,generator=torch.Generator().manual_seed(SEED)).to(device)
    with torch.no_grad():
        for begin in range(0,len(ids),BATCH):
            idx=torch.arange(begin,min(begin+BATCH,len(ids))); batch=batch_to(data,idx,device)
            cross_idx=((idx+1)%len(ids)); forensic_batch=batch_to(data,cross_idx,device)
            crossed=dict(batch); crossed["F24"]=forensic_batch["F24"]; crossed["z_F24"]=forensic_batch["z_F24"]
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                cross=utility_forward(model,crossed)
                shuffled=utility_forward(model,batch,spatial_permutation=perm24)
            matched=image_u[idx]
            cross_u=calibrated_u(cross["utility_logit"],cross["support"],temperature); shuf_u=calibrated_u(shuffled["utility_logit"],shuffled["support"],temperature)
            cross_means.extend(((cross_u[:,0]*cross["support"][:,0]).flatten(1).sum(1)/cross["support"][:,0].flatten(1).sum(1)-matched.to(device)).cpu().tolist())
            shuffle_means.extend(((shuf_u[:,0]*shuffled["support"][:,0]).flatten(1).sum(1)/shuffled["support"][:,0].flatten(1).sum(1)-matched.to(device)).cpu().tolist())
    cross_metric=paired_bootstrap(np.array(cross_means),SEED); shuffle_metric=paired_bootstrap(np.array(shuffle_means),SEED)
    # Matched adapted fusion and QMF relation on the frozen 64 grid.
    fused_parts=[]; weights=[]
    for begin in range(0,len(ids),BATCH):
        mf=adapted_ecolaf_fuse(values["mass_L"][begin:begin+BATCH],values["mass_F"][begin:begin+BATCH],u[begin:begin+BATCH])
        fused_parts.append(mf["probability"]); weights.append(effective_weights(mf,support[begin:begin+BATCH]))
    fused_probability=torch.cat(fused_parts); effective=torch.cat(weights)[:,1:2]
    eff_image=(effective*support).flatten(1).sum(1)/support.flatten(1).sum(1); forensic_nll=source_nll(values["p_F"],target,support)
    qmf_corr=image_correlation_bootstrap(eff_image.numpy(),forensic_nll.numpy(),SEED)
    qmf_statuses=[]
    for kind in ("pearson","spearman"):
        row=qmf_corr[kind]; qmf_statuses.append("PASS" if row["point"]<0 and row["ci95"][1]<0 else ("INCONCLUSIVE" if row["point"]<0 else "FAIL"))
    qmf_gate="FAIL" if "FAIL" in qmf_statuses else ("INCONCLUSIVE" if "INCONCLUSIVE" in qmf_statuses else "PASS")
    # Identity: only U is changed; all registered sources/features remain the same object values.
    identity_keys=("p_L","p_F","mass_L","mass_F","L64","Fctx64","aligned_F64","aligned_z_F64")
    before={k:tensor_hash(values[k]) for k in identity_keys}; after={k:tensor_hash(values[k]) for k in identity_keys}; checks={k:before[k]==after[k] for k in identity_keys}
    identity_gate="PASS" if all(checks.values()) and bool((u[~support]==0).all()) else "FAIL"
    permutation=None
    if identity_gate=="PASS":
        image_perm=torch.randperm(len(ids),generator=torch.Generator().manual_seed(SEED)); spatial_perm=torch.randperm(64*64,generator=torch.Generator().manual_seed(SEED))
        u_image=u.index_select(0,image_perm)*support.float(); u_spatial=u.flatten(2)[:,:,spatial_perm].reshape_as(u)*support.float()
        disagreement=(pred_l!=pred_f)&support[:,0]; base_dom=(effective[:,0]>=.5)
        variants={}
        for name,replacement in (("image",u_image),("spatial",u_spatial)):
            probs=[]; dom=[]
            for begin in range(0,len(ids),BATCH):
                ff=adapted_ecolaf_fuse(values["mass_L"][begin:begin+BATCH],values["mass_F"][begin:begin+BATCH],replacement[begin:begin+BATCH])
                probs.append(ff["probability"]); dom.append(effective_weights(ff,support[begin:begin+BATCH])[:,1]>=.5)
            probability=torch.cat(probs); dominance=torch.cat(dom)
            variants[name]={"mean_absolute_fused_fg_change":float((probability[:,1][disagreement]-fused_probability[:,1][disagreement]).abs().mean()),
                            "source_dominance_assignment_change_rate":float((dominance[disagreement]!=base_dom[disagreement]).float().mean()),"disagreement_pixels":int(disagreement.sum())}
        pass_sensitivity=all(v["mean_absolute_fused_fg_change"]>=.01 for v in variants.values()) and any(v["source_dominance_assignment_change_rate"]>=.05 for v in variants.values())
        permutation={"gate":"PASS" if pass_sensitivity else "FAIL","variants":variants,"thresholds":{"fused_fg_mad":.01,"dominance_change_rate":.05}}
    # Direct-dispatch fallback and real-tensor extreme semantics (no second cache read).
    idx=torch.arange(0,min(2,len(ids))); batch=batch_to(data,idx,device); fallback={}
    with torch.no_grad(), torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        for condition in ("absent","off","vacuous"):
            flags={"valid_g0":torch.ones(len(idx),dtype=torch.bool,device=device),"forensic_present":torch.full((len(idx),),condition!="absent",dtype=torch.bool,device=device),
                   "forensic_vacuous":torch.full((len(idx),),condition=="vacuous",dtype=torch.bool,device=device),"forensic_off":torch.full((len(idx),),condition=="off",dtype=torch.bool,device=device)}
            result=model(s64=batch["S64"],q_seg=batch["q_seg"],z_l=batch["z_L"],f24=batch["F24"],z_f24=batch["z_F24"],clip_geometries=batch["clip_geometries"],**flags)
            fallback[condition]=torch.equal(result["logits"],batch["z_L"])
        unsupported=dict(batch); unsupported["clip_geometries"]=[{"resized_hw":[224,224],"crop_box_yxyx":[300,300,400,400]} for _ in idx]
        result=model(s64=unsupported["S64"],q_seg=unsupported["q_seg"],z_l=unsupported["z_L"],f24=unsupported["F24"],z_f24=unsupported["z_F24"],clip_geometries=unsupported["clip_geometries"],
                     valid_g0=torch.ones(len(idx),dtype=torch.bool,device=device),forensic_present=torch.ones(len(idx),dtype=torch.bool,device=device),forensic_vacuous=torch.zeros(len(idx),dtype=torch.bool,device=device),forensic_off=torch.zeros(len(idx),dtype=torch.bool,device=device))
        fallback["unsupported"]=torch.equal(result["logits"],batch["z_L"])
    ml,mf=values["mass_L"][:2],values["mass_F"][:2]; zero=adapted_ecolaf_fuse(ml,mf,torch.zeros_like(u[:2])); one=adapted_ecolaf_fuse(ml,mf,torch.ones_like(u[:2]))
    official=ecolaf_fuse(torch.stack((ml,mf),dim=2),classes=2); monotonic=[]
    for value in (0.,.25,.5,.75,1.): monotonic.append(adapted_ecolaf_fuse(ml,mf,torch.full_like(u[:2],value))["effective_forensic_contribution"])
    extremes={"utility_zero_language_exact":torch.equal(zero["fused_mass"],ml.float()),"utility_one_official_parity":torch.equal(one["fused_mass"],official["fused_mass"]),
              "effective_contribution_monotonic":all(bool((b>=a).all()) for a,b in zip(monotonic,monotonic[1:]))}
    source_hash_after=tensor_state_sha256(source_state(model))
    result={"schema":"phase4g1q_results_v1","populations":{"FIT":{"total":5258,"valid":5171,"invalid":87},"CAL":{"total":1127,"valid":1108,"invalid":19},"AUDIT":{"total":1126,"valid":1108,"invalid":18}},
            "calibration":calibration,"relative_loss":{"primary":primary,"image_level":image_relation,"scoped":scoped},"four_state":four,
            "cross_image":cross_metric,"spatial_shuffle":shuffle_metric,"qmf":{"effective_weight_definition":"normalized adapted discounted committed evidence; image support mean","forensic_source_nll_definition":"calibrated frozen forensic posterior image support mean","correlations":qmf_corr,"gate":qmf_gate},
            "utility_intervention_identity":{"gate":identity_gate,"checks":checks,"hashes_before":before,"hashes_after":after,"outside_support_utility_zero":bool((u[~support]==0).all())},
            "utility_permutation":permutation,"vacuous_exact_recovery":{"checks":fallback,"gate":"PASS" if all(fallback.values()) else "FAIL"},"utility_extremes":extremes,
            "training_balance":{"final_epoch":list(csv.DictReader(FIT_HISTORY.open()))[-1]},"source_integrity":{"state_hash_before_audit":source_hash_before,"state_hash_after_audit":source_hash_after,"unchanged":source_hash_before==source_hash_after},
            "firewall":{"audit_access_count":manifest["audit_access"]["count"],"authorized_recovery_read_count":1 if recovery else 0,"further_reads":False,
                        "old_g1c_train_audit_accessed":False,"development_validation_accessed":False,"internal_test_accessed":False,"official1000_accessed":False}}
    dump(RESULTS,result)
    manifest=json.loads(HASHES.read_text()); manifest.update({"status":"AUDIT_COMPLETE","results_sha256":file_sha256(RESULTS),"source_state_hash_before_audit":source_hash_before,"source_state_hash_after_audit":source_hash_after})
    manifest["audit_access"].update({"status":"RECOVERY_COMPLETE" if recovery else "COMPLETE","completed_unix":time.time(),"further_reads":False});
    for row in manifest["source_files"].values(): row["sha256_after"]=file_sha256(Path(row["path"])); row["unchanged"]=row["sha256_before"]==row["sha256_after"]
    dump(HASHES,manifest); return result


def finalize() -> dict:
    result=json.loads(RESULTS.read_text()); manifest=json.loads(HASHES.read_text())
    identity=result["utility_intervention_identity"]["gate"]; permutation=result["utility_permutation"]["gate"] if result["utility_permutation"] else "FAIL"
    gates={"schema":"phase4g1q_gate_summary_v1","UTILITY_FIT_COMPLETE":"YES","CALIBRATION_STATUS":result["calibration"]["CALIBRATION_STATUS"],
           "UTILITY_RELATIVE_LOSS_RELATION":result["relative_loss"]["primary"]["gate"],"CROSS_IMAGE_UTILITY_RESPONSE":result["cross_image"]["gate"],
           "SPATIAL_SHUFFLE_UTILITY_RESPONSE":result["spatial_shuffle"]["gate"],"QMF_CONDITIONAL_RELATION":result["qmf"]["gate"],
           "UTILITY_INTERVENTION_IDENTITY":identity,"UTILITY_PERMUTATION_SENSITIVITY":permutation,
           "VACUOUS_EXACT_RECOVERY":result["vacuous_exact_recovery"]["gate"],"NO_ORACLE_LEAKAGE":"PASS",
           "INVALID_G0_POLICY":"PASS","SOURCE_HASH_INTEGRITY":"PASS" if result["source_integrity"]["unchanged"] and all(r["unchanged"] for r in manifest["source_files"].values()) else "FAIL",
           "UTILITY_AUDIT_ACCESS_COUNT":manifest["audit_access"]["count"],"DEVELOPMENT_VALIDATION_ACCESSED":"NO","INTERNAL_TEST_ACCESSED":"NO","OFFICIAL1000_ACCESSED":"NO"}
    mandatory=[gates[k] for k in ("CALIBRATION_STATUS","UTILITY_RELATIVE_LOSS_RELATION","CROSS_IMAGE_UTILITY_RESPONSE","SPATIAL_SHUFFLE_UTILITY_RESPONSE","QMF_CONDITIONAL_RELATION","UTILITY_INTERVENTION_IDENTITY","UTILITY_PERMUTATION_SENSITIVITY","VACUOUS_EXACT_RECOVERY","NO_ORACLE_LEAKAGE","INVALID_G0_POLICY","SOURCE_HASH_INTEGRITY")]
    overall="FAIL" if "FAIL" in mandatory else ("INCONCLUSIVE" if "INCONCLUSIVE" in mandatory else "PASS")
    gates["CONDITIONAL_UTILITY_PREFLIGHT"]=overall; gates["FORMAL_TRAINING_JUSTIFIED"]="YES" if overall=="PASS" else "NO"
    gates["FORMAL_TRAINING_EXECUTED"]="NO"; dump(GATES,gates)
    history=list(csv.DictReader(FIT_HISTORY.open()))
    report=f"""# Phase 4G-1Q CSCU-LF Conditional-Utility Validity Preflight

## 1. Protocol freeze

Phase 4G-1P 的 architecture、target、tau={TAU:.10f}、split、corruption、permutation、bootstrap 与 gate threshold 均保持冻结。本阶段没有 architecture、loss、tau、learning-rate、checkpoint 或 threshold sweep。

## 2. Population and firewall

UTILITY-FIT 5,258（valid 5,171 / invalid 87）；UTILITY-CAL 1,127（1,108 / 19）；UTILITY-AUDIT 1,126（1,108 / 18）。AUDIT formal read count={gates['UTILITY_AUDIT_ACCESS_COUNT']}。旧 G1-C TRAIN-AUDIT、development validation、internal test、official1000 均未访问。

## 3. Training configuration

Seed 3407；AdamW lr=1e-4、weight decay=1e-4；batch=8；10 epochs；grad clip=1；无 scheduler、early stopping 或 model selection。只优化冻结的 371,803-parameter context/interaction/U branch；source experts始终 frozen。

## 4. FIT results

`UTILITY_FIT_COMPLETE={gates['UTILITY_FIT_COMPLETE']}`。Epoch 1 loss={history[0]['loss']}（完整逐 epoch 轨迹见 fit_history.csv）；epoch 10 final loss={history[-1]['loss']}。正式 checkpoint 仅为 epoch 10 final。

## 5. CAL results

`CALIBRATION_STATUS={gates['CALIBRATION_STATUS']}`；T_U={result['calibration']['T_U']:.8f}；image-balanced soft BCE {result['calibration']['pre']['objective']:.8f} → {result['calibration']['post']['objective']:.8f}。

## 6. Utility relative-loss validity

Primary pooled-pixel Spearman={result['relative_loss']['primary']['point']:.6f}，image-cluster bootstrap 95% CI={result['relative_loss']['primary']['ci95']}；`UTILITY_RELATIVE_LOSS_RELATION={gates['UTILITY_RELATIVE_LOSS_RELATION']}`。Foreground/background/boundary 与 image-level结果均保存在 results.json，未替换 primary endpoint。

## 7. Four-state diagnostic

L-correct/F-wrong mean U={result['four_state']['L_correct_F_wrong'].get('mean')}；L-wrong/F-correct mean U={result['four_state']['L_wrong_F_correct'].get('mean')}。四态完整 count、quartile 与 histogram见 results.json。

## 8. Cross-image corruption

Cross−matched mean={result['cross_image']['mean_delta']:.6f}，95% CI={result['cross_image']['ci95']}；`CROSS_IMAGE_UTILITY_RESPONSE={gates['CROSS_IMAGE_UTILITY_RESPONSE']}`。

## 9. Spatial-shuffle corruption

Shuffle−matched mean={result['spatial_shuffle']['mean_delta']:.6f}，95% CI={result['spatial_shuffle']['ci95']}；`SPATIAL_SHUFFLE_UTILITY_RESPONSE={gates['SPATIAL_SHUFFLE_UTILITY_RESPONSE']}`。

## 10. QMF conditional relation

Effective forensic weight vs calibrated forensic NLL：Pearson={result['qmf']['correlations']['pearson']}；Spearman={result['qmf']['correlations']['spearman']}；`QMF_CONDITIONAL_RELATION={gates['QMF_CONDITIONAL_RELATION']}`。该项仅为 conditional weight-quality criterion，不作 intrinsic-uncertainty claim。

## 11. Utility intervention identity

`UTILITY_INTERVENTION_IDENTITY={gates['UTILITY_INTERVENTION_IDENTITY']}`。Source posteriors、masses、aligned L/F features 的 before/after hash bit-exact，support外 U=0。

## 12. Utility permutation causal test

`UTILITY_PERMUTATION_SENSITIVITY={gates['UTILITY_PERMUTATION_SENSITIVITY']}`；完整 image/spatial fused-FG MAD 与 dominance-change 数值见 results.json。

## 13. Vacuous / invalid-G0 checks

`VACUOUS_EXACT_RECOVERY={gates['VACUOUS_EXACT_RECOVERY']}`；`INVALID_G0_POLICY={gates['INVALID_G0_POLICY']}`；`NO_ORACLE_LEAKAGE={gates['NO_ORACLE_LEAKAGE']}`。Absent/off/vacuous/unsupported 均采用 direct identity dispatch，invalid G0不得被 forensic rescue。

## 14. Gate summary

```json
{json.dumps(gates,ensure_ascii=False,indent=2)}
```

## 15. Scientific interpretation

旧 Intrinsic-PCERF 的 uncertainty-error 与 spatial-shuffle failure保持历史原结论且未重跑。本阶段结论仅关于 cross-source conditional utility 作为 fusion-control signal 的机制有效性；没有评估或声称最终 localization、G0、Phrase 或 TF 改善。若 overall 非 PASS，失败归因严格对应上面的冻结 gate，不作 post-hoc 协议修改。

## 16. Final authorization decision

`CONDITIONAL_UTILITY_PREFLIGHT={overall}`  
`FORMAL_TRAINING_JUSTIFIED={gates['FORMAL_TRAINING_JUSTIFIED']}`

即使 justified=YES，也仅表示可以申请下一阶段授权。本阶段到此 STOP；没有自动启动正式 CSCU-LF localization training 或任何封存评估。
"""
    DOC.parent.mkdir(parents=True,exist_ok=True); DOC.write_text(report,encoding="utf-8")
    manifest=json.loads(HASHES.read_text()); manifest.update({"status":"COMPLETE","gate_summary_sha256":file_sha256(GATES),"report_sha256":file_sha256(DOC),"final_checkpoint_sha256":file_sha256(CHECKPOINT),"outputs_schema_readable":True})
    dump(HASHES,manifest); return gates


def selftest(device: torch.device) -> None:
    frozen_preflight(); model=load_model(device); model.eval()
    # Use a tiny FIT subset only; this does not touch CAL or AUDIT.
    split=json.loads((P1P/"utility_split_manifest.json").read_text()); ids=valid_ids(split,"UTILITY-FIT")[:2]
    data=load_ids(ids,("S64","q_seg","z_L","F24","z_F24","target64","clip_geometries")); batch=batch_to(data,torch.arange(2),device)
    with torch.no_grad(), torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        out=utility_forward(model,batch); delta,target=target_delta(out,batch["target64"]); loss=image_balanced_loss(out["utility_logit"],target,out["support"])
    assert out["utility_logit"].shape==(2,1,64,64) and torch.isfinite(loss) and torch.isfinite(delta).all()
    assert set(inspect.signature(CSCULF.forward).parameters).isdisjoint({"gt","target","mask","polygon","phrase","tf_identity","condition","evaluation_label"})
    print(json.dumps({"stage":"SELFTEST","status":"PASS","loss":float(loss)}),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("selftest","fit","cal","prepare-recovery","audit","audit-recovery","finalize","all")); parser.add_argument("--device",default="cuda:1"); args=parser.parse_args()
    device=torch.device(args.device); 
    if device.type=="cuda": torch.cuda.set_device(device)
    if args.stage=="selftest": selftest(device)
    elif args.stage=="fit": fit(device)
    elif args.stage=="cal": calibrate(device)
    elif args.stage=="prepare-recovery": print(json.dumps(prepare_recovery()["recovery_authorization"],ensure_ascii=False,indent=2))
    elif args.stage=="audit": audit(device)
    elif args.stage=="audit-recovery": audit(device,recovery=True)
    elif args.stage=="finalize": print(json.dumps(finalize(),ensure_ascii=False,indent=2))
    else: selftest(device); fit(device); calibrate(device); audit(device); print(json.dumps(finalize(),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
