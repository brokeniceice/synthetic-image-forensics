#!/usr/bin/env python3
"""Phase 4H-C: A1 pretrained vs A2 random-init direct U_F training arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from training import phase4g1q_conditional_utility as q
from training.phase4gf_formal_localization import full_inputs, load_dev
from training.phase4ha_utility_gated_rectification import baseline_records, gate_to_sam_grid, gated_embedding, load_phase4f, phase4f_records
from training.phase4hb_progressive_unfreezing_stage1 import diagnostics, make_phase4f_batch, train_ids
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import compare, summarize, tensor_state_sha256
from tools.phase4f import Phase4FStore, invalid_record, load_evidence_source, load_sam_runtime, mask_loss

PHASE = "phase4hc"; OUT = ROOT / "outputs" / PHASE; DOC = ROOT / "docs" / PHASE / "report.md"
CKPT_ROOT = Path("checkpoints/phase4hc_direct_utility_arms")
TRAIN_CACHE = Path("cache/phase4g1/g1c/frozen_sources")
CFG = yaml.safe_load((ROOT / "configs/phase4f_language_preserving_rectification.yaml").read_text())
G1S = Path("checkpoints/phase4g1s_mismatch_aware_utility/csculf_mismatch_utility_epoch10.pt")
HA_RESULTS = ROOT / "outputs/phase4ha/results.json"; HB_RESULTS = ROOT / "outputs/phase4hb/dev_results.json"
SEED, EPOCHS, BATCH, LR, WD, CLIP = 3407, 10, 8, 1e-4, 1e-4, 1.0
TAU, MARGIN = 0.041720069924898906, 0.1
HIST = {"A0_Phase4H-A": {"g0": .1727973844346831, "phrase": .22284259199832093, "tf": .2556609647449051},
        "B_Phase4H-B": {"g0": .17348278944076714, "phrase": .22098940309907283, "tf": .25110209046873033}}


def arm_paths(arm):
    root = OUT / arm; ckpt = CKPT_ROOT / arm
    return {"root": root, "history": root / "training_history.csv", "selector": root / "selector.json",
            "results": root / "dev_results.json", "selected": root / "selected_checkpoint.pt",
            "ckpt": ckpt, "manifest": ckpt / "execution_manifest.json"}


def dump(path, value): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def canonical_hash(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def trainable_state(model):
    names = {name for name, value in model.named_parameters() if value.requires_grad}
    return {name: value for name, value in model.state_dict().items() if name in names}


def load_utility(arm, device):
    seed_all(); source = torch.load(q.G1C_CKPT, map_location="cpu", weights_only=False); temps = torch.load(q.G1C_TEMPERATURES, map_location="cpu", weights_only=False)
    model = q.CSCULF(temps["T_L"], temps["T_F"]); model.load_frozen_source_heads(source["language_head"], source["forensic_head"])
    if arm == "a1": model.load_state_dict(torch.load(G1S, map_location="cpu", weights_only=False)["model_state"], strict=True)
    model.language_source.eval().requires_grad_(False); model.forensic_source.eval().requires_grad_(False)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != 371803: raise RuntimeError(f"trainable utility parameter drift: {count}")
    return model.to(device), tensor_state_sha256(trainable_state(model))


def config_contract():
    return {"phase": "Phase4H-C", "architecture": "unchanged CSCU d64/window7/heads4/blocks1/refinement NO",
            "arms": {"a1": "Phase4G-1S pretrained utility", "a2": "seed3407 random-init entire CSCU utility branch"},
            "only_difference": "utility branch initialization", "formula": "S64+U_F*Delta_F; no mapper",
            "trainable": "L/F context, channel/spatial interaction, local exchange, U_F output head; 371803 parameters",
            "frozen": "P1/LLM/LoRA/text_hidden_fcs/SAM/CLIP/Phase4C-A/dense head/Phase4F rectifier/source evidential heads",
            "loss": {"formula": "L_seg+L_relative+L_ranking", "weights": [1., 1., 1.], "tau": TAU, "margin": MARGIN,
                     "ranking_internal": {"cross": .5, "shuffle": .5}, "seg": "2*BCEWithLogits+0.5*soft-Dice"},
            "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WD, "batch": BATCH, "epochs": EPOCHS,
                          "grad_clip": CLIP, "scheduler": "none", "early_stopping": False, "seed": SEED},
            "population": {"train_fake": 8836, "valid": 8690, "invalid_accounting": 146, "real": 0,
                           "dev": 1106, "dev_invalid_iou_zero": 28},
            "order": "Python Random(seed+1009*epoch), canonical 8836 IDs", "cross": "cyclic next canonical ID; current geometry",
            "shuffle": "fixed randperm(576), CPU Generator seed3407, shared F24/z_F24",
            "selector": "each arm epochs1-10 DEV G0 mean IoU only; tie earlier; Phrase/TF/controls forbidden"}


def freeze_manifest(arm, init_hash):
    p = arm_paths(arm); contract = config_contract(); selector = json.loads((ROOT / "outputs/phase4f_language_preserving_rectification/selectors/forensic_rect.json").read_text())
    manifest = {"schema": "phase4hc_arm_manifest_v1", "arm": arm, "status": "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
                "config": contract, "config_sha256": canonical_hash(contract), "implementation_sha256": file_sha256(Path(__file__)),
                "initialization": {"kind": "pretrained_phase4g1s" if arm == "a1" else "random_seed3407", "trainable_state_sha256": init_hash,
                                   "source_checkpoint": str(G1S) if arm == "a1" else None,
                                   "source_checkpoint_sha256": file_sha256(G1S) if arm == "a1" else None},
                "source_files": {"phase4f_rectifier": {"path": selector["selected_checkpoint"], "sha256_before": selector["selected_checkpoint_sha256"]},
                    "sam": {"path": str(Path(CFG["experiment"]["runtime_root"]) / "p1_sam_runtime.pt"), "sha256_before": file_sha256(Path(CFG["experiment"]["runtime_root"]) / "p1_sam_runtime.pt")},
                    "forensic_adapter": {"path": CFG["evidence"]["forensic_checkpoint"], "sha256_before": file_sha256(Path(CFG["evidence"]["forensic_checkpoint"]))},
                    "g1c_source": {"path": str(q.G1C_CKPT), "sha256_before": file_sha256(q.G1C_CKPT)},
                    "temperatures": {"path": str(q.G1C_TEMPERATURES), "sha256_before": file_sha256(q.G1C_TEMPERATURES)}},
                "formal_optimizer_updates": 0, "firewall": {"internal_test_accessed": False, "official1000_accessed": False}}
    if p["manifest"].exists(): raise RuntimeError(f"{arm} already initialized")
    dump(p["manifest"], manifest); return manifest


def rank_loss(match, mismatch, support):
    value = F.relu(MARGIN - (match.float() - mismatch.float())) * support.float()
    return (value.flatten(1).sum(1) / support.flatten(1).sum(1).clamp_min(1)).mean()


def utility_forward(model, batch, *, permutation=None):
    with torch.autocast(device_type=batch["S64"].device.type, dtype=torch.bfloat16):
        return q.utility_forward(model, batch, spatial_permutation=permutation)


def write_history(path, rows):
    fields = ["epoch", "optimizer_updates", "seg_loss", "relative_loss", "ranking_loss", "cross_rank_loss", "shuffle_rank_loss", "total_loss",
              "mean_cross_delta", "mean_shuffle_delta", "traversal_exposures", "optimization_eligible_exposures", "invalid_g0_exposures",
              "dev_g0_mean_iou", "sample_order_sha256", "seconds"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def ids_hash(ids): return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def save_checkpoint(path, model, optimizer, arm, epoch, updates, dev, manifest):
    payload = {"schema": "phase4hc_direct_utility_checkpoint_v1", "arm": arm, "epoch": epoch, "optimizer_updates": updates,
               "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "optimizer": optimizer.state_dict(),
               "validation_g0": dev, "config_sha256": manifest["config_sha256"], "initialization": manifest["initialization"]}
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".pt.tmp"); torch.save(payload, tmp); tmp.replace(path)


def evaluate(model, store, sam, source, rectifier, dev, mode, condition, device, collect=True):
    ids = dev["sample_ids"]; cross = [(i + 1) % len(ids) for i in range(len(ids))]; perm = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    records, u_values, off_exact = [], [], []; model.eval(); model.language_source.eval(); model.forensic_source.eval()
    with torch.no_grad():
        for index, sid in enumerate(ids):
            if not bool(dev["valid"][index]):
                row = invalid_record(sid, dev["original_masks"][index]); row["valid_g0"] = False; records.append(row); continue
            partner = cross[index] if condition == "cross_image" else index; batch = full_inputs(dev, torch.tensor([index]), device)
            if condition == "cross_image":
                batch["F24"] = dev["F24"][partner:partner+1].to(device); batch["z_F24"] = dev["z_F24"][partner:partner+1].to(device)
            permutation = perm if condition == "spatial_shuffle" else None; evidence_ids = [ids[partner]] if partner != index else None
            s64, phase4f, support, _, sc, _ = make_phase4f_batch(store, [sid], source, rectifier, device, permutation=permutation, evidence_ids=evidence_ids)
            output = utility_forward(model, batch, permutation=permutation); u = gate_to_sam_grid(output["U"], sc) * support.float()
            if condition == "forensic_off": u = torch.zeros_like(u)
            adapted = gated_embedding(s64, phase4f, u); qseg = dev["q_seg"][index:index+1].to(device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(qseg, adapted.to(torch.bfloat16))
                if condition == "forensic_off": off_exact.append(torch.equal(low, sam(qseg, s64)))
            logits = inverse_sam_logits(low, dev["sam_geometries"][index]); row = metric_record(sid, logits, dev["original_masks"][index]); row["valid_g0"] = True; records.append(row)
            u_values.append(u[support.bool()].cpu())
            if (index + 1) % 200 == 0: print(json.dumps({"stage": "DEV_EVAL", "mode": mode, "condition": condition, "done": index + 1, "total": len(ids)}), flush=True)
    diag = diagnostics(u_values, u_values); diag = {"scope": diag["scope"], "U_F": diag["U_F"],
        "low_saturation_fraction_U_le_0.01": diag["low_saturation_fraction_g_le_0.01"],
        "high_saturation_fraction_U_ge_0.99": diag["high_saturation_fraction_g_ge_0.99"]}
    return summarize(records), records if collect else None, diag, {"checked": len(off_exact), "all_exact": all(off_exact) if off_exact else None}


def train(arm, device):
    p = arm_paths(arm); seed_all(); model, init_hash = load_utility(arm, device); manifest = freeze_manifest(arm, init_hash)
    store = Phase4FStore(CFG, "train"); val_store = Phase4FStore(CFG, "val"); ids = train_ids()
    if ids != store.sample_ids: raise RuntimeError("train order drift")
    data = q.load_ids(ids, ("valid_g0", "S64", "q_seg", "z_L", "F24", "z_F24", "target64", "clip_geometries")); valid = data["valid_g0"].bool(); id_to_i = {sid:i for i,sid in enumerate(ids)}
    sam = load_sam_runtime(CFG, device); source = load_evidence_source(CFG, "forensic_rect", device); rectifier, _ = load_phase4f(device)
    frozen = {"sam": tensor_state_sha256(sam.state_dict()), "source": tensor_state_sha256(source.state_dict()), "rectifier": tensor_state_sha256(rectifier.state_dict()), "heads": tensor_state_sha256(q.source_state(model))}
    params = [value for value in model.parameters() if value.requires_grad]; optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD); dev = load_dev("g0"); rows=[]; updates=0
    cross = torch.tensor([(i + 1) % len(ids) for i in range(len(ids))]); perm = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    for epoch in range(1, EPOCHS + 1):
        began=time.time(); order=list(ids); random.Random(SEED + 1009*epoch).shuffle(order); sums=defaultdict(float); traversal=eligible=invalid=0
        model.train(); model.language_source.eval(); model.forensic_source.eval()
        for begin in range(0, len(order), BATCH):
            all_batch=order[begin:begin+BATCH]; pos=torch.tensor([id_to_i[sid] for sid in all_batch]); idx=pos[valid.index_select(0,pos)]
            traversal += len(all_batch); eligible += len(idx); invalid += len(all_batch)-len(idx)
            if not len(idx): continue
            batch=full_inputs(data,idx,device); fi=cross.index_select(0,idx); crossed=dict(batch); crossed["F24"]=data["F24"].index_select(0,fi).to(device); crossed["z_F24"]=data["z_F24"].index_select(0,fi).to(device)
            batch_ids=[ids[i] for i in idx.tolist()]; s64,phase4f,support,targets,sc,qseg=make_phase4f_batch(store,batch_ids,source,rectifier,device)
            optimizer.zero_grad(set_to_none=True); matched=utility_forward(model,batch); cross_out=utility_forward(model,crossed); shuffle=utility_forward(model,batch,permutation=perm)
            u=gate_to_sam_grid(matched["U"],sc)*support.float(); adapted=gated_embedding(s64,phase4f,u)
            with torch.autocast(device_type=device.type,enabled=False): low=sam(qseg.to(torch.bfloat16),adapted.to(torch.bfloat16))
            seg=mask_loss(low,targets,CFG); _,soft=q.target_delta({"p_L":matched["p_L"],"p_F":matched["p_F"]},data["target64"].index_select(0,idx).to(device))
            relative=q.image_balanced_loss(matched["utility_logit"],soft,matched["support"]); cr=rank_loss(matched["U"],cross_out["U"],matched["support"]); sr=rank_loss(matched["U"],shuffle["U"],matched["support"]); ranking=.5*(cr+sr); total=seg["total"]+relative+ranking
            if not torch.isfinite(total): raise RuntimeError("nonfinite arm loss")
            total.backward(); norm=torch.nn.utils.clip_grad_norm_(params,CLIP)
            if not torch.isfinite(norm): raise RuntimeError("nonfinite arm gradient")
            optimizer.step(); updates+=1; n=len(idx)
            for key,value in (("seg",seg["total"]),("relative",relative),("ranking",ranking),("cr",cr),("sr",sr),("total",total)): sums[key]+=float(value.detach())*n
            den=matched["support"].flatten(1).sum(1); um=(matched["U"]*matched["support"]).flatten(1).sum(1)/den; uc=(cross_out["U"]*matched["support"]).flatten(1).sum(1)/den; us=(shuffle["U"]*matched["support"]).flatten(1).sum(1)/den
            sums["cross_delta"]+=float((uc-um).sum()); sums["shuffle_delta"]+=float((us-um).sum())
            if updates<=3 or updates%100==0: print(json.dumps({"stage":"TRAIN","arm":arm,"epoch":epoch,"update":updates,"loss":float(total.detach())}),flush=True)
        if (traversal,eligible,invalid)!=(8836,8690,146): raise RuntimeError("exposure mismatch")
        dev_metric,_,_,_=evaluate(model,val_store,sam,source,rectifier,dev,"g0","matched",device,collect=False)
        row={"epoch":epoch,"optimizer_updates":updates,"seg_loss":sums["seg"]/eligible,"relative_loss":sums["relative"]/eligible,"ranking_loss":sums["ranking"]/eligible,
             "cross_rank_loss":sums["cr"]/eligible,"shuffle_rank_loss":sums["sr"]/eligible,"total_loss":sums["total"]/eligible,"mean_cross_delta":sums["cross_delta"]/eligible,
             "mean_shuffle_delta":sums["shuffle_delta"]/eligible,"traversal_exposures":traversal,"optimization_eligible_exposures":eligible,"invalid_g0_exposures":invalid,
             "dev_g0_mean_iou":dev_metric["mean_foreground_iou"],"sample_order_sha256":ids_hash(order),"seconds":time.time()-began}
        rows.append(row); write_history(p["history"],rows); save_checkpoint(p["ckpt"]/f"epoch_{epoch}.pt",model,optimizer,arm,epoch,updates,dev_metric,manifest); print(json.dumps({"stage":"EPOCH_COMPLETE","arm":arm,**row}),flush=True)
    best=max(rows,key=lambda x:(x["dev_g0_mean_iou"],-x["epoch"])); chosen=p["ckpt"]/f"epoch_{best['epoch']}.pt"; p["root"].mkdir(parents=True,exist_ok=True); shutil.copy2(chosen,p["selected"])
    selector={"schema":"phase4hc_selector_v1","arm":arm,"status":"COMPLETE","primary":"DEV G0 mean IoU","population":1106,"invalid_iou_zero":28,"tie_break":"earlier epoch",
              "selected_epoch":best["epoch"],"selected_dev_g0":best["dev_g0_mean_iou"],"selected_checkpoint_sha256":file_sha256(chosen),"output_selected_checkpoint_sha256":file_sha256(p["selected"]),
              "candidates":[{"epoch":x["epoch"],"mean_g0_iou":x["dev_g0_mean_iou"]} for x in rows],"phrase_used":False,"tf_used":False,"controls_used":False,
              "initialization":manifest["initialization"],"sample_order_hashes":[x["sample_order_sha256"] for x in rows]}; dump(p["selector"],selector)
    after={"sam":tensor_state_sha256(sam.state_dict()),"source":tensor_state_sha256(source.state_dict()),"rectifier":tensor_state_sha256(rectifier.state_dict()),"heads":tensor_state_sha256(q.source_state(model))}
    if after!=frozen: raise RuntimeError("frozen source mutation")
    manifest.update({"status":"TRAINING_COMPLETE_SELECTOR_FROZEN","formal_optimizer_updates":updates,"frozen_hash_before":frozen,"frozen_hash_after":after,"selector_sha256":file_sha256(p["selector"]),"selected_sha256":file_sha256(p["selected"])}); dump(p["manifest"],manifest)


def evaluate_selected(arm,device):
    p=arm_paths(arm); state=torch.load(p["selected"],map_location="cpu",weights_only=False); model,_=load_utility("a2",device); model.load_state_dict(state["model_state"],strict=True); model.eval()
    store=Phase4FStore(CFG,"val"); sam=load_sam_runtime(CFG,device); source=load_evidence_source(CFG,"forensic_rect",device); rectifier,_=load_phase4f(device); dev={mode:load_dev(mode) for mode in ("g0","phrase","tf")}
    metrics={};records={};diags={};identities={}
    for mode,condition in (("g0","matched"),("phrase","matched"),("tf","matched"),("g0","cross_image"),("g0","spatial_shuffle"),("g0","forensic_off")):
        key={"phrase":"phrase","tf":"tf"}.get(mode,condition); metrics[key],records[key],diags[key],identities[key]=evaluate(model,store,sam,source,rectifier,dev[mode],mode,condition,device)
    ids=store.sample_ids; p1={m:baseline_records(m,ids) for m in ("g0","phrase","tf")}; p4f={m:phase4f_records(m) for m in ("g0","phrase","tf")}; ha=json.loads(HA_RESULTS.read_text()); hb=json.loads(HB_RESULTS.read_text())
    baselines={"A0": {"g0":ha["records"]["matched"],"phrase":ha["records"]["phrase"],"tf":ha["records"]["tf"]},
               "B": {"g0":hb["records"]["matched"],"phrase":hb["records"]["phrase"],"tf":hb["records"]["tf"]},"P1":p1,"Phase4F":p4f}
    stats={name:{} for name in baselines}
    for mode,key in (("g0","matched"),("phrase","phrase"),("tf","tf")):
        for name,base in baselines.items(): stats[name][mode]=compare(records[key],base[mode],seed=SEED)
    stats["matched_minus_cross"]=compare(records["matched"],records["cross_image"],seed=SEED); stats["matched_minus_shuffle"]=compare(records["matched"],records["spatial_shuffle"],seed=SEED)
    manifest=json.loads(p["manifest"].read_text()); result={"schema":"phase4hc_arm_results_v1","arm":arm,"status":"COMPLETE","selector":json.loads(p["selector"].read_text()),"metrics":metrics,"records":records,"statistics":stats,
        "utility_diagnostics":diags,"identities":identities,"initialization":manifest["initialization"],"source_hash_integrity":manifest["frozen_hash_before"]==manifest["frozen_hash_after"],
        "firewall":{"internal_test_accessed":False,"official1000_accessed":False}}; dump(p["results"],result)
    manifest.update({"status":"COMPLETE","results_sha256":file_sha256(p["results"])}); dump(p["manifest"],manifest)


def comparison():
    a1=json.loads(arm_paths("a1")["results"].read_text());a2=json.loads(arm_paths("a2")["results"].read_text())
    paired={};
    for mode,key in (("g0","matched"),("phrase","phrase"),("tf","tf")): paired[mode]=compare(a1["records"][key],a2["records"][key],seed=SEED)
    order_equal=json.loads(arm_paths("a1")["selector"].read_text())["sample_order_hashes"]==json.loads(arm_paths("a2")["selector"].read_text())["sample_order_hashes"]
    comp={"schema":"phase4hc_comparison_v1","status":"COMPLETE","metrics":{"A0":HIST["A0_Phase4H-A"],"B":HIST["B_Phase4H-B"],
          "A1":{"g0":a1["metrics"]["matched"]["mean_foreground_iou"],"phrase":a1["metrics"]["phrase"]["mean_foreground_iou"],"tf":a1["metrics"]["tf"]["mean_foreground_iou"]},
          "A2":{"g0":a2["metrics"]["matched"]["mean_foreground_iou"],"phrase":a2["metrics"]["phrase"]["mean_foreground_iou"],"tf":a2["metrics"]["tf"]["mean_foreground_iou"]}},
          "paired_A1_minus_A2":paired,"identical_data_order":order_equal,"initializations":{"A1":a1["initialization"],"A2":a2["initialization"]},
          "firewall":{"internal_test_accessed":False,"official1000_accessed":False}}; dump(OUT/"comparison.json",comp); finalize(comp,a1,a2)


def finalize(comp,a1,a2):
    d=comp["paired_A1_minus_A2"]["g0"]["foreground_iou"]; a1_gt_a2=d["bootstrap_95_ci"][0]>0; a2_gt_a1=d["bootstrap_95_ci"][1]<0
    a1_b=a1["statistics"]["B"]["g0"]["foreground_iou"]; a1_gt_b=a1_b["bootstrap_95_ci"][0]>0
    def order(r): return r["metrics"]["matched"]["mean_foreground_iou"]<r["metrics"]["phrase"]["mean_foreground_iou"]<r["metrics"]["tf"]["mean_foreground_iou"]
    def controls(r): return r["statistics"]["matched_minus_cross"]["foreground_iou"]["bootstrap_95_ci"][0]>0 and r["statistics"]["matched_minus_shuffle"]["foreground_iou"]["bootstrap_95_ci"][0]>0
    def collapse(r):
        x=r["utility_diagnostics"]["matched"]; return x["low_saturation_fraction_U_le_0.01"]>=.95 or x["high_saturation_fraction_U_ge_0.99"]>=.95 or x["U_F"]["std"]<.01
    def dominance(r):
        g=r["statistics"]["A0"]["g0"]["foreground_iou"]["mean_difference"]>0; p=r["statistics"]["A0"]["phrase"]["foreground_iou"]["mean_difference"]<-.01; t=r["statistics"]["A0"]["tf"]["foreground_iou"]["mean_difference"]<-.01
        return g and (p or t or not order(r))
    gates={"schema":"phase4hc_gate_summary_v1","A1_COMPLETE":"YES","A2_COMPLETE":"YES","IDENTICAL_DATA_ORDER":"PASS" if comp["identical_data_order"] else "FAIL",
      "A1_LANGUAGE_ORDER_PRESERVED":"YES" if order(a1) else "NO","A2_LANGUAGE_ORDER_PRESERVED":"YES" if order(a2) else "NO",
      "A1_MATCHED_GT_CROSS_SHUFFLE":"YES" if controls(a1) else "NO","A2_MATCHED_GT_CROSS_SHUFFLE":"YES" if controls(a2) else "NO",
      "A1_GATE_COLLAPSE":"YES" if collapse(a1) else "NO","A2_GATE_COLLAPSE":"YES" if collapse(a2) else "NO",
      "A1_SOURCE_HASH_INTEGRITY":"PASS" if a1["source_hash_integrity"] else "FAIL","A2_SOURCE_HASH_INTEGRITY":"PASS" if a2["source_hash_integrity"] else "FAIL",
      "UTILITY_PRETRAINING_HELPFUL":"YES" if a1_gt_a2 else ("NO" if a2_gt_a1 else "NOT_SUPPORTED"),
      "UTILITY_PRETRAINING_NECESSARY":"SUPPORTED" if a1_gt_a2 else "NOT_SUPPORTED",
      "A1_SIGNIFICANTLY_BEATS_PHASE4HB":"YES" if a1_gt_b else "NO","A1_FORENSIC_DOMINANCE":"YES" if dominance(a1) else "NO","A2_FORENSIC_DOMINANCE":"YES" if dominance(a2) else "NO",
      "INTERNAL_TEST_ACCESSED":"NO","OFFICIAL1000_ACCESSED":"NO"}; dump(OUT/"gate_summary.json",gates); write_report(comp,a1,a2,gates)


def write_report(comp,a1,a2,gates):
    m=comp["metrics"]; table="\n".join(["| Arm | G0 | Phrase | TF |","|---|---:|---:|---:|"]+[f"| {name} | {x['g0']:.6f} | {x['phrase']:.6f} | {x['tf']:.6f} |" for name,x in m.items()])
    def stat(x):
        v=x["foreground_iou"];return f"delta={v['mean_difference']:+.6f}, CI=[{v['bootstrap_95_ci'][0]:+.6f},{v['bootstrap_95_ci'][1]:+.6f}], W/T/L={v['wins']}/{v['ties']}/{v['losses']}, p={v['wilcoxon_pvalue']:.4g}"
    text=f"""# Phase 4H-C Direct U_F Fine-tuning — A1/A2 Parallel Arms

## Protocol

A1从Phase4G-1S mismatch-aware utility初始化，A2以seed3407随机初始化同一371,803参数CSCU utility branch。除此之外架构、数据顺序、cross/shuffle corruption、loss、optimizer和selector完全一致。两臂公式均为`S64_adapt=S64+U_F*Delta_F`，无mapper、late fusion、refinement或新decoder。Loss为`L_seg+L_relative+L_ranking`，权重1:1:1；tau={TAU:.10f}，margin=.1。两臂在GPU1/2并行训练，各自只用DEV G0 selector。

## Results

{table}

- A1 vs A0 G0: {stat(a1['statistics']['A0']['g0'])}
- A1 vs B G0: {stat(a1['statistics']['B']['g0'])}
- A2 vs A0 G0: {stat(a2['statistics']['A0']['g0'])}
- A1−A2 G0: {stat(comp['paired_A1_minus_A2']['g0'])}

## Arm diagnostics

A1 selected epoch={a1['selector']['selected_epoch']}；language order={gates['A1_LANGUAGE_ORDER_PRESERVED']}；matched>cross/shuffle={gates['A1_MATCHED_GT_CROSS_SHUFFLE']}；gate collapse={gates['A1_GATE_COLLAPSE']}。U_F matched diagnostics：`{json.dumps(a1['utility_diagnostics']['matched'],ensure_ascii=False)}`。

A2 selected epoch={a2['selector']['selected_epoch']}；language order={gates['A2_LANGUAGE_ORDER_PRESERVED']}；matched>cross/shuffle={gates['A2_MATCHED_GT_CROSS_SHUFFLE']}；gate collapse={gates['A2_GATE_COLLAPSE']}。random initialization hash=`{a2['initialization']['trainable_state_sha256']}`。U_F matched diagnostics：`{json.dumps(a2['utility_diagnostics']['matched'],ensure_ascii=False)}`。

## Interpretation and gates

```json
{json.dumps(gates,ensure_ascii=False,indent=2)}
```

结论必须同时结合G0 CI、Phrase/TF和language order；不得只按G0宣告成功。到此严格STOP，未访问internal test或official1000。
"""; DOC.parent.mkdir(parents=True,exist_ok=True); DOC.write_text(text,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("train","evaluate","compare"));parser.add_argument("--arm",choices=("a1","a2"));parser.add_argument("--device",default="cuda:1");args=parser.parse_args()
    if args.stage=="compare": comparison(); return
    if not args.arm: raise ValueError("--arm required")
    device=torch.device(args.device);torch.cuda.set_device(device)
    if args.stage=="train": train(args.arm,device)
    else: evaluate_selected(args.arm,device)


if __name__=="__main__": main()
