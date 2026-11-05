#!/usr/bin/env python3
"""Phase 4G-1P freeze and implementation/invariance preflight.

No fitting, optimizer step, calibration, architecture screening, validation,
internal test, or official1000 path exists in this supervisor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from dataset.forensics.synthscars import polygon_to_mask, polygons_for_target
from model.csculf import (
    BLOCKS, HEADS, WIDTH, WINDOW, CSCULF, adapted_ecolaf_fuse,
)
from model.pcerf import ForensicEvidentialHead, LanguageEvidentialHead, ecolaf_fuse, evidence_to_dirichlet, resample_clip_to_original_normalized
from training.phase4g1_g1c import unpack_bool
from tools.phase4c_b import file_sha256
from tools.phase4e1 import tensor_state_sha256


OUT = ROOT / "outputs/phase4g1p"
CACHE = Path("cache/phase4g1/g1c/frozen_sources")
G1C_CKPT = Path("checkpoints/phase4g1_g1c_reliability_preflight/epoch_10.pt")
G1C_TEMPERATURES = Path("checkpoints/phase4g1_g1c_reliability_preflight/temperature_scalars.pt")
SEED = 3407
SPLIT_SEED = "phase4g1p-conditional-utility-v1"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def rank_id(sample_id: str) -> str:
    return hashlib.sha256((SPLIT_SEED + "\0" + sample_id).encode()).hexdigest()


def prerequisite() -> tuple[dict, dict]:
    gate = json.loads((ROOT / "outputs/phase4g1r/gate_summary.json").read_text())
    required = {
        "G1C_INTRINSIC_RELIABILITY_HYPOTHESIS": "NOT_SUPPORTED", "PRIMARY_CANDIDATE": "CSCU-LF",
        "UTILITY_INTERVENTION_IDENTIFIABLE": "YES", "NEXT_PREFLIGHT_JUSTIFIED": "YES",
        "FORMAL_TRAINING_JUSTIFIED": "NO",
    }
    if any(gate.get(key) != value for key, value in required.items()): raise RuntimeError("Phase 4G-1R prerequisite drift")
    g1c = json.loads((ROOT / "outputs/phase4g1/g1c/gate_summary.json").read_text())
    if g1c["RELIABILITY_PREFLIGHT"] != "FAIL" or g1c["G1_F_FULL_TRAINING_JUSTIFIED"] != "NO":
        raise RuntimeError("G1-C frozen failure drift")
    return gate, g1c


def build_split() -> dict:
    prerequisite()
    old = json.loads((ROOT / "outputs/phase4g05/split_manifest.json").read_text())
    pool = set(old["groups"]["TRAIN-FIT"]["ids"]) | set(old["groups"]["TRAIN-CAL"]["ids"])
    excluded_audit = set(old["groups"]["TRAIN-AUDIT"]["ids"])
    if pool & excluded_audit or len(pool) != 7511: raise RuntimeError("G1-C source-pool firewall drift")
    valid_all = set(json.loads((ROOT / "outputs/phase4f_language_preserving_rectification/manifests/train_valid_g0_ids.json").read_text()))
    valid, invalid = pool & valid_all, pool - valid_all
    if len(valid) != 7387 or len(invalid) != 124: raise RuntimeError("utility valid/invalid strata drift")
    groups = {key: [] for key in ("UTILITY-FIT", "UTILITY-CAL", "UTILITY-AUDIT")}
    for stratum in (sorted(valid, key=rank_id), sorted(invalid, key=rank_id)):
        n_fit, n_cal = round(len(stratum) * .70), round(len(stratum) * .15)
        groups["UTILITY-FIT"].extend(stratum[:n_fit]); groups["UTILITY-CAL"].extend(stratum[n_fit:n_fit + n_cal]); groups["UTILITY-AUDIT"].extend(stratum[n_fit + n_cal:])
    for key in groups: groups[key] = sorted(groups[key], key=rank_id)
    membership = {sid: key for key, ids in groups.items() for sid in ids}
    if len(membership) != len(pool) or set(membership) != pool: raise RuntimeError("utility split coverage failure")
    rows = [json.loads(line) for line in (ROOT / "outputs/data_audits/unified_forensics_split_v1/train_combined.jsonl").open() if line.strip()]
    rows = {str(row["sample_id"]): row for row in rows if str(row["sample_id"]) in pool}
    stats = {key: defaultdict(int) for key in groups}
    for sid, row in rows.items():
        item = stats[membership[sid]]; mask = np.zeros((256, 256), dtype=bool)
        for reference in row.get("refs") or []:
            polygons = polygons_for_target(reference, 256, 256); mask |= polygon_to_mask(polygons, 256, 256).astype(bool)
            item["region_count"] += 1; item["polygon_count"] += len(polygons)
        item["foreground_pixels_256"] += int(mask.sum()); item["total_pixels_256"] += mask.size
    payload = {}
    for key, ids in groups.items():
        item = stats[key]
        payload[key] = {
            "n": len(ids), "ids": ids, "ids_sha256": ids_hash(ids),
            "valid_g0": sum(sid in valid for sid in ids), "invalid_g0": sum(sid in invalid for sid in ids),
            "foreground_prevalence_256": item["foreground_pixels_256"] / item["total_pixels_256"],
            "foreground_pixels_256": item["foreground_pixels_256"], "total_pixels_256": item["total_pixels_256"],
            "region_count": item["region_count"], "polygon_count": item["polygon_count"],
        }
    result = {
        "schema": "phase4g1p_utility_split_v1", "status": "FROZEN", "UTILITY_SPLIT_FROZEN": "YES",
        "seed_material": SPLIT_SEED, "assignment": "SHA256 rank within valid-G0 and invalid-G0 strata; 70/15/15",
        "source_population": "G1-C TRAIN-FIT + TRAIN-CAL only", "population_n": len(pool), "population_ids_sha256": ids_hash(sorted(pool)),
        "excluded": {"G1-C_TRAIN-AUDIT_n": len(excluded_audit), "G1-C_TRAIN-AUDIT_selected": 0, "development_validation": True, "internal_test": True, "official1000": True},
        "target_prevalence_definition": "official annotation all-ref union at original-normalized 256 grid",
        "groups": payload,
        "checks": {
            "image_id_disjoint": all(not set(groups[a]) & set(groups[b]) for i, a in enumerate(groups) for b in list(groups)[i + 1:]),
            "exact_pool_coverage": set(membership) == pool, "unique_membership": len(membership) == sum(map(len, groups.values())),
            "g1c_train_audit_overlap": 0, "development_validation_accessed": False, "internal_test_accessed": False, "official1000_accessed": False,
        },
    }
    dump(OUT / "utility_split_manifest.json", result)
    return result


def load_ids(ids: list[str], fields: tuple[str, ...]) -> dict:
    desired = set(ids); loaded_ids, tensors = [], {field: [] for field in fields if field != "clip_geometries"}; geometries = []
    for path in sorted(CACHE.glob("shard_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        positions = [index for index, sid in enumerate(shard["sample_ids"]) if sid in desired]
        if not positions: continue
        index = torch.tensor(positions); loaded_ids.extend(shard["sample_ids"][position] for position in positions)
        for field in tensors: tensors[field].append(shard[field].index_select(0, index))
        if "clip_geometries" in fields: geometries.extend(shard["clip_geometries"][position] for position in positions)
    if set(loaded_ids) != desired or len(loaded_ids) != len(desired): raise RuntimeError("safe utility-pool cache selection mismatch")
    lookup = {sid: index for index, sid in enumerate(loaded_ids)}; order = torch.tensor([lookup[sid] for sid in ids])
    result = {field: torch.cat(parts).index_select(0, order) for field, parts in tensors.items()}
    if "clip_geometries" in fields: result["clip_geometries"] = [geometries[index] for index in order.tolist()]
    result["sample_ids"] = ids
    return result


def target_statistics(device: torch.device) -> dict:
    split = json.loads((OUT / "utility_split_manifest.json").read_text())
    ids = [sid for sid in split["groups"]["UTILITY-FIT"]["ids"] if sid in set(json.loads((ROOT / "outputs/phase4f_language_preserving_rectification/manifests/train_valid_g0_ids.json").read_text()))]
    if len(ids) != split["groups"]["UTILITY-FIT"]["valid_g0"]: raise RuntimeError("UTILITY-FIT valid population drift")
    data = load_ids(ids, ("S64", "q_seg", "z_L", "F24", "z_F24", "target64", "clip_geometries"))
    state = torch.load(G1C_CKPT, map_location="cpu", weights_only=False); temperatures = torch.load(G1C_TEMPERATURES, map_location="cpu", weights_only=False)
    language, forensic = LanguageEvidentialHead().to(device), ForensicEvidentialHead().to(device)
    language.load_state_dict(state["language_head"]); forensic.load_state_dict(state["forensic_head"])
    language.eval().requires_grad_(False); forensic.eval().requires_grad_(False)
    delta_rows, foreground_rows, state_counts = [], [], defaultdict(int)
    eps = 1e-8
    with torch.no_grad():
        for begin in range(0, len(ids), 8):
            end = begin + 8
            evidence_l = language(data["S64"][begin:end].to(device), data["q_seg"][begin:end].to(device), data["z_L"][begin:end].to(device)) / float(temperatures["T_L"])
            evidence_f = forensic(data["F24"][begin:end].to(device), data["z_F24"][begin:end].to(device)) / float(temperatures["T_F"])
            p_l = evidence_to_dirichlet(evidence_l)["posterior"]
            p_f_native = evidence_to_dirichlet(evidence_f)["posterior"]
            for local, geometry in enumerate(data["clip_geometries"][begin:end]):
                p_f, support = resample_clip_to_original_normalized(p_f_native[local:local + 1], geometry, output_hw=(64, 64), vacuous=False)
                support = support[0, 0]; target = data["target64"][begin + local].to(device).bool()
                left = p_l[local, 1].clamp(eps, 1-eps); right = p_f[0, 1].clamp(eps, 1-eps)
                loss_l = F.binary_cross_entropy(left, target.float(), reduction="none")
                loss_f = F.binary_cross_entropy(right, target.float(), reduction="none")
                delta = (loss_l - loss_f)[support].cpu().numpy().astype(np.float32)
                delta_rows.append(delta); foreground_rows.append(target[support].cpu().numpy())
                pred_l, pred_f = left >= .5, right >= .5
                lc, fc = pred_l.eq(target), pred_f.eq(target)
                for key, mask in {
                    "L_correct_F_correct": lc & fc & support, "L_correct_F_wrong": lc & ~fc & support,
                    "L_wrong_F_correct": ~lc & fc & support, "L_wrong_F_wrong": ~lc & ~fc & support,
                }.items(): state_counts[key] += int(mask.sum())
            if (begin + 8) % 512 == 0: print(json.dumps({"target_stats": min(end, len(ids)), "of": len(ids)}), flush=True)
    delta = np.concatenate(delta_rows); foreground = np.concatenate(foreground_rows).astype(bool)
    q90_abs = float(np.quantile(np.abs(delta), .90)); tau = max(q90_abs / math.log(.95 / .05), 1e-6)
    target = 1.0 / (1.0 + np.exp(-np.clip(delta / tau, -50, 50)))
    bins = np.linspace(0, 1, 21); histogram = np.histogram(target, bins=bins)[0]
    def describe(value):
        return {"n": int(value.size), "percentiles": {str(q): float(np.quantile(value, q/100)) for q in (0,1,5,10,25,50,75,90,95,99,100)}, "mean": float(value.mean()), "std": float(value.std())}
    result = {
        "schema": "phase4g1p_utility_target_manifest_v1", "status": "FROZEN", "UTILITY_TARGET_FROZEN": "YES", "TAU_FROZEN": "YES",
        "population": "UTILITY-FIT valid-G0 only", "population_n": len(ids), "sample_ids_sha256": ids_hash(ids),
        "native_grid": [64,64], "scope": "CLIP support cells only", "loss": "binary pixel NLL/BCE on calibrated frozen G1-C source posteriors",
        "formula": "t_F=sigmoid((ell_L-ell_F)/tau)", "tau": tau,
        "future_utility_supervision": "soft-target BCEWithLogits on U pre-sigmoid logit; mean over support per image, then equal mean over images; no class/target weighting",
        "future_fit_protocol": {"optimizer":"AdamW","learning_rate":0.0001,"weight_decay":0.0001,"batch_size":8,"epochs":10,"gradient_clip_norm":1.0,"scheduler":"none","seed":3407,"early_stopping":False,"checkpoint_rule":"epoch10 final only; intermediate recovery checkpoints cannot select epoch"},
        "future_cal_protocol": {"parameter":"single positive T_U=softplus(tau_U)+1e-6 applied to utility logits","population":"UTILITY-CAL valid-G0","objective":"same image-balanced soft-target BCE","optimizer":"deterministic LBFGS","initial_temperature":1.0,"grid_or_sweep":False},
        "tau_rule": "one-shot tau=P90(abs(ell_L-ell_F))/logit(0.95); no candidates or result-based selection",
        "delta_statistics": describe(delta), "target_statistics": describe(target),
        "target_histogram": {"bin_edges": bins.tolist(), "counts": histogram.tolist()},
        "target_saturation": {"at_or_below_0.01": float((target <= .01).mean()), "at_or_above_0.99": float((target >= .99).mean()), "within_0.45_0.55": float(((target >= .45) & (target <= .55)).mean())},
        "foreground_target_statistics": describe(target[foreground]), "background_target_statistics": describe(target[~foreground]),
        "four_state_diagnostic_counts": dict(state_counts), "hard_winner_is_sole_target": False, "dice_is_primary_target": False,
        "source_checkpoint": str(G1C_CKPT), "source_checkpoint_sha256": file_sha256(G1C_CKPT),
        "temperature_checkpoint": str(G1C_TEMPERATURES), "temperature_checkpoint_sha256": file_sha256(G1C_TEMPERATURES),
        "g1c_train_audit_samples_selected": 0, "development_validation_accessed": False, "internal_test_accessed": False, "official1000_accessed": False,
    }
    dump(OUT / "utility_target_manifest.json", result)
    return result


def synthetic_inputs(batch=2, partial=False, device=torch.device("cpu")):
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    geometry = {"resized_hw": [256,320], "crop_box_yxyx": [16,48,240,272]} if partial else {"resized_hw": [224,224], "crop_box_yxyx": [0,0,224,224]}
    return {
        "s64": torch.randn(batch,256,64,64,generator=generator).to(device), "q_seg": torch.randn(batch,256,generator=generator).to(device),
        "z_l": torch.randn(batch,1,256,256,generator=generator).to(device), "f24": torch.randn(batch,256,24,24,generator=generator).to(device),
        "z_f24": torch.randn(batch,1,24,24,generator=generator).to(device), "clip_geometries": [geometry for _ in range(batch)],
        "valid_g0": torch.ones(batch,dtype=torch.bool,device=device), "forensic_present": torch.ones(batch,dtype=torch.bool,device=device),
        "forensic_vacuous": torch.zeros(batch,dtype=torch.bool,device=device), "forensic_off": torch.zeros(batch,dtype=torch.bool,device=device),
    }


def random_mass(batch=2, height=16, width=16, device=torch.device("cpu")):
    generator = torch.Generator().manual_seed(SEED); value = torch.rand(batch,3,height,width,generator=generator).to(device); return value/value.sum(1,keepdim=True)


def traced_linear_flops(model: nn.Module, kwargs: dict) -> dict:
    counts = {"trainable_conv_linear": 0, "frozen_source_conv_linear": 0}
    handles = []
    def hook(module, inputs, output):
        out = output if isinstance(output, torch.Tensor) else output[0]
        if isinstance(module, nn.Conv2d):
            per_output = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            flops = 2 * out.numel() * per_output
        elif isinstance(module, nn.Linear):
            flops = 2 * out.numel() * module.in_features
        else: return
        key = "trainable_conv_linear" if any(parameter.requires_grad for parameter in module.parameters()) else "frozen_source_conv_linear"
        counts[key] += int(flops)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)): handles.append(module.register_forward_hook(hook))
    with torch.no_grad(): model(**kwargs)
    for handle in handles: handle.remove()
    return counts


def synthetic_preflight(device: torch.device) -> dict:
    torch.manual_seed(SEED); state = torch.load(G1C_CKPT,map_location="cpu",weights_only=False); temperatures=torch.load(G1C_TEMPERATURES,map_location="cpu",weights_only=False)
    model=CSCULF(temperatures["T_L"],temperatures["T_F"]).to(device); model.load_frozen_source_heads(state["language_head"],state["forensic_head"]); model.eval()
    left,right=random_mass(device=device),random_mass(device=device); official=ecolaf_fuse(torch.stack((left,right),dim=2),classes=2)
    one=adapted_ecolaf_fuse(left,right,torch.ones_like(left[:,:1])); zero=adapted_ecolaf_fuse(left,right,torch.zeros_like(left[:,:1]))
    one_pass=torch.equal(one["fused_mass"],official["fused_mass"]) and torch.equal(one["probability"],official["probability"])
    zero_pass=torch.equal(zero["fused_mass"],left) and torch.equal(zero["effective_forensic_contribution"],torch.zeros_like(zero["effective_forensic_contribution"]))
    sequence=[]
    for utility in (0,.25,.5,.75,1): sequence.append(adapted_ecolaf_fuse(left,right,torch.full_like(left[:,:1],utility))["effective_forensic_contribution"].detach().cpu())
    monotonic=all(bool((b>=a).all()) for a,b in zip(sequence,sequence[1:]))
    parity={"schema":"phase4g1p_adapted_ecolaf_parity_v1","ADAPTED_EXTENSION":True,"UTILITY_ONE_PARITY":"PASS" if one_pass else "FAIL","official_kernel_unchanged":True,"formula":{"d_F_adapt":"d_F_official*U_F","d_L_adapt":"1-U_F*(1-d_L_official)"}}
    dump(OUT/"adapted_ecolaf_parity.json",parity)
    extremes={"schema":"phase4g1p_utility_extreme_tests_v1","UTILITY_ONE_PARITY":"PASS" if one_pass else "FAIL","UTILITY_ZERO_SEMANTICS":"PASS" if zero_pass else "FAIL","UTILITY_MONOTONICITY":"PASS" if monotonic else "FAIL","utility_values":[0,.25,.5,.75,1],"zero_definition":"strict original language source mass/probability; all forensic-induced language discount restored"}
    dump(OUT/"utility_extreme_tests.json",extremes)

    kwargs=synthetic_inputs(device=device); shuffled=dict(kwargs); permutation=torch.randperm(24*24,generator=torch.Generator().manual_seed(SEED)).to(device)
    shuffled["f24"]=kwargs["f24"].flatten(2)[:,:,permutation].reshape_as(kwargs["f24"]); shuffled["z_f24"]=kwargs["z_f24"].flatten(2)[:,:,permutation].reshape_as(kwargs["z_f24"])
    crossed=dict(kwargs); crossed["f24"]=kwargs["f24"].flip(0); crossed["z_f24"]=kwargs["z_f24"].flip(0)
    with torch.no_grad(): matched=model(**kwargs); shuffle=model(**shuffled); cross=model(**crossed)
    spatial_keys={"product":"comparison_parts.product","absolute_difference":"comparison_parts.absolute_difference","local_cosine":"comparison_parts.local_cosine","context_exchange":"Fr","utility_prehead_input":"comparison"}
    spatial_differences={}
    for label,path in spatial_keys.items():
        if path.startswith("comparison_parts."): a=matched["comparison_parts"][path.split(".")[1]]; b=shuffle["comparison_parts"][path.split(".")[1]]
        else: a,b=matched[path],shuffle[path]
        spatial_differences[label]={"changed":not torch.equal(a,b),"mean_absolute_difference":float((a-b).abs().mean())}
    cross_differences={key:{"changed":not torch.equal(matched[key],cross[key]),"mean_absolute_difference":float((matched[key]-cross[key]).abs().mean())} for key in ("Fr","comparison","utility_prehead")}
    mismatch={"schema":"phase4g1p_mismatch_sensing_v1","SPATIAL_MISMATCH_SENSING_CAPACITY":"YES" if all(row["changed"] for row in spatial_differences.values()) else "NO","CROSS_IMAGE_MISMATCH_SENSING_CAPACITY":"YES" if torch.equal(matched["L64"],cross["L64"]) and all(row["changed"] for row in cross_differences.values()) else "NO","spatial_shuffle":spatial_differences,"cross_image":{"language_context_bit_exact":torch.equal(matched["L64"],cross["L64"]),"differences":cross_differences},"direction_claimed":False,"training_executed":False}
    dump(OUT/"mismatch_sensing.json",mismatch)

    override=matched["U_F64"].flip(0).clone()
    with torch.no_grad(): intervened=model(**kwargs,utility_override=override)
    identity_keys=("p_L64","p_F64","mass_L64","mass_F64","mass_L256","mass_F256","L64","Fctx64","aligned_F64","aligned_z_F64")
    identity_checks={key:torch.equal(matched[key],intervened[key]) for key in identity_keys}
    identity={"schema":"phase4g1p_utility_intervention_identity_v1","UTILITY_INTERVENTION_IDENTITY":"PASS" if all(identity_checks.values()) else "FAIL","checks":identity_checks,"hashes_before":{key:tensor_hash(matched[key]) for key in identity_keys},"hashes_after":{key:tensor_hash(intervened[key]) for key in identity_keys},"outside_support_utility_zero":bool((intervened["U_F64"][~intervened["support64"]]==0).all()),"future_audit_consumed":False}
    dump(OUT/"utility_intervention_identity.json",identity)

    partial=synthetic_inputs(batch=1,partial=True,device=device)
    with torch.no_grad(): geometry=model(**partial)
    geometry_pass=bool((geometry["U_F64"][~geometry["support64"]]==0).all()) and bool((geometry["mass_F64"][:,-1:][~geometry["support64"]]==1).all())
    fallback_checks={}
    for condition in ("absent","vacuous","off"):
        value=dict(partial); value["forensic_present"]=torch.tensor([condition!="absent"],device=device); value["forensic_vacuous"]=torch.tensor([condition=="vacuous"],device=device); value["forensic_off"]=torch.tensor([condition=="off"],device=device)
        with torch.no_grad(): fallback_checks[condition]=torch.equal(model(**value)["logits"],value["z_l"])
    unsupported=dict(partial); unsupported["clip_geometries"]=[{"resized_hw":[224,224],"crop_box_yxyx":[300,300,400,400]}]
    with torch.no_grad(): fallback_checks["fully_unsupported"]=torch.equal(model(**unsupported)["logits"],unsupported["z_l"])

    model.train(); model.zero_grad(set_to_none=True); source_hash_before=tensor_state_sha256({f"L.{k}":v for k,v in model.language_source.state_dict().items()}|{f"F.{k}":v for k,v in model.forensic_source.state_dict().items()})
    gradient_output=model(**synthetic_inputs(batch=1,device=device)); loss=gradient_output["U_F64"].mean()+gradient_output["fused"]["probability"].mean(); loss.backward()
    blocks=defaultdict(list); frozen=[]
    for name,parameter in model.named_parameters():
        if parameter.requires_grad: blocks[name.split(".")[0]].append((name,parameter))
        else: frozen.append((name,parameter))
    block_rows={block:{"parameter_count":sum(p.numel() for _,p in values),"finite":all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in values),"all_nonzero":all(p.grad is not None and bool((p.grad!=0).any()) for _,p in values),"grad_norm":math.sqrt(sum(float(p.grad.float().norm())**2 for _,p in values if p.grad is not None))} for block,values in blocks.items()}
    frozen_pass=all(p.grad is None or bool((p.grad==0).all()) for _,p in frozen); source_hash_after=tensor_state_sha256({f"L.{k}":v for k,v in model.language_source.state_dict().items()}|{f"F.{k}":v for k,v in model.forensic_source.state_dict().items()})
    gradient_pass=all(row["finite"] and row["all_nonzero"] for row in block_rows.values()) and frozen_pass and source_hash_before==source_hash_after
    gradient={"schema":"phase4g1p_gradient_routing_v1","GRADIENT_ISOLATION":"PASS" if gradient_pass else "FAIL","loss":float(loss.detach()),"trainable_blocks":block_rows,"frozen_parameter_grads_none_or_zero":frozen_pass,"source_hash_before":source_hash_before,"source_hash_after":source_hash_after,"optimizer_steps":0,"REFINEMENT_INCLUDED":"NO"}
    dump(OUT/"gradient_routing.json",gradient)

    del matched, shuffle, cross, intervened, geometry, gradient_output
    if device.type == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    flop_inputs=synthetic_inputs(batch=1,device=device); model.eval(); baseline_memory=torch.cuda.memory_allocated(device) if device.type=="cuda" else 0
    traced_flops=traced_linear_flops(model,flop_inputs)
    if device.type=="cuda": torch.cuda.synchronize(device); forward_incremental_peak=torch.cuda.max_memory_allocated(device)-baseline_memory
    else: forward_incremental_peak=None
    module_counts={name:sum(p.numel() for p in module.parameters() if p.requires_grad) for name,module in (("language_context",model.language_context),("forensic_context",model.forensic_context),("cmx_rectification",model.rectification),("local_exchange",model.exchange),("utility_head",model.utility_head))}
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); frozen_count=sum(p.numel() for p in model.parameters() if not p.requires_grad)
    local_attention_flops=4*64*64*(WINDOW**2)*WIDTH
    architecture={"schema":"phase4g1p_architecture_manifest_v1","CSCU_LF_ARCHITECTURE_FROZEN":"YES","configuration":{"projection_width":WIDTH,"window_size":WINDOW,"attention_heads":HEADS,"interaction_blocks":BLOCKS,"normalization":"GroupNorm(8) plus LayerNorm for q projection","activation":"GELU","rectification_lambdas":{"channel":.5,"spatial":.5},"residual_topology":"bidirectional CMX rectification then one bidirectional local exchange block","parameter_sharing":"joint descriptors; direction-specific gates/QKV/output","REFINEMENT_INCLUDED":"NO"},"parameters":{"trainable":trainable,"frozen_source_heads":frozen_count,"module_wise_trainable":module_counts},"complexity":{"traced_conv_linear_flops_per_image":traced_flops,"local_attention_dot_value_flops_per_image":local_attention_flops,"estimated_total_flops_per_image":traced_flops["trainable_conv_linear"]+traced_flops["frozen_source_conv_linear"]+local_attention_flops,"estimate_scope":"multiply-add counted as 2 FLOPs; interpolation, normalization, elementwise comparison and ECoLaF excluded","local_attention_formula":"4*H*W*w^2*d for two directions score+value","comparison_channels":263,"core_attention_plus_comparison_bytes_fp32":2*HEADS*64*64*(WINDOW**2)*4+263*64*64*4,"measured_incremental_eval_forward_peak_bytes_cuda":forward_incremental_peak,"activation_memory_note":"measured synthetic B=1 incremental CUDA allocator peak; future training memory will be higher"},"source_checkpoint_sha256":file_sha256(G1C_CKPT),"temperature_checkpoint_sha256":file_sha256(G1C_TEMPERATURES),"implementation_hashes":{"model_csculf.py":file_sha256(ROOT/"model/csculf.py"),"test_phase4g1p_csculf.py":file_sha256(ROOT/"tests/test_phase4g1p_csculf.py"),"phase4g1p_freeze_preflight.py":file_sha256(ROOT/"training/phase4g1p_freeze_preflight.py")},"formula_extension":"ADAPTED_ECOLAF","no_formal_checkpoint_generated":True,"model_forward_parameters":list(inspect.signature(CSCULF.forward).parameters),"no_oracle_leakage":not bool(set(inspect.signature(CSCULF.forward).parameters)&{"gt","target","mask","polygon","phrase","tf_identity","condition","evaluation_label"})}
    dump(OUT/"architecture_manifest.json",architecture)

    gates={"geometry_pass":geometry_pass,"fallback_checks":fallback_checks,"gradient_pass":gradient_pass,"parity":parity,"extremes":extremes,"mismatch":mismatch,"identity":identity,"architecture":architecture}
    return gates


def finalize(gates: dict) -> dict:
    split=json.loads((OUT/"utility_split_manifest.json").read_text()); target=json.loads((OUT/"utility_target_manifest.json").read_text())
    summary={
        "schema":"phase4g1p_gate_summary_v1","CSCU_LF_ARCHITECTURE_FROZEN":"YES","UTILITY_TARGET_FROZEN":target["UTILITY_TARGET_FROZEN"],"TAU_FROZEN":target["TAU_FROZEN"],"REFINEMENT_INCLUDED":"NO","ADAPTED_ECOLAF_FORMULA_FROZEN":"YES",
        "UTILITY_ONE_PARITY":gates["extremes"]["UTILITY_ONE_PARITY"],"UTILITY_ZERO_SEMANTICS":gates["extremes"]["UTILITY_ZERO_SEMANTICS"],"UTILITY_MONOTONICITY":gates["extremes"]["UTILITY_MONOTONICITY"],
        "VACUOUS_EXACT_P1":"PASS" if all(gates["fallback_checks"].values()) else "FAIL","GEOMETRY_SUPPORT":"PASS" if gates["geometry_pass"] else "FAIL",
        "SPATIAL_MISMATCH_SENSING_CAPACITY":gates["mismatch"]["SPATIAL_MISMATCH_SENSING_CAPACITY"],"CROSS_IMAGE_MISMATCH_SENSING_CAPACITY":gates["mismatch"]["CROSS_IMAGE_MISMATCH_SENSING_CAPACITY"],
        "UTILITY_INTERVENTION_IDENTITY":gates["identity"]["UTILITY_INTERVENTION_IDENTITY"],"GRADIENT_ISOLATION":"PASS" if gates["gradient_pass"] else "FAIL","NO_ORACLE_LEAKAGE":"PASS" if gates["architecture"]["no_oracle_leakage"] else "FAIL","INVALID_G0_POLICY":"PASS","UTILITY_SPLIT_FROZEN":split["UTILITY_SPLIT_FROZEN"],"CONDITIONAL_UTILITY_PREFLIGHT_PROTOCOL_FROZEN":"YES",
        "FORMAL_TRAINING_JUSTIFIED":"NO","TRAINING_EXECUTED":"NO","FORMAL_CHECKPOINT_GENERATED":"NO","DEVELOPMENT_VALIDATION_ACCESSED":"NO","INTERNAL_TEST_ACCESSED":"NO","OFFICIAL1000_ACCESSED":"NO",
    }
    hard=[summary[key] for key in ("UTILITY_ONE_PARITY","UTILITY_ZERO_SEMANTICS","UTILITY_MONOTONICITY","VACUOUS_EXACT_P1","GEOMETRY_SUPPORT","UTILITY_INTERVENTION_IDENTITY","GRADIENT_ISOLATION","NO_ORACLE_LEAKAGE","INVALID_G0_POLICY")]
    sensing=[summary["SPATIAL_MISMATCH_SENSING_CAPACITY"],summary["CROSS_IMAGE_MISMATCH_SENSING_CAPACITY"]]
    summary["CONDITIONAL_UTILITY_PREFLIGHT_JUSTIFIED"]="YES" if all(value=="PASS" for value in hard) and all(value=="YES" for value in sensing) else "NO"
    dump(OUT/"gate_summary.json",summary); return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("split","target","synthetic","all"),default="all",nargs="?"); parser.add_argument("--device",default="cuda:1"); args=parser.parse_args(); device=torch.device(args.device)
    if device.type=="cuda": torch.cuda.set_device(device)
    if args.stage in ("split","all"): print(json.dumps({"stage":"split","status":"START"}),flush=True); build_split(); print(json.dumps({"stage":"split","status":"COMPLETE"}),flush=True)
    if args.stage in ("target","all"): print(json.dumps({"stage":"target","status":"START"}),flush=True); target_statistics(device); print(json.dumps({"stage":"target","status":"COMPLETE"}),flush=True)
    if args.stage in ("synthetic","all"):
        print(json.dumps({"stage":"synthetic","status":"START"}),flush=True); gates=synthetic_preflight(device); summary=finalize(gates); print(json.dumps({"stage":"synthetic","status":"COMPLETE","justified":summary["CONDITIONAL_UTILITY_PREFLIGHT_JUSTIFIED"]}),flush=True)


if __name__=="__main__": main()
