#!/usr/bin/env python3
"""Phase 4H-D: paired control for unfreezing the existing Phase4F rectifier."""

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

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import phase4g1q_conditional_utility as q
from training import phase4hc_direct_utility_arms as hc
from training.phase4gf_formal_localization import full_inputs, load_dev
from training.phase4hb_progressive_unfreezing_stage1 import diagnostics, train_ids
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import compare, summarize, tensor_state_sha256
from tools.phase4f import (
    Phase4FStore,
    evidence_feature,
    invalid_record,
    load_evidence_source,
    load_sam_runtime,
    mask_loss,
)

PHASE = "phase4hd"
OUT = ROOT / "outputs" / PHASE
DOC = ROOT / "docs" / PHASE / "report.md"
CKPT_ROOT = Path("checkpoints/phase4hd_rectifier_unfreeze_control")
A2_SELECTED = ROOT / "outputs/phase4hc/a2/selected_checkpoint.pt"
A2_RESULTS = ROOT / "outputs/phase4hc/a2/dev_results.json"
CFG = yaml.safe_load((ROOT / "configs/phase4f_language_preserving_rectification.yaml").read_text())
SEED, EPOCHS, BATCH, LR, WD, GRAD_CLIP = 3407, 10, 8, 1e-4, 1e-4, 1.0
TAU, MARGIN = hc.TAU, hc.MARGIN


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def arm_paths(arm: str) -> dict[str, Path]:
    root = OUT / arm
    return {
        "root": root,
        "history": root / "training_history.csv",
        "selector": root / "selector.json",
        "results": root / "dev_results.json",
        "selected": root / "selected_checkpoint.pt",
        "manifest": CKPT_ROOT / arm / "execution_manifest.json",
        "ckpt": CKPT_ROOT / arm,
    }


def load_common(device: torch.device):
    hc.seed_all()
    model, _ = hc.load_utility("a2", device)
    state = torch.load(A2_SELECTED, map_location="cpu", weights_only=False)
    if state.get("arm") != "a2" or state.get("epoch") != 3 or state.get("optimizer_updates") != 3315:
        raise RuntimeError("Phase4H-C A2 selected checkpoint provenance drift")
    model.load_state_dict(state["model_state"], strict=True)
    model.language_source.eval().requires_grad_(False)
    model.forensic_source.eval().requires_grad_(False)
    rectifier, rectifier_path = hc.load_phase4f(device)
    return model, rectifier, rectifier_path


def config_contract() -> dict:
    return {
        "phase": "Phase4H-D",
        "common_start": {
            "checkpoint": str(A2_SELECTED),
            "sha256": file_sha256(A2_SELECTED),
            "arm": "Phase4H-C A2",
            "selected_epoch": 3,
        },
        "arms": {
            "r0": {"utility": "trainable", "phase4f_rectifier": "frozen"},
            "r1": {"utility": "trainable", "phase4f_rectifier": "trainable complete module"},
        },
        "only_difference": "Phase4F rectifier requires_grad and optimizer inclusion",
        "architecture": "S64_adapt=S64+U_F*(S64_rect_Phase4F-S64); no mapper/late fusion/refinement/new module",
        "loss": {
            "formula": "L_seg+L_relative+L_ranking",
            "weights": [1.0, 1.0, 1.0],
            "tau": TAU,
            "margin": MARGIN,
            "ranking_internal": {"cross": 0.5, "shuffle": 0.5},
            "segmentation": "existing Phase4F 2*BCEWithLogits+0.5*soft-Dice",
        },
        "optimizer": {
            "name": "AdamW", "lr": LR, "weight_decay": WD, "batch": BATCH,
            "epochs": EPOCHS, "grad_clip": GRAD_CLIP, "scheduler": "none",
            "early_stopping": False, "seed": SEED,
        },
        "population": {"train_fake": 8836, "valid": 8690, "invalid_accounting": 146, "real": 0,
                       "dev": 1106, "dev_invalid_iou_zero": 28},
        "order": "Python Random(seed+1009*epoch), canonical 8836 IDs",
        "cross": "cyclic next canonical ID; current geometry",
        "shuffle": "fixed randperm(576), CPU Generator seed3407",
        "selector": "each arm epochs1-10 DEV G0 mean IoU only; tie earlier; Phrase/TF/controls forbidden",
        "dominance_risk": "R1 G0 point gain over R0 and Phrase or TF paired bootstrap CI upper<0, or language order fails",
    }


def audit(device: torch.device) -> None:
    if OUT.exists() or CKPT_ROOT.exists():
        raise RuntimeError("Phase4H-D output already exists; refusing to overwrite")
    model, rectifier, rectifier_path = load_common(device)
    store = Phase4FStore(CFG, "val")
    source = load_evidence_source(CFG, "forensic_rect", device)
    sid = store.sample_ids[0]
    s64, raw, _, sam_coordinates, clip_coordinates, _ = store.batch([sid], device)
    with torch.no_grad():
        evidence = evidence_feature(source, raw)
        valid = torch.ones(1, 576, dtype=torch.bool, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            value = rectifier(s64, evidence, sam_coordinates, clip_coordinates, valid)
    tree = str(rectifier)
    parameter_count = sum(p.numel() for p in rectifier.parameters())
    utility_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "schema": "phase4hd_rectifier_audit_v1",
        "status": "PASS",
        "audit_only_no_structure_change": True,
        "module_tree": tree,
        "rectifier_parameter_count": parameter_count,
        "rectifier_parameter_tensors": sum(1 for _ in rectifier.parameters()),
        "utility_trainable_parameter_count": utility_count,
        "inputs": {
            "image_embeddings": list(s64.shape), "evidence": list(evidence.shape),
            "sam_coordinates": list(sam_coordinates.shape), "evidence_coordinates": list(clip_coordinates.shape),
            "evidence_valid": list(valid.shape),
        },
        "outputs": {key: list(tensor.shape) for key, tensor in value.items()},
        "common_initialization": {
            "a2_checkpoint": str(A2_SELECTED), "a2_checkpoint_sha256": file_sha256(A2_SELECTED),
            "utility_state_sha256": tensor_state_sha256(model.state_dict()),
            "rectifier_checkpoint": str(rectifier_path), "rectifier_checkpoint_sha256": file_sha256(rectifier_path),
            "rectifier_state_sha256": tensor_state_sha256(rectifier.state_dict()),
        },
        "implementation_sha256": file_sha256(Path(__file__)),
        "firewall": {"internal_test_accessed": False, "official1000_accessed": False},
    }
    dump(OUT / "rectifier_audit.json", result)
    print(tree)
    print(json.dumps({key: result[key] for key in ("rectifier_parameter_count", "inputs", "outputs")}, ensure_ascii=False, indent=2))


def freeze_manifest(arm: str, model, rectifier, rectifier_path: Path) -> dict:
    p = arm_paths(arm)
    if p["manifest"].exists():
        raise RuntimeError(f"{arm} manifest already exists")
    audit_data = json.loads((OUT / "rectifier_audit.json").read_text())
    contract = config_contract()
    common = {
        "utility": tensor_state_sha256(model.state_dict()),
        "rectifier": tensor_state_sha256(rectifier.state_dict()),
    }
    if common["utility"] != audit_data["common_initialization"]["utility_state_sha256"] or common["rectifier"] != audit_data["common_initialization"]["rectifier_state_sha256"]:
        raise RuntimeError("common initialization differs from pre-training audit")
    manifest = {
        "schema": "phase4hd_arm_manifest_v1", "arm": arm,
        "status": "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        "config": contract, "config_sha256": canonical_hash(contract),
        "implementation_sha256": file_sha256(Path(__file__)),
        "imported_phase4hc_sha256": file_sha256(ROOT / "training/phase4hc_direct_utility_arms.py"),
        "initialization": common,
        "trainable": {"utility": 371803, "rectifier": 0 if arm == "r0" else sum(x.numel() for x in rectifier.parameters())},
        "sources": {
            "a2_selected": {"path": str(A2_SELECTED), "sha256_before": file_sha256(A2_SELECTED)},
            "phase4f_rectifier": {"path": str(rectifier_path), "sha256_before": file_sha256(rectifier_path)},
            "sam": {"path": str(Path(CFG["experiment"]["runtime_root"]) / "p1_sam_runtime.pt"),
                    "sha256_before": file_sha256(Path(CFG["experiment"]["runtime_root"]) / "p1_sam_runtime.pt")},
            "forensic_adapter": {"path": CFG["evidence"]["forensic_checkpoint"],
                                 "sha256_before": file_sha256(Path(CFG["evidence"]["forensic_checkpoint"]))},
        },
        "formal_optimizer_updates": 0,
        "firewall": {"internal_test_accessed": False, "official1000_accessed": False},
    }
    dump(p["manifest"], manifest)
    return manifest


def phase4f_batch(store, ids, source, rectifier, device, *, train_rectifier: bool,
                  permutation=None, evidence_ids=None):
    s64, raw, targets, sam_coordinates, clip_coordinates, qseg = store.batch(ids, device)
    if evidence_ids is not None:
        _, raw, _, _, _, _ = store.batch(evidence_ids, device)
    with torch.no_grad():
        evidence = evidence_feature(source, raw)
        if permutation is not None:
            evidence = evidence.flatten(2)[:, :, permutation].reshape_as(evidence)
    valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
    grad_context = torch.enable_grad() if train_rectifier else torch.no_grad()
    with grad_context:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            value = rectifier(s64, evidence, sam_coordinates, clip_coordinates, valid)
    image_embeddings = value["image_embeddings"] if train_rectifier else value["image_embeddings"].detach()
    return s64, image_embeddings, value["support"].reshape(len(ids), 1, 64, 64), targets, sam_coordinates, qseg


def write_history(path: Path, rows: list[dict]) -> None:
    fields = ["epoch", "optimizer_updates", "seg_loss", "relative_loss", "ranking_loss", "cross_rank_loss",
              "shuffle_rank_loss", "total_loss", "mean_cross_delta", "mean_shuffle_delta", "traversal_exposures",
              "optimization_eligible_exposures", "invalid_g0_exposures", "dev_g0_mean_iou",
              "sample_order_sha256", "seconds"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, model, rectifier, optimizer, arm, epoch, updates, dev, manifest) -> None:
    payload = {
        "schema": "phase4hd_checkpoint_v1", "arm": arm, "epoch": epoch, "optimizer_updates": updates,
        "utility_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "rectifier_state": {k: v.detach().cpu() for k, v in rectifier.state_dict().items()},
        "optimizer": optimizer.state_dict(), "validation_g0": dev, "config_sha256": manifest["config_sha256"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(arm: str, device: torch.device) -> None:
    if arm not in ("r0", "r1"):
        raise ValueError(arm)
    p = arm_paths(arm)
    hc.seed_all()
    model, rectifier, rectifier_path = load_common(device)
    train_rectifier = arm == "r1"
    rectifier.requires_grad_(train_rectifier)
    manifest = freeze_manifest(arm, model, rectifier, rectifier_path)
    store = Phase4FStore(CFG, "train")
    val_store = Phase4FStore(CFG, "val")
    ids = train_ids()
    if ids != store.sample_ids:
        raise RuntimeError("train order drift")
    data = q.load_ids(ids, ("valid_g0", "S64", "q_seg", "z_L", "F24", "z_F24", "target64", "clip_geometries"))
    valid_mask = data["valid_g0"].bool()
    id_to_i = {sid: i for i, sid in enumerate(ids)}
    sam = load_sam_runtime(CFG, device)
    source = load_evidence_source(CFG, "forensic_rect", device)
    frozen_before = {
        "sam": tensor_state_sha256(sam.state_dict()), "source": tensor_state_sha256(source.state_dict()),
        "heads": tensor_state_sha256(q.source_state(model)),
        "rectifier": tensor_state_sha256(rectifier.state_dict()),
    }
    params = [x for x in model.parameters() if x.requires_grad] + [x for x in rectifier.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    dev = load_dev("g0")
    rows, updates = [], 0
    cross = torch.tensor([(i + 1) % len(ids) for i in range(len(ids))])
    permutation = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    for epoch in range(1, EPOCHS + 1):
        began = time.time()
        order = list(ids)
        random.Random(SEED + 1009 * epoch).shuffle(order)
        sums = defaultdict(float)
        traversal = eligible = invalid = 0
        model.train(); model.language_source.eval(); model.forensic_source.eval()
        rectifier.train(mode=train_rectifier)
        for begin in range(0, len(order), BATCH):
            all_batch = order[begin:begin + BATCH]
            positions = torch.tensor([id_to_i[sid] for sid in all_batch])
            idx = positions[valid_mask.index_select(0, positions)]
            traversal += len(all_batch); eligible += len(idx); invalid += len(all_batch) - len(idx)
            if not len(idx):
                continue
            batch = full_inputs(data, idx, device)
            forensic_idx = cross.index_select(0, idx)
            crossed = dict(batch)
            crossed["F24"] = data["F24"].index_select(0, forensic_idx).to(device)
            crossed["z_F24"] = data["z_F24"].index_select(0, forensic_idx).to(device)
            batch_ids = [ids[i] for i in idx.tolist()]
            s64, phase4f, support, targets, sam_coordinates, qseg = phase4f_batch(
                store, batch_ids, source, rectifier, device, train_rectifier=train_rectifier)
            optimizer.zero_grad(set_to_none=True)
            matched = hc.utility_forward(model, batch)
            cross_out = hc.utility_forward(model, crossed)
            shuffle_out = hc.utility_forward(model, batch, permutation=permutation)
            gate = hc.gate_to_sam_grid(matched["U"], sam_coordinates) * support.float()
            adapted = hc.gated_embedding(s64, phase4f, gate)
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(qseg.to(torch.bfloat16), adapted.to(torch.bfloat16))
            seg = mask_loss(low, targets, CFG)
            _, soft = q.target_delta({"p_L": matched["p_L"], "p_F": matched["p_F"]}, data["target64"].index_select(0, idx).to(device))
            relative = q.image_balanced_loss(matched["utility_logit"], soft, matched["support"])
            cross_rank = hc.rank_loss(matched["U"], cross_out["U"], matched["support"])
            shuffle_rank = hc.rank_loss(matched["U"], shuffle_out["U"], matched["support"])
            ranking = 0.5 * (cross_rank + shuffle_rank)
            total = seg["total"] + relative + ranking
            if not torch.isfinite(total):
                raise RuntimeError("nonfinite loss")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite gradient")
            optimizer.step(); updates += 1
            n = len(idx)
            for key, value in (("seg", seg["total"]), ("relative", relative), ("ranking", ranking),
                               ("cr", cross_rank), ("sr", shuffle_rank), ("total", total)):
                sums[key] += float(value.detach()) * n
            den = matched["support"].flatten(1).sum(1)
            um = (matched["U"] * matched["support"]).flatten(1).sum(1) / den
            uc = (cross_out["U"] * matched["support"]).flatten(1).sum(1) / den
            us = (shuffle_out["U"] * matched["support"]).flatten(1).sum(1) / den
            sums["cross_delta"] += float((uc - um).sum())
            sums["shuffle_delta"] += float((us - um).sum())
            if updates <= 3 or updates % 100 == 0:
                print(json.dumps({"stage": "TRAIN", "arm": arm, "epoch": epoch, "update": updates,
                                  "loss": float(total.detach()), "grad_norm": float(norm)}), flush=True)
        if (traversal, eligible, invalid) != (8836, 8690, 146):
            raise RuntimeError("exposure mismatch")
        rectifier.eval()
        dev_metric, _, _, _ = hc.evaluate(model, val_store, sam, source, rectifier, dev, "g0", "matched", device, collect=False)
        row = {
            "epoch": epoch, "optimizer_updates": updates,
            "seg_loss": sums["seg"] / eligible, "relative_loss": sums["relative"] / eligible,
            "ranking_loss": sums["ranking"] / eligible, "cross_rank_loss": sums["cr"] / eligible,
            "shuffle_rank_loss": sums["sr"] / eligible, "total_loss": sums["total"] / eligible,
            "mean_cross_delta": sums["cross_delta"] / eligible, "mean_shuffle_delta": sums["shuffle_delta"] / eligible,
            "traversal_exposures": traversal, "optimization_eligible_exposures": eligible,
            "invalid_g0_exposures": invalid, "dev_g0_mean_iou": dev_metric["mean_foreground_iou"],
            "sample_order_sha256": ids_hash(order), "seconds": time.time() - began,
        }
        rows.append(row)
        write_history(p["history"], rows)
        save_checkpoint(p["ckpt"] / f"epoch_{epoch}.pt", model, rectifier, optimizer, arm, epoch, updates, dev_metric, manifest)
        print(json.dumps({"stage": "EPOCH_COMPLETE", "arm": arm, **row}), flush=True)
    best = max(rows, key=lambda x: (x["dev_g0_mean_iou"], -x["epoch"]))
    chosen = p["ckpt"] / f"epoch_{best['epoch']}.pt"
    p["root"].mkdir(parents=True, exist_ok=True)
    shutil.copy2(chosen, p["selected"])
    selector = {
        "schema": "phase4hd_selector_v1", "arm": arm, "status": "COMPLETE",
        "primary": "DEV G0 mean IoU", "population": 1106, "invalid_iou_zero": 28,
        "tie_break": "earlier epoch", "selected_epoch": best["epoch"],
        "selected_dev_g0": best["dev_g0_mean_iou"],
        "selected_checkpoint_sha256": file_sha256(chosen),
        "output_selected_checkpoint_sha256": file_sha256(p["selected"]),
        "candidates": [{"epoch": row["epoch"], "mean_g0_iou": row["dev_g0_mean_iou"]} for row in rows],
        "phrase_used": False, "tf_used": False, "controls_used": False,
        "sample_order_hashes": [row["sample_order_sha256"] for row in rows],
    }
    dump(p["selector"], selector)
    frozen_after = {
        "sam": tensor_state_sha256(sam.state_dict()), "source": tensor_state_sha256(source.state_dict()),
        "heads": tensor_state_sha256(q.source_state(model)),
        "rectifier": tensor_state_sha256(rectifier.state_dict()),
    }
    if any(frozen_before[k] != frozen_after[k] for k in ("sam", "source", "heads")):
        raise RuntimeError("frozen source mutation")
    if arm == "r0" and frozen_before["rectifier"] != frozen_after["rectifier"]:
        raise RuntimeError("R0 frozen rectifier mutation")
    manifest.update({
        "status": "TRAINING_COMPLETE_SELECTOR_FROZEN", "formal_optimizer_updates": updates,
        "frozen_hash_before": frozen_before, "frozen_hash_after": frozen_after,
        "rectifier_changed": frozen_before["rectifier"] != frozen_after["rectifier"],
        "selector_sha256": file_sha256(p["selector"]), "selected_sha256": file_sha256(p["selected"]),
    })
    dump(p["manifest"], manifest)


def evaluate_selected(arm: str, device: torch.device) -> None:
    p = arm_paths(arm)
    checkpoint = torch.load(p["selected"], map_location="cpu", weights_only=False)
    model, rectifier, _ = load_common(device)
    model.load_state_dict(checkpoint["utility_state"], strict=True)
    rectifier.load_state_dict(checkpoint["rectifier_state"], strict=True)
    model.eval(); rectifier.eval()
    store = Phase4FStore(CFG, "val")
    sam = load_sam_runtime(CFG, device)
    source = load_evidence_source(CFG, "forensic_rect", device)
    dev = {mode: load_dev(mode) for mode in ("g0", "phrase", "tf")}
    metrics, records, diags, identities = {}, {}, {}, {}
    for mode, condition in (("g0", "matched"), ("phrase", "matched"), ("tf", "matched"),
                            ("g0", "cross_image"), ("g0", "spatial_shuffle"), ("g0", "forensic_off")):
        key = {"phrase": "phrase", "tf": "tf"}.get(mode, condition)
        metrics[key], records[key], diags[key], identities[key] = hc.evaluate(
            model, store, sam, source, rectifier, dev[mode], mode, condition, device)
    a2 = json.loads(A2_RESULTS.read_text())
    baseline = {"g0": a2["records"]["matched"], "phrase": a2["records"]["phrase"], "tf": a2["records"]["tf"]}
    stats = {mode: compare(records[key], baseline[mode], seed=SEED)
             for mode, key in (("g0", "matched"), ("phrase", "phrase"), ("tf", "tf"))}
    controls = {
        "matched_minus_cross": compare(records["matched"], records["cross_image"], seed=SEED),
        "matched_minus_shuffle": compare(records["matched"], records["spatial_shuffle"], seed=SEED),
    }
    manifest = json.loads(p["manifest"].read_text())
    result = {
        "schema": "phase4hd_arm_results_v1", "arm": arm, "status": "COMPLETE",
        "selector": json.loads(p["selector"].read_text()), "metrics": metrics, "records": records,
        "statistics_vs_A2": stats, "control_statistics": controls,
        "utility_diagnostics": diags, "identities": identities,
        "source_integrity": {
            "sam": manifest["frozen_hash_before"]["sam"] == manifest["frozen_hash_after"]["sam"],
            "forensic_source": manifest["frozen_hash_before"]["source"] == manifest["frozen_hash_after"]["source"],
            "source_heads": manifest["frozen_hash_before"]["heads"] == manifest["frozen_hash_after"]["heads"],
            "r0_rectifier": None if arm == "r1" else manifest["frozen_hash_before"]["rectifier"] == manifest["frozen_hash_after"]["rectifier"],
        },
        "rectifier_changed": manifest["rectifier_changed"],
        "firewall": {"internal_test_accessed": False, "official1000_accessed": False},
    }
    dump(p["results"], result)
    manifest.update({"status": "COMPLETE", "results_sha256": file_sha256(p["results"])})
    dump(p["manifest"], manifest)


def fg(stat: dict) -> dict:
    return stat["foreground_iou"]


def language_order(result: dict) -> bool:
    m = result["metrics"]
    return m["matched"]["mean_foreground_iou"] < m["phrase"]["mean_foreground_iou"] < m["tf"]["mean_foreground_iou"]


def controls_pass(result: dict) -> bool:
    return (fg(result["control_statistics"]["matched_minus_cross"])["bootstrap_95_ci"][0] > 0 and
            fg(result["control_statistics"]["matched_minus_shuffle"])["bootstrap_95_ci"][0] > 0)


def gate_collapse(result: dict) -> bool:
    diag = result["utility_diagnostics"]["matched"]
    return (diag["low_saturation_fraction_U_le_0.01"] >= 0.95 or
            diag["high_saturation_fraction_U_ge_0.99"] >= 0.95 or diag["U_F"]["std"] < 0.01)


def compare_arms() -> None:
    r0 = json.loads(arm_paths("r0")["results"].read_text())
    r1 = json.loads(arm_paths("r1")["results"].read_text())
    paired = {}
    for mode, key in (("g0", "matched"), ("phrase", "phrase"), ("tf", "tf")):
        paired[mode] = compare(r1["records"][key], r0["records"][key], seed=SEED)
    order_equal = r0["selector"]["sample_order_hashes"] == r1["selector"]["sample_order_hashes"]
    metrics = {}
    a2 = json.loads(A2_RESULTS.read_text())
    metrics["A2"] = {k: a2["metrics"][{"g0": "matched"}.get(k, k)]["mean_foreground_iou"] for k in ("g0", "phrase", "tf")}
    for name, result in (("R0", r0), ("R1", r1)):
        metrics[name] = {k: result["metrics"][{"g0": "matched"}.get(k, k)]["mean_foreground_iou"] for k in ("g0", "phrase", "tf")}
    g0 = fg(paired["g0"]); phrase = fg(paired["phrase"]); tf = fg(paired["tf"])
    r1_significant = g0["bootstrap_95_ci"][0] > 0
    dominance = (g0["mean_difference"] > 0 and
                 (phrase["bootstrap_95_ci"][1] < 0 or tf["bootstrap_95_ci"][1] < 0 or not language_order(r1)))
    gates = {
        "schema": "phase4hd_gate_summary_v1", "R0_COMPLETE": "YES", "R1_COMPLETE": "YES",
        "IDENTICAL_DATA_ORDER": "PASS" if order_equal else "FAIL",
        "R0_LANGUAGE_ORDER_PRESERVED": "YES" if language_order(r0) else "NO",
        "R1_LANGUAGE_ORDER_PRESERVED": "YES" if language_order(r1) else "NO",
        "R0_MATCHED_GT_CROSS_SHUFFLE": "YES" if controls_pass(r0) else "NO",
        "R1_MATCHED_GT_CROSS_SHUFFLE": "YES" if controls_pass(r1) else "NO",
        "R0_GATE_COLLAPSE": "YES" if gate_collapse(r0) else "NO",
        "R1_GATE_COLLAPSE": "YES" if gate_collapse(r1) else "NO",
        "R0_SOURCE_INTEGRITY": "PASS" if all(v for v in r0["source_integrity"].values() if v is not None) else "FAIL",
        "R1_SOURCE_INTEGRITY": "PASS" if all(v for v in r1["source_integrity"].values() if v is not None) else "FAIL",
        "R0_RECTIFIER_FROZEN_INTEGRITY": "PASS" if not r0["rectifier_changed"] else "FAIL",
        "R1_RECTIFIER_CHANGED": "YES" if r1["rectifier_changed"] else "NO",
        "RECTIFIER_UNFREEZE_BENEFICIAL": "YES" if r1_significant else "NOT_SUPPORTED",
        "FORENSIC_DOMINANCE_RISK": "YES" if dominance else "NO",
        "INTERNAL_TEST_ACCESSED": "NO", "OFFICIAL1000_ACCESSED": "NO",
    }
    comparison = {
        "schema": "phase4hd_comparison_v1", "status": "COMPLETE", "metrics": metrics,
        "paired_R1_minus_R0": paired,
        "R0_vs_A2": r0["statistics_vs_A2"], "R1_vs_A2": r1["statistics_vs_A2"],
        "identical_data_order": order_equal,
        "firewall": {"internal_test_accessed": False, "official1000_accessed": False},
    }
    dump(OUT / "comparison.json", comparison)
    dump(OUT / "gate_summary.json", gates)
    write_report(comparison, r0, r1, gates)


def stat_line(stat: dict) -> str:
    v = fg(stat)
    return (f"delta={v['mean_difference']:+.6f}, CI=[{v['bootstrap_95_ci'][0]:+.6f},"
            f"{v['bootstrap_95_ci'][1]:+.6f}], W/T/L={v['wins']}/{v['ties']}/{v['losses']}, "
            f"Wilcoxon p={v['wilcoxon_pvalue']:.4g}")


def write_report(comp: dict, r0: dict, r1: dict, gates: dict) -> None:
    audit_data = json.loads((OUT / "rectifier_audit.json").read_text())
    rows = ["| Arm | Selected epoch | G0 | Phrase | TF |", "|---|---:|---:|---:|---:|"]
    for arm, result in (("R0", r0), ("R1", r1)):
        m = comp["metrics"][arm]
        rows.append(f"| {arm} | {result['selector']['selected_epoch']} | {m['g0']:.6f} | {m['phrase']:.6f} | {m['tf']:.6f} |")
    m = comp["metrics"]["A2"]
    rows.insert(2, f"| A2 start | 3 | {m['g0']:.6f} | {m['phrase']:.6f} | {m['tf']:.6f} |")
    report = f"""# Phase 4H-D Rectifier Unfreeze Paired Control

## Pre-training rectifier audit

`{audit_data['module_tree']}`

- Rectifier parameters: {audit_data['rectifier_parameter_count']}
- Inputs: `{json.dumps(audit_data['inputs'], ensure_ascii=False)}`
- Outputs: `{json.dumps(audit_data['outputs'], ensure_ascii=False)}`

## Frozen protocol

共同起点为Phase4H-C A2 selected epoch3。R0训练CSCU utility并冻结原Phase4F rectifier；R1训练相同CSCU utility及完整原Phase4F rectifier。两臂数据顺序、seed、loss、optimizer、LR、epoch、corruption与DEV G0 selector完全相同，唯一差异为rectifier是否进入optimizer。Phrase/TF及controls未参与selector。

## Results

{chr(10).join(rows)}

- R1 vs R0 G0: {stat_line(comp['paired_R1_minus_R0']['g0'])}
- R1 vs R0 Phrase: {stat_line(comp['paired_R1_minus_R0']['phrase'])}
- R1 vs R0 TF: {stat_line(comp['paired_R1_minus_R0']['tf'])}
- R0 vs A2 G0: {stat_line(comp['R0_vs_A2']['g0'])}
- R1 vs A2 G0: {stat_line(comp['R1_vs_A2']['g0'])}

## Gates

```json
{json.dumps(gates, ensure_ascii=False, indent=2)}
```

结论只依据R1对R0的paired comparison归因rectifier解冻收益。到此严格STOP，未访问internal test或official1000。
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "train", "evaluate", "compare"))
    parser.add_argument("--arm", choices=("r0", "r1"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.stage == "compare":
        compare_arms(); return
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.stage == "audit":
        audit(device); return
    if not args.arm:
        raise ValueError("--arm required")
    if args.stage == "train":
        train(args.arm, device)
    else:
        evaluate_selected(args.arm, device)


if __name__ == "__main__":
    main()
