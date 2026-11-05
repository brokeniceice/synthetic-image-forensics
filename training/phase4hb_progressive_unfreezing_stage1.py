#!/usr/bin/env python3
"""Phase 4H-B Stage 1: train only a bounded U_F-to-rectification mapper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import phase4g1q_conditional_utility as utility_code
from training.phase4gf_formal_localization import full_inputs, load_dev
from training.phase4ha_utility_gated_rectification import (
    HISTORICAL as HA_HISTORICAL,
    baseline_records,
    gate_to_sam_grid,
    gated_embedding,
    load_phase4f,
    load_utility,
    phase4f_records,
)
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import compare, summarize, tensor_state_sha256
from tools.phase4f import Phase4FStore, evidence_feature, invalid_record, load_evidence_source, load_sam_runtime, mask_loss

PHASE = "phase4hb"
OUT = ROOT / "outputs" / PHASE
DOC = ROOT / "docs" / PHASE / "report.md"
HISTORY = OUT / "training_history.csv"
DEV_RESULTS = OUT / "dev_results.json"
SELECTOR = OUT / "selector.json"
GATES = OUT / "gate_summary.json"
SELECTED = OUT / "selected_checkpoint.pt"
CKPT = Path("checkpoints/phase4hb_progressive_unfreezing_stage1")
MANIFEST = CKPT / "execution_manifest.json"
TRAIN_CACHE = Path("cache/phase4g1/g1c/frozen_sources")
CFG = yaml.safe_load((ROOT / "configs/phase4f_language_preserving_rectification.yaml").read_text())
HA_RESULTS = ROOT / "outputs/phase4ha/results.json"
SEED, EPOCHS, BATCH = 3407, 10, 8
LR, WEIGHT_DECAY, GRAD_CLIP = 1e-4, 1e-4, 1.0
LAMBDA_IDENTITY = 0.1
NONDEGRADATION_MARGIN = -0.010
COLLAPSE_FRACTION = 0.95
COLLAPSE_STD = 0.01
LOGIT_EPS = 1e-6


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seed_all() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class BoundedIdentityMapper(nn.Module):
    """Two-parameter logit-residual 1x1 mapper, exactly zero-residual at init."""

    def __init__(self) -> None:
        super().__init__()
        self.residual = nn.Conv2d(1, 1, 1, bias=True)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, utility: torch.Tensor, support: torch.Tensor | None = None) -> torch.Tensor:
        base = utility.float().clamp(LOGIT_EPS, 1.0 - LOGIT_EPS)
        gate = torch.sigmoid(torch.logit(base) + self.residual(utility.float()))
        if support is not None:
            gate = gate * support.float()
        return gate


def freeze_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    ha_gates = json.loads((ROOT / "outputs/phase4ha/gate_summary.json").read_text())
    required = {"P1_ENDPOINT_RECOVERY": "PASS", "PHASE4F_ENDPOINT_RECOVERY": "PASS", "PARETO_TRADEOFF_IMPROVEMENT": "YES"}
    if any(ha_gates.get(key) != value for key, value in required.items()):
        raise RuntimeError("Phase4H-A prerequisite drift")
    config = {
        "phase": "Phase4H-B Stage1", "mapper": "sigmoid(logit(clamp(U_F,1e-6,1-1e-6))+Conv1x1(U_F))",
        "mapper_inputs": ["U_F"], "mapper_parameters": 2, "mapper_initialization": "Conv1x1 weight=0,bias=0",
        "trainable": ["phi.residual.weight", "phi.residual.bias"],
        "frozen": ["P1_LLM_LoRA", "text_hidden_fcs", "SAM", "CLIP", "Phase4C-A", "forensic_dense_head", "Phase4F_rectifier", "Phase4G-1S_U_F"],
        "loss": {"formula": "L_seg+lambda_identity*L1(phi(U_F),U_F)", "lambda_identity": LAMBDA_IDENTITY,
                 "segmentation_contract": "Phase4F 2*BCEWithLogits+0.5*soft-Dice"},
        "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "batch": BATCH,
                      "grad_clip": GRAD_CLIP, "epochs": EPOCHS, "scheduler": "none", "early_stopping": False, "seed": SEED},
        "population": {"train_fake": 8836, "valid_optimizer": 8690, "invalid_accounting": 146, "real_optimizer": 0,
                       "dev": 1106, "dev_valid": 1078, "dev_invalid_iou_zero": 28},
        "selector": "epochs1-10 DEV G0 mean all-ref-union IoU only; tie earlier; epoch0 diagnostic not candidate",
        "nondegradation": {"margin": NONDEGRADATION_MARGIN, "rule": "paired bootstrap CI lower >= margin"},
        "gate_collapse": {"support_only": True, "low": "g<=0.01", "high": "g>=0.99",
                          "collapse": "low_fraction>=0.95 or high_fraction>=0.95 or std(g)<0.01"},
        "next_stage": "G0 vs Phase4H-A CI lower>0, Phrase/TF nondegraded, language order, controls, endpoints, no collapse",
    }
    phase4f_selector = json.loads((ROOT / "outputs/phase4f_language_preserving_rectification/selectors/forensic_rect.json").read_text())
    manifest = {
        "schema": "phase4hb_execution_manifest_v1", "status": "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        "config": config, "config_sha256": canonical_hash(config), "implementation_sha256": file_sha256(Path(__file__)),
        "source_files": {
            "phase4f_rectifier": {"path": phase4f_selector["selected_checkpoint"], "sha256_before": phase4f_selector["selected_checkpoint_sha256"]},
            "phase4g1s_utility": {"path": str(Path("checkpoints/phase4g1s_mismatch_aware_utility/csculf_mismatch_utility_epoch10.pt")), "sha256_before": file_sha256(Path("checkpoints/phase4g1s_mismatch_aware_utility/csculf_mismatch_utility_epoch10.pt"))},
            "sam_runtime": {"path": str(Path(CFG["experiment"]["runtime_root"]) / "p1_sam_runtime.pt"), "sha256_before": file_sha256(Path(CFG["experiment"]["runtime_root"]) / "p1_sam_runtime.pt")},
            "forensic_adapter": {"path": CFG["evidence"]["forensic_checkpoint"], "sha256_before": file_sha256(Path(CFG["evidence"]["forensic_checkpoint"]))},
        },
        "formal_optimizer_updates": 0, "firewall": {"internal_test_accessed": False, "official1000_accessed": False},
    }
    dump(MANIFEST, manifest)
    return manifest


def train_ids() -> list[str]:
    ids = []
    for path in sorted(TRAIN_CACHE.glob("shard_*.pt")):
        ids.extend(str(value) for value in torch.load(path, map_location="cpu", weights_only=False)["sample_ids"])
    if len(ids) != 8836 or len(set(ids)) != 8836:
        raise RuntimeError("formal train population drift")
    return ids


def make_phase4f_batch(store, ids, source, rectifier, device, *, permutation=None, evidence_ids=None):
    s64, raw, targets, sam_coordinates, clip_coordinates, qseg = store.batch(ids, device)
    if evidence_ids is not None:
        _, raw, _, _, _, _ = store.batch(evidence_ids, device)
    with torch.no_grad():
        evidence = evidence_feature(source, raw)
        if permutation is not None:
            evidence = evidence.flatten(2)[:, :, permutation].reshape_as(evidence)
        valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            value = rectifier(s64, evidence, sam_coordinates, clip_coordinates, valid)
    return s64, value["image_embeddings"].detach(), value["support"].reshape(len(ids), 1, 64, 64), targets, sam_coordinates, qseg


def frozen_utility(utility_model, batch, sam_coordinates, device, *, permutation=None):
    if permutation is not None:
        batch["F24"] = batch["F24"].flatten(2)[:, :, permutation].reshape_as(batch["F24"])
        batch["z_F24"] = batch["z_F24"].flatten(2)[:, :, permutation].reshape_as(batch["z_F24"])
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        utility = utility_code.utility_forward(utility_model, batch)["U"]
    return gate_to_sam_grid(utility, sam_coordinates).detach()


def endpoint_audit(store, sam, rectifier, source, dev, device) -> dict:
    """Local hard-gate audit; never mutates Phase4H-A artifacts."""
    p1_embedding = p4f_embedding = p1_decoder = p4f_decoder = True
    checked = decoder_checked = 0
    with torch.no_grad():
        for index, sid in enumerate(dev["sample_ids"]):
            if not bool(dev["valid"][index]):
                continue
            s64, raw, _, sc, cc, qseg = store.batch([sid], device)
            evidence = evidence_feature(source, raw)
            valid = torch.ones(1, 576, dtype=torch.bool, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                phase4f = rectifier(s64, evidence, sc, cc, valid)["image_embeddings"]
            zero = torch.zeros(1, 1, 64, 64, device=device); one = torch.ones_like(zero)
            endpoint0 = gated_embedding(s64, phase4f, zero); endpoint1 = gated_embedding(s64, phase4f, one)
            p1_embedding &= torch.equal(endpoint0, s64); p4f_embedding &= torch.equal(endpoint1, phase4f); checked += 1
            if decoder_checked < 16:
                with torch.autocast(device_type=device.type, enabled=False):
                    p1_decoder &= torch.equal(sam(qseg.to(torch.bfloat16), s64), sam(qseg.to(torch.bfloat16), endpoint0))
                    p4f_decoder &= torch.equal(sam(qseg.to(torch.bfloat16), phase4f.to(torch.bfloat16)), sam(qseg.to(torch.bfloat16), endpoint1.to(torch.bfloat16)))
                decoder_checked += 1
    result = {"valid_g0_embeddings_checked": checked, "decoder_subset_checked": decoder_checked,
              "P1": {"embedding_all_exact": p1_embedding, "decoder_subset_exact": p1_decoder, "pass": p1_embedding and p1_decoder},
              "Phase4F": {"embedding_all_exact": p4f_embedding, "decoder_subset_exact": p4f_decoder, "pass": p4f_embedding and p4f_decoder}}
    if not result["P1"]["pass"] or not result["Phase4F"]["pass"]:
        dump(GATES, {"STAGE1_TRAINING_COMPLETE": "NO", "P1_ENDPOINT_RECOVERY": "PASS" if result["P1"]["pass"] else "FAIL",
                     "PHASE4F_ENDPOINT_RECOVERY": "PASS" if result["Phase4F"]["pass"] else "FAIL",
                     "INTERNAL_TEST_ACCESSED": "NO", "OFFICIAL1000_ACCESSED": "NO"})
        raise RuntimeError("Phase4H-B hard endpoint failure")
    return result


def decode_loss(sam, qseg, adapted, targets):
    with torch.autocast(device_type=adapted.device.type, enabled=False):
        low = sam(qseg.to(torch.bfloat16), adapted.to(torch.bfloat16))
    return low, mask_loss(low, targets, CFG)


def diagnostics(u_values: list[torch.Tensor], g_values: list[torch.Tensor]) -> dict:
    u = torch.cat(u_values).float().numpy() if u_values else np.asarray([], dtype=np.float32)
    g = torch.cat(g_values).float().numpy() if g_values else np.asarray([], dtype=np.float32)
    def describe(value):
        percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        return {"n": int(value.size), "mean": float(value.mean()), "std": float(value.std()),
                "percentiles": {str(p): float(np.percentile(value, p)) for p in percentiles}}
    return {
        "scope": "Phase4F semantic support only", "U_F": describe(u), "g": describe(g),
        "mean_abs_g_minus_U_F": float(np.abs(g - u).mean()),
        "low_saturation_fraction_g_le_0.01": float((g <= 0.01).mean()),
        "high_saturation_fraction_g_ge_0.99": float((g >= 0.99).mean()),
    }


def evaluate(mapper, store, sam, source, rectifier, utility_model, dev, mode, condition, device, *, collect_records=True):
    ids = dev["sample_ids"]
    cross = [(index + 1) % len(ids) for index in range(len(ids))]
    permutation = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    records, u_values, g_values, off_exact = [], [], [], []
    mapper.eval()
    with torch.no_grad():
        for index, sid in enumerate(ids):
            if not bool(dev["valid"][index]):
                row = invalid_record(sid, dev["original_masks"][index]); row["valid_g0"] = False; records.append(row)
                continue
            evidence_index = cross[index] if condition == "cross_image" else index
            batch = full_inputs(dev, torch.tensor([index]), device)
            if condition == "cross_image":
                batch["F24"] = dev["F24"][evidence_index:evidence_index + 1].to(device)
                batch["z_F24"] = dev["z_F24"][evidence_index:evidence_index + 1].to(device)
            evidence_ids = [ids[evidence_index]] if evidence_index != index else None
            perm = permutation if condition == "spatial_shuffle" else None
            s64, phase4f, support, _, sc, qseg = make_phase4f_batch(store, [sid], source, rectifier, device, permutation=perm, evidence_ids=evidence_ids)
            # Phase4FStore(val) owns canonical G0 q_SEG only.  Phrase/TF must
            # decode with the mode-specific frozen q_SEG from the DEV cache.
            qseg = dev["q_seg"][index:index + 1].to(device=device, dtype=torch.bfloat16)
            u = frozen_utility(utility_model, batch, sc, device, permutation=perm)
            g = mapper(u, support)
            if condition == "forensic_off":
                g = torch.zeros_like(g)
            adapted = gated_embedding(s64, phase4f, g)
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(qseg.to(torch.bfloat16), adapted.to(torch.bfloat16))
                if condition == "forensic_off":
                    off_exact.append(torch.equal(low, sam(qseg.to(torch.bfloat16), s64)))
            logits = inverse_sam_logits(low, dev["sam_geometries"][index])
            row = metric_record(sid, logits, dev["original_masks"][index]); row["valid_g0"] = True; records.append(row)
            active = support.bool()
            u_values.append(u[active].cpu()); g_values.append(g[active].cpu())
            if (index + 1) % 200 == 0:
                print(json.dumps({"stage": "DEV_EVAL", "mode": mode, "condition": condition, "done": index + 1, "total": len(ids)}), flush=True)
    return summarize(records), records if collect_records else None, diagnostics(u_values, g_values), {"checked": len(off_exact), "all_exact": all(off_exact) if off_exact else None}


def write_history(rows: list[dict]) -> None:
    fields = ["epoch", "optimizer_updates", "seg_loss", "identity_loss", "total_loss", "traversal_exposures",
              "optimization_eligible_exposures", "invalid_g0_exposures", "dev_g0_mean_iou", "dev_phrase_mean_iou",
              "dev_tf_mean_iou", "mapper_weight", "mapper_bias", "seconds"]
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def save_checkpoint(path, mapper, optimizer, epoch, updates, dev_g0, config_hash):
    payload = {"schema": "phase4hb_stage1_checkpoint_v1", "epoch": epoch, "optimizer_updates": updates,
               "mapper_architecture": "BoundedIdentityMapper(logit residual Conv1x1 1->1)", "mapper_parameters": 2,
               "mapper_state": {key: value.detach().cpu() for key, value in mapper.state_dict().items()},
               "optimizer": optimizer.state_dict(), "dev_g0": dev_g0, "config_sha256": config_hash}
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".pt.tmp"); torch.save(payload, temporary); temporary.replace(path)


def train(device):
    manifest = freeze_manifest()
    if manifest["status"] != "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP" or manifest["formal_optimizer_updates"] != 0:
        raise RuntimeError("automatic rerun/recovery forbidden for non-frozen Phase4H-B state")
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    seed_all(); store = Phase4FStore(CFG, "train"); val_store = Phase4FStore(CFG, "val")
    ids = train_ids()
    if ids != store.sample_ids:
        raise RuntimeError("Phase4F/G1-C train order drift")
    data = utility_code.load_ids(ids, ("valid_g0", "S64", "q_seg", "z_L", "F24", "z_F24", "clip_geometries"))
    valid = data["valid_g0"].bool(); id_to_index = {sid: index for index, sid in enumerate(ids)}
    sam = load_sam_runtime(CFG, device); source = load_evidence_source(CFG, "forensic_rect", device)
    rectifier, _ = load_phase4f(device); utility_model = load_utility(device)
    frozen_models = (sam, source, rectifier, utility_model)
    frozen_hash_before = {name: tensor_state_sha256(model.state_dict()) for name, model in zip(("sam", "source", "rectifier", "utility"), frozen_models)}
    mapper = BoundedIdentityMapper().to(device)
    if sum(parameter.numel() for parameter in mapper.parameters()) != 2:
        raise RuntimeError("mapper architecture drift")
    if any(parameter.requires_grad for model in frozen_models for parameter in model.parameters()):
        raise RuntimeError("non-mapper parameter unexpectedly trainable")
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    dev = {mode: load_dev(mode) for mode in ("g0", "phrase", "tf")}
    endpoint = endpoint_audit(val_store, sam, rectifier, source, dev["g0"], device)
    epoch0 = {}
    for mode in ("g0", "phrase", "tf"):
        epoch0[mode] = evaluate(mapper, val_store, sam, source, rectifier, utility_model, dev[mode], mode, "matched", device, collect_records=False)[0]
    ha = json.loads(HA_RESULTS.read_text())["metrics"]
    epoch0_drift = {mode: epoch0[mode]["mean_foreground_iou"] - ha[{"g0": "matched", "phrase": "phrase", "tf": "tf"}[mode]]["mean_foreground_iou"] for mode in epoch0}
    if max(abs(value) for value in epoch0_drift.values()) > 5e-4:
        raise RuntimeError(f"epoch0 failed to recover Phase4H-A: {epoch0_drift}")
    rows = [{"epoch": 0, "optimizer_updates": 0, "seg_loss": "", "identity_loss": "", "total_loss": "",
             "traversal_exposures": 0, "optimization_eligible_exposures": 0, "invalid_g0_exposures": 0,
             "dev_g0_mean_iou": epoch0["g0"]["mean_foreground_iou"], "dev_phrase_mean_iou": epoch0["phrase"]["mean_foreground_iou"],
             "dev_tf_mean_iou": epoch0["tf"]["mean_foreground_iou"], "mapper_weight": 0.0, "mapper_bias": 0.0, "seconds": 0.0}]
    write_history(rows); updates = 0
    for epoch in range(1, EPOCHS + 1):
        began = time.time(); order = list(ids); random.Random(SEED + 1009 * epoch).shuffle(order)
        sums = defaultdict(float); traversal = eligible = invalid = 0; mapper.train()
        for begin in range(0, len(order), BATCH):
            all_batch = order[begin:begin + BATCH]; positions = torch.tensor([id_to_index[sid] for sid in all_batch]); keep = valid.index_select(0, positions); idx = positions[keep]
            traversal += len(all_batch); eligible += len(idx); invalid += len(all_batch) - len(idx)
            if not len(idx):
                continue
            batch_ids = [ids[index] for index in idx.tolist()]
            utility_batch = full_inputs(data, idx, device)
            s64, phase4f, support, targets, sc, qseg = make_phase4f_batch(store, batch_ids, source, rectifier, device)
            u = frozen_utility(utility_model, utility_batch, sc, device)
            optimizer.zero_grad(set_to_none=True); g = mapper(u, support); adapted = gated_embedding(s64, phase4f, g)
            _, seg = decode_loss(sam, qseg, adapted, targets)
            active = support.float(); identity = ((g - u).abs() * active).flatten(1).sum(1).div(active.flatten(1).sum(1).clamp_min(1)).mean()
            total = seg["total"] + LAMBDA_IDENTITY * identity
            if not torch.isfinite(total): raise RuntimeError("nonfinite Stage1 loss")
            total.backward(); norm = torch.nn.utils.clip_grad_norm_(mapper.parameters(), GRAD_CLIP)
            if not torch.isfinite(norm): raise RuntimeError("nonfinite mapper gradient")
            optimizer.step(); updates += 1; n = len(idx)
            sums["seg"] += float(seg["total"].detach()) * n; sums["identity"] += float(identity.detach()) * n; sums["total"] += float(total.detach()) * n
            if updates <= 3 or updates % 100 == 0:
                print(json.dumps({"stage": "TRAIN", "epoch": epoch, "update": updates, "loss": float(total.detach()),
                                  "weight": float(mapper.residual.weight.detach()), "bias": float(mapper.residual.bias.detach())}), flush=True)
        if (traversal, eligible, invalid) != (8836, 8690, 146):
            raise RuntimeError("formal exposure mismatch")
        dev_g0 = evaluate(mapper, val_store, sam, source, rectifier, utility_model, dev["g0"], "g0", "matched", device, collect_records=False)[0]
        row = {"epoch": epoch, "optimizer_updates": updates, "seg_loss": sums["seg"] / eligible, "identity_loss": sums["identity"] / eligible,
               "total_loss": sums["total"] / eligible, "traversal_exposures": traversal, "optimization_eligible_exposures": eligible,
               "invalid_g0_exposures": invalid, "dev_g0_mean_iou": dev_g0["mean_foreground_iou"], "dev_phrase_mean_iou": "", "dev_tf_mean_iou": "",
               "mapper_weight": float(mapper.residual.weight.detach()), "mapper_bias": float(mapper.residual.bias.detach()), "seconds": time.time() - began}
        rows.append(row); write_history(rows); save_checkpoint(CKPT / f"epoch_{epoch}.pt", mapper, optimizer, epoch, updates, dev_g0, manifest["config_sha256"])
        print(json.dumps({"stage": "EPOCH_COMPLETE", **row}), flush=True)
    best = max(rows[1:], key=lambda row: (float(row["dev_g0_mean_iou"]), -int(row["epoch"])))
    chosen = CKPT / f"epoch_{best['epoch']}.pt"; shutil.copy2(chosen, SELECTED)
    selector = {"schema": "phase4hb_selector_v1", "status": "COMPLETE", "primary": "DEV G0 mean all-ref-union IoU",
                "population": 1106, "invalid_iou_zero": 28, "tie_break": "earlier epoch", "epoch0_selector_eligible": False,
                "epoch0": {"metrics": epoch0, "phase4ha_drift": epoch0_drift}, "selected_epoch": int(best["epoch"]),
                "selected_dev_g0": float(best["dev_g0_mean_iou"]), "selected_checkpoint_sha256": file_sha256(chosen),
                "output_selected_checkpoint_sha256": file_sha256(SELECTED),
                "candidates": [{"epoch": int(row["epoch"]), "mean_g0_iou": float(row["dev_g0_mean_iou"])} for row in rows[1:]],
                "phrase_used": False, "tf_used": False, "controls_used": False, "endpoint_audit": endpoint,
                "frozen_config": manifest["config"]}
    dump(SELECTOR, selector)
    after = {name: tensor_state_sha256(model.state_dict()) for name, model in zip(("sam", "source", "rectifier", "utility"), frozen_models)}
    if frozen_hash_before != after:
        raise RuntimeError("frozen source state changed")
    manifest.update({"status": "TRAINING_COMPLETE_SELECTOR_FROZEN", "formal_optimizer_updates": updates,
                     "frozen_state_hash_before": frozen_hash_before, "frozen_state_hash_after": after,
                     "selector_sha256": file_sha256(SELECTOR), "selected_checkpoint_sha256": file_sha256(SELECTED)})
    dump(MANIFEST, manifest)


def evaluate_selected(device):
    selector = json.loads(SELECTOR.read_text()); state = torch.load(SELECTED, map_location="cpu", weights_only=False)
    mapper = BoundedIdentityMapper().to(device); mapper.load_state_dict(state["mapper_state"], strict=True); mapper.eval()
    store = Phase4FStore(CFG, "val"); sam = load_sam_runtime(CFG, device); source = load_evidence_source(CFG, "forensic_rect", device)
    rectifier, _ = load_phase4f(device); utility_model = load_utility(device); dev = {mode: load_dev(mode) for mode in ("g0", "phrase", "tf")}
    jobs = [("g0", "matched"), ("phrase", "matched"), ("tf", "matched"), ("g0", "cross_image"), ("g0", "spatial_shuffle"), ("g0", "forensic_off")]
    metrics, records, gate_diagnostics, identities = {}, {}, {}, {}
    for mode, condition in jobs:
        key = {"phrase": "phrase", "tf": "tf"}.get(mode, condition)
        metrics[key], records[key], gate_diagnostics[key], identities[key] = evaluate(mapper, store, sam, source, rectifier, utility_model, dev[mode], mode, condition, device)
    ids = store.sample_ids; p1 = {mode: baseline_records(mode, ids) for mode in ("g0", "phrase", "tf")}; p4f = {mode: phase4f_records(mode) for mode in ("g0", "phrase", "tf")}
    ha_result = json.loads(HA_RESULTS.read_text()); ha_records = {"g0": ha_result["records"]["matched"], "phrase": ha_result["records"]["phrase"], "tf": ha_result["records"]["tf"]}
    statistics = {name: {} for name in ("vs_P1", "vs_Phase4F", "vs_Phase4H-A")}
    for mode, key in (("g0", "matched"), ("phrase", "phrase"), ("tf", "tf")):
        statistics["vs_P1"][mode] = compare(records[key], p1[mode], seed=SEED)
        statistics["vs_Phase4F"][mode] = compare(records[key], p4f[mode], seed=SEED)
        statistics["vs_Phase4H-A"][mode] = compare(records[key], ha_records[mode], seed=SEED)
    statistics["matched_minus_cross"] = compare(records["matched"], records["cross_image"], seed=SEED)
    statistics["matched_minus_shuffle"] = compare(records["matched"], records["spatial_shuffle"], seed=SEED)
    result = {"schema": "phase4hb_dev_results_v1", "status": "COMPLETE", "selector": selector,
              "mapper": {"architecture": "2-parameter bounded logit-residual Conv1x1", "parameters": 2,
                         "weight": float(mapper.residual.weight.detach()), "bias": float(mapper.residual.bias.detach())},
              "metrics": metrics, "records": records, "statistics": statistics, "gate_diagnostics": gate_diagnostics,
              "identities": identities, "historical": {**HA_HISTORICAL, "Phase4H-A": {"g0": 0.1727973844346831, "phrase": 0.22284259199832093, "tf": 0.2556609647449051}},
              "evaluation_contract": {"population": 1106, "invalid_iou_zero": 28, "batch_size": 1, "threshold_logit": 0.0,
                                      "phrase_tf_selected_only_except_preregistered_epoch0_baseline": True},
              "firewall": {"internal_test_accessed": False, "official1000_accessed": False}}
    dump(DEV_RESULTS, result)


def finalize() -> dict:
    result = json.loads(DEV_RESULTS.read_text()); selector = json.loads(SELECTOR.read_text()); manifest = json.loads(MANIFEST.read_text())
    m, s, d = result["metrics"], result["statistics"], result["gate_diagnostics"]["matched"]
    delta_g0 = s["vs_Phase4H-A"]["g0"]["foreground_iou"]
    delta_phrase = s["vs_Phase4H-A"]["phrase"]["foreground_iou"]; delta_tf = s["vs_Phase4H-A"]["tf"]["foreground_iou"]
    beats_point = delta_g0["mean_difference"] > 0
    clear_benefit = delta_g0["bootstrap_95_ci"][0] > 0
    phrase_nonreg = delta_phrase["bootstrap_95_ci"][0] >= NONDEGRADATION_MARGIN
    tf_nonreg = delta_tf["bootstrap_95_ci"][0] >= NONDEGRADATION_MARGIN
    order = m["matched"]["mean_foreground_iou"] < m["phrase"]["mean_foreground_iou"] < m["tf"]["mean_foreground_iou"]
    controls = s["matched_minus_cross"]["foreground_iou"]["bootstrap_95_ci"][0] > 0 and s["matched_minus_shuffle"]["foreground_iou"]["bootstrap_95_ci"][0] > 0
    p1_endpoint = selector["endpoint_audit"]["P1"]["pass"] is True
    p4f_endpoint = selector["endpoint_audit"]["Phase4F"]["pass"] is True
    off = result["identities"]["forensic_off"]["all_exact"] is True
    gate = d["g"]; low = d["low_saturation_fraction_g_le_0.01"]; high = d["high_saturation_fraction_g_ge_0.99"]
    collapse = low >= COLLAPSE_FRACTION or high >= COLLAPSE_FRACTION or gate["std"] < COLLAPSE_STD
    next_stage = clear_benefit and phrase_nonreg and tf_nonreg and order and controls and p1_endpoint and p4f_endpoint and off and not collapse
    gates = {"schema": "phase4hb_gate_summary_v1", "STAGE1_TRAINING_COMPLETE": "YES",
             "SELECTED_EPOCH": selector["selected_epoch"], "SELECTED_DEV_G0": selector["selected_dev_g0"],
             "STAGE1_BEATS_PHASE4HA_G0": "YES" if beats_point else "NO",
             "STAGE1_G0_CLEAR_BENEFIT_CI_LOWER_GT_ZERO": "YES" if clear_benefit else "NO",
             "STAGE1_LANGUAGE_ORDER_PRESERVED": "YES" if order else "NO",
             "STAGE1_PHRASE_NONDEGRADED": "YES" if phrase_nonreg else "NO",
             "STAGE1_TF_NONDEGRADED": "YES" if tf_nonreg else "NO",
             "MATCHED_GT_CROSS_SHUFFLE": "YES" if controls else "NO",
             "P1_ENDPOINT_RECOVERY": "PASS" if p1_endpoint else "FAIL", "PHASE4F_ENDPOINT_RECOVERY": "PASS" if p4f_endpoint else "FAIL",
             "FORENSIC_OFF_EXACT_P1": "PASS" if off else "FAIL", "GATE_COLLAPSE": "YES" if collapse else "NO",
             "NEXT_UNFREEZE_STAGE_JUSTIFIED": "YES" if next_stage else "NO",
             "CURRENT_BEST_SCHEME": "Phase4H-B Stage1" if next_stage else "Phase4H-A frozen adaptive",
             "INTERNAL_TEST_ACCESSED": "NO", "OFFICIAL1000_ACCESSED": "NO"}
    dump(GATES, gates); write_report(result, gates)
    for row in manifest["source_files"].values():
        row["sha256_after"] = file_sha256(Path(row["path"])); row["unchanged"] = row["sha256_after"] == row["sha256_before"]
    manifest.update({"status": "COMPLETE", "dev_results_sha256": file_sha256(DEV_RESULTS), "gate_summary_sha256": file_sha256(GATES),
                     "training_history_sha256": file_sha256(HISTORY), "selector_sha256": file_sha256(SELECTOR),
                     "selected_checkpoint_sha256": file_sha256(SELECTED), "report_sha256": file_sha256(DOC), "stage2_started": False})
    dump(MANIFEST, manifest); return gates


def write_report(result, gates):
    m, s, d = result["metrics"], result["statistics"], result["gate_diagnostics"]["matched"]
    historical = result["historical"]
    table = "\n".join(["| Model | G0 | Phrase | TF |", "|---|---:|---:|---:|",
        f"| P1 | {historical['P1']['g0']:.6f} | {historical['P1']['phrase']:.6f} | {historical['P1']['tf']:.6f} |",
        f"| Phase4F | {historical['Phase4F']['g0']:.6f} | {historical['Phase4F']['phrase']:.6f} | {historical['Phase4F']['tf']:.6f} |",
        f"| Phase4H-A | {historical['Phase4H-A']['g0']:.6f} | {historical['Phase4H-A']['phrase']:.6f} | {historical['Phase4H-A']['tf']:.6f} |",
        f"| Phase4H-B Stage1 | {m['matched']['mean_foreground_iou']:.6f} | {m['phrase']['mean_foreground_iou']:.6f} | {m['tf']['mean_foreground_iou']:.6f} |"])
    def stat(value):
        x = value["foreground_iou"]
        return f"delta={x['mean_difference']:+.6f}, CI=[{x['bootstrap_95_ci'][0]:+.6f},{x['bootstrap_95_ci'][1]:+.6f}], W/T/L={x['wins']}/{x['ties']}/{x['losses']}, p={x['wilcoxon_pvalue']:.4g}"
    text = f"""# Phase 4H-B Adaptive Phase4F Progressive Unfreezing — Stage 1

## Frozen protocol

唯一trainable为2参数bounded mapper `phi(U_F)=sigmoid(logit(clamp(U_F))+Conv1x1(U_F))`；Conv weight/bias均以0初始化。其他P1/SAM/CLIP/Phase4C-A/Phase4F rectifier/Phase4G-1S utility全部冻结。Loss=`L_seg+0.1*L_identity`；AdamW lr1e-4/wd1e-4、batch8、10 epochs、seed3407、无scheduler/early stopping。全train Fake每epoch traversal 8836，8690 valid进入optimizer，146 invalid仅accounting，Real=0。

Epoch0 G0/Phrase/TF={result['selector']['epoch0']['metrics']['g0']['mean_foreground_iou']:.6f}/{result['selector']['epoch0']['metrics']['phrase']['mean_foreground_iou']:.6f}/{result['selector']['epoch0']['metrics']['tf']['mean_foreground_iou']:.6f}；仅作Phase4H-A近恒等baseline，不参与selector。Selector只用epochs1-10 direct batch1 DEV G0，selected epoch={result['selector']['selected_epoch']}。

## Development results

{table}

Controls：matched={m['matched']['mean_foreground_iou']:.6f}，cross={m['cross_image']['mean_foreground_iou']:.6f}，shuffle={m['spatial_shuffle']['mean_foreground_iou']:.6f}，off={m['forensic_off']['mean_foreground_iou']:.6f}。

## Paired comparisons to Phase4H-A

- G0：{stat(s['vs_Phase4H-A']['g0'])}
- Phrase：{stat(s['vs_Phase4H-A']['phrase'])}
- TF：{stat(s['vs_Phase4H-A']['tf'])}
- matched−cross：{stat(s['matched_minus_cross'])}
- matched−shuffle：{stat(s['matched_minus_shuffle'])}

## Mapper and gate diagnostics

Selected mapper weight={result['mapper']['weight']:+.8f}，bias={result['mapper']['bias']:+.8f}。

- U_F: mean={d['U_F']['mean']:.6f}, std={d['U_F']['std']:.6f}, percentiles={json.dumps(d['U_F']['percentiles'])}
- g: mean={d['g']['mean']:.6f}, std={d['g']['std']:.6f}, percentiles={json.dumps(d['g']['percentiles'])}
- mean |g-U_F|={d['mean_abs_g_minus_U_F']:.6f}
- g<=0.01={d['low_saturation_fraction_g_le_0.01']:.6f}; g>=0.99={d['high_saturation_fraction_g_ge_0.99']:.6f}
- GATE_COLLAPSE={gates['GATE_COLLAPSE']}

## Final gates

```json
{json.dumps(gates, ensure_ascii=False, indent=2)}
```

到此严格STOP。没有自动进入Stage2，internal test与official1000未访问。
"""
    DOC.parent.mkdir(parents=True, exist_ok=True); DOC.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("train", "evaluate", "finalize", "all")); parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args(); device = torch.device(args.device); torch.cuda.set_device(device)
    if args.stage == "train": train(device)
    elif args.stage == "evaluate": evaluate_selected(device)
    elif args.stage == "finalize": print(json.dumps(finalize(), ensure_ascii=False, indent=2))
    else: train(device); evaluate_selected(device); print(json.dumps(finalize(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
