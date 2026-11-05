#!/usr/bin/env python3
"""Preflight, matched training, and selected-checkpoint evaluation for Phase 4F."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase4f import (
    Phase4FStore, append, compare, decode, deterministic_order, dump, evaluate,
    evidence_feature, file_sha256, ids_sha256, load_evidence_source, load_rectifier,
    load_sam_runtime, mask_loss, q_index, summarize, tensor_state_sha256,
)

CFG = yaml.safe_load((ROOT / "configs/phase4f_language_preserving_rectification.yaml").read_text())
OUT = ROOT / CFG["experiment"]["output_root"]
CKPT = Path(CFG["experiment"]["checkpoint_root"])


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "train", "evaluate"), required=True)
    parser.add_argument("--arm", choices=tuple(CFG["arms"]))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_all(seed: int = 3407) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def selected_gamma() -> float:
    value = json.loads((OUT / "preflight/geometry_scale_audit.json").read_text())
    if value["status"] != "PASS":
        raise RuntimeError("PASS geometry scale audit required")
    return float(value["selected_gamma"])


def load_baseline_records(mode: str, store: Phase4FStore) -> list[dict]:
    _, records = store.q_for_mode(mode)
    by_id = {str(row["sample_id"]): row for row in records}
    result = []
    for sid in store.sample_ids:
        row = by_id[sid]
        result.append({"sample_id": sid, "foreground_iou": float(row["foreground_iou"]), "foreground_f1": float(row["foreground_f1"])})
    return result


def geometry_scale_audit(device: torch.device) -> dict:
    store = Phase4FStore(CFG, "train")
    source = load_evidence_source(CFG, "forensic_rect", device)
    ids = [sid for sid in deterministic_order(store.sample_ids, 0) if sid in store.valid_ids][:8]
    s64, raw, _, sc, cc, _ = store.batch(ids, device)
    evidence = evidence_feature(source, raw)
    candidates = []
    for gamma in CFG["architecture"]["gamma_candidates"]:
        rectifier = load_rectifier(CFG, float(gamma), device).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            value = rectifier(s64, evidence, sc, cc, torch.ones(len(ids), 576, dtype=torch.bool, device=device))
        base = s64.float().flatten(2).transpose(1, 2)
        residual = value["residual"].float().flatten(2).transpose(1, 2)
        ratio = residual.norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-12)
        support = value["support"]
        median_ratio = float(ratio[support].median())
        changed = (value["image_embeddings"].to(torch.bfloat16) != s64.to(torch.bfloat16)).flatten(2).any(1)
        survival = float(changed[support].float().mean())
        passed = (float(CFG["architecture"]["residual_ratio_range"][0]) <= median_ratio <= float(CFG["architecture"]["residual_ratio_range"][1]) and survival >= float(CFG["architecture"]["minimum_bf16_token_survival"]))
        candidates.append({"gamma": float(gamma), "median_residual_to_s64_norm_ratio": median_ratio, "bf16_token_survival": survival, "pass": passed})
        del rectifier
    passing = [row for row in candidates if row["pass"]]
    result = {
        "status": "PASS" if passing else "FAIL_NUMERICAL_INCOMPATIBILITY",
        "selection_uses_train_only": True,
        "sample_ids": ids,
        "candidates": candidates,
        "selected_gamma": passing[0]["gamma"] if passing else None,
        "formal_optimizer_updates": 0,
        "validation_performance_used": False,
        "internal_test_accessed": False,
        "official1000_accessed": False,
    }
    dump(OUT / "preflight/geometry_scale_audit.json", result)
    if not passing:
        raise RuntimeError("no preregistered gamma candidate passed")
    del store, source, s64, raw, evidence
    gc.collect(); torch.cuda.empty_cache()
    return result


def p1_equivalence_audit(device: torch.device) -> dict:
    gamma = selected_gamma()
    store = Phase4FStore(CFG, "val")
    sam = load_sam_runtime(CFG, device)
    rectifier = load_rectifier(CFG, gamma, device).eval()
    source = load_evidence_source(CFG, "forensic_rect", device)
    mode_results = {}
    all_pass = True
    for mode in ("g0", "phrase", "tf"):
        metrics, records = evaluate(CFG, store, sam, rectifier, source, mode, device, condition="rectifier_off")
        baseline = load_baseline_records(mode, store)
        differences = compare(records, baseline)
        max_iou = max(abs(a["foreground_iou"] - b["foreground_iou"]) for a, b in zip(records, baseline))
        max_f1 = max(abs(a["foreground_f1"] - b["foreground_f1"]) for a, b in zip(records, baseline))
        passed = max_iou <= 1e-6 and max_f1 <= 1e-6
        all_pass = all_pass and passed
        mode_results[mode] = {"metrics": metrics, "max_abs_iou_difference": max_iou, "max_abs_f1_difference": max_f1, "paired": differences, "pass": passed}
        root = OUT / "preflight/p1_equivalence"; root.mkdir(parents=True, exist_ok=True)
        (root / f"{mode}_rectifier_off.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    exact_rows = []
    low_cache = torch.load(ROOT / CFG["data"]["p1_g0_low_res"], map_location="cpu", weights_only=False)
    low_by_id = dict(zip(low_cache["sample_ids"], low_cache["low_res_logits"]))
    qcache, _ = store.q_for_mode("g0")
    exact_ids = [sid for sid in store.sample_ids if qcache[sid][1]][:16]
    with torch.no_grad():
        for sid in exact_ids:
            s64, raw, _, sc, cc, _ = store.batch([sid], device)
            evidence = evidence_feature(source, raw)
            q = qcache[sid][0].to(device=device, dtype=torch.bfloat16)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                low, _ = decode(sam, rectifier, q, s64, evidence, sc, cc, enabled=False)
            current = low[0, 0].detach().cpu().to(torch.bfloat16)
            baseline = low_by_id[sid].to(torch.bfloat16)
            exact_rows.append({"sample_id": sid, "tensor_exact": bool(torch.equal(current, baseline)), "max_abs_difference": float((current.float() - baseline.float()).abs().max())})
    exact = all(row["tensor_exact"] for row in exact_rows)
    all_pass = all_pass and exact
    result = {
        "status": "PASS" if all_pass else "IMPLEMENTATION_INVALID",
        "P1_PATH_EXACTLY_PRESERVED": "YES" if all_pass else "NO",
        "metric_tolerance": 1e-6,
        "modes": mode_results,
        "exact_low_res_subset": exact_rows,
        "rectifier_off_is_direct_bypass": True,
        "canonical_prompt_sha256": CFG["p1"]["prompt_hash"],
        "p1_checkpoint_sha256": file_sha256(Path(CFG["p1"]["checkpoint"])),
        "formal_optimizer_updates": 0,
        "internal_test_accessed": False,
        "official1000_accessed": False,
    }
    dump(OUT / "preflight/p1_equivalence_audit.json", result)
    if not all_pass:
        raise RuntimeError("rectifier-off does not recover P1")
    del store, sam, rectifier, source
    gc.collect(); torch.cuda.empty_cache()
    return result


def gradient_audit(device: torch.device) -> dict:
    gamma = selected_gamma()
    store = Phase4FStore(CFG, "train")
    sam = load_sam_runtime(CFG, device)
    results = {}
    initial_hashes = {}
    ids = [sid for sid in deterministic_order(store.sample_ids, 0) if sid in store.valid_ids][:8]
    for arm in CFG["arms"]:
        source = load_evidence_source(CFG, arm, device)
        rectifier = load_rectifier(CFG, gamma, device).train()
        initial_hashes[arm] = tensor_state_sha256(rectifier.state_dict())
        frozen_sam_hash = tensor_state_sha256(sam.state_dict())
        frozen_source_hash = tensor_state_sha256(source.state_dict())
        s64, raw, targets, sc, cc, q = store.batch(ids, device)
        evidence = evidence_feature(source, raw)
        rectifier.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            low, _ = decode(sam, rectifier, q, s64, evidence, sc, cc)
        losses = mask_loss(low, targets, CFG)
        losses["total"].backward()
        gradients = {name: (None if parameter.grad is None else float(parameter.grad.detach().float().norm())) for name, parameter in rectifier.named_parameters()}
        required_groups = {
            "Wq": [name for name in gradients if "q_proj" in name],
            "Wk": [name for name in gradients if "k_proj" in name],
            "Wv": [name for name in gradients if "v_proj" in name],
            "Wo": [name for name in gradients if "out_proj" in name],
            "Projection": [name for name in gradients if name.endswith("projection.weight") or name.endswith("projection.bias")],
            "gamma": [name for name in gradients if name.endswith("gamma")],
        }
        group_pass = {group: bool(names) and all(gradients[name] is not None and gradients[name] > 0 and math.isfinite(gradients[name]) for name in names) for group, names in required_groups.items()}
        frozen = {
            "P1_SAM_gradient_tensor_count": sum(parameter.grad is not None for parameter in sam.parameters()),
            "4C_A_source_gradient_tensor_count": sum(parameter.grad is not None for parameter in source.parameters()),
            "P1_SAM_hash_unchanged": tensor_state_sha256(sam.state_dict()) == frozen_sam_hash,
            "4C_A_source_hash_unchanged": tensor_state_sha256(source.state_dict()) == frozen_source_hash,
        }
        passed = all(group_pass.values()) and frozen["P1_SAM_gradient_tensor_count"] == 0 and frozen["4C_A_source_gradient_tensor_count"] == 0 and frozen["P1_SAM_hash_unchanged"] and frozen["4C_A_source_hash_unchanged"]
        results[arm] = {"pass": passed, "loss": float(losses["total"].detach()), "gradients": gradients, "required_groups": group_pass, "frozen": frozen, "formal_optimizer_updates": 0}
        del source, rectifier, evidence, low
        torch.cuda.empty_cache()
    matched_initialization = len(set(initial_hashes.values())) == 1
    result = {"status": "PASS" if all(row["pass"] for row in results.values()) and matched_initialization else "FAIL", "sample_ids": ids, "arms": results, "initial_rectifier_hashes": initial_hashes, "matched_initialization": matched_initialization, "formal_optimizer_updates": 0, "internal_test_accessed": False, "official1000_accessed": False}
    dump(OUT / "preflight/gradient_routing_audit.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("gradient routing audit failed")
    return result


def preflight(device: torch.device) -> None:
    frozen = json.loads((OUT / "manifests/frozen_protocol.json").read_text())
    if frozen["status"] != "COMPLETE":
        raise RuntimeError("frozen protocol manifest required")
    scale = geometry_scale_audit(device)
    equivalence = p1_equivalence_audit(device)
    gradients = gradient_audit(device)
    result = {"status": "PASS", "gamma": scale["selected_gamma"], "P1_PATH_EXACTLY_PRESERVED": equivalence["P1_PATH_EXACTLY_PRESERVED"], "gradient_routing": gradients["status"], "formal_optimizer_updates": 0, "internal_test_accessed": False, "official1000_accessed": False}
    dump(OUT / "preflight/integrated_preflight.json", result)
    print(json.dumps(result, indent=2))


def optimizer_scheduler(rectifier, total_updates: int):
    optimizer = torch.optim.AdamW(rectifier.parameters(), lr=float(CFG["optimizer"]["learning_rate"]), weight_decay=float(CFG["optimizer"]["weight_decay"]), betas=tuple(CFG["optimizer"]["betas"]))
    warm = round(total_updates * float(CFG["optimizer"]["warmup_fraction"]))
    def scale(step):
        if step < warm: return float(step + 1) / max(1, warm)
        progress = min(1.0, (step - warm) / max(1, total_updates - warm))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def save_checkpoint(path: Path, arm: str, epoch: int, updates: int, rectifier, optimizer, scheduler, metrics=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "phase4f_rectifier_v1", "arm": arm, "epoch": epoch, "optimizer_updates": updates, "gamma_initial": selected_gamma(), "rectifier": rectifier.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "validation": metrics}
    temporary = path.with_suffix(".pt.tmp"); torch.save(payload, temporary); temporary.replace(path)


def train(arm: str, device: torch.device) -> None:
    if not arm: raise ValueError("--arm required")
    preflight_state = json.loads((OUT / "preflight/integrated_preflight.json").read_text())
    if preflight_state["status"] != "PASS": raise RuntimeError("PASS integrated preflight required")
    gamma = selected_gamma(); store = Phase4FStore(CFG, "train"); val = Phase4FStore(CFG, "val")
    sam = load_sam_runtime(CFG, device); source = load_evidence_source(CFG, arm, device); rectifier = load_rectifier(CFG, gamma, device)
    sam_hash = tensor_state_sha256(sam.state_dict()); source_hash = tensor_state_sha256(source.state_dict()); initial_hash = tensor_state_sha256(rectifier.state_dict())
    total_updates = int(CFG["training"]["dataloader_steps_per_epoch"]) * int(CFG["training"]["epochs"])
    optimizer, scheduler = optimizer_scheduler(rectifier, total_updates)
    root = CKPT / arm; history_path = OUT / "training" / arm / "history.json"; history = []; start_epoch = 0; updates = 0
    existing = sorted(root.glob("epoch_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
    if existing:
        state = torch.load(existing[-1], map_location="cpu", weights_only=False); rectifier.load_state_dict(state["rectifier"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); start_epoch = int(state["epoch"]); updates = int(state["optimizer_updates"]); history = json.loads(history_path.read_text())
    else:
        metrics, records = evaluate(CFG, val, sam, rectifier, source, "g0", device)
        history = [{"epoch": 0, "optimizer_updates": 0, "train": None, "validation": metrics, "seconds": 0.0}]
        dump(history_path, history); save_checkpoint(root / "epoch_0.pt", arm, 0, 0, rectifier, optimizer, scheduler, metrics)
        evalroot = OUT / "evaluation" / arm; evalroot.mkdir(parents=True, exist_ok=True); (evalroot / "epoch_0_g0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    valid = store.valid_ids
    for epoch in range(start_epoch + 1, int(CFG["training"]["epochs"]) + 1):
        began = time.time(); order = deterministic_order(store.sample_ids, epoch); traversal = eligible = invalid = empty = 0; sums = {"total": 0.0, "bce": 0.0, "dice": 0.0}; rectifier.train(); batchlog = OUT / "training" / arm / f"epoch_{epoch}_batches.jsonl"
        for begin in range(0, len(order), int(CFG["training"]["batch_size"])):
            all_ids = order[begin:begin + int(CFG["training"]["batch_size"])]; ids = [sid for sid in all_ids if sid in valid]; traversal += len(all_ids); eligible += len(ids); invalid += len(all_ids) - len(ids)
            record = {"epoch": epoch, "dataloader_step": begin // int(CFG["training"]["batch_size"]) + 1, "n_total": len(all_ids), "n_valid_g0": len(ids), "n_invalid_g0": len(all_ids) - len(ids)}
            if not ids:
                empty += 1; record["status"] = "EMPTY_VALID_G0_BATCH"; append(batchlog, record); continue
            s64, raw, targets, sc, cc, q = store.batch(ids, device); evidence = evidence_feature(source, raw); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): low, _ = decode(sam, rectifier, q, s64, evidence, sc, cc)
            losses = mask_loss(low, targets, CFG)
            if not torch.isfinite(losses["total"]): raise RuntimeError("nonfinite mask loss")
            losses["total"].backward(); norm = torch.nn.utils.clip_grad_norm_(rectifier.parameters(), float(CFG["optimizer"]["gradient_clip_norm"]))
            if not torch.isfinite(norm): raise RuntimeError("nonfinite rectifier gradient")
            optimizer.step(); scheduler.step(); updates += 1; record.update({"status": "UPDATED", "optimizer_update": updates, "loss": float(losses["total"].detach()), "gradient_norm": float(norm)}); append(batchlog, record)
            for key in sums: sums[key] += float(losses[key].detach()) * len(ids)
            if updates <= 5 or updates % 50 == 0: print(json.dumps({"arm": arm, "epoch": epoch, "update": updates, "loss": float(losses["total"].detach()), "gamma": float(rectifier.rectification.gamma.detach())}), flush=True)
        if (traversal, eligible, invalid) != (8836, 8690, 146): raise RuntimeError("conditional exposure mismatch")
        metrics, records = evaluate(CFG, val, sam, rectifier, source, "g0", device)
        row = {"epoch": epoch, "optimizer_updates": updates, "train": {key: value / eligible for key, value in sums.items()} | {"traversal_exposures": traversal, "optimization_eligible_exposures": eligible, "invalid_g0_exposures": invalid, "empty_valid_g0_batches": empty, "sample_order_sha256": ids_sha256(order)}, "validation": metrics, "gamma": float(rectifier.rectification.gamma.detach()), "seconds": time.time() - began}
        history.append(row); dump(history_path, history); save_checkpoint(root / f"epoch_{epoch}.pt", arm, epoch, updates, rectifier, optimizer, scheduler, metrics)
        evalroot = OUT / "evaluation" / arm; (evalroot / f"epoch_{epoch}_g0.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records)); print(json.dumps({"arm": arm, **row}), flush=True)
    best = max(history, key=lambda row: (row["validation"]["mean_foreground_iou"], -row["epoch"]))
    selected = root / f"epoch_{best['epoch']}.pt"
    selector = {"status": "COMPLETE", "arm": arm, "primary": CFG["evaluation"]["selector"], "population": 1106, "invalid_g0_retained": 28, "tie_break": "earlier_epoch", "selected_epoch": best["epoch"], "selected_checkpoint": str(selected), "selected_checkpoint_sha256": file_sha256(selected), "selected_metrics": best["validation"], "candidates": history, "total_traversal_exposures": 88360, "total_optimization_eligible_exposures": 86900, "actual_optimizer_updates": updates, "initial_rectifier_hash": initial_hash, "P1_SAM_unchanged": tensor_state_sha256(sam.state_dict()) == sam_hash, "evidence_source_unchanged": tensor_state_sha256(source.state_dict()) == source_hash, "internal_test_used": False, "official1000_used": False}
    dump(OUT / "selectors" / f"{arm}.json", selector); print(json.dumps(selector, indent=2))


def evaluate_selected(arm: str, device: torch.device) -> None:
    selector = json.loads((OUT / "selectors" / f"{arm}.json").read_text()); gamma = selected_gamma(); rectifier = load_rectifier(CFG, gamma, device); state = torch.load(selector["selected_checkpoint"], map_location="cpu", weights_only=False); rectifier.load_state_dict(state["rectifier"]); rectifier.eval()
    store = Phase4FStore(CFG, "val"); sam = load_sam_runtime(CFG, device); source = load_evidence_source(CFG, arm, device); root = OUT / "final" / arm; root.mkdir(parents=True, exist_ok=True)
    results, recordsets = {}, {}
    jobs = [("g0", "matched"), ("phrase", "matched"), ("tf", "matched"), ("g0", "cross_image"), ("g0", "spatial_shuffle"), ("g0", "zero"), ("g0", "rectifier_off")]
    for mode, condition in jobs:
        name = "phrase_only" if mode == "phrase" else ("tf_full" if mode == "tf" else condition)
        metrics, records = evaluate(CFG, store, sam, rectifier, source, mode, device, condition=condition); results[name] = metrics; recordsets[name] = records; (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records)); dump(root / f"{name}_metrics.json", metrics)
    results["valid_g0_only_diagnostic"] = summarize([row for row in recordsets["matched"] if row["valid_g0"]]); results["valid_g0_only_diagnostic"]["label"] = "VALID-G0-ONLY N=1,078; not headline canonical G0"
    p1 = {mode: load_baseline_records(mode, store) for mode in ("g0", "phrase", "tf")}
    results["paired"] = {
        "g0_vs_p1": compare(recordsets["matched"], p1["g0"]),
        "phrase_vs_p1": compare(recordsets["phrase_only"], p1["phrase"]),
        "tf_vs_p1": compare(recordsets["tf_full"], p1["tf"]),
        "phrase_minus_g0": compare(recordsets["phrase_only"], recordsets["matched"]),
        "tf_minus_g0": compare(recordsets["tf_full"], recordsets["matched"]),
        "tf_minus_phrase": compare(recordsets["tf_full"], recordsets["phrase_only"]),
        "matched_minus_cross": compare(recordsets["matched"], recordsets["cross_image"]),
        "matched_minus_shuffle": compare(recordsets["matched"], recordsets["spatial_shuffle"]),
        "matched_minus_zero": compare(recordsets["matched"], recordsets["zero"]),
    }
    p1_gap = float(np.mean([row["foreground_iou"] for row in p1["tf"]])) - float(np.mean([row["foreground_iou"] for row in p1["g0"]]))
    current_gap = results["tf_full"]["mean_foreground_iou"] - results["matched"]["mean_foreground_iou"]
    results["oracle_gap"] = {"P1": p1_gap, "phase4f": current_gap, "retention_ratio": current_gap / p1_gap}
    detection = json.loads((ROOT / "outputs/phase4b_global_fepn_injection/detection_metrics.json").read_text())["P1"]
    results["canonical_detection"] = {**detection, "invariance": "exact by architecture: Phase 4F is downstream of frozen P1 generation and classification"}
    dump(root / "summary.json", results); print(json.dumps(results, indent=2))


def main() -> None:
    args = cli(); device = torch.device(args.device); torch.cuda.set_device(device); seed_all()
    if args.mode == "preflight": preflight(device)
    elif args.mode == "train": train(args.arm, device)
    else: evaluate_selected(args.arm, device)


if __name__ == "__main__":
    main()
