#!/usr/bin/env python3
"""Phase 4H-A: frozen Phase4F rectification gated by Phase4G-1S utility."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import phase4g1q_conditional_utility as utility_code
from training.phase4gf_formal_localization import full_inputs, load_dev
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import compare, summarize, tensor_state_sha256
from tools.phase4f import (
    Phase4FStore,
    evidence_feature,
    invalid_record,
    load_evidence_source,
    load_rectifier,
    load_sam_runtime,
    q_index,
)

PHASE = "phase4ha"
OUT = ROOT / "outputs" / PHASE
RESULTS = OUT / "results.json"
GATES = OUT / "gate_summary.json"
REPORT = ROOT / "docs" / PHASE / "report.md"
CFG = yaml.safe_load((ROOT / "configs/phase4f_language_preserving_rectification.yaml").read_text())
SELECTOR = ROOT / "outputs/phase4f_language_preserving_rectification/selectors/forensic_rect.json"
P4F_ROOT = ROOT / "outputs/phase4f_language_preserving_rectification/final/forensic_rect"
UTILITY_CKPT = Path("checkpoints/phase4g1s_mismatch_aware_utility/csculf_mismatch_utility_epoch10.pt")
SEED = 3407
HISTORICAL = {
    "P1": {"g0": 0.1482333978666042, "phrase": 0.24579772094855956, "tf": 0.3429279193646265},
    "Phase4F": {"g0": 0.17161424058491842, "phrase": 0.19357286978340313, "tf": 0.2052624796387254},
}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def selected_gamma() -> float:
    audit = json.loads((ROOT / "outputs/phase4f_language_preserving_rectification/preflight/geometry_scale_audit.json").read_text())
    if audit["status"] != "PASS":
        raise RuntimeError("Phase4F geometry audit is not PASS")
    return float(audit["selected_gamma"])


def load_utility(device: torch.device):
    seed_all()
    source = torch.load(utility_code.G1C_CKPT, map_location="cpu", weights_only=False)
    temperatures = torch.load(utility_code.G1C_TEMPERATURES, map_location="cpu", weights_only=False)
    model = utility_code.CSCULF(temperatures["T_L"], temperatures["T_F"])
    model.load_frozen_source_heads(source["language_head"], source["forensic_head"])
    state = torch.load(UTILITY_CKPT, map_location="cpu", weights_only=False)
    if state.get("epoch") != 10 or state.get("optimizer_updates") != 6470:
        raise RuntimeError("Phase4G-1S epoch10 utility checkpoint drift")
    model.load_state_dict(state["model_state"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_phase4f(device: torch.device):
    selector = json.loads(SELECTOR.read_text())
    checkpoint = Path(selector["selected_checkpoint"])
    if selector["selected_epoch"] != 9 or file_sha256(checkpoint) != selector["selected_checkpoint_sha256"]:
        raise RuntimeError("original selected Phase4F checkpoint drift")
    rectifier = load_rectifier(CFG, selected_gamma(), device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("arm") != "forensic_rect" or state.get("epoch") != 9:
        raise RuntimeError("Phase4F forensic rectifier payload drift")
    rectifier.load_state_dict(state["rectifier"], strict=True)
    return rectifier.eval().requires_grad_(False), checkpoint


def gate_to_sam_grid(utility64: torch.Tensor, sam_coordinates: torch.Tensor) -> torch.Tensor:
    """Sample original-normalized U_F at Phase4F SAM-token coordinates."""
    grid = (2.0 * sam_coordinates.float() - 1.0).reshape(utility64.shape[0], 64, 64, 2)
    value = F.grid_sample(utility64.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return value.clamp(0.0, 1.0)


def gated_embedding(s64: torch.Tensor, phase4f: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """S64 + g*Delta_F with exact representable endpoints."""
    if gate.shape != (s64.shape[0], 1, 64, 64):
        raise ValueError("aligned gate must be [B,1,64,64]")
    mixed = (s64.float() + gate.float() * (phase4f.float() - s64.float())).to(s64.dtype)
    mixed = torch.where(gate == 0, s64, mixed)
    mixed = torch.where(gate == 1, phase4f, mixed)
    return mixed


def baseline_records(mode: str, ids: list[str]) -> list[dict]:
    name = {"g0": "G0", "phrase": "phrase_only", "tf": "tf_full_context"}[mode]
    _, rows = q_index(Path(CFG["data"]["q_cache_root"]) / "validation", name)
    by_id = {str(row["sample_id"]): row for row in rows}
    return [{
        "sample_id": sid,
        "foreground_iou": float(by_id[sid]["foreground_iou"]),
        "foreground_f1": float(by_id[sid]["foreground_f1"]),
        "tp": int(by_id[sid]["tp"]), "fp": int(by_id[sid]["fp"]), "fn": int(by_id[sid]["fn"]),
    } for sid in ids]


def phase4f_records(mode: str) -> list[dict]:
    name = {"g0": "matched", "phrase": "phrase_only", "tf": "tf_full"}[mode]
    return [json.loads(line) for line in (P4F_ROOT / f"{name}.jsonl").read_text().splitlines() if line]


def endpoint_preflight(store, sam, rectifier, source, utility_model, dev, device) -> dict:
    """Verify both embedding endpoints over all valid G0, plus decoder identity."""
    utility_hash = tensor_state_sha256(utility_model.state_dict())
    rectifier_hash = tensor_state_sha256(rectifier.state_dict())
    sam_hash = tensor_state_sha256(sam.state_dict())
    source_hash = tensor_state_sha256(source.state_dict())
    p1_embedding_exact = phase4f_embedding_exact = True
    p1_decoder_exact = phase4f_decoder_exact = True
    checked = decoder_checked = 0
    first_evidence_exact = None
    with torch.no_grad():
        for index, sid in enumerate(dev["sample_ids"]):
            if not bool(dev["valid"][index]):
                continue
            s64, raw, _, sc, cc, _ = store.batch([sid], device)
            evidence = evidence_feature(source, raw)
            if first_evidence_exact is None:
                first_evidence_exact = bool(torch.equal(evidence.cpu(), dev["F24"][index:index + 1]))
            valid = torch.ones(1, 576, dtype=torch.bool, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                rectified = rectifier(s64, evidence, sc, cc, valid)["image_embeddings"]
            zero = torch.zeros(1, 1, 64, 64, device=device)
            one = torch.ones_like(zero)
            p1_endpoint = gated_embedding(s64, rectified, zero)
            phase4f_endpoint = gated_embedding(s64, rectified, one)
            p1_embedding_exact &= torch.equal(p1_endpoint, s64)
            phase4f_embedding_exact &= torch.equal(phase4f_endpoint, rectified)
            checked += 1
            if decoder_checked < 16:
                qseg = dev["q_seg"][index:index + 1].to(device=device, dtype=torch.bfloat16)
                with torch.autocast(device_type=device.type, enabled=False):
                    p1_direct = sam(qseg, s64)
                    p1_gated = sam(qseg, p1_endpoint)
                    p4f_direct = sam(qseg, rectified.to(torch.bfloat16))
                    p4f_gated = sam(qseg, phase4f_endpoint.to(torch.bfloat16))
                p1_decoder_exact &= torch.equal(p1_direct, p1_gated)
                phase4f_decoder_exact &= torch.equal(p4f_direct, p4f_gated)
                decoder_checked += 1
    p1_pass = p1_embedding_exact and p1_decoder_exact
    p4f_pass = phase4f_embedding_exact and phase4f_decoder_exact
    result = {
        "population_valid_g0_checked": checked,
        "decoder_subset_checked": decoder_checked,
        "P1_END_POINT": {"embedding_all_exact": p1_embedding_exact, "decoder_subset_exact": p1_decoder_exact, "pass": p1_pass},
        "PHASE4F_END_POINT": {"embedding_all_exact": phase4f_embedding_exact, "decoder_subset_exact": phase4f_decoder_exact, "pass": p4f_pass},
        "phase4f_evidence_matches_utility_cache_first_valid": first_evidence_exact,
        "frozen_state_hashes": {
            "utility_before": utility_hash, "utility_after": tensor_state_sha256(utility_model.state_dict()),
            "rectifier_before": rectifier_hash, "rectifier_after": tensor_state_sha256(rectifier.state_dict()),
            "sam_before": sam_hash, "sam_after": tensor_state_sha256(sam.state_dict()),
            "forensic_source_before": source_hash, "forensic_source_after": tensor_state_sha256(source.state_dict()),
        },
    }
    if not p1_pass or not p4f_pass:
        gates = {
            "P1_ENDPOINT_RECOVERY": "PASS" if p1_pass else "FAIL",
            "PHASE4F_ENDPOINT_RECOVERY": "PASS" if p4f_pass else "FAIL",
            "EVALUATION_COMPLETE": "NO",
            "STOP_REASON": "HARD_ENDPOINT_FAILURE",
            "INTERNAL_TEST_ACCESSED": "NO", "OFFICIAL1000_ACCESSED": "NO",
        }
        dump(RESULTS, {"schema": "phase4ha_results_v1", "status": "STOPPED_ENDPOINT_FAILURE", "endpoint_preflight": result})
        dump(GATES, gates)
        raise RuntimeError("Phase4H-A hard endpoint failure")
    return result


def evaluate_condition(store, sam, rectifier, source, utility_model, dev, mode, condition, device):
    ids = dev["sample_ids"]
    cross = {index: (index + 1) % len(ids) for index in range(len(ids))}
    permutation = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    records, gate_values = [], []
    off_exact = []
    with torch.no_grad():
        for index, sid in enumerate(ids):
            if not bool(dev["valid"][index]):
                row = invalid_record(sid, dev["original_masks"][index]); row["valid_g0"] = False; records.append(row)
                continue
            partner = cross[index] if condition == "cross_image" else index
            s64, raw, _, sc, cc, _ = store.batch([sid], device)
            if partner != index:
                _, raw, _, _, _, _ = store.batch([ids[partner]], device)
            evidence = evidence_feature(source, raw)
            utility_batch = full_inputs(dev, torch.tensor([index]), device)
            if condition == "cross_image":
                utility_batch["F24"] = dev["F24"][partner:partner + 1].to(device)
                utility_batch["z_F24"] = dev["z_F24"][partner:partner + 1].to(device)
            elif condition == "spatial_shuffle":
                evidence = evidence.flatten(2)[:, :, permutation].reshape_as(evidence)
                utility_batch["F24"] = utility_batch["F24"].flatten(2)[:, :, permutation].reshape_as(utility_batch["F24"])
                utility_batch["z_F24"] = utility_batch["z_F24"].flatten(2)[:, :, permutation].reshape_as(utility_batch["z_F24"])
            valid_evidence = torch.ones(1, 576, dtype=torch.bool, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                phase4f = rectifier(s64, evidence, sc, cc, valid_evidence)["image_embeddings"]
                utility = utility_code.utility_forward(utility_model, utility_batch)["U"]
            gate = gate_to_sam_grid(utility, sc)
            if condition == "forensic_off":
                gate = torch.zeros_like(gate)
            if float(gate.min()) < 0.0 or float(gate.max()) > 1.0:
                raise RuntimeError("utility gate left [0,1]")
            adapted = gated_embedding(s64, phase4f, gate)
            qseg = dev["q_seg"][index:index + 1].to(device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(qseg, adapted.to(torch.bfloat16))
                if condition == "forensic_off":
                    p1_low = sam(qseg, s64)
                    off_exact.append(torch.equal(low, p1_low))
            logits = inverse_sam_logits(low, dev["sam_geometries"][index])
            row = metric_record(sid, logits, dev["original_masks"][index]); row["valid_g0"] = True; records.append(row)
            gate_values.append(gate.detach().cpu())
            if (index + 1) % 200 == 0:
                print(json.dumps({"stage": "DEV_EVAL", "mode": mode, "condition": condition, "done": index + 1, "total": len(ids)}), flush=True)
    gates = torch.cat(gate_values) if gate_values else torch.empty(0)
    gate_summary = {
        "valid_images": len(gate_values),
        "mean": float(gates.mean()) if gates.numel() else None,
        "minimum": float(gates.min()) if gates.numel() else None,
        "maximum": float(gates.max()) if gates.numel() else None,
        "fraction_zero": float((gates == 0).float().mean()) if gates.numel() else None,
        "fraction_one": float((gates == 1).float().mean()) if gates.numel() else None,
    }
    return summarize(records), records, gate_summary, {"checked": len(off_exact), "all_exact": all(off_exact) if off_exact else None}


def run(device: torch.device) -> dict:
    if RESULTS.exists() or GATES.exists() or REPORT.exists():
        raise RuntimeError("Phase4H-A outputs already exist; automatic rerun forbidden")
    seed_all()
    prerequisite = json.loads((ROOT / "outputs/phase4g1s/gate_summary.json").read_text())
    required = {"MISMATCH_AWARE_UTILITY_PREFLIGHT": "PASS", "FORMAL_TRAINING_EXECUTED": "NO"}
    if any(prerequisite.get(key) != value for key, value in required.items()):
        raise RuntimeError("Phase4G-1S frozen utility prerequisite drift")
    store = Phase4FStore(CFG, "val")
    dev = {mode: load_dev(mode) for mode in ("g0", "phrase", "tf")}
    if any(value["sample_ids"] != store.sample_ids for value in dev.values()):
        raise RuntimeError("Phase4F/utility DEV identity mismatch")
    sam = load_sam_runtime(CFG, device)
    source = load_evidence_source(CFG, "forensic_rect", device)
    rectifier, rectifier_checkpoint = load_phase4f(device)
    utility_model = load_utility(device)
    if any(parameter.requires_grad for model in (sam, source, rectifier, utility_model) for parameter in model.parameters()):
        raise RuntimeError("Phase4H-A contains trainable parameters")
    endpoint = endpoint_preflight(store, sam, rectifier, source, utility_model, dev["g0"], device)
    jobs = [("g0", "matched"), ("phrase", "matched"), ("tf", "matched"),
            ("g0", "cross_image"), ("g0", "spatial_shuffle"), ("g0", "forensic_off")]
    metrics, records, gate_distributions, identities = {}, {}, {}, {}
    for mode, condition in jobs:
        key = {"phrase": "phrase", "tf": "tf"}.get(mode, condition)
        values = evaluate_condition(store, sam, rectifier, source, utility_model, dev[mode], mode, condition, device)
        metrics[key], records[key], gate_distributions[key], identities[key] = values
    ids = store.sample_ids
    p1 = {mode: baseline_records(mode, ids) for mode in ("g0", "phrase", "tf")}
    p4f = {mode: phase4f_records(mode) for mode in ("g0", "phrase", "tf")}
    statistics = {"adaptive_vs_P1": {}, "adaptive_vs_Phase4F": {}}
    for mode, key in (("g0", "matched"), ("phrase", "phrase"), ("tf", "tf")):
        statistics["adaptive_vs_P1"][mode] = compare(records[key], p1[mode], seed=SEED)
        statistics["adaptive_vs_Phase4F"][mode] = compare(records[key], p4f[mode], seed=SEED)
    statistics["matched_minus_cross"] = compare(records["matched"], records["cross_image"], seed=SEED)
    statistics["matched_minus_shuffle"] = compare(records["matched"], records["spatial_shuffle"], seed=SEED)
    frozen_hashes = endpoint["frozen_state_hashes"]
    frozen_integrity = all(frozen_hashes[key.replace("_before", "_after")] == value for key, value in frozen_hashes.items() if key.endswith("_before"))
    p1_endpoint = endpoint["P1_END_POINT"]["pass"]
    p4f_endpoint = endpoint["PHASE4F_END_POINT"]["pass"]
    off_exact = identities["forensic_off"]["all_exact"] is True
    adaptive_gt_p1 = metrics["matched"]["mean_foreground_iou"] > HISTORICAL["P1"]["g0"]
    phrase_gt_p4f = metrics["phrase"]["mean_foreground_iou"] > HISTORICAL["Phase4F"]["phrase"]
    tf_gt_p4f = metrics["tf"]["mean_foreground_iou"] > HISTORICAL["Phase4F"]["tf"]
    language_order = metrics["matched"]["mean_foreground_iou"] < metrics["phrase"]["mean_foreground_iou"] < metrics["tf"]["mean_foreground_iou"]
    matched_controls = statistics["matched_minus_cross"]["foreground_iou"]["bootstrap_95_ci"][0] > 0 and statistics["matched_minus_shuffle"]["foreground_iou"]["bootstrap_95_ci"][0] > 0
    tradeoff = adaptive_gt_p1 and phrase_gt_p4f and tf_gt_p4f and language_order
    gates = {
        "schema": "phase4ha_gate_summary_v1",
        "EVALUATION_COMPLETE": "YES",
        "P1_ENDPOINT_RECOVERY": "PASS" if p1_endpoint else "FAIL",
        "PHASE4F_ENDPOINT_RECOVERY": "PASS" if p4f_endpoint else "FAIL",
        "ADAPTIVE_BEATS_P1_G0": "YES" if adaptive_gt_p1 else "NO",
        "ADAPTIVE_PHRASE_BEATS_PHASE4F": "YES" if phrase_gt_p4f else "NO",
        "ADAPTIVE_TF_BEATS_PHASE4F": "YES" if tf_gt_p4f else "NO",
        "LANGUAGE_ORDER_PRESERVED": "YES" if language_order else "NO",
        "MATCHED_GT_CROSS_SHUFFLE": "YES" if matched_controls else "NO",
        "FORENSIC_OFF_EXACT_P1": "PASS" if off_exact else "FAIL",
        "FROZEN_SOURCE_INTEGRITY": "PASS" if frozen_integrity else "FAIL",
        "PARETO_TRADEOFF_IMPROVEMENT": "YES" if tradeoff else "NO",
        "NEXT_STEP": "NONE" if tradeoff else "ANALYZE_UTILITY_TO_PHASE4F_RECTIFICATION_STRENGTH_MAPPING_ONLY",
        "INTERNAL_TEST_ACCESSED": "NO", "OFFICIAL1000_ACCESSED": "NO",
    }
    result = {
        "schema": "phase4ha_results_v1", "status": "COMPLETE", "formal_optimizer_updates": 0,
        "definition": "S64_adapt = S64 + aligned(U_F) * (S64_rect_Phase4F - S64)",
        "alignment": "bilinear sample original-normalized U_F64 at original Phase4F SAM-token coordinates; zeros outside normalized image",
        "checkpoints": {
            "phase4f_rectifier": {"path": str(rectifier_checkpoint), "sha256": file_sha256(rectifier_checkpoint)},
            "phase4g1s_utility": {"path": str(UTILITY_CKPT), "sha256": file_sha256(UTILITY_CKPT)},
        },
        "endpoint_preflight": endpoint, "metrics": metrics, "records": records,
        "gate_distributions": gate_distributions, "statistics": statistics,
        "historical": HISTORICAL, "identities": identities,
        "evaluation_contract": {"population": 1106, "valid": 1078, "invalid_iou_zero": 28, "batch_size": 1, "threshold_logit": 0.0},
        "firewall": {"internal_test_accessed": False, "official1000_accessed": False},
    }
    dump(RESULTS, result)
    dump(GATES, gates)
    write_report(result, gates)
    return gates


def write_report(result: dict, gates: dict) -> None:
    m, s = result["metrics"], result["statistics"]
    def stat(block):
        value = block["foreground_iou"]
        return f"delta={value['mean_difference']:+.6f}, 95% CI [{value['bootstrap_95_ci'][0]:+.6f}, {value['bootstrap_95_ci'][1]:+.6f}], W/T/L={value['wins']}/{value['ties']}/{value['losses']}, Wilcoxon p={value['wilcoxon_pvalue']:.4g}"
    table = "\n".join([
        "| Path | G0 | Phrase | TF |", "|---|---:|---:|---:|",
        f"| P1 | {HISTORICAL['P1']['g0']:.6f} | {HISTORICAL['P1']['phrase']:.6f} | {HISTORICAL['P1']['tf']:.6f} |",
        f"| Phase4F | {HISTORICAL['Phase4F']['g0']:.6f} | {HISTORICAL['Phase4F']['phrase']:.6f} | {HISTORICAL['Phase4F']['tf']:.6f} |",
        f"| Adaptive Phase4F | {m['matched']['mean_foreground_iou']:.6f} | {m['phrase']['mean_foreground_iou']:.6f} | {m['tf']['mean_foreground_iou']:.6f} |",
    ])
    report = f"""# Phase 4H-A Phase4F-Preserving Utility-Gated Rectification

## Frozen protocol

本阶段没有训练、optimizer、可训练模块、gate tuning或sweep。直接复用原Phase4F forensic rectifier selected epoch9，并冻结Phase4G-1S epoch10 `U_F`。不使用CSCU-LF/ECoLaF final-mask fusion。定义为 `S64_adapt = S64 + g * (S64_rect_Phase4F - S64)`，其中 `g=U_F`；先把original-normalized U_F按SAM token coordinates对齐到原Phase4F padded S64 64×64网格。

硬端点：`P1_ENDPOINT_RECOVERY={gates['P1_ENDPOINT_RECOVERY']}`；`PHASE4F_ENDPOINT_RECOVERY={gates['PHASE4F_ENDPOINT_RECOVERY']}`。全1078 valid-G0 embedding逐tensor核验，另有16例decoder bit-exact核验。

## Development results

{table}

G0 controls：matched={m['matched']['mean_foreground_iou']:.6f}，cross={m['cross_image']['mean_foreground_iou']:.6f}，shuffle={m['spatial_shuffle']['mean_foreground_iou']:.6f}，forensic-off={m['forensic_off']['mean_foreground_iou']:.6f}。

## Paired statistics

- Adaptive vs P1 G0：{stat(s['adaptive_vs_P1']['g0'])}
- Adaptive vs Phase4F G0：{stat(s['adaptive_vs_Phase4F']['g0'])}
- Adaptive vs Phase4F Phrase：{stat(s['adaptive_vs_Phase4F']['phrase'])}
- Adaptive vs Phase4F TF：{stat(s['adaptive_vs_Phase4F']['tf'])}
- matched−cross：{stat(s['matched_minus_cross'])}
- matched−shuffle：{stat(s['matched_minus_shuffle'])}

## Answers

A. Adaptive G0高于P1：**{gates['ADAPTIVE_BEATS_P1_G0']}**。  
B. Phrase/TF高于Phase4F：**{gates['ADAPTIVE_PHRASE_BEATS_PHASE4F']} / {gates['ADAPTIVE_TF_BEATS_PHASE4F']}**。  
C. `G0 < Phrase < TF`：**{gates['LANGUAGE_ORDER_PRESERVED']}**。  
D. matched以paired CI lower>0优于cross/shuffle：**{gates['MATCHED_GT_CROSS_SHUFFLE']}**。  
E. forensic-off exact P1：**{gates['FORENSIC_OFF_EXACT_P1']}**。

`PARETO_TRADEOFF_IMPROVEMENT={gates['PARETO_TRADEOFF_IMPROVEMENT']}`。若失败，下一步严格限制为分析/重学U_F到Phase4F rectification strength的映射，不回到ECoLaF且不增加模块。

## Final gates

```json
{json.dumps(gates, ensure_ascii=False, indent=2)}
```

到此严格STOP。internal test与official1000未访问。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(json.dumps(run(device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
