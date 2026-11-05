#!/usr/bin/env python3
"""Authorized Phase G1-C reliability calibration preflight.

The supervisor is intentionally phase-bounded: it can build deterministic
frozen-source caches, fit only the two evidential heads on TRAIN-FIT, fit only
two scalar temperatures on TRAIN-CAL, and perform one frozen TRAIN-AUDIT read.
It has no development-validation, internal-test, official1000, G1-F, AHBFR,
or checkpoint-selection path.
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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.pcerf import (
    PCERF,
    LanguageEvidentialHead,
    ForensicEvidentialHead,
    dsmp_probability,
    ecolaf_fuse,
    evidence_to_dirichlet,
    normalized_cell_centers,
    replace_uncertainty,
    resample_clip_to_original_normalized,
    sam_lowres_to_original_normalized,
    tmc_expected_ce_kl,
)
from tools.phase4c_b import file_sha256
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import load_evidence_source, load_sam_runtime, q_index


CFG_PATH = ROOT / "configs/phase4g1_g1c_reliability_preflight.yaml"
CFG = yaml.safe_load(CFG_PATH.read_text())
P4F = yaml.safe_load((ROOT / CFG["source"]["phase4f_config"]).read_text())
OUT = ROOT / CFG["experiment"]["output_root"]
CKPT = Path(CFG["experiment"]["checkpoint_root"])
CACHE = Path(CFG["experiment"]["cache_root"])
SEED = int(CFG["experiment"]["seed"])
FOLD_CODE = {"TRAIN-FIT": 0, "TRAIN-CAL": 1, "TRAIN-AUDIT": 2}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def combined_head_hash(language, forensic) -> str:
    state = {f"language.{k}": v for k, v in language.state_dict().items()}
    state.update({f"forensic.{k}": v for k, v in forensic.state_dict().items()})
    return tensor_state_sha256(state)


def pack_bool(value: torch.Tensor) -> torch.Tensor:
    array = value.detach().cpu().numpy().astype(np.uint8).reshape(value.shape[0], -1)
    return torch.from_numpy(np.packbits(array, axis=1, bitorder="little").copy())


def unpack_bool(value: torch.Tensor, height: int = 256, width: int = 256) -> torch.Tensor:
    array = np.unpackbits(value.detach().cpu().numpy(), axis=1, count=height * width, bitorder="little")
    return torch.from_numpy(array.reshape(value.shape[0], height, width).astype(bool, copy=False))


def support_from_geometry(geometry: dict, output_hw=(256, 256)) -> torch.Tensor:
    height, width = output_hw
    centers = normalized_cell_centers(height, width)
    resized_h, resized_w = map(float, geometry["resized_hw"])
    top, left, bottom, right = map(float, geometry["crop_box_yxyx"])
    y = centers[..., 1] * resized_h
    x = centers[..., 0] * resized_w
    return (y >= top) & (y < bottom) & (x >= left) & (x < right)


def source_paths(source: str) -> list[Path]:
    root = ROOT / P4F["data"]["spatial_cache_root"] / "cache" / source / "train"
    complete = json.loads((root / "complete.json").read_text())
    paths = sorted(root.glob("shard_*.pt"))
    if complete["status"] != "COMPLETE" or len(paths) != int(complete["shards"]):
        raise RuntimeError(f"invalid frozen {source} cache")
    return paths


def frozen_preflight() -> tuple[dict, dict]:
    gate = json.loads((ROOT / CFG["source"]["gate_summary"]).read_text())
    required = {
        "FORMULA_PARITY": "PASS",
        "P1_LANGUAGE_ONLY_EXACT_RECOVERY": "PASS",
        "FORENSIC_VACUOUS_EXACT_RECOVERY": "PASS",
        "FORENSIC_OFF_EXACT_RECOVERY": "PASS",
        "GEOMETRY_VALIDATED": "YES",
        "VACUOUS_SUPPORT_VALIDATED": "YES",
        "GRADIENT_ISOLATION": "PASS",
        "NO_ORACLE_LEAKAGE": "PASS",
        "INVALID_G0_POLICY": "PASS",
        "PCERF_ARCHITECTURE_HARDENED": "YES",
        "G1_C_RELIABILITY_PREFLIGHT_JUSTIFIED": "YES",
        "PHASE4G1_FULL_TRAINING_JUSTIFIED": "NO",
    }
    if any(gate.get(key) != value for key, value in required.items()):
        raise RuntimeError("Phase 4G-0.5 gate prerequisite changed")
    split = json.loads((ROOT / CFG["source"]["split_manifest"]).read_text())
    expected = {"TRAIN-FIT": (6185, 6083), "TRAIN-CAL": (1326, 1304), "TRAIN-AUDIT": (1325, 1303)}
    for fold, (count, valid) in expected.items():
        row = split["groups"][fold]
        if int(row["n"]) != count or int(row["valid_g0"]) != valid:
            raise RuntimeError(f"frozen split drift: {fold}")
    p1_hash = file_sha256(Path(P4F["p1"]["checkpoint"]))
    forensic_hash = file_sha256(Path(P4F["evidence"]["forensic_checkpoint"]))
    if p1_hash != CFG["source"]["p1_checkpoint_sha256"] or forensic_hash != CFG["source"]["forensic_checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint hash drift")
    return gate, split


def build_cache(device: torch.device) -> dict:
    _, split = frozen_preflight()
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "source_cache_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("config_sha256") == file_sha256(CFG_PATH):
            if all(Path(row["path"]).is_file() and file_sha256(Path(row["path"])) == row["sha256"] for row in existing["shards"]):
                print(json.dumps({"cache": "REUSED", "shards": len(existing["shards"]), "n": existing["population_n"]}), flush=True)
                return existing
    membership = {sid: fold for fold, row in split["groups"].items() for sid in row["ids"]}
    sam_paths, clip_paths = source_paths("sam"), source_paths("clip")
    qcache, _ = q_index(Path(P4F["data"]["q_cache_root"]), "train_q_seg")
    sam_model = load_sam_runtime(P4F, device)
    forensic_model = load_evidence_source(P4F, "forensic_rect", device)
    sam_hash_before, forensic_hash_before = tensor_state_sha256(sam_model.state_dict()), tensor_state_sha256(forensic_model.state_dict())
    shards, population_ids, valid_count = [], [], 0
    started = time.time()
    with torch.no_grad():
        for shard_index, (sam_path, clip_path) in enumerate(zip(sam_paths, clip_paths)):
            sam = torch.load(sam_path, map_location="cpu", weights_only=False)
            clip = torch.load(clip_path, map_location="cpu", weights_only=False)
            sample_ids = [str(row["sample_id"]) for row in sam["records"]]
            if sample_ids != [str(row["sample_id"]) for row in clip["records"]]:
                raise RuntimeError("SAM/CLIP source ID mismatch")
            valid = torch.tensor([sid in qcache and qcache[sid][1] for sid in sample_ids], dtype=torch.bool)
            valid_positions = valid.nonzero(as_tuple=False)[:, 0].tolist()
            s64_raw = sam["features"].to(device=device, dtype=torch.bfloat16)
            raw_clip = clip["features"].to(device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                forensic_value = forensic_model(raw_clip, return_features=True)
            f24 = forensic_value["F_forensic"].detach().to(torch.bfloat16).cpu()
            z_f24 = forensic_value["logits"].detach().to(torch.bfloat16).cpu()
            s64 = torch.cat([
                sam_lowres_to_original_normalized(
                    s64_raw[i:i + 1], sam["records"][i]["geometry"], output_hw=(64, 64)
                ).to(torch.bfloat16).cpu()
                for i in range(len(sample_ids))
            ])
            q_seg = torch.full((len(sample_ids), 256), float("nan"), dtype=torch.bfloat16)
            z_l = torch.full((len(sample_ids), 1, 256, 256), float("nan"), dtype=torch.bfloat16)
            if valid_positions:
                q_valid = torch.stack([qcache[sample_ids[i]][0] for i in valid_positions]).to(device=device, dtype=torch.bfloat16)
                with torch.autocast(device_type=device.type, enabled=False):
                    z_raw = sam_model(q_valid, s64_raw[valid_positions])
                q_seg[valid_positions] = q_valid.cpu()
                for local, position in enumerate(valid_positions):
                    z_l[position] = sam_lowres_to_original_normalized(
                        z_raw[local:local + 1], sam["records"][position]["geometry"]
                    )[0].to(torch.bfloat16).cpu()
            target256 = torch.cat([
                sam_lowres_to_original_normalized(
                    sam["targets"][i:i + 1, None].float(), sam["records"][i]["geometry"], mode="nearest"
                ).bool().cpu()
                for i in range(len(sample_ids))
            ])[:, 0]
            target64 = F.interpolate(target256[:, None].float(), (64, 64), mode="nearest")[:, 0].to(torch.uint8)
            target24 = F.interpolate(clip["targets"][:, None].float(), (24, 24), mode="nearest")[:, 0].to(torch.uint8)
            support256 = torch.stack([support_from_geometry(row["geometry"]) for row in clip["records"]])
            payload = {
                "schema": CFG["source"]["cache_schema"],
                "sample_ids": sample_ids,
                "fold_codes": torch.tensor([FOLD_CODE[membership[sid]] for sid in sample_ids], dtype=torch.uint8),
                "valid_g0": valid,
                "q_seg": q_seg,
                "S64": s64,
                "z_L": z_l,
                "F24": f24,
                "z_F24": z_f24,
                "target64": target64,
                "target24": target24,
                "target256_packed": pack_bool(target256),
                "support256_packed": pack_bool(support256),
                "sam_geometries": [row["geometry"] for row in sam["records"]],
                "clip_geometries": [row["geometry"] for row in clip["records"]],
                "source_sam_shard": str(sam_path.resolve()),
                "source_clip_shard": str(clip_path.resolve()),
                "p1_checkpoint_sha256": CFG["source"]["p1_checkpoint_sha256"],
                "forensic_checkpoint_sha256": CFG["source"]["forensic_checkpoint_sha256"],
            }
            cache_path = CACHE / f"shard_{shard_index:06d}.pt"
            temporary = cache_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary); temporary.replace(cache_path)
            shard_row = {
                "index": shard_index,
                "path": str(cache_path),
                "sha256": file_sha256(cache_path),
                "n": len(sample_ids),
                "valid_g0": int(valid.sum()),
                "sample_ids_sha256": ids_hash(sample_ids),
            }
            shards.append(shard_row); population_ids.extend(sample_ids); valid_count += int(valid.sum())
            if (shard_index + 1) % 25 == 0 or shard_index + 1 == len(sam_paths):
                print(json.dumps({"cache_shards": shard_index + 1, "of": len(sam_paths), "samples": len(population_ids)}), flush=True)
    if len(population_ids) != 8836 or len(set(population_ids)) != 8836 or valid_count != 8690:
        raise RuntimeError("G1-C cache population mismatch")
    invariance = {
        "P1_SAM_state_hash_before": sam_hash_before,
        "P1_SAM_state_hash_after": tensor_state_sha256(sam_model.state_dict()),
        "forensic_state_hash_before": forensic_hash_before,
        "forensic_state_hash_after": tensor_state_sha256(forensic_model.state_dict()),
    }
    if invariance["P1_SAM_state_hash_before"] != invariance["P1_SAM_state_hash_after"] or invariance["forensic_state_hash_before"] != invariance["forensic_state_hash_after"]:
        raise RuntimeError("source module changed during cache generation")
    manifest = {
        "schema": "phase4g1_g1c_source_cache_manifest_v1",
        "status": "COMPLETE",
        "cache_root": str(CACHE),
        "config_sha256": file_sha256(CFG_PATH),
        "population": "canonical internal train Fake",
        "population_n": len(population_ids),
        "valid_g0": valid_count,
        "invalid_g0": len(population_ids) - valid_count,
        "population_ids_sha256": ids_hash(population_ids),
        "folds": {fold: {"n": row["n"], "valid_g0": row["valid_g0"], "ids_sha256": row["ids_sha256"]} for fold, row in split["groups"].items()},
        "invalid_representation": "q_seg and z_L are explicit NaN sentinels and are never consumed by LanguageHead",
        "tensor_contract": {
            "q_seg": ["N", 256], "S64": ["N", 256, 64, 64], "z_L": ["N", 1, 256, 256],
            "F24": ["N", 256, 24, 24], "z_F24": ["N", 1, 24, 24],
            "target64": ["N", 64, 64], "target24": ["N", 24, 24],
            "target256_packed": ["N", 8192], "support256_packed": ["N", 8192],
        },
        "checkpoint_hashes": {"P1": CFG["source"]["p1_checkpoint_sha256"], "Phase4C_A": CFG["source"]["forensic_checkpoint_sha256"]},
        "source_invariance": invariance,
        "shards": shards,
        "seconds": time.time() - started,
        "development_validation_accessed": False,
        "internal_test_accessed": False,
        "official1000_accessed": False,
    }
    dump(manifest_path, manifest)
    return manifest


def load_fold(fold: str, *, fields: tuple[str, ...], valid_only: bool = True) -> dict:
    _, split = frozen_preflight()
    desired = split["groups"][fold]["ids"]
    desired_set = set(desired)
    loaded_ids, tensors = [], {field: [] for field in fields if field not in ("clip_geometries", "sam_geometries")}
    clip_geometries, sam_geometries = [], []
    for path in sorted(CACHE.glob("shard_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        positions = [i for i, sid in enumerate(shard["sample_ids"]) if sid in desired_set and (not valid_only or bool(shard["valid_g0"][i]))]
        if not positions:
            continue
        loaded_ids.extend([shard["sample_ids"][i] for i in positions])
        index = torch.tensor(positions, dtype=torch.long)
        for field in tensors:
            tensors[field].append(shard[field].index_select(0, index))
        if "clip_geometries" in fields:
            clip_geometries.extend([shard["clip_geometries"][i] for i in positions])
        if "sam_geometries" in fields:
            sam_geometries.extend([shard["sam_geometries"][i] for i in positions])
    expected_ids = [sid for sid in desired if not valid_only or sid in set(loaded_ids)]
    if set(loaded_ids) != set(expected_ids) or len(loaded_ids) != len(set(loaded_ids)):
        raise RuntimeError(f"cache fold mismatch: {fold}")
    position_by_id = {sid: i for i, sid in enumerate(loaded_ids)}
    order = torch.tensor([position_by_id[sid] for sid in expected_ids], dtype=torch.long)
    result = {field: torch.cat(parts, dim=0).index_select(0, order) for field, parts in tensors.items()}
    if "clip_geometries" in fields:
        result["clip_geometries"] = [clip_geometries[i] for i in order.tolist()]
    if "sam_geometries" in fields:
        result["sam_geometries"] = [sam_geometries[i] for i in order.tolist()]
    result["sample_ids"] = expected_ids
    return result


def cache_parity(device: torch.device) -> dict:
    manifest = json.loads((OUT / "source_cache_manifest.json").read_text())
    if manifest["status"] != "COMPLETE":
        raise RuntimeError("complete cache required")
    _, split = frozen_preflight()
    valid_ids = set()
    for cache_path in sorted(CACHE.glob("shard_*.pt")):
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        valid_ids.update(sid for sid, valid in zip(cached["sample_ids"], cached["valid_g0"].tolist()) if valid)
    candidates = []
    for fold in ("TRAIN-FIT", "TRAIN-CAL"):
        candidates.extend([sid for sid in split["groups"][fold]["ids"] if sid in valid_ids][:8])
    selected = set(candidates)
    qcache, _ = q_index(Path(P4F["data"]["q_cache_root"]), "train_q_seg")
    sam_model = load_sam_runtime(P4F, device)
    forensic_model = load_evidence_source(P4F, "forensic_rect", device)
    rows = []
    with torch.no_grad():
        for cache_path in sorted(CACHE.glob("shard_*.pt")):
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            positions = [i for i, sid in enumerate(cached["sample_ids"]) if sid in selected and bool(cached["valid_g0"][i])]
            if not positions:
                continue
            sam = torch.load(cached["source_sam_shard"], map_location="cpu", weights_only=False)
            clip = torch.load(cached["source_clip_shard"], map_location="cpu", weights_only=False)
            raw_s64 = sam["features"][positions].to(device=device, dtype=torch.bfloat16)
            raw_clip = clip["features"][positions].to(device=device, dtype=torch.bfloat16)
            q = torch.stack([qcache[cached["sample_ids"][i]][0] for i in positions]).to(device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                forensic = forensic_model(raw_clip, return_features=True)
            with torch.autocast(device_type=device.type, enabled=False):
                zraw = sam_model(q, raw_s64)
            for local, position in enumerate(positions):
                direct_s64 = sam_lowres_to_original_normalized(raw_s64[local:local + 1], sam["records"][position]["geometry"], output_hw=(64, 64)).to(torch.bfloat16).cpu()[0]
                direct_zl = sam_lowres_to_original_normalized(zraw[local:local + 1], sam["records"][position]["geometry"]).to(torch.bfloat16).cpu()[0]
                checks = {
                    "q_seg_exact": torch.equal(cached["q_seg"][position], q[local].cpu()),
                    "S64_exact": torch.equal(cached["S64"][position], direct_s64),
                    "z_L_exact": torch.equal(cached["z_L"][position], direct_zl),
                    "F24_exact": torch.equal(cached["F24"][position], forensic["F_forensic"][local].to(torch.bfloat16).cpu()),
                    "z_F24_exact": torch.equal(cached["z_F24"][position], forensic["logits"][local].to(torch.bfloat16).cpu()),
                }
                rows.append({"sample_id": cached["sample_ids"][position], "checks": checks, "pass": all(checks.values())})
    result = {
        "schema": "phase4g1_g1c_cache_parity_v1",
        "status": "PASS" if len(rows) >= 12 and all(row["pass"] for row in rows) else "FAIL",
        "sample_rule": "first valid IDs from frozen TRAIN-FIT and TRAIN-CAL only; TRAIN-AUDIT excluded",
        "samples": rows,
        "P1_checkpoint_sha256": CFG["source"]["p1_checkpoint_sha256"],
        "forensic_checkpoint_sha256": CFG["source"]["forensic_checkpoint_sha256"],
        "development_validation_accessed": False,
        "internal_test_accessed": False,
        "official1000_accessed": False,
    }
    dump(OUT / "source_cache_parity.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("cache/direct-forward parity failed")
    return result


def save_fit_checkpoint(path: Path, epoch: int, updates: int, language, forensic, optimizer, initial_hash: str) -> None:
    payload = {
        "schema": "phase4g1_g1c_fit_checkpoint_v1",
        "epoch": epoch,
        "optimizer_updates": updates,
        "language_head": language.state_dict(),
        "forensic_head": forensic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "initial_head_hash": initial_hash,
        "current_head_hash": combined_head_hash(language, forensic),
        "checkpoint_rule": "epoch10_final_only_no_metric_selection",
        "not_deployment_checkpoint": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp"); torch.save(payload, temporary); temporary.replace(path)


def fit_heads(device: torch.device) -> dict:
    frozen_preflight()
    parity = json.loads((OUT / "source_cache_parity.json").read_text())
    if parity["status"] != "PASS":
        raise RuntimeError("PASS source cache parity required")
    data = load_fold("TRAIN-FIT", fields=("S64", "q_seg", "z_L", "F24", "z_F24", "target64", "target24"), valid_only=True)
    if len(data["sample_ids"]) != 6083:
        raise RuntimeError("TRAIN-FIT valid population mismatch")
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    language, forensic = LanguageEvidentialHead().to(device), ForensicEvidentialHead().to(device)
    initial_hash = combined_head_hash(language, forensic)
    optimizer = torch.optim.AdamW(
        list(language.parameters()) + list(forensic.parameters()),
        lr=float(CFG["fit"]["learning_rate"]), weight_decay=float(CFG["fit"]["weight_decay"]),
    )
    history_path = OUT / "fit_history.csv"
    existing = sorted(CKPT.glob("epoch_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
    start_epoch, updates = 0, 0
    if existing:
        state = torch.load(existing[-1], map_location="cpu", weights_only=False)
        language.load_state_dict(state["language_head"]); forensic.load_state_dict(state["forensic_head"])
        optimizer.load_state_dict(state["optimizer"]); start_epoch = int(state["epoch"]); updates = int(state["optimizer_updates"])
        if state["initial_head_hash"] != initial_hash:
            raise RuntimeError("fit initialization drift")
    fieldnames = [
        "epoch", "kl_coefficient", "optimizer_updates", "language_loss", "forensic_loss",
        "language_expected_ce", "forensic_expected_ce", "language_kl", "forensic_kl",
        "language_strength", "forensic_strength", "language_uncertainty", "forensic_uncertainty",
        "language_grad_norm", "forensic_grad_norm", "language_saturation_low_u", "forensic_saturation_low_u",
        "finite_count", "nonfinite_count", "seconds",
    ]
    if start_epoch == 0:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
    batch_size = int(CFG["fit"]["batch_size"]); n = len(data["sample_ids"])
    for epoch in range(start_epoch + 1, int(CFG["fit"]["epochs"]) + 1):
        began = time.time(); language.train(); forensic.train()
        generator = torch.Generator().manual_seed(SEED + 1009 * epoch)
        order = torch.randperm(n, generator=generator)
        sums = {key: 0.0 for key in fieldnames if key not in ("epoch", "kl_coefficient", "optimizer_updates", "finite_count", "nonfinite_count", "seconds")}
        sample_total = finite_count = nonfinite_count = 0
        coefficient = float(CFG["fit"]["kl_coefficients_by_epoch"][epoch - 1])
        for begin in range(0, n, batch_size):
            index = order[begin:begin + batch_size]; count = len(index)
            s64 = data["S64"].index_select(0, index).to(device=device, non_blocking=True)
            q = data["q_seg"].index_select(0, index).to(device=device, non_blocking=True)
            zl = data["z_L"].index_select(0, index).to(device=device, non_blocking=True)
            f24 = data["F24"].index_select(0, index).to(device=device, non_blocking=True)
            zf = data["z_F24"].index_select(0, index).to(device=device, non_blocking=True)
            target_l = data["target64"].index_select(0, index).to(device=device, non_blocking=True)
            target_f = data["target24"].index_select(0, index).to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            evidence_l = language(s64, q, zl)
            evidence_f = forensic(f24, zf)
            loss_l = tmc_expected_ce_kl(evidence_l, target_l, kl_coefficient=coefficient)
            loss_f = tmc_expected_ce_kl(evidence_f, target_f, kl_coefficient=coefficient)
            total = loss_l["total"] + loss_f["total"]
            finite = bool(torch.isfinite(total) and torch.isfinite(evidence_l).all() and torch.isfinite(evidence_f).all())
            if not finite:
                nonfinite_count += count
                raise RuntimeError(f"nonfinite TRAIN-FIT value at epoch {epoch}, offset {begin}")
            total.backward()
            language_grad = math.sqrt(sum(float(p.grad.detach().float().norm()) ** 2 for p in language.parameters() if p.grad is not None))
            forensic_grad = math.sqrt(sum(float(p.grad.detach().float().norm()) ** 2 for p in forensic.parameters() if p.grad is not None))
            torch.nn.utils.clip_grad_norm_(list(language.parameters()) + list(forensic.parameters()), float(CFG["fit"]["gradient_clip_norm"]))
            optimizer.step(); updates += 1; finite_count += count; sample_total += count
            opinion_l, opinion_f = evidence_to_dirichlet(evidence_l.detach()), evidence_to_dirichlet(evidence_f.detach())
            row = {
                "language_loss": float(loss_l["total"].detach()), "forensic_loss": float(loss_f["total"].detach()),
                "language_expected_ce": float(loss_l["expected_ce"].detach()), "forensic_expected_ce": float(loss_f["expected_ce"].detach()),
                "language_kl": float(loss_l["kl"].detach()), "forensic_kl": float(loss_f["kl"].detach()),
                "language_strength": float(opinion_l["strength"].mean()), "forensic_strength": float(opinion_f["strength"].mean()),
                "language_uncertainty": float(opinion_l["uncertainty"].mean()), "forensic_uncertainty": float(opinion_f["uncertainty"].mean()),
                "language_grad_norm": language_grad, "forensic_grad_norm": forensic_grad,
                "language_saturation_low_u": float((opinion_l["uncertainty"] < 0.01).float().mean()),
                "forensic_saturation_low_u": float((opinion_f["uncertainty"] < 0.01).float().mean()),
            }
            for key, value in row.items(): sums[key] += value * count
        summary = {key: value / sample_total for key, value in sums.items()}
        summary.update({"epoch": epoch, "kl_coefficient": coefficient, "optimizer_updates": updates, "finite_count": finite_count, "nonfinite_count": nonfinite_count, "seconds": time.time() - began})
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(summary)
        save_fit_checkpoint(CKPT / f"epoch_{epoch}.pt", epoch, updates, language, forensic, optimizer, initial_hash)
        print(json.dumps({"epoch": epoch, "updates": updates, "L": summary["language_loss"], "F": summary["forensic_loss"], "seconds": summary["seconds"]}), flush=True)
    final_state = torch.load(CKPT / "epoch_10.pt", map_location="cpu", weights_only=False)
    torch.save({"schema": "phase4g1_g1c_language_head_final_v1", "state_dict": final_state["language_head"], "epoch": 10, "not_deployment_checkpoint": True}, CKPT / "language_head_final.pt")
    torch.save({"schema": "phase4g1_g1c_forensic_head_final_v1", "state_dict": final_state["forensic_head"], "epoch": 10, "not_deployment_checkpoint": True}, CKPT / "forensic_head_final.pt")
    rows = list(csv.DictReader(history_path.open()))
    result = {
        "schema": "phase4g1_g1c_fit_manifest_v1",
        "TRAIN_FIT_COMPLETE": "YES" if len(rows) == 10 and int(rows[-1]["optimizer_updates"]) == 7610 else "NO",
        "population": "TRAIN-FIT valid canonical-G0",
        "population_n": n,
        "excluded_invalid_g0": 102,
        "optimizer": CFG["fit"],
        "final_epoch": 10,
        "optimizer_updates": int(rows[-1]["optimizer_updates"]),
        "initial_head_hash": initial_hash,
        "final_head_hash": final_state["current_head_hash"],
        "final_checkpoint": str((CKPT / "epoch_10.pt").resolve()),
        "language_checkpoint": str((CKPT / "language_head_final.pt").resolve()),
        "forensic_checkpoint": str((CKPT / "forensic_head_final.pt").resolve()),
        "checkpoint_selection": "epoch10 final by rule; no metric selection",
        "fused_loss_used": False,
        "development_validation_accessed": False,
        "internal_test_accessed": False,
        "official1000_accessed": False,
    }
    dump(OUT / "fit_manifest.json", result)
    if result["TRAIN_FIT_COMPLETE"] != "YES": raise RuntimeError("TRAIN-FIT incomplete")
    return result


def load_final_heads(device: torch.device):
    state = torch.load(CKPT / "epoch_10.pt", map_location="cpu", weights_only=False)
    language, forensic = LanguageEvidentialHead().to(device), ForensicEvidentialHead().to(device)
    language.load_state_dict(state["language_head"]); forensic.load_state_dict(state["forensic_head"])
    language.eval().requires_grad_(False); forensic.eval().requires_grad_(False)
    return language, forensic, state["current_head_hash"]


def collect_native_evidence(data: dict, language, forensic, device: torch.device, batch_size=8):
    left, right = [], []
    with torch.no_grad():
        for begin in range(0, len(data["sample_ids"]), batch_size):
            end = begin + batch_size
            left.append(language(data["S64"][begin:end].to(device), data["q_seg"][begin:end].to(device), data["z_L"][begin:end].to(device)).cpu())
            right.append(forensic(data["F24"][begin:end].to(device), data["z_F24"][begin:end].to(device)).cpu())
    return torch.cat(left), torch.cat(right)


def posterior_from_evidence(evidence: torch.Tensor, temperature: torch.Tensor | float):
    alpha = evidence.float() / temperature + 1.0
    return alpha / alpha.sum(dim=1, keepdim=True)


def ece_diagram(probability: torch.Tensor, target: torch.Tensor, bins: int = 15) -> tuple[float, list[dict]]:
    confidence, prediction = probability.max(dim=1)
    correctness = prediction.eq(target)
    rows, ece = [], 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        member = confidence.gt(lower) & (confidence.le(upper) if index + 1 < bins else confidence.le(1.0))
        count = int(member.sum())
        if count:
            accuracy = float(correctness[member].float().mean()); mean_confidence = float(confidence[member].mean())
            fraction = count / confidence.numel(); ece += abs(accuracy - mean_confidence) * fraction
        else:
            accuracy = mean_confidence = None
        rows.append({"bin": index, "lower": lower, "upper": upper, "count": count, "accuracy": accuracy, "confidence": mean_confidence})
    return ece, rows


def classification_metrics(probability: torch.Tensor, target: torch.Tensor, bins: int = 15) -> dict:
    target = target.long(); one_hot = F.one_hot(target, num_classes=2).movedim(-1, 1).float()
    selected = probability.gather(1, target[:, None]).clamp_min(1e-12)
    nll = float(-selected.log().mean())
    brier = float((probability - one_hot).square().sum(dim=1).mean())
    ece, diagram = ece_diagram(probability, target, bins)
    confidence, prediction = probability.max(dim=1); high = confidence >= float(CFG["audit"]["high_confidence_threshold"])
    high_error = float((prediction[high] != target[high]).float().mean()) if bool(high.any()) else None
    return {"nll": nll, "brier": brier, "ece": ece, "high_confidence_error_rate": high_error, "high_confidence_pixels": int(high.sum()), "reliability_diagram": diagram}


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value - float(CFG["calibration"]["temperature_epsilon"])))


def calibrate(device: torch.device) -> dict:
    fit = json.loads((OUT / "fit_manifest.json").read_text())
    if fit["TRAIN_FIT_COMPLETE"] != "YES": raise RuntimeError("completed TRAIN-FIT required")
    data = load_fold("TRAIN-CAL", fields=("S64", "q_seg", "z_L", "F24", "z_F24", "target64", "target24"), valid_only=True)
    if len(data["sample_ids"]) != 1304: raise RuntimeError("TRAIN-CAL valid population mismatch")
    language, forensic, head_hash = load_final_heads(device)
    evidence_l, evidence_f = collect_native_evidence(data, language, forensic, device)
    rows = {}
    temperatures = {}
    for name, evidence, target in (("language", evidence_l, data["target64"]), ("forensic", evidence_f, data["target24"])):
        evidence = evidence.to(device); target = target.to(device)
        tau = torch.nn.Parameter(torch.tensor(inverse_softplus(1.0), device=device))
        optimizer = torch.optim.LBFGS([tau], lr=float(CFG["calibration"]["learning_rate"]), max_iter=int(CFG["calibration"]["max_iter"]), line_search_fn=CFG["calibration"]["line_search_fn"])
        pre_probability = posterior_from_evidence(evidence, 1.0)
        pre = classification_metrics(pre_probability.cpu(), target.cpu(), int(CFG["audit"]["ece_bins"]))
        calls = 0
        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True); calls += 1
            temperature = F.softplus(tau) + float(CFG["calibration"]["temperature_epsilon"])
            probability = posterior_from_evidence(evidence, temperature)
            selected = probability.gather(1, target[:, None].long()).clamp_min(1e-12)
            loss = -selected.log().mean(); loss.backward(); return loss
        optimizer.step(closure)
        temperature = float((F.softplus(tau) + float(CFG["calibration"]["temperature_epsilon"])).detach())
        post_probability = posterior_from_evidence(evidence, temperature)
        post = classification_metrics(post_probability.cpu(), target.cpu(), int(CFG["audit"]["ece_bins"]))
        temperatures[name] = temperature
        rows[name] = {"temperature": temperature, "pre": pre, "post": post, "lbfgs_closure_calls": calls, "nll_non_increasing": post["nll"] <= pre["nll"] + 1e-10}
    passed = all(row["nll_non_increasing"] for row in rows.values())
    result = {
        "schema": "phase4g1_g1c_temperature_calibration_v1",
        "CALIBRATION_OPTIMIZATION": "PASS" if passed else "FAIL",
        "population": "TRAIN-CAL valid canonical-G0",
        "population_n": len(data["sample_ids"]),
        "excluded_invalid_g0": 22,
        "head_hash_before": head_hash,
        "head_hash_after": combined_head_hash(language, forensic),
        "objective": "source posterior NLL",
        "optimizer": CFG["calibration"],
        "sources": rows,
        "temperatures": {"T_L": temperatures["language"], "T_F": temperatures["forensic"]},
        "temperature_selection": "deterministic LBFGS only; no grid/sweep/IoU",
        "development_validation_accessed": False,
        "internal_test_accessed": False,
        "official1000_accessed": False,
    }
    if result["head_hash_before"] != result["head_hash_after"]: raise RuntimeError("heads changed during calibration")
    dump(OUT / "temperature_calibration.json", result)
    torch.save({"schema": "phase4g1_g1c_temperature_scalars_v1", "T_L": temperatures["language"], "T_F": temperatures["forensic"], "not_deployment_checkpoint": True}, CKPT / "temperature_scalars.pt")
    if not passed: print(json.dumps(result, indent=2), flush=True)
    return result


def mass_probability(mass: torch.Tensor) -> torch.Tensor:
    return mass[:, :-1] + mass[:, -1:] / float(mass.shape[1] - 1)


def boundary_mask(target: torch.Tensor) -> torch.Tensor:
    value = target[:, None].float()
    dilated = F.max_pool2d(value, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-value, 3, stride=1, padding=1)
    return (dilated != eroded)[:, 0]


def scoped_metrics(probability: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, uncertainty: torch.Tensor) -> dict:
    if not bool(mask.any()):
        raise RuntimeError("empty formal pixel scope")
    p = probability.permute(0, 2, 3, 1)[mask]
    y = target[mask]
    u = uncertainty[:, 0][mask]
    result = classification_metrics(p[:, :, None, None], y[:, None, None], int(CFG["audit"]["ece_bins"]))
    error = p.argmax(dim=1).ne(y)
    result.update({
        "pixels": int(mask.sum()),
        "mean_uncertainty": float(u.mean()),
        "mean_uncertainty_correct": float(u[~error].mean()) if bool((~error).any()) else None,
        "mean_uncertainty_error": float(u[error].mean()) if bool(error.any()) else None,
        "source_pixel_error_rate": float(error.float().mean()),
    })
    return result


def map_language(value: torch.Tensor) -> torch.Tensor:
    mapped = F.interpolate(value.float(), (256, 256), mode="bilinear", align_corners=False)
    return mapped / mapped.sum(dim=1, keepdim=True).clamp_min(1e-12)


def map_forensic(value: torch.Tensor, geometries: list[dict], *, vacuous: bool) -> tuple[torch.Tensor, torch.Tensor]:
    rows, supports = [], []
    for index, geometry in enumerate(geometries):
        mapped, support = resample_clip_to_original_normalized(value[index:index + 1], geometry, vacuous=vacuous)
        rows.append(mapped.cpu()); supports.append(support.cpu())
    return torch.cat(rows), torch.cat(supports)[:, 0]


def fusion_statistics(mass_l: torch.Tensor, mass_f: torch.Tensor, support: torch.Tensor, batch_size: int = 8) -> dict:
    output = {key: [] for key in ("probability", "conflict_l", "conflict_f", "discount_l", "discount_f", "weight_l", "weight_f")}
    for begin in range(0, len(mass_l), batch_size):
        ml, mf = mass_l[begin:begin + batch_size], mass_f[begin:begin + batch_size]
        fused = ecolaf_fuse(torch.stack((ml, mf), dim=2), classes=2)
        discount = fused["discount"]
        committed = torch.stack((1.0 - ml[:, -1], 1.0 - mf[:, -1]), dim=1)
        weighted = discount * committed
        weights = weighted / weighted.sum(dim=1, keepdim=True).clamp_min(1e-12)
        inactive = ~support[begin:begin + batch_size]
        weights[:, 1] = torch.where(inactive, torch.zeros_like(weights[:, 1]), weights[:, 1])
        weights[:, 0] = torch.where(inactive, torch.ones_like(weights[:, 0]), weights[:, 0])
        for key, tensor in {
            "probability": fused["probability"], "conflict_l": fused["conflict"][:, 0:1],
            "conflict_f": fused["conflict"][:, 1:2], "discount_l": discount[:, 0:1],
            "discount_f": discount[:, 1:2], "weight_l": weights[:, 0:1], "weight_f": weights[:, 1:2],
        }.items(): output[key].append(tensor.cpu())
    return {key: torch.cat(parts) for key, parts in output.items()}


def fg_iou(probability: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    prediction = probability.argmax(dim=1).bool()
    if mask is None: mask = torch.ones_like(target, dtype=torch.bool)
    intersection = (prediction & target & mask).flatten(1).sum(1).float()
    union = ((prediction | target) & mask).flatten(1).sum(1).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def per_image_nll(probability: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    selected = probability.gather(1, target[:, None].long()).clamp_min(1e-12)[:, 0]
    if mask is None: mask = torch.ones_like(target, dtype=torch.bool)
    return -(selected.log() * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)


def bootstrap_correlations(values: dict[str, np.ndarray], repeats: int, seed: int) -> dict:
    names = list(values); matrix = np.column_stack([values[name] for name in names]).astype(np.float64)
    rng = np.random.default_rng(seed); n, width = matrix.shape
    pearson = np.empty((repeats, width, width), dtype=np.float32)
    spearman = np.empty_like(pearson)
    for begin in range(0, repeats, 100):
        count = min(100, repeats - begin); index = rng.integers(0, n, size=(count, n))
        sampled_raw = matrix[index]
        sampled_rank = np.stack([stats.rankdata(sampled_raw[:, :, column], axis=1) for column in range(width)], axis=2)
        for sampled, destination in ((sampled_raw, pearson), (sampled_rank, spearman)):
            centered = sampled - sampled.mean(axis=1, keepdims=True)
            covariance = np.einsum("bni,bnj->bij", centered, centered)
            scale = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
            destination[begin:begin + count] = covariance / np.maximum(scale[:, :, None] * scale[:, None, :], 1e-30)
    result = {}
    point_p = np.corrcoef(matrix, rowvar=False); point_s = stats.spearmanr(matrix, axis=0).statistic
    for i, left in enumerate(names):
        result[left] = {}
        for j, right in enumerate(names):
            result[left][right] = {
                "pearson": {"point": float(point_p[i, j]), "ci95": np.quantile(pearson[:, i, j], [.025, .975]).tolist()},
                "spearman": {"point": float(point_s[i, j]), "ci95": np.quantile(spearman[:, i, j], [.025, .975]).tolist(), "bootstrap_method": "paired image bootstrap with ranks recomputed per resample"},
            }
    return result


def relation_gate(correlation: dict, *, positive: bool) -> str:
    point = correlation["point"]; lower, upper = correlation["ci95"]
    if positive:
        return "PASS" if point > 0 and lower > 0 else ("INCONCLUSIVE" if point > 0 else "FAIL")
    return "PASS" if point < 0 and upper < 0 else ("INCONCLUSIVE" if point < 0 else "FAIL")


def paired_delta(values: torch.Tensor, seed: int) -> dict:
    array = values.double().numpy(); rng = np.random.default_rng(seed); means = np.empty(int(CFG["audit"]["bootstrap_repeats"]), dtype=np.float64)
    for begin in range(0, len(means), 500):
        count = min(500, len(means) - begin)
        means[begin:begin + count] = array[rng.integers(0, len(array), size=(count, len(array)))].mean(axis=1)
    point = float(array.mean()); ci = np.quantile(means, [.025, .975]).tolist()
    return {"mean_delta": point, "ci95": ci, "gate": "PASS" if point < 0 and ci[1] < 0 else ("INCONCLUSIVE" if point < 0 else "FAIL")}


def opinion_maps(evidence_l: torch.Tensor, evidence_f: torch.Tensor, geometries: list[dict], t_l: float, t_f: float):
    native_l = evidence_to_dirichlet(evidence_l.float() / t_l)
    native_f = evidence_to_dirichlet(evidence_f.float() / t_f)
    mass_l = map_language(native_l["masses"])
    mass_f, support = map_forensic(native_f["masses"], geometries, vacuous=True)
    strength_l = F.interpolate(native_l["strength"], (256, 256), mode="bilinear", align_corners=False)
    strength_f, _ = map_forensic(native_f["strength"], geometries, vacuous=False)
    strength_f = torch.where(support[:, None], strength_f, torch.full_like(strength_f, 2.0))
    return mass_l, mass_f, support, strength_l, strength_f


def audit(device: torch.device) -> dict:
    marker_path = OUT / "audit_access.json"
    if marker_path.exists():
        raise RuntimeError("TRAIN-AUDIT formal access already consumed; rerun is forbidden")
    calibration = json.loads((OUT / "temperature_calibration.json").read_text())
    if calibration["CALIBRATION_OPTIMIZATION"] != "PASS":
        raise RuntimeError("successful TRAIN-CAL optimization required before formal audit")
    marker = {
        "schema": "phase4g1_g1c_audit_access_v1", "access_count": 1, "status": "STARTED",
        "rule": "formal TRAIN-AUDIT cache load is single-use even after interruption",
        "started_unix": time.time(),
    }
    dump(marker_path, marker)
    # This is the sole formal TRAIN-AUDIT cache traversal. All subsequent
    # controls operate only on these in-memory tensors.
    data = load_fold(
        "TRAIN-AUDIT",
        fields=("S64", "q_seg", "z_L", "F24", "z_F24", "target256_packed", "support256_packed", "clip_geometries"),
        valid_only=True,
    )
    if len(data["sample_ids"]) != 1303:
        raise RuntimeError("TRAIN-AUDIT valid population mismatch")
    marker.update({"status": "LOADED_IN_MEMORY", "population_n": 1303, "sample_ids_sha256": ids_hash(data["sample_ids"])})
    dump(marker_path, marker)
    target = unpack_bool(data["target256_packed"]); cached_support = unpack_bool(data["support256_packed"])
    language, forensic, head_hash = load_final_heads(device)
    t_l, t_f = calibration["temperatures"]["T_L"], calibration["temperatures"]["T_F"]
    evidence_l, evidence_f = collect_native_evidence(data, language, forensic, device)
    mass_l, mass_f, support, strength_l, strength_f = opinion_maps(evidence_l, evidence_f, data["clip_geometries"], t_l, t_f)
    if not torch.equal(support, cached_support):
        raise RuntimeError("cached forensic support drift")
    probability_l, probability_f = mass_probability(mass_l), mass_probability(mass_f)
    pre_l, pre_f, _, _, _ = opinion_maps(evidence_l, evidence_f, data["clip_geometries"], 1.0, 1.0)
    pre_probability_l, pre_probability_f = mass_probability(pre_l), mass_probability(pre_f)
    calibration_audit = {}
    for name, pre_probability, post_probability, scope in (
        ("language", pre_probability_l, probability_l, torch.ones_like(target)),
        ("forensic", pre_probability_f, probability_f, support),
    ):
        pre = scoped_metrics(pre_probability, target, scope, (pre_l if name == "language" else pre_f)[:, -1:])
        post = scoped_metrics(post_probability, target, scope, (mass_l if name == "language" else mass_f)[:, -1:])
        regression = (post["nll"] - pre["nll"]) / pre["nll"]
        calibration_audit[name] = {"pre": pre, "post": post, "relative_nll_regression": regression, "non_degrading": regression <= .01}
    calibration["TRAIN_AUDIT_generalization"] = calibration_audit
    calibration["CALIBRATION_NON_DEGRADING"] = "PASS" if all(row["non_degrading"] for row in calibration_audit.values()) else "FAIL"
    dump(OUT / "temperature_calibration.json", calibration)
    del pre_l, pre_f, pre_probability_l, pre_probability_f

    boundary = boundary_mask(target)
    pixel = {
        "schema": "phase4g1_g1c_pixel_metrics_v1", "population_n": 1303,
        "scope_definition": {"boundary": "3x3 morphological gradient", "forensic": "intersect CLIP support"}, "sources": {},
    }
    for name, probability, uncertainty, valid_scope in (
        ("language", probability_l, mass_l[:, -1:], torch.ones_like(target)),
        ("forensic", probability_f, mass_f[:, -1:], support),
    ):
        pixel["sources"][name] = {
            "all": scoped_metrics(probability, target, valid_scope, uncertainty),
            "foreground": scoped_metrics(probability, target, valid_scope & target, uncertainty),
            "background": scoped_metrics(probability, target, valid_scope & ~target, uncertainty),
            "boundary": scoped_metrics(probability, target, valid_scope & boundary, uncertainty),
        }
    dump(OUT / "pixel_metrics.json", pixel)

    matched = fusion_statistics(mass_l, mass_f, support)
    image_arrays, image_metrics = {}, {"schema": "phase4g1_g1c_image_metrics_v1", "population_n": 1303, "sources": {}}
    for name, probability, mass, strength, scope in (
        ("language", probability_l, mass_l, strength_l, torch.ones_like(target)),
        ("forensic", probability_f, mass_f, strength_f, support),
    ):
        suffix = "l" if name == "language" else "f"
        arrays = {
            "uncertainty": ((mass[:, -1] * scope).flatten(1).sum(1) / scope.flatten(1).sum(1)),
            "strength": ((strength[:, 0] * scope).flatten(1).sum(1) / scope.flatten(1).sum(1)),
            "weight": ((matched[f"weight_{suffix}"][:, 0] * scope).flatten(1).sum(1) / scope.flatten(1).sum(1)),
            "discount": ((matched[f"discount_{suffix}"][:, 0] * scope).flatten(1).sum(1) / scope.flatten(1).sum(1)),
            "fg_iou": fg_iou(probability, target, scope),
            "fg_error": 1.0 - fg_iou(probability, target, scope),
            "source_nll": per_image_nll(probability, target, scope),
        }
        image_arrays[name] = arrays
        correlations = bootstrap_correlations({key: value.numpy() for key, value in arrays.items()}, int(CFG["audit"]["bootstrap_repeats"]), SEED + (0 if name == "language" else 1))
        image_metrics["sources"][name] = {
            "means": {key: float(value.mean()) for key, value in arrays.items()},
            "correlations": correlations,
        }
    dump(OUT / "image_metrics.json", image_metrics)
    qmf = {"schema": "phase4g1_g1c_qmf_metrics_v1", "weight_definition": CFG["audit"]["effective_weight"], "sources": {}}
    uncertainty_gates, qmf_gates = {}, {}
    for name in ("language", "forensic"):
        correlations = image_metrics["sources"][name]["correlations"]
        uncertainty = correlations["uncertainty"]["fg_error"]["spearman"]
        qmf_relation = correlations["weight"]["source_nll"]
        uncertainty_gates[name] = relation_gate(uncertainty, positive=True)
        qmf_component_gates = {kind: relation_gate(qmf_relation[kind], positive=False) for kind in ("pearson", "spearman")}
        qmf_gates[name] = "PASS" if all(value == "PASS" for value in qmf_component_gates.values()) else ("FAIL" if "FAIL" in qmf_component_gates.values() else "INCONCLUSIVE")
        qmf["sources"][name] = {"weight_vs_source_nll": qmf_relation, "formal_component_gates": qmf_component_gates, "formal_combined_gate": qmf_gates[name]}
    dump(OUT / "qmf_metrics.json", qmf)

    # Frozen forensic corruptions; the P1 language masses never change.
    permutation = torch.randperm(24 * 24, generator=torch.Generator().manual_seed(SEED))
    donor = torch.roll(torch.arange(1303), shifts=-1)
    corruption_masses = {}
    with torch.no_grad():
        for name, f_value, z_value in (
            ("cross_image", data["F24"].index_select(0, donor), data["z_F24"].index_select(0, donor)),
            ("spatial_shuffle", data["F24"].flatten(2)[:, :, permutation].reshape_as(data["F24"]), data["z_F24"].flatten(2)[:, :, permutation].reshape_as(data["z_F24"])),
        ):
            parts = []
            for begin in range(0, 1303, 8):
                parts.append(forensic(f_value[begin:begin + 8].to(device), z_value[begin:begin + 8].to(device)).cpu())
            corrupted_evidence = torch.cat(parts)
            _, mapped, corrupted_support, _, corrupted_strength = opinion_maps(evidence_l, corrupted_evidence, data["clip_geometries"], t_l, t_f)
            if not torch.equal(corrupted_support, support): raise RuntimeError("corruption changed support")
            corruption_masses[name] = (mapped, corrupted_strength)
    matched_forensic = {
        "uncertainty": image_arrays["forensic"]["uncertainty"], "strength": image_arrays["forensic"]["strength"],
        "weight": image_arrays["forensic"]["weight"], "discount": image_arrays["forensic"]["discount"],
        "conflict": (matched["conflict_f"][:, 0] * support).flatten(1).sum(1) / support.flatten(1).sum(1),
    }
    corruption = {"schema": "phase4g1_g1c_corruption_metrics_v1", "primary_metric": "forensic effective source weight", "conditions": {"matched": {key: float(value.mean()) for key, value in matched_forensic.items()}}, "paired_responses": {}}
    response_gates = {}
    for offset, (name, (corrupted_mass, corrupted_strength)) in enumerate(corruption_masses.items()):
        fused = fusion_statistics(mass_l, corrupted_mass, support)
        rows = {
            "uncertainty": (corrupted_mass[:, -1] * support).flatten(1).sum(1) / support.flatten(1).sum(1),
            "strength": (corrupted_strength[:, 0] * support).flatten(1).sum(1) / support.flatten(1).sum(1),
            "weight": (fused["weight_f"][:, 0] * support).flatten(1).sum(1) / support.flatten(1).sum(1),
            "discount": (fused["discount_f"][:, 0] * support).flatten(1).sum(1) / support.flatten(1).sum(1),
            "conflict": (fused["conflict_f"][:, 0] * support).flatten(1).sum(1) / support.flatten(1).sum(1),
        }
        primary = paired_delta(rows["weight"] - matched_forensic["weight"], SEED + 10 + offset)
        response_gates[name] = primary["gate"]
        corruption["conditions"][name] = {key: float(value.mean()) for key, value in rows.items()}
        corruption["paired_responses"][name] = {"effective_weight_delta": primary, "mean_deltas": {key: float((value - matched_forensic[key]).mean()) for key, value in rows.items()}}
    vacuous_mass = torch.zeros_like(mass_f); vacuous_mass[:, -1] = 1.0
    vacuous_fusion = fusion_statistics(mass_l, vacuous_mass, support)
    corruption["conditions"]["zero_vacuous"] = {
        "uncertainty": 1.0, "strength": 2.0,
        "weight": float(vacuous_fusion["weight_f"].mean()),
        "discount": float(vacuous_fusion["discount_f"].mean()),
        "conflict": float(vacuous_fusion["conflict_f"].mean()),
        "dispatch": "exact P1 independently of diagnostic ECoLaF quantities",
    }
    corruption["CROSS_IMAGE_RESPONSE"] = response_gates["cross_image"]
    corruption["SPATIAL_SHUFFLE_RESPONSE"] = response_gates["spatial_shuffle"]
    dump(OUT / "corruption_metrics.json", corruption)

    # Reliability-only permutations preserve each source's class-belief direction.
    disagreement = probability_l.argmax(1).ne(probability_f.argmax(1)) & support
    baseline_dominance = matched["weight_f"][:, 0] > matched["weight_l"][:, 0]
    perm_results = {}
    image_perm = torch.randperm(1303, generator=torch.Generator().manual_seed(SEED))
    spatial_perm = torch.randperm(256 * 256, generator=torch.Generator().manual_seed(SEED))
    variants = {
        "image_level": (replace_uncertainty(mass_l, mass_l[image_perm, -1:]), replace_uncertainty(mass_f, mass_f[image_perm, -1:])),
        "spatial": (replace_uncertainty(mass_l, mass_l[:, -1:].flatten(2)[:, :, spatial_perm].reshape_as(mass_l[:, -1:])), replace_uncertainty(mass_f, mass_f[:, -1:].flatten(2)[:, :, spatial_perm].reshape_as(mass_f[:, -1:]))),
    }
    for name, (permuted_l, permuted_f) in variants.items():
        permuted_f = torch.where(support[:, None], permuted_f, torch.cat((torch.zeros_like(permuted_f[:, :2]), torch.ones_like(permuted_f[:, -1:])), dim=1))
        if not torch.equal(mass_probability(permuted_l).argmax(1), probability_l.argmax(1)) or not torch.equal(mass_probability(permuted_f).argmax(1)[support], probability_f.argmax(1)[support]):
            raise RuntimeError("reliability permutation changed source prediction")
        fused = fusion_statistics(permuted_l, permuted_f, support)
        dominance = fused["weight_f"][:, 0] > fused["weight_l"][:, 0]
        prob_change = (fused["probability"][:, 1] - matched["probability"][:, 1]).abs()
        weight_change = (fused["weight_f"][:, 0] - matched["weight_f"][:, 0]).abs()
        perm_results[name] = {
            "disagreement_pixels": int(disagreement.sum()),
            "mean_absolute_fused_fg_probability_change": float(prob_change[disagreement].mean()),
            "source_dominance_assignment_change_rate": float(dominance[disagreement].ne(baseline_dominance[disagreement]).float().mean()),
            "mean_absolute_effective_forensic_weight_change": float(weight_change[disagreement].mean()),
            "source_predictions_unchanged": True,
        }
    permutation_pass = all(row["mean_absolute_fused_fg_probability_change"] >= .01 for row in perm_results.values()) and any(row["source_dominance_assignment_change_rate"] >= .05 for row in perm_results.values())
    permutation_metrics = {"schema": "phase4g1_g1c_permutation_metrics_v1", "seed": SEED, "conditions": perm_results, "PERMUTATION_SENSITIVITY": "PASS" if permutation_pass else "FAIL"}
    dump(OUT / "permutation_metrics.json", permutation_metrics)

    # Re-run the identity dispatch with calibrated heads/temperatures.
    pcerf = PCERF().to(device); pcerf.language_head.load_state_dict(language.state_dict()); pcerf.forensic_head.load_state_dict(forensic.state_dict()); pcerf.eval().requires_grad_(False)
    with torch.no_grad():
        check = pcerf(
            s64=data["S64"][:2].to(device), q_seg=data["q_seg"][:2].to(device), z_l=data["z_L"][:2].to(device),
            f24=data["F24"][:2].to(device), z_f24=data["z_F24"][:2].to(device), valid_g0=torch.ones(2, dtype=torch.bool, device=device),
            forensic_present=torch.ones(2, dtype=torch.bool, device=device), forensic_vacuous=torch.ones(2, dtype=torch.bool, device=device),
            forensic_off=torch.zeros(2, dtype=torch.bool, device=device), clip_geometries=data["clip_geometries"][:2], temperature_l=t_l, temperature_f=t_f,
        )
    vacuous_gate = "PASS" if torch.equal(check["logits"].cpu(), data["z_L"][:2]) else "FAIL"
    after_hash = combined_head_hash(language, forensic)
    temp_state = torch.load(CKPT / "temperature_scalars.pt", map_location="cpu", weights_only=False)
    if after_hash != head_hash or temp_state["T_L"] != t_l or temp_state["T_F"] != t_f:
        raise RuntimeError("frozen audit parameters changed")
    overall_corruption = "PASS" if all(value == "PASS" for value in response_gates.values()) else ("FAIL" if any(value == "FAIL" for value in response_gates.values()) else "INCONCLUSIVE")
    formal = {
        "TRAIN_FIT_COMPLETE": "YES", "CALIBRATION_OPTIMIZATION": calibration["CALIBRATION_OPTIMIZATION"],
        "CALIBRATION_NON_DEGRADING": calibration["CALIBRATION_NON_DEGRADING"],
        "LANGUAGE_UNCERTAINTY_ERROR_RELATION": uncertainty_gates["language"], "FORENSIC_UNCERTAINTY_ERROR_RELATION": uncertainty_gates["forensic"],
        "LANGUAGE_QMF_RELATION": qmf_gates["language"], "FORENSIC_QMF_RELATION": qmf_gates["forensic"],
        "CROSS_IMAGE_RESPONSE": response_gates["cross_image"], "SPATIAL_SHUFFLE_RESPONSE": response_gates["spatial_shuffle"],
        "PERMUTATION_SENSITIVITY": permutation_metrics["PERMUTATION_SENSITIVITY"], "VACUOUS_EXACT_RECOVERY": vacuous_gate,
        "NO_ORACLE_LEAKAGE": "PASS", "INVALID_G0_POLICY": "PASS",
    }
    required = [value for key, value in formal.items() if key != "TRAIN_FIT_COMPLETE"]
    overall = "FAIL" if "FAIL" in required else ("INCONCLUSIVE" if "INCONCLUSIVE" in required else "PASS")
    gates = {
        "schema": "phase4g1_g1c_gate_summary_v1", **formal,
        "UNCERTAINTY_ERROR_RELATION": "PASS" if all(v == "PASS" for v in uncertainty_gates.values()) else ("FAIL" if "FAIL" in uncertainty_gates.values() else "INCONCLUSIVE"),
        "QMF_RELATION": "PASS" if all(v == "PASS" for v in qmf_gates.values()) else ("FAIL" if "FAIL" in qmf_gates.values() else "INCONCLUSIVE"),
        "CORRUPTION_RESPONSE": overall_corruption, "RELIABILITY_PREFLIGHT": overall,
        "G1_F_FULL_TRAINING_JUSTIFIED": "YES" if overall == "PASS" else "NO",
        "TRAINING_BALANCE_AUDIT_REQUIRED": "YES", "TRAINING_BALANCE_INTERVENTION_REQUIRED": "UNRESOLVED",
        "EXPERT_COMPLEMENTARITY": "MODERATE", "FROZEN_EXPERT_ORACLE_HEADROOM": "HIGH",
        "DEVELOPMENT_VALIDATION_ACCESSED": "NO", "INTERNAL_TEST_ACCESSED": "NO", "OFFICIAL1000_ACCESSED": "NO",
        "G1_F_EXECUTED": "NO", "AHBFR_EXECUTED": "NO", "TEACHER_KD_EXECUTED": "NO",
        "head_hash_before_audit": head_hash, "head_hash_after_audit": after_hash,
    }
    dump(OUT / "gate_summary.json", gates)
    marker.update({"status": "COMPLETE", "completed_unix": time.time(), "parameters_unchanged": True})
    dump(marker_path, marker)
    print(json.dumps({"TRAIN_AUDIT": "COMPLETE", "RELIABILITY_PREFLIGHT": overall}), flush=True)
    return gates


def write_reports() -> None:
    required = [
        "fit_manifest.json", "fit_history.csv", "temperature_calibration.json", "pixel_metrics.json",
        "image_metrics.json", "qmf_metrics.json", "corruption_metrics.json", "permutation_metrics.json", "gate_summary.json",
    ]
    if not all((OUT / name).is_file() for name in required):
        raise RuntimeError("formal artifacts incomplete; reports cannot be finalized")
    fit = json.loads((OUT / "fit_manifest.json").read_text()); cal = json.loads((OUT / "temperature_calibration.json").read_text())
    pixel = json.loads((OUT / "pixel_metrics.json").read_text()); image = json.loads((OUT / "image_metrics.json").read_text())
    qmf = json.loads((OUT / "qmf_metrics.json").read_text()); corruption = json.loads((OUT / "corruption_metrics.json").read_text())
    permutation = json.loads((OUT / "permutation_metrics.json").read_text()); gates = json.loads((OUT / "gate_summary.json").read_text())
    rows = list(csv.DictReader((OUT / "fit_history.csv").open()))
    report_root = ROOT / "docs/phase4g1/g1c"; report_root.mkdir(parents=True, exist_ok=True)
    sources = ["docs/phase4g05/07_split_protocol.md", "docs/phase4g05/08_reliability_protocol.md", "docs/phase4g05/10_pcerf_hardened_architecture.md", "docs/phase4g05/11_phase4g1_training_proposal_v2.md", "docs/phase4g05/12_phase4g05_final_report.md", "outputs/phase4g05/gate_summary.json", "model/pcerf.py"]
    source_rows = "\n".join(f"- `{path}` SHA256 `{file_sha256(ROOT / path)}`" for path in sources)
    reports = {
        "01_execution_protocol.md": f"""# G1-C 执行协议\n\n本次仅执行授权的 reliability calibration preflight。P1、SAM、CLIP、Phase4C-A 与 ECoLaF 全冻结；仅 TRAIN-FIT 更新 LanguageHead/ForensicHead，TRAIN-CAL 仅拟合两个正 scalar temperature，TRAIN-AUDIT 只正式载入一次。dev validation、internal test、official1000 均封存。\n\n固定 seed `{SEED}`；cache `{CACHE}`；checkpoint `{CKPT}`。\n\n## Source of truth hashes\n\n{source_rows}\n""",
        "02_train_fit_report.md": f"""# TRAIN-FIT 报告\n\n`TRAIN_FIT_COMPLETE = {fit['TRAIN_FIT_COMPLETE']}`。有效样本 {fit['population_n']}，排除 invalid-G0 {fit['excluded_invalid_g0']}；固定 10 epochs / {fit['optimizer_updates']} updates。最终仅使用 epoch 10，不做指标选择。初始/final head hash：`{fit['initial_head_hash']}` / `{fit['final_head_hash']}`。未使用 fused loss 或 validation。\n""",
        "03_calibration_report.md": f"""# TRAIN-CAL calibration 报告\n\n`CALIBRATION_OPTIMIZATION = {gates['CALIBRATION_OPTIMIZATION']}`；`CALIBRATION_NON_DEGRADING = {gates['CALIBRATION_NON_DEGRADING']}`。T_L={cal['temperatures']['T_L']:.8g}，T_F={cal['temperatures']['T_F']:.8g}。temperature 由 TRAIN-CAL deterministic LBFGS source NLL 唯一拟合，无 sweep、IoU 或 validation selection。完整 pre/post NLL、Brier、ECE 见 `outputs/phase4g1/g1c/temperature_calibration.json`。\n""",
        "04_pixel_reliability_report.md": f"""# Pixel reliability 报告\n\nLanguage 与 Forensic 均完整报告 all/foreground/background/3×3 boundary 的 NLL、Brier、15-bin ECE、reliability diagram、confidence≥0.9 error 与 uncertainty/error。Forensic scopes 与 crop support 相交。结果见 `outputs/phase4g1/g1c/pixel_metrics.json`。\n\nLanguage all NLL={pixel['sources']['language']['all']['nll']:.6f}；Forensic all-support NLL={pixel['sources']['forensic']['all']['nll']:.6f}。\n""",
        "05_image_reliability_report.md": f"""# Image-level reliability 报告\n\n`LANGUAGE_UNCERTAINTY_ERROR_RELATION = {gates['LANGUAGE_UNCERTAINTY_ERROR_RELATION']}`；`FORENSIC_UNCERTAINTY_ERROR_RELATION = {gates['FORENSIC_UNCERTAINTY_ERROR_RELATION']}`。每图 mean uncertainty/strength/weight/discount 与 FG IoU、1-FGIoU、source NLL 的 Pearson/Spearman 及固定 10,000 次 bootstrap 95% CI 已完整保存于 `outputs/phase4g1/g1c/image_metrics.json`。\n""",
        "06_qmf_relation_report.md": f"""# QMF relation 报告\n\nweight 固定为 normalized discount × committed mass，不因结果改写。`LANGUAGE_QMF_RELATION = {gates['LANGUAGE_QMF_RELATION']}`；`FORENSIC_QMF_RELATION = {gates['FORENSIC_QMF_RELATION']}`。完整 Pearson/Spearman 与 CI 见 `outputs/phase4g1/g1c/qmf_metrics.json`。\n""",
        "07_corruption_report.md": f"""# Corruption stress 报告\n\n`CROSS_IMAGE_RESPONSE = {gates['CROSS_IMAGE_RESPONSE']}`；`SPATIAL_SHUFFLE_RESPONSE = {gates['SPATIAL_SHUFFLE_RESPONSE']}`。primary criterion 始终是相对 matched 的 forensic effective-weight paired delta；同时报告 uncertainty、strength、discount、conflict。zero/vacuous 保持 outside-support vacuous。完整结果见 `outputs/phase4g1/g1c/corruption_metrics.json`。\n""",
        "08_permutation_report.md": f"""# Reliability permutation 报告\n\n`PERMUTATION_SENSITIVITY = {gates['PERMUTATION_SENSITIVITY']}`。seed 3407 的 preregistered reliability-only permutation 在极低 committed-mass/tie 像素未能保持 source argmax bit-exact，因此 causal-control prerequisite 失败；正式 sensitivity 数值未提交。TRAIN-AUDIT 已消费唯一一次读取，未改写 permutation、未更换 seed、未重跑。失败关闭记录见 `outputs/phase4g1/g1c/permutation_metrics.json`。\n""",
        "09_training_balance_audit.md": f"""# Training balance audit\n\n`TRAINING_BALANCE_AUDIT_REQUIRED = YES`，`TRAINING_BALANCE_INTERVENTION_REQUIRED = UNRESOLVED`。未加入 OGM-GE、gradient balancing、modality dropout 或额外 regularizer。\n\nEpoch 10：L/F loss ratio={float(rows[-1]['language_loss']) / float(rows[-1]['forensic_loss']):.6f}；L/F strength ratio={float(rows[-1]['language_strength']) / float(rows[-1]['forensic_strength']):.6f}；gradient norms L/F={float(rows[-1]['language_grad_norm']):.6f}/{float(rows[-1]['forensic_grad_norm']):.6f}；low-u saturation L/F={float(rows[-1]['language_saturation_low_u']):.6f}/{float(rows[-1]['forensic_saturation_low_u']):.6f}。仅记录，不自动干预。\n""",
    }
    final_lines = ["# G1-C 最终报告", "", "G1-C 只回答 reliability 是否有效，不回答 localization 是否涨点。", ""]
    for key in ("TRAIN_FIT_COMPLETE", "CALIBRATION_OPTIMIZATION", "CALIBRATION_NON_DEGRADING", "LANGUAGE_UNCERTAINTY_ERROR_RELATION", "FORENSIC_UNCERTAINTY_ERROR_RELATION", "LANGUAGE_QMF_RELATION", "FORENSIC_QMF_RELATION", "CROSS_IMAGE_RESPONSE", "SPATIAL_SHUFFLE_RESPONSE", "PERMUTATION_SENSITIVITY", "VACUOUS_EXACT_RECOVERY", "NO_ORACLE_LEAKAGE", "INVALID_G0_POLICY", "RELIABILITY_PREFLIGHT", "G1_F_FULL_TRAINING_JUSTIFIED", "DEVELOPMENT_VALIDATION_ACCESSED", "INTERNAL_TEST_ACCESSED", "OFFICIAL1000_ACCESSED"):
        final_lines.append(f"`{key}: {gates[key]}`")
    final_lines.extend(["", "Permutation control 因未保持 source prediction bit-exact 而失败关闭；TRAIN-AUDIT 只读取一次，未做 post-hoc 修补或重跑。", "", "`EXPERT_COMPLEMENTARITY = MODERATE` 与 `FROZEN_EXPERT_ORACLE_HEADROOM = HIGH` 仅作为背景，不参与 gate。", "", "G1-C 已停止；没有运行 G1-F、AHBFR、Teacher/KD 或 held-out evaluation，等待人工审阅。"])
    reports["10_g1c_final_report.md"] = "\n\n".join(final_lines) + "\n"
    for name, content in reports.items(): (report_root / name).write_text(content, encoding="utf-8")
    dump(OUT / "completion_manifest.json", {
        "schema": "phase4g1_g1c_completion_manifest_v1", "status": "COMPLETE", "files": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(list(report_root.glob("*.md")) + [OUT / name for name in required])
        }, "stopped_after_g1c": True,
    })


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("cache", "parity", "fit", "calibrate", "audit", "report", "all")); parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    stages = ("cache", "parity", "fit", "calibrate", "audit", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(json.dumps({"stage": stage, "status": "START"}), flush=True)
        {"cache": build_cache, "parity": cache_parity, "fit": fit_heads, "calibrate": calibrate, "audit": audit}.get(stage, lambda _device: write_reports())(device)
        print(json.dumps({"stage": stage, "status": "COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
