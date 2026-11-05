#!/usr/bin/env python3
"""Freeze the preregistered Phase 3A selector before any official-test inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3a_phrase_grounding"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p1-best",
        default="checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt",
    )
    parser.add_argument(
        "--p1-last",
        default="checkpoints/phase3a_phrase_grounding/p1/last/checkpoint/mp_rank_00_model_states.pt",
    )
    cli = parser.parse_args()
    metrics_path = OUT / "training/p1/metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    if len(rows) != 5000 or int(rows[-1]["optimizer_step"]) != 5000:
        raise RuntimeError("P1 training is not durably complete at step 5000")
    steps = [int(row["optimizer_step"]) for row in rows]
    if steps != list(range(1, 5001)):
        raise RuntimeError("Canonical P1 metrics are not the exact contiguous steps 1..5000")
    best_path = (ROOT / cli.p1_best).resolve()
    last_path = (ROOT / cli.p1_last).resolve()
    best = torch.load(best_path, map_location="cpu")
    last = torch.load(last_path, map_location="cpu")
    if int(last.get("optimizer_step", -1)) != 5000:
        raise RuntimeError("Last durable checkpoint is not step 5000")
    selector = {
        "status": "FROZEN_BEFORE_OFFICIAL_TEST",
        "selector": "min_validation_total_loss",
        "selected_checkpoint": str(best_path),
        "checkpoint_sha256": sha256(best_path),
        "optimizer_step": int(best["optimizer_step"]),
        "epoch": int(best["epoch"]),
        "best_val_total_loss": float(best["best_val_total_loss"]),
        "training_completed_step": 5000,
        "official_test_used_for_selection": False,
        "validation_localization_used_for_selection": False,
    }
    dump(OUT / "selection/p1_selector.json", selector)
    c0_path = (
        ROOT / "checkpoints/phase2a_unified_baseline/single/best/checkpoint/"
        "mp_rank_00_model_states.pt"
    ).resolve()
    dump(OUT / "selection/c0_selector.json", {
        "status": "HISTORICAL_FROZEN_REFERENCE_NOT_RETRAINED",
        "selector": "min_validation_total_loss",
        "selected_checkpoint": str(c0_path),
        "checkpoint_sha256": sha256(c0_path),
        "optimizer_step": 2500,
        "epoch": 5,
        "official_test_used_for_selection": False,
        "comparison_caveat": "UNPAIRED_TRAINING_COMPARISON",
    })
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "stage": "selector_frozen_official_test_not_started",
        "training_completed": True,
        "p1_selector_frozen": True,
        "official_test_started": False,
    })
    dump(manifest_path, manifest)
    print(json.dumps(selector, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
