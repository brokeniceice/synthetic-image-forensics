#!/usr/bin/env python3
"""Frozen-protocol controlled training for RINE-on-C1 (internal train/val only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPVisionModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.rine_on_c1 import (  # noqa: E402
    RINE_OFFICIAL_COMMIT,
    RINE_PROJ_DIM,
    RINE_Q,
    RINE_XI,
    RINEOnHFCLIP,
    official_rine_loss,
)


OUT = ROOT / "outputs/phase6b6_rine_training"
PROTOCOL = OUT / "protocol.json"
RINE_REPO = ROOT / "external/RINE_official"
CLIP_PATH = ROOT / "checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
INITIAL_HEAD = ROOT / "outputs/phase6b5_rine_preflight/rine_head_initialized_seed3407.pt"
PREFLIGHT = ROOT / "outputs/phase6b5_rine_preflight/preflight.json"
TRAIN_MANIFEST = ROOT / "outputs/phase5a3_legion_retrained/data/stage2/train.json"
VAL_MANIFEST = ROOT / "outputs/phase5a3_legion_retrained/data/stage2/val.json"
C1_RESULTS = ROOT / "outputs/phase6b1a_exact_stage2_control/results.json"
C1_VAL_CACHE = ROOT / "outputs/phase6b2_fusion/features/val/c1_exact_logits.pt"

BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEED = 3407
EXPECTED_TRAIN = 17672
EXPECTED_VAL = 2212
EXPECTED_INITIAL_HEAD_SHA = "6edc6f19bf9f4a61c05f38f53f88acfcdee52194143039435c45483687e8147a"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "train"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_manifest(path: Path, expected: int) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
        raise RuntimeError(f"Frozen manifest drift: {path}")
    if any(int(row["label"]) not in (0, 1) for row in rows):
        raise RuntimeError(f"Invalid label in {path}")
    return rows


def official_repo_identity() -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(RINE_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(RINE_REPO), "status", "--porcelain"], text=True
    ).strip()
    if commit != RINE_OFFICIAL_COMMIT or dirty:
        raise RuntimeError(f"Official RINE identity/cleanliness drift: {commit}, {dirty!r}")
    return {"path": str(RINE_REPO.resolve()), "commit": commit, "clean": True}


def prepare_protocol() -> None:
    if PROTOCOL.exists():
        raise RuntimeError(f"Protocol already exists and will not be overwritten: {PROTOCOL}")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or not preflight.get("ready_for_training"):
        raise RuntimeError("Phase 6B.5 readiness gate is not PASS")
    train_rows = read_manifest(TRAIN_MANIFEST, EXPECTED_TRAIN)
    val_rows = read_manifest(VAL_MANIFEST, EXPECTED_VAL)
    if sha256(INITIAL_HEAD) != EXPECTED_INITIAL_HEAD_SHA:
        raise RuntimeError("Frozen Phase 6B.5 initialization hash drift")
    protocol = {
        "schema": "phase6b6_rine_controlled_training_protocol_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "objective": "C1-Exact single-layer CLS versus RINE-on-C1 multi-layer CLS",
        "official_rine": official_repo_identity(),
        "architecture": {
            "backbone": "current frozen CLIP ViT-L/14@336",
            "all_transformer_blocks": 24, "hook": "each HF encoder layer_norm2 CLS",
            "q": RINE_Q, "projection_dim": RINE_PROJ_DIM,
            "tie": "alpha[1,24,1024], softmax over layer dimension",
            "dropout": 0.5, "binary_classifier_output": 1,
            "trainable": ["Q1", "TIE alpha", "Q2", "classifier"],
            "frozen": ["CLIP", "LLM", "R1", "SAM", "all localization paths"],
        },
        "initialization": {
            "path": str(INITIAL_HEAD.resolve()), "sha256": sha256(INITIAL_HEAD),
            "seed": SEED, "trained": False,
        },
        "loss": {
            "bce": "BCEWithLogitsLoss(reduction=sum)", "supervised_contrastive": True,
            "xi": RINE_XI, "temperature": 0.07, "base_temperature": 0.07,
        },
        "training": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "effective_global_batch": BATCH_SIZE, "gradient_accumulation": 1,
            "optimizer": "torch.optim.Adam", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "scheduler": "none",
            "shuffle": True, "drop_last": False, "seed": SEED,
            "steps_per_epoch": math.ceil(EXPECTED_TRAIN / BATCH_SIZE),
        },
        "selector": {
            "reason": "official code exposes no internal-validation checkpoint selector",
            "rule": "max validation ROC-AUC; tie Accuracy; tie earlier epoch",
            "threshold": 0.5, "positive_class": "Fake=1",
        },
        "preprocessing": {
            "choice": "exact current C1 RGB/CLIPImageProcessor at 336",
            "official_rine_224_random_augmentation": False,
            "reason": "controlled architecture comparison and Phase 6B.5 frozen interface",
        },
        "official_recipe_alignment": {
            "same": ["batch_size=128", "Adam", "lr=1e-3", "epochs=1",
                     "no active LR reduction", "BCE-sum + weighted SupCon",
                     "all-block ln_2 hooks", "Q1/TIE/Q2/classifier"],
            "declared_deviations": [
                "current C1 336 preprocessing replaces official RINE 224 augmentation",
                "seed=3407 from Phase 6B.5 replaces official seed=0",
                "internal frozen manifests replace official ProGAN classes",
            ],
        },
        "data": {
            "train": {"path": str(TRAIN_MANIFEST.resolve()), "n": len(train_rows),
                      "sha256": sha256(TRAIN_MANIFEST)},
            "validation": {"path": str(VAL_MANIFEST.resolve()), "n": len(val_rows),
                           "sha256": sha256(VAL_MANIFEST)},
            "resplit": False,
        },
        "firewall": {"internal_train": True, "internal_validation": True,
                     "internal_test": False, "OOD": False, "threshold_tuning": False,
                     "AIDE": False, "LLM_fusion": False},
    }
    atomic_json(PROTOCOL, protocol)
    atomic_json(OUT / "status.json", {"status": "PREPARED", "protocol_sha256": sha256(PROTOCOL)})
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)


class FrozenImages(Dataset):
    def __init__(self, rows: list[dict], processor: CLIPImageProcessor):
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Failed to decode frozen internal sample: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixels,
            "fake_label": 1 - int(row["label"]),
            "index": index,
        }


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities >= 0.5
    tp = int(((labels == 1) & predictions).sum())
    tn = int(((labels == 0) & ~predictions).sum())
    fp = int(((labels == 0) & predictions).sum())
    fn = int(((labels == 1) & ~predictions).sum())
    return {
        "n": int(len(labels)), "accuracy": float((labels == predictions).mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "fake_recall": float(tp / (tp + fn)), "tnr": float(tn / (tn + fp)),
        "fpr": float(fp / (tn + fp)), "f1": float(f1_score(labels, predictions)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": 0.5,
    }


def evaluate(model: RINEOnHFCLIP, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    labels, probabilities, indices = [], [], []
    bce_sum = 0.0
    supcon_weighted_sum = 0.0
    total_batch_sum = 0.0
    batches = 0
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
            targets = batch["fake_label"].to(device=device, dtype=torch.long, non_blocking=True)
            logits, embedding, _ = model(pixels)
            losses = official_rine_loss(logits, embedding, targets)
            count = len(targets)
            bce_sum += float(losses["bce"].item())
            supcon_weighted_sum += float(losses["supcon"].item()) * count
            total_batch_sum += float(losses["total"].item())
            batches += 1
            labels.extend(targets.cpu().tolist())
            probabilities.extend(torch.sigmoid(logits).squeeze(1).float().cpu().tolist())
            indices.extend(batch["index"].tolist())
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if indices != list(range(len(loader.dataset))):
        raise RuntimeError("Validation sample order changed")
    return {
        "metrics": metrics(labels_array, probabilities_array),
        "loss": {"bce_sum": bce_sum, "bce_per_image": bce_sum / len(labels),
                 "supcon_sample_weighted_mean": supcon_weighted_sum / len(labels),
                 "official_total_mean_per_batch": total_batch_sum / batches,
                 "batch_count": batches},
        "labels": torch.tensor(labels), "probabilities": torch.tensor(probabilities),
        "indices": indices,
    }


def tie_statistics(alpha: torch.Tensor) -> dict:
    weights = torch.softmax(alpha.detach().float(), dim=1)[0].cpu()
    channel_winners = weights.argmax(dim=0)
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(dim=0)
    rows = []
    for layer in range(weights.shape[0]):
        value = weights[layer]
        wins = int((channel_winners == layer).sum().item())
        rows.append({
            "layer_1based": layer + 1, "mean": float(value.mean()),
            "std": float(value.std(unbiased=False)), "min": float(value.min()),
            "max": float(value.max()), "winning_channels": wins,
            "winning_channel_fraction": wins / weights.shape[1],
        })
    return {
        "shape": list(weights.shape),
        "per_channel_sum_max_abs_error": float((weights.sum(0) - 1).abs().max()),
        "mean_entropy": float(entropy.mean()),
        "mean_effective_layer_count": float(entropy.exp().mean()),
        "global_min": float(weights.min()), "global_max": float(weights.max()),
        "top_layers_by_mean_weight": [
            row["layer_1based"] for row in sorted(rows, key=lambda item: item["mean"], reverse=True)[:5]
        ],
        "per_layer": rows,
    }


def source_snapshot() -> dict:
    paths = [ROOT / "model/GLaMM.py", ROOT / "model/SAM/build_sam.py",
             ROOT / "model/sam_forensic_rectifier.py",
             ROOT / "external/LEGION_official/model/Legion.py"]
    return {str(path.resolve()): sha256(path) for path in paths}


def train(device_name: str, workers: int) -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Run --mode prepare before training")
    protocol_hash = sha256(PROTOCOL)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_TRAINING":
        raise RuntimeError("Protocol state is not frozen")
    if protocol["training"] != {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "effective_global_batch": BATCH_SIZE, "gradient_accumulation": 1,
        "optimizer": "torch.optim.Adam", "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "scheduler": "none", "shuffle": True,
        "drop_last": False, "seed": SEED,
        "steps_per_epoch": math.ceil(EXPECTED_TRAIN / BATCH_SIZE),
    }:
        raise RuntimeError("Training protocol constant drift")
    if sha256(INITIAL_HEAD) != protocol["initialization"]["sha256"]:
        raise RuntimeError("Initialization changed after protocol freeze")
    official_repo_identity()
    before_sources = source_snapshot()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_rows = read_manifest(TRAIN_MANIFEST, EXPECTED_TRAIN)
    val_rows = read_manifest(VAL_MANIFEST, EXPECTED_VAL)
    processor = CLIPImageProcessor.from_pretrained(CLIP_PATH, local_files_only=True)
    train_dataset = FrozenImages(train_rows, processor)
    val_dataset = FrozenImages(val_rows, processor)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
        num_workers=workers, pin_memory=True, drop_last=False,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers,
        pin_memory=True, drop_last=False, persistent_workers=workers > 0,
    )

    vision = CLIPVisionModel.from_pretrained(CLIP_PATH, local_files_only=True)
    vision = vision.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    model = RINEOnHFCLIP(vision).to(device)
    initialization = torch.load(INITIAL_HEAD, map_location="cpu")
    if (initialization["q"], initialization["projection_dim"], initialization["xi"]) != (
        RINE_Q, RINE_PROJ_DIM, RINE_XI
    ):
        raise RuntimeError("Frozen initialized-head configuration drift")
    model.rine.load_state_dict(initialization["state_dict"], strict=True)
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if sum(parameter.numel() for _, parameter in trainable) != 6_323_201:
        raise RuntimeError("Trainable parameter count drift")
    if any(not name.startswith("rine.") for name, _ in trainable):
        raise RuntimeError("Non-RINE trainable parameter detected")
    optimizer = torch.optim.Adam(
        [parameter for _, parameter in trainable], lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "status.json", {
        "status": "RUNNING", "pid": os.getpid(), "device": device_name,
        "protocol_sha256": protocol_hash, "epoch": 0, "step": 0,
    })
    torch.cuda.reset_peak_memory_stats(device)
    epoch_records = []
    checkpoints = []
    start_time = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        bce_sum = 0.0
        supcon_weighted_sum = 0.0
        total_batch_sum = 0.0
        seen = 0
        for step, batch in enumerate(train_loader, start=1):
            pixels = batch["pixel_values"].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
            targets = batch["fake_label"].to(
                device=device, dtype=torch.long, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            logits, embedding, _ = model(pixels)
            losses = official_rine_loss(logits, embedding, targets)
            losses["total"].backward()
            if not bool(torch.isfinite(losses["total"]).item()):
                raise RuntimeError(f"Non-finite training loss at epoch={epoch}, step={step}")
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                   for _, parameter in trainable):
                raise RuntimeError(f"Invalid trainable gradient at epoch={epoch}, step={step}")
            if any(parameter.grad is not None for parameter in vision.parameters()):
                raise RuntimeError("Frozen CLIP received a training gradient")
            optimizer.step()
            count = len(targets)
            seen += count
            bce_sum += float(losses["bce"].detach().item())
            supcon_weighted_sum += float(losses["supcon"].detach().item()) * count
            total_batch_sum += float(losses["total"].detach().item())
            if step == 1 or step % 10 == 0 or step == len(train_loader):
                elapsed = time.monotonic() - start_time
                event = {"event": "train_progress", "epoch": epoch, "step": step,
                         "steps": len(train_loader), "seen": seen,
                         "loss": float(losses["total"].detach().item()),
                         "elapsed_seconds": elapsed}
                print(json.dumps(event), flush=True)
                atomic_json(OUT / "status.json", {
                    "status": "RUNNING", "pid": os.getpid(), "device": device_name,
                    "protocol_sha256": protocol_hash, "epoch": epoch, "step": step,
                    "steps": len(train_loader), "seen": seen,
                    "elapsed_seconds": elapsed,
                })
        if seen != EXPECTED_TRAIN:
            raise RuntimeError(f"Epoch coverage drift: {seen}")

        validation = evaluate(model, val_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": {
                "bce_sum": bce_sum, "bce_per_image": bce_sum / seen,
                "supcon_sample_weighted_mean": supcon_weighted_sum / seen,
                "official_total_mean_per_batch": total_batch_sum / len(train_loader),
                "batch_count": len(train_loader), "sample_count": seen,
            },
            "validation_loss": validation["loss"],
            "validation_metrics": validation["metrics"],
            "tie_statistics": tie_statistics(model.rine.alpha),
        }
        epoch_records.append(epoch_record)
        checkpoint_path = OUT / f"checkpoint_epoch{epoch}.pt"
        torch.save({
            "schema": "phase6b6_rine_checkpoint_v1", "epoch": epoch,
            "protocol_sha256": protocol_hash,
            "rine_state_dict": {name: value.detach().cpu()
                                for name, value in model.rine.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": validation["metrics"],
        }, checkpoint_path)
        checkpoints.append({"epoch": epoch, "path": str(checkpoint_path.resolve()),
                            "sha256": sha256(checkpoint_path)})
        torch.save({
            "schema": "phase6b6_validation_predictions_v1", "epoch": epoch,
            "sample_ids": [row["sample_id"] for row in val_rows],
            "labels_fake_positive": validation["labels"],
            "prob_fake": validation["probabilities"],
        }, OUT / f"validation_predictions_epoch{epoch}.pt")
        print(json.dumps({"event": "validation_complete", **epoch_record}, ensure_ascii=False), flush=True)

    selected = max(
        epoch_records,
        key=lambda row: (row["validation_metrics"]["roc_auc"],
                         row["validation_metrics"]["accuracy"], -row["epoch"]),
    )
    selected_checkpoint = next(item for item in checkpoints if item["epoch"] == selected["epoch"])
    selected_payload = torch.load(selected_checkpoint["path"], map_location="cpu")
    torch.save(selected_payload, OUT / "selected_checkpoint.pt")
    selected_sha = sha256(OUT / "selected_checkpoint.pt")

    c1_result = json.loads(C1_RESULTS.read_text(encoding="utf-8"))
    c1_metrics = c1_result["metrics"]
    rine_metrics = selected["validation_metrics"]
    metric_keys = ("accuracy", "roc_auc", "fake_recall", "tnr", "fpr", "f1")
    delta = {key: rine_metrics[key] - c1_metrics[key] for key in metric_keys}

    c1_cache = torch.load(C1_VAL_CACHE, map_location="cpu")
    c1_positions = {sample_id: index for index, sample_id in enumerate(c1_cache["sample_ids"])}
    selected_predictions = torch.load(
        OUT / f"validation_predictions_epoch{selected['epoch']}.pt", map_location="cpu"
    )
    if selected_predictions["sample_ids"] != [row["sample_id"] for row in val_rows]:
        raise RuntimeError("Saved RINE validation order drift")
    c1_logits = torch.stack([
        c1_cache["logits_real_fake"][c1_positions[sample_id]]
        for sample_id in selected_predictions["sample_ids"]
    ])
    c1_labels = torch.tensor([
        int(c1_cache["labels"][c1_positions[sample_id]])
        for sample_id in selected_predictions["sample_ids"]
    ])
    if not torch.equal(c1_labels, selected_predictions["labels_fake_positive"]):
        raise RuntimeError("C1/RINE validation label alignment failure")
    c1_pred = c1_logits.softmax(1)[:, 1] >= 0.5
    rine_pred = selected_predictions["prob_fake"] >= 0.5
    labels = selected_predictions["labels_fake_positive"].bool()
    paired = {
        "same_sample_ids": True,
        "prediction_disagreement_count": int((c1_pred != rine_pred).sum()),
        "both_correct": int(((c1_pred == labels) & (rine_pred == labels)).sum()),
        "c1_only_correct": int(((c1_pred == labels) & (rine_pred != labels)).sum()),
        "rine_only_correct": int(((c1_pred != labels) & (rine_pred == labels)).sum()),
        "both_wrong": int(((c1_pred != labels) & (rine_pred != labels)).sum()),
    }

    after_sources = source_snapshot()
    if before_sources != after_sources:
        raise RuntimeError("Localization source changed during training")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("Protocol changed after training started")
    if official_repo_identity()["commit"] != RINE_OFFICIAL_COMMIT:
        raise RuntimeError("Official source changed")

    results = {
        "schema": "phase6b6_rine_training_results_v1", "status": "COMPLETE",
        "protocol_sha256": protocol_hash,
        "candidate": "RINE-on-C1", "selected_epoch": selected["epoch"],
        "selector": protocol["selector"], "epoch_records": epoch_records,
        "selected_checkpoint": {**selected_checkpoint,
                                "copied_path": str((OUT / "selected_checkpoint.pt").resolve()),
                                "copied_sha256": selected_sha},
        "comparison": {"c1_exact": c1_metrics, "rine_on_c1": rine_metrics,
                       "delta_rine_minus_c1": delta, "paired": paired},
        "runtime": {"seconds": time.monotonic() - start_time,
                    "gpu": torch.cuda.get_device_name(device),
                    "device": device_name,
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))},
        "invariance": {"localization_source_hashes_before": before_sources,
                       "localization_source_hashes_after": after_sources,
                       "exact_equal": before_sources == after_sources,
                       "clip_gradient_count": sum(parameter.grad is not None
                                                  for parameter in vision.parameters())},
        "firewall": {"internal_train": True, "internal_validation": True,
                     "internal_test": False, "OOD": False, "threshold_tuning": False,
                     "AIDE": False, "LLM_fusion": False},
    }
    atomic_json(OUT / "results.json", results)
    atomic_json(OUT / "status.json", {
        "status": "COMPLETE", "selected_epoch": selected["epoch"],
        "selected_checkpoint_sha256": selected_sha,
        "protocol_sha256": protocol_hash,
    })
    model.close_hooks()
    print(json.dumps({"event": "phase6b6_complete", **results}, ensure_ascii=False), flush=True)


def main() -> None:
    args = arguments()
    if args.mode == "prepare":
        prepare_protocol()
    else:
        train(args.device, args.workers)


if __name__ == "__main__":
    main()
