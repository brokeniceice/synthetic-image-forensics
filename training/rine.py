"""Train the frozen-CLIP RINE head used by C1."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPVisionModel
from model.rine_on_c1 import RINE_Q, RINE_PROJ_DIM, RINE_XI, RINEOnHFCLIP, official_rine_loss
BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, SEED = 128, 1, 1e-3, 0., 3407
EXPECTED_TRAIN, EXPECTED_VAL = 17672, 2212
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


def read_manifest(path, expected):
    source = path.read_text()
    rows = json.loads(source) if path.suffix == '.json' else [json.loads(s) for s in source.splitlines() if s.strip()]
    if len(rows) != expected or len({r['sample_id'] for r in rows}) != expected:
        raise ValueError('Training population mismatch')
    for row in rows:
        # Unified labels are Fake=1. Native RINE manifests use Real=1.
        if 'class_label' in row:
            row['label'] = 1 - int(row['class_label'])
        if row['label'] not in (0, 1):
            raise ValueError('Invalid training label')
    return rows

def train(args):
    OUT = args.output
    INITIAL_HEAD = args.initial_head
    CLIP_PATH = args.clip
    TRAIN_MANIFEST, VAL_MANIFEST = args.train_manifest, args.val_manifest
    workers, device_name = args.workers, args.device
    if OUT.exists():
        raise ValueError('Refusing to overwrite an existing training directory')
    if sha256(INITIAL_HEAD) != '6edc6f19bf9f4a61c05f38f53f88acfcdee52194143039435c45483687e8147a':
        raise ValueError('Use the recorded RINE head initialization')
    protocol_hash = hashlib.sha256((sha256(TRAIN_MANIFEST)+sha256(VAL_MANIFEST)+sha256(INITIAL_HEAD)).encode()).hexdigest()
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
            "schema": "rine_checkpoint_v1", "epoch": epoch,
            "protocol_sha256": protocol_hash,
            "rine_state_dict": {name: value.detach().cpu()
                                for name, value in model.rine.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": validation["metrics"],
        }, checkpoint_path)
        checkpoints.append({"epoch": epoch, "path": str(checkpoint_path.resolve()),
                            "sha256": sha256(checkpoint_path)})
        torch.save({
            "schema": "rine_validation_predictions_v1", "epoch": epoch,
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

    atomic_json(OUT / 'results.json', dict(status='COMPLETE', selected_epoch=selected['epoch'],
        selected_checkpoint_sha256=selected_sha, epoch_records=epoch_records,
        optimizer='torch.optim.Adam', batch_size=BATCH_SIZE, seed=SEED))
    model.close_hooks()

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip', type=Path, default=Path('checkpoints/clip-vit-large-patch14-336'))
    p.add_argument('--initial-head', type=Path, required=True)
    p.add_argument('--train-manifest', type=Path, default=Path('data/manifests/train_combined.jsonl'))
    p.add_argument('--val-manifest', type=Path, default=Path('data/manifests/val_combined.jsonl'))
    p.add_argument('--output', type=Path, default=Path('outputs/rine'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=8)
    train(p.parse_args(argv))

if __name__ == '__main__':
    main()
