"""Pure geometry, loss, metric, and statistics helpers for Phase 3C.1."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SEED = 3407
SOURCES = ("sam", "clip", "npr", "srm", "focal")
PREPROCESS_VERSIONS = {
    "sam": "ResizeLongestSide1024-PadRightBottom-IMG_MEAN_STD:v1",
    "clip": "CLIPImageProcessor-ResizeShortest336-CenterCrop336-Rescale-Normalize:v1",
    "npr": "ResizeSquare256-CenterCrop224-ImageNetNormalize:v1",
    "srm": "ResizeSquare256-CenterCrop224-ImageNetNormalize:v1",
    "focal": "ResizeSquare1024-ToTensor-FP32:v1",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ordered_ids_sha256(values: Sequence[str]) -> str:
    return sha256_text("".join(f"{value}\n" for value in values))


def geometry_for(source: str, original_hw: Sequence[int]) -> dict[str, Any]:
    """Return the exact forward/inverse geometry without assuming feature H/W."""
    h, w = map(int, original_hw)
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid original geometry: {(h, w)}")
    base = {"source": source, "original_hw": [h, w]}
    if source in {"npr", "srm"}:
        return {
            **base, "kind": "resize_square_center_crop", "resized_hw": [256, 256],
            "crop_box_yxyx": [16, 16, 240, 240], "model_input_hw": [224, 224],
            "inverse": "paste crop into 256 square with outside background, resize to original",
        }
    if source == "focal":
        return {
            **base, "kind": "resize_square", "resized_hw": [1024, 1024],
            "model_input_hw": [1024, 1024],
            "inverse": "resize logits directly to original",
        }
    if source == "sam":
        scale = 1024.0 / max(h, w)
        rh, rw = int(h * scale + 0.5), int(w * scale + 0.5)
        return {
            **base, "kind": "resize_longest_pad_right_bottom", "resized_hw": [rh, rw],
            "padding_tlbr": [0, 0, 1024 - rh, 1024 - rw],
            "model_input_hw": [1024, 1024],
            "inverse": "remove right/bottom padding then resize logits to original",
        }
    if source == "clip":
        short, long = (w, h) if w <= h else (h, w)
        new_short, new_long = 336, int(336 * long / short)
        rh, rw = ((new_long, new_short) if w <= h else (new_short, new_long))
        top, left = (rh - 336) // 2, (rw - 336) // 2
        return {
            **base, "kind": "resize_shortest_center_crop", "resized_hw": [rh, rw],
            "crop_box_yxyx": [top, left, top + 336, left + 336],
            "model_input_hw": [336, 336],
            "inverse": "paste crop into resized canvas with outside background, resize to original",
        }
    raise ValueError(f"unknown feature source: {source}")


def transform_mask(mask: torch.Tensor, geometry: Mapping[str, Any]) -> torch.Tensor:
    """Apply the source's image geometry to an original-space union mask."""
    value = torch.as_tensor(mask).float()
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value.any(dim=0, keepdim=True)[None].float()
    else:
        raise ValueError(f"mask must be [H,W] or [N,H,W], got {tuple(value.shape)}")
    resized = F.interpolate(value, size=tuple(geometry["resized_hw"]), mode="nearest")
    kind = geometry["kind"]
    if kind in {"resize_square", "resize_longest_pad_right_bottom"}:
        if kind == "resize_longest_pad_right_bottom":
            target_h, target_w = geometry["model_input_hw"]
            resized = F.pad(
                resized, (0, target_w - resized.shape[-1], 0, target_h - resized.shape[-2])
            )
        return resized[0, 0].bool()
    top, left, bottom, right = geometry["crop_box_yxyx"]
    return resized[0, 0, top:bottom, left:right].bool()


def inverse_logits(logits: torch.Tensor, geometry: Mapping[str, Any]) -> torch.Tensor:
    """Map feature-grid logits to original image space with unseen regions as background."""
    value = torch.as_tensor(logits).float()
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3 and value.shape[0] == 1:
        value = value[None]
    elif value.ndim != 4:
        raise ValueError(f"logits must be [H,W], [1,H,W], or [B,1,H,W], got {tuple(value.shape)}")
    if value.shape[0] != 1 or value.shape[1] != 1:
        raise ValueError("inverse_logits operates on one one-channel sample")
    input_hw = tuple(geometry["model_input_hw"])
    model_logits = F.interpolate(value, size=input_hw, mode="bilinear", align_corners=False)
    kind = geometry["kind"]
    if kind == "resize_square":
        canvas = model_logits
    elif kind == "resize_longest_pad_right_bottom":
        rh, rw = geometry["resized_hw"]
        canvas = model_logits[..., :rh, :rw]
    else:
        rh, rw = geometry["resized_hw"]
        canvas = torch.full((1, 1, rh, rw), -100.0, dtype=model_logits.dtype, device=model_logits.device)
        top, left, bottom, right = geometry["crop_box_yxyx"]
        canvas[..., top:bottom, left:right] = model_logits
    return F.interpolate(
        canvas, size=tuple(geometry["original_hw"]), mode="bilinear", align_corners=False
    )[0, 0]


def probe_loss(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Phase 3C.1 fixed objective: ``2.0 * BCE + 0.5 * Dice``."""
    if logits.shape != targets.shape:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets)
    probabilities = logits.float().sigmoid()
    dims = tuple(range(1, probabilities.ndim))
    # Numerically match model.GLaMM.calculate_dice_loss exactly: its scale
    # factor and epsilon are part of the frozen project's mask objective.
    intersection = 2.0 * (probabilities / 1000.0 * targets).sum(dim=dims)
    union = (probabilities / 1000.0).sum(dim=dims) + (targets / 1000.0).sum(dim=dims)
    dice = (1.0 - (intersection + 1e-6) / (union + 1e-6)).mean()
    return {"bce": bce, "dice": dice, "total": 2.0 * bce + 0.5 * dice}


def binary_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    prediction = torch.as_tensor(logits).gt(0)
    truth = torch.as_tensor(target).bool()
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    tn = int((~prediction & ~truth).sum())
    union = tp + fp + fn
    fg_iou = tp / union if union else 1.0
    f1_den = 2 * tp + fp + fn
    fg_f1 = 2 * tp / f1_den if f1_den else 1.0
    bg_union = tn + fp + fn
    bg_iou = tn / bg_union if bg_union else 1.0
    return {
        "foreground_iou": float(fg_iou), "foreground_f1": float(fg_f1),
        "background_iou": float(bg_iou), "fg_bg_miou": float((fg_iou + bg_iou) / 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    iou = np.asarray([float(row["foreground_iou"]) for row in records], dtype=np.float64)
    f1 = np.asarray([float(row["foreground_f1"]) for row in records], dtype=np.float64)
    miou = np.asarray([float(row["fg_bg_miou"]) for row in records], dtype=np.float64)
    return {
        "n": int(len(records)), "mean_foreground_iou": float(iou.mean()),
        "median_foreground_iou": float(np.median(iou)), "mean_foreground_f1": float(f1.mean()),
        "mean_fg_bg_miou": float(miou.mean()), "iou_gt_0_30_fraction": float((iou > 0.30).mean()),
        "iou_gt_0_50_fraction": float((iou > 0.50).mean()),
    }


def paired_statistics(
    left: Sequence[float], right: Sequence[float], *, seed: int = SEED, repeats: int = 10000,
) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError(f"invalid paired arrays: {a.shape} vs {b.shape}")
    delta = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(repeats, len(delta)))
    boot = delta[indices].mean(axis=1)
    nonzero = delta[delta != 0]
    statistic, pvalue = (0.0, 1.0) if not len(nonzero) else wilcoxon(delta)
    eps = 1e-12
    return {
        "n": int(len(delta)), "mean_difference": float(delta.mean()),
        "median_difference": float(np.median(delta)),
        "bootstrap_95_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "wins": int((delta > eps).sum()), "ties": int((np.abs(delta) <= eps).sum()),
        "losses": int((delta < -eps).sum()), "wilcoxon_statistic": float(statistic),
        "wilcoxon_pvalue": float(pvalue),
    }


def tensor_sha256(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape)).encode() + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
