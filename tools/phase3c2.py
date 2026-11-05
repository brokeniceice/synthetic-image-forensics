"""Pure helpers for Phase 3C.2 deterministic corruption and paired gates."""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image


CONDITIONS = ("original", "jpeg70", "jpeg80", "gaussian5", "gaussian10")


def deterministic_seed(seed: int, sample_id: str, condition: str) -> int:
    payload = f"{seed}:{sample_id}:{condition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def corrupt_rgb(image: np.ndarray, condition: str, *, seed: int, sample_id: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB HWC uint8 image, got {image.shape}")
    if condition == "original":
        return image.copy()
    if condition in {"jpeg70", "jpeg80"}:
        quality = int(condition.removeprefix("jpeg"))
        buffer = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(
            buffer, format="JPEG", quality=quality, subsampling=2, optimize=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
    if condition in {"gaussian5", "gaussian10"}:
        sigma = float(condition.removeprefix("gaussian"))
        rng = np.random.default_rng(deterministic_seed(seed, sample_id, condition))
        noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape).astype(np.float32)
        return np.rint(np.clip(noisy, 0.0, 255.0)).astype(np.uint8)
    raise ValueError(f"unsupported robustness condition: {condition}")


def mcnemar_exact(base_predictions, candidate_predictions, labels) -> dict:
    from scipy.stats import binomtest

    base = np.asarray(base_predictions) == np.asarray(labels)
    candidate = np.asarray(candidate_predictions) == np.asarray(labels)
    base_only = int(np.logical_and(base, ~candidate).sum())
    candidate_only = int(np.logical_and(~base, candidate).sum())
    discordant = base_only + candidate_only
    p_value = float(binomtest(min(base_only, candidate_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "base_correct_candidate_wrong": base_only,
        "base_wrong_candidate_correct": candidate_only,
        "net_corrected": candidate_only - base_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def apply_classification_gate(record: dict, fake_prediction: bool) -> dict:
    row = dict(record)
    row["classification_gates_localization"] = True
    row["classification_gate_passed"] = bool(fake_prediction)
    row["eval_mode"] = "classification_gated_G0"
    if not fake_prediction:
        foreground_pixels = int(row.get("tp", 0)) + int(row.get("fn", 0))
        background_pixels = int(row.get("tn", 0)) + int(row.get("fp", 0))
        background_iou = (
            background_pixels / (background_pixels + foreground_pixels)
            if background_pixels + foreground_pixels else 1.0
        )
        row.update({
            "intersection": 0,
            "tp": 0,
            "fp": 0,
            "fn": foreground_pixels,
            "tn": background_pixels,
            "union": foreground_pixels,
            "image_iou": 0.0,
            "image_pixel_f1": 0.0,
            "foreground_iou": 0.0,
            "foreground_f1": 0.0,
            "background_iou": background_iou,
            "fg_bg_miou": background_iou / 2.0,
        })
    return row
