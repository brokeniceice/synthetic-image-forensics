"""World-size and accumulation-window invariant loss normalization helpers."""

from __future__ import annotations

import torch


LOSS_COUNT_KEYS = {
    "ce_loss": "text_tokens",
    "cls_loss": "classification_samples",
    "mask_bce_loss": "valid_masks",
    "mask_dice_loss": "valid_masks",
}


def batch_supervision_counts(batch) -> dict[str, int]:
    labels = batch["labels"]
    text_tokens = int(labels[:, 1:].ne(-100).sum().item())
    cls_labels = batch.get("cls_labels")
    classification_samples = 0 if cls_labels is None else int(cls_labels.ge(0).sum().item())
    valid_masks = 0
    for is_valid, mask in zip(batch["seg_valid"].tolist(), batch["masks_list"]):
        if is_valid:
            if mask is None or mask.numel() == 0:
                raise ValueError("seg_valid=True sample has no mask while counting global denominator")
            valid_masks += int(mask.shape[0])
    return {
        "text_tokens": text_tokens,
        "classification_samples": classification_samples,
        "valid_masks": valid_masks,
    }


def sum_window_counts(counts) -> dict[str, int]:
    return {
        key: sum(int(item[key]) for item in counts)
        for key in ("text_tokens", "classification_samples", "valid_masks")
    }


def all_reduce_window_counts(local_counts: dict[str, int], device) -> dict[str, int]:
    values = torch.tensor(
        [local_counts["text_tokens"], local_counts["classification_samples"], local_counts["valid_masks"]],
        dtype=torch.long, device=device,
    )
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return dict(zip(("text_tokens", "classification_samples", "valid_masks"), values.tolist()))


def globally_normalized_components(output, local_counts, global_counts, *, world_size: int,
                                   accumulation_steps: int):
    """Scale local means so DP averaging + GAS averaging yields global means."""
    averaging_factor = float(world_size * accumulation_steps)
    scaled = {}
    for loss_key, count_key in LOSS_COUNT_KEYS.items():
        denominator = int(global_counts[count_key])
        local_count = int(local_counts[count_key])
        if denominator == 0:
            scaled[loss_key] = output[loss_key] * 0.0
        else:
            scaled[loss_key] = output[loss_key] * (averaging_factor * local_count / denominator)
    scaled["loss"] = sum(scaled.values())
    return scaled


def global_objective_from_local_means(local_outputs, local_counts, global_counts):
    """Compute scalar window metrics corresponding to the exact global objective."""
    values = {}
    for loss_key, count_key in LOSS_COUNT_KEYS.items():
        numerator = sum(
            float(output[loss_key].detach().float().cpu()) * int(count[count_key])
            for output, count in zip(local_outputs, local_counts)
        )
        values[loss_key] = (numerator, int(global_counts[count_key]))
    return values
