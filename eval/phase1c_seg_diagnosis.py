"""Pure helpers for Phase 1C autoregressive SEG-activation diagnosis."""

from __future__ import annotations

import collections
import math
from typing import Any, Iterable, Mapping, Sequence

import torch


FAILURE_REASONS = (
    "EOS_BEFORE_SEG",
    "MAX_NEW_TOKENS_BEFORE_SEG",
    "REPETITION_LOOP",
    "EMPTY_GENERATION",
    "INVALID_SPECIAL_TOKEN",
    "DECODE_STOP_CONDITION",
    "OTHER",
)


def audit_weight_tying(embedding_weight: torch.nn.Parameter, lm_head_weight: torch.nn.Parameter,
                       *, config_tie_word_embeddings: bool) -> dict[str, Any]:
    return {
        "parameter_identity": embedding_weight is lm_head_weight,
        "data_ptr_equal": embedding_weight.data_ptr() == lm_head_weight.data_ptr(),
        "embedding_data_ptr": embedding_weight.data_ptr(),
        "lm_head_data_ptr": lm_head_weight.data_ptr(),
        "config_tie_word_embeddings": bool(config_tie_word_embeddings),
        "embedding_shape": list(embedding_weight.shape),
        "lm_head_shape": list(lm_head_weight.shape),
    }


def module_participates_in_loss(forward_hook_count: int, gradient_norm: float) -> bool:
    if forward_hook_count < 0 or gradient_norm < 0:
        raise ValueError("Participation audit counts and norms must be non-negative")
    return bool(forward_hook_count or gradient_norm > 0)


def repetition_statistics(token_ids: Sequence[int]) -> dict[str, Any]:
    ids = [int(value) for value in token_ids]
    longest_run = 0
    current_run = 0
    previous = None
    for token in ids:
        current_run = current_run + 1 if token == previous else 1
        longest_run = max(longest_run, current_run)
        previous = token

    repeated_ngram_rates = {}
    for width in (2, 3, 4):
        ngrams = [tuple(ids[index:index + width]) for index in range(max(0, len(ids) - width + 1))]
        if not ngrams:
            repeated_ngram_rates[str(width)] = 0.0
            continue
        counts = collections.Counter(ngrams)
        repeated = sum(count - 1 for count in counts.values())
        repeated_ngram_rates[str(width)] = repeated / len(ngrams)
    loop = bool(
        longest_run >= 4
        or (len(ids) >= 12 and repeated_ngram_rates["3"] >= 0.5)
    )
    return {
        "longest_identical_token_run": longest_run,
        "repeated_ngram_rates": repeated_ngram_rates,
        "is_repetition_loop": loop,
    }


def classify_generation_stop(
    token_ids: Sequence[int], *, seg_token_id: int, eos_token_id: int | None,
    max_new_tokens: int, invalid_special_token_ids: Iterable[int] = (),
) -> dict[str, Any]:
    ids = [int(value) for value in token_ids]
    seg_position = next((index for index, value in enumerate(ids) if value == seg_token_id), None)
    first_eos_position = (
        next((index for index, value in enumerate(ids) if value == eos_token_id), None)
        if eos_token_id is not None else None
    )
    repetition = repetition_statistics(ids)
    invalid = any(value in set(invalid_special_token_ids) for value in ids)
    if seg_position is not None and (first_eos_position is None or seg_position < first_eos_position):
        reason = "SEG_TRIGGERED"
    elif not ids:
        reason = "EMPTY_GENERATION"
    elif invalid:
        reason = "INVALID_SPECIAL_TOKEN"
    elif first_eos_position is not None:
        reason = "EOS_BEFORE_SEG"
    elif repetition["is_repetition_loop"]:
        reason = "REPETITION_LOOP"
    elif len(ids) >= max_new_tokens:
        reason = "MAX_NEW_TOKENS_BEFORE_SEG"
    elif len(ids) < max_new_tokens:
        reason = "DECODE_STOP_CONDITION"
    else:
        reason = "OTHER"
    return {
        "stop_reason": reason,
        "first_eos_position": first_eos_position,
        "seg_position": seg_position,
        "num_generated_tokens": len(ids),
        "repetition_statistics": repetition,
    }


def probability_trace(
    scores: Sequence[torch.Tensor], generated_token_ids: Sequence[int], *,
    seg_token_id: int, eos_token_id: int | None,
) -> list[dict[str, Any]]:
    if len(scores) != len(generated_token_ids):
        raise ValueError(f"Got {len(scores)} score steps for {len(generated_token_ids)} generated tokens")
    trace = []
    for step, (score, generated_token) in enumerate(zip(scores, generated_token_ids)):
        logits = torch.as_tensor(score).detach().float()
        if logits.ndim == 2:
            if logits.shape[0] != 1:
                raise ValueError("Phase 1C traces require batch size 1")
            logits = logits[0]
        if logits.ndim != 1:
            raise ValueError(f"Expected vocabulary logits, got {tuple(logits.shape)}")
        probabilities = logits.softmax(dim=-1)
        seg_logit = logits[seg_token_id]
        row = {
            "step": step,
            "generated_token_id": int(generated_token),
            "p_seg": float(probabilities[seg_token_id].cpu()),
            "rank_seg": int(logits.gt(seg_logit).sum().item()) + 1,
        }
        if eos_token_id is not None:
            row.update({
                "p_eos": float(probabilities[eos_token_id].cpu()),
                "rank_eos": int(logits.gt(logits[eos_token_id]).sum().item()) + 1,
                "p_seg_minus_p_eos": float(
                    (probabilities[seg_token_id] - probabilities[eos_token_id]).cpu()
                ),
            })
        trace.append(row)
    return trace


def summarize_probability_trace(trace: Sequence[Mapping[str, Any]], *, eos_position: int | None) -> dict[str, Any]:
    if not trace:
        return {
            "max_p_seg": None, "step_of_max_p_seg": None, "min_rank_seg": None,
            "p_seg_before_eos": None, "p_eos_before_eos": None,
        }
    max_row = max(trace, key=lambda row: row["p_seg"])
    before_eos = None
    if eos_position is not None and 0 <= eos_position < len(trace):
        before_eos = trace[eos_position]
    return {
        "max_p_seg": float(max_row["p_seg"]),
        "step_of_max_p_seg": int(max_row["step"]),
        "min_rank_seg": min(int(row["rank_seg"]) for row in trace),
        "p_seg_before_eos": None if before_eos is None else float(before_eos["p_seg"]),
        "p_eos_before_eos": None if before_eos is None else float(before_eos.get("p_eos", math.nan)),
        "max_p_seg_minus_p_eos": max(
            (float(row["p_seg_minus_p_eos"]) for row in trace if "p_seg_minus_p_eos" in row),
            default=None,
        ),
        "last_step_p_seg": float(trace[-1]["p_seg"]),
        "last_step_p_eos": float(trace[-1].get("p_eos", math.nan)),
    }


def explanation_drift(
    generated_token_ids: Sequence[int], gt_token_ids: Sequence[int], *,
    stop_token_ids: Iterable[int], ignored_prefix_token_ids: Iterable[int] = (),
) -> dict[str, Any]:
    stop = set(int(value) for value in stop_token_ids)
    ignored = set(int(value) for value in ignored_prefix_token_ids)
    generated = []
    for value in generated_token_ids:
        value = int(value)
        if value in stop:
            break
        if not generated and value in ignored:
            continue
        generated.append(value)
    gt = [int(value) for value in gt_token_ids]
    overlap = 0
    for generated_value, gt_value in zip(generated, gt):
        if generated_value != gt_value:
            break
        overlap += 1
    return {
        "generated_explanation_token_length": len(generated),
        "gt_explanation_token_length": len(gt),
        "exact_prefix_overlap_tokens": overlap,
        "exact_prefix_overlap_over_gt": overlap / len(gt) if gt else None,
        "exact_explanation_match": generated == gt,
    }


def summarize_failure_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = [record for record in records if record.get("stop_reason") != "SEG_TRIGGERED"]
    counts = collections.Counter(record.get("stop_reason", "OTHER") for record in failures)
    lengths = [int(record.get("num_generated_tokens", 0)) for record in records]
    sorted_lengths = sorted(lengths)
    if sorted_lengths:
        middle = len(sorted_lengths) // 2
        median_length = (
            sorted_lengths[middle] if len(sorted_lengths) % 2
            else (sorted_lengths[middle - 1] + sorted_lengths[middle]) / 2
        )
    else:
        median_length = None
    success_p = [record.get("max_p_seg") for record in records if record.get("stop_reason") == "SEG_TRIGGERED"]
    failure_p = [record.get("max_p_seg") for record in failures]
    success_p = [float(value) for value in success_p if value is not None]
    failure_p = [float(value) for value in failure_p if value is not None]
    return {
        "num_samples": len(records),
        "num_failures": len(failures),
        "failure_reason_counts": {reason: counts.get(reason, 0) for reason in FAILURE_REASONS},
        "eos_before_seg_rate": counts.get("EOS_BEFORE_SEG", 0) / len(records) if records else None,
        "max_token_before_seg_rate": counts.get("MAX_NEW_TOKENS_BEFORE_SEG", 0) / len(records) if records else None,
        "repetition_rate": sum(
            bool(record.get("repetition_statistics", {}).get("is_repetition_loop")) for record in records
        ) / len(records) if records else None,
        "mean_generated_token_length": sum(lengths) / len(lengths) if lengths else None,
        "median_generated_token_length": median_length,
        "successful_max_p_seg_mean": sum(success_p) / len(success_p) if success_p else None,
        "failed_max_p_seg_mean": sum(failure_p) / len(failure_p) if failure_p else None,
    }
