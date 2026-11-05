"""Reproducible detection/localization protocols for unified forensics.

The mode functions in this module are deliberately independent.  They share
only metric and record helpers, preventing classification gates or GT context
from leaking from one protocol into another.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import torch


VERDICT_NAMES = ("real", "fake")
FORENSICS_EVAL_MODES = (
    "detection",
    "gt_fake_generate",
    "tf_full_context",
    "tf_minimal_context",
    "joint",
)
PHASE1C_DIAGNOSTIC_MODES = (
    "unified_fake_generate",
    "unified_prompt_gt_fake_prefix",
    "legacy_gt_fake_generate",
)
PROTOCOL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/forensics_eval_phase1a.json"
PROTOCOL_CONFIG = json.loads(PROTOCOL_CONFIG_PATH.read_text(encoding="utf-8"))
TF_MINIMAL_CONTEXT_TEMPLATE = PROTOCOL_CONFIG["tf_minimal_context_template"]
MASK_LOGIT_THRESHOLD = float(PROTOCOL_CONFIG["mask_logit_threshold"])
CONTENT_GROUPS = tuple(PROTOCOL_CONFIG["content_groups"])


class ForensicsEvaluationBackend(Protocol):
    """Model-specific operations required by the protocol layer."""

    def detection(
        self, sample: Mapping[str, Any], *, user_prompt: str = "canonical"
    ) -> Mapping[str, Any]: ...

    def generate_localization(
        self, sample: Mapping[str, Any], *, provide_gt_fake: bool
    ) -> Mapping[str, Any]: ...

    def teacher_forced_localization(
        self, sample: Mapping[str, Any], *, context: str, user_prompt: str = "canonical"
    ) -> Mapping[str, Any]: ...


def _as_binary_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(mask)
    if mask.ndim == 3:
        if mask.shape[0] == 0:
            raise ValueError("Cannot convert an empty mask tensor")
        mask = mask.bool().any(dim=0)
    elif mask.ndim != 2:
        raise ValueError(f"Expected [H,W] or [N,H,W] mask, got {tuple(mask.shape)}")
    return mask.bool()


def compute_binary_mask_metrics(pred_logits: torch.Tensor, gt_mask: torch.Tensor) -> dict[str, float | int]:
    """Compute the one canonical localization metric implementation.

    Multiple predicted masks are combined as a pixelwise union (equivalent to
    max over logits before applying the fixed ``> 0`` threshold).
    """
    logits = torch.as_tensor(pred_logits)
    if logits.ndim == 3:
        if logits.shape[0] == 0:
            return compute_empty_prediction_metrics(gt_mask)
        logits = logits.amax(dim=0)
    elif logits.ndim != 2:
        raise ValueError(f"Expected [H,W] or [N,H,W] logits, got {tuple(logits.shape)}")
    prediction = logits > MASK_LOGIT_THRESHOLD
    target = _as_binary_mask(gt_mask).to(device=prediction.device)
    if prediction.shape != target.shape:
        raise ValueError(f"Pred/GT mask shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    return _metrics_from_binary_masks(prediction, target)


def compute_empty_prediction_metrics(gt_mask: torch.Tensor) -> dict[str, float | int]:
    target = _as_binary_mask(gt_mask)
    return _metrics_from_binary_masks(torch.zeros_like(target), target)


def _metrics_from_binary_masks(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    prediction, target = prediction.bool(), target.bool()
    tp = int((prediction & target).sum().item())
    fp = int((prediction & ~target).sum().item())
    fn = int((~prediction & target).sum().item())
    intersection = tp
    union = tp + fp + fn
    image_iou = intersection / union if union else 1.0
    f1_denominator = 2 * tp + fp + fn
    image_f1 = 2 * tp / f1_denominator if f1_denominator else 1.0
    return {
        "intersection": intersection,
        "union": union,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "image_iou": float(image_iou),
        "image_pixel_f1": float(image_f1),
    }


def _verdict_name(value: Any) -> str:
    if torch.is_tensor(value):
        value = value.item()
    if isinstance(value, str):
        value = value.lower()
        if value not in VERDICT_NAMES:
            raise ValueError(f"Unknown verdict: {value}")
        return value
    return VERDICT_NAMES[int(value)]


def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().item()
    return float(value)


def _sample_record(sample: Mapping[str, Any], eval_mode: str) -> dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id"),
        "image_path": sample.get("image_path"),
        "source": sample.get("source"),
        "content_type": sample.get("content_category"),
        "gt_label": _verdict_name(sample.get("cls_label", sample.get("class_label"))),
        "eval_mode": eval_mode,
    }


def _prediction_fields(output: Mapping[str, Any]) -> dict[str, Any]:
    cls_pred = _verdict_name(output["cls_pred"])
    lm_pred = _verdict_name(output["lm_verdict_pred"])
    return {
        "cls_prob_fake": _scalar(output["cls_prob_fake"]),
        "cls_pred": cls_pred,
        "lm_verdict_prob_fake": _scalar(output["lm_verdict_prob_fake"]),
        "lm_verdict_pred": lm_pred,
        "cls_lm_agree": bool(output.get("cls_lm_agree", cls_pred == lm_pred)),
    }


def build_forensics_prediction_records(
    batch: Mapping[str, Any], output: Mapping[str, torch.Tensor]
) -> list[dict[str, Any]]:
    """Backward-compatible Phase 0.5 classification record builder."""
    cls_pred = output["cls_pred"].detach().cpu()
    cls_prob = output["cls_prob"].detach().float().cpu()
    lm_pred = output["lm_verdict_pred"].detach().cpu()
    lm_prob = output["lm_verdict_prob"].detach().float().cpu()
    agree = output["cls_lm_agree"].detach().cpu()
    count = cls_pred.numel()
    fields: Sequence[tuple[str, Sequence[Any]]] = (
        ("sample_ids", batch.get("sample_ids") or [None] * count),
        ("image_paths", batch.get("image_paths") or [None] * count),
        ("sources", batch.get("sources") or [None] * count),
        ("content_categories", batch.get("content_categories") or [None] * count),
    )
    for name, values in fields:
        if len(values) != count:
            raise ValueError(f"{name} has {len(values)} values for {count} predictions")

    labels = batch.get("cls_labels")
    labels = labels.detach().cpu() if torch.is_tensor(labels) else labels
    records = []
    for index in range(count):
        records.append({
            "sample_id": fields[0][1][index],
            "image_path": fields[1][1][index],
            "source": fields[2][1][index],
            "content_type": fields[3][1][index],
            "gt_label": None if labels is None else VERDICT_NAMES[int(labels[index])],
            "cls_pred": VERDICT_NAMES[int(cls_pred[index])],
            "cls_prob": float(cls_prob[index]),
            "lm_verdict_pred": VERDICT_NAMES[int(lm_pred[index])],
            "lm_verdict_prob": float(lm_prob[index]),
            "cls_lm_agree": bool(agree[index]),
        })
    return records


def _binary_summary(labels: list[int], predictions: list[int], probabilities: list[float]) -> dict[str, Any]:
    if not labels:
        return {key: None for key in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    tp = sum(gt == 1 and pred == 1 for gt, pred in zip(labels, predictions))
    tn = sum(gt == 0 and pred == 0 for gt, pred in zip(labels, predictions))
    fp = sum(gt == 0 and pred == 1 for gt, pred in zip(labels, predictions))
    fn = sum(gt == 1 and pred == 0 for gt, pred in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    positive_scores = [p for y, p in zip(labels, probabilities) if y == 1]
    negative_scores = [p for y, p in zip(labels, probabilities) if y == 0]
    if positive_scores and negative_scores:
        wins = sum(
            float(pos > neg) + 0.5 * float(pos == neg)
            for pos in positive_scores for neg in negative_scores
        )
        roc_auc = wins / (len(positive_scores) * len(negative_scores))
    else:
        roc_auc = None
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "fake_precision": precision,
        "fake_recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_detection(
    samples: Iterable[Mapping[str, Any]], backend: ForensicsEvaluationBackend, *,
    user_prompt: str = "canonical",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate both authenticity heads on every Real and Fake sample."""
    records = []
    for sample in samples:
        prediction = backend.detection(sample, user_prompt=user_prompt)
        record = _sample_record(sample, "detection")
        record.update(_prediction_fields(prediction))
        records.append(record)

    return records, summarize_detection_records(records)


def summarize_detection_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate stored detection records without rerunning model inference."""
    labels = [int(record["gt_label"] == "fake") for record in records]
    cls_predictions = [int(record["cls_pred"] == "fake") for record in records]
    lm_predictions = [int(record["lm_verdict_pred"] == "fake") for record in records]
    metrics = {
        "num_samples": len(records),
        "classification_head": _binary_summary(
            labels, cls_predictions,
            [record.get("cls_prob_fake", record.get("cls_prob")) for record in records],
        ),
        "lm_verdict": _binary_summary(
            labels, lm_predictions,
            [record.get("lm_verdict_prob_fake", record.get("lm_verdict_prob")) for record in records],
        ),
        "cls_lm_agreement": (
            sum(record["cls_lm_agree"] for record in records) / len(records) if records else None
        ),
        "outcome_counts": {
            "cls_correct_lm_correct": 0,
            "cls_correct_lm_wrong": 0,
            "cls_wrong_lm_correct": 0,
            "cls_wrong_lm_wrong": 0,
        },
    }
    for record in records:
        cls_correct = record["cls_pred"] == record["gt_label"]
        lm_correct = record["lm_verdict_pred"] == record["gt_label"]
        key = f"cls_{'correct' if cls_correct else 'wrong'}_lm_{'correct' if lm_correct else 'wrong'}"
        metrics["outcome_counts"][key] += 1
    return metrics


def _eligible_fake(sample: Mapping[str, Any]) -> bool:
    label = sample.get("cls_label", sample.get("class_label"))
    if torch.is_tensor(label):
        label = label.item()
    return int(label) == 1 and bool(sample.get("seg_valid", False))


def _localization_record(
    sample: Mapping[str, Any], output: Mapping[str, Any], eval_mode: str, *,
    uses_gt_authenticity: bool, uses_gt_explanation: bool, classification_gate: bool,
) -> dict[str, Any]:
    record = _sample_record(sample, eval_mode)
    record.update(_prediction_fields(output))
    record.update({
        "generated_explanation": output.get("generated_explanation"),
        "gt_explanation": " ".join(
            str((sample.get("manifest_row") or {}).get("explanation") or "").split()
        ),
        "seg_triggered": bool(output.get("seg_triggered", False)),
        "has_pred_mask": output.get("pred_mask") is not None,
        "uses_gt_authenticity": uses_gt_authenticity,
        "uses_gt_explanation": uses_gt_explanation,
        "classification_gates_localization": classification_gate,
    })
    for key in (
        "generation_mode", "generated_text", "generated_token_ids", "prompt_token_ids",
        "seg_probability_trace",
        "prompt_template_id", "prompt_sha256", "raw_prompt_text", "assistant_prefix",
        "prompt_ends_with_eos", "num_generated_tokens", "stop_reason", "contains_fake_token",
        "contains_real_token", "contains_seg_token", "first_eos_position", "seg_position",
        "max_new_tokens", "repetition_statistics", "max_p_seg", "step_of_max_p_seg",
        "min_rank_seg", "p_seg_before_eos", "p_eos_before_eos", "max_p_seg_minus_p_eos",
        "last_step_p_seg", "last_step_p_eos", "generated_explanation_token_length",
        "gt_explanation_token_length", "exact_prefix_overlap_tokens",
        "exact_prefix_overlap_over_gt", "exact_explanation_match",
    ):
        if key in output:
            record[key] = output[key]
    record["repetition_flag"] = bool(
        record.get("repetition_statistics", {}).get("is_repetition_loop", False)
    )
    gt_mask = sample.get("masks")
    if gt_mask is None:
        raise ValueError(f"Eligible Fake sample {sample.get('sample_id')} has no GT mask")
    gate_passed = not classification_gate or record["cls_pred"] == "fake"
    valid_prediction = record["seg_triggered"] and record["has_pred_mask"] and gate_passed
    if valid_prediction:
        metrics = compute_binary_mask_metrics(output["pred_mask"], gt_mask)
    else:
        metrics = compute_empty_prediction_metrics(gt_mask)
    record.update(metrics)
    record["seg_trigger_failure"] = not (record["seg_triggered"] and record["has_pred_mask"])
    record["grounding_activation_failure"] = bool(
        classification_gate and gate_passed and record["seg_trigger_failure"]
    )
    record["classification_gate_passed"] = gate_passed
    return record


def _summarize_record_group(records: Sequence[Mapping[str, Any]], *, autoregressive: bool,
                            joint: bool) -> dict[str, Any]:
    count = len(records)
    intersection = sum(record["intersection"] for record in records)
    union = sum(record["union"] for record in records)
    tp = sum(record["tp"] for record in records)
    fp = sum(record["fp"] for record in records)
    fn = sum(record["fn"] for record in records)
    metrics = {
        "num_gt_fake": count,
        "mean_iou": sum(record["image_iou"] for record in records) / count if count else None,
        "global_iou": intersection / union if union else (1.0 if count else None),
        "mean_pixel_f1": sum(record["image_pixel_f1"] for record in records) / count if count else None,
        "global_pixel_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else (1.0 if count else None),
        "intersection": intersection,
        "union": union,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    if autoregressive:
        triggered = sum(record["seg_triggered"] and record["has_pred_mask"] for record in records)
        metrics.update({
            "seg_trigger_rate": triggered / count if count else None,
            "seg_trigger_failure_count": count - triggered,
        })
        lengths = [record.get("num_generated_tokens") for record in records]
        lengths = sorted(int(value) for value in lengths if value is not None)
        if lengths:
            middle = len(lengths) // 2
            median = (
                lengths[middle] if len(lengths) % 2
                else (lengths[middle - 1] + lengths[middle]) / 2
            )
            metrics.update({
                "mean_generated_token_length": sum(lengths) / len(lengths),
                "median_generated_token_length": median,
                "eos_before_seg_rate": sum(
                    record.get("stop_reason") == "EOS_BEFORE_SEG" for record in records
                ) / count,
                "max_token_before_seg_rate": sum(
                    record.get("stop_reason") == "MAX_NEW_TOKENS_BEFORE_SEG" for record in records
                ) / count,
                "repetition_rate": sum(
                    bool(record.get("repetition_statistics", {}).get("is_repetition_loop"))
                    for record in records
                ) / count,
            })
    if joint:
        passed = sum(record["cls_pred"] == "fake" for record in records)
        metrics.update({
            "fake_recall": passed / count if count else None,
            "classification_pass_rate": passed / count if count else None,
            "joint_mean_iou": metrics["mean_iou"],
            "joint_global_iou": metrics["global_iou"],
            "joint_mean_pixel_f1": metrics["mean_pixel_f1"],
            "joint_global_pixel_f1": metrics["global_pixel_f1"],
        })
    return metrics


def summarize_localization_records(records: Sequence[Mapping[str, Any]], *, autoregressive: bool,
                                   joint: bool = False) -> dict[str, Any]:
    """Report overall and fixed content-category localization summaries."""
    overall = _summarize_record_group(records, autoregressive=autoregressive, joint=joint)
    by_content = {}
    for group in CONTENT_GROUPS:
        selected = [record for record in records if str(record.get("content_type", "")).lower() == group]
        by_content[group] = _summarize_record_group(selected, autoregressive=autoregressive, joint=joint)
    return {**overall, "overall": overall, "by_content_type": by_content}


def evaluate_gt_fake_generation_localization(
    samples: Iterable[Mapping[str, Any]], backend: ForensicsEvaluationBackend
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Default GT-conditioned localization: Unified prompt + structural [FAKE] prefix."""
    records = []
    for sample in samples:
        if not _eligible_fake(sample):
            continue
        output = backend.generate_localization(sample, provide_gt_fake=True)
        records.append(_localization_record(
            sample, output, "gt_fake_generate", uses_gt_authenticity=True,
            uses_gt_explanation=False, classification_gate=False,
        ))
    return records, summarize_localization_records(records, autoregressive=True)


def evaluate_known_fake_prompt_generation_localization(samples, backend):
    """Phase 1C G2: corrected continuation with the old known-Fake-style prompt."""
    records = []
    for sample in samples:
        if not _eligible_fake(sample):
            continue
        output = backend.generate_localization(
            sample, provide_gt_fake=True, generation_mode="gt_fake_generate"
        )
        records.append(_localization_record(
            sample, output, "gt_fake_generate", uses_gt_authenticity=True,
            uses_gt_explanation=False, classification_gate=False,
        ))
    return records, summarize_localization_records(records, autoregressive=True)


def _evaluate_fake_generation_mode(samples, backend, *, mode: str, uses_gt_authenticity: bool):
    records = []
    for sample in samples:
        if not _eligible_fake(sample):
            continue
        output = backend.generate_localization(
            sample, provide_gt_fake=uses_gt_authenticity, generation_mode=mode
        )
        records.append(_localization_record(
            sample, output, mode, uses_gt_authenticity=uses_gt_authenticity,
            uses_gt_explanation=False, classification_gate=False,
        ))
    return records, summarize_localization_records(records, autoregressive=True)


def evaluate_unified_fake_generation_localization(samples, backend):
    """G0: training-distribution prompt, fixed CLS, otherwise free generation."""
    return _evaluate_fake_generation_mode(
        samples, backend, mode="unified_fake_generate", uses_gt_authenticity=False
    )


def evaluate_unified_gt_fake_prefix_localization(samples, backend):
    """G1: training-distribution prompt plus a structural GT [FAKE] prefix."""
    return _evaluate_fake_generation_mode(
        samples, backend, mode="unified_prompt_gt_fake_prefix", uses_gt_authenticity=True
    )


def evaluate_legacy_gt_fake_generation_localization(samples, backend):
    """Reproduce Phase-1B G2, including its terminal-EOS prefix implementation."""
    return _evaluate_fake_generation_mode(
        samples, backend, mode="legacy_gt_fake_generate", uses_gt_authenticity=True
    )


def evaluate_teacher_forced_full_context(
    samples: Iterable[Mapping[str, Any]], backend: ForensicsEvaluationBackend, *,
    user_prompt: str = "canonical",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Teacher-forced GT verdict + GT explanation diagnostic upper bound."""
    records = []
    for sample in samples:
        if not _eligible_fake(sample):
            continue
        output = backend.teacher_forced_localization(
            sample, context="full", user_prompt=user_prompt
        )
        records.append(_localization_record(
            sample, output, "tf_full_context", uses_gt_authenticity=True,
            uses_gt_explanation=True, classification_gate=False,
        ))
    return records, summarize_localization_records(records, autoregressive=False)


def evaluate_teacher_forced_minimal_context(
    samples: Iterable[Mapping[str, Any]], backend: ForensicsEvaluationBackend, *,
    user_prompt: str = "canonical",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Teacher-forced fixed minimal template diagnostic (no GT explanation)."""
    records = []
    for sample in samples:
        if not _eligible_fake(sample):
            continue
        output = backend.teacher_forced_localization(
            sample, context="minimal", user_prompt=user_prompt
        )
        records.append(_localization_record(
            sample, output, "tf_minimal_context", uses_gt_authenticity=True,
            uses_gt_explanation=False, classification_gate=False,
        ))
    return records, summarize_localization_records(records, autoregressive=False)


def evaluate_joint_localization(
    samples: Iterable[Mapping[str, Any]], backend: ForensicsEvaluationBackend
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fully unified generation with classification-head gating on GT Fake samples."""
    records = []
    for sample in samples:
        if not _eligible_fake(sample):
            continue
        output = backend.generate_localization(sample, provide_gt_fake=False)
        records.append(_localization_record(
            sample, output, "joint", uses_gt_authenticity=False,
            uses_gt_explanation=False, classification_gate=True,
        ))
    return records, summarize_localization_records(records, autoregressive=True, joint=True)


MODE_EVALUATORS = {
    "detection": evaluate_detection,
    "gt_fake_generate": evaluate_gt_fake_generation_localization,
    "tf_full_context": evaluate_teacher_forced_full_context,
    "tf_minimal_context": evaluate_teacher_forced_minimal_context,
    "joint": evaluate_joint_localization,
}


def write_evaluation_outputs(output_dir: str | Path, mode: str, records: Sequence[Mapping[str, Any]],
                             metrics: Mapping[str, Any]) -> tuple[Path, Path]:
    if mode not in (*FORENSICS_EVAL_MODES, *PHASE1C_DIAGNOSTIC_MODES):
        raise ValueError(f"Unknown evaluation mode: {mode}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"forensics_predictions_{mode}.jsonl"
    metrics_path = output_dir / f"forensics_metrics_{mode}.json"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return predictions_path, metrics_path
