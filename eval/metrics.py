from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
VERDICT_NAMES = ('real', 'fake')
PROTOCOL_CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs/evaluation.json'
PROTOCOL_CONFIG = json.loads(PROTOCOL_CONFIG_PATH.read_text(encoding='utf-8'))
MASK_LOGIT_THRESHOLD = float(PROTOCOL_CONFIG['mask_logit_threshold'])
CONTENT_GROUPS = tuple(PROTOCOL_CONFIG['content_groups'])

def _as_binary_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(mask)
    if mask.ndim == 3:
        if mask.shape[0] == 0:
            raise ValueError('Cannot convert an empty mask tensor')
        mask = mask.bool().any(dim=0)
    elif mask.ndim != 2:
        raise ValueError(f'Expected [H,W] or [N,H,W] mask, got {tuple(mask.shape)}')
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
        raise ValueError(f'Expected [H,W] or [N,H,W] logits, got {tuple(logits.shape)}')
    prediction = logits > MASK_LOGIT_THRESHOLD
    target = _as_binary_mask(gt_mask).to(device=prediction.device)
    if prediction.shape != target.shape:
        raise ValueError(f'Pred/GT mask shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}')
    return _metrics_from_binary_masks(prediction, target)

def compute_empty_prediction_metrics(gt_mask: torch.Tensor) -> dict[str, float | int]:
    target = _as_binary_mask(gt_mask)
    return _metrics_from_binary_masks(torch.zeros_like(target), target)

def _metrics_from_binary_masks(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    (prediction, target) = (prediction.bool(), target.bool())
    tp = int((prediction & target).sum().item())
    fp = int((prediction & ~target).sum().item())
    fn = int((~prediction & target).sum().item())
    intersection = tp
    union = tp + fp + fn
    image_iou = intersection / union if union else 1.0
    f1_denominator = 2 * tp + fp + fn
    image_f1 = 2 * tp / f1_denominator if f1_denominator else 1.0
    return {'intersection': intersection, 'union': union, 'tp': tp, 'fp': fp, 'fn': fn, 'image_iou': float(image_iou), 'image_pixel_f1': float(image_f1)}

def build_forensics_prediction_records(batch: Mapping[str, Any], output: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    """Backward-compatible selected model classification record builder."""
    cls_pred = output['cls_pred'].detach().cpu()
    cls_prob = output['cls_prob'].detach().float().cpu()
    lm_pred = output['lm_verdict_pred'].detach().cpu()
    lm_prob = output['lm_verdict_prob'].detach().float().cpu()
    agree = output['cls_lm_agree'].detach().cpu()
    count = cls_pred.numel()
    fields: Sequence[tuple[str, Sequence[Any]]] = (('sample_ids', batch.get('sample_ids') or [None] * count), ('image_paths', batch.get('image_paths') or [None] * count), ('sources', batch.get('sources') or [None] * count), ('content_categories', batch.get('content_categories') or [None] * count))
    for (name, values) in fields:
        if len(values) != count:
            raise ValueError(f'{name} has {len(values)} values for {count} predictions')
    labels = batch.get('cls_labels')
    labels = labels.detach().cpu() if torch.is_tensor(labels) else labels
    records = []
    for index in range(count):
        records.append({'sample_id': fields[0][1][index], 'image_path': fields[1][1][index], 'source': fields[2][1][index], 'content_type': fields[3][1][index], 'gt_label': None if labels is None else VERDICT_NAMES[int(labels[index])], 'cls_pred': VERDICT_NAMES[int(cls_pred[index])], 'cls_prob': float(cls_prob[index]), 'lm_verdict_pred': VERDICT_NAMES[int(lm_pred[index])], 'lm_verdict_prob': float(lm_prob[index]), 'cls_lm_agree': bool(agree[index])})
    return records

def _binary_summary(labels: list[int], predictions: list[int], probabilities: list[float]) -> dict[str, Any]:
    if not labels:
        return {key: None for key in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')}
    tp = sum((gt == 1 and pred == 1 for (gt, pred) in zip(labels, predictions)))
    tn = sum((gt == 0 and pred == 0 for (gt, pred) in zip(labels, predictions)))
    fp = sum((gt == 0 and pred == 1 for (gt, pred) in zip(labels, predictions)))
    fn = sum((gt == 1 and pred == 0 for (gt, pred) in zip(labels, predictions)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    positive_scores = [p for (y, p) in zip(labels, probabilities) if y == 1]
    negative_scores = [p for (y, p) in zip(labels, probabilities) if y == 0]
    if positive_scores and negative_scores:
        wins = sum((float(pos > neg) + 0.5 * float(pos == neg) for pos in positive_scores for neg in negative_scores))
        roc_auc = wins / (len(positive_scores) * len(negative_scores))
    else:
        roc_auc = None
    return {'accuracy': (tp + tn) / len(labels), 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc, 'fake_precision': precision, 'fake_recall': recall, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

def summarize_detection_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate stored detection records without rerunning model inference."""
    labels = [int(record['gt_label'] == 'fake') for record in records]
    cls_predictions = [int(record['cls_pred'] == 'fake') for record in records]
    lm_predictions = [int(record['lm_verdict_pred'] == 'fake') for record in records]
    metrics = {'num_samples': len(records), 'classification_head': _binary_summary(labels, cls_predictions, [record.get('cls_prob_fake', record.get('cls_prob')) for record in records]), 'lm_verdict': _binary_summary(labels, lm_predictions, [record.get('lm_verdict_prob_fake', record.get('lm_verdict_prob')) for record in records]), 'cls_lm_agreement': sum((record['cls_lm_agree'] for record in records)) / len(records) if records else None, 'outcome_counts': {'cls_correct_lm_correct': 0, 'cls_correct_lm_wrong': 0, 'cls_wrong_lm_correct': 0, 'cls_wrong_lm_wrong': 0}}
    for record in records:
        cls_correct = record['cls_pred'] == record['gt_label']
        lm_correct = record['lm_verdict_pred'] == record['gt_label']
        key = f"cls_{('correct' if cls_correct else 'wrong')}_lm_{('correct' if lm_correct else 'wrong')}"
        metrics['outcome_counts'][key] += 1
    return metrics

def _summarize_record_group(records: Sequence[Mapping[str, Any]], *, autoregressive: bool, joint: bool) -> dict[str, Any]:
    count = len(records)
    intersection = sum((record['intersection'] for record in records))
    union = sum((record['union'] for record in records))
    tp = sum((record['tp'] for record in records))
    fp = sum((record['fp'] for record in records))
    fn = sum((record['fn'] for record in records))
    metrics = {'num_gt_fake': count, 'mean_iou': sum((record['image_iou'] for record in records)) / count if count else None, 'global_iou': intersection / union if union else 1.0 if count else None, 'mean_pixel_f1': sum((record['image_pixel_f1'] for record in records)) / count if count else None, 'global_pixel_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0 if count else None, 'intersection': intersection, 'union': union, 'tp': tp, 'fp': fp, 'fn': fn}
    if autoregressive:
        triggered = sum((record['seg_triggered'] and record['has_pred_mask'] for record in records))
        metrics.update({'seg_trigger_rate': triggered / count if count else None, 'seg_trigger_failure_count': count - triggered})
        lengths = [record.get('num_generated_tokens') for record in records]
        lengths = sorted((int(value) for value in lengths if value is not None))
        if lengths:
            middle = len(lengths) // 2
            median = lengths[middle] if len(lengths) % 2 else (lengths[middle - 1] + lengths[middle]) / 2
            metrics.update({'mean_generated_token_length': sum(lengths) / len(lengths), 'median_generated_token_length': median, 'eos_before_seg_rate': sum((record.get('stop_reason') == 'EOS_BEFORE_SEG' for record in records)) / count, 'max_token_before_seg_rate': sum((record.get('stop_reason') == 'MAX_NEW_TOKENS_BEFORE_SEG' for record in records)) / count, 'repetition_rate': sum((bool(record.get('repetition_statistics', {}).get('is_repetition_loop')) for record in records)) / count})
    if joint:
        passed = sum((record['cls_pred'] == 'fake' for record in records))
        metrics.update({'fake_recall': passed / count if count else None, 'classification_pass_rate': passed / count if count else None, 'joint_mean_iou': metrics['mean_iou'], 'joint_global_iou': metrics['global_iou'], 'joint_mean_pixel_f1': metrics['mean_pixel_f1'], 'joint_global_pixel_f1': metrics['global_pixel_f1']})
    return metrics

def summarize_localization_records(records: Sequence[Mapping[str, Any]], *, autoregressive: bool, joint: bool=False) -> dict[str, Any]:
    """Report overall and fixed content-category localization summaries."""
    overall = _summarize_record_group(records, autoregressive=autoregressive, joint=joint)
    by_content = {}
    for group in CONTENT_GROUPS:
        selected = [record for record in records if str(record.get('content_type', '')).lower() == group]
        by_content[group] = _summarize_record_group(selected, autoregressive=autoregressive, joint=joint)
    return {**overall, 'overall': overall, 'by_content_type': by_content}
