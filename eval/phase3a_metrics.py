"""Deterministic parsing and artifact contracts for Phase 3A."""

from __future__ import annotations

import re
from typing import Any, Mapping


PREDICTION_REQUIRED_FIELDS = {
    "sample_id", "generated_token_ids", "decoded_text", "generated_localization_phrase",
    "seg_position", "mask_logits_path", "binary_mask_path", "tp", "fp", "fn", "tn",
    "foreground_iou", "foreground_f1", "fg_bg_miou", "mask_logit_threshold",
}


def parse_phrase_aligned_generation(text: str) -> dict[str, Any]:
    """Parse the frozen P1 output grammar without filling missing fields."""
    raw = str(text)
    verdict_match = re.search(r"\[(REAL|FAKE)\]", raw)
    target_matches = list(re.finditer(r"Target\s+regions?\s*:\s*", raw, flags=re.IGNORECASE))
    seg_matches = list(re.finditer(r"\[SEG\]", raw))
    verdict = verdict_match.group(1) if verdict_match else None
    target = None
    explanation = None
    status = []
    if verdict_match is None:
        status.append("MISSING_VERDICT")
    if len(target_matches) > 1:
        status.append("REPEATED_TARGET_FIELD")
    if target_matches:
        target_match = target_matches[0]
        end = seg_matches[0].start() if seg_matches and seg_matches[0].start() > target_match.end() else len(raw)
        target = raw[target_match.end():end].strip()
        if not target:
            status.append("EMPTY_TARGET_FIELD")
        if verdict_match:
            explanation = raw[verdict_match.end():target_match.start()].strip()
    elif verdict == "FAKE":
        status.append("MISSING_TARGET_FIELD")
    if verdict == "FAKE" and not seg_matches:
        status.append("MISSING_SEG")
    if len(seg_matches) > 1:
        status.append("MULTIPLE_SEG")
    return {
        "parse_success": not status,
        "parse_status": status or ["PASS"],
        "verdict": verdict,
        "explanation": explanation,
        "target_region": target,
        "target_field_present": bool(target_matches),
        "seg_triggered": bool(seg_matches),
        "seg_count": len(seg_matches),
        "raw_generation": raw,
    }


def validate_prediction_record(record: Mapping[str, Any]) -> None:
    missing = sorted(PREDICTION_REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"Phase 3A prediction record missing fields: {missing}")
    if float(record["mask_logit_threshold"]) != 0.0:
        raise ValueError("Phase 3A primary mask threshold must remain logit > 0")

