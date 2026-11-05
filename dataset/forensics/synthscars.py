"""Image-grouped SynthScars adapter for unified forensic supervision.

The official JSON is annotation-centric: an outer annotation id maps to a
record containing an image name, a caption, and one or more grounded refs.
This adapter groups records by image, retains every source annotation id/ref,
and exposes one boolean-union evidence mask per image for the first unified
training stage. Per-ref masks remain available for later phrase-mask training.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


SCHEMA_VERSION = "synthscars_image_grouped_v1"
FAKE_CLASS_LABEL = 1
FAKE_VERDICT_TOKEN = "[FAKE]"


class SynthScarsFormatError(ValueError):
    """Raised when an official SynthScars annotation violates the expected schema."""


def _normalise_text(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SynthScarsFormatError(f"{context}: {field} must be a non-empty string")
    return " ".join(value.strip().strip('"').split())


def _normalise_optional_text(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str):
        raise SynthScarsFormatError(f"{context}: {field} must be a string")
    return " ".join(value.strip().strip('"').split())


def _normalise_polygon(polygon: Any, context: str) -> List[float]:
    if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
        raise SynthScarsFormatError(
            f"{context}: polygon must be a flat list with at least three coordinate pairs"
        )
    try:
        values = [float(value) for value in polygon]
    except (TypeError, ValueError) as error:
        raise SynthScarsFormatError(f"{context}: polygon contains a non-numeric coordinate") from error
    if not all(math.isfinite(value) for value in values):
        raise SynthScarsFormatError(f"{context}: polygon contains a non-finite coordinate")
    return values


def _normalise_bbox(bbox: Any, context: str) -> List[float] | None:
    # The released SynthScars JSON includes the bbox key but stores null for all
    # current train/test refs. Keep that value traceable; masks come from polygon.
    if bbox is None:
        return None
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise SynthScarsFormatError(f"{context}: bbox must be null or contain four values")
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError) as error:
        raise SynthScarsFormatError(f"{context}: bbox contains a non-numeric value") from error
    if not all(math.isfinite(value) for value in values):
        raise SynthScarsFormatError(f"{context}: bbox contains a non-finite value")
    return values


def _parse_ref(raw_ref: Any, annotation_id: str, ref_index: int) -> Dict[str, Any]:
    context = f"annotation {annotation_id}, ref {ref_index}"
    if not isinstance(raw_ref, dict):
        raise SynthScarsFormatError(f"{context}: ref must be an object")
    raw_polygons = raw_ref.get("segmentation")
    if not isinstance(raw_polygons, list) or not raw_polygons:
        raise SynthScarsFormatError(f"{context}: segmentation must contain at least one polygon")
    return {
        "ref_id": f"{annotation_id}:{ref_index}",
        "annotation_id": annotation_id,
        "phrase": _normalise_text(raw_ref.get("sentence"), "sentence", context),
        # Three released train refs have an empty explanation. Preserve and
        # audit them; their annotation-level caption is still non-empty.
        "explanation": _normalise_optional_text(raw_ref.get("explanation"), "explanation", context),
        "bbox": _normalise_bbox(raw_ref.get("bbox"), context),
        "polygons": [
            _normalise_polygon(polygon, f"{context}, polygon {polygon_index}")
            for polygon_index, polygon in enumerate(raw_polygons)
        ],
    }


def _parse_annotation(raw_annotation: Any, row_index: int) -> Dict[str, Any]:
    context = f"row {row_index}"
    if not isinstance(raw_annotation, dict) or len(raw_annotation) != 1:
        raise SynthScarsFormatError(f"{context}: expected one outer annotation-id key")
    annotation_id, raw_record = next(iter(raw_annotation.items()))
    annotation_id = str(annotation_id)
    if not isinstance(raw_record, dict):
        raise SynthScarsFormatError(f"{context}, annotation {annotation_id}: record must be an object")
    raw_image_name = raw_record.get("img_file_name")
    if not isinstance(raw_image_name, str) or not raw_image_name.strip():
        raise SynthScarsFormatError(f"{context}: img_file_name must be a non-empty string")
    image_name = raw_image_name.strip()
    if Path(image_name).name != image_name:
        raise SynthScarsFormatError(f"{context}: img_file_name must be a bare file name")
    raw_refs = raw_record.get("refs")
    if not isinstance(raw_refs, list) or not raw_refs:
        raise SynthScarsFormatError(f"{context}, annotation {annotation_id}: refs must be non-empty")
    refs = [_parse_ref(ref, annotation_id, index) for index, ref in enumerate(raw_refs)]
    for ref in refs:
        ref["source_image_name"] = image_name
    return {
        "annotation_id": annotation_id,
        "image_name": image_name,
        "caption": _normalise_text(raw_record.get("caption"), "caption", context),
        "refs": refs,
    }


def _build_merged_explanation(captions: Sequence[str], refs: Sequence[Mapping[str, Any]]) -> str:
    unique_captions = list(dict.fromkeys(captions))
    if len(unique_captions) == 1:
        return unique_captions[0]

    clauses = []
    seen = set()
    for ref in refs:
        key = (ref["phrase"].casefold(), ref["explanation"].casefold())
        if key in seen:
            continue
        seen.add(key)
        phrase = ref["phrase"].rstrip(" .:")
        explanation = ref["explanation"].strip()
        if explanation:
            if explanation[-1] not in ".!?":
                explanation += "."
            clauses.append(f"{phrase}: {explanation}")
        else:
            clauses.append(f"{phrase}.")
    return "Upon examining the image, I found the following synthetic artifacts. " + " ".join(clauses)


def polygon_to_mask(polygons: Sequence[Sequence[float]], height: int, width: int) -> np.ndarray:
    """Rasterize COCO polygons and return a strictly binary uint8 mask."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: {(height, width)}")
    union = np.zeros((height, width), dtype=bool)
    for polygon in polygons:
        rle = coco_mask.frPyObjects([list(polygon)], height, width)
        decoded = coco_mask.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        union |= decoded.astype(bool)
    return union.astype(np.uint8)


class SynthScarsAdapter:
    """Load an official split and expose one deterministic sample per image."""

    def __init__(self, root: Path | str, split: str = "train") -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.annotation_path = self.root / split / "annotations" / f"{split}.json"
        self.images_dir = self.root / split / "images"
        if not self.annotation_path.is_file():
            raise FileNotFoundError(self.annotation_path)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(self.images_dir)

        with self.annotation_path.open("r", encoding="utf-8") as handle:
            raw_annotations = json.load(handle)
        if not isinstance(raw_annotations, list):
            raise SynthScarsFormatError(f"{self.annotation_path}: top-level JSON must be a list")

        parsed = [_parse_annotation(row, index) for index, row in enumerate(raw_annotations)]
        self.annotation_count = len(parsed)
        grouped: MutableMapping[str, List[Dict[str, Any]]] = OrderedDict()
        for annotation in parsed:
            image_identity = Path(annotation["image_name"]).stem
            grouped.setdefault(image_identity, []).append(annotation)
        self.samples = [
            self._group_image(image_identity, annotations) for image_identity, annotations in grouped.items()
        ]

    def _group_image(self, image_identity: str, annotations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        image_names = list(dict.fromkeys(str(annotation["image_name"]) for annotation in annotations))
        image_variants = []
        for image_name in image_names:
            with Image.open(self.images_dir / image_name) as image:
                width, height = image.size
            image_variants.append({"image_name": image_name, "image_size": [height, width]})
        canonical = max(
            image_variants,
            key=lambda variant: (
                variant["image_size"][0] * variant["image_size"][1],
                Path(variant["image_name"]).suffix.lower() == ".png",
                variant["image_name"],
            ),
        )
        variant_sizes = {variant["image_name"]: variant["image_size"] for variant in image_variants}
        refs = []
        for annotation in annotations:
            for raw_ref in annotation["refs"]:
                ref = dict(raw_ref)
                ref["source_image_size"] = list(variant_sizes[ref["source_image_name"]])
                refs.append(ref)
        captions = [str(annotation["caption"]) for annotation in annotations]
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_id": f"synthscars:{self.split}:{image_identity}",
            "source": "SynthScars",
            "split": self.split,
            "image_identity": image_identity,
            "image_name": canonical["image_name"],
            "image_relpath": str(Path(self.split) / "images" / canonical["image_name"]),
            "image_variants": image_variants,
            "annotation_relpath": str(Path(self.split) / "annotations" / f"{self.split}.json"),
            "annotation_ids": [str(annotation["annotation_id"]) for annotation in annotations],
            "annotation_captions": captions,
            "class_label": FAKE_CLASS_LABEL,
            "verdict_token": FAKE_VERDICT_TOKEN,
            "explanation": _build_merged_explanation(captions, refs),
            "refs": refs,
            "has_class_label": True,
            "has_explanation": True,
            "has_mask_label": True,
            "generator": None,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get_sample(index, decode_masks=True)

    def get_sample(self, index: int, decode_masks: bool = True) -> Dict[str, Any]:
        sample = dict(self.samples[index])
        sample["refs"] = [dict(ref) for ref in self.samples[index]["refs"]]
        image_path = self.root / sample["image_relpath"]
        sample["image_path"] = str(image_path)
        if not decode_masks:
            return sample
        with Image.open(image_path) as image:
            width, height = image.size
        ref_masks = [
            polygon_to_mask(polygons_for_target(ref, height, width), height, width) for ref in sample["refs"]
        ]
        union = np.zeros((height, width), dtype=bool)
        for ref_mask in ref_masks:
            union |= ref_mask.astype(bool)
        sample["image_size"] = [height, width]
        sample["ref_phrases"] = [ref["phrase"] for ref in sample["refs"]]
        sample["ref_explanations"] = [ref["explanation"] for ref in sample["refs"]]
        sample["ref_masks"] = ref_masks
        sample["ref_mask_valid"] = [bool(np.any(ref_mask)) for ref_mask in ref_masks]
        sample["invalid_ref_ids"] = [
            ref["ref_id"] for ref, is_valid in zip(sample["refs"], sample["ref_mask_valid"]) if not is_valid
        ]
        sample["union_evidence_mask"] = union.astype(np.uint8)
        return sample

    def iter_manifest_records(self) -> Iterable[Dict[str, Any]]:
        for index in range(len(self)):
            yield self.get_sample(index, decode_masks=False)

    @property
    def duplicate_image_samples(self) -> List[Dict[str, Any]]:
        return [sample for sample in self.samples if len(sample["annotation_ids"]) > 1]


def polygons_for_target(ref: Mapping[str, Any], target_height: int, target_width: int) -> List[List[float]]:
    """Scale a ref's original polygon coordinates to a canonical image size."""
    source_height, source_width = ref["source_image_size"]
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"Invalid source image size for {ref['ref_id']}: {ref['source_image_size']}")
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scaled = []
    for polygon in ref["polygons"]:
        values = []
        for index, coordinate in enumerate(polygon):
            values.append(float(coordinate) * (scale_x if index % 2 == 0 else scale_y))
        scaled.append(values)
    return scaled


def ref_signature(ref: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Stable exact-match signature used only for duplicate annotation auditing."""
    polygons = tuple(tuple(float(value) for value in polygon) for polygon in ref["polygons"])
    return (
        str(ref["phrase"]).casefold(),
        str(ref["explanation"]).casefold(),
        None if ref["bbox"] is None else tuple(float(value) for value in ref["bbox"]),
        polygons,
    )
