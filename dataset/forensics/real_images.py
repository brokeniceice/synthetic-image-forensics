"""Unified adapter for manifest-pinned real-image sources."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from PIL import Image


SCHEMA_VERSION = "real_images_unified_v1"
REAL_CLASS_LABEL = 0
REAL_VERDICT_TOKEN = "[REAL]"
REAL_EXPLANATION = "No reliable synthetic artifact is found."

TRAIN_CANDIDATE_SOURCES = ("OpenImagesV7", "PASS", "COCO2017", "FFHQ", "iNaturalist")
HELDOUT_SOURCES = ("RAISE-1k",)
ALL_SOURCES = TRAIN_CANDIDATE_SOURCES + HELDOUT_SOURCES


class RealManifestError(ValueError):
    """Raised when a pinned real-source manifest cannot be mapped safely."""


def _required_text(record: Mapping[str, Any], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RealManifestError(f"{context}: {key} must be a non-empty string")
    return value.strip()


def _normalise_category(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    value = value.strip().lower()
    aliases = {
        "human": "human",
        "person": "human",
        "people": "human",
        "object": "object",
        "animal": "animal",
        "scene": "scene",
    }
    return aliases.get(value, "unknown")


def _raise_content_category(keywords: str) -> str:
    tags = {tag.strip().lower() for tag in keywords.split(";") if tag.strip()}
    if "people" in tags:
        return "human"
    if tags & {"landscape", "outdoor", "indoor", "buildings", "nature"}:
        return "scene"
    if "objects" in tags:
        return "object"
    return "unknown"


def _source_role(source: str) -> str:
    return "heldout_test" if source in HELDOUT_SOURCES else "candidate_train"


def _map_source_record(source: str, record: Mapping[str, Any], row_index: int) -> Dict[str, Any]:
    context = f"{source} manifest row {row_index}"
    if source == "OpenImagesV7":
        record_id = _required_text(record, "ImageID", context)
        image_name = f"{record_id}.jpg"
        category = _normalise_category(record.get("category"))
        labels = list(record.get("labels") or [])
        license_name = record.get("License")
        original_page = record.get("OriginalLandingURL")
        author = record.get("Author")
    elif source == "PASS":
        record_id = _required_text(record, "image_hash", context)
        image_name = f"{record_id}.jpg"
        category = "unknown"
        labels = []
        license_name = record.get("licensename")
        original_page = None
        author = record.get("unickname")
    elif source == "COCO2017":
        record_id = str(record.get("id"))
        if record_id == "None":
            raise RealManifestError(f"{context}: id is required")
        image_name = _required_text(record, "file_name", context)
        category = _normalise_category(record.get("category"))
        labels = list(record.get("instance_categories") or [])
        license_metadata = record.get("license_metadata") or {}
        license_name = license_metadata.get("name")
        original_page = record.get("flickr_url") or record.get("coco_url")
        author = None
    elif source == "FFHQ":
        image_id = record.get("image_id")
        if not isinstance(image_id, int):
            raise RealManifestError(f"{context}: image_id must be an integer")
        record_id = str(image_id)
        image_name = f"{image_id:05d}.png"
        category = "human"
        labels = ["aligned face"]
        metadata = record.get("metadata") or {}
        license_name = metadata.get("license")
        original_page = metadata.get("photo_url")
        author = metadata.get("author")
    elif source == "iNaturalist":
        record_id = f"{record.get('observation_id')}:{record.get('photo_id')}"
        if "None" in record_id:
            raise RealManifestError(f"{context}: observation_id and photo_id are required")
        image_name = _required_text(record, "file_name", context)
        category = "animal"
        taxon = record.get("taxon") or {}
        labels = [value for value in (record.get("group"), taxon.get("name"), taxon.get("preferred_common_name")) if value]
        license_name = record.get("photo_license")
        original_page = record.get("observation_url")
        author = record.get("observer")
    elif source == "RAISE-1k":
        record_id = _required_text(record, "raise_id", context)
        image_name = _required_text(record, "file_name", context)
        keywords = str(record.get("keywords") or "")
        category = _raise_content_category(keywords)
        labels = [tag.strip().lower() for tag in keywords.split(";") if tag.strip()]
        license_name = "RAISE non-commercial research/education"
        original_page = "https://loki.disi.unitn.it/RAISE/download.html"
        author = None
    else:
        raise RealManifestError(f"Unsupported real source: {source}")

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"real:{source}:{record_id}",
        "source": source,
        "source_role": _source_role(source),
        "source_record_id": record_id,
        "source_record_index": row_index,
        "source_manifest_relpath": str(Path(source) / "manifests" / "selected.jsonl"),
        "image_name": image_name,
        "image_relpath": str(Path(source) / "images" / image_name),
        "class_label": REAL_CLASS_LABEL,
        "verdict_token": REAL_VERDICT_TOKEN,
        "explanation": REAL_EXPLANATION,
        "content_category": category,
        "content_labels": labels,
        "content_label_source": "source_manifest" if category != "unknown" else "unlabeled",
        "license": license_name,
        "original_page": original_page,
        "author": author,
        "has_class_label": True,
        "has_explanation": True,
        "has_mask_label": False,
        "refs": [],
        "annotation_ids": [],
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RealManifestError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise RealManifestError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records


def _checksum_key(source: str, record: Mapping[str, Any]) -> str:
    if source == "OpenImagesV7":
        return _required_text(record, "ImageID", f"{source} checksum")
    if source == "PASS":
        return _required_text(record, "image_hash", f"{source} checksum")
    return _required_text(record, "file_name", f"{source} checksum")


class UnifiedRealImageAdapter:
    """Combine pinned manifests while retaining source roles and provenance."""

    def __init__(self, datasets_root: Path | str, sources: Sequence[str] = ALL_SOURCES) -> None:
        self.datasets_root = Path(datasets_root).expanduser().resolve()
        self.sources = tuple(sources)
        unsupported = sorted(set(self.sources) - set(ALL_SOURCES))
        if unsupported:
            raise ValueError(f"Unsupported sources: {unsupported}")

        self.samples: List[Dict[str, Any]] = []
        self.source_manifest_paths: Dict[str, Path] = {}
        for source in self.sources:
            manifest_path = self.datasets_root / source / "manifests" / "selected.jsonl"
            images_dir = self.datasets_root / source / "images"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            if not images_dir.is_dir():
                raise FileNotFoundError(images_dir)
            self.source_manifest_paths[source] = manifest_path
            records = _load_jsonl(manifest_path)
            mapped = [_map_source_record(source, record, index) for index, record in enumerate(records)]
            if source == "FFHQ":
                for sample, record in zip(mapped, records):
                    image_metadata = record.get("image") or {}
                    sample["identity_hash"] = {
                        "algorithm": "md5",
                        "value": image_metadata.get("file_md5"),
                    }
                    sample["bytes"] = image_metadata.get("file_size")
            else:
                checksum_path = self.datasets_root / source / "manifests" / "checksums_sha256.jsonl"
                if not checksum_path.is_file():
                    raise FileNotFoundError(checksum_path)
                checksum_records = _load_jsonl(checksum_path)
                checksum_index = {_checksum_key(source, record): record for record in checksum_records}
                for sample in mapped:
                    if source == "OpenImagesV7":
                        checksum_key = sample["source_record_id"]
                    elif source == "PASS":
                        checksum_key = sample["source_record_id"]
                    else:
                        checksum_key = sample["image_name"]
                    checksum = checksum_index.get(checksum_key)
                    if checksum is None:
                        raise RealManifestError(f"{source}: no checksum for {checksum_key}")
                    sample["identity_hash"] = {"algorithm": "sha256", "value": checksum["sha256"]}
                    sample["bytes"] = checksum.get("bytes")
            self.samples.extend(mapped)

        ids = [sample["sample_id"] for sample in self.samples]
        if len(ids) != len(set(ids)):
            duplicates = [sample_id for sample_id, count in Counter(ids).items() if count > 1]
            raise RealManifestError(f"Duplicate unified sample ids: {duplicates[:10]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get_sample(index, decode_mask=True)

    def get_sample(self, index: int, decode_mask: bool = True) -> Dict[str, Any]:
        sample = dict(self.samples[index])
        image_path = self.datasets_root / sample["image_relpath"]
        sample["image_path"] = str(image_path)
        sample["ref_phrases"] = []
        sample["ref_explanations"] = []
        sample["ref_masks"] = []
        if decode_mask:
            with Image.open(image_path) as image:
                width, height = image.size
            sample["image_size"] = [height, width]
            # Authentic samples intentionally have no segmentation target.  Do
            # not represent "not supervised" as an all-zero foreground mask:
            # doing so would turn Real images into negative mask supervision.
            sample["union_evidence_mask"] = None
        return sample

    def iter_manifest_records(self, role: str | None = None) -> Iterable[Dict[str, Any]]:
        for sample in self.samples:
            if role is None or sample["source_role"] == role:
                yield dict(sample)

    def indices_for_role(self, role: str) -> List[int]:
        return [index for index, sample in enumerate(self.samples) if sample["source_role"] == role]
