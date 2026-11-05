"""Data, geometry, metrics, and hashing helpers for Phase 4E-1."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset.forensics.synthscars import polygon_to_mask, polygons_for_target
from tools.phase3c1 import binary_metrics, inverse_logits, paired_statistics, transform_mask

ROOT = Path(__file__).resolve().parents[1]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(str(tuple(value.shape)).encode() + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def sam_coordinates(geometry: dict, grid: int = 32) -> torch.Tensor:
    resized_h, resized_w = map(float, geometry["resized_hw"])
    y = (torch.arange(grid, dtype=torch.float32) + 0.5) * 1024.0 / grid / resized_h
    x = (torch.arange(grid, dtype=torch.float32) + 0.5) * 1024.0 / grid / resized_w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def clip_coordinates(geometry: dict, grid: int = 24) -> torch.Tensor:
    resized_h, resized_w = map(float, geometry["resized_hw"])
    top, left, _, _ = geometry["crop_box_yxyx"]
    y = (float(top) + (torch.arange(grid, dtype=torch.float32) + 0.5) * 336.0 / grid) / resized_h
    x = (float(left) + (torch.arange(grid, dtype=torch.float32) + 0.5) * 336.0 / grid) / resized_w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def region_targets(row: dict, geometry: dict) -> torch.Tensor:
    height, width = map(int, geometry["original_hw"])
    values = []
    for ref in row.get("refs") or []:
        mask = torch.from_numpy(
            polygon_to_mask(polygons_for_target(ref, height, width), height, width).astype(np.float32)
        )
        sam_mask = transform_mask(mask, geometry).float()[None, None]
        # Token occupancy target: preserve any authoritative positive pixel in each S32 cell.
        values.append(F.adaptive_max_pool2d(sam_mask, (32, 32))[0, 0].to(torch.uint8))
    if not values:
        raise RuntimeError(f"Fake sample lacks region target: {row.get('sample_id')}")
    return torch.stack(values)


class FrozenStore:
    """RAM-resident immutable train/validation cache with ID-addressed access."""

    def __init__(self, cfg: dict, split: str, *, keep_original: bool = False) -> None:
        self.cfg, self.split = cfg, split
        manifest_rows = rows(ROOT / cfg["data"]["manifest_dir"] / f"{split}_combined.jsonl")
        self.rows = {str(row["sample_id"]): row for row in manifest_rows if int(row["class_label"]) == 1}
        self.sample_ids: list[str] = []
        self.sam_features, self.clip_features = [], []
        self.geometries, self.original_masks = {}, {}
        self.locations = {}
        spatial = ROOT / cfg["data"]["spatial_cache_root"]
        sam_paths = sorted((spatial / "sam" / split).glob("shard_*.pt"))
        clip_paths = sorted((spatial / "clip" / split).glob("shard_*.pt"))
        if len(sam_paths) != len(clip_paths) or not sam_paths:
            raise RuntimeError("spatial cache shard mismatch")
        for shard_index, (sam_path, clip_path) in enumerate(zip(sam_paths, clip_paths)):
            sam = torch.load(sam_path, map_location="cpu", weights_only=False)
            clip = torch.load(clip_path, map_location="cpu", weights_only=False)
            sam_ids = [str(row["sample_id"]) for row in sam["records"]]
            clip_ids = [str(row["sample_id"]) for row in clip["records"]]
            if sam_ids != clip_ids:
                raise RuntimeError("SAM/CLIP cache ID mismatch")
            self.sam_features.append(sam["features"])
            self.clip_features.append(clip["features"])
            for local, (sid, record) in enumerate(zip(sam_ids, sam["records"])):
                self.locations[sid] = (shard_index, local)
                self.geometries[sid] = record["geometry"]
                if keep_original:
                    self.original_masks[sid] = sam["original_masks"][local].bool()
            self.sample_ids.extend(sam_ids)
        if set(self.sample_ids) != set(self.rows):
            raise RuntimeError("spatial cache/manifest Fake set mismatch")
        hidden_root = Path("cache/phase4e1_tf_fdg_raw_hidden") / split
        self.tf_hidden, self.g0_hidden, self.phrase_hidden = {}, {}, {}
        for path in sorted(hidden_root.glob("shard_*.pt")):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            for index, sid in enumerate(shard["sample_ids"]):
                self.tf_hidden[sid] = shard["tf_hidden"][index]
            for position, hidden in zip(shard["g0_positions"].tolist(), shard["g0_hidden"]):
                self.g0_hidden[shard["sample_ids"][position]] = hidden
            if shard.get("phrase_hidden") is not None:
                for sid, hidden in zip(shard["sample_ids"], shard["phrase_hidden"]):
                    self.phrase_hidden[sid] = hidden
        if set(self.tf_hidden) != set(self.sample_ids):
            raise RuntimeError("TF hidden cache population mismatch")
        self.targets = {sid: region_targets(self.rows[sid], self.geometries[sid]) for sid in self.sample_ids}
        self.sam_coords = {sid: sam_coordinates(self.geometries[sid]) for sid in self.sample_ids}
        clip_geometry_by_id = {}
        for path in clip_paths:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            for record in shard["records"]:
                clip_geometry_by_id[str(record["sample_id"])] = record["geometry"]
        self.clip_coords = {sid: clip_coordinates(clip_geometry_by_id[sid]) for sid in self.sample_ids}

    def spatial_batch(self, sample_ids: list[str], device: torch.device):
        sam = torch.stack([self.sam_features[self.locations[sid][0]][self.locations[sid][1]] for sid in sample_ids])
        clip = torch.stack([self.clip_features[self.locations[sid][0]][self.locations[sid][1]] for sid in sample_ids])
        sam_coord = torch.stack([self.sam_coords[sid] for sid in sample_ids])
        clip_coord = torch.stack([self.clip_coords[sid] for sid in sample_ids])
        return (
            sam.to(device=device, dtype=torch.bfloat16, non_blocking=True),
            clip.to(device=device, dtype=torch.bfloat16, non_blocking=True),
            sam_coord.to(device=device, non_blocking=True),
            clip_coord.to(device=device, non_blocking=True),
        )

    def hidden_batch(self, sample_ids: list[str], mode: str, device: torch.device) -> torch.Tensor:
        source = self.tf_hidden if mode == "tf" else (self.phrase_hidden if mode == "phrase" else self.g0_hidden)
        return torch.stack([source[sid] for sid in sample_ids]).to(device=device, dtype=torch.bfloat16)


def deterministic_order(sample_ids: list[str], seed: int, epoch: int) -> list[str]:
    result = list(sample_ids)
    random.Random(seed + 1009 * epoch).shuffle(result)
    return result


def summarize(records: list[dict]) -> dict:
    iou = np.asarray([row["foreground_iou"] for row in records], dtype=np.float64)
    f1 = np.asarray([row["foreground_f1"] for row in records], dtype=np.float64)
    tp, fp, fn = (sum(int(row[key]) for row in records) for key in ("tp", "fp", "fn"))
    return {
        "n": len(records), "mean_foreground_iou": float(iou.mean()),
        "median_foreground_iou": float(np.median(iou)), "mean_foreground_f1": float(f1.mean()),
        "global_foreground_iou": tp / max(1, tp + fp + fn),
        "global_foreground_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "threshold_logit": 0.0,
    }


def compare(left: list[dict], right: list[dict], seed: int = 3407) -> dict:
    if [row["sample_id"] for row in left] != [row["sample_id"] for row in right]:
        raise RuntimeError("paired result identity mismatch")
    return {
        key: paired_statistics([row[key] for row in left], [row[key] for row in right], seed=seed)
        for key in ("foreground_iou", "foreground_f1")
    }


def metric(sample_id: str, logits32: torch.Tensor, original_mask: torch.Tensor, geometry: dict) -> dict:
    original_logits = inverse_logits(logits32, geometry)
    value = binary_metrics(original_logits, original_mask)
    return {"sample_id": sample_id, **value}
