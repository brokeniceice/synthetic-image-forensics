"""Frozen data/runtime helpers for Phase 4F."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.clip_forensic_adapter import CLIPSpatialArm
from model.sam_forensic_rectifier import FrozenP1SAMPath, GeometryAwareSAMRectifier
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record, spatial_cache_paths
from tools.phase4e1 import clip_coordinates, compare, sam_coordinates, summarize, tensor_state_sha256

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


def ids_sha256(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()


def deterministic_order(sample_ids: list[str], epoch: int, seed: int = 3407) -> list[str]:
    result = list(sample_ids)
    random.Random(seed + 1009 * epoch).shuffle(result)
    return result


def q_index(root: Path, context: str) -> tuple[dict[str, tuple[torch.Tensor, bool]], list[dict]]:
    values, records = {}, []
    for path in sorted((root / context).glob("shard_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        valid = shard.get("valid", torch.ones(len(shard["records"]), dtype=torch.bool))
        for index, record in enumerate(shard["records"]):
            sid = str(record["sample_id"])
            values[sid] = (shard["q_seg"][index], bool(valid[index]))
            records.append(record)
    return values, records


class Phase4FStore:
    """RAM-resident immutable S64/F24/union-target store."""

    def __init__(self, cfg: dict, split: str) -> None:
        self.cfg, self.split = cfg, split
        root = ROOT / cfg["data"]["spatial_cache_root"]
        sam_paths = spatial_cache_paths(root, "sam", split)
        clip_paths = spatial_cache_paths(root, "clip", split)
        if len(sam_paths) != len(clip_paths):
            raise RuntimeError("SAM/CLIP shard mismatch")
        self.sample_ids, self.locations = [], {}
        self.sam_features, self.clip_features, self.targets = [], [], []
        self.original_masks, self.geometries, self.sam_coords, self.clip_coords = {}, {}, {}, {}
        for shard_index, (sam_path, clip_path) in enumerate(zip(sam_paths, clip_paths)):
            sam = torch.load(sam_path, map_location="cpu", weights_only=False)
            clip = torch.load(clip_path, map_location="cpu", weights_only=False)
            sam_ids = [str(row["sample_id"]) for row in sam["records"]]
            clip_ids = [str(row["sample_id"]) for row in clip["records"]]
            if sam_ids != clip_ids:
                raise RuntimeError("SAM/CLIP ID mismatch")
            self.sam_features.append(sam["features"])
            self.clip_features.append(clip["features"])
            self.targets.append(sam["targets"])
            original = sam.get("original_masks")
            for local, (sid, sam_meta, clip_meta) in enumerate(zip(sam_ids, sam["records"], clip["records"])):
                self.locations[sid] = (shard_index, local)
                self.sample_ids.append(sid)
                geometry = sam_meta["geometry"]
                self.geometries[sid] = geometry
                self.sam_coords[sid] = sam_coordinates(geometry, grid=64)
                self.clip_coords[sid] = clip_coordinates(clip_meta["geometry"], grid=24)
                if original is not None:
                    self.original_masks[sid] = original[local].bool()
        expected = int(cfg["data"]["train_fake"] if split == "train" else cfg["data"]["val_fake"])
        if len(self.sample_ids) != expected or len(set(self.sample_ids)) != expected:
            raise RuntimeError(f"{split} population mismatch")
        qroot = Path(cfg["data"]["q_cache_root"])
        if split == "train":
            self.q, self.q_records = q_index(qroot, "train_q_seg")
        else:
            self.q, self.q_records = q_index(qroot / "validation", "G0")
        self.valid_ids = {sid for sid, (_, valid) in self.q.items() if valid}

    def _values(self, sid: str):
        shard, local = self.locations[sid]
        return self.sam_features[shard][local], self.clip_features[shard][local], self.targets[shard][local]

    def batch(self, sample_ids: list[str], device: torch.device):
        values = [self._values(sid) for sid in sample_ids]
        return (
            torch.stack([row[0] for row in values]).to(device=device, dtype=torch.bfloat16, non_blocking=True),
            torch.stack([row[1] for row in values]).to(device=device, dtype=torch.bfloat16, non_blocking=True),
            torch.stack([row[2] for row in values]).to(device=device, non_blocking=True),
            torch.stack([self.sam_coords[sid] for sid in sample_ids]).to(device=device, non_blocking=True),
            torch.stack([self.clip_coords[sid] for sid in sample_ids]).to(device=device, non_blocking=True),
            torch.stack([self.q[sid][0] for sid in sample_ids]).to(device=device, dtype=torch.bfloat16, non_blocking=True),
        )

    def q_for_mode(self, mode: str) -> tuple[dict[str, tuple[torch.Tensor, bool]], list[dict]]:
        if self.split != "val":
            raise RuntimeError("language modes only exist for validation")
        name = {"g0": "G0", "phrase": "phrase_only", "tf": "tf_full_context"}[mode]
        return q_index(Path(self.cfg["data"]["q_cache_root"]) / "validation", name)


def load_sam_runtime(cfg: dict, device: torch.device) -> FrozenP1SAMPath:
    path = Path(cfg["experiment"]["runtime_root"]) / "p1_sam_runtime.pt"
    value = torch.load(path, map_location="cpu", weights_only=False)
    model = FrozenP1SAMPath()
    model.prompt_encoder.load_state_dict(value["prompt_encoder"], strict=True)
    model.mask_decoder.load_state_dict(value["mask_decoder"], strict=True)
    return model.to(device=device, dtype=torch.bfloat16).freeze()


def load_evidence_source(cfg: dict, arm: str, device: torch.device) -> CLIPSpatialArm:
    is_clip = arm == "clip_rect"
    key = "clip_checkpoint" if is_clip else "forensic_checkpoint"
    path = Path(cfg["evidence"][key])
    if file_sha256(path) != cfg["evidence"][key + "_sha256"]:
        raise RuntimeError("Phase 4C-A evidence checkpoint hash mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if int(value["epoch"]) != 4:
        raise RuntimeError("Phase 4C-A selected epoch mismatch")
    model = CLIPSpatialArm(blocks=0 if is_clip else 3)
    model.load_state_dict(value["model"], strict=True)
    model._phase4f_arm = arm
    return model.to(device).eval().requires_grad_(False)


def evidence_feature(source: CLIPSpatialArm, raw_clip: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        value = source(raw_clip, return_features=True)
    return (value["F0"] if source._phase4f_arm == "clip_rect" else value["F_forensic"]).detach()


def load_rectifier(cfg: dict, gamma: float, device: torch.device) -> GeometryAwareSAMRectifier:
    torch.manual_seed(int(cfg["experiment"]["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["experiment"]["seed"]))
    return GeometryAwareSAMRectifier(
        gamma_init=gamma,
        heads=int(cfg["architecture"]["heads"]),
        locality_sigma=float(cfg["architecture"]["locality_sigma"]),
    ).to(device)


def decode(
    sam: FrozenP1SAMPath,
    rectifier: GeometryAwareSAMRectifier,
    q_seg: torch.Tensor,
    s64: torch.Tensor,
    evidence: torch.Tensor,
    sam_coords: torch.Tensor,
    clip_coords: torch.Tensor,
    *,
    enabled: bool = True,
) -> tuple[torch.Tensor, dict]:
    valid = torch.ones(evidence.shape[0], 576, dtype=torch.bool, device=evidence.device)
    value = rectifier(s64, evidence, sam_coords, clip_coords, valid, enabled=enabled)
    # Historical P1 executes its already-BF16 SAM prompt/mask path without an
    # outer autocast context.  Autocast changes the TwoWayTransformer kernels
    # enough to break exact rectifier-off recovery, despite identical weights.
    with torch.autocast(device_type=value["image_embeddings"].device.type, enabled=False):
        low = sam(q_seg.to(torch.bfloat16), value["image_embeddings"].to(torch.bfloat16))
    return low, value


def mask_loss(low: torch.Tensor, targets: torch.Tensor, cfg: dict) -> dict[str, torch.Tensor]:
    logits = F.interpolate(low.float(), size=targets.shape[-2:], mode="bilinear", align_corners=False)
    target = targets.float()[:, None]
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    intersection = 2 * (probability / 1000 * target).flatten(1).sum(1)
    union = (probability / 1000).flatten(1).sum(1) + (target / 1000).flatten(1).sum(1)
    dice = (1 - (intersection + 1e-6) / (union + 1e-6)).mean()
    total = float(cfg["loss"]["mask_bce_weight"]) * bce + float(cfg["loss"]["mask_dice_weight"]) * dice
    return {"bce": bce, "dice": dice, "total": total}


def invalid_record(sid: str, target: torch.Tensor) -> dict:
    truth = target.bool()
    return {
        "sample_id": sid, "foreground_iou": 0.0, "foreground_f1": 0.0,
        "tp": 0, "fp": 0, "fn": int(truth.sum()), "valid_g0": False,
    }


def evaluate(
    cfg: dict,
    store: Phase4FStore,
    sam: FrozenP1SAMPath,
    rectifier: GeometryAwareSAMRectifier,
    source: CLIPSpatialArm,
    mode: str,
    device: torch.device,
    *,
    condition: str = "matched",
) -> tuple[dict, list[dict]]:
    rectifier.eval()
    qcache, _ = store.q_for_mode(mode)
    ids = list(store.sample_ids)
    cross = {sid: ids[(index + 1) % len(ids)] for index, sid in enumerate(ids)}
    permutation = torch.randperm(576, generator=torch.Generator().manual_seed(3407)).to(device)
    records = []
    with torch.no_grad():
        for position, sid in enumerate(ids, 1):
            if sid not in qcache or not qcache[sid][1]:
                records.append(invalid_record(sid, store.original_masks[sid]))
                continue
            evidence_sid = cross[sid] if condition == "cross_image" else sid
            s64, raw, _, sc, cc, _ = store.batch([sid], device)
            if evidence_sid != sid:
                _, raw, _, _, _, _ = store.batch([evidence_sid], device)
            evidence = evidence_feature(source, raw)
            if condition == "spatial_shuffle":
                evidence = evidence.flatten(2)[:, :, permutation].reshape_as(evidence)
            elif condition == "zero":
                evidence = torch.zeros_like(evidence)
            q = qcache[sid][0].to(device=device, dtype=torch.bfloat16)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                low, _ = decode(sam, rectifier, q, s64, evidence, sc, cc, enabled=condition != "rectifier_off")
            logits = inverse_sam_logits(low, store.geometries[sid])
            row = metric_record(sid, logits, store.original_masks[sid])
            row["valid_g0"] = True
            records.append(row)
            if position % 100 == 0:
                print(json.dumps({"mode": mode, "condition": condition, "done": position, "total": len(ids)}), flush=True)
    return summarize(records), records


__all__ = [
    "Phase4FStore", "append", "compare", "decode", "deterministic_order", "dump",
    "evaluate", "evidence_feature", "file_sha256", "ids_sha256", "load_evidence_source",
    "load_rectifier", "load_sam_runtime", "mask_loss", "q_index", "rows", "tensor_state_sha256",
]
