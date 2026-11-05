"""Shared frozen-protocol utilities for Phase 4B-G."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def dump(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def balanced_schedule(rows: list[dict], *, exposures: int, seed: int) -> dict:
    if exposures % 4:
        raise ValueError("Phase 4B-G exposure count must be divisible by effective batch size four")
    by_label = {label: [i for i, row in enumerate(rows) if int(row["class_label"]) == label]
                for label in (0, 1)}
    rng = random.Random(seed)
    for values in by_label.values():
        rng.shuffle(values)
    cursors = {0: 0, 1: 0}
    indices = []
    for _ in range(exposures // 4):
        batch = []
        for label in (0, 1):
            for _ in range(2):
                if cursors[label] == len(by_label[label]):
                    rng.shuffle(by_label[label]); cursors[label] = 0
                batch.append(by_label[label][cursors[label]])
                cursors[label] += 1
        rng.shuffle(batch)
        indices.extend(batch)
    sample_ids = [str(rows[index]["sample_id"]) for index in indices]
    labels = [int(rows[index]["class_label"]) for index in indices]
    result = {"seed": seed, "construction": "deterministic_2real_2fake_per_optimizer_step",
              "indices": indices, "sample_ids": sample_ids, "class_labels": labels,
              "exposures": exposures, "class_histogram": dict(Counter(labels))}
    result["sha256"] = canonical_sha256({key: result[key] for key in
        ("seed", "construction", "indices", "sample_ids", "class_labels")})
    return result


class FrozenFeatureStore:
    """CPU-resident sample-ID keyed immutable FEPN feature cache."""

    def __init__(self, path: str | Path, *, expected_sha256: str | None = None) -> None:
        self.path = Path(path)
        if expected_sha256 is not None and file_sha256(self.path) != expected_sha256:
            raise RuntimeError(f"feature cache hash mismatch: {self.path}")
        state = torch.load(self.path, map_location="cpu")
        self.features = state["features"].contiguous()
        self.sample_ids = list(map(str, state["sample_ids"]))
        self.index = {sid: index for index, sid in enumerate(self.sample_ids)}
        if len(self.index) != len(self.sample_ids):
            raise RuntimeError("duplicate sample IDs in FEPN feature cache")
        if tuple(self.features.shape) != (len(self.sample_ids), 128):
            raise RuntimeError(f"unexpected cache shape: {tuple(self.features.shape)}")

    def get(self, sample_ids: list[str], *, device, dtype=torch.bfloat16) -> torch.Tensor:
        indices = torch.tensor([self.index[str(sid)] for sid in sample_ids], dtype=torch.long)
        return self.features.index_select(0, indices).to(device=device, dtype=dtype)
