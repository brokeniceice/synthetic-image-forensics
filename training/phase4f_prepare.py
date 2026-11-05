#!/usr/bin/env python3
"""Freeze the lightweight P1 SAM runtime and Phase 4F manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.sam_forensic_rectifier import FrozenP1SAMPath
from tools.phase4f import Phase4FStore, dump, file_sha256, ids_sha256, q_index, tensor_state_sha256

CFG = yaml.safe_load((ROOT / "configs/phase4f_language_preserving_rectification.yaml").read_text())
OUT = ROOT / CFG["experiment"]["output_root"]
RUNTIME = Path(CFG["experiment"]["runtime_root"])


def strip(state: dict, prefix: str) -> dict:
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def extract_runtime() -> Path:
    destination = RUNTIME / "p1_sam_runtime.pt"
    if destination.exists():
        return destination
    p1 = Path(CFG["p1"]["checkpoint"])
    if file_sha256(p1) != CFG["p1"]["checkpoint_sha256"]:
        raise RuntimeError("canonical P1 checkpoint hash mismatch")
    index_path = ROOT / CFG["p1"]["base_model"] / "pytorch_model.bin.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    prompt_keys = [key for key in index if key.startswith("model.grounding_encoder.prompt_encoder.")]
    shard_names = sorted({index[key] for key in prompt_keys})
    base = {}
    base_hashes = {}
    for shard_name in shard_names:
        path = index_path.parent / shard_name
        base_hashes[shard_name] = file_sha256(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        base.update({key: state[key] for key in prompt_keys if key in state})
        del state
    prompt = strip(base, "model.grounding_encoder.prompt_encoder.")
    checkpoint = torch.load(p1, map_location="cpu", weights_only=False)
    module = checkpoint["module"]
    p1_prompt = strip(module, "base_model.model.model.grounding_encoder.prompt_encoder.")
    prompt.update(p1_prompt)
    mask = strip(module, "base_model.model.model.grounding_encoder.mask_decoder.")
    runtime = FrozenP1SAMPath()
    missing_prompt, unexpected_prompt = runtime.prompt_encoder.load_state_dict(prompt, strict=True)
    missing_mask, unexpected_mask = runtime.mask_decoder.load_state_dict(mask, strict=True)
    if missing_prompt or unexpected_prompt or missing_mask or unexpected_mask:
        raise RuntimeError("P1 SAM runtime state mismatch")
    payload = {
        "schema": "phase4f_p1_sam_runtime_v1",
        "prompt_encoder": runtime.prompt_encoder.state_dict(),
        "mask_decoder": runtime.mask_decoder.state_dict(),
        "prompt_encoder_hash": tensor_state_sha256(runtime.prompt_encoder.state_dict()),
        "mask_decoder_hash": tensor_state_sha256(runtime.mask_decoder.state_dict()),
        "p1_checkpoint": str(p1),
        "p1_checkpoint_sha256": file_sha256(p1),
        "base_index_sha256": file_sha256(index_path),
        "base_shard_sha256": base_hashes,
    }
    RUNTIME.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    dump(RUNTIME / "p1_sam_runtime_manifest.json", {key: value for key, value in payload.items() if key not in ("prompt_encoder", "mask_decoder")} | {"runtime_sha256": file_sha256(destination)})
    return destination


def main() -> None:
    runtime = extract_runtime()
    train = Phase4FStore(CFG, "train")
    val = Phase4FStore(CFG, "val")
    modes = {}
    for mode, name in (("g0", "G0"), ("phrase", "phrase_only"), ("tf", "tf_full_context")):
        q, records = q_index(Path(CFG["data"]["q_cache_root"]) / "validation", name)
        modes[mode] = {"n": len(q), "valid": sum(valid for _, valid in q.values()), "ids_sha256": ids_sha256([str(row["sample_id"]) for row in records])}
    train_valid = sorted(train.valid_ids)
    train_invalid = sorted(set(train.sample_ids) - train.valid_ids)
    val_valid = sorted(val.valid_ids)
    val_invalid = sorted(set(val.sample_ids) - val.valid_ids)
    if (len(train_valid), len(train_invalid), len(val_valid), len(val_invalid)) != (8690, 146, 1078, 28):
        raise RuntimeError("valid-G0 frozen counts mismatch")
    manifest = {
        "status": "COMPLETE",
        "phase": "Phase 4F",
        "p1_runtime": str(runtime),
        "p1_runtime_sha256": file_sha256(runtime),
        "p1_checkpoint_sha256": file_sha256(Path(CFG["p1"]["checkpoint"])),
        "canonical_prompt_sha256": CFG["p1"]["prompt_hash"],
        "hidden_extraction": CFG["p1"]["hidden_extraction"],
        "train": {"n": len(train.sample_ids), "ids_sha256": ids_sha256(train.sample_ids), "valid_g0": len(train_valid), "invalid_g0": len(train_invalid), "valid_ids_sha256": ids_sha256(train_valid), "invalid_ids_sha256": ids_sha256(train_invalid)},
        "validation": {"n": len(val.sample_ids), "ids_sha256": ids_sha256(val.sample_ids), "valid_g0": len(val_valid), "invalid_g0": len(val_invalid), "valid_ids_sha256": ids_sha256(val_valid), "invalid_ids_sha256": ids_sha256(val_invalid), "modes": modes},
        "internal_test_accessed": False,
        "official1000_accessed": False,
    }
    dump(OUT / "manifests/frozen_protocol.json", manifest)
    dump(OUT / "manifests/train_valid_g0_ids.json", train_valid)
    dump(OUT / "manifests/train_invalid_g0_ids.json", train_invalid)
    dump(OUT / "manifests/val_valid_g0_ids.json", val_valid)
    dump(OUT / "manifests/val_invalid_g0_ids.json", val_invalid)
    schedules = {str(epoch): deterministic for epoch in []}
    from tools.phase4f import deterministic_order
    schedules = {str(epoch): deterministic_order(train.sample_ids, epoch) for epoch in range(1, 11)}
    dump(OUT / "manifests/training_schedules.json", {"seed": 3407, "epochs": {epoch: {"ids_sha256": ids_sha256(ids), "sample_ids": ids} for epoch, ids in schedules.items()}})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
