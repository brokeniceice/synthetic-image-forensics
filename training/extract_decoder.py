"""Extract the selected frozen P1 prompt encoder and mask decoder."""
from pathlib import Path
import argparse
import json
import torch
from model.sam_forensic_rectifier import FrozenP1SAMPath
from forensics.checkpoints import file_sha256
from tools.spatial_metrics import tensor_state_sha256

def strip(state: dict, prefix: str) -> dict:
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def extract_runtime(p1, base_model, destination):
    destination = Path(destination)
    p1 = Path(p1)
    if destination.exists():
        return destination
    if file_sha256(p1) != 'fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326':
        raise RuntimeError("canonical P1 checkpoint hash mismatch")
    index_path = Path(base_model) / "pytorch_model.bin.index.json"
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
        "schema": "corrected_p1_sam_runtime_v1",
        "prompt_encoder": runtime.prompt_encoder.state_dict(),
        "mask_decoder": runtime.mask_decoder.state_dict(),
        "prompt_encoder_hash": tensor_state_sha256(runtime.prompt_encoder.state_dict()),
        "mask_decoder_hash": tensor_state_sha256(runtime.mask_decoder.state_dict()),
        "p1_checkpoint": str(p1),
        "p1_checkpoint_sha256": file_sha256(p1),
        "base_index_sha256": file_sha256(index_path),
        "base_shard_sha256": base_hashes,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--p1', required=True)
    p.add_argument('--base-model', default='checkpoints/GLaMM-FullScope')
    p.add_argument('--output', default='checkpoints/p1_sam_runtime.pt')
    args = p.parse_args()
    print(extract_runtime(args.p1, args.base_model, args.output))

if __name__ == '__main__':
    main()
