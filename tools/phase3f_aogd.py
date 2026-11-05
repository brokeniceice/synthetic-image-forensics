"""Shared fixed-trajectory representation helpers for Phase 3F AOGD."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from eval.forensics import compute_binary_mask_metrics
from eval.inference_trace import unwrap_glamm
from model.GLaMM import extract_seg_predictor_hidden


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_group(name: str) -> str:
    if "lora_" in name:
        return "lora"
    if "text_hidden_fcs" in name:
        return "text_hidden_fcs"
    if "grounding_encoder.mask_decoder" in name:
        return "mask_decoder"
    return "frozen"


def tensor_collection_hash(items) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def group_hashes(model) -> dict:
    return {
        group: tensor_collection_hash((name, parameter) for name, parameter in model.named_parameters()
                                      if parameter_group(name) == group)
        for group in ("lora", "text_hidden_fcs", "mask_decoder", "frozen")
    }


def configure_lora_only(model) -> dict:
    counts = {group: 0 for group in ("lora", "text_hidden_fcs", "mask_decoder", "frozen")}
    tensors = {group: 0 for group in counts}
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        parameter.requires_grad = group == "lora"
        counts[group] += parameter.numel()
        tensors[group] += 1
    if counts["lora"] <= 0:
        raise RuntimeError("LoRA parameter group is empty")
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if parameter_group(name) != "lora"):
        raise RuntimeError("Phase 3F parameter boundary leak")
    return {"trainable": ["lora"], "parameter_counts": counts, "tensor_counts": tensors,
            "trainable_parameters": counts["lora"]}


def grad_norm(model, group="lora") -> float:
    values = [parameter.grad.detach().float().pow(2).sum()
              for name, parameter in model.named_parameters()
              if parameter_group(name) == group and parameter.grad is not None]
    return math.sqrt(sum(float(value) for value in values)) if values else 0.0


def replay_batch(backend, sample, replay_row, question: str):
    batch = backend._batch(sample, "", question=question)
    ids = torch.tensor(replay_row["full_input_token_ids"], dtype=torch.long, device=batch["input_ids"].device).unsqueeze(0)
    batch["input_ids"] = ids
    batch["attention_masks"] = torch.ones_like(ids, dtype=torch.bool)
    return batch


def capture_representations(core, batch):
    _, hidden_all = core._inference_path(
        batch["input_ids"], batch["global_enc_images"], batch["attention_masks"],
        batch["offset"], batch["bboxes"],
    )
    last = core._get_last_hidden_state(hidden_all)
    raw, raw_positions = extract_seg_predictor_hidden(last, batch["input_ids"], core.seg_token_idx)
    projected, positions = core._extract_projected_seg_predictor_hidden(
        hidden_all, batch["input_ids"], batch["offset"]
    )
    if len(raw) != 1 or len(projected) != 1 or raw[0].shape[0] != 1 or projected[0].shape[0] != 1:
        raise RuntimeError(f"fixed trajectory must have exactly one predictor state: raw={list(map(tuple, [x.shape for x in raw]))}")
    return raw[0][0], projected[0][0], raw_positions, positions


def cosine_loss(student, teacher):
    return 1.0 - torch.nn.functional.cosine_similarity(
        student.float().reshape(1, -1), teacher.float().reshape(1, -1), dim=-1
    ).mean()


def mask_metrics_from_projected(core, batch, projected):
    image_embeddings = core.get_grounding_encoder_embs(batch["grounding_enc_images"])
    masks = core._generate_and_postprocess_masks(
        [projected.reshape(1, -1)], image_embeddings, batch["resize_list"], batch["label_list"]
    )
    logits = masks[0][0].detach().float().cpu()
    target = torch.as_tensor(batch["masks_list"][0]).bool().any(dim=0).cpu()
    metrics = compute_binary_mask_metrics(logits, target)
    return {"foreground_iou": float(metrics["image_iou"]), "foreground_f1": float(metrics["image_pixel_f1"])}


def load_jsonl_index(path: Path) -> dict:
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[row["sample_id"]] = row
    return result


def core_model(model):
    return unwrap_glamm(model)
