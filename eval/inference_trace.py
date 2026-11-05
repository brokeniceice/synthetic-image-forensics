"""Read-only tensor tracing for frozen GLaMM localization inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import torch

from model.GLaMM import extract_seg_predictor_hidden


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    save_tokens: bool = True
    save_hidden: bool = True
    save_projection: bool = True
    save_sam_prompt: bool = True
    save_masks: bool = True


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_descriptor(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
        "sha256": tensor_sha256(tensor), "min": float(value.min()) if value.numel() else None,
        "max": float(value.max()) if value.numel() else None,
        "mean": float(value.mean()) if value.numel() else None,
        "norm": float(value.norm()) if value.numel() else 0.0,
    }


def raw_seg_positions(input_ids: torch.Tensor, seg_token_idx: int) -> tuple[int, int]:
    positions = input_ids[0].eq(seg_token_idx).nonzero(as_tuple=False).flatten()
    if positions.numel() != 1:
        raise ValueError(f"Expected exactly one [SEG], found {positions.numel()}")
    seg_position = int(positions[0])
    if seg_position == 0:
        raise ValueError("[SEG] cannot be the first raw token")
    return seg_position, seg_position - 1


@torch.no_grad()
def trace_full_forward(model, batch: Mapping[str, Any], input_ids: torch.Tensor, *,
                       original_size: tuple[int, int]) -> dict[str, Any]:
    """Replay one exact sequence through the canonical full-forward mask path."""
    if input_ids.shape[0] != 1:
        raise ValueError("Phase 2D.1 trace currently requires batch size one")
    return trace_full_forward_batch(model, batch, input_ids, original_sizes=[original_size])[0]


@torch.no_grad()
def trace_full_forward_batch(model, batch: Mapping[str, Any], input_ids: torch.Tensor, *,
                             original_sizes: list[tuple[int, int]]) -> list[dict[str, Any]]:
    """Replay a fixed padded batch, preserving the historical GEMM batch shape."""
    if input_ids.shape[0] != len(original_sizes):
        raise ValueError("original_sizes must contain one entry per sequence")
    core = unwrap_glamm(model)
    attention = input_ids.ne(core.config.pad_token_id)
    output, hidden_states = core._inference_path(
        input_ids, batch["global_enc_images"], attention, batch["offset"], batch["bboxes"]
    )
    last_hidden = core._get_last_hidden_state(hidden_states)
    raw_hidden, expanded_positions = extract_seg_predictor_hidden(
        last_hidden, input_ids, core.seg_token_idx
    )
    projected = [core.model.text_hidden_fcs[0](value) for value in raw_hidden]
    image_embeddings = core.get_grounding_encoder_embs(batch["grounding_enc_images"])
    traces = []
    for index, (predictor_hidden, projected_embedding, image_embedding, original_size) in enumerate(
            zip(raw_hidden, projected, image_embeddings, original_sizes)):
        row_ids = input_ids[index:index + 1]
        if predictor_hidden.shape[0] == 0:
            empty_mask = predictor_hidden.new_empty((0, *original_size))
            traces.append({
                "input_ids": row_ids.detach(), "attention_mask": attention[index:index + 1].detach(),
                "position_ids": None, "seg_token_id": int(core.seg_token_idx),
                "seg_token_position_raw": None, "predictor_position_raw": None,
                "predictor_position_expanded": None, "llm_predictor_hidden": predictor_hidden.detach(),
                "projected_embedding": projected_embedding.detach(), "image_embedding": image_embedding.detach(),
                "sparse_prompt_embedding": predictor_hidden.new_empty((0, core.config.out_dim)),
                "dense_prompt_embedding": predictor_hidden.new_empty((0,)),
                "low_res_mask_logits": predictor_hidden.new_empty((0, 256, 256)),
                "postprocessed_mask_logits": empty_mask, "binary_mask": empty_mask.bool(),
                "lm_logits_at_predictor": predictor_hidden.new_empty((0, core.config.vocab_size)),
                "execution_semantics": f"full_sequence_causal_forward_no_cache_batch_{input_ids.shape[0]}",
            })
            continue
        if predictor_hidden.shape[0] != 1:
            raise ValueError("Trace requires at most one predictor hidden per sequence")
        sparse, dense = core.model.grounding_encoder.prompt_encoder(
            points=None, boxes=None, masks=None, text_embeds=projected_embedding.unsqueeze(1)
        )
        sparse = sparse.to(projected_embedding.dtype)
        low_res, _ = core.model.grounding_encoder.mask_decoder(
            image_embeddings=image_embedding.unsqueeze(0),
            image_pe=core.model.grounding_encoder.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        postprocessed = core.model.grounding_encoder.postprocess_masks(
            low_res, input_size=batch["resize_list"][index], original_size=original_size
        )[:, 0]
        seg_raw, predictor_raw = raw_seg_positions(row_ids, core.seg_token_idx)
        expanded = int(expanded_positions[index][0])
        traces.append({
            "input_ids": row_ids.detach(), "attention_mask": attention[index:index + 1].detach(),
            "position_ids": None, "seg_token_id": int(core.seg_token_idx),
            "seg_token_position_raw": seg_raw, "predictor_position_raw": predictor_raw,
            "predictor_position_expanded": expanded,
            "llm_predictor_hidden": predictor_hidden.detach(),
            "projected_embedding": projected_embedding.detach(), "image_embedding": image_embedding.detach(),
            "sparse_prompt_embedding": sparse.detach(), "dense_prompt_embedding": dense.detach(),
            "low_res_mask_logits": low_res.detach(), "postprocessed_mask_logits": postprocessed.detach(),
            "binary_mask": postprocessed.gt(0).detach(),
            "lm_logits_at_predictor": output.logits[index:index + 1, expanded].detach(),
            "execution_semantics": f"full_sequence_causal_forward_no_cache_batch_{input_ids.shape[0]}",
        })
    return traces


def trace_descriptors(trace: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: tensor_descriptor(value) if torch.is_tensor(value) else value
        for key, value in trace.items()
    }
def unwrap_glamm(model):
    """Return the GLaMMForCausalLM beneath an optional PEFT wrapper."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model
