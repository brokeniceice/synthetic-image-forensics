"""Language-preserving dense-evidence rectification for Phase 4F."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.SAM.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from model.tf_fdg import CrossAttentiveSemanticRectification


def build_prompt_encoder() -> PromptEncoder:
    return PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    )


def build_mask_decoder() -> MaskDecoder:
    return MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )


class FrozenP1SAMPath(nn.Module):
    """Only the frozen P1 prompt encoder and mask decoder; S64 is precomputed."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_encoder = build_prompt_encoder()
        self.mask_decoder = build_mask_decoder()

    def freeze(self) -> "FrozenP1SAMPath":
        return self.eval().requires_grad_(False)

    def forward(self, q_seg: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        if q_seg.ndim == 2:
            q_seg = q_seg[:, None]
        if q_seg.shape[0] != image_embeddings.shape[0]:
            raise ValueError("q_seg/image_embeddings batch mismatch")
        # SAM's native decoder contract repeats one image embedding for the
        # prompt batch.  P1 therefore decodes images one by one; retain that
        # exact contract while allowing the rectifier/loss to use batch eight.
        values = []
        for index in range(q_seg.shape[0]):
            current = q_seg[index:index + 1]
            sparse, dense = self.prompt_encoder(points=None, boxes=None, masks=None, text_embeds=current)
            sparse = sparse.to(current.dtype)
            low_res, _ = self.mask_decoder(
                image_embeddings=image_embeddings[index:index + 1],
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            values.append(low_res)
        return torch.cat(values, dim=0)


class GeometryAwareSAMRectifier(nn.Module):
    """Inject F24 only into the dense S64 image side of the original P1 SAM path."""

    def __init__(self, gamma_init: float, heads: int = 8, locality_sigma: float = 0.25) -> None:
        super().__init__()
        self.rectification = CrossAttentiveSemanticRectification(
            dim=256, heads=heads, gamma_init=gamma_init,
        )
        self.rectification.cross_attention.locality_sigma = float(locality_sigma)

    @staticmethod
    def semantic_support(
        sam_coordinates: torch.Tensor,
        evidence_coordinates: torch.Tensor,
        evidence_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sam_coordinates.ndim == 2:
            sam_coordinates = sam_coordinates[None]
        if evidence_coordinates.ndim == 2:
            evidence_coordinates = evidence_coordinates[None]
        if evidence_valid is None:
            minimum = evidence_coordinates.amin(dim=1)
            maximum = evidence_coordinates.amax(dim=1)
        else:
            valid = evidence_valid if evidence_valid.ndim == 2 else evidence_valid[None]
            if valid.shape != evidence_coordinates.shape[:2]:
                raise ValueError("evidence_valid/evidence_coordinates shape mismatch")
            if not bool(valid.any(dim=1).all()):
                raise ValueError("every sample must contain at least one valid evidence token")
            minimum = torch.where(
                valid[..., None], evidence_coordinates,
                torch.full_like(evidence_coordinates, float("inf")),
            ).amin(dim=1)
            maximum = torch.where(
                valid[..., None], evidence_coordinates,
                torch.full_like(evidence_coordinates, float("-inf")),
            ).amax(dim=1)
        return ((sam_coordinates >= minimum[:, None]) & (sam_coordinates <= maximum[:, None])).all(-1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        evidence: torch.Tensor,
        sam_coordinates: torch.Tensor,
        evidence_coordinates: torch.Tensor,
        evidence_valid: torch.Tensor | None = None,
        *,
        enabled: bool = True,
        semantic_support: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not enabled:
            return {
                "image_embeddings": image_embeddings,
                "residual": torch.zeros_like(image_embeddings),
                "attention": image_embeddings.new_empty(0),
                "support": torch.ones(
                    image_embeddings.shape[0], 4096, dtype=torch.bool, device=image_embeddings.device
                ),
            }
        semantic = image_embeddings.flatten(2).transpose(1, 2)
        forensic = evidence.flatten(2).transpose(1, 2)
        expected = forensic.shape[:2]
        if evidence_coordinates.shape[-2:] != (expected[1], 2):
            raise ValueError("evidence/evidence_coordinates token-count mismatch")
        if evidence_valid is not None and evidence_valid.shape[-1] != expected[1]:
            raise ValueError("evidence/evidence_valid token-count mismatch")
        support = (
            self.semantic_support(sam_coordinates, evidence_coordinates, evidence_valid)
            if semantic_support is None else semantic_support
        )
        if support.shape != semantic.shape[:2]:
            raise ValueError("semantic_support must match flattened SAM tokens")
        rectified, attention, residual = self.rectification(
            semantic, forensic, sam_coordinates, evidence_coordinates,
            evidence_valid, support,
        )
        grid = rectified.transpose(1, 2).reshape_as(image_embeddings)
        residual_grid = residual.transpose(1, 2).reshape_as(image_embeddings)
        return {
            "image_embeddings": grid,
            "residual": residual_grid,
            "attention": attention,
            "support": support,
        }
