"""Phase 4E-0.5 TF-FDG architecture specification and audit implementation.

This module is deliberately not wired into any training entry point.  It
defines the frozen tensor contracts, geometry-aware rectification, coordinated
query decoder, soft union, assignment, and KD losses used by the Phase 4E-0.5
synthetic implementation audits.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def normalized_grid(height: int, width: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Cell-centre coordinates ``[H*W,2]`` in original-normalized ``(x,y)`` order."""
    yy = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xx = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def coordinate_sincos(coordinates: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Deterministic sine/cosine encoding for normalized ``(x,y)`` coordinates."""
    if embed_dim % 4:
        raise ValueError("coordinate encoding requires embed_dim divisible by four")
    quarter = embed_dim // 4
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(quarter, device=coordinates.device, dtype=coordinates.dtype)
        / max(1, quarter - 1)
    )
    values = []
    for axis in (0, 1):
        phase = coordinates[..., axis, None] * frequency * (2.0 * math.pi)
        values.extend((phase.sin(), phase.cos()))
    return torch.cat(values, dim=-1)


def soft_union(slot_logits: torch.Tensor, epsilon: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerically stable probabilistic OR over ``K`` slot logits.

    ``p_union = 1 - prod_k(1 - sigmoid(z_k))``.  Returns probability and
    continuous logit; no threshold is applied.
    """
    probabilities = slot_logits.float().sigmoid().clamp(epsilon, 1.0 - epsilon)
    log_not_union = torch.log1p(-probabilities).sum(dim=1, keepdim=True)
    union_probability = (-torch.expm1(log_not_union)).clamp(epsilon, 1.0 - epsilon)
    union_logit = torch.logit(union_probability, eps=epsilon)
    return union_probability, union_logit


def coordinate_resample(
    source_tokens: torch.Tensor, source_coordinates: torch.Tensor,
    target_coordinates: torch.Tensor, source_valid: torch.Tensor | None = None,
    neighbours: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically resample tokens through original-normalized coordinates.

    The method uses inverse-distance weighting over the nearest valid source
    cells. Target cells outside the source field of view (notably outside the
    CLIP center crop) are returned as zero with ``support=False``.
    """
    batch, source_count, dim = source_tokens.shape
    source_coordinates = source_coordinates if source_coordinates.ndim == 3 else source_coordinates[None].expand(batch, -1, -1)
    target_coordinates = target_coordinates if target_coordinates.ndim == 3 else target_coordinates[None].expand(batch, -1, -1)
    valid = torch.ones(batch, source_count, dtype=torch.bool, device=source_tokens.device) if source_valid is None else source_valid
    if valid.ndim == 1:
        valid = valid[None].expand(batch, -1)
    distance2 = (target_coordinates[:, :, None] - source_coordinates[:, None]).square().sum(-1)
    distance2 = distance2.masked_fill(~valid[:, None], float("inf"))
    values, indices = distance2.topk(min(neighbours, source_count), dim=-1, largest=False)
    gather_index = indices[..., None].expand(-1, -1, -1, dim)
    expanded = source_tokens[:, None].expand(-1, target_coordinates.shape[1], -1, -1)
    selected = expanded.gather(2, gather_index)
    weights = values.clamp_min(1e-8).reciprocal()
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    resampled = (selected * weights[..., None]).sum(2)
    minimum = torch.where(valid[..., None], source_coordinates, torch.full_like(source_coordinates, float("inf"))).amin(1)
    maximum = torch.where(valid[..., None], source_coordinates, torch.full_like(source_coordinates, float("-inf"))).amax(1)
    support = ((target_coordinates >= minimum[:, None]) & (target_coordinates <= maximum[:, None])).all(-1)
    return resampled * support[..., None], support


class QueryGenerator(nn.Module):
    def __init__(self, hidden_dim: int = 4096, query_dim: int = 256, slots: int = 4) -> None:
        super().__init__()
        self.hidden_dim, self.query_dim, self.slots = hidden_dim, query_dim, slots
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1024), nn.GELU(),
            nn.Linear(1024, slots * query_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden).reshape(hidden.shape[0], self.slots, self.query_dim)


class GeometryAwareCrossAttention(nn.Module):
    """Cross-attention with fixed original-coordinate bias and validity mask."""

    def __init__(self, dim: int = 256, heads: int = 8, locality_sigma: float = 0.25) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.locality_sigma = locality_sigma
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor,
        query_coordinates: torch.Tensor, key_coordinates: torch.Tensor,
        key_valid: torch.Tensor | None = None, value: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, nq, _ = query.shape
        value = key if value is None else value
        nk = key.shape[1]
        q = self.q_proj(query).reshape(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(b, nk, self.heads, self.head_dim).transpose(1, 2)
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        q_coord = query_coordinates if query_coordinates.ndim == 3 else query_coordinates[None].expand(b, -1, -1)
        k_coord = key_coordinates if key_coordinates.ndim == 3 else key_coordinates[None].expand(b, -1, -1)
        distance2 = (q_coord[:, :, None] - k_coord[:, None]).square().sum(dim=-1)
        score = score - distance2[:, None] / (2.0 * self.locality_sigma ** 2)
        if key_valid is not None:
            valid = key_valid if key_valid.ndim == 2 else key_valid[None].expand(b, -1)
            score = score.masked_fill(~valid[:, None, None], torch.finfo(score.dtype).min)
        attention = score.softmax(dim=-1)
        value = torch.matmul(attention, v).transpose(1, 2).reshape(b, nq, self.dim)
        return self.out_proj(value), attention


class CrossAttentiveSemanticRectification(nn.Module):
    """``S_rect = S + gamma * Projection(CrossAttention(LN(S), LN(F), F))``."""

    def __init__(self, dim: int = 256, heads: int = 8, gamma_init: float = 0.01) -> None:
        super().__init__()
        if gamma_init == 0:
            raise ValueError("gamma must be nonzero at initialization")
        self.semantic_norm = nn.LayerNorm(dim)
        self.forensic_norm = nn.LayerNorm(dim)
        self.cross_attention = GeometryAwareCrossAttention(dim, heads)
        self.projection = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(
        self, semantic: torch.Tensor, forensic: torch.Tensor,
        semantic_coordinates: torch.Tensor, forensic_coordinates: torch.Tensor,
        forensic_valid: torch.Tensor | None = None, semantic_support: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic_pe = coordinate_sincos(semantic_coordinates, semantic.shape[-1])
        forensic_pe = coordinate_sincos(forensic_coordinates, forensic.shape[-1])
        response, attention = self.cross_attention(
            self.semantic_norm(semantic) + semantic_pe,
            self.forensic_norm(forensic) + forensic_pe,
            semantic_coordinates, forensic_coordinates, forensic_valid, value=forensic,
        )
        residual = self.gamma * self.projection(response)
        if semantic_support is not None:
            residual = residual * semantic_support[..., None]
        return semantic + residual, attention, residual


class DualPathDecoderLayer(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.semantic_norm = nn.LayerNorm(dim)
        self.semantic_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.forensic_norm = nn.LayerNorm(dim)
        self.forensic_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 1024), nn.GELU(), nn.Linear(1024, dim))

    def forward(self, queries: torch.Tensor, semantic: torch.Tensor, forensic: torch.Tensor):
        value, _ = self.self_attention(self.self_norm(queries), self.self_norm(queries), self.self_norm(queries))
        queries = queries + value
        value, semantic_attention = self.semantic_attention(
            self.semantic_norm(queries), semantic, semantic,
            need_weights=True, average_attn_weights=False,
        )
        queries = queries + value
        value, forensic_attention = self.forensic_attention(
            self.forensic_norm(queries), forensic, forensic,
            need_weights=True, average_attn_weights=False,
        )
        queries = queries + value
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries, semantic_attention, forensic_attention


class TFFDGStudent(nn.Module):
    """CPU-auditable tensor contract for the frozen four-layer TF-FDG student."""

    def __init__(self, slots: int = 4, dim: int = 256, gamma_init: float = 0.01,
                 *, use_forensic: bool = True, use_rectification: bool = True) -> None:
        super().__init__()
        self.slots, self.dim = slots, dim
        self.use_forensic = bool(use_forensic)
        self.use_rectification = bool(use_rectification)
        self.query_generator = QueryGenerator(slots=slots, query_dim=dim)
        self.semantic_pyramid = nn.Conv2d(dim, dim, 1)
        self.forensic_pyramid = nn.Conv2d(dim, dim, 1)
        self.rectification = CrossAttentiveSemanticRectification(dim, gamma_init=gamma_init)
        self.decoder = nn.ModuleList([DualPathDecoderLayer(dim) for _ in range(4)])
        self.mask_feature = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.mask_embedding = nn.Linear(dim, dim)

    def forward(
        self, hidden: torch.Tensor, semantic_grid: torch.Tensor, forensic_grid: torch.Tensor,
        semantic_coordinates: torch.Tensor | None = None,
        forensic_coordinates: torch.Tensor | None = None,
        forensic_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        semantic_grid = F.adaptive_avg_pool2d(self.semantic_pyramid(semantic_grid), (32, 32))
        forensic_grid = self.forensic_pyramid(forensic_grid)
        semantic = semantic_grid.flatten(2).transpose(1, 2)
        forensic = forensic_grid.flatten(2).transpose(1, 2)
        semantic_coordinates = semantic_coordinates if semantic_coordinates is not None else normalized_grid(32, 32, device=hidden.device)
        forensic_coordinates = forensic_coordinates if forensic_coordinates is not None else normalized_grid(24, 24, device=hidden.device)
        forensic_32_tokens, semantic_support = coordinate_resample(
            forensic, forensic_coordinates, semantic_coordinates, forensic_valid,
        )
        if self.use_forensic and self.use_rectification:
            semantic, rect_attention, residual = self.rectification(
                semantic, forensic, semantic_coordinates, forensic_coordinates, forensic_valid, semantic_support,
            )
        else:
            rect_attention = semantic.new_zeros(
                semantic.shape[0], self.rectification.cross_attention.heads,
                semantic.shape[1], forensic.shape[1],
            )
            residual = torch.zeros_like(semantic)
        queries = self.query_generator(hidden)
        semantic_attentions, forensic_attentions, features = [], [], []
        for index, layer in enumerate(self.decoder):
            if self.use_forensic:
                queries, sem_attn, for_attn = layer(queries, semantic, forensic)
            else:
                value, _ = layer.self_attention(
                    layer.self_norm(queries), layer.self_norm(queries), layer.self_norm(queries)
                )
                queries = queries + value
                value, sem_attn = layer.semantic_attention(
                    layer.semantic_norm(queries), semantic, semantic,
                    need_weights=True, average_attn_weights=False,
                )
                queries = queries + value
                queries = queries + layer.ffn(layer.ffn_norm(queries))
                for_attn = queries.new_zeros(
                    queries.shape[0], layer.forensic_attention.num_heads,
                    queries.shape[1], forensic.shape[1],
                )
            semantic_attentions.append(sem_attn)
            forensic_attentions.append(for_attn)
            if index in (1, 3):
                features.append(queries)
        forensic_32 = forensic_32_tokens.transpose(1, 2).reshape(semantic_grid.shape)
        if not self.use_forensic:
            forensic_32 = torch.zeros_like(forensic_32)
        fused = self.mask_feature(torch.cat((semantic.transpose(1, 2).reshape_as(semantic_grid), forensic_32), dim=1))
        slot_logits = torch.einsum("bkd,bdhw->bkhw", self.mask_embedding(queries), fused) / math.sqrt(self.dim)
        union_probability, union_logit = soft_union(slot_logits)
        return {
            "queries": queries, "semantic_attention": semantic_attentions,
            "forensic_attention": forensic_attentions, "decoder_features": features,
            "slot_logits": slot_logits, "union_probability": union_probability,
            "union_logit": union_logit, "rectification_attention": rect_attention,
            "rectification_residual": residual, "semantic_tokens": semantic,
            "forensic_tokens": forensic,
        }


@dataclass(frozen=True)
class AssignmentWeights:
    mask: float = 1.0
    logit: float = 1.0
    attention: float = 0.25


def _dice_cost(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    s, t = student.sigmoid().flatten(1), teacher.sigmoid().flatten(1)
    return 1.0 - (2.0 * torch.einsum("ip,jp->ij", s, t) + 1e-6) / (
        s.sum(1)[:, None] + t.sum(1)[None] + 1e-6
    )


def _logit_cost(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return (student[:, None] - teacher[None]).abs().flatten(2).mean(-1)


def _js_cost(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    s = student.flatten(1).softmax(-1)
    t = teacher.flatten(1).softmax(-1)
    midpoint = 0.5 * (s[:, None] + t[None])
    return 0.5 * (
        (s[:, None] * (s[:, None].clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(-1)
        + (t[None] * (t[None].clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(-1)
    )


def hungarian_teacher_assignment(
    student_slot_logits: torch.Tensor, teacher_slot_logits: torch.Tensor,
    student_attention: torch.Tensor, teacher_attention: torch.Tensor,
    weights: AssignmentWeights = AssignmentWeights(),
) -> torch.Tensor:
    """Return teacher index for each student slot; assignment cost is stop-gradient."""
    with torch.no_grad():
        cost = (
            weights.mask * _dice_cost(student_slot_logits.detach(), teacher_slot_logits.detach())
            + weights.logit * _logit_cost(student_slot_logits.detach(), teacher_slot_logits.detach())
            + weights.attention * _js_cost(student_attention.detach(), teacher_attention.detach())
        )
        row, col = linear_sum_assignment(cost.float().cpu().numpy())
    order = np.empty(len(row), dtype=np.int64)
    order[row] = col
    return torch.as_tensor(order, device=student_slot_logits.device, dtype=torch.long)


def region_aware_slot_loss(slot_logits: torch.Tensor, region_masks: torch.Tensor) -> dict[str, torch.Tensor | str]:
    """Authoritative region supervision for ``M<=K`` and union-only overflow.

    Assignment cost is exactly ``DiceCost + BCECost`` and is detached. For
    ``M>K`` no arbitrary region bundles are invented; the caller uses the
    returned soft-union logit with the authoritative union target and mandatory
    collapse diagnostics.
    """
    if slot_logits.ndim != 3 or region_masks.ndim != 3:
        raise ValueError("expected slot logits [K,H,W] and region masks [M,H,W]")
    slots, regions = slot_logits.shape[0], region_masks.shape[0]
    # The frozen protocol requires FP32 mask/union losses even when module
    # compute is autocast BF16.  Keep this promotion explicit so Hungarian
    # cost and BCE/Dice never mix BF16 probabilities with FP32 targets.
    loss_logits = slot_logits.float()
    union_probability, union_logit = soft_union(loss_logits[None])
    union_target = region_masks.bool().any(0, keepdim=True).float()[None]
    union_loss = F.binary_cross_entropy_with_logits(union_logit, union_target)
    if regions > slots:
        return {"mode": "union_only_overflow", "slot_loss": slot_logits.sum() * 0.0, "union_loss": union_loss}
    with torch.no_grad():
        probability = loss_logits.sigmoid().flatten(1)
        target = region_masks.float().flatten(1)
        dice = 1.0 - (2 * torch.einsum("ip,jp->ij", probability, target) + 1e-6) / (
            probability.sum(1)[:, None] + target.sum(1)[None] + 1e-6
        )
        expanded_logit = loss_logits[:, None].expand(-1, regions, -1, -1)
        expanded_target = region_masks[None].float().expand(slots, -1, -1, -1)
        bce = F.binary_cross_entropy_with_logits(expanded_logit, expanded_target, reduction="none").flatten(2).mean(-1)
        row, col = linear_sum_assignment((dice + bce).cpu().numpy())
    targets = torch.zeros_like(loss_logits)
    for slot_index, region_index in zip(row, col):
        targets[slot_index] = region_masks[region_index]
    slot_bce = F.binary_cross_entropy_with_logits(loss_logits, targets)
    slot_probability = loss_logits.sigmoid()
    slot_dice = 1.0 - (2 * (slot_probability * targets).flatten(1).sum(1) + 1e-6) / (
        slot_probability.flatten(1).sum(1) + targets.flatten(1).sum(1) + 1e-6
    )
    return {"mode": "region_hungarian", "slot_loss": slot_bce + slot_dice.mean(), "union_loss": union_loss}


def slot_collapse_diagnostics(slot_logits: torch.Tensor, queries: torch.Tensor, attention: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pre-registered differentiable quantities used by the formal diagnostic."""
    probability = slot_logits.sigmoid()
    mass = probability.flatten(-2).mean(-1)
    confidence = (probability - 0.5).abs().mul(2).flatten(-2).mean(-1)
    active = (mass >= 0.01) & (confidence >= 0.10)
    query = F.normalize(queries, dim=-1)
    query_cosine = torch.matmul(query, query.transpose(-1, -2))
    flat_mask = probability.flatten(-2)
    intersection = torch.minimum(flat_mask[:, :, None], flat_mask[:, None]).sum(-1)
    union = torch.maximum(flat_mask[:, :, None], flat_mask[:, None]).sum(-1).clamp_min(1e-8)
    mask_soft_iou = intersection / union
    attn = attention.mean(1).clamp_min(1e-8)
    midpoint = 0.5 * (attn[:, :, None] + attn[:, None])
    attention_js = 0.5 * (
        (attn[:, :, None] * (attn[:, :, None].log() - midpoint.log())).sum(-1)
        + (attn[:, None] * (attn[:, None].log() - midpoint.log())).sum(-1)
    )
    union_all, _ = soft_union(slot_logits)
    contribution = []
    for index in range(slot_logits.shape[1]):
        keep = torch.cat((slot_logits[:, :index], slot_logits[:, index + 1:]), dim=1)
        union_without, _ = soft_union(keep)
        contribution.append((union_all - union_without).clamp_min(0).flatten(1).mean(1))
    return {
        "query_pairwise_cosine": query_cosine, "attention_pairwise_js": attention_js,
        "mask_pairwise_soft_iou": mask_soft_iou, "slot_mass": mass,
        "slot_confidence": confidence, "active_slot": active,
        "active_slot_count": active.sum(-1), "union_contribution": torch.stack(contribution, dim=1),
    }


def kd_losses(student: dict, teacher: dict, permutation: torch.Tensor, temperature: float = 2.0) -> dict[str, torch.Tensor]:
    """Frozen multi-level KD definitions after slot correspondence is known."""
    teacher_queries = teacher["queries"][:, permutation].detach()
    student_queries = F.normalize(student["queries"], dim=-1)
    teacher_queries = F.normalize(teacher_queries, dim=-1)
    relation = F.smooth_l1_loss(
        torch.matmul(student_queries, student_queries.transpose(-1, -2)),
        torch.matmul(teacher_queries, teacher_queries.transpose(-1, -2)),
    )
    attention_terms = []
    for student_map, teacher_map in zip(student["forensic_attention"][-2:], teacher["forensic_attention"][-2:]):
        # MultiheadAttention exposes normalized weights rather than its
        # pre-softmax score.  log(p) recovers an equivalent score up to the
        # additive log-normalizer; applying softmax(log(p)/T) therefore
        # implements the specified temperature distribution without applying
        # a second softmax directly to already-small probabilities.
        s = student_map.mean(1).clamp_min(1e-8).log()
        t = teacher_map.mean(1)[:, permutation].detach().clamp_min(1e-8).log()
        attention_terms.append(F.kl_div(
            F.log_softmax(s / temperature, dim=-1), F.softmax(t / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature ** 2 / s.shape[1])
    attention = torch.stack(attention_terms).mean()
    feature = torch.stack([
        F.l1_loss(F.normalize(s, dim=-1), F.normalize(t[:, permutation].detach(), dim=-1))
        for s, t in zip(student["decoder_features"], teacher["decoder_features"])
    ]).mean()
    teacher_probability = (teacher["union_logit"].detach() / temperature).sigmoid()
    logit = F.binary_cross_entropy_with_logits(
        student["union_logit"] / temperature, teacher_probability,
    ) * temperature ** 2
    return {"relation": relation, "attention": attention, "feature": feature, "logit": logit}


def clone_frozen_teacher(student: TFFDGStudent) -> TFFDGStudent:
    teacher = copy.deepcopy(student).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher
