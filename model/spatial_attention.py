from __future__ import annotations
import math
import torch
import torch.nn as nn

def coordinate_sincos(coordinates: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Deterministic sine/cosine encoding for normalized ``(x,y)`` coordinates."""
    if embed_dim % 4:
        raise ValueError('coordinate encoding requires embed_dim divisible by four')
    quarter = embed_dim // 4
    frequency = torch.exp(-math.log(10000.0) * torch.arange(quarter, device=coordinates.device, dtype=coordinates.dtype) / max(1, quarter - 1))
    values = []
    for axis in (0, 1):
        phase = coordinates[..., axis, None] * frequency * (2.0 * math.pi)
        values.extend((phase.sin(), phase.cos()))
    return torch.cat(values, dim=-1)

class GeometryAwareCrossAttention(nn.Module):
    """Cross-attention with fixed original-coordinate bias and validity mask."""

    def __init__(self, dim: int=256, heads: int=8, locality_sigma: float=0.25) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        (self.dim, self.heads, self.head_dim) = (dim, heads, dim // heads)
        self.locality_sigma = locality_sigma
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, query_coordinates: torch.Tensor, key_coordinates: torch.Tensor, key_valid: torch.Tensor | None=None, value: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        (b, nq, _) = query.shape
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
        return (self.out_proj(value), attention)

class CrossAttentiveSemanticRectification(nn.Module):
    """``S_rect = S + gamma * Projection(CrossAttention(LN(S), LN(F), F))``."""

    def __init__(self, dim: int=256, heads: int=8, gamma_init: float=0.01) -> None:
        super().__init__()
        if gamma_init == 0:
            raise ValueError('gamma must be nonzero at initialization')
        self.semantic_norm = nn.LayerNorm(dim)
        self.forensic_norm = nn.LayerNorm(dim)
        self.cross_attention = GeometryAwareCrossAttention(dim, heads)
        self.projection = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, semantic: torch.Tensor, forensic: torch.Tensor, semantic_coordinates: torch.Tensor, forensic_coordinates: torch.Tensor, forensic_valid: torch.Tensor | None=None, semantic_support: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic_pe = coordinate_sincos(semantic_coordinates, semantic.shape[-1])
        forensic_pe = coordinate_sincos(forensic_coordinates, forensic.shape[-1])
        (response, attention) = self.cross_attention(self.semantic_norm(semantic) + semantic_pe, self.forensic_norm(forensic) + forensic_pe, semantic_coordinates, forensic_coordinates, forensic_valid, value=forensic)
        residual = self.gamma * self.projection(response)
        if semantic_support is not None:
            residual = residual * semantic_support[..., None]
        return (semantic + residual, attention, residual)
