"""Frozen Phase 4G-1P CSCU-LF architecture.

This module exposes conditional utility as an explicit intervention node.  It
contains no ground-truth, condition-identity, selector, or evaluation input.
The P1/SAM and forensic source paths are consumed as frozen tensors; the two
G1-C evidential source heads are frozen source-posterior materializers.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pcerf import (
    ECOLAF_LOG_EPS,
    ForensicEvidentialHead,
    LanguageEvidentialHead,
    dsmp_probability,
    ecolaf_dempster,
    ecolaf_discount,
    evidence_to_dirichlet,
    resample_clip_to_original_normalized,
)


WIDTH = 64
WINDOW = 7
HEADS = 4
BLOCKS = 1
RECTIFICATION_LAMBDA_CHANNEL = 0.5
RECTIFICATION_LAMBDA_SPATIAL = 0.5


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if channels >= 8 else 1, channels)


class LanguageContext64(nn.Module):
    """Trainable context projection; frozen source tensors are detached."""

    def __init__(self, width: int = WIDTH):
        super().__init__()
        self.s = nn.Sequential(nn.Conv2d(256, width, 1), _gn(width), nn.GELU())
        self.q = nn.Sequential(nn.Linear(256, width), nn.LayerNorm(width), nn.GELU())
        self.z = nn.Sequential(nn.Conv2d(1, 16, 1), _gn(16), nn.GELU())
        self.merge = nn.Sequential(nn.Conv2d(2 * width + 16, width, 3, padding=1), _gn(width), nn.GELU())

    def forward(self, s64: torch.Tensor, q_seg: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        s = self.s(s64.detach().float())
        q = self.q(q_seg.detach().float())[:, :, None, None].expand(-1, -1, 64, 64)
        z = self.z(F.interpolate(z_l.detach().float(), (64, 64), mode="bilinear", align_corners=False))
        return self.merge(torch.cat((s, q, z), dim=1))


class ForensicContext64(nn.Module):
    def __init__(self, width: int = WIDTH):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(256, width, 1), _gn(width), nn.GELU())
        self.z = nn.Sequential(nn.Conv2d(1, 16, 1), _gn(16), nn.GELU())
        self.merge = nn.Sequential(nn.Conv2d(width + 16 + 1, width, 3, padding=1), _gn(width), nn.GELU())

    def forward(self, f64: torch.Tensor, z_f64: torch.Tensor, support64: torch.Tensor) -> torch.Tensor:
        support = support64.float()
        value = self.merge(torch.cat((self.f(f64.detach().float()), self.z(z_f64.detach().float()), support), dim=1))
        return value * support


class CMXJointRectification(nn.Module):
    """Minimum-sufficient faithful channel+spatial bidirectional CMX lineage."""

    def __init__(self, width: int = WIDTH):
        super().__init__()
        hidden = max(width // 4, 8)
        self.channel = nn.Sequential(nn.Conv2d(4 * width, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 2 * width, 1))
        self.spatial = nn.Conv2d(4, 2, 7, padding=3)

    def forward(self, language: torch.Tensor, forensic: torch.Tensor, support: torch.Tensor):
        pooled = torch.cat(
            (language.mean((2, 3), keepdim=True), language.amax((2, 3), keepdim=True),
             forensic.mean((2, 3), keepdim=True), forensic.amax((2, 3), keepdim=True)), dim=1,
        )
        channel_l, channel_f = self.channel(pooled).sigmoid().chunk(2, dim=1)
        spatial_input = torch.cat(
            (language.mean(1, keepdim=True), language.amax(1, keepdim=True),
             forensic.mean(1, keepdim=True), forensic.amax(1, keepdim=True)), dim=1,
        )
        spatial_l, spatial_f = self.spatial(spatial_input).sigmoid().chunk(2, dim=1)
        language_rectified = language + RECTIFICATION_LAMBDA_CHANNEL * channel_l * forensic + RECTIFICATION_LAMBDA_SPATIAL * spatial_l * forensic
        forensic_rectified = forensic + RECTIFICATION_LAMBDA_CHANNEL * channel_f * language + RECTIFICATION_LAMBDA_SPATIAL * spatial_f * language
        forensic_rectified = forensic_rectified * support.float()
        return language_rectified, forensic_rectified, {
            "channel_L": channel_l, "channel_F": channel_f,
            "spatial_L": spatial_l, "spatial_F": spatial_f,
        }


class LocalCrossExchange(nn.Module):
    """One bidirectional 7x7 local cross-attention block at 64x64."""

    def __init__(self, width: int = WIDTH, heads: int = HEADS, window: int = WINDOW):
        super().__init__()
        if width % heads or window % 2 != 1:
            raise ValueError("width must divide heads and local window must be odd")
        self.width, self.heads, self.window, self.head_dim = width, heads, window, width // heads
        self.q_l, self.k_l, self.v_l = (nn.Conv2d(width, width, 1) for _ in range(3))
        self.q_f, self.k_f, self.v_f = (nn.Conv2d(width, width, 1) for _ in range(3))
        self.out_l, self.out_f = nn.Conv2d(width, width, 1), nn.Conv2d(width, width, 1)
        self.norm_l, self.norm_f = _gn(width), _gn(width)

    def _attend(self, query, key, value, key_support=None):
        batch, _, height, width = query.shape; cells = height * width; neighbors = self.window ** 2
        q = query.view(batch, self.heads, self.head_dim, cells).permute(0, 1, 3, 2)
        k = F.unfold(key, self.window, padding=self.window // 2).view(batch, self.heads, self.head_dim, neighbors, cells).permute(0, 1, 4, 3, 2)
        v = F.unfold(value, self.window, padding=self.window // 2).view(batch, self.heads, self.head_dim, neighbors, cells).permute(0, 1, 4, 3, 2)
        score = (q.unsqueeze(-2) * k).sum(-1) / self.head_dim ** 0.5
        if key_support is not None:
            mask = F.unfold(key_support.float(), self.window, padding=self.window // 2).view(batch, 1, cells, neighbors)
            score = score.masked_fill(mask == 0, -1e4)
        attention = score.softmax(-1)
        if key_support is not None:
            attention = attention * mask
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        output = (attention.unsqueeze(-1) * v).sum(-2).permute(0, 1, 3, 2).reshape(batch, self.width, height, width)
        return output, attention

    def forward(self, language: torch.Tensor, forensic: torch.Tensor, support: torch.Tensor):
        from_f, attention_l = self._attend(self.q_l(language), self.k_f(forensic), self.v_f(forensic), support)
        from_l, attention_f = self._attend(self.q_f(forensic), self.k_l(language), self.v_l(language))
        language_out = self.norm_l(language + self.out_l(from_f))
        forensic_out = self.norm_f(forensic + self.out_f(from_l)) * support.float()
        return language_out, forensic_out, {"attention_L_from_F": attention_l, "attention_F_from_L": attention_f}


def build_comparison_features(
    language: torch.Tensor,
    forensic: torch.Tensor,
    probability_l: torch.Tensor,
    probability_f: torch.Tensor,
    conflict: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    product = language * forensic
    absolute_difference = (language - forensic).abs()
    cosine = F.cosine_similarity(language, forensic, dim=1, eps=1e-8)[:, None]
    parts = {
        "Lr": language,
        "Fr": forensic,
        "product": product,
        "absolute_difference": absolute_difference,
        "local_cosine": cosine,
        "p_L": probability_l,
        "p_F": probability_f,
        "conflict": conflict,
    }
    comparison = torch.cat(tuple(parts.values()), dim=1) * support.float()
    return comparison, parts


class ConditionalUtilityHead(nn.Module):
    def __init__(self, width: int = WIDTH):
        super().__init__()
        comparison_channels = 4 * width + 1 + 2 + 2 + 2
        self.net = nn.Sequential(nn.Conv2d(comparison_channels, width, 3, padding=1), _gn(width), nn.GELU(), nn.Conv2d(width, 1, 1))

    def forward(self, comparison: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prelogit = self.net[:-1](comparison)
        utility = self.net[-1](prelogit).sigmoid() * support.float()
        return utility, prelogit


def adapted_ecolaf_fuse(mass_l: torch.Tensor, mass_f: torch.Tensor, utility_f: torch.Tensor) -> dict[str, torch.Tensor]:
    """ECoLaF-style ADAPTED EXTENSION with strict U=0 and U=1 semantics.

    d_F^adapt = d_F^official * U.  The companion language discount is
    restored continuously as d_L^adapt = 1-U*(1-d_L^official), ensuring that
    U=0 removes every forensic-induced effect, not only forensic singleton
    mass.  U=1 selects the official discounted/fused tensors exactly.
    """
    if mass_l.shape != mass_f.shape or utility_f.shape != mass_l[:, :1].shape:
        raise ValueError("adapted fusion expects aligned [B,3,H,W] masses and [B,1,H,W] utility")
    if bool((utility_f < 0).any()) or bool((utility_f > 1).any()):
        raise ValueError("conditional utility must be in [0,1]")
    stacked = torch.stack((mass_l.float(), mass_f.float()), dim=2)
    official_discounted, conflict, official_discount = ecolaf_discount(stacked, classes=2)
    u = utility_f[:, 0]
    discount_l = 1.0 - u * (1.0 - official_discount[:, 0])
    discount_f = official_discount[:, 1] * u
    discounts = torch.stack((discount_l, discount_f), dim=1)
    singleton = stacked[:, :-1] * discounts[:, None]
    ignorance = 1.0 - singleton.sum(dim=1, keepdim=True)
    adapted_discounted = torch.cat((singleton, ignorance), dim=1).clamp(0.0, 1.0)
    one = utility_f[:, None] == 1.0
    zero = utility_f[:, None] == 0.0
    adapted_discounted = torch.where(one, official_discounted, adapted_discounted)
    vacuous_f = torch.zeros_like(mass_f); vacuous_f[:, -1] = 1.0
    language_only_pair = torch.stack((mass_l, vacuous_f), dim=2)
    adapted_discounted = torch.where(zero, language_only_pair, adapted_discounted)
    combined = ecolaf_dempster(adapted_discounted)
    official_combined = ecolaf_dempster(official_discounted)
    combined = torch.where(utility_f == 1.0, official_combined, combined)
    combined = torch.where(utility_f == 0.0, mass_l.float(), combined)
    probability = dsmp_probability(combined)
    official_probability = dsmp_probability(official_combined)
    probability = torch.where(utility_f == 1.0, official_probability, probability)
    probability = torch.where(utility_f == 0.0, dsmp_probability(mass_l.float()), probability)
    effective_forensic_contribution = discount_f[:, None] * (1.0 - mass_f[:, -1:])
    return {
        "official_discounted_masses": official_discounted,
        "official_discount": official_discount,
        "adapted_discounted_masses": adapted_discounted,
        "adapted_discount": discounts,
        "conflict": conflict,
        "fused_mass": combined,
        "probability": probability,
        "effective_forensic_contribution": effective_forensic_contribution,
    }


class CSCULF(nn.Module):
    """Frozen CSCU-LF A3 graph; REFINEMENT_INCLUDED=NO."""

    def __init__(self, temperature_l: float = 1.0, temperature_f: float = 1.0):
        super().__init__()
        self.language_source = LanguageEvidentialHead()
        self.forensic_source = ForensicEvidentialHead()
        self.language_source.requires_grad_(False)
        self.forensic_source.requires_grad_(False)
        self.register_buffer("temperature_l", torch.tensor(float(temperature_l)))
        self.register_buffer("temperature_f", torch.tensor(float(temperature_f)))
        self.language_context = LanguageContext64()
        self.forensic_context = ForensicContext64()
        self.rectification = CMXJointRectification()
        self.exchange = LocalCrossExchange()
        self.utility_head = ConditionalUtilityHead()

    def load_frozen_source_heads(self, language_state: dict, forensic_state: dict) -> None:
        self.language_source.load_state_dict(language_state)
        self.forensic_source.load_state_dict(forensic_state)
        self.language_source.eval().requires_grad_(False)
        self.forensic_source.eval().requires_grad_(False)

    def _map_forensic(self, tensor: torch.Tensor, geometries: list[Mapping[str, object]], *, output_hw=(64, 64), vacuous=False):
        values, supports = [], []
        for index, geometry in enumerate(geometries):
            value, support = resample_clip_to_original_normalized(tensor[index:index + 1], geometry, output_hw=output_hw, vacuous=vacuous)
            values.append(value); supports.append(support)
        return torch.cat(values), torch.cat(supports)

    def forward(
        self, *, s64, q_seg, z_l, f24, z_f24, clip_geometries,
        valid_g0, forensic_present, forensic_vacuous, forensic_off,
        utility_override: torch.Tensor | None = None,
    ):
        batch = z_l.shape[0]
        for flag in (valid_g0, forensic_present, forensic_vacuous, forensic_off):
            if flag.shape != (batch,): raise ValueError("all route flags must be [B]")
        if z_l.shape[-2:] != (256, 256): raise ValueError("z_L must use canonical 256 grid")
        with torch.set_grad_enabled(self.training and torch.is_grad_enabled()):
            evidence_l = self.language_source(s64.detach(), q_seg.detach(), z_l.detach()) / self.temperature_l
            evidence_f = self.forensic_source(f24.detach(), z_f24.detach()) / self.temperature_f
        opinion_l = evidence_to_dirichlet(evidence_l)
        opinion_f = evidence_to_dirichlet(evidence_f)
        mass_l64 = opinion_l["masses"]
        mass_f64, support64 = self._map_forensic(opinion_f["masses"], clip_geometries, output_hw=(64, 64), vacuous=True)
        probability_l64 = opinion_l["posterior"]
        probability_f64 = mass_f64[:, :-1] + mass_f64[:, -1:] / 2.0
        f64, mapped_support = self._map_forensic(f24.detach(), clip_geometries, output_hw=(64, 64), vacuous=False)
        z_f64, _ = self._map_forensic(z_f24.detach(), clip_geometries, output_hw=(64, 64), vacuous=False)
        if not torch.equal(support64, mapped_support): raise RuntimeError("forensic geometry/support drift")
        l64 = self.language_context(s64, q_seg, z_l)
        fctx64 = self.forensic_context(f64, z_f64, support64)
        lr, fr, rectification = self.rectification(l64, fctx64, support64)
        lr, fr, exchange = self.exchange(lr, fr, support64)
        source_conflict = ecolaf_discount(torch.stack((mass_l64, mass_f64), dim=2), classes=2)[1]
        comparison, comparison_parts = build_comparison_features(lr, fr, probability_l64, probability_f64, source_conflict, support64)
        utility64, utility_prehead = self.utility_head(comparison, support64)
        if utility_override is not None:
            if utility_override.shape != utility64.shape: raise ValueError("utility override must match U_F64")
            utility64 = utility_override.to(utility64).clamp(0.0, 1.0) * support64.float()
        utility256 = F.interpolate(utility64, (256, 256), mode="bilinear", align_corners=False)
        mass_l256 = F.interpolate(mass_l64, (256, 256), mode="bilinear", align_corners=False)
        mass_l256 = mass_l256 / mass_l256.sum(1, keepdim=True).clamp_min(ECOLAF_LOG_EPS)
        mass_f256, support256 = self._map_forensic(opinion_f["masses"], clip_geometries, output_hw=(256, 256), vacuous=True)
        utility256 = utility256 * support256.float()
        fused = adapted_ecolaf_fuse(mass_l256, mass_f256, utility256)
        fused_logits = torch.logit(fused["probability"][:, 1:2].clamp(1e-6, 1.0 - 1e-6)).to(z_l.dtype)
        source_active = valid_g0.bool() & forensic_present.bool() & ~forensic_vacuous.bool() & ~forensic_off.bool() & support256.flatten(1).any(1)
        active_pixels = source_active[:, None, None, None] & support256
        logits = torch.where(active_pixels, fused_logits, z_l)
        formal_invalid = ~valid_g0.bool()
        logits = torch.where(formal_invalid[:, None, None, None], torch.zeros_like(logits), logits)
        return {
            "logits": logits, "formal_valid": ~formal_invalid, "L64": l64, "Fctx64": fctx64,
            "Lr": lr, "Fr": fr, "rectification": rectification, "exchange": exchange,
            "comparison": comparison, "comparison_parts": comparison_parts, "utility_prehead": utility_prehead,
            "U_F64": utility64, "U_F256": utility256, "p_L64": probability_l64, "p_F64": probability_f64,
            "mass_L64": mass_l64, "mass_F64": mass_f64, "mass_L256": mass_l256, "mass_F256": mass_f256,
            "aligned_F64": f64, "aligned_z_F64": z_f64, "support64": support64, "support256": support256,
            "fused": fused, "active_pixels": active_pixels,
        }
