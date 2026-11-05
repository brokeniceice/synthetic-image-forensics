from __future__ import annotations
from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.evidence import ForensicEvidentialHead, LanguageEvidentialHead, resample_clip_to_original_normalized
WIDTH = 64
WINDOW = 7
HEADS = 4
RECTIFICATION_LAMBDA_CHANNEL = 0.5
RECTIFICATION_LAMBDA_SPATIAL = 0.5

def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if channels >= 8 else 1, channels)

class LanguageContext64(nn.Module):
    """Trainable context projection; frozen source tensors are detached."""

    def __init__(self, width: int=WIDTH):
        super().__init__()
        self.s = nn.Sequential(nn.Conv2d(256, width, 1), _gn(width), nn.GELU())
        self.q = nn.Sequential(nn.Linear(256, width), nn.LayerNorm(width), nn.GELU())
        self.z = nn.Sequential(nn.Conv2d(1, 16, 1), _gn(16), nn.GELU())
        self.merge = nn.Sequential(nn.Conv2d(2 * width + 16, width, 3, padding=1), _gn(width), nn.GELU())

    def forward(self, s64: torch.Tensor, q_seg: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        s = self.s(s64.detach().float())
        q = self.q(q_seg.detach().float())[:, :, None, None].expand(-1, -1, 64, 64)
        z = self.z(F.interpolate(z_l.detach().float(), (64, 64), mode='bilinear', align_corners=False))
        return self.merge(torch.cat((s, q, z), dim=1))

class ForensicContext64(nn.Module):

    def __init__(self, width: int=WIDTH):
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

    def __init__(self, width: int=WIDTH):
        super().__init__()
        hidden = max(width // 4, 8)
        self.channel = nn.Sequential(nn.Conv2d(4 * width, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 2 * width, 1))
        self.spatial = nn.Conv2d(4, 2, 7, padding=3)

    def forward(self, language: torch.Tensor, forensic: torch.Tensor, support: torch.Tensor):
        pooled = torch.cat((language.mean((2, 3), keepdim=True), language.amax((2, 3), keepdim=True), forensic.mean((2, 3), keepdim=True), forensic.amax((2, 3), keepdim=True)), dim=1)
        (channel_l, channel_f) = self.channel(pooled).sigmoid().chunk(2, dim=1)
        spatial_input = torch.cat((language.mean(1, keepdim=True), language.amax(1, keepdim=True), forensic.mean(1, keepdim=True), forensic.amax(1, keepdim=True)), dim=1)
        (spatial_l, spatial_f) = self.spatial(spatial_input).sigmoid().chunk(2, dim=1)
        language_rectified = language + RECTIFICATION_LAMBDA_CHANNEL * channel_l * forensic + RECTIFICATION_LAMBDA_SPATIAL * spatial_l * forensic
        forensic_rectified = forensic + RECTIFICATION_LAMBDA_CHANNEL * channel_f * language + RECTIFICATION_LAMBDA_SPATIAL * spatial_f * language
        forensic_rectified = forensic_rectified * support.float()
        return (language_rectified, forensic_rectified, {'channel_L': channel_l, 'channel_F': channel_f, 'spatial_L': spatial_l, 'spatial_F': spatial_f})

class LocalCrossExchange(nn.Module):
    """One bidirectional 7x7 local cross-attention block at 64x64."""

    def __init__(self, width: int=WIDTH, heads: int=HEADS, window: int=WINDOW):
        super().__init__()
        if width % heads or window % 2 != 1:
            raise ValueError('width must divide heads and local window must be odd')
        (self.width, self.heads, self.window, self.head_dim) = (width, heads, window, width // heads)
        (self.q_l, self.k_l, self.v_l) = (nn.Conv2d(width, width, 1) for _ in range(3))
        (self.q_f, self.k_f, self.v_f) = (nn.Conv2d(width, width, 1) for _ in range(3))
        (self.out_l, self.out_f) = (nn.Conv2d(width, width, 1), nn.Conv2d(width, width, 1))
        (self.norm_l, self.norm_f) = (_gn(width), _gn(width))

    def _attend(self, query, key, value, key_support=None):
        (batch, _, height, width) = query.shape
        cells = height * width
        neighbors = self.window ** 2
        q = query.view(batch, self.heads, self.head_dim, cells).permute(0, 1, 3, 2)
        k = F.unfold(key, self.window, padding=self.window // 2).view(batch, self.heads, self.head_dim, neighbors, cells).permute(0, 1, 4, 3, 2)
        v = F.unfold(value, self.window, padding=self.window // 2).view(batch, self.heads, self.head_dim, neighbors, cells).permute(0, 1, 4, 3, 2)
        score = (q.unsqueeze(-2) * k).sum(-1) / self.head_dim ** 0.5
        if key_support is not None:
            mask = F.unfold(key_support.float(), self.window, padding=self.window // 2).view(batch, 1, cells, neighbors)
            score = score.masked_fill(mask == 0, -10000.0)
        attention = score.softmax(-1)
        if key_support is not None:
            attention = attention * mask
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        output = (attention.unsqueeze(-1) * v).sum(-2).permute(0, 1, 3, 2).reshape(batch, self.width, height, width)
        return (output, attention)

    def forward(self, language: torch.Tensor, forensic: torch.Tensor, support: torch.Tensor):
        (from_f, attention_l) = self._attend(self.q_l(language), self.k_f(forensic), self.v_f(forensic), support)
        (from_l, attention_f) = self._attend(self.q_f(forensic), self.k_l(language), self.v_l(language))
        language_out = self.norm_l(language + self.out_l(from_f))
        forensic_out = self.norm_f(forensic + self.out_f(from_l)) * support.float()
        return (language_out, forensic_out, {'attention_L_from_F': attention_l, 'attention_F_from_L': attention_f})

def build_comparison_features(language: torch.Tensor, forensic: torch.Tensor, probability_l: torch.Tensor, probability_f: torch.Tensor, conflict: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    product = language * forensic
    absolute_difference = (language - forensic).abs()
    cosine = F.cosine_similarity(language, forensic, dim=1, eps=1e-08)[:, None]
    parts = {'Lr': language, 'Fr': forensic, 'product': product, 'absolute_difference': absolute_difference, 'local_cosine': cosine, 'p_L': probability_l, 'p_F': probability_f, 'conflict': conflict}
    comparison = torch.cat(tuple(parts.values()), dim=1) * support.float()
    return (comparison, parts)

class ConditionalUtilityHead(nn.Module):

    def __init__(self, width: int=WIDTH):
        super().__init__()
        comparison_channels = 4 * width + 1 + 2 + 2 + 2
        self.net = nn.Sequential(nn.Conv2d(comparison_channels, width, 3, padding=1), _gn(width), nn.GELU(), nn.Conv2d(width, 1, 1))

    def forward(self, comparison: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prelogit = self.net[:-1](comparison)
        utility = self.net[-1](prelogit).sigmoid() * support.float()
        return (utility, prelogit)

class CSCULF(nn.Module):
    """Frozen CSCU-LF A3 graph; REFINEMENT_INCLUDED=NO."""

    def __init__(self, temperature_l: float=1.0, temperature_f: float=1.0):
        super().__init__()
        self.language_source = LanguageEvidentialHead()
        self.forensic_source = ForensicEvidentialHead()
        self.language_source.requires_grad_(False)
        self.forensic_source.requires_grad_(False)
        self.register_buffer('temperature_l', torch.tensor(float(temperature_l)))
        self.register_buffer('temperature_f', torch.tensor(float(temperature_f)))
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
        (values, supports) = ([], [])
        for (index, geometry) in enumerate(geometries):
            (value, support) = resample_clip_to_original_normalized(tensor[index:index + 1], geometry, output_hw=output_hw, vacuous=vacuous)
            values.append(value)
            supports.append(support)
        return (torch.cat(values), torch.cat(supports))

    def forward(self, batch):
        from forensics.spatial import _utility_forward
        return _utility_forward(self, batch)
