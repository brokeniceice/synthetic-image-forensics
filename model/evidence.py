from __future__ import annotations
from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
FUSION_DTYPE = torch.float32
ECOLAF_LOG_EPS = 1e-10

def evidence_to_dirichlet(evidence: torch.Tensor) -> dict[str, torch.Tensor]:
    """TMC subjective opinion for class evidence ``[B,K,H,W]``.

    Evidence is required to be finite and non-negative. Computation is always
    float32 even when frozen source features are stored in BF16.
    """
    value = evidence.to(FUSION_DTYPE)
    if not torch.isfinite(value).all():
        raise ValueError('evidence must be finite')
    if bool((value < 0).any()):
        raise ValueError('evidence must be non-negative')
    classes = value.shape[1]
    alpha = value + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    belief = value / strength
    uncertainty = torch.full_like(strength, float(classes)) / strength
    masses = torch.cat((belief, uncertainty), dim=1)
    return {'evidence': value, 'alpha': alpha, 'strength': strength, 'belief': belief, 'uncertainty': uncertainty, 'masses': masses, 'posterior': alpha / strength}

def ecolaf_discount(masses: torch.Tensor, *, classes: int, conflict_lambda: float=2.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official ECoLaF adaptive conflict-discount layer.

    ``masses`` is ``[B,K+1,M,H,W]``: K singleton masses, one Ω mass, and M
    experts. The equations and indexing follow official commit
    ``543cdf6c2cfe390f6d3e39ef01fbc66705781f08``.
    """
    value = masses.to(FUSION_DTYPE)
    if value.ndim < 4 or value.shape[1] != classes + 1:
        raise ValueError('expected [B,K+1,M,...] ECoLaF masses')
    experts = value.shape[2]
    if experts < 2:
        raise ValueError('ECoLaF discount requires at least two experts')
    indices = torch.zeros((experts, experts - 1), dtype=torch.long, device=value.device)
    count = 0
    for left in range(experts - 1):
        for right in range(left + 1, experts):
            indices[left, right - 1] = count
            indices[right, left] = count
            count += 1
    pairs = torch.triu_indices(experts, experts, offset=1, device=value.device)
    differences = torch.index_select(value, 2, pairs[0]) - torch.index_select(value, 2, pairs[1])
    distance_term = differences.square().sum(dim=1)
    distance_term = distance_term + 2.0 / classes * differences[:, -1] * differences[:, :-1].sum(dim=1)
    conflict_scale = 1.0 - (2.0 * classes + 1.0) / (classes + 1.0) ** 2
    pairwise_conflict = conflict_scale * torch.sqrt(torch.clamp_min(distance_term / 2.0, 0.0))
    per_expert_conflict = pairwise_conflict[:, indices].sum(dim=2) / float(experts - 1)
    discount = torch.clamp_min(1.0 - per_expert_conflict.unsqueeze(1).pow(conflict_lambda), 0.0)
    discount = discount.pow(1.0 / conflict_lambda)
    singleton = value[:, :-1] * discount
    ignorance = 1.0 - singleton.sum(dim=1, keepdim=True)
    discounted = torch.cat((singleton, ignorance), dim=1).clamp(0.0, 1.0)
    return (discounted, per_expert_conflict, discount.squeeze(1))

def normalized_cell_centers(height: int, width: int, *, device=None) -> torch.Tensor:
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / float(height)
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / float(width)
    (yy, xx) = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy), dim=-1)

def resample_clip_to_original_normalized(tensor24: torch.Tensor, geometry: Mapping[str, object], *, output_hw: tuple[int, int]=(256, 256), vacuous: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a CLIP crop tensor to original-image normalized cell centers.

    Outside the center-crop support, ordinary features/logits are zeroed only
    as a storage placeholder. Evidential masses are explicitly set to vacuous
    ``[0,...,0,1]`` when ``vacuous=True``.
    """
    if tensor24.ndim != 4:
        raise ValueError('tensor24 must be [B,C,H,W]')
    (out_h, out_w) = output_hw
    centers = normalized_cell_centers(out_h, out_w, device=tensor24.device)
    (resized_h, resized_w) = map(float, geometry['resized_hw'])
    (top, left, bottom, right) = map(float, geometry['crop_box_yxyx'])
    y_resized = centers[..., 1] * resized_h
    x_resized = centers[..., 0] * resized_w
    support = (y_resized >= top) & (y_resized < bottom) & (x_resized >= left) & (x_resized < right)
    (crop_h, crop_w) = (bottom - top, right - left)
    grid_x = 2.0 * ((x_resized - left) / crop_w) - 1.0
    grid_y = 2.0 * ((y_resized - top) / crop_h) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(tensor24.shape[0], -1, -1, -1)
    sampled = F.grid_sample(tensor24.to(FUSION_DTYPE), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    support4 = support[None, None].expand(tensor24.shape[0], 1, -1, -1)
    sampled = torch.where(support4, sampled, torch.zeros_like(sampled))
    if vacuous:
        if sampled.shape[1] < 2:
            raise ValueError('vacuous mapping expects singleton+omega masses')
        vacuous_mass = torch.zeros_like(sampled)
        vacuous_mass[:, -1] = 1.0
        sampled = torch.where(support4, sampled, vacuous_mass)
        sampled = sampled / sampled.sum(dim=1, keepdim=True).clamp_min(ECOLAF_LOG_EPS)
    return (sampled, support4)

def sam_lowres_to_original_normalized(lowres: torch.Tensor, geometry: Mapping[str, object], *, output_hw: tuple[int, int]=(256, 256), mode: str='bilinear') -> torch.Tensor:
    """Remove SAM right/bottom padding and express a tensor on normalized cells."""
    if lowres.ndim != 4:
        raise ValueError('SAM low-resolution tensor must be [B,C,H,W]')
    (resized_h, resized_w) = map(int, geometry['resized_hw'])
    (raw_h, raw_w) = lowres.shape[-2:]
    valid_h = max(1, round(raw_h * resized_h / 1024.0))
    valid_w = max(1, round(raw_w * resized_w / 1024.0))
    valid = lowres[..., :valid_h, :valid_w].to(FUSION_DTYPE)
    kwargs = {'size': output_hw, 'mode': mode}
    if mode in ('bilinear', 'bicubic'):
        kwargs['align_corners'] = False
    return F.interpolate(valid, **kwargs)

class LanguageEvidentialHead(nn.Module):
    """Full language head with explicit source-information ablation hooks."""

    def __init__(self, channels: int=64):
        super().__init__()
        self.s64 = nn.Sequential(nn.Conv2d(256, channels, 1), nn.GELU())
        self.q_seg = nn.Sequential(nn.Linear(256, channels), nn.GELU())
        self.z_l = nn.Sequential(nn.Conv2d(1, 16, 1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels + 16, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, 2, 1))

    def forward(self, s64: torch.Tensor, q_seg: torch.Tensor, z_l: torch.Tensor, *, use_s64: bool=True, use_q_seg: bool=True, use_z_l: bool=True) -> torch.Tensor:
        s = self.s64(s64.float())
        q = self.q_seg(q_seg.float())[:, :, None, None].expand(-1, -1, 64, 64)
        z = self.z_l(F.interpolate(z_l.float(), (64, 64), mode='bilinear', align_corners=False))
        if not use_s64:
            s = torch.zeros_like(s)
        if not use_q_seg:
            q = torch.zeros_like(q)
        if not use_z_l:
            z = torch.zeros_like(z)
        return F.softplus(self.fuse(torch.cat((s, q, z), dim=1)))

class ForensicEvidentialHead(nn.Module):
    """Full forensic head with explicit feature/logit ablation hooks."""

    def __init__(self, channels: int=64):
        super().__init__()
        self.f24 = nn.Sequential(nn.Conv2d(256, channels, 1), nn.GELU())
        self.z_f24 = nn.Sequential(nn.Conv2d(1, 16, 1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(channels + 16, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, 2, 1))

    def forward(self, f24: torch.Tensor, z_f24: torch.Tensor, *, use_f24: bool=True, use_z_f24: bool=True) -> torch.Tensor:
        f = self.f24(f24.float())
        z = self.z_f24(z_f24.float())
        if not use_f24:
            f = torch.zeros_like(f)
        if not use_z_f24:
            z = torch.zeros_like(z)
        return F.softplus(self.fuse(torch.cat((f, z), dim=1)))
