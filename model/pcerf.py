"""Phase 4G-0.5 PCERF architecture-hardening primitives.

This module deliberately separates three contracts:

* TMC-style non-negative evidence -> Dirichlet opinion conversion;
* the official ECoLaF conflict discount, scalable Dempster fusion, and DSmP
  probability transform;
* project-level source availability dispatch, for which an absent/vacuous
  forensic source is the neutral element and therefore returns P1 bit-exactly.

It is an implementation/preflight surface, not authorization for Phase 4G-1
training or validation-based architecture selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


FUSION_DTYPE = torch.float32
ECOLAF_LOG_EPS = 1e-10
DSMP_EPS = 1e-4
PROBABILITY_EPS = 1e-6


def multiseg_flatten(
    grouped: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """Flatten variable-K SEG states and return image indices plus offsets."""
    if not grouped:
        raise ValueError("multiseg grouping must contain at least one image")
    if any(value.ndim < 1 for value in grouped):
        raise ValueError("each grouped tensor must have a leading K dimension")
    trailing = grouped[0].shape[1:]
    if any(value.shape[1:] != trailing for value in grouped):
        raise ValueError("grouped SEG tensors have inconsistent trailing shapes")
    device = grouped[0].device
    counts = torch.tensor([value.shape[0] for value in grouped], dtype=torch.long, device=device)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    image_index = torch.repeat_interleave(torch.arange(len(grouped), device=device), counts)
    if int(offsets[-1]) == 0:
        flat = grouped[0].new_empty((0, *trailing))
    else:
        flat = torch.cat(grouped, dim=0)
    return flat, image_index, offsets


def multiseg_regroup(flat: torch.Tensor, offsets: torch.Tensor) -> list[torch.Tensor]:
    """Inverse of :func:`multiseg_flatten`, preserving image and phrase order."""
    offsets = torch.as_tensor(offsets, dtype=torch.long, device=flat.device).reshape(-1)
    if offsets.numel() < 2 or int(offsets[0]) != 0 or int(offsets[-1]) != flat.shape[0]:
        raise ValueError("invalid multiseg offsets")
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError("multiseg offsets must be monotonic")
    return [flat[int(start):int(end)] for start, end in zip(offsets[:-1], offsets[1:])]


def multiseg_repeat_images(value: torch.Tensor, image_index: torch.Tensor) -> torch.Tensor:
    """Repeat each image-side tensor once per SEG without creating parameters."""
    if value.ndim < 1:
        raise ValueError("image-side value must have a batch dimension")
    index = torch.as_tensor(image_index, dtype=torch.long, device=value.device)
    if index.numel() and (int(index.min()) < 0 or int(index.max()) >= value.shape[0]):
        raise IndexError("multiseg image index out of range")
    return value.index_select(0, index)


def multiseg_r1_batch(
    grouped_q_seg: list[torch.Tensor],
    grouped_z_l: list[torch.Tensor],
    image_batch: Mapping[str, object],
) -> tuple[dict[str, object], torch.LongTensor]:
    """Build the existing R1 utility input at ``T=sum(K)`` with shared weights.

    Phrase-specific tensors (q_seg and preliminary z_L) are flattened in
    order. Image-side tensors are repeated by image index; no module or
    parameter is copied.
    """
    q_flat, image_index, offsets = multiseg_flatten(grouped_q_seg)
    z_flat, z_image_index, z_offsets = multiseg_flatten(grouped_z_l)
    if not torch.equal(image_index, z_image_index) or not torch.equal(offsets, z_offsets):
        raise ValueError("q_seg/z_L multiseg grouping mismatch")
    required = ("S64", "F24", "z_F24", "clip_geometries")
    if any(key not in image_batch for key in required):
        raise KeyError(f"R1 image batch requires {required}")
    result: dict[str, object] = {"q_seg": q_flat, "z_L": z_flat}
    for key in ("S64", "F24", "z_F24"):
        value = image_batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a tensor")
        if value.shape[0] != len(grouped_q_seg):
            raise ValueError(f"{key} image batch mismatch")
        result[key] = multiseg_repeat_images(value, image_index)
    geometries = list(image_batch["clip_geometries"])
    if len(geometries) != len(grouped_q_seg):
        raise ValueError("clip geometry image batch mismatch")
    result["clip_geometries"] = [geometries[int(index)] for index in image_index.tolist()]
    return result, offsets


def tmc_expected_ce_kl(
    evidence: torch.Tensor,
    target: torch.Tensor,
    *,
    kl_coefficient: float,
) -> dict[str, torch.Tensor]:
    """TMC expected cross-entropy plus annealed incorrect-evidence KL.

    ``target`` is an integer class map matching ``evidence`` spatially. The
    reduction is the unweighted mean over all pixels, exactly as frozen for
    G1-C source-head fitting.
    """
    if target.shape != (evidence.shape[0], *evidence.shape[2:]):
        raise ValueError("target must be [B,...] and spatially match evidence")
    value = evidence.float()
    classes = value.shape[1]
    alpha = value + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    one_hot = F.one_hot(target.long(), num_classes=classes).movedim(-1, 1).float()
    expected_ce_map = (one_hot * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=1)
    alpha_tilde = value * (1.0 - one_hot) + 1.0
    beta = torch.ones_like(alpha_tilde)
    sum_alpha = alpha_tilde.sum(dim=1, keepdim=True)
    sum_beta = beta.sum(dim=1, keepdim=True)
    log_b_alpha = torch.lgamma(sum_alpha) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
    log_b_beta_inverse = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(sum_beta)
    digamma_sum = torch.digamma(sum_alpha)
    kl_map = (
        ((alpha_tilde - beta) * (torch.digamma(alpha_tilde) - digamma_sum)).sum(dim=1, keepdim=True)
        + log_b_alpha
        + log_b_beta_inverse
    )[:, 0]
    expected_ce = expected_ce_map.mean()
    kl = kl_map.mean()
    total = expected_ce + float(kl_coefficient) * kl
    return {"total": total, "expected_ce": expected_ce, "kl": kl}


def evidence_to_dirichlet(evidence: torch.Tensor) -> dict[str, torch.Tensor]:
    """TMC subjective opinion for class evidence ``[B,K,H,W]``.

    Evidence is required to be finite and non-negative. Computation is always
    float32 even when frozen source features are stored in BF16.
    """
    value = evidence.to(FUSION_DTYPE)
    if not torch.isfinite(value).all():
        raise ValueError("evidence must be finite")
    if bool((value < 0).any()):
        raise ValueError("evidence must be non-negative")
    classes = value.shape[1]
    alpha = value + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    belief = value / strength
    uncertainty = torch.full_like(strength, float(classes)) / strength
    masses = torch.cat((belief, uncertainty), dim=1)
    return {
        "evidence": value,
        "alpha": alpha,
        "strength": strength,
        "belief": belief,
        "uncertainty": uncertainty,
        "masses": masses,
        "posterior": alpha / strength,
    }


def tmc_dempster_two(alpha_left: torch.Tensor, alpha_right: torch.Tensor) -> torch.Tensor:
    """Literal two-view TMC Dempster combination, spatially vectorized."""
    if alpha_left.shape != alpha_right.shape:
        raise ValueError("TMC alpha tensors must have identical shapes")
    classes = alpha_left.shape[1]
    beliefs, uncertainties = [], []
    for alpha in (alpha_left.to(FUSION_DTYPE), alpha_right.to(FUSION_DTYPE)):
        strength = alpha.sum(dim=1, keepdim=True)
        beliefs.append((alpha - 1.0) / strength)
        uncertainties.append(torch.full_like(strength, float(classes)) / strength)
    pair = beliefs[0].unsqueeze(2) * beliefs[1].unsqueeze(1)
    conflict = pair.sum(dim=(1, 2)) - torch.diagonal(pair, dim1=1, dim2=2).sum(dim=-1)
    denominator = 1.0 - conflict.unsqueeze(1)
    belief = (
        beliefs[0] * beliefs[1]
        + beliefs[0] * uncertainties[1]
        + beliefs[1] * uncertainties[0]
    ) / denominator
    uncertainty = uncertainties[0] * uncertainties[1] / denominator
    strength = float(classes) / uncertainty
    return belief * strength + 1.0


def ecolaf_discount(
    masses: torch.Tensor,
    *,
    classes: int,
    conflict_lambda: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official ECoLaF adaptive conflict-discount layer.

    ``masses`` is ``[B,K+1,M,H,W]``: K singleton masses, one Ω mass, and M
    experts. The equations and indexing follow official commit
    ``543cdf6c2cfe390f6d3e39ef01fbc66705781f08``.
    """
    value = masses.to(FUSION_DTYPE)
    if value.ndim < 4 or value.shape[1] != classes + 1:
        raise ValueError("expected [B,K+1,M,...] ECoLaF masses")
    experts = value.shape[2]
    if experts < 2:
        raise ValueError("ECoLaF discount requires at least two experts")
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
    distance_term = distance_term + (2.0 / classes) * differences[:, -1] * differences[:, :-1].sum(dim=1)
    conflict_scale = 1.0 - (2.0 * classes + 1.0) / (classes + 1.0) ** 2
    pairwise_conflict = conflict_scale * torch.sqrt(torch.clamp_min(distance_term / 2.0, 0.0))
    per_expert_conflict = pairwise_conflict[:, indices].sum(dim=2) / float(experts - 1)
    discount = torch.clamp_min(1.0 - per_expert_conflict.unsqueeze(1).pow(conflict_lambda), 0.0)
    discount = discount.pow(1.0 / conflict_lambda)
    singleton = value[:, :-1] * discount
    ignorance = 1.0 - singleton.sum(dim=1, keepdim=True)
    discounted = torch.cat((singleton, ignorance), dim=1).clamp(0.0, 1.0)
    return discounted, per_expert_conflict, discount.squeeze(1)


def ecolaf_dempster(masses: torch.Tensor) -> torch.Tensor:
    """Official ECoLaF scalable Dempster fusion in log space."""
    value = masses.to(FUSION_DTYPE)
    singleton_plus_omega = torch.cat(
        (value[:, :-1] + value[:, None, -1], value[:, None, -1]), dim=1
    )
    log_product = torch.log(torch.relu(singleton_plus_omega) + ECOLAF_LOG_EPS).sum(dim=2)
    log_product = log_product - log_product.max(dim=1, keepdim=True).values
    product = torch.exp(log_product)
    product_omega = product[:, -1, None]
    singleton = product[:, :-1] - product_omega
    combined = torch.cat((singleton, product_omega), dim=1)
    return F.normalize(combined, p=1.0, dim=1)


def dsmp_probability(masses: torch.Tensor, *, eps: float = DSMP_EPS) -> torch.Tensor:
    """Official ECoLaF DSmP transform from K+1 masses to K probabilities."""
    value = masses.to(FUSION_DTYPE)
    omega = value[:, None, -1]
    return (value + (value * omega + eps * omega) / (1.0 - omega + eps * (value.shape[1] - 1)))[:, :-1]


def ecolaf_fuse(masses: torch.Tensor, *, classes: int = 2) -> dict[str, torch.Tensor]:
    discounted, conflict, discount = ecolaf_discount(masses, classes=classes)
    fused_mass = ecolaf_dempster(discounted)
    probability = dsmp_probability(fused_mass)
    return {
        "discounted_masses": discounted,
        "conflict": conflict,
        "discount": discount,
        "fused_mass": fused_mass,
        "probability": probability,
    }


def normalized_cell_centers(height: int, width: int, *, device=None) -> torch.Tensor:
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / float(height)
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / float(width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def resample_clip_to_original_normalized(
    tensor24: torch.Tensor,
    geometry: Mapping[str, object],
    *,
    output_hw: tuple[int, int] = (256, 256),
    vacuous: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a CLIP crop tensor to original-image normalized cell centers.

    Outside the center-crop support, ordinary features/logits are zeroed only
    as a storage placeholder. Evidential masses are explicitly set to vacuous
    ``[0,...,0,1]`` when ``vacuous=True``.
    """
    if tensor24.ndim != 4:
        raise ValueError("tensor24 must be [B,C,H,W]")
    out_h, out_w = output_hw
    centers = normalized_cell_centers(out_h, out_w, device=tensor24.device)
    resized_h, resized_w = map(float, geometry["resized_hw"])
    top, left, bottom, right = map(float, geometry["crop_box_yxyx"])
    y_resized = centers[..., 1] * resized_h
    x_resized = centers[..., 0] * resized_w
    support = (y_resized >= top) & (y_resized < bottom) & (x_resized >= left) & (x_resized < right)
    crop_h, crop_w = bottom - top, right - left
    grid_x = 2.0 * ((x_resized - left) / crop_w) - 1.0
    grid_y = 2.0 * ((y_resized - top) / crop_h) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(tensor24.shape[0], -1, -1, -1)
    sampled = F.grid_sample(
        tensor24.to(FUSION_DTYPE), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    support4 = support[None, None].expand(tensor24.shape[0], 1, -1, -1)
    sampled = torch.where(support4, sampled, torch.zeros_like(sampled))
    if vacuous:
        if sampled.shape[1] < 2:
            raise ValueError("vacuous mapping expects singleton+omega masses")
        vacuous_mass = torch.zeros_like(sampled)
        vacuous_mass[:, -1] = 1.0
        sampled = torch.where(support4, sampled, vacuous_mass)
        sampled = sampled / sampled.sum(dim=1, keepdim=True).clamp_min(ECOLAF_LOG_EPS)
    return sampled, support4


def sam_lowres_to_original_normalized(
    lowres: torch.Tensor,
    geometry: Mapping[str, object],
    *,
    output_hw: tuple[int, int] = (256, 256),
    mode: str = "bilinear",
) -> torch.Tensor:
    """Remove SAM right/bottom padding and express a tensor on normalized cells."""
    if lowres.ndim != 4:
        raise ValueError("SAM low-resolution tensor must be [B,C,H,W]")
    resized_h, resized_w = map(int, geometry["resized_hw"])
    raw_h, raw_w = lowres.shape[-2:]
    valid_h = max(1, round(raw_h * resized_h / 1024.0))
    valid_w = max(1, round(raw_w * resized_w / 1024.0))
    valid = lowres[..., :valid_h, :valid_w].to(FUSION_DTYPE)
    kwargs = {"size": output_hw, "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    return F.interpolate(valid, **kwargs)


class LanguageEvidentialHead(nn.Module):
    """Full language head with explicit source-information ablation hooks."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.s64 = nn.Sequential(nn.Conv2d(256, channels, 1), nn.GELU())
        self.q_seg = nn.Sequential(nn.Linear(256, channels), nn.GELU())
        self.z_l = nn.Sequential(nn.Conv2d(1, 16, 1), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels + 16, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2, 1),
        )

    def forward(
        self,
        s64: torch.Tensor,
        q_seg: torch.Tensor,
        z_l: torch.Tensor,
        *,
        use_s64: bool = True,
        use_q_seg: bool = True,
        use_z_l: bool = True,
    ) -> torch.Tensor:
        s = self.s64(s64.float())
        q = self.q_seg(q_seg.float())[:, :, None, None].expand(-1, -1, 64, 64)
        z = self.z_l(F.interpolate(z_l.float(), (64, 64), mode="bilinear", align_corners=False))
        if not use_s64:
            s = torch.zeros_like(s)
        if not use_q_seg:
            q = torch.zeros_like(q)
        if not use_z_l:
            z = torch.zeros_like(z)
        return F.softplus(self.fuse(torch.cat((s, q, z), dim=1)))


class ForensicEvidentialHead(nn.Module):
    """Full forensic head with explicit feature/logit ablation hooks."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.f24 = nn.Sequential(nn.Conv2d(256, channels, 1), nn.GELU())
        self.z_f24 = nn.Sequential(nn.Conv2d(1, 16, 1), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + 16, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2, 1),
        )

    def forward(
        self,
        f24: torch.Tensor,
        z_f24: torch.Tensor,
        *,
        use_f24: bool = True,
        use_z_f24: bool = True,
    ) -> torch.Tensor:
        f = self.f24(f24.float())
        z = self.z_f24(z_f24.float())
        if not use_f24:
            f = torch.zeros_like(f)
        if not use_z_f24:
            z = torch.zeros_like(z)
        return F.softplus(self.fuse(torch.cat((f, z), dim=1)))


@dataclass(frozen=True)
class PCERFAblation:
    use_s64: bool = True
    use_q_seg: bool = True
    use_z_l: bool = True
    use_f24: bool = True
    use_z_f24: bool = True
    language_stream: bool = True
    forensic_stream: bool = True
    conflict_discount: bool = True
    static_fusion: bool = False
    constant_reliability: bool = False
    sample_only_reliability: bool = False


def replace_uncertainty(
    masses: torch.Tensor,
    uncertainty: torch.Tensor,
) -> torch.Tensor:
    """Keep class-belief direction while replacing evidential reliability."""
    belief = masses[:, :-1]
    direction = belief / belief.sum(dim=1, keepdim=True).clamp_min(ECOLAF_LOG_EPS)
    no_belief = belief.sum(dim=1, keepdim=True) <= ECOLAF_LOG_EPS
    direction = torch.where(no_belief, torch.full_like(direction, 1.0 / belief.shape[1]), direction)
    value = uncertainty.clamp(0.0, 1.0)
    return torch.cat((direction * (1.0 - value), value), dim=1)


class PCERF(nn.Module):
    """Hardened, identity-preserving PCERF tensor graph.

    The forward signature contains deployment-time tensors only. Ground truth,
    TF/Phrase identity, authoritative phrases, polygons, and evaluation labels
    are intentionally impossible to pass through this API.
    """

    def __init__(self):
        super().__init__()
        self.language_head = LanguageEvidentialHead()
        self.forensic_head = ForensicEvidentialHead()

    def forward(
        self,
        *,
        s64: torch.Tensor,
        q_seg: torch.Tensor,
        z_l: torch.Tensor,
        f24: torch.Tensor,
        z_f24: torch.Tensor,
        valid_g0: torch.Tensor,
        forensic_present: torch.Tensor,
        forensic_vacuous: torch.Tensor,
        forensic_off: torch.Tensor,
        clip_geometries: list[Mapping[str, object]] | None = None,
        reliability_permutation: torch.Tensor | None = None,
        temperature_l: torch.Tensor | float = 1.0,
        temperature_f: torch.Tensor | float = 1.0,
        ablation: PCERFAblation = PCERFAblation(),
    ) -> dict[str, torch.Tensor]:
        batch = z_l.shape[0]
        if z_l.shape[-2:] != (256, 256):
            raise ValueError("canonical z_L must be [B,1,256,256]")
        for name, flag in {
            "valid_g0": valid_g0,
            "forensic_present": forensic_present,
            "forensic_vacuous": forensic_vacuous,
            "forensic_off": forensic_off,
        }.items():
            if flag.shape != (batch,):
                raise ValueError(f"{name} must have shape [B]")

        e_l = self.language_head(
            s64, q_seg, z_l,
            use_s64=ablation.use_s64,
            use_q_seg=ablation.use_q_seg,
            use_z_l=ablation.use_z_l,
        )
        e_f = self.forensic_head(
            f24, z_f24,
            use_f24=ablation.use_f24,
            use_z_f24=ablation.use_z_f24,
        )
        temperature_l = torch.as_tensor(temperature_l, dtype=e_l.dtype, device=e_l.device)
        temperature_f = torch.as_tensor(temperature_f, dtype=e_f.dtype, device=e_f.device)
        if temperature_l.numel() != 1 or temperature_f.numel() != 1:
            raise ValueError("PCERF calibration temperatures must be scalar")
        if not bool(torch.isfinite(temperature_l)) or not bool(torch.isfinite(temperature_f)):
            raise ValueError("PCERF calibration temperatures must be finite")
        if not bool(temperature_l > 0) or not bool(temperature_f > 0):
            raise ValueError("PCERF calibration temperatures must be positive")
        e_l = e_l / temperature_l
        e_f = e_f / temperature_f
        opinion_l_native = evidence_to_dirichlet(e_l)
        opinion_f_native = evidence_to_dirichlet(e_f)
        mass_l = F.interpolate(opinion_l_native["masses"], (256, 256), mode="bilinear", align_corners=False)
        mass_l = mass_l / mass_l.sum(dim=1, keepdim=True).clamp_min(ECOLAF_LOG_EPS)

        mapped_forensic, support_rows = [], []
        for index in range(batch):
            if clip_geometries is None:
                mapped = F.interpolate(
                    opinion_f_native["masses"][index:index + 1], (256, 256),
                    mode="bilinear", align_corners=False,
                )
                support = torch.ones((1, 1, 256, 256), dtype=torch.bool, device=z_l.device)
            else:
                mapped, support = resample_clip_to_original_normalized(
                    opinion_f_native["masses"][index:index + 1],
                    clip_geometries[index], output_hw=(256, 256), vacuous=True,
                )
            mapped_forensic.append(mapped)
            support_rows.append(support)
        mass_f = torch.cat(mapped_forensic, dim=0)
        support = torch.cat(support_rows, dim=0)

        if ablation.constant_reliability:
            constant_l = torch.full_like(mass_l[:, -1:], 0.5)
            constant_f = torch.full_like(mass_f[:, -1:], 0.5)
            mass_l = replace_uncertainty(mass_l, constant_l)
            mass_f = replace_uncertainty(mass_f, constant_f)
        if ablation.sample_only_reliability:
            sample_u_l = mass_l[:, -1:].mean(dim=(2, 3), keepdim=True).expand_as(mass_l[:, -1:])
            sample_u_f = mass_f[:, -1:].mean(dim=(2, 3), keepdim=True).expand_as(mass_f[:, -1:])
            mass_l = replace_uncertainty(mass_l, sample_u_l)
            mass_f = replace_uncertainty(mass_f, sample_u_f)
        if reliability_permutation is not None:
            if reliability_permutation.shape != (batch,):
                raise ValueError("sample reliability permutation must have shape [B]")
            permutation = reliability_permutation.to(device=z_l.device, dtype=torch.long)
            mass_l = replace_uncertainty(mass_l, mass_l[permutation, -1:])
            mass_f = replace_uncertainty(mass_f, mass_f[permutation, -1:])

        source_active = (
            valid_g0.bool()
            & forensic_present.bool()
            & ~forensic_vacuous.bool()
            & ~forensic_off.bool()
        )
        if not ablation.forensic_stream:
            source_active = torch.zeros_like(source_active)
        active_pixels = support & source_active[:, None, None, None]
        stacked = torch.stack((mass_l, mass_f), dim=2)
        if ablation.static_fusion:
            probability = 0.5 * (dsmp_probability(mass_l) + dsmp_probability(mass_f))
            fused = {
                "discounted_masses": stacked,
                "conflict": torch.zeros((batch, 2, 256, 256), device=z_l.device),
                "discount": torch.ones((batch, 2, 256, 256), device=z_l.device),
                "fused_mass": torch.full_like(mass_l, float("nan")),
                "probability": probability,
            }
        elif ablation.conflict_discount:
            fused = ecolaf_fuse(stacked, classes=2)
        else:
            fused_mass = ecolaf_dempster(stacked)
            fused = {
                "discounted_masses": stacked,
                "conflict": torch.zeros((batch, 2, 256, 256), device=z_l.device),
                "discount": torch.ones((batch, 2, 256, 256), device=z_l.device),
                "fused_mass": fused_mass,
                "probability": dsmp_probability(fused_mass),
            }
        probability_fg = fused["probability"][:, 1:2].clamp(PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
        fused_logits = torch.logit(probability_fg)
        # Source-faithful identity dispatch. This is deliberately not a learned
        # residual: unsupported/absent/vacuous/off forensic evidence is the DS
        # neutral element, and the original P1 tensor is selected bit-for-bit.
        logits = torch.where(active_pixels, fused_logits.to(z_l.dtype), z_l)
        if not ablation.language_stream:
            forensic_probability = dsmp_probability(mass_f)[:, 1:2].clamp(
                PROBABILITY_EPS, 1.0 - PROBABILITY_EPS
            )
            forensic_logits = torch.logit(forensic_probability).to(z_l.dtype)
            unsupported = torch.full_like(z_l, torch.logit(torch.tensor(PROBABILITY_EPS)).item())
            logits = torch.where(active_pixels, forensic_logits, unsupported)
        formal_failure = ~valid_g0.bool()
        logits = torch.where(formal_failure[:, None, None, None], torch.zeros_like(logits), logits)
        return {
            "logits": logits,
            "formal_valid": ~formal_failure,
            "formal_score_override": torch.where(
                formal_failure,
                torch.zeros_like(formal_failure, dtype=torch.float32),
                torch.full_like(formal_failure, float("nan"), dtype=torch.float32),
            ),
            "E_L": e_l,
            "E_F": e_f,
            "alpha_L": opinion_l_native["alpha"],
            "alpha_F": opinion_f_native["alpha"],
            "mass_L": mass_l,
            "mass_F": mass_f,
            "uncertainty_L": opinion_l_native["uncertainty"],
            "uncertainty_F": opinion_f_native["uncertainty"],
            "conflict": fused["conflict"],
            "discount": fused["discount"],
            "fused_mass": fused["fused_mass"],
            "fused_probability": fused["probability"],
            "forensic_support": support,
            "active_pixels": active_pixels,
            "sample_reliability_L": 1.0 - opinion_l_native["uncertainty"].mean(dim=(1, 2, 3)),
            "sample_reliability_F": 1.0 - opinion_f_native["uncertainty"].mean(dim=(1, 2, 3)),
        }
