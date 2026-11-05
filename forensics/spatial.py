"""Spatial computations used by the final NoRelative model."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from model.csculf import CSCULF, build_comparison_features
from model.pcerf import evidence_to_dirichlet, ecolaf_discount

def clip_grid(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise RuntimeError(f"CLIP patch tensor is not [B,N,C]: {tuple(tokens.shape)}")
    side = int(round(tokens.shape[1] ** 0.5))
    if side * side != tokens.shape[1]:
        raise RuntimeError(f"CLIP_SPATIAL_UNAVAILABLE: token count {tokens.shape[1]} is not a square grid")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)

def _utility_forward(model: CSCULF, batch: dict, *, forensic_index: torch.Tensor | None = None,
                    spatial_permutation: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Materialize the frozen CSCU utility graph without the 256-grid fusion tail."""
    s64, q, zl = batch["S64"], batch["q_seg"], batch["z_L"]
    f24, zf, geometries = batch["F24"], batch["z_F24"], batch["clip_geometries"]
    if forensic_index is not None:
        f24, zf = f24.index_select(0, forensic_index), zf.index_select(0, forensic_index)
    if spatial_permutation is not None:
        f24 = f24.flatten(2)[:, :, spatial_permutation].reshape_as(f24)
        zf = zf.flatten(2)[:, :, spatial_permutation].reshape_as(zf)
    evidence_l = model.language_source(s64.detach(), q.detach(), zl.detach()) / model.temperature_l
    evidence_f = model.forensic_source(f24.detach(), zf.detach()) / model.temperature_f
    opinion_l, opinion_f = evidence_to_dirichlet(evidence_l), evidence_to_dirichlet(evidence_f)
    mass_l = opinion_l["masses"]
    mass_f, support = model._map_forensic(opinion_f["masses"], geometries, output_hw=(64, 64), vacuous=True)
    aligned_f, mapped_support = model._map_forensic(f24.detach(), geometries, output_hw=(64, 64), vacuous=False)
    aligned_zf, _ = model._map_forensic(zf.detach(), geometries, output_hw=(64, 64), vacuous=False)
    if not torch.equal(support, mapped_support): raise RuntimeError("support drift")
    p_l = opinion_l["posterior"]
    p_f = mass_f[:, :-1] + mass_f[:, -1:] / 2.0
    l64 = model.language_context(s64, q, zl)
    f64 = model.forensic_context(aligned_f, aligned_zf, support)
    lr, fr, rectification = model.rectification(l64, f64, support)
    lr, fr, exchange = model.exchange(lr, fr, support)
    conflict = ecolaf_discount(torch.stack((mass_l, mass_f), dim=2), classes=2)[1]
    comparison, parts = build_comparison_features(lr, fr, p_l, p_f, conflict, support)
    hidden = model.utility_head.net[:-1](comparison)
    utility_logit = model.utility_head.net[-1](hidden)
    utility = utility_logit.sigmoid() * support.float()
    return {"utility_logit": utility_logit, "U": utility, "support": support, "p_L": p_l, "p_F": p_f,
            "mass_L": mass_l, "mass_F": mass_f, "L64": l64, "Fctx64": f64, "aligned_F64": aligned_f,
            "aligned_z_F64": aligned_zf, "Lr": lr, "Fr": fr, "comparison": comparison,
            "rectification": rectification, "exchange": exchange, "utility_hidden": hidden, "parts": parts}

def utility_forward(model, batch):
    with torch.autocast(device_type=batch['S64'].device.type, dtype=torch.bfloat16):
        return _utility_forward(model, batch)

def gate_to_sam_grid(utility64: torch.Tensor, sam_coordinates: torch.Tensor) -> torch.Tensor:
    """Sample original-normalized U_F at Phase4F SAM-token coordinates."""
    grid = (2.0 * sam_coordinates.float() - 1.0).reshape(utility64.shape[0], 64, 64, 2)
    value = F.grid_sample(utility64.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return value.clamp(0.0, 1.0)

def gated_embedding(s64: torch.Tensor, phase4f: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """S64 + g*Delta_F with exact representable endpoints."""
    if gate.shape != (s64.shape[0], 1, 64, 64):
        raise ValueError("aligned gate must be [B,1,64,64]")
    mixed = (s64.float() + gate.float() * (phase4f.float() - s64.float())).to(s64.dtype)
    mixed = torch.where(gate == 0, s64, mixed)
    mixed = torch.where(gate == 1, phase4f, mixed)
    return mixed

