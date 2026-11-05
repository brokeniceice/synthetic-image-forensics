"""Classification-only residual forensic fusion for Phase 2C."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


BRANCH_FEATURES = {
    "npr": ("npr",),
    "srm": ("srm",),
    "npr_srm": ("npr", "srm"),
}


@dataclass
class ForensicFusionOutput:
    logits: torch.Tensor
    delta_logits: torch.Tensor
    scaled_delta_logits: torch.Tensor


class ResidualForensicFusion(nn.Module):
    """Add bounded trainable forensic evidence to frozen Phase 2A logits.

    The module intentionally does not own or call GLaMM.  Baseline logits and
    the fixed CLS hidden state are detached inputs, which makes contamination
    of the language/generation/localization graph impossible by construction.
    """

    def __init__(
        self,
        variant: str,
        *,
        semantic_dim: int,
        npr_dim: int = 512,
        srm_dim: int = 512,
        d_fuse: int = 256,
    ) -> None:
        super().__init__()
        if variant not in BRANCH_FEATURES:
            raise ValueError(f"unsupported forensic fusion variant: {variant}")
        self.variant = variant
        self.d_fuse = int(d_fuse)
        self.semantic_input_norm = nn.LayerNorm(int(semantic_dim))
        self.semantic_projector = nn.Sequential(
            nn.Linear(int(semantic_dim), self.d_fuse), nn.LayerNorm(self.d_fuse)
        )
        if "npr" in BRANCH_FEATURES[variant]:
            self.npr_projector = nn.Sequential(
                nn.Linear(int(npr_dim), self.d_fuse), nn.LayerNorm(self.d_fuse)
            )
        if "srm" in BRANCH_FEATURES[variant]:
            self.srm_projector = nn.Sequential(
                nn.Linear(int(srm_dim), self.d_fuse), nn.LayerNorm(self.d_fuse)
            )
        input_dim = self.d_fuse * (1 + len(BRANCH_FEATURES[variant]))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, self.d_fuse),
            nn.GELU(),
            nn.Linear(self.d_fuse, 2),
        )
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, base_logits, h_cls, *, npr=None, srm=None):
        base_logits = base_logits.detach().float()
        parts = [self.semantic_projector(self.semantic_input_norm(h_cls.detach().float()))]
        if "npr" in BRANCH_FEATURES[self.variant]:
            if npr is None:
                raise ValueError(f"{self.variant} requires NPR features")
            parts.append(self.npr_projector(npr.detach().float()))
        if "srm" in BRANCH_FEATURES[self.variant]:
            if srm is None:
                raise ValueError(f"{self.variant} requires SRM features")
            parts.append(self.srm_projector(srm.detach().float()))
        delta = self.fusion_mlp(torch.cat(parts, dim=-1))
        scaled_delta = self.alpha * delta
        return ForensicFusionOutput(
            logits=base_logits + scaled_delta,
            delta_logits=delta,
            scaled_delta_logits=scaled_delta,
        )

    def trainable_parameter_audit(self):
        return {
            "variant": self.variant,
            "d_fuse": self.d_fuse,
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
            "alpha_initial": float(self.alpha.detach()),
        }
