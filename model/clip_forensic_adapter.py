"""Phase 4C-A lightweight local specialization over frozen CLIP grids."""

from __future__ import annotations

import torch
import torch.nn as nn


class LocalForensicBlock(nn.Module):
    def __init__(self, channels: int = 256, groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.pointwise(self.activation(self.depthwise(self.norm(value))))


class CLIPSpatialArm(nn.Module):
    """Matched projection-only or three-block forensic-adapter arm."""

    def __init__(self, input_channels: int = 1024, channels: int = 256,
                 blocks: int = 0, groups: int = 32):
        super().__init__()
        if blocks not in (0, 3):
            raise ValueError("Phase 4C-A permits exactly 0 or 3 local blocks")
        self.projection = nn.Conv2d(input_channels, channels, 1)
        self.forensic_blocks = nn.Sequential(
            *(LocalForensicBlock(channels, groups) for _ in range(blocks))
        )
        self.dense_head = nn.Conv2d(channels, 1, 1, bias=True)

    def forward(self, clip_feature: torch.Tensor, *, return_features: bool = False):
        base = self.projection(clip_feature)
        forensic = self.forensic_blocks(base)
        logits = self.dense_head(forensic)
        if return_features:
            return {"logits": logits, "F0": base, "F_forensic": forensic}
        return logits

