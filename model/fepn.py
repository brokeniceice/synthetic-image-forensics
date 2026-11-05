"""Minimal standalone Forensic Evidence Prompt Network for Phase 4A."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedResidualOperator(nn.Module):
    """Deterministic depthwise Laplacian on denormalized RGB.

    The kernel and CLIP normalization constants are buffers, never parameters.
    """

    def __init__(self, mean, std) -> None:
        super().__init__()
        kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
        self.register_buffer("kernel", kernel.reshape(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).reshape(1, 3, 1, 1))

    def forward(self, normalized_rgb: torch.Tensor) -> torch.Tensor:
        rgb = normalized_rgb.float() * self.std + self.mean
        return F.conv2d(F.pad(rgb, (1, 1, 1, 1), mode="reflect"), self.kernel, groups=3)


class ConvGNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int = 3, stride: int = 1) -> None:
        padding = kernel // 2
        groups = min(16, cout)
        while cout % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(cin, cout, kernel, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
        )


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(channels, channels, 3),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.body(value))


class FEPNv0(nn.Module):
    """Small dual-view convolutional encoder with FPN-style top-down fusion."""

    def __init__(self, image_mean, image_std) -> None:
        super().__init__()
        self.residual_operator = FixedResidualOperator(image_mean, image_std)
        self.rgb_stem = nn.Sequential(ConvGNAct(3, 32, 3, 2), ConvGNAct(32, 64, 3, 2))
        self.residual_stem = nn.Sequential(ConvGNAct(3, 32, 3, 2), ConvGNAct(32, 64, 3, 2))
        self.fusion = ConvGNAct(128, 96, 1)
        self.level1 = nn.Sequential(ResidualConvBlock(96), ResidualConvBlock(96))
        self.level2 = nn.Sequential(ConvGNAct(96, 128, 3, 2), ResidualConvBlock(128))
        self.level3 = nn.Sequential(ConvGNAct(128, 192, 3, 2), ResidualConvBlock(192))
        self.lateral1 = nn.Conv2d(96, 128, 1)
        self.lateral2 = nn.Conv2d(128, 128, 1)
        self.lateral3 = nn.Conv2d(192, 128, 1)
        self.smooth2 = ConvGNAct(128, 128, 3)
        self.smooth1 = ConvGNAct(128, 128, 3)
        self.dense_head = nn.Sequential(ConvGNAct(128, 64, 3), nn.Conv2d(64, 1, 1))
        self.global_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def encode(self, normalized_rgb: torch.Tensor):
        residual = self.residual_operator(normalized_rgb)
        rgb_feature = self.rgb_stem(normalized_rgb)
        residual_feature = self.residual_stem(residual.to(dtype=normalized_rgb.dtype))
        level1 = self.level1(self.fusion(torch.cat([rgb_feature, residual_feature], dim=1)))
        level2 = self.level2(level1)
        level3 = self.level3(level2)
        top3 = F.interpolate(self.lateral3(level3).float(), size=level2.shape[-2:], mode="nearest").to(level2.dtype)
        pyramid2 = self.smooth2(
            self.lateral2(level2) + top3
        )
        top2 = F.interpolate(pyramid2.float(), size=level1.shape[-2:], mode="nearest").to(level1.dtype)
        dense = self.smooth1(
            self.lateral1(level1) + top2
        )
        return dense, residual

    def forward(self, normalized_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        dense, residual = self.encode(normalized_rgb)
        global_logits = self.global_head(dense.mean(dim=(2, 3))).squeeze(1)
        dense_logits = self.dense_head(dense)
        return {
            "global_logits": global_logits,
            "dense_logits": dense_logits,
            "dense_features": dense,
            "residual_view": residual,
        }


class ForensicEvidenceProjector(nn.Module):
    """Frozen Phase 4B-G interface: one global FEPN vector to K LLM tokens."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256,
                 num_tokens: int = 4, llm_dim: int = 4096) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.llm_dim = int(llm_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_tokens * self.llm_dim),
        )

    def forward(self, pooled_feature: torch.Tensor) -> torch.Tensor:
        if pooled_feature.ndim != 2 or pooled_feature.shape[1] != self.input_dim:
            raise ValueError(f"expected [B,{self.input_dim}], got {tuple(pooled_feature.shape)}")
        projected = self.network(pooled_feature)
        return projected.reshape(projected.shape[0], self.num_tokens, self.llm_dim)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def parameter_manifest(model: nn.Module, input_resolution: int = 336) -> dict:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    buffers = sum(value.numel() for value in model.buffers())
    with torch.no_grad():
        output = model(torch.zeros(1, 3, input_resolution, input_resolution))
    return {
        "architecture": "FEPN-v0",
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "buffer_elements": int(buffers),
        "input_shape": [3, input_resolution, input_resolution],
        "dense_feature_shape": list(output["dense_features"].shape[1:]),
        "dense_logit_shape": list(output["dense_logits"].shape[1:]),
        "residual_operator_trainable_parameters": 0,
    }
