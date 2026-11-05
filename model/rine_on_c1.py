"""RINE classification head adapted to the frozen C1 Hugging Face CLIP tower.

The trainable topology is a direct transcription of RINE commit
9b7fd5857cc205d0412be6aeee0d7611b95bd620, src/models.py.  The only adapter
is the tensor-layout bridge from Hugging Face CLIP layer_norm2 outputs
([batch, tokens, channels]) to RINE's [batch, blocks, channels] input.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


RINE_OFFICIAL_COMMIT = "9b7fd5857cc205d0412be6aeee0d7611b95bd620"
RINE_Q = 2
RINE_PROJ_DIM = 1024
RINE_XI = 0.2
RINE_DROPOUT = 0.5


class RINEHead(nn.Module):
    """Official Q1/TIE/Q2/classifier topology, without an owned backbone."""

    def __init__(
        self,
        block_count: int = 24,
        input_dim: int = 1024,
        nproj: int = RINE_Q,
        proj_dim: int = RINE_PROJ_DIM,
    ) -> None:
        super().__init__()
        self.block_count = block_count
        self.input_dim = input_dim
        self.nproj = nproj
        self.proj_dim = proj_dim

        # Names and initialization follow official src/models.py.
        self.alpha = nn.Parameter(torch.randn(1, block_count, proj_dim))
        proj1_layers: List[nn.Module] = [nn.Dropout()]
        for index in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(input_dim if index == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)

        proj2_layers: List[nn.Module] = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend(
                [nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout()]
            )
        self.proj2 = nn.Sequential(*proj2_layers)
        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, block_cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        expected = (self.block_count, self.input_dim)
        if block_cls.ndim != 3 or tuple(block_cls.shape[1:]) != expected:
            raise ValueError(
                f"RINE expects [batch,{expected[0]},{expected[1]}], got {tuple(block_cls.shape)}"
            )
        g = self.proj1(block_cls.float())
        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)
        z = self.proj2(z)
        return self.head(z), z


class RINEOnHFCLIP(nn.Module):
    """Frozen HF CLIP plus hooks matching official OpenAI CLIP ``ln_2`` hooks."""

    def __init__(self, vision_model: nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.vision_model.requires_grad_(False)
        self.vision_model.eval()

        layers = self.vision_model.vision_model.encoder.layers
        if len(layers) != 24:
            raise ValueError(f"Expected CLIP ViT-L/14 with 24 blocks, found {len(layers)}")
        self._captured: List[torch.Tensor] = []
        self._capture_enabled = False
        self._hook_names = [
            f"vision_model.encoder.layers.{index}.layer_norm2"
            for index in range(len(layers))
        ]
        self._handles = [
            layer.layer_norm2.register_forward_hook(self._capture_layer_norm2)
            for layer in layers
        ]
        self.rine = RINEHead(block_count=len(layers), input_dim=1024)

    def _capture_layer_norm2(self, module, inputs, output) -> None:
        del module, inputs
        # The Phase6D.3 C1 model shares this frozen CLIP tower with the normal
        # LLaVA visual path.  Capture only the explicitly requested RINE pass;
        # retaining ordinary visual-forward activations here causes an
        # inference-time GPU memory leak without contributing to Q2.
        if self._capture_enabled:
            self._captured.append(output)

    @property
    def hook_names(self) -> List[str]:
        return list(self._hook_names)

    def train(self, mode: bool = True):
        super().train(mode)
        # ``super().train`` recursively toggles the tower, so restore the
        # backbone's frozen inference behavior on every call.
        self.vision_model.eval()
        return self

    def extract_block_cls(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, object]:
        self._captured.clear()
        self._capture_enabled = True
        try:
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=True)
        finally:
            self._capture_enabled = False
        if len(self._captured) != len(self._hook_names):
            raise RuntimeError(
                f"Expected {len(self._hook_names)} layer_norm2 hooks, captured {len(self._captured)}"
            )
        if any(value.ndim != 3 for value in self._captured):
            raise RuntimeError("A captured HF layer_norm2 output is not [batch,tokens,channels]")
        block_cls = torch.stack([value[:, 0, :] for value in self._captured], dim=1)
        return block_cls, outputs

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_cls, _ = self.extract_block_cls(pixel_values)
        logits, embedding = self.rine(block_cls)
        return logits, embedding, block_cls

    def close_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class OfficialSupConLoss(nn.Module):
    """Exact SupCon implementation shipped in RINE ``src/utils.py``."""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        if len(features.shape) < 3:
            raise ValueError("features needs at least [batch, views, ...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both labels and mask")
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


def official_rine_loss(
    logits: torch.Tensor,
    embedding: torch.Tensor,
    fake_labels: torch.Tensor,
    xi: float = RINE_XI,
) -> dict:
    """Return the exact RINE BCE-sum plus weighted supervised contrastive loss."""
    bce = nn.BCEWithLogitsLoss(reduction="sum")(
        logits, fake_labels.float().view(-1, 1)
    )
    supcon = OfficialSupConLoss()(
        F.normalize(embedding, dim=1).unsqueeze(1), fake_labels
    )
    return {"total": bce + xi * supcon, "bce": bce, "supcon": supcon}
