"""Frozen FOCAL ViT-L forensic feature extractor.

FOCAL uses the SAM ViT-L image encoder as a dense forensic feature extractor.
Only the image-encoder weights are loaded; SAM's prompt encoder and mask
decoder are intentionally not constructed.
"""

from pathlib import Path

import torch
import torch.nn as nn

from model.SAM.modeling import ImageEncoderViT


class FrozenFOCALViT(nn.Module):
    """Load the author's FOCAL-ViT weights and return pooled 512-D features."""

    _WEIGHT_PREFIXES = (
        "module.net.image_encoder.",
        "net.image_encoder.",
        "module.image_encoder.",
        "image_encoder.",
    )

    def __init__(self, weights_path, micro_batch_size=1):
        super().__init__()
        if micro_batch_size < 1:
            raise ValueError("FOCAL micro_batch_size 必须大于 0")
        self.weights_path = str(Path(weights_path))
        self.micro_batch_size = int(micro_batch_size)
        self.image_encoder = ImageEncoderViT(
            depth=24,
            embed_dim=1024,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=lambda channels: nn.LayerNorm(channels, eps=1e-6),
            num_heads=16,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[5, 11, 17, 23],
            window_size=14,
            out_chans=256,
        )
        self._load_focal_weights(self.weights_path)
        self.requires_grad_(False)
        self.eval()

    def _load_focal_weights(self, weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            for container_key in ("model", "state_dict"):
                if container_key in checkpoint and isinstance(checkpoint[container_key], dict):
                    checkpoint = checkpoint[container_key]
                    break
        if not isinstance(checkpoint, dict):
            raise ValueError(f"无法识别 FOCAL 权重格式：{weights_path}")

        encoder_state = {}
        for key, value in checkpoint.items():
            for prefix in self._WEIGHT_PREFIXES:
                if key.startswith(prefix):
                    encoder_state[key[len(prefix):]] = value
                    break
        if not encoder_state:
            raise ValueError(f"权重中没有找到 FOCAL image_encoder 参数：{weights_path}")
        self.image_encoder.load_state_dict(encoder_state, strict=True)
        del checkpoint, encoder_state

    def train(self, mode=True):
        """The published extractor always remains frozen and in eval mode."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        """Process CPU or GPU images in small chunks and return ``[B, 512]``."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"FOCAL 输入应为 [B,3,H,W]，实际为 {tuple(images.shape)}")
        device = next(self.image_encoder.parameters()).device
        pooled_features = []
        for image_chunk in images.split(self.micro_batch_size):
            image_chunk = image_chunk.to(device=device, dtype=torch.float32, non_blocking=True)
            # The published FOCAL extractor is trained and evaluated in FP32.
            # Keep it outside the NPR trainer's optional AMP context so the
            # frozen feature distribution matches the author's implementation.
            with torch.cuda.amp.autocast(enabled=False):
                spatial_features = self.image_encoder(image_chunk)
            mean_features = spatial_features.mean(dim=(2, 3))
            max_features = spatial_features.amax(dim=(2, 3))
            pooled_features.append(torch.cat([mean_features, max_features], dim=1))
        return torch.cat(pooled_features, dim=0)

    @torch.no_grad()
    def forward_spatial(self, images):
        """Return the published FP32 image-encoder map before mean/max pooling.

        The historical :meth:`forward` implementation is deliberately left
        untouched so adding this diagnostic API cannot change its numerics.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"FOCAL 输入应为 [B,3,H,W]，实际为 {tuple(images.shape)}")
        device = next(self.image_encoder.parameters()).device
        spatial = []
        for image_chunk in images.split(self.micro_batch_size):
            image_chunk = image_chunk.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=False):
                spatial.append(self.image_encoder(image_chunk))
        return torch.cat(spatial, dim=0)
