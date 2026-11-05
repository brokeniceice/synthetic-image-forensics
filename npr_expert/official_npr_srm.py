"""官方 NPR 主干与独立的 SRM 残差分支。"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _IncompatibleKeys

from .focal_extractor import FrozenFOCALViT


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        identity = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        outputs += identity
        return self.relu(outputs)


class OfficialNPRSRM(nn.Module):
    """官方 NPR 加 SRM，返回形状为 ``[B, 1]`` 的伪造类别 logit。

    NPR 分支保持官方结构，在第二个残差阶段后直接做全局平均池化。SRM
    分支也只产生全局特征；两个分支相加后由单输出线性层完成二分类。
    正 logit 表示伪造类别，训练时使用 BCEWithLogitsLoss。
    """

    def __init__(
        self,
        focal_weights=None,
        focal_micro_batch_size=1,
        focal_gate_init=0.0,
        use_cached_focal=False,
        focal_only=False,
    ):
        super().__init__()
        self.focal_only = bool(focal_only)
        if self.focal_only and focal_weights is None and not use_cached_focal:
            raise ValueError("focal_only 要求启用 FOCAL extractor 或缓存特征")
        self.inplanes = 64
        self.unfoldSize = 2
        self.unfoldIndex = 0
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)

        self.register_buffer("srm_kernels", self._build_srm_kernels(), persistent=False)
        self.srm_stem = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.srm_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.srm_gate = nn.Parameter(torch.tensor(0.1))
        self.focal_extractor = None
        self._init_weights()
        if focal_weights is not None:
            self.focal_extractor = FrozenFOCALViT(
                focal_weights,
                micro_batch_size=focal_micro_batch_size,
            )
        if focal_weights is not None or use_cached_focal:
            self.focal_norm = nn.LayerNorm(512)
            if not self.focal_only:
                self.focal_gate = nn.Parameter(torch.tensor(float(focal_gate_init)))
        if self.focal_only:
            for module in (
                self.conv1,
                self.bn1,
                self.layer1,
                self.layer2,
                self.srm_stem,
            ):
                module.requires_grad_(False)
            self.srm_gate.requires_grad_(False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def _build_srm_kernels():
        filter_1 = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ) / 4.0
        filter_2 = torch.tensor(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=torch.float32,
        ) / 12.0
        filter_3 = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ) / 2.0
        return torch.stack([filter_1, filter_2, filter_3]).unsqueeze(1).repeat(3, 1, 1, 1)

    @staticmethod
    def interpolate(images, factor):
        dtype = images.dtype
        images = images.float()
        images = F.interpolate(images, scale_factor=factor, mode="nearest", recompute_scale_factor=True)
        images = F.interpolate(images, scale_factor=1 / factor, mode="nearest", recompute_scale_factor=True)
        return images.to(dtype=dtype)

    def srm_residual_features(self, images):
        kernels = self.srm_kernels.to(device=images.device, dtype=torch.float32)
        residuals = F.conv2d(images.float(), kernels, padding=2, groups=3)
        return residuals.clamp(-3.0, 3.0).to(dtype=images.dtype)

    @property
    def uses_focal(self):
        return hasattr(self, "focal_norm")

    def train(self, mode=True):
        super().train(mode)
        if self.focal_extractor is not None:
            self.focal_extractor.eval()
        return self

    def state_dict(self, *args, **kwargs):
        """Exclude the immutable 1.2 GB FOCAL extractor from small checkpoints."""
        state = super().state_dict(*args, **kwargs)
        focal_marker = "focal_extractor."
        for key in [key for key in state if focal_marker in key]:
            del state[key]
        return state

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=False)
        missing = [key for key in result.missing_keys if not key.startswith("focal_extractor.")]
        unexpected = list(result.unexpected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "NPR+SRM+FOCAL checkpoint 不兼容："
                f"missing={missing}, unexpected={unexpected}"
            )
        return _IncompatibleKeys(missing, unexpected)

    def forward(self, images, focal_images=None):
        features = None
        if not self.focal_only:
            branches = self.extract_branch_features(images)
            features = branches["npr"] + branches["srm_gated"]
        if self.uses_focal:
            if focal_images is None:
                raise ValueError("启用 FOCAL extractor 时必须提供 focal_images")
            if focal_images.ndim == 2:
                target_device = images.device if features is None else features.device
                focal_features = focal_images.to(device=target_device, dtype=torch.float32)
            elif self.focal_extractor is not None:
                focal_features = self.focal_extractor(focal_images)
            else:
                raise ValueError("当前模型只有缓存模式，focal_images 必须是 [B,512] 特征")
            focal_features = self.focal_norm(focal_features.float())
            features = (
                focal_features
                if self.focal_only
                else features + self.focal_gate * focal_features
            )
        return self.fc1(features)

    def extract_branch_features(self, images):
        """Return the trained NPR and SRM image-level representations.

        This is a read-only decomposition of the existing expert graph.  It is
        used by Phase 2C to ablate the two signals without inventing new NPR or
        SRM algorithms.  ``srm_gated`` is the exact residual contribution used
        by the historical joint expert; ``srm_raw`` is retained for provenance
        and diagnostics.
        """
        if self.focal_only:
            raise ValueError("focal_only has no NPR/SRM branch features")
        npr = self.extract_npr_features(images)
        srm_raw = self.extract_srm_features(images)
        srm_gated = self.srm_gate.float() * srm_raw
        return {"npr": npr, "srm_raw": srm_raw, "srm_gated": srm_gated}

    def extract_npr_features(self, images):
        """Official NPR branch through layer2 and global average pooling."""
        npr_images = images - self.interpolate(images, 0.5)
        npr = self.maxpool(self.relu(self.bn1(self.conv1(npr_images * 2.0 / 3.0))))
        return self.avgpool(self.layer2(self.layer1(npr))).flatten(1).float()

    def extract_npr_spatial_features(self, images):
        """Return the official NPR ``layer2`` map before global pooling.

        This read-only Phase 3C.1 API intentionally duplicates the existing
        branch graph instead of routing :meth:`extract_npr_features` through a
        new helper.  The historical classification path therefore remains
        byte-for-byte unchanged.
        """
        npr_images = images - self.interpolate(images, 0.5)
        npr = self.maxpool(self.relu(self.bn1(self.conv1(npr_images * 2.0 / 3.0))))
        return self.layer2(self.layer1(npr))

    def extract_srm_features(self, images):
        """Frozen Phase 2C SRM backend feature before the historical gate."""
        return self.srm_pool(
            self.srm_stem(self.srm_residual_features(images))
        ).flatten(1).float()

    def extract_srm_spatial_features(self, images):
        """Return the frozen SRM stem map before ``srm_pool``."""
        return self.srm_stem(self.srm_residual_features(images))

    def branch_logits(self, images):
        """Expose standalone branch scores under the historical shared head."""
        branches = self.extract_branch_features(images)
        return {
            "npr": self.fc1(branches["npr"]),
            "srm": self.fc1(branches["srm_gated"]),
            "npr_srm": self.fc1(branches["npr"] + branches["srm_gated"]),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path, freeze=False, **model_kwargs):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model = cls(**model_kwargs)
        model.load_state_dict(state_dict, strict=True)
        if freeze:
            model.requires_grad_(False).eval()
        return model

    @torch.no_grad()
    def predict(self, images, focal_images=None, threshold=0.5):
        logits = self(images, focal_images=focal_images)
        fake_probabilities = torch.sigmoid(logits.float())
        return {
            "logits": logits,
            "fake_probabilities": fake_probabilities,
            "predictions": fake_probabilities.ge(threshold).long().squeeze(-1),
        }
