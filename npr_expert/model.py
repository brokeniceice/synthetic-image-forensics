"""独立 NPR+SRM 专家网络的统一调用接口。"""

from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn

from .official_npr_srm import OfficialNPRSRM


@dataclass
class NPRExpertOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor


class NPRExpert(nn.Module):
    """与 GLaMM 解耦的 NPR+SRM 真假二分类器。"""

    def __init__(self, **model_kwargs):
        super().__init__()
        self.model = OfficialNPRSRM(**model_kwargs)

    def forward(self, images, focal_images=None):
        logits = self.model(images, focal_images=focal_images)
        fake_probabilities = torch.sigmoid(logits.float())
        probabilities = torch.cat([1.0 - fake_probabilities, fake_probabilities], dim=-1)
        predictions = fake_probabilities.ge(0.5).long().squeeze(-1)
        return NPRExpertOutput(
            logits=logits,
            probabilities=probabilities,
            predictions=predictions,
        )

    def freeze(self):
        self.requires_grad_(False)
        self.eval()
        return self

    def unfreeze(self):
        self.requires_grad_(True)
        return self

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        *,
        freeze=False,
        **model_kwargs,
    ):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model = cls(**model_kwargs)
        model.model.load_state_dict(state_dict, strict=True)
        if freeze:
            model.freeze()
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
