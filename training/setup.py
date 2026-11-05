"""Shared argument construction and device transfer for detector training."""
from __future__ import annotations
import torch
from training import backbone as glamm_train

def build_train_args(config):
    args = glamm_train.parse_args([])
    model = config["model"]
    loss = config["loss"]
    args.version = model["version"]
    args.vision_tower = model["vision_tower"]
    args.vision_pretrained = model.get("vision_pretrained")
    args.pretrained = bool(model["pretrained"])
    args.token_strategy = model["token_strategy"]
    args.model_max_length = int(model["model_max_length"])
    args.image_size = int(model["image_size"])
    args.out_dim = int(model["out_dim"])
    args.precision = model["precision"]
    args.lora_r = int(model["lora_r"])
    args.lora_alpha = int(model["lora_alpha"])
    args.lora_dropout = float(model["lora_dropout"])
    args.lora_target_modules = model["lora_target_modules"]
    args.train_mask_decoder = bool(model["train_mask_decoder"])
    args.ce_loss_weight = float(loss["text_weight"])
    args.cls_loss_weight = float(loss["classification_weight"])
    args.bce_loss_weight = float(loss["mask_bce_weight"])
    args.dice_loss_weight = float(loss["mask_dice_weight"])
    args.per_sample_text_loss_normalization = bool(loss.get("per_sample_text_loss_normalization", False))
    args.local_rank = 0
    return args


def move_batch(batch, device, dtype):
    for key, value in tuple(batch.items()):
        if torch.is_tensor(value):
            value = value.to(device)
            if key in {"global_enc_images", "grounding_enc_images"}:
                value = value.to(dtype)
            batch[key] = value
        elif isinstance(value, list):
            batch[key] = [
                item.to(device) if torch.is_tensor(item) else item
                for item in value
            ]
    return batch


