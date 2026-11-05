"""Assemble the same C1 adapter topology as the frozen evaluation."""
from pathlib import Path
import hashlib
import json
import torch
import train as glamm_train


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def configure_args(config):
    args = glamm_train.parse_args([])
    model, loss = config['model'], config['loss']
    for key in ('version', 'vision_tower', 'pretrained', 'token_strategy',
                'model_max_length', 'image_size', 'out_dim', 'precision',
                'lora_r', 'lora_alpha', 'lora_dropout', 'lora_target_modules',
                'train_mask_decoder', 'freeze_region_encoder'):
        setattr(args, key, model[key])
    args.vision_pretrained = model.get('vision_pretrained')
    args.ce_loss_weight = float(loss['text_weight'])
    args.cls_loss_weight = float(loss['classification_weight'])
    args.bce_loss_weight = float(loss['mask_bce_weight'])
    args.dice_loss_weight = float(loss['mask_dice_weight'])
    args.per_sample_text_loss_normalization = bool(loss.get('per_sample_text_loss_normalization', False))
    args.lr = float(config['optimizer']['lora_lr'])
    for key in ('lora_lr', 'classification_head_lr', 'text_hidden_fcs_lr',
                'mask_decoder_lr', 'embedding_lr', 'lm_head_lr'):
        setattr(args, key, float(config['optimizer'][key]))
    args.local_rank = 0
    return args


def load_c1(config, checkpoint, device):
    expected = json.loads((Path(__file__).resolve().parents[1] / 'configs/paper/protocol.json').read_text())['checkpoint_sha256']
    observed_c1 = file_sha256(checkpoint)
    if observed_c1 != expected['c1']:
        raise ValueError('C1 checkpoint differs from the final paper asset')
    if file_sha256(config['forensics']['rine_checkpoint']) != expected['rine']:
        raise ValueError('RINE checkpoint differs from the final paper asset')
    base_assets = json.loads((Path(__file__).resolve().parents[1] / 'configs/paper/base_assets.json').read_text())
    for group, directory in [('glamm', config['model']['version']), ('clip', config['model']['vision_tower'])]:
        for filename, digest in base_assets[group].items():
            if file_sha256(Path(directory) / filename) != digest:
                raise ValueError(f'Base asset differs from the paper checkpoint: {group}/{filename}')
    args = configure_args(config)
    args.local_rank = device.index or 0
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    model = glamm_train.initialize_model(args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, args)
    model.enable_rine_conditioning(Path(config['forensics']['rine_checkpoint']).resolve())
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (int(payload.get('epoch', -1)), int(payload.get('optimizer_step', -1))) != (5, 2500):
        raise ValueError('The final C1 checkpoint must be epoch 5 / optimizer step 2500')
    missing, unexpected = model.load_state_dict(payload['module'], strict=False)
    if unexpected:
        raise ValueError(f'Unexpected C1 checkpoint keys: {unexpected}')
    model.gradient_checkpointing_disable()
    model.requires_grad_(False).to(device=device, dtype=torch.bfloat16).eval()
    model.rine_conditioner.rine.float().eval()
    return model, tokenizer, {'c1_sha256': observed_c1,
                              'missing_frozen_keys': len(missing)}
