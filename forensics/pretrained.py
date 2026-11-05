"""Load complete model shards with a separately cached CLIP visual encoder."""
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import GenerationConfig
from transformers.modeling_utils import no_init_weights

from forensics.checkpoints import configure_args
from model.GLaMM import GLaMMForCausalLM
from model.llava.model.language_model.llava_llama import LlavaConfig
from training import backbone


def is_clip_weight(name):
    return '.vision_tower.' in name or '.rine_conditioner.vision_model.' in name


def load_module(directory, module, prefix, device):
    """Materialize stored tensors without changing their precision."""
    directory = Path(directory)
    index = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
    expected = {name for name in module.state_dict() if not is_clip_weight('.' + name)}
    stored = {name[len(prefix):] for name in index if name.startswith(prefix)}
    if expected != stored:
        raise ValueError(f'Incomplete {prefix} weights: missing={sorted(expected-stored)}, unexpected={sorted(stored-expected)}')
    by_shard = {}
    for name in sorted(stored):
        by_shard.setdefault(index[prefix + name], []).append(name)
    for filename, names in by_shard.items():
        if Path(filename).name != filename:
            raise ValueError('Invalid model shard filename')
        with safe_open(directory / filename, framework='pt', device='cpu') as handle:
            for name in names:
                tensor = handle.get_tensor(prefix + name)
                # Accelerate otherwise casts values to the uninitialized parameter dtype.
                set_module_tensor_to_device(module, name, device, value=tensor, dtype=tensor.dtype)
    module.requires_grad_(False).eval()


def load_backbone(config, directory, manifest, device):
    args = configure_args(config)
    args.local_rank = device.index or 0
    tokenizer = backbone.setup_tokenizer_and_special_tokens(args)
    language = LlavaConfig.from_dict(manifest['language_config'])
    language.mm_vision_tower = config['model']['vision_tower']
    language.vision_tower = config['model']['vision_tower']
    kwargs = {key: getattr(args, key) for key in (
        'train_mask_decoder', 'out_dim', 'ce_loss_weight', 'dice_loss_weight',
        'bce_loss_weight', 'cls_loss_weight', 'seg_token_idx', 'cls_token_idx',
        'real_token_idx', 'fake_token_idx', 'token_strategy',
        'per_sample_text_loss_normalization', 'vision_pretrained', 'vision_tower',
        'use_mm_start_end', 'mm_vision_select_layer', 'pretrain_mm_mlp_adapter',
        'tune_mm_mlp_adapter', 'freeze_mm_mlp_adapter', 'mm_use_im_start_end',
        'with_region', 'bbox_token_idx', 'eop_token_idx', 'bop_token_idx')}
    kwargs['num_level_reg_features'] = 4
    # The complete checkpoint supplies every non-CLIP tensor; no base model download.
    with init_empty_weights(include_buffers=False), no_init_weights():
        model = GLaMMForCausalLM(language, **kwargs)
    model = backbone.prepare_model_for_training(model, tokenizer, args)
    model.enable_rine_conditioning()
    model.to(dtype=torch.bfloat16)
    load_module(directory, model, 'backbone.', device)
    model.to(device=device)
    model.gradient_checkpointing_disable()
    model.rine_conditioner.rine.float().eval()
    model.get_base_model().generation_config = GenerationConfig.from_dict(manifest['generation_config'])
    return model, tokenizer, {'format_version': 2}
