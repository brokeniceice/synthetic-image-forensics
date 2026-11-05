import torch
import argparse
import transformers
from peft import LoraConfig, get_peft_model
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from tools.utils import DEFAULT_CLS_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN

def parse_args(args):
    parser = argparse.ArgumentParser(description='GLaMM Model Training')
    parser.add_argument('--version', default='MBZUAI/GLaMM-GranD-Pretrained')
    parser.add_argument('--vision_pretrained', default='./checkpoints/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument('--vision-tower', default='openai/clip-vit-large-patch14-336', type=str)
    parser.add_argument('--conv_type', default='llava_v1', type=str, choices=['llava_v1', 'llava_llama_2'])
    parser.add_argument('--tune_mm_mlp_adapter', action='store_true')
    parser.add_argument('--freeze_mm_mlp_adapter', action='store_true')
    parser.add_argument('--mm_use_im_start_end', action='store_true', default=True)
    parser.add_argument('--out_dim', default=256, type=int)
    parser.add_argument('--image_size', default=1024, type=int, help='Image size for grounding image encoder')
    parser.add_argument('--model_max_length', default=1536, type=int)
    parser.add_argument('--lora_target_modules', default='q_proj,v_proj', type=str)
    parser.add_argument('--with_region', action='store_true', default=True)
    parser.add_argument('--mm_vision_select_layer', default=-2, type=int)
    parser.add_argument('--pretrain_mm_mlp_adapter', default='', type=str)
    parser.add_argument('--precision', default='bf16', type=str)
    parser.add_argument('--use_forensics_data', action='store_true', help='Use frozen unified Real/Fake manifests')
    parser.add_argument('--forensics_manifest_dir', default='outputs/data_audits/unified_forensics_split_v1')
    parser.add_argument('--forensics_datasets_root', default='datasets')
    parser.add_argument('--synthscars_root', default='datasets/SynthScars')
    parser.add_argument('--token_strategy', default='fixed_cls_query', choices=['generated_cls', 'fixed_cls_query'])
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--lora_lr', default=None, type=float)
    parser.add_argument('--classification_head_lr', default=None, type=float)
    parser.add_argument('--text_hidden_fcs_lr', default=None, type=float)
    parser.add_argument('--mask_decoder_lr', default=None, type=float)
    parser.add_argument('--embedding_lr', default=None, type=float)
    parser.add_argument('--lm_head_lr', default=None, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--steps_per_epoch', default=500, type=int)
    parser.add_argument('--batch_size', default=2, type=int, help='batch size per device per step')
    parser.add_argument('--grad_accumulation_steps', default=10, type=int)
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--ce_loss_weight', default=1.0, type=float)
    parser.add_argument('--dice_loss_weight', default=0.5, type=float)
    parser.add_argument('--bce_loss_weight', default=2.0, type=float)
    parser.add_argument('--cls_loss_weight', default=1.0, type=float)
    parser.add_argument('--per_sample_text_loss_normalization', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)
    parser.add_argument('--train_mask_decoder', action='store_true', default=True)
    parser.add_argument('--freeze_region_encoder', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use_mm_start_end', action='store_true', default=True)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank')
    parser.add_argument('--no_eval', action='store_true')
    return parser.parse_args(args)

def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.version, model_max_length=args.model_max_length, padding_side='right', use_fast=False)
    print('\x1b[92m' + '---- Initialized tokenizer from: {} ----'.format(args.version) + '\x1b[0m')
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens([DEFAULT_CLS_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN], special_tokens=True)
    if not args.pretrained:
        if args.use_mm_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        reg_tokens = ['<bbox>', '<point>']
        segmentation_tokens = ['[SEG]']
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)
    args.bbox_token_idx = tokenizer('<bbox>', add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer('[SEG]', add_special_tokens=False).input_ids[0]
    args.cls_token_idx = tokenizer(DEFAULT_CLS_TOKEN, add_special_tokens=False).input_ids[0]
    real_ids = tokenizer(DEFAULT_REAL_TOKEN, add_special_tokens=False).input_ids
    fake_ids = tokenizer(DEFAULT_FAKE_TOKEN, add_special_tokens=False).input_ids
    if len(real_ids) != 1 or len(fake_ids) != 1:
        raise ValueError('[REAL] and [FAKE] must each encode as exactly one token')
    args.real_token_idx = real_ids[0]
    args.fake_token_idx = fake_ids[0]
    args.bop_token_idx = tokenizer('<p>', add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer('</p>', add_special_tokens=False).input_ids[0]
    print('Forensic special token ids:', {'[CLS]': args.cls_token_idx, '[REAL]': args.real_token_idx, '[FAKE]': args.fake_token_idx, '[SEG]': args.seg_token_idx}, 'token_strategy=', args.token_strategy)
    return tokenizer

def initialize_model(args, tokenizer):
    """ Initialize the GLaMM model. """
    model_args = {k: getattr(args, k) for k in ['train_mask_decoder', 'out_dim', 'ce_loss_weight', 'dice_loss_weight', 'bce_loss_weight', 'cls_loss_weight', 'seg_token_idx', 'cls_token_idx', 'real_token_idx', 'fake_token_idx', 'token_strategy', 'per_sample_text_loss_normalization', 'vision_pretrained', 'vision_tower', 'use_mm_start_end', 'mm_vision_select_layer', 'pretrain_mm_mlp_adapter', 'tune_mm_mlp_adapter', 'freeze_mm_mlp_adapter', 'mm_use_im_start_end', 'with_region', 'bbox_token_idx', 'eop_token_idx', 'bop_token_idx']}
    model_args['num_level_reg_features'] = 4
    model = GLaMMForCausalLM.from_pretrained(args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args)
    print('\x1b[92m' + '---- Initialized model from: {} ----'.format(args.version) + '\x1b[0m')
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model

def prepare_model_for_training(model, tokenizer, args):
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    print('\x1b[92m' + '---- Initialized Global Image Encoder (vision tower) from: {} ----'.format(args.vision_tower) + '\x1b[0m')
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)
    if not args.pretrained:
        model.get_model().initialize_glamm_model(model.get_model().config)
    else:
        for param in model.get_model().grounding_encoder.parameters():
            param.requires_grad = False
        if model.get_model().config.train_mask_decoder:
            model.get_model().grounding_encoder.mask_decoder.train()
            for param in model.get_model().grounding_encoder.mask_decoder.parameters():
                param.requires_grad = True
        model.get_model().text_hidden_fcs.train()
        for param in model.get_model().text_hidden_fcs.parameters():
            param.requires_grad = True
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False
    lora_r = args.lora_r
    if lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)
    model.resize_token_embeddings(len(tokenizer))
    set_trainable_modules(model)
    if args.freeze_region_encoder:
        freeze_unused_region_encoder(model)
    return model

def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """

    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for (name, module) in model.named_modules():
            if isinstance(module, linear_cls) and all((x not in name for x in ['grounding_encoder', 'vision_tower', 'mm_projector', 'text_hidden_fcs'])) and any((x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))
    lora_target_modules = args.lora_target_modules.split(',')
    lora_module_names = find_proj_layers(model, lora_target_modules)
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout, bias='none', task_type='CAUSAL_LM')
    return lora_config

def set_trainable_modules(model):
    """ Make specified modules in the model trainable. """
    trainable_modules = ['lm_head', 'embed_tokens', 'mask_decoder', 'text_hidden_fcs', 'region_encoder', 'classification_head']
    for (name, param) in model.named_parameters():
        if any((module in name for module in trainable_modules)):
            print(f'Making trainable: {name}, Shape: {param.shape}')
            param.requires_grad = True

    def count_parameters(model):
        total_params = sum((p.numel() for p in model.parameters()))
        trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
        print('\x1b[92m' + '---- Total parameters: ----{}'.format(total_params) + '\x1b[0m')
        print('\x1b[92m' + '---- Trainable parameters: ----{}'.format(trainable_params) + '\x1b[0m')
    count_parameters(model)

def freeze_unused_region_encoder(model):
    """Remove the unused bbox region encoder from Unified Forensics optimization."""
    matched = 0
    for (name, parameter) in model.named_parameters():
        if 'region_encoder' in name:
            parameter.requires_grad = False
            matched += parameter.numel()
    if matched == 0:
        raise ValueError('No region_encoder parameters were found to freeze')
    return matched

def build_optimizer_parameter_groups(model, args):
    """Build disjoint named groups for the finalized Unified baseline policy."""
    lr_values = {'lora': args.lora_lr if args.lora_lr is not None else args.lr, 'classification_head': args.classification_head_lr if args.classification_head_lr is not None else args.lr, 'text_hidden_fcs': args.text_hidden_fcs_lr if args.text_hidden_fcs_lr is not None else args.lr, 'mask_decoder': args.mask_decoder_lr if args.mask_decoder_lr is not None else args.lr, 'embeddings': args.embedding_lr if args.embedding_lr is not None else args.lr, 'lm_head': args.lm_head_lr if args.lm_head_lr is not None else args.lr, 'forensic_projector': args.lr}
    grouped = {name: [] for name in lr_values}
    unknown = []
    seen = set()
    for (parameter_name, parameter) in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if 'lora_' in parameter_name:
            group_name = 'lora'
        elif 'classification_head' in parameter_name:
            group_name = 'classification_head'
        elif 'forensic_projector' in parameter_name:
            group_name = 'forensic_projector'
        elif 'text_hidden_fcs' in parameter_name:
            group_name = 'text_hidden_fcs'
        elif 'grounding_encoder.mask_decoder' in parameter_name:
            group_name = 'mask_decoder'
        elif 'embed_tokens' in parameter_name:
            group_name = 'embeddings'
        elif 'lm_head' in parameter_name:
            group_name = 'lm_head'
        else:
            unknown.append(parameter_name)
            continue
        if id(parameter) in seen:
            raise ValueError(f'Parameter occurs in multiple optimizer groups: {parameter_name}')
        seen.add(id(parameter))
        grouped[group_name].append(parameter)
    if unknown:
        raise ValueError(f'Unclassified trainable parameters: {unknown[:20]}')
    output = []
    for (group_name, parameters) in grouped.items():
        if not parameters and group_name != 'forensic_projector':
            raise ValueError(f'Final optimizer group is empty: {group_name}')
        if not parameters:
            continue
        output.append({'name': group_name, 'params': parameters, 'lr': float(lr_values[group_name])})
    return output

def validation_total_loss_from_output(output_dict):
    """Return and verify the full text+classification+conditional-mask objective."""
    expected = output_dict['ce_loss'] + output_dict['cls_loss'] + output_dict['mask_bce_loss'] + output_dict['mask_dice_loss']
    if not torch.allclose(output_dict['loss'], expected, atol=1e-06, rtol=1e-06):
        raise ValueError('Model loss does not equal text+cls+BCE+Dice validation objective')
    return output_dict['loss']

def update_best_validation_total_loss(current_total_loss, best_total_loss):
    """Primary selector: validation total loss only, lower is better."""
    current = float(current_total_loss)
    best = float(best_total_loss)
    return (current < best, min(current, best))
