"""
train.py - GLaMM Model Training on Mixed Datasets

Trains the GLaMM model using Caption, Region, and Segmentation datasets with a random sampling approach. This method
is crucial for developing a versatile model capable of handling diverse applications effectively.
"""
import os
import sys
import time
import tqdm
import random
import json
import torch
import argparse
import deepspeed
import numpy as np
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib

from dataset.dataset import custom_collate_fn, HybridSegDataset, HybridRegDataset, HybridCapDataset
from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics import (
    build_forensics_prediction_records,
    compute_binary_mask_metrics,
    compute_empty_prediction_metrics,
    summarize_detection_records,
    summarize_localization_records,
)
from tools.utils import (DEFAULT_CLS_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter,
                         DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN, dict_to_cuda, Summary, intersectionAndUnionGPU)

from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset
from dataset.region_datasets.RefCOCO_VG_Region_ds import RefCocoGRegDataset, VisualGenomeRegDataset
from dataset.caption_datasets.COCO_Caption_ds import CocoCapDataset
from dataset.gcg_datasets.GranDf_gcg_ds import OpenPsgGCGDataset, Flickr30kGCGDataset, RefCOCOgGCGDataset


def parse_args(args):
    parser = argparse.ArgumentParser(description="GLaMM Model Training")

    # Model-specific settings
    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--precision", default='bf16', type=str)

    # Dataset settings
    parser.add_argument("--use_cap_data", action="store_true", help="Use caption data")
    parser.add_argument("--use_reg_data", action="store_true", help="Use region data")
    parser.add_argument("--use_segm_data", action="store_true", help="Use segmentation data")
    parser.add_argument("--weight_cap", default=0.15, type=float, help="Sampling weight for caption data")
    parser.add_argument("--weight_reg", default=0.40, type=float, help="Sampling weight for region data")
    parser.add_argument("--weight_segm", default=0.45, type=float, help="Sampling weight for segmentation data")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--seg_dataset", default="Semantic_Segm||Refer_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG",
                        type=str, help="Choose from: Semantic_Segm, Refer_Segm, RefCoco_GCG, GranDf_GCG, PSG_GCG, Flickr_GCG, GrandRefer_Segm")
    parser.add_argument("--segm_sample_rates", default="5,4,3,3,3,1", type=str)
    parser.add_argument("--reg_dataset", default="RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg",
                        type=str, help="Choose from: RefCoco_Reg, RefCocoG_Reg, RefCocoP_Reg, VisGen_Reg, Flickr_Reg, GrandRefer_Reg")
    parser.add_argument("--reg_sample_rates", default="1,1,1,1", type=str)
    parser.add_argument("--cap_dataset", default="CocoCap||LLaVaInstruct", type=str,
                        help="Choose from: CocoCap, LLaVaInstruct, GrandCaptionDataset")
    parser.add_argument("--cap_sample_rates", default="1,1", type=str)
    parser.add_argument("--semantic_segm_data", default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary", type=str)
    parser.add_argument("--refer_segm_data", default="refcoco||refcoco+||refcocog||refclef", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--use_forensics_data", action="store_true", help="Use frozen unified Real/Fake manifests")
    parser.add_argument("--weight_forensics", default=1.0, type=float)
    parser.add_argument("--forensics_manifest_dir", default="outputs/data_audits/unified_forensics_split_v1")
    parser.add_argument("--forensics_datasets_root", default="datasets")
    parser.add_argument("--synthscars_root", default="datasets/SynthScars")
    parser.add_argument("--token_strategy", default="fixed_cls_query",
                        choices=["generated_cls", "fixed_cls_query"])

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--lora_lr", default=None, type=float)
    parser.add_argument("--classification_head_lr", default=None, type=float)
    parser.add_argument("--text_hidden_fcs_lr", default=None, type=float)
    parser.add_argument("--mask_decoder_lr", default=None, type=float)
    parser.add_argument("--embedding_lr", default=None, type=float)
    parser.add_argument("--lm_head_lr", default=None, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--cls_loss_weight", default=1.0, type=float)
    parser.add_argument("--per_sample_text_loss_normalization", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--freeze_region_encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    # Evaluation settings
    parser.add_argument("--val_dataset", default="CocoCapVal|RefCOCOgRegVal|RefCOCOgSegmVal", type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal")
    parser.add_argument("--mask_validation", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./output", type=str)
    parser.add_argument("--exp_name", default="GlamFinetuneOS", type=str)

    return parser.parse_args(args)


def initialize_environment(args):
    """ Set up logging and model directories. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        return SummaryWriter(args.log_dir)
    return None


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens([DEFAULT_CLS_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN], special_tokens=True)

    if not args.pretrained:
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        # modifications specific for regions
        reg_tokens = ['<bbox>', '<point>']
        # Adding special tokens for pixel grounding
        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)

    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.cls_token_idx = tokenizer(DEFAULT_CLS_TOKEN, add_special_tokens=False).input_ids[0]
    real_ids = tokenizer(DEFAULT_REAL_TOKEN, add_special_tokens=False).input_ids
    fake_ids = tokenizer(DEFAULT_FAKE_TOKEN, add_special_tokens=False).input_ids
    if len(real_ids) != 1 or len(fake_ids) != 1:
        raise ValueError("[REAL] and [FAKE] must each encode as exactly one token")
    args.real_token_idx = real_ids[0]
    args.fake_token_idx = fake_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]
    print(
        "Forensic special token ids:",
        {"[CLS]": args.cls_token_idx, "[REAL]": args.real_token_idx,
         "[FAKE]": args.fake_token_idx, "[SEG]": args.seg_token_idx},
        "token_strategy=", args.token_strategy,
    )

    return tokenizer


def initialize_model(args, tokenizer):
    """ Initialize the GLaMM model. """
    model_args = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
                   "cls_loss_weight", "seg_token_idx", "cls_token_idx", "real_token_idx", "fake_token_idx",
                   "token_strategy", "per_sample_text_loss_normalization", "vision_pretrained", "vision_tower", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end",
                   "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}
    model_args["num_level_reg_features"] = 4

    model = GLaMMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args
    )
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def prepare_model_for_training(model, tokenizer, args):
    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Initialize vision tower
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)

    # Initialize GLaMM model and adjust requires_grad
    if not args.pretrained:
        model.get_model().initialize_glamm_model(model.get_model().config)
    else:
        for param in model.get_model().grounding_encoder.parameters():
            param.requires_grad = False
        if model.get_model().config.train_mask_decoder:
            model.get_model().grounding_encoder.mask_decoder.train()
            for param in model.get_model().grounding_encoder.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        model.get_model().text_hidden_fcs.train()
        for param in model.get_model().text_hidden_fcs.parameters():
            param.requires_grad = True

    # Set requires_grad for vision tower and mm projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    if lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Configure LoRA if applicable
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Make certain modules trainable
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
        for name, module in model.named_modules():
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["grounding_encoder", "vision_tower", "mm_projector", "text_hidden_fcs"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)

    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    return lora_config


def set_trainable_modules(model):
    """ Make specified modules in the model trainable. """
    trainable_modules = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "region_encoder",
                         "classification_head"]
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + "---- Total parameters: ----{}".format(total_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters: ----{}".format(trainable_params) + '\033[0m')

    count_parameters(model)


def freeze_unused_region_encoder(model):
    """Remove the unused bbox region encoder from Unified Forensics optimization."""
    matched = 0
    for name, parameter in model.named_parameters():
        if "region_encoder" in name:
            parameter.requires_grad = False
            matched += parameter.numel()
    if matched == 0:
        raise ValueError("No region_encoder parameters were found to freeze")
    return matched


def build_optimizer_parameter_groups(model, args):
    """Build disjoint named groups for the finalized Unified baseline policy."""
    lr_values = {
        "lora": args.lora_lr if args.lora_lr is not None else args.lr,
        "classification_head": (
            args.classification_head_lr if args.classification_head_lr is not None else args.lr
        ),
        "text_hidden_fcs": args.text_hidden_fcs_lr if args.text_hidden_fcs_lr is not None else args.lr,
        "mask_decoder": args.mask_decoder_lr if args.mask_decoder_lr is not None else args.lr,
        "embeddings": args.embedding_lr if args.embedding_lr is not None else args.lr,
        "lm_head": args.lm_head_lr if args.lm_head_lr is not None else args.lr,
        "forensic_projector": args.lr,
    }
    grouped = {name: [] for name in lr_values}
    unknown = []
    seen = set()
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in parameter_name:
            group_name = "lora"
        elif "classification_head" in parameter_name:
            group_name = "classification_head"
        elif "forensic_projector" in parameter_name:
            group_name = "forensic_projector"
        elif "text_hidden_fcs" in parameter_name:
            group_name = "text_hidden_fcs"
        elif "grounding_encoder.mask_decoder" in parameter_name:
            group_name = "mask_decoder"
        elif "embed_tokens" in parameter_name:
            group_name = "embeddings"
        elif "lm_head" in parameter_name:
            group_name = "lm_head"
        else:
            unknown.append(parameter_name)
            continue
        if id(parameter) in seen:
            raise ValueError(f"Parameter occurs in multiple optimizer groups: {parameter_name}")
        seen.add(id(parameter))
        grouped[group_name].append(parameter)
    if unknown:
        raise ValueError(f"Unclassified trainable parameters: {unknown[:20]}")
    output = []
    for group_name, parameters in grouped.items():
        if not parameters and group_name != "forensic_projector":
            raise ValueError(f"Final optimizer group is empty: {group_name}")
        if not parameters:
            continue
        output.append({"name": group_name, "params": parameters, "lr": float(lr_values[group_name])})
    return output


def initialize_datasets_and_loaders(args, tokenizer):
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # Common dataset arguments
    common_ds_args = {"dataset_dir": args.dataset_dir, "tokenizer": tokenizer,
                      "global_image_encoder": args.vision_tower,
                      "epoch_samples": args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
                      "precision": args.precision, "image_size": args.image_size,
                      "num_classes_per_sample": args.num_classes_per_sample}

    # Training datasets
    cap_train_dataset = HybridCapDataset(
        **common_ds_args, dataset=args.cap_dataset, sample_rate=[float(x) for x in args.cap_sample_rates.split(",")],
        batch_size=args.batch_size, ) if args.use_cap_data else None
    reg_train_dataset = HybridRegDataset(
        **common_ds_args, dataset=args.reg_dataset, sample_rate=[float(x) for x in args.reg_sample_rates.split(",")],
        batch_size=args.batch_size, ) if args.use_reg_data else None
    seg_train_dataset = HybridSegDataset(
        **common_ds_args, dataset=args.seg_dataset, sample_rate=[float(x) for x in args.segm_sample_rates.split(",")],
        semantic_segm_data=args.semantic_segm_data, refer_segm_data=args.refer_segm_data,
        batch_size=args.batch_size, ) if args.use_segm_data else None

    # Validation datasets
    val_datasets = []
    if not args.no_eval:
        val_dataset_classes = {'CocoCapVal': CocoCapDataset,
                               'RefCOCOgRegVal': RefCocoGRegDataset,
                               'VisGenomeRegVal': VisualGenomeRegDataset,
                               'RefCOCOgSegmVal': ReferSegmDataset,
                               'PsgGCGVal': OpenPsgGCGDataset,
                               'RefCocoGCGVal': RefCOCOgGCGDataset,
                               'FlickrGCGVal': Flickr30kGCGDataset,
                               }
        for val_dataset_name in args.val_dataset.split('|'):
            val_dataset_class = val_dataset_classes.get(val_dataset_name)
            if val_dataset_class:
                if val_dataset_class == ReferSegmDataset:
                    # Modify this if other datasets in refer_segm_data need to be included in val
                    refer_segm_data = 'refcocog'
                    all_datasets = refer_segm_data.split("||")
                    for d in all_datasets:
                        val_dataset_class = val_dataset_class(
                            **common_ds_args, validation=True, refer_segm_data=d, split='val'
                        )
                        val_dataset_class._set_len(len(val_dataset_class.refer_segm_data[d]['images']))
                        val_datasets.append(val_dataset_class)
                else:
                    val_datasets.append(val_dataset_class(**common_ds_args, validation=True))

    forensics_train_dataset = None
    if args.use_forensics_data:
        forensics_train_dataset = UnifiedForensicsDataset(
            args.forensics_manifest_dir, tokenizer, args.vision_tower, split="train",
            datasets_root=args.forensics_datasets_root, synthscars_root=args.synthscars_root,
            image_size=args.image_size,
        )
        if not args.no_eval:
            val_datasets.append(UnifiedForensicsDataset(
                args.forensics_manifest_dir, tokenizer, args.vision_tower, split="val",
                datasets_root=args.forensics_datasets_root, synthscars_root=args.synthscars_root,
                image_size=args.image_size,
            ))

    return cap_train_dataset, reg_train_dataset, seg_train_dataset, forensics_train_dataset, val_datasets


def setup_data_loaders(args, cap_train_dataset, reg_train_dataset, seg_train_dataset, forensics_train_dataset,
                       val_datasets, tokenizer):
    sampler_args = {"shuffle": False, "drop_last": False}
    train_loader_args = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.workers,
                         "pin_memory": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers,
                       "pin_memory": False}
    collate_fn_args_train = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=False, token_strategy=args.token_strategy
    )
    inference_mode = args.mask_validation
    collate_fn_args_val = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=inference_mode, token_strategy=args.token_strategy
    )

    # Training loaders
    cap_train_loader = torch.utils.data.DataLoader(
        cap_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            cap_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if cap_train_dataset is not None else None
    reg_train_loader = torch.utils.data.DataLoader(
        reg_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            reg_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if reg_train_dataset is not None else None
    seg_train_loader = torch.utils.data.DataLoader(
        seg_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            seg_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if seg_train_dataset is not None else None
    forensics_train_loader = torch.utils.data.DataLoader(
        forensics_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            forensics_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if forensics_train_dataset is not None else None

    # Validation loader
    val_loader = None
    if val_datasets:
        combined_val_datasets = ConcatDataset(val_datasets)
        val_loader = torch.utils.data.DataLoader(
            combined_val_datasets, **val_loader_args, collate_fn=collate_fn_args_val,
            sampler=torch.utils.data.distributed.DistributedSampler(combined_val_datasets, **sampler_args), )

    return cap_train_loader, reg_train_loader, seg_train_loader, forensics_train_loader, val_loader


def initialize_deepspeed(model, tokenizer, args):
    optimizer_groups = build_optimizer_parameter_groups(model, args)
    group_lrs = [group["lr"] for group in optimizer_groups]
    ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.grad_accumulation_steps,
                 "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.0,
                                                           "betas": (args.beta1, args.beta2)}},
                 "scheduler": {"type": "WarmupDecayLR",
                                          "params": {"total_num_steps": args.epochs * args.steps_per_epoch,
                                          "warmup_min_lr": [0.0] * len(group_lrs),
                                          "warmup_max_lr": group_lrs,
                                          "warmup_num_steps": 100, "warmup_type": "linear"}},
                 "fp16": {"enabled": args.precision == "fp16"}, "bf16": {"enabled": args.precision == "bf16"},
                 "gradient_clipping": 1.0,
                 "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                                       "reduce_scatter": True, "reduce_bucket_size": 5e8,
                                       "allgather_bucket_size": 5e8}, }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=optimizer_groups, collate_fn=partial(
            custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank, token_strategy=args.token_strategy
        ), config=ds_config
    )

    return model_engine, optimizer, scheduler


def resume_training_from_checkpoint(model_engine, args):
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")


def main(args):
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    model = prepare_model_for_training(model, tokenizer, args)

    model_engine, optimizer, scheduler = initialize_deepspeed(model, tokenizer, args)
    resume_training_from_checkpoint(model_engine, args)

    cap_train_dataset, reg_train_dataset, seg_train_dataset, forensics_train_dataset, val_datasets = (
        initialize_datasets_and_loaders(args, tokenizer))
    cap_train_loader, reg_train_loader, seg_train_loader, forensics_train_loader, val_loader = (
        setup_data_loaders(args, cap_train_dataset, reg_train_dataset, seg_train_dataset,
                           forensics_train_dataset, val_datasets, tokenizer))

    # Determine active datasets and their weights
    active_dataloaders = []
    weights = []

    if args.use_cap_data:
        active_dataloaders.append(('cap', cap_train_loader))
        weights.append(args.weight_cap)
    if args.use_reg_data:
        active_dataloaders.append(('reg', reg_train_loader))
        weights.append(args.weight_reg)
    if args.use_segm_data:
        active_dataloaders.append(('seg', seg_train_loader))
        weights.append(args.weight_segm)
    if args.use_forensics_data:
        active_dataloaders.append(('forensics', forensics_train_loader))
        weights.append(args.weight_forensics)

    # Assert that at least one dataset is active
    assert active_dataloaders, "Error: At least one dataset (segm, reg, or cap) must be active."

    dataset_iters = {'cap': iter(cap_train_loader) if args.use_cap_data else None,
                     'reg': iter(reg_train_loader) if args.use_reg_data else None,
                     'seg': iter(seg_train_loader) if args.use_segm_data else None,
                     'forensics': iter(forensics_train_loader) if args.use_forensics_data else None, }

    writer = initialize_environment(args)
    if args.local_rank == 0:
        tokenizer.save_pretrained(os.path.join(args.log_dir, "tokenizer"))

    if args.eval_only:
        cur_val_loss = validate_model_performance(val_loader, model_engine, 0, writer, args)[0]
        exit()

    epoch_seeds = [random.randint(0, 100000) for _ in range(args.epochs)]
    dataset_choices = [idx for idx, _ in enumerate(active_dataloaders)]

    best_giou, best_ciou, best_val_loss = 0.0, 0.0, np.inf
    for epoch in range(args.start_epoch, args.epochs):
        random.seed(epoch_seeds[epoch])

        step_choices = random.choices(dataset_choices, weights=weights, k=args.steps_per_epoch)

        dataset_iters = train(
            active_dataloaders, model_engine, epoch, scheduler, writer, dataset_iters, args, step_choices
        )

        if args.mask_validation:
            giou, ciou = validate_model_performance(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_giou
            best_giou = max(giou, best_giou)
            best_ciou = ciou if is_best else best_ciou
            if args.local_rank == 0:  # Log the progress
                print(f"Epoch: {epoch}, giou: {giou}, ciou: {ciou}, best_giou: {best_giou}, best_ciou: {best_ciou}")
            save_checkpoint(model_engine, args, epoch, 'giou-ciou', f"{giou:.4f}-{ciou:.4f}", is_best)
        else:
            cur_val_loss = validate_model_performance(val_loader, model_engine, epoch, writer, args)
            is_best, best_val_loss = update_best_validation_total_loss(cur_val_loss, best_val_loss)
            if args.local_rank == 0:  # Log the progress
                print(f"Epoch: {epoch}, Current Validation Loss: {cur_val_loss:.4f}, Best Validation Loss: {best_val_loss:}")
            save_checkpoint(model_engine, args, epoch, 'loss', f"{cur_val_loss:.4f}", is_best)


def save_checkpoint(model_engine, args, epoch, metric_name, metric_value, is_best):
    """Always save ``last`` and additionally refresh ``best`` when selected."""
    destinations = ["ckpt_model_last_epoch"]
    if is_best:
        destinations.append("ckpt_model_best")
    for save_dir_name in destinations:
        save_dir = os.path.join(args.log_dir, save_dir_name)
        if args.local_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_filename = f"epoch_{epoch}_val_{metric_name}_{metric_value}.pth"
            torch.save(
                {"epoch": epoch, f"val_{metric_name}": metric_value},
                os.path.join(save_dir, ckpt_filename),
            )
        torch.distributed.barrier()
        model_engine.save_checkpoint(save_dir)


def train(active_datasets, model, epoch, scheduler, writer, dataset_iters, args, step_choices):
    """Main training loop."""

    def get_next_input(iterator, data_loader):
        """Retrieve next input from the iterator, or reinitialize if necessary."""
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    def log_progress():
        """Log training progress."""
        if global_step % args.print_freq == 0:
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                for key, tracker in trackers.items():
                    writer.add_scalar(f"train/{key}", tracker.avg, global_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)

            for tracker in trackers.values():
                tracker.reset()

    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    trackers = {"loss": AverageMeter("Loss", ":.4f"),
                "ce_loss": AverageMeter("CeLoss", ":.4f"),
                "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                "mask_loss": AverageMeter("MaskLoss", ":.4f"),
                "cls_loss": AverageMeter("ClsLoss", ":.4f"),
                "seg_missing_pred_count": AverageMeter("SegMissingPred", ":.1f"),
                "seg_count_mismatch_count": AverageMeter("SegCountMismatch", ":.1f"),
                "seg_unexpected_pred_count": AverageMeter("SegUnexpectedPred", ":.1f")}
    progress = ProgressMeter(args.steps_per_epoch, list(trackers.values()), prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            # Select data loader based on step choice
            dataset_type, data_loader = active_datasets[step_choices[global_step]]
            data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
            dataset_iters[dataset_type] = new_iter

            data_time.update(time.time() - end)
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                if data_batch[key] is not None:
                    data_batch[key] = data_batch[key].bfloat16()

            output_dict = model(**data_batch)

            # Update training metrics
            for key, tracker in trackers.items():
                if key in output_dict:
                    tracker.update(output_dict[key].item(), data_batch["global_enc_images"].size(0))

            model.backward(output_dict["loss"])
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return dataset_iters


def validate_model_performance(validation_loader, training_model, current_epoch, tensorboard_writer, args):
    if args.mask_validation:
        # For use with only segmentation/GCG type datasets
        trackers = {"intersection": AverageMeter("Intersec", ":.4f", Summary.SUM),
                    "union": AverageMeter("Union", ":.4f", Summary.SUM),
                    "gIoU": AverageMeter("gIoU", ":.4f", Summary.SUM)}

        training_model.eval()
        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()
            if os.environ.get("GLAMM_PRESERVE_CUDA_CACHE", "0") != "1":
                torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                results = training_model(**data_batch)

            predictions = results["pred_masks"]
            gt_masks = results["gt_masks"][0].int()
            # Note: An error at this line may suggest that the dataset used for validation does not support
            # segmentation tasks. Ensure that the dataset is appropriate for segmentation analysis.
            predicted_masks = (predictions[0] > 0).int()
            assert len(predictions) == 1

            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            for target, prediction in zip(gt_masks, predicted_masks):
                intersect, union_, _ = intersectionAndUnionGPU(
                    prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                )
                intersection += intersect
                union += union_
                accuracy_iou += intersect / (union_ + 1e-5)
                # handles no-object targets
                accuracy_iou[union_ == 0] += 1.0

            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            accuracy_iou = accuracy_iou.cpu().numpy() / gt_masks.shape[0]
            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            trackers["gIoU"].update(accuracy_iou, n=gt_masks.shape[0])

        for meter in trackers.values():
            meter.all_reduce()

        iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
        class_iou = iou_per_class[1]
        global_iou = trackers["gIoU"].avg[1]

        if args.local_rank == 0:
            tensorboard_writer.add_scalar("val/giou", global_iou, current_epoch)
            tensorboard_writer.add_scalar("val/ciou", class_iou, current_epoch)
            print("giou: {:.4f}, ciou: {:.4f}".format(global_iou, class_iou))

        return global_iou, class_iou
    else:
        # Initializing performance trackers
        trackers = {"loss": AverageMeter("Loss", ":.4f"), "ce_loss": AverageMeter("CeLoss", ":.4f"),
                    "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                    "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                    "mask_loss": AverageMeter("MaskLoss", ":.4f"),
                    "cls_loss": AverageMeter("ClsLoss", ":.4f")}

        # Prepare model for validation phase
        # Hack to get the loss
        training_model.train()
        forensics_records = []
        tf_localization_records = []

        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                if data_batch[key] is not None:
                    data_batch[key] = data_batch[key].bfloat16()
            if os.environ.get("GLAMM_PRESERVE_CUDA_CACHE", "0") != "1":
                torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                predictions = training_model(**data_batch)
            cls_labels = data_batch.get("cls_labels")
            if cls_labels is not None and cls_labels.ge(0).all():
                forensics_records.extend(build_forensics_prediction_records(data_batch, predictions))
                pred_masks = predictions.get("pred_masks")
                if pred_masks is not None:
                    for index, is_valid in enumerate(data_batch["seg_valid"].detach().cpu().tolist()):
                        if not is_valid:
                            continue
                        gt_mask = data_batch["masks_list"][index]
                        pred_mask = pred_masks[index]
                        has_prediction = pred_mask is not None and pred_mask.shape[0] > 0
                        mask_metrics = (
                            compute_binary_mask_metrics(pred_mask, gt_mask)
                            if has_prediction else compute_empty_prediction_metrics(gt_mask)
                        )
                        tf_localization_records.append({
                            **mask_metrics,
                            "content_type": data_batch["content_categories"][index],
                            "seg_triggered": has_prediction,
                            "has_pred_mask": has_prediction,
                        })
            # Update performance metrics)
            for key, tracker in trackers.items():
                tracker.update(predictions[key].item(), data_batch["global_enc_images"].size(0))

        # Synchronize metrics across processes
        for tracker in trackers.values():
            tracker.all_reduce()
        if args.distributed:
            gathered_records = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_records, forensics_records)
            forensics_records = [record for records in gathered_records for record in records]
            gathered_tf_records = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_tf_records, tf_localization_records)
            tf_localization_records = [record for records in gathered_tf_records for record in records]
        if args.local_rank == 0 and forensics_records:
            prediction_path = os.path.join(args.log_dir, f"forensics_predictions_epoch_{current_epoch}.jsonl")
            with open(prediction_path, "w", encoding="utf-8") as handle:
                for record in forensics_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            detection_metrics = summarize_detection_records(forensics_records)
            tf_metrics = summarize_localization_records(tf_localization_records, autoregressive=False)
            metrics_path = os.path.join(args.log_dir, f"forensics_metrics_epoch_{current_epoch}.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "classification": detection_metrics["classification_head"],
                    "lm_verdict": detection_metrics["lm_verdict"],
                    "cls_lm_agreement": detection_metrics["cls_lm_agreement"],
                    "tf_full_context": tf_metrics,
                }, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            tensorboard_writer.add_scalar(
                "val/classification_accuracy", detection_metrics["classification_head"]["accuracy"],
                current_epoch,
            )
            tensorboard_writer.add_scalar(
                "val/lm_verdict_accuracy", detection_metrics["lm_verdict"]["accuracy"], current_epoch
            )
            tensorboard_writer.add_scalar(
                "val/cls_lm_agreement", detection_metrics["cls_lm_agreement"], current_epoch
            )
            if tf_metrics["mean_iou"] is not None:
                tensorboard_writer.add_scalar("val/tf_mean_iou", tf_metrics["mean_iou"], current_epoch)
                tensorboard_writer.add_scalar("val/tf_global_iou", tf_metrics["global_iou"], current_epoch)
                tensorboard_writer.add_scalar(
                    "val/tf_mean_pixel_f1", tf_metrics["mean_pixel_f1"], current_epoch
                )
                tensorboard_writer.add_scalar(
                    "val/tf_global_pixel_f1", tf_metrics["global_pixel_f1"], current_epoch
                )
        # Primary validation objective must match the complete training loss;
        # CE alone is a diagnostic component and must never masquerade as total.
        avg_val_loss = trackers["loss"].avg
        # Tensorboard logging for primary process
        if args.local_rank == 0:
            tensorboard_writer.add_scalar("val/total_loss", avg_val_loss, current_epoch)

        return avg_val_loss


def validation_total_loss_from_output(output_dict):
    """Return and verify the full text+classification+conditional-mask objective."""
    expected = (
        output_dict["ce_loss"] + output_dict["cls_loss"]
        + output_dict["mask_bce_loss"] + output_dict["mask_dice_loss"]
    )
    if not torch.allclose(output_dict["loss"], expected, atol=1e-6, rtol=1e-6):
        raise ValueError("Model loss does not equal text+cls+BCE+Dice validation objective")
    return output_dict["loss"]


def update_best_validation_total_loss(current_total_loss, best_total_loss):
    """Primary selector: validation total loss only, lower is better."""
    current = float(current_total_loss)
    best = float(best_total_loss)
    return current < best, min(current, best)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
