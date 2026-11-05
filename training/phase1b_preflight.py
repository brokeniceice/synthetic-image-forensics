#!/usr/bin/env python3
"""Run the bounded Phase 1B unified-baseline training preflight."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train as glamm_train
from dataset.dataset import custom_collate_fn
from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics import (
    evaluate_detection,
    evaluate_gt_fake_generation_localization,
    evaluate_joint_localization,
    evaluate_teacher_forced_full_context,
    write_evaluation_outputs,
)
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from tools.utils import (
    DEFAULT_CLS_TOKEN,
    DEFAULT_FAKE_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_REAL_TOKEN,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase 1B 64-sample controlled overfit")
    parser.add_argument("--config", default="configs/phase1b_overfit.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eval-only-base", action="store_true")
    parser.add_argument("--eval-only-checkpoint", default=None)
    return parser.parse_args(argv)


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def numeric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {key: None for key in ("min", "mean", "median", "std", "p90", "p95", "p99", "max")}
    return {
        "min": float(values.min()), "mean": float(values.mean()), "median": float(np.median(values)),
        "std": float(values.std()), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def select_balanced_subset(rows, config):
    seed = int(config["experiment"]["seed"])
    rng = random.Random(seed)
    groups = [str(value).lower() for value in config["data"]["content_groups"]]
    per_group = {
        "real": int(config["data"]["num_real"]) // len(groups),
        "fake": int(config["data"]["num_fake"]) // len(groups),
    }
    selected = {domain: {} for domain in ("real", "fake")}
    for domain in ("real", "fake"):
        for group in groups:
            candidates = [
                row for row in rows
                if row["forensics_domain"] == domain
                and str(row.get("content_category", "")).lower() == group
            ]
            rng.shuffle(candidates)
            if len(candidates) < per_group[domain]:
                raise ValueError(f"Not enough {domain}/{group}: {len(candidates)}")
            selected[domain][group] = candidates[:per_group[domain]]

    # Alternate Real/Fake and rotate content groups so every two-sample batch
    # is mixed while the complete subset remains category-balanced.
    ordered = []
    for item_index in range(max(per_group.values())):
        for group in groups:
            if item_index < len(selected["real"][group]):
                ordered.extend([selected["real"][group][item_index], selected["fake"][group][item_index]])
    expected = int(config["data"]["num_real"]) + int(config["data"]["num_fake"])
    if len(ordered) != expected or len({row["sample_id"] for row in ordered}) != expected:
        raise AssertionError("Controlled subset size/uniqueness mismatch")
    return ordered


def token_length_audit(rows, tokenizer, model_max_length):
    conv = conversation_lib.default_conversation.copy()
    assistant_prefix = conv.sep + conv.roles[1] + ": "
    cls_prefix = assistant_prefix + DEFAULT_CLS_TOKEN + " "
    truncate_len = model_max_length - 575
    answer_lengths = {"real": [], "fake": []}
    explanation_lengths, fake_sequence_lengths, seg_positions = [], [], []
    truncated_fake = seg_truncated = 0

    for row in rows:
        domain = row["forensics_domain"]
        target = UnifiedForensicsDataset._target(row)
        answer_lengths[domain].append(len(tokenizer(target, add_special_tokens=False).input_ids) + 1)
        if domain != "fake":
            continue
        explanation = " ".join(str(row.get("explanation") or "").split())
        explanation_lengths.append(len(tokenizer(explanation, add_special_tokens=False).input_ids))
        _, conversations = UnifiedForensicsDataset._conversation(row)
        prompt = conversations[0].replace(DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        prompt = prompt.replace(assistant_prefix, cls_prefix)
        ids = tokenizer_image_token(prompt, tokenizer)
        fake_sequence_lengths.append(len(ids))
        seg_id = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        positions = [index for index, value in enumerate(ids) if value == seg_id]
        if len(positions) != 1:
            raise ValueError(f"Expected one [SEG] in {row['sample_id']}, found {positions}")
        seg_positions.append(positions[0])
        if len(ids) > truncate_len:
            truncated_fake += 1
        if positions[0] >= truncate_len:
            seg_truncated += 1

    return {
        "truncate_raw_token_limit": truncate_len,
        "fake_sequence_token_length": numeric_summary(fake_sequence_lengths),
        "fake_explanation_token_length": numeric_summary(explanation_lengths),
        "seg_position": numeric_summary(seg_positions),
        "real_answer_token_length": numeric_summary(answer_lengths["real"]),
        "fake_answer_token_length": numeric_summary(answer_lengths["fake"]),
        "num_fake_affected_by_truncation": truncated_fake,
        "num_fake_with_seg_beyond_naive_cutoff": seg_truncated,
        "num_fake_with_seg_truncated_after_preservation_policy": 0,
        "num_fake": len(fake_sequence_lengths),
    }


def module_group(name):
    if "lora_" in name:
        return "LoRA"
    if "vision_tower" in name:
        return "vision_tower"
    if "mm_projector" in name:
        return "mm_projector"
    if "region_encoder" in name:
        return "region_encoder"
    if "grounding_encoder.mask_decoder" in name:
        return "SAM_mask_decoder"
    if "grounding_encoder" in name:
        return "grounding_encoder"
    if "text_hidden_fcs" in name:
        return "text_hidden_fcs"
    if "classification_head" in name:
        return "classification_head"
    if "embed_tokens" in name:
        return "token_embeddings"
    if "lm_head" in name:
        return "lm_head"
    if ".layers." in name or ".norm." in name:
        return "LLM_base"
    return "other"


def parameter_audit(model, optimizer):
    lr_by_parameter = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            lr_by_parameter[id(parameter)] = float(group["lr"])
    groups = collections.defaultdict(lambda: {
        "total_parameters": 0, "trainable_parameters": 0, "frozen_parameters": 0,
        "trainable_tensors": 0, "frozen_tensors": 0, "learning_rates": set(),
    })
    for name, parameter in model.named_parameters():
        info = groups[module_group(name)]
        info["total_parameters"] += parameter.numel()
        key = "trainable_parameters" if parameter.requires_grad else "frozen_parameters"
        tensor_key = "trainable_tensors" if parameter.requires_grad else "frozen_tensors"
        info[key] += parameter.numel()
        info[tensor_key] += 1
        if parameter.requires_grad and id(parameter) in lr_by_parameter:
            info["learning_rates"].add(lr_by_parameter[id(parameter)])
    output = {}
    for name, info in groups.items():
        output[name] = {**info, "learning_rates": sorted(info["learning_rates"])}
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "optimizer_learning_rates": {
            "base": sorted({float(group.get("initial_lr", group["lr"])) for group in optimizer.param_groups}),
            "current_at_audit": sorted({float(group["lr"]) for group in optimizer.param_groups}),
        },
        "modules": dict(sorted(output.items())),
    }


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


def grad_norm_for_names(model, needles):
    squares = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and any(needle in name for needle in needles):
            squares.append(parameter.grad.detach().float().pow(2).sum())
    return float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0


def embedding_row_grad_norm(model, token_id):
    weight = model.get_input_embeddings().weight
    if weight.grad is None:
        return 0.0
    return float(weight.grad[token_id].detach().float().norm().item())


def make_collate(tokenizer):
    return lambda items: custom_collate_fn(
        items, tokenizer=tokenizer, use_mm_start_end=True,
        inference=False, token_strategy="fixed_cls_query",
    )


def loss_isolation_and_gradient_audit(model, dataset, tokenizer, device, dtype):
    real = next(dataset[index] for index in range(len(dataset)) if dataset.rows[index]["forensics_domain"] == "real")
    fake = next(dataset[index] for index in range(len(dataset)) if dataset.rows[index]["forensics_domain"] == "fake")
    collate = make_collate(tokenizer)
    cases = {"real_only": [real], "fake_only": [fake], "mixed": [real, fake]}
    outputs, gradients = {}, {}
    model.eval()
    for case_name, items in cases.items():
        model.zero_grad(set_to_none=True)
        batch = move_batch(collate(items), device, dtype)
        result = model(**batch)
        outputs[case_name] = {
            key: float(result[key].detach().float().cpu())
            for key in ("loss", "ce_loss", "cls_loss", "mask_bce_loss", "mask_dice_loss")
        }
        if case_name != "mixed":
            result["loss"].backward()
            gradients[case_name] = {
                "mask_decoder": grad_norm_for_names(model, ["grounding_encoder.mask_decoder"]),
                "classification_head": grad_norm_for_names(model, ["classification_head"]),
                "text_hidden_fcs": grad_norm_for_names(model, ["text_hidden_fcs"]),
                "embeddings": grad_norm_for_names(model, ["embed_tokens"]),
                "cls_embedding": embedding_row_grad_norm(model, model.cls_token_idx),
                "real_embedding": embedding_row_grad_norm(model, model.real_token_idx),
                "fake_embedding": embedding_row_grad_norm(model, model.fake_token_idx),
            }
    fake_mask = outputs["fake_only"]["mask_bce_loss"] + outputs["fake_only"]["mask_dice_loss"]
    mixed_mask = outputs["mixed"]["mask_bce_loss"] + outputs["mixed"]["mask_dice_loss"]
    end_to_end_relative_difference = abs(fake_mask - mixed_mask) / max(abs(fake_mask), 1e-8)
    gt_probe = fake["masks"].to(device)
    pred_probe = torch.zeros_like(gt_probe, device=device)
    zero_ce = type("Output", (), {"loss": pred_probe.sum() * 0})()
    same_pred_fake = model._compute_loss_components(
        [pred_probe], [gt_probe], zero_ce, torch.tensor([True], device=device)
    )
    same_pred_mixed = model._compute_loss_components(
        [torch.ones_like(pred_probe), pred_probe], [None, gt_probe], zero_ce,
        torch.tensor([False, True], device=device),
    )
    same_pred_difference = abs(
        float(same_pred_fake["mask_loss"].detach().float())
        - float(same_pred_mixed["mask_loss"].detach().float())
    )
    checks = {
        "real_mask_losses_zero": outputs["real_only"]["mask_bce_loss"] == 0
        and outputs["real_only"]["mask_dice_loss"] == 0,
        "fake_mask_losses_positive": outputs["fake_only"]["mask_bce_loss"] > 0
        and outputs["fake_only"]["mask_dice_loss"] > 0,
        "mixed_fake_normalization_equivalent": same_pred_difference < 1e-7,
        "real_mask_decoder_grad_zero": gradients["real_only"]["mask_decoder"] == 0,
        "fake_mask_decoder_grad_positive": gradients["fake_only"]["mask_decoder"] > 0,
        "cls_embedding_grad_positive": gradients["real_only"]["cls_embedding"] > 0
        and gradients["fake_only"]["cls_embedding"] > 0,
        "real_embedding_grad_positive": gradients["real_only"]["real_embedding"] > 0,
        "fake_embedding_grad_positive": gradients["fake_only"]["fake_embedding"] > 0,
    }
    if not all(checks.values()):
        raise AssertionError({"checks": checks, "outputs": outputs, "gradients": gradients})
    model.zero_grad(set_to_none=True)
    return {"losses": outputs, "gradients": gradients, "checks": checks,
            "same_prediction_mask_loss_absolute_difference": same_pred_difference,
            "end_to_end_mixed_vs_fake_relative_difference_due_to_prediction_change":
                end_to_end_relative_difference}


def classification_leakage_audit(model, sample, tokenizer, device, dtype):
    captured = []
    head = model.classification_head
    hook = head.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().float().cpu()))
    model.eval()
    try:
        items = []
        for answer in (
            "[REAL] No identifiable synthetic artifact evidence is detected.",
            "[FAKE] Different future explanation tokens. [SEG]",
        ):
            item = copy.copy(dict(sample))
            item["conversations"] = [GLaMMForensicsBackend._conversation(answer)]
            items.append(item)
        batch = custom_collate_fn(
            items, tokenizer=tokenizer, inference=True, token_strategy="fixed_cls_query"
        )
        cls_positions = batch["input_ids"].eq(model.cls_token_idx).nonzero(as_tuple=False)
        if cls_positions[:, 1].unique().numel() != 1:
            raise AssertionError(f"A/B [CLS] positions differ: {cls_positions.tolist()}")
        batch["grounding_enc_images"] = None
        batch = move_batch(batch, device, dtype)
        with torch.no_grad():
            model(**batch)
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0].shape[0] != 2:
        raise AssertionError(f"Expected one two-row CLS capture, got {[tuple(x.shape) for x in captured]}")
    max_abs_difference = float((captured[0][0] - captured[0][1]).abs().max().item())
    passed = torch.allclose(captured[0][0], captured[0][1], atol=2e-3, rtol=2e-3)
    if not passed:
        raise AssertionError(f"Future-label leakage check failed: max diff {max_abs_difference}")
    return {"passed": bool(passed), "max_abs_difference": max_abs_difference}


def prediction_metrics(output, labels):
    cls = output["cls_pred"].detach()
    lm = output["lm_verdict_pred"].detach()
    labels = labels.detach()
    fake = labels.eq(1)
    return {
        "cls_correct": int(cls.eq(labels).sum().item()),
        "lm_correct": int(lm.eq(labels).sum().item()),
        "agree": int(cls.eq(lm).sum().item()),
        "fake_correct": int((cls.eq(1) & fake).sum().item()),
        "num": int(labels.numel()),
        "num_fake": int(fake.sum().item()),
    }


def run_evaluation(model, tokenizer, dataset, output_root, device, dtype, max_new_tokens):
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=dtype,
        use_mm_start_end=True, max_new_tokens=max_new_tokens,
    )
    evaluators = {
        "detection": evaluate_detection,
        "gt_fake_generate": evaluate_gt_fake_generation_localization,
        "tf_full_context": evaluate_teacher_forced_full_context,
        "joint": evaluate_joint_localization,
    }
    summaries = {}
    model.eval()
    for mode, evaluator in evaluators.items():
        records, metrics = evaluator(dataset, backend)
        write_evaluation_outputs(Path(output_root) / mode, mode, records, metrics)
        summaries[mode] = metrics
    return summaries


def exposure_update(counter, batch):
    for label, source, content in zip(
        batch["cls_labels"].detach().cpu().tolist(), batch["sources"], batch["content_categories"]
    ):
        counter["domain"]["fake" if label == 1 else "real"] += 1
        counter["source"][str(source)] += 1
        counter["content_type"][str(content).lower()] += 1


def serialize_counter(counter):
    return {key: dict(sorted(value.items())) for key, value in counter.items()}


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


def main(argv=None):
    cli = parse_args(argv)
    config_path = Path(cli.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = (REPO_ROOT / config["experiment"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    (output_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]

    train_args = build_train_args(config)
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(train_args)
    token_audit = {}
    for token in (DEFAULT_CLS_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN, "[SEG]"):
        ids = tokenizer(token, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise AssertionError(f"{token} is not a single token: {ids}")
        token_audit[token] = {"token_id": ids[0]}

    manifest = Path(config["data"]["manifest_dir"]) / "train_combined.jsonl"
    rows = read_jsonl(manifest)
    subset_rows = select_balanced_subset(rows, config)
    write_jsonl(output_dir / "dataset_subset.jsonl", subset_rows)
    subset_manifest_dir = output_dir / "subset_manifests"
    write_jsonl(subset_manifest_dir / "train_combined.jsonl", subset_rows)
    length_audit = token_length_audit(rows, tokenizer, int(config["model"]["model_max_length"]))
    (output_dir / "data_audit.json").write_text(
        json.dumps({"token_lengths": length_audit}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if length_audit["num_fake_with_seg_truncated_after_preservation_policy"]:
        raise AssertionError("At least one Fake [SEG] is truncated")
    if cli.prepare_only:
        print(json.dumps({"subset": len(subset_rows), "token_lengths": length_audit}, indent=2))
        return

    device = torch.device(cli.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[train_args.precision]
    model = glamm_train.initialize_model(train_args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, train_args)
    model.to(device=device, dtype=dtype)

    dataset = UnifiedForensicsDataset(
        subset_manifest_dir, tokenizer, train_args.vision_tower, split="train",
        datasets_root=config["data"]["datasets_root"],
        synthscars_root=config["data"]["synthscars_root"], image_size=train_args.image_size,
    )
    if cli.eval_only_base:
        initial_metrics = run_evaluation(
            model, tokenizer, dataset, output_dir / "evaluation" / "initial", device, dtype,
            int(config["training"]["final_generation_max_new_tokens"]),
        )
        summary_path = output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        summary["initial_evaluation"] = initial_metrics
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "base_evaluation_complete", "output_dir": str(output_dir)}, indent=2))
        return
    if cli.eval_only_checkpoint:
        checkpoint = torch.load(cli.eval_only_checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
        if unexpected:
            raise ValueError(f"Unexpected checkpoint keys: {unexpected}")
        final_metrics = run_evaluation(
            model, tokenizer, dataset, output_dir / "evaluation", device, dtype,
            int(config["training"]["final_generation_max_new_tokens"]),
        )
        summary_path = output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        summary["final_evaluation"] = final_metrics
        summary["evaluation_checkpoint_missing_frozen_keys"] = len(missing)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "evaluation_complete", "output_dir": str(output_dir)}, indent=2))
        return
    loader = DataLoader(
        dataset, batch_size=int(config["training"]["batch_size_per_device"]), shuffle=False,
        num_workers=int(config["training"]["num_workers"]), collate_fn=make_collate(tokenizer),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"]["betas"]), weight_decay=float(config["training"]["weight_decay"]),
    )
    max_steps = int(config["training"]["max_optimizer_steps"])
    warmup = int(config["training"]["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / max(1, warmup) if step < warmup
        else max(0.0, (max_steps - step) / max(1, max_steps - warmup)),
    )
    parameter_report = parameter_audit(model, optimizer)
    for token, info in token_audit.items():
        info["embedding_trainable"] = bool(model.get_input_embeddings().weight.requires_grad)

    isolation = loss_isolation_and_gradient_audit(model, dataset, tokenizer, device, dtype)
    leakage = classification_leakage_audit(model, dataset[0], tokenizer, device, dtype)
    parameter_report.update({
        "special_tokens": token_audit,
        "loss_isolation": isolation,
        "classification_label_leakage": leakage,
    })
    parameter_report_text = json.dumps(parameter_report, indent=2, ensure_ascii=False) + "\n"
    (output_dir / "trainable_parameters.json").write_text(parameter_report_text, encoding="utf-8")
    canonical_preflight_dir = REPO_ROOT / "outputs" / "phase1b_preflight"
    canonical_preflight_dir.mkdir(parents=True, exist_ok=True)
    (canonical_preflight_dir / "trainable_parameters.json").write_text(
        parameter_report_text, encoding="utf-8"
    )

    initial_metrics = {}
    if config["training"].get("eval_at_start", True):
        initial_metrics = run_evaluation(
            model, tokenizer, dataset, output_dir / "evaluation" / "initial", device, dtype,
            int(config["training"]["final_generation_max_new_tokens"]),
        )
    else:
        for mode in ("detection", "gt_fake_generate", "tf_full_context", "joint"):
            existing = output_dir / "evaluation" / "initial" / mode / f"forensics_metrics_{mode}.json"
            if existing.is_file():
                initial_metrics[mode] = json.loads(existing.read_text(encoding="utf-8"))

    metrics_path = output_dir / "metrics.jsonl"
    train_log = (output_dir / "train.log").open("w", encoding="utf-8")
    metrics_handle = metrics_path.open("w", encoding="utf-8")
    exposure = {
        "domain": collections.Counter(), "source": collections.Counter(),
        "content_type": collections.Counter(),
    }
    loader_iterator = iter(loader)
    grad_accumulation = int(config["training"]["gradient_accumulation_steps"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    first_step_gradients = None
    for step in range(max_steps):
        loss_sums = collections.Counter()
        prediction_sums = collections.Counter()
        batch_compositions = []
        for _ in range(grad_accumulation):
            try:
                batch = next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(loader)
                batch = next(loader_iterator)
            exposure_update(exposure, batch)
            batch_compositions.append({
                "num_real": int(batch["cls_labels"].eq(0).sum()),
                "num_fake": int(batch["cls_labels"].eq(1).sum()),
            })
            batch = move_batch(batch, device, dtype)
            result = model(**batch)
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError(f"Non-finite loss at step {step}: {result['loss']}")
            (result["loss"] / grad_accumulation).backward()
            for key in ("loss", "ce_loss", "cls_loss", "mask_bce_loss", "mask_dice_loss"):
                loss_sums[key] += float(result[key].detach().float().cpu()) / grad_accumulation
            prediction_sums.update(prediction_metrics(result, batch["cls_labels"]))
        if first_step_gradients is None:
            first_step_gradients = {
                "classification_head": grad_norm_for_names(model, ["classification_head"]),
                "lora": grad_norm_for_names(model, ["lora_"]),
                "token_embeddings": grad_norm_for_names(model, ["embed_tokens"]),
                "text_hidden_fcs": grad_norm_for_names(model, ["text_hidden_fcs"]),
                "mask_decoder": grad_norm_for_names(model, ["grounding_encoder.mask_decoder"]),
                "cls_embedding": embedding_row_grad_norm(model, model.cls_token_idx),
                "real_embedding": embedding_row_grad_norm(model, model.real_token_idx),
                "fake_embedding": embedding_row_grad_norm(model, model.fake_token_idx),
            }
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(config["training"]["gradient_clip_norm"])
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        num = prediction_sums["num"]
        num_fake = prediction_sums["num_fake"]
        record = {
            "record_type": "train_step", "step": step + 1, **dict(loss_sums),
            "classification_accuracy": prediction_sums["cls_correct"] / num,
            "classification_fake_recall": prediction_sums["fake_correct"] / max(1, num_fake),
            "lm_verdict_accuracy": prediction_sums["lm_correct"] / num,
            "cls_lm_agreement": prediction_sums["agree"] / num,
            "gradient_norm_before_clip": float(torch.as_tensor(gradient_norm).float().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "batches": batch_compositions,
        }
        metrics_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        metrics_handle.flush()
        message = json.dumps(record, ensure_ascii=False)
        print(message, flush=True)
        train_log.write(message + "\n")
        train_log.flush()
    metrics_handle.close()
    train_log.close()

    final_metrics = run_evaluation(
        model, tokenizer, dataset, output_dir / "evaluation", device, dtype,
        int(config["training"]["final_generation_max_new_tokens"]),
    )
    summary = {
        "initial_evaluation": initial_metrics,
        "final_evaluation": final_metrics,
        "first_training_step_gradient_norms": first_step_gradients,
        "effective_exposure": serialize_counter(exposure),
        "optimizer_steps": max_steps,
        "gradient_accumulation_steps": grad_accumulation,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    torch.save({"trainable_state_dict": trainable_state, "config": config}, output_dir / "final_checkpoint.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    print(json.dumps({"status": "complete", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
