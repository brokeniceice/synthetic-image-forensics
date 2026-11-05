"""CLI and GLaMM backend for the Phase 1A forensic evaluation protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset import custom_collate_fn
from dataset.forensics.unified import (
    CANONICAL_PROMPT_SHA256,
    CANONICAL_PROMPT_TEMPLATE_ID,
    CANONICAL_UNIFIED_QUESTION,
    CANONICAL_UNIFIED_USER_CONTENT,
    UnifiedForensicsDataset,
)
from eval.forensics import (
    FORENSICS_EVAL_MODES,
    MODE_EVALUATORS,
    TF_MINIMAL_CONTEXT_TEMPLATE,
    write_evaluation_outputs,
)
from eval.phase1c_seg_diagnosis import (
    classify_generation_stop,
    explanation_drift,
    probability_trace,
    summarize_probability_trace,
)
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from tools.utils import DEFAULT_CLS_TOKEN, DEFAULT_FAKE_TOKEN, DEFAULT_REAL_TOKEN, IMAGE_TOKEN_INDEX


FORENSICS_QUESTION = (
    "Analyze the synthetic artifacts in this image, explain the forensic evidence, "
    "and localize the corresponding artifact regions."
)
UNIFIED_FORENSICS_QUESTION = CANONICAL_UNIFIED_QUESTION


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase 1A unified forensic evaluator")
    parser.add_argument("--version", required=True, help="Phase 0.5-compatible model/checkpoint path")
    parser.add_argument("--forensics_manifest_dir", required=True)
    parser.add_argument("--dataset_dir", default="datasets")
    parser.add_argument("--synthscars_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--forensics_eval_mode", default="all", choices=(*FORENSICS_EVAL_MODES, "all"))
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--vision_pretrained", default=None)
    parser.add_argument("--conv_type", default="llava_v1", choices=("llava_v1", "llava_llama_2"))
    parser.add_argument("--precision", default="bf16", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--max_samples", default=None, type=int,
                        help="Optional smoke-test cap; omitted means the complete frozen test split")
    parser.add_argument("--use_mm_start_end", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


class _LimitedDataset:
    def __init__(self, dataset, limit=None):
        self.dataset = dataset
        self.limit = len(dataset) if limit is None else min(len(dataset), limit)

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        if index < 0 or index >= self.limit:
            raise IndexError(index)
        return self.dataset[index]


class GLaMMForensicsBackend:
    """Translate explicit protocol contexts into GLaMM forward/generate calls."""

    def __init__(self, model, tokenizer, *, device, dtype, use_mm_start_end=True, max_new_tokens=256,
                 forensic_evidence_provider=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_mm_start_end = use_mm_start_end
        self.max_new_tokens = max_new_tokens
        self.forensic_evidence_provider = forensic_evidence_provider

    def _forensic_evidence_tokens(self, samples):
        """Return optional continuous evidence tokens without changing raw token IDs."""
        provider = getattr(self, "forensic_evidence_provider", None)
        if provider is None:
            return None
        tokens = provider(list(samples))
        if tokens is None or tokens.ndim != 3 or tokens.shape[0] != len(samples):
            raise ValueError("forensic evidence provider must return [B,K,H]")
        return tokens.to(device=self.device, dtype=self.dtype)

    def _model_forward_with_optional_evidence(self, batch, evidence_tokens):
        if evidence_tokens is None:
            return self.model.model_forward(**batch)
        return self.model.model_forward(**batch, forensic_evidence_tokens=evidence_tokens)

    @staticmethod
    def _prediction_fields(output: Mapping[str, Any], index=0) -> dict[str, Any]:
        return {
            "cls_pred": output["cls_pred"][index],
            "cls_prob_fake": output["cls_prob"][index],
            "lm_verdict_pred": output["lm_verdict_pred"][index],
            "lm_verdict_prob_fake": output["lm_verdict_prob"][index],
            "cls_lm_agree": output["cls_lm_agree"][index],
        }

    @staticmethod
    def _conversation(assistant_content: str, *, question: str = FORENSICS_QUESTION,
                      continue_assistant: bool = False) -> str:
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        user_content = (
            CANONICAL_UNIFIED_USER_CONTENT
            if question == CANONICAL_UNIFIED_QUESTION
            else "The <image> provides an overview of the picture.\n" + question
        )
        conv.append_message(conv.roles[0], user_content)
        conv.append_message(conv.roles[1], assistant_content)
        prompt = conv.get_prompt()
        # ``get_prompt`` terminates non-empty assistant messages with sep2.
        # A prefilled continuation is a prefix, not a completed assistant turn;
        # keeping the EOS separator changes the KV trajectory and was the
        # Phase-1A legacy G2 implementation bug.
        assistant_terminator = conv.sep2 or conv.sep
        if (continue_assistant and assistant_content and assistant_terminator
                and prompt.endswith(assistant_terminator)):
            prompt = prompt[:-len(assistant_terminator)]
        return prompt

    def _batch(self, sample: Mapping[str, Any], assistant_content: str, *,
               question: str = FORENSICS_QUESTION, continue_assistant: bool = False):
        return self._batch_many(
            [sample], assistant_content, question=question,
            continue_assistant=continue_assistant,
        )

    def _batch_many(self, samples, assistant_content: str, *,
                    question: str = FORENSICS_QUESTION, continue_assistant: bool = False):
        evaluation_samples = []
        for sample in samples:
            evaluation_sample = copy.copy(dict(sample))
            evaluation_sample["conversations"] = [self._conversation(
                assistant_content, question=question, continue_assistant=continue_assistant
            )]
            evaluation_samples.append(evaluation_sample)
        batch = custom_collate_fn(
            evaluation_samples, tokenizer=self.tokenizer, use_mm_start_end=self.use_mm_start_end,
            inference=True, token_strategy="fixed_cls_query",
        )
        for key, value in tuple(batch.items()):
            if torch.is_tensor(value):
                value = value.to(self.device)
                if key in {"global_enc_images", "grounding_enc_images"}:
                    value = value.to(self.dtype)
                batch[key] = value
        return batch

    def generate_localization_batch(self, samples, *, provide_gt_fake: bool,
                                    generation_mode: str | None = None):
        """Batched equivalent of ``generate_localization`` for equal-length prompts."""
        samples = list(samples)
        if not samples:
            return []
        if generation_mode is None:
            generation_mode = "unified_prompt_gt_fake_prefix" if provide_gt_fake else "joint"
        modes = {
            "unified_fake_generate": (UNIFIED_FORENSICS_QUESTION, "", False),
            "unified_prompt_gt_fake_prefix": (UNIFIED_FORENSICS_QUESTION, "[FAKE]", True),
            "gt_fake_generate": (FORENSICS_QUESTION, "[FAKE]", True),
            "legacy_gt_fake_generate": (FORENSICS_QUESTION, "[FAKE]", False),
            "joint": (UNIFIED_FORENSICS_QUESTION, "", False),
        }
        if generation_mode not in modes:
            raise ValueError(f"Unknown generation mode: {generation_mode}")
        question, assistant_content, continue_assistant = modes[generation_mode]
        batch = self._batch_many(
            samples, assistant_content, question=question,
            continue_assistant=continue_assistant,
        )
        original_sizes = [tuple(label.shape) for label in batch["label_list"]]
        evidence_tokens = self._forensic_evidence_tokens(samples)
        with torch.no_grad():
            evaluation_kwargs = {}
            if evidence_tokens is not None:
                evaluation_kwargs["forensic_evidence_tokens"] = evidence_tokens
            sequences, pred_masks, cls_output, details = self.model.evaluate(
                batch["global_enc_images"], batch["grounding_enc_images"], batch["input_ids"],
                batch["resize_list"], original_sizes, max_tokens_new=self.max_new_tokens,
                bboxes=batch["bboxes"], force_cls_token=False, return_generation_details=True,
                **evaluation_kwargs,
            )
        prompt_length = batch["input_ids"].shape[1]
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        results = []
        for index, sample in enumerate(samples):
            generated_ids = sequences[index, prompt_length:]
            # Batched generation pads rows that finish before their peers.  Keep
            # the same sequence/score semantics as batch-size-one evaluation.
            if eos_token_id is not None:
                eos = generated_ids.eq(eos_token_id).nonzero(as_tuple=False).flatten()
                if eos.numel():
                    generated_ids = generated_ids[:int(eos[0]) + 1]
            generated_token_ids = [int(value) for value in generated_ids.detach().cpu().tolist()]
            seg_token_id = self.model.seg_token_idx
            seg_triggered = seg_token_id in generated_token_ids
            decoded_ids = generated_ids[generated_ids.ne(IMAGE_TOKEN_INDEX)]
            generated_text = self.tokenizer.decode(decoded_ids, skip_special_tokens=False).strip()
            generated_explanation = generated_text.replace("[SEG]", "").strip()
            stop = classify_generation_stop(
                generated_token_ids, seg_token_id=seg_token_id, eos_token_id=eos_token_id,
                max_new_tokens=int(details.get("max_new_tokens", self.max_new_tokens)),
                invalid_special_token_ids=tuple(
                    value for value in (
                        getattr(self.model, "cls_token_idx", None),
                        getattr(self.model, "real_token_idx", None),
                    ) if value is not None
                ),
            )
            scores = [score[index] for score in details.get("scores", ())[:len(generated_token_ids)]]
            trace = probability_trace(
                scores, generated_token_ids, seg_token_id=seg_token_id, eos_token_id=eos_token_id
            ) if scores else []
            trace_summary = summarize_probability_trace(
                trace, eos_position=stop["first_eos_position"]
            )
            row = sample.get("manifest_row") or {}
            gt_explanation = " ".join(str(row.get("explanation") or "").split())
            gt_ids = self.tokenizer(gt_explanation, add_special_tokens=False).input_ids
            drift = explanation_drift(
                generated_token_ids, gt_ids,
                stop_token_ids=tuple(
                    value for value in (seg_token_id, eos_token_id) if value is not None
                ),
                ignored_prefix_token_ids=(getattr(self.model, "fake_token_idx"),),
            )
            result = self._prediction_fields(cls_output, index=index)
            pred_mask = pred_masks[index] if pred_masks[index].shape[0] else None
            result.update({
                "generated_explanation": generated_explanation,
                "seg_triggered": seg_triggered,
                "pred_mask": pred_mask,
                "generation_mode": generation_mode,
                "generated_text": generated_text,
                "generated_token_ids": generated_token_ids,
                "contains_fake_token": getattr(self.model, "fake_token_idx", None) in generated_token_ids,
                "contains_real_token": getattr(self.model, "real_token_idx", None) in generated_token_ids,
                "contains_seg_token": seg_token_id in generated_token_ids,
                "max_new_tokens": int(details.get("max_new_tokens", self.max_new_tokens)),
                "prompt_token_ids": [int(value) for value in batch["input_ids"][index].detach().cpu().tolist()],
                "prompt_ends_with_eos": bool(
                    eos_token_id is not None and int(batch["input_ids"][index, -1]) == eos_token_id
                ),
                "prompt_template_id": (
                    CANONICAL_PROMPT_TEMPLATE_ID if question == CANONICAL_UNIFIED_QUESTION
                    else "legacy_known_fake_prompt_v1"
                ),
                "prompt_sha256": (
                    CANONICAL_PROMPT_SHA256 if question == CANONICAL_UNIFIED_QUESTION
                    else hashlib.sha256((
                        "The <image> provides an overview of the picture.\n" + question
                    ).encode("utf-8")).hexdigest()
                ),
                "raw_prompt_text": batch.get("conversation_list", [None] * len(samples))[index],
                # Ordered generated [SEG] predictor states.  Consumers that do
                # not request Phase 6C.2 replay ignore this diagnostic field.
                "projected_seg_embeddings": details.get(
                    "projected_seg_embeddings", [None] * len(samples)
                )[index],
                "assistant_prefix": "[CLS] [FAKE]" if assistant_content else "[CLS]",
                "seg_probability_trace": trace,
                **stop, **trace_summary, **drift,
            })
            results.append(result)
        return results

    def _causal_forward(self, sample: Mapping[str, Any], assistant_content: str, *,
                        question: str = FORENSICS_QUESTION):
        batch = self._batch(sample, assistant_content, question=question)
        evidence_tokens = self._forensic_evidence_tokens([sample])
        with torch.no_grad():
            output = self._model_forward_with_optional_evidence(batch, evidence_tokens)
        return batch, output

    @staticmethod
    def authoritative_phrase(sample: Mapping[str, Any]) -> str:
        """Return the frozen phrase-aligned target used by Phase 3A diagnostics."""
        field = sample.get("localization_field") or {}
        phrase = " ".join(str(field.get("normalized_training_phrase") or "").split())
        if not phrase:
            raise ValueError(f"Fake sample {sample.get('sample_id')} has no authoritative phrase")
        return phrase

    @classmethod
    def phrase_only_content(cls, sample: Mapping[str, Any]) -> str:
        return f"[FAKE] Target regions: {cls.authoritative_phrase(sample)} [SEG]"

    @classmethod
    def tf_phrase_content(cls, sample: Mapping[str, Any]) -> str:
        row = cls._row_for_sample(sample)
        explanation = " ".join(str(row.get("explanation") or "").split())
        if not explanation:
            raise ValueError(f"Fake sample {sample.get('sample_id')} has no GT explanation")
        return (
            f"[FAKE] {explanation}\nTarget regions: "
            f"{cls.authoritative_phrase(sample)} [SEG]"
        )

    def detection(self, sample: Mapping[str, Any], *, user_prompt: str = "canonical") -> Mapping[str, Any]:
        questions = {"legacy": FORENSICS_QUESTION, "canonical": UNIFIED_FORENSICS_QUESTION}
        if user_prompt not in questions:
            raise ValueError(f"Unknown detection user prompt: {user_prompt}")
        batch = self._batch(sample, "", question=questions[user_prompt])
        # Detection consumes only the global encoder and fixed [CLS] state.
        # Avoid running SAM when no localization representation is requested.
        batch["grounding_enc_images"] = None
        evidence_tokens = self._forensic_evidence_tokens([sample])
        with torch.no_grad():
            output = self._model_forward_with_optional_evidence(batch, evidence_tokens)
        return self._prediction_fields(output)

    def generated_fake_vs_prefilled_fake_alignment(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """Compare continuation state after a generated vs correctly-prefilled [FAKE]."""
        free_batch = self._batch(
            sample, "", question=UNIFIED_FORENSICS_QUESTION, continue_assistant=False
        )
        prefix_batch = self._batch(
            sample, "[FAKE]", question=UNIFIED_FORENSICS_QUESTION, continue_assistant=True
        )
        with torch.no_grad():
            generated = self.model.generate(
                images=free_batch["global_enc_images"], input_ids=free_batch["input_ids"],
                bboxes=free_batch["bboxes"], max_new_tokens=1, num_beams=1,
                do_sample=False, use_cache=True,
            )
        generated_token_id = int(generated[0, -1])
        path_a_ids = generated
        path_b_ids = prefix_batch["input_ids"]
        raw_ids_equal = bool(
            path_a_ids.shape == path_b_ids.shape and torch.equal(path_a_ids, path_b_ids)
        )

        def continuation_state(batch, ids):
            attention = torch.ones_like(ids)
            with torch.no_grad():
                output, hidden_states = self.model._inference_path(
                    ids, batch["global_enc_images"], attention, batch["offset"], batch["bboxes"]
                )
            fake_positions = ids[0].eq(self.model.fake_token_idx).nonzero(as_tuple=False).flatten()
            if fake_positions.numel() != 1:
                return output, hidden_states, None, None
            raw_position = int(fake_positions[0])
            images_before = int(ids[0, :raw_position].eq(IMAGE_TOKEN_INDEX).sum())
            expanded_position = raw_position + images_before * 575
            last_hidden = hidden_states[-1]
            hidden = (
                last_hidden[0, expanded_position] if last_hidden.ndim == 3
                else last_hidden[expanded_position]
            ).detach().float().cpu().reshape(-1)
            logits_tensor = output.logits
            logits = (
                logits_tensor[0, expanded_position] if logits_tensor.ndim == 3
                else logits_tensor[expanded_position]
            ).detach().float().cpu().reshape(-1)
            return output, hidden_states, hidden, logits

        _, hidden_a_all, hidden_a, logits_a = continuation_state(free_batch, path_a_ids)
        _, hidden_b_all, hidden_b, logits_b = continuation_state(prefix_batch, path_b_ids)
        aligned = generated_token_id == self.model.fake_token_idx and hidden_a is not None and hidden_b is not None
        hidden_max_abs = None
        logits_max_abs = None
        hidden_cosine = None
        logits_cosine = None
        top_a = []
        top_b = []
        if aligned:
            hidden_max_abs = float((hidden_a - hidden_b).abs().max())
            logits_max_abs = float((logits_a - logits_b).abs().max())
            hidden_cosine = float(torch.nn.functional.cosine_similarity(hidden_a, hidden_b, dim=0))
            logits_cosine = float(torch.nn.functional.cosine_similarity(logits_a, logits_b, dim=0))
            top_a = [int(value) for value in logits_a.topk(10).indices.tolist()]
            top_b = [int(value) for value in logits_b.topk(10).indices.tolist()]
        fake_raw_position_a = next(
            (index for index, value in enumerate(path_a_ids[0].tolist()) if value == self.model.fake_token_idx), None
        )
        fake_raw_position_b = next(
            (index for index, value in enumerate(path_b_ids[0].tolist()) if value == self.model.fake_token_idx), None
        )
        return {
            "sample_id": sample.get("sample_id"),
            "fake_token_id": int(self.model.fake_token_idx),
            "generated_token_id": generated_token_id,
            "generated_fake": generated_token_id == self.model.fake_token_idx,
            "path_a_raw_token_ids": [int(value) for value in path_a_ids[0].detach().cpu().tolist()],
            "path_b_raw_token_ids": [int(value) for value in path_b_ids[0].detach().cpu().tolist()],
            "raw_token_ids_equal": raw_ids_equal,
            "raw_sequence_length_a": int(path_a_ids.shape[1]),
            "raw_sequence_length_b": int(path_b_ids.shape[1]),
            "fake_raw_position_a": fake_raw_position_a,
            "fake_raw_position_b": fake_raw_position_b,
            "position_id_a": fake_raw_position_a,
            "position_id_b": fake_raw_position_b,
            "attention_mask_length_a": int(path_a_ids.shape[1]),
            "attention_mask_length_b": int(path_b_ids.shape[1]),
            "expanded_sequence_length_a": int(hidden_a_all[-1].shape[-2]),
            "expanded_sequence_length_b": int(hidden_b_all[-1].shape[-2]),
            "hidden_max_abs_difference": hidden_max_abs,
            "hidden_cosine_similarity": hidden_cosine,
            "logits_max_abs_difference": logits_max_abs,
            "logits_cosine_similarity": logits_cosine,
            "hidden_a_first16": None if hidden_a is None else hidden_a[:16].tolist(),
            "hidden_b_first16": None if hidden_b is None else hidden_b[:16].tolist(),
            "top10_next_token_ids_a": top_a,
            "top10_next_token_ids_b": top_b,
            "prefilled_prompt_ends_with_eos": bool(
                self.tokenizer.eos_token_id is not None
                and int(path_b_ids[0, -1]) == self.tokenizer.eos_token_id
            ),
        }

    def teacher_forced_localization(self, sample: Mapping[str, Any], *, context: str,
                                    user_prompt: str = "canonical") -> Mapping[str, Any]:
        if context == "full":
            if sample.get("target_protocol") == "phrase_aligned":
                assistant_content = self.tf_phrase_content(sample)
            else:
                row = self._row_for_sample(sample)
                explanation = " ".join(str(row.get("explanation") or "").split())
                if not explanation:
                    raise ValueError(f"Fake sample {sample.get('sample_id')} has no GT explanation")
                assistant_content = f"[FAKE] {explanation} [SEG]"
        elif context == "minimal":
            assistant_content = f"[FAKE] {TF_MINIMAL_CONTEXT_TEMPLATE}"
        else:
            raise ValueError(f"Unknown teacher-forced context: {context}")
        questions = {
            "legacy": FORENSICS_QUESTION,
            "canonical": UNIFIED_FORENSICS_QUESTION,
        }
        if user_prompt not in questions:
            raise ValueError(f"Unknown teacher-forced user prompt: {user_prompt}")
        _, output = self._causal_forward(sample, assistant_content, question=questions[user_prompt])
        result = self._prediction_fields(output)
        pred_masks = output["pred_masks"][0]
        result.update({
            "generated_explanation": None,
            "seg_triggered": True,
            "pred_mask": pred_masks if pred_masks is not None and pred_masks.shape[0] else None,
        })
        return result

    def phrase_only_localization(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """Oracle diagnostic using only the authoritative target phrase before ``[SEG]``."""
        phrase = self.authoritative_phrase(sample)
        assistant_content = self.phrase_only_content(sample)
        _, output = self._causal_forward(sample, assistant_content)
        result = self._prediction_fields(output)
        pred_masks = output["pred_masks"][0]
        result.update({
            "generated_explanation": None,
            "seg_triggered": True,
            "pred_mask": pred_masks if pred_masks is not None and pred_masks.shape[0] else None,
            "oracle_phrase": phrase,
            "oracle_template": "[FAKE] Target regions: <authoritative phrase(s)> [SEG]",
        })
        return result

    @staticmethod
    def _row_for_sample(sample: Mapping[str, Any]) -> Mapping[str, Any]:
        row = sample.get("manifest_row")
        if row is None:
            raise ValueError("Evaluation sample is missing manifest_row")
        return row

    def generate_localization(self, sample: Mapping[str, Any], *, provide_gt_fake: bool,
                              generation_mode: str | None = None) -> Mapping[str, Any]:
        """Generate one localization trajectory without ever reading GT explanation.

        ``legacy_gt_fake_generate`` preserves the Phase-1B terminal-EOS prefix
        exactly for reproducibility.  The corrected G2 uses the same natural
        language prompt but treats ``[FAKE]`` as a continuation prefix.
        """
        if generation_mode is None:
            generation_mode = "unified_prompt_gt_fake_prefix" if provide_gt_fake else "joint"
        modes = {
            "unified_fake_generate": (UNIFIED_FORENSICS_QUESTION, "", False),
            "unified_prompt_gt_fake_prefix": (UNIFIED_FORENSICS_QUESTION, "[FAKE]", True),
            "gt_fake_generate": (FORENSICS_QUESTION, "[FAKE]", True),
            "legacy_gt_fake_generate": (FORENSICS_QUESTION, "[FAKE]", False),
            "joint": (UNIFIED_FORENSICS_QUESTION, "", False),
        }
        if generation_mode not in modes:
            raise ValueError(f"Unknown generation mode: {generation_mode}")
        question, assistant_content, continue_assistant = modes[generation_mode]
        try:
            batch = self._batch(
                sample, assistant_content, question=question, continue_assistant=continue_assistant
            )
        except TypeError as error:
            if "unexpected keyword" not in str(error):
                raise
            batch = self._batch(sample, assistant_content)
        original_sizes = [tuple(batch["label_list"][0].shape)]
        evidence_tokens = self._forensic_evidence_tokens([sample])
        with torch.no_grad():
            evaluation_args = (
                batch["global_enc_images"], batch["grounding_enc_images"], batch["input_ids"],
                batch["resize_list"], original_sizes,
            )
            evaluation_kwargs = {
                "max_tokens_new": self.max_new_tokens,
                "bboxes": batch["bboxes"],
                "force_cls_token": False,
            }
            if evidence_tokens is not None:
                evaluation_kwargs["forensic_evidence_tokens"] = evidence_tokens
            try:
                evaluation = self.model.evaluate(
                    *evaluation_args, **evaluation_kwargs, return_generation_details=True
                )
            except TypeError as error:
                if "return_generation_details" not in str(error):
                    raise
                evaluation = self.model.evaluate(*evaluation_args, **evaluation_kwargs)
        if len(evaluation) == 4:
            sequences, pred_masks, cls_output, generation_details = evaluation
        else:  # Backward-compatible test doubles and older external backends.
            sequences, pred_masks, cls_output = evaluation
            generation_details = {"scores": (), "max_new_tokens": self.max_new_tokens}
        prompt_length = batch["input_ids"].shape[1]
        generated_ids = sequences[0, prompt_length:]
        seg_token_id = self.model.seg_token_idx
        seg_triggered = bool(generated_ids.eq(seg_token_id).any().item())
        decoded_ids = generated_ids[generated_ids.ne(IMAGE_TOKEN_INDEX)]
        generated_text = self.tokenizer.decode(decoded_ids, skip_special_tokens=False).strip()
        generated_explanation = generated_text.replace("[SEG]", "").strip()
        generated_token_ids = [int(value) for value in generated_ids.detach().cpu().tolist()]
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        stop = classify_generation_stop(
            generated_token_ids, seg_token_id=seg_token_id, eos_token_id=eos_token_id,
            max_new_tokens=int(generation_details.get("max_new_tokens", self.max_new_tokens)),
            invalid_special_token_ids=tuple(
                value for value in (
                    getattr(self.model, "cls_token_idx", None),
                    getattr(self.model, "real_token_idx", None),
                ) if value is not None
            ),
        )
        scores = generation_details.get("scores", ())
        trace = probability_trace(
            scores, generated_token_ids, seg_token_id=seg_token_id, eos_token_id=eos_token_id
        ) if scores else []
        trace_summary = summarize_probability_trace(trace, eos_position=stop["first_eos_position"])
        row = sample.get("manifest_row") or {}
        gt_explanation = " ".join(str(row.get("explanation") or "").split())
        gt_explanation_ids = self.tokenizer(
            gt_explanation, add_special_tokens=False
        ).input_ids if gt_explanation and callable(self.tokenizer) else []
        drift = explanation_drift(
            generated_token_ids, gt_explanation_ids,
            stop_token_ids=tuple(value for value in (seg_token_id, eos_token_id) if value is not None),
            ignored_prefix_token_ids=tuple(
                value for value in (getattr(self.model, "fake_token_idx", None),) if value is not None
            ),
        )
        result = self._prediction_fields(cls_output)
        pred_mask = pred_masks[0] if pred_masks and pred_masks[0].shape[0] else None
        result.update({
            "generated_explanation": generated_explanation,
            "seg_triggered": seg_triggered,
            "pred_mask": pred_mask,
            "generation_mode": generation_mode,
            "generated_text": generated_text,
            "generated_token_ids": generated_token_ids,
            "contains_fake_token": getattr(self.model, "fake_token_idx", None) in generated_token_ids,
            "contains_real_token": getattr(self.model, "real_token_idx", None) in generated_token_ids,
            "contains_seg_token": seg_token_id in generated_token_ids,
            "max_new_tokens": int(generation_details.get("max_new_tokens", self.max_new_tokens)),
            "prompt_token_ids": [int(value) for value in batch["input_ids"][0].detach().cpu().tolist()],
            "prompt_ends_with_eos": bool(
                eos_token_id is not None and int(batch["input_ids"][0, -1]) == eos_token_id
            ),
            "prompt_template_id": (
                CANONICAL_PROMPT_TEMPLATE_ID if question == CANONICAL_UNIFIED_QUESTION
                else "legacy_known_fake_prompt_v1"
            ),
            "prompt_sha256": (
                CANONICAL_PROMPT_SHA256 if question == CANONICAL_UNIFIED_QUESTION
                else hashlib.sha256(
                    ("The <image> provides an overview of the picture.\n" + question).encode("utf-8")
                ).hexdigest()
            ),
            "raw_prompt_text": batch.get("conversation_list", [None])[0],
            "projected_seg_embeddings": generation_details.get(
                "projected_seg_embeddings", [None]
            )[0],
            "assistant_prefix": "[CLS] [FAKE]" if assistant_content else "[CLS]",
            "seg_probability_trace": trace,
            **stop,
            **trace_summary,
            **drift,
        })
        return result


def _dtype_for_precision(precision: str):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def load_backend_and_dataset(args):
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    tokenizer = AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens([DEFAULT_CLS_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN], special_tokens=True)
    token_ids = {}
    for token in (DEFAULT_CLS_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN, "[SEG]"):
        ids = tokenizer(token, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"{token} must encode as exactly one token, got {ids}")
        token_ids[token] = ids[0]

    dtype = _dtype_for_precision(args.precision)
    model = GLaMMForCausalLM.from_pretrained(
        args.version,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        vision_pretrained=args.vision_pretrained,
        seg_token_idx=token_ids["[SEG]"],
        cls_token_idx=token_ids[DEFAULT_CLS_TOKEN],
        real_token_idx=token_ids[DEFAULT_REAL_TOKEN],
        fake_token_idx=token_ids[DEFAULT_FAKE_TOKEN],
        token_strategy="fixed_cls_query",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.to(device=torch.device(args.device), dtype=dtype).eval()

    dataset = UnifiedForensicsDataset(
        args.forensics_manifest_dir,
        tokenizer,
        args.vision_tower,
        split="test",
        datasets_root=args.dataset_dir,
        synthscars_root=args.synthscars_root,
    )
    return GLaMMForensicsBackend(
        model, tokenizer, device=args.device, dtype=dtype,
        use_mm_start_end=args.use_mm_start_end, max_new_tokens=args.max_new_tokens,
    ), _LimitedDataset(dataset, args.max_samples)


def main(argv=None):
    args = parse_args(argv)
    backend, dataset = load_backend_and_dataset(args)
    modes = FORENSICS_EVAL_MODES if args.forensics_eval_mode == "all" else (args.forensics_eval_mode,)
    output_dir = Path(args.output_dir)
    for mode in modes:
        records, metrics = MODE_EVALUATORS[mode](dataset, backend)
        prediction_path, metrics_path = write_evaluation_outputs(output_dir, mode, records, metrics)
        print(f"[{mode}] samples={len(records)} predictions={prediction_path} metrics={metrics_path}")


if __name__ == "__main__":
    main()
