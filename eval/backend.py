from __future__ import annotations
import copy
import hashlib
from typing import Any, Mapping
import torch
from dataset.collate import custom_collate_fn
from dataset.forensics.unified import CANONICAL_PROMPT_SHA256, CANONICAL_PROMPT_TEMPLATE_ID, CANONICAL_UNIFIED_QUESTION, CANONICAL_UNIFIED_USER_CONTENT
from model.llava import conversation as conversation_lib
from tools.utils import IMAGE_TOKEN_INDEX
FORENSICS_QUESTION = 'Analyze the synthetic artifacts in this image, explain the forensic evidence, and localize the corresponding artifact regions.'
UNIFIED_FORENSICS_QUESTION = CANONICAL_UNIFIED_QUESTION

class GLaMMForensicsBackend:
    """Translate explicit protocol contexts into GLaMM forward/generate calls."""

    def __init__(self, model, tokenizer, *, device, dtype, use_mm_start_end=True, max_new_tokens=256, forensic_evidence_provider=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_mm_start_end = use_mm_start_end
        self.max_new_tokens = max_new_tokens
        self.forensic_evidence_provider = forensic_evidence_provider

    def _forensic_evidence_tokens(self, samples):
        """Return optional continuous evidence tokens without changing raw token IDs."""
        provider = getattr(self, 'forensic_evidence_provider', None)
        if provider is None:
            return None
        tokens = provider(list(samples))
        if tokens is None or tokens.ndim != 3 or tokens.shape[0] != len(samples):
            raise ValueError('forensic evidence provider must return [B,K,H]')
        return tokens.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _conversation(assistant_content: str, *, question: str=FORENSICS_QUESTION, continue_assistant: bool=False) -> str:
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        user_content = CANONICAL_UNIFIED_USER_CONTENT if question == CANONICAL_UNIFIED_QUESTION else 'The <image> provides an overview of the picture.\n' + question
        conv.append_message(conv.roles[0], user_content)
        conv.append_message(conv.roles[1], assistant_content)
        prompt = conv.get_prompt()
        assistant_terminator = conv.sep2 or conv.sep
        if continue_assistant and assistant_content and assistant_terminator and prompt.endswith(assistant_terminator):
            prompt = prompt[:-len(assistant_terminator)]
        return prompt

    def _batch(self, sample: Mapping[str, Any], assistant_content: str, *, question: str=FORENSICS_QUESTION, continue_assistant: bool=False):
        return self._batch_many([sample], assistant_content, question=question, continue_assistant=continue_assistant)

    def _batch_many(self, samples, assistant_content: str, *, question: str=FORENSICS_QUESTION, continue_assistant: bool=False):
        evaluation_samples = []
        for sample in samples:
            evaluation_sample = copy.copy(dict(sample))
            evaluation_sample['conversations'] = [self._conversation(assistant_content, question=question, continue_assistant=continue_assistant)]
            evaluation_samples.append(evaluation_sample)
        batch = custom_collate_fn(evaluation_samples, tokenizer=self.tokenizer, use_mm_start_end=self.use_mm_start_end, inference=True, token_strategy='fixed_cls_query')
        for (key, value) in tuple(batch.items()):
            if torch.is_tensor(value):
                value = value.to(self.device)
                if key in {'global_enc_images', 'grounding_enc_images'}:
                    value = value.to(self.dtype)
                batch[key] = value
        return batch

    def generate_explanation_batch(self, samples):
        samples = list(samples)
        if not samples:
            return []
        batch = self._batch_many(samples, '', question=UNIFIED_FORENSICS_QUESTION, continue_assistant=False)
        original_sizes = [tuple(label.shape) for label in batch['label_list']]
        evidence_tokens = self._forensic_evidence_tokens(samples)
        with torch.no_grad():
            evaluation_kwargs = {}
            if evidence_tokens is not None:
                evaluation_kwargs['forensic_evidence_tokens'] = evidence_tokens
            (sequences, pred_masks, cls_output, details) = self.model.evaluate(batch['global_enc_images'], batch['grounding_enc_images'], batch['input_ids'], batch['resize_list'], original_sizes, max_tokens_new=self.max_new_tokens, bboxes=batch['bboxes'], force_cls_token=False, return_generation_details=True, **evaluation_kwargs)
        prompt_length = batch['input_ids'].shape[1]
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        results = []
        for (index, sample) in enumerate(samples):
            generated_ids = sequences[index, prompt_length:]
            if eos_token_id is not None:
                eos = generated_ids.eq(eos_token_id).nonzero(as_tuple=False).flatten()
                if eos.numel():
                    generated_ids = generated_ids[:int(eos[0]) + 1]
            generated_token_ids = [int(value) for value in generated_ids.detach().cpu().tolist()]
            decoded_ids = generated_ids[generated_ids.ne(IMAGE_TOKEN_INDEX)]
            generated_text = self.tokenizer.decode(decoded_ids, skip_special_tokens=False).strip()
            results.append(dict(generated_text=generated_text, generated_token_ids=generated_token_ids, prompt_token_ids=[int(value) for value in batch['input_ids'][index].detach().cpu().tolist()]))
        return results
