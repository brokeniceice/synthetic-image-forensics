import os
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import warnings

from model.SAM import build_sam_vit_h
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel
from tools.utils import IMAGE_TOKEN_INDEX


def _release_cuda_cache_unless_retained():
    """Keep allocator high-water reservations for shared-GPU training when requested."""
    if os.environ.get("GLAMM_PRESERVE_CUDA_CACHE", "0") != "1":
        torch.cuda.empty_cache()


def per_sample_causal_text_loss(logits: torch.Tensor, labels: torch.Tensor,
                                ignore_index: int = -100) -> torch.Tensor:
    """Mean causal CE per sample, then mean over supervised samples."""
    shift_logits = logits[:, :-1].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1),
        ignore_index=ignore_index, reduction="none",
    ).reshape_as(shift_labels)
    valid = shift_labels.ne(ignore_index)
    counts = valid.sum(dim=1)
    supervised = counts.gt(0)
    if not supervised.any():
        return logits.sum() * 0.0
    sample_means = (token_losses * valid).sum(dim=1) / counts.clamp_min(1)
    return sample_means[supervised].mean()


def causal_text_loss_per_sample(logits: torch.Tensor, labels: torch.Tensor,
                                ignore_index: int = -100):
    shift_logits = logits[:, :-1].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1),
        ignore_index=ignore_index, reduction="none",
    ).reshape_as(shift_labels)
    valid = shift_labels.ne(ignore_index)
    counts = valid.sum(dim=1)
    means = (token_losses * valid).sum(dim=1) / counts.clamp_min(1)
    return means, counts.gt(0)


def diagnostic_domain_text_losses(logits, labels, cls_labels):
    means, supervised = causal_text_loss_per_sample(logits, labels)
    cls_labels = torch.as_tensor(cls_labels, device=means.device, dtype=torch.long).reshape(-1)
    if cls_labels.numel() != means.numel():
        raise ValueError(f"Got {cls_labels.numel()} domain labels for {means.numel()} LM sequences")
    zero = means.sum() * 0.0
    real = supervised & cls_labels.eq(0)
    fake = supervised & cls_labels.eq(1)
    return means[real].mean() if real.any() else zero, means[fake].mean() if fake.any() else zero


def mask_gradient_rows(gradient: torch.Tensor, trainable_row_ids) -> torch.Tensor:
    """Option-B audit helper: retain gradients only for explicitly allowed vocabulary rows."""
    masked = torch.zeros_like(gradient)
    row_ids = torch.as_tensor(trainable_row_ids, device=gradient.device, dtype=torch.long)
    if row_ids.numel():
        if row_ids.lt(0).any() or row_ids.ge(gradient.shape[0]).any():
            raise IndexError("Vocabulary gradient row id is out of range")
        masked.index_copy_(0, row_ids, gradient.index_select(0, row_ids))
    return masked


def extract_seg_predictor_hidden(last_hidden_state: torch.Tensor, input_ids: torch.Tensor,
                                 seg_token_idx: int, image_expansion: int = 575,
                                 image_token_idx: int = IMAGE_TOKEN_INDEX,
                                 min_token_index: int = 0):
    """Extract the causal states that predict ``[SEG]``.

    A causal LM logit at sequence position ``k`` predicts token ``k + 1``.
    Consequently, when ``[SEG]`` is present at text-token position ``s``, the
    grounding representation is the hidden state at expanded position
    ``s - 1`` (plus the visual-token expansion offset), never the hidden state
    of ``[SEG]`` itself.  Both teacher-forced and generated sequences use this
    helper so that their indexing convention cannot diverge.
    """
    if last_hidden_state.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("Expected hidden states [B,L,H] and input ids [B,T]")
    if last_hidden_state.shape[0] != input_ids.shape[0]:
        raise ValueError("Hidden-state and token batch sizes differ")

    predictor_states, predictor_positions = [], []
    text_positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    for row_index, row in enumerate(input_ids):
        seg_mask = row.eq(seg_token_idx)
        if min_token_index:
            seg_mask[:min_token_index] = False
        image_mask = row.eq(image_token_idx)
        images_before = image_mask.long().cumsum(dim=0) - image_mask.long()
        expanded_seg_positions = text_positions + images_before * image_expansion
        positions = expanded_seg_positions[seg_mask] - 1
        if positions.numel() and (positions.lt(0).any() or positions.ge(last_hidden_state.shape[1]).any()):
            raise IndexError(
                f"[SEG] predictor position is outside expanded hidden sequence: {positions.tolist()} "
                f"vs length {last_hidden_state.shape[1]}"
            )
        predictor_positions.append(positions)
        predictor_states.append(last_hidden_state[row_index, positions])
    return predictor_states, predictor_positions


def calculate_dice_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, mask_count: float, scale_factor=1000,
                        epsilon=1e-6):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (ground_truth / scale_factor).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss


def compute_sigmoid_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, mask_count: float):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss


class GLaMMBaseModel:
    def __init__(self, config, **kwargs):
        super(GLaMMBaseModel, self).__init__(config)
        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)

        # Set config attributes if they don't exist
        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)
        )
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 512))

        self.initialize_glamm_model(self.config)

    def initialize_glamm_model(self, config):
        # Initialize the visual model
        self.grounding_encoder = build_sam_vit_h(self.vision_pretrained)
        self._configure_grounding_encoder(config)

        # Initialize the text projection layer
        self._initialize_text_projection_layer()

    def _configure_grounding_encoder(self, config):
        # Freezing visual model parameters
        for param in self.grounding_encoder.parameters():
            param.requires_grad = False

        # Training mask decoder if specified
        if config.train_mask_decoder:
            self._train_mask_decoder()

    def _train_mask_decoder(self):
        self.grounding_encoder.mask_decoder.train()
        for param in self.grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0), ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()
        self.text_hidden_fcs.train()


class GLaMMModel(GLaMMBaseModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(GLaMMModel, self).__init__(config, **kwargs)
        self._configure_model_settings()

    def _configure_model_settings(self):
        self.config.use_cache = False
        self.config.vision_module = self.config.mm_vision_module
        self.config.select_feature_type = "patch"
        self.config.image_aspect = "square"
        self.config.image_grid_points = None
        self.config.tune_mlp_adapter = False
        self.config.freeze_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.use_image_patch_token = False


class GLaMMForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = GLaMMModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.classification_head = nn.Linear(config.hidden_size, 2)
        self.register_buffer("seg_missing_pred_total", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("seg_count_mismatch_total", torch.zeros((), dtype=torch.long), persistent=False)
        self.post_init()

    def enable_rine_conditioning(self, rine_checkpoint, *, projector_dtype=None):
        """Install the frozen Phase6B.6 RINE-Q2 branch and trainable C1 heads.

        The existing CLIP tower is shared (and remains frozen); RINE performs a
        separate deterministic forward through that tower.  Only Q2 is exposed
        to the LLM through one directly injected continuous token.
        """
        from model.rine_on_c1 import RINEOnHFCLIP

        hidden = int(self.config.hidden_size)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden, 512), nn.ReLU(), nn.Linear(512, 2)
        )
        self.forensic_projector = nn.Sequential(
            nn.Linear(1024, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        vision = self.get_vision_tower().vision_tower
        self.rine_conditioner = RINEOnHFCLIP(vision)
        payload = torch.load(rine_checkpoint, map_location="cpu", weights_only=False)
        self.rine_conditioner.rine.load_state_dict(payload["rine_state_dict"], strict=True)
        self.rine_conditioner.requires_grad_(False)
        self.rine_conditioner.eval()
        if projector_dtype is not None:
            self.classification_head.to(dtype=projector_dtype)
            self.forensic_projector.to(dtype=projector_dtype)
        self.forensic_evidence_token_count = 1

    def enable_h2_classification_head(self, *, head_dtype=None):
        """Install the frozen Phase 6D.3 C0 matched H2 classifier only."""
        hidden = int(self.config.hidden_size)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden, 512), nn.ReLU(), nn.Linear(512, 2)
        )
        if head_dtype is not None:
            self.classification_head.to(dtype=head_dtype)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "rine_conditioner"):
            self.rine_conditioner.eval()
            self.rine_conditioner.requires_grad_(False)
        return self

    def _rine_evidence_token(self, global_enc_images):
        if not hasattr(self, "rine_conditioner"):
            return None
        self.rine_conditioner.eval()
        with torch.no_grad():
            rine_logits, q2, _ = self.rine_conditioner(global_enc_images)
        self._last_rine_binary_logits = rine_logits.detach().float().reshape(-1)
        projected = self.forensic_projector(q2.to(dtype=self.forensic_projector[0].weight.dtype))
        if projected.requires_grad:
            projected.register_hook(
                lambda grad: setattr(self, "_last_frc_gradient_norm", float(grad.detach().float().norm()))
            )
        self._last_rine_q2_stats = q2.detach().float()
        self._last_frc_stats = projected.detach().float()
        return projected.unsqueeze(1)

    def _set_model_configurations(self, config, kwargs):
        config.mm_use_image_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_module = kwargs.get("vision_module", "openai/clip-vit-large-patch14-336")
        self._initialize_loss_weights(kwargs)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 1)
        config.num_reg_features = kwargs.get("num_level_reg_features", 4)
        config.with_region = kwargs.get("with_region", True)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 32002)
        self.seg_token_idx = kwargs.pop("seg_token_idx", getattr(config, "seg_token_idx", None))
        self.cls_token_idx = kwargs.pop("cls_token_idx", getattr(config, "cls_token_idx", None))
        self.real_token_idx = kwargs.pop("real_token_idx", getattr(config, "real_token_idx", None))
        self.fake_token_idx = kwargs.pop("fake_token_idx", getattr(config, "fake_token_idx", None))
        self.token_strategy = kwargs.pop("token_strategy", getattr(config, "token_strategy", "generated_cls"))
        self.per_sample_text_loss_normalization = kwargs.pop(
            "per_sample_text_loss_normalization",
            getattr(config, "per_sample_text_loss_normalization", False),
        )
        config.seg_token_idx = self.seg_token_idx
        config.cls_token_idx = self.cls_token_idx
        config.real_token_idx = self.real_token_idx
        config.fake_token_idx = self.fake_token_idx
        config.token_strategy = self.token_strategy
        config.per_sample_text_loss_normalization = self.per_sample_text_loss_normalization

    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.cls_loss_weight = kwargs.pop("cls_loss_weight", 1.0)

    def get_grounding_encoder_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            return torch.cat([self._encode_single_image(img) for img in pixel_values], dim=0)

    def _encode_single_image(self, image):
        _release_cuda_cache_unless_retained()
        return self.model.grounding_encoder.image_encoder(image.unsqueeze(0))

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            if kwargs["past_key_values"] is None:
                kwargs.pop("past_key_values")
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(self, global_enc_images: torch.FloatTensor, grounding_enc_images: torch.FloatTensor,
                      bboxes: torch.FloatTensor, input_ids: torch.LongTensor, labels: torch.LongTensor,
                      attention_masks: torch.LongTensor, offset: torch.LongTensor, masks_list: List[torch.FloatTensor],
                      label_list: List[torch.Tensor], resize_list: List[tuple], inference: bool = False,
                      cls_labels: torch.LongTensor = None, seg_valid: torch.BoolTensor = None,
                      precomputed_grounding_embeddings: torch.FloatTensor = None,
                      forensic_evidence_tokens: torch.FloatTensor = None,
                      multiseg_cache_only: bool = False, **kwargs, ):

        if hasattr(self, "rine_conditioner"):
            if forensic_evidence_tokens is not None:
                raise ValueError("C1 RINE conditioning forbids an external evidence token override")
            forensic_evidence_tokens = self._rine_evidence_token(global_enc_images)

        # Handle inference or training paths
        if inference:
            output, output_hidden_states = self._inference_path(
                input_ids, global_enc_images, attention_masks, offset, bboxes,
                forensic_evidence_tokens=forensic_evidence_tokens,
            )
        else:
            output, output_hidden_states = self._training_path(
                global_enc_images, bboxes, input_ids, labels, attention_masks, offset,
                forensic_evidence_tokens=forensic_evidence_tokens,
            )

        cls_logits, cls_valid_mask = self._extract_cls_logits(output_hidden_states, input_ids)
        image_cls_logits, image_cls_valid_mask = self._aggregate_cls_logits(cls_logits, cls_valid_mask, offset)
        cls_loss = self._compute_cls_loss(cls_logits, cls_valid_mask, cls_labels, offset)
        lm_verdict_logits, lm_verdict_valid = self._extract_lm_verdict_logits(output.logits, input_ids)
        image_lm_verdict_logits, image_lm_verdict_valid = self._aggregate_cls_logits(
            lm_verdict_logits, lm_verdict_valid, offset
        )

        if grounding_enc_images is not None:
            # Extract grounding encoder image embeddings
            image_embeddings = (
                precomputed_grounding_embeddings
                if precomputed_grounding_embeddings is not None
                else self.get_grounding_encoder_embs(grounding_enc_images)
            )
            assert image_embeddings.shape[0] == len(offset) - 1

            # Extract the causal state immediately preceding each [SEG].
            pred_embeddings, _ = self._extract_projected_seg_predictor_hidden(
                output_hidden_states, input_ids, offset
            )

            if multiseg_cache_only:
                return {
                    "projected_seg_embeddings": [value.detach() for value in pred_embeddings],
                    "grounding_image_embeddings": image_embeddings.detach(),
                }

            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                pred_embeddings, image_embeddings, resize_list, label_list
            )

        else:
            pred_masks = None

        if inference:
            cls_probabilities = image_cls_logits.float().softmax(dim=-1)
            lm_probabilities = image_lm_verdict_logits.float().softmax(dim=-1)
            return {
                "pred_masks": pred_masks,
                "gt_masks": masks_list,
                "cls_logits": image_cls_logits,
                "cls_probabilities": cls_probabilities,
                "cls_predictions": image_cls_logits.argmax(dim=-1),
                "cls_valid_mask": image_cls_valid_mask,
                "lm_verdict_logits": image_lm_verdict_logits,
                "lm_verdict_probabilities": lm_probabilities,
                "lm_verdict_predictions": image_lm_verdict_logits.argmax(dim=-1),
                "lm_verdict_valid_mask": image_lm_verdict_valid,
                "cls_lm_agree": image_cls_logits.argmax(dim=-1).eq(
                    image_lm_verdict_logits.argmax(dim=-1)
                ) & image_cls_valid_mask & image_lm_verdict_valid,
                "cls_pred": image_cls_logits.argmax(dim=-1),
                "cls_prob": cls_probabilities[:, 1],
                "lm_verdict_pred": image_lm_verdict_logits.argmax(dim=-1),
                "lm_verdict_prob": lm_probabilities[:, 1],
            }

        # Calculate losses
        loss_dict = self._calculate_losses(pred_masks, masks_list, output, seg_valid)
        loss_dict["cls_loss"] = cls_loss
        loss_dict["loss"] = loss_dict["loss"] + cls_loss
        if cls_labels is not None:
            domain_labels = torch.as_tensor(cls_labels, device=input_ids.device, dtype=torch.long).reshape(-1)
            if offset is not None and domain_labels.numel() == len(offset) - 1:
                domain_labels = torch.repeat_interleave(
                    domain_labels, (offset[1:] - offset[:-1]).to(device=input_ids.device)
                )
            real_text_loss, fake_text_loss = diagnostic_domain_text_losses(
                output.logits, output.get("expanded_labels", labels), domain_labels
            )
        else:
            real_text_loss = fake_text_loss = output.loss * 0.0
        loss_dict["real_text_loss"] = real_text_loss
        loss_dict["fake_text_loss"] = fake_text_loss
        loss_dict["cls_logits"] = image_cls_logits
        loss_dict["cls_valid_mask"] = image_cls_valid_mask
        loss_dict["lm_verdict_logits"] = image_lm_verdict_logits
        loss_dict["lm_verdict_valid_mask"] = image_lm_verdict_valid
        loss_dict["lm_verdict_probabilities"] = image_lm_verdict_logits.float().softmax(dim=-1)
        loss_dict["lm_verdict_predictions"] = image_lm_verdict_logits.argmax(dim=-1)
        loss_dict["cls_lm_agree"] = image_cls_logits.argmax(dim=-1).eq(
            image_lm_verdict_logits.argmax(dim=-1)
        ) & image_cls_valid_mask & image_lm_verdict_valid
        loss_dict["cls_pred"] = image_cls_logits.argmax(dim=-1)
        loss_dict["cls_prob"] = image_cls_logits.float().softmax(dim=-1)[:, 1]
        loss_dict["lm_verdict_pred"] = image_lm_verdict_logits.argmax(dim=-1)
        loss_dict["lm_verdict_prob"] = image_lm_verdict_logits.float().softmax(dim=-1)[:, 1]
        # Expose teacher-forced masks for validation diagnostics. This is a
        # non-loss output and does not alter training gradients or weighting.
        loss_dict["pred_masks"] = pred_masks
        # The grounding image encoder is frozen and evaluated under no_grad.
        # Phase 3B may reuse these exact embeddings for a second mask-only
        # context branch on the same image, avoiding redundant SAM encoding.
        loss_dict["grounding_image_embeddings"] = (
            image_embeddings.detach() if grounding_enc_images is not None else None
        )
        # Phase 6C.2 may cache the frozen selected-L2 teacher-forced SEG
        # representations for the downstream R1-specific training stage.
        # This diagnostic output is detached and does not alter any loss or
        # trainable path.  Grouping remains one tensor [K_i, out_dim] per image.
        loss_dict["projected_seg_embeddings"] = (
            [value.detach() for value in pred_embeddings]
            if grounding_enc_images is not None else None
        )
        return loss_dict

    def _create_expanded_token_mask(self, input_ids, token_idx, target_length=None, min_token_index=0):
        """Align a text-token mask with the sequence after image-token expansion."""
        if token_idx is None:
            length = target_length if target_length is not None else input_ids.shape[1]
            return torch.zeros((input_ids.shape[0], length), dtype=torch.bool, device=input_ids.device)

        image_expansion = 575 + int(getattr(self, "forensic_evidence_token_count", 0))
        expanded_lengths = input_ids.shape[1] + input_ids.eq(IMAGE_TOKEN_INDEX).sum(dim=1) * image_expansion
        if target_length is None:
            target_length = int(expanded_lengths.max().item())

        expanded_masks = torch.zeros(
            (input_ids.shape[0], target_length), dtype=torch.bool, device=input_ids.device)
        token_positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        for batch_idx, row in enumerate(input_ids):
            row_mask = row.eq(token_idx)
            if min_token_index:
                row_mask[:min_token_index] = False
            image_mask = row.eq(IMAGE_TOKEN_INDEX)
            images_before = image_mask.long().cumsum(dim=0) - image_mask.long()
            expanded_positions = token_positions + images_before * image_expansion
            valid = row_mask & expanded_positions.ge(0) & expanded_positions.lt(target_length)
            expanded_masks[batch_idx, expanded_positions[valid]] = True
        return expanded_masks

    @staticmethod
    def _get_last_hidden_state(output_hidden_states):
        return output_hidden_states[-1] if isinstance(output_hidden_states, (tuple, list)) else output_hidden_states

    def _extract_cls_logits(self, output_hidden_states, input_ids, min_token_index=0):
        last_hidden_state = self._get_last_hidden_state(output_hidden_states)
        cls_token_mask = self._create_expanded_token_mask(
            input_ids, self.cls_token_idx, target_length=last_hidden_state.shape[1],
            min_token_index=min_token_index,
        )
        cls_valid_mask = cls_token_mask.any(dim=1)
        cls_positions = cls_token_mask.float().argmax(dim=1)
        batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
        cls_hidden_states = last_hidden_state[batch_indices, cls_positions]
        cls_logits = self.classification_head(cls_hidden_states)
        return cls_logits, cls_valid_mask

    def _extract_lm_verdict_logits(self, lm_logits, input_ids, min_token_index=0):
        """Return next-token Real/Fake logits at the fixed CLS query position."""
        if self.real_token_idx is None or self.fake_token_idx is None:
            empty = lm_logits.new_zeros((input_ids.shape[0], 2))
            valid = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            return empty, valid
        cls_token_mask = self._create_expanded_token_mask(
            input_ids,
            self.cls_token_idx,
            target_length=lm_logits.shape[1],
            min_token_index=min_token_index,
        )
        valid = cls_token_mask.any(dim=1)
        positions = cls_token_mask.float().argmax(dim=1)
        batch_indices = torch.arange(lm_logits.shape[0], device=lm_logits.device)
        next_token_logits = lm_logits[batch_indices, positions]
        verdict_logits = next_token_logits[:, [self.real_token_idx, self.fake_token_idx]]
        return verdict_logits, valid

    @staticmethod
    def _aggregate_cls_logits(cls_logits, cls_valid_mask, offset):
        if offset is None:
            return cls_logits, cls_valid_mask

        image_logits, image_valid = [], []
        for start, end in zip(offset[:-1].tolist(), offset[1:].tolist()):
            valid = cls_valid_mask[start:end]
            if valid.any():
                image_logits.append(cls_logits[start:end][valid].mean(dim=0))
                image_valid.append(True)
            else:
                image_logits.append(cls_logits[start:end].mean(dim=0))
                image_valid.append(False)
        return torch.stack(image_logits), torch.tensor(image_valid, dtype=torch.bool, device=cls_logits.device)

    def _compute_cls_loss(self, cls_logits, cls_valid_mask, cls_labels, offset):
        if cls_labels is None or not cls_valid_mask.any():
            return cls_logits.sum() * 0.0

        cls_labels = torch.as_tensor(cls_labels, device=cls_logits.device, dtype=torch.long).reshape(-1)
        if offset is not None and cls_labels.numel() == len(offset) - 1:
            repeats = (offset[1:] - offset[:-1]).to(device=cls_logits.device)
            cls_labels = torch.repeat_interleave(cls_labels, repeats)
        if cls_labels.numel() != cls_logits.shape[0]:
            raise ValueError(
                f"cls_labels must contain one label per image or conversation; got {cls_labels.numel()} labels "
                f"for {cls_logits.shape[0]} conversations"
            )
        supervised = cls_valid_mask & cls_labels.ge(0)
        if not supervised.any():
            return cls_logits.sum() * 0.0
        return F.cross_entropy(cls_logits[supervised].float(), cls_labels[supervised]) * self.cls_loss_weight

    def _inference_path(self, input_ids, global_enc_images, attention_masks, offset, bboxes,
                        forensic_evidence_tokens=None):
        global_enc_images = self._prepare_global_enc_image(global_enc_images, offset)
        output = super().forward(
            images=global_enc_images,
            attention_mask=attention_masks,
            input_ids=input_ids,
            output_hidden_states=True,
            bboxes=bboxes,
            forensic_evidence_tokens=forensic_evidence_tokens,
        )
        return output, output.hidden_states

    def _training_path(self, global_enc_images, bboxes, input_ids, labels, attention_masks, offset,
                       forensic_evidence_tokens=None):
        if global_enc_images is not None:
            global_enc_images = self._prepare_global_enc_image(global_enc_images, offset)
        bboxes_list = bboxes

        # A mask-only replay branch deliberately carries no LM targets.  Some
        # transformers versions return NaN for cross entropy when every label
        # is IGNORE_INDEX, so omit labels for that forward and attach an exact
        # differentiable zero instead.  Hidden states/logits remain available
        # to the [SEG] predictor and mask decoder.
        all_text_ignored = labels is not None and not labels.ne(-100).any()
        forward_labels = None if all_text_ignored else labels
        output = super().forward(
            images=global_enc_images, attention_mask=attention_masks, input_ids=input_ids, labels=forward_labels,
            output_hidden_states=True, bboxes=bboxes_list,
            forensic_evidence_tokens=forensic_evidence_tokens, )
        # The optional continuous evidence tokens are inserted beside the
        # original visual tokens by the shared multimodal preparation path.
        expanded_labels = labels if all_text_ignored else output.get("expanded_labels", labels)
        if all_text_ignored:
            output.loss = output.logits.sum() * 0.0
        elif self.per_sample_text_loss_normalization:
            output.loss = per_sample_causal_text_loss(output.logits, expanded_labels)
        output["expanded_labels"] = expanded_labels
        output_hidden_states = output.hidden_states
        return output, output_hidden_states

    def _prepare_global_enc_image(self, global_enc_image, offset):
        global_enc_image_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            global_enc_image_i = global_enc_image[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
            global_enc_image_list.append(global_enc_image_i)
        return torch.cat(global_enc_image_list, dim=0)

    def _extract_projected_seg_predictor_hidden(self, output_hidden_states, input_ids, offset=None,
                                                min_token_index=0):
        """Return projected causal predictor states, grouped per image."""
        last_hidden_state = self._get_last_hidden_state(output_hidden_states)
        per_conversation, positions = extract_seg_predictor_hidden(
            last_hidden_state,
            input_ids,
            self.seg_token_idx,
            image_expansion=575 + int(getattr(self, "forensic_evidence_token_count", 0)),
            min_token_index=min_token_index,
        )
        projected = [self.model.text_hidden_fcs[0](states) for states in per_conversation]
        if offset is None:
            return projected, positions

        grouped = []
        for start, end in zip(offset[:-1].tolist(), offset[1:].tolist()):
            values = projected[start:end]
            grouped.append(
                torch.cat(values, dim=0)
                if values else last_hidden_state.new_empty((0, self.config.out_dim))
            )
        return grouped, positions

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list, label_list, infer=False):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            orig_size = label_list[i] if infer else label_list[i].shape
            if pred_embedding.numel() == 0:
                pred_masks.append(pred_embedding.new_empty((0, *tuple(orig_size))))
                continue
            sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.model.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, )
            # During inference, we have original size list in place of label list
            pred_mask = self.model.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size, )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks

    def _calculate_losses(self, pred_masks, masks_list, output, seg_valid=None):
        loss_components = self._compute_loss_components(pred_masks, masks_list, output, seg_valid)
        return loss_components

    def _compute_loss_components(self, pred_masks, masks_list, output, seg_valid=None):
        # Initialize loss components
        ce_loss = output.loss * self.ce_loss_weight
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
        num_masks = 0
        batch_size = len(masks_list) if masks_list is not None else (len(pred_masks) if pred_masks else 0)
        if seg_valid is None:
            seg_valid = torch.tensor(
                [mask is not None for mask in (masks_list or [])], dtype=torch.bool, device=ce_loss.device
            )
        else:
            seg_valid = torch.as_tensor(seg_valid, dtype=torch.bool, device=ce_loss.device).reshape(-1)
        if seg_valid.numel() != batch_size:
            raise ValueError(f"seg_valid has {seg_valid.numel()} entries for batch size {batch_size}")
        missing_pred_count = 0
        count_mismatch_count = 0
        unexpected_pred_count = 0

        pred_masks = pred_masks or [None] * batch_size
        masks_list = masks_list or [None] * batch_size
        if len(pred_masks) != batch_size or len(masks_list) != batch_size:
            raise ValueError(
                f"Segmentation batch length mismatch: pred={len(pred_masks)} gt={len(masks_list)} batch={batch_size}"
            )
        for batch_idx, (pred_mask, gt_mask, is_valid) in enumerate(zip(pred_masks, masks_list, seg_valid.tolist())):
            pred_count = 0 if pred_mask is None else pred_mask.shape[0]
            if not is_valid:
                unexpected_pred_count += int(pred_count > 0)
                continue
            if gt_mask is None or gt_mask.numel() == 0:
                raise ValueError(f"seg_valid=True sample {batch_idx} has no ground-truth mask")
            if pred_count == 0:
                missing_pred_count += 1
                continue
            if pred_count != gt_mask.shape[0]:
                count_mismatch_count += 1
                continue
            mask_bce_loss += (
                compute_sigmoid_cross_entropy(pred_mask, gt_mask, mask_count=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                calculate_dice_loss(pred_mask, gt_mask, mask_count=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        if missing_pred_count or count_mismatch_count:
            self.seg_missing_pred_total.add_(missing_pred_count)
            self.seg_count_mismatch_total.add_(count_mismatch_count)
            warnings.warn(
                "Fake segmentation supervision was skipped because required [SEG]/pred masks were missing "
                f"or mismatched: missing={missing_pred_count}, count_mismatch={count_mismatch_count}",
                RuntimeWarning,
            )

        # Normalize the losses
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        # Aggregate all loss components
        total_loss = ce_loss + mask_loss
        return {"loss": total_loss, "ce_loss": ce_loss, "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss, "mask_loss": mask_loss,
                "seg_missing_pred_count": torch.tensor(missing_pred_count, device=ce_loss.device),
                "seg_count_mismatch_count": torch.tensor(count_mismatch_count, device=ce_loss.device),
                "seg_unexpected_pred_count": torch.tensor(unexpected_pred_count, device=ce_loss.device), }

    def evaluate(self, global_enc_images, grounding_enc_images, input_ids, resize_list, orig_sizes, max_tokens_new=32,
                 bboxes=None, force_cls_token=True, return_generation_details=False,
                 forensic_evidence_tokens=None):
        with torch.no_grad():
            if hasattr(self, "rine_conditioner"):
                if forensic_evidence_tokens is not None:
                    raise ValueError("C1 RINE evaluation forbids an external evidence token override")
                # Generation bypasses model_forward, so construct the same
                # frozen-Q2 -> trainable-projector token explicitly once and
                # reuse it for autoregressive generation and the full replay.
                forensic_evidence_tokens = self._rine_evidence_token(global_enc_images)
            generation_kwargs = {}
            if self.token_strategy == "fixed_cls_query":
                if self.cls_token_idx is None or not input_ids.eq(self.cls_token_idx).any(dim=1).all():
                    raise ValueError("fixed_cls_query generation requires [CLS] in every input prompt")
            elif force_cls_token and self.cls_token_idx is not None:
                generation_kwargs["forced_decoder_ids"] = [[input_ids.shape[1], self.cls_token_idx]]
            generation_outputs = self.generate(
                images=global_enc_images, input_ids=input_ids, bboxes=bboxes, max_new_tokens=max(2, max_tokens_new),
                num_beams=1, output_hidden_states=False, return_dict_in_generate=True,
                output_scores=return_generation_details, use_cache=True,
                forensic_evidence_tokens=forensic_evidence_tokens, **generation_kwargs)

            generated_output_ids = generation_outputs.sequences
            generated_attention = generated_output_ids.ne(self.config.pad_token_id)
            full_output = super().forward(
                images=global_enc_images,
                attention_mask=generated_attention,
                input_ids=generated_output_ids,
                output_hidden_states=True,
                bboxes=bboxes,
                forensic_evidence_tokens=forensic_evidence_tokens,
            )
            output_hidden_states = full_output.hidden_states

            predicted_embeddings, _ = self._extract_projected_seg_predictor_hidden(
                output_hidden_states, generated_output_ids
            )
            cls_logits, cls_valid_mask = self._extract_cls_logits(
                output_hidden_states,
                generated_output_ids,
                min_token_index=0 if self.token_strategy == "fixed_cls_query" else input_ids.shape[1],
            )
            lm_verdict_logits, lm_verdict_valid = self._extract_lm_verdict_logits(
                full_output.logits,
                generated_output_ids,
                min_token_index=0 if self.token_strategy == "fixed_cls_query" else input_ids.shape[1],
            )
            cls_results = {
                "logits": cls_logits,
                "probabilities": cls_logits.float().softmax(dim=-1),
                "predictions": cls_logits.argmax(dim=-1),
                "valid_mask": cls_valid_mask,
                "lm_verdict_logits": lm_verdict_logits,
                "lm_verdict_probabilities": lm_verdict_logits.float().softmax(dim=-1),
                "lm_verdict_predictions": lm_verdict_logits.argmax(dim=-1),
                "lm_verdict_valid_mask": lm_verdict_valid,
                "cls_lm_agree": cls_logits.argmax(dim=-1).eq(lm_verdict_logits.argmax(dim=-1))
                & cls_valid_mask & lm_verdict_valid,
                "cls_pred": cls_logits.argmax(dim=-1),
                "cls_prob": cls_logits.float().softmax(dim=-1)[:, 1],
                "lm_verdict_pred": lm_verdict_logits.argmax(dim=-1),
                "lm_verdict_prob": lm_verdict_logits.float().softmax(dim=-1)[:, 1],
            }
            if grounding_enc_images is None:
                pred_masks = []
            elif not any(embedding.numel() for embedding in predicted_embeddings):
                last_hidden_state = self._get_last_hidden_state(output_hidden_states)
                pred_masks = [
                    last_hidden_state.new_empty((0, *tuple(original_size)))
                    for original_size in orig_sizes
                ]
            else:
                image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
                # Generate and post-process masks
                pred_masks = self._generate_and_postprocess_masks(
                    predicted_embeddings, image_embeddings, resize_list, orig_sizes, infer=True
                )
        if return_generation_details:
            generation_details = {
                "scores": tuple(generation_outputs.scores),
                "prompt_length": int(input_ids.shape[1]),
                "max_new_tokens": int(max_tokens_new),
                # Read-only Phase 6C.2 hook: retain the generated, ordered
                # predictor states so the frozen L2 autoregressive trajectory
                # can be replayed through one shared R1 rectifier.  This does
                # not alter generation or the P1/SAM mask path.
                "projected_seg_embeddings": [value.detach() for value in predicted_embeddings],
            }
            return generated_output_ids, pred_masks, cls_results, generation_details
        return generated_output_ids, pred_masks, cls_results
