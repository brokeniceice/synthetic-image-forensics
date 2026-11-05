"""Frozen C1 classification and P1-decoder spatial inference.

Localization reproduces known-Fake G1; explanation reproduces autonomous E03 G0.
Classification uses the canonical prompt and does not require labels or masks.
"""
from pathlib import Path
import copy
import json
import torch
import yaml
from transformers import CLIPImageProcessor
from model.llava import conversation as conv
from model.SAM.utils.transforms import ResizeLongestSide
from model.csculf import CSCULF
from model.clip_forensic_adapter import CLIPSpatialArm
from model.sam_forensic_rectifier import FrozenP1SAMPath, GeometryAwareSAMRectifier
from eval.backend import GLaMMForensicsBackend
from forensics.checkpoints import load_c1
from forensics.generation import (require, append_seg, expected_predictor_positions,
                                  tensor_sha, text_diagnostics)

QUESTION = 'Determine whether this image is authentic and explain the forensic evidence.'
TASKS = dict(localization=dict(budget=400, question=QUESTION),
             explanation=dict(budget=512, question='Determine whether this image is real or synthetic. Explain the visible evidence in English, describing any synthetic artifacts and their locations. If synthetic artifacts are present, mask them.'))
TRAIN_CENTER = -9.190834884678093


class Runtime:
    def __init__(self, config, c1_checkpoint=None, spatial_bundle=None, device='cuda:0', *, packed_weights=None, complete_package=None):
        self.device = torch.device(device)
        require(self.device.type == 'cuda', 'BF16 inference requires a CUDA device')
        torch.cuda.set_device(self.device)
        torch.set_num_threads(4)
        torch.manual_seed(3407)
        torch.cuda.manual_seed_all(3407)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.cfg = copy.deepcopy(config) if isinstance(config, dict) else yaml.safe_load(Path(config).read_text())
        conv.default_conversation = conv.conv_templates['llava_v1']
        if complete_package:
            from forensics.pretrained import load_backbone
            directory, manifest = complete_package
            self.model, self.tokenizer, self.identity = load_backbone(self.cfg, directory, manifest, self.device)
        else:
            self.model, self.tokenizer, self.identity = load_c1(self.cfg, c1_checkpoint, self.device,
                rine_payload=packed_weights['rine'] if packed_weights else None,
                source_hashes=packed_weights['source_hashes'] if packed_weights else None)
        self.owner = self.model.evaluate.__self__
        require(self.owner is self.model.get_base_model(), 'Wrong generation owner')
        require(self.owner.forensic_evidence_token_count == 1 and self.owner.token_strategy == 'fixed_cls_query', 'C1 token strategy drift')
        self.backend = GLaMMForensicsBackend(self.model, self.tokenizer, device=self.device, dtype=torch.bfloat16)
        self.processor = CLIPImageProcessor.from_pretrained(self.cfg['model']['vision_tower'], local_files_only=True)
        self.transform = ResizeLongestSide(self.cfg['model']['image_size'])
        self.base_generation = copy.deepcopy(self.owner.generation_config)
        for key in ('forced_bos_token_id', 'forced_eos_token_id', 'forced_decoder_ids', 'suppress_tokens', 'begin_suppress_tokens'):
            require(getattr(self.base_generation, key, None) is None, 'Unexpected generation constraint: ' + key)
        require(getattr(self.base_generation, 'min_length', 0) == 0 and getattr(self.base_generation, 'min_new_tokens', None) in (None, 0), 'EOS is not free')
        require(getattr(self.base_generation, 'repetition_penalty', 1.0) == 1.0 and getattr(self.base_generation, 'no_repeat_ngram_size', 0) == 0, 'Unexpected repetition constraint')
        self.control_ids = [self.owner.real_token_idx, self.owner.cls_token_idx, self.owner.fake_token_idx]
        require(len(set(self.control_ids)) == 3 and self.owner.seg_token_idx not in self.control_ids and self.tokenizer.eos_token_id not in self.control_ids, 'Invalid suppression tokens')
        self.spatial_ready = spatial_bundle is not None or packed_weights is not None or complete_package is not None
        if self.spatial_ready:
            if complete_package:
                selected = manifest['spatial_config']
            else:
                selected = packed_weights['spatial'] if packed_weights is not None else torch.load(spatial_bundle, map_location='cpu', weights_only=False)
                require(selected['schema'] == 'paper_spatial_bundle_v1', 'Unknown spatial bundle')
                require(selected['c1_sha256'] == self.identity['c1_sha256'], 'C1/spatial checkpoint linkage mismatch')
                expected = json.loads((Path(__file__).resolve().parents[1] / 'configs/paper/protocol.json').read_text())['checkpoint_sha256']
                require(selected['source_hashes']['r1'] == expected['r1'] and selected['source_hashes']['forensic_source'] == expected['forensic_source'], 'Spatial bundle differs from final NoRelative assets')
            self.utility = CSCULF().to(self.device)
            self.rectifier = GeometryAwareSAMRectifier(selected['gamma'], heads=8, locality_sigma=0.25).to(self.device)
            self.sam = FrozenP1SAMPath()
            self.sam.to(device=self.device, dtype=torch.bfloat16)
            self.source = CLIPSpatialArm(blocks=3).to(self.device)
            if complete_package:
                from forensics.pretrained import load_module
                for name in ('utility', 'rectifier', 'sam', 'source'):
                    load_module(directory, getattr(self, name), name + '.', self.device)
            else:
                self.utility.load_state_dict(selected['utility_state'], strict=True)
                self.rectifier.load_state_dict(selected['rectifier_state'], strict=True)
                self.sam.prompt_encoder.load_state_dict(selected['prompt_encoder'], strict=True)
                self.sam.mask_decoder.load_state_dict(selected['mask_decoder'], strict=True)
                self.source.load_state_dict(selected['forensic_source_state'], strict=True)
            for module in (self.utility, self.rectifier, self.sam, self.source):
                module.eval().requires_grad_(False)
            if not complete_package:
                self.identity['spatial_sources'] = selected['source_hashes']

    @torch.inference_mode()
    def classify_many(self, rows, center=TRAIN_CENTER):
        import cv2
        images = []
        for row in rows:
            bgr = cv2.imread(row['image_path'], cv2.IMREAD_COLOR)
            require(bgr is not None, 'Image decode failed: ' + row['image_path'])
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pixels = self.processor(images=images, return_tensors='pt')['pixel_values']
        samples = [dict(image_path=row['image_path'], sample_id=row['sample_id'],
            global_enc_image=pixel, grounding_enc_image=None, bboxes=None,
            conversations=[''], masks=None, label=None, resize=None,
            questions=[], sampled_classes=[], cls_label=0, seg_valid=False,
            source='image_only', content_category=None, manifest_row={})
            for row, pixel in zip(rows, pixels)]
        batch = self.backend._batch_many(samples, '', question=QUESTION)
        batch['grounding_enc_images'] = None
        output = self.model.model_forward(**batch)
        # Preserve archived arithmetic: subtract in native BF16, then cast to FP32.
        margins = (output['cls_logits'][:, 1] - output['cls_logits'][:, 0]).float().cpu().tolist()
        scores = output['cls_logits'].float().softmax(-1)[:, 1].cpu().tolist()
        return [dict(sample_id=row['sample_id'], margin=margin, score_fake=score,
                     prediction_raw=int(margin > 0), center=float(center),
                     prediction_centered=int(margin > center), prompt=QUESTION)
                for row, margin, score in zip(rows, margins, scores)]

    def sample(self, row):
            import torch, cv2
            from dataset.forensics.unified import UnifiedForensicsDataset as U
            bgr = cv2.imread(row['image_path']); require(bgr is not None, 'Image decode failed')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); hw = rgb.shape[:2]; resized = self.transform.apply_image(rgb)
            # Image and geometry only. Placeholder labels/masks contain no annotation.
            return dict(image_path=row['image_path'], sample_id=row['sample_id'],
                global_enc_image=self.processor.preprocess(rgb, return_tensors='pt')['pixel_values'][0],
                grounding_enc_image=U.grounding_enc_processor(torch.from_numpy(resized).permute(2, 0, 1).contiguous()),
                bboxes=None, conversations=[], masks=torch.zeros((1, *hw)), label=torch.full(hw, U.IGNORE_LABEL, dtype=torch.long),
                resize=resized.shape[:2], questions=[], sampled_classes=[], cls_label=0, seg_valid=False,
                manifest_row={'sample_id': row['sample_id'], 'explanation': ''}, source='image_only', content_category=None)

    @torch.inference_mode()
    def infer(self, row, task):
            import torch
            from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
            from tools.utils import IMAGE_TOKEN_INDEX
            require(self.spatial_ready, 'Localization requires a complete model package')
            spec = TASKS[task]; sample = self.sample(row)
            batch = self.backend._batch_many([sample], '[FAKE]', question=spec['question'], continue_assistant=True)
            prompt = batch['input_ids'][0].cpu().tolist()
            require(prompt[-2:] == [self.owner.cls_token_idx, self.owner.fake_token_idx] and prompt.count(IMAGE_TOKEN_INDEX) == 1, 'G1 prompt boundary drift')
            evidence = self.owner._rine_evidence_token(batch['global_enc_images'])
            require(bool(torch.isfinite(evidence).all()), 'Invalid RINE token')
            evidence_hash = tensor_sha(evidence)
            generation = copy.deepcopy(self.base_generation); generation._from_model_config = False
            result = self.owner.generate(images=batch['global_enc_images'], input_ids=batch['input_ids'], bboxes=batch['bboxes'],
                forensic_evidence_tokens=evidence, generation_config=generation, do_sample=False, num_beams=1,
                suppress_tokens=self.control_ids, max_new_tokens=spec['budget'], output_hidden_states=False,
                return_dict_in_generate=True, output_scores=True, use_cache=True)
            suffix = result.sequences[0, len(prompt):].cpu().tolist()
            require(suffix and suffix == [int(score[0].argmax()) for score in result.scores], 'Actual greedy argmax mismatch')
            require(all(bool(torch.isneginf(score[:, self.control_ids]).all()) for score in result.scores), 'Suppression bypassed')
            require(not any(i in self.control_ids for i in suffix), 'Suppressed token emitted')
            require(bool(torch.isfinite(result.scores[0][0, [self.tokenizer.eos_token_id, self.owner.seg_token_idx]]).all()), 'EOS/SEG not free')
            text = self.tokenizer.decode(suffix, skip_special_tokens=False).strip()
            nseg = suffix.count(self.owner.seg_token_idx)
            replay_ids = prompt + suffix
            injection = dict(appended_seg_count=0, appended_query_is_forced=False, natural_seg_count=nseg,
                             text_was_regenerated=False, removed_terminal_ids=[])
            if nseg == 0:
                replay_ids, injection = append_seg(prompt, suffix, seg_id=self.owner.seg_token_idx,
                    eos_id=self.tokenizer.eos_token_id, pad_id=self.owner.config.pad_token_id)
                require(injection['natural_seg_count'] == 0 and injection['appended_seg_count'] == 1, 'Bad fallback')
            del result
            sequence = torch.tensor([replay_ids], dtype=torch.long, device=self.device)
            replay = LlavaLlamaForCausalLM.forward(self.owner, images=batch['global_enc_images'], input_ids=sequence,
                attention_mask=sequence.ne(self.owner.config.pad_token_id), output_hidden_states=True,
                bboxes=batch['bboxes'], forensic_evidence_tokens=evidence, use_cache=False, return_dict=True)
            queries, positions = self.owner._extract_projected_seg_predictor_hidden(replay.hidden_states, sequence)
            q = queries[0].detach()
            expected = expected_predictor_positions(replay_ids, self.owner.seg_token_idx, IMAGE_TOKEN_INDEX, 576)
            require(positions[0].tolist() == expected, 'Query predictor geometry mismatch')
            require(self.owner._get_last_hidden_state(replay.hidden_states).shape[1] == len(replay_ids) + 576, 'RINE expansion mismatch')
            require(len(q) == (nseg or 1) and q.ndim == 2 and bool(torch.isfinite(q).all()), 'Invalid localization queries')
            require(tensor_sha(evidence) == evidence_hash, 'RINE replay changed')
            del replay, queries
            binary = self.native(sample, q)
            meta = dict(generated_text=text, generated_ids=suffix, generated_token_ids=suffix, generated_token_count=len(suffix),
                prefix_input_count=1, answer_token_count=len(suffix) + 1, generated_budget_scope='continuation_only',
                truncated=len(suffix) >= spec['budget'] and suffix[-1] != self.tokenizer.eos_token_id,
                prompt=spec['question'], prompt_token_ids=prompt, generation_mode='G1_suppress_control_tokens_missing_SEG_fallback',
                inference_condition='known-FAKE conditional', budget=spec['budget'], natural_seg_count=nseg,
                natural_seg_triggered=nseg > 0, seg_triggered=True, seg_count=len(q), valid_q_seg=True,
                fallback_applied=nseg == 0, forced_seg_count=int(nseg == 0), has_pred_mask=True, status='OK',
                appended_token_changes_original_text=False, replay_token_ids=replay_ids, injection=injection,
                query_sha256=tensor_sha(q), query_shape=list(q.shape), predictor_positions=expected, rine_token_sha256=evidence_hash,
                actual_generate=dict(generate_calls=1, suppression_verified_all_step_scores=True,
                    greedy_argmax_equals_returned_tokens=True, suppress_tokens=self.control_ids,
                    eos_free=True, seg_free=True, do_sample=False, num_beams=1, score_steps=len(suffix)),
                diagnostics=text_diagnostics(text, suffix, self.tokenizer.eos_token_id))
            meta['diagnostics']['truncated'] = meta['truncated']
            meta['diagnostics']['max_new_tokens'] = spec['budget']
            meta['diagnostics']['budget_scope'] = 'continuation_only; FAKE prefix is input'
            return meta, binary, q.cpu()

    def native(self, sample, qvalues):
            import torch
            from model.evidence import sam_lowres_to_original_normalized as align
            from forensics import spatial as hc, spatial as ha
            from forensics.spatial import clip_grid
            from tools.geometry import geometry_for
            from tools.spatial_metrics import sam_coordinates, clip_coordinates
            from tools.masks import inverse_sam_logits
            device = self.device; hw = tuple(sample['label'].shape)
            raw = self.model.get_grounding_encoder_embs(sample['grounding_enc_image'][None].to(device, dtype=torch.bfloat16))
            ct, _ = self.model.get_model().get_vision_tower()(sample['global_enc_image'][None].to(device, dtype=torch.bfloat16))
            with torch.autocast('cuda', dtype=torch.bfloat16): forensic = self.source(clip_grid(ct), return_features=True)
            sg, cg = geometry_for('sam', hw), geometry_for('clip', hw)
            sc, cc = sam_coordinates(sg, grid=64)[None].to(device), clip_coordinates(cg, grid=24)[None].to(device)
            s64 = align(raw, sg, output_hw=(64, 64)).to(torch.bfloat16)
            f24, zf = forensic['F_forensic'].to(torch.bfloat16), forensic['logits'].to(torch.bfloat16)
            with torch.autocast('cuda', dtype=torch.bfloat16): corrected = self.rectifier(raw, f24, sc, cc, torch.ones(1, 576, dtype=torch.bool, device=device))
            masks = []
            for query in qvalues:
                query = query[None].to(device, dtype=torch.bfloat16)
                with torch.autocast('cuda', enabled=False): low0 = self.sam(query, raw.to(torch.bfloat16))
                batch = dict(S64=s64, q_seg=query, z_L=align(low0, sg, output_hw=(256, 256)).to(torch.bfloat16),
                    F24=f24, z_F24=zf, clip_geometries=[cg], valid_g0=torch.ones(1, dtype=torch.bool, device=device),
                    forensic_present=torch.ones(1, dtype=torch.bool, device=device), forensic_vacuous=torch.zeros(1, dtype=torch.bool, device=device), forensic_off=torch.zeros(1, dtype=torch.bool, device=device))
                u = hc.utility_forward(self.utility, batch)
                gate = ha.gate_to_sam_grid(u['U'], sc) * corrected['support'].reshape(1, 1, 64, 64).float()
                require(bool(torch.isfinite(gate).all()) and bool(((gate >= 0) & (gate <= 1)).all()), 'Invalid gate')
                adapted = ha.gated_embedding(raw, corrected['image_embeddings'], gate)
                with torch.autocast('cuda', enabled=False): low = self.sam(query, adapted.to(torch.bfloat16))
                require(bool(torch.isfinite(low).all()), 'Nonfinite localization logits')
                masks.append(inverse_sam_logits(low, sg))
            return torch.stack(masks).amax(0).gt(0).cpu().numpy().reshape(hw)

    @torch.inference_mode()
    def explain(self, row):
        import eval.backend as evaluation
        question = TASKS['explanation']['question']
        sample = self.sample(row)
        sample['questions'] = [question]
        old_question = evaluation.UNIFIED_FORENSICS_QUESTION
        old_budget = self.backend.max_new_tokens
        old_sampling = self.model.generation_config.do_sample
        try:
            evaluation.UNIFIED_FORENSICS_QUESTION = question
            self.backend.max_new_tokens = 512
            self.model.generation_config.do_sample = False
            out = self.backend.generate_explanation_batch([sample])[0]
        finally:
            evaluation.UNIFIED_FORENSICS_QUESTION = old_question
            self.backend.max_new_tokens = old_budget
            self.model.generation_config.do_sample = old_sampling
        tokens = out['generated_token_ids']
        return dict(sample_id=row['sample_id'], generated_text=out['generated_text'],
            generated_ids=tokens, generated_token_count=len(tokens),
            truncated=len(tokens) >= 512 and tokens[-1] != self.tokenizer.eos_token_id,
            prompt=question, prompt_token_ids=out['prompt_token_ids'],
            generation_mode='G0_autonomous_E03', budget=512,
            inference_condition='no GT verdict prefix; no control-token suppression')

    @classmethod
    def from_pretrained(cls, model_id_or_path, *, revision=None, device='cuda:0', local_files_only=False):
        from forensics.checkpoints import file_sha256
        directory = Path(model_id_or_path).expanduser()
        if not directory.is_dir():
            from huggingface_hub import snapshot_download
            directory = Path(snapshot_download(repo_id=str(model_id_or_path), revision=revision,
                                              local_files_only=local_files_only))
        manifest = json.loads((directory / 'config.json').read_text())
        if manifest.get('format_version') == 2:
            require(manifest['model_type'] == 'synthetic_image_forensics', 'Unsupported model package')
            from huggingface_hub import snapshot_download
            clip = manifest['clip']
            clip_directory = snapshot_download(repo_id=clip['repo_id'], revision=clip['revision'],
                allow_patterns=['config.json', 'preprocessor_config.json', 'pytorch_model.bin'],
                local_files_only=local_files_only)
            config = copy.deepcopy(manifest['backbone_config'])
            config['model']['version'] = str(directory)
            config['model']['vision_tower'] = clip_directory
            runtime = cls(config, device=device, complete_package=(directory, manifest))
            runtime.identity['model_package'] = str(directory)
            runtime.identity['clip'] = clip
            return runtime
        require(manifest['model_type'] == 'synthetic_image_forensics' and manifest['format_version'] == 1, 'Unsupported model package')
        require(file_sha256(directory / 'forensics.pt') == manifest['weights_sha256'], 'Model package checksum mismatch')
        expected_protocol = json.loads((Path(__file__).resolve().parents[1] / 'configs/paper/protocol.json').read_text())
        require(manifest['protocol'] == expected_protocol, 'Model package protocol differs from the paper model')
        weights = torch.load(directory / 'forensics.pt', map_location='cpu', weights_only=False)
        require(weights['source_hashes'] == expected_protocol['checkpoint_sha256'], 'Model package source identity mismatch')
        config = manifest['backbone_config']
        config['model']['version'] = str(directory / 'base_model')
        config['model']['vision_tower'] = str(directory / 'clip')
        runtime = cls(config, weights['c1'], device=device, packed_weights=weights)
        runtime.identity['model_package'] = str(directory)
        runtime.identity['package_weights_sha256'] = manifest['weights_sha256']
        return runtime
