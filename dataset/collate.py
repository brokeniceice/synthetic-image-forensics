import torch
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from tools.utils import DEFAULT_CLS_TOKEN, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN

def _stack_optional_tensors(values, field_name):
    """Stack an all-present field without letting batch order define semantics."""
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f'Mixed presence for {field_name}: each sample must explicitly provide the same tensor field')
    return torch.stack(values, dim=0)

def _truncate_training_batch_preserving_seg(input_ids, targets, attention_masks, tokenizer, truncate_len, target_protocols=None, multiseg_pair_counts=None):
    """Truncate long training rows without ever dropping an existing [SEG]."""
    seg_ids = tokenizer('[SEG]', add_special_tokens=False).input_ids
    if len(seg_ids) != 1:
        raise ValueError('[SEG] must encode as exactly one token')
    seg_id = seg_ids[0]
    (kept_ids, kept_targets) = ([], [])
    preserve_count = 0
    protocols = target_protocols or [None] * len(input_ids)
    pair_counts = multiseg_pair_counts or [0] * len(input_ids)
    phrase_suffix_ids = tokenizer('Target regions:', add_special_tokens=False).input_ids[1:]
    for (ids, labels, attention, protocol, pair_count) in zip(input_ids, targets, attention_masks, protocols, pair_counts):
        valid_len = int(attention.sum().item())
        (ids, labels) = (ids[:valid_len], labels[:valid_len])
        if valid_len > truncate_len:
            if protocol == 'native_multiseg' and int(pair_count) > 0:
                raise ValueError('native_multiseg must be atomically pair-truncated by the forensic adapter before generic token truncation')
            seg_positions = ids.eq(seg_id).nonzero(as_tuple=False).flatten()
            if seg_positions.numel() and int(seg_positions[-1]) >= truncate_len:
                tail_start = int(seg_positions[-1])
                if protocol == 'phrase_aligned':
                    before_seg = ids[:tail_start].tolist()
                    matches = [start for start in range(len(before_seg) - len(phrase_suffix_ids) + 1) if before_seg[start:start + len(phrase_suffix_ids)] == phrase_suffix_ids]
                    if not matches:
                        raise ValueError('phrase_aligned row lacks the Target regions field before [SEG]')
                    tail_start = max(0, matches[-1] - 1)
                tail_len = valid_len - tail_start
                prefix_budget = truncate_len - tail_len
                if prefix_budget <= 0:
                    raise ValueError('Training sequence suffix beginning at [SEG] exceeds truncation budget')
                ids = torch.cat([ids[:prefix_budget], ids[tail_start:]], dim=0)
                labels = torch.cat([labels[:prefix_budget], labels[tail_start:]], dim=0)
                preserve_count += 1
            else:
                (ids, labels) = (ids[:truncate_len], labels[:truncate_len])
        kept_ids.append(ids)
        kept_targets.append(labels)
    padded_ids = torch.nn.utils.rnn.pad_sequence(kept_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_targets = torch.nn.utils.rnn.pad_sequence(kept_targets, batch_first=True, padding_value=IGNORE_INDEX)
    return (padded_ids, padded_targets, padded_ids.ne(tokenizer.pad_token_id), preserve_count)

def custom_collate_fn(batch, tokenizer=None, use_mm_start_end=True, inference=False, local_rank=-1, token_strategy='generated_cls'):
    (image_path_list, global_enc_image_list, grounding_enc_image_list) = ([], [], [])
    (bboxes_list, conversation_list, masks_list) = ([], [], [])
    (label_list, resize_list, questions_list) = ([], [], [])
    (selected_labels_list, cls_labels_list, seg_valid_list, offset_list) = ([], [], [], [0])
    (sample_ids, sources, content_categories) = ([], [], [])
    (prompt_template_ids, prompt_sha256s, target_protocols) = ([], [], [])
    (multiseg_pair_counts, multiseg_ref_indices, multiseg_dropped_pairs) = ([], [], [])
    cnt = 0
    for sample in batch:
        if isinstance(sample, dict):
            image_path = sample['image_path']
            global_enc_image = sample['global_enc_image']
            grounding_enc_image = sample.get('grounding_enc_image')
            bboxes = sample.get('bboxes')
            conversations = sample['conversations']
            masks = sample.get('masks')
            label = sample.get('label')
            resize = sample.get('resize')
            questions = sample.get('questions', [])
            sampled_classes = sample.get('sampled_classes', [])
            cls_label = sample.get('cls_label')
            seg_valid = bool(sample.get('seg_valid', False))
            sample_ids.append(sample.get('sample_id'))
            sources.append(sample.get('source'))
            content_categories.append(sample.get('content_category'))
            prompt_template_ids.append(sample.get('prompt_template_id'))
            prompt_sha256s.append(sample.get('prompt_sha256'))
            target_protocols.append(sample.get('target_protocol'))
            multiseg_pair_counts.append(int(sample.get('multiseg_pair_count', 0)))
            multiseg_ref_indices.append(list(sample.get('multiseg_ref_indices', [])))
            multiseg_dropped_pairs.append(list(sample.get('multiseg_dropped_pairs', [])))
        elif len(sample) == 10:
            (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, resize, questions, sampled_classes) = sample
            cls_label = None
            seg_valid = masks is not None
            sample_ids.append(None)
            sources.append(None)
            content_categories.append(None)
            prompt_template_ids.append(None)
            prompt_sha256s.append(None)
            target_protocols.append(None)
            multiseg_pair_counts.append(0)
            multiseg_ref_indices.append([])
            multiseg_dropped_pairs.append([])
        elif len(sample) == 11:
            (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, resize, questions, sampled_classes, cls_label) = sample
            seg_valid = masks is not None
            sample_ids.append(None)
            sources.append(None)
            content_categories.append(None)
            prompt_template_ids.append(None)
            prompt_sha256s.append(None)
            target_protocols.append(None)
            multiseg_pair_counts.append(0)
            multiseg_ref_indices.append([])
            multiseg_dropped_pairs.append([])
        else:
            raise ValueError(f'Expected a mapping or 10-/11-field dataset sample, got {len(sample)} fields')
        image_path_list.append(image_path)
        global_enc_image_list.append(global_enc_image)
        grounding_enc_image_list.append(grounding_enc_image)
        bboxes_list.append(bboxes)
        conversation_list.extend(conversations)
        masks_list.append(None if masks is None else masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        selected_labels_list.append(sampled_classes)
        cls_labels_list.append(cls_label)
        seg_valid_list.append(seg_valid)
        offset_list.append((cnt := (cnt + len(conversations))))
    if use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        conversation_list = [conv.replace(DEFAULT_IMAGE_TOKEN, replace_token) for conv in conversation_list]
    conv = conversation_lib.default_conversation.copy()
    if token_strategy not in {'generated_cls', 'fixed_cls_query'}:
        raise ValueError(f'Unsupported token strategy: {token_strategy}')
    if not inference or token_strategy == 'fixed_cls_query':
        assistant_prefix = conv.sep + conv.roles[1] + ': '
        cls_prefix = assistant_prefix + DEFAULT_CLS_TOKEN + ' '
        assistant_bare = conv.sep + conv.roles[1] + ':'
        updated_conversations = []
        for conversation in conversation_list:
            conversation = conversation.replace(assistant_prefix, cls_prefix)
            if token_strategy == 'fixed_cls_query' and conversation.endswith(assistant_bare):
                conversation += ' ' + DEFAULT_CLS_TOKEN
            updated_conversations.append(conversation)
        conversation_list = updated_conversations
    input_ids = torch.nn.utils.rnn.pad_sequence([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversation_list], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    seg_ids = tokenizer('[SEG]', add_special_tokens=False).input_ids
    if len(seg_ids) != 1:
        raise ValueError('[SEG] must encode as exactly one token')
    for (index, protocol) in enumerate(target_protocols):
        if protocol != 'native_multiseg' or inference:
            continue
        seg_count = int(input_ids[index].eq(seg_ids[0]).sum())
        mask_count = 0 if masks_list[index] is None else int(masks_list[index].shape[0])
        declared = multiseg_pair_counts[index]
        if not seg_count == mask_count == declared == len(multiseg_ref_indices[index]):
            raise ValueError(f'native_multiseg count mismatch at batch row {index}: phrase={declared} SEG={seg_count} mask={mask_count}')
        if multiseg_ref_indices[index] != sorted(multiseg_ref_indices[index]):
            raise ValueError(f'native_multiseg ref order drift at batch row {index}')
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ': '
    sep2 = conv.sep2
    if inference:
        targets.fill_(IGNORE_INDEX)
    else:
        for (conversation, target) in zip(conversation_list, targets):
            _process_conversation(conversation, target, tokenizer, sep, sep2)
    if token_strategy == 'fixed_cls_query':
        cls_token_ids = tokenizer(DEFAULT_CLS_TOKEN, add_special_tokens=False).input_ids
        if len(cls_token_ids) != 1:
            raise ValueError(f'{DEFAULT_CLS_TOKEN} must encode as exactly one token')
        cls_mask = input_ids.eq(cls_token_ids[0])
        if not cls_mask.any(dim=1).all():
            raise ValueError('fixed_cls_query requires one [CLS] token in every conversation')
        targets[cls_mask] = IGNORE_INDEX
    seg_preserving_truncation_count = 0
    if not inference:
        truncate_len = tokenizer.model_max_length - 575 if tokenizer.model_max_length > 575 else tokenizer.model_max_length
        if input_ids.shape[1] > truncate_len:
            (input_ids, targets, attention_masks, seg_preserving_truncation_count) = _truncate_training_batch_preserving_seg(input_ids, targets, attention_masks, tokenizer, truncate_len, target_protocols=target_protocols, multiseg_pair_counts=multiseg_pair_counts)
    return {'image_paths': image_path_list, 'global_enc_images': torch.stack(global_enc_image_list, dim=0), 'grounding_enc_images': _stack_optional_tensors(grounding_enc_image_list, 'grounding_enc_images'), 'bboxes': None if not any((box is not None for box in bboxes_list)) else bboxes_list, 'input_ids': input_ids, 'labels': targets, 'attention_masks': attention_masks, 'masks_list': masks_list, 'label_list': label_list, 'resize_list': resize_list, 'offset': torch.LongTensor(offset_list), 'questions_list': questions_list, 'sampled_classes_list': selected_labels_list, 'cls_labels': None if all((label is None for label in cls_labels_list)) else torch.tensor([-100 if label is None else int(label) for label in cls_labels_list], dtype=torch.long), 'seg_valid': torch.tensor(seg_valid_list, dtype=torch.bool), 'inference': inference, 'conversation_list': conversation_list, 'sample_ids': sample_ids, 'sources': sources, 'content_categories': content_categories, 'prompt_template_ids': prompt_template_ids, 'prompt_sha256s': prompt_sha256s, 'token_strategy': token_strategy, 'seg_preserving_truncation_count': seg_preserving_truncation_count, 'multiseg_pair_counts': multiseg_pair_counts, 'multiseg_ref_indices': multiseg_ref_indices, 'multiseg_dropped_pairs': multiseg_dropped_pairs}

def _process_conversation(conversation, target, tokenizer, sep, sep2):
    total_len = target.ne(tokenizer.pad_token_id).sum().item()
    rounds = conversation.split(sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX
    for rou in rounds:
        if not rou:
            break
        parts = rou.split(sep)
        assert len(parts) == 2, (len(parts), rou)
        parts[0] += sep
        if DEFAULT_IMAGE_TOKEN in conversation:
            round_len = len(tokenizer_image_token(rou, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
        else:
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
        target[cur_len:cur_len + instruction_len] = IGNORE_INDEX
        cur_len += round_len
    target[cur_len:] = IGNORE_INDEX
    if cur_len < tokenizer.model_max_length:
        assert cur_len == total_len
