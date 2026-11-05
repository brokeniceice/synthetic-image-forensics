"""Token replay helpers shared with the frozen evaluation implementation."""
import hashlib
import re

BUDGET = 400

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def append_seg(prompt, generated, *, seg_id, eos_id, pad_id):
    """Keep the original G0 answer before its terminator; append exactly one SEG."""
    require(prompt and seg_id not in prompt and eos_id not in prompt, 'Invalid G0 prompt boundary')
    generated = list(generated)
    end = generated.index(eos_id) if eos_id in generated else len(generated)
    removed = generated[end:]
    require(all(token in (eos_id, pad_id) for token in removed), 'Unexpected nonpadding text after EOS')
    retained = generated[:end]
    full = list(prompt) + retained + [seg_id]
    positions = [i for i, value in enumerate(full) if value == seg_id]
    require(positions[-1] == len(full) - 1 and full[:-1] == list(prompt) + retained,
            'Append changed the retained G0 prefix')
    return full, {
        'generated_ids_retained': retained, 'removed_terminal_ids': removed,
        'terminal_eos_removed': bool(removed), 'injection_position_text': len(full) - 1,
        'natural_seg_positions_text': positions[:-1], 'all_seg_positions_text': positions,
        'natural_seg_count': len(positions) - 1, 'appended_seg_count': 1,
        'appended_query_is_forced': True, 'text_was_regenerated': False,
    }

def expected_predictor_positions(input_ids, seg_id, image_id, image_expansion):
    return [i - 1 + input_ids[:i].count(image_id) * image_expansion
            for i, token in enumerate(input_ids) if token == seg_id]

def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()

def text_diagnostics(text, token_ids, eos_id):
    lower = text.lower()
    phrases = ('no identifiable synthetic artifact evidence', 'no synthetic artifacts',
               'no evidence of manipulation', 'no signs of manipulation', 'appears authentic',
               'image is real', 'image is authentic')
    lexical = re.findall(r"[a-z0-9']+", lower)
    grams = [tuple(lexical[i:i + 4]) for i in range(max(0, len(lexical) - 3))]
    repeated = 1 - len(set(grams)) / len(grams) if grams else 0.0
    max_repeats, repeated_block = 1, None
    for width in range(1, min(32, len(token_ids) // 2) + 1):
        for start in range(len(token_ids) - 2 * width + 1):
            block = token_ids[start:start + width]
            count = 1
            while token_ids[start + count * width:start + (count + 1) * width] == block:
                count += 1
            if count > max_repeats:
                max_repeats, repeated_block = count, block
    return {'negative_phrases_found': [p for p in phrases if p in lower],
            'contains_real_token_text': '[REAL]' in text, 'word_count': len(lexical),
            'fourgram_repeat_fraction': repeated, 'max_consecutive_token_block_repeats': max_repeats,
            'most_repeated_token_block': repeated_block,
            'possible_repetition': repeated >= 0.3 or max_repeats >= 4,
            'truncated': len(token_ids) >= BUDGET and (not token_ids or token_ids[-1] != eos_id),
            'ends_with_eos': bool(token_ids and token_ids[-1] == eos_id),
            'semantic_grounding': 'NOT_HUMAN_REVIEWED; token suppression/coverage alone is not success'}

