from __future__ import annotations
from typing import Any, Sequence

def geometry_for(source: str, original_hw: Sequence[int]) -> dict[str, Any]:
    """Return the exact forward/inverse geometry without assuming feature H/W."""
    (h, w) = map(int, original_hw)
    if h <= 0 or w <= 0:
        raise ValueError(f'invalid original geometry: {(h, w)}')
    base = {'source': source, 'original_hw': [h, w]}
    if source == 'sam':
        scale = 1024.0 / max(h, w)
        (rh, rw) = (int(h * scale + 0.5), int(w * scale + 0.5))
        return {**base, 'kind': 'resize_longest_pad_right_bottom', 'resized_hw': [rh, rw], 'padding_tlbr': [0, 0, 1024 - rh, 1024 - rw], 'model_input_hw': [1024, 1024], 'inverse': 'remove right/bottom padding then resize logits to original'}
    if source == 'clip':
        (short, long) = (w, h) if w <= h else (h, w)
        (new_short, new_long) = (336, int(336 * long / short))
        (rh, rw) = (new_long, new_short) if w <= h else (new_short, new_long)
        (top, left) = ((rh - 336) // 2, (rw - 336) // 2)
        return {**base, 'kind': 'resize_shortest_center_crop', 'resized_hw': [rh, rw], 'crop_box_yxyx': [top, left, top + 336, left + 336], 'model_input_hw': [336, 336], 'inverse': 'paste crop into resized canvas with outside background, resize to original'}
    raise ValueError(f'unknown feature source: {source}')
