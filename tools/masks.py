from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch.nn.functional as F
from eval.metrics import compute_binary_mask_metrics

def spatial_cache_paths(spatial_root: Path, source: str, split: str):
    if source not in ('clip', 'sam'):
        raise ValueError(f'unsupported spatial cache source: {source}')
    root = spatial_root / 'cache' / source / split
    complete = json.loads((root / 'complete.json').read_text())
    if complete['status'] != 'COMPLETE' or not complete['source_parameter_hash_exact']:
        raise RuntimeError(f'invalid frozen {source.upper()} cache: {root}')
    paths = sorted(root.glob('shard_*.pt'))
    if len(paths) != int(complete['shards']):
        raise RuntimeError(f'{source.upper()} cache shard-count mismatch')
    return paths

def file_sha256(path: Path):
    d = hashlib.sha256()
    with path.open('rb') as h:
        for block in iter(lambda : h.read(8 << 20), b''):
            d.update(block)
    return d.hexdigest()

def inverse_sam_logits(low, geometry):
    model = F.interpolate(low.float(), size=(1024, 1024), mode='bilinear', align_corners=False)
    (rh, rw) = geometry['resized_hw']
    canvas = model[..., :rh, :rw]
    return F.interpolate(canvas, size=tuple(geometry['original_hw']), mode='bilinear', align_corners=False)[0, 0]

def metric_record(sample_id, logits, target):
    m = compute_binary_mask_metrics(logits.detach().cpu(), target.detach().cpu())
    pred = logits.detach().cpu().gt(0)
    truth = target.detach().cpu().bool()
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    return {'sample_id': sample_id, 'foreground_iou': float(m['image_iou']), 'foreground_f1': float(m['image_pixel_f1']), 'tp': tp, 'fp': fp, 'fn': fn}
