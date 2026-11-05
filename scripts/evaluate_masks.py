#!/usr/bin/env python3
"""Evaluate original-size binary masks against reference union masks."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True, help='JSONL: sample_id, prediction_mask, target_mask; paths relative to manifest')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    from PIL import Image
    records, seen = [], set()
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line); sid = str(row['sample_id'])
        if sid in seen:
            raise ValueError('Duplicate sample: ' + sid)
        seen.add(sid)
        pred = np.asarray(Image.open(args.manifest.parent / row['prediction_mask']).convert('L')) > 0
        target = np.asarray(Image.open(args.manifest.parent / row['target_mask']).convert('L')) > 0
        if pred.shape != target.shape:
            raise ValueError('Mask shape mismatch: ' + sid)
        tp = int((pred & target).sum()); fp = int((pred & ~target).sum()); fn = int((~pred & target).sum()); tn = int((~pred & ~target).sum())
        valid = bool(row.get('valid_q_seg', True))
        fg = (tp/(tp+fp+fn) if tp+fp+fn else 1.) if valid else 0.
        bg = tn/(tn+fp+fn) if tn+fp+fn else 1.
        records.append(dict(sample_id=sid, tp=tp, fp=fp, fn=fn, tn=tn, iou=fg, miou=(fg+bg)/2,
                            valid_q_seg=valid, f1=(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.) if valid else 0.))
    if not records:
        p.error('Manifest is empty')
    tp, fp, fn, tn = (sum(r[k] for r in records) for k in ('tp','fp','fn','tn'))
    result = dict(n=len(records), mean_foreground_iou=sum(r['iou'] for r in records)/len(records),
                  mean_foreground_f1=sum(r['f1'] for r in records)/len(records),
                  global_foreground_iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                  global_foreground_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                  mean_fg_bg_miou=sum(r['miou'] for r in records)/len(records),
                  global_fg_bg_miou=((tp/(tp+fp+fn) if tp+fp+fn else 1.)+(tn/(tn+fp+fn) if tn+fp+fn else 1.))/2,
                  invalid_queries=sum(not r['valid_q_seg'] for r in records),
                  confusion_pixels=dict(tp=tp,fp=fp,fn=fn,tn=tn))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
