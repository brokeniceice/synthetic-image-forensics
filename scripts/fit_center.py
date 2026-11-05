#!/usr/bin/env python3
"""Fit the label-free mean TRAIN margin; never optimize a VAL/test metric."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-scores', type=Path, required=True, help='JSONL with sample_id and margin, or two logits ordered [Real, Fake]')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    seen, values = set(), []
    with args.train_scores.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line); sid = str(row['sample_id'])
            if sid in seen:
                raise ValueError('Duplicate TRAIN sample: ' + sid)
            seen.add(sid)
            value = float(row['margin']) if 'margin' in row else float(row['logits'][1]) - float(row['logits'][0])
            if not math.isfinite(value):
                raise ValueError('Nonfinite TRAIN margin')
            values.append(value)
    if not values:
        p.error('TRAIN scores are empty')
    result = dict(n=len(values), center=math.fsum(values)/len(values),
        definition='mean TRAIN (Fake - Real) logit margin; label-free; no threshold search',
        score_file_sha256=hashlib.sha256(args.train_scores.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
