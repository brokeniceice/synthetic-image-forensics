#!/usr/bin/env python3
"""Score raw and fixed TRAIN-centered classification margins."""
import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True, help='JSONL with sample_id and label (Real=0, Fake=1)')
    p.add_argument('--center', type=float, default=-9.190834884678093)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    def read(path):
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        by_id = {str(r['sample_id']): r for r in rows}
        if len(rows) != len(by_id) or not rows:
            raise ValueError('Empty or duplicate sample IDs: ' + str(path))
        return by_id
    pred, refs = read(args.predictions), read(args.references)
    if set(pred) != set(refs):
        raise ValueError('Prediction/reference populations differ')
    scores, labels = [], []
    for sid, row in pred.items():
        score = float(row['margin']); label = refs[sid]['label']
        if not math.isfinite(score) or label not in (0, 1):
            raise ValueError('Invalid margin or label: ' + sid)
        scores.append(score); labels.append(label)
    def summarize(center):
        tp=fp=tn=fn=0
        for score, label in zip(scores, labels):
            decision = int(score > center)
            tp += int(decision == 1 and label == 1); fp += int(decision == 1 and label == 0)
            tn += int(decision == 0 and label == 0); fn += int(decision == 0 and label == 1)
        return dict(n=len(scores), accuracy=(tp+tn)/len(scores), f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                    tp=tp,fp=fp,tn=tn,fn=fn,threshold=center)
    result = dict(raw=summarize(0.), center=summarize(args.center))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
