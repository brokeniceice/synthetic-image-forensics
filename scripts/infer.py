#!/usr/bin/env python3
"""Image-only classification, conditional localization, or autonomous explanation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Hugging Face model repository ID or downloaded model directory')
    p.add_argument('--revision', default=None, help='Optional Hugging Face commit/tag')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--task', choices=['classification', 'localization', 'explanation'], required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--manifest', type=Path, help='JSONL with unique sample_id and image_path; relative images resolve against manifest directory')
    p.add_argument('--output', type=Path, required=True, help='A new output directory')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--classification-batch-size', type=int, default=8)
    p.add_argument('--center', type=float, default=-9.190834884678093)
    args = p.parse_args()
    if args.classification_batch_size < 1:
        p.error('Batch size must be positive')
    if args.image:
        rows = [dict(sample_id=args.image.stem, image_path=str(args.image.resolve()))]
    else:
        rows = [json.loads(s) for s in args.manifest.read_text().splitlines() if s.strip()]
        for row in rows:
            image = Path(row['image_path'])
            row['image_path'] = str((args.manifest.parent / image).resolve() if not image.is_absolute() else image)
    ids = [str(row['sample_id']) for row in rows]
    if len(ids) != len(set(ids)) or not ids:
        p.error('Input sample IDs must be unique and nonempty')
    for row in rows:
        if not Path(row['image_path']).is_file():
            p.error('Missing image: ' + row['image_path'])
    args.output.mkdir(parents=True, exist_ok=False)
    from forensics.runtime import Runtime
    runtime = Runtime.from_pretrained(args.model, revision=args.revision, device=args.device, local_files_only=args.local_files_only)
    (args.output / 'run.json').write_text(json.dumps(dict(task=args.task, identity=runtime.identity,
        condition={'classification': 'canonical classification', 'localization': 'known-Fake conditional G1', 'explanation': 'autonomous G0 E03'}[args.task],
        center=args.center, classification_batch_size=args.classification_batch_size, n=len(rows)), indent=2) + '\n')
    from PIL import Image
    import numpy as np
    with (args.output / 'predictions.jsonl').open('w') as handle:
        if args.task == 'classification':
            for start in range(0, len(rows), args.classification_batch_size):
                part = rows[start:start+args.classification_batch_size]
                for result in runtime.classify_many(part, center=args.center):
                    handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                handle.flush()
                print(json.dumps(dict(done=start+len(part),total=len(rows))),flush=True)
            return
        for i, row in enumerate(rows):
            if args.task == 'explanation':
                result = runtime.explain(row)
            else:
                result, mask, _ = runtime.infer(row, args.task)
                result['sample_id'] = row['sample_id']
                name = hashlib.sha256(str(row['sample_id']).encode()).hexdigest() + '.png'
                (args.output / 'masks').mkdir(exist_ok=True)
                Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(args.output / 'masks' / name)
                result['mask_path'] = 'masks/' + name
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
            print(json.dumps(dict(done=i+1, total=len(rows), sample_id=row['sample_id'])), flush=True)


if __name__ == '__main__':
    main()
