#!/usr/bin/env python3
"""Generate predictions for the classification, localization or explanation experiment."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forensics.cli import make_parser, read_inputs, save_mask


def main():
    parser = make_parser(__doc__)
    parser.add_argument('--task', choices=['classification', 'localization', 'explanation'], required=True)
    parser.add_argument('--classification-batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.classification_batch_size < 1:
        parser.error('Batch size must be positive')
    rows = read_inputs(args, parser)
    args.output.mkdir(parents=True, exist_ok=False)
    from forensics.runtime import Runtime
    runtime = Runtime.from_pretrained(args.model, revision=args.revision,
        device=args.device, local_files_only=args.local_files_only)
    (args.output / 'run.json').write_text(json.dumps(dict(task=args.task, n=len(rows),
        center=args.center, classification_batch_size=args.classification_batch_size), indent=2) + '\n')
    with (args.output / 'predictions.jsonl').open('w') as handle:
        if args.task == 'classification':
            for start in range(0, len(rows), args.classification_batch_size):
                part = rows[start:start + args.classification_batch_size]
                for result in runtime.classify_many(part, center=args.center):
                    handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                handle.flush()
                print(json.dumps(dict(done=start + len(part), total=len(rows))), flush=True)
            return
        for i, row in enumerate(rows):
            if args.task == 'explanation':
                generated = runtime.explain(row)
                result = dict(sample_id=row['sample_id'], generated_text=generated['generated_text'],
                              truncated=generated['truncated'])
            else:
                generated, mask, _ = runtime.infer(row, 'localization')
                result = dict(sample_id=row['sample_id'], generated_text=generated['generated_text'],
                    truncated=generated['truncated'], valid_q_seg=generated['valid_q_seg'],
                    mask_path=save_mask(args.output, row['sample_id'], mask))
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
            print(json.dumps(dict(done=i + 1, total=len(rows), sample_id=row['sample_id'])), flush=True)


if __name__ == '__main__':
    main()
