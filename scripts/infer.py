#!/usr/bin/env python3
"""Joint image forensics: classify, explain, and produce the final artifact mask."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forensics.cli import make_parser, read_inputs, save_mask


def main():
    parser = make_parser(__doc__)
    parser.add_argument('--max-new-tokens', type=int, default=400)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('Token budget must be positive')
    rows = read_inputs(args, parser)
    args.output.mkdir(parents=True, exist_ok=False)
    from forensics.runtime import Runtime
    runtime = Runtime.from_pretrained(args.model, revision=args.revision,
        device=args.device, local_files_only=args.local_files_only)
    (args.output / 'run.json').write_text(json.dumps(dict(mode='joint', n=len(rows),
        center=args.center, max_new_tokens=args.max_new_tokens), indent=2) + '\n')
    with (args.output / 'predictions.jsonl').open('w') as handle:
        for i, row in enumerate(rows):
            result, mask = runtime.predict(row, center=args.center, max_new_tokens=args.max_new_tokens)
            result['mask_path'] = save_mask(args.output, row['sample_id'], mask)
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
            print(json.dumps(dict(done=i + 1, total=len(rows), sample_id=row['sample_id'])), flush=True)


if __name__ == '__main__':
    main()
