"""Shared image input and mask output for the command-line tools."""
import argparse
import hashlib
import json
from pathlib import Path


def make_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', required=True, help='Hugging Face model ID or local model directory')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--local-files-only', action='store_true')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--manifest', type=Path, help='JSONL with sample_id and image_path')
    parser.add_argument('--output', type=Path, required=True, help='A new output directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--center', type=float, default=-9.190834884678093)
    return parser


def read_inputs(args, parser):
    if args.image:
        rows = [dict(sample_id=args.image.stem, image_path=str(args.image.resolve()))]
    else:
        rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
        for row in rows:
            image = Path(row['image_path'])
            row['image_path'] = str((args.manifest.parent / image).resolve() if not image.is_absolute() else image)
    ids = [str(row['sample_id']) for row in rows]
    if not ids or len(ids) != len(set(ids)):
        parser.error('Input sample IDs must be unique and nonempty')
    for row in rows:
        if not Path(row['image_path']).is_file():
            parser.error('Missing image: ' + row['image_path'])
    return rows


def save_mask(output, sample_id, mask):
    import numpy as np
    from PIL import Image
    name = hashlib.sha256(str(sample_id).encode()).hexdigest() + '.png'
    (output / 'masks').mkdir(exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(output / 'masks' / name)
    return 'masks/' + name
