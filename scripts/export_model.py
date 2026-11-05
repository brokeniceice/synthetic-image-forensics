#!/usr/bin/env python3
"""Maintainer utility: export one self-contained Hugging Face model directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_asset(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    # A hard link is a regular file and avoids duplicating immutable local weights.
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model', type=Path, required=True)
    p.add_argument('--clip', type=Path, required=True)
    p.add_argument('--c1', type=Path, required=True)
    p.add_argument('--rine', type=Path, required=True)
    p.add_argument('--spatial-bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root/'configs/paper/protocol.json').read_text())
    base_assets = json.loads((root/'configs/paper/base_assets.json').read_text())
    expected = protocol['checkpoint_sha256']
    for path, digest in [(args.c1,expected['c1']),(args.rine,expected['rine'])]:
        if sha(path) != digest:
            raise ValueError('Wrong selected checkpoint: ' + str(path))
    for group, directory in [('glamm',args.base_model),('clip',args.clip)]:
        for filename,digest in base_assets[group].items():
            if sha(directory/filename) != digest:
                raise ValueError('Base asset differs from the paper model: ' + filename)
    import torch
    import yaml
    torch.set_num_threads(4)
    c1 = torch.load(args.c1,map_location='cpu',weights_only=False)
    rine = torch.load(args.rine,map_location='cpu',weights_only=False)
    spatial = torch.load(args.spatial_bundle,map_location='cpu',weights_only=False)
    if spatial['c1_sha256'] != expected['c1'] or spatial['source_hashes']['r1'] != expected['r1'] or spatial['source_hashes']['forensic_source'] != expected['forensic_source']:
        raise ValueError('Spatial bundle is not the final P1-decoder NoRelative model')
    args.output.mkdir(parents=True,exist_ok=False)
    # Preserve all tensor values/dtypes and parameter names. Do not merge LoRA.
    weights = dict(c1=dict(module=c1['module'],epoch=c1['epoch'],optimizer_step=c1['optimizer_step']),
                   rine=dict(rine_state_dict=rine['rine_state_dict']),spatial=spatial,
                   source_hashes=expected)
    torch.save(weights,args.output/'forensics.pt')
    for group,directory,target in [('glamm',args.base_model,'base_model'),('clip',args.clip,'clip')]:
        for filename in base_assets[group]:
            copy_asset(directory/filename,args.output/target/filename)
    # Include any additional tokenizer files required by the selected snapshots.
    for directory,target in [(args.base_model,'base_model'),(args.clip,'clip')]:
        for file in directory.iterdir():
            if file.is_file() and file.suffix in ('.json','.txt','.model','.md') and not (args.output/target/file.name).exists():
                copy_asset(file,args.output/target/file.name)
    config = yaml.safe_load((root/'configs/c1.yaml').read_text())
    config['model']['version']='base_model';config['model']['vision_tower']='clip'
    config['forensics']['rine_checkpoint']='forensics.pt'
    manifest=dict(model_type='synthetic_image_forensics',format_version=1,
                  weights_sha256=sha(args.output/'forensics.pt'),protocol=protocol,
                  backbone_config=config)
    (args.output/'config.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(root/'LICENSE',args.output/'LICENSE')
    shutil.copy2(root/'NOTICE',args.output/'NOTICE')
    (args.output/'README.md').write_text('''---
language:
- en
tags:
- image-forensics
- image-classification
- image-segmentation
- multimodal
---
# Synthetic Image Forensics

Frozen C1 + P1 SAM decoder + NoRelative R1 model package. Downloading this
repository provides all inference weights; manual checkpoint assembly is not
required. Install the matching code from
https://github.com/brokeniceice/synthetic-image-forensics .

```python
from forensics.runtime import Runtime
runtime = Runtime.from_pretrained("LOCAL_DIRECTORY_OR_THIS_REPOSITORY_ID")
```

The package contains the selected GLaMM base and tokenizer, CLIP tower, and
one combined file containing the unchanged C1, RINE and spatial tensors.
LoRA remains a separate computation to preserve the BF16 inference results.
Classification uses the fixed TRAIN center; localization uses known-Fake G1
with 400 continuation tokens; explanation uses autonomous G0 with 512 tokens.
Exact provenance and evaluation settings are in config.json.

Source code, base weights and other upstream assets retain their respective
licenses. See LICENSE and NOTICE. This package contains no dataset images,
reference annotations, optimizer state or training logs.
''')
    files=[p for p in args.output.rglob('*') if p.is_file()]
    print(json.dumps(dict(output=str(args.output),files=len(files),bytes=sum(p.stat().st_size for p in files),weights_sha256=manifest['weights_sha256'])),flush=True)


if __name__=='__main__':
    main()
