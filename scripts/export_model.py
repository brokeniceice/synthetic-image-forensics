#!/usr/bin/env python3
"""Export complete model shards; load the frozen CLIP encoder separately."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLIP = dict(repo_id='openai/clip-vit-large-patch14-336',
            revision='ce19dc912ca5cd21c8a653c79e251e808ccabcd1')


def export_runtime(runtime, output, shard_bytes=4_000_000_000):
    from safetensors.torch import save_file
    from forensics.pretrained import is_clip_weight
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    modules = dict(backbone=runtime.model, utility=runtime.utility,
                   rectifier=runtime.rectifier, sam=runtime.sam, source=runtime.source)
    tensors = {}
    for prefix, module in modules.items():
        for name, value in module.state_dict().items():
            key = prefix + '.' + name
            if not is_clip_weight(key):
                tensors[key] = value.detach().cpu().contiguous().clone()
    shards, current, size = [], {}, 0
    for name, tensor in tensors.items():
        count = tensor.numel() * tensor.element_size()
        if current and size + count > shard_bytes:
            shards.append(current)
            current, size = {}, 0
        current[name] = tensor
        size += count
    if current:
        shards.append(current)
    index = dict(metadata=dict(total_size=sum(t.numel()*t.element_size() for t in tensors.values())), weight_map={})
    for number, shard in enumerate(shards, 1):
        filename = f'model-{number:05d}-of-{len(shards):05d}.safetensors'
        save_file(shard, output / filename, metadata={'format': 'pt'})
        index['weight_map'].update({name: filename for name in shard})
        print(json.dumps(dict(shard=number, total=len(shards), file=filename)), flush=True)
    (output / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
    runtime.tokenizer.save_pretrained(output)
    language = runtime.owner.config.to_dict()
    language['_name_or_path'] = ''
    for key in ('mm_vision_tower', 'vision_tower', 'mm_vision_module', 'vision_module'):
        language[key] = CLIP['repo_id']
    config = copy.deepcopy(runtime.cfg)
    config = {key: config[key] for key in ('model', 'loss', 'optimizer')}
    config['model']['version'] = '.'
    config['model']['vision_tower'] = CLIP['repo_id']
    config['model']['vision_pretrained'] = None
    manifest = dict(model_type='synthetic_image_forensics', format_version=2,
        clip=CLIP, backbone_config=config, language_config=language,
        generation_config=runtime.owner.generation_config.to_dict(),
        spatial_config=dict(gamma=float(runtime.rectifier.rectification.gamma.detach().cpu())))
    # Gamma is restored from its tensor; this value only constructs the module.
    (output / 'config.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for name in ('LICENSE', 'NOTICE'):
        shutil.copy2(ROOT / name, output / name)
    (output / 'README.md').write_text('''---
language:
- en
tags:
- image-forensics
- image-classification
- image-segmentation
- multimodal
---
# Synthetic Image Forensics

Complete model weights for synthetic-image classification, artifact localization
and explanation generation. Install the code and dependencies from
[GitHub](https://github.com/brokeniceice/synthetic-image-forensics).

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \\
  --model broken-ice/synthetic-image-forensics \\
  --task classification --image path/to/image.png \\
  --output outputs/classification
```

Choose `classification`, `localization` or `explanation` with `--task`.
Localization assumes a synthetic input image. Use a new output directory.
For multiple images, replace `--image ...` with `--manifest data/images.jsonl`.
Manifest records contain `sample_id` and `image_path`; see the GitHub README
for data formats, evaluation and training commands.

The model downloads `openai/clip-vit-large-patch14-336` automatically at the
revision specified in `config.json`. CLIP is cached separately and is not
included in these weight shards. All other model components are included;
no base-model or adapter checkpoint assembly is required.

`--model` also accepts a downloaded model directory. Once both the model and
CLIP are cached, use `--local-files-only` to run offline.

See LICENSE and NOTICE for licensing and attribution.
''')
    return dict(output=str(output), tensors=len(tensors), shards=len(shards),
                weight_bytes=index['metadata']['total_size'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Model repository ID or local package')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    from forensics.runtime import Runtime
    runtime = Runtime.from_pretrained(args.model, device=args.device)
    print(json.dumps(export_runtime(runtime, args.output)), flush=True)


if __name__ == '__main__':
    main()
