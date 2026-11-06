"""Read the frozen R1 decoder and evidence source from the released model."""
from functools import lru_cache
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open


@lru_cache(maxsize=2)
def released_components(model, revision=None, local_files_only=False):
    directory = Path(model).expanduser()

    def file(name):
        if directory.is_dir():
            return directory / name
        return Path(hf_hub_download(repo_id=model, filename=name, revision=revision,
                                   local_files_only=local_files_only))

    config = json.loads(file('config.json').read_text())
    if config.get('model_type') != 'synthetic_image_forensics':
        raise ValueError('Expected a Synthetic Image Forensics model')
    if config['format_version'] == 1:
        # Earlier releases already contain these components in the combined file.
        value = torch.load(file('forensics.pt'), map_location='cpu', weights_only=False)['spatial']
        return {name: value[name] for name in ('prompt_encoder', 'mask_decoder', 'forensic_source_state')}
    if config['format_version'] != 2:
        raise ValueError('Unsupported model format')
    index = json.loads(file('model.safetensors.index.json').read_text())['weight_map']
    prefixes = {'sam.prompt_encoder.': 'prompt_encoder',
                'sam.mask_decoder.': 'mask_decoder', 'source.': 'forensic_source_state'}
    selected = {}
    for key, shard in index.items():
        for prefix, destination in prefixes.items():
            if key.startswith(prefix):
                selected.setdefault(shard, []).append((key, destination, key[len(prefix):]))
    states = {destination: {} for destination in prefixes.values()}
    for shard, entries in selected.items():
        if Path(shard).name != shard:
            raise ValueError('Invalid model shard filename')
        with safe_open(file(shard), framework='pt', device='cpu') as handle:
            for key, destination, name in entries:
                states[destination][name] = handle.get_tensor(key)
    if not all(states.values()):
        raise ValueError('The model is missing frozen R1 components')
    return states


def initialization_states(config):
    args = config['initialization']
    return released_components(str(args['model']), args.get('revision'),
                               bool(args.get('local_files_only', False)))
