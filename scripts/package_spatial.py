#!/usr/bin/env python3
"""Pack final R1, frozen P1 SAM and forensic source into a portable asset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r1', type=Path, required=True)
    p.add_argument('--p1-runtime', type=Path, required=True)
    p.add_argument('--forensic-source', type=Path, required=True)
    p.add_argument('--gamma', type=float, required=True, help='Selected C1 gamma used at rectifier initialization')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists')
    expected = json.loads((Path(__file__).resolve().parents[1] / 'configs/paper/protocol.json').read_text())['checkpoint_sha256']
    if sha(args.r1) != expected['r1'] or sha(args.forensic_source) != expected['forensic_source']:
        raise ValueError('Use the final NoRelative R1 and the selected forensic source')
    import torch
    from model.csculf import CSCULF
    from model.sam_forensic_rectifier import FrozenP1SAMPath, GeometryAwareSAMRectifier
    from model.clip_forensic_adapter import CLIPSpatialArm
    r1 = torch.load(args.r1, map_location='cpu', weights_only=False)
    sam = torch.load(args.p1_runtime, map_location='cpu', weights_only=False)
    source = torch.load(args.forensic_source, map_location='cpu', weights_only=False)
    expected = 'fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326'
    if sam.get('p1_checkpoint_sha256') != expected:
        raise ValueError('This release uses the frozen P1 SAM decoder')
    utility = CSCULF(); utility.load_state_dict(r1['utility_state'], strict=True)
    rectifier = GeometryAwareSAMRectifier(args.gamma, heads=8, locality_sigma=0.25)
    rectifier.load_state_dict(r1['rectifier_state'], strict=True)
    runtime = FrozenP1SAMPath()
    runtime.prompt_encoder.load_state_dict(sam['prompt_encoder'], strict=True)
    runtime.mask_decoder.load_state_dict(sam['mask_decoder'], strict=True)
    forensic = CLIPSpatialArm(blocks=3); forensic.load_state_dict(source['model'], strict=True)
    payload = dict(schema='paper_spatial_bundle_v1', c1_sha256=r1['c1_sha256'], gamma=args.gamma,
        utility_state=utility.state_dict(), rectifier_state=rectifier.state_dict(),
        prompt_encoder=runtime.prompt_encoder.state_dict(), mask_decoder=runtime.mask_decoder.state_dict(),
        forensic_source_state=forensic.state_dict(),
        source_hashes=dict(r1=sha(args.r1), p1_runtime=sha(args.p1_runtime), forensic_source=sha(args.forensic_source)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print('Wrote', args.output, 'sha256=' + sha(args.output))


if __name__ == '__main__':
    main()
