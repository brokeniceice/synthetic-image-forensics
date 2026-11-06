from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from model.clip_forensic_adapter import CLIPSpatialArm
from model.sam_forensic_rectifier import FrozenP1SAMPath, GeometryAwareSAMRectifier
from tools.masks import spatial_cache_paths
from tools.spatial_metrics import clip_coordinates, sam_coordinates
from training.initialization import initialization_states
ROOT = Path(__file__).resolve().parents[1]

class SpatialFeatureStore:
    """RAM-resident immutable S64/F24/union-target store."""

    def __init__(self, cfg: dict, split: str) -> None:
        (self.cfg, self.split) = (cfg, split)
        root = ROOT / cfg['data']['spatial_cache_root']
        sam_paths = spatial_cache_paths(root, 'sam', split)
        clip_paths = spatial_cache_paths(root, 'clip', split)
        if len(sam_paths) != len(clip_paths):
            raise RuntimeError('SAM/CLIP shard mismatch')
        (self.sample_ids, self.locations) = ([], {})
        (self.sam_features, self.clip_features, self.targets) = ([], [], [])
        (self.original_masks, self.geometries, self.sam_coords, self.clip_coords) = ({}, {}, {}, {})
        for (shard_index, (sam_path, clip_path)) in enumerate(zip(sam_paths, clip_paths)):
            sam = torch.load(sam_path, map_location='cpu', weights_only=False)
            clip = torch.load(clip_path, map_location='cpu', weights_only=False)
            sam_ids = [str(row['sample_id']) for row in sam['records']]
            clip_ids = [str(row['sample_id']) for row in clip['records']]
            if sam_ids != clip_ids:
                raise RuntimeError('SAM/CLIP ID mismatch')
            self.sam_features.append(sam['features'])
            self.clip_features.append(clip['features'])
            self.targets.append(sam['targets'])
            original = sam.get('original_masks')
            for (local, (sid, sam_meta, clip_meta)) in enumerate(zip(sam_ids, sam['records'], clip['records'])):
                self.locations[sid] = (shard_index, local)
                self.sample_ids.append(sid)
                geometry = sam_meta['geometry']
                self.geometries[sid] = geometry
                self.sam_coords[sid] = sam_coordinates(geometry, grid=64)
                self.clip_coords[sid] = clip_coordinates(clip_meta['geometry'], grid=24)
                if original is not None:
                    self.original_masks[sid] = original[local].bool()
        expected = int(cfg['data']['train_fake'] if split == 'train' else cfg['data']['val_fake'])
        if len(self.sample_ids) != expected or len(set(self.sample_ids)) != expected:
            raise RuntimeError(f'{split} population mismatch')

    def _values(self, sid: str):
        (shard, local) = self.locations[sid]
        return (self.sam_features[shard][local], self.clip_features[shard][local], self.targets[shard][local])

def load_sam_runtime(cfg: dict, device: torch.device) -> FrozenP1SAMPath:
    value = initialization_states(cfg)
    model = FrozenP1SAMPath()
    model.prompt_encoder.load_state_dict(value['prompt_encoder'], strict=True)
    model.mask_decoder.load_state_dict(value['mask_decoder'], strict=True)
    return model.to(device=device, dtype=torch.bfloat16).freeze()

def load_evidence_source(cfg: dict, device: torch.device) -> CLIPSpatialArm:
    value = initialization_states(cfg)
    model = CLIPSpatialArm(blocks=3)
    model.load_state_dict(value['forensic_source_state'], strict=True)
    return model.to(device).eval().requires_grad_(False)

def evidence_feature(source: CLIPSpatialArm, raw_clip: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        value = source(raw_clip, return_features=True)
    return value['F_forensic'].detach()

def load_rectifier(cfg: dict, gamma: float, device: torch.device) -> GeometryAwareSAMRectifier:
    torch.manual_seed(int(cfg['experiment']['seed']))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg['experiment']['seed']))
    return GeometryAwareSAMRectifier(gamma_init=gamma, heads=int(cfg['architecture']['heads']), locality_sigma=float(cfg['architecture']['locality_sigma'])).to(device)

def mask_loss(low: torch.Tensor, targets: torch.Tensor, cfg: dict) -> dict[str, torch.Tensor]:
    logits = F.interpolate(low.float(), size=targets.shape[-2:], mode='bilinear', align_corners=False)
    target = targets.float()[:, None]
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    intersection = 2 * (probability / 1000 * target).flatten(1).sum(1)
    union = (probability / 1000).flatten(1).sum(1) + (target / 1000).flatten(1).sum(1)
    dice = (1 - (intersection + 1e-06) / (union + 1e-06)).mean()
    total = float(cfg['loss']['mask_bce_weight']) * bce + float(cfg['loss']['mask_dice_weight']) * dice
    return {'bce': bce, 'dice': dice, 'total': total}

def invalid_record(sid: str, target: torch.Tensor) -> dict:
    truth = target.bool()
    return {'sample_id': sid, 'foreground_iou': 0.0, 'foreground_f1': 0.0, 'tp': 0, 'fp': 0, 'fn': int(truth.sum()), 'valid_g0': False}
from collections import defaultdict
from forensics.spatial import gate_to_sam_grid, gated_embedding
from training.utility import full_inputs, utility_forward

def load_c1_cache(cfg: dict, split: str, expected_ids: list[str]) -> dict:
    values = defaultdict(list)
    ids = []
    for path in sorted((Path(cfg['inputs']['c1_queries']) / split).glob('shard_*.pt')):
        x = torch.load(path, map_location='cpu', weights_only=False)
        if x.get('c1_sha256') != cfg['c1_sha256']:
            raise RuntimeError('C1 cache checkpoint drift')
        ids += x['sample_ids']
        for key in ('q_seg', 'valid', 'seg_count'):
            values[key].append(x[key])
    if ids != expected_ids:
        raise RuntimeError(f'{split} C1 cache incomplete/order drift')
    return {'sample_ids': ids, **{k: torch.cat(v) for (k, v) in values.items()}}

def c1_language_batch(data: dict, cache: dict, idx: torch.Tensor, store: SpatialFeatureStore, sam, device: torch.device) -> tuple[dict, torch.Tensor]:
    batch = full_inputs(data, idx, device)
    qseg = cache['q_seg'].index_select(0, idx).to(device=device, dtype=torch.bfloat16)
    ids = [cache['sample_ids'][i] for i in idx.tolist()]
    (raw, _, _, _, _) = spatial_batch(store, ids, device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        low = sam(qseg, raw.to(torch.bfloat16))
    zl = []
    for (j, sid) in enumerate(ids):
        from model.evidence import sam_lowres_to_original_normalized
        zl.append(sam_lowres_to_original_normalized(low[j:j + 1], store.geometries[sid], output_hw=(256, 256)))
    batch['q_seg'] = qseg
    batch['z_L'] = torch.cat(zl).to(device=device, dtype=torch.bfloat16)
    return (batch, raw)

def spatial_batch(store: SpatialFeatureStore, ids: list[str], device: torch.device):
    """spatial rectification spatial payload without the obsolete P1 q_seg side field."""
    values = [store._values(sid) for sid in ids]
    return (torch.stack([row[0] for row in values]).to(device=device, dtype=torch.bfloat16, non_blocking=True), torch.stack([row[1] for row in values]).to(device=device, dtype=torch.bfloat16, non_blocking=True), torch.stack([row[2] for row in values]).to(device=device, non_blocking=True), torch.stack([store.sam_coords[sid] for sid in ids]).to(device=device, non_blocking=True), torch.stack([store.clip_coords[sid] for sid in ids]).to(device=device, non_blocking=True))

def corrected_batch(store, ids, source, rectifier, device):
    (s64, raw, targets, sc, cc) = spatial_batch(store, ids, device)
    with torch.no_grad():
        evidence = evidence_feature(source, raw)
    valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        value = rectifier(s64, evidence, sc, cc, valid)
    return (s64, value['image_embeddings'], value['support'].reshape(len(ids), 1, 64, 64), targets, sc)

def evaluate(model, rectifier, sam, source, store, dev, cache, device):
    records = []
    model.eval()
    model.language_source.eval()
    model.forensic_source.eval()
    rectifier.eval()
    with torch.no_grad():
        for (i, sid) in enumerate(dev['sample_ids']):
            if not bool(cache['valid'][i]):
                row = invalid_record(sid, dev['original_masks'][i])
                row['valid_g0'] = False
                records.append(row)
                continue
            idx = torch.tensor([i])
            (batch, _) = c1_language_batch(dev, cache, idx, store, sam, device)
            (s64, p4f, support, _, sc) = corrected_batch(store, [sid], source, rectifier, device)
            out = utility_forward(model, batch)
            gate = gate_to_sam_grid(out['U'], sc) * support.float()
            adapted = gated_embedding(s64, p4f, gate)
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(batch['q_seg'], adapted.to(torch.bfloat16))
            logits = inverse_sam_logits(low, dev['sam_geometries'][i])
            row = metric_record(sid, logits, dev['original_masks'][i])
            row['valid_g0'] = True
            records.append(row)
            if (i + 1) % 200 == 0:
                print(json.dumps({'stage': 'INTERNAL_VAL', 'done': i + 1, 'total': len(dev['sample_ids'])}), flush=True)
    return (summarize(records), records)

def load_ids(cfg: dict, ids: list[str], fields: tuple[str, ...]) -> dict:
    desired = set(ids)
    (loaded_ids, tensors) = ([], {field: [] for field in fields if field != 'clip_geometries'})
    geometries = []
    for path in sorted(Path(cfg['inputs']['utility_features']).glob('shard_*.pt')):
        shard = torch.load(path, map_location='cpu', weights_only=False)
        positions = [index for (index, sid) in enumerate(shard['sample_ids']) if sid in desired]
        if not positions:
            continue
        index = torch.tensor(positions)
        loaded_ids.extend((shard['sample_ids'][position] for position in positions))
        for field in tensors:
            tensors[field].append(shard[field].index_select(0, index))
        if 'clip_geometries' in fields:
            geometries.extend((shard['clip_geometries'][position] for position in positions))
    if set(loaded_ids) != desired or len(loaded_ids) != len(desired):
        raise RuntimeError('safe utility-pool cache selection mismatch')
    lookup = {sid: index for (index, sid) in enumerate(loaded_ids)}
    order = torch.tensor([lookup[sid] for sid in ids])
    result = {field: torch.cat(parts).index_select(0, order) for (field, parts) in tensors.items()}
    if 'clip_geometries' in fields:
        result['clip_geometries'] = [geometries[index] for index in order.tolist()]
    result['sample_ids'] = ids
    return result

def load_dev(cfg, mode):
    CACHE = Path(cfg['inputs']['dev'])
    shared = json.loads((CACHE / 'dev_shared/complete.json').read_text())
    modes = json.loads((CACHE / f'dev_{mode}/complete.json').read_text())
    result = defaultdict(list)
    ids = []
    for (sr, mr) in zip(shared['shards'], modes['shards']):
        s = torch.load(sr['path'], map_location='cpu', weights_only=False)
        m = torch.load(mr['path'], map_location='cpu', weights_only=False)
        if s['sample_ids'] != m['sample_ids']:
            raise RuntimeError('dev mode/shared ID drift')
        ids += s['sample_ids']
        for k in ('S64', 'F24', 'z_F24', 'target256'):
            result[k].append(s[k])
        for k in ('valid', 'q_seg', 'z_L'):
            result[k].append(m[k])
        for k in ('original_masks', 'sam_geometries', 'clip_geometries'):
            result[k].extend(s[k])
    value = {k: torch.cat(v) if k not in ('original_masks', 'sam_geometries', 'clip_geometries') else v for (k, v) in result.items()}
    value['sample_ids'] = ids
    return value

def train_ids(cfg):
    ids = []
    for path in sorted(Path(cfg['inputs']['utility_features']).glob('shard_*.pt')):
        ids.extend((str(sid) for sid in torch.load(path, map_location='cpu', weights_only=False)['sample_ids']))
    if len(ids) != 8836 or len(set(ids)) != 8836:
        raise RuntimeError('Spatial TRAIN population mismatch')
    return ids
from tools.masks import inverse_sam_logits, metric_record
from tools.spatial_metrics import summarize
