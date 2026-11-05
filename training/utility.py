"""Final conditional utility initialization and ranking loss."""
from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn.functional as F
from model.csculf import CSCULF
from forensics.spatial import _utility_forward, gate_to_sam_grid, gated_embedding
from tools.spatial_metrics import tensor_state_sha256
SEED = 3407
MARGIN = 0.1

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def trainable_state(model):
    names = {name for name, value in model.named_parameters() if value.requires_grad}
    return {name: value for name, value in model.state_dict().items() if name in names}


def rank_loss(match, mismatch, support):
    value = F.relu(MARGIN - (match.float() - mismatch.float())) * support.float()
    return (value.flatten(1).sum(1) / support.flatten(1).sum(1).clamp_min(1)).mean()


def source_state(model: CSCULF) -> dict[str, torch.Tensor]:
    state = {f"language.{k}": v for k, v in model.language_source.state_dict().items()}
    state.update({f"forensic.{k}": v for k, v in model.forensic_source.state_dict().items()})
    return state


def full_inputs(data,idx,device):
    batch={k:data[k].index_select(0,idx).to(device) for k in ("S64","q_seg","z_L","F24","z_F24")}; batch["clip_geometries"]=[data["clip_geometries"][i] for i in idx.tolist()]; n=len(idx); batch.update({"valid_g0":torch.ones(n,dtype=torch.bool,device=device),"forensic_present":torch.ones(n,dtype=torch.bool,device=device),"forensic_vacuous":torch.zeros(n,dtype=torch.bool,device=device),"forensic_off":torch.zeros(n,dtype=torch.bool,device=device)}); return batch


def utility_forward(model, batch, *, permutation=None):
    with torch.autocast(device_type=batch['S64'].device.type, dtype=torch.bfloat16):
        return _utility_forward(model, batch, spatial_permutation=permutation)

def load_utility(cfg, device):
    seed_all()
    source = torch.load(cfg['inputs']['source_heads'], map_location='cpu', weights_only=False)
    temperatures = torch.load(cfg['inputs']['temperatures'], map_location='cpu', weights_only=False)
    model = CSCULF(temperatures['T_L'], temperatures['T_F'])
    model.load_frozen_source_heads(source['language_head'], source['forensic_head'])
    model.language_source.eval().requires_grad_(False)
    model.forensic_source.eval().requires_grad_(False)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 371803:
        raise RuntimeError('Utility trainable scope mismatch')
    return model.to(device), tensor_state_sha256(trainable_state(model))
