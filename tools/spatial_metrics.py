from __future__ import annotations
import hashlib
import numpy as np
import torch

def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for (name, tensor) in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b'\x00')
        digest.update(str(value.dtype).encode() + b'\x00')
        digest.update(str(tuple(value.shape)).encode() + b'\x00')
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def sam_coordinates(geometry: dict, grid: int=32) -> torch.Tensor:
    (resized_h, resized_w) = map(float, geometry['resized_hw'])
    y = (torch.arange(grid, dtype=torch.float32) + 0.5) * 1024.0 / grid / resized_h
    x = (torch.arange(grid, dtype=torch.float32) + 0.5) * 1024.0 / grid / resized_w
    (yy, xx) = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

def clip_coordinates(geometry: dict, grid: int=24) -> torch.Tensor:
    (resized_h, resized_w) = map(float, geometry['resized_hw'])
    (top, left, _, _) = geometry['crop_box_yxyx']
    y = (float(top) + (torch.arange(grid, dtype=torch.float32) + 0.5) * 336.0 / grid) / resized_h
    x = (float(left) + (torch.arange(grid, dtype=torch.float32) + 0.5) * 336.0 / grid) / resized_w
    (yy, xx) = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

def summarize(records: list[dict]) -> dict:
    iou = np.asarray([row['foreground_iou'] for row in records], dtype=np.float64)
    f1 = np.asarray([row['foreground_f1'] for row in records], dtype=np.float64)
    (tp, fp, fn) = (sum((int(row[key]) for row in records)) for key in ('tp', 'fp', 'fn'))
    return {'n': len(records), 'mean_foreground_iou': float(iou.mean()), 'median_foreground_iou': float(np.median(iou)), 'mean_foreground_f1': float(f1.mean()), 'global_foreground_iou': tp / max(1, tp + fp + fn), 'global_foreground_f1': 2 * tp / max(1, 2 * tp + fp + fn), 'threshold_logit': 0.0}
