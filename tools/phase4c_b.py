"""Shared helpers for Phase 4C-B caches, readers, masks, metrics, and hashes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.forensics import compute_binary_mask_metrics
from model.clip_forensic_adapter import CLIPSpatialArm

ROOT=Path(__file__).resolve().parents[1]


def spatial_cache_paths(phase3c1_root: Path, source: str, split: str):
    if source not in ("clip", "sam"):
        raise ValueError(f"unsupported spatial cache source: {source}")
    root = phase3c1_root / "cache" / source / split
    complete = json.loads((root / "complete.json").read_text())
    if complete["status"] != "COMPLETE" or not complete["source_parameter_hash_exact"]:
        raise RuntimeError(f"invalid frozen {source.upper()} cache: {root}")
    paths = sorted(root.glob("shard_*.pt"))
    if len(paths) != int(complete["shards"]):
        raise RuntimeError(f"{source.upper()} cache shard-count mismatch")
    return paths


def load_spatial_shard(path: Path, expected_source: str):
    value = torch.load(path, map_location="cpu")
    if (value.get("schema") != "phase3c1_spatial_cache_v1" or
            value.get("source") != expected_source):
        raise RuntimeError(f"invalid {expected_source.upper()} cache payload: {path}")
    return value


def dump(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def append(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as h: h.write(json.dumps(value,ensure_ascii=False)+"\n")


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def file_sha256(path: Path):
    d=hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda:h.read(8<<20),b""): d.update(block)
    return d.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def tensor_hash(named):
    d=hashlib.sha256()
    for name,value in sorted(named):
        x=value.detach().cpu().contiguous(); d.update(name.encode()+b"\0"); d.update(str(x.dtype).encode()+b"\0"); d.update(str(tuple(x.shape)).encode()+b"\0"); d.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return d.hexdigest()


def load_source_model(cfg, arm, device):
    blocks=0 if arm=="clip_reader" else 3
    key="clip_proj_checkpoint" if arm=="clip_reader" else "forensic_adapter_checkpoint"
    state=torch.load(Path(cfg["phase4c_a"][key]),map_location="cpu")
    if int(state["epoch"])!=4: raise RuntimeError("Phase 4C-A selected epoch mismatch")
    model=CLIPSpatialArm(blocks=blocks); model.load_state_dict(state["model"]); model.requires_grad_(False); model.to(device).eval()
    return model


def source_feature(model, raw_clip, arm):
    with torch.no_grad():
        value=model(raw_clip,return_features=True)
    return (value["F0"] if arm=="clip_reader" else value["F_forensic"]).detach()


def decode_low_res(core, q_final, image_embedding):
    sparse,dense=core.model.grounding_encoder.prompt_encoder(points=None,boxes=None,masks=None,text_embeds=q_final)
    sparse=sparse.to(q_final.dtype)
    low,_=core.model.grounding_encoder.mask_decoder(
        image_embeddings=image_embedding.unsqueeze(0),
        image_pe=core.model.grounding_encoder.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
    return low


def sam_input_loss(low, target):
    logits=F.interpolate(low.float(),size=target.shape[-2:],mode="bilinear",align_corners=False)
    target=target.float().reshape(1,1,*target.shape[-2:])
    bce=F.binary_cross_entropy_with_logits(logits,target)
    p=logits.sigmoid(); intersection=2*(p/1000*target).sum(); union=(p/1000).sum()+(target/1000).sum()
    dice=1-(intersection+1e-6)/(union+1e-6)
    return {"bce":bce,"dice":dice,"total":2*bce+.5*dice}


def inverse_sam_logits(low, geometry):
    model=F.interpolate(low.float(),size=(1024,1024),mode="bilinear",align_corners=False)
    rh,rw=geometry["resized_hw"]; canvas=model[...,:rh,:rw]
    return F.interpolate(canvas,size=tuple(geometry["original_hw"]),mode="bilinear",align_corners=False)[0,0]


def metric_record(sample_id, logits, target):
    m=compute_binary_mask_metrics(logits.detach().cpu(),target.detach().cpu())
    pred=logits.detach().cpu().gt(0); truth=target.detach().cpu().bool(); tp=int((pred&truth).sum()); fp=int((pred&~truth).sum()); fn=int((~pred&truth).sum())
    return {"sample_id":sample_id,"foreground_iou":float(m["image_iou"]),"foreground_f1":float(m["image_pixel_f1"]),"tp":tp,"fp":fp,"fn":fn}


def summarize(records):
    a=np.asarray([r["foreground_iou"] for r in records]); f=np.asarray([r["foreground_f1"] for r in records]); tp=sum(r["tp"] for r in records); fp=sum(r["fp"] for r in records); fn=sum(r["fn"] for r in records)
    return {"n":len(records),"mean_foreground_iou":float(a.mean()),"median_foreground_iou":float(np.median(a)),"mean_foreground_f1":float(f.mean()),"global_foreground_iou":tp/(tp+fp+fn),"global_foreground_f1":2*tp/(2*tp+fp+fn),"threshold_logit":0.0}


def paired(left,right,repeats=10000,seed=3407):
    if [x["sample_id"] for x in left] != [x["sample_id"] for x in right]: raise RuntimeError("paired identity mismatch")
    result={}
    rng=np.random.default_rng(seed)
    for key in ("foreground_iou","foreground_f1"):
        d=np.asarray([a[key]-b[key] for a,b in zip(left,right)],np.float64); boot=[]
        for _ in range(20): boot.append(d[rng.integers(0,len(d),size=(repeats//20,len(d)))].mean(1))
        boot=np.concatenate(boot); eps=1e-12
        result[key]={"n":len(d),"mean_difference":float(d.mean()),"median_difference":float(np.median(d)),"bootstrap_95_ci":[float(x) for x in np.quantile(boot,[.025,.975])],"wins":int((d>eps).sum()),"ties":int((abs(d)<=eps).sum()),"losses":int((d<-eps).sum()),"bootstrap_repeats":repeats}
    return result


def diagnostic(values):
    a=np.asarray(values,np.float64); return {"n":len(a),"mean":float(a.mean()),"std":float(a.std()),"min":float(a.min()),"median":float(np.median(a)),"max":float(a.max())}
