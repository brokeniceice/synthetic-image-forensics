"""Train R1 through rectifier fitting, utility fitting and joint fitting."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
import torch
import yaml
from training import utility as hc
from training import spatial_data as e2
from training.spatial_data import SpatialFeatureStore, evidence_feature, invalid_record, load_evidence_source, load_rectifier, load_sam_runtime, mask_loss, train_ids, load_dev
from tools.masks import inverse_sam_logits, metric_record, file_sha256
from tools.spatial_metrics import summarize, tensor_state_sha256
SEED = 3407
C1_SHA = '85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff'
CFG = {}
OUT = Path('outputs/spatial')
CKPTS = Path('checkpoints/spatial')

def require(test, message):
    if not test:
        raise RuntimeError(message)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def hash_state(module):
    return tensor_state_sha256(module.state_dict())


def rect_metrics(rectifier, store, dev, cache, sam, source, device):
    records = []
    rectifier.eval()
    with torch.no_grad():
        for i, sid in enumerate(dev["sample_ids"]):
            if not bool(cache["valid"][i]):
                row = invalid_record(sid, dev["original_masks"][i])
                row["valid_g0"] = False
                records.append(row)
                continue
            s64, raw, _, sc, cc = e2.spatial_batch(store, [sid], device)
            evidence = evidence_feature(source, raw)
            valid = torch.ones(1, 576, dtype=torch.bool, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = rectifier(s64, evidence, sc, cc, valid)
            query = cache["q_seg"][i:i + 1].to(device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type="cuda", enabled=False):
                low = sam(query, output["image_embeddings"].to(torch.bfloat16))
            logits = inverse_sam_logits(low, dev["sam_geometries"][i])
            row = metric_record(sid, logits, dev["original_masks"][i])
            row["valid_g0"] = True
            records.append(row)
    return summarize(records)


def stage1(device):
    stage = "rectifier"
    require(not (OUT / stage / "selector.json").exists(), "rectifier already selected")
    hc.seed_all()
    ids, dev, train_store, val_store, cache, val_cache = load_inputs()
    sam = load_sam_runtime(CFG, device)
    source = load_evidence_source(CFG, device)
    gamma = c1_gamma()
    rectifier = load_rectifier(CFG, gamma, device)
    require(sum(p.numel() for p in rectifier.parameters() if p.requires_grad) == 329985, "rectifier trainable scope drift")
    frozen = {"sam": hash_state(sam), "source": hash_state(source)}
    init = hash_state(rectifier)
    # Match spatial rectification: AdamW, warmup+cosine, ten full traversals, epoch 0 eligible.
    optimizer, scheduler = optimizer_scheduler(rectifier, 11050)
    history, updates = [], 0
    stage_ckpts = CKPTS / stage
    require(not list(stage_ckpts.glob("epoch_*.pt")), "rectifier stage already started; refusing overwrite")
    initial = rect_metrics(rectifier, val_store, dev, val_cache, sam, source, device)
    history.append({"epoch": 0, "updates": 0, "dev_g0_mean_iou": initial["mean_foreground_iou"]})
    dump(OUT / stage / "history.json", history)
    stage_ckpts.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 0, "rectifier_state": rectifier.state_dict(), "c1_sha256": C1_SHA}, stage_ckpts / "epoch_0.pt")
    valid = cache["valid"].bool()
    id_to_i = {sid: i for i, sid in enumerate(ids)}
    for epoch in range(1, 11):
        began = time.time()
        order = list(ids)
        random.Random(SEED + 1009 * epoch).shuffle(order)
        rectifier.train()
        traversal = eligible = invalid = 0
        for start in range(0, len(order), 8):
            batch_ids = order[start:start + 8]
            positions = torch.tensor([id_to_i[sid] for sid in batch_ids])
            idx = positions[valid.index_select(0, positions)]
            traversal += len(batch_ids); eligible += len(idx); invalid += len(batch_ids) - len(idx)
            if len(idx) == 0:
                continue
            active_ids = [ids[j] for j in idx.tolist()]
            s64, raw, target, sc, cc = e2.spatial_batch(train_store, active_ids, device)
            with torch.no_grad():
                evidence = evidence_feature(source, raw)
            query = cache["q_seg"].index_select(0, idx).to(device=device, dtype=torch.bfloat16)
            token_valid = torch.ones(len(idx), 576, dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = rectifier(s64, evidence, sc, cc, token_valid)
            with torch.autocast(device_type="cuda", enabled=False):
                low = sam(query, output["image_embeddings"].to(torch.bfloat16))
            loss = mask_loss(low, target, CFG)["total"]
            require(bool(torch.isfinite(loss)), "nonfinite rectifier loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(rectifier.parameters(), float(CFG["optimizer"]["gradient_clip_norm"]))
            require(bool(torch.isfinite(norm)), "nonfinite rectifier gradient")
            optimizer.step(); scheduler.step(); updates += 1
            if updates <= 3 or updates % 100 == 0:
                print(json.dumps({"stage": stage, "epoch": epoch, "update": updates, "loss": float(loss.detach())}), flush=True)
        require((traversal, eligible, invalid) == (8836, 8741, 95), "rectifier exposure drift")
        metrics = rect_metrics(rectifier, val_store, dev, val_cache, sam, source, device)
        row = {"epoch": epoch, "updates": updates, "dev_g0_mean_iou": metrics["mean_foreground_iou"],
               "sample_order_sha256": hashlib.sha256('\n'.join(order).encode()).hexdigest(), "seconds": time.time() - began}
        history.append(row)
        dump(OUT / stage / "history.json", history)
        temp = stage_ckpts / f"epoch_{epoch}.pt.tmp"
        torch.save({"epoch": epoch, "rectifier_state": rectifier.state_dict(), "c1_sha256": C1_SHA,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, temp)
        os.replace(temp, stage_ckpts / f"epoch_{epoch}.pt")
        print(json.dumps({"stage": stage, **row}), flush=True)
    require(frozen == {"sam": hash_state(sam), "source": hash_state(source)}, "frozen backbone drift")
    selector = select(stage, history, "dev_g0_mean_iou", "rectifier_state")
    dump(OUT / stage / "provenance.json", {"status": "COMPLETE", "seed": SEED, "loss_coefficients": [0.0, 1.0], "updates": updates, "initial_state_sha256": init,
         "selected": selector, "frozen_hash_before_after": frozen, "c1_valid_train": 8741, "c1_valid_dev": 1090,
         "p1_trained_rectifier_loaded": False, "official1000_used": False})


def later_stage(stage, device):
    require(stage in ("utility", "joint"), "bad stage")
    require(not (OUT / stage / "selector.json").exists(), f"{stage} already selected")
    hc.seed_all()
    ids, dev, train_store, val_store, cache, val_cache = load_inputs()
    model, _ = hc.load_utility(CFG, device)
    random_utility_hash = tensor_state_sha256(hc.trainable_state(model))
    rectifier = load_rectifier(CFG, c1_gamma(), device)
    rect_state, rect_selector = selected_state("rectifier", "rectifier_state")
    rectifier.load_state_dict(rect_state, strict=True)
    if stage == "joint":
        utility_state, utility_selector = selected_state("utility", "utility_state")
        model.load_state_dict(utility_state, strict=True)
    else:
        utility_selector = None
        rectifier.requires_grad_(False)
    sam = load_sam_runtime(CFG, device)
    source = load_evidence_source(CFG, device)
    frozen = {"sam": hash_state(sam), "source": hash_state(source), "heads": tensor_state_sha256(hc.source_state(model))}
    if stage == "utility":
        frozen["rectifier"] = hash_state(rectifier)
    data = e2.load_ids(CFG, ids, ("valid_g0", "S64", "q_seg", "z_L", "F24", "z_F24", "target64", "clip_geometries"))
    params = [p for p in model.parameters() if p.requires_grad] + [p for p in rectifier.parameters() if p.requires_grad]
    expected_count = 371803 if stage == "utility" else 371803 + 329985
    require(sum(p.numel() for p in params) == expected_count, f"{stage} trainable scope drift")
    optimizer = torch.optim.AdamW(params, lr=float(CFG['joint_training']['learning_rate']), weight_decay=float(CFG['joint_training']['weight_decay']))
    stage_ckpts = CKPTS / stage
    require(not list(stage_ckpts.glob("epoch_*.pt")), f"{stage} already started; refusing overwrite")
    stage_ckpts.mkdir(parents=True, exist_ok=True)
    valid = cache["valid"].bool()
    id_to_i = {sid: i for i, sid in enumerate(ids)}
    cross = torch.tensor([(i + 1) % len(ids) for i in range(len(ids))])
    permutation = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    history, updates = [], 0
    for epoch in range(1, 11):
        began = time.time()
        order = list(ids)
        random.Random(SEED + 1009 * epoch).shuffle(order)
        model.train(); model.language_source.eval(); model.forensic_source.eval()
        rectifier.train() if stage == "joint" else rectifier.eval()
        sums = defaultdict(float)
        traversal = eligible = invalid = 0
        for begin in range(0, len(order), 8):
            all_ids = order[begin:begin + 8]
            positions = torch.tensor([id_to_i[sid] for sid in all_ids])
            idx = positions[valid.index_select(0, positions)]
            traversal += len(all_ids); eligible += len(idx); invalid += len(all_ids) - len(idx)
            if len(idx) == 0:
                continue
            batch, _ = e2.c1_language_batch(data, cache, idx, train_store, sam, device)
            crossed = dict(batch)
            fi = cross.index_select(0, idx)
            crossed["F24"] = data["F24"].index_select(0, fi).to(device)
            crossed["z_F24"] = data["z_F24"].index_select(0, fi).to(device)
            active_ids = [ids[i] for i in idx.tolist()]
            s64, corrected, support, target, sc = e2.corrected_batch(train_store, active_ids, source, rectifier, device)
            optimizer.zero_grad(set_to_none=True)
            matched = hc.utility_forward(model, batch)
            crossed_out = hc.utility_forward(model, crossed)
            shuffled = hc.utility_forward(model, batch, permutation=permutation)
            gate = hc.gate_to_sam_grid(matched["U"], sc) * support.float()
            adapted = hc.gated_embedding(s64, corrected, gate)
            with torch.autocast(device_type="cuda", enabled=False):
                low = sam(batch["q_seg"], adapted.to(torch.bfloat16))
            seg = mask_loss(low, target, CFG)["total"]
            cr = hc.rank_loss(matched["U"], crossed_out["U"], matched["support"])
            sr = hc.rank_loss(matched["U"], shuffled["U"], matched["support"])
            total = seg + 0.5 * (cr + sr)
            require(bool(torch.isfinite(total)), f"nonfinite {stage} loss")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, float(CFG['joint_training']['gradient_clip_norm']))
            require(bool(torch.isfinite(norm)), f"nonfinite {stage} gradient")
            optimizer.step(); updates += 1
            sums["loss"] += float(total.detach()) * len(idx)
            if updates <= 3 or updates % 100 == 0:
                print(json.dumps({"stage": stage, "epoch": epoch, "update": updates, "loss": float(total.detach())}), flush=True)
        require((traversal, eligible, invalid) == (8836, 8741, 95), f"{stage} exposure drift")
        metrics, _ = e2.evaluate(model, rectifier, sam, source, val_store, dev, val_cache, device)
        row = {"epoch": epoch, "updates": updates, "dev_g0_mean_iou": metrics["mean_foreground_iou"],
               "mean_train_loss": sums["loss"] / eligible, "sample_order_sha256": hashlib.sha256('\n'.join(order).encode()).hexdigest(), "seconds": time.time() - began}
        history.append(row)
        dump(OUT / stage / "history.json", history)
        temp = stage_ckpts / f"epoch_{epoch}.pt.tmp"
        torch.save({"epoch": epoch, "utility_state": model.state_dict(), "rectifier_state": rectifier.state_dict(),
                    "c1_sha256": C1_SHA, "optimizer": optimizer.state_dict()}, temp)
        os.replace(temp, stage_ckpts / f"epoch_{epoch}.pt")
        print(json.dumps({"stage": stage, **row}), flush=True)
    after = {"sam": hash_state(sam), "source": hash_state(source), "heads": tensor_state_sha256(hc.source_state(model))}
    if stage == "utility":
        after["rectifier"] = hash_state(rectifier)
    require(after == frozen, f"{stage} frozen state drift")
    selector = select(stage, history, "dev_g0_mean_iou", "utility_state+rectifier_state")
    dump(OUT / stage / "provenance.json", {"status": "COMPLETE", "seed": SEED, "loss_coefficients": [0.0, 1.0], "updates": updates, "selected": selector,
         "rectifier_input_selector_sha256": rect_selector["selected_checkpoint_sha256"],
         "utility_input_selector_sha256": utility_selector["selected_checkpoint_sha256"] if utility_selector else None,
         "random_utility_state_sha256": random_utility_hash, "trainable_parameters": expected_count,
         "frozen_hash_before": frozen, "frozen_hash_after": after,
         "p1_trained_utility_loaded": False, "p1_trained_rectifier_loaded": False,
         "official1000_used": False, "external_ood_used": False})


def optimizer_scheduler(rectifier, total_updates: int):
    optimizer = torch.optim.AdamW(rectifier.parameters(), lr=float(CFG["optimizer"]["learning_rate"]), weight_decay=float(CFG["optimizer"]["weight_decay"]), betas=tuple(CFG["optimizer"]["betas"]))
    warm = round(total_updates * float(CFG["optimizer"]["warmup_fraction"]))
    def scale(step):
        if step < warm: return float(step + 1) / max(1, warm)
        progress = min(1.0, (step - warm) / max(1, total_updates - warm))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)



def c1_gamma():
    return float(CFG['inputs']['gamma'])

def load_inputs():
    ids = train_ids(CFG)
    dev = load_dev(CFG, 'g0')
    train_store = SpatialFeatureStore(CFG, 'train')
    val_store = SpatialFeatureStore(CFG, 'val')
    require(ids == train_store.sample_ids and dev['sample_ids'] == val_store.sample_ids, 'Spatial population/order mismatch')
    cache = e2.load_c1_cache(CFG, 'train', ids)
    val_cache = e2.load_c1_cache(CFG, 'val', dev['sample_ids'])
    require((len(ids), int(cache['valid'].sum()), len(dev['sample_ids']), int(val_cache['valid'].sum())) == (8836, 8741, 1106, 1090), 'C1 query population mismatch')
    return ids, dev, train_store, val_store, cache, val_cache

def select(stage, history, key, payload_key):
    best = max(history, key=lambda row: (row[key], -row['epoch']))
    path = CKPTS / stage / f"epoch_{best['epoch']}.pt"
    selected_path = OUT / stage / 'selected_checkpoint.pt'
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, selected_path)
    info = dict(status='COMPLETE', stage=stage, selected_epoch=best['epoch'], selected_metric=best[key],
                selected_checkpoint=str(selected_path.resolve()), selected_checkpoint_sha256=file_sha256(selected_path),
                c1_sha256=C1_SHA, checkpoint_payload=payload_key)
    dump(OUT / stage / 'selector.json', info)
    return info

def selected_state(stage, key):
    selector = json.loads((OUT / stage / 'selector.json').read_text())
    path = Path(selector['selected_checkpoint'])
    require(file_sha256(path) == selector['selected_checkpoint_sha256'], 'Selected checkpoint changed')
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(state['c1_sha256'] == C1_SHA and key in state, 'Selected checkpoint linkage mismatch')
    return state[key], selector

def main(argv=None):
    global CFG, OUT, CKPTS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['rectifier', 'utility', 'joint'])
    p.add_argument('--config', default='configs/r1.yaml')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args(argv)
    CFG = yaml.safe_load(Path(args.config).read_text())
    CFG['c1_sha256'] = C1_SHA
    OUT = Path(CFG['experiment']['output_root'])
    CKPTS = Path(CFG['experiment']['checkpoint_root'])
    require(CFG['experiment']['seed'] == 3407 and CFG['joint_training']['relative_weight'] == 0.0 and CFG['joint_training']['ranking_weight'] == 1.0, 'Final spatial recipe mismatch')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    if args.stage == 'rectifier':
        stage1(device)
    else:
        later_stage(args.stage, device)

if __name__ == '__main__':
    main()
