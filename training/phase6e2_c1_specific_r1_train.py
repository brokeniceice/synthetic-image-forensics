#!/usr/bin/env python3
"""Train C1-specific R1 from the exact pre-Phase4H-D R1 initialization."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import phase4g1q_conditional_utility as q
from training import phase4hc_direct_utility_arms as hc
from training import phase4hd_rectifier_unfreeze_control as hd
from training.phase4gf_formal_localization import full_inputs, load_dev
from training.phase4hb_progressive_unfreezing_stage1 import train_ids
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import summarize, tensor_state_sha256
from tools.phase4f import Phase4FStore, evidence_feature, invalid_record, load_evidence_source, load_rectifier, load_sam_runtime, mask_loss

SEED, EPOCHS, BATCH, LR, WD, GRAD_CLIP = hd.SEED, hd.EPOCHS, hd.BATCH, hd.LR, hd.WD, hd.GRAD_CLIP
CFG = hd.CFG
CACHE = Path("cache/phase6e2_c1_specific_r1")
ARM = os.environ.get("PHASE6E2_ARM", "main")
if ARM not in ("main", "i1", "i2"): raise RuntimeError(f"unsupported Phase6E.2 arm: {ARM}")
BASE_OUT = ROOT / "outputs/phase6e2_c1_specific_r1"
OUT = BASE_OUT if ARM == "main" else BASE_OUT / ARM
BASE_CKPTS = Path("checkpoints/phase6e2_c1_specific_r1")
CKPTS = BASE_CKPTS if ARM == "main" else BASE_CKPTS / ARM
SELECTED = OUT / "selected_checkpoint.pt"
OLD_SELECTED = ROOT / "outputs/phase4hd/r1/selected_checkpoint.pt"
C1_SHA = "85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def load_c1_cache(split: str, expected_ids: list[str]) -> dict:
    values = defaultdict(list)
    ids = []
    for path in sorted((CACHE / split).glob("shard_*.pt")):
        x = torch.load(path, map_location="cpu", weights_only=False)
        if x.get("c1_sha256") != C1_SHA:
            raise RuntimeError("C1 cache checkpoint drift")
        ids += x["sample_ids"]
        for key in ("q_seg", "valid", "seg_count"):
            values[key].append(x[key])
    if ids != expected_ids:
        raise RuntimeError(f"{split} C1 cache incomplete/order drift")
    return {"sample_ids": ids, **{k: torch.cat(v) for k, v in values.items()}}


def c1_language_batch(data: dict, cache: dict, idx: torch.Tensor, store: Phase4FStore,
                      sam, device: torch.device) -> tuple[dict, torch.Tensor]:
    batch = full_inputs(data, idx, device)
    qseg = cache["q_seg"].index_select(0, idx).to(device=device, dtype=torch.bfloat16)
    ids = [cache["sample_ids"][i] for i in idx.tolist()]
    # Phase4FStore.batch() also fetches the historical P1 G0 q_seg.  That
    # tensor is intentionally absent for the 146 samples on which P1 did not
    # emit exactly one SEG, and it is not an input to this C1-specific arm.
    # Fetch only the shared spatial tensors so a valid C1 SEG can supervise
    # those samples without accidentally depending on the old P1 trigger.
    raw, _, _, _, _ = phase4f_spatial_batch(store, ids, device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        low = sam(qseg, raw.to(torch.bfloat16))
    zl = []
    for j, sid in enumerate(ids):
        from model.pcerf import sam_lowres_to_original_normalized
        zl.append(sam_lowres_to_original_normalized(low[j:j + 1], store.geometries[sid], output_hw=(256, 256)))
    batch["q_seg"] = qseg
    batch["z_L"] = torch.cat(zl).to(device=device, dtype=torch.bfloat16)
    return batch, raw


def phase4f_spatial_batch(store: Phase4FStore, ids: list[str], device: torch.device):
    """Phase4F spatial payload without the obsolete P1 q_seg side field."""
    values = [store._values(sid) for sid in ids]
    return (
        torch.stack([row[0] for row in values]).to(device=device, dtype=torch.bfloat16, non_blocking=True),
        torch.stack([row[1] for row in values]).to(device=device, dtype=torch.bfloat16, non_blocking=True),
        torch.stack([row[2] for row in values]).to(device=device, non_blocking=True),
        torch.stack([store.sam_coords[sid] for sid in ids]).to(device=device, non_blocking=True),
        torch.stack([store.clip_coords[sid] for sid in ids]).to(device=device, non_blocking=True),
    )


def phase4f_batch(store, ids, source, rectifier, device):
    s64, raw, targets, sc, cc = phase4f_spatial_batch(store, ids, device)
    with torch.no_grad():
        evidence = evidence_feature(source, raw)
    valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        value = rectifier(s64, evidence, sc, cc, valid)
    return s64, value["image_embeddings"], value["support"].reshape(len(ids), 1, 64, 64), targets, sc


def evaluate(model, rectifier, sam, source, store, dev, cache, device):
    records = []
    model.eval(); model.language_source.eval(); model.forensic_source.eval(); rectifier.eval()
    with torch.no_grad():
        for i, sid in enumerate(dev["sample_ids"]):
            if not bool(cache["valid"][i]):
                row = invalid_record(sid, dev["original_masks"][i]); row["valid_g0"] = False; records.append(row); continue
            idx = torch.tensor([i])
            batch, _ = c1_language_batch(dev, cache, idx, store, sam, device)
            s64, p4f, support, _, sc = phase4f_batch(store, [sid], source, rectifier, device)
            out = hc.utility_forward(model, batch)
            gate = hc.gate_to_sam_grid(out["U"], sc) * support.float()
            adapted = hc.gated_embedding(s64, p4f, gate)
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(batch["q_seg"], adapted.to(torch.bfloat16))
            logits = inverse_sam_logits(low, dev["sam_geometries"][i])
            row = metric_record(sid, logits, dev["original_masks"][i]); row["valid_g0"] = True; records.append(row)
            if (i + 1) % 200 == 0:
                print(json.dumps({"stage": "INTERNAL_VAL", "done": i + 1, "total": len(dev["sample_ids"])}), flush=True)
    return summarize(records), records


def write_history(rows):
    path = OUT / "training_curve.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    device = torch.device("cuda:0"); torch.cuda.set_device(device); hc.seed_all()
    cache_status = json.loads((BASE_OUT / "cache/status.json").read_text())
    if cache_status.get("status") != "COMPLETE" or cache_status.get("c1_sha256") != C1_SHA:
        raise RuntimeError("C1 cache not complete")
    if SELECTED.exists() or list(CKPTS.glob("epoch_*.pt")):
        raise RuntimeError("Phase6E.2 training output exists; refusing implicit continuation/overwrite")
    if ARM == "main":
        model, rectifier, rectifier_path = hd.load_common(device)
    elif ARM == "i1":
        # I1 isolates utility initialization: scratch A2 utility plus the
        # selected Phase4F epoch-9 rectifier used at the historical R1 start.
        model, _ = hc.load_utility("a2", device)
        rectifier, rectifier_path = hc.load_phase4f(device)
    else:
        # I2: reproduce the original seed-3407 scratch constructors for both
        # trainable branches.  No Phase4H-C or Phase4F trained state is loaded.
        model, _ = hc.load_utility("a2", device)
        scale = json.loads((ROOT / "outputs/phase4f_language_preserving_rectification/preflight/geometry_scale_audit.json").read_text())
        if scale.get("status") != "PASS": raise RuntimeError("Phase4F geometry initialization audit drift")
        rectifier = load_rectifier(CFG, float(scale["selected_gamma"]), device)
        rectifier_path = None
    rectifier.requires_grad_(True)
    old_audit = json.loads((ROOT / "outputs/phase4hd/rectifier_audit.json").read_text())
    init = {"utility_state_sha256": tensor_state_sha256(model.state_dict()),
            "rectifier_state_sha256": tensor_state_sha256(rectifier.state_dict())}
    expected = old_audit["common_initialization"]
    if ARM == "main":
        if init["utility_state_sha256"] != expected["utility_state_sha256"] or init["rectifier_state_sha256"] != expected["rectifier_state_sha256"]:
            raise RuntimeError("pre-Phase4H-D initialization was not reproduced")
    else:
        a2 = json.loads((ROOT / "outputs/phase4hc/a2/selector.json").read_text())
        random_utility_hash = tensor_state_sha256(hc.trainable_state(model))
        if random_utility_hash != a2["initialization"]["trainable_state_sha256"]:
            raise RuntimeError(f"{ARM} random utility initialization drift")
        if ARM == "i1":
            if init["rectifier_state_sha256"] != expected["rectifier_state_sha256"]:
                raise RuntimeError("I1 Phase4F epoch9 rectifier initialization drift")
        else:
            random_rectifier_hash = tensor_state_sha256(rectifier.state_dict())
            if random_rectifier_hash != "5099ffcd68b1c4aecfbbedc29cf0935adf034236f08e8788d737db940dc05cea":
                raise RuntimeError("I2 random rectifier initialization drift")
    manifest = {"schema": "phase6e2_initialization_provenance_v1", "status": "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        "base": "frozen C1 epoch5/step2500 actual canonical-G0 SEG query cache", "c1_sha256": C1_SHA,
        "arm": ARM,
        "new_r1_initialization": ({**init, "kind": "A2_epoch3_plus_Phase4F_epoch9",
            "utility_source": str(hd.A2_SELECTED.resolve()), "utility_source_sha256": file_sha256(hd.A2_SELECTED),
            "rectifier_source": str(rectifier_path.resolve()), "rectifier_source_sha256": file_sha256(rectifier_path)}
            if ARM == "main" else ({**init, "kind": "I1_random_utility_plus_Phase4F_epoch9_rectifier",
                "seed": SEED, "phase4hc_a2_random_utility_trainable_sha256": random_utility_hash,
                "rectifier_source": str(rectifier_path.resolve()), "rectifier_source_sha256": file_sha256(rectifier_path),
                "trained_utility_checkpoint_loaded": False, "trained_rectifier_checkpoint_loaded": True}
            if ARM == "i1" else {**init, "kind": "I2_random_utility_plus_random_rectifier",
                "seed": SEED, "phase4hc_a2_random_utility_trainable_sha256": random_utility_hash,
                "phase4f_random_rectifier_sha256": random_rectifier_hash,
                "phase4f_selected_gamma_initial": float(scale["selected_gamma"]),
                "trained_utility_checkpoint_loaded": False, "trained_rectifier_checkpoint_loaded": False})),
        "old_selected_checkpoint": {"path": str(OLD_SELECTED.resolve()), "sha256": file_sha256(OLD_SELECTED),
            "loaded_as_initialization": False, "explicitly_forbidden": True},
        "old_new_init_difference_audit": ("new init hashes exactly equal Phase4H-D pre-training audit; old selected was not torch.load'ed"
            if ARM == "main" else ("I1 replaces only A2 epoch3 utility with its exact seed-3407 scratch initialization; Phase4F epoch9 rectifier is unchanged"
            if ARM == "i1" else "I2 uses exact seed-3407 scratch constructors; no trained A2/Phase4F/old-R1 state loaded")),
        "recipe": hd.config_contract(), "differences": ["base q_seg and derived z_L come from frozen C1 rather than P1",
            "actual C1 exactly-one-SEG validity controls optimization/validation eligibility"],
        "sam_path": "unchanged frozen Phase4F/P1 SAM runtime for matched old/new R1 comparison",
        "trainable": {"utility": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "rectifier": sum(p.numel() for p in rectifier.parameters() if p.requires_grad)},
        "frozen": {"C1": True, "LoRA": True, "H2": True, "FRC_projector": True, "RINE_eval": True},
        "firewall": {"internal_test": False, "official1000_before_selection": False}}
    if manifest["trainable"] != {"utility": 371803, "rectifier": 329985}:
        raise RuntimeError(f"trainable scope drift: {manifest['trainable']}")
    dump(OUT / "initialization_provenance.json", manifest)
    train_store, val_store = Phase4FStore(CFG, "train"), Phase4FStore(CFG, "val")
    ids = train_ids(); dev = load_dev("g0")
    train_cache, val_cache = load_c1_cache("train", ids), load_c1_cache("val", dev["sample_ids"])
    if ids != train_store.sample_ids or dev["sample_ids"] != val_store.sample_ids:
        raise RuntimeError("Phase4F spatial population/order drift")
    missing_old_q_train = [sid for sid in ids if sid not in train_store.q]
    missing_old_q_val = [sid for sid in dev["sample_ids"] if sid not in val_store.q]
    dependency_audit = {
        "schema": "phase6e2_c1_query_dependency_audit_v1",
        "status": "PASS",
        "spatial_population_order_exact": True,
        "old_p1_q_seg_is_not_consumed": True,
        "train_old_p1_q_missing": len(missing_old_q_train),
        "train_c1_valid_among_old_p1_q_missing": sum(
            bool(train_cache["valid"][i]) for i, sid in enumerate(ids) if sid not in train_store.q),
        "validation_old_p1_q_missing": len(missing_old_q_val),
        "validation_c1_valid_among_old_p1_q_missing": sum(
            bool(val_cache["valid"][i]) for i, sid in enumerate(dev["sample_ids"]) if sid not in val_store.q),
        "eligibility_source": "frozen C1 canonical-G0 exactly-one-SEG validity",
    }
    dump(OUT / "c1_query_dependency_audit.json", dependency_audit)
    data = q.load_ids(ids, ("valid_g0", "S64", "q_seg", "z_L", "F24", "z_F24", "target64", "clip_geometries"))
    valid = train_cache["valid"].bool(); id_to_i = {sid: i for i, sid in enumerate(ids)}
    sam = load_sam_runtime(CFG, device); source = load_evidence_source(CFG, "forensic_rect", device)
    frozen_before = {"sam": tensor_state_sha256(sam.state_dict()), "source": tensor_state_sha256(source.state_dict()),
                     "heads": tensor_state_sha256(q.source_state(model))}
    params = [p for p in model.parameters() if p.requires_grad] + [p for p in rectifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    cross = torch.tensor([(i + 1) % len(ids) for i in range(len(ids))])
    permutation = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    history, updates = [], 0
    for epoch in range(1, EPOCHS + 1):
        began = time.time(); order = list(ids); random.Random(SEED + 1009 * epoch).shuffle(order)
        sums = defaultdict(float); traversal = eligible = invalid = 0
        model.train(); model.language_source.eval(); model.forensic_source.eval(); rectifier.train()
        for begin in range(0, len(order), BATCH):
            all_batch = order[begin:begin + BATCH]
            positions = torch.tensor([id_to_i[sid] for sid in all_batch])
            idx = positions[valid.index_select(0, positions)]
            traversal += len(all_batch); eligible += len(idx); invalid += len(all_batch) - len(idx)
            if not len(idx): continue
            batch, _ = c1_language_batch(data, train_cache, idx, train_store, sam, device)
            fi = cross.index_select(0, idx); crossed = dict(batch)
            crossed["F24"] = data["F24"].index_select(0, fi).to(device)
            crossed["z_F24"] = data["z_F24"].index_select(0, fi).to(device)
            batch_ids = [ids[i] for i in idx.tolist()]
            s64, p4f, support, targets, sc = phase4f_batch(train_store, batch_ids, source, rectifier, device)
            optimizer.zero_grad(set_to_none=True)
            matched = hc.utility_forward(model, batch); crossed_out = hc.utility_forward(model, crossed)
            shuffled = hc.utility_forward(model, batch, permutation=permutation)
            gate = hc.gate_to_sam_grid(matched["U"], sc) * support.float()
            adapted = hc.gated_embedding(s64, p4f, gate)
            with torch.autocast(device_type=device.type, enabled=False): low = sam(batch["q_seg"], adapted.to(torch.bfloat16))
            seg = mask_loss(low, targets, CFG)
            _, soft = q.target_delta({"p_L": matched["p_L"], "p_F": matched["p_F"]}, data["target64"].index_select(0, idx).to(device))
            relative = q.image_balanced_loss(matched["utility_logit"], soft, matched["support"])
            cr = hc.rank_loss(matched["U"], crossed_out["U"], matched["support"])
            sr = hc.rank_loss(matched["U"], shuffled["U"], matched["support"])
            ranking = .5 * (cr + sr); total = seg["total"] + relative + ranking
            if not torch.isfinite(total): raise RuntimeError("nonfinite loss")
            total.backward(); norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            if not torch.isfinite(norm): raise RuntimeError("nonfinite gradient")
            optimizer.step(); updates += 1; n = len(idx)
            for key, value in (("seg_loss", seg["total"]), ("relative_loss", relative), ("ranking_loss", ranking),
                               ("cross_rank_loss", cr), ("shuffle_rank_loss", sr), ("total_loss", total)):
                sums[key] += float(value.detach()) * n
            if updates <= 3 or updates % 100 == 0:
                print(json.dumps({"stage": "TRAIN", "epoch": epoch, "update": updates,
                                  "loss": float(total.detach()), "grad_norm": float(norm)}), flush=True)
        val_metric, _ = evaluate(model, rectifier, sam, source, val_store, dev, val_cache, device)
        row = {"epoch": epoch, "optimizer_updates": updates,
               **{k: sums[k] / eligible for k in ("seg_loss", "relative_loss", "ranking_loss", "cross_rank_loss", "shuffle_rank_loss", "total_loss")},
               "traversal_exposures": traversal, "optimization_eligible_exposures": eligible,
               "invalid_g0_exposures": invalid, "dev_g0_mean_iou": val_metric["mean_foreground_iou"],
               "dev_g0_mean_f1": val_metric["mean_foreground_f1"], "sample_order_sha256": ids_hash(order),
               "seconds": time.time() - began}
        history.append(row); write_history(history)
        payload = {"schema": "phase6e2_c1_specific_r1_checkpoint_v1", "epoch": epoch, "optimizer_updates": updates,
                   "utility_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "rectifier_state": {k: v.detach().cpu() for k, v in rectifier.state_dict().items()},
                   "optimizer": optimizer.state_dict(), "validation_g0": val_metric, "c1_sha256": C1_SHA,
                   "initialization_provenance_sha256": file_sha256(OUT / "initialization_provenance.json")}
        CKPTS.mkdir(parents=True, exist_ok=True); path = CKPTS / f"epoch_{epoch}.pt"; tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp); os.replace(tmp, path)
        print(json.dumps({"stage": "EPOCH_COMPLETE", **row}), flush=True)
    best = max(history, key=lambda x: (x["dev_g0_mean_iou"], -x["epoch"]))
    chosen = CKPTS / f"epoch_{best['epoch']}.pt"; shutil.copy2(chosen, SELECTED)
    selector = {"schema": "phase6e2_selector_v1", "status": "COMPLETE", "primary": "internal validation canonical G0 mean IoU",
                "tie_break": "earlier epoch", "selected_epoch": best["epoch"], "selected_dev_g0": best["dev_g0_mean_iou"],
                "selected_checkpoint_sha256": file_sha256(SELECTED), "candidates": history,
                "official1000_used": False, "internal_test_used": False}
    dump(OUT / "selector.json", selector)
    frozen_after = {"sam": tensor_state_sha256(sam.state_dict()), "source": tensor_state_sha256(source.state_dict()),
                    "heads": tensor_state_sha256(q.source_state(model))}
    if frozen_after != frozen_before: raise RuntimeError("frozen runtime/source mutation")
    manifest.update({"status": "TRAINING_COMPLETE_SELECTED_FROZEN", "selected_checkpoint_sha256": file_sha256(SELECTED),
                     "selected_epoch": best["epoch"], "formal_optimizer_updates": updates,
                     "actual_population": {"train": len(ids), "train_valid": int(train_cache["valid"].sum()),
                                           "train_invalid": int((~train_cache["valid"]).sum()), "validation": len(dev["sample_ids"]),
                                           "validation_valid": int(val_cache["valid"].sum()), "validation_invalid": int((~val_cache["valid"]).sum())},
                     "frozen_hash_before": frozen_before, "frozen_hash_after": frozen_after})
    dump(OUT / "initialization_provenance.json", manifest)


if __name__ == "__main__":
    main()
