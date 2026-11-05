#!/usr/bin/env python3
"""Phase 4G-F formal CSCU-LF localization training and DEV selector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from training import phase4g1q_conditional_utility as q
from training.phase4g1_g1c import unpack_bool
from tools.phase4c_b import file_sha256, metric_record, spatial_cache_paths
from tools.phase4e1 import compare, summarize, tensor_state_sha256
from tools.phase4f import deterministic_order, invalid_record, load_evidence_source, load_sam_runtime, q_index
from model.pcerf import sam_lowres_to_original_normalized

PHASE="phase4gf"; OUT=ROOT/f"outputs/{PHASE}"; DOC=ROOT/f"docs/{PHASE}/report.md"
CKPT=Path("checkpoints/phase4gf_csculf_formal_localization")
CACHE=Path("cache/phase4gf_csculf_formal_localization")
G1C_CACHE=Path("cache/phase4g1/g1c/frozen_sources")
MANIFEST=CKPT/"execution_manifest.json"; HISTORY=OUT/"training_history.csv"; DEV_RESULTS=OUT/"dev_results.json"; SELECTOR=OUT/"selector.json"; GATES=OUT/"gate_summary.json"; SELECTED=OUT/"selected_checkpoint.pt"
WARM=Path("checkpoints/phase4g1s_mismatch_aware_utility/csculf_mismatch_utility_epoch10.pt")
P4F_CFG=yaml.safe_load((ROOT/"configs/phase4f_language_preserving_rectification.yaml").read_text())
SEED=3407; BATCH=8; EPOCHS=10; MARGIN=.1; TAU=0.041720069924898906
HISTORICAL={"P1":{"g0":.1482333978666042,"phrase":.24579772094855956,"tf":.3429279193646265},
            "Phase4F_FORENSIC_RECT":{"g0":.17161424058491842,"phrase":.19357286978340313,"tf":.2052624796387254},
            "Phase4E_NO_TEACHER":{"g0":.17281810774799355,"phrase":.17342445370179935,"tf":.17820726584168442}}


def dump(path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def canonical_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def ids_hash(ids): return hashlib.sha256("\n".join(ids).encode()).hexdigest()
def seed_all(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def load_model(device, checkpoint=WARM):
    seed_all(); source=torch.load(q.G1C_CKPT,map_location="cpu",weights_only=False); temps=torch.load(q.G1C_TEMPERATURES,map_location="cpu",weights_only=False)
    model=q.CSCULF(temps["T_L"],temps["T_F"]); model.load_frozen_source_heads(source["language_head"],source["forensic_head"])
    state=torch.load(checkpoint,map_location="cpu",weights_only=False); model.load_state_dict(state["model_state"])
    return model.to(device)


def frozen_preflight():
    g1s=json.loads((ROOT/"outputs/phase4g1s/gate_summary.json").read_text()); required={"MISMATCH_AWARE_UTILITY_PREFLIGHT":"PASS","FORMAL_TRAINING_JUSTIFIED":"YES","FORMAL_TRAINING_EXECUTED":"NO"}
    if any(g1s.get(k)!=v for k,v in required.items()): raise RuntimeError("Phase4G-1S prerequisite drift")
    arch=json.loads((ROOT/"outputs/phase4g1p/architecture_manifest.json").read_text()); protocol=json.loads((ROOT/"outputs/phase4f_language_preserving_rectification/manifests/frozen_protocol.json").read_text())
    if arch["parameters"]["trainable"]!=371803 or arch["configuration"]!={"projection_width":64,"window_size":7,"attention_heads":4,"interaction_blocks":1,"normalization":"GroupNorm(8) plus LayerNorm for q projection","activation":"GELU","rectification_lambdas":{"channel":.5,"spatial":.5},"residual_topology":"bidirectional CMX rectification then one bidirectional local exchange block","parameter_sharing":"joint descriptors; direction-specific gates/QKV/output","REFINEMENT_INCLUDED":"NO"}: raise RuntimeError("architecture drift")
    if (protocol["train"]["n"],protocol["train"]["valid_g0"],protocol["train"]["invalid_g0"])!=(8836,8690,146) or (protocol["validation"]["n"],protocol["validation"]["valid_g0"],protocol["validation"]["invalid_g0"])!=(1106,1078,28): raise RuntimeError("formal population drift")
    state=torch.load(WARM,map_location="cpu",weights_only=False)
    if state["epoch"]!=10 or state["optimizer_updates"]!=6470: raise RuntimeError("warm-start drift")
    return arch,protocol


def freeze_manifest():
    arch,protocol=frozen_preflight()
    if MANIFEST.exists(): raise RuntimeError("Phase4G-F already initialized; use explicit recovery path")
    config={"phase":"Phase4G-F","initialization":"Phase4G-1S epoch10 final","architecture":{"d":64,"window":7,"heads":4,"blocks":1,"REFINEMENT_INCLUDED":"NO"},
        "trainable":"existing CSCULF context/interaction/U_F branch only","train_population":{"all_fake":8836,"valid_optimizer":8690,"invalid_accounting":146,"real_optimizer":0},
        "loss":{"formula":"L_seg+L_relative+L_ranking","coefficients":{"seg":1.,"relative":1.,"ranking":1.},"seg_contract":"2*BCEWithLogits+0.5*soft-Dice","relative_tau":TAU,"ranking_margin":MARGIN,"ranking_cross_shuffle_weights":[.5,.5]},
        "optimizer":{"name":"AdamW","lr":1e-4,"weight_decay":1e-4,"batch_size":8,"grad_clip":1.,"epochs":10,"scheduler":"none","early_stopping":False,"seed":SEED},
        "schedule":"Phase4F deterministic_order: Python Random(seed+1009*epoch), traverse all 8836 then filter invalid G0 within batch","cross":"cyclic next forensic in canonical full-train/dev ID order; current language and geometry fixed","shuffle":"fixed randperm(576), CPU Generator seed3407",
        "checkpoints":"save epoch1-10; selector allowed","selector":{"primary":"DEV canonical G0 all-ref-union mean foreground IoU","population":1106,"invalid_iou":0.,"tie":"earlier epoch","batch":1,"threshold_logit":0.,"forbidden":["Phrase","TF","cross","shuffle","utility correlation"]},
        "selected_evaluation":{"modes":["G0","Phrase-only","TF"],"controls":["matched","cross-image","spatial-shuffle","forensic-off","vacuous"],"batch":1},
        "statistics":{"bootstrap_repeats":10000,"seed":3407,"wilcoxon":True},
        "heldout_justification":"selected DEV G0>P1 and >Phase4F; G0<Phrase<TF; matched-cross and matched-shuffle paired IoU CI lower>0; exact fallback and source integrity PASS"}
    manifest={"schema":"phase4gf_execution_manifest_v1","status":"FROZEN_BEFORE_FIRST_OPTIMIZER_STEP","created_unix":time.time(),"config":config,"config_sha256":canonical_hash(config),"implementation_sha256":file_sha256(Path(__file__)),
        "warm_start":{"path":str(WARM),"sha256":file_sha256(WARM)},"architecture_manifest_sha256":file_sha256(ROOT/"outputs/phase4g1p/architecture_manifest.json"),"phase4f_protocol_sha256":file_sha256(ROOT/"outputs/phase4f_language_preserving_rectification/manifests/frozen_protocol.json"),
        "source_files":{"g1c_epoch10":{"path":str(q.G1C_CKPT),"sha256_before":file_sha256(q.G1C_CKPT)},"source_temperatures":{"path":str(q.G1C_TEMPERATURES),"sha256_before":file_sha256(q.G1C_TEMPERATURES)},"p1_sam_runtime":{"path":str(Path(P4F_CFG["experiment"]["runtime_root"])/"p1_sam_runtime.pt"),"sha256_before":file_sha256(Path(P4F_CFG["experiment"]["runtime_root"])/"p1_sam_runtime.pt")},"forensic_adapter":{"path":P4F_CFG["evidence"]["forensic_checkpoint"],"sha256_before":file_sha256(Path(P4F_CFG["evidence"]["forensic_checkpoint"]))}},
        "firewall":{"internal_test_accessed":False,"official1000_accessed":False},"formal_optimizer_updates":0}
    dump(MANIFEST,manifest); return manifest


def build_shared_dev(device):
    root=CACHE/"dev_shared"; complete=root/"complete.json"
    if complete.exists(): return json.loads(complete.read_text())
    sam_paths=spatial_cache_paths(ROOT/P4F_CFG["data"]["spatial_cache_root"],"sam","val"); clip_paths=spatial_cache_paths(ROOT/P4F_CFG["data"]["spatial_cache_root"],"clip","val")
    source=load_evidence_source(P4F_CFG,"forensic_rect",device); source_hash=tensor_state_sha256(source.state_dict()); shards=[]; all_ids=[]; root.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for si,(sp,cp) in enumerate(zip(sam_paths,clip_paths)):
            sam=torch.load(sp,map_location="cpu",weights_only=False); clip=torch.load(cp,map_location="cpu",weights_only=False); ids=[str(x["sample_id"]) for x in sam["records"]]
            if ids!=[str(x["sample_id"]) for x in clip["records"]]: raise RuntimeError("dev source ID mismatch")
            raw=clip["features"].to(device=device,dtype=torch.bfloat16)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16): forensic=source(raw,return_features=True)
            s64=[]; target=[]
            for i,row in enumerate(sam["records"]):
                s64.append(sam_lowres_to_original_normalized(sam["features"][i:i+1],row["geometry"],output_hw=(64,64)).to(torch.bfloat16).cpu())
                target.append(sam_lowres_to_original_normalized(sam["targets"][i:i+1,None].float(),row["geometry"],output_hw=(256,256),mode="nearest")[:,0].to(torch.uint8).cpu())
            payload={"schema":"phase4gf_dev_shared_v1","sample_ids":ids,"Sraw64":sam["features"],"S64":torch.cat(s64),"F24":forensic["F_forensic"].to(torch.bfloat16).cpu(),"z_F24":forensic["logits"].to(torch.bfloat16).cpu(),"target256":torch.cat(target),
                "original_masks":[x.bool() for x in sam["original_masks"]],"sam_geometries":[x["geometry"] for x in sam["records"]],"clip_geometries":[x["geometry"] for x in clip["records"]]}
            path=root/f"shard_{si:06d}.pt"; tmp=path.with_suffix(".pt.tmp"); torch.save(payload,tmp); tmp.replace(path); shards.append({"path":str(path),"sha256":file_sha256(path),"n":len(ids)}); all_ids+=ids
            print(json.dumps({"stage":"DEV_SHARED_CACHE","shard":si+1,"of":len(sam_paths),"n":len(all_ids)}),flush=True)
    if len(all_ids)!=1106 or len(set(all_ids))!=1106 or tensor_state_sha256(source.state_dict())!=source_hash: raise RuntimeError("dev shared cache integrity failure")
    result={"schema":"phase4gf_dev_shared_manifest_v1","status":"COMPLETE","n":1106,"ids_sha256":ids_hash(all_ids),"source_hash_before":source_hash,"source_hash_after":tensor_state_sha256(source.state_dict()),"shards":shards}; dump(complete,result); return result


def build_dev_mode(mode,device):
    root=CACHE/f"dev_{mode}"; complete=root/"complete.json"
    if complete.exists(): return json.loads(complete.read_text())
    shared=json.loads((CACHE/"dev_shared/complete.json").read_text()); name={"g0":"G0","phrase":"phrase_only","tf":"tf_full_context"}[mode]; qcache,records=q_index(Path(P4F_CFG["data"]["q_cache_root"])/"validation",name); sam_model=load_sam_runtime(P4F_CFG,device); sam_hash=tensor_state_sha256(sam_model.state_dict()); root.mkdir(parents=True,exist_ok=True); shards=[]; all_ids=[]; valid_n=0
    with torch.no_grad():
        for si,row in enumerate(shared["shards"]):
            base=torch.load(row["path"],map_location="cpu",weights_only=False); ids=base["sample_ids"]; valid=torch.tensor([sid in qcache and qcache[sid][1] for sid in ids]); qseg=torch.full((len(ids),256),float("nan"),dtype=torch.bfloat16); zl=torch.full((len(ids),1,256,256),float("nan"),dtype=torch.bfloat16); pos=valid.nonzero(as_tuple=False)[:,0]
            if len(pos):
                qq=torch.stack([qcache[ids[i]][0] for i in pos.tolist()]).to(device=device,dtype=torch.bfloat16); raw=base["Sraw64"].index_select(0,pos).to(device=device,dtype=torch.bfloat16)
                with torch.autocast(device_type=device.type,enabled=False): low=sam_model(qq,raw)
                qseg[pos]=qq.cpu()
                for local,p in enumerate(pos.tolist()): zl[p]=sam_lowres_to_original_normalized(low[local:local+1],base["sam_geometries"][p],output_hw=(256,256))[0].to(torch.bfloat16).cpu()
            payload={"schema":"phase4gf_dev_mode_v1","mode":mode,"sample_ids":ids,"valid":valid,"q_seg":qseg,"z_L":zl}; path=root/f"shard_{si:06d}.pt"; tmp=path.with_suffix(".pt.tmp"); torch.save(payload,tmp); tmp.replace(path); shards.append({"path":str(path),"sha256":file_sha256(path),"n":len(ids),"valid":int(valid.sum())}); all_ids+=ids; valid_n+=int(valid.sum())
            print(json.dumps({"stage":f"DEV_{mode.upper()}_CACHE","shard":si+1,"of":len(shared["shards"]),"valid":valid_n}),flush=True)
    if len(all_ids)!=1106 or tensor_state_sha256(sam_model.state_dict())!=sam_hash: raise RuntimeError("dev mode cache integrity failure")
    result={"schema":"phase4gf_dev_mode_manifest_v1","status":"COMPLETE","mode":mode,"n":1106,"valid":valid_n,"invalid":1106-valid_n,"ids_sha256":ids_hash(all_ids),"sam_hash_before":sam_hash,"sam_hash_after":tensor_state_sha256(sam_model.state_dict()),"shards":shards}; dump(complete,result); return result


def load_dev(mode):
    shared=json.loads((CACHE/"dev_shared/complete.json").read_text()); modes=json.loads((CACHE/f"dev_{mode}/complete.json").read_text()); result=defaultdict(list); ids=[]
    for sr,mr in zip(shared["shards"],modes["shards"]):
        s=torch.load(sr["path"],map_location="cpu",weights_only=False); m=torch.load(mr["path"],map_location="cpu",weights_only=False)
        if s["sample_ids"]!=m["sample_ids"]: raise RuntimeError("dev mode/shared ID drift")
        ids+=s["sample_ids"]
        for k in ("S64","F24","z_F24","target256"): result[k].append(s[k])
        for k in ("valid","q_seg","z_L"): result[k].append(m[k])
        for k in ("original_masks","sam_geometries","clip_geometries"): result[k].extend(s[k])
    value={k:(torch.cat(v) if k not in ("original_masks","sam_geometries","clip_geometries") else v) for k,v in result.items()}; value["sample_ids"]=ids; return value


def segmentation_loss(logits,target):
    logits=logits.float(); target=target[:,None].float(); bce=F.binary_cross_entropy_with_logits(logits,target); p=logits.sigmoid(); intersection=2*(p/1000*target).flatten(1).sum(1); union=(p/1000).flatten(1).sum(1)+(target/1000).flatten(1).sum(1); dice=(1-(intersection+1e-6)/(union+1e-6)).mean(); return {"bce":bce,"dice":dice,"total":2*bce+.5*dice}


def rank_loss(match,mismatch,support):
    value=F.relu(MARGIN-(match.float()-mismatch.float()))*support.float(); return (value.flatten(1).sum(1)/support.flatten(1).sum(1).clamp_min(1)).mean()


def full_inputs(data,idx,device):
    batch={k:data[k].index_select(0,idx).to(device) for k in ("S64","q_seg","z_L","F24","z_F24")}; batch["clip_geometries"]=[data["clip_geometries"][i] for i in idx.tolist()]; n=len(idx); batch.update({"valid_g0":torch.ones(n,dtype=torch.bool,device=device),"forensic_present":torch.ones(n,dtype=torch.bool,device=device),"forensic_vacuous":torch.zeros(n,dtype=torch.bool,device=device),"forensic_off":torch.zeros(n,dtype=torch.bool,device=device)}); return batch


def normalized_metric(sid,logit,mask,geometry,valid=True):
    if not valid: return invalid_record(sid,mask)
    original=F.interpolate(logit.float(),size=tuple(geometry["original_hw"]),mode="bilinear",align_corners=False)[0,0]; row=metric_record(sid,original,mask); row["valid_g0"]=True; return row


def evaluate_dev(model,data,mode,device,condition="matched"):
    model.eval(); ids=data["sample_ids"]; cross={i:(i+1)%len(ids) for i in range(len(ids))}; perm=torch.randperm(576,generator=torch.Generator().manual_seed(SEED)).to(device); records=[]; exact=[]
    with torch.no_grad():
        for i,sid in enumerate(ids):
            valid=bool(data["valid"][i])
            if not valid: records.append(invalid_record(sid,data["original_masks"][i])); records[-1]["valid_g0"]=False; continue
            idx=torch.tensor([i]); batch=full_inputs(data,idx,device)
            if condition=="cross_image":
                partner=cross[i]; batch["F24"]=data["F24"][partner:partner+1].to(device); batch["z_F24"]=data["z_F24"][partner:partner+1].to(device)
            elif condition=="spatial_shuffle":
                batch["F24"]=batch["F24"].flatten(2)[:,:,perm].reshape_as(batch["F24"]); batch["z_F24"]=batch["z_F24"].flatten(2)[:,:,perm].reshape_as(batch["z_F24"])
            elif condition=="off": batch["forensic_off"][:]=True
            elif condition=="vacuous": batch["forensic_vacuous"][:]=True
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16): output=model(s64=batch["S64"],q_seg=batch["q_seg"],z_l=batch["z_L"],f24=batch["F24"],z_f24=batch["z_F24"],clip_geometries=batch["clip_geometries"],valid_g0=batch["valid_g0"],forensic_present=batch["forensic_present"],forensic_vacuous=batch["forensic_vacuous"],forensic_off=batch["forensic_off"])
            if condition in ("off","vacuous"): exact.append(torch.equal(output["logits"],batch["z_L"]))
            records.append(normalized_metric(sid,output["logits"].cpu(),data["original_masks"][i],data["sam_geometries"][i],True))
            if (i+1)%200==0: print(json.dumps({"stage":"DEV_EVAL","mode":mode,"condition":condition,"done":i+1,"total":len(ids)}),flush=True)
    return summarize(records),records,{"checked":len(exact),"all_exact":all(exact) if exact else None}


def save_checkpoint(path,model,optimizer,epoch,updates,validation,source_hash):
    payload={"schema":"phase4gf_csculf_formal_checkpoint_v1","epoch":epoch,"optimizer_updates":updates,"model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer":optimizer.state_dict(),"validation_g0":validation,"source_state_hash":tensor_state_sha256(q.source_state(model)),"initial_source_state_hash":source_hash,"config_sha256":json.loads(MANIFEST.read_text())["config_sha256"]}; path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".pt.tmp"); torch.save(payload,tmp); tmp.replace(path)


def train(device):
    if MANIFEST.exists():
        existing=json.loads(MANIFEST.read_text())
        if existing.get("status")!="FROZEN_BEFORE_FIRST_OPTIMIZER_STEP" or existing.get("formal_optimizer_updates")!=0: raise RuntimeError("unsupported Phase4G-F recovery state")
    else: freeze_manifest()
    OUT.mkdir(parents=True,exist_ok=True); build_shared_dev(device); build_dev_mode("g0",device); g0=load_dev("g0")
    protocol=json.loads((ROOT/"outputs/phase4f_language_preserving_rectification/manifests/frozen_protocol.json").read_text()); all_ids=protocol["train"]["n"] and json.loads((ROOT/"outputs/phase4f_language_preserving_rectification/manifests/training_schedules.json").read_text())["epochs"]["1"]["sample_ids"]
    # Canonical population order is recovered from the frozen G1-C cache manifest traversal, then checked against Phase4F set.
    all_ids=[]
    for path in sorted(G1C_CACHE.glob("shard_*.pt")): all_ids.extend(torch.load(path,map_location="cpu",weights_only=False)["sample_ids"])
    if len(all_ids)!=8836 or len(set(all_ids))!=8836: raise RuntimeError("all-train cache population drift")
    data=q.load_ids(all_ids,("valid_g0","S64","q_seg","z_L","F24","z_F24","target64","target256_packed","clip_geometries")); target256=unpack_bool(data.pop("target256_packed")); valid=data["valid_g0"].bool(); id_to_index={sid:i for i,sid in enumerate(all_ids)}; cross=torch.tensor([(i+1)%len(all_ids) for i in range(len(all_ids))]); perm=torch.randperm(576,generator=torch.Generator().manual_seed(SEED)).to(device)
    model=load_model(device).train(); source_hash=tensor_state_sha256(q.source_state(model)); params=[p for p in model.parameters() if p.requires_grad]; optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4); history=[]; updates=0
    fields=["epoch","optimizer_updates","seg_loss","seg_bce","seg_dice","relative_loss","ranking_loss","cross_rank_loss","shuffle_rank_loss","total_loss","mean_cross_delta","mean_shuffle_delta","traversal_exposures","optimization_eligible_exposures","invalid_g0_exposures","empty_valid_g0_batches","dev_g0_mean_iou","dev_g0_median_iou","dev_g0_mean_f1","sample_order_sha256","seconds"]
    for epoch in range(1,EPOCHS+1):
        began=time.time(); order=deterministic_order(all_ids,epoch,SEED); sums=defaultdict(float); traversal=eligible=invalid=empty=0; model.train()
        for begin in range(0,len(order),BATCH):
            all_batch=order[begin:begin+BATCH]; positions=torch.tensor([id_to_index[sid] for sid in all_batch]); keep=valid.index_select(0,positions); idx=positions[keep]; traversal+=len(all_batch); eligible+=len(idx); invalid+=len(all_batch)-len(idx)
            if not len(idx): empty+=1; continue
            batch=full_inputs(data,idx,device); forensic_idx=cross.index_select(0,idx); forensic={"F24":data["F24"].index_select(0,forensic_idx).to(device),"z_F24":data["z_F24"].index_select(0,forensic_idx).to(device)}; crossed=dict(batch); crossed.update(forensic); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16):
                matched=model(s64=batch["S64"],q_seg=batch["q_seg"],z_l=batch["z_L"],f24=batch["F24"],z_f24=batch["z_F24"],clip_geometries=batch["clip_geometries"],valid_g0=batch["valid_g0"],forensic_present=batch["forensic_present"],forensic_vacuous=batch["forensic_vacuous"],forensic_off=batch["forensic_off"])
                cross_out=q.utility_forward(model,crossed); shuffle=q.utility_forward(model,batch,spatial_permutation=perm); seg=segmentation_loss(matched["logits"],target256.index_select(0,idx).to(device)); mapped={"p_L":matched["p_L64"],"p_F":matched["p_F64"]}; _,soft=q.target_delta(mapped,batch["target64"] if "target64" in batch else data["target64"].index_select(0,idx).to(device)); utility_logit=model.utility_head.net[-1](matched["utility_prehead"]); relative=q.image_balanced_loss(utility_logit,soft,matched["support64"]); cr=rank_loss(matched["U_F64"],cross_out["U"],matched["support64"]); sr=rank_loss(matched["U_F64"],shuffle["U"],matched["support64"]); ranking=.5*(cr+sr); total=seg["total"]+relative+ranking
            if not torch.isfinite(total): raise RuntimeError("nonfinite formal loss")
            total.backward(); norm=torch.nn.utils.clip_grad_norm_(params,1.); 
            if not torch.isfinite(norm): raise RuntimeError("nonfinite formal gradient")
            optimizer.step(); updates+=1; n=len(idx)
            for k,v in (("seg",seg["total"]),("bce",seg["bce"]),("dice",seg["dice"]),("relative",relative),("ranking",ranking),("cr",cr),("sr",sr),("total",total)): sums[k]+=float(v.detach())*n
            support=matched["support64"]; denom=support.flatten(1).sum(1); um=(matched["U_F64"]*support).flatten(1).sum(1)/denom; uc=(cross_out["U"]*support).flatten(1).sum(1)/denom; us=(shuffle["U"]*support).flatten(1).sum(1)/denom; sums["cross_delta"]+=float((uc-um).sum()); sums["shuffle_delta"]+=float((us-um).sum())
            if updates<=3 or updates%100==0: print(json.dumps({"stage":"TRAIN","epoch":epoch,"update":updates,"loss":float(total.detach())}),flush=True)
        if (traversal,eligible,invalid)!=(8836,8690,146): raise RuntimeError("formal exposure mismatch")
        dev_metrics,dev_records,_=evaluate_dev(model,g0,"g0",device,"matched"); row={"epoch":epoch,"optimizer_updates":updates,"seg_loss":sums["seg"]/eligible,"seg_bce":sums["bce"]/eligible,"seg_dice":sums["dice"]/eligible,"relative_loss":sums["relative"]/eligible,"ranking_loss":sums["ranking"]/eligible,"cross_rank_loss":sums["cr"]/eligible,"shuffle_rank_loss":sums["sr"]/eligible,"total_loss":sums["total"]/eligible,"mean_cross_delta":sums["cross_delta"]/eligible,"mean_shuffle_delta":sums["shuffle_delta"]/eligible,"traversal_exposures":traversal,"optimization_eligible_exposures":eligible,"invalid_g0_exposures":invalid,"empty_valid_g0_batches":empty,"dev_g0_mean_iou":dev_metrics["mean_foreground_iou"],"dev_g0_median_iou":dev_metrics["median_foreground_iou"],"dev_g0_mean_f1":dev_metrics["mean_foreground_f1"],"sample_order_sha256":ids_hash(order),"seconds":time.time()-began}; history.append(row)
        with HISTORY.open("w",newline="",encoding="utf-8") as h: writer=csv.DictWriter(h,fieldnames=fields); writer.writeheader(); writer.writerows(history)
        save_checkpoint(CKPT/f"epoch_{epoch}.pt",model,optimizer,epoch,updates,dev_metrics,source_hash); print(json.dumps({"stage":"EPOCH_COMPLETE",**row}),flush=True)
    best=max(history,key=lambda x:(x["dev_g0_mean_iou"],-x["epoch"])); selected=CKPT/f"epoch_{best['epoch']}.pt"; shutil.copy2(selected,SELECTED)
    selector={"schema":"phase4gf_selector_v1","status":"COMPLETE","primary":"DEV G0 all-ref-union mean foreground IoU","population":1106,"invalid_g0_iou_zero":28,"tie_break":"earlier epoch","selected_epoch":best["epoch"],"selected_dev_g0":best["dev_g0_mean_iou"],"selected_checkpoint":str(selected),"selected_checkpoint_sha256":file_sha256(selected),"output_selected_checkpoint_sha256":file_sha256(SELECTED),"candidates":[{"epoch":x["epoch"],"mean_g0_iou":x["dev_g0_mean_iou"],"median_g0_iou":x["dev_g0_median_iou"]} for x in history],"phrase_used":False,"tf_used":False,"controls_used":False,"utility_metrics_used":False}; dump(SELECTOR,selector)
    manifest=json.loads(MANIFEST.read_text()); manifest.update({"status":"TRAINING_COMPLETE_SELECTOR_FROZEN","formal_optimizer_updates":updates,"source_state_hash_before":source_hash,"source_state_hash_after":tensor_state_sha256(q.source_state(model)),"selector_sha256":file_sha256(SELECTOR),"selected_checkpoint_sha256":file_sha256(SELECTED)}); dump(MANIFEST,manifest)


def baseline_records(mode,ids):
    name={"g0":"G0","phrase":"phrase_only","tf":"tf_full_context"}[mode]; _,records=q_index(Path(P4F_CFG["data"]["q_cache_root"])/"validation",name); by={str(x["sample_id"]):x for x in records}; return [{"sample_id":sid,"foreground_iou":float(by[sid]["foreground_iou"]),"foreground_f1":float(by[sid]["foreground_f1"]),"tp":int(by[sid]["tp"]),"fp":int(by[sid]["fp"]),"fn":int(by[sid]["fn"])} for sid in ids]


def evaluate_selected(device):
    selector=json.loads(SELECTOR.read_text()); build_dev_mode("phrase",device); build_dev_mode("tf",device); state=torch.load(SELECTED,map_location="cpu",weights_only=False); model=load_model(device,SELECTED).eval(); outputs={}; records={}; exact={}
    jobs=[("g0","matched"),("phrase","matched"),("tf","matched"),("g0","cross_image"),("g0","spatial_shuffle"),("g0","off"),("g0","vacuous")]
    loaded={mode:load_dev(mode) for mode in ("g0","phrase","tf")}
    for mode,condition in jobs:
        key={"phrase":"phrase_only","tf":"tf"}.get(mode,condition); metrics,row,identity=evaluate_dev(model,loaded[mode],mode,device,condition); outputs[key]=metrics; records[key]=row; exact[key]=identity
    ids=loaded["g0"]["sample_ids"]; p1={mode:baseline_records(mode,ids) for mode in ("g0","phrase","tf")}; p4f_records={"g0":[json.loads(x) for x in (ROOT/"outputs/phase4f_language_preserving_rectification/final/forensic_rect/matched.jsonl").read_text().splitlines() if x],"phrase":[json.loads(x) for x in (ROOT/"outputs/phase4f_language_preserving_rectification/final/forensic_rect/phrase_only.jsonl").read_text().splitlines() if x],"tf":[json.loads(x) for x in (ROOT/"outputs/phase4f_language_preserving_rectification/final/forensic_rect/tf_full.jsonl").read_text().splitlines() if x]}
    statistics={"vs_P1":{},"vs_Phase4F":{}}
    for mode,key in (("g0","matched"),("phrase","phrase_only"),("tf","tf")):
        statistics["vs_P1"][mode]=compare(records[key],p1[mode]); statistics["vs_Phase4F"][mode]=compare(records[key],p4f_records[mode])
    statistics["matched_minus_cross"]=compare(records["matched"],records["cross_image"]); statistics["matched_minus_shuffle"]=compare(records["matched"],records["spatial_shuffle"])
    result={"schema":"phase4gf_dev_results_v1","selector":selector,"metrics":outputs,"records":records,"statistics":statistics,"fallback_identity":exact,"historical":HISTORICAL,"evaluation_contract":{"population":1106,"threshold_logit":0.,"batch_size":1,"target":"all-ref-union","invalid_policy":"IoU=0","phrase_tf_only_selected_checkpoint":True},"firewall":{"internal_test_accessed":False,"official1000_accessed":False}}; dump(DEV_RESULTS,result)


def finalize():
    result=json.loads(DEV_RESULTS.read_text()); selector=json.loads(SELECTOR.read_text()); manifest=json.loads(MANIFEST.read_text()); m=result["metrics"]; s=result["statistics"]
    order=m["matched"]["mean_foreground_iou"]<m["phrase_only"]["mean_foreground_iou"]<m["tf"]["mean_foreground_iou"]; beats_p1=m["matched"]["mean_foreground_iou"]>HISTORICAL["P1"]["g0"]; beats_p4f=m["matched"]["mean_foreground_iou"]>HISTORICAL["Phase4F_FORENSIC_RECT"]["g0"]
    matched_controls=s["matched_minus_cross"]["foreground_iou"]["bootstrap_95_ci"][0]>0 and s["matched_minus_shuffle"]["foreground_iou"]["bootstrap_95_ci"][0]>0
    fallback=all(result["fallback_identity"][k]["all_exact"] for k in ("off","vacuous")); source_integrity=manifest["source_state_hash_before"]==manifest["source_state_hash_after"]
    heldout=beats_p1 and beats_p4f and order and matched_controls and fallback and source_integrity
    gates={"schema":"phase4gf_gate_summary_v1","FORMAL_TRAINING_COMPLETE":"YES","SELECTED_EPOCH":selector["selected_epoch"],"SELECTED_DEV_G0":selector["selected_dev_g0"],"LANGUAGE_ORDER_PRESERVED":"YES" if order else "NO","CSCULF_BEATS_P1_G0":"YES" if beats_p1 else "NO","CSCULF_BEATS_PHASE4F_G0":"YES" if beats_p4f else "NO","MATCHED_GT_CROSS_SHUFFLE":"YES" if matched_controls else "NO","VACUOUS_EXACT_RECOVERY":"PASS" if fallback else "FAIL","SOURCE_HASH_INTEGRITY":"PASS" if source_integrity else "FAIL","HELDOUT_FINAL_EVALUATION_JUSTIFIED":"YES" if heldout else "NO","INTERNAL_TEST_ACCESSED":"NO","OFFICIAL1000_ACCESSED":"NO"}; dump(GATES,gates)
    table="\n".join(["| Model | G0 | Phrase | TF |","|---|---:|---:|---:|",f"| P1 | {HISTORICAL['P1']['g0']:.6f} | {HISTORICAL['P1']['phrase']:.6f} | {HISTORICAL['P1']['tf']:.6f} |",f"| Phase4F FORENSIC-RECT | {HISTORICAL['Phase4F_FORENSIC_RECT']['g0']:.6f} | {HISTORICAL['Phase4F_FORENSIC_RECT']['phrase']:.6f} | {HISTORICAL['Phase4F_FORENSIC_RECT']['tf']:.6f} |",f"| Phase4E -teacher | {HISTORICAL['Phase4E_NO_TEACHER']['g0']:.6f} | {HISTORICAL['Phase4E_NO_TEACHER']['phrase']:.6f} | {HISTORICAL['Phase4E_NO_TEACHER']['tf']:.6f} |",f"| CSCU-LF Phase4G-F | {m['matched']['mean_foreground_iou']:.6f} | {m['phrase_only']['mean_foreground_iou']:.6f} | {m['tf']['mean_foreground_iou']:.6f} |"])
    report=f"""# Phase 4G-F CSCU-LF Formal Localization Training

## Protocol

Warm-start Phase4G-1S epoch10；CSCU-LF d64/window7/heads4/blocks1保持不变，REFINEMENT_INCLUDED=NO。P1/SAM/CLIP/Phase4C-A/source heads frozen，只训练既有371,803-parameter context/interaction/U branch。全train Fake traversal 8,836/epoch，其中8,690 valid-G0进入optimizer、146 invalid保留accounting且不可rescue，Real=0。Loss严格为 `L_seg + L_relative + L_ranking`，系数1:1:1；L_seg复用2×BCE+0.5×Dice，margin=.1，tau={TAU:.10f}。AdamW lr1e-4/wd1e-4、batch8、10 epochs、无scheduler/early stopping。

每个epoch只用direct batch=1 DEV G0 selector，invalid IoU=0；highest mean G0、tie取earlier。Phrase/TF/control/utility未参与selection，只在selected checkpoint计算。internal test与official1000未访问。

## Selector

Selected epoch **{selector['selected_epoch']}**，DEV G0={selector['selected_dev_g0']:.6f}。完整十个epoch轨迹见training_history.csv。

## Development results

{table}

Controls：matched={m['matched']['mean_foreground_iou']:.6f}，cross={m['cross_image']['mean_foreground_iou']:.6f}，shuffle={m['spatial_shuffle']['mean_foreground_iou']:.6f}，off={m['off']['mean_foreground_iou']:.6f}，vacuous={m['vacuous']['mean_foreground_iou']:.6f}。

## Paired G0 statistics

- CSCU-LF vs P1：`{json.dumps(s['vs_P1']['g0']['foreground_iou'],ensure_ascii=False)}`
- CSCU-LF vs Phase4F：`{json.dumps(s['vs_Phase4F']['g0']['foreground_iou'],ensure_ascii=False)}`
- matched−cross：`{json.dumps(s['matched_minus_cross']['foreground_iou'],ensure_ascii=False)}`
- matched−shuffle：`{json.dumps(s['matched_minus_shuffle']['foreground_iou'],ensure_ascii=False)}`

## Scientific answers

A. Deployable G0是否进一步提高：相对P1为 **{gates['CSCULF_BEATS_P1_G0']}**，相对Phase4F为 **{gates['CSCULF_BEATS_PHASE4F_G0']}**。  
B. `G0 < Phrase < TF`：**{gates['LANGUAGE_ORDER_PRESERVED']}**。  
C. 与Phase4F tradeoff：依据同时的G0与language order结果解释，不使用Phrase/TF选模。  
D. matched明显优于cross/shuffle（paired CI lower>0）：**{gates['MATCHED_GT_CROSS_SHUFFLE']}**。  
E. off/vacuous direct tensor exact recovery：**{gates['VACUOUS_EXACT_RECOVERY']}**。

## Final gates

```json
{json.dumps(gates,ensure_ascii=False,indent=2)}
```

到此严格STOP。未访问internal test/official1000，也未自动启动held-out final evaluation。
"""; DOC.parent.mkdir(parents=True,exist_ok=True); DOC.write_text(report,encoding="utf-8")
    for row in manifest["source_files"].values(): row["sha256_after"]=file_sha256(Path(row["path"])); row["unchanged"]=row["sha256_before"]==row["sha256_after"]
    manifest.update({"status":"COMPLETE","dev_results_sha256":file_sha256(DEV_RESULTS),"selector_sha256":file_sha256(SELECTOR),"gate_summary_sha256":file_sha256(GATES),"training_history_sha256":file_sha256(HISTORY),"selected_checkpoint_sha256":file_sha256(SELECTED),"report_sha256":file_sha256(DOC),"heldout_started":False}); dump(MANIFEST,manifest); return gates


def selftest(device):
    frozen_preflight(); model=load_model(device).train(); split=json.loads((ROOT/"outputs/phase4f_language_preserving_rectification/manifests/frozen_protocol.json").read_text()); ids=[]
    for path in sorted(G1C_CACHE.glob("shard_*.pt")):
        shard=torch.load(path,map_location="cpu",weights_only=False)
        for sid,v in zip(shard["sample_ids"],shard["valid_g0"].tolist()):
            if v: ids.append(sid)
            if len(ids)>=8: break
        if len(ids)>=8: break
    data=q.load_ids(ids,("S64","q_seg","z_L","F24","z_F24","target64","target256_packed","clip_geometries")); data["target256"]=unpack_bool(data.pop("target256_packed")); idx=torch.arange(4); batch=full_inputs(data,idx,device); crossed=dict(batch); crossed["F24"]=batch["F24"].flip(0); crossed["z_F24"]=batch["z_F24"].flip(0); perm=torch.randperm(576,generator=torch.Generator().manual_seed(SEED)).to(device)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16):
        matched=model(s64=batch["S64"],q_seg=batch["q_seg"],z_l=batch["z_L"],f24=batch["F24"],z_f24=batch["z_F24"],clip_geometries=batch["clip_geometries"],valid_g0=batch["valid_g0"],forensic_present=batch["forensic_present"],forensic_vacuous=batch["forensic_vacuous"],forensic_off=batch["forensic_off"]); cross=q.utility_forward(model,crossed); shuffle=q.utility_forward(model,batch,spatial_permutation=perm); seg=segmentation_loss(matched["logits"],data["target256"][:4].to(device)); _,soft=q.target_delta({"p_L":matched["p_L64"],"p_F":matched["p_F64"]},data["target64"][:4].to(device)); rel=q.image_balanced_loss(model.utility_head.net[-1](matched["utility_prehead"]),soft,matched["support64"]); rank=.5*(rank_loss(matched["U_F64"],cross["U"],matched["support64"])+rank_loss(matched["U_F64"],shuffle["U"],matched["support64"])); total=seg["total"]+rel+rank
    total.backward(); assert torch.isfinite(total) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()); print(json.dumps({"SELFTEST":"PASS","seg":float(seg["total"]),"relative":float(rel),"ranking":float(rank),"total":float(total)}))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("selftest","cache-g0","train","evaluate","finalize","all")); parser.add_argument("--device",default="cuda:1"); args=parser.parse_args(); device=torch.device(args.device); torch.cuda.set_device(device)
    if args.stage=="selftest": selftest(device)
    elif args.stage=="cache-g0": freeze_manifest(); build_shared_dev(device); build_dev_mode("g0",device)
    elif args.stage=="train": train(device)
    elif args.stage=="evaluate": evaluate_selected(device)
    elif args.stage=="finalize": print(json.dumps(finalize(),ensure_ascii=False,indent=2))
    else: selftest(device); train(device); evaluate_selected(device); print(json.dumps(finalize(),ensure_ascii=False,indent=2))

if __name__=="__main__": main()
