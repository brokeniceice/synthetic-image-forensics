"""Finite, sealed C1/R1 ablation contract; no mutation of historical sources."""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'training'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
BASE=ROOT/'outputs/c1_native_paper_audit/ablation_20261018'
OLD=ROOT/'outputs/phase6e3_c1_native_staged'
COMBO=ROOT/'outputs/c1_native_paper_audit/g1_fallback_eval'
FS=ROOT/'outputs/c1_native_paper_audit/fakeshield_g1'
PYTHON=sys.executable
DATASETS={'synthscars':1000,'loki':229,'xaigd':2419,'pal4vst':1441}
CLASSIFICATION={'internal':2208,'genimage':100000,'loki':2217,'aigi_holmes':99999}
ARMS={'Full':(1.,1.),'SegOnly':(0.,0.),'NoRelative':(0.,1.),'NoRanking':(1.,0.)}
CONFIGS=[('Full',3407),('SegOnly',3407),('NoRelative',3407),('NoRanking',3407)]
C1_SHA='85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff'
R1_SHA='1446074c8233ccce560f57cde2b609e8600337453ad620fa60cf285a97b90a5b'
def require(ok,why):
    if not ok:raise RuntimeError(why)
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def immutable(p,v):
    p=Path(p)
    if p.exists():require(read(p)==v,'Immutable receipt conflict: '+str(p))
    else:write(p,v)
def name(arm,seed):return f'{arm}_s{seed}'
def stage_dir(arm,seed,stage):
    if seed==3407 and (stage=='rectifier' or arm=='Full'):return OLD/stage
    if stage=='rectifier':return BASE/'training'/f'shared_s{seed}'/stage
    return BASE/'training'/name(arm,seed)/stage

def selected(arm,seed,stage='joint'):
    folder=stage_dir(arm,seed,stage);s=read(folder/'selector.json');p=Path(s['selected_checkpoint'])
    require(s['status']=='COMPLETE' and s['c1_sha256']==C1_SHA,'Invalid selector')
    require(sha(p)==s['selected_checkpoint_sha256'],'Selected checkpoint drift')
    require(not s['official1000_used_for_selection'] and not s['external_ood_used_for_selection'],'Selection firewall')
    prov=read(folder/'provenance.json');require(prov['status']=='COMPLETE','Incomplete training provenance')
    if not (seed==3407 and (stage=='rectifier' or arm=='Full')):
        require(prov['seed']==seed and prov['loss_coefficients']==list(ARMS['Full' if stage=='rectifier' else arm]),'Wrong seed/loss training')
        require(prov['updates']==11050,'Training exposure mismatch')
    return p,s

def verify_contract(large=False):
    c=read(BASE/'protocol.json')
    for group in ['sources','files']+(['assets','training_cache'] if large else []):
        for p,h in c[group].items():require(sha(p)==h,'Contract drift: '+p)
    return c


