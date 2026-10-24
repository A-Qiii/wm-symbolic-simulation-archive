from pathlib import Path
import json,hashlib,datetime
import numpy as np
import torch
from model import PredicateReader,targets
from metrics import summarize,known,GROUPS,metric
P=Path('/home/robot/research/projects/wm_symbolic');O=P/'experiments/state_predicates_20261014';A=P/'experiments/state_predicates_augmented_20261014';E=P/'experiments/state_predicates_joint_20261014'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dataset():
    out={};om=read(O/'data/metadata.json');am=read(A/'data/metadata_train_dev.json')
    with np.load(O/'data/inputs_targets.npz') as z:
        for sp,name in [('train','old_train'),('tune','old_tune'),('calibration','old_cal')]:
            sel=z['splits']==sp;y,v=targets(torch.tensor(z['cat'][sel]),torch.tensor(z['valid'][sel]));out[name]={k:z[k][sel].copy() for k in ['objects','context','robot']};out[name].update(y=y.numpy(),v=v.numpy(),roots=np.array([r['root_id'] for r in om if r['split']==sp]),meta=[r for r in om if r['split']==sp])
    with np.load(A/'data/supplement_train_dev.npz') as z:
        for sp,name in [('train','new_train'),('dev','new_dev')]:
            sel=z['splits']==sp;out[name]={k:z[k][sel].copy() for k in ['objects','context','robot']};out[name].update(y=z['y'][sel].copy(),v=z['eval_valid'][sel].copy(),original_train_v=z['train_valid'][sel].copy(),roots=np.array([r['root_id'] for r in am if r['split']==sp]),meta=[r for r in am if r['split']==sp])
    return out
def infer(m,d):
    m.eval()
    with torch.no_grad():return m.probabilities(**{k:torch.as_tensor(d[k],device='cuda') for k in ['objects','context','robot']}).cpu().numpy()
def loadmodel(path):
    m=PredicateReader().cuda();ck=torch.load(path,map_location='cuda',weights_only=False);m.load_state_dict(ck['model']);return m
def evaluate_predictions(p,d,tau=(.5,.5,.5)):
    return summarize(p,d['y'],d['v'],tau)
def raw_gate(metrics):
    bad=[];excess=0.
    for domain,groups in metrics.items():
        for g,q in groups.items():
            fp=q['FPR'];rec=q['positive_recall'];reasons=[]
            if fp is not None and fp>.01+1e-10:reasons.append('FPR_above_1pct');excess+=(fp-.01)
            if rec is not None and rec<.95-1e-10:reasons.append('recall_below_95pct');excess+=(.95-rec)
            if reasons:bad.append({'domain':domain,'predicate':g,'reasons':reasons,'FP':q['FP'],'FN':q['FN'],'positive':q['positive'],'negative':q['negative']})
    return {'passed':not bad,'failed_domain_predicates':len(bad),'excess':excess,'failures':bad}
def joint_score(preds,ds):
    vals=[]
    for domain,p in preds.items():
        d=ds[domain];y=d['y'];v=d['v'];bce=-(y*np.log(np.clip(p,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-p,1e-7,1-1e-7)))
        for cols in GROUPS.values():
            rs=[]
            for root in sorted(set(d['roots'])):
                idx=d['roots']==root;vv=v[idx][:,cols];yy=y[idx][:,cols];ll=bce[idx][:,cols];classes=[]
                for cls in [0,1]:
                    mask=vv&(yy==cls)
                    if mask.any():classes.append(float(ll[mask].mean()))
                if classes:rs.append(np.mean(classes))
            vals.append(np.mean(rs))
    return float(np.mean(vals))
def rank_predictions(preds,ds):
    metrics={n:evaluate_predictions(p,ds[n]) for n,p in preds.items()};gate=raw_gate(metrics);score=joint_score(preds,ds)
    return (gate['failed_domain_predicates'],gate['excess'],score),{'gate':gate,'joint_root_class_macro_BCE':score,'metrics':metrics}
def calibrate_joint(preds,ds):
    taus=[];details={}
    for gi,(name,cols) in enumerate(GROUPS.items()):
        grid=[];winner=None
        for t in [.5,.6,.7,.8,.85,.9,.95,.975,.99,.995]:
            ts=[.5,.5,.5];ts[gi]=t;met={n:metric(p[:,cols],ds[n]['y'][:,cols],ds[n]['v'][:,cols],known(p,ts)[:,cols]) for n,p in preds.items()}
            eligible=True
            for q in met.values():
                eligible &= q['coverage']>=.95 and q['known_error'] is not None and q['known_error']<=.02
                if q['negative']:eligible &= q['known_false_positive']/q['negative']<=.01
                if q['positive']:eligible &= q['known_true_positive']/q['positive']>=.95
            cov=min(q['coverage'] for q in met.values());grid.append({'threshold':t,'eligible':bool(eligible),'metrics':met})
            if eligible and (winner is None or cov>winner['min_coverage']):winner={'threshold':t,'min_coverage':cov}
        taus.append(winner['threshold'] if winner else 1.000001);details[name]={'winner':winner,'grid':grid}
    metrics={n:evaluate_predictions(p,ds[n],taus) for n,p in preds.items()};raw=raw_gate(metrics)
    return {'thresholds':taus,'eligible':raw['passed'] and all(v['winner'] is not None for v in details.values()),'raw_gate':raw,'metrics':metrics,'details':details}
