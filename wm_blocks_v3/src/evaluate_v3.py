"""One frozen, same-new-layout v1/v3 comparison; no fitting or test tuning."""
from paths import GPU_LOCK
from paths import ROOT as B, RUNTIME as P
import json,hashlib,fcntl
from pathlib import Path
import numpy as np,torch
from models import RGBState,Dynamics
from models_v2 import StateActorDynamics
from models_v3 import DiagonalStateDynamics
from evaluate import action,decode,gt,same
def read(p):return json.load(open(p))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def models():
 new=read(B/'configs/v3_models_frozen.json');assert new['development_gate_passed'];mid=read(B/'configs/v2_models_frozen.json')['models'];records=[]
 for r in new['models']:
  m=next(x for x in mid if x['seed']==r['seed'])
  records.extend([{'version':'v1',**r['baseline'],'normalization':r['normalization']},{'version':'v2',**m},{'version':'v3',**r}])
 return records
def main():
 cache=B/'cache/v3/confirmation';assert(cache/'complete.json').exists();meta=read(cache/'state_rows.json');feat=np.load(cache/'all_features.npz')['features'];idx={(r['episode'],r['boundary']):i for i,r in enumerate(meta)};manifest=read(B/'configs/v3_confirmation_manifest.json');out=B/'reports/v3/confirmation';out.mkdir(parents=True,exist_ok=True)
 if(out/'complete.json').exists():return
 results=[];summaries=[]
 for rec in models():
  for key,shakey in [('checkpoint','sha256'),('state_checkpoint','state_sha256')]:assert hashlib.sha256(Path(rec[key]).read_bytes()).hexdigest()==rec[shakey]
  est=RGBState(1567).cuda();est.load_state_dict(torch.load(rec['state_checkpoint'],map_location='cuda',weights_only=False)['model']);net=(Dynamics('motion_matrix')if rec['version']=='v1'else(StateActorDynamics()if rec['version']=='v2'else DiagonalStateDynamics())).cuda();net.load_state_dict(torch.load(rec['checkpoint'],map_location='cuda',weights_only=False)['model']);est.eval().requires_grad_(False);net.eval().requires_grad_(False);norm=np.load(rec['normalization']);counts={};cases=[]
  for spec in manifest['episodes']:
   e=read(B/'data/v3/confirmation'/spec['id']/'episode.json');row={'version':rec['version'],'seed':rec['seed'],'episode':spec['id'],'group':spec['group'],'recipe':spec['recipe'],'physical_valid':e['physical_valid'],'planned_actions':len(spec['actions'])}
   if not e['physical_valid']:cases.append(row);continue
   L=len(spec['actions']);x=torch.tensor((feat[[idx[(spec['id'],i)]for i in range(L+1)]]-norm['mean'])/norm['std'],device='cuda')
   with torch.no_grad():
    post=Dynamics.pack(*est(x));current=[decode(post[i:i+1],est)for i in range(L+1)];one=[];free=[];s=post[:1]
    for t,a in enumerate(spec['actions']):
     one.append(decode(net(post[t:t+1],*action(a),est.sd),est));s=net(s,*action(a),est.sd);free.append(decode(s,est))
    feedback_s=post[1:2]
    for a in spec['actions'][1:]:feedback_s=net(feedback_s,*action(a),est.sd)
    feedback=decode(feedback_s,est)
   truth=[gt(s)for s in e['states']];row.update({'truth':truth,'current':current,'one':one,'free':free,'feedback_after_first_action_endpoint':feedback,'correct':{'current':[same(p,t)for p,t in zip(current,truth)],'one':[same(p,t)for p,t in zip(one,truth[1:])],'free_endpoint':same(free[-1],truth[-1]),'free_interval':all(same(p,t)for p,t in zip(free,truth[1:])),'feedback_endpoint':same(feedback,truth[-1])}});cases.append(row)
  for recipe in sorted({r['recipe']for r in cases}):
   xs=[r for r in cases if r['recipe']==recipe];valid=[r for r in xs if r['physical_valid']];s={'version':rec['version'],'seed':rec['seed'],'recipe':recipe,'planned':len(xs),'valid':len(valid),'current':[sum(sum(r['correct']['current'])for r in valid),sum(len(r['truth'])for r in valid)],'one':[sum(sum(r['correct']['one'])for r in valid),sum(r['planned_actions']for r in valid)]}
   for k in ['free_endpoint','free_interval','feedback_endpoint']:s[k]=[sum(r['correct'][k]for r in valid),len(xs)]
   summaries.append(s)
  results.extend(cases);save(out/f'{rec["version"]}_seed{rec["seed"]}.json',{'cases':cases});print('V3_CONFIRM_MODEL',rec['version'],rec['seed'],flush=True);del est,net;torch.cuda.empty_cache()
 save(out/'all_predictions.json',results);save(out/'summary.json',summaries);save(out/'complete.json',{'models':9,'same_new_test_layouts':True,'test_not_used_for_tuning':True,'models_manifest_sha256':hashlib.sha256((B/'configs/v3_models_frozen.json').read_bytes()).hexdigest()})
if __name__=='__main__':
 lock=GPU_LOCK.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 try:main()
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
