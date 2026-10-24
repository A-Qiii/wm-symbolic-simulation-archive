"""Read saved actual and world-model z only; no encoder/transition/optimizer instantiated."""
from config import *
import torch,sys,fcntl
sys.path.insert(0,str(REP/'source'))
from model import Query,probs

def main():
 verify();assert not(E/'results/actual_predictions.json').exists()
 for p,h in read(E/'configs/final_heads_frozen.json')['sha256'].items():assert sha(p)==h,p
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);norm=np.load(E/'configs/query_normalization.npz');cfg=read(E/'configs/protocol.json');actual=[];trajectory=[];pairs=read(COMBO/'data/combo/public/pairs.json');pairbyid={r['id']:r for r in pairs};root_split={r:s for s,rr in cfg['root_split'].items()for r in rr};original=read(COMBO/'results/predictions.json');oldlookup={(p['condition'],p['id']):p for p in original};maxdiff=0.
 def infer(q,z):
  with torch.inference_mode():return np.concatenate([probs(q(torch.tensor(z[k:k+128],device='cuda'))).cpu().numpy()for k in range(0,len(z),128)])
 for seed in [0,1]:
  for arm in ['original','control','coverage']:
   path=cfg['initial_heads'][str(seed)]['path']if arm=='original'else str(E/'models'/f'{arm}_seed{seed}/update0600.pt');q=Query().cuda().eval().requires_grad_(False);q.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state'])
   for f in sorted((E/'data/evaluation').glob('*.npz')):
    d=np.load(f);pp=infer(q,d['z']);meta=read(f.with_name(f.stem+'_meta.json'))
    for i,p in enumerate(pp):actual.append({'cohort':f.stem,'state_index':i,'state_id':str(d['state_ids'][i]),'root_id':str(d['root_ids'][i]),'seed':seed,'query_arm':arm,'probability':p,'dual_relation':meta[i]['dual_relation'],'environment':meta[i]['environment']})
   for wm in ['mixed','latent']:
    name=f'{wm}_seed{seed}';d=np.load(COMBO/'results'/f'latents_{name}.npz');pps={key:infer(q,(d[key]-norm['mean'])/norm['std'])for key in PATHS};assert len(d['pair_ids'])==216
    for i,pid in enumerate(d['pair_ids']):
     p=pairbyid[str(pid)];r={k:p[k]for k in ['id','root_id','environment','target','other_target','branch','slot','skill','before','after']}|{'seed':seed,'query_arm':arm,'world_model':wm,'split':root_split[p['root_id']]}
     for key in PATHS:
      r[key+'_probability']=pps[key][i]
      if arm=='original':maxdiff=max(maxdiff,float(np.max(np.abs(pps[key][i]-oldlookup[(name,str(pid))][key+'_probability']))))
     trajectory.append(r)
   print('QUERY_PREDICTED',seed,arm,flush=True)
 assert maxdiff<1e-5,maxdiff
 save(E/'results/actual_predictions.json',actual);save(E/'results/trajectory_predictions.json',trajectory);save(E/'results/prediction_audit.json',{'utc':now(),'original_predictions_reproduction_max_delta':maxdiff,'actual_rows':len(actual),'trajectory_rows':len(trajectory),'encoder_transition_instantiated':False,'optimizer_instantiated':False,'new_physical_calls':0,'D_evaluation_only_after_final_head_freeze':True,'input':'only normalizedsavedz, no semantic truth or plan supplied toquery'})
if __name__=='__main__':main()
