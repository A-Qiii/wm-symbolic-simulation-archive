from config import *
from model import *
def main():
 verify();assert not(E/'results/actual_predictions.json').exists();torch.set_num_threads(4)
 for p,h in read(E/'configs/final_heads_frozen.json')['sha256'].items():assert sha(p)==h
 oldactual=read(WHOLE/'results/actual_predictions.json');oldtraj=read(WHOLE/'results/trajectory_predictions.json');norm=np.load(E/'configs/object_normalization.npz');qn=np.load(E/'configs/query_normalization.npz');actual=[];traj=[];rename={'original':'original','control':'whole_control','coverage':'whole_coverage'}
 for r in oldactual:actual.append(r|{'query_arm':rename[r['query_arm']]})
 for r in oldtraj:traj.append(r|{'query_arm':rename[r['query_arm']]})
 for seed in [0,1]:
  for arm in ['control','coverage']:
   q=make().eval();q.load_state_dict(torch.load(E/'models'/f'{arm}_seed{seed}_update0600.pt',weights_only=True,map_location='cpu')['state'])
   def infer(raw):
    with torch.inference_mode():return q(torch.tensor(((raw[:,:3]-norm['mean'])/norm['std']).astype(np.float32))).squeeze(-1).sigmoid().numpy()
   for file in sorted((WHOLE/'data/evaluation').glob('*.npz')):
    d=np.load(file);ip=infer(d['z']*qn['std']+qn['mean']);rr=sorted([r for r in oldactual if r['cohort']==file.stem and r['seed']==seed and r['query_arm']=='original'],key=lambda r:r['state_index'])
    for r,pp in zip(rr,ip):pr=list(r['probability']);pr[3:6]=pp.tolist();actual.append(r|{'query_arm':'in_'+arm,'probability':pr})
   for wm in ['mixed','latent']:
    d=np.load(COMBO/'results'/f'latents_{wm}_seed{seed}.npz');ips={p:infer(d[p])for p in PATHS};rr={r['id']:r for r in oldtraj if r['seed']==seed and r['world_model']==wm and r['query_arm']=='original'}
    for i,pid in enumerate(d['pair_ids']):
     r=rr[str(pid)];new=r|{'query_arm':'in_'+arm}
     for p in PATHS:pr=list(r[p+'_probability']);pr[3:6]=ips[p][i].tolist();new[p+'_probability']=pr
     traj.append(new)
   print('OBJECT_IN_PREDICTED',seed,arm,flush=True)
 save(E/'results/actual_predictions.json',actual);save(E/'results/trajectory_predictions.json',traj);save(E/'results/prediction_audit.json',{'utc':now(),'actual_rows':len(actual),'trajectory_rows':len(traj),'Holding_drawer_source':'bitwise copy of original formal Query probabilities on same input','semantic_truth_and_actions_used_for_prediction':False,'world_models_or_encoder_instantiated':False,'no_trueHolding_mask':True,'checkpoint_selection':False})
if __name__=='__main__':main()
