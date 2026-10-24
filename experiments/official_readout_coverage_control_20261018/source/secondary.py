from coverage_common import *
from evaluate import metrics
import torch,time,collections,shutil
def main():
 assert not(E/'results/secondary_metrics.json').exists();gate=read(E/'results/secondary_gate.json');eligible=[x['head']for x in gate['head_decisions']if x['eligible_for_saved_prediction_diagnostic']];assert eligible==['In'],'This bounded run authorized by its frozen actual gate only'
 pairs=read(D/'data/combo/public/pairs.json');labels=read(D/'data/combo/_private/labels.json');refs={str(D/'data/combo/public/pairs.json'):sha(D/'data/combo/public/pairs.json'),str(D/'data/combo/_private/labels.json'):sha(D/'data/combo/_private/labels.json')};norm=np.load(E/'configs/normalization_In.npz',allow_pickle=False)
 for seed in [0,1]:
  for wm in['mixed','latent']:
   p=D/'results'/f'latents_{wm}_seed{seed}.npz';refs[str(p)]=sha(p)
 save(E/'configs/secondary_frozen.json',{'utc':now(),'eligible_heads':eligible,'In_only':True,'Holding_prediction_reads':False,'no_T_forward':True,'D_previously_seen_diagnostic_only':True,'sha256':refs,'head_seal_sha256':sha(E/'configs/final_heads_frozen.json')});shutil.copy2(__file__,E/'source/secondary.py');torch.set_num_threads(4);start=time.perf_counter();out=[];pred=[];strata=[]
 y=np.array([labels[str(p['after'])]['y'][3:6]for p in pairs],np.float32);v=np.array([labels[str(p['after'])]['v'][3:6]for p in pairs],np.float32);held=np.array([labels[str(p['after'])]['y'][:3]for p in pairs],bool);roots=np.array([p['root_id']for p in pairs])
 with torch.inference_mode():
  for seed in[0,1]:
   heads={}
   for arm in ARMS:
    q=make().eval().requires_grad_(False);q.load_state_dict(torch.load(E/'models'/f'In_{arm}_seed{seed}_update0600.pt',weights_only=True,map_location='cpu')['state']);heads[arm]=q
   for wm in['mixed','latent']:
    latent=np.load(D/'results'/f'latents_{wm}_seed{seed}.npz',allow_pickle=False);assert latent['pair_ids'].tolist()==[p['id']for p in pairs]
    for method in['one_step','recursive']:
     z=latent[method];x=torch.tensor(((z[:,:3]-norm['mean'])/norm['std']).astype(np.float32))
     for arm,q in heads.items():
      prob=q(x).squeeze(-1).sigmoid().numpy();m=metrics(prob,y,v,roots);out.append({'head':'In','seed':seed,'world_model':wm,'method':method,'arm':arm,**m,'self_held_object_In_FP':int(((prob>=.5)&held&v.astype(bool)).sum()),'self_held_object_cells':int((held&v.astype(bool)).sum())})
      for branch in sorted({p['branch']for p in pairs}):
       for slot in sorted({p['slot']for p in pairs if p['branch']==branch}):
        ix=np.array([i for i,p in enumerate(pairs)if p['branch']==branch and p['slot']==slot]);strata.append({'head':'In','seed':seed,'world_model':wm,'method':method,'arm':arm,'branch':branch,'slot':slot,**metrics(prob[ix],y[ix],v[ix],roots[ix])})
      for i,p in enumerate(pairs):pred.append({'head':'In','seed':seed,'world_model':wm,'method':method,'arm':arm,'pair_id':p['id'],'root_id':p['root_id'],'after':p['after'],'probability':prob[i]})
 save(E/'results/secondary_metrics.json',out);save(E/'results/secondary_stage_metrics.json',strata);save(E/'results/secondary_predictions.json',pred);save(E/'results/secondary_audit.json',{'utc':now(),'heads_read_on_saved_predicted_z':['In'],'Holding_stopped_by_actual_gate':True,'T_forward_calls':0,'physics_calls':0,'extra_fits':0,'D_previously_seen':True,'full9_secondary_not_run':True,'seconds':time.perf_counter()-start,'meaning':'Readout response to unchanged previously saved predictions; not a new test or verification of transition dynamics.'});print('SECONDARY_IN_ONLY_COMPLETE',len(out),'summaries',flush=True)
if __name__=='__main__':main()
