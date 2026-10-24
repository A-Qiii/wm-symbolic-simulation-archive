from config import *
import torch
from train import features

def main():
 verify();p=read(E/'results/predictions.json');metrics=read(E/'results/metrics.json');joint=read(E/'results/joint_metrics.json');training=read(E/'results/training.json');assert training['fit_count']==4 and training['validation_reads_for_training']==training['D_reads_for_training']==training['world_model_or_encoder_updates']==training['original_query_updates']==0
 roots=read(E/'configs/root_split.json');assert len(set(sum(roots.values(),[])))==18 and all(len(v)==6 for v in roots.values())
 for normkind in ['object','context']:
  data=np.load(E/'data/train/states.npz');x=features(data['z'],normkind);norm=np.load(E/'models'/f'normalization_{normkind}.npz');assert np.array_equal(x.mean(0),norm['mean'])and np.array_equal(np.maximum(x.std(0),.01),norm['std'])
 for name,h in read(E/'configs/probe_models_frozen.json')['sha256'].items():assert sha(name)==h,name
 for r in metrics:
  rr=[x for x in p if x['split']==r['split']and x['model']==r['model']]
  g,v=r['group'],r['value']
  if g=='condition':rr=[x for x in rr if (v=='held'and x['holding_count']>0)or(v=='empty'and x['holding_count']==0)or(v=='dual'and x['dual_relation'])or(v=='held_without_inside'and x['holding_count']>0 and not x['dual_relation'])]
  elif g=='object_status':rr=[x for x in rr if x['object_is_held']==(v=='held_object')]
  elif g=='object_color':rr=[x for x in rr if x['object_color']==v]
  elif g=='held_color':rr=[x for x in rr if x['held_color']==v]
  elif g=='dual_object_color':rr=[x for x in rr if x['dual_relation']and x['object_color']==v]
  elif g=='dual_held_color':rr=[x for x in rr if x['dual_relation']and x['held_color']==v]
  elif g=='root':rr=[x for x in rr if x['root_id']==v]
  assert len(rr)==r['N'];assert sum(x['truth']and x['prediction']for x in rr)==r['TP']and sum(x['truth']and not x['prediction']for x in rr)==r['FN']and sum(not x['truth']and x['prediction']for x in rr)==r['FP']and sum(not x['truth']and not x['prediction']for x in rr)==r['TN']
 for r in joint:
  rr=[x for x in p if x['split']==r['split']and x['model']==r['model']];groups={}
  for x in rr:groups.setdefault(x['state_index'],[]).append(x)
  ss=[v for v in groups.values()if r['condition']=='all'or(r['condition']=='dual'and v[0]['dual_relation'])or(r['condition']=='held'and v[0]['holding_count']>0)or(r['condition']=='empty'and v[0]['holding_count']==0)];assert len(ss)==r['state_N']and sum(all(x['truth']==x['prediction']for x in v)for v in ss)==r['three_In_correct']
 for mf,h in read(E/'configs/frozen.json')['prior_manifests'].items():
  assert sha(mf)==h
  for fn,digest in read(mf)['sha256'].items():assert sha(fn)==digest,fn
 out={'passed':True,'utc':now(),'probe_fits':4,'updates_per_fit':500,'roots_disjoint':True,'normalization_train_only_verified':True,'metric_rows_recounted':len(metrics),'joint_rows_recounted':len(joint),'prior_seals_unchanged':True,'D_not_training_or_selection':True,'new_queries_not_adopted':True}
 seal=E/'configs/delivery_manifest.json'
 if seal.exists():
  for fn,h in read(seal)['sha256'].items():assert sha(fn)==h,fn
  out['sealed_files_verified']=len(read(seal)['sha256'])
 print('PROBE_VERIFICATION',json.dumps(out),flush=True)
if __name__=='__main__':main()
