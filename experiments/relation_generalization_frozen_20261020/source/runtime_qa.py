from common import *
def same_arrays(a,b):
 with np.load(a)as x,np.load(b)as y:return set(x.files)==set(y.files)and all(np.array_equal(x[k],y[k])for k in x.files)
def main():
 guard();verify();eps=read(S/'cache/episodes.json');items=read(S/'cache/items.json');by={e['episode_id']:e for e in eps};matched=0;probe_matches=0;probe_probability=[]
 for e in eps:
  d=Path(e['rows'][0]['labels_path']).parent.parent
  if e['template']=='order_known4':
   other=by[e['episode_id'].replace('order_known4','order_novel4')];dd=Path(other['rows'][0]['labels_path']).parent.parent
   assert all(same_arrays(d/f'root_observation_{j}.npz',dd/f'root_observation_{j}.npz')for j in range(4));matched+=1
  for index in e['probes']:
   item=items[index]
   if item['completion_control']!=12:continue
   normal=items[e['boundaries'][item['step']]]
   assert all(same_arrays(a,b)for a,b in zip(item['public_inputs'],normal['public_inputs']));probe_matches+=1
 for seed in [0,1]:
  z=dict(np.load(S/f'cache/front_seed{seed}.npz'))
  assert z['objects'].shape==(len(items),3,512)and z['context'].shape==(len(items),512)and z['availability'].shape==(len(items),3)
  assert all(np.isfinite(v).all()for v in z.values())
  for arm in ARMS:
   p=dict(np.load(S/f'results/predictions_{arm}_seed{seed}.npz'));q=dict(np.load(S/f'results/probes_{arm}_seed{seed}.npz'));ds=[]
   for j in np.flatnonzero(q['completion_control']==12):
    ids=np.flatnonzero((p['episode_id']==q['episode_id'][j])&(p['step']==q['step'][j])&(p['mode']=='current'));assert len(ids)==1;ds.append(float(np.abs(p['prob'][ids[0]]-q['prob'][j]).max()))
   maximum=max(ds,default=0.);assert maximum<1e-4
   probe_probability.append({'arm':arm,'seed':seed,'pairs':len(ds),'max_difference':maximum})
 save(S/'results/runtime_qa.json',{'utc':now(),'passed':True,'matched_known_novel_public_initial_states':matched,'last_probe_equals_actual_boundary_public_windows':probe_matches,'last_probe_current_probability_comparison':probe_probability,'frozen_design_and_weights_verified':True,'all_base_feature_arrays_finite_and_expected_shapes':True,'no_metric_or_threshold_change':True})
 print('RUNTIME_QA_PASSED',matched,probe_matches,flush=True)
if __name__=='__main__':main()
