from common import *
import ast,collections,torch
def main():
 sys.addaudithook(guard);checks={};checks['frozen_inputs_unchanged']=verify_inputs();ds={s:dataset(s)for s in SPLITS};roots=read(S/'configs/roots.json');selection=read(S/'configs/selected_models.json');compute=read(S/'results/compute_complete.json')
 assert compute['fits']==4 and compute['updates']==6000 and compute['candidates_saved']==8
 assert not(S/'results/FAILED.json').exists()
 for a in SPLITS:
  assert set(ds[a]['root_id'])==set(roots[a]);assert all(str(r).startswith(('A_','B_'))for r in ds[a]['root_id'])
  assert len(ds[a]['root_id'])==len(roots[a])*3
  for b in SPLITS:
   if a!=b:assert not set(roots[a])&set(roots[b])
  assert ds[a]['valid'].all();assert np.array_equal(rel_from_cat(ds[a]['cat']),ds[a]['y']);assert np.isfinite(ds[a]['physical']).all()
  assert np.array_equal(ds[a]['initial_state'][...,:9],ds[a]['y']);assert(ds[a]['initial_state'][...,9:]==1).all()
  assert ds[a]['action'].shape[-1]==7 and (ds[a]['action'][...,:6].sum(-1)==1).all()
 checks['root_split_disjoint_and_valid_labels']=True
 norm=np.load(S/'configs/physical_normalization.npz');phys=ds['train']['physical'][:,:4].reshape(-1,96)
 assert np.array_equal(norm['mean'],phys.mean(0,dtype=np.float64).astype(np.float32));assert np.array_equal(norm['std'],np.maximum(phys.std(0,dtype=np.float64),.001).astype(np.float32));checks['normalization_train_before_only']=True
 initials={};params=[]
 for selected in selection['selection']:
  receipt=read(S/f'results/train_{selected["arm"]}_seed{selected["seed"]}.json');chosen=min(receipt['candidates'],key=lambda r:(r['validation_objective'],r['update']));assert chosen==receipt['selected'];assert chosen['sha256']==selected['sha256'];assert receipt['updates']==1500
  for c in receipt['candidates']:assert sha(c['path'])==c['sha256']
  ck=torch.load(selected['path'],weights_only=True,map_location='cpu');assert ck['updates']==selected['update'];initials[selected['arm'],selected['seed']]=ck['initial_state'];params.append(receipt['parameters'])
 for seed in [0,1]:
  aa=initials['relations',seed];bb=initials['physics_enriched',seed];assert aa.keys()==bb.keys();assert all(torch.equal(aa[k],bb[k])for k in aa)
  schedule=np.load(S/f'configs/schedule_seed{seed}.npy');assert schedule.shape==(1500,32);assert schedule.min()>=0 and schedule.max()<120
 assert len(set(params))==1;checks['four_paired_fits_and_dev_selection']=True;checks['shared_parameter_count']=params[0]
 selected_at=datetime.datetime.fromisoformat(selection['utc']).timestamp()
 prediction_files=list((S/'data').glob('predictions_*.npz'));assert len(prediction_files)==18
 for f in prediction_files:
  split=next(s for s in SPLITS if f.name.endswith('_'+s+'.npz'));d=np.load(f);assert np.array_equal(d['episode_id'],ds[split]['episode_id']);assert np.array_equal(d['root_id'],ds[split]['root_id'])
  for mode in ['one_step','free']:
   a=d[mode];assert a.shape==(len(ds[split]['root_id']),4,15);assert np.isfinite(a).all();assert(a>=0).all()and(a<=1).all();assert np.allclose(a[...,6:9].sum(-1),1);assert np.allclose(a[...,9:12].sum(-1),1)
  if split=='test':assert f.stat().st_mtime>selected_at
 checks['selected_before_test_forecasts_and_probabilities_valid']=True
 # Verify reused source files against their earlier sealed manifests, then exact alignment.
 old=read(B/'configs/delivery_manifest.json')['sha256'];supp=read(H/'manifest.json')['sha256'];n=0
 for split in SPLITS:
  d=ds[split]
  for seed in [0,1]:
   p=visual_path(seed,split);expected=supp[str(p.relative_to(H))]if split=='test'else old[str(p)];assert sha(p)==expected;raw=np.load(p);v=np.load(S/f'data/visual_seed{seed}_{split}.npz');index={(str(e),str(m),int(t)):i for i,(e,m,t)in enumerate(zip(raw['episode_id'],raw['mode'],raw['step']))}
   for e,eid in enumerate(d['episode_id']):
    for t in range(5):
     k=index[str(eid),'current',t];assert np.array_equal(v['current'][e,t],rel_from_prob(raw['prob'][k]));assert np.array_equal(raw['cat'][k,:4],d['cat'][e,t])
    for t in range(1,5):
     k=index[str(eid),'one_step',t];assert np.array_equal(v['one_step'][e,t-1],rel_from_prob(raw['prob'][k]));assert raw['controls'][k]==d['controls'][e,t-1];key=(str(eid),'free'+str(t),t)
     if key in index:assert np.array_equal(v['free'][e,t-1],rel_from_prob(raw['prob'][index[key]]))
     else:assert np.isnan(v['free'][e,t-1]).all()
   n+=1
 checks['six_cached_visual_sources_match_prior_seals_and_labels']=n
 # Recount reported scores from stored forecast values with independent NumPy operations.
 records=read(S/'data/forecast_records.json');assert len(records)==14100;groups=collections.defaultdict(list)
 for r in records:
  p=np.asarray(r['prob']);y=np.asarray(r['truth']);z=np.r_[p[:6]>=.5,np.eye(3)[p[6:9].argmax()]];c=[bool(np.array_equal(z[:3],y[:3])),bool(np.array_equal(z[3:6],y[3:6])),bool(np.array_equal(z[6:],y[6:]))];c.append(all(c));assert c==r['correct']
  groups[r['split'],str(r['seed']),r['model'],r['mode'],str(r['horizon'])].append(r)
 metrics=list(csv.DictReader((S/'results/relation_metrics.csv').open()));count=0
 for m in metrics:
  j=FIELDS.index(m['field']);rr=groups[m['split'],m['seed'],m['model'],m['mode'],m['horizon']]
  if m['stratum']!='all':
   k=3 if m['stratum'].startswith('initial_all_')else j;v=m['stratum'].endswith('correct');rr=[r for r in rr if r['visual_initial_correct'][k]==v]
  if m['actual_status']!='all':rr=[r for r in rr if r['actual_status']==m['actual_status']]
  if m['subset']=='changed':rr=[r for r in rr if r['changed'][j]]
  root=collections.defaultdict(list)
  for r in rr:root[r['root_id']].append(r['correct'][j])
  assert int(m['n'])==len(rr)and int(m['correct'])==sum(r['correct'][j]for r in rr)and int(m['roots'])==len(root);assert np.isclose(float(m['root_accuracy']),np.mean([np.mean(v)for v in root.values()]));count+=1
 checks['recounted_relation_metric_rows']=count
 for c in read(S/'data/chains.json'):
  for model,v in c['models'].items():
   a=correct(v['free_prob'],np.array(c['truth'])[1:]);assert np.array_equal(a,v['correct_by_field']);assert np.array_equal(np.cumprod(a[:,-1]).astype(bool),v['prefix_all_correct']);first=int(np.flatnonzero(~a[:,-1])[0]+1)if not a[:,-1].all()else None;assert first==v['first_wrong_step']
 checks['first_errors_and_prefixes_recomputed']=True
 # No model input path takes future labels, actual status, or actual duration.
 tree=ast.parse((S/'source/train.py').read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Model');roll=next(n for n in cl.body if isinstance(n,ast.FunctionDef)and n.name=='rollout');assert [a.arg for a in roll.args.args]==['self','initial','actions'];outputs=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='outputs');code=ast.unparse(outputs);assert 'x[:, 0]'in code and 'x[:, :4]'in code;checks['input_path_static_audit']='rollout(initial, actions); one-step actual before; free initial only. Labels enter loss/scoring only.'
 # Preserve semantic mismatch instead of silently equating relation truth with executor success.
 p=next(r for r in read(S/'data/provenance.json')if r['episode_id']=='A_core1200_22_b2'and r['t']==3);lab=read(p['labels_path']);assert p['status']=='executed_failure'and p['controls']==311;assert lab['after']['categorical'][2]==3;assert lab['after']['continuous']['objects']['pink']['drawer_support_paths']==[['block_pink','block_blue','drawer']]
 semantic={'utc':now(),'episode':'A_core1200_22_b2','step':3,'controls':311,'execution_success':False,'pink_in_drawer_ownership_label':True,'support_path':['block_pink','block_blue','drawer'],'executor_in_drawer_definition':'Direct load-bearing contact with drawer_link, force > .01','ownership_definition':'Upward load-bearing support path, including through another object','source_sha256':{str(p):sha(p)for p in [P/'src/wm_symbolic/physical_oracle.py',P/'src/wm_symbolic/core_ownership.py']},'labels_path':p['labels_path'],'reason':p['reason'],'not_counterfactual_causal_proof':True};save(S/'results/execution_label_semantics.json',semantic);checks['actual_failure_and_ownership_semantics_preserved']=True
 report=(S/'REPORT.md').read_text();assert '支撑链'in report and '120/120转移'in report and '未被标签确认入屉'not in report;assert len(list((S/'figures').glob('*.png')))==9
 checks['report_case_semantics_and_nine_figures']=True
 save(S/'results/final_verification.json',{'utc':now(),'passed':True,'checks':checks,'new_model_forwards':0,'new_fits':0,'no_new_data_or_C_D_or_upload':True,'limitations':['Historical test and already-seen relation-action combinations/sequences','Only 10 independent evaluation roots; wrong-initial stratum 1 root','Not a new planner or HULC validation; full guide incomplete','Physical state enriched but not proven Markov']})
 manifest={str(p.relative_to(S)):sha(p)for p in sorted(S.rglob('*'))if p.is_file()and p.name!='manifest.json'and 'cache'not in p.relative_to(S).parts and '__pycache__'not in p.parts};save(S/'manifest.json',{'utc':now(),'status':'COMPLETED_BOUNDED_DIAGNOSTIC','files':len(manifest),'bytes':sum((S/p).stat().st_size for p in manifest),'sha256':manifest});print('VERIFIED_SEALED',len(manifest),sha(S/'manifest.json'),flush=True)
if __name__=='__main__':main()
