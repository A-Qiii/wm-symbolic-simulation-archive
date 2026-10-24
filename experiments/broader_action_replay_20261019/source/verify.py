from common import *
import ast,collections
def main():
 sys.addaudithook(guard);checks={};checks['frozen_inputs_unchanged']=verify_inputs();comp=read(S/'results/compute_complete.json');assert comp['new_fits']==0 and comp['collection']==0 and comp['DINO_forwards']==0 and comp['spatial_encoder_forwards']==0 and comp['C_D_samples']==0 and comp['uploads']==0;assert not(S/'results/FAILED.json').exists()
 for path,h in comp['model_weights_unchanged'].items():assert sha(path)==h
 checks['unchanged_model_checkpoint_files']=len(comp['model_weights_unchanged'])
 roots=read(S/'configs/roots.json');oldroots=read(D/'configs/roots.json');assert len(set(sum(roots.values(),[])))==60;assert not set(sum(roots.values(),[]))&set(sum(oldroots.values(),[]));ind=list(csv.DictReader((S/'results/root_independence.csv').open()));assert len(ind)==300
 for r in ind:assert(r['used_in_model_training']=='True')==(r['model']=='coverage_object_head'and r['split']=='train');assert(r['all_models_unseen_root']=='True')==(r['split']!='train')
 checks['per_model_independence_rows']=300
 ds={s:dataset(s)for s in SPLITS};source_rows=read(R/'data/data_manifest.json')['records'];by={(r['episode_id'],r['step']):r for r in source_rows};verified=0
 for split,d in ds.items():
  assert set(d['root_id'])==set(roots[split]);assert len(d['root_id'])==len(roots[split])*3
  for e,eid in enumerate(d['episode_id']):
   for t in range(4):
    r=by[str(eid),t];lab=read(r['labels_path']);assert np.array_equal(d['cat'][e,t],lab['before']['categorical'][:4]);assert np.array_equal(d['cat'][e,t+1],lab['after']['categorical'][:4]);assert d['controls'][e,t]==lab['control_samples_including_start']-1;assert d['action'][e,t,:6].argmax()==SKILLS.index(r['skill']);assert d['action'][e,t,6]==r['executor']['budget_steps']/900;assert d['status'][e,t]==(0 if lab['success']else 1 if d['controls'][e,t]==0 else 2);verified+=1
  assert np.array_equal(d['y'],rel_from_cat(d['cat']));assert np.array_equal(d['initial_state'][...,:9],d['y']);assert np.isfinite(d['physical']).all()
 checks['source_transitions_aligned']=verified
 for seed in [0,1]:
  a=np.load(S/f'data/predictions_joint_seed{seed}_test.npz');b=np.load(H/f'data/predictions_joint_seed{seed}_test.npz');assert set(a.files)==set(b.files)
  for k in b.files:assert np.array_equal(a[k],b[k],equal_nan=True)if b[k].dtype.kind in 'fc'else np.array_equal(a[k],b[k])
  for split in SPLITS:
   native=np.load(S/f'data/predictions_joint_seed{seed}_{split}.npz');assert np.isfinite(native['prob']).all()
   for source in ['rssm','old_object','coverage_object']:
    raw=native if source=='rssm'else np.load(S/f'data/predictions_{source}_seed{seed}_{split}.npz');ix={(str(e),int(t)):i for i,(e,t)in enumerate(zip(raw['episode_id'],raw['step']))if source!='rssm'or raw['mode'][i]=='current'};cur=np.load(S/f'data/current_{source}_seed{seed}_{split}.npz');d=ds[split]
    assert np.array_equal(cur['episode_id'],d['episode_id'])
    for e,eid in enumerate(d['episode_id']):
     for t in range(5):assert np.array_equal(cur['prob'][e,t],rel_from_prob(raw['prob'][ix[str(eid),t]]));assert np.array_equal(raw['cat'][ix[str(eid),t],:4],d['cat'][e,t])
 checks['native_old_test_exact_replay_rows']=1500;checks['all_current_interfaces_aligned']=True
 forecasts=read(S/'data/forecast_records.json');assert len(forecasts)==20160;groups=collections.defaultdict(list)
 for r in forecasts:
  p=np.asarray(r['prob']);y=np.asarray(r['truth']);z=np.r_[p[:6]>=.5,np.eye(3)[p[6:9].argmax()]];v=[np.array_equal(z[:3],y[:3]),np.array_equal(z[3:6],y[3:6]),np.array_equal(z[6:],y[6:])];v.append(all(v));assert v==r['correct'];assert int(((z[:6]>0)&(y[:6]==0)).sum())==r['extra_binary'];assert int(((z[:6]==0)&(y[:6]>0)).sum())==r['missing_binary'];groups[r['split'],str(r['seed']),r['model'],r['mode'],str(r['horizon'])].append(r)
 metrics=list(csv.DictReader((S/'results/relation_metrics.csv').open()))
 for m in metrics:
  rr=groups[m['split'],m['seed'],m['model'],m['mode'],m['horizon']];j=FIELDS.index(m['field']);rr=[r for r in rr if(m['scene']=='all'or r['scene']==m['scene'])and(m['initial_class']=='all'or r['initial_class']==m['initial_class'])and(m['subset']=='all'or r['changed'][j])];byroot=collections.defaultdict(list)
  for r in rr:byroot[r['root_id']].append(r['correct'][j])
  assert int(m['n'])==len(rr)and int(m['roots'])==len(byroot)and float(m['count'])==sum(r['correct'][j]for r in rr);assert np.isclose(float(m['root_mean']),np.mean([np.mean(v)for v in byroot.values()]))
 checks['relation_metrics_recomputed']=len(metrics)
 chains=read(S/'data/chains.json')
 for c in chains:
  for v in c['models'].values():
   corr=correct(v['free'],np.asarray(c['truth'])[1:]);assert np.array_equal(corr,v['correct']);assert np.array_equal(np.cumprod(corr[:,-1]).astype(bool),v['prefix']);assert v['first_error']==(int(np.flatnonzero(~corr[:,-1])[0]+1)if not corr[:,-1].all()else None)
 checks['chain_first_errors_and_prefixes_recomputed']=len(chains)
 current=read(S/'data/current_records.json');assert len(current)==5400
 for r in current:
  c=correct(r['prob'],np.array(r['truth']));assert r['all_correct']==bool(c[-1])and r['holding_identity']==bool(c[0]);p=np.array(r['prob']);y=np.array(r['truth']);confidence=np.r_[np.where(y[:6]>0,p[:6],1-p[:6]),p[6:9][y[6:9].argmax()]];assert r['hard_correct_but_mixed']==bool(c[-1]and confidence.min()<.8)
 checks['current_and_confidence_records_recomputed']=len(current)
 # Independently rebuild the reported new counterexample against old relation-action support.
 train=dict(np.load(D/'data/dataset_train.npz'));outputs=collections.defaultdict(set)
 for e in range(len(train['episode_id'])):
  for t in range(4):outputs[tuple(train['y'][e,t]),int(train['action'][e,t,:6].argmax())].add((tuple(train['y'][e,t+1]),int(train['status'][e,t])))
 observed=[]
 for split,d in ds.items():
  for e in range(len(d['episode_id'])):
   for t in range(4):
    key=(tuple(d['y'][e,t]),int(d['action'][e,t,:6].argmax()));out=(tuple(d['y'][e,t+1]),int(d['status'][e,t]))
    if key in outputs and out not in outputs[key]:observed.append((str(d['episode_id'][e]),t))
 assert observed==[('B_coverage_gap_04_b0',0)];checks['observed_relation_action_counterexample']=observed
 cases=read(S/'results/cases.json');assert len(cases)==4 and len({c['root_id']for c in cases})==4 and all(c['split']!='train'for c in cases)
 failure=next(c for c in cases if c['case_type']=='accurate_state_generalization_failure');assert failure['controls'][0]==416 and failure['status'][0]=='executed_failure';assert all(np.array_equal(hard(x),np.array(y))for x,y in zip(failure['current']['coverage_object'],failure['truth']));assert all(not correct(failure['current']['rssm'][t],np.array(failure['truth'][t]))[-1]for t in range(1,5));checks['failure_correct_object_observation_but_wrong_native_posterior']=True
 comp_source=(S/'source/compute.py').read_text();assert 'torch.optim'not in comp_source and '.backward('not in comp_source;assert "enc=data.cached_encoded(ids,'cuda')"in comp_source;tree=ast.parse((S/'source/cached_evaluator.py').read_text());assert not any(isinstance(n,ast.Attribute)and n.attr=='encode'for n in ast.walk(tree));assert not list(S.glob('models/*.pt'));checks['static_no_training_or_spatial_encoding']=True
 report=(S/'REPORT.md').read_text();assert '未实施'in report and '不能严格证明状态非Markov'in report and '不能称作端到端RSSM改进'in report;assert len(list((S/'figures').glob('*.png')))==8;checks['figures_and_report_limitations']=True
 # Protect actual displayed camera frames against their prior sealed values.
 seal=read(R/'manifest.json')['sha256'];imagefiles={}
 for c in cases:
  for p in ds[c['split']]['public_frame'][c['episode_index']]:
   path=Path(str(p));h=sha(path);expected=seal.get(str(path),seal.get(str(path.relative_to(R))));assert h==expected;imagefiles[str(path)]=h
 save(S/'results/displayed_frames.json',{'utc':now(),'sha256':imagefiles});checks['original_displayed_frames_unchanged']=len(imagefiles)
 save(S/'results/final_verification.json',{'utc':now(),'passed':True,'checks':checks,'verification_model_forwards':0,'new_fits':0,'new_collection':0,'C_D_samples':0,'uploads':0,'limitations':['Only20 unseen-by-training roots; historical exploratory reuse','Coverage train40 are not heldout for coverage head and share old training semantic templates','Object heads not integrated into original RSSM','Outcome conflict refutes deterministic success generalization, not a formal non-Markov proof','Next history-ablation diagnostic is recommended only, not implemented']})
 files={str(p.relative_to(S)):sha(p)for p in sorted(S.rglob('*'))if p.is_file()and p.name!='manifest.json'and 'cache'not in p.relative_to(S).parts and '__pycache__'not in p.parts};save(S/'manifest.json',{'utc':now(),'status':'COMPLETE_FROZEN_AB_REPLAY','files':len(files),'bytes':sum((S/p).stat().st_size for p in files),'sha256':files});print('VERIFIED_SEALED',len(files),sha(S/'manifest.json'),flush=True)
if __name__=='__main__':main()
