from study import *

def main():
 guard('verify');assert not(U/'SEALED.json').exists();frozen=read(U/'configs/training_frozen.json')
 for p,h in frozen['sha256'].items():assert sha(p)==h,p
 historical=[]
 for folder in [R,I,P/'experiments/paired_outcome_training_20261019',C,P/'experiments/observation_retention_audit_20261019']:
  seal=read(folder/'SEALED.json');manifest=read(folder/'MANIFEST.json');assert sha(folder/'MANIFEST.json')==seal['manifest_sha256']
  for r in manifest['files']:
   p=folder/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
  historical.append({'path':str(folder),'files':len(manifest['files']),'manifest_sha256':seal['manifest_sha256'],'all_unchanged':True})
 assert sha(U/'source/model.py')==sha(C/'source/model.py') and sha(U/'source/training_loop.py')==sha(C/'source/training_loop.py')
 reuse=read(U/'results/control_reuse_audit.json');assert reuse['reuse_allowed'] and reuse['initial_full_state_exact'] and len(reuse['checks'])==4 and all(r['stored']==r['recomputed']for r in reuse['checks'])
 models=read(U/'configs/models_frozen.json');complete=read(U/'results/training_complete.json');assert complete['fits']==2 and complete['updates']==2000 and complete['new_collection']==complete['C_D_samples']==0
 for p,h in models['sha256'].items():assert sha(p)==h,p
 exposures=list(csv.DictReader((U/'configs/exposure_counts.csv').open()));schedule_checks=[]
 _,newdata,_,_=prepared();oldpaired=NewEpisodes('train');oldpaired.episodes=[e for e in oldpaired.episodes if e['branch']in [1,2]]
 assert len(newdata.episodes)==36 and len(oldpaired.episodes)==24
 for i,episode in enumerate(oldpaired.episodes):assert newdata.episodes[(i//2)*3]['root_id']==episode['root_id']

 for seed in [0,1]:
  with np.load(U/f'configs/schedule_seed{seed}.npz')as z:zz={k:z[k]for k in z.files}
  with np.load(C/f'configs/schedule_seed{seed}.npz')as z:old={k:z[k]for k in z.files}
  assert zz['new'].shape==(1000,2,4) and np.array_equal(zz['old'],old['old']) and np.array_equal(zz['latent'],old['latent']) and np.array_equal(zz['new']//3,old['new']//2)
  counts=np.bincount(zz['new'].reshape(-1),minlength=36).reshape(12,3);assert(counts.max(1)-counts.min(1)<=1).all() and counts.sum()==8000
  for b in [0,1,2]:assert sum(int(r['balanced_samples'])for r in exposures if r['seed']==str(seed)and r['branch']==str(b))==int(counts[:,b].sum())
  new=read(U/f'results/train_balanced_seed{seed}.json');control=read(C/f'results/train_state_update_seed{seed}.json');assert new['initial_hash']==control['initial_hash'] and new['updates']==control['updates']==1000 and new['old_samples']==new['new_samples']==8000
  assert [r['update']for r in new['candidates']]==[500,1000] and new['selected']==min(new['candidates'],key=lambda r:(r['development_objective'],r['update'])) and all(np.isfinite(r['objective'])for r in new['curve'])
  schedule_checks.append({'seed':seed,'old_draws_exact':True,'new_root_per_draw_exact':True,'latent_rng_exact':True,'branch_samples':counts.sum(0).tolist(),'new_unique_episodes':36,'new_unique_transitions':216})
 ef=read(U/'configs/evaluation_frozen.json')
 for p,h in ef['sha256'].items():assert sha(p)==h,p
 assert frozen['utc']<models['utc']<ef['utc'];infer=read(U/'results/inference_complete.json');assert infer['weights_unchanged'] and not infer['new_training'] and not infer['new_blind_evaluation']
 for r in infer['reports']:assert r['paired_prefault_probabilities_exact'] and r['suffix_future_observations_used']==0 and r['samples']==8 and r['prediction_rows']==r['roots']*66
 refs=read(U/'results/prediction_references.json');assert len(refs)==24;arrays={}
 for r in refs:
  assert sha(r['path'])==r['sha256']
  with np.load(r['path'])as z:d={k:z[k]for k in z.files}
  assert len(d['prob'])==(792 if r['split']=='train'else 396) and np.isfinite(d['prob']).all() and d['cat_valid'][:,:4].all()
  arrays[r['variant'],r['seed'],r['split']]=d
 for seed in [0,1]:
  for split in ['train','development','evaluation']:
   a=arrays['control',seed,split]
   for v in ['updated','control_final1000','updated_final1000']:
    b=arrays[v,seed,split]
    for k in ['root_id','episode_id','branch','mode','step','cat','cat_valid','status','skill','controls']:assert np.array_equal(a[k],b[k]),(v,seed,split,k)
 metrics=read(U/'results/analysis_complete.json')['metrics'];gates=read(U/'results/gates.json');assert not gates['full_guide_acceptance']
 def count(v,s,sp,m):return next(r['sum']for r in metrics if(r['variant'],r['seed'],r['split'],r['metric'])==(v,s,sp,m))
 events=list(csv.DictReader((U/'results/forgetting_events.csv').open()))
 for g in gates['gates']:
  s=g['seed'];assert g['training_joint_gate']==all(g['training_components'].values()) and g['development_gate']==all(g['development_components'].values())
  assert g['training_components']['paired_current_future12']==(count('updated',s,'train','h2_and_suffix3_pair_joint')==12)
  for sp,bb in g['retention_root_sets'].items():
   for branch,rr in bb.items():
    for v in ['control','updated']:
     actual=sorted(set(r['root_id']for r in events if r['variant']==v and r['seed']==str(s)and r['split']==sp and r['branch']==branch));assert actual==rr[v]
    assert rr['new_forgetting_roots']==sorted(set(rr['updated'])-set(rr['control']))
 assert all(np.isfinite(r['mean']) and np.isfinite(r['ci_low']) and np.isfinite(r['ci_high'])for r in metrics)
 assert len(list((U/'figures').glob('*.png')))==4 and read(U/'results/report_complete.json')['figures']==4
 save(U/'results/verification.json',{'utc':now(),'passed':True,'frozen_input_hashes_unchanged':len(frozen['sha256']),'historical_seals':historical,'same_residual_model_and_training_loop':True,'same_initial_weights_optimizer_verified':True,'schedules':schedule_checks,'two_fits_2000_updates_only':True,'same_evaluation_labels_and_order':True,'original_selection_and_threshold_unchanged':True,'root_forgetting_sets_verified':True,'no_new_collection_C_D_upload':True,'full_guide_acceptance':False})
 files=[]
 for p in sorted(U.rglob('*')):
  if p.is_file() and '__pycache__'not in p.parts and 'cache'not in p.relative_to(U).parts and p.name not in ['MANIFEST.json','SEALED.json']:files.append({'path':str(p.relative_to(U)),'bytes':p.stat().st_size,'sha256':sha(p)})
 save(U/'MANIFEST.json',{'utc':now(),'files':files});save(U/'SEALED.json',{'utc':now(),'manifest_sha256':sha(U/'MANIFEST.json'),'files':len(files),'bytes':sum(r['bytes']for r in files),'verification_passed':True,'no_further_training_or_selection':True});print(json.dumps(read(U/'SEALED.json')),flush=True)

if __name__=='__main__':main()
