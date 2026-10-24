from study import *

def main():
 guard('verify');assert not(U/'SEALED.json').exists();frozen=read(U/'configs/training_frozen.json')
 for p,h in frozen['sha256'].items():assert sha(p)==h,p
 historical=[]
 for folder in [R,I,C]:
  seal=read(folder/'SEALED.json');manifest=read(folder/'MANIFEST.json');assert sha(folder/'MANIFEST.json')==seal['manifest_sha256']
  for r in manifest['files']:
   p=folder/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
  historical.append({'path':str(folder),'files':len(manifest['files']),'manifest_sha256':seal['manifest_sha256'],'all_unchanged':True})
 reuse=read(U/'results/control_reuse_audit.json');assert reuse['reuse_allowed'] and len(reuse['checks'])==4 and all(c['stored']==c['recomputed']for c in reuse['checks'])
 for r in reuse['initialization']:assert r['all_shared_tensors_exact'] and r['zero_residual_forward_exact'] and r['new_optimizer_moments_absent'] and r['new_parameters']==262144
 models=read(U/'configs/models_frozen.json');complete=read(U/'results/training_complete.json');assert complete['fits']==2 and complete['updates']==2000 and complete['new_collection']==complete['C_D_samples']==complete['control_refits']==0
 for p,h in models['sha256'].items():assert sha(p)==h,p
 new_weights=[]
 for seed in [0,1]:
  assert sha(U/f'configs/schedule_seed{seed}.npz')==sha(C/f'configs/schedule_seed{seed}.npz')
  new=read(U/f'results/train_state_update_seed{seed}.json');old=read(C/f'results/train_paired_seed{seed}.json')
  assert new['updates']==old['updates']==1000 and new['old_samples']==new['new_samples']==old['old_samples']==old['new_samples']==8000
  assert [c['update']for c in new['candidates']]==[500,1000] and new['selected']==min(new['candidates'],key=lambda c:(c['development_objective'],c['update']))
  assert all(np.isfinite(r['objective'])for r in new['curve'])
  cp=torch.load(new['selected']['path'],map_location='cpu',weights_only=False);matrix=cp['model']['observation_to_h.weight'];assert matrix.shape==(512,512) and torch.isfinite(matrix).all() and torch.count_nonzero(matrix)>0
  new_weights.append({'seed':seed,'selected_update':new['selected']['update'],'weight_norm':float(matrix.norm()),'nonzero_elements':int(torch.count_nonzero(matrix))})
  del cp
 # Exact legacy optimizer loop except only construction/shared load and new optimizer parameter group.
 text=(C/'source/exact_original_training_loop.py').read_text().replace('m = WorldModel(skill_count=6, event_count=3).cuda()','m = ObservationStateWorldModel(skill_count=6, event_count=3).cuda()').replace("m.load_state_dict(cp['model'])","load_shared(m, cp['model'])").replace('torch.optim.AdamW(m.parameters(),','torch.optim.AdamW(shared_parameters(m),').replace("opt.load_state_dict(copy.deepcopy(cp['optimizer']))","opt.load_state_dict(copy.deepcopy(cp['optimizer']))\n        add_new_optimizer_group(opt, m)")
 assert (U/'source/training_loop.py').read_text()==text
 ef=read(U/'configs/evaluation_frozen.json')
 for p,h in ef['sha256'].items():assert sha(p)==h,p
 assert frozen['utc']<models['utc']<ef['utc'];assert read(U/'configs/evaluation_requirements.json')['utc']<ef['utc']
 gradients=read(U/'results/gradient_start_check.json');assert len(gradients)==2 and all(r['zero_initialized_weight_gradient_norm']>0 and r['optimizer_steps']==0 for r in gradients)
 infer=read(U/'results/inference_complete.json');assert infer['model_weights_unchanged'] and not infer['new_blind_evaluation'] and not infer['new_training_during_evaluation']
 for r in infer['reports']:assert r['paired_prefault_probabilities_exact'] and r['suffix_future_observations_used']==0 and r['samples']==8 and r['prediction_rows']==r['roots']*66
 refs=read(U/'results/prediction_references.json');assert len(refs)==24;arrays={}
 for ref in refs:
  assert sha(ref['path'])==ref['sha256']
  with np.load(ref['path'])as z:d={k:z[k]for k in z.files}
  assert len(d['prob'])==(792 if ref['split']=='train'else 396) and np.isfinite(d['prob']).all() and d['cat_valid'][:,:4].all()
  for rid in set(d['root_id']):
   ix=lambda b:np.where((d['root_id']==rid)&(d['branch']==b)&(d['mode']=='one_step')&(d['step']==2))[0][0]
   assert np.array_equal(d['prob'][ix(1)],d['prob'][ix(2)])
  arrays[ref['variant'],ref['seed'],ref['split']]=d
 for seed in [0,1]:
  for sp in ['train','development','evaluation']:
   ref=arrays['control',seed,sp]
   for v in ['updated','control_final1000','updated_final1000']:
    d=arrays[v,seed,sp]
    for k in ['root_id','episode_id','branch','mode','step','cat','cat_valid','status','skill','controls']:assert np.array_equal(ref[k],d[k]),(v,seed,sp,k)
 a=read(U/'results/analysis_complete.json');metrics=a['metrics'];gates=read(U/'results/gates.json');assert not gates['full_guide_acceptance']
 def count(v,s,sp,m):return next(r['sum']for r in metrics if (r['variant'],r['seed'],r['split'],r['metric'])==(v,s,sp,m))
 for g in gates['gates']:
  s=g['seed'];assert g['training_necessary_gate']==(count('updated',s,'train','h0_held')==12 and count('updated',s,'train','h2_pair_joint')==12)
  assert g['training_joint_current_future_gate']==(g['training_necessary_gate'] and count('updated',s,'train','h2_and_suffix3_pair_joint')==12)
  assert g['development_support_gate']==all(g['development_components'].values())
 assert all(np.isfinite(r['mean']) and np.isfinite(r['ci_low']) and np.isfinite(r['ci_high'])for r in metrics)
 o=read(U/'results/same_observation_control.json');assert o['all_same_E_loses_pair_discrimination'] and o['rows']==192
 rr=list(csv.DictReader((U/'results/same_observation_control.csv').open()));assert len(rr)==192
 assert all(r['same_E_h_q_probabilities_exact']=='True' and r['same_E_h2_pair_joint']=='False' and r['same_E_h2_and_suffix3_pair_joint']=='False' for r in rr)
 for v in ['control','updated']:
  delta=[float(r['actual_E_pair_h_rms_difference'])for r in rr if r['variant']==v]
  assert all(np.isfinite(x) and x>=0 for x in delta)
  if v=='control':assert all(x==0 for x in delta)
 assert len(list((U/'figures').glob('*.png')))==4 and read(U/'results/report_complete.json')['figures']==4
 assert 'RGB' in (U/'REPORT.md').read_text() and '探索性' in (U/'REPORT.md').read_text()
 save(U/'results/verification.json',{'utc':now(),'passed':True,'frozen_input_hashes_unchanged':len(frozen['sha256']),'historical_seals':historical,'zero_init_forward_exact':True,'shared_optimizer_reused_new_moments_absent':True,'live_initial_gradient_checks':gradients,'trained_residual_weights':new_weights,'same_old_new_latent_schedule':True,'same_loss_and_optimizer_loop_except_added_parameters':True,'two_fits_2000_updates_only':True,'same_scoring_labels_and_order':True,'same_E_control_passed':True,'original_gates_not_relaxed_and_joint_gate_added':True,'no_new_collection_C_D_upload':True,'full_guide_acceptance':False})
 files=[]
 for p in sorted(U.rglob('*')):
  if p.is_file() and '__pycache__'not in p.parts and 'cache'not in p.relative_to(U).parts and p.name not in ['MANIFEST.json','SEALED.json']:files.append({'path':str(p.relative_to(U)),'bytes':p.stat().st_size,'sha256':sha(p)})
 save(U/'MANIFEST.json',{'utc':now(),'files':files});save(U/'SEALED.json',{'utc':now(),'manifest_sha256':sha(U/'MANIFEST.json'),'files':len(files),'bytes':sum(r['bytes']for r in files),'verification_passed':True,'no_further_training_or_selection':True});print(json.dumps(read(U/'SEALED.json')),flush=True)

if __name__=='__main__':main()
