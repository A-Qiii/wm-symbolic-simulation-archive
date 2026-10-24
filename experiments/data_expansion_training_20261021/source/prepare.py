from common import *
import shutil
def main():
 assert not(S/'configs/design_frozen.json').exists()
 for name in ['configs','cache','models','results','reports','logs','figures']:(S/name).mkdir(parents=True,exist_ok=True)
 started=time.time();expected='f74cd2f7a7da565bdaf893293b5e51e3dada20a9eb0778c52468209b627f546c'
 assert sha(D/'MANIFEST.json')==expected
 for f in read(M/'MANIFEST.json')['files']:assert sha(M/f['path'])==f['sha256']
 for path,h in read(M/'configs/design_frozen.json')['sha256'].items():assert sha(Path(path))==h,path
 combined=read(D/'data/combined_training_manifest.json');membership=read(D/'data/membership.json')['episodes'];assert combined['episode_count']==868 and combined['transition_count']==4968
 assert len(membership['train'])==640 and len(membership['development'])==80 and len(membership['combination_test'])==80
 assert not any(set(membership[a])&set(membership[b])for a in membership for b in membership if a!=b)
 assert not read(D/'results/final_data_audit.json')['quarantined_families']
 old=read(O/'cache/train_metadata.json');assert len(old)==228 and all(e['kind']=='core'for e in old[:120])and all(e['kind']=='role'for e in old[120:])
 roots={r['root_id']:r for r in read(D/'data/roots_manifest.json')};by=collections.defaultdict(list)
 for row in read(D/'data/data_manifest.json')['records']:by[row['episode_id']].append(row)
 eps={}
 for split,ids in membership.items():
  eps[split]=[]
  for eid in sorted(ids):
   rr=sorted(by[eid],key=lambda r:r['step']);root=roots[rr[0]['root_id']];assert len(rr)==6 and all(r['split']==split for r in rr)
   eps[split].append({'episode_id':eid,'root_id':root['root_id'],'family_id':root['family_id'],'split':split,'length':6,'kind':'expanded','state_kind':root['state_kind'],'environment':root['environment'],'branch':rr[0]['branch_index'],'branch_mode':root['branch_modes'][rr[0]['branch_index']],'fault_step':root['fault_step'],'rows':rr})
 bad=read(D/'results/setup_failure_analysis.json')['mismatched_episodes'];assert len(bad)==2
 for item in bad:
  assert item['episode_id']in set(membership['train']);rr=by[item['episode_id']];actual=read(rr[0]['labels_path'])['before']['categorical'];assert actual==item['actual_initial_categorical']and not any(x in [3,6]for x in actual[:3])
 save(S/'results/failed_setup_membership_check.json',{'episodes':bad,'retained_in_training':True,'actual_labels_used':True,'intended_state_not_model_input':True})
 save(S/'configs/episodes.json',eps)
 shutil.copy2(M/'configs/input_normalization.json',S/'configs/input_normalization.json')
 schedules=[]
 for seed in [0,1]:
  z=np.load(M/f'configs/schedule_seed{seed}.npz');ids=z['ids'].copy();starts=z['starts'].copy();rng=np.random.default_rng(2610219000+seed)
  mask=ids>=120;ids[mask]=rng.integers(120,868,size=int(mask.sum()))
  assert np.all((ids<120)==(z['ids']<120))and np.array_equal(starts,z['starts'])
  oldlens=np.array([e['length']for e in old]);newlens=np.r_[oldlens,np.full(640,6)];assert np.array_equal(oldlens[z['ids']],newlens[ids])
  exposure=int((newlens[ids]-starts).sum());assert exposure==[89834,89839][seed]
  np.savez(S/f'configs/schedule_seed{seed}.npz',ids=ids,starts=starts)
  schedules.append({'seed':seed,'updates':1500,'episode_draws':24000,'full_start_draws':12000,'middle_start_draws':12000,'effective_transition_draws':exposure,'core_draws':12000,'six_step_draws':12000,'unique_new_training_episodes_seen':int(len(set(ids[ids>=228].tolist()))),'new_episode_draws':int((ids>=228).sum()),'old_schedule_sha256':sha(M/f'configs/schedule_seed{seed}.npz'),'new_schedule_sha256':sha(S/f'configs/schedule_seed{seed}.npz')})
 save(S/'configs/sampling_audit.json',schedules)
 protocol={'utc':now(),'scope':'Old228 versus expanded868 data, existing frozen-vision ObjectAug/cold_mix architecture only; no closed-loop or architecture changes',
  'data':{'old_training_episodes':228,'new_training_episodes':640,'combined_training_episodes':868,'old_training_transitions':1128,'new_training_transitions':3840,'combined_training_transitions':4968,'new_development_episodes':80,'old_development_regression_episodes':78,'heldout_combination_episodes':80,'two_failed_setup_episodes':'Retained in train with actual all-table labels; intended state kind is reporting metadata, never a model input','heldout':'New A/B family-disjoint combination:>=2In and every object In or Held. Test payload inaccessible before model freeze. No C/D.'},
  'comparison':{'baseline':'Reuse sealed M cold_mix seeds0,1 step1500, not continued finetuning','new':'Two new fits from exact matching initializations, seeds0,1;1500 updates each','optimizer':'AdamW3e-4,weight_decay1e-4,clip1;BF16 autocast;two microbatches of8','sampling':'Per microbatch4 old four-step core +4 six-step. Keep original core IDs, crop-start schedule and sequence lengths. Expand six-step pool108->748. No uniform868 claim. Same24000 draws and89834/89839 effective transitions per seed.','loss':'Unchanged M losses.py:current+.5*(prior+multi3+terminal_relative_h2+outcome)','normalization':'Original old-training-only geometry means/std and class weights unchanged','encoder':'Existing DINO and CELL learned encoder per seed frozen; old228 and old78 cached features reused exactly. New boundary-only causal four-frame dual-camera cache, fresh visible registry per episode, no probe-window insertion','selection':'Fixed step1500 only; no checkpoint or threshold selection, no planned hyperparameter repairs'},
  'evaluation':{'thresholds':'Held/In0.5;drawer argmax; masks exclude invalid labels from denominators','cohorts':['old_development78','new_development80','combination_test80_after_freeze'],
   'primary':'Cold initial relations at every nonterminal origin; h0 and middle separately. Complete observed-path state and one-step from origin0. Free1/3/5 and full6 from cold origins. Held/In TP,FP,positive/negative denominators; joint Held3+In3+drawer on fully valid states. Shared GT-stable non-target In and Held intervals, both-correct and prior-correct forgetting denominators.',
   'heldout_subset':'Report all test conditions AND conditions whose actual target state satisfies frozen>=2In/allInOrHeld combination. Eligibility from labels only after model/metric freeze, no outcome-driven replacement.',
   'strata':'Prepared state kind, actual initial relation, environment, requested skill, nominal/miss/order branch, color and horizon; report exact eligible denominators and two seeds separately.',
   'single_item_diagnostic':'For middle origins on development only, replace zero recurrent prior with existing causal public-prefix prior. Same current window, model, actions and future targets. No GT initializer; no fitting. Diagnose earliest initial-versus-transition error, not repair individual cases.',
   'uncertainty':'If bootstrap used, paired resampling by layout family,2000 draws fixed seed; not independent windows or extra training-seed evidence'},
  'budget':{'new_fits':2,'updates_total':3000,'training_wall_minutes':45,'total_wall_minutes':90,'single_GPU_serial':True,'no_downloads_or_uploads':True}}
 save(S/'configs/protocol.json',protocol)
 save(S/'configs/launch.json',{'started_epoch':time.time(),'deadline_epoch':time.time()+5400,'utc':now()})
 oldpaths=[M/f'models/cold_mix_seed{seed}_step1500.pt'for seed in [0,1]]
 save(S/'configs/baselines_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in oldpaths},'fits':[read(M/f'results/fit_cold_mix_seed{seed}.json')for seed in [0,1]]})
 paths=list((S/'source').glob('*.py'))+[S/'configs'/name for name in ['protocol.json','episodes.json','sampling_audit.json','input_normalization.json','baselines_frozen.json']]+[S/f'configs/schedule_seed{seed}.npz'for seed in [0,1]]+[D/'MANIFEST.json',D/'data/combined_training_manifest.json',O/'cache/train_metadata.json',O/'cache/development_metadata.json',CELL/'configs/normalization.json']
 paths += [CELL/f'models/joint_seed{seed}/latest.pt'for seed in [0,1]]
 paths += [M/'source/mid_train.py',M/'source/mid_data.py',OLD/'source/data.py',O/'source/data.py',P/'experiments/object_identity_readout_20261019/source/run.py',P/'src/wm_symbolic/vision_cache.py',P/'src/wm_symbolic/world_model.py',P/'src/wm_symbolic/public_geometry.py']
 save(S/'configs/design_frozen.json',{'utc':now(),'before_new_fits_or_new_test_payload':True,'sha256':{str(p):sha(p)for p in paths}})
 save(S/'results/preparation.json',{'utc':now(),'old559_files_verified':True,'expansion_manifest_verified':True,'failed_setup_episodes_retained':2,'seconds':time.time()-started})
 print('DESIGN_FROZEN',json.dumps(schedules),flush=True)
if __name__=='__main__':main()
