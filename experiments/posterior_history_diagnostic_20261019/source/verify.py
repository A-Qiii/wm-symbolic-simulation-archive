from common import *
import collections,ast
def main():
 sys.addaudithook(guard);checks={'frozen_inputs_unchanged':verify_inputs()};comp=read(S/'results/compute_complete.json');assert comp['training_updates']==comp['collection']==comp['C_D_samples']==comp['uploads']==0
 for p,h in comp['model_weights_unchanged'].items():assert sha(p)==h
 checks['unchanged_RSSM_checkpoints']=len(comp['model_weights_unchanged']);amend=read(S/'configs/implementation_amendment.json');assert sha(S/'source/compute.py')==amend['unchanged_original_compute_sha256'];assert sha(S/'source/resume.py')==amend['resume_sha256'];assert(S/'results/FAILED_attempt1.json').exists();assert not(S/'results/FAILED.json').exists();checks['implementation_repair_preserved_without_protocol_change']=True
 align=read(S/'results/alignment.json');assert len(align)==240;assert sum(r['controls']==0 for r in align)==42;assert all(r['final_frame_matches_public']and r['trace_window_robot_exact']and r['label_endpoint_positions_exact']for r in align);fail=next(r for r in align if(r['episode_id'],r['step'])==('B_coverage_gap_04_b0',1));assert fail['window_control_indices_local']==[413,414,415,416]and fail['static_pixels_changed']>0 and fail['gripper_pixels_changed']>0;checks['alignment_transitions']=240;checks['failed_action_window_steps']=fail['window_control_indices_local'];pixel=read(S/'results/fresh_pixel_alignment.json');assert pixel['passed']and pixel['fresh_DINO_windows']==5 and pixel['fresh_spatial_batches']==2;checks['fresh_image_cache_and_E_exact']=True
 oldrows=0
 for seed in [0,1]:
  for split in SPLITS:
   a=np.load(S/f'data/predictions_joint_seed{seed}_{split}.npz');b=np.load(G/f'data/predictions_joint_seed{seed}_{split}.npz');assert set(a.files)==set(b.files)
   for k in b.files:assert np.array_equal(a[k],b[k],equal_nan=True)if b[k].dtype.kind in 'fc'else np.array_equal(a[k],b[k])
   oldrows+=len(a['prob'])
 checks['all_original_prediction_rows_exact']=oldrows
 raw=np.load(S/'data/paired_predictions.npz');records=read(S/'data/records.json');assert len(records)==5040;ix={(r['seed'],r['episode_id'],r['condition'],r['mode'],r['origin'],r['target']):r for r in records};assert len(ix)==len(records);h0=0;free=0;one=0
 for r in records:
  p=np.asarray(r['task_prob']);y=np.asarray(r['truth']);z=np.r_[p[:6]>=.5,np.eye(3)[p[6:].argmax()]];cc=[np.array_equal(z[:3],y[:3]),np.array_equal(z[3:6],y[3:6]),np.array_equal(z[6:],y[6:])];cc.append(all(cc));assert cc==r['correct']
  assert np.isfinite(r['prob']).all();assert np.array_equal(p,rel_from_prob(r['prob']));assert np.array_equal(y,rel_from_cat(np.array(r['cat'])[:4]));assert int(((z[:3]>0)&(y[:3]==0)).sum())==r['extra_held'];assert int(((z[3:6]>0)&(y[3:6]==0)).sum())==r['extra_in']
  if r['condition']=='continuous'and r['mode']=='current'and r['target']==0:
   other=ix[r['seed'],r['episode_id'],'reset','current',0,0];assert np.array_equal(r['prob'],other['prob']);h0+=1
  if r['condition']=='continuous'and r['mode']=='free_root':
   other=ix[r['seed'],r['episode_id'],'reset','free_root',0,r['target']];assert np.array_equal(r['prob'],other['prob']);free+=1
 for seed in [0,1]:
  for split in SPLITS:
   old=np.load(G/f'data/predictions_joint_seed{seed}_{split}.npz');oi={(str(e),str(m),int(t)):j for j,(e,m,t)in enumerate(zip(old['episode_id'],old['mode'],old['step']))}
   for r in records:
    if r['seed']!=seed or r['split']!=split or r['condition']!='continuous':continue
    mode='current'if r['mode']=='current'else'one_step'if r['mode']=='one_step'else'free'+str(r['target'])if r['mode']=='free_root'else None
    if mode:assert np.array_equal(r['prob'],old['prob'][oi[r['episode_id'],mode,r['target']]]);one+=1
 checks.update(h0_paired_rows_exact=h0,root_open_loop_paired_rows_exact=free,continuous_main_modes_recomputed_exact=one)
 pairs=list(csv.DictReader((S/'results/current_pairs.csv').open()));cur=[r for r in records if r['mode']=='current'and r['target']>0 and r['condition']=='continuous']
 def cohort(r,n):return n=='all'or n=='after_failure'and r['target_previous_status']=='failed'or n=='after_first_failure'and r['after_first_failure']or n=='after_refusal'and r['target_previous_status']=='refused'or n=='after_success'and r['target_previous_status']=='success'or n=='held_positive'and r['held_positive']or n=='drawer_positive'and r['drawer_positive']or n=='stable_held'and r['stable_held']or n=='stable_in'and r['stable_in']or n=='occluded_true_relation'and r['occluded_true_relation']
 for p in pairs:
  rr=[r for r in cur if r['seed']==int(p['seed'])and(p['scene']=='all'or r['scene']==p['scene'])and(p['split']=='all'or r['split']==p['split'])and cohort(r,p['cohort'])];j=FIELDS.index(p['field']);a=np.array([r['correct'][j]for r in rr]);b=np.array([ix[r['seed'],r['episode_id'],'reset','current',r['origin'],r['target']]['correct'][j]for r in rr]);delta=b.astype(int)-a.astype(int);roots=collections.defaultdict(list)
  for r,d in zip(rr,delta):roots[r['root_id']].append(d)
  assert len(rr)==int(p['n'])and len(roots)==int(p['roots']);assert int(a.sum())==int(p['continuous_correct'])and int(b.sum())==int(p['reset_correct']);assert int((delta==1).sum())==int(p['corrected'])and int((delta==-1).sum())==int(p['regressed']);assert np.isclose(float(p['root_mean']),np.mean([np.mean(v)for v in roots.values()]))
 checks['current_root_paired_metric_rows_recomputed']=len(pairs)
 for seed in [0,1]:
  for condition in ['continuous','reset']:
   rr=[r for r in records if r['seed']==seed and r['condition']==condition and r['mode']=='current'and r['target']>0];assert len(rr)==240;assert len({r['root_id']for r in rr})==20
   expected={'continuous':[224,221],'reset':[74,83]}[condition][seed];assert sum(r['correct'][3]for r in rr)==expected
 checks['all20_roots_and_key_scores_verified']=True
 code=(S/'source/compute.py').read_text();assert 'prior,_=m.imagine(posts[-1],actions[:,t])'in code and 'm.observe(prior,enc[:,t+1])'in code;assert 'm.initial(enc[:,t].repeat_interleave(sample,0))'in code;assert 'states[origin]'in code;assert "state,_=m.imagine(state,actions[:,target-1]"in code;assert 'torch.equal(torch.cuda.get_rng_state(),before)'in code;assert 'torch.optim'not in code and '.backward('not in code
 # The suffix loop uses actions only; no future E lookup inside this loop.
 tree=ast.parse(code);func=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='extra');loops=[n for n in ast.walk(func)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='target'];assert len(loops)==1;assert not any(isinstance(n,ast.Name)and n.id=='enc'for n in ast.walk(loops[0]));checks['suffix_no_future_observations_and_paired_rng_static_audit']=True
 rng=read(S/'results/rng_pairs.json');assert len(rng)==160 and all(r['same_observation_feature']and r['same_random_numbers']for r in rng);checks['posterior_random_state_pairs']=len(rng)
 cases=read(S/'results/cases.json');assert len(cases)==4 and len({c['root_id']for c in cases})==4;assert len(list((S/'figures').glob('*.png')))==8;assert '不能定位唯一模块'in(S/'REPORT.md').read_text();checks['report_figures_and_case_roots']=8
 save(S/'results/final_verification.json',{'utc':now(),'passed':True,'checks':checks,'verification_forwards':0,'no_training_collection_C_D_upload':True,'limitations':['No wall-clock timestamps: verified control-index chronology','Joint reset changes h and conditional z; cannot identify unique module','One-seed recovery is one root, not four independent cases','Root open-loop unchanged; observation correction is not failure prediction','Reset severely damages persistent/visible/occluded relations','First-attempt pixel/E resource peak not saved']})
 files={str(p.relative_to(S)):sha(p)for p in sorted(S.rglob('*'))if p.is_file()and p.name!='manifest.json'and 'cache'not in p.relative_to(S).parts and '__pycache__'not in p.parts};save(S/'manifest.json',{'utc':now(),'status':'COMPLETE_FROZEN_POSTERIOR_DIAGNOSTIC','files':len(files),'bytes':sum((S/p).stat().st_size for p in files),'sha256':files});print('VERIFIED_SEALED',len(files),sha(S/'manifest.json'),flush=True)
if __name__=='__main__':main()
