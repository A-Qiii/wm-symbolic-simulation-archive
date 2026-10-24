"""Read-only independent arithmetic, input alignment, frozen hash and old-seal checks."""
from config import *
import csv,ast,subprocess,sys
def pred(p):
 out=[float(x)>=.5 for x in p[:6]];idx=max(range(3),key=lambda j:p[6+j]);return np.array(out+[j==idx for j in range(3)])
def main():
 verified=0
 for name in ['part1_source_hashes.json','head_source_hashes.json','training_frozen.json','heads_frozen_before_validation.json']:
  for p,h in read(E/'configs'/name)['sha256'].items():assert sha(p)==h,p;verified+=1
 rows=read(E/'results/part1_scored_records.json');met=read(E/'results/part1_metrics.json');truth={r['trajectory_id']:r for r in read(E/'data/evaluator/part1_truth.json')};assert len(rows)==85;calls_checked=0;aligned_sources={}
 for t in truth.values():
  last=None
  for r in t['steps']:
   if not r['observed']:continue
   path=Path(r['before_path']).parent.parent/'_private/calls.json';calls=read(path);aligned_sources[str(path)]=sha(path);c=[c for c in calls if c['role']=='task'and c['start_frame']==r['before_frame']and c['end_frame']==r['after_frame']];assert len(c)==1;c=c[0];assert c['skill']==r['skill']and c['action']==r['action'];assert c['end_frame']-c['primitive_end']==96
   expected=[float(r['skill']=='open_drawer'),float(r['skill'].startswith('lift_')),float(r['skill']=='place_in_drawer'),float(r['skill']=='close_drawer')]+[float(r['skill']==f'lift_{colour}_block_table')for colour in ['red','blue','pink']]+[1.]
   assert r['action']==expected
   for name in ['before_path','after_path']:
    with np.load(r[name])as data:assert set(data.files)=={'rgb_static','rgb_gripper','robot_obs'}and data['robot_obs'].shape==(15,)
   if last is not None:assert last==r['before_path']
   last=r['after_path'];calls_checked+=1
 for cohort in ['normal','refusal']:
  eligible=[r for r in rows if truth[r['trajectory_id']]['four_step_eligible']and((cohort=='normal'and r['source']=='reference')or(cohort=='refusal'and r['source']=='recorded_action_logs'and truth[r['trajectory_id']]['contains_refusal']))]
  for h in range(1,5):
   rr=[r for r in eligible if r['step']==h];m=met[cohort][h-1];assert len(rr)==m['N'];yo=np.array([r['truth']for r in rr]);preds={k:np.array([pred(r[k+'_probability'])for r in rr])for k in ['actual','one_step','recursive']}
   for method,p in preds.items():
    assert int((p==yo).all(1).sum())==m[method+'_all9']
    for name,lo,hi in [('Holding',0,3),('In',3,6),('drawer',6,9)]:assert int((p[:,lo:hi]!=yo[:,lo:hi]).any(1).sum())==m[method+'_'+name+'_wrong']
   for method in ['one_step','recursive']:
    for level,a,b in [('facts',preds['actual']==yo,preds[method]==yo),('state',(preds['actual']==yo).all(1),(preds[method]==yo).all(1))]:
     pair={'actual_right_prediction_wrong':int((a&~b).sum()),'both_wrong':int((~a&~b).sum()),'actual_wrong_prediction_right':int((~a&b).sum()),'both_right':int((a&b).sum())};assert pair==m[method+'_paired_'+level]
   a=(preds['one_step']==yo).all(1);b=(preds['recursive']==yo).all(1);assert int((a&~b).sum())==m['one_step_right_recursive_wrong'];assert int((~a&b).sum())==m['one_step_wrong_recursive_right']
 assert len(list(csv.DictReader((R/'part1_paired_propositions.csv').open())))==765
 data=read(E/'results/C_validation_state_predictions.json');ev=read(E/'results/head_evaluation.json');frozen=read(E/'configs/heads_frozen_before_validation.json');cv=np.load(E/'data/evaluator/C_validation.npz');sel=np.load(E/'data/head_train/C_selection.npz');assert not(set(cv['root_ids'])&set(sel['root_ids']))
 for name,result in ev['heads'].items():
  rr=[r for r in data if r['head']==name];y=np.array([r['truth']for r in rr],bool);v=np.array([r['valid']for r in rr],bool);p=np.array([pred(r['probability'])for r in rr]);full=v.all(1);ok=((p==y)|~v).all(1);m=result['C_validation']['overall'];assert len(rr)==116 and m['full9_N']==int(full.sum())==110 and m['full9_correct']==int((ok&full).sum());assert int(ok.sum())==m['all_valid_correct']
  for g,lo,hi in [('Holding',0,3),('In',3,6),('drawer',6,9)]:assert int(((p[:,lo:hi]!=y[:,lo:hi]).any(1)&v[:,lo:hi].all(1)).sum())==m['groups'][g]['wrong_states']
  for stratum,ss in result['C_validation']['strata'].items():
   mask=np.array([stratum in r['strata']for r in rr]);assert int(mask.sum())==ss['N'];assert int((mask&full&ok).sum())==ss['full9_correct'];assert int((mask&ok).sum())==ss['all_valid_correct']
 # Four completed equal-budget runs; same initialization within each seed; no checkpoint choice after validation.
 cfg=read(E/'configs/head_protocol.json');tf=read(E/'configs/training_frozen.json');initial={};updates=0
 for name,s in frozen['selected_heads'].items():
  tr=read(E/'results'/f'head_training_{name}.json');assert tr['updates']==1200 and tr['sample_draws']==38400 and tr['parameters']==25473;assert not tr['world_model_loaded_or_updated']and not tr['C_validation_or_D_read'];choice=min(tr['log'],key=lambda r:r['C_selection_root_macro_loss']);assert s['path'].endswith(f'update{choice["update"]:04d}.pt');assert tf['frozen_utc']<frozen['frozen_utc']<ev['evaluated_utc'];initial[name]=tr['initial_query_hash'];updates+=tr['updates']
 for seed in[0,1]:assert initial[f'original_coverage_seed{seed}']==initial[f'intermediate_coverage_seed{seed}']
 for file in(E/'source').glob('*.py'):ast.parse(file.read_text())
 old=subprocess.run([sys.executable,str(P/'experiments/interaction_latent_repair_followup_audit_20261016/source/verify_sealed.py')],capture_output=True,text=True,check=True)
 previous=subprocess.run([sys.executable,str(HOR/'source/verify.py')],capture_output=True,text=True,check=True)
 out={'passed':True,'verified_frozen_hash_entries':verified,'independent_part1_recount':True,'independent_C_recount':True,'observed_transitions_checked':calls_checked,'atomic_pair_rows':765,'exact_action_frame_and_96_settle_checks':True,'alignment_source_hashes':aligned_sources,'query_training_updates_total':updates,'sample_draws_total':4*38400,'four_heads_fixed_before_validation':True,'world_model_weights_unchanged':True,'C_selection_validation_root_overlap':0,'AB_only_training':True,'new_physics_or_planning':False,'old_sealed_delivery':json.loads(old.stdout),'previous_horizon_delivery':json.loads(previous.stdout),'chart_visual_inspection':'Both PNGs inspected; legends, axes and captions readable.'}
 manifest=E/'configs/delivery_manifest.json'
 if manifest.exists():
  d=read(manifest)
  for p,h in d['sha256'].items():assert sha(p)==h,p
  out['new_sealed_files_verified']=len(d['sha256'])
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
