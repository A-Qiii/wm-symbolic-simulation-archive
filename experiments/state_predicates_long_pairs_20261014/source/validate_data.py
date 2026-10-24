from pathlib import Path
import json,hashlib,argparse,shutil,gzip
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_long_pairs_20261014';B=P/'data/predicate_long_pairs_20261014'
PUBLIC={'rgb_static','rgb_gripper','depth_static','depth_gripper','robot_obs','seg_static','seg_gripper'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','full'],required=True);args=ap.parse_args();p=read(E/'configs/protocol.json');m=read(B/'data_manifest.json');expected=2 if args.phase=='pilot'else 30
 assert len(m['roots'])==expected and len(m['records'])==4*expected and len(m['windows'])==5*expected
 for path,h in p['frozen_sha256'].items():assert sha(path)==h,path
 assert sha(B/'roots_manifest.json')==p['roots_manifest_sha256'];roots=read(B/'roots_manifest.json');rmap={r['root_id']:r for r in roots};wmap={w['window_id']:w for w in m['windows']};audits=[];nbytes=0;nfiles=0;distinct_hashes=set();allkeys=set()
 if args.phase=='pilot':assert {r['root_id']for r in m['roots']}==set(p['pilot_roots'])
 else:assert {r['root_id']for r in m['roots']}==set(rmap)
 for s in m['roots']:
  assert s['collection_error']is None and s['physics_guard_passed'];assert s['attempted_macros']==s['completed_macro_pairs']==4;assert s['physics_ticks']==8*s['controls'];assert s['initial_history_hold']['controls']==96
  d=B/'episodes'/s['root_id'];idx=[json.loads(x)for x in(d/'public/frames.jsonl').read_text().splitlines()];cmd=[json.loads(x)for x in(d/'public/commands.jsonl').read_text().splitlines()];assert len(idx)==s['frames']and len(cmd)==s['controls']
  for i,row in enumerate(idx):
   assert row['frame']==i and row['physics_ticks_since_reset']==8*i and abs(row['sim_seconds']-i/30)<1e-10
   fp=d/'public'/row['file'];assert fp.stat().st_size==row['bytes']and sha(fp)==row['sha256'];nbytes+=row['bytes'];nfiles+=1;distinct_hashes.add(row['sha256'])
  assert [x['arrival_frame']for x in cmd]==list(range(1,s['frames']));assert all(set(x)=={'arrival_frame','physics_ticks_since_reset','sim_seconds','target_xyz','target_quaternion','gripper'}for x in cmd)
  with gzip.open(d/'_private/trace.json.gz','rt')as f:tr=json.load(f)
  assert len(tr)==s['frames']
  for w in read(d/'public/windows.json'):
   a=w['anchor_frame'];inds=[a-k for k in [63,42,21,0]];assert w['history_frame_indices']==inds and min(inds)>=0;assert len(set(w['public_frames']))==4
   assert np.allclose(w['simulation_times'],np.array(inds)/30,rtol=0,atol=1e-10)and abs(w['simulation_times'][-1]-w['simulation_times'][0]-2.1)<1e-10
   lab=read(d/'_private'/(w['window_id']+'.json'));assert [q['frame']for q in lab['frames']]==inds
   for fi,fp in zip(inds,w['public_frames']):
    with np.load(fp)as z:
     assert all(k in PUBLIC or k.startswith('calib_')for k in z.files);assert PUBLIC<=set(z.files);allkeys.update(z.files);assert z['rgb_static'].shape==(200,200,3)and z['rgb_gripper'].shape==(84,84,3);assert z['robot_obs'].shape==(15,);assert np.array_equal(z['robot_obs'],tr[fi]['robot_obs'])
   t=lab['all_7_targets'];cat=np.array(tr[a]['labels']['categorical'][:4]);cv=np.array(tr[a]['labels']['categorical_valid'][:4],bool);assert t['y']==np.r_[cat[:3]==6,cat[:3]==3,cat[3]==2].tolist();assert t['valid']==np.r_[cv[:3],cv[:3],cv[3]].tolist()
  pp=read(d/'public/pairs.json');pr=read(d/'_private/pairs.json');assert [q['skill']for q in pp]==rmap[s['root_id']]['planned_skills']
  for q,z in zip(pp,pr):
   assert q['pair_id']==z['pair_id'];assert wmap[q['before_window_id']]['anchor_frame']==z['start_frame'];assert wmap[q['after_window_id']]['anchor_frame']==z['end_frame'];assert z['end_frame']-z['primitive_end_frame']==96;assert z['primitive_end_frame']-z['start_frame']==z['primitive']['controls']<=900;assert z['posthold']['controls']==96;assert z['macro_controls']>=96
   assert len(z['before_targets']['y'])==len(z['after_targets']['y'])==len(z['before_targets']['valid'])==len(z['after_targets']['valid'])==7
   assert all(q['declared_arguments'].get('color')==q['skill'].split('_')[1] for _ in [0])if q['skill'].startswith('lift_')else q['declared_arguments']=={}
  audits.append({'root_id':s['root_id'],'frames':s['frames'],'pairs':len(pp),'all7_valid_pairs':sum(q['both_all7_valid']for q in pr),'controller_failed_or_refused':sum(q['primitive']['status']!='success'for q in pr),'posthold_held_loss_macros':len(s['posthold_any_held_losses'])})
 measured=nbytes/nfiles;remaining=30-len(m['roots']);worstframes=remaining*(1+96+4*(900+96));projected=worstframes*measured;free=shutil.disk_usage(B).free;required=1.5*projected+5*1024**3;assert free>=required
 result={'passed':True,'phase':args.phase,'roots':expected,'complete_causal_macro_pairs':len(m['records']),'unique_windows':len(m['windows']),'all7_valid_pairs':sum(a['all7_valid_pairs']for a in audits),'public_frame_files_hashed':nfiles,'distinct_public_file_hashes':len(distinct_hashes),'public_frame_bytes':nbytes,'measured_bytes_per_frame':measured,'remaining_worstcase_frames':worstframes,'remaining_projected_public_bytes':projected,'disk_required_with_margin_bytes':required,'free_bytes':free,'public_keys':sorted(allkeys),'actual_physics_tick_alignment':True,'all7_label_semantics_and_validity_checked':True,'no_future_or_padded_frames':True,'frozen_sources_unchanged':True,'root_audits':audits};save(E/'results'/('pilot_gate.json'if args.phase=='pilot'else'data_validation.json'),result);print(json.dumps({k:v for k,v in result.items()if k not in ['root_audits','public_keys']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
