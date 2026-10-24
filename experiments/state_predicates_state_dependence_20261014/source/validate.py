from experiment import *
import argparse,gzip,shutil,collections
import numpy as np
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','full'],required=True);a=ap.parse_args();q=verify_frozen();m=read(B/'data_manifest.json');roots=read(B/'roots_manifest.json');rmap={r['root_id']:r for r in roots};expected=set(q['pilot_roots'])if a.phase=='pilot'else set(rmap);eps=[b['episode_id']for r in roots if r['root_id']in expected for b in r['branches']];assert{r['episode_id']for r in m['episodes']}==set(eps)
 assert len(m['records'])<=len(eps)<=54 and len(m['windows'])==2*len(m['records']);wmap={w['window_id']:w for w in m['windows']};nbytes=nfiles=0;outcomes=collections.Counter();validfacts=0;invalidfacts=0;prior_initial={};maxreset=0.;uncovered=[];declarations=0;refusedholds=0
 for s in m['episodes']:
  assert s['collection_error']is None and s['physics_guard_passed'];assert s['physics_ticks']==8*s['controls'];d=B/'episodes'/s['episode_id'];idx=[json.loads(x)for x in(d/'public/frames.jsonl').read_text().splitlines()];cmd=[json.loads(x)for x in(d/'public/commands.jsonl').read_text().splitlines()];assert len(idx)==s['frames']and len(cmd)==s['controls']
  for i,r in enumerate(idx):
   assert r['frame']==i and r['physics_ticks']==i*8 and abs(r['sim_seconds']-i/30)<1e-10;fp=d/'public'/r['file'];assert fp.stat().st_size==r['bytes']and sha(fp)==r['sha256'];nbytes+=r['bytes'];nfiles+=1
  assert[x['arrival_frame']for x in cmd]==list(range(1,s['frames']));assert all(set(x)=={'arrival_frame','physics_ticks','sim_seconds','target_xyz','target_quaternion','gripper'}for x in cmd)
  with np.load(d/'_private/initial_state.npz')as z:state={k:z[k].copy()for k in z.files}
  if s['root_id']in prior_initial:
   diff=max(float(np.abs(state[k]-prior_initial[s['root_id']][k]).max())for k in state);maxreset=max(maxreset,diff);assert diff<=1e-5
  else:prior_initial[s['root_id']]=state
  with gzip.open(d/'_private/trace.json.gz','rt')as f:tr=json.load(f)
  assert len(tr)==s['frames'];assert all(r['physics_ticks']==r['frame']*8 for r in tr)
  for h in s['holds']:assert h['controls']==96 and h['end_frame']-h['start_frame']==96
  assert[s['holds'][i]['name']for i in range(2)]==['initial_history_hold','preparation_boundary_hold']
  for w in read(d/'public/windows.json'):
   inds=[w['anchor_frame']-k for k in [63,42,21,0]];assert w['history_frame_indices']==inds and min(inds)>=0 and len(set(w['public_frames']))==4;assert abs(w['simulation_times'][-1]-w['simulation_times'][0]-2.1)<1e-10
   lab=read(d/'_private'/(w['window_id']+'.json'));assert[q['frame']for q in lab['frames']]==inds
   for fi,fp in zip(inds,w['public_frames']):
    with np.load(fp)as z:
     assert set(z.files)=={'rgb_static','rgb_gripper','robot_obs'};assert z['rgb_static'].shape==(200,200,3)and z['rgb_gripper'].shape==(84,84,3)and z['robot_obs'].shape==(15,);assert np.array_equal(z['robot_obs'],tr[fi]['robot_obs'])
   t=lab['all_7_targets'];row=tr[w['anchor_frame']];cat=np.array(row['labels']['categorical'][:4]);cv=np.array(row['labels']['categorical_valid'][:4],bool);assert t['y']==np.r_[cat[:3]==6,cat[:3]==3,cat[3]==2].tolist();assert t['valid']==np.r_[cv[:3],cv[:3],cv[3]].tolist()
  branch=rmap[s['root_id']]['branches'][s['branch']]
  assert[p['skill']for p in s['preparation_attempts']]==branch['preparation_skills'][:len(s['preparation_attempts'])]
  if not s['covered']:
   assert s['uncovered_reason']and not s['diagnostic_declared'];uncovered.append({'episode_id':s['episode_id'],'reason':s['uncovered_reason']});continue
  declarations+=1;pub=read(d/'public/pairs.json')[0];priv=read(d/'_private/pairs.json')[0];decl=read(d/'public/diagnostic_declaration.json');assert pub['skill']==branch['declared_skill']and pub['declared_arguments']==branch['declared_arguments'];assert decl['start_frame']==priv['start_frame'];assert sha(d/'public/diagnostic_declaration.json')==priv['declaration_sha256_before_execution'];assert pub['before_window_id']==wmap[pub['before_window_id']]['window_id'];assert wmap[pub['before_window_id']]['anchor_frame']==priv['start_frame'];assert wmap[pub['after_window_id']]['anchor_frame']==priv['end_frame'];assert priv['end_frame']-priv['primitive_end_frame']==96;assert priv['primitive_end_frame']-priv['start_frame']==priv['primitive']['controls']<=900
  if priv['primitive']['status']=='refused':assert priv['primitive']['controls']==0 and priv['macro_controls']==96;refusedholds+=1
  valid=np.array(priv['before_targets']['valid'])&np.array(priv['after_targets']['valid']);validfacts+=int(valid.sum());invalidfacts+=int((~valid).sum());outcomes[priv['outcome_group']]+=1
 measured=nbytes/max(1,nfiles);remaining=54-len(eps);worst=remaining*(1+96+2*900+96+900+96);projected=worst*measured;free=shutil.disk_usage(B).free;need=1.5*projected+5*1024**3;assert free>need
 result={'created_utc':now(),'passed':True,'phase':a.phase,'independent_roots':len(expected),'episodes':len(eps),'planned_diagnostic_calls':len(eps),'covered_completed_calls':len(m['records']),'uncovered':uncovered,'declarations_before_execution_verified':declarations,'real_96_control_postholds_after_refusal':refusedholds,'outcomes':dict(outcomes),'all7_valid_facts_at_both_ends':validfacts,'uncertain_fact_endpoints_retained':invalidfacts,'windows':len(m['windows']),'causal_2p1s_unique_real_frames_verified':True,'public_RGB_robot_allowlist_verified':True,'matched_initial_scene_and_robot_max_abs_difference':maxreset,'public_frames_hashed':nfiles,'public_frame_bytes':nbytes,'measured_bytes_per_frame':measured,'remaining_worstcase_frames':worst,'remaining_projected_public_bytes':projected,'required_free_bytes_with_margin':need,'free_bytes':free,'no_success_quota':True,'frozen_files_unchanged':True,'model_predictions':0,'training_updates':0};save(E/'results'/('pilot_gate.json'if a.phase=='pilot'else'data_validation.json'),result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
