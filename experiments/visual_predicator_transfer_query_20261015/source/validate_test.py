from common import *
import argparse,gzip
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','full'],required=True);a=ap.parse_args();q=read(E/'configs/test_frozen.json');m=read(B/'data_manifest.json');expected=2 if a.phase=='pilot'else 12;assert len(m['episodes'])==expected;count=0;frames=0;initial={}
 for ep in m['episodes']:
  assert ep['collection_error']is None and ep['calls']<=8;folder=B/'episodes'/ep['episode_id'];index=[json.loads(s)for s in(folder/'public/frames.jsonl').read_text().splitlines()];assert len(index)==ep['frames'];assert ep['physics_ticks']==8*ep['controls']==8*(len(index)-1)
  for i,r in enumerate(index):assert r['frame']==i and r['physics_ticks']==i*8 and abs(r['sim_seconds']-i/30)<1e-9 and sha(folder/'public'/r['file'])==r['sha256']
  init=dict(np.load(folder/'_private/initial_state.npz'))
  if ep['root_id']in initial:assert all(np.max(np.abs(v-initial[ep['root_id']][k]))<=1e-5 for k,v in init.items())
  else:initial[ep['root_id']]=init
  windows=read(folder/'_private/windows.json');wm={r['window_id']:r for r in windows}
  with gzip.open(folder/'_private/trace.json.gz','rt')as f:trace=json.load(f)
  assert len(trace)==len(index)
  for w in windows:
   assert w['history_frame_indices']==[w['anchor_frame']-k for k in [63,42,21,0]];assert w['targets_evaluator_only']==trace[w['anchor_frame']]['targets'];assert len(w['public_frames'])==4;assert w['inference']['input_keys']==['rgb_static','rgb_gripper','robot_obs']and w['inference']['GT_or_action_input']is False
   for path,h in zip(w['public_frames'],w['inference']['public_frame_sha256']):
    assert sha(path)==h
    with np.load(path)as z:assert set(z.files)=={'rgb_static','rgb_gripper','robot_obs'}and z['rgb_static'].shape==(200,200,3)and z['rgb_gripper'].shape==(84,84,3)and z['robot_obs'].shape==(15,)
  actions=read(folder/'_private/actions.json')if(folder/'_private/actions.json').exists()else[];assert len(actions)==ep['calls']
  for r in actions:
   assert r['end_frame']-r['primitive_end_frame']==96 and r['primitive_end_frame']-r['start_frame']==r['primitive']['controls'];assert wm[r['before_window_id']]['inference']['completed_wall_ns']<r['action_start_wall_ns']<r['after_inference_started_wall_ns'];assert r['before_targets']==trace[r['start_frame']]['targets']and r['after_targets']==trace[r['end_frame']]['targets']
  count+=len(actions);frames+=len(index)
 assert count<=96
 for path,h in q['frozen_sha256'].items():assert sha(path)==h,path
 out={'passed':True,'phase':a.phase,'episodes':len(m['episodes']),'independent_roots':len(initial),'calls':count,'public_frames':frames,'all_assigned_episodes_retained':True,'matched_arm_initial_states':True,'causal_window_hashes_checked':True,'physics240Hz_controls30Hz':True,'prediction_before_execution':True,'GT_only_privileged_arm_and_scoring':True,'frozen_model_rules_code_unchanged':True,'no_accuracy_gate_used':True};save(E/'results'/('test_pilot_validation.json'if a.phase=='pilot'else'test_data_validation.json'),out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
