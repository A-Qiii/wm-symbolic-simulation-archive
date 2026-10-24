from common import *
import gzip,argparse
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--phase',choices=['train_dev','evaluation'],required=True);args=pa.parse_args();guard('train'if args.phase=='train_dev'else'eval');verify_freeze();records=[r for r in read(S/'data/data_manifest.json')['records']if(r['split']=='evaluation')==(args.phase=='evaluation')];counts=collections.Counter();alignment=[];ncommands=0;changed=0
 for r in records:
  d=Path(r['labels_path']).parent;lab=read(r['labels_path']);res=read(d/'result_private.json');commands=read(d/'controls.json');assert lab['control_samples_including_start']==len(commands)+1;assert res['physics_guard_passed'];assert res['controller_sha256']==read(S/'configs/protocol.json')['controller']['sha256'];active=r['branch_index']==2 and r['step']==1
  for a in commands:
   assert a['declared_execution_disturbance']==active;assert a['requested_gripper']==a['gripper'];assert np.array_equal(a['requested_xyz'],a['target_xyz'])and np.array_equal(a['requested_quaternion'],a['target_quaternion']);assert a['applied_gripper']==(1 if active and a['requested_gripper']==-1 else a['requested_gripper']);changed+=a['applied_gripper']!=a['requested_gripper'];ncommands+=1
  initial=dict(np.load(r['obs_before']));final=dict(np.load(r['obs_after']));window=dict(np.load(d/'boundary_history_3.npz'));assert set(final)==set(window)and all(np.array_equal(final[k],window[k])for k in final)
  if not commands:assert all(np.array_equal(initial[k],final[k])for k in initial)
  with gzip.open(d/'trace_private.json.gz','rt')as f:trace=json.load(f)
  assert len(trace)==len(commands)
  if trace:
   assert np.allclose(trace[-1]['robot_obs'],final['robot_obs'],atol=0,rtol=0)
   for c in COLORS:assert np.allclose(trace[-1]['objects']['block_'+c][:3],lab['after']['continuous']['objects'][c]['position_m'],atol=1e-12,rtol=0)
  counts['success'if lab['success']else'executed_failure'if commands else'refused']+=1
  alignment.append({'episode_id':r['episode_id'],'step':r['step'],'controls':len(commands),'real_final_window_exact':True,'trace_endpoint_exact':True,'disturbance':active})
 roots=read(S/'results/root_audits.json');selected=[a for a in roots if(a['split']=='evaluation')==(args.phase=='evaluation')];assert all(a['paired_prefix_public_arrays_exact']for a in selected)
 families=collections.defaultdict(set)
 for root in read(S/'data/roots_manifest.json'):families[root['family_id']].add(root['split'])
 assert all(len(s)==1 for s in families.values());save(S/f'results/audit_{args.phase}.json',{'utc':now(),'passed':True,'transitions':len(records),'independent_roots':len(selected),'statuses':dict(counts),'requested_applied_commands':ncommands,'open_gripper_overrides':changed,'valid_matched_roots':sum(a['valid_matched_root']for a in selected),'paired_public_prefix_exact_all':True,'root_family_leakage':False,'per_transition':alignment,'no_timestamps_invented':'Control index and exact states used; source has no wall-clock frame timestamps.'});print('AUDIT_COMPLETE',args.phase,len(records),dict(counts),flush=True)
if __name__=='__main__':main()
