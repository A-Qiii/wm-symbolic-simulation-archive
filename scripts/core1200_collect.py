"""Physical collection for frozen six-skill study; private targets separated."""
from pathlib import Path
from collections import deque, Counter
import argparse, json, hashlib, datetime, time, os, fcntl, signal, gzip
import numpy as np
import pybullet as p
import hydra, imageio.v2 as imageio
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter, save_npz
from wm_symbolic.physical_oracle import PhysicalOracle
from wm_symbolic.calibrated_labels import calibrated_labels_at
from wm_symbolic.core_ownership import drawer_ownership
from evaluate_physical_oracle import physics_guard

P=Path('/home/robot/research/projects/wm_symbolic'); B=P/'data/core1200_v1'; R=P/'runs/core1200_v1'
COLORS=['red','blue','pink']; stop=False
def requested(sig,frame):
 global stop
 stop=True
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(path)
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def label(env):
 info=env.env.get_info(); result=calibrated_labels_at(info,env.config.scene); oracle=PhysicalOracle(env);held=oracle.held()
 for j,c in enumerate(COLORS):
  name='block_'+c
  if name in held:result['categorical'][j]=6;result['categorical_valid'][j]=True
  elif result['categorical'][j]==6:result['categorical'][j]=7;result['categorical_valid'][j]=False
 result['holding_derived']=[name in held for name in ['block_'+c for c in COLORS]]
 result['hand_empty_derived']=not bool(held) if all(result['categorical_valid'][:3]) else None
 result['held_definition']='bilateral force-bearing descendant finger contacts; no external support'
 return drawer_ownership(plain(result),info)
def hist_save(d,prefix,history):
 frames=list(history);frames=[frames[0]]*(4-len(frames))+frames
 for j,obs in enumerate(frames):save_npz(d/f'{prefix}_{j}.npz',obs)
def run_episode(root,branch,oracle):
 eid=f"{root['root_id']}_b{branch}";d=B/'episodes'/eid
 if d.exists():d.rename(d.with_name(eid+'_interrupted_'+str(int(time.time()))))
 d.mkdir(parents=True);env=CalvinAdapter(P,root['environment'],root['seed'],tabletop=True,segmentation=True)
 try:
  for obj in env.env.scene.movable_objects:
   val=root['layout'][obj.name];obj.initial_pos=np.array(val[:3]);obj.initial_orn=np.array(p.getQuaternionFromEuler(val[3:]))
  obs=env.reset()
  for _ in range(24):obs=env.step([obs['robot_obs'][:3],obs['robot_obs'][3:6],1])
  # Setup is physically executed and logged, never counted as requested actions.
  setup=[];setup_controls=[]
  def setup_record(obs,act):setup_controls.append(act)
  ctl=PhysicalOracle(env,setup_record)
  for skill in root['setup_skills']:
   with physics_guard(env):res=ctl.run(skill)
   setup.append({'skill':skill,**res})
  dump(d/'setup_private.json',{'results':setup,'controls':setup_controls,'planned_initial_state':root['initial_state'],'actual_initial_labels':label(env)})
  state=env.env.get_state_obs();np.savez_compressed(d/'root_reset_private.npz',**state)
  rootd=B/'roots'/root['root_id'];rootd.mkdir(parents=True,exist_ok=True)
  check={'branch':branch,'setup_all_successful':all(x['status']=='success' for x in setup)}
  ref=rootd/'branch0_initial_private.npz'
  if branch==0:np.savez_compressed(ref,**state)
  else:
   with np.load(ref) as z:
    check['scene_max_abs_difference']=float(np.max(np.abs(z['scene_obs']-state['scene_obs'])))
    check['robot_max_abs_difference']=float(np.max(np.abs(z['robot_obs']-state['robot_obs'])))
   if max(check['scene_max_abs_difference'],check['robot_max_abs_difference'])>1e-5:raise RuntimeError('Branches do not share the same physical root: '+str(check))
  dump(d/'root_replay_check.json',check)
  history=deque(list(env.history),maxlen=4);hist_save(d,'root_observation',history)
  rows=[]
  for step,skill in enumerate(root['branches'][branch]):
   s=d/f'{step:02d}_{skill}';s.mkdir();initial=env.observe();startinfo=env.env.get_info();before=label(env);startstate=env.env.get_state_obs();save_npz(s/'initial_public.npz',initial)
   frames={cam:imageio.get_writer(str(s/(cam+'.mp4')),fps=30,macro_block_size=1,codec='libx264',quality=7) for cam in initial['rgb_obs']}
   for cam,w in frames.items():w.append_data(initial['rgb_obs'][cam])
   actions=[];trace=[];startpos=np.array([startinfo['scene_info']['movable_objects']['block_'+c]['current_pos'] for c in COLORS]);maxdisp=np.zeros(3);first=[None]*3
   def record(obs,act):
    actions.append(act);history.append(obs)
    for cam,w in frames.items():w.append_data(obs['rgb_obs'][cam])
    positions=np.array([env.env.scene.get_info()['movable_objects']['block_'+c]['current_pos'] for c in COLORS]);dist=np.linalg.norm(positions-startpos,axis=1);maxdisp[:]=np.maximum(maxdisp,dist)
    for j in range(3):
     if first[j] is None and dist[j]>.02:first[j]=len(actions)
    contacts=[list(c) for c in p.getContactPoints(physicsClientId=env.env.cid) if c[9]>.01]
    trace.append({'robot_obs':obs['robot_obs'].tolist(),'scene_obs':env.env.scene.get_obs().tolist(),'objects':{o.name:o.get_state().tolist() for o in env.env.scene.movable_objects},'contacts':contacts})
   ctl=PhysicalOracle(env,record)
   try:
    with physics_guard(env):ctl.execute(skill)
    result=ctl.last_audit_private
   finally:
    for w in frames.values():w.close()
   final=env.observe();endstate=env.env.get_state_obs();after=label(env);save_npz(s/'final_public.npz',final)
   if not actions:
    assert np.array_equal(startstate['scene_obs'],endstate['scene_obs']) and np.array_equal(startstate['robot_obs'],endstate['robot_obs'])
    assert before==after,'Zero-action labels changed'
   hist_save(s,'boundary_history',history)
   official=skill in oracle.get_task_info_for_set(startinfo,env.env.get_info(),{skill})
   goal_success=bool(official and result['status']=='success')
   continuous=np.stack([startstate['scene_obs']]+[np.array(x['scene_obs']) for x in trace]);side=float(np.max(np.abs(continuous[:,[0,2,3,4,5]]-continuous[0,[0,2,3,4,5]])))
   result.update(skill=skill,official_success=official,goal_success=goal_success,zero_action_state_identical=not actions,continuous_unrequested_device_max_m=side,strict_device_side_effect=side>.003,unreviewed_incidental_motion=side>.003,control_samples_including_start=len(actions)+1,physics_guard_passed=True,controller_sha256=sha(P/'src/wm_symbolic/physical_oracle.py'))
   labels={'schema':'core_labels_v1','before':before,'after':after,'success':goal_success,'events':(maxdisp>.02).astype(int).tolist(),'events_valid':[True]*3,'event_max_displacement_m':maxdisp.tolist(),'event_first_control_step':first,'role':'offline_targets_only','control_samples_including_start':len(actions)+1}
   dump(s/'labels_private.json',labels);dump(s/'result_private.json',result);dump(s/'controls.json',actions)
   with gzip.open(s/'trace_private.json.gz','wt') as f:json.dump(trace,f)
   dump(s/'endpoint_info_private.json',plain({'before':startinfo,'after':env.env.get_info()}))
   row={'episode_id':eid,'root_id':root['root_id'],'split':root['split'],'dev_role':root['dev_role'],'environment':root['environment'],'step':step,'skill':skill,'executor':{'checkpoint':PhysicalOracle.VERSION,'budget_steps':900},'obs_before':str(s/'initial_public.npz'),'obs_after':str(s/'final_public.npz'),'labels_path':str(s/'labels_private.json'),'low_level_log':str(s/'controls.json'),'collection_style':'matched_order','controller_sha256':result['controller_sha256']}
   rows.append(row);dump(d/'committed_rows.json',rows)
   print(json.dumps({'episode':eid,'step':step,'controls':len(actions),'status':result['status'] if root['split']!='test' else 'sealed'}),flush=True)
  dump(d/'COMPLETE.json',{'rows':4,'root_id':root['root_id'],'controller_unchanged':True});return rows
 finally:env.close()
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--limit',type=int,default=1200);pa.add_argument('--splits',nargs='+',default=['train','dev']);args=pa.parse_args()
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
 for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sig,requested)
 protocol=json.loads((R/'protocol.json').read_text());assert sha(P/'src/wm_symbolic/physical_oracle.py')==protocol['controller']['controller_sha256']
 roots=json.loads((B/'roots_manifest.json').read_text());m=json.loads((B/'data_manifest.json').read_text());done={r['episode_id'] for r in m['records']};begin=time.monotonic()
 oracle=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
 # Interleave native A/B training roots, so the first128 probe contains both.
 roots.sort(key=lambda r:({'train':0,'dev':1,'test':2}[r['split']],int(r['root_id'].split('_')[-1]),r['environment']))
 for root in roots:
  if root['split'] not in args.splits:continue
  for b in range(3):
   eid=f"{root['root_id']}_b{b}"
   if eid in done:continue
   if len(m['records'])>=args.limit or stop:break
   rows=run_episode(root,b,oracle);m['records'].extend(rows);m['real_transitions']=len(m['records']);dump(B/'data_manifest.json',m)
   prog={'stage':'collecting','pid':os.getpid(),'transitions':len(m['records']),'target':1200,'split_counts':dict(Counter(x['split'] for x in m['records'])),'last_episode':eid,'elapsed_this_run_seconds':time.monotonic()-begin,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_pipeline_paused':True}
   dump(R/'progress.json',prog)
   if len(m['records'])>=50 and not (R/'storage_first50.json').exists():
    sizes=[]
    for rr in m['records'][:50]:sizes.append(sum(f.stat().st_size for f in Path(rr['obs_before']).parent.rglob('*') if f.is_file()))
    dump(R/'storage_first50.json',{'transitions':50,'bytes':sum(sizes),'bytes_per_transition':sum(sizes)/50,'projected1200_bytes':sum(sizes)*24,'includes':'control videos, endpoint/history RGBD+visible masks, actions, full private physical trace; excludes setup/root duplicate and DINO features'})
  if len(m['records'])>=args.limit or stop:break
 prog=json.loads((R/'progress.json').read_text());prog.update(stage='paused_at_episode_boundary' if stop else 'collection_requested_batch_complete',active_pid=None);dump(R/'progress.json',prog);print('BATCH_COMPLETE',len(m['records']),flush=True)
if __name__=='__main__':main()
