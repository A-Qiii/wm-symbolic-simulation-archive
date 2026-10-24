from common import *
import labels_physical as truth
import time,sys,traceback,fcntl,platform,importlib.metadata
from collections import deque
import pybullet as pb
import hydra
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from wm_symbolic.physical_oracle_v2 import PhysicalOracleV2
from wm_symbolic.physical_oracle import ExecutionFailure
sys.path.insert(0,str(P/'scripts'))
from evaluate_physical_oracle_v2 import physics_guard

def full_snapshot(env):
 out=[]
 for k in range(pb.getNumBodies(physicsClientId=env.cid)):
  u=pb.getBodyUniqueId(k,physicsClientId=env.cid);n=pb.getNumJoints(u,physicsClientId=env.cid)
  out.append({'uid':u,'pose':pb.getBasePositionAndOrientation(u,physicsClientId=env.cid),'velocity':pb.getBaseVelocity(u,physicsClientId=env.cid),'joints':pb.getJointStates(u,list(range(n)),physicsClientId=env.cid)if n else [],'links':[pb.getLinkState(u,j,computeLinkVelocity=1,computeForwardKinematics=1,physicsClientId=env.cid)for j in range(n)]})
 return out
def same(a,b):return json.dumps(a,default=json_default,sort_keys=True)==json.dumps(b,default=json_default,sort_keys=True)

class MotorAdapter:
 def __init__(self,env):self.env=env;self.controls=0;self.last_grip=1
 def observe(self):return {'robot_obs':self.env.get_state_obs()['robot_obs']}
 def step(self,action):
  if self.controls>=900:raise ExecutionFailure('whole_root_control_budget_exhausted')
  self.env.robot.apply_action(action)
  for _ in range(self.env.action_repeat):pb.stepSimulation(physicsClientId=self.env.cid)
  self.env.scene.step();self.controls+=1;self.last_grip=int(action[2]);return self.observe()

def force_labels(ctl):
 out=[]
 for cutoff in [.005,.01,.015]:
  held=[]
  for c in COLORS:
   cs=[x for x in ctl.contacts('block_'+c)if x[9]>cutoff];links={x[4]for x in cs if x[2]==ctl.robot.robot_uid}
   held.append(bool(all(links&fam for fam in ctl.finger_families)and not any(x[2]!=ctl.robot.robot_uid for x in cs)))
  out.append(held)
 return out

def run_root(spec):
 rid=spec['root_id'];d=E/'data/private'/rid;d.mkdir();env=ctl=None;rows=[];checks=[];hist=deque(maxlen=12);captured=set();orig_image=pb.getCameraImage;orig_step=pb.stepSimulation;mode='initialize';caps=[];ticks=0;tracefile=(d/'control_trace_private.jsonl').open('w');stage='initial_settle'
 def step(*a,**kw):
  nonlocal ticks
  assert mode in ['initialize','move'],'physics advanced within capture';ticks+=1;return orig_step(*a,**kw)
 def image(*a,**kw):
  assert mode in ['primary','baseline'],'intermediate RGB prohibited';kw['flags']=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if mode=='primary'else 0
  out=orig_image(*a,**kw);w,h=map(int,out[:2]);caps.append({'rgb':np.asarray(out[2],np.uint8).reshape(h,w,4)[:,:,:3].copy(),'depth':np.asarray(out[3],np.float32).reshape(h,w).copy(),'mask':np.asarray(out[4],np.int32).reshape(h,w).copy(),'meta':{'tick':ticks,'mode':mode,'viewMatrix':kw['viewMatrix'],'projectionMatrix':kw['projectionMatrix'],'width':w,'height':h,'flags':kw['flags']}});return out
 pb.getCameraImage=image;pb.stepSimulation=step
 try:
  with initialize_config_dir(config_dir=str(P/'vendor/calvin/calvin_env/conf'),job_name='relation32'):
   cfg=compose(config_name='config_data_collection',overrides=[f'scene=calvin_scene_{spec["scene"]}','cameras=static_and_gripper','use_vr=false',f'env.seed={spec["seed"]}','env.use_egl=true'])
  cfg.scene.objects.fixed_objects.table.joints.base__drawer.initial_state=spec['drawer_initial_q_m']
  for c,b in spec['blocks'].items():cfg.scene.objects.movable_objects['block_'+c].initial_pos=b['initial_pos'];cfg.scene.objects.movable_objects['block_'+c].initial_orn=b['initial_orn']
  OmegaConf.save(cfg,d/'resolved_config.yaml');env=hydra.utils.instantiate(cfg.env,use_scene_info=True);pb.setRealTimeSimulation(0,physicsClientId=env.cid)
  getobs=env.get_obs;env.get_obs=env.get_state_obs
  try:env.reset()
  finally:env.get_obs=getobs
  adapter=MotorAdapter(env)
  def record(obs,act):
   snap=truth.snapshot(adapter,ctl);tracefile.write(json.dumps({'control':adapter.controls,'tick':ticks,'stage':stage,'action':act,'physical':snap,'scene_obs':env.scene.get_obs(),'robot_obs':obs['robot_obs']},default=json_default)+'\n');hist.append({'control':adapter.controls,'tick':ticks,'physical':snap,'holding_at_cutoffs':force_labels(ctl)})
  ctl=PhysicalOracleV2(adapter,record,max_controls=900);robot=env.robot.robot_uid;families=[sorted(f)for f in ctl.finger_families];mapping={'robot_uid':robot,'object_uids':{c:ctl.objects['block_'+c].uid for c in COLORS},'finger_families':families,'robot_links':{pb.getJointInfo(robot,j,physicsClientId=env.cid)[12].decode():j for j in range(pb.getNumJoints(robot,physicsClientId=env.cid))}}
  assert families==[sorted([mapping['robot_links'][a],mapping['robot_links'][b]])for a,b in [('panda_leftfinger','finger_left_tip'),('panda_rightfinger','finger_right_tip')]]
  save(d/'body_link_mapping.json',mapping);save(d/'physics_parameters.json',pb.getPhysicsEngineParameters(physicsClientId=env.cid))
  def capture(anchor):
   nonlocal mode,caps
   if anchor in captured:return
   assert len(captured)<8;idx=protocol['anchors'].index(anchor);sd=d/f's{idx:02d}';sd.mkdir();caps=[];before=full_snapshot(env);physical=truth.snapshot(adapter,ctl);lab=truth.evaluate(physical);cutoffs=force_labels(ctl);assert len(hist)==12
   assert list(lab['y'][:3])==cutoffs[1]
   stable=[all(all(bool(h['holding_at_cutoffs'][f][j])==bool(lab['y'][j])for f in range(3))and all(sum(z)<=1 for z in h['holding_at_cutoffs'])for h in hist)for j in range(3)]
   mode='primary';env.get_obs();assert len(caps)==2;mode='baseline'
   for cam in env.cameras:cam.render()
   mode='frozen';assert len(caps)==4;after=full_snapshot(env);physical_after=truth.snapshot(adapter,ctl);assert same(before,after)and same(physical,physical_after)
   for a,b in zip(caps[:2],caps[2:]):assert np.array_equal(a['rgb'],b['rgb'])and np.array_equal(a['depth'],b['depth'])
   save(sd/'snapshot_before.json',before);save(sd/'snapshot_after.json',after);save(sd/'physical_truth.json',{'physical':physical,'labels_v2_1':lab,'holding_stability_valid':stable,'holding_cutoffs':cutoffs,'last12_controls':list(hist),'In_visual_observability':'not established: visible drawer front is not evidence for full cavity containment/support'})
   pub=E/'data/public'/f'{rid}_s{idx:02d}.npz';save_raw_npz(pub,rgb_static=caps[0]['rgb'],rgb_gripper=caps[1]['rgb'])
   for ci,cam in enumerate(env.cameras):
    cap=caps[ci];seg=cap['mask'];body=seg&((1<<24)-1);link=(seg>>24)-1;fingers=[(seg>=0)&(body==robot)&np.isin(link,f)for f in families];union=fingers[0]|fingers[1];gb=bbox(union);fc=[int(f.sum())for f in fingers];h,w=seg.shape
    save_raw_npz(sd/f'native_{cam.name}.npz',rgb=cap['rgb'],depth_buffer=cap['depth'],object_link_mask=seg);save(sd/f'camera_{cam.name}.json',cap['meta']|{'camera':cam.name,'near':cam.nearval,'far':cam.farval});save(sd/f'baseline_{cam.name}.json',caps[ci+2]['meta']|{'RGB_depth_bitwise_equal':True})
    for j,c in enumerate(COLORS):
     bm=(seg>=0)&(body==mapping['object_uids'][c]);bc=int(bm.sum());visible=bc>=4 and all(n>=4 for n in fc);eligible=visible and stable[j]
     rows.append({'record_id':f'{rid}_s{idx:02d}_{cam.name}_{c}','root_id':rid,'scene':spec['scene'],'split':spec['split'],'sample':idx,'anchor':anchor,'camera':cam.name,'color':c,'target_color':spec['target_color'],'path':str(pub),'key':'rgb_'+cam.name,'width':w,'height':h,'RGB_sha256':hashlib.sha256(cap['rgb'].tobytes()).hexdigest(),'object_bbox':bbox(bm),'gripper_bbox':gb,'object_pixels':bc,'finger_pixels':fc,'y':bool(lab['y'][j]),'physical_valid':stable[j],'visually_eligible':visible,'eligible':eligible,'unknown_reasons':([]if stable[j]else['force_or_temporal_ambiguity'])+([]if bc>=4 else['block_lt4_visible_pixels'])+([]if all(n>=4 for n in fc)else['finger_family_lt4_visible_pixels']),'In_y':bool(lab['y'][j+3]),'In_physical_valid':bool(lab['v'][j+3]),'In_visual_eligible':False})
   check={'root_id':rid,'sample':idx,'anchor':anchor,'tick':ticks,'control':adapter.controls,'four_same_tick_calls':len({x['meta']['tick']for x in caps})==1,'full_snapshot_and_contacts_equal':True,'RGB_depth_bitwise_equal':True,'physical_valid':stable,'holding':lab['y'][:3],'public_path':str(pub)};checks.append(check);save(sd/'integrity.json',check);captured.add(anchor);save(d/'records.json',rows);mode='move'
  origmove=ctl.move;orighold=ctl.hold
  def wrapped_move(xyz,orn,grip,phase,*a,**kw):
   out=origmove(xyz,orn,grip,phase,*a,**kw)
   if phase in {'above_named_block','descend_named_block','lift_named_block'}:capture({'above_named_block':'above_named_block','descend_named_block':'descend_open_near_block','lift_named_block':'lifted'}[phase])
   return out
  def wrapped_hold(n,grip):
   out=orighold(n,grip)
   if n==20 and grip==-1 and ctl.phase=='descend_named_block':capture('closed_on_support')
   return out
  ctl.move=wrapped_move;ctl.hold=wrapped_hold;mode='move';result={}
  with physics_guard(adapter,ctl):
   orighold(24,1);capture('initial_settled');initial_target=ctl.xyz('block_'+spec['target_color']).copy()
   stage='frozen_grasp';result=ctl.run('lift_'+spec['target_color']+'_block_table');stage='post_grasp_anchors';save(d/'frozen_controller_result.json',result)
   try:
    ctl.phase='post_run_hold';orighold(12,adapter.last_grip);capture('post_run_stable')
    orn=ctl.pose()[3:];origmove([initial_target[0]+spec['post_lift_translate_x_m'],initial_target[1],.60],orn,-1,'fixed_translate_after_run',settle=12);capture('translated_closed')
    origmove(initial_target+[0,0,.003],orn,-1,'fixed_return_to_table',speed=.0025,settle=12);orighold(24,1);capture('released_near_table')
   except ExecutionFailure as ex:result['post_anchor_failure']=str(ex)
  summary={'root_id':rid,'controller_result':result,'captured_anchors':sorted(captured,key=protocol['anchors'].index),'missing_anchors':[a for a in protocol['anchors']if a not in captured],'controls':adapter.controls,'ticks':ticks,'states':len(captured),'physics_guard_passed':True};save(d/'root_summary.json',summary);return rows,checks,summary
 finally:
  pb.getCameraImage=orig_image;pb.stepSimulation=orig_step;tracefile.close()
  if ctl is not None:ctl.close()
  if env is not None and env.ownsPhysicsClient:env.close();env.ownsPhysicsClient=False

def main():
 global protocol
 install_guard([]);protocol=read(E/'configs/protocol.json');assert all(sha(p)==h for p,h in read(E/'configs/precollection_frozen.json')['sha256'].items());assert not(E/'results/attempts.json').exists();assert len(protocol['roots'])==8
 lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.perf_counter();rows=[];checks=[];attempts=[];summaries=[]
 save(E/'configs/runtime.json',{'utc':now(),'python':sys.version,'platform':platform.platform(),'pybullet':importlib.metadata.version('pybullet'),'pid':os.getpid(),'renderer':'native CALVIN EGL','new_downloads':0})
 try:
  for spec in protocol['roots']:
   attempts.append({'root_id':spec['root_id'],'status':'started','utc':now()});save(E/'results/attempts.json',attempts)
   try:
    rr,cc,ss=run_root(spec);rows+=rr;checks+=cc;summaries.append(ss);assert len(checks)<=64
    attempts[-1]['status']='completed_with_actual_outcomes';save(E/'data/records.json',rows);save(E/'results/integrity.json',checks);save(E/'results/root_summaries.json',summaries);save(E/'results/attempts.json',attempts)
    for boundary in [50]:
     fn=E/f'results/storage_first{boundary}.json'
     if len(checks)>=boundary and not fn.exists():
      selected=checks[:boundary];private=sum(p.stat().st_size for c in selected for p in (E/'data/private'/c['root_id']/f's{c["sample"]:02d}').rglob('*')if p.is_file());public=sum(Path(c['public_path']).stat().st_size for c in selected);save(fn,{'states':boundary,'private_bytes':private,'public_RGB_bytes':public,'bytes_per_state':(private+public)/boundary,'projected_64_state_bytes':64*(private+public)/boundary,'excludes':'shared configs/root metadata; raw baseline renders verified in memory, their call metadata saved'})
    print('ROOT',spec['root_id'],'states',ss['states'],'status',ss['controller_result']['status'],'reason',ss['controller_result']['reason'],'eligible_pos',sum(r['eligible']and r['y']for r in rr),'eligible_neg',sum(r['eligible']and not r['y']for r in rr),flush=True)
   except Exception:
    attempts[-1].update(status='infrastructure_or_integrity_failure',traceback=traceback.format_exc());save(E/'results/attempts.json',attempts);raise
  save(E/'results/collection_summary.json',{'utc':now(),'roots':len(summaries),'states':len(checks),'primary_images':2*len(checks),'camera_calls':4*len(checks),'records':len(rows),'seconds':time.perf_counter()-start,'physics_ticks':sum(s['ticks']for s in summaries),'controls':sum(s['controls']for s in summaries),'all_integrity_pass':True,'fits':0,'data_bytes':sum(p.stat().st_size for p in(E/'data').rglob('*')if p.is_file())})
 finally:lock.close()
if __name__=='__main__':main()
