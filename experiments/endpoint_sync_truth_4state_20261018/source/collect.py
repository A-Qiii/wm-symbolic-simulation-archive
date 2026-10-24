from common import *
from geometry import classify
import time,sys,shutil,traceback,fcntl,os,platform,importlib.metadata,contextlib
import pybullet as pb
import hydra
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf

def snapshot(env):
 cid=env.cid;out=[]
 for k in range(pb.getNumBodies(physicsClientId=cid)):
  uid=pb.getBodyUniqueId(k,physicsClientId=cid);n=pb.getNumJoints(uid,physicsClientId=cid)
  out.append({'uid':uid,'body_info':pb.getBodyInfo(uid,physicsClientId=cid),'base_pose':pb.getBasePositionAndOrientation(uid,physicsClientId=cid),'base_velocity':pb.getBaseVelocity(uid,physicsClientId=cid),'joints':pb.getJointStates(uid,list(range(n)),physicsClientId=cid)if n else [],'link_states':[pb.getLinkState(uid,j,computeLinkVelocity=1,computeForwardKinematics=1,physicsClientId=cid)for j in range(n)]})
 return out
def serial_equal(a,b):return json.dumps(a,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.decode()if isinstance(x,bytes)else x,sort_keys=True)==json.dumps(b,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.decode()if isinstance(x,bytes)else x,sort_keys=True)
def record_target(mask,semantic_complete=True):
 h,w=mask.shape;b=bbox(mask)
 return {'visible_pixels':int(mask.sum()),'visible_bbox_xyxy':b,'bbox_valid':bool(b is not None and semantic_complete),'touches_image_border':bool(mask[0].any()or mask[-1].any()or mask[:,0].any()or mask[:,-1].any()),'visibility_state':'observed_visible_pixels'if mask.any()else'no_observed_pixels','occlusion_status':'unknown','out_of_frame_status':'unknown','zero_pixels_not_nonexistence':True}

def run_state(spec):
 s=spec['scene'];folder=E/'data'/s;priv=folder/'annotation_private';pub=folder/'public';priv.mkdir(parents=True);pub.mkdir()
 tstart=time.perf_counter();state={'ticks':0,'phase':'initialization','calls':[]};captured=[];env=None
 original_step=pb.stepSimulation;original_image=pb.getCameraImage
 def step(*args,**kwargs):
  if state['phase']!='initialization':raise RuntimeError('Physics tick prohibited after primary render')
  state['ticks']+=1
  if state['ticks']>8:raise RuntimeError('Initialization tick cap exceeded')
  return original_step(*args,**kwargs)
 def image_call(*args,**kwargs):
  if state['phase']=='initialization':state['phase']='primary';state['snapshot_before_primary']=snapshot(env)
  phase=state['phase'];assert phase in ['primary','baseline']
  if len(state['calls'])>=4:raise RuntimeError('Fixed camera-call cap exceeded')
  kwargs['flags']=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if phase=='primary'else 0
  t=time.perf_counter();r=original_image(*args,**kwargs);elapsed=time.perf_counter()-t
  w,h=int(r[0]),int(r[1]);rgb=np.asarray(r[2],dtype=np.uint8).reshape(h,w,4)[:,:,:3].copy();z=np.asarray(r[3],dtype=np.float32).reshape(h,w).copy();seg=np.asarray(r[4],dtype=np.int32).reshape(h,w).copy()
  meta={'call_index':len(state['calls']),'phase':phase,'tick':state['ticks'],'frame_id':spec['root_id'],'width':w,'height':h,'flags':kwargs['flags'],'renderer_argument':kwargs.get('renderer','omitted as in stock CALVIN; EGL plugin active'),'viewMatrix':list(kwargs['viewMatrix']),'projectionMatrix':list(kwargs['projectionMatrix']),'render_seconds':elapsed}
  state['calls'].append(meta);captured.append({'rgb':rgb,'zbuffer':z,'segmentation':seg,'meta':meta});return r
 pb.stepSimulation=step;pb.getCameraImage=image_call
 try:
  with initialize_config_dir(config_dir=str(P/'vendor/calvin/calvin_env/conf'),job_name='sync_truth_tech'):
   cfg=compose(config_name='config_data_collection',overrides=[f'scene=calvin_scene_{s}','cameras=static_and_gripper','use_vr=false',f'env.seed={spec["seed"]}','env.use_egl=true'])
  cfg.scene.objects.fixed_objects.table.joints.base__drawer.initial_state=spec['drawer_initial_q_m']
  for color,x in zip(['red','blue','pink'],[-.10,.10,.30]):
   cfg.scene.objects.movable_objects['block_'+color].initial_pos=[x,-.08,.46];cfg.scene.objects.movable_objects['block_'+color].initial_orn=[0.,0.,0.]
  OmegaConf.save(cfg,priv/'resolved_config.yaml');env=hydra.utils.instantiate(cfg.env,use_scene_info=True);pb.setRealTimeSimulation(0,physicsClientId=env.cid)
  rgbd=env.reset();assert len(captured)==2 and state['phase']=='primary';assert [cam.name for cam in env.cameras]==['static','gripper'];initialization_seconds=time.perf_counter()-tstart
  state['phase']='frozen';before=state.pop('snapshot_before_primary');afterprimary=snapshot(env);assert serial_equal(before,afterprimary)
  names={};bodymap=[]
  for k in range(pb.getNumBodies(physicsClientId=env.cid)):
   uid=pb.getBodyUniqueId(k,physicsClientId=env.cid);joints=[pb.getJointInfo(uid,j,physicsClientId=env.cid)for j in range(pb.getNumJoints(uid,physicsClientId=env.cid))]
   bodymap.append({'uid':uid,'body_info':pb.getBodyInfo(uid,physicsClientId=env.cid),'joints':[{'index':j[0],'joint_name':j[1],'type':j[2],'child_link_name':j[12],'parent_index':j[16]}for j in joints],'visual_shapes':pb.getVisualShapeData(uid,physicsClientId=env.cid),'collision_shapes_by_link':{str(j):pb.getCollisionShapeData(uid,j,physicsClientId=env.cid)for j in range(-1,len(joints))}})
   names[uid]={j[12].decode():j[0]for j in joints}
  robot=env.robot.robot_uid;table=next(o.uid for o in env.scene.fixed_objects if o.name=='table');drawer=names[table]['drawer_link'];families=[[names[robot][a],names[robot][b]]for a,b in [('panda_leftfinger','finger_left_tip'),('panda_rightfinger','finger_right_tip')]]
  save(priv/'body_link_mapping.json',{'all_bodies':bodymap,'target_mapping':{'robot_uid':robot,'table_uid':table,'drawer_link':drawer,'finger_families':families}})
  save(priv/'snapshot_before.json',before);save(priv/'native_serialization.json',env.serialize());save(priv/'native_state_obs.json',env.get_state_obs());save(priv/'physics_parameters.json',pb.getPhysicsEngineParameters(physicsClientId=env.cid))
  v,f=mesh(P/f'vendor/calvin/calvin_env/data/calvin_table_{s}/meshes/drawer_link.obj');face_classes=np.asarray(read(E/'configs/mesh_semantics.json')[s]['class_by_triangle_index'],np.uint8)
  ls=pb.getLinkState(table,drawer,computeForwardKinematics=1,physicsClientId=env.cid);rot=np.array(pb.getMatrixFromQuaternion(ls[5])).reshape(3,3);scale=float(cfg.scene.global_scaling);world=v*scale@rot.T+np.asarray(ls[4]);tri=world[f]
  save(priv/'drawer_visual_transform.json',{'world_link_position':ls[4],'world_link_quaternion':ls[5],'mesh_global_scale':scale,'visual_origin_xyz':[0,0,0],'visual_origin_rpy':[0,0,0],'source_visual_mesh':str(P/f'vendor/calvin/calvin_env/data/calvin_table_{s}/meshes/drawer_link.obj')})
  np.savez_compressed(priv/'drawer_world_mesh.npz',vertices=world,faces=f,semantic_class_by_face=face_classes)
  summaries=[];offline_start=time.perf_counter()
  for cam,cap in zip(env.cameras,captured[:2]):
   name=cam.name;near=cam.nearval;far=cam.farval;metric=far*near/(far-(far-near)*cap['zbuffer']);assert np.array_equal(cap['rgb'],rgbd['rgb_obs']['rgb_'+name]);assert np.array_equal(metric,rgbd['depth_obs']['depth_'+name])
   np.savez_compressed(priv/f'primary_{name}.npz',rgb=cap['rgb'],depth_buffer=cap['zbuffer'],depth_m=metric,object_link_mask=cap['segmentation'])
   cap['meta'].update(camera=name,near=near,far=far,fov=cam.fov,aspect=cam.aspect);save(priv/f'camera_{name}.json',cap['meta'])
   sem,face_idx,err,stat=classify(cap['segmentation'],cap['zbuffer'],cap['meta']['viewMatrix'],cap['meta']['projectionMatrix'],tri,face_classes,table,drawer,near,far)
   seg=cap['segmentation'];body=seg&((1<<24)-1);link=(seg>>24)-1;fingers=[(seg>=0)&(body==robot)&np.isin(link,fam)for fam in families];whole=(seg>=0)&(body==table)&(link==drawer);front=np.isin(sem,[1,2]);unknown=sem==255
   assert np.all((sem>0)==whole)and not np.any(front&~whole)and np.all(~(fingers[0]&fingers[1]));assert int(front.sum()+(sem==3).sum()+unknown.sum())==int(whole.sum())
   np.savez_compressed(priv/f'semantics_{name}.npz',drawer_parts=sem,drawer_face_id=face_idx,depth_error_m=err,finger1=fingers[0],finger2=fingers[1],whole_drawer=whole,front_plus_handle=front,unknown_drawer_pixels=unknown)
   labels={'finger1':record_target(fingers[0]),'finger2':record_target(fingers[1]),'whole_drawer_diagnostic':record_target(whole),'drawer_front_plus_handle':record_target(front,not unknown.any())};save(priv/f'visible_labels_{name}.json',labels)
   summaries.append({'scene':s,'camera':name,'frame_id':spec['root_id'],'tick':state['ticks'],'semantic':stat,'labels':labels})
  offline_seconds=time.perf_counter()-offline_start
  np.savez_compressed(pub/'rgb.npz',rgb_static=captured[0]['rgb'],rgb_gripper=captured[1]['rgb'])
  # No edits were made; flag-off same-state baseline additionally checks that
  # enabling object/link ID output itself leaves stock RGB/depth unchanged.
  state['phase']='baseline'
  for cam in env.cameras:cam.render()
  assert len(captured)==4
  checks=[]
  for cam,a,b in zip(env.cameras,captured[:2],captured[2:]):
   checks.append({'camera':cam.name,'RGB_bitwise_equal':bool(np.array_equal(a['rgb'],b['rgb'])),'depth_buffer_bitwise_equal':bool(np.array_equal(a['zbuffer'],b['zbuffer'])),'RGB_max_abs_difference':int(np.abs(a['rgb'].astype(int)-b['rgb'].astype(int)).max()),'depth_max_abs_difference':float(np.abs(a['zbuffer']-b['zbuffer']).max())})
   np.savez_compressed(priv/f'baseline_flag_off_{cam.name}.npz',rgb=b['rgb'],depth_buffer=b['zbuffer'])
  after=snapshot(env);save(priv/'snapshot_after.json',after);assert serial_equal(before,after);assert all(x['RGB_bitwise_equal']and x['depth_buffer_bitwise_equal']for x in checks)
  save(priv/'render_call_ledger.json',state['calls']);save(priv/'integrity.json',{'before_after_full_snapshot_equal':True,'same_tick_all4calls':len({x['tick']for x in state['calls']})==1,'init_physics_ticks':state['ticks'],'post_primary_physics_ticks':0,'flag_off_checks':checks,'no_visual_or_physics_mutation_after_primary':True})
  return {'spec':spec,'summaries':summaries,'initialization_and_primary_seconds':initialization_seconds,'offline_semantic_seconds':offline_seconds,'render_calls':len(state['calls']),'primary_records':2,'init_physics_ticks':state['ticks'],'post_primary_physics_ticks':0,'state_invariant':True,'RGB_depth_bitwise_unchanged':True,'elapsed_seconds_before_close':time.perf_counter()-tstart,'directory':str(folder),'data_bytes_before_summary':sum(p.stat().st_size for p in folder.rglob('*')if p.is_file())}
 finally:
  pb.stepSimulation=original_step;pb.getCameraImage=original_image
  if env is not None and env.ownsPhysicsClient:env.close();env.ownsPhysicsClient=False

def main():
 protocol=read(E/'configs/protocol.json');frozen=read(E/'configs/pre_run_frozen.json');assert all(sha(p)==h for p,h in frozen['sha256'].items())
 assert not(E/'results/attempts.json').exists(),'No automatic repeat'
 lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(E/'configs/runtime.json',{'python':sys.version,'platform':platform.platform(),'pybullet_version':importlib.metadata.version('pybullet'),'pybullet_API':pb.getAPIVersion(),'numpy':np.__version__,'renderer':'Native CALVIN EGL path on local GPU, no cloud, no fallback renderer','pid':os.getpid(),'utc':now()})
 attempts=[];results=[];start=time.perf_counter()
 try:
  for spec in protocol['fixed_states']:
   attempts.append({'scene':spec['scene'],'status':'started','utc':now()});save(E/'results/attempts.json',attempts)
   try:
    result=run_state(spec);results.append(result);attempts[-1]['status']='completed';save(E/'results/results.json',results);save(E/'results/attempts.json',attempts);print('STATE_COMPLETE',spec['scene'],[(r['camera'],r['semantic']['unknown_pixels'])for r in result['summaries']],flush=True)
   except Exception:
    attempts[-1].update(status='failed',error=traceback.format_exc());save(E/'results/attempts.json',attempts);raise
  save(E/'results/collection_summary.json',{'utc':now(),'states':len(results),'primary_camera_records':sum(r['primary_records']for r in results),'camera_calls_including_baseline':sum(r['render_calls']for r in results),'initialization_physics_ticks':sum(r['init_physics_ticks']for r in results),'post_primary_physics_ticks':0,'elapsed_seconds_including_close':time.perf_counter()-start,'training_updates':0,'model_loads':0,'new_skill_calls':0,'semantic_complete_records':sum(s['semantic']['all_drawer_pixels_classified']for r in results for s in r['summaries'])})
 finally:lock.close()
if __name__=='__main__':main()
