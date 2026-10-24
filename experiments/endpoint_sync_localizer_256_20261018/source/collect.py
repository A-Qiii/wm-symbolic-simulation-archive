from common import *
from geometry import classify
import time,sys,traceback,fcntl,os,platform,importlib.metadata
import pybullet as pb
import hydra
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf

def snapshot(env):
 out=[]
 for k in range(pb.getNumBodies(physicsClientId=env.cid)):
  u=pb.getBodyUniqueId(k,physicsClientId=env.cid);n=pb.getNumJoints(u,physicsClientId=env.cid)
  out.append({'uid':u,'info':pb.getBodyInfo(u,physicsClientId=env.cid),'pose':pb.getBasePositionAndOrientation(u,physicsClientId=env.cid),'velocity':pb.getBaseVelocity(u,physicsClientId=env.cid),'joints':pb.getJointStates(u,list(range(n)),physicsClientId=env.cid)if n else [],'links':[pb.getLinkState(u,j,computeLinkVelocity=1,computeForwardKinematics=1,physicsClientId=env.cid)for j in range(n)]})
 return out
def same(a,b):return json.dumps(a,default=json_default,sort_keys=True)==json.dumps(b,default=json_default,sort_keys=True)
def target(m,complete=True):
 return {'visible_pixels':int(m.sum()),'bbox':bbox(m),'bbox_valid':bool(m.any()and complete),'border_pixels':bool(m[0].any()or m[-1].any()or m[:,0].any()or m[:,-1].any()),'visibility':'observed_visible'if m.any()else'zero_pixels_cause_unknown','occlusion':'unknown','out_of_view':'unknown','nonexistence_inferred':False}

def run_root(spec):
 root=spec['root_id'];privroot=E/'data/annotation_private'/root;privroot.mkdir();env=None;original_step=pb.stepSimulation;original_image=pb.getCameraImage;state={'ticks':0,'mode':'initialize','captures':[],'sample':0};rootrows=[];anns=[];checks=[]
 def step(*a,**kw):
  assert state['mode']in['initialize','move'],'No physics within synchronous capture'
  state['ticks']+=1;return original_step(*a,**kw)
 def image_call(*a,**kw):
  if state['mode']=='initialize':state['before']=snapshot(env);state['mode']='primary'
  assert state['mode']in['primary','baseline']
  kw['flags']=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if state['mode']=='primary'else 0
  out=original_image(*a,**kw);w,h=map(int,out[:2]);rgb=np.asarray(out[2],np.uint8).reshape(h,w,4)[:,:,:3].copy();z=np.asarray(out[3],np.float32).reshape(h,w).copy();seg=np.asarray(out[4],np.int32).reshape(h,w).copy();j=len(state['captures']);folder=privroot/f's{state["sample"]:02d}';folder.mkdir(exist_ok=True)
  meta={'root_id':root,'sample':state['sample'],'tick':state['ticks'],'camera_index':j%2,'call_index':j,'phase':state['mode'],'width':w,'height':h,'viewMatrix':list(kw['viewMatrix']),'projectionMatrix':list(kw['projectionMatrix']),'flags':kw['flags']}
  save_raw_npz(folder/f'call{j:02d}.npz',rgb=rgb,depth_buffer=z,object_link_mask=seg);save(folder/f'call{j:02d}.json',meta);state['captures'].append({'rgb':rgb,'z':z,'seg':seg,'meta':meta});return out
 pb.stepSimulation=step;pb.getCameraImage=image_call
 try:
  with initialize_config_dir(config_dir=str(P/'vendor/calvin/calvin_env/conf'),job_name='sync256'):
   cfg=compose(config_name='config_data_collection',overrides=[f'scene=calvin_scene_{spec["scene"]}','cameras=static_and_gripper','use_vr=false',f'env.seed={spec["seed"]}','env.use_egl=true'])
  cfg.scene.objects.fixed_objects.table.joints.base__drawer.initial_state=spec['drawer_initial_q_m']
  for c,b in spec['blocks'].items():
   cfg.scene.objects.movable_objects['block_'+c].initial_pos=b['initial_pos'];cfg.scene.objects.movable_objects['block_'+c].initial_orn=b['initial_orn']
  OmegaConf.save(cfg,privroot/'resolved_config.yaml');env=hydra.utils.instantiate(cfg.env,use_scene_info=True);pb.setRealTimeSimulation(0,physicsClientId=env.cid);env.reset();assert len(state['captures'])==2
  names={};mapping=[]
  for k in range(pb.getNumBodies(physicsClientId=env.cid)):
   uid=pb.getBodyUniqueId(k,physicsClientId=env.cid);joints=[pb.getJointInfo(uid,j,physicsClientId=env.cid)for j in range(pb.getNumJoints(uid,physicsClientId=env.cid))];names[uid]={j[12].decode():j[0]for j in joints};mapping.append({'uid':uid,'body_info':pb.getBodyInfo(uid,physicsClientId=env.cid),'joints':joints})
  robot=env.robot.robot_uid;table=next(o.uid for o in env.scene.fixed_objects if o.name=='table');drawer=names[table]['drawer_link'];families=[[names[robot][a],names[robot][b]]for a,b in [('panda_leftfinger','finger_left_tip'),('panda_rightfinger','finger_right_tip')]]
  save(privroot/'body_link_mapping.json',{'bodies':mapping,'robot':robot,'table':table,'drawer':drawer,'finger_families':families});save(privroot/'physics_parameters.json',pb.getPhysicsEngineParameters(physicsClientId=env.cid))
  vertices,faces=mesh(P/f'vendor/calvin/calvin_env/data/calvin_table_{spec["scene"]}/meshes/drawer_link.obj');classes=np.asarray(read(E/'configs/mesh_semantics.json')[spec['scene']]['class_by_triangle_index'],np.uint8)
  for sample in range(8):
   state['sample']=sample
   if sample:
    state['mode']='move';state['captures']=[];wp=spec['waypoints'][sample-1]
    for period in range(wp['control_periods']):
     env.robot.apply_action([wp['xyz'],wp['euler'],wp['grip']])
     for _ in range(wp['physics_ticks_per_period']):pb.stepSimulation(physicsClientId=env.cid)
     env.scene.step()
    state['before']=snapshot(env);state['mode']='primary';env.get_obs()
   assert len(state['captures'])==2;state['mode']='baseline'
   for cam in env.cameras:cam.render()
   state['mode']='frozen';after=snapshot(env);before=state.pop('before');assert same(before,after);caps=state['captures'];assert len(caps)==4
   for a,b in zip(caps[:2],caps[2:]):assert np.array_equal(a['rgb'],b['rgb'])and np.array_equal(a['z'],b['z'])
   folder=privroot/f's{sample:02d}';save(folder/'snapshot_before.json',before);save(folder/'snapshot_after.json',after);save(folder/'native_serialization.json',env.serialize());obs=env.get_state_obs();save(folder/'native_state_obs.json',obs)
   pub=E/'data/public'/f'{root}_s{sample:02d}.npz';save_raw_npz(pub,rgb_static=caps[0]['rgb'],rgb_gripper=caps[1]['rgb']);ls=pb.getLinkState(table,drawer,computeForwardKinematics=1,physicsClientId=env.cid);rot=np.array(pb.getMatrixFromQuaternion(ls[5])).reshape(3,3);world=vertices*float(cfg.scene.global_scaling)@rot.T+np.asarray(ls[4]);tri=world[faces];save_raw_npz(folder/'drawer_world_mesh.npz',vertices=world,faces=faces,semantic_class_by_face=classes)
   save(folder/'drawer_visual_transform.json',{'link_world_position':ls[4],'link_world_quaternion':ls[5],'mesh_global_scale':float(cfg.scene.global_scaling),'visual_origin_xyz':[0,0,0],'visual_origin_rpy':[0,0,0]})
   for cam,cap in zip(env.cameras,caps[:2]):
    h,w=cap['seg'].shape;meta=cap['meta'];meta.update(camera=cam.name,near=cam.nearval,far=cam.farval,fov=cam.fov,aspect=cam.aspect);save(folder/f'camera_{cam.name}.json',meta);metric=cam.farval*cam.nearval/(cam.farval-(cam.farval-cam.nearval)*cap['z']);save_raw_npz(folder/f'depth_{cam.name}.npz',depth_m=metric)
    sem,face_idx,err,stats=classify(cap['seg'],cap['z'],meta['viewMatrix'],meta['projectionMatrix'],tri,classes,table,drawer,cam.nearval,cam.farval);seg=cap['seg'];body=seg&((1<<24)-1);link=(seg>>24)-1;fingers=[(seg>=0)&(body==robot)&np.isin(link,fam)for fam in families];whole=(seg>=0)&(body==table)&(link==drawer);front=np.isin(sem,[1,2]);unknown=sem==255
    assert np.all((sem>0)==whole)and not np.any(fingers[0]&fingers[1]);save_raw_npz(folder/f'semantics_{cam.name}.npz',drawer_parts=sem,drawer_face_id=face_idx,depth_error_m=err,finger1=fingers[0],finger2=fingers[1],whole_drawer=whole,front_plus_handle=front,unknown_drawer_pixels=unknown)
    labels={'drawer_front':target(front,not unknown.any()),'finger1':target(fingers[0]),'finger2':target(fingers[1])};valid=[labels['drawer_front']['bbox_valid'],all(labels[f'finger{j}']['bbox_valid']for j in [1,2])]
    bb=[[labels['drawer_front']['bbox']]if valid[0]else[],[labels[f'finger{j}']['bbox']for j in [1,2]]if valid[1]else[]]
    def patches(mask):
     yy,xx=np.where(mask);return set(((yy*GRID//h)*GRID+xx*GRID//w).tolist())
    hp=patches(sem==2);fp=patches(fingers[0]|fingers[1]);positive_patches=set()
    for box in bb[1]:
     xy=((np.array(box[:2])+box[2:])/2/[w,h]*GRID).astype(int);positive_patches.add(int(xy[1]*GRID+xy[0]))
    hard=sorted(hp-fp-positive_patches)if valid[1]else[]
    image_id=len(rootrows)+len(read(E/'data/public/images.json'))if(E/'data/public/images.json').exists()else len(rootrows)
    row={'image_id':image_id,'root_id':root,'scene':spec['scene'],'split':spec['split'],'sample':sample,'camera':cam.name,'path':str(pub),'key':'rgb_'+cam.name,'width':w,'height':h,'RGB_sha256':hashlib.sha256(cap['rgb'].tobytes()).hexdigest()};rootrows.append(row)
    ann={'image_id':image_id,'root_id':root,'sample':sample,'camera':cam.name,'class_valid':valid,'boxes':bb,'targets':labels,'semantic':stats,'handle_hardnegative_patches':hard,'semantic_file':str(folder/f'semantics_{cam.name}.npz'),'class_unknown_reasons':['unknown_drawer_pixels'if unknown.any()else('zero_visible_front'if not front.any()else None),'one_or_more_finger_families_zero_pixels'if not valid[1]else None]};anns.append(ann);save(folder/f'labels_{cam.name}.json',ann)
   integrity={'root_id':root,'sample':sample,'tick':state['ticks'],'same_tick_4calls':len({c['meta']['tick']for c in caps})==1,'full_snapshot_equal':True,'RGB_depth_bitwise_equal':True,'post_primary_ticks':0,'tcp_xyz':obs['robot_obs'][:3],'target_error_m':float(np.linalg.norm(np.array(obs['robot_obs'][:3])-spec['waypoints'][sample-1]['xyz']))if sample else None};checks.append(integrity);save(folder/'integrity.json',integrity)
  return rootrows,anns,checks
 finally:
  pb.stepSimulation=original_step;pb.getCameraImage=original_image
  if env is not None and env.ownsPhysicsClient:env.close();env.ownsPhysicsClient=False

def main():
 protocol=read(E/'configs/protocol.json');assert all(sha(p)==h for p,h in read(E/'configs/precollection_frozen.json')['sha256'].items());assert not(E/'results/attempts.json').exists()
 lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.perf_counter();attempts=[];rows=[];anns=[];checks=[]
 save(E/'configs/runtime.json',{'utc':now(),'python':sys.version,'platform':platform.platform(),'pybullet':importlib.metadata.version('pybullet'),'pid':os.getpid(),'EGL_VISIBLE_DEVICES':os.environ.get('EGL_VISIBLE_DEVICES'),'renderer':'native CALVIN EGL','new_models':0})
 try:
  for spec in protocol['roots']:
   attempts.append({'root_id':spec['root_id'],'status':'started','utc':now()});save(E/'results/attempts.json',attempts)
   try:
    rr,aa,cc=run_root(spec);rows+=rr;anns+=aa;checks+=cc;assert [r['image_id']for r in rows]==list(range(len(rows)));save(E/'data/public/images.json',rows);save(E/'data/annotation_private/labels.json',anns);save(E/'results/integrity.json',checks);attempts[-1]['status']='completed';save(E/'results/attempts.json',attempts)
    for boundary in [50,128]:
     fn=E/f'results/storage_first{boundary}.json'
     if len(checks)>=boundary and not fn.exists():
      dirs=[E/'data/annotation_private'/c['root_id']/f's{c["sample"]:02d}'for c in checks[:boundary]];private=sum(p.stat().st_size for d in dirs for p in d.rglob('*')if p.is_file());public=sum(Path(rows[2*i]['path']).stat().st_size for i in range(boundary));save(fn,{'states':boundary,'images':2*boundary,'private_state_bytes':private,'public_RGB_bytes':public,'bytes_per_state_excluding_root_metadata':(private+public)/boundary,'projected_256_state_bytes':256*(private+public)/boundary,'note':'includes duplicate flag-off integrity renders; excludes shared root/config metadata'})
    print('ROOT_COMPLETE',spec['root_id'],'images',len(rows),'front_valid',sum(a['class_valid'][0]for a in aa),'finger_valid',sum(a['class_valid'][1]for a in aa),flush=True)
   except Exception:
    attempts[-1].update(status='failed',traceback=traceback.format_exc());save(E/'results/attempts.json',attempts);raise
  save(E/'results/collection_summary.json',{'utc':now(),'states':len(checks),'images':len(rows),'roots':len(attempts),'seconds':time.perf_counter()-start,'camera_calls':len(checks)*4,'primary_images':len(rows),'training_updates':0,'all_integrity_pass':all(c['same_tick_4calls']and c['full_snapshot_equal']and c['RGB_depth_bitwise_equal']for c in checks),'physics_ticks_total':sum(c['tick']for c in checks if c['sample']==7),'raw_and_annotation_bytes':sum(p.stat().st_size for p in (E/'data').rglob('*')if p.is_file())})
 finally:lock.close()
if __name__=='__main__':main()
