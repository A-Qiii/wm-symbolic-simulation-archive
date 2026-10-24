"""One-pass state reconstruction audit. No policy, encoder, model or training."""
from pathlib import Path
import sys,json,hashlib,datetime,fcntl,shutil,itertools,time
import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/official_AB_state_pilot_20261018'
EV=P/'experiments/combination_generalization_20261017/evaluator_review_v1'
REP=P/'experiments/interaction_latent_repair_20261016'
sys.path.insert(0,str(EV/'source'));from native_evaluator import NativeEvaluator
sys.path.insert(0,str(REP/'source'));from labels_v2_1 import snapshot
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x))+'\n')
COLORS=['red','blue','pink']
def color_masks(rgb):
 x=rgb.astype(float);r,g,b=x[...,0],x[...,1],x[...,2]
 return np.array([(r>65)&(r>1.5*g)&(r>1.5*b),(b>55)&(b>1.4*r)&(b>1.4*g),(r>65)&(b>65)&(np.minimum(r,b)>1.4*g)&(b>.55*r)&(r>.55*b)])
def distances(env,ctl):
 out={}
 for name,obj in ctl.objects.items():
  ds=[]
  for family in ctl.finger_families:
   points=[c for link in family for c in pb.getClosestPoints(obj.uid,ctl.robot.robot_uid,.05,linkIndexB=link,physicsClientId=env.env.cid)]
   ds.append(min([float(c[8])for c in points]+[.05]))
  out[name]=ds
 return out
def sample(env,ctl,ev):
 public=env.observe();state=env.env.get_state_obs();physical=snapshot(env,ctl);labels=ev.evaluate(physical)
 return public,{'state':state,'physical':physical,'restored_evaluator_labels':labels,'nearest_each_finger_family_m':distances(env,ctl)}
def main():
 assert read(E/'results/combined_retrieval_complete.json')['frames']<=128
 assert not(E/'results/restoration.json').exists()
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 roots=read(E/'configs/combined_frame_selection.json')['roots'];nframes=sum(len(r['frame_ids'])for r in roots);sources=[Path(__file__),P/'src/wm_symbolic/env_adapter.py',P/'src/wm_symbolic/physical_oracle.py',EV/'source/native_evaluator.py',REP/'source/labels_v2_1.py']
 for scene in ['A','B']:sources.append(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{scene}.yaml')
 protocol={'utc':now(),'native_scenes_only':['A','B'],'roots':4,'frames':nframes,'restore_procedure':'fresh native world reload per frame; scene/robot positions restored without env.reset automatic integration; zero movable velocities explicitly; performCollisionDetection then record t0; exactly1 native 1/240s physics tick then record t1','missing_state':'historical contact manifold, joint/base velocities, applied motor targets/forces are not in frame; native robot.reset holds recorded finger width, not original unknown force','raw_restored_labels':'frozen native evaluator output for comparison only; never presumed historical GT','original_label_mask':'all positives requiring historical support/force unknown; allow conservative geometric negatives only after scene color mapping and stability evidence; no gripper-closure proxy','known_negative_buffers_m':{'max_one_tick_position_drift':.002,'Holding_min_gap_all_finger_families':.01,'In_min_outside_margin':.01,'In_floor_below_tolerance':-.013},'scene_mapping_evidence':'native URDF order + original RGB color purity on restored segmentation; require every native color dominant at least50% of visible original-color pixels and unique identity; median color-mask IoU>=.3 for every color on frames with>=5 rendered pixels. Report full confusion matrix; no manual slot swapping','color_thresholds':'red r>65,r>1.5g,r>1.5b;blue b>55,b>1.4r,b>1.4g;pink r,b>65,min(r,b)>1.4g,.55<r/b<1/.55','no_model_training':True,'no_features_or_model_forward':True,'source_sha256':{str(p):sha(p)for p in sources}}
 write(E/'configs/restoration_protocol.json',protocol);shutil.copy2(__file__,E/'source/restore.py');rows=[];scene_evidence={};start=time.monotonic()
 for scene in ['A','B']:
  env=CalvinAdapter(P,scene,170818,segmentation=True);ev=NativeEvaluator(scene);matrix=np.zeros((3,3),np.int64);evidence_frames=[]
  try:
   write(E/'results'/f'native_evaluator_{scene}.json',ev.provenance)
   order=[obj.name for obj in env.env.scene.movable_objects];write(E/'results'/f'native_config_{scene}.json',{'object_slot_order':order,'calibration':env.calibration(),'time_step':pb.getPhysicsEngineParameters(physicsClientId=env.env.cid)['fixedTimeStep']})
   for root in [r for r in roots if r['scene']==scene]:
    for fid in root['frame_ids']:
     original_path=E/'data/frames'/root['root_id']/f'episode_{fid:07d}.npz';original=np.load(original_path,allow_pickle=False);folder=E/'data/restored'/root['root_id']/f'{fid:07d}';folder.mkdir(parents=True,exist_ok=False)
     # A new physical world prevents prior-frame velocities and contact caches from leaking.
     oldids=[o.uid for o in env.env.scene.movable_objects];oldrobot=env.env.robot.robot_uid;env.env.load();assert oldids==[o.uid for o in env.env.scene.movable_objects]and oldrobot==env.env.robot.robot_uid
     env.env.scene.reset(original['scene_obs']);env.env.robot.reset(original['robot_obs']);env.env.robot.gripper_action=int(original['robot_obs'][-1])
     for obj in env.env.scene.movable_objects:pb.resetBaseVelocity(obj.uid,[0,0,0],[0,0,0],physicsClientId=env.env.cid)
     pb.performCollisionDetection(physicsClientId=env.env.cid);ctl=PhysicalOracle(env)
     obs0,r0=sample(env,ctl,ev);pb.stepSimulation(physicsClientId=env.env.cid);obs1,r1=sample(env,ctl,ev)
     write(folder/'physical_t0.json',r0);write(folder/'physical_t1.json',r1)
     np.savez_compressed(folder/'rendered.npz',**{f'{phase}_{k}':obs['rgb_obs'][k]for phase,obs in [('t0',obs0),('t1',obs1)]for k in ['rgb_static','rgb_gripper']},seg_t0_static=obs0['segmentation']['static'],seg_t1_static=obs1['segmentation']['static'])
     masks=color_masks(original['rgb_static']);seg=obs0['segmentation']['static'];frame_matrix=np.zeros((3,3),np.int64);visible=[];overlaps=[]
     for j,c in enumerate(COLORS):
      uid=ctl.objects['block_'+c].uid;mask=(seg&((1<<24)-1))==uid;visible.append(int(mask.sum()))
      for k in range(3):frame_matrix[j,k]=int((mask&masks[k]).sum())
      union=int((mask|masks[j]).sum());overlaps.append({'color':c,'original_color_pixels':int(masks[j].sum()),'rendered_visible_pixels':int(mask.sum()),'intersection':int((mask&masks[j]).sum()),'IoU':float((mask&masks[j]).sum()/union)if union else None,'original_bbox_xyxy':[int(np.where(masks[j])[1].min()),int(np.where(masks[j])[0].min()),int(np.where(masks[j])[1].max()),int(np.where(masks[j])[0].max())]if masks[j].any()else None,'rendered_bbox_xyxy':[int(np.where(mask)[1].min()),int(np.where(mask)[0].min()),int(np.where(mask)[1].max()),int(np.where(mask)[0].max())]if mask.any()else None})
     matrix+=frame_matrix;evidence_frames.append({'root_id':root['root_id'],'frame_id':fid,'native_color_vs_original_RGB_pixels':frame_matrix.tolist(),'visible_object_pixels':visible,'color_geometry_overlap':overlaps})
     drift={}
     for c in COLORS:
      o0=r0['physical']['objects']['block_'+c];o1=r1['physical']['objects']['block_'+c]
      drift[c]={'position_m':float(np.linalg.norm(np.array(o1['position'])-o0['position'])),'rotation_rad':float((Rotation.from_quat(o1['quaternion'])*Rotation.from_quat(o0['quaternion']).inv()).magnitude()),'finger_gaps_t0':r0['nearest_each_finger_family_m']['block_'+c],'finger_gaps_t1':r1['nearest_each_finger_family_m']['block_'+c]}
     row={'scene':scene,'root_id':root['root_id'],'frame_id':fid,'sample_roles':root['frame_roles'][str(fid)],'source_path':str(original_path),'source_sha256':sha(original_path),'folder':str(folder),'native_slot_order':order,'t0_scene_obs_max_abs_error':float(np.abs(np.array(r0['state']['scene_obs'])-original['scene_obs']).max()),'t0_robot_joint_width_max_abs_error':float(np.abs(np.array(r0['state']['robot_obs'])[6:14]-original['robot_obs'][6:14]).max()),'t0_tcp_position_error_m':float(np.linalg.norm(np.array(r0['state']['robot_obs'])[:3]-original['robot_obs'][:3])),'rgb_MAE_0_255':{k:float(np.abs(obs0['rgb_obs'][k].astype(float)-original[k]).mean())for k in ['rgb_static','rgb_gripper']},'one_tick_objects':drift,'contacts_t0':len(r0['physical']['contacts']),'contacts_t1':len(r1['physical']['contacts']),'force_contacts_t0':sum(c[9]>.01 for c in r0['physical']['contacts']),'force_contacts_t1':sum(c[9]>.01 for c in r1['physical']['contacts']),'restored_t0':r0['restored_evaluator_labels'],'restored_t1':r1['restored_evaluator_labels'],'original_gripper_action':int(original['robot_obs'][-1]),'original_gripper_width':float(original['robot_obs'][6])};rows.append(row)
    write(E/'results/restoration_partial.json',rows);print('RESTORED_EPISODE',root['root_id'],len(rows),flush=True)
   ious={c:[a['IoU']for ef in evidence_frames for a in ef['color_geometry_overlap']if a['color']==c and a['rendered_visible_pixels']>=5 and a['IoU']is not None]for c in COLORS};median_iou={c:float(np.median(v))if v else 0. for c,v in ious.items()};totals=matrix.sum(1);purity=np.diag(matrix)/np.maximum(totals,1);identity=bool(np.all(totals>=5)and np.array_equal(matrix.argmax(1),np.arange(3))and np.all(purity>=.5)and min(median_iou.values())>=.3);scene_evidence[scene]={'native_slot_order':order,'native_object_rows_original_color_columns':matrix.tolist(),'row_purity':purity.tolist(),'median_color_mask_IoU':median_iou,'minimum_pixels_per_color':5,'native_identity_supported':identity,'frames':evidence_frames,'note':'RGB purity is mapping evidence, not a force/holding label; unknown or occluded cases remain visible in records'}
  finally:env.close()
 write(E/'results/color_mapping_evidence.json',scene_evidence)
 for row in rows:
  mapping=scene_evidence[row['scene']]['native_identity_supported'];y=[False]*6;v=[False]*6;why=[]
  for j,c in enumerate(COLORS):
   d=row['one_tick_objects'][c];stable=d['position_m']<=.002;far=min(d['finger_gaps_t0']+d['finger_gaps_t1'])>.01
   v[j]=bool(mapping and stable and far);why.append('known_negative_both_finger_families_geometrically_far'if v[j]else'unknown_historical_force_contact_not_recorded_or_mapping_drift_uncertain')
  for j,c in enumerate(COLORS):
   gs=[row['restored_t0']['objects'][c],row['restored_t1']['objects'][c]];outside=all(g['minimum_side_rim_clearance_m']<=-.01 or g['floor_clearance_m']<=-.013 for g in gs);stable=row['one_tick_objects'][c]['position_m']<=.002
   v[3+j]=bool(mapping and stable and outside);why.append('known_negative_robustly_outside_drawer_geometry'if v[3+j]else'unknown_historical_support_force_or_containment_boundary')
  row['original_provisional_labels']={'y':y,'v':v,'fact_order':[f'Holding({c})'for c in COLORS]+[f'In({c},drawer)'for c in COLORS],'reason':why,'positive_labels_admitted':0,'force_from_restoration_not_historical_GT':True}
 write(E/'results/restoration.json',rows);write(E/'results/restoration_budget.json',{'utc':now(),'independent_episodes':4,'frames':nframes,'native_world_reloads':nframes,'integration_ticks':nframes,'seconds':time.monotonic()-start,'native_scene_identity_supported':{s:r['native_identity_supported']for s,r in scene_evidence.items()},'known_H_negative_fact_cells':sum(sum(r['original_provisional_labels']['v'][:3])for r in rows),'known_In_negative_fact_cells':sum(sum(r['original_provisional_labels']['v'][3:])for r in rows),'known_positive_cells':0,'model_updates':0,'new_fixed_skill_transitions':0,'original_physical_contact_GT_available':False});print('RESTORATION_COMPLETE',flush=True)
if __name__=='__main__':main()
