from common import *
import csv,collections,types,shutil,sys
import pybullet as pb
import hydra
from omegaconf import OmegaConf
from wm_symbolic.physical_oracle_v2 import PhysicalOracleV2
from wm_symbolic.physical_oracle import ExecutionFailure

def csvout(path,rows):
 with path.open('w')as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def replay(rid):
 d=OLD/'data/private'/rid;cfg=OmegaConf.load(d/'resolved_config.yaml');cfg.env.use_egl=False;env=None;ctl=None
 original_image=pb.getCameraImage;original_step=pb.stepSimulation;calls={'image':0,'step_after_restore':0}
 def image(*a,**kw):calls['image']+=1;raise RuntimeError('No rendering in retrospective geometry replay')
 def step(*a,**kw):calls['step_after_restore']+=1;raise RuntimeError('No physical rollout in retrospective geometry replay')
 pb.getCameraImage=image
 try:
  env=hydra.utils.instantiate(cfg.env,use_scene_info=True);saved=read(d/'s00/snapshot_before.json')
  # Restore saved geometry in a disposable DIRECT audit client only. No data or
  # execution root is changed, and forces from the restored world are not labels.
  for b in saved:
   pb.resetBasePositionAndOrientation(b['uid'],*b['pose'],physicsClientId=env.cid);pb.resetBaseVelocity(b['uid'],*b['velocity'],physicsClientId=env.cid)
   for j,state in enumerate(b['joints']):pb.resetJointState(b['uid'],j,state[0],state[1],physicsClientId=env.cid)
  pb.stepSimulation=step;errors=[]
  for b in saved:
   for j,expected in enumerate(b['links']):
    actual=pb.getLinkState(b['uid'],j,computeForwardKinematics=1,physicsClientId=env.cid)
    errors.append(float(np.max(np.abs(np.array(actual[4])-expected[4]))))
  assert max(errors)<1e-6
  adapter=types.SimpleNamespace(env=env,observe=lambda:{'robot_obs':env.get_state_obs()['robot_obs']})
  ctl=PhysicalOracleV2(adapter);pairs=[(env.robot.robot_uid,ctl.shadow.robot)]+list(ctl.shadow.body_map.items());aabb_errors=[];frame_errors=[];shape_checks=0;names={}
  for actual,query in pairs:
   for j in range(-1,pb.getNumJoints(actual,physicsClientId=env.cid)):
    aabb_errors.append(float(np.max(np.abs(np.array(pb.getAABB(actual,j,physicsClientId=env.cid))-np.array(pb.getAABB(query,j,physicsClientId=ctl.shadow.cid))))))
    aa=pb.getCollisionShapeData(actual,j,physicsClientId=env.cid);bb=pb.getCollisionShapeData(query,j,physicsClientId=ctl.shadow.cid);assert len(aa)==len(bb)
    for a,b in zip(aa,bb):
     assert a[2]==b[2]
     for k in [3,5,6]:assert np.allclose(a[k],b[k],atol=1e-9,rtol=0)
     shape_checks+=1
    if j>=0:
     a=pb.getLinkState(actual,j,computeForwardKinematics=1,physicsClientId=env.cid);b=pb.getLinkState(query,j,computeForwardKinematics=1,physicsClientId=ctl.shadow.cid);frame_errors.append(float(np.max(np.abs(np.array(a[4])-b[4]))));names[f'{actual}:{j}']=pb.getJointInfo(actual,j,physicsClientId=env.cid)[12].decode()
  assert max(aabb_errors)<1e-6 and max(frame_errors)<1e-6
  expected=read(d/'frozen_controller_result.json');target=expected['target'];reason=None
  try:ctl.plan_lift(target)
  except ExecutionFailure as ex:reason=str(ex)
  cand=ctl.planning_audit;old=expected['planning_private'];assert len(cand)==len(old)==16
  categorical=['tilt_world_x_degrees','grasp_z_offset_m','yaw_flip','valid','reason','obstacle','robot_link','obstacle_link','stage','samples'];differences=[]
  for i,(a,b)in enumerate(zip(cand,old)):
   for key in categorical:
    if a.get(key)!=b.get(key):differences.append({'candidate':i,'key':key,'replay':a.get(key),'saved':b.get(key)})
   for key in ['distance_m','error_m','minimum_excess_clearance_m','max_IK_error_m']:
    if key in a and key in b and abs(a[key]-b[key])>1e-6:differences.append({'candidate':i,'key':key,'replay':a[key],'saved':b[key]})
  assert not differences,differences
  out={'root_id':rid,'purpose':'retrospective geometry-only replay, not rollout, not independent configuration or new evaluation','same_saved_initial_link_frames_max_abs_error_m':max(errors),'native_query_aabb_max_abs_error_m':max(aabb_errors),'native_query_link_frame_max_abs_error_m':max(frame_errors),'collision_shapes_compared':shape_checks,'native_link_names':names,'candidate_records_reproduced':16,'valid_candidates':sum(x['valid']for x in cand),'rejection_reason':reason,'all_candidate_fields_match_saved_with_numeric_atol_1e_6':True,'calls':calls,'planning':cand};save(E/'results/replays'/f'{rid}.json',out);return out
 finally:
  pb.getCameraImage=original_image;pb.stepSimulation=original_step
  if ctl is not None:ctl.close()
  if env is not None and env.ownsPhysicsClient:env.close();env.ownsPhysicsClient=False

def main():
 assert not E.exists()
 for sub in ['source','configs','data/public','data/private','results/replays','logs']:(E/sub).mkdir(parents=True,exist_ok=True)
 R.mkdir();(R/'figures').mkdir()
 seal=read(OLD/'configs/delivery_manifest.json');assert sha(OLD/'configs/delivery_manifest.json')=='4823571c19297461d26903e6d804ff5b2bcebe999d9170eb89c7bcd2a030975c'
 allowed=list(seal['sha256'])+[str(OLD/'configs/delivery_manifest.json')];opened=install_guard(allowed)
 for p,h in seal['sha256'].items():assert sha(p)==h,p
 proto=read(OLD/'configs/protocol.json');data=read(OLD/'data/records.json');surfaces={s:OmegaConf.to_container(OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{s}.yaml').surfaces.table)for s in 'ABC'};rows=[];blocks=[]
 for spec in proto['roots']:
  rid=spec['root_id'];d=OLD/'data/private'/rid;snap=read(d/'s00/physical_truth.json');result=read(d/'frozen_controller_result.json');lo,hi=np.array(surfaces[spec['scene']]);outside=[];badcontacts=[]
  for c in COLORS:
   initial=np.array(spec['blocks'][c]['initial_pos']);out=not np.all((initial[:2]>=lo[:2])&(initial[:2]<=hi[:2]));outside+=[c]if out else[];obj=snap['physical']['objects']['block_'+c];cs=[]
   for x in snap['physical']['contacts']:
    if x[9]<=.01:continue
    if x[1]==obj['uid']and x[2]==snap['physical']['table_uid']:link=x[4]
    elif x[2]==obj['uid']and x[1]==snap['physical']['table_uid']:link=x[3]
    else:continue
    if link>=0:cs.append({'table_link':link,'force_N':x[9],'gap_m':x[8]})
   if cs:badcontacts.append({'color':c,'contacts':cs})
   blocks.append({'root_id':rid,'split':spec['split'],'scene':spec['scene'],'color':c,'is_target':c==spec['target_color'],'requested_x_m':initial[0],'requested_y_m':initial[1],'outside_native_table_xy':out,'settled_x_m':obj['position'][0],'settled_y_m':obj['position'][1],'settled_z_m':obj['position'][2],'initial_displacement_m':float(np.linalg.norm(np.array(obj['position'])-initial)),'speed_m_s':float(np.linalg.norm(obj['linear_velocity'])),'non_base_table_contact_links':','.join(map(str,sorted({x['table_link']for x in cs}))),'max_non_base_contact_force_N':max([x['force_N']for x in cs],default=0)})
  rr=[r for r in data if r['root_id']==rid];rows.append({'root_id':rid,'split':spec['split'],'scene':spec['scene'],'target_color':spec['target_color'],'target_outside_native_table':spec['target_color']in outside,'outside_colors':','.join(outside),'initial_non_base_contact_colors':','.join(x['color']for x in badcontacts),'initial_non_base_contacts':badcontacts,'controller_status':result['status'],'reason':result['reason'],'controls':result['controls'],'valid_grasp_candidates':sum(x['valid']for x in result['planning_private']),'stable_positive_roots_contribution':int(any(r['physical_valid']and r['y']for r in rr)),'eligible_positive_queries':sum(r['eligible']and r['y']for r in rr),'eligible_negative_queries':sum(r['eligible']and not r['y']for r in rr)})
 csvout(R/'original_root_failures.csv',[{k:v for k,v in row.items()if k!='initial_non_base_contacts'}for row in rows]);csvout(R/'original_block_surface_audit.csv',blocks)
 prior=P/'runs/physical_oracle_v2';history=[]
 for rid,color in [('A_acceptance_v2_00','red'),('B_acceptance_v2_01','blue'),('C_acceptance_v2_02','pink'),('C_acceptance_v2_00','red')]:
  d=prior/'independent_acceptance'/rid/'negative_preconditions'/f'02_lift_{color}_block_table';result=read(d/'result.json');trace=read(d/'trace_private.json');controls=read(d/'controls.json');target='block_'+color;zs=[x['objects'][target][2]for x in trace];scenes=np.array([x['scene_obs']for x in trace]);row={'root_id':rid,'target_color':color,'result':result['status'],'reason':result['reason'],'conservative_pass':result['conservative_pass'],'trace_frames':len(trace),'control_rows':len(controls),'target_z_first_last_m':[zs[0],zs[-1]]if zs else None,'target_z_gain_m':zs[-1]-zs[0]if zs else None,'unrequested_max_displacement_m':result['max_unrequested_object_displacement_m'],'max_penetration_m':result['max_robot_contact_penetration_m'],'non_drawer_scene_max_delta_from_first':np.max(np.abs(scenes[:,[0,2,3,4,5]]-scenes[0,[0,2,3,4,5]]),axis=0).tolist()if len(scenes)else None,'result_path':str(d/'result.json'),'trace_sha256':sha(d/'trace_private.json'),'initial_xyz':{c:trace[0]['objects']['block_'+c][:3]for c in COLORS}if trace else None};history.append(row)
 replays=[replay(rid)for rid in ['relation32_A_00','relation32_A_03','relation32_B_09','relation32_C_24']]
 # Original failed gate and every root remain the immutable historical outcome.
 outcome={'utc':now(),'old_manifest_sha256':sha(OLD/'configs/delivery_manifest.json'),'old_files_verified':len(seal['sha256']),'original_full_protocol':proto,'original_data_gate':read(OLD/'results/data_audit.json'),'native_table_surfaces':surfaces,'root_audit':rows,'block_surface_audit':blocks,'historical_trajectories':history,'controller_not_promoted':read(prior/'CONTROLLER_NOT_PROMOTED.json'),'prior_causal_clearance_evidence':read(prior/'causal_clearance_development_evidence.json'),'geometry_replays':[{k:v for k,v in x.items()if k not in ['planning','native_link_names']}for x in replays],'finding':'Fixed world-x layout is not scene-aware: B right objects and C left objects lie outside the declared table sampling surface. All six monitor-triggered roots have initial button contacts involving outside-surface objects. A00 rejection is separately reproducible correct-geometry conservative clearance rejection, not coordinate-copy failure. The12mm margin has prior evidence preventing slider motion and remains unchanged.','supported_correction':'Change only x-slot placement from fixed world coordinates to each native scene table surface fractions [.18,.5,.82], already used by the previous controller test generator. Retain y range, yaw, z, x jitter, controller, force labels and margins. New configurations only, all attempts retained, engineering pilot not model evaluation.','scope_limit':'Native table surface is a sampling support, not a physical impossibility boundary. Initial out-of-support contacts and motion are directly observed; no claim that every old failure is caused by this one defect or that corrected collection will always succeed.','old_inputs_opened':sorted(opened)}
 save(E/'results/existing_audit.json',outcome);save(R/'existing_audit.json',outcome);save(E/'configs/original_gate_and_lineage.json',{'original_protocol':proto,'original_gate':read(OLD/'results/data_audit.json'),'old_manifest_sha256':sha(OLD/'configs/delivery_manifest.json'),'must_not_merge_pilot':True})
 refs=[P/'src/wm_symbolic/physical_oracle.py',P/'src/wm_symbolic/physical_oracle_v2.py',P/'scripts/evaluate_physical_oracle_v2.py',P/'scripts/prepare_physical_v2.py',prior/'CONTROLLER_NOT_PROMOTED.json',prior/'causal_clearance_development_evidence.json']+[P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{s}.yaml'for s in 'ABC'];save(E/'configs/audit_input_references.json',{'sha256':{str(p):sha(p)for p in refs},'old_experiment_full_manifest_verified':sha(OLD/'configs/delivery_manifest.json')})
 for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,E/'source'/f.name)
 print('AUDIT_COMPLETE',json.dumps({'out_of_surface_blocks':sum(x['outside_native_table_xy']for x in blocks),'roots_with_outside_objects':sum(bool(x['outside_colors'])for x in rows),'replays':len(replays),'all_geometry_reproduced':True}),flush=True)
if __name__=='__main__':main()
