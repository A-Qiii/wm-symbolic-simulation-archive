from pathlib import Path
import json,gzip,hashlib,sys,time,fcntl,traceback,argparse,datetime
import numpy as np
import pybullet as p
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_long_pairs_20261014';B=P/'data/predicate_long_pairs_20261014';sys.path.insert(0,str(P/'scripts'))
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
from wm_symbolic.physical_oracle import PhysicalOracle
from core1200_collect import label,plain
from evaluate_physical_oracle import physics_guard
C=['red','blue','pink']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(plain(x),ensure_ascii=False,indent=2)+'\n')
def targets(lab):
 cat=np.array(lab['categorical'][:4]);cv=np.array(lab['categorical_valid'][:4],bool)
 return {'y':np.r_[cat[:3]==6,cat[:3]==3,cat[3]==2].tolist(),'valid':np.r_[cv[:3],cv[:3],cv[3]].tolist()}
def near(a,b):
 d=np.linalg.norm(a[:,:3]-b[:,:3],axis=1);yaw=np.abs((a[:,5]-b[:,5]+np.pi/2)%np.pi-np.pi/2)
 return bool(np.all(d<=.012)or(np.all(d<=.020)and np.all(yaw<=.25)))
def collect(root,protocol,prior):
 folder=B/'episodes'/root['root_id'];folder.mkdir(exist_ok=False);(folder/'public').mkdir();(folder/'_private').mkdir();save(folder/'_private/root_provenance.json',root)
 trace=[];attempts=[];windows=[];pairs=[];private_pairs=[];failure=None;env=None;original_step=None;ticks=0;segment='reset';macro_index=None;started=time.monotonic();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 public_index=(folder/'public/frames.jsonl').open('w');commands=(folder/'public/commands.jsonl').open('w');actual=None
 try:
  env=CalvinAdapter(P,root['environment'],root['seed'],tabletop=True,segmentation=True)
  for obj in env.env.scene.movable_objects:
   val=root['layout'][obj.name];obj.initial_pos=np.array(val[:3]);obj.initial_orn=np.array(p.getQuaternionFromEuler(val[3:]))
  for door in env.env.scene.doors:
   if door.name=='base__drawer':door.initial_state=root['drawer_reset_m']
  ob=env.reset();auditctl=PhysicalOracle(env);actual=np.array([auditctl.objects['block_'+c].get_state()for c in C]);assert actual.shape==(3,6)
  collisions=[src for prev,src in prior if near(actual,prev)];assert not collisions,('actual_initialized_layout_near_old',collisions)
  assert env.env.control_freq==30 and env.env.action_repeat==8
  OmegaConf.save(env.config,folder/'_private/complete_config.yaml',resolve=True);np.savez_compressed(folder/'_private/initial_state.npz',**env.env.get_state_obs())
  original_step=p.stepSimulation
  def counted_step(*args,**kwargs):
   nonlocal ticks
   out=original_step(*args,**kwargs);ticks+=1;return out
  p.stepSimulation=counted_step
  def record(obs,action):
   fi=len(trace);assert ticks==8*fi,(fi,ticks);lab=label(env);objects={};contacts={}
   for c in C:
    obj=auditctl.objects['block_'+c];cs=[q for q in p.getContactPoints(bodyA=obj.uid,physicsClientId=env.env.cid)if q[9]>.01];contacts[c]=[list(q)for q in cs];robot=[q for q in cs if q[2]==auditctl.robot.robot_uid];links={q[4]for q in robot};bi=all(links&family for family in auditctl.finger_families);ext=[q for q in cs if q[2]!=auditctl.robot.robot_uid];held=bool(bi and not ext);assert held==bool(lab['holding_derived'][C.index(c)])
    objects[c]={'position_m':auditctl.xyz('block_'+c).tolist(),'held':held,'bilateral':bool(bi),'external_support_contacts':len(ext),'robot_force_contacts':len(robot),'object_uid_evaluator_only':obj.uid,'visible_pixels':{cam:int((((m&((1<<24)-1))==obj.uid)&(m>=0)).sum())for cam,m in obs['segmentation'].items()}}
   row={'frame':fi,'physics_ticks_since_reset':ticks,'sim_seconds':ticks/240,'segment':segment,'macro_index':macro_index,'action_private':action,'robot_obs':obs['robot_obs'].tolist(),'drawer_position_m':auditctl.joint('drawer_link'),'scene_obs_private':env.env.scene.get_obs().tolist(),'labels':lab,'targets':targets(lab),'objects_audit':objects,'contacts_private':contacts}
   fp=folder/'public'/f'{fi:05d}.npz';save_npz(fp,obs);entry={'frame':fi,'physics_ticks_since_reset':ticks,'sim_seconds':ticks/240,'record_elapsed_wall_seconds':time.monotonic()-started,'file':fp.name,'bytes':fp.stat().st_size,'sha256':sha(fp)};public_index.write(json.dumps(entry)+'\n')
   if 'target_xyz'in action:commands.write(json.dumps({'arrival_frame':fi,'physics_ticks_since_reset':ticks,'sim_seconds':ticks/240,**{k:action[k]for k in ['target_xyz','target_quaternion','gripper']}})+'\n')
   trace.append(row)
  record(ob,{'phase':'initial_reset_only'})
  def hold(name,grip):
   nonlocal segment
   segment=name;start=len(trace)-1;ctl=PhysicalOracle(env,record);ctl.phase=name
   with physics_guard(env):ctl.hold(96,grip)
   rows=trace[start:];tcp=np.array([r['robot_obs'][:3]for r in rows]);dis={c:float(max(np.linalg.norm(np.array(q['objects_audit'][c]['position_m'])-rows[0]['objects_audit'][c]['position_m'])for q in rows)*1000)for c in C}
   loss={c:[q['frame']for q in rows[1:]if not q['objects_audit'][c]['held']]for c in C if rows[0]['objects_audit'][c]['held']}
   before=rows[0]['targets'];after=rows[-1]['targets'];changed=[k for k in range(7)if before['valid'][k]and after['valid'][k]and before['y'][k]!=after['y'][k]]
   return {'start_frame':start,'end_frame':len(trace)-1,'controls':len(trace)-1-start,'gripper_public_command':grip,'TCP_max_displacement_mm':float(np.linalg.norm(tcp-tcp[0],axis=1).max()*1000),'objects_max_displacement_mm':dis,'drawer_max_drift_mm':float(max(abs(q['drawer_position_m']-rows[0]['drawer_position_m'])for q in rows)*1000),'held_loss_frames':loss,'valid_changed_columns':changed,'start_targets':before,'end_targets':after}
  def window(anchor):
   inds=[anchor-d for d in [63,42,21,0]];assert min(inds)>=0;wid=root['root_id']+f'__t{anchor:05d}'
   w={'window_id':wid,'root_id':root['root_id'],'environment':root['environment'],'split':root['split'],'dev_role':root['dev_role'],'anchor_frame':anchor,'history_frame_indices':inds,'simulation_times':[trace[i]['sim_seconds']for i in inds],'physics_ticks':[trace[i]['physics_ticks_since_reset']for i in inds],'public_frames':[str(folder/'public'/f'{i:05d}.npz')for i in inds]};windows.append(w)
   save(folder/'_private'/(wid+'.json'),{'frames':[trace[i]for i in inds],'all_7_targets':trace[anchor]['targets']});return wid
  initial_hold=hold('initial_history_hold',1);before_id=window(len(trace)-1)
  for mi,skill in enumerate(root['planned_skills']):
   macro_index=mi;start=len(trace)-1;segment='primitive';ctl=PhysicalOracle(env,record);attempt={'macro_index':mi,'skill':skill,'start_frame':start,'status':'started'};attempts.append(attempt);save(folder/'_private/attempts.json',attempts)
   with physics_guard(env):result=ctl.run(skill)
   primitive_end=len(trace)-1;assert primitive_end-start==result['controls'];grip=int(env.observe()['robot_obs'][-1]);assert grip in [-1,1],grip;h=hold('macro_posthold',grip);end=len(trace)-1;after_id=window(end)
   pairid=root['root_id']+f'__m{mi}';pub={'pair_id':pairid,'root_id':root['root_id'],'environment':root['environment'],'split':root['split'],'dev_role':root['dev_role'],'macro_index':mi,'skill':skill,'declared_arguments':{'color':skill.split('_')[1]}if skill.startswith('lift_')else{},'macro_contract':'frozen_skill_plus_posthold96_v1','declared_primitive_budget':900,'declared_posthold_controls':96,'before_window_id':before_id,'after_window_id':after_id};pairs.append(pub)
   y0=trace[start]['targets'];y1=trace[end]['targets'];changed=[k for k in range(7)if y0['valid'][k]and y1['valid'][k]and y0['y'][k]!=y1['y'][k]];j=C.index(root['target_color']);end_goal=(y1['y'][6]and y1['valid'][6])if skill=='open_drawer'else((not y1['y'][6]and y1['valid'][6]and trace[end]['drawer_position_m']<.02)if skill=='close_drawer'else (y1['y'][j]and y1['valid'][j]if skill.startswith('lift_')else y1['y'][3+j]and y1['valid'][3+j]and not any(y1['y'][:3])))
   priv={'pair_id':pairid,'start_frame':start,'primitive_end_frame':primitive_end,'end_frame':end,'primitive':result,'posthold':h,'before_targets':y0,'after_targets':y1,'valid_changed_columns':changed,'both_all7_valid':all(y0['valid'])and all(y1['valid']),'endpoint_requested_goal_sustained':bool(end_goal),'macro_controls':end-start,'macro_sim_seconds':(end-start)/30};private_pairs.append(priv);attempt.update(status='completed',primitive_status=result['status'],primitive_reason=result.get('reason'),end_frame=end);save(folder/'_private/attempts.json',attempts);before_id=after_id
  np.savez_compressed(folder/'_private/final_state.npz',**env.env.get_state_obs())
 except Exception as ex:
  failure={'type':type(ex).__name__,'message':str(ex),'traceback':traceback.format_exc(),'last_frame':len(trace)-1};save(folder/'_private/collection_error.json',failure)
 finally:
  if original_step is not None:p.stepSimulation=original_step
  if env is not None:env.close()
  public_index.close();commands.close()
  with gzip.open(folder/'_private/trace.json.gz','wt')as f:json.dump(plain(trace),f)
 save(folder/'public/windows.json',windows);save(folder/'public/pairs.json',pairs);save(folder/'_private/pairs.json',private_pairs)
 summary={'root_id':root['root_id'],'environment':root['environment'],'split':root['split'],'dev_role':root['dev_role'],'target_color':root['target_color'],'initial_drawer':root['initial_drawer'],'actual_initialized_layout':actual.tolist()if actual is not None else None,'frames':len(trace),'controls':max(0,len(trace)-1),'physics_ticks':ticks,'attempted_macros':len(attempts),'completed_macro_pairs':len(pairs),'not_attempted_skills':root['planned_skills'][len(attempts):],'attempts':attempts,'initial_history_hold':initial_hold if 'initial_hold'in locals()else None,'controller_failures':[q for q in private_pairs if q['primitive']['status']!='success'],'posthold_any_held_losses':[q['pair_id']for q in private_pairs if any(q['posthold']['held_loss_frames'].values())],'collection_error':failure,'physics_guard_passed':failure is None,'started_utc':stamp,'elapsed_seconds':time.monotonic()-started};save(folder/'_private/summary.json',summary)
 return pairs,windows,private_pairs,summary
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','remaining'],required=True);args=ap.parse_args();protocol=read(E/'configs/protocol.json')
 for path,h in protocol['frozen_sha256'].items():assert sha(path)==h,path
 assert sha(B/'roots_manifest.json')==protocol['roots_manifest_sha256'];lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if args.phase=='remaining':assert read(E/'results/pilot_gate.json')['passed']and read(E/'results/cache_pilot.json')['passed']
 manifest=read(B/'data_manifest.json');manifest.setdefault('windows',[]);manifest.setdefault('private_pairs',[]);done={r['root_id']for r in manifest['roots']};prior=[(np.array(x['layout']),x['source'])for x in read(E/'configs/root_exclusion.json')['prior_layouts']];prior +=[(np.array(x['actual_initialized_layout']),x['root_id'])for x in manifest['roots']if x['actual_initialized_layout']is not None]
 for root in read(B/'roots_manifest.json'):
  if root['root_id']in done:continue
  if (root['root_id']in protocol['pilot_roots'])!=(args.phase=='pilot'):continue
  rows,windows,private,s=collect(root,protocol,prior);manifest['records']+=rows;manifest['windows']+=windows;manifest['private_pairs']+=private;manifest['roots'].append(s);save(B/'data_manifest.json',manifest)
  if s['actual_initialized_layout']is not None:prior.append((np.array(s['actual_initialized_layout']),root['root_id']))
  save(E/'results/progress.json',{'stage':'collection_'+args.phase,'roots':len(manifest['roots']),'pairs':len(manifest['records']),'last_root':root['root_id'],'training':False});print(json.dumps({'root':root['root_id'],'frames':s['frames'],'pairs':len(rows),'controller_failures':[(x['pair_id'],x['primitive']['reason'])for x in s['controller_failures']],'hold_losses':s['posthold_any_held_losses'],'error':s['collection_error']},ensure_ascii=False),flush=True)
  if s['collection_error']:raise RuntimeError('Preserved infrastructure/guard error; root will not be silently rerun')
 save(E/'results'/('collection_'+args.phase+'.json'),{'roots':len(manifest['roots']),'pairs':len(manifest['records']),'frames':sum(s['frames']for s in manifest['roots']),'new_reader_inference':False});print('COLLECTION_PHASE_COMPLETE',args.phase,flush=True)
if __name__=='__main__':main()
