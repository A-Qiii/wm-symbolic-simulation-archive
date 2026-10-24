from common import *
from labels_v2_1 import snapshot,evaluate
import sys,time,gzip,traceback,argparse,fcntl,shutil
import pybullet as p
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(P/'scripts'))
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
from core1200_collect import label as old_label,plain
from evaluate_physical_oracle import physics_guard
CASES=['table_open','single_inside','edge_straddle','outside_front','stacked','separated','empty_mid','empty_closed','outside_front_supported']

def episode(root,case):
 eid=root['root_id']+'__'+case;folder=B/'episodes'/eid;folder.mkdir();(folder/'public').mkdir();(folder/'_private').mkdir();trace=[];windows=[];actions=[];env=None;error=None;original=None;ticks=0;segment='initial';calls=[]
 try:
  env=CalvinAdapter(P,root['environment'],root['seed'],tabletop=True,segmentation=False);ctl=PhysicalOracle(env);q=.005+.005*root['replicate']if case=='empty_closed'else .09+.04*root['replicate']if case=='empty_mid'else 0. if case=='chain'else .22
  link=p.getLinkState(ctl.table.uid,ctl.links['drawer_link'],physicsClientId=env.env.cid);rot=Rotation.from_quat(link[5]);origin=np.array(link[4])+rot.apply([0,0,q-ctl.joint('drawer_link')]);layout={k:np.array(v).copy()for k,v in root['layout'].items()};j=COLORS.index(root['color']);k=(j+1)%3;d=root['cavity_local_offsets'];dx,dz=d['x_jitter'],d['z_jitter'];yaw=d['yaw']
  def put(which,local):layout['block_'+COLORS[which]]=np.r_[origin+rot.apply(local),[0,0,yaw]]
  if case=='single_inside':put(j,[dx,.044,.08+dz])
  elif case=='edge_straddle':put(j,[-.174+dx*.15,.104,.08+dz])
  elif case=='outside_front':put(j,[dx,.104,.184+dz*.1])
  elif case=='outside_front_supported':put(j,[dx,.104,.172+dz*.1])
  elif case=='stacked':put(k,[dx,.044,.08+dz]);put(j,[dx,.086,.08+dz])
  elif case=='separated':put(k,[-.07+dx,.044,.08+dz]);put(j,[.07+dx,.044,.08+dz])
  for obj in env.env.scene.movable_objects:obj.initial_pos=layout[obj.name][:3];obj.initial_orn=np.array(p.getQuaternionFromEuler(layout[obj.name][3:]))
  for door in env.env.scene.doors:
   if door.name=='base__drawer':door.initial_state=q
  obs=env.reset();ctl=PhysicalOracle(env);initial=env.env.get_state_obs();np.savez_compressed(folder/'_private/initial_state.npz',**initial);save(folder/'_private/provenance.json',{'root':root,'condition':case,'intended_layout':layout,'drawer_q':q,'paired_reset_branch_not_transition':case!='chain'});original=p.stepSimulation
  def counted(*args,**kw):
   nonlocal ticks
   out=original(*args,**kw);ticks+=1;return out
  p.stepSimulation=counted
  def record(ob,action):
   f=len(trace);assert ticks==f*8;physical=snapshot(env,ctl);new=evaluate(physical);old=old_label(env);trace.append({'frame':f,'segment':segment,'control':action,'physics_ticks':ticks,'physical':physical,'new_labels':new,'old_labels':old});np.savez_compressed(folder/'public'/f'{f:05d}.npz',rgb_static=ob['rgb_obs']['rgb_static'],rgb_gripper=ob['rgb_obs']['rgb_gripper'],robot_obs=ob['robot_obs'])
  record(obs,{'phase':'reset_only'})
  def hold(n,name):
   nonlocal segment
   segment=name;h=PhysicalOracle(env,record,max_controls=n+1)
   with physics_guard(env):h.hold(n,int(env.observe()['robot_obs'][-1]))
  def window(anchor,role):
   if anchor<63:return
   previous=next((r for r in windows if r['anchor']==anchor),None)
   if previous:
    if role not in previous['roles']:previous['roles'].append(role)
    return
   inds=[anchor-k for k in [63,42,21,0]];paths=[folder/'public'/f'{i:05d}.npz'for i in inds];wid=eid+f'__t{anchor:05d}';windows.append({'window_id':wid,'root_id':root['root_id'],'episode_id':eid,'environment':root['environment'],'split':root['split'],'condition':case,'anchor':anchor,'indices':inds,'public_frames':[str(x)for x in paths],'public_sha256':[sha(x)for x in paths],'roles':[role],'new_labels':trace[anchor]['new_labels'],'old_labels':trace[anchor]['old_labels']})
  hold(127,'settle');window(63,'settled_state');window(127,'before_action')
  requests=['open_drawer','lift_'+root['color']+'_block_table','place_in_drawer','close_drawer']if case=='chain'else['close_drawer']if case in ['table_open','single_inside','stacked','separated','outside_front_supported']else[]
  for step,skill in enumerate(requests):
   start=len(trace)-1;segment='skill_'+str(step);controller=PhysicalOracle(env,record);before=trace[start]['new_labels']
   with physics_guard(env):res=controller.run(skill)
   endprimitive=len(trace)-1;assert endprimitive-start==res['controls']and res['controls']<=900;hold(96,'posthold_'+str(step));end=len(trace)-1;window(end,'after_action');row={'step':step,'skill':skill,'family':'lift'if skill.startswith('lift')else skill.split('_')[0],'object':j if skill.startswith('lift')or skill=='place_in_drawer'else None,'start':start,'primitive_end':endprimitive,'end':end,'native':res,'before':before,'after':trace[end]['new_labels']};actions.append(row)
   subseq=trace[start+1:endprimitive+1]
   if skill=='close_drawer':
    for threshold in [.14,.07]:
     hit=next((r for r in subseq if r['physical']['drawer_q']<=threshold),None)
     if hit:window(hit['frame'],f'closing_cross_{threshold}')
   hit=next((r for r in subseq if r['control'].get('phase')=='descend_named_block'and r['control'].get('gripper')==-1),None)
   if hit:window(hit['frame'],'grasp_contact')
   lower=[r for r in subseq if r['control'].get('phase')=='lower_into_drawer']
   if len(lower)>=64:window(lower[63]['frame'],'suspended_lowering')
   save(folder/'_private/actions.json',actions)
   if case=='chain'and res['status']!='success':break
  np.savez_compressed(folder/'_private/final_state.npz',**env.env.get_state_obs())
 except Exception as exc:error={'error':str(exc),'traceback':traceback.format_exc()};save(folder/'_private/error.json',error)
 finally:
  if original is not None:p.stepSimulation=original
  if env is not None:env.close()
  with gzip.open(folder/'_private/full_physics_trace.json.gz','wt')as f:json.dump(plain(trace),f)
  save(folder/'_private/windows.json',windows);save(folder/'_private/actions.json',actions)
 summary={'episode_id':eid,'root_id':root['root_id'],'condition':case,'environment':root['environment'],'split':root['split'],'frames':len(trace),'windows':len(windows),'calls':len(actions),'native_statuses':[r['native']['status']for r in actions],'final_labels':trace[-1]['new_labels']if trace else None,'error':error};save(folder/'_private/summary.json',summary);return summary

def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for path,h in read(E/'configs/frozen.json')['sha256'].items():assert sha(path)==h,path
 roots=read(E/'configs/close_extension_roots.json');assert len(roots)==9;manifest=read(B/'manifest.json')
 for root in roots:
  if any(x['root_id']==root['root_id']for x in manifest['episodes']):continue
  row=episode(root,'outside_front_supported');manifest['episodes'].append(row);save(B/'manifest.json',manifest);print(json.dumps({k:v for k,v in row.items()if k!='final_labels'}),flush=True)
  if row['error']:raise RuntimeError('Retain infrastructure failure; no replacement')
 assert sum(r['calls']for r in manifest['episodes'])==9
 print('NINE_CLOSE_DIAGNOSTICS_COMPLETED',flush=True)
if __name__=='__main__':main()
