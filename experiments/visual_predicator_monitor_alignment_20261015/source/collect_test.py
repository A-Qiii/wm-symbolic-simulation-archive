"""Real physics, frozen learned rules, separate visual and explicitly privileged GT arms."""
from common import *
from symbolic import search,goal_satisfied,col
import sys,time,gzip,traceback,subprocess,argparse
import pybullet as p
sys.path.insert(0,str(P/'scripts'))
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
from core1200_collect import label,plain
from evaluate_physical_oracle import physics_guard
def targets(lab):
 c=np.array(lab['categorical'][:4]);v=np.array(lab['categorical_valid'][:4],bool);return {'y':np.r_[c[:3]==6,c[:3]==3,c[3]==2].tolist(),'v':np.r_[v[:3],v[:3],v[3]].tolist()}
def near(a,b):
 d=np.linalg.norm(a[:,:3]-b[:,:3],axis=1);yaw=np.abs((a[:,5]-b[:,5]+np.pi/2)%np.pi-np.pi/2);return bool(np.all(d<=.012)or(np.all(d<=.020)and np.all(yaw<=.25)))
class Worker:
 def __init__(self):
  self.log=(E/'logs'/f'test_worker_{time.time_ns()}.log').open('w');cmd=f'source {P}/world_model_env.sh\nexec python {E}/source/worker.py';self.proc=subprocess.Popen(['/bin/bash','-c',cmd],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1);line=self.proc.stdout.readline();assert line and json.loads(line).get('ready'),'worker start failed'
 def request(self,req):
  self.proc.stdin.write(json.dumps(req)+'\n');self.proc.stdin.flush();line=self.proc.stdout.readline();assert line,'worker stopped';r=json.loads(line);assert r.get('ok'),r;return r
 def close(self):
  self.proc.stdin.write('{"op":"stop"}\n');self.proc.stdin.flush();assert self.proc.wait(timeout=30)==0;self.log.close()
def collect(root,arm,worker,prior):
 eid=root['root_id']+'__'+arm;folder=B/'episodes'/eid;folder.mkdir();(folder/'public').mkdir();(folder/'_private').mkdir();save(folder/'_private/provenance.json',{'root':root,'arm':arm});trace=[];windows=[];actions=[];decisions=[];window_cache={};env=None;original_step=None;ticks=0;segment='reset';error=None;reason=None;initial=None
 index=(folder/'public/frames.jsonl').open('w');commands=(folder/'public/commands.jsonl').open('w');operators=read(E/'configs/rules_frozen.json')['operators']
 try:
  env=CalvinAdapter(P,'D',root['seed'],tabletop=True,segmentation=False)
  for obj in env.env.scene.movable_objects:
   ar=root['layout'][obj.name];obj.initial_pos=np.array(ar[:3]);obj.initial_orn=np.array(p.getQuaternionFromEuler(ar[3:]))
  for door in env.env.scene.doors:
   if door.name=='base__drawer':door.initial_state=root['drawer_reset_m']
  ob=env.reset();ctl0=PhysicalOracle(env);initial=np.array([ctl0.objects['block_'+c].get_state()for c in COLORS]);assert not any(near(initial,x)for x in prior);np.savez_compressed(folder/'_private/initial_state.npz',**env.env.get_state_obs());assert env.env.control_freq==30 and env.env.action_repeat==8;original_step=p.stepSimulation
  def counted(*a,**kw):
   nonlocal ticks
   out=original_step(*a,**kw);ticks+=1;return out
  p.stepSimulation=counted
  def record(obs,action):
   fi=len(trace);assert ticks==fi*8;lab=label(env);r={'frame':fi,'physics_ticks':ticks,'sim_seconds':ticks/240,'segment':segment,'action_private':action,'targets':targets(lab),'labels_private':lab};trace.append(r);fp=folder/'public'/f'{fi:05d}.npz';np.savez_compressed(fp,rgb_static=obs['rgb_obs']['rgb_static'],rgb_gripper=obs['rgb_obs']['rgb_gripper'],robot_obs=obs['robot_obs']);index.write(json.dumps({'frame':fi,'physics_ticks':ticks,'sim_seconds':ticks/240,'file':fp.name,'sha256':sha(fp)})+'\n')
   if 'target_xyz'in action:commands.write(json.dumps({'arrival_frame':fi,**{k:action[k]for k in ['target_xyz','target_quaternion','gripper']}})+'\n')
  record(ob,{'phase':'reset_only'})
  def hold(name,grip):
   nonlocal segment
   segment=name;ctl=PhysicalOracle(env,record);ctl.phase=name;start=len(trace)-1
   with physics_guard(env):ctl.hold(96,grip)
   assert len(trace)-1-start==96
  def observe(anchor,role):
   if anchor in window_cache:window_cache[anchor]['roles'].append(role);return window_cache[anchor]
   inds=[anchor-k for k in [63,42,21,0]];assert min(inds)>=0;wid=eid+f'__t{anchor:05d}';paths=[str(folder/'public'/f'{i:05d}.npz')for i in inds];tickbefore=ticks;result=worker.request({'window_id':wid,'public_frames':paths});assert ticks==tickbefore;w={'window_id':wid,'episode_id':eid,'root_id':root['root_id'],'arm':arm,'anchor_frame':anchor,'history_frame_indices':inds,'public_frames':paths,'roles':[role],'inference':result};windows.append(w);window_cache[anchor]=w;save(folder/'public'/(wid+'.json'),w);return w
  hold('initial_hold',1);cur=observe(len(trace)-1,'planning_endpoint');seen=set()
  for step in range(8):
   truth=trace[-1]['targets'];state=cur['inference']['decoded']if arm=='visual_learned_rules'else[int(y)if v else-1 for y,v in zip(truth['y'],truth['v'])];decision=search(state,root['goal'],operators,depth=8-step,expansions=512);dr={'step':step,'window_id':cur['window_id'],'input_state':state,'source':'public_visual_inference'if arm=='visual_learned_rules'else'explicit_GT_symbol_diagnostic','goal':root['goal'],'search':decision,'decided_wall_ns':time.time_ns()};decisions.append(dr);save(folder/'public/decisions.json',decisions)
   if decision['status']!='solved':reason=decision['status'];break
   if not decision['plan']:reason='reported_goal_reached';break
   chosen=decision['plan'][0];key=(tuple(state),chosen['family'],chosen['object'])
   if key in seen:reason='repeated_state_action';break
   seen.add(key);family=chosen['family'];x=chosen['object'];skill='open_drawer'if family=='open'else'close_drawer'if family=='close'else'place_in_drawer'if family=='place'else'lift_'+COLORS[x]+'_block_table';start=len(trace)-1;beforecur=cur;startwall=time.time_ns();assert beforecur['inference']['completed_wall_ns']<startwall;declaration={'step':step,'skill':skill,'family':family,'object':x,'predicted_after':chosen['predicted_state'],'before_window_id':cur['window_id'],'action_start_wall_ns':startwall,'decision_sha256':sha(folder/'public/decisions.json'),'start_frame':start};save(folder/'public'/f'declaration_{step}.json',declaration);segment='skill_'+str(step);ctl=PhysicalOracle(env,record)
   with physics_guard(env):res=ctl.run(skill)
   primitive_end=len(trace)-1;assert primitive_end-start==res['controls']and res['controls']<=900;hold('posthold_'+str(step),int(env.observe()['robot_obs'][-1]));end=len(trace)-1;cur=observe(end,'planning_endpoint')
   ar=declaration|{'after_window_id':cur['window_id'],'primitive_end_frame':primitive_end,'end_frame':end,'primitive':res,'macro_controls':end-start,'before_targets':trace[start]['targets'],'after_targets':trace[end]['targets'],'after_inference_started_wall_ns':cur['inference']['started_wall_ns']};actions.append(ar);save(folder/'_private/actions.json',actions)
   # Preregistered intermediate perception probes do not enter planning.
   rows=trace[start+1:primitive_end+1];nearclosed=[r['frame']for r in rows if r['action_private'].get('phase')=='descend_named_block'and r['action_private'].get('gripper')==-1];lower=[r['frame']for r in rows if r['action_private'].get('phase')=='lower_into_drawer']
   if nearclosed and nearclosed[0]>=63:observe(nearclosed[0],'contact_probe')
   if len(lower)>=64 and lower[63]-lower[0]==63:observe(lower[63],'lowering_probe')
   if res['status']!='success':
    # Keep native failed status. Reconcile only an actually observed learned effect;
    # no GT is consulted in the visual arm, no controller behavior is changed.
    aftertruth=trace[end]['targets'];observed=cur['inference']['decoded']if arm=='visual_learned_rules'else[int(y)if v else-1 for y,v in zip(aftertruth['y'],aftertruth['v'])]
    op=next(o for o in operators if o['id']==chosen['operator_id']);checks=[observed[col(atom,x)]==1 for atom in op['add']]+[observed[col(atom,x)]==0 for atom in op['delete']]
    confirmed=res['status']=='failed'and bool(checks)and all(checks)
    ar['monitor_alignment']={'native_status_unchanged':res['status'],'learned_effects_observed':bool(confirmed),'checks':checks,'source':'visual_query'if arm=='visual_learned_rules'else'explicit_GT_diagnostic','action':'replan_from_observation'if confirmed else'stop'};save(folder/'_private/actions.json',actions)
    if not confirmed:reason='controller_'+res['status'];break
  else:reason='call_budget_exhausted'
  final=trace[-1]['targets'];actual=bool(all(final['v'][int(j)]and final['y'][int(j)]==bool(value)for j,value in root['goal']));np.savez_compressed(folder/'_private/final_state.npz',**env.env.get_state_obs())
 except Exception as ex:error={'type':type(ex).__name__,'message':str(ex),'traceback':traceback.format_exc()};save(folder/'_private/error.json',error);actual=False
 finally:
  if original_step is not None:p.stepSimulation=original_step
  if env is not None:env.close()
  index.close();commands.close()
  with gzip.open(folder/'_private/trace.json.gz','wt')as f:json.dump(plain(trace),f)
 for w in windows:w['targets_evaluator_only']=trace[w['anchor_frame']]['targets']
 save(folder/'_private/windows.json',windows);summary={'root_id':root['root_id'],'episode_id':eid,'arm':arm,'goal':root['goal'],'goal_text':root['goal_text'],'calls':len(actions),'frames':len(trace),'controls':len(trace)-1,'physics_ticks':ticks,'windows':len(windows),'actual_goal_success':actual,'stop_reason':reason,'reported_goal_success':reason=='reported_goal_reached','false_completion':reason=='reported_goal_reached'and not actual,'collection_error':error,'initial_objects':initial.tolist()if initial is not None else None,'action_statuses':[a['primitive']['status']for a in actions]};save(folder/'_private/summary.json',summary);return summary
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','remaining'],required=True);args=ap.parse_args();q=read(E/'configs/test_frozen.json')
 for path,h in q['frozen_sha256'].items():assert sha(path)==h,path
 if args.phase=='remaining':assert read(E/'results/test_pilot_validation.json')['passed']
 m=read(B/'data_manifest.json');prior=[np.array(r['layout'])for r in read(E/'configs/root_exclusion.json')['prior_layouts']];worker=Worker()
 try:
  for root in read(B/'roots_manifest.json'):
   if(root['root_id']==q['pilot_root'])!=(args.phase=='pilot'):continue
   others={r['root_id']:np.array(r['initial_objects'])for r in m['episodes']if r['root_id']!=root['root_id']and r['initial_objects']is not None}
   for arm in q['arms']:
    eid=root['root_id']+'__'+arm
    if any(r['episode_id']==eid for r in m['episodes']):continue
    s=collect(root,arm,worker,prior+list(others.values()));m['episodes'].append(s);save(B/'data_manifest.json',m);save(E/'results/test_progress.json',{'episodes':len(m['episodes']),'last':s,'calls':sum(x['calls']for x in m['episodes'])});print(json.dumps(s,ensure_ascii=False),flush=True)
    if s['collection_error']:raise RuntimeError('Infrastructure error preserved; no retry/replacement')
 finally:worker.close()
if __name__=='__main__':main()
