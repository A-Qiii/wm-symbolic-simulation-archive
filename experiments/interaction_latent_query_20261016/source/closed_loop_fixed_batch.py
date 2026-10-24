from common import *
from labels_v2_1 import snapshot,evaluate
import pybullet as pb,fcntl,time,gzip,sys,traceback,subprocess
sys.path.insert(0,str(P/'scripts'))
from evaluate_physical_oracle import physics_guard
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
class Worker:
 def __init__(self):
  self.log=(E/'logs'/f'bridge_{time.time_ns()}.log').open('w');cmd=f'source {P}/world_model_env.sh\nexec python {E}/source/worker_fixed_batch.py';self.proc=subprocess.Popen(['/bin/bash','-c',cmd],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1);line=self.proc.stdout.readline();assert line and json.loads(line).get('ready'),'worker start failed'
 def request(self,req):
  self.proc.stdin.write(json.dumps(req)+'\n');self.proc.stdin.flush();line=self.proc.stdout.readline();assert line,'worker stopped';r=json.loads(line);assert r.get('ok'),r;return r
 def close(self):
  self.proc.stdin.write('{"op":"stop"}\n');self.proc.stdin.flush();assert self.proc.wait(timeout=30)==0;self.log.close()
def run(root,kind,worker):
 folder=E/'data/closed_loop'/root['root_id']/kind;folder.mkdir(parents=True,exist_ok=False);(folder/'public').mkdir();(folder/'_private').mkdir();env=None;trace=[];decisions=[];actions=[];original=None;ticks=0;error=None;start=time.monotonic();obs=None
 try:
  env=CalvinAdapter(P,'D',root['seed'],tabletop=True,segmentation=False)
  for obj in env.env.scene.movable_objects:val=root['layout'][obj.name];obj.initial_pos=np.array(val[:3]);obj.initial_orn=np.array(pb.getQuaternionFromEuler(val[3:]))
  for door in env.env.scene.doors:
   if door.name=='base__drawer':door.initial_state=root['drawer_reset_m']
  obs=env.reset();ctl=PhysicalOracle(env);original=pb.stepSimulation
  def counted(*a,**kw):
   nonlocal ticks
   r=original(*a,**kw);ticks+=1;return r
  pb.stepSimulation=counted
  def record(ob,ac):
   assert ticks==len(trace)*8;physical=snapshot(env,ctl);trace.append({'frame':len(trace),'control':ac,'physical':physical,'labels':evaluate(physical)})
  record(obs,{'phase':'reset'})
  with physics_guard(env):PhysicalOracle(env,record,max_controls=128).hold(127,1)
  np.savez_compressed(folder/'_private/initial_state.npz',**env.env.get_state_obs());initial=evaluate(snapshot(env,ctl))
  for step in range(7):
   obs=env.observe();path=folder/'public'/f'boundary{step:02d}.npz';np.savez_compressed(path,rgb_static=obs['rgb_obs']['rgb_static'],rgb_gripper=obs['rgb_obs']['rgb_gripper'],robot_obs=obs['robot_obs'])
   tickbefore=ticks;response=worker.request({'kind':kind,'public_path':str(path),'target':COLORS.index(root['color'])});assert ticks==tickbefore;pr=response['current_probability'];plan=response['search']
   decision={'step':step,'current_probability':pr,'search':plan,'public_path':str(path),'actual_labels_evaluator_only':evaluate(snapshot(env,ctl))};decisions.append(decision);save(folder/'_private/decisions.json',decisions)
   if plan['status']!='plan_found'or step==6:break
   choice=plan['plan'][0];frame=len(trace)-1
   with physics_guard(env):native=PhysicalOracle(env,record).run(choice['skill'])
   primitive_end=len(trace)-1
   with physics_guard(env):PhysicalOracle(env,record,max_controls=97).hold(96,int(env.observe()['robot_obs'][-1]))
   actions.append({'step':step,'skill':choice['skill'],'declared_target':choice['target'],'start_frame':frame,'primitive_end':primitive_end,'end_frame':len(trace)-1,'native':native});save(folder/'_private/actions.json',actions)
  final=evaluate(snapshot(env,ctl));target=COLORS.index(root['color']);success=bool(final['v'][3+target]and final['v'][6]and final['y'][3+target]and final['y'][6]);reported=decisions[-1]['search']['status']=='perceived_goal_satisfied';np.savez_compressed(folder/'_private/final_state.npz',**env.env.get_state_obs())
 except Exception as exc:error={'message':str(exc),'traceback':traceback.format_exc()};initial=None;final=None;success=False;reported=False
 finally:
  if original is not None:pb.stepSimulation=original
  if env is not None:env.close()
  with gzip.open(folder/'_private/trace.json.gz','wt')as f:json.dump(trace,f)
  summary={'root_id':root['root_id'],'kind':kind,'seed':0,'target':root['color'],'initial_labels':initial,'final_labels':final,'physical_strict_goal_success':success,'reported_completion':reported,'false_completion':reported and not success,'actions':len(actions),'decisions':len(decisions),'last_status':decisions[-1]['search']['status']if decisions else None,'native_statuses':[a['native']['status']for a in actions],'physics_ticks':ticks,'controls':len(trace)-1,'elapsed_seconds':time.monotonic()-start,'error':error};save(folder/'_private/summary.json',summary)
 return summary
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for p,h in read(E/'configs/final_before_D.json')['sha256'].items():assert sha(p)==h,p
 for p,h in read(E/'configs/bridge_frozen.json')['sha256'].items():assert sha(p)==h,p
 assert all(sha(p)==h for p,h in read(E/'configs/inference_batch_frozen.json')['sha256'].items());assert read(E/'results/runtime_bridge_preflight.json')['passed'];results=[];roots=[r for r in read(E/'data/D_raw/roots_manifest.json')if r['closed_loop_matched']];worker=Worker()
 try:
  for kind in ['generic','ae','interaction']:
   for root in roots:
    pp=E/'data/closed_loop'/root['root_id']/kind/'_private/summary.json'
    r=read(pp)if pp.exists()else run(root,kind,worker);results.append(r);save(E/'results/closed_loop.json',results);print({k:v for k,v in r.items()if k not in ['initial_labels','final_labels']},flush=True)
    if r['error']:raise RuntimeError('Preserved infrastructure failure, no replacement')
 finally:worker.close()
 print('CLOSED_LOOP_COMPLETE',len(results),flush=True)
if __name__=='__main__':main()
