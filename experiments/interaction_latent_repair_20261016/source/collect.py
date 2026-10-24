from common import *
from labels_v2_1 import snapshot,evaluate
import pybullet as pb,fcntl,time,gzip,sys,traceback,subprocess,argparse
import imageio.v2 as imageio
from PIL import Image,ImageDraw
sys.path.insert(0,str(P/'scripts'))
from evaluate_physical_oracle import physics_guard
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
ARMS=['canonical_control','canonical_matched']
class Worker:
 def __init__(self):
  self.log=(E/'logs'/f'worker_{time.time_ns()}.log').open('w');cmd=f'source {P}/world_model_env.sh\nexec python {E}/source/worker.py';self.proc=subprocess.Popen(['/bin/bash','-c',cmd],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1);line=self.proc.stdout.readline();assert line and json.loads(line).get('ready'),line
 def request(self,r):
  self.proc.stdin.write(json.dumps(r)+'\n');self.proc.stdin.flush();line=self.proc.stdout.readline();assert line,'worker stopped';v=json.loads(line);assert v.get('ok'),v;return v
 def close(self):
  self.proc.stdin.write('{"op":"stop"}\n');self.proc.stdin.flush();assert self.proc.wait(timeout=30)==0;self.log.close()
class Episode:
 def __init__(self,root,condition,folder,q):
  self.root=root;self.condition=condition;self.folder=folder;folder.mkdir(parents=True,exist_ok=False);(folder/'public').mkdir();(folder/'_private').mkdir();self.trace=[];self.calls=[];self.boundaries=[];self.ticks=0;self.segment='settle';self.env=None;self.original=None;self.writer=None
  self.env=CalvinAdapter(P,'D',root['seed'],tabletop=True,segmentation=False)
  for obj in self.env.env.scene.movable_objects:
   v=root['layout'][obj.name];obj.initial_pos=np.array(v[:3]);obj.initial_orn=np.array(pb.getQuaternionFromEuler(v[3:]))
  for door in self.env.env.scene.doors:
   if door.name=='base__drawer':door.initial_state=q
  obs=self.env.reset();self.ctl=PhysicalOracle(self.env);self.original=pb.stepSimulation
  def counted(*a,**kw):
   r=self.original(*a,**kw);self.ticks+=1;return r
  pb.stepSimulation=counted;self.writer=imageio.get_writer(folder/'continuous.mp4',fps=30,codec='libx264',quality=7,macro_block_size=None,ffmpeg_params=['-movflags','+faststart']);self.record(obs,{'phase':'reset'});self.hold(127,'settle',1)
  save(folder/'_private/provenance.json',{'root':root,'condition':condition,'drawer_reset_m':q,'real_controls_only':True,'video':'Every control observation,30fps simulation time; no interpolation; wrist nearest-neighbour resized; fixed camera200x200 retained; compressed H264.'});np.savez_compressed(folder/'_private/initial_state.npz',**self.env.env.get_state_obs())
 def record(self,obs,control):
  f=len(self.trace);assert self.ticks==f*8;self.trace.append({'frame':f,'ticks':self.ticks,'segment':self.segment,'control':control});im=Image.new('RGB',(400,240),'#14202b');im.paste(Image.fromarray(obs['rgb_obs']['rgb_static']),(0,16));im.paste(Image.fromarray(obs['rgb_obs']['rgb_gripper']).resize((200,200),Image.Resampling.NEAREST),(200,16));dr=ImageDraw.Draw(im);dr.text((4,2),'STATIC',fill='white');dr.text((204,2),'WRIST',fill='white');dr.text((4,219),f'frame {f} | {f/30:.2f}s | {self.segment}',fill='white');self.writer.append_data(np.asarray(im))
 def hold(self,n,name,grip=None):
  self.segment=name
  with physics_guard(self.env):PhysicalOracle(self.env,self.record,max_controls=n+1).hold(n,int(self.env.observe()['robot_obs'][-1])if grip is None else grip)
 def boundary(self,name):
  path=self.folder/'public'/f'{name}.npz';obs=self.env.observe();np.savez_compressed(path,rgb_static=obs['rgb_obs']['rgb_static'],rgb_gripper=obs['rgb_obs']['rgb_gripper'],robot_obs=obs['robot_obs']);physical=snapshot(self.env,self.ctl);row={'name':name,'public_path':str(path),'frame':len(self.trace)-1,'physical':physical,'labels':evaluate(physical)};self.boundaries.append(row);save(self.folder/'_private/boundaries.json',self.boundaries);return row
 def call(self,skill,role):
  start=len(self.trace)-1;self.segment=role+'_'+skill
  with physics_guard(self.env):native=PhysicalOracle(self.env,self.record,max_controls=900).run(skill)
  end=len(self.trace)-1;assert end-start==native['controls']and native['controls']<=900;assert native['attempts']<=2;self.hold(96,role+'_posthold');r={'role':role,'skill':skill,'start_frame':start,'primitive_end':end,'end_frame':len(self.trace)-1,'native':native,'action':action(skill,None,96),'executor_arguments':{'skill':skill}};self.calls.append(r);save(self.folder/'_private/calls.json',self.calls);return r
 def close(self):
  if self.original is not None:pb.stepSimulation=self.original
  if self.writer is not None:self.writer.close()
  if self.env is not None:self.env.close()
  with gzip.open(self.folder/'_private/control_trace.json.gz','wt')as f:json.dump(self.trace,f)
  save(self.folder/'_private/calls.json',self.calls)
def offline_episode(root,condition):
 folder=E/'data/D2_raw'/root['root_id']/condition;ep=None;tasks=[];error=None;start=time.monotonic();planned=root['normal_reference_skills']if condition=='reference'else['place_in_drawer'];prep=[];prep_ok=True
 try:
  q=root['drawer_reset_m']if condition=='reference'else(.22 if condition.endswith('_open')else 0.);ep=Episode(root,condition,folder,q)
  if condition.startswith('held_'):
   p=ep.call('lift_'+root['target']+'_block_table','preparation');prep.append(p);lab=evaluate(snapshot(ep.env,ep.ctl));t=COLORS.index(root['target']);prep_ok=p['native']['status']=='success'and lab['y'][t]and sum(lab['y'][:3])==1 and lab['y'][7 if condition.endswith('_open')else 6]
  elif condition.startswith('empty_'):
   lab=evaluate(snapshot(ep.env,ep.ctl));prep_ok=not any(lab['y'][:3])and lab['y'][7 if condition.endswith('_open')else 6]
  if not prep_ok:
   for i,skill in enumerate(planned):tasks.append({'slot':i,'skill':skill,'status':'unreached_preparation','native':None})
  else:
   before=ep.boundary('boundary00');stopped=False
   for i,skill in enumerate(planned):
    if stopped:tasks.append({'slot':i,'skill':skill,'status':'unreached_prior_failure','native':None});continue
    r=ep.call(skill,'task');after=ep.boundary(f'boundary{i+1:02d}');tasks.append({'slot':i,'skill':skill,'status':r['native']['status'],'native':r['native'],'before':before,'after':after,'call':r});before=after
    if condition=='reference'and r['native']['status']!='success':stopped=True
 except Exception as ex:error={'message':str(ex),'traceback':traceback.format_exc()}
 finally:
  if ep:ep.close()
 summary={'root_id':root['root_id'],'condition':condition,'target':root['target'],'planned_tasks':len(planned),'tasks':tasks,'preparation':prep,'preparation_valid':prep_ok,'error':error,'elapsed_seconds':time.monotonic()-start,'frames':len(ep.trace)if ep else 0,'folder':str(folder)};save(folder/'_private/summary.json',summary);return summary
def closed_episode(root,arm,worker):
 folder=E/'data/closed_loop'/root['root_id']/arm;ep=None;decisions=[];error=None;start=time.monotonic();success=False;reported=False;final=None
 try:
  ep=Episode(root,'closed_loop',folder,root['drawer_reset_m'])
  for step in range(7):
   b=ep.boundary(f'boundary{step:02d}');ticks=ep.ticks;response=worker.request({'arm':arm,'public_path':b['public_path'],'target':COLORS.index(root['target'])});assert ep.ticks==ticks;decisions.append({'step':step,'current_probability':response['current_probability'],'search':response['search'],'public_path':b['public_path'],'actual_labels_evaluator_only':b['labels'],'frame':b['frame']});save(folder/'_private/decisions.json',decisions);plan=response['search']
   if plan['status']!='plan_found'or step==6:break
   choice=plan['plan'][0];assert choice['skill']!='place_in_drawer'or choice['target']is None;ep.call(choice['skill'],'task')
  final=ep.boundaries[-1]['labels'];j=3+COLORS.index(root['target']);success=bool(final['v'][j]and final['v'][6]and final['y'][j]and final['y'][6]);reported=decisions[-1]['search']['status']=='perceived_goal_satisfied'
 except Exception as ex:error={'message':str(ex),'traceback':traceback.format_exc()}
 finally:
  if ep:ep.close()
 summary={'root_id':root['root_id'],'arm':arm,'seed':0,'target':root['target'],'initial_open':root['drawer_reset_m']>.18,'physical_success':success,'reported_completion':reported,'false_completion':reported and not success,'calls':len(ep.calls)if ep else 0,'native_statuses':[r['native']['status']for r in ep.calls]if ep else[],'final_labels':final,'error':error,'frames':len(ep.trace)if ep else 0,'elapsed_seconds':time.monotonic()-start,'video':str(folder/'continuous.mp4')};save(folder/'_private/summary.json',summary);return summary
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['offline','closed_loop'],required=True);a=ap.parse_args();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for p,h in read(E/'configs/before_D2_frozen.json')['sha256'].items():assert sha(p)==h,p
 roots=read(E/'configs/D2_roots.json');results=[];worker=Worker()if a.phase=='closed_loop'else None
 try:
  for root in roots:
   for case in(['reference','held_open','held_closed','empty_open','empty_closed']if a.phase=='offline'else ARMS):
    folder=(E/'data/D2_raw'/root['root_id']/case)if a.phase=='offline'else(E/'data/closed_loop'/root['root_id']/case);summary=folder/'_private/summary.json'
    if summary.exists():r=read(summary)
    else:
     assert not folder.exists(),'Partial episode exists; preserve it, do not rerun';r=offline_episode(root,case)if a.phase=='offline'else closed_episode(root,case,worker)
    results.append(r);save(E/'results'/('D2_collection.json'if a.phase=='offline'else'closed_loop.json'),results);print('COLLECT',a.phase,root['root_id'],case,'error',r['error'],'frames',r['frames'],flush=True)
    if r['error']:raise RuntimeError('Preserved infrastructure failure; no replacement')
 finally:
  if worker:worker.close()
 if a.phase=='offline':assert sum(r['planned_tasks']for r in results)==90;assert sum(len(r['preparation'])for r in results)<=24
 else:assert len(results)==24 and sum(r['calls']for r in results)<=144
 print('COLLECTION_COMPLETE',a.phase,len(results),flush=True)
if __name__=='__main__':main()
