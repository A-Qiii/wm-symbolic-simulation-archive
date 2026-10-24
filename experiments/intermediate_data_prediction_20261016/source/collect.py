"""Fixed-controller physical collection; no model, query, planner or optimizer is loaded."""
from config import *
import sys,time,gzip,traceback,argparse,fcntl
import pybullet as pb
import imageio.v2 as imageio
from PIL import Image,ImageDraw
sys.path.insert(0,str(REP/'source'))
from labels_v2_1 import snapshot,evaluate
sys.path.insert(0,str(P/'scripts'))
from evaluate_physical_oracle import physics_guard
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
class Episode:
 def __init__(self,root,branch):
  self.root=root;self.branch=branch;self.folder=E/'data/raw'/root['root_id']/branch;self.folder.mkdir(parents=True,exist_ok=False)
  for s in ['public','_private']:(self.folder/s).mkdir()
  self.frames=0;self.ticks=0;self.last=None;self.points={};self.calls=[];self.env=None;self.writer=None;self.tracefile=None;self.original=None;self.skill=None;self.call_index=-1;self.segment='reset'
 def start(self):
  self.env=CalvinAdapter(P,self.root['environment'],self.root['seed'],tabletop=True,segmentation=False)
  for obj in self.env.env.scene.movable_objects:
   a=self.root['layout'][obj.name];obj.initial_pos=np.array(a[:3]);obj.initial_orn=np.array(pb.getQuaternionFromEuler(a[3:]))
  for door in self.env.env.scene.doors:
   if door.name=='base__drawer':door.initial_state=0.
  obs=self.env.reset();self.ctl=PhysicalOracle(self.env);self.original=pb.stepSimulation
  def counted(*a,**kw):
   r=self.original(*a,**kw);self.ticks+=1;return r
  pb.stepSimulation=counted;self.writer=imageio.get_writer(self.folder/'continuous.mp4',fps=30,codec='libx264',quality=7,macro_block_size=None,ffmpeg_params=['-movflags','+faststart']);self.tracefile=gzip.open(self.folder/'_private/physical_control_trace.jsonl.gz','wt')
  self.record(obs,{'phase':'reset','gripper':1});self.segment='reset_settle';self.hold(127,1)
  np.savez_compressed(self.folder/'_private/initial_state.npz',**self.env.env.get_state_obs());save(self.folder/'_private/provenance.json',{'root':self.root,'branch':self.branch,'reset_only_initial_placement':True,'reset_is_not_transition':True,'controller_version':self.ctl.VERSION,'controller_sha256':sha(P/'src/wm_symbolic/physical_oracle.py'),'video':'Every control frame,dual camera,30fps simulation time; no interpolation.','physical_trace':'Full object quaternion/position/velocity,drawer pose,contacts,held,control at every recorded control frame.'})
 def point(self,tags,record=None):
  r=self.last if record is None else record;f=r['frame']
  if f not in self.points:
   path=self.folder/'public'/f'{f:06d}.npz';np.savez_compressed(path,**r['obs']);self.points[f]={'frame':f,'ticks':r['ticks'],'public_path':str(path),'call_index':r['call_index'],'skill':r['skill'],'segment':r['segment'],'phase':r['phase'],'physical':r['physical'],'labels':evaluate(r['physical']),'tags':[]}
  for tag in tags:
   if tag not in self.points[f]['tags']:self.points[f]['tags'].append(tag)
  return self.points[f]
 def record(self,obs,control):
  f=self.frames;assert self.ticks==f*8
  phase=control.get('phase','idle');grip=control.get('gripper');key=(self.call_index,self.segment,phase,grip)
  if self.last is not None and key!=self.last['key']:
   tags=[f'phase_end:{self.last["call_index"]}:{self.last["phase"]}']
   if self.last['skill']=='place_in_drawer'and self.last['grip']==-1 and grip==1:tags.append('before_release')
   self.point(tags)
  physical=snapshot(self.env,self.ctl);public={'rgb_static':obs['rgb_obs']['rgb_static'].copy(),'rgb_gripper':obs['rgb_obs']['rgb_gripper'].copy(),'robot_obs':obs['robot_obs'].copy()}
  self.last={'frame':f,'ticks':self.ticks,'call_index':self.call_index,'skill':self.skill,'segment':self.segment,'phase':phase,'grip':grip,'key':key,'obs':public,'physical':physical}
  line={'frame':f,'ticks':self.ticks,'call_index':self.call_index,'skill':self.skill,'segment':self.segment,'control':control,'robot_obs':public['robot_obs'],'physical':physical};self.tracefile.write(json.dumps(line,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x))+'\n')
  im=Image.new('RGB',(400,240),'#14202b');im.paste(Image.fromarray(public['rgb_static']),(0,16));im.paste(Image.fromarray(public['rgb_gripper']).resize((200,200),Image.Resampling.NEAREST),(200,16));d=ImageDraw.Draw(im);d.text((4,2),'STATIC',fill='white');d.text((204,2),'WRIST',fill='white');d.text((4,219),f'{f:05d} | {self.branch} | {self.call_index} {phase}',fill='white');self.writer.append_data(np.asarray(im));self.frames+=1
 def hold(self,n,grip=None):
  grip=int(self.env.observe()['robot_obs'][-1])if grip is None else grip
  with physics_guard(self.env):PhysicalOracle(self.env,self.record,max_controls=n+1).hold(n,grip)
 def call(self,index,skill):
  before=self.point([f'before_call:{index}']);self.call_index=index;self.skill=skill;self.segment='skill';start=self.frames-1
  with physics_guard(self.env):native=PhysicalOracle(self.env,self.record,max_controls=900).run(skill)
  end=self.frames-1;assert end-start==native['controls']and native['controls']<=900 and native['attempts']<=2
  self.point([f'primitive_end:{index}']);self.segment='posthold';self.hold(96);after=self.point([f'after_call:{index}'])
  row={'slot':index,'skill':skill,'status':native['status'],'native':native,'before_frame':before['frame'],'after_frame':after['frame'],'start_frame':start,'primitive_end':end,'end_frame':self.frames-1,'action':action(skill),'executor_arguments':{'skill':skill}};self.calls.append(row);self.flush();return row
 def flush(self):
  save(self.folder/'_private/states.json',list(self.points.values()));save(self.folder/'_private/calls.json',self.calls)
 def close(self):
  if self.last is not None:self.point(['episode_end']);self.flush()
  if self.original is not None:pb.stepSimulation=self.original
  if self.writer is not None:self.writer.close()
  if self.tracefile is not None:self.tracefile.close()
  if self.env is not None:self.env.close()
def episode(root,branch):
 ep=Episode(root,branch);start=time.monotonic();slots=[];error=None
 try:
  ep.start();stopped=False
  for i,skill in enumerate(root['branches'][branch]):
   if stopped:slots.append({'slot':i,'skill':skill,'status':'unreached_prior_failure'});continue
   row=ep.call(i,skill);slots.append(row)
   if(branch=='normal'and row['status']!='success')or(branch=='held_closed'and(row['status']=='failed'or(i==0 and row['status']!='success'))):stopped=True
 except Exception as ex:error={'message':str(ex),'traceback':traceback.format_exc()}
 finally:ep.close()
 result={'root_id':root['root_id'],'environment':root['environment'],'split':root['split'],'target':root['target'],'branch':branch,'slots':slots,'actual_calls':len(ep.calls),'frames':ep.frames,'saved_states':len(ep.points),'error':error,'elapsed_seconds':time.monotonic()-start,'folder':str(ep.folder)};save(ep.folder/'_private/summary.json',result);return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['ABC','D3'],required=True);a=ap.parse_args();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for p,h in read(E/'configs/collection_frozen.json')['sha256'].items():assert sha(p)==h,p
 if a.phase=='D3':
  for p,h in read(E/'configs/before_D3_frozen.json')['sha256'].items():assert sha(p)==h,p
 roots=[r for r in read(E/'configs/roots.json')if(r['environment']=='D')==(a.phase=='D3')];results=[];start=time.monotonic()
 for root in roots:
  for branch in ['normal','held_closed']:
   folder=E/'data/raw'/root['root_id']/branch;sfile=folder/'_private/summary.json'
   if sfile.exists():r=read(sfile)
   else:
    assert not folder.exists(),'Partial episode retained, no replay';assert time.monotonic()-start<90*60,'Soft wall budget reached';r=episode(root,branch)
   results.append(r);save(E/'results'/f'collection_{a.phase}.json',results);print('COLLECT',a.phase,root['root_id'],branch,[x['status']for x in r['slots']],r['frames'],round(r['elapsed_seconds'],1),'seconds',flush=True)
   if r['error']:raise RuntimeError('Infrastructure failure preserved: '+r['error']['message'])
 print('COLLECTION_COMPLETE',a.phase,len(results),'calls',sum(r['actual_calls']for r in results),flush=True)
if __name__=='__main__':main()
