"""Actual selected-action execution; seed0 main/MLP + covered controls.
Three-seed matched predictions are separately frozen in evaluate.py.
"""
from paths import GPU_LOCK
from paths import ROOT as B, RUNTIME as P
import os,json,time,fcntl,hashlib,subprocess
from pathlib import Path
import numpy as np,torch
from models import RGBState,Dynamics
from rgb_front import RGBFront
from evaluate import action,decode
def read(p):return json.load(open(p))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def same_goal(symbol,goal):return symbol['region']==[goal['large'],goal['small']]and not any(symbol['held'])
class Runtime:
 def __init__(self,rec):
  self.rec=rec;self.norm=np.load(Path(rec['checkpoint']).parent/'feature_normalization.npz');self.est=RGBState(1567).cuda();self.est.load_state_dict(torch.load(rec['state_checkpoint'],map_location='cuda',weights_only=False)['model']);self.est.eval().requires_grad_(False);self.net=Dynamics(rec['condition']).cuda();self.net.load_state_dict(torch.load(rec['checkpoint'],map_location='cuda',weights_only=False)['model']);self.net.eval().requires_grad_(False)
 @torch.no_grad()
 def observe(self,feature):
  x=torch.tensor((feature-self.norm['mean'])/self.norm['std'],device='cuda',dtype=torch.float32)[None];return Dynamics.pack(*self.est(x))
 @torch.no_grad()
 def predict(self,state,a):return self.net(state,*action(a),self.est.sd)
 @torch.no_grad()
 def symbols(self,state):return decode(state,self.est)
def programs(task):
 pick=lambda n:['Pick',n,None];place=lambda n,t:['Place',n,t]
 if task=='state':return [[],[place('small','B')],[place('small','A')]]
 if task=='transition':return [[],[pick('large'),place('large','A')],[pick('large'),place('large','B')]]
 return [[],[pick('large'),place('large','B')],[pick('small'),place('small','A'),pick('large'),place('large','B')],[pick('small'),place('small','large_top'),pick('large'),place('large','B')]]
def choose(rt,state,prefix,candidates,goal):
 options=[]
 for i,pr in enumerate(candidates):
  if pr[:len(prefix)]!=prefix:continue
  current=state;preds=[];kept=True
  for a in pr[len(prefix):]:
   current=rt.predict(current,a);sym=rt.symbols(current);preds.append({'action':a,**sym})
   if goal.get('always_small')is not None and sym['region'][1]!=goal['always_small']:kept=False
  final=rt.symbols(current);passing=kept and same_goal(final,goal);options.append({'candidate':i,'program':pr,'remaining':len(pr)-len(prefix),'predictions':preds,'passes':passing})
 good=[x for x in options if x['passes']and x['remaining']>0];selected=min(good,key=lambda x:(x['remaining'],x['candidate']))if good else None;return selected,options
class Worker:
 def __init__(self,out):
  self.err=open(out/'worker_stderr.log','a');self.misc=open(out/'worker_stdout.log','a');self.p=subprocess.Popen(['bash','-c','source "$1"\nshift\nexec "$@"','bash',str(P/'env.sh'),'python','-u',str(B/'src/live_worker.py')],cwd=B,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,bufsize=1)
 def call(self,q):
  self.p.stdin.write(json.dumps(q)+'\n');self.p.stdin.flush()
  while True:
   line=self.p.stdout.readline()
   if not line:raise RuntimeError('Simulator worker ended')
   if line.startswith('BLOCK_RPC '):
    x=json.loads(line[len('BLOCK_RPC '):]);assert x['ok'],x;return x
   self.misc.write(line);self.misc.flush()
 def close(self):
  if self.p.poll()is None:self.call({'command':'close'});self.p.stdin.close();self.p.wait(timeout=30)
  self.err.close();self.misc.close()

def task_specs(plan,experiment,basic=False):
 x=plan['experiments'][experiment];tests=[e for e in x['episodes']if e['split']=='test'];lookup={e['id']:e for e in tests}
 if not basic:
  for e in tests:
   if experiment=='state':large=0 if 'paired_A_'in e['recipe']else 1;goal={'large':large,'small':1,'always_small':None};context=[['Pick','small',None]]
   else:goal={'large':0,'small':0 if'loaded'in e['recipe']else 1,'always_small':None if'loaded'in e['recipe']else 1};context=[]
   yield {'id':e['id'],'group':e['group'],'layout':e['layout'],'goal':goal,'context':context,'task':experiment,'split':'test','purpose':'held joint-state generalization'if experiment=='state'else'withheld loaded reverse transfer'}
 else:
  for group in x['groups']:
   if group['split']!='test':continue
   # Same held-out layout group; original A-start two-goal tasks are distinct
   # frozen tasks, not training examples.
   dx=next(e['layout']['large_xy'][0]-.18 for e in tests if e['group']==group['id']);dy=next(e['layout']['large_xy'][1]+.14 for e in tests if e['group']==group['id']);yaw=next(e['layout']['yaw']for e in tests if e['group']==group['id'])
   for stacked in [True,False]:
    original=next(e for e in tests if e['group']==group['id']and (('loaded'in e['recipe'])==stacked));layout={**original['layout'],'large_xy':[-.08+dx,-.14+dy],'stacked':stacked};layout.pop('small_xy',None)
    for goalid in [1,2]:yield {'id':group['id']+f'_basic_{"stacked"if stacked else"beside"}_goal{goalid}','group':group['id'],'layout':layout,'goal':{'large':1,'small':0 if goalid==1 else 1,'always_small':0 if goalid==1 else None},'context':[],'task':'basic','split':'test','purpose':'covered multi-action task ability','goal_id':goalid,'stacked_private':stacked}

def main():
 frozen=read(B/'configs/all_models_frozen.json');assert frozen['development_gate_passed'];assert(B/'reports/v1/confirmation/complete.json').exists();plan=read(B/'configs/experiment_plan.json');out=B/'reports/v1/live';out.mkdir(parents=True,exist_ok=True);front=RGBFront();worker=Worker(out);records=[]
 try:
  modelrecords=[r for r in frozen['models']if r['seed']==0]
  for rec in modelrecords:
   assert hashlib.sha256(Path(rec['checkpoint']).read_bytes()).hexdigest()==rec['sha256'];rt=Runtime(rec);modelid=f'{rec["experiment"]}_{rec["arm"]}_{rec["condition"]}_seed0';specs=list(task_specs(plan,rec['experiment']))
   if rec['experiment']=='transition'and rec['arm']=='holdout'and rec['condition']=='motion_matrix':specs+=list(task_specs(plan,'transition',True))
   for i,spec in enumerate(specs):
    folder=out/modelid/spec['id']
    if(folder/'policy.json').exists()and(folder/'score_private.json').exists():records.append({'model':modelid,'case':spec['id'],'score_file':str(folder/'score_private.json')});continue
    if folder.exists():raise RuntimeError('Incomplete live execution retained: '+str(folder))
    capture=(rec['arm']=='holdout'and rec['condition']=='motion_matrix'and(i<2 or(spec['task']=='basic'and spec.get('goal_id')==1 and spec.get('stacked_private')and i<44)))
    reply=worker.call({'command':'start','folder':str(folder),'spec':spec,'capture':capture});log=[];prefix=[];stop='invalid_initialization';next_reply=reply
    if reply['initial_valid']:
     stop=None
     for a in spec['context']:
      next_reply=worker.call({'command':'execute','action':a})
      if next_reply['status']!='completed':stop='context_skill_failure';break
     if stop is None:
      while True:
       raw,_=front.extract(next_reply['public_paths']);state=rt.observe(raw);current=rt.symbols(state);selected,cands=choose(rt,state,prefix,programs(spec['task']),spec['goal']);entry={'prefix':prefix.copy(),'current':current,'selected':selected,'candidates':cands,'public_paths':next_reply['public_paths'],'control_count':next_reply['control_count'],'feature_sha256':hashlib.sha256(raw.tobytes()).hexdigest()}
       if spec['task']in ['state','transition']:
        boundary=len(prefix)+len(spec['context']);cache=B/'cache/v1/test'/f'{spec["id"]}__b{boundary}.npz'
        reference=next(e for e in plan['experiments'][rec['experiment']]['episodes']if e['id']==spec['id'])['actions'];actual=spec['context']+prefix
        if cache.exists()and reference[:len(actual)]==actual:
         cached=np.load(cache)['features'];entry['feature_max_abs_diff_vs_same_reference_boundary']=float(np.max(np.abs(raw-cached)));entry['feature_mean_abs_diff_vs_same_reference_boundary']=float(np.mean(np.abs(raw-cached)))
       log.append(entry)
       if same_goal(current,spec['goal']):stop='goal_reached';break
       if spec['goal'].get('always_small')is not None and current['region'][1]!=spec['goal']['always_small']:stop='model_current_constraint_violation';break
       if selected is None:stop='no_feasible_prediction';break
       a=selected['program'][len(prefix)];next_reply=worker.call({'command':'execute','action':a});prefix.append(a)
       if next_reply['status']!='completed':stop='low_level_skill_failure';break
       if len(prefix)>4:raise RuntimeError('Candidate depth invariant violated')
    save(folder/'policy.json',{'model':rec,'task_spec':spec,'context_actions':spec['context'],'selected_actions':prefix,'stop':stop,'boundaries':log,'private_initialization_metadata_seen_by_model':False,'GT_inputs_to_model':False});worker.call({'command':'finish','stop':stop});records.append({'model':modelid,'case':spec['id'],'score_file':str(folder/'score_private.json')});save(B/'STATUS.json',{'stage':'live_tasks','completed':len(records),'model':modelid,'case':spec['id'],'stop':stop});print('LIVE',len(records),modelid,spec['id'],stop,flush=True)
   del rt;torch.cuda.empty_cache()
 finally:worker.close();front.close()
 # Read independent scores only after all decisions are complete.
 result=[]
 for row in records:result.append({**row,**read(Path(row['score_file']))})
 save(out/'all_results.json',result);save(out/'complete.json',{'tasks':len(result),'success':sum(r['success']for r in result),'seed':0,'three_seed_offline_prediction_results_separate':True,'main_model_basic_tasks':80,'combination_tasks_per_model':40,'models':8,'no_test_tuning':True});save(B/'STATUS.json',{'stage':'live_tasks_complete','tasks':len(result),'next':'results analysis and final model demonstration videos'});print('LIVE_COMPLETE',len(result),flush=True)

if __name__=='__main__':
 lock=GPU_LOCK.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 try:main()
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
