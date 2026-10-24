from experiment import *
import sys,gzip,time,fcntl,traceback,argparse,subprocess
import numpy as np
import pybullet as p
from omegaconf import OmegaConf
sys.path.insert(0,str(P/'scripts'))
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
from core1200_collect import label,plain
from evaluate_physical_oracle import physics_guard
def targets(lab):
 cat=np.array(lab['categorical'][:4]);v=np.array(lab['categorical_valid'][:4],bool)
 return {'y':np.r_[cat[:3]==6,cat[:3]==3,cat[3]==2].tolist(),'valid':np.r_[v[:3],v[:3],v[3]].tolist()}
def near(a,b):
 d=np.linalg.norm(a[:,:3]-b[:,:3],axis=1);yaw=np.abs((a[:,5]-b[:,5]+np.pi/2)%np.pi-np.pi/2)
 return bool(np.all(d<=.012)or(np.all(d<=.020)and np.all(yaw<=.25)))
def collect(root,branch,prior,own_initial,worker):
 eid=branch['episode_id'];folder=B/'episodes'/eid;folder.mkdir();(folder/'public').mkdir();(folder/'_private').mkdir();save(folder/'_private/provenance.json',{'root':root,'branch':branch})
 trace=[];preparations=[];pairs=[];private_pairs=[];windows=[];failure=None;env=None;original_step=None;ticks=0;segment='reset';started=time.monotonic();initial=None;declaration=None;coverage_reason=None;holds=[]
 index=(folder/'public/frames.jsonl').open('w');commands=(folder/'public/commands.jsonl').open('w')
 try:
  env=CalvinAdapter(P,root['environment'],root['seed'],tabletop=True,segmentation=False)
  for obj in env.env.scene.movable_objects:
   a=root['layout'][obj.name];obj.initial_pos=np.array(a[:3]);obj.initial_orn=np.array(p.getQuaternionFromEuler(a[3:]))
  for door in env.env.scene.doors:
   if door.name=='base__drawer':door.initial_state=root['drawer_reset_m']
  ob=env.reset();auditctl=PhysicalOracle(env);initial=np.array([auditctl.objects['block_'+c].get_state()for c in COLORS]);assert not any(near(initial,a)for a,_ in prior),'root_near_previous'
  if own_initial is not None:assert float(np.abs(initial-own_initial).max())<=1e-5,'matched_branch_initial_objects_mismatch'
  state=env.env.get_state_obs();np.savez_compressed(folder/'_private/initial_state.npz',**state);assert env.env.control_freq==30 and env.env.action_repeat==8;OmegaConf.save(env.config,folder/'_private/config.yaml',resolve=True)
  original_step=p.stepSimulation
  def counted(*args,**kw):
   nonlocal ticks
   r=original_step(*args,**kw);ticks+=1;return r
  p.stepSimulation=counted
  def record(obs,action):
   fi=len(trace);assert ticks==fi*8;(folder/'public').mkdir(exist_ok=True);lab=label(env);objects={};contacts={}
   for j,c in enumerate(COLORS):
    name='block_'+c;obj=auditctl.objects[name];cs=[q for q in p.getContactPoints(bodyA=obj.uid,physicsClientId=env.env.cid)if q[9]>.01];contacts[c]=[list(q)for q in cs];rc=[q for q in cs if q[2]==auditctl.robot.robot_uid];links={q[4]for q in rc};bi=all(links&fam for fam in auditctl.finger_families);ext=[q for q in cs if q[2]!=auditctl.robot.robot_uid];held=bool(bi and not ext);assert held==bool(lab['holding_derived'][j]);objects[c]={'position_m':auditctl.xyz(name).tolist(),'held':held,'bilateral':bool(bi),'external_support_contacts':len(ext)}
   row={'frame':fi,'physics_ticks':ticks,'sim_seconds':ticks/240,'segment':segment,'action_private':action,'robot_obs':obs['robot_obs'].tolist(),'drawer_m':auditctl.joint('drawer_link'),'labels':lab,'targets':targets(lab),'objects_audit':objects,'contacts_private':contacts};trace.append(row)
   fp=folder/'public'/f'{fi:05d}.npz';np.savez_compressed(fp,rgb_static=obs['rgb_obs']['rgb_static'],rgb_gripper=obs['rgb_obs']['rgb_gripper'],robot_obs=obs['robot_obs']);index.write(json.dumps({'frame':fi,'physics_ticks':ticks,'sim_seconds':ticks/240,'file':fp.name,'bytes':fp.stat().st_size,'sha256':sha(fp)})+'\n')
   if 'target_xyz'in action:commands.write(json.dumps({'arrival_frame':fi,'physics_ticks':ticks,'sim_seconds':ticks/240,**{k:action[k]for k in ['target_xyz','target_quaternion','gripper']}})+'\n')
  record(ob,{'phase':'reset_only'})
  def hold(name,grip):
   nonlocal segment
   segment=name;start=len(trace)-1;ctl=PhysicalOracle(env,record);ctl.phase=name
   with physics_guard(env):ctl.hold(96,grip)
   rows=trace[start:];before=rows[0]['targets'];after=rows[-1]['targets'];loss={c:[q['frame']for q in rows[1:]if not q['objects_audit'][c]['held']]for c in COLORS if rows[0]['objects_audit'][c]['held']}
   result={'name':name,'start_frame':start,'end_frame':len(trace)-1,'controls':len(trace)-1-start,'gripper_public_command':grip,'held_loss_frames':loss,'valid_changed_columns':[j for j in range(7)if before['valid'][j]and after['valid'][j]and before['y'][j]!=after['y'][j]],'objects_max_displacement_mm':{c:max(float(np.linalg.norm(np.array(q['objects_audit'][c]['position_m'])-rows[0]['objects_audit'][c]['position_m'])*1000)for q in rows)for c in COLORS},'start_targets':before,'end_targets':after};holds.append(result);return result
  hold('initial_history_hold',1)
  for pi,skill in enumerate(branch['preparation_skills']):
   segment='preparation';start=len(trace)-1;entry={'index':pi,'skill':skill,'start_frame':start,'declared_utc':now(),'status':'started'};preparations.append(entry);save(folder/'_private/preparation_attempts.json',preparations);ctl=PhysicalOracle(env,record)
   with physics_guard(env):res=ctl.run(skill)
   entry.update(status='completed',result=res,end_frame=len(trace)-1);save(folder/'_private/preparation_attempts.json',preparations)
   if res['status']!='success':coverage_reason='preparation_'+res['status']+':'+str(res.get('reason'));break
  hold('preparation_boundary_hold',int(env.observe()['robot_obs'][-1]))
  actualheld=auditctl.held();expected=[]if branch['expected_held_colour']is None else['block_'+branch['expected_held_colour']];drawer=auditctl.joint('drawer_link');condition={'inside_target_matches':branch.get('expected_inside_colour')is None or trace[-1]['targets']['y'][3+COLORS.index(branch['expected_inside_colour'])], 'held_matches':sorted(actualheld)==expected,'drawer_matches':drawer>.18 if branch['expected_drawer']=='open'else drawer<.02,'named_target_on_table':not branch['require_named_target_on_table']or auditctl.on_table('block_'+root['target_color'])};save(folder/'_private/preparation_condition.json',{'expected':branch,'actual_held':actualheld,'actual_drawer_m':drawer,'checks':condition})
  if not all(condition.values()):coverage_reason=coverage_reason or'preparation_condition_not_reached'
  def window(anchor):
   inds=[anchor-k for k in [63,42,21,0]];assert min(inds)>=0;wid=eid+f'__t{anchor:05d}';w={'window_id':wid,'root_id':root['root_id'],'episode_id':eid,'environment':root['environment'],'split':root['split'],'dev_role':root['dev_role'],'anchor_frame':anchor,'history_frame_indices':inds,'simulation_times':[trace[i]['sim_seconds']for i in inds],'physics_ticks':[trace[i]['physics_ticks']for i in inds],'public_frames':[str(folder/'public'/f'{i:05d}.npz')for i in inds]};windows.append(w);save(folder/'_private'/(wid+'.json'),{'frames':[trace[i]for i in inds],'all_7_targets':trace[anchor]['targets']});return wid
  if coverage_reason is None:
   start=len(trace)-1;before=window(start);declaration={'pair_id':eid+'__diagnostic','root_id':root['root_id'],'episode_id':eid,'environment':root['environment'],'split':root['split'],'dev_role':root['dev_role'],'skill':branch['declared_skill'],'declared_arguments':branch['declared_arguments'],'macro_contract':'frozen_skill_plus_posthold96_v1','declared_primitive_budget':900,'declared_posthold_controls':96,'declared_utc':now(),'start_frame':start};save(folder/'public/diagnostic_declaration.json',declaration);dh=sha(folder/'public/diagnostic_declaration.json');before_meta=windows[-1];ticks_before=ticks;pred=worker.request({'op':'before','window_id':before,'public_frames':before_meta['public_frames'],'declaration':{k:declaration[k]for k in ['skill','declared_arguments','declared_primitive_budget','declared_posthold_controls']}});assert ticks==ticks_before and len(trace)-1==start;save(folder/'public/prediction_before_action.json',pred);ph=sha(folder/'public/prediction_before_action.json');start_wall=time.time_ns();assert pred['completed_wall_ns']<start_wall;save(folder/'public/action_start.json',{'started_wall_ns':start_wall,'prediction_record_sha256':ph,'declaration_sha256':dh,'physics_ticks_before':ticks});segment='diagnostic';ctl=PhysicalOracle(env,record)
   with physics_guard(env):res=ctl.run(branch['declared_skill'])
   primitive_end=len(trace)-1;assert primitive_end-start==res['controls'];post=hold('diagnostic_posthold',int(env.observe()['robot_obs'][-1]));end=len(trace)-1;after=window(end);ticks_after=ticks;after_inference=worker.request({'op':'after','window_id':after,'public_frames':windows[-1]['public_frames']});assert ticks==ticks_after;save(folder/'public/observed_after_action.json',after_inference);pub=declaration|{'before_window_id':before,'after_window_id':after,'preaction_prediction_record_path':str(folder/'public/prediction_before_action.json'),'preaction_prediction_record_sha256':ph,'after_observation_record_path':str(folder/'public/observed_after_action.json')};pairs.append(pub)
   y0=trace[start]['targets'];y1=trace[end]['targets'];valid=[bool(x and z)for x,z in zip(y0['valid'],y1['valid'])];changed=[j for j in range(7)if valid[j]and y0['y'][j]!=y1['y'][j]];group='uncertain_labels'if not all(valid)else res['status']+('_changed'if changed else'_nochange')
   private_pairs.append({'pair_id':pub['pair_id'],'root_id':root['root_id'],'episode_id':eid,'family':branch['family'],'condition':branch['condition'],'contrast_role':branch['role'],'planned_colour_evaluator_only':root['target_color'],'start_frame':start,'primitive_end_frame':primitive_end,'end_frame':end,'macro_controls':end-start,'primitive':res,'posthold':post,'before_targets':y0,'after_targets':y1,'valid_changed_columns':changed,'both_all7_valid':all(valid),'outcome_group':group,'declaration_sha256_before_execution':dh,'prediction_input_excludes_private_fields':True});assert sha(folder/'public/diagnostic_declaration.json')==dh
 except Exception as ex:
  failure={'type':type(ex).__name__,'message':str(ex),'traceback':traceback.format_exc(),'last_frame':len(trace)-1};save(folder/'_private/error.json',failure)
 finally:
  if original_step is not None:p.stepSimulation=original_step
  if env is not None:env.close()
  index.close();commands.close()
  with gzip.open(folder/'_private/trace.json.gz','wt')as f:json.dump(plain(trace),f)
 save(folder/'public/windows.json',windows);save(folder/'public/pairs.json',pairs);save(folder/'_private/pairs.json',private_pairs)
 summary={'root_id':root['root_id'],'episode_id':eid,'branch':branch['branch'],'family':branch['family'],'condition':branch['condition'],'split':root['split'],'environment':root['environment'],'frames':len(trace),'controls':max(0,len(trace)-1),'physics_ticks':ticks,'actual_initial_objects':initial.tolist()if initial is not None else None,'preparation_attempts':preparations,'holds':holds,'covered':len(pairs)==1,'uncovered_reason':coverage_reason,'diagnostic_declared':declaration is not None,'completed_pairs':len(pairs),'collection_error':failure,'physics_guard_passed':failure is None,'elapsed_seconds':time.monotonic()-started};save(folder/'_private/summary.json',summary);return summary,pairs,windows,private_pairs
class Worker:
 def __init__(self):
  self.log=(E/'logs'/('worker_'+str(time.time_ns())+'.log')).open('w')
  command='source /home/robot/research/projects/wm_symbolic/world_model_env.sh\nexec python /home/robot/research/projects/wm_symbolic/experiments/state_predicates_independent_D_20261015/source/worker.py'
  self.proc=subprocess.Popen(['/bin/bash','-c',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
  first=self.proc.stdout.readline();assert first,'worker failed to start';ready=json.loads(first);assert ready.get('ready'),ready
 def request(self,req):
  self.proc.stdin.write(json.dumps(req)+'\n');self.proc.stdin.flush();line=self.proc.stdout.readline();assert line,'worker exited unexpectedly';result=json.loads(line)
  if not result.get('ok'):raise RuntimeError('Frozen prediction worker: '+str(result))
  return result
 def close(self):
  self.proc.stdin.write(json.dumps({'op':'stop'})+'\n');self.proc.stdin.flush();assert self.proc.wait(timeout=30)==0;self.log.close()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','remaining'],required=True);a=ap.parse_args();q=frozen()
 if a.phase=='remaining':assert read(E/'results/pilot_gate.json')['passed']
 assert read(E/'results/online_AB_preflight.json')['passed'];worker=Worker();m=read(B/'data_manifest.json');done={s['episode_id']for s in m['episodes']};prior=[(np.array(x['layout']),x['source'])for x in read(E/'configs/root_exclusion.json')['prior_layouts']]
 for root in read(B/'roots_manifest.json'):
  if(root['root_id']in q['pilot_roots'])!=(a.phase=='pilot'):continue
  oldsame=[s for s in m['episodes']if s['root_id']==root['root_id']];own=np.array(oldsame[0]['actual_initial_objects'])if oldsame else None
  others={s['root_id']:np.array(s['actual_initial_objects'])for s in m['episodes']if s['root_id']!=root['root_id']and s['actual_initial_objects']is not None}
  for branch in root['branches']:
   if branch['episode_id']in done:continue
   s,pp,ww,priv=collect(root,branch,prior+[(v,k)for k,v in others.items()],own,worker);m['episodes'].append(s);m['records']+=pp;m['windows']+=ww;m['private_pairs']+=priv;save(B/'data_manifest.json',m);done.add(branch['episode_id'])
   if own is None and s['actual_initial_objects']is not None:own=np.array(s['actual_initial_objects'])
   progress={'phase':'collection_'+a.phase,'episodes':len(m['episodes']),'roots_started':len({s['root_id']for s in m['episodes']}),'pairs':len(m['records']),'uncovered':sum(not s['covered']for s in m['episodes']),'last_episode':branch['episode_id'],'last_outcome':priv[0]['outcome_group']if priv else s['uncovered_reason'],'last_error':s['collection_error']};save(E/'results/progress.json',progress);print(json.dumps(progress,ensure_ascii=False),flush=True)
   if s['collection_error']:raise RuntimeError('Infrastructure/guard failure retained; no silent retry')
 worker.close()
 save(E/'results'/('collection_'+a.phase+'.json'),{'completed_episodes':len(m['episodes']),'pairs':len(m['records']),'training':False,'predictions_saved_before_action':True})
if __name__=='__main__':main()
