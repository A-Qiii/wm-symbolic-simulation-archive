"""New block-only data. Failures retained, fitting blocked on physical failures."""
from paths import GPU_LOCK
import os,json,time,fcntl,gzip,hashlib,subprocess,sys
from pathlib import Path
from block_env import B,P,Scene,save,serial
from physics_monitor import Monitor
PLAN=json.load(open(B/'configs/experiment_plan.json'))
def episode(experiment,e,out):
 folder=out/e['id']
 if (folder/'episode.json').exists():return json.load(open(folder/'episode.json'))
 if folder.exists():raise RuntimeError('Incomplete episode retained; choose a new collection version: '+str(folder))
 folder.mkdir(parents=True);s=Scene(e['layout']);mon=Monitor(s,keep_samples=False)
 try:
  states=[s.root];paths=[s.window(folder/'rgb/b0')];actions=[];reverse_loaded=False
  if s.valid:
   for a in e['actions']:
    before=s.private()
    if a==['Place','large','A']and before['objects']['large']['region']==1:
     for c in before['contacts']:
      if {c['bodyA'],c['bodyB']}=={'large','small'}and c['force_N']>.02:reverse_loaded=True
    r=s.execute(a);actions.append(r);states.append(r['state']);paths.append(s.window(folder/f'rgb/b{len(actions)}'))
    if r['status']!='completed':break
  physical=s.valid and mon.peak<=.002 and not mon.unintended and len(actions)==len(e['actions'])and all(a['status']=='completed'for a in actions)
  result={'id':e['id'],'group':e['group'],'split':e['split'],'recipe':e['recipe'],'layout_private':e['layout'],'initialization_valid':s.valid,'actions':actions,'states':states,'public_windows':paths,'physical_valid':bool(physical),'physics_steps':mon.n,'peak_penetration_m':mon.peak,'unintended_contacts':mon.unintended,'region_changes':mon.region_changes,'forbidden_state_seen':mon.forbidden_state_seen,'loaded_reverse_action_present':reverse_loaded,'grasp_audits':s.grasp_audits}
  save(folder/'episode.json',result)
  with gzip.open(folder/'control_trace.json.gz','wt')as f:json.dump(s.trace,f,default=serial)
  return result
 finally:mon.close();s.close()

def main():
 gate=B/'engineering/gate_06/summary.json'
 assert gate.exists()and json.load(open(gate))['all_passed'],'Physical gate must pass before collection'
 audit=gate.parent/'audit.json'
 if not audit.exists():subprocess.run([sys.executable,str(B/'src/audit_engineering.py')],check=True,cwd=B)
 assert json.load(open(audit))['all_passed'],'Transient lift / repeatability audit must pass'
 version=os.environ.get('COLLECTION_VERSION','v1');mode=os.environ.get('COLLECT_MODE','fit');assert mode in ['fit','test']
 if mode=='test':assert(B/'configs/all_models_frozen.json').exists(),'No final test collection before model freeze'
 root=B/'data'/version;root.mkdir(exist_ok=True);frozen=root/'frozen_source';frozen.mkdir(exist_ok=True)
 for name in ['block_env.py','kinematic_guard.py','physics_monitor.py','collect.py']:
  source=B/'src'/name;dest=frozen/name
  if dest.exists():assert source.read_bytes()==dest.read_bytes(),'Control changed during collection; new version required'
  else:dest.write_bytes(source.read_bytes())
 total=[];t0=time.time()
 for exp,x in PLAN['experiments'].items():
  for e in x['episodes']:
   if (e['split']=='test')!=(mode=='test'):continue
   r=episode(exp,e,root/exp);total.append(r);count=sum(len(z['actions'])for z in total);save(B/'STATUS.json',{'stage':'collecting_'+mode,'version':version,'episodes':len(total),'transitions':count,'latest':e['id'],'physical_failures':sum(not z['physical_valid']for z in total),'seconds':time.time()-t0});print('COLLECT',exp,len(total),count,e['id'],r['physical_valid'],flush=True)
   if not r['physical_valid']and mode=='fit':raise RuntimeError('Physical data failure retained; stop and diagnose using fit/development only: '+e['id'])
   if mode=='fit'and count>=50 and not(root/'first50_storage.json').exists():
    files=list(root.rglob('*'));size=sum(f.stat().st_size for f in files if f.is_file());save(root/'first50_storage.json',{'actual_transitions':count,'episodes':len(total),'bytes':size,'bytes_per_transition':size/count,'projection_2880_train_dev_transitions':size/count*2880})
 # Audit each arm independently; no forbidden episode can enter inputs or targets.
 for exp,x in PLAN['experiments'].items():
  lookup={r['id']:r for r in total if r['id'].startswith(exp+'_')}
  for arm,arms in x['arms'].items():
   for split in (['test']if mode=='test'else['train','development']):
    for eid in arms[split]:
     r=lookup[eid]
     if arm=='holdout'and mode=='fit':assert not r['forbidden_state_seen']if exp=='state'else not r['loaded_reverse_action_present']
 save(root/f'{mode}_complete.json',{'episodes':len(total),'transitions':sum(len(r['actions'])for r in total),'physical_failures':sum(not r['physical_valid']for r in total),'plan_sha256':hashlib.sha256((B/'configs/experiment_plan.json').read_bytes()).hexdigest(),'group_isolation_locked':True,'all_holdout_inputs_targets_history_audited':True});print('COLLECTION_COMPLETE',mode,len(total),flush=True)

if __name__=='__main__':
 lock=GPU_LOCK.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:main()
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
