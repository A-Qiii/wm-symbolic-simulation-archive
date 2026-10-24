from pathlib import Path
import json,hashlib,datetime
import numpy as np
import pybullet as p
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'runs/physical_oracle_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 B.mkdir(exist_ok=False)
 regression=[]
 for version in ['v1','v2']:
  base=P/('runs/physical_oracle_'+version)
  regression.extend(dict(r,root_base=str(base),source_version=version,split='regression') for r in json.loads((base/'roots_manifest.json').read_text()) if r['split']=='acceptance')
 protocol={'version':'physical_oracle_v3','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'v2 valid-request planning failure in closed-drawer red lift; retain all16 direct grasp candidates, add a bounded fully collision-checked front-clearance route if direct routes have no feasible candidate.',
 'unchanged':'Actual900-step and2-attempt limits;12mm moving-mechanism/2mm other planning clearance; original3mm mechanism,10mm non-target drift and10mm penetration acceptance thresholds. No precondition relabeling, no unrequested drawer opening.',
 'regression_roots':45,'regression_note':'v1 exposed18 plusv2 exposed27, both explicitly regression now. Historical results remainunchanged.',
 'new_acceptance_roots':27,'new_seeds':'101000+100*environment_index+root_index(0..8)','new_distribution':'SameABC 3tableblock spacing/yaw distribution; nooutcome screening; noD.',
 'formal_collection_or_training_authorized':False,'main_manifest_sha256':sha(P/'data/data_manifest.json')}
 (B/'protocol.json').write_text(json.dumps(protocol,indent=2));(B/'regression_roots.json').write_text(json.dumps(regression,indent=2))
 rows=[]
 for ei,e in enumerate('ABC'):
  env=CalvinAdapter(P,e,101000+100*ei,tabletop=True)
  try:
   lo,hi=np.asarray(env.env.scene.surfaces['table'])
   for i in range(9):
    seed=101000+100*ei+i;rng=np.random.RandomState(seed);names=rng.permutation(['block_red','block_blue','block_pink'])
    layout={name:[float(lo[0]+(hi[0]-lo[0])*frac+rng.uniform(-.008,.008)),float(rng.uniform(-.13,-.065)),.46] for name,frac in zip(names,[.18,.5,.82])}
    for o in env.env.scene.movable_objects:o.initial_pos=np.array(layout[o.name]);o.initial_orn=np.array(p.getQuaternionFromEuler([0,0,rng.uniform(-np.pi,np.pi)]))
    obs=env.reset()
    for _ in range(24):obs=env.step([obs['robot_obs'][:3],obs['robot_obs'][3:6],1])
    root=env.evaluator_snapshot();rid=f'{e}_acceptance_v3_{i:02d}';d=B/'roots'/rid;d.mkdir(parents=True)
    np.savez_compressed(d/'reset_evaluator_only.npz',robot_obs=root['robot_obs'],scene_obs=root['scene_obs']);save_npz(d/'initial_public.npz',obs)
    rows.append({'root_id':rid,'split':'acceptance','environment':e,'seed':seed,'chain_color':['red','blue','pink'][i%3],'root_base':str(B),'reset_sha256':sha(d/'reset_evaluator_only.npz')})
  finally:env.close()
 assert not ({r['reset_sha256'] for r in rows}&{r['reset_sha256'] for r in regression})
 (B/'roots_manifest.json').write_text(json.dumps(rows,indent=2))
 for f in [B/'protocol.json',B/'regression_roots.json',B/'roots_manifest.json',*B.glob('roots/*/reset_evaluator_only.npz')]:f.chmod(0o444)
 print(json.dumps({'fresh':27,'regression':45,'before_development':True}))
if __name__=='__main__':main()
