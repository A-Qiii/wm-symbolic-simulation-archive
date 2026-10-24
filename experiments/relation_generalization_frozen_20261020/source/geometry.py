from common import *
import ast,copy,itertools,fcntl,xml.etree.ElementTree as ET
import pybullet as pb
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
DATA=P/'vendor/calvin/calvin_env/data'
node=next(n for n in ast.parse((P/'experiments/spatial_coverage_readout_run_20261019/source/preflight_feasible.py').read_text()).body if isinstance(n,ast.ClassDef)and n.name=='Geometry');exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged Geometry','exec'))
def near(e,a,excluded):
 for x in excluded:
  if x['environment']!=e:continue
  b=np.asarray(x['poses'])[:,:3]
  if any(np.all(np.linalg.norm(a[:,:3]-b[list(order)],axis=1)<=.020)for order in itertools.permutations(range(3))):return x['source']
def main():
 guard();verify();budget();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(S/'configs/roots_frozen.json').exists();geos={};accepted=[];audit=[];roots=[];excluded=read(S/'configs/AB_exclusion_inventory.json')['entries']
 try:
  geos={e:Geometry(e)for e in 'AB'}
  for fam in read(S/'configs/family_role_proposal.json'):
   rng=np.random.default_rng(fam['geometry_seed']);e=fam['environment'];g=geos[e]
   for attempt in range(128):
    budget();centers=[-.10,.06,.22]if e=='A'else[-.22,-.08,.08];layout={'block_'+c:[float(centers[i]+rng.uniform(-.022,.022)),float(rng.uniform(-.145,-.065)),.46,0.,0.,float(rng.uniform(-np.pi,np.pi))]for i,c in enumerate(fam['color_assignment_to_x_sorted_slots'])};a=np.array([layout['block_'+c]for c in COLORS]);dup=near(e,a,excluded)
    r={'reasons':['near_previous_family'],'match':dup}if dup else g.check({'root_id':fam['family_id'],'layout':layout},reach=True,stable=True);r.update(family_id=fam['family_id'],attempt=attempt,layout=layout);audit.append(r)
    if not r['reasons']:break
   else:raise RuntimeError('128 geometry-only proposals exhausted')
   ff={**fam,'layout':layout,'accepted_attempt':attempt};accepted.append(ff);excluded.append({'environment':e,'poses':a,'source':fam['family_id']})
   for h,t in fam['collect_union_roles']:
    for template in ['recovery_seen6','order_known4','order_novel4']:
     chain=['place_in_drawer',f'lift_{t}_block_table','place_in_drawer',f'lift_{t}_block_table','place_in_drawer','close_drawer']if template=='recovery_seen6'else([f'lift_{t}_block_table','place_in_drawer','close_drawer','open_drawer']if template=='order_known4'else['close_drawer','open_drawer',f'lift_{t}_block_table','place_in_drawer'])
     setup=['open_drawer',f'lift_{h}_block_table']+([]if template=='recovery_seen6'else['place_in_drawer'])
     roots.append({'root_id':f"{fam['family_id']}_H{h}_T{t}_{template}",'family_id':fam['family_id'],'environment':e,'split':'independent','dev_role':None,'seed':fam['geometry_seed'],'initial_state':'open_held'if template=='recovery_seen6'else'open_empty_H_in','held_color':h,'matched_color':t,'primary_color':t,'layout':layout,'setup_skills':setup,'branches':[chain,chain,chain],'collect_branches':[1,2],'template':template,'fault_step':next(i for i,s in enumerate(chain)if s.startswith('lift_'))})
   save(S/'results/geometric_admission.json',audit);print('GEOMETRY',fam['family_id'],'accepted',attempt,flush=True)
  assert len(accepted)==4 and len(roots)==36
  save(S/'data/families_manifest.json',accepted);save(S/'data/roots_manifest.json',roots)
  save(S/'configs/roots_frozen.json',{'utc':now(),'families':4,'template_roots':36,'episodes':72,'task_calls':336,'before_rollouts':True,'geometry_only':True,'sha256':{str(S/p):sha(S/p)for p in ['data/families_manifest.json','data/roots_manifest.json']}})
 finally:
  for g in geos.values():g.close()
if __name__=='__main__':main()
