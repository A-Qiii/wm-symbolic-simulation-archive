from common import *
import ast,copy,fcntl,xml.etree.ElementTree as ET
import pybullet as pb
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
DATA=P/'vendor/calvin/calvin_env/data'
H=P/'experiments/failure_recovery_control_20261019'
source=(R/'source/preflight_feasible.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)and n.name=='Geometry');exec(compile(ast.Module(body=[node],type_ignores=[]),'<unchanged geometric checks>','exec'))
def near(e,a,excluded):
 for x in excluded:
  if x['environment']!=e:continue
  b=np.asarray(x['poses'])[:,:3]
  if any(np.all(np.linalg.norm(a[:,:3]-b[list(order)],axis=1)<=.020)for order in itertools.permutations(range(3))):return x['source']
def main():
 guard('preflight');budget_check();assert not(S/'configs/roots_frozen.json').exists();start=time.monotonic();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 families=read(S/'configs/family_role_proposal.json');excluded=read(H/'configs/AB_exclusion_inventory.json')['entries'];excluded=copy.deepcopy(excluded)
 for r in read(H/'data/roots_manifest.json'):excluded.append({'environment':r['environment'],'poses':[r['layout']['block_'+c]for c in COLORS],'source':r['root_id']})
 save(S/'configs/AB_exclusion_inventory.json',{'entries':excluded,'source':str(H/'configs/AB_exclusion_inventory.json')})
 accepted=[];audit=[];roots=[];geos={}
 try:
  geos={e:Geometry(e)for e in 'AB'}
  save(S/'results/object_dimensions.json',{e:{c:{'dimensions_m':g.dims[c],'asset':str(g.cfg.objects.movable_objects['block_'+c].file)}for c in COLORS}for e,g in geos.items()})
  for fam in families:
   budget_check();rng=np.random.default_rng(fam['geometry_seed']);e=fam['environment'];g=geos[e];ok=False
   for attempt in range(128):
    centers=[-.10,.06,.22]if e=='A'else[-.22,-.08,.08];layout={}
    for slot,c in enumerate(fam['color_assignment_to_x_sorted_slots']):layout['block_'+c]=[float(centers[slot]+rng.uniform(-.022,.022)),float(rng.uniform(-.145,-.065)),.46,0.,0.,float(rng.uniform(-np.pi,np.pi))]
    rr={'root_id':fam['family_id'],'layout':layout};a=np.array([layout['block_'+c]for c in COLORS]);dup=near(e,a,excluded)
    result={'root_id':fam['family_id'],'reasons':['near_family'],'match':dup}if dup else g.check(rr,reach=True,stable=True);result.update(attempt=attempt,layout=layout);audit.append(result)
    if not result['reasons']:
     ff={**fam,'layout':layout,'accepted_attempt':attempt};accepted.append(ff);excluded.append({'environment':e,'poses':a,'source':fam['family_id']});ok=True;break
   save(S/'results/geometric_admission.json',audit);print('GEOMETRY',fam['family_id'],ok,attempt+1,flush=True)
   if not ok:save(S/'results/BLOCKED_PREFLIGHT.json',{'utc':now(),'family':fam['family_id'],'reason':'128 geometry-only candidates exhausted','no_collection':True});return
   # All role variants retain the exact same assets and base layout; no task rollout during admission.
   for h,t in ff['collect_union_roles']:
    rid=f"{fam['family_id']}_H{h}_T{t}"
    chain=['place_in_drawer',f'lift_{t}_block_table','place_in_drawer',f'lift_{t}_block_table','place_in_drawer','close_drawer']
    roots.append({'root_id':rid,'family_id':fam['family_id'],'environment':e,'split':fam['split'],'dev_role':None,'seed':fam['geometry_seed'],'initial_state':'open_held','held_color':h,'matched_color':t,'primary_color':t,'layout':layout,'setup_skills':['open_drawer',f'lift_{h}_block_table'],'branches':[chain,chain,chain],'arms':[arm for arm in ARMS if [h,t]in fam.get(arm+'_roles',[])],'collect_branches':[1,2]})
  save(S/'data/families_manifest.json',accepted);save(S/'data/roots_manifest.json',roots)
  assert len(accepted)==24 and len(roots)==126
  for arm in ARMS:
   rr=[r for r in roots if arm in r['arms']];assert len(rr)==36;assert all(sum(r['family_id']==f['family_id']for r in rr)==3 for f in accepted if f['split']=='train')
  for e in 'AB':assert collections.Counter((r['held_color'],r['matched_color'])for r in roots if 'random'in r['arms']and r['environment']==e)==collections.Counter((r['held_color'],r['matched_color'])for r in roots if 'balanced'in r['arms']and r['environment']==e)
  save(S/'results/role_feasibility.json',{'utc':now(),'all126_role_variants_geometrically_feasible':True,'checks':'Each original object footprint, dimensions, pair clearance, settled support, and actual-size IK approach checked. Role changes only H setup and T request; all objects checked in every geometry. This is not a task-success certificate. Test motor setups/rollouts remain locked until model selection.','families':24,'role_versions':126,'object_shape_size_changes':False,'geometry_calls':len(audit)})
  save(S/'configs/roots_frozen.json',{'utc':now(),'before_task_collection':True,'geometry_only':True,'families':24,'roles':126,'seconds':time.monotonic()-start,'sha256':{str(S/p):sha(S/p)for p in ['data/families_manifest.json','data/roots_manifest.json','configs/protocol.json','configs/family_role_proposal.json']}})
 finally:
  for g in geos.values():g.close()
if __name__=='__main__':main()
