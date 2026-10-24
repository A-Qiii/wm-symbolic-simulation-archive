from common import *
import ast,xml.etree.ElementTree as ET,fcntl,copy
import pybullet as pb
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
DATA=P/'vendor/calvin/calvin_env/data'
source=(R/'source/preflight_feasible.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)and n.name=='Geometry');exec(compile(ast.Module(body=[node],type_ignores=[]),'<original geometry-only admission>','exec'))
def near_family(env,poses,excluded):
 for e,b,name in excluded:
  if e!=env:continue
  # Ignore colors when checking physical-family duplication.
  if any(np.all(np.linalg.norm(poses[:,:3]-b[list(order),:3],axis=1)<=.020)for order in itertools.permutations(range(3))):return name
 return None
def main():
 guard('preflight');verify_freeze();assert not(S/'configs/roots_frozen.json').exists();lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();excluded=[];refs=[]
 for path in [P/'data/core1200_v1/roots_manifest.json',R/'data/roots_manifest.json']:
  for r in read(path):
   if r['environment']in 'AB':excluded.append((r['environment'],np.array([r['layout']['block_'+c]for c in COLORS]),r['root_id']))
  refs.append(path)
 inventory=[r for r in read(R/'configs/historical_snapshot_inventory.json')if r['environment']in 'AB']
 for path in (R/'data/roots').glob('*_*/branch0_initial_private.npz'):
  if path.parent.name[0]in 'AB':inventory.append({'environment':path.parent.name[0],'path':str(path)})
 for r in inventory:
  e=r['environment'];names=list(OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml').objects.movable_objects);z=np.load(r['path']);a=dict(zip(names,z['scene_obs'][6:].reshape(3,6)));excluded.append((e,np.array([a['block_'+c]for c in COLORS]),r['path']));refs.append(Path(r['path']))
 save(S/'configs/AB_exclusion_inventory.json',{'entries':[{'environment':e,'poses':b,'source':n}for e,b,n in excluded],'files_sha256':{str(p):sha(p)for p in set(refs)}})
 geos={};accepted=[];audit=[]
 try:
  geos={e:Geometry(e)for e in 'AB'}
  for recipe in read(S/'configs/root_recipes.json'):
   rng=np.random.default_rng(recipe['seed']);env=recipe['environment'];success=False
   for attempt in range(128):
    r=copy.deepcopy(recipe);centers=np.array([-.10,.06,.22])if env=='A'else np.array([-.22,-.08,.08]);order=rng.permutation(COLORS);layout={}
    for slot,c in enumerate(order):layout['block_'+c]=[float(centers[slot]+rng.uniform(-.022,.022)),float(rng.uniform(-.145,-.065)),.46,0.,0.,float(rng.uniform(-np.pi,np.pi))]
    r['layout']=layout;a=np.array([layout['block_'+c]for c in COLORS]);dup=near_family(env,a,excluded)
    row={'root_id':r['root_id'],'reasons':['near_family'],'match':dup}if dup else geos[env].check(r,reach=True,stable=True);row.update(attempt=attempt,layout=layout);audit.append(row)
    if not row['reasons']:
     accepted.append(r);excluded.append((env,a,r['root_id']));success=True;break
   save(S/'results/geometric_admission.json',audit);save(S/'configs/accepted_roots_progress.json',accepted);print('GEOMETRY',r['root_id'],success,attempt+1,flush=True)
   if not success:save(S/'results/BLOCKED.json',{'utc':now(),'reason':'Finite geometric admission exhausted','root':r['root_id'],'attempts':128,'collection':0});return
  save(S/'data/roots_manifest.json',accepted);save(S/'configs/roots_frozen.json',{'utc':now(),'roots':24,'no_physical_skill_outcomes_used':True,'sha256':{str(S/'data/roots_manifest.json'):sha(S/'data/roots_manifest.json'),str(S/'configs/protocol.json'):sha(S/'configs/protocol.json')},'seconds':time.monotonic()-start});print('ROOTS_FROZEN',time.monotonic()-start,flush=True)
 finally:
  for g in geos.values():g.close()
if __name__=='__main__':main()
