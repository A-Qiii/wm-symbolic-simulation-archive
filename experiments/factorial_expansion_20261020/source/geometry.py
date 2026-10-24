from common import *
import ast,copy,itertools,fcntl,xml.etree.ElementTree as ET
import pybullet as pb
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
DATA=P/'vendor/calvin/calvin_env/data'
node=next(n for n in ast.parse((P/'experiments/spatial_coverage_readout_run_20261019/source/preflight_feasible.py').read_text()).body if isinstance(n,ast.ClassDef)and n.name=='Geometry')
exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged Geometry','exec'))
def near(e,a,excluded):
 for x in excluded:
  if x['environment']!=e:continue
  b=np.asarray(x['poses'])[:,:3]
  if any(np.all(np.linalg.norm(a[:,:3]-b[list(order)],axis=1)<=.020)for order in itertools.permutations(range(3))):return x['source']
def recipes(fam,j,kind):
 colors=list(fam['color_assignment_to_x_sorted_slots']);a,b,c=colors[j:]+colors[:j]
 O='open_drawer';C='close_drawer';P='place_in_drawer';L=lambda x:f'lift_{x}_block_table'
 opened=(fam['index_in_split']//5+j)%2==0
 setup=[];fault=-1;mode='order_variant'
 if kind==0:
  setup=[O]if opened else[]
  chain=[L(a),L(a),P,C,O,C]if opened else[O,L(a),L(a),P,C,O]
  other=chain;fault=chain.index(L(a));mode='gripper_miss'
 elif kind==1:
  setup=[O,L(a)];chain=[P,L(b),L(b),P,C,O];other=chain;fault=1;mode='gripper_miss'
 elif kind==2:
  setup=[O,L(a),P]+([]if opened else[C])
  chain=[L(b),L(b),P,C,O,C]if opened else[O,L(b),L(b),P,C,O]
  other=chain;fault=chain.index(L(b));mode='gripper_miss'
 elif kind==3:
  setup=[O,L(a),P,L(b)];chain=[L(a),P,C,O,L(b),C];other=[C,P,O,C,L(b),O]
 elif kind==4:
  setup=[O,L(a),P,L(b),P]
  if fam['split']=='combination_test':
   chain=[L(c),L(c),P,C,O,C];other=chain;fault=0;mode='gripper_miss'
  else:chain=[C,L(a),O,P,L(b),C];other=[P,C,L(b),O,L(a),C]
 else:
  setup=[O,L(a),P,L(b),P,L(c)];chain=[P,C,O,L(c),P,C];other=[C,P,O,C,L(c),O]
 return dict(root_id=f"{fam['family_id']}_state{j}_k{kind}",family_id=fam['family_id'],environment=fam['environment'],
  split=fam['split'],dev_role=None,seed=fam['geometry_seed'],state_kind=kind,role_colors=[a,b,c],
  intended_drawer_open=opened if kind in[0,2]else True,
  initial_state=f'k{kind}_'+''.join([a,b,c])+('_open'if opened else'_closed'),held_color=a,matched_color=b,primary_color=b,
  layout=fam['layout'],setup_skills=setup,branches=[chain,chain,other],collect_branches=[1,2],
  branch_modes=['unused','nominal',mode],fault_step=fault,template='factorial6',pilot=fam['pilot'])
def main():
 guard();verify();budget();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not(S/'configs/roots_frozen.json').exists()
 geos={};accepted=[];audit=[];roots=[];excluded=read(S/'configs/AB_exclusion_inventory.json')['entries']
 try:
  geos={e:Geometry(e)for e in 'AB'}
  for fam in read(S/'configs/family_proposal.json'):
   rng=np.random.default_rng(fam['geometry_seed']);e=fam['environment'];g=geos[e]
   for attempt in range(128):
    budget();centers=[-.10,.06,.22]if e=='A'else[-.22,-.08,.08]
    layout={'block_'+c:[float(centers[i]+rng.uniform(-.022,.022)),float(rng.uniform(-.145,-.065)),.46,0.,0.,float(rng.uniform(-np.pi,np.pi))]for i,c in enumerate(fam['color_assignment_to_x_sorted_slots'])}
    a=np.array([layout['block_'+c]for c in COLORS]);dup=near(e,a,excluded)
    r={'reasons':['near_previous_family'],'match':dup}if dup else g.check({'root_id':fam['family_id'],'layout':layout},reach=True,stable=True)
    r.update(family_id=fam['family_id'],attempt=attempt,layout=layout);audit.append(r)
    if not r['reasons']:break
   else:
    save(S/'results/geometric_admission.json',audit);raise RuntimeError('128 geometry-only proposals exhausted: '+fam['family_id'])
   ff={**fam,'layout':layout,'accepted_attempt':attempt};accepted.append(ff);excluded.append({'environment':e,'poses':a,'source':fam['family_id']})
   roots += [recipes(ff,j,k)for j,k in enumerate(fam['state_kinds'])]
   save(S/'data/families_manifest.json',accepted);save(S/'data/roots_manifest.json',roots)
   if len(accepted)%10==0:save(S/'results/geometric_admission.json',audit)
   print('GEOMETRY',len(accepted),fam['family_id'],'accepted',attempt,flush=True)
  assert len(accepted)==200 and len(roots)==400
  save(S/'results/geometric_admission.json',audit)
  save(S/'configs/roots_frozen.json',{'utc':now(),'families':200,'roots':400,'episodes':800,'task_calls':4800,'before_rollouts':True,'geometry_only':True,'sha256':{str(S/p):sha(S/p)for p in ['data/families_manifest.json','data/roots_manifest.json']}})
 finally:
  for g in geos.values():g.close()
if __name__=='__main__':main()
