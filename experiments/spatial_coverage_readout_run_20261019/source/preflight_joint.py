from common import *
from preflight_feasible import Geometry,near
import copy,collections,time,fcntl
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

def polygon(v,dim):
 xy=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.asarray(dim[:2])/2;c,s=np.cos(v[5]),np.sin(v[5]);return xy@np.array([[c,s],[-s,c]])+v[:2]
def distance(a,b):
 overlap=True
 for poly in [a,b]:
  for i in range(2):
   edge=poly[i+1]-poly[i];axis=np.array([-edge[1],edge[0]]);aa=a@axis;bb=b@axis
   if aa.max()<bb.min()or bb.max()<aa.min():overlap=False
 if overlap:return 0.
 best=1.
 for pts,poly in [(a,b),(b,a)]:
  for v in pts:
   for i in range(4):
    q=poly[i];edge=poly[(i+1)%4]-q;t=np.clip(np.dot(v-q,edge)/np.dot(edge,edge),0,1);best=min(best,float(np.linalg.norm(v-q-t*edge)))
 return best

def main():
 sys.addaudithook(guard);start=time.monotonic();assert not(S/'configs/roots_frozen.json').exists();lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 recipes=read(S/'configs/recipes_before_screening.json')['roots'];train=recipes[:40];domains={(x['template'],x['color']):x['valid_x_grid']for x in read(S/'results/feasible_domain_audit.json')}
 save(S/'configs/joint_rule_before_solver.json',{'utc':now(),'reason':'Independent Latin ranks create overlapping footprints. Jointly allocate the same per-scene/color/class quantile ranks under original within-class ordering and >=14mm rectangle clearance; then bounded actual mesh/IK/reset verification.','solver_time_limit_seconds':60,'candidate_limit_each':256,'x_jitter_m':.001,'sole_design_factor':'x coverage on geometrically admitted subset; fixed y,yaw,class,actions','candidate_acceptance_not_action_success':True,'no_outcome_or_test_scores':True,'prior_attempts_retained':True})
 geos={};audit=[];accepted=[]
 try:
  geos={e:Geometry(e)for e in 'AB'};groups=collections.defaultdict(list)
  for i,r in enumerate(train):
   for c in COLORS:groups[(r['environment'],c,cls(r['layout']['block_'+c][0]))].append(i)
  variables=[];idx={};centers={};polys={}
  for key,indices in sorted(groups.items()):
   e,c,k=key;n=len(indices)
   for i in indices:
    values=domains[train[i]['paired_old_root'],c]
    for rank in range(n):
     x=values[min(int((rank+.5)/n*len(values)),len(values)-1)];idx[i,c,rank]=len(variables);variables.append((i,c,rank));centers[i,c,rank]=x;v=copy.deepcopy(train[i]['layout']['block_'+c]);v[0]=x;polys[i,c,rank]=polygon(v,geos[e].dims[c])
  rr=[];cc=[];vv=[];lower=[];upper=[]
  def constraint(cols,lo,hi):
   row=len(lower);lower.append(lo);upper.append(hi)
   for col in cols:rr.append(row);cc.append(col);vv.append(1.)
  for key,indices in groups.items():
   c=key[1];n=len(indices)
   for i in indices:constraint([idx[i,c,k]for k in range(n)],1,1)
   for k in range(n):constraint([idx[i,c,k]for i in indices],1,1)
  for i,r in enumerate(train):
   for a,c in enumerate(COLORS):
    for d in COLORS[:a]:
     nc=len(groups[r['environment'],c,cls(r['layout']['block_'+c][0])]);nd=len(groups[r['environment'],d,cls(r['layout']['block_'+d][0])]);same=cls(r['layout']['block_'+c][0])==cls(r['layout']['block_'+d][0]);original_order=r['layout']['block_'+c][0]<r['layout']['block_'+d][0]
     for k in range(nc):
      for l in range(nd):
       if (same and (centers[i,c,k]<centers[i,d,l])!=original_order)or distance(polys[i,c,k],polys[i,d,l])<.014:constraint([idx[i,c,k],idx[i,d,l]],0,1)
  matrix=coo_matrix((vv,(rr,cc)),shape=(len(lower),len(variables))).tocsc();rng=np.random.default_rng(261019904)
  result=milp(rng.uniform(0,1,len(variables)),integrality=np.ones(len(variables)),bounds=Bounds(0,1),constraints=LinearConstraint(matrix,np.array(lower),np.array(upper)),options={'time_limit':60,'mip_rel_gap':.01})
  save(S/'results/joint_assignment_solver.json',{'utc':now(),'variables':len(variables),'constraints':len(lower),'status':result.status,'message':result.message,'has_feasible_solution':result.x is not None})
  if result.x is None:
   save(S/'results/BLOCKED_PREFLIGHT.json',{'utc':now(),'reason':'Finite joint quantile assignment has no feasible returned solution','solver_status':result.message,'no_collection':True,'no_training':True});return
  chosen={}
  for j,value in enumerate(result.x):
   if value>.5:
    i,c,k=variables[j];chosen[i,c]=(k,centers[i,c,k])
  assert len(chosen)==120
  assign=[{'root':train[i]['root_id'],'color':c,'rank':k,'center_x':x}for(i,c),(k,x)in sorted(chosen.items())];save(S/'configs/joint_quantile_assignment.json',assign)
  excluded=[]
  for row in read(S/'configs/historical_exclusions.json')['items']:excluded.append((row['environment'],np.array(row['poses']),row['source']))
  # Extend exclusion to all indexed historical A/B serialized post-setup roots.
  from omegaconf import OmegaConf
  inventory=read(S/'configs/historical_snapshot_inventory.json')
  for row in inventory:
   e=row['environment'];path=Path(row['path']);names=list(OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml').objects.movable_objects)
   with np.load(path)as z:poses=z['scene_obs'][6:].reshape(3,6)
   lookup=dict(zip(names,poses));excluded.append((e,np.array([lookup['block_'+c]for c in COLORS]),str(path)))
  for i,recipe in enumerate(recipes):
   e=recipe['environment'];rng=np.random.default_rng(recipe['seed']);accepted_this=None
   for attempt in range(256):
    r=copy.deepcopy(recipe)
    if r['split']=='train':
     for c in COLORS:
      _,center=chosen[i,c];k=cls(r['layout']['block_'+c][0]);lo,hi=(-.35,-.151667)if k==0 else(-.141667,.141667)if k==1 else(.151667,.35);r['layout']['block_'+c][0]=float(np.clip(center+rng.uniform(-.001,.001),lo+1e-6,hi-1e-6))
    else:
     lo,hi=(-.2,.35)if e=='A'else(-.35,.15);order=sorted(COLORS,key=lambda c:r['layout']['block_'+c][0])
     for slot,c in enumerate(order):r['layout']['block_'+c]=[lo+(hi-lo)*(.16+.34*slot)+rng.uniform(-.014,.014),rng.uniform(-.14,-.05),.46,0,0,rng.uniform(-np.pi,np.pi)]
     if r['split']=='gap':
      c=r['gap_target_color'];r['layout']['block_'+c][0]=rng.uniform(.151667,.249974)if r['gap_target_class']==2 else rng.uniform(.087206,.141667)
    dup=near(r,excluded)
    if dup:a={'root_id':r['root_id'],'reasons':['near_duplicate'],'matches':dup}
    else:a=geos[e].check(r,reach=True,stable=True)
    a.update(attempt=attempt,candidate_layout=r['layout']);audit.append(a)
    if not a['reasons']:
     accepted_this=r;accepted.append(r);excluded.append((e,np.array([r['layout']['block_'+c]for c in COLORS]),r['root_id']));break
   save(S/'results/joint_candidate_audit.json',audit);save(S/'configs/accepted_joint_so_far.json',accepted);print('JOINT_ROOT',recipe['root_id'],'accepted'if accepted_this else'EXHAUSTED',attempt+1,flush=True)
   if not accepted_this:
    save(S/'results/BLOCKED_PREFLIGHT.json',{'utc':now(),'root':recipe['root_id'],'attempts':256,'reason':'Joint assignment actual check exhausted','reasons':dict(collections.Counter(v for x in audit if x['root_id']==recipe['root_id']for v in x['reasons'])),'accepted_roots':len(accepted),'no_collection':True,'no_training':True});return
  save(S/'data/roots_manifest.json',accepted);refs=[S/'data/roots_manifest.json',S/'configs/protocol.json',S/'configs/domain_rule_before_queries.json',S/'configs/joint_rule_before_solver.json',S/'configs/joint_quantile_assignment.json']
  save(S/'configs/roots_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in refs},'roots':60,'historical_exclusions':len(excluded)-60,'seconds':time.monotonic()-start,'accepted_mesh_checks':60})
  save(S/'results/PREFLIGHT_COMPLETE.json',{'utc':now(),'roots':60,'previous_blocker_superseded':True,'formal_collection':0})
 finally:
  for g in geos.values():g.close()
if __name__=='__main__':main()
