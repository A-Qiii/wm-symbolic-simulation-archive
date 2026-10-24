from common import *
import copy,collections,xml.etree.ElementTree as ET,fcntl,time,traceback
import pybullet as pb
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
DATA=P/'vendor/calvin/calvin_env/data'
LIMIT=256
class Geometry:
 def __init__(self,e):
  self.e=e;self.cfg=OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml');self.cid=pb.connect(pb.DIRECT)
  self.table=pb.loadURDF(str(DATA/self.cfg.objects.fixed_objects.table.file),useFixedBase=True,globalScaling=.8,physicsClientId=self.cid)
  pb.performCollisionDetection(physicsClientId=self.cid)
  self.dims={};self.blocks={}
  for c in COLORS:
   obj=self.cfg.objects.movable_objects['block_'+c];root=ET.fromstring((DATA/obj.file).read_text());self.dims[c]=np.array(list(map(float,root.find('.//collision/geometry/box').get('size').split())))*.8
  self.env=CalvinAdapter(P,e,260101,use_egl=False,tabletop=True)
  self.robot=self.env.env.robot
 def support(self,c,v):
  dx,dy=self.dims[c][:2]/2;xx=np.linspace(-dx,dx,int(np.ceil(2*dx/.002))+1);yy=np.linspace(-dy,dy,int(np.ceil(2*dy/.002))+1);a,b=np.meshgrid(xx,yy);xy=np.stack([a.ravel(),b.ravel()],1);co,si=np.cos(v[5]),np.sin(v[5]);xy=xy@np.array([[co,si],[-si,co]])+v[:2]
  rays=pb.rayTestBatch(np.c_[xy,np.full(len(xy),.52)].tolist(),np.c_[xy,np.full(len(xy),.40)].tolist(),physicsClientId=self.cid)
  bad=[i for i,r in enumerate(rays)if r[0]!=self.table or r[1]!=-1 or abs(r[3][2]-.44)>.001 or r[4][2]<.99]
  return not bad,{'points':len(rays),'bad':len(bad),'first_bad_xy':xy[bad[0]].tolist()if bad else None}
 def check(self,root,reach=False,stable=False):
  layout=root['layout'];rec={'root_id':root['root_id'],'reasons':[],'support':{},'minimum_pair_clearance_m':1.}
  for c in COLORS:
   ok,a=self.support(c,layout['block_'+c]);rec['support'][c]=a
   if not ok:rec['reasons'].append('footprint_support_'+c)
  if rec['reasons']:return rec
  uids=[]
  for c in COLORS:
   v=layout['block_'+c];uid=pb.loadURDF(str(DATA/self.cfg.objects.movable_objects['block_'+c].file),v[:3],pb.getQuaternionFromEuler(v[3:]),globalScaling=.8,physicsClientId=self.cid);uids.append(uid)
   for q in pb.getClosestPoints(uid,self.table,.001,physicsClientId=self.cid):
    if q[4]!=-1 or q[8]<-.001:rec['reasons'].append('scene_collision_'+c)
  for i in range(3):
   for j in range(i):
    points=pb.getClosestPoints(uids[i],uids[j],1.,physicsClientId=self.cid);d=min(x[8]for x in points);rec['minimum_pair_clearance_m']=min(rec['minimum_pair_clearance_m'],d)
    if d<.010:rec['reasons'].append('pair_clearance_below10mm')
  for u in uids:pb.removeBody(u,physicsClientId=self.cid)
  # No graphics or label/model outcomes participate in candidate screening.
  if not rec['reasons']and(reach or stable):
   env=self.env
   for obj in env.env.scene.movable_objects:
    v=layout[obj.name];obj.initial_pos=np.array(v[:3]);obj.initial_orn=np.array(pb.getQuaternionFromEuler(v[3:]))
   obs=env.reset()
   for _ in range(24):obs=env.step([obs['robot_obs'][:3],obs['robot_obs'][3:6],1])
   snapshot={o.name:o.get_state()for o in env.env.scene.movable_objects};rec['reset_positions']={n:v[:3].tolist()for n,v in snapshot.items()}
   rec['reset_max_position_drift_m']=max(float(np.linalg.norm(snapshot[n][:3]-v[:3]))for n,v in layout.items())
   ctl=PhysicalOracle(env);rec['all_table_supported']=all(ctl.on_table('block_'+c)for c in COLORS)
   if rec['reset_max_position_drift_m']>.003 or not rec['all_table_supported']:rec['reasons'].append('reset_stability')
   if reach:
    r=self.robot;cid=env.env.cid;uid=r.robot_uid;initial=[pb.getJointState(uid,j,physicsClientId=cid)[0]for j in r.arm_joint_ids];reachrows=[]
    # Query-only geometric IK validation in a disposable preflight world. Never used by learned inputs or collection.
    for c in COLORS:
     v=layout['block_'+c];orn=pb.getQuaternionFromEuler([np.pi,0,v[5]]);err=[];collision=[]
     for z in [.62,.59,.54,.50,.463]:
      target=[v[0],v[1],z]
      for _ in range(8):
       sol=pb.calculateInverseKinematics(uid,r.tcp_link_id,target,orn,lowerLimits=r.ll,upperLimits=r.ul,jointRanges=r.jr,restPoses=r.rp,maxNumIterations=100,residualThreshold=1e-5,physicsClientId=cid)
       q=np.clip(sol[:7],r.ll_real,r.ul_real)
       for j,a in zip(r.arm_joint_ids,q):pb.resetJointState(uid,j,float(a),physicsClientId=cid)
      actual=pb.getLinkState(uid,r.tcp_link_id,computeForwardKinematics=True,physicsClientId=cid);err.append(float(np.linalg.norm(np.array(actual[0])-target)))
      # Full-arm/table collision check; ground base contact is excluded, penetration >1mm is not.
      for cp in pb.getClosestPoints(uid,ctl.table.uid,.0,physicsClientId=cid):
       if cp[3]>0 and cp[8]<-.001:collision.append({'z':z,'robot_link':cp[3],'table_link':cp[4],'distance':cp[8]})
     reachrows.append({'color':c,'max_position_error_m':max(err),'scene_penetrations':collision})
     for j,a in zip(r.arm_joint_ids,initial):pb.resetJointState(uid,j,a,physicsClientId=cid)
    rec['reach']=reachrows
    if any(a['max_position_error_m']>.03 or a['scene_penetrations']for a in reachrows):rec['reasons'].append('geometric_approach')
  rec['accepted']=not rec['reasons'];return rec
 def close(self):self.env.close();pb.disconnect(self.cid)
def near(root,excluded):
 a=np.array([root['layout']['block_'+c]for c in COLORS]);matches=[]
 for e,b,name in excluded:
  if e!=root['environment']:continue
  dist=np.linalg.norm(a[:,:3]-b[:,:3],axis=1);dy=np.abs((a[:,5]-b[:,5]+np.pi/2)%np.pi-np.pi/2)
  if np.all(dist<=.012)or(np.all(dist<=.020)and np.all(dy<=.25)):matches.append(name)
 return matches
def main():
 start=time.monotonic();sys.addaudithook(guard);assert not(S/'configs/roots_frozen.json').exists()
 old=read(P/'data/core1200_v1/roots_manifest.json');members=read(O/'configs/protocol.json')['members']['train'];templates=[r for r in old if r['root_id']in members];templates.sort(key=lambda r:r['root_id'])
 save(S/'configs/protected_before.json',seals())
 protocol={'utc':now(),'authorization':'User continuous execution relayed by parent; bounded proposal accepted','proposal_sha256':sha(Q/'manifest.json'),'controller_sha256':sha(P/'src/wm_symbolic/physical_oracle.py'),'counts':{'train':40,'old_band':10,'gap':10,'branches':3,'skills_per_branch':4},'candidate_limit_per_root':LIMIT,'geometry':{'footprint':'Entire rotated footprint sampled on <=2mm grid including all edges/corners; base-link top z=.440 +/-1mm and upward normals required. Finite geometric test, not a mathematical proof.','pair_clearance_m':.010,'scene_penetration_limit_m':.001,'reset_drift_limit_m':.003,'approach':'Five downward TCP poses z .62,.59,.54,.50,.463; IK within30mm plus whole-arm/table penetration check. Query world only; does not certify motor success.'},'sampling':'Conditional Latin strata by scene/color/class; fixed y/yaw, same-class original order. Max256 deterministic draws per root. Geometric rejections retained. No outcome-driven replacement after accept.','heads':{'seeds':[0,1],'updates':1500,'batch':64,'lr':.0003,'weight_decay':.0001,'loss':'.75 unweighted objectCE + .25 drawerCE','norm':str(B/'cells/roots40/configs/normalization.json'),'paired_schedule':'explicit old root/branch/boundary to new mapping'},'evaluation':'Recipes frozen before fits, capture only after heads frozen; C20 exploratory only. h0 paired2000rootbootstrap; per seed.','h0_gate':'Exact paired labels required. Any mismatch retained and stops formal fit; no replacement.','fixed_encoder_limitation':'E previously trained on old narrow40; not on new wide40','no_upload':True}
 save(S/'configs/protocol.json',protocol)
 excluded=[]
 for r in old:excluded.append((r['environment'],np.array([r['layout']['block_'+c]for c in COLORS]),r['root_id']))
 # All historical serialized reset states with matching native object order, AB only.
 for path in (P/'data').glob('roots/*/*/reset_evaluator_only.npz'):
  e=path.parent.name[:1]
  if e not in 'AB':continue
  cfg=OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml');names=list(cfg.objects.movable_objects)
  with np.load(path)as z:poses=z['scene_obs'][6:].reshape(3,6)
  lookup=dict(zip(names,poses));excluded.append((e,np.array([lookup['block_'+c]for c in COLORS]),str(path)))
 for version in ['v1','v2','v3']:
  for path in(P/f'runs/physical_oracle_{version}/roots').glob('*/reset_evaluator_only.npz'):
   e=path.parent.name[:1]
   if e not in 'AB':continue
   cfg=OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml');names=list(cfg.objects.movable_objects)
   with np.load(path)as z:poses=z['scene_obs'][6:].reshape(3,6)
   lookup=dict(zip(names,poses));excluded.append((e,np.array([lookup['block_'+c]for c in COLORS]),str(path)))
 save(S/'configs/historical_exclusions.json',{'items':[{'environment':e,'poses':b,'source':n}for e,b,n in excluded]})
 rng=np.random.default_rng(261019901);groups=collections.defaultdict(list)
 for i,r in enumerate(templates):
  for c in COLORS:groups[(r['environment'],c,cls(r['layout']['block_'+c][0]))].append(i)
 strata={}
 for key,indices in sorted(groups.items()):
  for rank,i in zip(rng.permutation(len(indices)),indices):strata[i,key[1]]=(int(rank),len(indices))
 # Deterministically assign Latin bins compatible with fixed within-class object order.
 def violations():
  bad=[]
  for i,t in enumerate(templates):
   mids=sorted([c for c in COLORS if cls(t['layout']['block_'+c][0])==1],key=lambda c:t['layout']['block_'+c][0])
   if len(mids)==2:
    l,r=mids;lr,ln=strata[i,l];rr,rn=strata[i,r]
    if (lr+.5)/ln >= (rr+.5)/rn:bad.append((i,l,r))
  return bad
 for iteration in range(100000):
  bad=violations()
  if not bad:break
  i,c,d=bad[int(rng.integers(len(bad)))];color=c if rng.random()<.5 else d;key=(templates[i]['environment'],color,1);j=int(rng.choice(groups[key]));before=len(bad)
  strata[i,color],strata[j,color]=strata[j,color],strata[i,color]
  if len(violations())>before and rng.random()>.02:strata[i,color],strata[j,color]=strata[j,color],strata[i,color]
 assert not violations(),'Latin assignment incompatible with original ordering'
 save(S/'configs/strata_assignment.json',{'algorithm':'Fixed seeded swaps, <=100000; reject reversed bin midpoints; no physics/outcome/model scores','iterations':iteration,'violations':0})
 candidates=[]
 for i,t in enumerate(templates):
  r=copy.deepcopy(t);r.update(root_id=f"{t['environment']}_coverage_train_{i%20:02d}",paired_old_root=t['root_id'],seed=261019100+i,split='train',dev_role=None);candidates.append(r)
 # Eval quotas predeclared; use untouched old templates only as state/action recipes.
 for cohort in ['old_band','gap']:
  for ei,e in enumerate('AB'):
   kinds=['closed_empty','open_empty','open_held','closed_empty','closed_empty']if e=='A'else['closed_empty','open_empty','open_held','closed_empty','open_empty']
   for j,kind in enumerate(kinds):
    pool=[r for r in templates if r['environment']==e and r['initial_state']==kind];t=pool[j%len(pool)];r=copy.deepcopy(t);r.update(root_id=f'{e}_coverage_{cohort}_{j:02d}',paired_old_root=None,action_template=t['root_id'],seed=261019300+(100 if cohort=='gap'else 0)+ei*10+j,split=cohort,dev_role=None)
    if cohort=='gap':
     targetclass=2 if e=='A'and j in[0,2,4]else 1;allowed=[c for c in COLORS if cls(r['layout']['block_'+c][0])==targetclass and c!=r['held_color']]
     if not allowed:
      # Held-right template cannot support a table-right target: select predetermined next template of same type.
      for t in pool:
       allowed=[c for c in COLORS if cls(t['layout']['block_'+c][0])==targetclass and c!=t['held_color']]
       if allowed:
        oldid=r['root_id'];seed=r['seed'];r=copy.deepcopy(t);r.update(root_id=oldid,paired_old_root=None,action_template=t['root_id'],seed=seed,split=cohort,dev_role=None);break
     assert allowed;r['gap_target_color']=allowed[j%len(allowed)];r['gap_target_class']=targetclass
    candidates.append(r)
 save(S/'configs/recipes_before_screening.json',{'roots':candidates,'strata':[{ 'template_index':i,'color':c,'rank':v[0],'count':v[1]}for(i,c),v in strata.items()]})
 geos={};audits=[];accepted=[];oldchecks=[]
 try:
  for e in 'AB':geos[e]=Geometry(e)
  for t in templates:
   a=geos[t['environment']].check(t,reach=True,stable=True);oldchecks.append(a);print('OLD_CHECK',t['root_id'],a['reasons'],flush=True)
  save(S/'results/old40_geometry.json',oldchecks)
  for i,recipe in enumerate(candidates):
   rng=np.random.default_rng(recipe['seed']);e=recipe['environment'];g=geos[e];original=copy.deepcopy(recipe['layout']);accepted_this=None
   for attempt in range(LIMIT):
    r=copy.deepcopy(recipe)
    if r['split']=='train':
     for c in COLORS:
      k=cls(original['block_'+c][0]);lo,hi=(-.35,-.151667)if k==0 else(-.141667,.141667)if k==1 else(.151667,.35);rank,n=strata[i,c];r['layout']['block_'+c][0]=lo+(hi-lo)*(rank+rng.random())/n
    else:
     lo,hi=(-.2,.35)if e=='A'else(-.35,.15)
     order=sorted(COLORS,key=lambda c:original['block_'+c][0])
     for slot,c in enumerate(order):
      r['layout']['block_'+c]=[lo+(hi-lo)*(.16+.34*slot)+rng.uniform(-.014,.014),rng.uniform(-.14,-.05),.46,0,0,rng.uniform(-np.pi,np.pi)]
     if r['split']=='gap':
      c=r['gap_target_color'];k=r['gap_target_class'];r['layout']['block_'+c][0]=rng.uniform(.151667,.249974)if k==2 else rng.uniform(.087206,.141667)
    # Within-class left/right order, including held pre-setup layout.
    orderok=all((original['block_'+c][0]<original['block_'+d][0])==(r['layout']['block_'+c][0]<r['layout']['block_'+d][0])for c in COLORS for d in COLORS if c!=d and cls(original['block_'+c][0])==cls(original['block_'+d][0]))
    duplicate=near(r,excluded)
    if not orderok or duplicate:a={'root_id':r['root_id'],'reasons':['within_class_order']if not orderok else['near_duplicate'],'matches':duplicate}
    else:a=g.check(r,reach=True,stable=True)
    a.update(attempt=attempt,candidate_layout=r['layout']);audits.append(a)
    if not a['reasons']:
     accepted_this=r;accepted.append(r);excluded.append((e,np.array([r['layout']['block_'+c]for c in COLORS]),r['root_id']));break
   save(S/'results/candidate_audit.json',audits);save(S/'configs/accepted_so_far.json',accepted)
   print('ROOT_SCREEN',recipe['root_id'],'accepted'if accepted_this else'EXHAUSTED','attempts',attempt+1,flush=True)
   if accepted_this is None:
    save(S/'results/BLOCKED_PREFLIGHT.json',{'utc':now(),'root':recipe['root_id'],'attempts':LIMIT,'reasons':dict(collections.Counter(x for a in audits if a['root_id']==recipe['root_id']for x in a['reasons'])),'accepted_roots':len(accepted),'no_collection':True,'no_training':True});return
  save(S/'data/roots_manifest.json',accepted);save(S/'configs/roots_frozen.json',{'utc':now(),'sha256':{str(S/'data/roots_manifest.json'):sha(S/'data/roots_manifest.json'),str(S/'configs/protocol.json'):sha(S/'configs/protocol.json')},'roots':60,'historical_exclusions':len(excluded)-60,'seconds':time.monotonic()-start})
 finally:
  for g in geos.values():g.close()
if __name__=='__main__':main()
