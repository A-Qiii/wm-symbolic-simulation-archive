from pathlib import Path
import json,hashlib,datetime,os,numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
NAME='uninterrupted_action_prediction54_20261018'
E=P/'experiments'/NAME;R=P/'reports'/NAME
DESIGN=P/'experiments/uninterrupted_action_prediction_pilot_design_20261018'
COLORS=['red','blue','pink'];ANCHORS=[0,12,24,36,60,72]
PUBLIC_KEYS={'rgb_static_t0','rgb_gripper_t0','robot_obs_t0','action_program','duration_seconds'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def default(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def encoded(x):return json.dumps(x,sort_keys=True,default=default,separators=(',',':'))
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=default)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def npz(p,**a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial')
 with tmp.open('wb')as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
 with np.load(p,allow_pickle=False)as z:
  assert set(z.files)==set(a)
  for k,v in a.items():assert np.array_equal(z[k],v),k
def action_program(robot_obs,action):
 from scipy.spatial.transform import Rotation
 assert np.asarray(robot_obs).shape==(15,)
 x=np.r_[robot_obs[:3],Rotation.from_euler('xyz',robot_obs[3:6]).as_quat(),1 if action=='hold_open'else-1]
 a=np.repeat(x[None],72,axis=0)
 assert action in ['hold_closed','lift_closed','hold_open']
 if action=='lift_closed':a[:,2]+=np.r_[np.zeros(12),np.arange(1,49)*.0025,np.full(12,.12)]
 return a
def public_input(static,gripper,robot_obs,program):
 a={'rgb_static_t0':np.asarray(static).copy(),'rgb_gripper_t0':np.asarray(gripper).copy(),'robot_obs_t0':np.asarray(robot_obs,dtype=np.float64).copy(),'action_program':np.asarray(program,dtype=np.float64).copy(),'duration_seconds':np.array(2.4,dtype=np.float64)}
 validate_public(a);return a
def validate_public(a):
 assert set(a)==PUBLIC_KEYS,'input allowlist violation'
 for k,shape in [('rgb_static_t0',(200,200,3)),('rgb_gripper_t0',(84,84,3))]:assert a[k].shape==shape and a[k].dtype==np.uint8
 assert a['robot_obs_t0'].shape==(15,)and a['action_program'].shape==(72,8)and a['duration_seconds'].shape==()
 assert float(a['duration_seconds'])==2.4
 assert all(np.isfinite(a[k]).all()for k in a)
 assert np.isin(a['action_program'][:,-1],[-1,1]).all()
def contact_features(contacts,mapping):
 out={};robot=mapping['robot_uid'];table=mapping['table_uid'];families=[set(x)for x in mapping['finger_families']]
 for color,uid in mapping['object_uids'].items():
  own=[]
  for c in contacts:
   if c[1]==uid:own.append((c[2],c[4],np.asarray(c[7]),float(c[8]),float(c[9])))
   elif c[2]==uid:own.append((c[1],c[3],-np.asarray(c[7]),float(c[8]),float(c[9])))
  held=[];bilateral=[];support=[]
  for f,nz in zip([.005,.01,.015],[.45,.5,.55]):
   cs=[c for c in own if c[4]>f];links={c[1]for c in cs if c[0]==robot};both=all(links&fam for fam in families)
   bilateral.append(bool(both));held.append(bool(both and not any(c[0]!=robot for c in cs)))
   support.append(any(c[0]==table and c[1]==-1 and c[2][2]>nz and c[3]<=.003 for c in cs))
  out[color]={'holding_cutoffs':held,'bilateral_cutoffs':bilateral,'support_cutoffs':support,'table_upward_force_N':sum(c[4]*c[2][2]for c in own if c[0]==table and c[1]==-1 and c[4]>.01 and c[2][2]>.5 and c[3]<=.003)}
 return out
def labels_from_history(history):
 out={}
 for color in COLORS:
  f=history[-1]['features'][color];held=bool(f['holding_cutoffs'][1]);support=bool(f['support_cutoffs'][1]);both=bool(f['bilateral_cutoffs'][1])
  full=len(history)==12
  stable=full and all(all(v==held for v in h['features'][color]['holding_cutoffs'])and all(sum(h['features'][c]['holding_cutoffs'][j]for c in COLORS)<=1 for j in range(3))for h in history)
  ss=full and all(all(v==support for v in h['features'][color]['support_cutoffs'])for h in history)
  bs=full and all(all(v==both for v in h['features'][color]['bilateral_cutoffs'])for h in history)
  out[color]={'Holding':held,'holding_stable':bool(stable),'SupportTable':support,'support_stable':bool(ss),'bilateral':both,'bilateral_stable':bool(bs),'table_upward_force_N':f['table_upward_force_N'],'history_controls':len(history)}
 return out
def install_write_guard():
 import sys
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]if len(args)>2 else 0
   writing=(isinstance(mode,str)and any(c in mode for c in 'wa+'))or(isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
   if writing and p.is_relative_to(P)and not(p.is_relative_to(E)or p.is_relative_to(R)or p==P/'runs/gpu.lock'):raise RuntimeError('Writing historical project artifact denied: '+str(p))
 sys.addaudithook(guard)
