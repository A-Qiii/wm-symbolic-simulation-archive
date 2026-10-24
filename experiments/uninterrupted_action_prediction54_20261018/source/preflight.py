"""Pure-array and mocked-interface checks. No environment/client/simulation/render."""
from common import *
from safety import DENIED,physical_guard,opening_sweep
from gates import evaluate_gates
import ast,types,importlib.util
def must_raise(fn):
 try:fn()
 except (AssertionError,RuntimeError):return
 raise AssertionError('expected rejection')
def main():
 checks=[];obs=np.zeros(15);obs[2]=.5
 for action in ['hold_closed','lift_closed','hold_open']:
  a=action_program(obs,action);assert a.shape==(72,8);assert np.all(a[:,7]==(1 if action=='hold_open'else-1))
  assert np.array_equal(a[:12],np.repeat(a[:1],12,axis=0))
  if action=='lift_closed':assert np.allclose(a[12:60,2],.5+np.arange(1,49)*.0025)and np.all(a[60:,2]==.62)
  else:assert np.array_equal(a,np.repeat(a[:1],72,axis=0))
 checks.append('exact preset72 programs and lift timing')
 inp=public_input(np.zeros((200,200,3),np.uint8),np.zeros((84,84,3),np.uint8),obs,action_program(obs,'hold_closed'))
 for name in ['contacts','phase','scene_obs','rgb_static_t72','target_color','root_id','actual_abort_seconds']:must_raise(lambda name=name:validate_public(dict(inp,**{name:np.zeros(1)})))
 bad={**inp,'duration_seconds':np.array(.4)};must_raise(lambda:validate_public(bad));checks.append('strict public allowlist rejects all injected privileged/future metadata and abort duration')
 from model_input_reader import load_input
 must_raise(lambda:load_input(E/'data/private/forbidden.npz'));checks.append('predictor reader denies private/target directory before opening')
 class Fake:
  def __init__(self):self.calls=[];self.collision=False
  def __getattr__(self,name):
   if name in DENIED+['setJointMotorControl2','setJointMotorControlArray']:
    def fn(*a,**kw):self.calls.append((name,a,kw));return None
    return fn
   raise AttributeError(name)
  def getJointState(self,*a,**kw):return (.01,0,0,0)
  def getClosestPoints(self,*a,**kw):
   self.calls.append(('closest',a,kw))
   if self.collision:
    c=[0]*14;c[3]=9;c[4]=-1;c[8]=.001;return [c]
   return []
 fake=Fake();env=types.SimpleNamespace(cid=0,robot=types.SimpleNamespace(robot_uid=3));shadow=types.SimpleNamespace(cid=1,actual=0,robot=7,name_map={'table':8,'block_red':9},r=types.SimpleNamespace(robot_uid=3,gripper_joint_ids=[9,11],gripper_joint_limits=[0,.04]),sync=lambda:None)
 audit={'forbidden_attempts':[],'shadow_only_reset_calls':0,'native_robot_motor_calls':0}
 with physical_guard(fake,env,shadow,audit):
  must_raise(lambda:fake.restoreState(stateId=0,physicsClientId=0));must_raise(lambda:fake.resetJointState(3,9,0,physicsClientId=0));must_raise(lambda:fake.setJointMotorControl2(bodyIndex=99,physicsClientId=0))
  fake.resetJointState(7,9,0,physicsClientId=1);fake.setJointMotorControl2(bodyIndex=3,physicsClientId=0)
 assert len(audit['forbidden_attempts'])==3 and audit['shadow_only_reset_calls']==1 and audit['native_robot_motor_calls']==1
 checks.append('restore/actual-reset/nonrobot motor denial and isolated shadow exemption')
 fake.calls=[];s=opening_sweep(fake,shadow,'block_red');assert s['valid']and s['samples_tested']==9
 resets=[c for c in fake.calls if c[0]=='resetJointState'];assert len(resets)==18 and all(c[2]['physicsClientId']==1 for c in resets)
 assert all(c[1][1]==8 for c in fake.calls if c[0]=='closest')
 fake.collision=True;assert not opening_sweep(fake,shadow,'block_red')['valid'];checks.append('nine opening widths check non-target geometry only on shadow; collision refuses')
 m={'robot_uid':1,'table_uid':2,'object_uids':{'red':3,'blue':4,'pink':5},'finger_families':[[9,10],[11,12]]}
 def contact(uid,other,link,force,nz=1):
  c=[0]*14;c[1]=uid;c[2]=other;c[3]=-1;c[4]=link;c[7]=[0,0,nz];c[8]=0.;c[9]=force;return c
 cs=[contact(3,1,9,.1),contact(3,1,11,.1),contact(3,2,-1,.3)];f=contact_features(cs,m);hist=[{'features':f}for _ in range(12)];lab=labels_from_history(hist)['red'];assert not lab['Holding']and lab['holding_stable']and lab['SupportTable']and lab['support_stable']and lab['bilateral_stable']
 assert labels_from_history(hist[:5])['red']['holding_stable']==False
 f2=contact_features(cs[:2],m);assert f2['red']['holding_cutoffs']==[True]*3
 h=hist[:11]+[{'features':f2}];assert not labels_from_history(h)['red']['holding_stable'];checks.append('bilateral table-supported negative and unstable/short-history unknown')
 empty=evaluate_gates([],True);assert not empty['passed']and not empty['four_baseline_comparison_justified'];checks.append('empty data cannot pass gates')
 source=(E/'source/collect.py').read_text();tree=ast.parse(source);calls=[n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
 assert 'run'not in calls and 'held'not in calls and 'restoreState'not in calls and 'saveState'not in calls and calls.count('reset')==1
 assert 'torch'not in source and 'V3Runtime'not in source
 for p in(E/'source').glob('*.py'):ast.parse(p.read_text())
 checks.append('source has one initialization reset and no run/held-conditioned selection/restore/fork/model imports')
 for p,h in read(E/'configs/precollection_frozen.json')['sha256'].items():assert sha(p)==h,p
 proto=read(E/'configs/protocol.json');assert len(proto['roots'])==54 and set(proto['anchors'])=={0,12,24,36,60,72};assert proto['budget']['root_controls_max']==900
 import collect
 assert not issubclass(collect.Runtime,object)or collect.Runtime.__bases__==(object,)
 checks.append('collector import and all frozen source hashes verified; no Runtime instantiated')
 save(E/'results/preflight.json',{'utc':now(),'passed':True,'checks':checks,'check_count':len(checks),'physics_steps':0,'renderer_calls':0,'native_clients_created':0,'synthetic_arrays_and_mock_contacts':'unit checks only, never collected data','historical_dedup':read(E/'configs/historical_exclusion.json')['new_vs_historical_and_new_pair_position_only_and_pose_checks_pass']});print(encoded({'preflight':'PASS','checks':len(checks),'physics':0,'renders':0}))
if __name__=='__main__':main()
