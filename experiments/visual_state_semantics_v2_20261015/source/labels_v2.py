"""Evaluator-only geometric and load-bearing labels. Never import into a visual model."""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
COLORS=['red','blue','pink'];SIZES=np.array([[.07,.05,.05],[.05,.05,.05],[.1,.05,.05]])*.8
SIGNS=np.array(list(itertools.product([-1,1],repeat=3)))
LO=np.array([-.225,.025,-.05])*.8;HI=np.array([.225,.10,.20])*.8
SPEC={'version':'contained_supported_drawer3_v2','In':'Full object OBB inside native drawer cavity (sidewalls AND below rim), upward load-bearing path to drawer, not Held. All supporting movable ancestors must also fit the cavity. A protruding stacked block is NOT In.','box_dimensions_m':SIZES.tolist(),'drawer_local_lo_m':LO.tolist(),'drawer_local_hi_m':HI.tolist(),'geometry_signed_side_and_rim_margin_m':.002,'floor_penetration_tolerance_m':.003,'support_force_N':.01,'support_up_normal_z':.5,'support_max_gap_m':.003,'support_force_uncertainty_N':[.005,.015],'support_normal_uncertainty':[.45,.55],'Closed_q_max_m':.02,'Open_q_min_m':.18,'drawer_threshold_uncertainty_m':.003,'Intermediate':'0.02<q<0.18; boundary bands invalid for all three labels. Closed and Open are separate queries, never negations.','Holding':'Unchanged bilateral force-bearing finger-family contacts with no external body support, from frozen PhysicalOracle.','model_access':'No simulator fields; labels, poses, contacts and scenario condition are evaluator-only.'}
def classify_drawer(q):
 cls=0 if q<=.02 else 1 if q>=.18 else 2;y=[cls==j for j in range(3)];valid=min(abs(q-.02),abs(q-.18))>.003;return y,[valid]*3

def evaluate(snapshot):
 # snapshot contains complete physical contacts and world poses, not public model input.
 objects=snapshot['objects'];held=set(snapshot['held']);inv=Rotation.from_quat(snapshot['drawer_world_quaternion']).inv();origin=np.array(snapshot['drawer_world_position']);byuid={o['uid']:name for name,o in objects.items()};geometry={};edges={};uncertain={}
 for j,c in enumerate(COLORS):
  name='block_'+c;o=objects[name];world=Rotation.from_quat(o['quaternion']).apply(SIGNS*SIZES[j]/2)+np.array(o['position']);local=inv.apply(world-origin);clear=np.r_[local[:,0].min()-LO[0],HI[0]-local[:,0].max(),local[:,2].min()-LO[2],HI[2]-local[:,2].max(),HI[1]-local[:,1].max()];floor=float(local[:,1].min()-LO[1]);score=float(clear.min());inside=bool(score>=.002 and floor>=-.003);outside=bool(score<=-.002 or floor<-.003);geometry[name]={'world_corners':world.tolist(),'drawer_local_corners':local.tolist(),'side_rim_clearances_m':clear.tolist(),'minimum_side_rim_clearance_m':score,'floor_clearance_m':floor,'inside':inside,'outside':outside,'boundary':not(inside or outside)};edges[name]=[];uncertain[name]=[]
  uid=o['uid']
  for contact in snapshot['contacts']:
   if contact[1]==uid:other,link,n=contact[2],contact[4],contact[7]
   elif contact[2]==uid:other,link,n=contact[1],contact[3],[-v for v in contact[7]]
   else:continue
   dest='drawer'if other==snapshot['table_uid']and link==snapshot['drawer_link']else byuid.get(other)
   if dest is None:continue
   force=float(contact[9]);nz=float(n[2]);gap=float(contact[8]);edge={'to':dest,'force_N':force,'world_up_normal_z':nz,'gap_m':gap}
   if force>.01 and nz>.5 and gap<=.003:edges[name].append(edge)
   if gap<=.003 and ((.005<force<.015 and nz>.45)or(.45<nz<.55 and force>.005)):uncertain[name].append(edge)
 def paths(name,visited,require_geometry):
  if name in visited or name in held:return []
  if require_geometry and not geometry[name]['inside']:return []
  out=[]
  for edge in edges[name]:
   if edge['to']=='drawer':out.append([name,'drawer'])
   elif edge['to']in objects:
    out += [[name]+rest for rest in paths(edge['to'],visited|{name},require_geometry)]
  return out
 y=[('block_'+c)in held for c in COLORS];v=[True]*3;details={}
 for c in COLORS:
  name='block_'+c;rawpaths=paths(name,set(),False);safe=paths(name,set(),True);g=geometry[name];value=bool(safe);ambiguous=g['boundary']and bool(rawpaths);ambiguous|=not g['outside']and name not in held and bool(uncertain[name]);valid=not ambiguous
  if name in held or g['outside']:value=False;valid=True
  y.append(value);v.append(valid);details[c]=g|{'load_bearing_paths':rawpaths,'contained_load_bearing_paths':safe,'support_edges':edges[name],'uncertain_support_edges':uncertain[name],'held':name in held}
 dy,dv=classify_drawer(snapshot['drawer_q']);return {'y':y+dy,'v':v+dv,'drawer_class':['Closed','Open','Intermediate'][int(np.argmax(dy))],'drawer_q_m':snapshot['drawer_q'],'objects':details,'version':SPEC['version']}

def snapshot(env,oracle):
 import pybullet as p
 state=p.getLinkState(oracle.table.uid,oracle.links['drawer_link'],physicsClientId=env.env.cid);objects={}
 for name,obj in oracle.objects.items():
  pos,quat=p.getBasePositionAndOrientation(obj.uid,physicsClientId=env.env.cid);vel=p.getBaseVelocity(obj.uid,physicsClientId=env.env.cid);objects[name]={'uid':obj.uid,'position':list(pos),'quaternion':list(quat),'linear_velocity':list(vel[0]),'angular_velocity':list(vel[1])}
 return {'objects':objects,'contacts':[list(c)for c in p.getContactPoints(physicsClientId=env.env.cid)],'held':oracle.held(),'table_uid':oracle.table.uid,'drawer_link':oracle.links['drawer_link'],'drawer_world_position':list(state[4]),'drawer_world_quaternion':list(state[5]),'drawer_q':oracle.joint('drawer_link')}
