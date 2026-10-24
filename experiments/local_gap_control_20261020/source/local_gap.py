import pathlib,json,os,datetime,sys,hashlib,collections,itertools
import numpy as np
P=pathlib.Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/local_gap_control_20261020';OLD=P/'experiments/rgbd_observation_control_20261020';O=P/'experiments/object_recurrent_state_run_20261019';sys.path.insert(0,str(P/'src'))
from wm_symbolic.public_geometry import colour_masks,visible_world_points
COL=['red','blue','pink'];PAIRS=[(0,1),(0,2),(1,2)]
def features(frame,details=False):
 objects=[[]for _ in COL];scene=[];edge_info=[]
 for vi,view in enumerate(['static','gripper']):
  need=['rgb_'+view,'depth_'+view,'seg_'+view,'calib_'+view+'_fov','calib_'+view+'_viewMatrix']
  if any(k not in frame for k in need):continue
  dep=frame['depth_'+view];seg=frame['seg_'+view];xyz=visible_world_points(frame,view);valid=np.isfinite(xyz).all(-1)&np.isfinite(dep)&(dep>.01)&(dep<5)&(seg>=0);cm=colour_masks(frame['rgb_'+view]);object_pixels=np.zeros(dep.shape,bool)
  for ci,color in enumerate(COL):
   candidates=[]
   for uid in np.unique(seg[cm[color]&valid]):
    region=seg==uid;colored=region&cm[color]&valid;pts=xyz[colored]
    if len(pts)<6:continue
    span=np.quantile(pts,.95,axis=0)-np.quantile(pts,.05,axis=0)
    if span.max()>.13:continue
    candidates.append((int(colored.sum()),region&valid))
   if candidates:
    _,region=max(candidates,key=lambda x:x[0]);object_pixels|=region;objects[ci].append(xyz[region])
  edges=np.zeros(dep.shape,bool)
  # Visible image boundaries, not CAD edges: instance boundary OR >1cm adjacent depth jump.
  for axis in [0,1]:
   aa=[slice(None),slice(None)];bb=[slice(None),slice(None)];aa[axis]=slice(1,None);bb[axis]=slice(None,-1);aa=tuple(aa);bb=tuple(bb);jump=(seg[aa]!=seg[bb])|(abs(dep[aa]-dep[bb])>.01);edges[aa]|=jump;edges[bb]|=jump
  chosen=edges&valid&~object_pixels;pts=xyz[chosen];scene.append(pts)
  if details:edge_info.append({'view':view,'points':pts,'pixels':np.argwhere(chosen)})
 obs={}
 for i,arrays in enumerate(objects):
  if arrays:
   pts=np.concatenate(arrays);lo=np.quantile(pts,.05,axis=0);hi=np.quantile(pts,.95,axis=0);obs[i]={'center':np.median(pts,axis=0),'lo':lo,'hi':hi,'points':pts}
 out=np.zeros(32,np.float32);candidate=[(np.linalg.norm(obs[i]['center']-obs[j]['center']),p,i,j)for p,(i,j)in enumerate(PAIRS)if i in obs and j in obs]
 detail={'visible_objects':sorted(obs),'pair':None,'gaps':[],'scene_edge_points':sum(len(x)for x in scene)}
 if not candidate or not scene:return out,detail
 distance,pairid,i,j=min(candidate);pts=np.concatenate(scene);out[24+pairid]=1;out[27:30]=obs[j]['center']-obs[i]['center'];out[30]=distance;out[31]=1;detail['pair']=[COL[i],COL[j]]
 for slot,ci in enumerate([i,j]):
  obj=obs[ci];gap=[]
  for axis in range(3):
   for si,sign in enumerate([-1,1]):
    k=axis*2+si;other=[d for d in range(3)if d!=axis];edge=obj['lo'][axis]if sign<0 else obj['hi'][axis];dist=(pts[:,axis]-edge)*sign;mask=(dist>=0)&(dist<=.35)
    for a in other:mask&=(pts[:,a]>=obj['lo'][a]-.015)&(pts[:,a]<=obj['hi'][a]+.015)
    ids=np.flatnonzero(mask);supported=len(ids)>=3
    if supported:
     # 10th percentile resists one-pixel outliers; no dynamics or outcome rule.
     value=float(np.quantile(dist[ids],.1));out[slot*12+k]=value;out[slot*12+6+k]=1;representative=int(ids[np.argmin(abs(dist[ids]-value))]);point=pts[representative].tolist()
    else:value=None;point=None
    gap.append({'axis':'xyz'[axis],'sign':sign,'valid':supported,'support_points':len(ids),'distance_m':value,'point':point})
  detail['gaps'].append({'color':COL[ci],'center':obj['center'].tolist(),'visible_bounds':[obj['lo'].tolist(),obj['hi'].tolist()],'directions':gap})
 return out,detail

def main():
 assert not S.exists(),S
 for d in ['source','configs','cache','results','reports','models','logs']:(S/d).mkdir(parents=True,exist_ok=True)
 start=datetime.datetime.fromisoformat('2026-10-20T00:43:36+00:00').timestamp()
 protocol={'status':'VISIBILITY_AND_FEATURE_DESIGN_FROZEN_BEFORE_SCORING_OR_TRAINING','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'maximum_minutes':40,'hypothesis':'Visible local object-environment geometry improves unseen-layout action consequences without a hand-coded success predicate.','arms':'Existing32-D coarse scene grid versus32-D local visible-edge distances; all original16-D per-object geometry and old frozen inputs unchanged.','coordinate_frame':'Calibrated world/base XYZ; no drawer true joint or object pose.','regions':'Known colored objects routed from current RGB within anonymous visible instance regions, numeric mask IDs never features. Scene is all remaining visible regions, including robot if visible; no semantic drawer/body-link ID selection.','edges':'4-neighbor instance boundary or adjacent depth discontinuity>0.01m. These are visible point-cloud boundaries, not ground-truth physical edges.','pair_selection':'Choose closest pair among visible object median centers; generic proximity, no Held/In or action outcome. Pair ordering by existing color slots.','32_dimensions':'Two closest objects each6 directional[-x,+x,-y,+y,-z,+z] edge distances and6 validity bits=24; pair onehot3 +relative center3+center distance1+pair-valid1=8.','direction_query':'Visible robust AABB from5/95% object points; environment edge points in directional column, orthogonal bounds expanded15mm, positive distance<=350mm; >=3points needed,10th percentile distance. No neighbor means missing, never zero/free-space truth.','normalization':'Training-only; zero invalid distances after normalization with per-direction masks. All other inputs bit-identical to preceding geometry arm.','visibility_gate':'Before any training, on all original train/development role h5 boundaries, require>=90% both objects visible and>=90% selected object-forward(+Y) queries supported in each split. Inspect public RGB-D projections for support on environment beyond objects, not only object occlusion contour. Forward query is inspection of existing fixed setup, not a success rule. If key gap not observable stop this feature path.','training_if_visible':'Same2,490,068 parameters, same original1128 transitions and schedules,1500 updates per arm per seed, seeds0/1; no iterative tuning.','evaluation':'Training/development close outcomes by success,jam,joint-limit, macro recall and training-majority baseline; one/four-step plus Held/In and normal observed trajectories. Development is exploratory.','independent_test_trigger':'Both seeds improve development close one-step and four-step macro recall with positive success-recall change, neither below majority accuracy, h0 Held and normal whole no more than5pp regression, In forgetting not worse by>2pp. Freeze a separate independent test plan only then.','forbidden_features':['private contacts','true object poses','drawer true joint position','execution outcome','future observation'],'original_gates_unchanged':True}
 (S/'configs/protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2));(S/'configs/launch.json').write_text(json.dumps({'start_epoch':start,'deadline_epoch':start+2400,'training_deadline_epoch':start+1800},indent=2));(S/'source/local_gap.py').write_text(pathlib.Path(__file__).read_text())
 rows=[];snapshots=[]
 for split in ['train','development']:
  eps=json.load(open(O/f'cache/{split}_metadata.json'))
  for e in eps:
   if e['kind']!='role':continue
   f=e['features'][5]['public_inputs'][-1]
   with np.load(f)as z:frame={k:z[k]for k in z.files}
   x,d=features(frame,True);rows.append({'split':split,'episode_id':e['episode_id'],'family_id':e['family_id'],'public_path':f,'feature':x.tolist(),**d})
  print('VISIBILITY',split,len([r for r in rows if r['split']==split]),flush=True)
 summaries=[]
 for split in ['train','development']:
  rr=[r for r in rows if r['split']==split];pair=sum(r['pair']is not None for r in rr);front=sum(q['directions'][3]['valid']for r in rr for q in r['gaps']);opp=len(rr)*2;summaries.append({'split':split,'boundaries':len(rr),'families':len({r['family_id']for r in rr}),'pair_visible':pair,'pair_visible_rate':pair/len(rr),'forward_supported_objects':front,'forward_object_opportunities':opp,'forward_supported_rate':front/opp,'passed_numerical_gate':pair/len(rr)>=.9 and front/opp>=.9})
 (S/'results/visibility_rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(S/'results/visibility_preflight.json').write_text(json.dumps({'summaries':summaries,'all_numerical_gates_passed':all(x['passed_numerical_gate']for x in summaries),'visual_support_review_pending':True,'training_updates':0},ensure_ascii=False,indent=2));print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
