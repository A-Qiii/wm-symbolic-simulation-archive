import pathlib,json,sys,numpy as np,collections
P=pathlib.Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/local_gap_control_20261020';sys.path.insert(0,str(P/'src'))
from wm_symbolic.public_geometry import colour_masks,visible_world_points
rows=json.load(open(S/'results/visibility_rows.json'));audits=[]
for r in rows:
 with np.load(r['public_path'])as z:f={k:z[k]for k in z.files}
 allpoints=[];allids=[];edgepoints=[];edgeids=[]
 for view in ['static','gripper']:
  d=f['depth_'+view];seg=f['seg_'+view];xyz=visible_world_points(f,view);valid=np.isfinite(xyz).all(-1)&(d>.01)&(d<5)&(seg>=0);col=colour_masks(f['rgb_'+view]);obj=np.zeros(seg.shape,bool)
  for mask in col.values():
   cand=[]
   for uid in np.unique(seg[mask&valid]):
    q=mask&valid&(seg==uid);pts=xyz[q]
    if len(pts)>=6 and(np.quantile(pts,.95,axis=0)-np.quantile(pts,.05,axis=0)).max()<=.13:cand.append((len(pts),uid))
   if cand:obj|=seg==max(cand)[1]
  env=valid&~obj;edge=np.zeros(d.shape,bool)
  for axis in [0,1]:
   a=[slice(None),slice(None)];b=[slice(None),slice(None)];a[axis]=slice(1,None);b[axis]=slice(None,-1);a,b=tuple(a),tuple(b);q=(seg[a]!=seg[b])|(abs(d[a]-d[b])>.01);edge[a]|=q;edge[b]|=q
  allpoints.append(xyz[env]);allids.append(seg[env]);edgepoints.append(xyz[env&edge]);edgeids.append(seg[env&edge])
 pts=np.concatenate(allpoints);ids=np.concatenate(allids);ep=np.concatenate(edgepoints);ei=np.concatenate(edgeids);bottom=min(r['gaps'],key=lambda g:g['center'][2]);lo,hi=np.array(bottom['visible_bounds']);near=(abs(pts[:,2]-lo[2])<=.02)
 for ax in [0,1]:near&=(pts[:,ax]>=lo[ax]-.03)&(pts[:,ax]<=hi[ax]+.03)
 c=collections.Counter(ids[near].tolist());support=c.most_common(1)[0][0]if c else None;rr={'split':r['split'],'episode_id':r['episode_id'],'support_region_found':support is not None,'support_region_pixels':c.get(support,0),'objects':[]}
 for g in r['gaps']:
  lo,hi=np.array(g['visible_bounds']);dist=ep[:,1]-hi[1];q=(dist>=0)&(dist<=.35)
  for ax in [0,2]:q&=(ep[:,ax]>=lo[ax]-.015)&(ep[:,ax]<=hi[ax]+.015)
  same=q&(ei==support);other=q&(ei!=support);point=None;gap=None
  if other.sum()>=3:
   gap=float(np.quantile(dist[other],.1));ix=np.flatnonzero(other);point=ep[ix[np.argmin(abs(dist[ix]-gap))]].tolist()
  rr['objects'].append({'color':g['color'],'center':g['center'],'visible_bounds':g['visible_bounds'],'same_support_forward_points':int(same.sum()),'external_region_forward_points':int(other.sum()),'external_forward_valid':int(other.sum())>=3,'external_forward_distance_m':gap,'external_point':point,'original_gap_m':g['directions'][3]['distance_m']})
 audits.append(rr)
summary=[]
for split in ['train','development']:
 rr=[r for r in audits if r['split']==split];oo=[o for r in rr for o in r['objects']];upper=[max(r['objects'],key=lambda o:o['center'][2])for r in rr];summary.append({'split':split,'boundaries':len(rr),'support_region_found':sum(r['support_region_found']for r in rr),'external_forward_supported_objects':sum(o['external_forward_valid']for o in oo),'object_opportunities':len(oo),'external_forward_support_rate':sum(o['external_forward_valid']for o in oo)/len(oo),'upper_object_external_forward_supported':sum(o['external_forward_valid']for o in upper),'upper_object_opportunities':len(upper)})
(S/'results/external_region_visibility.json').write_text(json.dumps({'definition':'Support region estimated solely by dominant anonymous visible region near lower object bottom; no true scene coordinates/contact/body-link semantics. External means a different visible region, not guaranteed physical obstacle identity. This is visibility audit, not a second fitted model.','summary':summary,'rows':audits},ensure_ascii=False,indent=2));print(json.dumps(summary,indent=2))
