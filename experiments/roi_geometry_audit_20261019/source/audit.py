from pathlib import Path
import os,sys,json,hashlib,csv,collections,datetime,time
import numpy as np
import yaml
P=Path('/home/robot/research/projects/wm_symbolic')
O=P/'experiments/object_identity_readout_20261019'
D=P/'experiments/roi_geometry_audit_20261019'
COLORS=['red','blue','pink'];SPLITS=['train','test','externalC']
os.environ['MPLCONFIGDIR']=str(D/'cache/matplotlib');os.environ['PYTHONDONTWRITEBYTECODE']='1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.item()if isinstance(x,np.generic)else x.tolist())+'\n')
def csvwrite(p,rows):
 with Path(p).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def guard(event,args):
 if event in ['socket.connect','subprocess.Popen','os.system']:raise RuntimeError('No network/process execution in audit')
 if event=='open'and isinstance(args[0],(str,bytes)):
  p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]if len(args)>2 else 0
  writing=(isinstance(mode,str)and any(x in mode for x in 'wa+'))or(isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and p.is_relative_to(P)and not p.is_relative_to(D):raise RuntimeError('Sealed/original project write forbidden: '+str(p))
  if p.is_relative_to(P/'data/core1200_v1')and '/D_core1200_'in str(p):raise RuntimeError('D access forbidden')
def protected():
 out={}
 for m in [P/'experiments/rssm_scaling_run_20261019/configs/delivery_manifest.json',P/'experiments/rssm_h2_supplement_20261019/manifest.json',O/'manifest.json']:
  base=O if m==O/'manifest.json'else m.parent
  for n,h in read(m)['sha256'].items():
   f=Path(n)if Path(n).is_absolute()else base/n;assert sha(f)==h,str(f);out[str(f)]=h
  out[str(m)]=sha(m)
 return out
def hull(points):
 pts=sorted(set(map(tuple,points)))
 if len(pts)<3:return np.array(pts)
 def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
 lo=[];hi=[]
 for p in pts:
  while len(lo)>=2 and cross(lo[-2],lo[-1],p)<=0:lo.pop()
  lo.append(p)
 for p in reversed(pts):
  while len(hi)>=2 and cross(hi[-2],hi[-1],p)<=0:hi.pop()
  hi.append(p)
 return np.array(lo[:-1]+hi[:-1])
def inside(points,polygon):
 if len(polygon)<3:return np.zeros(len(points),bool)
 a=polygon;b=np.roll(a,-1,axis=0);v=b-a;q=points[:,None,:]-a
 return ((v[None,:,0]*q[:,:,1]-v[None,:,1]*q[:,:,0])>=-1e-9).all(1)
def project(pos,view,proj,w,h):
 clip=proj@view@np.r_[pos,1.];q=clip[:3]/clip[3];return np.array([(q[0]+1)*w/2,(1-q[1])*h/2])
def collect():
 started=time.perf_counter();assert not(D/'results/complete.json').exists();sys.addaudithook(guard)
 before=protected();save(D/'configs/protected_before.json',before)
 members=read(O/'configs/protocol.json')['members'];roots={r['root_id']:r for r in read(P/'data/core1200_v1/roots_manifest.json')};allowed=set().union(*(set(members[s])for s in SPLITS))
 items=[r for r in read(P/'data/core1200_v1/features/index.json')['items']if r['root_id']in allowed]
 records={(r['episode_id'],r['step']):r for r in read(P/'data/core1200_v1/data_manifest.json')['records']if r['root_id']in allowed}
 scenes={e:yaml.safe_load((P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml').read_text())for e in 'ABC'}
 scene_audit={};sources={}
 for e,sc in scenes.items():
  table=sc['objects']['fixed_objects']['table'];mesh=(P/'vendor/calvin/calvin_env/data'/table['file']).parent.parent/'meshes/base_link.obj';vertices=np.array([[float(x)for x in l.split()[1:4]]for l in mesh.read_text().splitlines()if l.startswith('v ')])*sc['global_scaling'];bounds=vertices.min(0),vertices.max(0)
  scene_audit[e]={'origin':table['initial_pos'],'orientation':table['initial_orn'],'mesh_xyz_bounds_m':np.array(bounds),'cuts_x_m':np.linspace(bounds[0][0],bounds[1][0],4)[1:3],'sampling_table_bounds_m':sc['surfaces']['table'],'mesh_sha256':sha(mesh)}
  assert table['initial_pos']==[0,0,0]and table['initial_orn']==[0,0,0]
 for f in [P/'src/wm_symbolic/calibrated_labels.py',P/'src/wm_symbolic/state_labels.py',P/'src/wm_symbolic/vision_cache.py',P/'src/wm_symbolic/world_model.py',P/'src/wm_symbolic/public_geometry.py']+[P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml'for e in 'ABC']:sources[str(f)]=sha(f)
 save(D/'configs/source_hashes.json',sources)
 output=[];calibs={};label_errors=[];geometric_max=0.;counters=collections.Counter()
 for split in SPLITS:
  route=dict(np.load(O/f'data/routes_{split}.npz'));routes={(eid,int(h)):i for i,(eid,h)in enumerate(zip(route['episode_id'],route['step']))}
  pred={}
  for seed in [0,1]:
   a=dict(np.load(O/f'data/predictions_object_seed{seed}_{split}.npz'));pred[seed]={(eid,int(h)):a['prob'][i]for i,(eid,h)in enumerate(zip(a['episode_id'],a['step']))}
  for item in sorted([x for x in items if x['root_id']in members[split]],key=lambda x:(x['episode_id'],x['step'])):
   eid,h=item['episode_id'],item['step'];root=item['root_id'];env=root[0];rec=records[eid,max(h-1,0)];label=read(rec['labels_path'])['before'if h==0 else'after'];ix=routes[eid,h]
   with np.load(item['path'])as z:geom=z['geometry'];valid=z['valid'];ids=z['visible_registry_ids']
   with np.load(item['public_inputs'][-1])as z:
    frames={k:z[k]for k in z.files if k.startswith('calib_')or k.startswith('seg_')}
   for vi,cam in enumerate(['static','gripper']):
    w=int(frames['calib_'+cam+'_width']);height=int(frames['calib_'+cam+'_height']);view=frames['calib_'+cam+'_viewMatrix'].reshape(4,4).T;proj=frames['calib_'+cam+'_projectionMatrix'].reshape(4,4).T
    calibration={k.replace('calib_'+cam+'_',''):v.tolist()for k,v in frames.items()if k.startswith('calib_'+cam+'_')};sig=hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest();calibs.setdefault((env,cam),{}).setdefault(sig,{'calibration':calibration,'records':0});calibs[env,cam][sig]['records']+=1
    seg=frames['seg_'+cam].copy();seg[(seg>=0)&((seg&((1<<24)-1))==0)]=0
    for j,color in enumerate(COLORS):
     loc=int(label['categorical'][j]);pos=np.array(label['continuous']['objects'][color]['position_m']);cuts=np.array(scene_audit[env]['cuts_x_m']);slot=int(route['slots'][ix,j]);visible=slot>=0 and bool(valid[(6+vi)*64+slot*4])
     if loc<3:
      expected=int(pos[0]>=cuts[0])+int(pos[0]>=cuts[1]);assert label['categorical_valid'][j]
      if expected!=loc:label_errors.append([eid,h,color,loc,expected])
     r={'split':split,'environment':env,'root_id':root,'episode_id':eid,'step':h,'h0_unique':h==0 and eid.endswith('_b0'),'color':color,'camera':cam,'GT_location':loc,'GT_x_m':pos[0],'GT_y_m':pos[1],'GT_z_m':pos[2],'GT_right_boundary_margin_mm':(pos[0]-cuts[1])*1000,'visible':bool(visible),'pixels':0,'initial_state':roots[root]['initial_state'],'object_seed0_prediction':int(pred[0][eid,h][j*8:j*8+8].argmax()),'object_seed1_prediction':int(pred[1][eid,h][j*8:j*8+8].argmax())}
     for name in ['u_px','v_px','width_px','height_px','u','v','width','height','padded_u','padded_v','projected_GT_u_px','projected_GT_v_px','center_projection_error_px']:r[name]=None
     if visible:
      start=(6+vi)*64+slot*4;g=geom[start];yy,xx=np.where(seg==ids[slot]);assert len(xx)>=2
      box=np.array([xx.min()+(224-w)//2,yy.min()+(224-height)//2,xx.max()+1+(224-w)//2,yy.max()+1+(224-height)//2])/224
      geometric_max=max(geometric_max,float(np.abs(g[:4]-box).max()));assert np.allclose(g[:4],box,atol=3e-8)
      u=(xx.min()+xx.max()+1)/2;v=(yy.min()+yy.max()+1)/2;bw=xx.max()-xx.min()+1;bh=yy.max()-yy.min()+1;q=project(pos,view,proj,w,height)
      r.update(u_px=u,v_px=v,width_px=bw,height_px=bh,u=u/w,v=v/height,width=bw/w,height=bh/height,padded_u=float((g[0]+g[2])/2),padded_v=float((g[1]+g[3])/2),projected_GT_u_px=q[0],projected_GT_v_px=q[1],center_projection_error_px=float(np.linalg.norm([u-q[0],v-q[1]])),pixels=len(xx))
     output.append(r);counters[split,cam,loc,bool(visible)]+=1
  print('COLLECTED',split,len([r for r in output if r['split']==split]),flush=True)
 csvwrite(D/'results/records.csv',output)
 save(D/'results/calibration.json',{'scenes':scene_audit,'camera_signatures':{e+'_'+c:v for(e,c),v in calibs.items()},'table_label_mismatches':label_errors,'cached_ROI_vs_pixels_max_difference':geometric_max,'definitions':{'class0':'x < -0.14666666666666667 m','class1':'-0.14666666666666667 <= x < 0.14666666666666672 m','class2':'x >= 0.14666666666666672 m','valid':'Distance from either cut > 0.005 m; table support checked before band assignment','frame':'Table local equals world because all table origins/orientations zero','geometry':'Latest frame of each 4-frame boundary; native coordinates converted from stored padded224 boxes; public color routing saved before this audit; GT used for grouping/scoring only','depth':'Saved in public archives, not fed to the frozen RGB/box model; not processed by this audit'}})
 assert not label_errors
 analyze(output)
 after=protected();assert before==after
 save(D/'results/complete.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.perf_counter()-started,'rows':len(output),'roots':{s:len(members[s])for s in SPLITS},'protected_files_unchanged':len(before),'fits':0,'model_forward_passes':0,'new_samples':0,'uploads':0,'scope':'CPU descriptive geometry audit; no classifier/threshold fitting; latest frame aligned to GT, h0 independent roots primary; all-boundary secondary'})
 print('AUDIT_COMPLETE',round(time.perf_counter()-started,2),flush=True)
def analyze(rows):
 supports=[];coverage=[];overlap=[];errors=[]
 def selected(split,cohort,cam,color,loc):return [r for r in rows if r['split']==split and r['camera']==cam and r['GT_location']==loc and(color=='all'or r['color']==color)and(cohort=='all_boundaries'or r['h0_unique'])]
 for cohort in ['h0','all_boundaries']:
  for cam in ['static','gripper']:
   for color in ['all']+COLORS:
    for loc in [0,1,2]:
     train=selected('train',cohort,cam,color,loc);tv=[r for r in train if r['visible']];tx=np.array([[r[k]for k in ['u','v','width','height']]for r in tv]);poly=hull(tx[:,:2])if len(tx)else np.empty((0,2))
     for split in SPLITS:
      rr=selected(split,cohort,cam,color,loc);vv=[r for r in rr if r['visible']];r={'cohort':cohort,'camera':cam,'color':color,'location':loc,'split':split,'records':len(rr),'roots':len(set(x['root_id']for x in rr)),'visible':len(vv)}
      for key in ['GT_x_m','u_px','v_px','width_px','height_px']:
       vals=[x[key]for x in (rr if key=='GT_x_m'else vv)]
       for q,name in [(0,'min'),(.05,'p05'),(.5,'median'),(.95,'p95'),(1,'max')]:r[key+'_'+name]=float(np.quantile(vals,q))if vals else None
      supports.append(r)
      if split!='train'and vv and tv:
       x=np.array([[r[k]for k in ['u','v','width','height']]for r in vv]);minmax=((x>=tx.min(0)-1e-9)&(x<=tx.max(0)+1e-9)).all(1);ih=inside(x[:,:2],poly);dist=np.sqrt(((x[:,None,:2]-tx[None,:,:2])**2).sum(-1)).min(1)
       gx=np.array([r['GT_x_m']for r in train]);evalx=np.array([r['GT_x_m']for r in vv]);xinside=(evalx>=gx.min())&(evalx<=gx.max())
       coverage.append({'cohort':cohort,'camera':cam,'color':color,'location':loc,'split':split,'visible':len(vv),'roots':len(set(r['root_id']for r in vv)),'training_visible':len(tv),'training_roots':len(set(r['root_id']for r in tv)),'inside_training_4D_axis_box':int(minmax.sum()),'inside_training_2D_center_convex_hull':int(ih.sum()),'inside_training_GT_x_range':int(xinside.sum()),'nearest_same_class_train_center_distance_native_fraction_median':float(np.median(dist)),'nearest_same_class_train_center_distance_native_fraction_max':float(dist.max())})
    # Descriptive cross-class overlap, never a classifier or selected threshold.
    for split in SPLITS:
     a=[r for r in selected(split,cohort,cam,color,1)if r['visible']];b=[r for r in selected(split,cohort,cam,color,2)if r['visible']]
     if not a or not b:continue
     x=np.array([[r['u'],r['v']]for r in a]);y=np.array([[r['u'],r['v']]for r in b]);lo=max(x[:,0].min(),y[:,0].min());hi=min(x[:,0].max(),y[:,0].max());width=max(0,hi-lo)
     overlap.append({'cohort':cohort,'camera':cam,'color':color,'split':split,'middle_visible':len(a),'right_visible':len(b),'u_interval_overlap_native_fraction':float(width),'middle_centers_inside_right_hull':int(inside(x,hull(y)).sum()),'right_centers_inside_middle_hull':int(inside(y,hull(x)).sum()),'middle_hull_vertices':len(hull(x)),'right_hull_vertices':len(hull(y))})
 for split in SPLITS:
  for seed in [0,1]:
   for color in COLORS:
    rr=[r for r in rows if r['camera']=='static'and r['split']==split and r['color']==color];right=[r for r in rr if r['GT_location']==2];bad=[r for r in right if r[f'object_seed{seed}_prediction']==1]
    errors.append({'split':split,'seed':seed,'color':color,'true_right':len(right),'right_to_middle':len(bad),'error_roots':len(set(r['root_id']for r in bad)),'min_GT_margin_mm':min((r['GT_right_boundary_margin_mm']for r in bad),default=None),'median_GT_margin_mm':float(np.median([r['GT_right_boundary_margin_mm']for r in bad]))if bad else None})
 csvwrite(D/'results/support_ranges.csv',supports);csvwrite(D/'results/train_coverage.csv',coverage);csvwrite(D/'results/class_overlap.csv',overlap);csvwrite(D/'results/right_to_middle.csv',errors)
if __name__=='__main__':collect()
