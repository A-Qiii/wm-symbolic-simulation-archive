from pathlib import Path
import numpy as np,json,hashlib,datetime,collections,csv,os,sys
from scipy.spatial import ConvexHull
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'experiments/spatial_coverage_readout_run_20261019';O=P/'experiments/object_identity_readout_20261019';S=P/'experiments/spatial_coverage_error_audit_20261019';COLORS=['red','blue','pink'];ARMS=['control','coverage'];SPLITS=['old_band','gap','externalC']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v)for v in x]
 return x
def save(p,x):Path(p).write_text(json.dumps(plain(x),ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for q in iter(lambda:f.read(4*1024**2),b''):h.update(q)
 return h.hexdigest()
def guard(event,args):
 if event=='socket.connect':raise RuntimeError('Offline audit')
 if event=='open'and isinstance(args[0],(str,bytes)):
  p=Path(os.fsdecode(args[0])).resolve();m=args[1];f=args[2]if len(args)>2 else 0;writing=(isinstance(m,str)and any(c in m for c in 'wa+'))or(isinstance(f,int)and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and p.is_relative_to(P)and not p.is_relative_to(S):raise RuntimeError('Sealed experiment is read-only')
def csvwrite(p,rows):
 if not rows:return
 with Path(p).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def hull_inside(q,points):
 points=np.unique(np.asarray(points),axis=0)
 if len(points)<3:return False
 try:
  h=ConvexHull(points);return bool(np.all(h.equations[:,:2]@q+h.equations[:,2]<=1e-7))
 except:return False

def main():
 sys.addaudithook(guard);save(S/'protocol.json',{'utc':now(),'mode':'READ_ONLY_EXISTING_DATA','new_simulations':0,'new_model_forwards':0,'new_fits':0,'sources':str(R),'source_manifest_sha256':sha(R/'manifest.json'),'classification':'common_fixed=both controls wrong and both coverage correct; common_unfixed=both controls wrong and both coverage wrong; common_regression=both controls correct and both coverage wrong; seed_discordant_regression=at least one correct-to-wrong but not both. h0 primary independent roots; all-boundaries earliest per root descriptive.'})
 arrays={(s,k,a):dict(np.load(R/f'data/predictions_{a}_seed{k}_{s}.npz'))for s in SPLITS for k in[0,1]for a in ARMS};features={(s,k):dict(np.load((O if s=='externalC'else R)/f'data/features_seed{k}_{s}.npz'))for s in SPLITS+['train']for k in[0,1]}
 idx={(x['episode_id'],int(x['step'])):x for x in read(R/'data/features/index.json')['items']+read(P/'data/core1200_v1/features/index.json')['items']};rows={(x['episode_id'],int(x['step'])):x for x in read(R/'data/data_manifest.json')['records']+read(P/'data/core1200_v1/data_manifest.json')['records']};routes={s:dict(np.load((O if s=='externalC'else R)/f'data/routes_{s}.npz'))for s in SPLITS+['train']};ridx={s:{(str(e),int(h)):i for i,(e,h)in enumerate(zip(a['episode_id'],a['step']))}for s,a in routes.items()};objrecords=[]
 for split in SPLITS+['train']:
  a=features[split,0]
  for i,(eid,h)in enumerate(zip(a['episode_id'],a['step'])):
   eid=str(eid);h=int(h);row=rows[eid,max(h-1,0)];label=read(row['labels_path'])['before'if h==0 else'after'];cache=np.load(idx[eid,h]['path']);ri=ridx[split][eid,h]
   for j,c in enumerate(COLORS):
    slot=int(routes[split]['slots'][ri,j]);g=cache['geometry'][384+slot*4]if slot>=0 else np.zeros(8);bbox=(g[:4]*224-np.array([12,12,12,12])).tolist()if g[4]else None;xy=np.array(label['continuous']['objects'][c]['position_m']);objrecords.append({'split':split,'array_index':i,'root_id':str(a['root_id'][i]),'episode_id':eid,'boundary':h,'h0':h==0 and eid.endswith('_b0'),'color':c,'object_index':j,'truth':int(a['cat'][i,j]),'valid':bool(label['categorical_valid'][j]),'x':float(xy[0]),'y':float(xy[1]),'z':float(xy[2]),'availability':float(a['availability'][i,j]),'static_visible':bool(g[4]),'static_u':(bbox[0]+bbox[2])/2 if bbox else None,'static_v':(bbox[1]+bbox[3])/2 if bbox else None,'static_bbox':bbox,'source_frame':idx[eid,h]['public_inputs'][-1]})
 save(S/'object_records.json',objrecords);lookup={(r['split'],r['array_index'],r['object_index']):r for r in objrecords};facts=[]
 for split in SPLITS:
  base=arrays[split,0,'control']
  for i in range(len(base['cat'])):
   y=base['cat'][i];valid=base['valid'][i];out={}
   for seed in[0,1]:
    for arm in ARMS:
     q=arrays[split,seed,arm]['prob'][i];out[seed,arm]={'cat':[int(q[j*8:j*8+8].argmax())for j in range(3)]+[int(q[24:].argmax())],'holding':[c for j,c in enumerate(COLORS)if q[6+8*j]>=.5],'in':[bool(q[3+8*j]>=.5)for j in range(3)]}
   baseinfo={'split':split,'root_id':str(base['root_id'][i]),'episode_id':str(base['episode_id'][i]),'boundary':int(base['step'][i]),'h0':int(base['step'][i])==0 and str(base['episode_id'][i]).endswith('_b0'),'array_index':i}
   checks=[]
   for j,c in enumerate(COLORS):
    if not valid[j]:continue
    if y[j]<3:checks.append(('table_'+['left','middle','right'][y[j]],c,int(y[j]),{(k,a):out[k,a]['cat'][j]for k in[0,1]for a in ARMS},j))
    checks.append(('inside_drawer'if y[j]==3 else'outside_drawer',c,bool(y[j]==3),{(k,a):out[k,a]['in'][j]for k in[0,1]for a in ARMS},j))
   if valid[:3].all():checks.append(('holding_identity_positive'if any(y[:3]==6)else'empty_hand_identity','all',[c for j,c in enumerate(COLORS)if y[j]==6],{(k,a):out[k,a]['holding']for k in[0,1]for a in ARMS},-1))
   if valid[3]:checks.append(('drawer_open_state','drawer',int(y[3]),{(k,a):out[k,a]['cat'][3]for k in[0,1]for a in ARMS},-1))
   for field,color,truth,pred,j in checks:
    oc=[pred[k,'control']==truth for k in[0,1]];nc=[pred[k,'coverage']==truth for k in[0,1]]
    category='common_fixed'if not any(oc)and all(nc)else'common_unfixed'if not any(oc)and not any(nc)else'common_regression'if all(oc)and not any(nc)else'seed_discordant_regression'if any(o and not n for o,n in zip(oc,nc))else'other'
    if category=='other':continue
    facts.append({**baseinfo,'field':field,'color':color,'object_index':j,'truth':truth,'category':category,'control_seed0':pred[0,'control'],'coverage_seed0':pred[0,'coverage'],'control_seed1':pred[1,'control'],'coverage_seed1':pred[1,'coverage'],'object_record':lookup[split,i,j]if j>=0 else None})
 save(S/'facts.json',facts);inventory=[];representatives=[]
 for cohort in['h0','all_boundaries']:
  for split in SPLITS:
   for field in sorted(set(f['field']for f in facts)):
    for cat in['common_fixed','common_unfixed','common_regression','seed_discordant_regression']:
     q=[f for f in facts if f['split']==split and f['field']==field and f['category']==cat and (cohort!='h0'or f['h0'])];roots=sorted(set(f['root_id']for f in q));inventory.append({'cohort':cohort,'split':split,'field':field,'category':cat,'facts':len(q),'independent_roots':len(roots)})
     for root in roots:
      f=min((f for f in q if f['root_id']==root),key=lambda f:(f['boundary'],f['episode_id'],f['color']));representatives.append({**f,'cohort':cohort})
 csvwrite(S/'case_counts.csv',inventory);save(S/'independent_case_representatives.json',representatives)
 # C coordinate/visibility coverage uses only existing recorded data; h0 is the independent primary view.
 coverage=[];nearest=[]
 for scope in['h0','all_boundaries']:
  training=[r for r in objrecords if r['split']=='train'and r['valid']and r['truth']<3 and(scope!='h0'or r['h0'])]
  tests=[r for r in objrecords if r['split']=='externalC'and r['valid']and r['truth']<3 and(scope!='h0'or r['h0'])]
  for r in tests:
   same=[a for a in training if a['truth']==r['truth']and a['color']==r['color']];pooled=[a for a in training if a['truth']==r['truth']];vis=[a for a in same if a['static_visible']];correct=[]
   for seed in[0,1]:p=arrays['externalC',seed,'coverage']['prob'][r['array_index']];correct.append(int(p[r['object_index']*8:r['object_index']*8+8].argmax())==r['truth'])
   nearest_coord=min(same,key=lambda a:abs(a['x']-r['x']));rec={**r,'scope':scope,'new_train_class_x_min':min(a['x']for a in pooled),'new_train_class_x_max':max(a['x']for a in pooled),'same_color_class_x_min':min(a['x']for a in same),'same_color_class_x_max':max(a['x']for a in same),'within_pooled_x':min(a['x']for a in pooled)<=r['x']<=max(a['x']for a in pooled),'within_same_color_x':min(a['x']for a in same)<=r['x']<=max(a['x']for a in same),'within_same_color_xy_hull':hull_inside([r['x'],r['y']],[[a['x'],a['y']]for a in same]),'within_same_color_static_hull':bool(r['static_visible']and hull_inside([r['static_u'],r['static_v']],[[a['static_u'],a['static_v']]for a in vis])),'correct_seed0':correct[0],'correct_seed1':correct[1],'both_wrong':not any(correct),'nearest_same_color_class_root':nearest_coord['root_id'],'nearest_same_color_class_x':nearest_coord['x'],'nearest_x_distance_m':abs(nearest_coord['x']-r['x'])};coverage.append(rec)
   if scope=='h0':
    j=r['object_index'];i=r['array_index'];candidate=[a for a in training if a['color']==r['color']]
    for seed in[0,1]:
     q=features['externalC',seed]['object'][i,j].astype(float);q/=max(np.linalg.norm(q),1e-9);vectors=np.stack([features['train',seed]['object'][a['array_index'],j]for a in candidate]).astype(float);vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-9);dist=1-vectors@q;order=np.argsort(dist)[:3];same_ix=[z for z,a in enumerate(candidate)if a['truth']==r['truth']];other_ix=[z for z,a in enumerate(candidate)if a['truth']!=r['truth']];own=min(same_ix,key=lambda z:dist[z]);other=min(other_ix,key=lambda z:dist[z]);nearest.append({'root_id':r['root_id'],'color':r['color'],'seed':seed,'truth':r['truth'],'nearest_correct_class':{'root_id':candidate[own]['root_id'],'truth':candidate[own]['truth'],'cosine_distance':float(dist[own])},'nearest_wrong_class':{'root_id':candidate[other]['root_id'],'truth':candidate[other]['truth'],'cosine_distance':float(dist[other])},'nearest_object_features':[{'root_id':candidate[k]['root_id'],'truth':candidate[k]['truth'],'x':candidate[k]['x'],'cosine_distance':float(dist[k])}for k in order]})
 save(S/'C_coverage_records.json',coverage);save(S/'C_object_feature_neighbors.json',nearest)
 summary=[]
 for scope in['h0','all_boundaries']:
  for k in[0,1,2]:
   for subset in['all','both_wrong']:
    q=[r for r in coverage if r['scope']==scope and r['truth']==k and(subset=='all'or r['both_wrong'])]
    summary.append({'scope':scope,'class':k,'subset':subset,'objects':len(q),'roots':len(set(r['root_id']for r in q)),'within_pooled_x':sum(r['within_pooled_x']for r in q),'within_same_color_x':sum(r['within_same_color_x']for r in q),'within_same_color_xy_hull':sum(r['within_same_color_xy_hull']for r in q),'within_same_color_static_hull':sum(r['within_same_color_static_hull']for r in q),'static_visible':sum(r['static_visible']for r in q),'public_roi_available':sum(r['availability']>0 for r in q)})
 save(S/'C_coverage_summary.json',summary);save(S/'compute_complete.json',{'utc':now(),'facts':len(facts),'independent_primary':'h0 b0','new_forwards':0,'new_fits':0,'new_simulations':0,'uploads':0});print('READONLY_ANALYSIS_COMPLETE',len(facts))
if __name__=='__main__':main()
