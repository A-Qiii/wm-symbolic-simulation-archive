from common import *
import csv,collections
import labels_physical as truth

def summarize(rows,relation='Holding',eligible=True):
 if relation=='Holding':known=[r for r in rows if r['eligible']if eligible]if eligible else[r for r in rows if r['physical_valid']];label='y'
 else:known=[r for r in rows if r['In_physical_valid']];label='In_y'
 pos=[r for r in known if r[label]];neg=[r for r in known if not r[label]]
 return {'records':len(rows),'known_records':len(known),'positive_records':len(pos),'negative_records':len(neg),'unknown_records':len(rows)-len(known),'positive_roots':len({r['root_id']for r in pos}),'negative_roots':len({r['root_id']for r in neg}),'roots':len({r['root_id']for r in rows})}
def physical_force(snap,robot,families,cutoff):
 values=[]
 for c in COLORS:
  uid=snap['objects']['block_'+c]['uid'];links=set();external=False
  for x in snap['contacts']:
   if x[9]<=cutoff:continue
   if x[1]==uid:other,link=x[2],x[4]
   elif x[2]==uid:other,link=x[1],x[3]
   else:continue
   if other==robot:links.add(link)
   else:external=True
  values.append(bool(all(links&set(f)for f in families)and not external))
 return values
def main():
 install_guard([]);rows=read(E/'data/records.json');checks=read(E/'results/integrity.json');protocol=read(E/'configs/protocol.json');old=read(E/'configs/historical_exclusion.json');roots=read(E/'results/root_summaries.json');frozen=read(E/'configs/precollection_frozen.json')
 assert len(roots)==32 and len(checks)<=256 and all(sha(p)==h for p,h in frozen['sha256'].items());assert len({r['record_id']for r in rows})==len(rows)
 byroot=collections.defaultdict(set)
 for r in rows:byroot[r['root_id']].add(r['split'])
 assert all(len(s)==1 for s in byroot.values())and len(byroot)==32
 verify={'states':0,'rendered_images':0,'query_boxes':0,'force_trace_checks':0,'native_truth_recomputed':0};oldrgb=set(old['old_RGB_array_hashes']);oldfiles=set(old['old_file_hashes']);overlap=[];dups=collections.defaultdict(list)
 for ck in checks:
  d=E/'data/private'/ck['root_id']/f's{ck["sample"]:02d}';mapping=read(d.parent/'body_link_mapping.json');s=read(d/'physical_truth.json');lab=truth.evaluate(s['physical']);assert lab==s['labels_v2_1'];verify['native_truth_recomputed']+=1
  assert read(d/'snapshot_before.json')==read(d/'snapshot_after.json');assert len(s['last12_controls'])==12
  for frame in s['last12_controls']:
   for i,cut in enumerate([.005,.01,.015]):assert physical_force(frame['physical'],mapping['robot_uid'],mapping['finger_families'],cut)==frame['holding_at_cutoffs'][i];verify['force_trace_checks']+=1
  stable=[all(all(h['holding_at_cutoffs'][f][j]==lab['y'][j]for f in range(3))and all(sum(z)<=1 for z in h['holding_at_cutoffs'])for h in s['last12_controls'])for j in range(3)];assert stable==s['holding_stability_valid'];pub=Path(ck['public_path']);ph=sha(pub)
  if ph in oldfiles:overlap.append({'type':'public_NPZ_old_exact','path':str(pub)})
  with np.load(pub)as public:
   for cam in ['static','gripper']:
    imageid=f'{ck["root_id"]}_s{ck["sample"]:02d}_{cam}'
    with np.load(d/f'native_{cam}.npz')as z:
     assert np.array_equal(public['rgb_'+cam],z['rgb']);seg=z['object_link_mask'];body=seg&((1<<24)-1);link=(seg>>24)-1;fm=[(seg>=0)&(body==mapping['robot_uid'])&np.isin(link,f)for f in mapping['finger_families']];gb=bbox(fm[0]|fm[1]);fc=[int(f.sum())for f in fm];rh=hashlib.sha256(z['rgb'].tobytes()).hexdigest();dups[rh].append(imageid)
     if rh in oldrgb:overlap.append({'type':'individual_RGB_old_exact','image_id':imageid})
     rr=[r for r in rows if r['root_id']==ck['root_id']and r['sample']==ck['sample']and r['camera']==cam];assert len(rr)==3
     for r in rr:
      j=COLORS.index(r['color']);bm=(seg>=0)&(body==mapping['object_uids'][r['color']]);assert bbox(bm)==r['object_bbox']and gb==r['gripper_bbox']and fc==r['finger_pixels'];assert int(bm.sum())==r['object_pixels'];assert r['physical_valid']==stable[j]and r['y']==lab['y'][j]and r['In_y']==lab['y'][j+3]and r['In_physical_valid']==lab['v'][j+3];assert r['eligible']==(r['physical_valid']and r['visually_eligible']);assert r['visually_eligible']==(r['object_pixels']>=4 and min(fc)>=4);assert r['RGB_sha256']==rh;verify['query_boxes']+=1
     verify['rendered_images']+=1
  verify['states']+=1
 strata=[];failures=[]
 for split in ['train','dev','test']:
  rs=[r for r in rows if r['split']==split]
  for kind,names in [('all',['all']),('color',COLORS),('camera',['static','gripper']),('root',sorted({r['root_id']for r in rs})),('anchor',protocol['anchors'])]:
   for name in names:
    subset=rs if kind=='all'else[r for r in rs if r[{'color':'color','camera':'camera','root':'root_id','anchor':'anchor'}[kind]]==name]
    for relation,eligibility in [('Holding','physical'),('Holding','eligible'),('In','physical')]:strata.append({'split':split,'group_type':kind,'group':name,'relation':relation,'scope':eligibility,**summarize(subset,relation,eligibility=='eligible')})
  for color in COLORS:
   ss=summarize([r for r in rs if r['color']==color]);gate=protocol['data_gates']['train_per_color'if split=='train'else'dev_test_per_color']
   for key,minval in gate.items():
    if ss[key]<minval:failures.append({'split':split,'group_type':'color','group':color,'quantity':key,'required':minval,'actual':ss[key],'deficit':minval-ss[key]})
  for cam in ['static','gripper']:
   ss=summarize([r for r in rs if r['camera']==cam]);gate=protocol['data_gates']['each_camera_train'if split=='train'else'each_camera_dev_test']
   for key,minval in gate.items():
    if ss[key]<minval:failures.append({'split':split,'group_type':'camera','group':cam,'quantity':key,'required':minval,'actual':ss[key],'deficit':minval-ss[key]})
 for x in overlap:failures.append({'split':'all','group_type':'integrity','group':'historical_overlap',**x})
 crosssplit=[]
 for h,ids in dups.items():
  if len(ids)>1:
   splits={next(r['split']for r in rows if i.startswith(r['root_id']))for i in ids}
   if len(splits)>1:crosssplit.append({'RGB_sha256':h,'images':ids,'splits':sorted(splits)})
 if crosssplit:failures.append({'group_type':'integrity','group':'cross_split_exact_RGB','actual':len(crosssplit)})
 audit={'utc':now(),'passed':not failures,'failures':failures,'verification':verify,'historical_overlap':overlap,'within_new_exact_RGB_duplicates':[{'sha256':h,'images':ids}for h,ids in dups.items()if len(ids)>1],'cross_split_exact_RGB':crosssplit,'unknown_reasons':dict(collections.Counter(x for r in rows for x in r['unknown_reasons'])),'controller_status':dict(collections.Counter(r['controller_result']['status']for r in roots)),'controller_failure_reasons':dict(collections.Counter(r['controller_result']['reason']for r in roots if r['controller_result']['reason'])),'post_anchor_failures':[{'root_id':r['root_id'],'reason':r['controller_result']['post_anchor_failure']}for r in roots if' post_anchor_failure'.strip()in r['controller_result']],'missing_anchors':[{'root_id':r['root_id'],'missing':r['missing_anchors']}for r in roots if r['missing_anchors']],'strata':strata,'stop_on_gate_failure':True,'normalization_or_training_started':False}
 save(E/'results/data_audit.json',audit);save(R/'data_audit.json',audit)
 for name,data in [('availability',strata),('gate_failures',failures)]:
  if data:
   fields=list(dict.fromkeys(k for r in data for k in r))
   with(R/f'{name}.csv').open('w')as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(data)
 save(E/'configs/data_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in(E/'data').rglob('*')if p.is_file()},'passed':not failures});print('DATA_GATE',not failures,'failures',len(failures),'states',len(checks),flush=True)
 for r in strata:
  if r['group_type']=='all':print(r,flush=True)
 for f in failures:print('GATE_FAIL',f,flush=True)
if __name__=='__main__':main()
