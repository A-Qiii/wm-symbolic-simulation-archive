from common import *
import csv,collections

def main():
 install_guard([]);assert not(E/'results/data_gate.json').exists();protocol=read(E/'configs/protocol.json');rows=read(E/'data/public/images.json');anns=read(E/'data/annotation_private/labels.json');checks=read(E/'results/integrity.json');assert len(rows)==len(anns)==256 and len(checks)==128;assert [r['image_id']for r in rows]==[r['image_id']for r in anns]==list(range(256));assert len({r['root_id']for r in rows})==16;assert all(c['full_snapshot_equal']and c['same_tick_4calls']and c['RGB_depth_bitwise_equal']for c in checks)
 old=read(E/'configs/historical_registry.json');hashes=set(old['old_file_hashes'].values());oldrgb=set(old['old_RGB_array_hashes']);rootset=set(old['old_root_ids']);assert not rootset&{r['root_id']for r in rows};dups={'old_initial_snapshot':[],'old_primary_raw_capture':[],'old_dual_RGB_file':[],'old_individual_RGB_array':[],'within_new_initial_snapshots':[],'within_new_dual_RGB':[]};initialseen={};pairseen={}
 for r in rows:
  assert r['scene']=='C'and r['split']=='fresh';assert r['RGB_sha256']==hashlib.sha256(np.load(r['path'])[r['key']].tobytes()).hexdigest()
  if r['RGB_sha256']in oldrgb:dups['old_individual_RGB_array'].append({'root':r['root_id'],'sample':r['sample'],'camera':r['camera']})
 for r in rows[::2]:
  d=E/'data/annotation_private'/r['root_id']/f's{r["sample"]:02d}';z=np.load(r['path']);assert set(z.files)=={'rgb_static','rgb_gripper'};pair=hashlib.sha256(z['rgb_static'].tobytes()+z['rgb_gripper'].tobytes()).hexdigest()
  if pair in pairseen:dups['within_new_dual_RGB'].append([pairseen[pair],[r['root_id'],r['sample']]])
  pairseen[pair]=[r['root_id'],r['sample']]
  if sha(r['path'])in hashes:dups['old_dual_RGB_file'].append([r['root_id'],r['sample']])
  for j in [0,1]:
   if sha(d/f'call{j:02d}.npz')in hashes:dups['old_primary_raw_capture'].append([r['root_id'],r['sample'],j])
  if r['sample']==0:
   h=sha(d/'snapshot_before.json')
   if h in hashes:dups['old_initial_snapshot'].append(r['root_id'])
   if h in initialseen:dups['within_new_initial_snapshots'].append([initialseen[h],r['root_id']])
   initialseen[h]=r['root_id']
 for r,a in zip(rows,anns):
  z=np.load(a['semantic_file']);sem=z['drawer_parts'];assert np.array_equal(sem>0,z['whole_drawer']);assert np.array_equal(z['front_plus_handle'],np.isin(sem,[1,2]));assert np.array_equal(z['unknown_drawer_pixels'],sem==255);masks=[z[k]for k in ['front_plus_handle','finger1','finger2']]
  for name,m in zip(['drawer_front','finger1','finger2'],masks):assert bbox(m)==a['targets'][name]['bbox']and int(m.sum())==a['targets'][name]['visible_pixels']
  assert a['class_valid']==[bool(masks[0].any()and not(sem==255).any()),bool(masks[1].any()and masks[2].any())]
 coverage=[];fails=[]
 for key in ['old_initial_snapshot','old_dual_RGB_file','within_new_initial_snapshots']:
  if dups[key]:fails.append({'duplicate':key,'rows':dups[key]})
 # A dual-array match is detected even when compressed file serialization differs.
 for i in range(0,len(rows),2):
  if rows[i]['RGB_sha256']in oldrgb and rows[i+1]['RGB_sha256']in oldrgb:fails.append({'both_old_camera_array_hashes':[rows[i]['root_id'],rows[i]['sample']]})
 for cam in ['static','gripper']:
  ids=[i for i,r in enumerate(rows)if r['camera']==cam]
  for cid,cat in enumerate(CATS):
   valid=[i for i in ids if anns[i]['class_valid'][cid]];hard=[i for i in ids if anns[i]['handle_hardnegative_patches']];q={'camera':cam,'category':cat,'images':len(ids),'roots':16,'known_images':len(valid),'unknown_images':len(ids)-len(valid),'usable_positive_boxes':sum(len(anns[i]['boxes'][cid])for i in valid),'positive_roots':len({rows[i]['root_id']for i in valid}),'certified_absence_images':0,'handle_hardnegative_images':len(hard)if cid==1 else 0,'handle_hardnegative_roots':len({rows[i]['root_id']for i in hard})if cid==1 else 0,'unknown_reasons':dict(collections.Counter(anns[i]['class_unknown_reasons'][cid]for i in ids if not anns[i]['class_valid'][cid]))};coverage.append(q)
   if q['usable_positive_boxes']<12 or q['positive_roots']<4:fails.append({'coverage':q})
   if cid==1 and(q['handle_hardnegative_images']<8 or q['handle_hardnegative_roots']<4):fails.append({'handle_context':q})
 save(E/'results/deduplication.json',{'utc':now(),'old_root_id_seed_signature_intersections':0,'snapshot_file_hash_count':len(initialseen),'old_hash_registry_entries':len(old['old_file_hashes']),'old_primary_RGB_array_hash_count':len(oldrgb),'intersections':dups,'no_old_sample_files_read':True,'limits':'array comparisons cover known sync256 images; technical/historical sample comparisons also use saved file hashes and root registries; no claim over unregistered history'})
 save(E/'results/coverage.json',coverage);save(E/'results/data_gate.json',{'utc':now(),'passed':not fails,'failures':fails,'states':128,'primary_images':256,'roots':16,'coverage':coverage,'all_synchronous':True,'mask_box_consistency':True,'unknown_not_absence':True})
 with(R/'coverage.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
 files=[f for f in(E/'data').rglob('*')if f.is_file()];save(E/'configs/data_frozen.json',{'utc':now(),'sha256':{str(f):sha(f)for f in files}});print('NEW_C_DATA_GATE',not fails,'failures',len(fails),flush=True)
 for q in coverage:print(q,flush=True)
if __name__=='__main__':main()
