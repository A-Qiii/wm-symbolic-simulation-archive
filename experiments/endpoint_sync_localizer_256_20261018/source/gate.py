from common import *
import csv
def main():
 assert not(E/'results/data_gate.json').exists();p=read(E/'configs/protocol.json');rows=read(E/'data/public/images.json');anns=read(E/'data/annotation_private/labels.json');checks=read(E/'results/integrity.json');assert len(rows)==len(anns)==512 and len(checks)==256
 assert [r['image_id']for r in rows]==[a['image_id']for a in anns]==list(range(512));assert all(c['full_snapshot_equal']and c['RGB_depth_bitwise_equal']and c['same_tick_4calls']for c in checks)
 groups={s:{r['root_id']for r in rows if r['split']==s}for s in ['train','dev','final']};assert not(groups['train']&groups['dev']or groups['train']&groups['final']or groups['dev']&groups['final']);assert [len(groups[s])for s in groups]==[16,8,8]
 paired={};duplicates=[];rgb_ok=True
 for i in range(0,512,2):
  r=rows[i];z=np.load(r['path'],allow_pickle=False);assert set(z.files)=={'rgb_static','rgb_gripper'}
  for rr in rows[i:i+2]:assert hashlib.sha256(z[rr['key']].tobytes()).hexdigest()==rr['RGB_sha256']
  sig=hashlib.sha256(z['rgb_static'].tobytes()+z['rgb_gripper'].tobytes()).hexdigest()
  if sig in paired and paired[sig]['split']!=r['split']:duplicates.append([paired[sig],r])
  paired[sig]=r
 for r,a in zip(rows,anns):
  z=np.load(a['semantic_file'],allow_pickle=False);native=z['whole_drawer'];sem=z['drawer_parts'];assert np.array_equal(sem>0,native);assert np.array_equal(z['front_plus_handle'],np.isin(sem,[1,2]));assert np.array_equal(z['unknown_drawer_pixels'],sem==255)
  masks=[z['front_plus_handle'],z['finger1'],z['finger2']]
  for name,m in zip(['drawer_front','finger1','finger2'],masks):assert bbox(m)==a['targets'][name]['bbox']and int(m.sum())==a['targets'][name]['visible_pixels']
  assert a['class_valid']==[bool(masks[0].any()and not(sem==255).any()),bool(masks[1].any()and masks[2].any())]
 coverage=[];failures=[]
 if duplicates:failures.append({'cross_split_identical_dual_RGB':len(duplicates)})
 for split in ['train','dev','final']:
  for cam in ['static','gripper']:
   ix=[i for i,r in enumerate(rows)if r['split']==split and r['camera']==cam]
   for c,cat in enumerate(CATS):
    positive=[i for i in ix if anns[i]['class_valid'][c]and anns[i]['boxes'][c]];nbox=sum(len(anns[i]['boxes'][c])for i in positive);nroots=len({rows[i]['root_id']for i in positive});hard=[i for i in ix if anns[i]['handle_hardnegative_patches']];nhr=len({rows[i]['root_id']for i in hard});known=sum(anns[i]['class_valid'][c]for i in ix)
    row={'split':split,'camera':cam,'category':cat,'images':len(ix),'roots':len(groups[split]),'known_images':known,'unknown_images':len(ix)-known,'usable_positive_boxes':nbox,'positive_roots':nroots,'whole_image_certified_negative':0,'hardnegative_handle_images':len(hard)if c==1 else 0,'hardnegative_handle_roots':nhr if c==1 else 0,'hardnegative_handle_patches':sum(len(anns[i]['handle_hardnegative_patches'])for i in hard)if c==1 else 0,'visible_instances_including_unknown_classes':sum(int(anns[i]['targets'][name]['visible_pixels']>0)for i in ix for name in (['drawer_front']if c==0 else['finger1','finger2']))};coverage.append(row)
    if nbox<p['gates']['minimum_valid_positive_boxes_per_class_camera'][split]or nroots<p['gates']['minimum_positive_roots_per_class_camera'][split]:failures.append({'positive_coverage':row})
    if c==1 and(len(hard)<p['gates']['minimum_handle_hardnegative_images_per_camera'][split]or nhr<p['gates']['minimum_handle_hardnegative_roots_per_camera'][split]):failures.append({'handle_negative_coverage':row})
 save(E/'results/coverage.json',coverage)
 with (R/'coverage.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
 save(E/'results/data_gate.json',{'utc':now(),'passed':not failures,'failures':failures,'roots_disjoint':True,'dual_RGB_cross_split_duplicates':duplicates,'RGB_only_public_schema':True,'same_capture_integrity':True,'annotation_mask_box_consistency':True,'final_holdout_review':'aggregate coverage and integrity only; no holdout images inspected and no model metrics computed','whole_image_absence_labels':0,'coverage':coverage})
 files=[f for f in (E/'data').rglob('*')if f.is_file()];save(E/'configs/data_frozen.json',{'utc':now(),'sha256':{str(f):sha(f)for f in files}});print('DATA_GATE',not failures,'FAILURES',len(failures),flush=True)
 for x in coverage:print(x,flush=True)
if __name__=='__main__':main()
