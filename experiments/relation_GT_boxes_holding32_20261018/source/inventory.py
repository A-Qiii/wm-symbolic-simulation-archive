from pathlib import Path
import json,collections,hashlib,numpy as np,datetime
P=Path('/home/robot/research/projects/wm_symbolic');W=Path('/home/robot/Documents/Codex/2026-10-13/task-4/relation_boxes_src')
read=lambda p:json.loads(p.read_text())
old=P/'experiments/visual_state_semantics_v2_20261015/data/windows.json';windows=read(old);assert all(r['environment']in 'ABC'for r in windows);counts=[]
for split in ['train','dev']:
 rr=[r for r in windows if r['split']==split]
 for rel,offset in [('Holding',0),('In',3)]:
  for j,c in enumerate(['red','blue','pink']):
   pos=[r for r in rr if r['new_labels']['v'][offset+j]and r['new_labels']['y'][offset+j]];neg=[r for r in rr if r['new_labels']['v'][offset+j]and not r['new_labels']['y'][offset+j]];counts.append({'split':split,'relation':rel,'color':c,'positive':len(pos),'negative':len(neg),'unknown':len(rr)-len(pos)-len(neg),'positive_roots':len({r['root_id']for r in pos}),'negative_roots':len({r['root_id']for r in neg})})
public=sorted({f for r in windows for f in r['public_frames']});schemas=collections.Counter();withmask=0
for f in public:
 with np.load(f,allow_pickle=False)as z:
  keys=tuple(sorted(z.files));schemas[str(keys)]+=1;withmask+=any(k.startswith('seg')or 'mask'in k for k in keys)
sync=[]
for name in ['endpoint_sync_localizer_256_20261018','endpoint_focal_calibration_freshC16_20261018']:
 e=P/'experiments'/name;rows=[r for r in read(e/'data/public/images.json')if r['scene']in 'ABC'];states={(r['root_id'],r['sample'])for r in rows};hascontact=0
 for root,s in states:
  d=read(e/f'data/annotation_private/{root}/s{s:02d}/native_serialization.json')
  def contains(o):
   if isinstance(o,dict):return any('contact'in k.lower()or contains(v)for k,v in o.items())
   if isinstance(o,list):return any(contains(v)for v in o)
   return False
  hascontact+=contains(d)
 sync.append({'corpus':name,'ABC_roots':len({r['root_id']for r in rows}),'states':len(states),'images':len(rows),'native_same_frame_masks':len(rows),'serialization_with_contact_records':hascontact,'strict_Holding_or_supported_In_labels_derivable_from_saved_record':False,'reason':'poses/joints/velocities preserved but no force-bearing contact records; do not infer Holding from closed jaws or In from geometry alone'})
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_physical_corpus':{'windows':len(windows),'roots':len({r['root_id']for r in windows}),'labels_versions':sorted({r['new_labels']['version']for r in windows}),'unique_public_frames_checked':len(public),'with_native_mask_fields':withmask,'schema_counts':dict(schemas),'physical_label_counts':counts,'native_box_condition_usable':False,'reason':'has synchronous RGB/physical labels but no corresponding native instance/link masks; previous audit did not prove exact old RGB mask recovery'},'native_mask_corpora':sync,'usable_current_intersection_for_strict_GT_box_relation_comparison':0,'decision':'new bounded synchronized supplement required; Holding first, In geometry/support metadata audited but no all-negative In training','old_D_sample_reads':0}
(W/'inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
