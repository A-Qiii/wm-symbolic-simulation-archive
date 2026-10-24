from pathlib import Path
import json,gzip,hashlib,datetime,shutil,csv,xml.etree.ElementTree as ET
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
NAME='endpoint_simulator_truth_audit_20261018';E=P/'experiments'/NAME;R=P/'reports'/NAME
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 assert not E.exists()and not R.exists()
 for d in ['source','configs','data','logs']:(E/d).mkdir(parents=True)
 (R/'figures').mkdir(parents=True)
 shutil.copy2(__file__,E/'source/audit.py')
 manual=P/'experiments/endpoint_box_full40_label_audit_20261018'
 save(manual/'configs/paused_for_truth_check.json',{'utc':now(),'reason':'Latest user request prioritizes simulator truth feasibility.','prepared_images':40,'prepared_class_cells':80,'review_sheets_actually_viewed_this_round':[[0,1,2,3],[4,5,6,7]],'new_final_label_decisions':0,'cleaned40_frozen':False,'saved_prediction_evaluation_runs':0,'previous_user_semantics_v2_8_images_remain_sealed':True,'continuation':'Do not resume subjective difficult-boundary guesses until reliable automatic labeling route is assessed.'})
 for f in Path(__file__).parent.parent.joinpath('full40audit_src').glob('*.py'):shutil.copy2(f,manual/'source'/f.name)
 (P/'reports/endpoint_box_full40_label_audit_20261018/STATUS.md').write_text('40图原始RGB和历史标签参考页已准备；本轮仅查看0–7号审图页，尚未生成任何新的最终标签。此前8图v2裁决保持原封存。本轮没有读取保存预测、重评或训练。按用户最新问题，暂停逐图疑难边界判断，优先核查同期仿真真值。\n')
 refs={}
 def reg(p):
  p=Path(p);refs[str(p)]=sha(p);return p
 base=P/'experiments/endpoint_box_localizer_20261018';meta=read(reg(base/'data/public/images.json'))
 node=read(reg(P/'experiments/visual_node_edge_readout_20261018/data/public/states.json'))
 cov=read(reg(P/'experiments/official_readout_coverage_control_20261018/data/public/states.json'));covmap={r['state_id']:r for r in cov}
 rows=[];mappings=[]
 for m in meta[::2]:
  n=node[m['index']];assert n['state_id']==m['state_id'];chain=[m['path'],m['source'],n['source_path']]
  if m['state_id'].startswith('official:'):
   c=covmap[m['state_id']];raw=Path(c['source_observation_path']);chain.append(str(raw));trace=reg(raw.parent/'physics_history.jsonl.gz')
   with gzip.open(trace,'rt')as f:rec=next(json.loads(line)for line in f if '"physics_step":96,'in line)
   assert rec['new_observation_path']==str(raw)and rec['new_observation_sha256']==sha(raw)
   assert rec['physics_step']==96
   kind='official_state_initialized_resimulation_step96';state_fields={k:len(v)for k,v in rec['state'].items()};physical_fields=list(rec['physical']);mapping={k:rec['physical'][k]for k in ['robot_uid','table_uid','drawer_link','finger_families']};mappings.append(mapping)
   config=reg(P/f'experiments/official_AB_state_expansion_20261018/configs/resolved_native_{m["scene"]}.yaml')
  else:
   raw=Path(n['source_path']);trace=reg(raw.parent.parent/'_private/states.json');ss=read(trace);rec=next(r for r in ss if r['frame']==int(raw.stem));assert rec['public_path']==str(raw)
   kind='native_collection_recorded_control_frame';state_fields={'robot_obs':15};physical_fields=list(rec['physical']);mapping={k:rec['physical'][k]for k in ['table_uid','drawer_link']}
   config=reg(raw.parent.parent/'_private/provenance.json')
  with np.load(reg(raw),allow_pickle=False)as z:
   fields=list(z.files);same=[]
   for c,cam in enumerate(['rgb_static','rgb_gripper']):
    r=meta[m['image_id']+c];assert hashlib.sha256(z[cam].tobytes()).hexdigest()==r['RGB_sha256'];same.append(r['image_id'])
  for path in chain:reg(path)
  row={'image_ids':same,'root_id':m['root_id'],'split':m['split'],'scene':m['scene'],'state_id':m['state_id'],'source_type':kind,'source_observation':str(raw),'RGB_bitwise_matches_current_images':True,'raw_fields':fields,'same_capture_segmentation_available':any('seg'in k or'mask'in k for k in fields),'same_capture_depth_available':all('depth_'+c in fields for c in ['static','gripper']),'frame_state_evidence':str(trace),'frame_state_fields':state_fields,'physical_fields':physical_fields,'runtime_mapping':mapping,'config_or_provenance':str(config),'exact_recovery_proven':False,'recovery_limit':'No per-camera captured matrices or separate two finger joint positions; robot_obs stores total opening width. No exact renderer replay proved.','lineage':chain}
  rows.append(row)
 assert len(rows)==20 and not any(r['same_capture_segmentation_available']for r in rows)
 assert all(x==mappings[0]for x in mappings)
 save(E/'data/source_inventory20_states.json',rows)
 # Code/resource inspection, no PyBullet import or simulation creation.
 srcs=['src/wm_symbolic/env_adapter.py','src/wm_symbolic/physical_oracle.py','vendor/calvin/calvin_env/calvin_env/camera/camera.py','vendor/calvin/calvin_env/calvin_env/camera/static_camera.py','vendor/calvin/calvin_env/calvin_env/camera/gripper_camera.py','vendor/calvin/calvin_env/calvin_env/envs/play_table_env.py','vendor/calvin/calvin_env/calvin_env/robot/robot.py','vendor/calvin/calvin_env/conf/config_data_collection.yaml','vendor/calvin/calvin_env/conf/robot/panda_longer_finger.yaml','experiments/official_AB_state_expansion_20261018/source/official_expansion_run.py','experiments/intermediate_data_prediction_20261016/source/collect.py','experiments/independent_identity_D12_20261017/source/collect.py']
 for p in srcs:
  if(P/p).exists():reg(P/p)
 resource=P/'vendor/calvin/calvin_env/data';robot=ET.parse(reg(resource/'franka_panda/panda_longer_finger.urdf')).getroot()
 robot_links={l.attrib['name']:[g.attrib for g in l.findall('./visual/geometry/mesh')]for l in robot.findall('link')if 'finger'in l.attrib['name']}
 tables={}
 for s in 'ABCD':
  path=reg(resource/f'calvin_table_{s}/urdf/calvin_table_{s}.urdf');tree=ET.parse(path).getroot();l=next(x for x in tree.findall('link')if x.attrib['name']=='drawer_link');mesh=reg(resource/f'calvin_table_{s}/meshes/drawer_link.obj');lines=mesh.read_text().splitlines()
  tables[s]={'urdf':str(path),'all_link_names':[x.attrib['name']for x in tree.findall('link')],'drawer_visuals':[g.attrib for g in l.findall('./visual/geometry/mesh')],'mesh':str(mesh),'mesh_objects_groups_materials':[a for a in lines if a.startswith(('o ','g ','usemtl '))],'vertices':sum(a.startswith('v ')for a in lines),'faces':sum(a.startswith('f ')for a in lines),'dedicated_front_or_handle_link':False}
 save(E/'data/resource_mapping.json',{'loaded_AB_runtime_mapping_all16_states':mappings[0],'robot_finger_visual_links':robot_links,'drawer_meshes':tables,'limit':'drawer_link is whole moving drawer mesh; it is not the requested front-panel-plus-handle semantic subset. Body UID must be recorded per world, never universally hard-coded.'})
 masks=[]
 for s in 'AB':
  p=reg(P/f'runs/masks_{s}.npz')
  with np.load(p,allow_pickle=False)as z:
   for cam in ['static','gripper']:
    seg=z['seg_'+cam];rgb=z['rgb_'+cam];assert seg.shape==rgb.shape[:2]and seg.dtype==np.int32
    vals,cnt=np.unique(seg,return_counts=True);decoded=[{'encoded':int(v),'body':int(v)&((1<<24)-1),'link':(int(v)>>24)-1,'pixels':int(n)}for v,n in zip(vals,cnt)if v!=-1]
    masks.append({'source':str(p),'scene':s,'camera':cam,'shape':list(seg.shape),'dtype':str(seg.dtype),'unique_encoded_values':len(vals),'decoded':decoded,'both_finger_families_present':all(any(r['body']==0 and r['link']in family for r in decoded)for family in [[9,10],[11,12]]),'current40_image_match':any(hashlib.sha256(rgb.tobytes()).hexdigest()==m['RGB_sha256']for m in meta)})
 reg(P/'runs/mask_probe.json');save(E/'data/archived_mask_evidence.json',masks)
 assert all(x['both_finger_families_present']and not x['current40_image_match']for x in masks)
 save(E/'configs/protocol.json',{'utc':now(),'mode':'Read-only code/resource/source-record inspection plus archival mask decoding, no new simulation.','simulation_creations':0,'new_renders':0,'physics_steps':0,'model_loads_or_forwards':0,'training_updates':0,'data_collection':0,'old_files_modified':0,'label_conclusions':'No old RGB receives reconstructed segmentation. No new final labels or prediction evaluation.','archived_mask_examples_not_current40':True})
 save(E/'configs/source_references.json',{'sha256':refs})
 save(E/'data/summary.json',{'current_images':40,'states':20,'source_RGB_hash_matches':40,'simultaneous_mask_images':0,'simultaneous_depth_images':32,'AB_resim_images':32,'native_C_images':4,'native_D_images':4,'historical_probe_mask_images':4,'historical_probe_matches_current40':0,'all_AB_mapping_identical':True,'exact_current40_recovery_proved':False})
 with(R/'source_inventory.csv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=['image_ids','split','root_id','source_type','same_capture_segmentation_available','same_capture_depth_available','RGB_bitwise_matches_current_images','source_observation']);w.writeheader();w.writerows({k:r[k]for k in w.fieldnames}for r in rows)
 print(json.dumps(read(E/'data/summary.json'),ensure_ascii=False))
if __name__=='__main__':main()
