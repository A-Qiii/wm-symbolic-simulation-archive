"""One evidence-backed label correction; original labels and statuses retained."""
from pathlib import Path
import json,hashlib,datetime
from wm_symbolic.core_ownership import drawer_ownership
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'data/core1200_v1';R=P/'runs/core1200_v1'
def correct_manifest():
 path=B/'data_manifest.json';m=json.loads(path.read_text());changes=[]
 for r in m['records']:
  if r['split']=='test':continue
  original=Path(r.get('labels_original_path',r['labels_path']));x=json.loads(original.read_text());info=json.loads((original.parent/'endpoint_info_private.json').read_text());new=dict(x)
  for when in ['before','after']:
   new[when]=drawer_ownership(x[when],info[when])
   if new[when]['categorical']!=x[when]['categorical']:changes.append({'episode_id':r['episode_id'],'step':r['step'],'boundary':when,'original':x[when]['categorical'],'corrected':new[when]['categorical'],'support_paths':{c:v.get('drawer_support_paths') for c,v in new[when]['continuous']['objects'].items() if 'drawer_support_paths' in v}})
  new['schema']='core_labels_ownership_v2';new['original_labels_sha256']=hashlib.sha256(original.read_bytes()).hexdigest();dest=original.parent/'labels_ownership_v2_private.json';dest.write_text(json.dumps(new,indent=2));r['labels_original_path']=str(original);r['labels_path']=str(dest)
 m['label_version']='core_ownership_v2';tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(m,indent=2));tmp.replace(path)
 audit={'correction':'One offline container-ownership correction from direct-contact to upward load-bearing support paths through movable blocks','reason':'First actual second placement at A_core1200_02_b2: red sits on pink, pink on drawer; force-bearing normal Z~1.0 and static/gripper images agree. Direct-contact annotation incorrectly classified red as other.','first_evidence':'data/core1200_v1/episodes/A_core1200_02_b2/03_place_in_drawer/endpoint_info_private.json','controller_changed':False,'original_labels_preserved':True,'official_and_controller_status_unchanged':True,'raw_images_controls_unchanged':True,'test_inspected':False,'changed_boundary_labels':len(changes),'changes':changes,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (R/'ownership_label_correction.json').write_text(json.dumps(audit,indent=2));return audit
if __name__=='__main__':print(json.dumps(correct_manifest(),indent=2))
