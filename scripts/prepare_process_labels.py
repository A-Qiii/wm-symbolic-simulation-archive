"""Add labels alongside frozen v1; do not change an active training manifest."""
from pathlib import Path
import json
import yaml
from wm_symbolic.process_events import add_device_events,EVENT_NAMES
from wm_symbolic.calibrated_labels import calibrated_labels_at
P=Path('/home/robot/research/projects/wm_symbolic');original=json.loads((P/'data/data_manifest.json').read_text());manifest=json.loads(json.dumps(original))
for r in manifest['records']:
 old=Path(r['labels_path']);label=json.loads(old.read_text());trace=Path(r['raw_evaluator_trace']);infos=[json.loads(line)['info'] for line in trace.open()]
 enriched=add_device_events(label,infos)
 cfg=yaml.safe_load((P/'vendor/calvin/calvin_env/conf/scene'/f"calvin_scene_{r['environment']}.yaml").read_text())
 enriched['before']=calibrated_labels_at(infos[0],cfg);enriched['after']=calibrated_labels_at(infos[-1],cfg)
 target=old.parent/'labels_process_v2.json';target.write_text(json.dumps(enriched,indent=2));r['labels_path']=str(target);r['label_version']='calibrated_geometry_and_process_v2'
manifest['event_names']=list(EVENT_NAMES);manifest['labels_specification']='configs/label_specification_process_v2.json'
(P/'data/data_manifest_process_v2.json').write_text(json.dumps(manifest,indent=2))
spec={'extends':'configs/label_specification.json','event_names':list(EVENT_NAMES),'device_event_semantics':'Any control-step sample including macro start/end equals the specified physical state; endpoint return does not erase event',
 'table_regions':'Thirds of the calibrated physical native table mesh in table-local coordinates; never placement-sampling bounds',
 'evidence':['runs/device_event_audit.json','runs/label_geometry_audit.json'],'original_labels_preserved':True,'main_manifest_unchanged':True,'test_data_used':False}
(P/'configs/label_specification_process_v2.json').write_text(json.dumps(spec,indent=2));print('Prepared',len(manifest['records']),'additional process-event label records; active manifest unchanged')
