"""Offline-only frozen v2 labels for appended real transitions."""
from pathlib import Path
import json,collections,yaml
from wm_symbolic.state_labels import annotate_trace
from wm_symbolic.calibrated_labels import calibrated_labels_at
from wm_symbolic.process_events import add_device_events,EVENT_NAMES
from wm_symbolic.checkpointing import PauseRequest
pause=PauseRequest()
P=Path('/home/robot/research/projects/wm_symbolic')
m=json.loads((P/'data/data_manifest.json').read_text());counts=collections.Counter();roots={};skills=collections.Counter()
for r in m['records']:
    trace=Path(r.get('raw_evaluator_trace',r['labels_path']))
    infos=[json.loads(line)['info'] for line in trace.open()]
    cfg=yaml.safe_load((P/'vendor/calvin/calvin_env/conf/scene'/f"calvin_scene_{r['environment']}.yaml").read_text())
    label=add_device_events(annotate_trace(infos,cfg,r['success']),infos)
    label['before']=calibrated_labels_at(infos[0],cfg);label['after']=calibrated_labels_at(infos[-1],cfg)
    target=trace.parent/'labels_process_v2.json';target.write_text(json.dumps(label,indent=2))
    r.update(raw_evaluator_trace=str(trace),labels_path=str(target),label_version='calibrated_geometry_and_process_v2')
    if pause.requested:
        (P/'runs/PAUSED_BY_USER.json').write_text(json.dumps({'status':'paused','stage':'annotation','resume_requires_user_authorization':True},indent=2));raise SystemExit(0)
    counts[r['split']]+=1;roots.setdefault(r['root_id'],set()).add(r['split']);skills[(r['split'],r['skill'])]+=1
assert all(len(s)==1 for s in roots.values()),'Physical root leakage'
m.update(event_names=list(EVENT_NAMES),labels_specification='configs/label_specification_process_v2.json')
for name in ['data_manifest_process_v2.json','data_manifest.json']:
    path=P/'data'/name;temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(m,indent=2));temporary.replace(path)
report={'transitions':len(m['records']),'split_counts':dict(counts),'roots_disjoint':True,
        'root_counts':dict(collections.Counter(next(iter(s)) for s in roots.values())),
        'skill_counts':{f'{split}:{skill}':count for (split,skill),count in skills.items()},'labels':'frozen process_v2'}
(P/'runs/label_distribution_process_v2.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
