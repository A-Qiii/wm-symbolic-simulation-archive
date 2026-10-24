from pathlib import Path
import json,collections
from omegaconf import OmegaConf
from wm_symbolic.state_labels import annotate_trace,SPEC
P=Path('/home/robot/research/projects/wm_symbolic')
manifest=json.loads((P/'data/data_manifest.json').read_text());distribution=collections.Counter();changes=collections.Counter();events=collections.Counter()
for r in manifest['records']:
 path=Path(r.get('raw_evaluator_trace',r['labels_path']));infos=[json.loads(line)['info'] for line in path.open()]
 scene=OmegaConf.to_container(OmegaConf.load(P/'vendor/calvin/calvin_env/conf/scene'/f"calvin_scene_{r['environment']}.yaml"))
 label=annotate_trace(infos,scene,r['success']);target=path.parent/'labels_evaluator_only.json';target.write_text(json.dumps(label,indent=2))
 r['raw_evaluator_trace']=str(path);r['labels_path']=str(target)
 for j,(v,ok) in enumerate(zip(label['after']['categorical'],label['after']['categorical_valid'])):
  distribution[f'{j}:{v}' if ok else f'{j}:masked']+=1
  if ok and label['before']['categorical_valid'][j] and v!=label['before']['categorical'][j]:changes[str(j)]+=1
 for j,v in enumerate(label['events']):events[str(j)]+=v
manifest['labels_specification']='configs/label_specification.json'
manifest['root_split']={'train':'A/B native roots','dev':'C seeds 2000+','calibration':'C seeds 3000+','test':'D seeds 4000+'}
(P/'configs/label_specification.json').write_text(json.dumps(SPEC,indent=2));(P/'data/data_manifest.json').write_text(json.dumps(manifest,indent=2))
(P/'runs/label_distribution.json').write_text(json.dumps({'transitions':len(manifest['records']),'categorical_after':dict(distribution),'changed_categorical':dict(changes),'moved_events':dict(events),'training_not_started':True},indent=2))
print('Annotated',len(manifest['records']),'real transitions',dict(changes),flush=True)
