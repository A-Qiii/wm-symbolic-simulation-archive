from pathlib import Path
import json
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wm_symbolic.state_labels import labels_at,LOCATIONS
P=Path('/home/robot/research/projects/wm_symbolic');m=json.loads((P/'data/data_manifest.json').read_text());chosen=[];seen=set();roundtrip=[]
for r in m['records']:
 label=json.loads(Path(r['labels_path']).read_text())
 for j,c in enumerate(['red','blue','pink']):
  value=label['after']['categorical'][j]
  if (value in [3,6]) and (c,value) not in seen and label['after']['categorical_valid'][j]:
   seen.add((c,value));chosen.append((r,c,value,label))
  start=np.array(label['before']['continuous']['objects'][c]['position_m']);end=np.array(label['after']['continuous']['objects'][c]['position_m'])
  if label['events'][j] and np.linalg.norm(end-start)<.02:roundtrip.append({'episode_id':r['episode_id'],'step':r['step'],'colour':c,'max_m':label['event_max_displacement_m'][j],'end_m':float(np.linalg.norm(end-start))})
selected=chosen[:6];fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
for ax,item in zip(axes.flat,selected):
 r,c,v,label=item;obs=np.load(r['obs_after']);ax.imshow(obs['rgb_static']);ax.set_title(f'{c}: {LOCATIONS[v]}\n{r["episode_id"]}, step {r["step"]}',fontsize=9);ax.set_xlabel('camera pixel x');ax.set_ylabel('camera pixel y')
for ax in list(axes.flat)[len(selected):]:ax.axis('off')
fig.suptitle('Actual RGB frames and offline state-label audit (not model predictions)');fig.savefig(P/'reports/figures/label_audit.png',dpi=150);plt.close(fig)
f=P/'runs/stage0_container_reference_20261013_152502/C_root002_3_close_drawer/trajectory_evaluator_only.jsonl'
last=json.loads(f.read_text().splitlines()[-1])['info'];cfg=yaml.safe_load((P/'vendor/calvin/calvin_env/conf/scene/calvin_scene_C.yaml').read_text());absolute=labels_at(last,cfg)
result={'audited_examples':[{'episode_id':r['episode_id'],'step':r['step'],'colour':c,'label':LOCATIONS[v],'path':r['obs_after']} for r,c,v,label in selected],
 'moved_then_returned_examples':roundtrip,'reference_chain_final_absolute_label':absolute,
 'container_reference_terminal_conjunction':bool(absolute['categorical'][2]==3 and absolute['categorical'][3]==0 and absolute['categorical_valid'][2] and absolute['categorical_valid'][3]),
 'note':'Current facts come from absolute geometry/contact; success flags never substitute for state. This audit is evaluator-only.'}
(P/'runs/label_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({'roundtrip_events':len(roundtrip),'container_final_goal':result['container_reference_terminal_conjunction']}))
