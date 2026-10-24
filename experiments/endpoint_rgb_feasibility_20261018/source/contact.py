from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'experiments/visual_node_edge_readout_20261018'
rows=json.loads((B/'data/public/states.json').read_text());chosen=[]
for scene in 'AB':
 chosen.append(next(r for r in rows if r['scene']==scene and r['train']and r['state_id'].startswith('official:')))
for scene in 'ABCD':
 pool=[r for r in rows if r['scene']==scene and (r['train']if scene in'AB'else True)]
 for j in [0,len(pool)//2,len(pool)-1]:
  if pool[j]['index']not in[r['index']for r in chosen]:chosen.append(pool[j])
fig,axes=plt.subplots(len(chosen),2,figsize=(7,len(chosen)*2.5))
for i,r in enumerate(chosen):
 z=np.load(r['path']);assert set(z.files)=={'rgb_static','rgb_gripper'}
 for cam,key in enumerate(['rgb_static','rgb_gripper']):
  ax=axes[i,cam];ax.imshow(z[key]);ax.set(title=f"{r['scene']} idx{r['index']} {'static'if cam==0 else'wrist'}");ax.grid(alpha=.35)
fig.tight_layout();fig.savefig('/tmp/endpoint_candidates.png',dpi=100);Path('/tmp/endpoint_sample_rows.json').write_text(json.dumps(chosen,indent=2));print('sample_states',len(chosen))
