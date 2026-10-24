from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/robot/research/projects/wm_symbolic')
S=P/'runs/stage0_skills_20261013_151214'
out=Path('/home/robot/Documents/Codex/2026-10-13/task-4/figures');out.mkdir(exist_ok=True)
r=json.loads((S/'summary.json').read_text())
families=['drawer','slider','lift','device','push']
values=np.array([[r['budget_comparison'][f][str(b)] for b in [120,240,360]] for f in families])
fig,ax=plt.subplots(figsize=(7,4.5));im=ax.imshow(values,vmin=0,vmax=3,cmap='Blues')
ax.set_xticks(range(3),['120 (4 s)','240 (8 s)','360 (12 s)']);ax.set_yticks(range(5),['Open drawer','Move slider','Lift red block','Light switch','Push blue block'])
ax.set_xlabel('Fixed control budget (steps; simulated seconds at 30 Hz)');ax.set_title('Actual HULC success on 3 native C roots')
for y in range(5):
 for x in range(3):ax.text(x,y,f'{values[y,x]}/3',ha='center',va='center',color='white' if values[y,x]>1 else '#173456',fontsize=13)
fig.colorbar(im,ax=ax,ticks=[0,1,2,3],label='Successful roots out of 3')
fig.tight_layout();fig.savefig(out/'skill_budget.png',dpi=180);plt.close(fig)
d=S/'C_root001_lift_red_block_table';trace=[json.loads(x) for x in (d/'trajectory_evaluator_only.jsonl').read_text().splitlines()]
steps=np.array([x['step'] for x in trace]);tcp=np.array([x['info']['robot_info']['tcp_pos'] for x in trace])
fig,axs=plt.subplots(1,3,figsize=(11.5,3.5),gridspec_kw={'width_ratios':[1,1,1.6]})
for ax,name,label in zip(axs[:2],['before','after'],['Before: requested red','After: blue was lifted']):
 with np.load(d/f'{name}.npz') as obs:ax.imshow(obs['rgb_static'])
 ax.set_title(label);ax.set_xticks([]);ax.set_yticks([])
for name,color in [('block_red','#b62838'),('block_blue','#2369b8')]:
 pos=np.array([x['info']['scene_info']['movable_objects'][name]['current_pos'] for x in trace])
 axs[2].plot(steps,np.linalg.norm(tcp-pos,axis=1),color=color,label=name.replace('block_','')+' block')
axs[2].axvline(47,color='#555',ls='--',label='First closing contact with blue')
axs[2].set_xlabel('Executed control step');axs[2].set_ylabel('Gripper-to-block distance (m)');axs[2].legend(fontsize=8);axs[2].set_title('Offline evidence; never policy input')
fig.tight_layout();fig.savefig(out/'wrong_object.png',dpi=180);plt.close(fig)
print(out)
