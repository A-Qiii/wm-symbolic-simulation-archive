from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/robot/research/projects/wm_symbolic');out=P/'runs/wm_probe_seed0'
rows=[json.loads(x) for x in (out/'training.jsonl').read_text().splitlines()];x=[r['update'] for r in rows]
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
colours={'obs':'#555eaa','sym':'#147eab','out':'#da7135','kl':'#5c6a73'}
for key,label in [('obs','Observation reconstruction'),('sym','Current + one-step symbols'),('out','Outcome + events')]:axes[0].plot(x,[r[key] for r in rows],label=label,color=colours[key])
axes[0].set(yscale='log',xlabel='Optimizer update',ylabel='Normalized training loss (log scale)',title='128 real transitions; minibatch losses');axes[0].legend(fontsize=8)
axes[1].plot(x,[r['kl'] for r in rows],color=colours['kl'],label='Balanced KL');axes[1].axhline(1,color='#aaa',ls='--',label='1 nat floor (32 variables summed)');axes[1].set(xlabel='Optimizer update',ylabel='Balanced KL (nats)',title='Latent regularization');axes[1].legend(fontsize=8)
d=json.loads((out/'sampling_diagnostic.json').read_text());e=json.loads((out/'metrics_sampling8_01000.json').read_text())
values=[d['eval_argmax']['prior_categorical_accuracy'],d['eval_sample']['prior_categorical_accuracy'],e['prior_accuracy']]
bars=axes[2].bar(['Argmax latent\n618 facts','One sampled trace\n618 facts','8-sample mean\n618 facts'],values,color=['#8e979f','#66a5cb','#175b85']);axes[2].set(ylim=(0,1.1),ylabel='One-step categorical accuracy',title='Same frozen checkpoint; training set only')
for bar,v in zip(bars,values):axes[2].text(bar.get_x()+bar.get_width()/2,v+.02,f'{v:.1%}',ha='center')
fig.savefig(P/'reports/figures/probe_training.png',dpi=150);plt.close(fig)
print(P/'reports/figures/probe_training.png')
