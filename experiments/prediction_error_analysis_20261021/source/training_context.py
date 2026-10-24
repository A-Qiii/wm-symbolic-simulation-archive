from pathlib import Path
import json,csv,collections
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');A=P/'experiments/prediction_error_analysis_20261021';S=P/'experiments/data_expansion_training_20261021';M=P/'experiments/midstart_training_control_20261020';O=P/'experiments/object_recurrent_state_run_20261019'
old=json.loads((O/'cache/train_metadata.json').read_text());new=json.loads((S/'cache/train_metadata.json').read_text());eps=old+new;cat=np.concatenate([np.load(O/'cache/train_seed0.npz')['cat'],np.load(S/'cache/train_seed0.npz')['cat']]);col=['red','blue','pink'];out=[];joint=[]
for seed in [0,1]:
 for arm,root in [('old228',M),('new868',S)]:
  z=np.load(root/f'configs/schedule_seed{seed}.npz');cc=collections.Counter();jc=collections.Counter()
  for i,st in zip(z['ids'].flatten(),z['starts'].flatten()):
   e=eps[i];group='old_core'if i<120 else'old_role'if i<228 else'new640'
   for t in range(int(st),e['length']):
    skill=e['rows'][t]['skill'];c=cat[i,t];d=cat[i,t+1];typ='other';ob=-1
    if skill.startswith('lift_'):
     ob=col.index(skill.split('_')[1]);typ='lift_target_In'if c[ob]==3 else'lift_target_Held'if c[ob]==6 else'lift_target_Table'if c[ob]in [0,1,2]else'lift_target_other'
    elif skill=='place_in_drawer':typ='place_single_Held'if sum(c[:3]==6)==1 else'place_no_single_Held'
    elif skill in ['open_drawer','close_drawer']:typ=skill
    cc[group,typ]+=1;jc[group,skill,f'In{sum(c[:3]==3)}_Held{sum(c[:3]==6)}_drawer{c[3]}',int(c[ob])if ob>=0 else-1]+=1
    if skill.startswith('lift_')and c[3]==d[3]:cc[group,'lift_true_drawer_stable']+=1
    if skill=='close_drawer'and d[3]==1:cc[group,'close_ends_middle']+=1
  out.extend({'seed':seed,'arm':arm,'group':g,'context':c,'transitions':v}for(g,c),v in cc.items());joint.extend({'seed':seed,'arm':arm,'group':g,'skill':sk,'before_context':c,'target_cat':ob,'transitions':v}for(g,sk,c,ob),v in jc.items())
for n,rr in [('training_conditional_exposure',out),('training_joint_context_exposure',joint)]:
 (A/f'results/{n}.json').write_text(json.dumps(rr,indent=2)+'\n')
 with(A/f'results/{n}.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
for seed in [0,1]:
 for typ in ['lift_target_In','lift_target_Table','place_single_Held','place_no_single_Held','close_ends_middle','lift_true_drawer_stable']:
  print(seed,typ,{arm:sum(r['transitions']for r in out if r['seed']==seed and r['arm']==arm and r['context']==typ)for arm in ['old228','new868']})
