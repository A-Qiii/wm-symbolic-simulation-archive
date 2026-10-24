from pathlib import Path
import json,csv,collections
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'experiments/expansion_mechanism_analysis_20261021';S=P/'experiments/data_expansion_training_20261021';O=P/'experiments/object_recurrent_state_run_20261019';M=P/'experiments/midstart_training_control_20261020'
def rd(p):return json.loads(p.read_text())
def out(n,r):
 (B/f'results/{n}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 if r:
  with(B/f'results/{n}.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in r for k in x)));w.writeheader();w.writerows({k:json.dumps(v)if isinstance(v,(dict,list))else v for k,v in x.items()}for x in r)
old=rd(O/'cache/train_metadata.json');new=rd(S/'cache/train_metadata.json');eps=old+new;c=np.concatenate([np.load(O/'cache/train_seed0.npz')['cat'],np.load(S/'cache/train_seed0.npz')['cat']]);res=[]
for group,lo,hi in [('old228',0,228),('new640',228,868)]:
 for drawer in [0,1,2]:
  term=nonterm=0
  for i in range(lo,hi):
   for t in range(eps[i]['length']+1):
    v=c[i,t]
    if sum(v[:3]==3)==2 and sum(v[:3]==6)==0 and v[3]==drawer:
     if t==eps[i]['length']:term+=1
     else:nonterm+=1
  res.append({'group':group,'drawer':drawer,'nonterminal_boundaries':nonterm,'terminal_boundaries':term})
starts=[];grad=[]
for se in [0,1]:
 for arm,root,tag in [('old228',M,'cold_mix'),('new868',S,'new868')]:
  z=np.load(root/f'configs/schedule_seed{se}.npz');co=collections.Counter()
  for i,t in zip(z['ids'].flatten(),z['starts'].flatten()):
   v=c[i,t]
   if sum(v[:3]==3)==2 and sum(v[:3]==6)==0:co[int(v[3])]+=1
  starts.append({'seed':se,'arm':arm,'In2_Held0_drawer0':co[0],'In2_Held0_drawer1':co[1],'In2_Held0_drawer2':co[2]})
  with(root/f'logs/{tag}_seed{se}.csv').open()as f:r=list(csv.DictReader(f))
  for st in [1,1001]:
   rr=[x for x in r if int(x['update'])>=st];grad.append({'seed':se,'arm':arm,'updates_from':st,'updates':len(rr),'grad_preclip_above1':sum(float(x['grad_norm'])>1 for x in rr),'max_grad_preclip':max(float(x['grad_norm'])for x in rr)})
out('terminal_vs_continuation_support',res);out('cold_start_state_support',starts);out('gradient_clip_audit',grad)
# Within-family same requested lift, same other-object symbolic states and drawer state;
# target Table->Held versus In->In. This controls coarse labels, not full pixel/physical state.
phy=rd(B/'results/physical_outcomes.json');family={so:{e['episode_id']:e['family_id']for e in rd(O/'cache/development_metadata.json'if so=='old_development'else S/f'cache/{so}_metadata.json')}for so in ['old_development','development','combination_test']};matched=[]
for so in ['old_development','development','combination_test']:
 g=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in phy:
  if r['source']!=so or r['declared_intervention']or not r['skill'].startswith('lift_'):continue
  ob=['red','blue','pink'].index(r['skill'].split('_')[1]);c=r['before_cat'];d=r['after_cat'];kind='table'if c[ob]in [0,1,2]and d[ob]==6 else'inside'if c[ob]==3 and d[ob]==3 else None
  if kind is None:continue
  others=tuple(c[j]for j in range(3)if j!=ob);key=(family[so][r['episode_id']],r['skill'],c[3],others);g[key][kind].append(r)
 pairlist=[(k,sorted(v['table'],key=lambda r:(r['episode_id'],r['step']))[0],sorted(v['inside'],key=lambda r:(r['episode_id'],r['step']))[0])for k,v in g.items()if v['table']and v['inside']]
 for se in [0,1]:
  for arm in ['old228','new868']:
   z=dict(np.load(S/f'results/predictions_{so}_{arm}_seed{se}.npz'));ix={(str(e),int(o),str(m),int(t)):i for i,(e,o,m,t)in enumerate(zip(z['episode_id'],z['origin'],z['mode'],z['relative_step']))};p=z['prob']
   for key,t,ins in pairlist:
    ob=['red','blue','pink'].index(key[1].split('_')[1]);ii=[ix[r['episode_id'],r['step']-1,'free',1]for r in [t,ins]];pp=p[ii];held=pp[:,[6,14,22][ob]]>=.5;inside=pp[:,[3,11,19][ob]]>=.5
    matched.append({'source':so,'seed':se,'arm':arm,'family':key[0],'skill':key[1],'drawer_before':int(key[2]),'other_object_states':list(key[3]),'table_episode':t['episode_id'],'table_step':t['step'],'inside_episode':ins['episode_id'],'inside_step':ins['step'],'table_target_correct':bool(held[0]and not inside[0]),'inside_target_correct':bool(inside[1]and not held[1]),'both_state_conditions_correct':bool(held[0]and not inside[0]and inside[1]and not held[1]),'table_pHeld':float(pp[0,[6,14,22][ob]]),'inside_pHeld':float(pp[1,[6,14,22][ob]])})
out('within_family_state_contrast_pairs',matched)
print('PAIRS',[(so,arm,se,len(rr),sum(r['both_state_conditions_correct']for r in rr))for so in ['old_development','development','combination_test']for arm in ['old228','new868']for se in [0,1]if(rr:=[r for r in matched if(r['source'],r['arm'],r['seed'])==(so,arm,se)])])
