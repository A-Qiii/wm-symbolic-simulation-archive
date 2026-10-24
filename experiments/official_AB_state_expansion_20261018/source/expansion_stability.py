"""Frozen long-window validity, used only for evaluation, never model inputs."""
import numpy as np
COLORS=['red','blue','pink'];WINDOW=range(32,97)
def cv(values):
 a=np.asarray(values,dtype=float);return float(a.std()/a.mean())if a.mean()>0 else None
def forces(physical):
 out={};robot=physical['robot_uid'];families=physical['finger_families']
 for color in COLORS:
  uid=physical['objects']['block_'+color]['uid'];mx=[0.,0.];tot=[0.,0.];ext=0.;table_force=0.
  for c in physical['contacts']:
   if c[1]==uid:body,link,n=c[2],c[4],c[7]
   elif c[2]==uid:body,link,n=c[1],c[3],[-x for x in c[7]]
   else:continue
   f=max(0.,float(c[9]))
   if body==robot:
    for j,family in enumerate(families):
     if link in family:mx[j]=max(mx[j],f);tot[j]+=f
   else:ext=max(ext,f)
   if body==physical['table_uid']and link==-1 and f>=.015 and n[2]>=.55 and c[8]<=.003:table_force+=f
  out[color]={'family_max_normal_force_N':mx,'family_total_normal_force_N':tot,'max_external_normal_force_N':ext,'certain_table_base_support_force_N':table_force}
 return out
def evidence(physical,labels):
 out=forces(physical)
 for j,c in enumerate(COLORS):
  x=out[c];f=x['family_max_normal_force_N'];ext=x['max_external_normal_force_N'];y=labels['y'][j]
  hc=bool(min(f)>=.015 and ext<=.005)if y else bool(ext>=.015 or min(f)<=.005)
  g=labels['objects'][c];ic=bool(labels['v'][j+3]and(g['outside']or hc)and(not labels['y'][j+3]or g['certain_contained_load_bearing_paths']))
  x.update(Holding_confident=hc,In_confident=ic)
 return out
def path_force(labels,source,dest):
 g=labels['objects'][source.removeprefix('block_')]
 return sum(e['force_N']for e in g['support_edges']if e['to']==dest and e['force_N']>=.015 and e['world_up_normal_z']>=.55 and e['gap_m']<=.003)
def classify(ticks):
 assert len(ticks)==97 and[t['physics_step']for t in ticks]==list(range(97))
 window=[ticks[s]for s in WINDOW];final=ticks[96];valid=[];reasons=[];details=[]
 for j in range(6):
  color=COLORS[j%3];group='Holding'if j<3 else'In';value=final['labels']['y'][j]
  fail=[]
  for t in window:
   reason='label_changed'if t['labels']['y'][j]!=value else'native_boundary'if not t['labels']['v'][j]else'force_evidence_ambiguous'if not t['confidence'][color][group+'_confident']else None
   if reason:fail.append({'step':t['physics_step'],'reason':reason})
  force_detail={};force_ok=True
  if value and group=='Holding':
   cvs=[cv([t['confidence'][color]['family_total_normal_force_N'][k]for t in window])for k in[0,1]];force_ok=all(x is not None and x<=.25 for x in cvs);force_detail={'family_total_force_CV':cvs,'passes':force_ok}
  if value and group=='In':
   common=set(tuple(p)for p in window[0]['labels']['objects'][color]['certain_contained_load_bearing_paths'])
   for t in window[1:]:common&={tuple(p)for p in t['labels']['objects'][color]['certain_contained_load_bearing_paths']}
   paths=[]
   for path in sorted(common):
    cvs=[cv([path_force(t['labels'],a,b)for t in window])for a,b in zip(path[:-1],path[1:])];paths.append({'path':list(path),'edge_total_force_CV':cvs,'passes':all(x is not None and x<=.25 for x in cvs)})
   force_ok=any(p['passes']for p in paths);force_detail={'common_certain_paths':paths,'passes':force_ok}
  ok=not fail and force_ok;valid.append(ok);reasons.append(fail[0]['reason']if fail else'force_variation_or_path_instability'if not force_ok else'stable_native_force_32_to_96')
  details.append({'fact':(f'Holding({color})'if j<3 else f'In({color},drawer)'),'first_window_failure':fail[0]if fail else None,'failing_tick_count':len(fail),'force_stability':force_detail})
 states=[]
 for j,c in enumerate(COLORS):
  table=all(t['confidence'][c]['certain_table_base_support_force_N']>0 for t in window)
  state='unknown'
  if valid[j]and final['labels']['y'][j]:state='Holding'
  elif valid[j+3]and final['labels']['y'][j+3]:state='In_drawer'
  elif valid[j]and valid[j+3]:state='On_table_negative'if table else'Other_known_negative'
  states.append({'color':c,'state':state,'table_base_support_all_window':table})
 return {'fact_order':[f'Holding({c})'for c in COLORS]+[f'In({c},drawer)'for c in COLORS],'y':final['labels']['y'][:6],'v':valid,'validity_reason':reasons,'diagnostics':details,'object_states':states,'empty_hand':bool(all(valid[:3])and not any(final['labels']['y'][:3])),'stable_window':[32,96],'total_window_ticks':65,'final_step':96}
