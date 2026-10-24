from common import *
def flags(z):
 p=rel(z['prob']);y=truth(z['cat']);valid=z['cat_valid'];ok=correct(p,y);ok[:,:2]&=valid[:,:3].all(1)[:,None];ok[:,2]&=valid[:,3];ok[:,3]=ok[:,:3].all(1);return p,y,ok
def main():
 guard();verify();step_rows=[];episode_rows=[];family_rows=[];ret=[];probes=[];close=[];first=[];semantic=[];allpoint=[]
 for path in sorted((S/'results').glob('predictions_*.npz')):
  z=dict(np.load(path));p,y,ok=flags(z);arm=str(z['arm'][0]);seed=int(z['seed'][0])
  for template in sorted(set(z['template'])):
   for mode in sorted(set(z['mode'][z['template']==template])):
    for branch in ['all',1,2]:
     select=(z['template']==template)&(z['mode']==mode)
     if branch!='all':select&=z['branch']==branch
     ix=np.flatnonzero(select);meta={'arm':arm,'seed':seed,'template':str(template),'mode':str(mode),'branch':branch}
     for t in sorted(set(z['step'][ix])):
      ids=ix[z['step'][ix]==t];step_rows.append({**meta,'step':int(t),'n':len(ids),'Held_correct':int(ok[ids,0].sum()),'In_correct':int(ok[ids,1].sum()),'drawer_correct':int(ok[ids,2].sum()),'all_correct':int(ok[ids,3].sum()),'invalid_GT':int((~z['cat_valid'][ids,:4].all(1)).sum())})
  for eid in sorted(set(z['episode_id'])):
   ids=np.flatnonzero(z['episode_id']==eid);i=ids[0];meta={k:str(z[k][i])for k in ['episode_id','root_id','family_id','template','held_color','target_color']};meta.update(arm=arm,seed=seed,branch=int(z['branch'][i]));ix={(str(z['mode'][j]),int(z['step'][j])):j for j in ids};length=max(z['step'][ids]);h=COLORS.index(meta['held_color']);target=COLORS.index(meta['target_color'])
   for mode in sorted(set(z['mode'][ids])):
    jj=sorted([j for j in ids if z['mode'][j]==mode],key=lambda j:z['step'][j]);rr={**meta,'mode':str(mode),'boundaries':len(jj),'Held_whole':bool(ok[jj,0].all()),'In_whole':bool(ok[jj,1].all()),'drawer_whole':bool(ok[jj,2].all()),'all_whole':bool(ok[jj,3].all()),'all_step_accuracy':float(ok[jj,3].mean())};episode_rows.append(rr)
    for j in jj:allpoint.append({**meta,'mode':str(mode),'step':int(z['step'][j]),'Held':bool(ok[j,0]),'In':bool(ok[j,1]),'drawer':bool(ok[j,2]),'all_relation':bool(ok[j,3])})
    bad=next((j for j in jj if not ok[j,3]),None)
    if bad is not None:first.append({**meta,'mode':str(mode),'step':int(z['step'][bad]),'skill':str(z['skill'][bad]),'truth':y[bad].tolist(),'predicted':p[bad].tolist()})
   for t in range(1,length+1):
    cur=ix['current',t];prev=ix['current',t-1];prior=ix['one_step',t];skill=str(z['skill'][cur]);action_target=target if skill.startswith('lift_')else next((j for j in range(3)if y[prev,j]),-1)if skill=='place_in_drawer'else -1
    for relation,offset in [('Held',0),('In',3)]:
     for ob in range(3):
      if relation=='In'and ob==action_target:continue
      stable=bool(y[prev,offset+ob]and y[cur,offset+ob]);pp=bool(p[prior,offset+ob]>=.5);po=bool(p[cur,offset+ob]>=.5);ret.append({**meta,'step':t,'relation':relation,'object':COLORS[ob],'stable_true':stable,'prior_positive':pp,'posterior_positive':po,'forgot':stable and pp and not po,'retained_from_previous_observation':stable and p[prev,offset+ob]>=.5 and po})
    semantic.append({**meta,'step':t,'skill':skill,'status':str(z['status'][cur]),'task_success':bool(z['task_success'][cur]),'true_target_Held':bool(y[cur,target]),'true_target_In':bool(y[cur,3+target]),'true_H_In':bool(y[cur,3+h]),'true_hand_empty':bool(not y[cur,:3].any()),'current_target_Held_correct':bool((p[cur,target]>=.5)==y[cur,target]),'one_step_target_Held_correct':bool((p[prior,target]>=.5)==y[cur,target]),'current_H_In_correct':bool((p[cur,3+h]>=.5)==y[cur,3+h]),'one_step_H_In_correct':bool((p[prior,3+h]>=.5)==y[cur,3+h])})
   for j in ids:
    if z['skill'][j]=='close_drawer':close.append({**meta,'mode':str(z['mode'][j]),'step':int(z['step'][j]),'status':str(z['status'][j]),'task_success':bool(z['task_success'][j]),'true_drawer':int(z['cat'][j,3]),'pred_drawer':int(z['prob'][j,24:27].argmax()),'drawer_correct':bool(ok[j,2]),'all_correct':bool(ok[j,3])})
  q=dict(np.load(S/f'results/probes_{arm}_seed{seed}.npz'));pp,yy,oo=flags(q)
  for j in range(len(q['arm'])):
   meta={k:str(q[k][j])for k in ['episode_id','root_id','family_id','template','held_color','target_color']};meta.update(arm=arm,seed=seed,branch=int(q['branch'][j]),action_index=int(q['action_index'][j]),completion_control=int(q['completion_control'][j]));prev=np.flatnonzero((q['episode_id']==q['episode_id'][j])&(q['action_index']==q['action_index'][j])&(q['completion_control']==q['completion_control'][j]-4));target=COLORS.index(meta['target_color']);h=COLORS.index(meta['held_color'])
   row={**meta,'true_target_Held':bool(yy[j,target]),'target_Held_readout':bool(pp[j,target]>=.5),'true_H_In':bool(yy[j,3+h]),'H_In_readout':bool(pp[j,3+h]>=.5),'Held_all_correct':bool(oo[j,0]),'In_all_correct':bool(oo[j,1]),'drawer_correct':bool(oo[j,2]),'all_correct':bool(oo[j,3]),'positive_Held_interval':bool(q['continuous_Held'][j,target]),'positive_H_In_interval':bool(q['continuous_In'][j,h]),'Held_interval_both_readouts':False,'H_In_interval_both_readouts':False,'Held_previous_correct':False,'Held_forgot':False,'H_In_forgot':False}
   if len(prev)==1:
    k=prev[0];row.update(Held_interval_both_readouts=bool(pp[k,target]>=.5 and pp[j,target]>=.5),H_In_interval_both_readouts=bool(pp[k,3+h]>=.5 and pp[j,3+h]>=.5),Held_previous_correct=bool(pp[k,target]>=.5),Held_forgot=bool(q['continuous_Held'][j,target]and pp[k,target]>=.5 and pp[j,target]<.5),H_In_forgot=bool(q['continuous_In'][j,h]and pp[k,3+h]>=.5 and pp[j,3+h]<.5))
   probes.append(row)
 summaries=[]
 for seed in [0,1]:
  for arm in ARMS:
   for template in sorted({r['template']for r in episode_rows}):
    for mode in sorted({r['mode']for r in episode_rows if r['template']==template}):
     for branch in ['all',1,2]:
      rr=[r for r in episode_rows if r['arm']==arm and r['seed']==seed and r['template']==template and r['mode']==mode and(branch=='all'or r['branch']==branch)];meta={'arm':arm,'seed':seed,'template':template,'mode':mode,'branch':branch};summaries.append({**meta,'episodes':len(rr),**{k:float(np.mean([r[k]for r in rr]))for k in ['Held_whole','In_whole','drawer_whole','all_whole','all_step_accuracy']}})
     for fid in sorted({r['family_id']for r in episode_rows}):
      rr=[r for r in episode_rows if r['arm']==arm and r['seed']==seed and r['template']==template and r['mode']==mode and r['family_id']==fid];family_rows.append({'arm':arm,'seed':seed,'template':template,'mode':mode,'family_id':fid,'episodes':len(rr),'all_whole':float(np.mean([r['all_whole']for r in rr])),'all_step_accuracy':float(np.mean([r['all_step_accuracy']for r in rr]))})
 probe_summary=[]
 for seed in [0,1]:
  for arm in ARMS:
   for template in ['ALL']+sorted({r['template']for r in probes}):
    rr=[r for r in probes if r['arm']==arm and r['seed']==seed and(template=='ALL'or r['template']==template)];h=[r for r in rr if r['positive_Held_interval']];i=[r for r in rr if r['positive_H_In_interval']];hp=[r for r in rr if r['true_target_Held']];runs=collections.defaultdict(list)
    for r in rr:runs[(r['episode_id'],r['action_index'])].append(r)
    complete=[v for v in runs.values()if len(v)==3];positive_runs=[v for v in complete if all(r['true_target_Held']for r in v)]
    probe_summary.append({'arm':arm,'seed':seed,'template':template,'windows':len(rr),'all_capture_runs':len(runs),'complete_runs':len(complete),'positive_Held_frames':len(hp),'positive_Held_frame_correct':sum(r['target_Held_readout']for r in hp),'continuous_Held_intervals':len(h),'continuous_Held_both_correct':sum(r['Held_interval_both_readouts']for r in h),'Held_previous_correct_intervals':sum(r['Held_previous_correct']for r in h),'Held_forgot':sum(r['Held_forgot']for r in h),'continuous_H_In_intervals':len(i),'continuous_H_In_both_correct':sum(r['H_In_interval_both_readouts']for r in i),'H_In_forgot':sum(r['H_In_forgot']for r in i),'positive_complete_Held_runs':len(positive_runs),'positive_complete_Held_runs_all_correct':sum(all(r['target_Held_readout']for r in v)for v in positive_runs)})
 contrasts=[]
 for seed in [0,1]:
  for template in sorted({r['template']for r in family_rows}):
   for mode in sorted({r['mode']for r in family_rows if r['template']==template}):
    vals={a:{r['family_id']:r['all_whole']for r in family_rows if r['seed']==seed and r['arm']==a and r['template']==template and r['mode']==mode}for a in ARMS};fids=sorted(vals['baseline']);d=np.array([vals['geometry'][f]-vals['baseline'][f]for f in fids]);rng=np.random.default_rng(261024100);boots=d[rng.integers(len(d),size=(2000,len(d)))].mean(1);contrasts.append({'seed':seed,'template':template,'mode':mode,'difference':float(d.mean()),'lo':float(np.quantile(boots,.025)),'hi':float(np.quantile(boots,.975)),'families':len(d)})
 for name,rows in [('per_step_metrics',step_rows),('episode_metrics',episode_rows),('family_metrics',family_rows),('retention_opportunities',ret),('probe_metrics',probes),('probe_summary',probe_summary),('close_metrics',close),('semantic_outcomes',semantic),('summary',summaries),('contrasts',contrasts),('point_metrics',allpoint)]:csvwrite(S/f'results/{name}.csv',rows);save(S/f'results/{name}.json',rows)
 save(S/'results/first_errors.json',first);save(S/'results/analysis.json',{'utc':now(),'summaries':summaries,'probe_summary':probe_summary,'contrasts':contrasts,'no_model_selection_or_training':True,'no_acceptance_gate_added':True,'old_failed_gates_unchanged':True,'independent_new_geometry':True,'new_order_paired_with_same_multiset_and_initial_state':True,'hold_transition_not_supported_or_claimed':True,'closed_failures_included_in_full_scores':True});print('SCORING_COMPLETE',len(episode_rows),len(probes),flush=True)
if __name__=='__main__':main()
