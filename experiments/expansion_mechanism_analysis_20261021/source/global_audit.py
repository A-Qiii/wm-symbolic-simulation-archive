from pathlib import Path
import json,csv,collections
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'experiments/expansion_mechanism_analysis_20261021';S=P/'experiments/data_expansion_training_20261021';A=P/'experiments/prediction_error_analysis_20261021'
def rd(p):return json.loads(p.read_text())
def out(n,r):
 (B/f'results/{n}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 if r:
  with(B/f'results/{n}.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in r for k in x)));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in x.items()}for x in r)
windows=rd(B/'results/matched_windows.json');phy={(r['source'],r['episode_id'],r['step']):r for r in rd(B/'results/physical_outcomes.json')};roots=[];contrast=[];nontarget=[];decode=[];difficult=[]
for so in ['old_development','development','combination_test']:
 for seed in [0,1]:
  for arm in ['old228','new868']:
   z=dict(np.load(S/f'results/predictions_{so}_{arm}_seed{seed}.npz'));p=z['prob'];pred=np.c_[p[:,[6,14,22]]>=.5,p[:,[3,11,19]]>=.5];yy=np.c_[z['cat'][:,:3]==6,z['cat'][:,:3]==3];valid=z['cat_valid'][:,:4].all(1);ok=(pred==yy).all(1)&(p[:,24:27].argmax(1)==z['cat'][:,3]);arg=p[:,:24].reshape(-1,3,8).argmax(2);argok=(np.c_[arg==6,arg==3]==yy).all(1)&(p[:,24:27].argmax(1)==z['cat'][:,3]);ix={(str(e),int(o),str(m),int(t)):i for i,(e,o,m,t)in enumerate(zip(z['episode_id'],z['origin'],z['mode'],z['relative_step']))}
   eids=sorted(set(str(e)for e in z['episode_id'][(z['length']==6)&(z['origin']==0)]))
   for t in range(7):
    allids=[[ix[e,0,'free',k]for k in range(t+1)]for e in eids];assert all(valid[ii].all()for ii in allids);roots.append({'source':so,'seed':seed,'arm':arm,'step':t,'same_root_episodes':len(eids),'endpoint_correct':sum(bool(ok[ii[-1]])for ii in allids),'all0_to_step_correct':sum(bool(ok[ii].all())for ii in allids),'endpoint_correct_after_earlier_error':sum(bool(ok[ii[-1]]and not ok[ii].all())for ii in allids)})
   rr=[r for r in windows if(r['source'],r['seed'],r['arm'])==(so,seed,arm)];jointwrong=thresholdonly=0
   for r in rr:
    eid=r['episode_id'];o=r['origin']
    for t in range(1,4):
     fi=ix[eid,o,'free',t];oi=ix[eid,o,'one_step',t];pr=ix[eid,o,'free',t-1];realprev=ix[eid,o,'current',t-1];ph=phy[so,eid,o+t];sk=ph['skill'];target=int(z['target'][fi]);c=z['cat'][fi,:3];before=z['cat'][pr,:3]
     jointwrong+=int(not ok[fi]);thresholdonly+=int(not ok[fi]and argok[fi])
     if sk.startswith('lift_'):
      ob=target;typ='target_In_stays_In'if before[ob]==3 and c[ob]==3 else'target_Table_to_Held'if before[ob]in [0,1,2]and c[ob]==6 else'target_Table_stays_Table'if before[ob]in [0,1,2]and c[ob]in [0,1,2]else'other'
      for mode,i in [('free',fi),('observed',oi)]:contrast.append({'source':so,'seed':seed,'arm':arm,'episode_id':eid,'family_id':r['family_id'],'origin':o,'relative_step':t,'skill':sk,'condition':typ,'mode':mode,'target_before':int(before[ob]),'target_after':int(c[ob]),'pred_Held':bool(pred[i,ob]),'pred_In':bool(pred[i,ob+3]),'target_Held_In_correct':bool((pred[i,[ob,ob+3]]==yy[i,[ob,ob+3]]).all()),'joint_correct':bool(ok[i]),'pHeld':float(p[i,[6,14,22][ob]]),'declared_intervention':ph['declared_intervention']})
     for mode,i,prev in [('free',fi,pr),('observed',oi,realprev)]:
      for ob in range(3):
       if ob==target:continue
       for off,name in [(0,'Held'),(3,'In')]:
        if yy[fi,off+ob]and yy[pr,off+ob]and z['cat_valid'][fi,ob]and z['cat_valid'][pr,ob]:nontarget.append({'source':so,'seed':seed,'arm':arm,'mode':mode,'skill':sk,'relation':name,'before_correct':bool(pred[prev,off+ob]),'after_correct':bool(pred[i,off+ob]),'both_correct':bool(pred[prev,off+ob]and pred[i,off+ob]),'forgot':bool(pred[prev,off+ob]and not pred[i,off+ob]),'declared_intervention':ph['declared_intervention']})
   decode.append({'source':so,'seed':seed,'arm':arm,'matched_future_predictions':len(rr)*3,'joint_wrong':jointwrong,'joint_wrong_but_argmax_relation_readout_would_match':thresholdonly})
out('same_root_horizons',roots);out('target_state_condition_contrasts',contrast);out('non_target_kept_relations',nontarget);out('decoder_confidence_diagnostic',decode)
for n,data,factors in [('target_contrast_summary',contrast,['source','seed','arm','mode','condition']),('non_target_action_summary',nontarget,['source','seed','arm','mode','skill','relation'])]:
 rows=[]
 for key in sorted({tuple(r[f]for f in factors)for r in data}):
  rr=[r for r in data if tuple(r[f]for f in factors)==key];row={**dict(zip(factors,key)),'forecasts':len(rr)}
  if n.startswith('target'):row.update(target_relation_correct=sum(r['target_Held_In_correct']for r in rr),pred_Held=sum(r['pred_Held']for r in rr),pred_In=sum(r['pred_In']for r in rr),mean_pHeld=float(np.mean([r['pHeld']for r in rr])),declared_intervention=sum(r['declared_intervention']for r in rr))
  else:row.update(both_correct=sum(r['both_correct']for r in rr),forgot=sum(r['forgot']for r in rr),previous_correct=sum(r['before_correct']for r in rr),after_correct=sum(r['after_correct']for r in rr))
  rows.append(row)
 out(n,rows)
# Same physical drawer transition predicted from different origins, with actual actions/results fixed.
events=rd(B/'results/drawer_forecasts.json');dd={(r['source'],r['arm'],r['seed'],r['episode_id'],r['absolute_step'],r['relative_step']):r for r in events};match=[]
for key,r in dd.items():
 if r['relative_step']!=3:continue
 so,arm,se,eid,st,t=key;q=dd.get((so,arm,se,eid,st,1))
 if q:match.append({'source':so,'arm':arm,'seed':se,'episode_id':eid,'absolute_step':st,'skill':r['skill'],'condition':r['condition'],'free3_drawer_correct':r['free_drawer_correct'],'cold1_drawer_correct':q['free_drawer_correct'],'free3_joint_correct':r['free_joint_correct'],'cold1_joint_correct':q['free_joint_correct'],'actual_status':r['status'],'declared_intervention':r['declared_intervention']})
out('same_transition_horizon_pairs',match)
print('ROOT_NEWDEV',[r for r in roots if r['source']=='development'and r['arm']=='new868']);print('DECODE_DEV',[r for r in decode if r['source']=='development'])
