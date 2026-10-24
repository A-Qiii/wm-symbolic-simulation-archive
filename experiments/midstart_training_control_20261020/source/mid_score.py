from common import *
def main():
 guard();summary=[];whole=[];first=[];initial=[];ret=[];horizon=[];probe=[]
 for path in sorted((S/'results').glob('*.npz')):
  if path.name.startswith('schedule'):continue
  z=dict(np.load(path));p=rel(z['prob']);y=truth(z['cat']);ok=correct(p,y);valid=z['cat_valid'];ok[:,:2]&=valid[:,:3].all(1)[:,None];ok[:,2]&=valid[:,3];ok[:,3]=ok[:,:3].all(1);arm=str(z['arm'][0]);seed=int(z['seed'][0]);source=str(z['source'][0]);meta={'source':source,'arm':arm,'seed':seed}
  if 'probes_'in path.name:
   for template in ['ALL']+sorted(set(z['template'])):
    indices=np.flatnonzero(np.ones(len(p),bool)if template=='ALL'else z['template']==template);counts=collections.Counter()
    for j in indices:
     h=COLORS.index(z['held_color'][j]);t=COLORS.index(z['target_color'][j]);prev=np.flatnonzero((z['episode_id']==z['episode_id'][j])&(z['action_index']==z['action_index'][j])&(z['completion_control']==z['completion_control'][j]-4))
     for name,off,ob,field in [('Held',0,t,'continuous_Held'),('In',3,h,'continuous_In')]:
      if len(prev)!=1 or not z[field][j,ob]:continue
      k=prev[0];a=p[k,off+ob]>=.5;b=p[j,off+ob]>=.5;counts[name+'_positive_intervals']+=1;counts[name+'_both_correct']+=int(a and b);counts[name+'_previous_correct']+=int(a);counts[name+'_forgot']+=int(a and not b)
    probe.append({**meta,'template':str(template),**counts})
   continue
  startgroups=['full','middle']if source=='development'else['full']
  for startgroup in startgroups:
   base=(z['origin']==0)if startgroup=='full'else(z['origin']>0)
   for template in ['ALL']+sorted(set(z['template'])):
    for branch in ['all']+sorted(set(z['branch'])):
     sel=base&(True if template=='ALL'else z['template']==template)&(True if branch=='all'else z['branch']==branch);mm={**meta,'startgroup':startgroup,'template':str(template),'branch':branch}
     for mode in sorted(set(z['mode'])):
      ids=np.flatnonzero(sel&(z['mode']==mode))
      if not len(ids):continue
      summary.append({**mm,'mode':str(mode),'boundaries':len(ids),**{k:int(ok[ids,j].sum())for j,k in enumerate(['Held_correct','In_correct','drawer_correct','all_correct'])}})
      groups=collections.defaultdict(list)
      for j in ids:groups[(str(z['episode_id'][j]),int(z['origin'][j]))].append(j)
      whole.append({**mm,'mode':str(mode),'trajectories':len(groups),**{k:sum(bool(ok[jj,j].all())for jj in groups.values())for j,k in enumerate(['Held_correct','In_correct','drawer_correct','all_correct'])}})
     ids=np.flatnonzero(sel&(z['mode']=='current')&(z['relative_step']==0))
     if len(ids):
      rr={**mm,'starts':len(ids),'all_correct':int(ok[ids,3].sum()),'Held_all_correct':int(ok[ids,0].sum()),'In_all_correct':int(ok[ids,1].sum()),'drawer_correct':int(ok[ids,2].sum())}
      for name,off in [('Held',0),('In',3)]:
       pp=p[ids,off:off+3]>=.5;yy=y[ids,off:off+3].astype(bool);vv=valid[ids,:3];rr.update({name+'_positive':int((yy&vv).sum()),name+'_TP':int((pp&yy&vv).sum()),name+'_negative':int((~yy&vv).sum()),name+'_FP':int((pp&~yy&vv).sum())})
      rr['drawer_confusion']=[[int(((z['cat'][ids,3]==a)&(p[ids,6:9].argmax(1)==b)).sum())for b in range(3)]for a in range(3)];initial.append(rr)
     for h in range(1,7):
      ids=np.flatnonzero(sel&(z['mode']=='free')&(z['relative_step']==h))
      if len(ids):horizon.append({**mm,'horizon':h,'boundaries':len(ids),**{k:int(ok[ids,j].sum())for j,k in enumerate(['Held_correct','In_correct','drawer_correct','all_correct'])}})
  groups=collections.defaultdict(list)
  for j in range(len(p)):groups[(str(z['episode_id'][j]),int(z['origin'][j]),str(z['mode'][j]))].append(j)
  for (eid,origin,mode),jj in groups.items():
   jj=sorted(jj,key=lambda j:z['step'][j]);bad=next((j for j in jj if not ok[j,3]),None)
   if bad is not None:first.append({**meta,'episode_id':eid,'family_id':str(z['family_id'][bad]),'template':str(z['template'][bad]),'branch':int(z['branch'][bad]),'origin':origin,'mode':mode,'relative_step':int(z['relative_step'][bad]),'step':int(z['step'][bad]),'skill':str(z['skill'][bad]),'wrong_relations':','.join(k for j,k in enumerate(['Held','In','drawer'])if not ok[bad,j]),'probability':p[bad].tolist(),'truth':y[bad].tolist()})
   if mode!='current':continue
   for a,b in zip(jj,jj[1:]):
    skill=str(z['skill'][b]);target=COLORS.index(skill.split('_')[1])if skill.startswith('lift_')else next((i for i in range(3)if y[a,i]),-1)if skill=='place_in_drawer'else -1
    for ob in range(3):
     if ob==target or not(y[a,3+ob]and y[b,3+ob]):continue
     pa=p[a,3+ob]>=.5;pb=p[b,3+ob]>=.5;ret.append({**meta,'episode_id':eid,'family_id':str(z['family_id'][a]),'template':str(z['template'][a]),'branch':int(z['branch'][a]),'origin':origin,'step':int(z['step'][b]),'object':COLORS[ob],'both_correct':bool(pa and pb),'previous_correct':bool(pa),'forgot':bool(pa and not pb)})
 for name,rr in [('summary',summary),('whole',whole),('initial',initial),('horizon',horizon),('retention',ret),('probes',probe),('first_errors',first)]:save(S/f'results/{name}.json',rr);csvwrite(S/f'results/{name}.csv',rr)
 check=[]
 for seed in [0,1]:
  rr={a:next(r for r in initial if r['source']=='development'and r['arm']==a and r['seed']==seed and r['startgroup']=='middle'and r['template']=='ALL'and r['branch']=='all')for a in ARMS};delta=rr['cold_mix']['In_TP']/rr['cold_mix']['In_positive']-rr['full']['In_TP']/rr['full']['In_positive'];joint=rr['cold_mix']['all_correct']>=rr['full']['all_correct'];check.append({'seed':seed,'initial_In_recall_improvement':delta,'joint_not_worse':joint,'passed':delta>=.2 and joint})
 save(S/'results/independent_validation_trigger.json',{'utc':now(),'criteria_frozen_before_training':True,'passed':all(r['passed']for r in check),'checks':check});print('SCORE_COMPLETE',json.dumps(check),flush=True)
if __name__=='__main__':main()
