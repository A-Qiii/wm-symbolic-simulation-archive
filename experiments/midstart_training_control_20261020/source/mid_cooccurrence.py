from common import *
def state_name(x):return'Table'if x<3 else{3:'In',6:'Held'}.get(int(x),'Other'+str(int(x)))
def signature(cat,i):return'|'.join(f'{COLORS[j]}:{state_name(cat[j])}'for j in range(3)if j!=i)+f'|drawer:{int(cat[3])}'
def main():
 guard();tr=dict(np.load(O/'cache/train_seed0.npz'));dev=dict(np.load(O/'cache/development_seed0.npz'));te=read(O/'cache/train_metadata.json');de=read(O/'cache/development_metadata.json');dix={e['episode_id']:i for i,e in enumerate(de)};support=collections.Counter();weighted={};examples={};rows=[]
 for ei,e in enumerate(te):
  for t in range(1,e['length']):
   cat=tr['cat'][ei,t]
   for i in range(3):
    if cat[i]==3:support[(COLORS[i],signature(cat,i))]+=1
 for seed in [0,1]:
  z=np.load(S/f'configs/schedule_seed{seed}.npz');ct=collections.Counter()
  for ei,t in zip(z['ids'].reshape(-1),z['starts'].reshape(-1)):
   cat=tr['cat'][ei,t]
   for i in range(3):
    if cat[i]==3:ct[(COLORS[i],signature(cat,i))]+=1
  weighted[seed]=ct
  for arm in ARMS:
   z=dict(np.load(S/f'results/development_{arm}_seed{seed}.npz'));ix={(str(z['episode_id'][j]),int(z['origin'][j]),str(z['mode'][j]),int(z['step'][j])):j for j in range(len(z['prob']))}
   for j in range(len(z['prob'])):
    if z['mode'][j]!='current'or z['origin'][j]<=0 or z['relative_step'][j]!=0:continue
    eid=str(z['episode_id'][j]);t=int(z['origin'][j]);ei=dix[eid];cat=z['cat'][j];history=ix[eid,0,'current',t];cn=ix[eid,t,'one_step',t+1];hn=ix[eid,0,'one_step',t+1]
    for i in range(3):
     if cat[i]!=3:continue
     key=(COLORS[i],signature(cat,i));others=[k for k in range(3)if k!=i];table=[k for k in others if cat[k]<3]
     for path,k,nextk in [('cold',j,cn),('history',history,hn)]:
      p=rel(z['prob'][k]);y=truth(cat);npred=rel(z['prob'][nextk]);ny=truth(z['cat'][nextk]);rows.append({'seed':seed,'arm':arm,'path':path,'episode_id':eid,'family_id':str(z['family_id'][j]),'origin':t,'In_object':COLORS[i],'other_combination':key[1],'train_unique_origins':support[key],'train_sampled_initial_support':weighted[seed][key],'In_correct':bool(p[3+i]>=.5),'other_Held_all_correct':bool(np.array_equal(p[others]>=.5,y[others])),'other_table_opportunities':len(table),'invented_Held_on_table':sum(p[k]>=.5 for k in table),'invented_Held_on_table_visible':sum(p[k]>=.5 and dev['availability'][ei,t,k]>0 for k in table),'invented_Held_on_table_unavailable':sum(p[k]>=.5 and dev['availability'][ei,t,k]<=0 for k in table),'next_all_correct':bool(correct(npred[None],ny[None])[0,3]),'all_correct':bool(correct(p[None],y[None])[0,3])})
 summary=[];keys=sorted(set((r['arm'],r['seed'],r['path'],r['In_object'],r['other_combination'])for r in rows))
 for arm,seed,path,color,combo in keys:
  rr=[r for r in rows if(r['arm'],r['seed'],r['path'],r['In_object'],r['other_combination'])==(arm,seed,path,color,combo)];summary.append({'arm':arm,'seed':seed,'path':path,'In_object':color,'other_combination':combo,'train_unique_origins':rr[0]['train_unique_origins'],'train_sampled_initial_support':rr[0]['train_sampled_initial_support'],'development_origins':len(rr),**{k:int(sum(r[k]for r in rr))for k in ['In_correct','other_Held_all_correct','other_table_opportunities','invented_Held_on_table','invented_Held_on_table_visible','invented_Held_on_table_unavailable','next_all_correct','all_correct']}})
 compact=[]
 for seed in [0,1]:
  for arm in ARMS:
   for path in ['cold','history']:
    for stratum in ['ALL','unsupported','supported']:
     rr=[r for r in rows if r['seed']==seed and r['arm']==arm and r['path']==path and(stratum=='ALL'or(r['train_unique_origins']==0 if stratum=='unsupported'else r['train_unique_origins']>0))];compact.append({'seed':seed,'arm':arm,'path':path,'stratum':stratum,'In_object_origins':len(rr),**{k:int(sum(r[k]for r in rr))for k in ['In_correct','other_Held_all_correct','other_table_opportunities','invented_Held_on_table','invented_Held_on_table_visible','invented_Held_on_table_unavailable','next_all_correct','all_correct']}})
 counts=[{'In_object':k[0],'other_combination':k[1],'unique_training_origins':v,'sampled_initial_seed0':weighted[0][k],'sampled_initial_seed1':weighted[1][k]}for k,v in sorted(support.items())]
 result={'utc':now(),'requested_while_training_no_protocol_change':True,'training_support':counts,'summary':summary,'compact':compact,'rows':rows,'limits':['Posthoc association audit, not a randomized causal test of co-occurrence bias.','Same In identity may have different occlusion, geometry, drawer or dynamics. Do not force history-invariant predictions for actually partially observable states.','Unit is In-object at a nonterminal boundary; two In objects can contribute twice and multiple boundaries share episodes/families.','Only task symbols and subsequent consequences are compared. No hidden-vector alignment target is used.'],'next_round_if_needed':'Small balanced paired design only, not executed here: for each In identity cross another object Held vs on-table, while matching drawer, visible geometry, settling and robot/gripper state as far as physically compatible; use multiple public histories reaching verified same current observations/dynamics and separately include intentionally aliased occluded states. Freeze counterbalanced support, false-positive and next-action outcomes before collection. Do not label physically different hidden states as identical solely because a camera view matches.'};save(S/'results/cooccurrence_audit.json',result);csvwrite(S/'results/cooccurrence_combinations.csv',summary);csvwrite(S/'results/cooccurrence_compact.csv',compact);print('COOCCURRENCE_AUDIT_COMPLETE',len(rows),len(summary),flush=True)
if __name__=='__main__':main()
