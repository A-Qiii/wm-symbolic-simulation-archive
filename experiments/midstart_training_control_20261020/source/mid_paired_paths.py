from common import *
def main():
 guard();rows=[];summ=[]
 for seed in [0,1]:
  for arm in ARMS:
   z=dict(np.load(S/f'results/development_{arm}_seed{seed}.npz'));p=rel(z['prob']);y=truth(z['cat']);ok=correct(p,y);ix={(str(z['episode_id'][j]),int(z['origin'][j]),str(z['mode'][j]),int(z['step'][j])):j for j in range(len(p))}
   for j in range(len(p)):
    if z['mode'][j]!='current'or z['origin'][j]<=0 or z['relative_step'][j]!=0:continue
    eid=str(z['episode_id'][j]);origin=int(z['origin'][j]);h=ix[eid,0,'current',origin];cnext=ix[eid,origin,'one_step',origin+1];hnext=ix[eid,0,'one_step',origin+1];assert np.array_equal(z['cat'][j],z['cat'][h]);assert np.array_equal(z['cat'][cnext],z['cat'][hnext])
    for mode,a,b in [('initial',j,h),('same_next_action',cnext,hnext)]:
     rows.append({'arm':arm,'seed':seed,'episode_id':eid,'family_id':str(z['family_id'][j]),'origin':origin,'comparison':mode,'skill':str(z['skill'][a]),'cold_prob':p[a].tolist(),'history_prob':p[b].tolist(),'truth':y[a].tolist(),'cold_Held_correct':bool(ok[a,0]),'history_Held_correct':bool(ok[b,0]),'cold_In_correct':bool(ok[a,1]),'history_In_correct':bool(ok[b,1]),'cold_drawer_correct':bool(ok[a,2]),'history_drawer_correct':bool(ok[b,2]),'cold_all_correct':bool(ok[a,3]),'history_all_correct':bool(ok[b,3]),'initial_any_In':bool(y[j,3:6].any())})
   for mode in ['initial','same_next_action']:
    for subset in ['all','initial_In_positive']:
     rr=[r for r in rows if r['arm']==arm and r['seed']==seed and r['comparison']==mode and(subset=='all'or r['initial_any_In'])];r={'arm':arm,'seed':seed,'comparison':mode,'subset':subset,'boundaries':len(rr)}
     for reln in ['Held','In','drawer','all']:
      for init in ['cold','history']:r[init+'_'+reln+'_correct']=sum(x[init+'_'+reln+'_correct']for x in rr)
      r['history_right_cold_wrong_'+reln]=sum(x['history_'+reln+'_correct']and not x['cold_'+reln+'_correct']for x in rr);r['cold_right_history_wrong_'+reln]=sum(x['cold_'+reln+'_correct']and not x['history_'+reln+'_correct']for x in rr)
     for init in ['cold','history']:
      r[init+'_In_TP']=sum(int((np.array(x[init+'_prob'][3:6])>=.5)[np.array(x['truth'][3:6],bool)].sum())for x in rr)
     r['In_positive']=sum(sum(x['truth'][3:6])for x in rr);summ.append(r)
 save(S/'results/same_boundary_initialization_paths.json',{'utc':now(),'posthoc_addition_requested_while_training':True,'no_training_or_selection_change':True,'same_current_input':'Both predictions index the exact same original cached object/context/availability and public four-control-frame window at the same physical boundary. History path arrives through earlier public observations/actions; cold path resets recurrent prior. Same next-action prior reads no future observation.','state_support_interpretation':'Intermediate physical state and downstream transition were already represented in original training. Missing support is the mapping from empty recurrent prior plus current observation to a usable internal state, and its compatibility with the learned transition function.','rows':rows,'summary':summ});csvwrite(S/'results/same_boundary_initialization_paths.csv',summ);print('PAIRED_PATHS_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
