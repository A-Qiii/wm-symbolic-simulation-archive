from common import *
from mid_cooccurrence import state_name
def relsig(cat):return'|'.join(f'{c}:{state_name(cat[i])}'for i,c in enumerate(COLORS))+f'|drawer:{int(cat[3])}'
def abstract(cat):return'|'.join(f'{k}:{v}'for k,v in sorted(collections.Counter(state_name(x)for x in cat[:3]).items()))+f'|drawer:{int(cat[3])}'
def main():
 guard();eps=read(O/'cache/train_metadata.json');z=np.load(O/'cache/train_seed0.npz');states=collections.Counter();abstracts=collections.Counter();conditions=collections.Counter();origins=collections.defaultdict(set)
 for i,e in enumerate(eps):
  for t in range(e['length']):
   sig=relsig(z['cat'][i,t]);act=e['rows'][t]['skill'];states[sig]+=1;abstracts[abstract(z['cat'][i,t])]+=1;conditions[(sig,act)]+=1;origins[(sig,act)].add(e['root_id'])
 coverage=[{'relation_combination':s,'requested_action':a,'original_training_occurrences':n,'training_roots':len(origins[s,a]),'retained_complete_trajectories_in_both_mixed_arms':True}for(s,a),n in sorted(conditions.items())]
 evaluated=[]
 for source in ['development','exploratory','independent']:
  path=S/f'results/{source}_full_seed0.npz'
  if not path.exists():continue
  q=np.load(path);ix={(str(q['episode_id'][j]),int(q['step'][j])):j for j in range(len(q['prob']))if q['mode'][j]=='current'and q['origin'][j]==0}
  for(eid,t),j in ix.items():
   if(eid,t+1)not in ix:continue
   nxt=ix[eid,t+1];sig=relsig(q['cat'][j]);act=str(q['skill'][nxt]);evaluated.append({'source':source,'episode_id':eid,'family_id':str(q['family_id'][j]),'template':str(q['template'][j]),'branch':int(q['branch'][j]),'step':t,'relation_combination':sig,'abstract_combination':abstract(q['cat'][j]),'requested_action':act,'state_seen_original_training':states[sig]>0,'abstract_state_seen':abstracts[abstract(q['cat'][j])]>0,'state_action_seen_original_training':conditions[sig,act]>0})
 summary=[]
 for source in sorted(set(r['source']for r in evaluated)):
  for template in ['ALL']+sorted(set(r['template']for r in evaluated if r['source']==source)):
   rr=[r for r in evaluated if r['source']==source and(template=='ALL'or r['template']==template)];summary.append({'source':source,'template':template,'action_boundaries':len(rr),'identity_state_seen':sum(r['state_seen_original_training']for r in rr),'abstract_state_seen':sum(r['abstract_state_seen']for r in rr),'identity_state_action_seen':sum(r['state_action_seen_original_training']for r in rr),'unseen_identity_combinations':sorted(set(r['relation_combination']for r in rr if not r['state_seen_original_training'])),'unseen_state_action_conditions':sorted(set(r['relation_combination']+' -> '+r['requested_action']for r in rr if not r['state_action_seen_original_training']))})
 # Feasibility inventory only. No new split/model is selected or trained.
 all_eps=[]
 for split in ['train','development','historical_evaluation']:
  e=read(O/f'cache/{split}_metadata.json');v=np.load(O/f'cache/{split}_seed0.npz')['cat']
  for i,x in enumerate(e):all_eps.append((split,x,v[i,:x['length']+1]))
 proposals=[]
 for ic in range(3):
  for hc in range(3):
   if ic==hc:continue
   heldout=[];safe=[]
   for split,e,c in all_eps:
    pair=(c[:,ic]==3)&(c[:,hc]==6);(heldout if pair.any()else safe).append((split,e,c))
   bad_roots={e['root_id']for split,e,c in heldout};safe=[(split,e,c)for split,e,c in safe if e['root_id']not in bad_roots]
   proposals.append({'heldout_complete_relation_combination':f'In({COLORS[ic]}) AND Held({COLORS[hc]})','episodes_containing_combination':len(heldout),'roots_containing_combination':len(bad_roots),'families_containing_combination':len({e['family_id']for split,e,c in heldout}),'episodes_remaining_after_root_exclusion':len(safe),'remaining_roots':len({e['root_id']for split,e,c in safe}),'remaining_In_atom_boundaries':sum(int((c[:,ic]==3).sum())for split,e,c in safe),'remaining_Held_atom_boundaries':sum(int((c[:,hc]==6).sum())for split,e,c in safe),'caveat':'Counts before stricter family-level split/role balancing. Entire roots containing the combination must be excluded from training, including every prefix, sibling branch and cached warmup. All existing checkpoints already saw these combinations and cannot serve as withheld-combination models.'})
 save(S/'results/composition_coverage.json',{'utc':now(),'training_coverage':coverage,'evaluation_coverage':evaluated,'summary':summary,'future_feasibility_only':proposals,'interpretation':['Mixing nonterminal starts adds an initialization route for already-existing state/action transitions; it does not expand the union of physical state/action combinations.','All original episodes remain represented by full samples in both mixed arms. Therefore newly available initial In support must not be called compositional generalization.','Fresh geometry is new layout within A/B; known4 vs novel4 order novelty is separate from relation-combination novelty.','Do not equate coarse symbolic signature coverage with full visual or dynamical state coverage.','Next study: freeze partial-combination training and completely withheld combinations, preserve constituent atom/action support, enforce root/family split, train anew, and compare reusable object updates/interaction messages/learning objectives if failure remains. Do not keep adding examples of each failed combination.'],'next_training_started':False});csvwrite(S/'results/training_relation_action_coverage.csv',coverage);csvwrite(S/'results/evaluation_relation_action_coverage.csv',evaluated);csvwrite(S/'results/future_composition_feasibility.csv',proposals);print('COMPOSITION_COVERAGE_COMPLETE',len(coverage),len(evaluated),flush=True)
if __name__=='__main__':main()
