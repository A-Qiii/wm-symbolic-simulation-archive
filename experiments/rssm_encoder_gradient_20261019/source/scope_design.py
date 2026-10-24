from common import *
from collections import defaultdict,Counter
def main():
 guard('scope');manifest=read(P/'data/core1200_v1/data_manifest.json');by=defaultdict(list)
 for r in manifest['records']:
  if r['split']in['train','dev']:by[r['episode_id']].append(r)
 roots={};sequences=defaultdict(set);grams={2:defaultdict(set),3:defaultdict(set),4:defaultdict(set)};continuity=Counter();controls=Counter();labels={}
 for eid,rows in by.items():
  rows.sort(key=lambda r:r['step']);rs=[read(r['labels_path'])for r in rows];labels[eid]=rs;r0=rows[0];root=r0['root_id'];split=r0['split'];seq=tuple(r['skill']for r in rows);sequences[split].add(seq)
  init=rs[0]['before']['categorical'];condition='open_held'if 6 in init[:3]else'open_empty'if init[3]==2 else'closed_empty'
  ent=roots.setdefault(root,{'root':root,'split':split,'environment':r0['environment'],'initial':condition,'episodes':[],'skills':Counter()});ent['episodes'].append(eid);ent['skills'].update(seq)
  for k in grams:
   for t in range(len(seq)-k+1):grams[k][seq[t:t+k]].add(root)
  for i,lab in enumerate(rs):
   c=lab['control_samples_including_start']-1;controls[split+'_requests']+=1;controls[split+('_nonzero'if c else'_zero')]+=1
   if i<len(rs)-1:
    assert lab['after']['categorical']==rs[i+1]['before']['categorical'];continuity[split+'_label_links']+=1
  controls[split+'_rootstart3']+=1;controls[split+'_rootstart3_all_actuated']+=int(all(r['control_samples_including_start']>1 for r in rs[:3]))
 train={r for r,x in roots.items()if x['split']=='train'};dev=set(roots)-train;candidates=[]
 for k,d in grams.items():
  for seq,present in sorted(d.items()):
   removed=train&present;retained=train-removed;test=dev&present;skills=Counter()
   for r in retained:skills.update(roots[r]['skills'])
   candidates.append({'length':k,'sequence':list(seq),'removed_train_roots':sorted(removed),'remaining_train_roots':len(retained),'remaining_environments':dict(Counter(roots[r]['environment']for r in retained)),'remaining_initial_conditions':dict(Counter(roots[r]['initial']for r in retained)),'remaining_skill_requests':dict(skills),'C_roots_containing_sequence':sorted(test),'all6skills_remaining':len(skills)==6,'train_sequence_occurrences_after_root_exclusion':0})
 # Enumerate all candidates, then a metadata-only example, never use outcome/model accuracy.
 feasible=[c for c in candidates if c['length']==3 and c['remaining_train_roots']>=30 and len(c['C_roots_containing_sequence'])>=4 and c['all6skills_remaining']and len(c['remaining_initial_conditions'])==3]
 example=feasible[0]if feasible else None
 # Nested balanced samples: each environment/initial-condition stratum hash ranked, then proportionate prefix.
 strata=defaultdict(list)
 for r in train:strata[(roots[r]['environment'],roots[r]['initial'])].append(r)
 for key in strata:strata[key].sort(key=lambda r:hashlib.sha256(('future_scale_plan_261019:'+r).encode()).hexdigest())
 subsets=[]
 for size in [20,40,60]:
  selected=[]
  for key,rr in sorted(strata.items()):
   take=size*len(rr)/60;assert int(take)==take;selected+=rr[:int(take)]
  assert len(selected)==size
  subsets.append({'roots':size,'transitions':size*12,'episodes':size*3,'root_ids':sorted(selected),'strata':dict(Counter(roots[r]['environment']+'/'+roots[r]['initial']for r in selected)),'fixed_updates6000_episode_draws_per_root_average':6000*16/size,'fixed_updates6000_epoch_equivalent':6000*16/(size*3)})
 for a,b in zip(subsets,subsets[1:]):assert set(a['root_ids'])<=set(b['root_ids'])
 out={'utc':now(),'executed_new_fits':0,'new_collection':0,'root_counts':dict(Counter(x['split']for x in roots.values())),'initial_conditions':{s:dict(Counter(x['initial']for x in roots.values()if x['split']==s))for s in ['train','dev']},'controls':dict(controls),'label_continuity':dict(continuity),'four_request_sequences':{s:len(v)for s,v in sequences.items()},'C_unseen_four_request_sequences':len(sequences['dev']-sequences['train']),'composition_candidates':candidates,'metadata_only_example':example,'nested_root_design':subsets,'fixed_eval_C_roots':sorted(dev),'limits':['Already viewed C and data-informed metadata design: exploratory re-split, not a new blind test.','Whole roots and all sibling branches are excluded together. Original trajectories never spliced.','Holding out an action sequence is not necessarily holding out a physical relation or a state-action condition; those require separate label/coverage definitions.','Training 6000 updates at all sizes fixes compute, not exposures per root. A separate equal-epoch schedule would vary compute. Do not mix the two effects.','Known6skills and fixed32outputs do not become open vocabulary by adding samples. New relation/object/skill/composition transfer needs predeclared fresh coverage and roots.','No threshold of perfect local predicates is required to interpret a mechanism effect; report uncertainty, failure kinds, and downstream consequence.']}
 save(E/'results/scope_design.json',out);print(json.dumps({k:v for k,v in out.items()if k not in ['composition_candidates','nested_root_design','fixed_eval_C_roots']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
