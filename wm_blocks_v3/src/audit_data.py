"""Private sampling-quality / split audit; relation facts never become labels."""
from paths import ROOT as B, RUNTIME as P
import json,sys,collections
from pathlib import Path
def support(state,upper,lower):
 for c in state['contacts']:
  if c['force_N']<=.02:continue
  if c['bodyA']==upper and c['bodyB']==lower and c['normal_on_B'][2]>.5:return True
  if c['bodyB']==upper and c['bodyA']==lower and c['normal_on_B'][2]<-.5:return True
 return False
def main():
 mode=sys.argv[1]if len(sys.argv)>1 else'fit';partial='--partial'in sys.argv;root=B/'data/v1';plan=json.load(open(B/'configs/experiment_plan.json'));rows={};bad=[];outcomes=[]
 for exp,x in plan['experiments'].items():
  lookup={e['id']:e for e in x['episodes']}
  for file in sorted((root/exp).glob('*/episode.json')):
   try:r=json.load(open(file))
   except json.JSONDecodeError:
    if partial:continue
    raise
   if (r['split']=='test')!=(mode=='test'):continue
   e=lookup[r['id']];initial=r['states'][0];stacked=bool(r['layout_private']['stacked']);actual_stacked=support(initial,'small','large');small_table=support(initial,'small','fixed:table');large_region=0 if r['layout_private']['large_xy'][0]<.075 else 1;small_region=1 if e['recipe'].endswith('smallHeld')or'smallB'in e['recipe']else large_region;actual_regions=[initial['objects'][n]['region']for n in ['large','small']];valid=actual_stacked==stacked and (actual_stacked or small_table)and actual_regions==[large_region,small_region]
   if not valid:bad.append({'id':r['id'],'stacked_intended':stacked,'stacked_actual':actual_stacked,'small_table_support':small_table,'expected_regions':[large_region,small_region],'actual_regions':actual_regions})
   rows[r['id']]=r
   for i,a in enumerate(r['actions']):
    if a['action'][:2]!=['Place','large']:continue
    before=r['states'][i];after=r['states'][i+1];outcomes.append({'id':r['id'],'action':i,'loaded_before':support(before,'small','large'),'from_region':before['objects']['large']['region'],'target':a['action'][2],'large_after':after['objects']['large']['region'],'small_after':after['objects']['small']['region'],'status':a['status']})
 exposure=[]
 if not partial:
  for exp,x in plan['experiments'].items():
   for arm,sets in x['arms'].items():
    for split in (['test']if mode=='test'else['train','development']):
     ids=sets[split];assert all(e in rows for e in ids);acts=collections.Counter();combos=collections.Counter()
     for eid in ids:
      r=rows[eid]
      for a in r['actions']:acts[str(a['action'])]+=1
      for s in r['states']:combos[str((s['objects']['large']['region'],s['objects']['small']['held']))]+=1
      if arm=='holdout'and mode=='fit':assert not r['forbidden_state_seen']if exp=='state'else not r['loaded_reverse_action_present']
     exposure.append({'experiment':exp,'arm':arm,'split':split,'episodes':len(ids),'transitions':sum(len(rows[e]['actions'])for e in ids),'actions':dict(acts),'large_region_small_held':dict(combos)})
 result={'mode':mode,'partial':partial,'episodes_checked':len(rows),'bad_initializations':bad,'all_initial_strata_correct':not bad,'actual_large_place_outcomes':outcomes,'arm_exposure':exposure,'support_only_private_sampling_audit_not_model_supervision':True};path=root/(mode+'_quality_partial.json'if partial else mode+'_quality.json');path.write_text(json.dumps(result,indent=2)+'\n');print('DATA_QUALITY',mode,len(rows),'bad initialization',len(bad),'full arm exposure audit',not partial,flush=True)
 if mode=='fit':assert not bad,'Invalid intended starting configurations; fitting remains blocked'
if __name__=='__main__':main()
