"""Independent standard-library recount from the original sealed predictions."""
from analysis import *
def main():
 inp=read(E/'configs/input_manifest.json')
 for p,h in inp['sha256'].items():assert sha(p)==h,p
 for p,h in read(D/'configs/delivery_manifest.json')['sha256'].items():assert sha(p)==h,p
 predictions=read(D/'results/predictions.json');labels=read(D/'data/combo/_private/labels.json');eligible={r['root_id']for r in read(D/'results/root_contexts.json')if r['context_matched']};roots={r['root_id']:r for r in read(D/'configs/roots.json')}
 def dec(p):
  out=[bool(v>=.5)for v in p];drawer=max(enumerate(p[6:]),key=lambda x:x[1])[0]
  for i in range(3):out[6+i]=i==drawer
  return out
 pl={(p['seed'],p['world_model'],p['query_arm'],p['root_id'],p['branch'],p['slot']):p for p in predictions}
 boundaries=read(E/'results/boundary_errors.json');bc=0
 for row in boundaries:
  key=(row['seed'],'latent'if row['path']=='actual'else row['world_model'],row['query_arm'],row['root_id'],row['branch'],row['slot']);p=pl[key];truth=labels[str(p['after'])];out=dec(p[row['path']+'_probability'])
  assert row['classification']==out and row['truth']==truth['y'] and row['valid']==truth['v'];assert row['wrong_facts']==[FACTS[j]for j in range(9)if truth['v'][j]and out[j]!=truth['y'][j]];bc+=1
 first=read(E/'results/first_errors.json');old=read(D/'results/chain_root_results.json');oldidx={(r['seed'],r['world_model'],r['query_arm'],r['path'],r['root_id'],r['branch']):r for r in old};fc=0
 for r in first:
  o=oldidx[(r['seed'],'latent'if r['path']=='actual'else r['world_model'],r['query_arm'],r['path'],r['root_id'],r['branch'])]
  assert r['first_error_step']==o['first_wrong_step'] and r['first_error_facts']==o['first_wrong_facts'] and r['all4_correct']==o['all4_correct'];fc+=1
 # Aggregate directly from probabilities, rather than trusting the saved pair rows.
 counts=collections.defaultdict(collections.Counter);rootsets=collections.defaultdict(set);facts=collections.defaultdict(collections.Counter)
 for p in predictions:
  truth=labels[str(p['after'])];a=dec(p['actual_probability']);group=roots[p['root_id']]['pair_group']
  for path in ['one_step','recursive']:
   b=dec(p[path+'_probability']);meta=(p['seed'],p['world_model'],p['query_arm'],path)
   for g,idx in GROUPS.items():
    if all(truth['v'][j]for j in idx):
     ae=any(a[j]!=truth['y'][j]for j in idx);be=any(b[j]!=truth['y'][j]for j in idx);cat={ (False,False):'both_correct',(True,False):'actual_only_wrong',(False,True):'predicted_only_wrong',(True,True):'both_wrong'}[(ae,be)]
    else:cat='unknown'
    for scope,cohort,branch,step in itertools.product(['all',group],['all_observed','eligible_roots']if p['root_id']in eligible else['all_observed'],['all',p['branch']],[0,p['slot']+1]):
     key=(*meta,g,scope,cohort,branch,step);counts[key][cat]+=1;rootsets[key].add(p['root_id'])
   for j,fact in enumerate(FACTS):
    ae=a[j]!=truth['y'][j];be=b[j]!=truth['y'][j];cat={(False,False):'both_correct',(True,False):'actual_only_wrong',(False,True):'predicted_only_wrong',(True,True):'both_wrong'}[(ae,be)]if truth['v'][j]else'unknown'
    for cohort in ['all_observed','eligible_roots']if p['root_id']in eligible else['all_observed']:facts[(*meta,cohort,fact)][cat]+=1
 pc=0
 for r in read(E/'results/actual_prediction_summary.json'):
  key=tuple(r[k]for k in ['seed','world_model','query_arm','path','group','scope','cohort','branch','step']);ct=counts[key]
  assert sum(ct.values())==r['N'] and len(rootsets[key])==r['root_N']
  for category in ['both_wrong','actual_only_wrong','predicted_only_wrong','both_correct','unknown']:assert ct[category]==r[category]
  pc+=1
 factrows=[]
 for key,c in sorted(facts.items()):factrows.append(dict(zip(['seed','world_model','query_arm','path','cohort','fact'],key),N=sum(c.values()),**{k:c[k]for k in ['both_wrong','actual_only_wrong','predicted_only_wrong','both_correct','unknown']}))
 emit('fact_level_actual_prediction_pairs',factrows)
 # Joint all-eight errors are a union of the two branch chains, not every boundary wrong.
 jc=0
 for r in read(E/'results/joint_root_errors.json'):
  errors={g:False for g in GROUPS};wrong_boundaries=0
  for branch,slot in itertools.product(BRANCHES[1:],range(4)):
   p=pl[(r['seed'],'latent'if r['path']=='actual'else r['world_model'],r['query_arm'],r['root_id'],branch,slot)];truth=labels[str(p['after'])]['y'];out=dec(p[r['path']+'_probability']);wrong_boundaries+=out!=truth
   for g,idx in GROUPS.items():errors[g]|=any(out[j]!=truth[j]for j in idx)
  assert wrong_boundaries==r['wrong_boundaries']
  for g in GROUPS:assert errors[g]==r[g+'_any_error']
  jc+=1
 shutil.copy2(__file__,E/'source/verify.py')
 result={'utc':now(),'passed':True,'independent_boundary_checks':bc,'first_error_and_chain_comparisons_to_original':fc,'independent_contingency_summaries':pc,'independent_joint_root_checks':jc,'original_sealed_files_unchanged':len(read(D/'configs/delivery_manifest.json')['sha256']),'input_files_unchanged':len(inp['sha256']),'model_forward_calls':0,'training_updates':0,'collection_calls':0}
 save(E/'results/verification.json',result);print('LOCALIZATION_VERIFIED',result)
if __name__=='__main__':main()
