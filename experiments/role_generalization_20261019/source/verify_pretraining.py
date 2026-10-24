from common import *
def main():
 guard('verify');verify_freeze();roots=read(S/'data/roots_manifest.json');eps={}
 for arm in ARMS:
  rr=sorted([r for r in roots if arm in r['arms']],key=lambda r:(r['family_id'],r['held_color'],r['matched_color']));eps[arm]=[(r,b)for r in rr for b in[1,2]];assert len(eps[arm])==72
 receipts=[]
 for seed in[0,1]:
  rng=np.random.default_rng(261020000+seed);old=rng.integers(120,size=(1000,2,4));cycles=[]
  while len(cycles)<7992:cycles.extend(rng.permutation(72).tolist())
  shared={}
  for ix,((a,ab),(b,bb))in enumerate(zip(eps['random'],eps['balanced'])):
   assert a['family_id']==b['family_id']and ab==bb
   if a['root_id']==b['root_id']:shared.setdefault(a['family_id'],[]).append(ix)
  chosen=rng.choice(sorted(shared),size=8,replace=False);branches=rng.permutation([1]*4+[2]*4)
  for fid,branch in zip(chosen,branches):cycles.append(int(rng.choice([ix for ix in shared[fid]if eps['random'][ix][1]==branch])))
  counts={}
  for arm in ARMS:
   counts[arm]=collections.Counter((eps[arm][i][0]['environment'],eps[arm][i][0]['held_color'],eps[arm][i][0]['matched_color'],eps[arm][i][1])for i in cycles)
  assert counts['random']==counts['balanced'];assert np.ptp(np.bincount(cycles,minlength=72))<=1
  fam=collections.Counter(eps['random'][i][0]['family_id']for i in cycles);assert max(fam.values())-min(fam.values())<=1
  receipts.append({'seed':seed,'draws':len(cycles),'old_draws':int(old.size),'global_environment_role_branch_exposure_exact':True,'unique_episode_exposure_range':[int(np.bincount(cycles).min()),int(np.bincount(cycles).max())],'family_exposure_range':[min(fam.values()),max(fam.values())],'counts':[{'environment':e,'held':h,'target':t,'branch':b,'draws':n}for(e,h,t,b),n in sorted(counts['random'].items())]})
 # Verify data-policy roots are held away from development/test even though role IDs differ.
 split=collections.defaultdict(set)
 for r in roots:split[r['family_id']].add(r['split'])
 assert all(len(x)==1 for x in split.values());assert len(split)==24
 save(S/'results/schedule_dry_run.json',{'utc':now(),'passed':True,'model_or_simulator_calls':0,'schedules':receipts,'family_split_leakage':False})
 save(S/'configs/scoring_code_frozen.json',{'utc':now(),'before_model_fit_and_any_test_rollout':True,'sha256':{str(S/'source'/n):sha(S/'source'/n)for n in['metrics.py','evaluate.py','audit_collection.py','train_exact_exposure.py','verify_pretraining.py']}})
 print('DRY_RUN_PASS',flush=True)
if __name__=='__main__':main()
