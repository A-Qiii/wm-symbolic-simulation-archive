from config import *
import ast,csv
def main():
 hashes=0
 for fn in ['before_collection_frozen.json','encoding_and_scoring_frozen.json','raw_D5_hashes.json']:
  for p,h in read(E/'configs'/fn)['sha256'].items():assert sha(p)==h,p;hashes+=1
 previous=read(ABL/'configs/delivery_manifest.json')['sha256']
 for p,h in previous.items():assert sha(p)==h,p
 assert not read(ABL/'results/gate.json')['passed'];assert read(E/'results/collection_verification.json')['passed']
 protocol=read(E/'configs/protocol.json');roots=read(E/'configs/roots.json');pairs=read(E/'data/D5/public/pairs.json');labs=read(E/'data/D5/_private/labels.json');pred=read(E/'results/predictions.json');assert len(roots)==30 and len(pred)==len(pairs)*4;lookup={(p['condition'],p['root_id'],p['slot']):p for p in pred};rinfo=read(E/'results/root_coverage.json');rootresults=read(E/'results/root_results.json');prim=read(E/'results/primary.json');coverage=read(E/'results/coverage_and_physical_failures.json');N=sum(r['full9_eligible']for r in rinfo)
 for r in rootresults:
  ok=[]
  for j in range(4):
   p=lookup.get((r['condition'],r['root_id'],j))
   if p is None:ok.append(None)
   else:
    lab=labs[str(p['after'])];ok.append(bool(np.array_equal(discrete(p[r['method']+'_probability']),lab['y']))if all(lab['v'])else None)
  assert ok==r['step_correct'];assert r['all4_correct']==(all(x is True for x in ok)if all(x is not None for x in ok)else None)
 for r in prim:
  rr=[x for x in rootresults if x['condition']==r['condition']and x['method']==r['method']];n=sum(x['all4_correct']is True for x in rr);assert n==r['correct_roots']and r['evaluable_complete_roots']==N and r['planned_roots']==30;assert r['confirmed_lower_bound_all30']==n/30 and r['missingness_upper_bound_all30']==(n+30-N)/30
 stages=read(E/'results/stage_metrics.json');facts=read(E/'results/fact_metrics.json')
 for r in stages:
  rr=[p for p in pred if p['condition']==r['condition']and p['global_step']==r['global_step']and(r['color']=='all'or p['target']==r['color'])];y=np.array([labs[str(p['after'])]['y']for p in rr]);v=np.array([labs[str(p['after'])]['v']for p in rr]);pr=discrete([p[r['method']+'_probability']for p in rr]);full=v.all(1);assert len(rr)==r['observed_roots']and int(full.sum())==r['full9_evaluable_roots']and int((full&(pr==y).all(1)).sum())==r['full9_correct']
  for g,sl in [('Holding',slice(0,3)),('In',slice(3,6)),('drawer',slice(6,9))]:assert int((v[:,sl].all(1)&(pr[:,sl]==y[:,sl]).all(1)).sum())==r[g+'_correct']
 for r in facts:
  rr=[p for p in pred if p['condition']==r['condition']and p['global_step']==r['global_step']and(r['color']=='all'or p['target']==r['color'])];j=FACTS.index(r['fact']);valid=np.array([labs[str(p['after'])]['v'][j]for p in rr]);vbefore=np.array([labs[str(p['before'])]['v'][j]for p in rr]);y=np.array([labs[str(p['after'])]['y'][j]for p in rr]);yb=np.array([labs[str(p['before'])]['y'][j]for p in rr]);pr=discrete([p[r['method']+'_probability']for p in rr])[:,j];pb=discrete([p[r['method']+'_before_probability']for p in rr])[:,j]
  assert int((valid&y&pr).sum())==r['state_TP']and int((valid&~y&pr).sum())==r['state_FP'];assert int((valid&y&~pr).sum())==r['state_FN']and int((valid&~y&~pr).sum())==r['state_TN'];vc=valid&vbefore;on=~yb&y;po=~pb&pr;assert int((vc&on).sum())==r['change_positive_N'];assert int((vc&on&po).sum())==r['predicted_change_TP']and int((vc&~on&po).sum())==r['change_FP']
 eligible=[r for r in rinfo if r['full9_eligible']];boot=dict(np.load(E/'results/root_bootstrap_indices.npz'));assert boot['root_ids'].tolist()==[r['root_id']for r in eligible];rng=np.random.default_rng(16972000);blocks=[]
 for color in COLORS:
  ii=np.array([i for i,r in enumerate(eligible)if r['color']==color],int)
  if len(ii):blocks.append(rng.choice(ii,size=(10000,len(ii)),replace=True))
 expected=np.concatenate(blocks,1)if blocks else np.empty((10000,0),int);assert np.array_equal(boot['indices'],expected);prr=read(E/'results/paired_root_outcomes.json')
 for r in read(E/'results/paired_effects.json'):
  rows=[x for x in prr if x['seed']==r['seed']and x['method']==r['method']];assert [x['root_id']for x in rows]==boot['root_ids'].tolist();dd=np.array([int(x['latent_correct'])-int(x['mixed_correct'])for x in rows]);assert int((dd==1).sum())==r['gains']and int((dd==-1).sum())==r['losses'];assert np.allclose(np.quantile(dd[expected].mean(1),[.025,.975]),r['paired_root_bootstrap95'])
 for p in pred:
  if p['condition'].startswith('mixed'):
   other=lookup[(p['condition'].replace('mixed','latent'),p['root_id'],p['slot'])];assert p['actual_probability']==other['actual_probability']and p['actual_logits']==other['actual_logits']
 assert len(list(csv.DictReader((R/'paired_propositions.csv').open())))==len(pairs)*2*9
 for p in(E/'source').glob('*.py'):ast.parse(p.read_text())
 audit=read(E/'results/prediction_audit.json');assert audit['model_updates']==audit['physics_calls']==audit['private_truth_reads']==audit['recursive_latent_corrections']==0
 result={'passed':True,'checked_utc':now(),'frozen_hash_entries':hashes,'prior_ablation_files_unchanged':len(previous),'old_gate_still_failed':True,'new_protocol_separate':True,'network_updates':0,'new_root_count':30,'observed_boundaries':len(pairs),'primary_eligible_complete_roots':N,'all30coverage_and_missingness_retained':True,'all_root_correctness_recounted':len(rootresults),'stage_metrics_recounted':len(stages),'fact_and_change_metrics_recounted':len(facts),'paired_root_bootstrap_reproduced':True,'same_bootstrap_root_draws_for_both_seeds':True,'actual_readout_identical_between_objectives':True,'paired_fact_rows':len(pairs)*18,'post_test_parameter_selection':False,'formal_adoption':False}
 if(E/'configs/delivery_manifest.json').exists():
  seal=read(E/'configs/delivery_manifest.json')['sha256']
  for p,h in seal.items():assert sha(p)==h,p
  result['sealed_files_verified']=len(seal)
 print('VERIFICATION_RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
