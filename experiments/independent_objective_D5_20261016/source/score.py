from config import *
import collections,csv,math
def csvout(p,rows):
 with p.open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def wilson(n,N):
 if N==0:return [None,None]
 z=1.959963984540054;p=n/N;den=1+z*z/N;c=(p+z*z/(2*N))/den;h=z*math.sqrt(p*(1-p)/N+z*z/(4*N*N))/den;return[max(0,c-h),min(1,c+h)]
def main():
 assert not(E/'results/primary.json').exists()
 roots=read(E/'configs/roots.json');episodes=read(E/'results/collection_D5.json');pairs=read(E/'data/D5/public/pairs.json');labs=read(E/'data/D5/_private/labels.json');pred=read(E/'results/predictions.json');names=['mixed_seed0','latent_seed0','mixed_seed1','latent_seed1'];methods=['actual','one_step','recursive'];lookup={(p['condition'],p['root_id'],p['slot']):p for p in pred};eplook={ep['root_id']:ep for ep in episodes};pairlook={(p['root_id'],p['slot']):p for p in pairs};rootinfo=[];rootresults=[]
 for root in roots:
  rid=root['root_id'];rr=[pairlook[(rid,j)]for j in range(4)if(rid,j)in pairlook];complete=len(rr)==4;valid=[all(labs[str(p['after'])]['v'])for p in rr];ep=eplook.get(rid);row={'root_id':rid,'color':root['target'],'attempted':ep is not None,'observed_boundaries':len(rr),'full9_valid_boundaries':sum(valid),'complete':complete,'full9_eligible':complete and all(valid),'any_evaluable':any(valid),'native_all4_success':bool(ep and len(ep['slots'])==4 and all(s['status']=='success'for s in ep['slots'])),'slot_statuses':[next((s['status']for s in ep['slots']if s['slot']==j),'unreached')if ep else'root_not_collected'for j in range(4)]};rootinfo.append(row)
  for name in names:
   for method in methods:
    ok=[]
    for j in range(4):
     p=lookup.get((name,rid,j))
     if p is None:ok.append(None);continue
     lab=labs[str(p['after'])];ok.append(bool(np.array_equal(discrete(p[method+'_probability']),lab['y']))if all(lab['v'])else None)
    prefix=[all(x is True for x in ok[:j])if all(x is not None for x in ok[:j])else None for j in range(1,5)];rootresults.append({'condition':name,'method':method,'root_id':rid,'color':root['target'],'eligible':row['full9_eligible'],'step_correct':ok,'prefix_correct':prefix,'all4_correct':all(x is True for x in ok)if row['full9_eligible']else None})
 save(E/'results/root_coverage.json',rootinfo);save(E/'results/root_results.json',rootresults)
 neligible=sum(r['full9_eligible']for r in rootinfo);primary=[]
 for name in names:
  for method in methods:
   rr=[r for r in rootresults if r['condition']==name and r['method']==method];n=sum(r['all4_correct']is True for r in rr);primary.append({'condition':name,'method':method,'correct_roots':n,'evaluable_complete_roots':neligible,'planned_roots':30,'complete_observed_roots':sum(r['complete']for r in rootinfo),'any_evaluable_roots':sum(r['any_evaluable']for r in rootinfo),'proportion':n/neligible if neligible else None,'wilson95':wilson(n,neligible),'confirmed_lower_bound_all30':n/30,'missingness_upper_bound_all30':(n+30-neligible)/30})
 save(E/'results/primary.json',primary)
 # One root resample matrix shared across models/seeds; strata preserve observed target-color counts.
 eligible=[r for r in rootinfo if r['full9_eligible']];rng=np.random.default_rng(16972000);blocks=[]
 for color in COLORS:
  ii=np.array([i for i,r in enumerate(eligible)if r['color']==color],int)
  if len(ii):blocks.append(rng.choice(ii,size=(10000,len(ii)),replace=True))
 boot=np.concatenate(blocks,1)if blocks else np.empty((10000,0),int);np.savez_compressed(E/'results/root_bootstrap_indices.npz',indices=boot,root_ids=np.array([r['root_id']for r in eligible]));paired=[];pairedroots=[]
 rlookup={(r['condition'],r['method'],r['root_id']):r for r in rootresults}
 for seed in [0,1]:
  for method in methods:
   diffs=[]
   for r in eligible:
    a=rlookup[(f'mixed_seed{seed}',method,r['root_id'])]['all4_correct'];b=rlookup[(f'latent_seed{seed}',method,r['root_id'])]['all4_correct'];diffs.append(int(b)-int(a));pairedroots.append({'seed':seed,'method':method,'root_id':r['root_id'],'color':r['color'],'mixed_correct':a,'latent_correct':b,'difference':int(b)-int(a)})
   dd=np.asarray(diffs);draw=dd[boot].mean(1)if neligible else np.full(10000,np.nan);ci=np.quantile(draw,[.025,.975]).tolist()if neligible else[None,None];paired.append({'seed':seed,'method':method,'N_roots':neligible,'gains':int((dd==1).sum()),'losses':int((dd==-1).sum()),'ties':int((dd==0).sum()),'difference':float(dd.mean())if neligible else None,'paired_root_bootstrap95':ci,'bootstrap_seed':16972000,'resamples':10000,'stratification':'targetcolor;observedeligiblecounts','same_draws_both_seeds':True})
 save(E/'results/paired_effects.json',paired);save(E/'results/paired_root_outcomes.json',pairedroots);csvout(R/'paired_root_outcomes.csv',pairedroots)
 # Per-state and per-change confusion tables, masks kept explicit.
 stage=[];facts=[]
 for name in names:
  for method in methods:
   for color in ['all',*COLORS]:
    for step in range(1,5):
     rr=[p for p in pred if p['condition']==name and p['slot']==step-1 and(color=='all'or p['target']==color)]
     if not rr:continue
     y=np.array([labs[str(p['after'])]['y']for p in rr],bool);v=np.array([labs[str(p['after'])]['v']for p in rr],bool);yb=np.array([labs[str(p['before'])]['y']for p in rr],bool);vb=np.array([labs[str(p['before'])]['v']for p in rr],bool);pr=discrete([p[method+'_probability']for p in rr]);prev=discrete([p[method+'_before_probability']for p in rr]);full=v.all(1)
     row={'condition':name,'method':method,'color':color,'global_step':step,'observed_roots':len(rr),'full9_evaluable_roots':int(full.sum()),'full9_correct':int((full&(pr==y).all(1)).sum()),'raw_latent_MSE':float(np.mean([p[method+'_raw_MSE']for p in rr])),'raw_feature_error':float(np.mean([p[method+'_feature_groups']for p in rr]))}
     for g,ss in [('Holding',slice(0,3)),('In',slice(3,6)),('drawer',slice(6,9))]:gv=v[:,ss].all(1);row[g+'_N']=int(gv.sum());row[g+'_correct']=int((gv&(pr[:,ss]==y[:,ss]).all(1)).sum())
     stage.append(row)
     for j,fact in enumerate(FACTS):
      vv=v[:,j];on=~yb[:,j]&y[:,j];po=~prev[:,j]&pr[:,j];vc=vv&vb[:,j];pos=vv&y[:,j];neg=vv&~y[:,j];pon=vc&on;non=vc&~on;TP=int((pos&pr[:,j]).sum());FP=int((neg&pr[:,j]).sum());FN=int((pos&~pr[:,j]).sum());TN=int((neg&~pr[:,j]).sum());ct=int((pon&po).sum());cf=int((non&po).sum())
      facts.append({'condition':name,'method':method,'color':color,'global_step':step,'fact':fact,'valid_after_N':int(vv.sum()),'state_TP':TP,'state_FN':FN,'state_FP':FP,'state_TN':TN,'state_positive_N':int(pos.sum()),'state_negative_N':int(neg.sum()),'state_recall':TP/int(pos.sum())if pos.any()else None,'state_FPR':FP/int(neg.sum())if neg.any()else None,'change_positive_N':int(pon.sum()),'predicted_change_TP':ct,'change_recall':ct/int(pon.sum())if pon.any()else None,'after_state_hit_on_true_change':int((pon&pr[:,j]).sum()),'no_positive_change_N':int(non.sum()),'change_FP':cf,'change_FPR':cf/int(non.sum())if non.any()else None})
 save(E/'results/stage_metrics.json',stage);save(E/'results/fact_metrics.json',facts);csvout(R/'stage_metrics.csv',stage);csvout(R/'fact_state_and_change_metrics.csv',facts)
 # Direct per-fact arm pairs retain root identity and signed margin, not calibrated confidence.
 pairedfacts=[]
 for p in pred:
  if not p['condition'].startswith('mixed'):continue
  q=lookup[(p['condition'].replace('mixed','latent'),p['root_id'],p['slot'])];lab=labs[str(p['after'])]
  assert p['actual_probability']==q['actual_probability']and p['actual_logits']==q['actual_logits']
  for j,fact in enumerate(FACTS):
   row={'seed':p['seed'],'root_id':p['root_id'],'target':p['target'],'global_step':p['global_step'],'fact':fact,'truth':lab['y'][j],'valid':lab['v'][j]}
   for n,rr,method in [('actual',p,'actual'),('mixed_one',p,'one_step'),('latent_one',q,'one_step'),('mixed_recursive',p,'recursive'),('latent_recursive',q,'recursive')]:
    ll=rr[method+'_logits'];signed=ll[j]if j<6 else ll[j]-max(ll[k]for k in range(6,9)if k!=j);signed*=1 if lab['y'][j]else-1;row[n+'_prediction']=bool(discrete(rr[method+'_probability'])[j]);row[n+'_signed_logit_margin']=float(signed)
   pairedfacts.append(row)
 csvout(R/'paired_propositions.csv',pairedfacts)
 statuses=collections.Counter(s['status']for ep in episodes for s in ep['slots']);coverage={'planned_roots':30,'attempted_roots':len(episodes),'any_evaluable_roots':sum(r['any_evaluable']for r in rootinfo),'complete_observed_roots':sum(r['complete']for r in rootinfo),'full9_evaluable_complete_roots':neligible,'native_all4_success_roots':sum(r['native_all4_success']for r in rootinfo),'observed_boundaries':len(pairs),'started_calls':sum(ep['started_calls']for ep in episodes),'slot_status_counts':dict(statuses),'not_collected_roots':[r['root_id']for r in rootinfo if not r['attempted']],'incomplete_roots':[r['root_id']for r in rootinfo if not r['complete']],'unknown_label_roots':[r['root_id']for r in rootinfo if r['complete']and not r['full9_eligible']],'no_replacement_or_external_retry':True}
 save(E/'results/coverage_and_physical_failures.json',coverage);save(E/'results/scoring_audit.json',{'finished_utc':now(),'old_gate_unchanged':sha(ABL/'results/gate.json')==read(E/'configs/before_collection_frozen.json')['sha256'][str(ABL/'results/gate.json')],'main_unit':'root','query_seeds_are_not_independent_roots':True,'new_gate_created':False,'models_or_thresholds_selected_on_D':False,'network_updates':0,'labels_only_after_predictions':True,'bootstrap_shared_across_seeds':True,'perfact_pairs':len(pairedfacts)});print('SCORED_D5',json.dumps(coverage),json.dumps([r for r in paired if r['method']=='recursive']),flush=True)
if __name__=='__main__':main()
