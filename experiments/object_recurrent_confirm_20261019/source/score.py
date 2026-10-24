from common import *
from metric_functions import analyze
def interval(values):
 fids=sorted(values);a=np.array([values[f]for f in fids]);rng=np.random.default_rng(2610209901);groups=[np.array([i for i,f in enumerate(fids)if f.startswith(e+'_')])for e in 'AB'];assert all(len(g)==4 for g in groups);draw=np.concatenate([rng.choice(g,size=(2000,4),replace=True)for g in groups],1);bs=a[draw].mean(1);return {'difference':float(a.mean()),'lo':float(np.quantile(bs,.025)),'hi':float(np.quantile(bs,.975)),'families':8,'resampling':'paired, stratified4A+4B families,2000 draws'}
def main():
 guard();verify();budget();roles=[];ret=[];events=[];first=[]
 for arm in ARMS:
  for seed in [0,1]:
   r,f,e,x=analyze(S/f'results/predictions_{arm}_seed{seed}_confirm.npz');roles+=r;ret+=f;events+=e;first+=x
 csvwrite(S/'results/root_metrics.csv',roles);csvwrite(S/'results/retention_opportunities.csv',ret);csvwrite(S/'results/outcome_metrics.csv',events);save(S/'results/first_errors.json',first);families=[];summaries=[]
 keys=['h0_relation','h0_held','h0_full','h2_pair','normal_whole','normal_whole_full','disturbed_whole','joint3_actual','primary_joint3','primary_joint4','terminal','H_In_h2_prior_correct_both','H_In_h2_post_correct_both']
 for arm in ARMS:
  for seed in [0,1]:
   rr=[r for r in roles if r['arm']==arm and r['seed']==seed];assert len(rr)==48;fs=[]
   for fid in sorted({r['family_id']for r in rr}):
    xs=[r for r in rr if r['family_id']==fid];assert len(xs)==6;ff=[r for r in ret if r['arm']==arm and r['seed']==seed and r['family_id']==fid];stable=[r for r in ff if r['stable_trueIn']];pc=[r for r in stable if r['prior_correct']];f={'arm':arm,'seed':seed,'family_id':fid,'valid_pair_rate':np.mean([r['valid_pair']for r in xs]),**{k:np.mean([r[k]for r in xs])for k in keys},'stable_In_opportunities':len(stable),'prior_correct_stable_In':len(pc),'forget_count':sum(r['forgot']for r in stable),'forget_rate':sum(r['forgot']for r in stable)/max(len(stable),1)};families.append(f);fs.append(f)
   ff=[r for r in ret if r['arm']==arm and r['seed']==seed];stable=[r for r in ff if r['stable_trueIn']];pc=[r for r in stable if r['prior_correct']];positive=[r for r in ff if r['event_valid']and r['nontarget_moved']];negative=[r for r in ff if r['event_valid']and not r['nontarget_moved']];summaries.append({'arm':arm,'seed':seed,'roles':48,'families':8,'valid_pair_rate':np.mean([r['valid_pair']for r in rr]),**{k:np.mean([r[k]for r in fs])for k in keys},'stable_In_opportunities':len(stable),'prior_correct_stable_In':len(pc),'forget_count':sum(r['forgot']for r in stable),'forget_rate':sum(r['forgot']for r in stable)/max(len(stable),1),'forget_conditional_rate':sum(r['forgot']for r in stable)/max(len(pc),1),'new_In_opportunities':sum(r['new_trueIn']for r in ff),'moved_positives':len(positive),'moved_recall':np.mean([r['predicted_moved']for r in positive])if positive else None,'moved_false_positive_rate':np.mean([r['predicted_moved']for r in negative])if negative else None})
 csvwrite(S/'results/family_metrics.csv',families);csvwrite(S/'results/summary.csv',summaries);contrasts=[]
 for k in keys+['forget_rate']:
  ds=[]
  for seed in [0,1]:
   xx={arm:{r['family_id']:r[k]for r in families if r['arm']==arm and r['seed']==seed}for arm in ARMS};d={fid:xx['object'][fid]-xx['global'][fid]for fid in xx['global']};ds.append(d);contrasts.append({'metric':k,'seed':str(seed),**interval(d)})
  contrasts.append({'metric':k,'seed':'mean',**interval({fid:(ds[0][fid]+ds[1][fid])/2 for fid in ds[0]})})
 checks=[]
 for seed in [0,1]:
  pair={r['arm']:r for r in summaries if r['seed']==seed};obj=pair['object'];glob=pair['global']
  for k,cut in [('valid_pair_rate',.90),('h0_held',.95),('h0_relation',.95),('h2_pair',.90),('H_In_h2_post_correct_both',.90)]:checks.append({'seed':seed,'criterion':k,'value':obj[k],'rule':'>='+str(cut),'passed':obj[k]>=cut})
  for k in ['primary_joint3','normal_whole']:checks.append({'seed':seed,'criterion':k+'_improvement','value':obj[k]-glob[k],'rule':'>0','passed':obj[k]>glob[k]})
  checks.append({'seed':seed,'criterion':'forget_absolute','value':obj['forget_rate'],'rule':'<=0.02','passed':obj['forget_rate']<=.02});checks.append({'seed':seed,'criterion':'forget_not_worse','value':obj['forget_rate']-glob['forget_rate'],'rule':'<=0','passed':obj['forget_rate']<=glob['forget_rate']})
 for k in ['primary_joint3','normal_whole']:
  ci=next(r for r in contrasts if r['metric']==k and r['seed']=='mean');checks.append({'seed':'mean','criterion':k+'_paired_CI_lower','value':ci['lo'],'rule':'>0','passed':ci['lo']>0})
 csvwrite(S/'results/contrasts.csv',contrasts);save(S/'results/analysis.json',{'utc':now(),'relation_confirmation_gate':all(x['passed']for x in checks),'checks':checks,'summaries':summaries,'contrasts':contrasts,'old_full_state_training_fit_gate':False,'old_gate_unchanged':True,'new_families':True,'scope':'Known objects and skills, AB layout transfer and observation recovery; not complete planning safety acceptance','training_updates':0});print('SCORING_COMPLETE',all(x['passed']for x in checks),flush=True)
if __name__=='__main__':main()
