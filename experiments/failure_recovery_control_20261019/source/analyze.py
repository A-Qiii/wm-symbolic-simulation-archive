from common import *
NAMES=['held','in_drawer','drawer','joint']
def load(variant,seed,split):
 with np.load(S/f'results/predictions_{variant}_seed{seed}_{split}.npz')as z:d={k:z[k]for k in z.files}
 d['rel']=rel(d['prob']);d['truth']=truth(d['cat']);d['ok']=correct(d['rel'],d['truth']);d['valid']=d['cat_valid'][:,:4].all(-1);d['nll']=nll(d['rel'],d['truth']);return d
def main():
 guard('analysis');metrics=[];rootvalues=[];firsterrors=[];posterior=[];allsets={};audits={a['root_id']:a for a in read(S/'results/root_audits.json')}
 for split in ['development','evaluation']:
  for seed in [0,1]:
   variants=['original','nominal','recovery']+[a+'_final1000'for a in ARMS if(S/f'results/predictions_{a}_final1000_seed{seed}_{split}.npz').exists()]
   for variant in variants:
    d=load(variant,seed,split);allsets[variant,seed,split]=d
    def add(name,mask,values,kind='accuracy'):
     ids=np.where(mask & d['valid'])[0];v=np.asarray(values)[ids];roots=d['root_id'][ids];rv=[]
     for rid in sorted(set(roots)):
      vv=v[roots==rid];r={'split':split,'seed':seed,'variant':variant,'metric':name,'root_id':rid,'value':float(vv.mean()),'n':len(vv),'kind':kind};rootvalues.append(r);rv.append(r['value'])
     if not len(ids):return
     rng=np.random.default_rng(271019);boots=np.mean(np.array(rv)[rng.integers(len(rv),size=(10000,len(rv)))],axis=1);ci=np.quantile(boots,[.025,.975]);metrics.append({'split':split,'seed':seed,'variant':variant,'metric':name,'value':float(v.mean()),'numerator':float(v.sum()),'denominator':len(v),'roots':len(rv),'root_macro_value':float(np.mean(rv)),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'kind':kind})
    curr=d['mode']=='current';h2=curr&(d['step']==2);failure=d['branch']==2;nominal=d['branch']<2;validfailure=np.array([audits[str(r)]['branches'][2]['valid_observed_failure']for r in d['root_id']]);suffix=d['mode']=='suffix';one=d['mode']=='one_step';heldidx=np.array([COLORS.index(str(c))for c in d['held_color']]);retained=d['rel'][np.arange(len(heldidx)),3+heldidx]>=.5;true_retained=d['truth'][np.arange(len(heldidx)),3+heldidx]>0;empty=~(d['rel'][:,:3]>=.5).any(-1)
    for j,name in enumerate(NAMES):
     add('failure_h2_'+name,h2&failure&validfailure,d['ok'][:,j]);add('nominal_h2_'+name,h2&nominal,d['ok'][:,j]);add('all_current_'+name,curr,d['ok'][:,j]);add('nominal_current_'+name,curr&nominal,d['ok'][:,j])
     for step in [3,4,5]:add(f'failure_suffix{step-2}_'+name,suffix&failure&(d['step']==step),d['ok'][:,j])
    add('failure_h2_empty_and_retained',h2&failure&validfailure,empty&retained);add('failure_h2_nll',h2&failure&validfailure,d['nll'],'nll');add('failure_current_retained_positive',curr&failure&(d['step']>=2)&true_retained,retained);add('failure_suffix_retained_positive',suffix&failure&true_retained,retained);add('nominal_suffix_joint',suffix&nominal,d['ok'][:,-1]);add('post_h2_one_step_joint',one&(d['base']>=2),d['ok'][:,-1]);add('next_place_success_brier',one&(d['step']==3),(d['success_prob']-d['success_target'])**2,'brier');add('next_place_success_accuracy',one&(d['step']==3),(d['success_prob']>=.5)==d['success_target'])
    # Episode and paired-root summaries are represented at a single existing row per unit.
    pairmask=np.zeros(len(curr),bool);pairvalue=np.zeros(len(curr));sufmask=np.zeros(len(curr),bool);sufvalue=np.zeros(len(curr));nomask=np.zeros(len(curr),bool)
    for rid in sorted(set(d['root_id'])):
     a=np.where(h2&(d['root_id']==rid)&(d['branch']==1))[0][0];b=np.where(h2&(d['root_id']==rid)&failure)[0][0]
     if audits[str(rid)]['valid_matched_root']:pairmask[b]=True;pairvalue[b]=d['ok'][a,0]&d['ok'][b,0]
    for eid in sorted(set(d['episode_id'])):
     ids=np.where(suffix&(d['episode_id']==eid))[0];assert len(ids)==3;i=ids[0];sufvalue[i]=d['ok'][ids,-1].all();sufmask[i]=d['branch'][i]==2;nomask[i]=d['branch'][i]<2
     current=np.where(curr&(d['episode_id']==eid))[0];bad=[int(i)for i in current if d['valid'][i]and not d['ok'][i,-1]];f=bad[0]if bad else None;firsterrors.append({'split':split,'seed':seed,'variant':variant,'root_id':str(d['root_id'][i]),'episode_id':str(eid),'branch':int(d['branch'][i]),'first_wrong_boundary':int(d['step'][f])if f is not None else -1,'wrong_components':','.join(NAMES[j]for j in range(3)if not d['ok'][f,j])if f is not None else'none'})
     prior=np.where(one&(d['step']==2)&(d['episode_id']==eid))[0][0];post=np.where(h2&(d['episode_id']==eid))[0][0]
     posterior.append({'split':split,'seed':seed,'variant':variant,'root_id':str(d['root_id'][post]),'episode_id':str(eid),'branch':int(d['branch'][post]),'prior_held_correct':bool(d['ok'][prior,0]),'posterior_held_correct':bool(d['ok'][post,0]),'corrected_held':bool(not d['ok'][prior,0]and d['ok'][post,0]),'regressed_held':bool(d['ok'][prior,0]and not d['ok'][post,0]),'prior_joint':bool(d['ok'][prior,-1]),'posterior_joint':bool(d['ok'][post,-1]),'held_probability_change_L1':float(np.abs(d['rel'][post,:3]-d['rel'][prior,:3]).sum()),'actual_empty':bool(not d['truth'][post,:3].any()),'retained_A':bool(retained[post]),'h2_nll':float(d['nll'][post])})
    add('paired_h2_held_both',pairmask,pairvalue);add('failure_suffix3_full_prefix',sufmask,sufvalue);add('nominal_suffix3_full_prefix',nomask,sufvalue)
 csvwrite(S/'results/metrics.csv',metrics);csvwrite(S/'results/root_metrics.csv',rootvalues);csvwrite(S/'results/first_errors.csv',firsterrors);csvwrite(S/'results/posterior_updates.csv',posterior)
 differences=[]
 for split in ['development','evaluation']:
  names=sorted(set(r['metric']for r in rootvalues))
  for metric in names:
   for seedset in [[0],[1],[0,1]]:
    pair={}
    for r in rootvalues:
     if r['split']==split and r['metric']==metric and r['variant']in ARMS and r['seed']in seedset:pair[r['root_id'],r['seed'],r['variant']]=r['value']
    roots=sorted(set(k[0]for k in pair));values=[]
    for rid in roots:
     dd=[pair[rid,seed,'recovery']-pair[rid,seed,'nominal']for seed in seedset if(rid,seed,'recovery')in pair and(rid,seed,'nominal')in pair]
     if dd:values.append(float(np.mean(dd)))
    if not values:continue
    rng=np.random.default_rng(271019);boot=np.mean(np.array(values)[rng.integers(len(values),size=(10000,len(values)))],1);ci=np.quantile(boot,[.025,.975]);differences.append({'split':split,'seeds':','.join(map(str,seedset)),'metric':metric,'recovery_minus_nominal':float(np.mean(values)),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'independent_roots':len(values),'positive_roots':sum(x>0 for x in values),'negative_roots':sum(x<0 for x in values),'ties':sum(x==0 for x in values)})
 csvwrite(S/'results/paired_differences.csv',differences)
 # Record direct changes in actual encoder/posterior/deterministic states; this is evidence of change, not causal attribution.
 changes=[]
 for seed in [0,1]:
  with np.load(S/f'results/latent_nominal_seed{seed}_evaluation.npz')as a,np.load(S/f'results/latent_recovery_seed{seed}_evaluation.npz')as b:
   assert np.array_equal(a['episode_id'],b['episode_id'])and np.array_equal(a['step'],b['step']);mask=(a['step']==2)&np.char.endswith(a['episode_id'],'_b2')
   for rid in sorted(set(a['root_id'][mask])):
    ix=mask&(a['root_id']==rid);changes.append({'seed':seed,'root_id':str(rid),'E_RMS':float(np.sqrt(np.mean((a['E'][ix].astype(float)-b['E'][ix].astype(float))**2))),'h_RMS':float(np.sqrt(np.mean((a['h'][ix].astype(float)-b['h'][ix].astype(float))**2))),'q_mean_TV':float(np.abs(a['q'][ix].astype(float)-b['q'][ix].astype(float)).sum(-1).mean()/2)})
 csvwrite(S/'results/latent_changes.csv',changes);assert all(np.isfinite(r['value'])for r in metrics);save(S/'results/analysis_complete.json',{'utc':now(),'metrics':len(metrics),'paired_differences':len(differences),'independent_evaluation_roots':6,'bootstrap_replicates':10000,'bootstrap_unit':'whole root; both seeds averaged within same root, never treated as12 independent roots','all_numeric_values_finite':True,'threshold':.5,'no_threshold_tuning':True,'metrics':metrics,'differences':differences})
 print('ANALYSIS_COMPLETE',len(metrics),flush=True)
if __name__=='__main__':main()
