from study import *
def main():
 guard('analysis');refs=read(U/'results/prediction_references.json');metrics=[];rootrows=[];updates=[];raw={};forgetting=[]
 for ref in refs:
  assert sha(ref['path'])==ref['sha256']
  with np.load(ref['path'])as z:d={k:z[k]for k in z.files}
  p=rel(d['prob']);y=truth(d['cat']);ok=correct(p,y);valid=d['cat_valid'][:,:4].all(-1);assert valid.all();variant=ref['variant'];seed=ref['seed'];split=ref['split'];raw[variant,seed,split]=d
  for rid in sorted(set(d['root_id'])):
   ix=lambda mode,step,branch:np.where((d['root_id']==rid)&(d['mode']==mode)&(d['step']==step)&(d['branch']==branch))[0][0]
   a,b=ix('current',2,1),ix('current',2,2);pa,pb=ix('one_step',2,1),ix('one_step',2,2);assert np.array_equal(p[pa],p[pb]);held=COLORS.index(str(d['held_color'][a]));target=COLORS.index(str(d['target_color'][a]));h0=ix('current',0,0);suffix=[];nexts=[]
   for branch in [1,2]:
    ids=[ix('suffix',t,branch)for t in [3,4,5]];suffix.append(bool(ok[ids,3].all()));nexts.append(bool(ok[ix('one_step',3,branch),3]));pre=ix('one_step',2,branch);post=ix('current',2,branch);updates.append({'variant':variant,'seed':seed,'split':split,'root_id':str(rid),'branch':branch,'prior_held_correct':bool(ok[pre,0]),'posterior_held_correct':bool(ok[post,0]),'corrected_held':bool(not ok[pre,0]and ok[post,0]),'regressed_held':bool(ok[pre,0]and not ok[post,0]),'prior_target_Held':float(p[pre,target]),'posterior_target_Held':float(p[post,target]),'A_InDrawer':float(p[post,3+held])})
   ids=np.where((d['root_id']==rid)&(d['mode']=='current')&(d['branch']<2))[0];future=lambda step,branch:ok[ix('suffix',step,branch),3]
   vals={'h0_held':bool(ok[h0,0]),'h0_joint':bool(ok[h0,3]),'h2_b1_held':bool(ok[a,0]),'h2_b2_held':bool(ok[b,0]),'h2_pair_held':bool(ok[[a,b],0].all()),'h2_pair_joint':bool(ok[[a,b],3].all()),'h2_pair_A_retained':bool((p[[a,b],3+held]>=.5).all()),'h2_b2_A_retained':bool(p[b,3+held]>=.5),'h2_pair_target_probability_gap':float(p[a,target]-p[b,target]),'next_b1_joint':nexts[0],'next_b2_joint':nexts[1],'next_pair_joint':all(nexts),'suffix3_b1_full':suffix[0],'suffix3_b2_full':suffix[1],'h2_and_suffix3_pair_joint':bool(ok[[a,b],3].all() and all(suffix)),'suffix3_pair_full':all(suffix),'normal_current_joint':float(ok[ids,3].mean()),'b0_current_joint':float(ok[ids[d['branch'][ids]==0],3].mean()),'b0_full_trajectory':bool(ok[ids[d['branch'][ids]==0],3].all()),'b0_h2_future3_joint':bool(ok[ix('current',2,0),3] and all(ok[ix('suffix',t,0),3]for t in [3,4,5])),'b1_current_joint':float(ok[ids[d['branch'][ids]==1],3].mean()),'b1_observation_correction':bool(not ok[pa,0]and ok[a,0]),'b2_observation_correction':bool(not ok[pb,0]and ok[b,0]),'observation_any_correction':bool((not ok[pa,0]and ok[a,0])or(not ok[pb,0]and ok[b,0])),'observation_any_regression':bool((ok[pa,0]and not ok[a,0])or(ok[pb,0]and not ok[b,0]))}
   for k in [1,2,3]:vals[f'suffix{k}_b0_joint']=bool(future(k+2,0));vals[f'suffix{k}_b1_joint']=bool(future(k+2,1));vals[f'suffix{k}_b2_joint']=bool(future(k+2,2))
   for branch in [0,1,2]:
    h0i=ix('current',0,branch);acolor=COLORS.index(str(d['held_color'][h0i]));bcolor=COLORS.index(str(d['target_color'][h0i]))
    for t in range(1,7):
     before=ix('current',t-1,branch);pr=ix('one_step',t,branch);post=ix('current',t,branch);target=acolor if t==1 else bcolor if t in [2,3,4,5]else None
     for j,color in enumerate(COLORS):
      if j!=target and y[before,3+j] and y[post,3+j] and p[pr,3+j]>=.5 and p[post,3+j]<.5:forgetting.append({'variant':variant,'seed':seed,'split':split,'root_id':str(rid),'branch':branch,'step':t,'color':color,'prior':float(p[pr,3+j]),'posterior':float(p[post,3+j])})
   for name,value in vals.items():rootrows.append({'variant':variant,'seed':seed,'split':split,'root_id':str(rid),'metric':name,'value':float(value)})
 groups=collections.defaultdict(list)
 for r in rootrows:groups[r['variant'],r['seed'],r['split'],r['metric']].append(r['value'])
 for(v,s,sp,m),vv in groups.items():
  a=np.array(vv);rng=np.random.default_rng(281019);boot=np.mean(a[rng.integers(len(a),size=(10000,len(a)))],1);ci=np.quantile(boot,[.025,.975]);metrics.append({'variant':v,'seed':s,'split':sp,'metric':m,'sum':float(a.sum()),'roots':len(a),'mean':float(a.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1])})
 def get(v,s,sp,m):return next(r for r in metrics if(r['variant'],r['seed'],r['split'],r['metric'])==(v,s,sp,m))
 gates=[]
 def lost(v,s,sp,b):return set(r['root_id']for r in forgetting if(r['variant'],r['seed'],r['split'],r['branch'])==(v,s,sp,b))
 for seed in [0,1]:
  retention={sp:{str(b):{'control':sorted(lost('control',seed,sp,b)),'updated':sorted(lost('updated',seed,sp,b)),'new_forgetting_roots':sorted(lost('updated',seed,sp,b)-lost('control',seed,sp,b))}for b in [0,1,2]}for sp in ['train','development']}
  tc={'paired_current_future12':get('updated',seed,'train','h2_and_suffix3_pair_joint')['sum']==12,'h0_held12':get('updated',seed,'train','h0_held')['sum']==12,'b0_whole_trajectory_improves':get('updated',seed,'train','b0_full_trajectory')['sum']>get('control',seed,'train','b0_full_trajectory')['sum'],'b0_current_future_improves':get('updated',seed,'train','b0_h2_future3_joint')['sum']>get('control',seed,'train','b0_h2_future3_joint')['sum'],'no_new_forgetting_roots':all(not r['new_forgetting_roots']for r in retention['train'].values())}
  dc={m+'_no_regression':get('updated',seed,'development',m)['sum']>=get('control',seed,'development',m)['sum']for m in ['h0_held','h2_and_suffix3_pair_joint','b0_full_trajectory']};dc['no_new_forgetting_roots']=all(not r['new_forgetting_roots']for r in retention['development'].values())
  gates.append({'seed':seed,'training_components':tc,'training_joint_gate':all(tc.values()),'development_components':dc,'development_gate':all(dc.values()),'retention_root_sets':retention,'historical_evaluation_not_used_for_gate':True})
 contrasts=[]
 for split in ['train','development','evaluation']:
  for metric in sorted(set(r['metric']for r in rootrows)):
   for seeds in [[0],[1],[0,1]]:
    rr={};
    for r in rootrows:
     if r['split']==split and r['metric']==metric and r['variant']in ['control','updated']and r['seed']in seeds:rr[r['root_id'],r['seed'],r['variant']]=r['value']
    values=[np.mean([rr[r,s,'updated']-rr[r,s,'control']for s in seeds])for r in sorted(set(k[0]for k in rr))];rng=np.random.default_rng(281019);boot=np.mean(np.array(values)[rng.integers(len(values),size=(10000,len(values)))],1);ci=np.quantile(boot,[.025,.975]);contrasts.append({'split':split,'seeds':','.join(map(str,seeds)),'metric':metric,'paired_minus_control':float(np.mean(values)),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'independent_roots':len(values)})
 csvwrite(U/'results/forgetting_events.csv',forgetting);csvwrite(U/'results/metrics.csv',metrics);csvwrite(U/'results/root_metrics.csv',rootrows);csvwrite(U/'results/posterior_updates.csv',updates);csvwrite(U/'results/contrasts.csv',contrasts);save(U/'results/gates.json',{'utc':now(),'gates':gates,'training_all_seeds':all(r['training_joint_gate']for r in gates),'development_all_seeds':all(r['development_gate']for r in gates),'full_guide_acceptance':False});save(U/'results/analysis_complete.json',{'utc':now(),'metrics':metrics,'contrasts':contrasts,'root_bootstrap_replicates':10000,'bootstrap_unit':'whole root; seeds averaged inside root for combined contrasts','historical_eval_exploratory_only':True,'threshold':.5});print('GATES',json.dumps(gates),flush=True)
if __name__=='__main__':main()
