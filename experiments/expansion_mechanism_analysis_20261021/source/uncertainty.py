from pathlib import Path
import json,collections,csv,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'experiments/expansion_mechanism_analysis_20261021';S=P/'experiments/data_expansion_training_20261021'
def rd(p):return json.loads(p.read_text())
def out(n,r):
 (B/f'results/{n}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 if r:
  with(B/f'results/{n}.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bootstrap_replicates':10000,'primary_cluster':'family_id:20 independent layout families, retains both roots and branches and overlapping windows','sensitivity_cluster':'root_id:40 common physical roots, retains both branches and all windows; correlations across roots of same family remain, so primaryfamily is conservative choice','paired_resampling':'Same selected clusters for observed rate, nonlinear reference, model conditions; no independent-window binomialCI, seeds separately fixed weights not doubleddata','main_difference':'D_equal = observed free three-boundary all-correct fraction minus matched first-step success fraction cubed','secondary_differences':'D_marginal = same joint minus product of actual three marginal success fractions; D_unmatched = joint minus all480one-step fraction cubed, shown to explain original4.4pp only','interval':'Percentile95% group pairedbootstrap, estimates data-layout sampling variability conditional on fitted models; not training-seed variance','acceptance':'No practical tolerance or planning risk threshold specified. CI including0 does not establish equivalence or acceptance. Cube-root per-step equivalent is descriptive only.'};(B/'configs/uncertainty_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
windows=rd(B/'results/matched_windows.json');meta={e['episode_id']:e for e in rd(S/'cache/development_metadata.json')};stats=[];paired=[];cor=[];comparison=[]
def calc(x):
 n=x[:,0];p=x[:,1:4]/n[:,None];whole=x[:,4]/n;glob=x[:,5]/x[:,6];return{'p1':p[:,0],'p2':p[:,1],'p3':p[:,2],'whole':whole,'reference_equal_p1_cubed':p[:,0]**3,'reference_actual_marginal_product':p.prod(1),'reference_unmatched480_cubed':glob**3,'D_equal':whole-p[:,0]**3,'D_marginal':whole-p.prod(1),'D_unmatched480':whole-glob**3,'equivalent_per_step':np.cbrt(whole),'equivalent_minus_actual_p1':np.cbrt(whole)-p[:,0]}
for seed in [0,1]:
 vectors={}
 for arm in ['old228','new868']:
  rr=[r for r in windows if(r['source'],r['seed'],r['arm'])==('development',seed,arm)];z=dict(np.load(S/f'results/predictions_development_{arm}_seed{seed}.npz'));p=z['prob'];y=np.c_[z['cat'][:,:3]==6,z['cat'][:,:3]==3];pr=np.c_[p[:,[6,14,22]]>=.5,p[:,[3,11,19]]>=.5];ok=(pr==y).all(1)&(p[:,24:27].argmax(1)==z['cat'][:,3]);sel=(z['mode']=='free')&(z['relative_step']==1);full={str(e):[0,0]for e in meta}
  for i in np.flatnonzero(sel):full[str(z['episode_id'][i])][0]+=int(ok[i]);full[str(z['episode_id'][i])][1]+=1
  for grouping in ['family_id','root_id']:
   g=collections.defaultdict(lambda:np.zeros(7,float))
   for r in rr:
    group=meta[r['episode_id']][grouping];a=np.array(r['free_correct'],float);g[group][:5]+=np.r_[1,a,int(a.all())]
   for eid,(a,n)in full.items():g[meta[eid][grouping]][5:]+=np.array([a,n])
   keys=sorted(g);v=np.stack([g[k]for k in keys]);vectors[arm,grouping]=v;draw=np.random.default_rng(2610214400+seed+(0 if grouping=='family_id'else 10)).integers(0,len(v),size=(10000,len(v)));boot=calc(v[draw].sum(1));point=calc(v.sum(0,keepdims=True))
   for name,values in boot.items():stats.append({'seed':seed,'arm':arm,'cluster':grouping,'clusters':len(v),'metric':name,'estimate':float(point[name][0]),'CI95_low':float(np.quantile(values,.025)),'CI95_high':float(np.quantile(values,.975))})
  for mode in ['free','observed']:
   arr=np.array([r[mode+'_correct']for r in rr],bool)
   for a,b in [(0,1),(0,2),(1,2)]:
    cor.append({'seed':seed,'arm':arm,'mode':mode,'step_pair':f'{a+1},{b+1}','phi':float(np.corrcoef(arr[:,a],arr[:,b])[0,1]),'both_error':int((~arr[:,a]&~arr[:,b]).sum()),'first_error_second_correct':int((~arr[:,a]&arr[:,b]).sum()),'first_correct_second_error':int((arr[:,a]&~arr[:,b]).sum()),'both_correct':int((arr[:,a]&arr[:,b]).sum())})
  f=np.array([r['free_correct']for r in rr],bool);o=np.array([r['observed_correct']for r in rr],bool)
  for step in range(3):
   for selection in ['all','free_prefix_correct']:
    mask=np.ones(len(f),bool)if selection=='all'or step==0 else f[:,:step].all(1);a=f[mask,step];b=o[mask,step];comparison.append({'seed':seed,'arm':arm,'step':step+1,'selection':selection,'windows':int(mask.sum()),'free_correct':int(a.sum()),'observed_correct':int(b.sum()),'observation_gains':int((~a&b).sum()),'observation_losses':int((a&~b).sum())})
 for grouping in ['family_id','root_id']:
  a=vectors['old228',grouping];b=vectors['new868',grouping];draw=np.random.default_rng(2610214400+seed+(0 if grouping=='family_id'else 10)).integers(0,len(a),size=(10000,len(a)));ba=calc(a[draw].sum(1));bb=calc(b[draw].sum(1));pa=calc(a.sum(0,keepdims=True));pb=calc(b.sum(0,keepdims=True))
  for metric in ['p1','p3','whole','D_equal','D_marginal']:
   d=bb[metric]-ba[metric];paired.append({'seed':seed,'cluster':grouping,'metric_new_minus_old':metric,'estimate':float(pb[metric][0]-pa[metric][0]),'CI95_low':float(np.quantile(d,.025)),'CI95_high':float(np.quantile(d,.975))})
out('grouped_bootstrap',stats);out('paired_model_bootstrap',paired);out('temporal_correlations',cor);out('observation_matched_gains',comparison)
print('NEW_PRIMARY',[r for r in stats if r['arm']=='new868'and r['cluster']=='family_id'and r['metric']in ['D_equal','D_marginal','D_unmatched480','whole','p1','equivalent_minus_actual_p1']])
