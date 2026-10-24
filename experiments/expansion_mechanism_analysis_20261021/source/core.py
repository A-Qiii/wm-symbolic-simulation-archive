from pathlib import Path
import json,csv,collections,hashlib,datetime,os,sys
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/data_expansion_training_20261021';A=P/'experiments/prediction_error_analysis_20261021';B=P/'experiments/expansion_mechanism_analysis_20261021';O=P/'experiments/object_recurrent_state_run_20261019'
def guard(ev,args):
 if ev=='socket.connect':raise RuntimeError('Offline analysis')
 if ev!='open'or not isinstance(args[0],(str,bytes)):return
 p=Path(os.fsdecode(args[0])).resolve();m=args[1];f=args[2]if len(args)>2 else 0
 if p.is_relative_to(P)and not p.is_relative_to(B)and((isinstance(m,str)and any(x in m for x in 'wa+'))or(isinstance(f,int)and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))):raise RuntimeError('Protected write '+str(p))
sys.addaudithook(guard)
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return{str(k):plain(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return[plain(v)for v in x]
 return x
def rd(p):return json.loads(p.read_text())
def out(n,rows):
 (B/f'results/{n}.json').write_text(json.dumps(plain(rows),ensure_ascii=False,indent=2)+'\n')
 if isinstance(rows,list)and rows:
  with(B/f'results/{n}.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows({k:json.dumps(plain(v),ensure_ascii=False)if isinstance(v,(list,dict))else plain(v)for k,v in r.items()}for r in rows)
def pack(z):
 p=z['prob'];y=np.c_[z['cat'][:,:3]==6,z['cat'][:,:3]==3];pred=np.c_[p[:,[6,14,22]]>=.5,p[:,[3,11,19]]>=.5];err=np.c_[(pred[:,:3]!=y[:,:3]).any(1),(pred[:,3:]!=y[:,3:]).any(1),p[:,24:27].argmax(1)!=z['cat'][:,3]];valid=z['cat_valid'][:,:4].all(1);ok=~err.any(1);ix={(str(e),int(o),str(m),int(t)):i for i,(e,o,m,t)in enumerate(zip(z['episode_id'],z['origin'],z['mode'],z['relative_step']))};return y,pred,err,valid,ok,ix
protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Second posthoc mechanism analysis, no training/inference/collection/original writes. New development diagnostic focus; holdout descriptive only.','matched_windows':'Same free origin/episode with >=3remaining. One_step at same origin andrelative1..3 has exactly same start; receives actual public observations after each preceding action. Never use origin0 one_step for matching different origins.','chain_rule':'P(C1)*P(C2|C1)*P(C3|C1C2)=count(C1C2C3)/allvalid matchedwindows. Future boundaries1..3 only; initial correctness separate.','paired':'Old/new identical episode,origin,actions,target labels; initial state/action/relation strata all retained with counts, no outcome-driven sampling.','mechanism':'Compare drawer GT and free/observed forecasts, actual control status, continuous distance tolabel thresholds, training analogous before/action/after exposure. No blanket attribution to rollout instability.','training_audit':'Read actual objective, per-update logs, schedules andtargets; count actual supervision horizons, fixed classweights, per-class exposure andmovingmean. No new LR/epoch/architecture ablation exists.'};(B/'configs/protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
meta={so:{e['episode_id']:e for e in rd(path)}for so,path in [('development',S/'cache/development_metadata.json'),('old_development',O/'cache/development_metadata.json'),('combination_test',S/'cache/combination_test_metadata.json')]};logs={(r['source'],r['episode_id'],r['step']):r for r in rd(A/'results/execution_log_audit.json')};windows=[];hazard=[];recover=[];stepcompare=[];allz={}
for so in ['old_development','development','combination_test']:
 for seed in [0,1]:
  for arm in ['old228','new868']:
   z=dict(np.load(S/f'results/predictions_{so}_{arm}_seed{seed}.npz'));allz[so,seed,arm]=z;y,pred,err,v,ok,ix=pack(z)
   for eid,e in meta[so].items():
    for origin in range(e['length']-2):
     ids=[ix[eid,origin,'free',t]for t in range(4)];obs=[ix[eid,origin,'one_step',t]for t in range(1,4)];assert v[ids+obs].all();assert np.array_equal(z['prob'][ids[1]],z['prob'][obs[0]])
     acts=[str(z['skill'][i])for i in ids[1:]];c=z['cat'][ids[0],:4];fs=[s for(so0,id0,s),r in logs.items()if so0==so and id0==eid and r['declared_controls']];freeok=ok[ids[1:]];obsok=ok[obs];ferrors=['+'.join(np.array(['Held','In','drawer'])[err[i]])or'none'for i in ids[1:]]
     row={'source':so,'seed':seed,'arm':arm,'episode_id':eid,'family_id':e['family_id'],'origin':origin,'initial_correct':bool(ok[ids[0]]),'initial_cat':c.tolist(),'initial_relation':f'In{sum(c[:3]==3)}_Held{sum(c[:3]==6)}_drawer{c[3]}','first_action':acts[0],'action_sequence':' > '.join(acts),'free_pattern':''.join('1'if b else'0'for b in freeok),'observed_pattern':''.join('1'if b else'0'for b in obsok),'free_correct':freeok.tolist(),'observed_correct':obsok.tolist(),'free_errors':ferrors,'endpoint_correct':bool(freeok[-1]),'whole_correct':bool(freeok.all()),'whole_including_initial_correct':bool(ok[ids].all()),'observed_whole_correct':bool(obsok.all()),'future_declared_intervention':any(origin<s<=origin+3 for s in fs),'previous_intervention':any(s<=origin for s in fs)};windows.append(row)
     for j,(fi,oi)in enumerate(zip(ids[1:],obs),1):
      stepcompare.append({'source':so,'seed':seed,'arm':arm,'episode_id':eid,'origin':origin,'relative_step':j,'skill':acts[j-1],'free_correct':bool(ok[fi]),'observed_correct':bool(ok[oi]),'free_error':ferrors[j-1],'observed_error':'+'.join(np.array(['Held','In','drawer'])[err[oi]])or'none'})
     if freeok[2]and not freeok[:2].all():
      lastbad=2 if not freeok[1]else 1;bi=ids[lastbad];recovered_at=lastbad+1;ri=ids[recovered_at];prev=ids[recovered_at-1];log=logs[so,eid,origin+recovered_at]
      recover.append({'source':so,'arm':arm,'seed':seed,'episode_id':eid,'origin':origin,'pattern':row['free_pattern'],'initial_correct':row['initial_correct'],'last_bad_step':lastbad,'last_bad_relations':ferrors[lastbad-1],'recovery_at_step':recovered_at,'recovery_skill':str(z['skill'][ri]),'last_skill':acts[2],'GT_drawer_before_recovery':int(z['cat'][prev,3]),'GT_drawer_after_recovery':int(z['cat'][ri,3]),'pred_drawer_before_recovery':int(z['prob'][prev,24:27].argmax()),'pred_drawer_after_recovery':int(z['prob'][ri,24:27].argmax()),'actual_success':log['actual_success_label'],'actual_declared_intervention':bool(log['declared_controls'])})
   rr=[r for r in windows if(r['source'],r['seed'],r['arm'])==(so,seed,arm)]
   for cohort in ['all','no_future_intervention']:
    ss=rr if cohort=='all'else[r for r in rr if not r['future_declared_intervention']]
    for mode in ['free','observed']:
     arr=np.array([r[mode+'_correct']for r in ss],bool);n=len(arr);s1=int(arr[:,0].sum());s12=int(arr[:,:2].all(1).sum());s123=int(arr.all(1).sum());marg=arr.mean(0);hazard.append({'source':so,'seed':seed,'arm':arm,'cohort':cohort,'mode':mode,'windows':n,'step1_correct':s1,'step12_correct':s12,'step123_correct':s123,'p1':s1/n,'p2_given1':s12/s1 if s1 else None,'p3_given12':s123/s12 if s12 else None,'conditional_product':s123/n,'marginal_step1':marg[0],'marginal_step2':marg[1],'marginal_step3':marg[2],'product_of_marginals':float(np.prod(marg)),'p1_cubed':(s1/n)**3,'pattern_counts':dict(collections.Counter(''.join('1'if v else'0'for v in row)for row in arr))})
out('matched_windows',windows);out('conditional_chain',hazard);out('matched_observation_comparison',stepcompare);out('recovery_windows',recover)
paired=[];gains=[]
for so in meta:
 for seed in [0,1]:
  a={(r['episode_id'],r['origin']):r for r in windows if(r['source'],r['seed'],r['arm'])==(so,seed,'old228')};b={(r['episode_id'],r['origin']):r for r in windows if(r['source'],r['seed'],r['arm'])==(so,seed,'new868')};assert a.keys()==b.keys()
  for key,r in a.items():
   q=b[key];assert r['initial_cat']==q['initial_cat']and r['action_sequence']==q['action_sequence']
   for metric in ['step1','endpoint','whole','whole_including_initial','observed_whole']:
    field=metric+'_correct';aa=r['free_correct'][0]if metric=='step1'else r[field];bb=q['free_correct'][0]if metric=='step1'else q[field];outcome='both_correct'if aa and bb else'gain'if bb else'loss'if aa else'both_wrong';row={'source':so,'seed':seed,'episode_id':key[0],'origin':key[1],'metric':metric,'outcome':outcome,'initial_relation':r['initial_relation'],'first_action':r['first_action'],'action_sequence':r['action_sequence'],'future_declared_intervention':r['future_declared_intervention'],'old_initial_correct':r['initial_correct'],'new_initial_correct':q['initial_correct'],'old_pattern':r['free_pattern'],'new_pattern':q['free_pattern'],'old_first_future_error':next((r['free_errors'][j]for j in range(3)if not r['free_correct'][j]),'none'),'new_first_future_error':next((q['free_errors'][j]for j in range(3)if not q['free_correct'][j]),'none')};gains.append(row)
for key in sorted({(r['source'],r['seed'],r['metric'])for r in gains}):
 rr=[r for r in gains if(r['source'],r['seed'],r['metric'])==key]
 for factor in ['all','initial_relation','first_action','action_sequence','future_declared_intervention','old_first_future_error']:
  for level in ['all']if factor=='all'else sorted({str(r[factor])for r in rr}):
   tt=rr if factor=='all'else[r for r in rr if str(r[factor])==level];c=collections.Counter(r['outcome']for r in tt);paired.append({'source':key[0],'seed':key[1],'metric':key[2],'factor':factor,'level':level,'windows':len(tt),'old_correct':c['both_correct']+c['loss'],'new_correct':c['both_correct']+c['gain'],'gains':c['gain'],'losses':c['loss'],'net_gain':c['gain']-c['loss']})
out('paired_windows',gains);out('paired_strata',paired)
print('CONDITIONAL',[r for r in hazard if r['source']=='development'and r['cohort']=='all'],flush=True)
