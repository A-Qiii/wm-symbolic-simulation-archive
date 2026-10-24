from common import *
import csv,collections,time
def writecsv(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def stats(a,mask):
 p=a['prob'][mask].astype(np.float64);y=a['cat'][mask];g=np.stack([p[:,j*8:j*8+8].argmax(1)for j in range(3)]+[p[:,24:27].argmax(1)],1);n=len(y)
 H=p[:,[6,14,22]]>=.5;I=p[:,[3,11,19]]>=.5;hy=y[:,:3]==6;iy=y[:,:3]==3
 s={'n':float(n),'object_n':float(3*n),'exact4':float((g==y).all(1).sum()),'nine':float(((H==hy).all(1)&(I==iy).all(1)&(g[:,3]==y[:,3])).sum()),'drawer_correct':float((g[:,3]==y[:,3]).sum()),'NLL_sum':float(np.stack([-np.log(np.clip(p[np.arange(n),off+y[:,j]],1e-9,1))for j,off in enumerate([0,8,16,24])],1).mean(1).sum()),'holding_exact':float((H==hy).all(1).sum()),'holding_positive_states':float(hy.any(1).sum()),'holding_identity_correct_positive':float(((H==hy).all(1)&hy.any(1)).sum()),'holding_identity_wrong_positive':float((hy.any(1)&H.any(1)&~(H==hy).all(1)).sum()),'holding_missed_positive':float((hy.any(1)&~H.any(1)).sum()),'holding_multiple':float((H.sum(1)>1).sum())}
 for name,z,q in [('Holding',hy,H),('InDrawer',iy,I)]:
  s.update({name+'_TP':float((z&q).sum()),name+'_FN':float((z&~q).sum()),name+'_FP':float((~z&q).sum()),name+'_TN':float((~z&~q).sum())})
 for j,c in enumerate(COLORS):
  for k in range(8):s[f'{c}_true{k}']=float((y[:,j]==k).sum());s[f'{c}_correct{k}']=float(((y[:,j]==k)&(g[:,j]==k)).sum())
 for k in range(8):s[f'loc_true{k}']=sum(s[f'{c}_true{k}']for c in COLORS);s[f'loc_correct{k}']=sum(s[f'{c}_correct{k}']for c in COLORS)
 for k in range(3):s[f'drawer_true{k}']=float((y[:,3]==k).sum());s[f'drawer_correct{k}']=float(((y[:,3]==k)&(g[:,3]==k)).sum())
 return s
def ratio(x,y):
 x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float));return np.divide(x,y,out=np.full(x.shape,np.nan),where=y>0)
def derive(s,support):
 d={'exact4':ratio(s['exact4'],s['n']),'relation9_exact':ratio(s['nine'],s['n']),'NLL4':ratio(s['NLL_sum'],s['n']),'drawer_accuracy':ratio(s['drawer_correct'],s['n']),'holding_exact':ratio(s['holding_exact'],s['n']),'holding_identity_positive':ratio(s['holding_identity_correct_positive'],s['holding_positive_states']),'holding_wrong_identity_positive':ratio(s['holding_identity_wrong_positive'],s['holding_positive_states']),'holding_missed_positive':ratio(s['holding_missed_positive'],s['holding_positive_states']),'holding_multiple_rate':ratio(s['holding_multiple'],s['n'])}
 for name in ['Holding','InDrawer']:
  tp,fn,fp,tn=[s[name+'_'+k]for k in ['TP','FN','FP','TN']];d[name+'_recall']=ratio(tp,tp+fn);d[name+'_precision']=ratio(tp,tp+fp);d[name+'_FPR']=ratio(fp,fp+tn);d[name+'_balanced_accuracy']=(ratio(tp,tp+fn)+ratio(tn,tn+fp))/2
 for prefix in ['loc']+COLORS:
  for k in range(8):d[f'{prefix}_recall{k}']=ratio(s[f'{prefix}_correct{k}'],s[f'{prefix}_true{k}'])
  ks=support[prefix]
  d[prefix+'_macro_recall']=np.mean(np.stack([d[f'{prefix}_recall{k}']for k in ks]),axis=0)if ks else np.asarray(s['n'])*np.nan
 d['location_accuracy']=ratio(sum(s[f'loc_correct{k}']for k in range(8)),s['object_n'])
 ks=[k for k in support['loc']if k<3];d['table_macro_recall']=np.mean(np.stack([d[f'loc_recall{k}']for k in ks]),axis=0)if ks else np.asarray(s['n'])*np.nan
 for k in range(3):d[f'drawer_recall{k}']=ratio(s[f'drawer_correct{k}'],s[f'drawer_true{k}'])
 return d
def interval(x):
 x=np.asarray(x);x=x[np.isfinite(x)];return (float(np.quantile(x,.025)),float(np.quantile(x,.975)),len(x))if len(x)else(None,None,0)
def number(x):return float(x)if np.isfinite(x)else None
def main():
 start=time.perf_counter();assert read(S/'results/compute_complete.json')['status']=='COMPLETED_FIXED_PROTOCOL';sys.addaudithook(audit_writes)
 rm={r['root_id']:r for r in read(P/'data/core1200_v1/roots_manifest.json')};metrics=[];paired=[];rootrows=[];confusions=[];examples=[];rawcounts=[]
 for seed in [0,1]:
  for split in SPLITS:
   arrays={arm:dict(np.load(S/f'data/predictions_{arm}_seed{seed}_{split}.npz'))for arm in ARMS};a=arrays['global'];b=arrays['object']
   for k in ['root_id','episode_id','step','cat','availability']:assert np.array_equal(a[k],b[k])
   for arm,aa in arrays.items():
    p=aa['prob'];y=aa['cat'];g=np.stack([p[:,j*8:j*8+8].argmax(1)for j in range(3)]+[p[:,24:27].argmax(1)],1)
    for i in range(len(p)):
     examples.append({'seed':seed,'arm':arm,'split':split,'root_id':aa['root_id'][i],'episode_id':aa['episode_id'][i],'step':int(aa['step'][i]),'initial_state':rm[aa['root_id'][i]]['initial_state'],'truth':','.join(map(str,y[i])),'prediction':','.join(map(str,g[i])),'exact4':bool(np.array_equal(y[i],g[i])),'holding_truth':','.join(map(str,(y[i,:3]==6).astype(int))),'holding_prediction':','.join(map(str,(p[i,[6,14,22]]>=.5).astype(int))),'available_colors':int((aa['availability'][i]>0).sum())})
   initial=np.array([rm[r]['initial_state']=='open_held'for r in a['root_id']])
   for cohort in ['h0','all_boundaries']:
    cm=(a['step']==0)&np.char.endswith(a['episode_id'],'_b0')if cohort=='h0'else np.ones(len(a['step']),bool)
    for stratum,sm in [('all',np.ones(len(cm),bool)),('initial_held',initial),('initial_not_held',~initial)]:
     mask=cm&sm;roots=sorted(set(a['root_id'][mask]));nroot=len(roots);assert nroot>0
     ident={'seed':seed,'split':split,'cohort':cohort,'stratum':stratum,'records':int(mask.sum()),'roots':nroot};pairs={};boots={}
     rng=np.random.default_rng(261019901+seed);indices=rng.integers(nroot,size=(2000,nroot));w=np.zeros((2000,nroot));np.add.at(w,(np.repeat(np.arange(2000),nroot),indices.ravel()),1)
     for arm,aa in arrays.items():
      rs=[stats(aa,mask&(aa['root_id']==r))for r in roots];keys=list(rs[0]);mat=np.array([[r[k]for k in keys]for r in rs]);total=dict(zip(keys,mat.sum(0)));boot=dict(zip(keys,(w@mat).T));support={prefix:[k for k in range(8)if total[f'{prefix}_true{k}']>0]for prefix in ['loc']+COLORS}
      value=derive(total,support);samples=derive(boot,support);pairs[arm]=value;boots[arm]=samples
      for r,st in zip(roots,rs):rootrows.append({**ident,'arm':arm,'root_id':r,**st})
      rawcounts.append({**ident,'arm':arm,**total,'location_macro_supported_classes':','.join(map(str,support['loc']))})
      for name,val in value.items():
       lo,hi,n=interval(samples[name]);metrics.append({**ident,'arm':arm,'metric':name,'value':number(val),'lo95':lo,'hi95':hi,'valid_bootstrap_replicates':n})
      p=aa['prob'][mask];y=aa['cat'][mask];g=np.stack([p[:,j*8:j*8+8].argmax(1)for j in range(3)],1)
      for j,color in enumerate(COLORS):
       conf=np.zeros((8,8),int);np.add.at(conf,(y[:,j],g[:,j]),1)
       for truth in range(8):
        for pred in range(8):confusions.append({**ident,'arm':arm,'color':color,'truth':truth,'prediction':pred,'count':int(conf[truth,pred])})
     for metric in pairs['global']:
      delta=pairs['object'][metric]-pairs['global'][metric];lo,hi,n=interval(boots['object'][metric]-boots['global'][metric]);paired.append({**ident,'metric':metric,'contrast':'object_minus_global','global_value':number(pairs['global'][metric]),'object_value':number(pairs['object'][metric]),'difference':number(delta),'lo95':lo,'hi95':hi,'valid_bootstrap_replicates':n})
   print('SCORED',seed,split,flush=True)
 writecsv(S/'results/metrics.csv',metrics);writecsv(S/'results/paired_differences.csv',paired);writecsv(S/'results/root_statistics.csv',rootrows);writecsv(S/'results/confusions.csv',confusions);writecsv(S/'results/per_example.csv',examples);writecsv(S/'results/counts.csv',rawcounts)
 primary=[r for r in paired if r['split']in ['test','externalC']and r['stratum']=='all'and r['metric']in ['loc_macro_recall','table_macro_recall','Holding_recall','Holding_FPR','holding_identity_positive','InDrawer_recall','InDrawer_FPR','exact4','relation9_exact','NLL4']]
 held=[r for r in rawcounts if r['split']in ['test','externalC']and r['stratum']=='initial_held']
 save(S/'results/analysis.json',{'utc':now(),'primary':primary,'initial_held_counts':held,'metric_rows':len(metrics),'paired_rows':len(paired),'root_rows':len(rootrows),'per_example_rows':len(examples),'seconds':time.perf_counter()-start,'bootstrap':{'unit':'independent physical root','paired':True,'replicates':2000,'seeds_separate':True,'macro_class_set':'Fixed classes supported in original reporting subset; bootstrap draws missing any required class omitted for that macro metric; valid replicate counts reported','multiple_comparison_adjustment':False}})
 for r in primary:
  if r['cohort']=='all_boundaries'and r['metric']in ['loc_macro_recall','holding_identity_positive','InDrawer_recall','exact4']:print('PRIMARY',json.dumps(r),flush=True)
 print('ANALYSIS_COMPLETE',round(time.perf_counter()-start,3),flush=True)
if __name__=='__main__':main()
