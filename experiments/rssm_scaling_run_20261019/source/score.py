from common import *
import csv,itertools

CAT=['red_location','blue_location','pink_location','drawer','slider']
SLICES=[slice(0,8),slice(8,16),slice(16,24),slice(24,27),slice(27,30)]
SPLITS=['train','common20','validation','test','externalC']
def writecsv(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def ci(v):
 a=np.array(list(v),float)
 if not len(a):return None,None,None
 rng=np.random.default_rng(261019900);b=a[rng.integers(len(a),size=(2000,len(a)))].mean(1)
 return float(a.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))
def rms(v,mask,roots):return {str(r):float(v[(roots==r)&mask].mean())for r in sorted(set(roots))if ((roots==r)&mask).any()}
def ratios(tp,fp,fn,tn):
 precision=tp/(tp+fp)if tp+fp else None;recall=tp/(tp+fn)if tp+fn else None;spec=tn/(tn+fp)if tn+fp else None
 return precision,recall,(recall+spec)/2 if recall is not None and spec is not None else None

def score(a,meta,rootmeta):
 roots=a['root_id'];p=a['prob'];y=a['cat'];cv=a['cat_valid'].astype(bool);bv=a['binary_valid'].astype(bool);n=len(roots)
 assert p.shape==(n,32)and np.isfinite(p).all()and cv.all()and bv.all()
 assert all(np.allclose(p[:,sl].sum(-1),1,atol=1e-5)for sl in SLICES)
 pred=np.stack([p[:,sl].argmax(-1)for sl in SLICES],1);nll=np.stack([-np.log(np.clip(p[np.arange(n),off+y[:,j]],1e-9,1))for j,off in enumerate([0,8,16,24,27])],1)
 facts=[]
 for j,name in enumerate(CAT):facts.append((name,(pred[:,j]==y[:,j]).astype(float),nll[:,j],cv[:,j],y[:,j]!=a['base_cat'][:,j],a['base_valid'][:,j].astype(bool)))
 for j,name in enumerate(['light','LED']):
  z=a['binary'][:,j];q=p[:,30+j];loss=-(z*np.log(np.clip(q,1e-9,1))+(1-z)*np.log(np.clip(1-q,1e-9,1)))
  facts.append((name,((q>=.5)==z).astype(float),loss,bv[:,j],z!=a['base_binary'][:,j],a['base_binary_valid'][:,j].astype(bool)))
 for j,color in enumerate(['red','blue','pink']):
  for relation,klass in [('Holding',6),('In',3)]:
   z=y[:,j]==klass;q=p[:,8*j+klass];loss=-(z*np.log(np.clip(q,1e-9,1))+(1-z)*np.log(np.clip(1-q,1e-9,1)))
   facts.append((relation+'_'+color,((q>=.5)==z).astype(float),loss,cv[:,j],z!=(a['base_cat'][:,j]==klass),a['base_valid'][:,j].astype(bool)))
 hold=p[:,[6,14,22]]>=.5;ins=p[:,[3,11,19]]>=.5;hy=y[:,:3]==6;iy=y[:,:3]==3
 facts += [('primary4',(pred[:,:4]==y[:,:4]).mean(1),nll[:,:4].mean(1),cv[:,:4].all(1),None,None),('all5_exact',(pred==y).all(1).astype(float),None,cv.all(1),None,None),('all7_exact',((pred==y).all(1)&((p[:,30:]>=.5)==a['binary']).all(1)).astype(float),None,cv.all(1)&bv.all(1),None,None),('relation9_exact',((hold==hy).all(1)&(ins==iy).all(1)&(pred[:,3]==y[:,3])).astype(float),None,cv[:,:4].all(1),None,None),('multiple_holding',(hold.sum(1)>1).astype(float),None,cv[:,:3].all(1),None,None)]
 it=np.array([rootmeta[r]['initial_state']for r in roots]);rows=[];rootrows=[];relrows=[]
 for mode in sorted(set(a['mode'])):
  maskmode=a['mode']==mode
  strata={'all':np.ones(n,bool)}
  if mode in ['current','one_step','free1','free3','free4','probe']:
   strata.update(initial_closed_empty=it=='closed_empty',initial_open_empty=it=='open_empty',initial_open_held=it=='open_held',initial_not_held=it!='open_held')
  if mode not in ['current','probe']:
   strata.update(zero_control=a['controls']==0,nonzero_control=a['controls']>0,executor_success=a['status']=='success',executed_failure=a['status']=='executed_failure')
  if mode.startswith('free'):strata['all_prefix_actuated']=a['all_prefix_actuated']
  if mode=='one_step':
   for skill in sorted(set(a['skill'][maskmode])):strata['skill:'+skill]=a['skill']==skill
  for st,ss in strata.items():
   sel=maskmode&ss
   for name,acc,loss,valid,changed,bvalid in facts:
    changes=['all','changed','unchanged']if changed is not None and st=='all'and mode in ['one_step','free3']else['all']
    for ch in changes:
     mask=sel&valid
     if ch!='all':mask=mask&bvalid&(changed if ch=='changed'else~changed)
     for metric,values in [('accuracy',acc),('NLL',loss)]:
      if values is None:continue
      rv=rms(values,mask,roots);v,lo,hi=ci(rv.values());ident={**meta,'mode':mode,'stratum':st,'fact':name,'change':ch,'metric':metric}
      rows.append({**ident,'records':int(mask.sum()),'roots':len(rv),'value':v,'lo95':lo,'hi95':hi,'micro':float(values[mask].mean())if mask.any()else None})
      for rid,value in rv.items():rootrows.append({**ident,'root_id':rid,'value':value})
   if mode in ['current','one_step','free3','probe']:
    for name,truth,guess in [('Holding',hy,hold),('In',iy,ins)]:
     counts=[]
     for rid in sorted(set(roots[sel])):
      m=(sel&(roots==rid))[:,None]&cv[:,:3]
      counts.append([int((m&guess&truth).sum()),int((m&guess&~truth).sum()),int((m&~guess&truth).sum()),int((m&~guess&~truth).sum())])
     counts=np.array(counts,dtype=int).reshape(-1,4);total=counts.sum(0);point=ratios(*total);rng=np.random.default_rng(261019900)
     boot=np.array([ratios(*v)for v in counts[rng.integers(len(counts),size=(2000,len(counts)))].sum(1)],dtype=float)if len(counts)else np.empty((0,3))
     row={**meta,'mode':mode,'stratum':st,'relation':name,'roots':len(counts),'answers':int(total.sum()),**dict(zip(['TP','FP','FN','TN'],map(int,total)))}
     for j,k in enumerate(['precision','recall','BA']):
      b=boot[:,j];b=b[np.isfinite(b)];row[k]=point[j];row[k+'_lo95']=float(np.quantile(b,.025))if len(b)else None;row[k+'_hi95']=float(np.quantile(b,.975))if len(b)else None
     relrows.append(row)
 return rows,rootrows,relrows

def main():
 guard('score');assert read(B/'results/compute_complete.json')['heldout_after_allfits_frozen']
 rootmeta={r['root_id']:r for r in read(P/'data/core1200_v1/roots_manifest.json')if r['environment']in ['A','B','C']}
 rows=[];relrows=[];paired_pool={};root_count=0
 rootkeys=['size','seed','arm','split','mode','stratum','fact','change','metric','root_id','value']
 with (REPORT/'root_metrics.csv').open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=rootkeys);writer.writeheader()
  for size in [20,40]:
   for seed in [0,1]:
    for arm in ARMS:
     for split in SPLITS:
      for prefix in ['predictions','probe_predictions']:
       path=B/f'cells/roots{size}/data/{prefix}_{arm}_seed{seed}_{split}.npz';a=dict(np.load(path));rr,rt,rl=score(a,{'size':size,'seed':seed,'arm':arm,'split':split},rootmeta);rows+=rr;relrows+=rl;writer.writerows(rt);root_count+=len(rt)
       for r in rt:
        if r['split']=='train':continue # Different training roots: no false paired newroot contrast.
        key=tuple(r[k]for k in ['seed','split','mode','stratum','fact','change','metric']);paired_pool.setdefault(key,{}).setdefault((size,arm),{})[r['root_id']]=r['value']
      print('SCORED',size,seed,arm,split,flush=True)
 writecsv(REPORT/'all_metrics.csv',rows);writecsv(REPORT/'relation_confusions.csv',relrows)
 pairs=[];keynames=['seed','split','mode','stratum','fact','change','metric']
 for key,d in paired_pool.items():
  common=set.intersection(*(set(v)for v in d.values()));assert all(set(v)==common for v in d.values())
  contrasts={'size40_minus20_joint':{r:d[40,'joint'][r]-d[20,'joint'][r]for r in common},'size40_minus20_control':{r:d[40,'route_current_only'][r]-d[20,'route_current_only'][r]for r in common},'joint_minus_control_roots20':{r:d[20,'joint'][r]-d[20,'route_current_only'][r]for r in common},'joint_minus_control_roots40':{r:d[40,'joint'][r]-d[40,'route_current_only'][r]for r in common}}
  contrasts['size_improvement_difference_joint_minus_control']={r:contrasts['size40_minus20_joint'][r]-contrasts['size40_minus20_control'][r]for r in common}
  for name,rv in contrasts.items():
   v,lo,hi=ci(rv[r]for r in sorted(rv));pairs.append({**dict(zip(keynames,key)),'contrast':name,'roots':len(rv),'value':v,'lo95':lo,'hi95':hi})
 writecsv(REPORT/'paired_contrasts.csv',pairs)
 standardized=[]
 # Fixed original 50/30/20 initial-state weights, descriptive external gap only.
 for size,seed,arm,mode,fact,metric in itertools.product([20,40],[0,1],ARMS,['probe','free3'],['primary4','relation9_exact','all5_exact'],['NLL','accuracy']):
  if fact!='primary4'and metric=='NLL':continue
  for split in ['test','externalC']:
   matches=[next(r for r in rows if all(r[k]==v for k,v in {'size':size,'seed':seed,'arm':arm,'split':split,'mode':mode,'fact':fact,'metric':metric,'stratum':'initial_'+st,'change':'all'}.items()))for st in ['closed_empty','open_empty','open_held']]
   standardized.append({'size':size,'seed':seed,'arm':arm,'split':split,'mode':mode,'fact':fact,'metric':metric,'value':sum(w*r['value']for w,r in zip([.5,.3,.2],matches)),'weights':'.5closed_empty+.3open_empty+.2open_held','interpretation':'descriptive_only_no_causal_environment_effect'})
 writecsv(REPORT/'initial_state_standardized.csv',standardized)
 primary=[r for r in rows if r['mode']in ['probe','free3']and r['fact']in ['primary4','relation9_exact','all5_exact','all7_exact']and r['stratum']=='all'and r['change']=='all']
 pc=[r for r in pairs if r['split']in ['test','externalC']and r['mode']in ['probe','free3']and r['fact']in ['primary4','relation9_exact','all5_exact']and r['stratum']=='all'and r['change']=='all']
 save(B/'results/scores.json',{'utc':now(),'primary':primary,'paired_primary':pc,'metric_rows':len(rows),'root_rows':root_count,'paired_rows':len(pairs),'relation_rows':len(relrows),'root_bootstrap_replicates':2000,'NLL_floor':1e-9,'no_seed_pooling':True,'no_D':True,'no_new_fit':True,'fullprefix_available':False})
 print('SCORING_COMPLETE',len(rows),root_count,flush=True)

if __name__=='__main__':main()
