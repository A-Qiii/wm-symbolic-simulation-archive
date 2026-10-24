from common import *
import csv,math
METRICS=['precision','recall','balanced_accuracy','F1','NLL']
def vec(y,p):
 y=np.asarray(y,bool);p=np.asarray(p,float);q=p>=.5;c=np.clip(p,1e-7,1-1e-7)
 return np.array([sum(y&q),sum(~y&q),sum(y&~q),sum(~y&~q),-sum(y*np.log(c)+(~y)*np.log1p(-c)),len(y)],float)
def vals(a):
 tp,fp,fn,tn,loss,n=np.moveaxis(np.asarray(a,float),-1,0)
 def div(x,d):return np.divide(x,d,out=np.full_like(x,np.nan,dtype=float),where=d>0)
 return dict(precision=div(tp,tp+fp),recall=div(tp,tp+fn),balanced_accuracy=(div(tp,tp+fn)+div(tn,tn+fp))/2,F1=div(2*tp,2*tp+fp+fn),NLL=div(loss,n))
def finite(x):return float(x)if np.isfinite(x)else None
def main():
 guard('score');rows=read(B/'data/records.json');boot=np.load(B/'results/bootstrap_draws.npz');roots=boot['root_ids'];weights=boot['root_multiplicities'];assert weights.shape==(2000,12)
 alias={1101,1107,1113,1119,1510,1516,1522,1528};metrics=[];cis=[];diffs=[];matrices={};errors=[];alias_predictions=[];verified=0
 for seed in [0,1]:
  for arm in ARMS:
   for split in ['train','dev']:
    z=np.load(E/f'results/predictions_{arm}_seed{seed}_{split}.npz');ids=z['record_indices'];pr=z['probability'];rr=[rows[int(i)]for i in ids];truth=read(B/f'data/supervision/{split}_targets.json');assert np.array_equal(ids,[r['record_index']for r in truth]);y=np.array([r['y']for r in truth],dtype=bool);assert np.array_equal(y,[r['y']for r in rr]);assert all(r['eligible']for r in rr)
    groups=[('all','all',np.arange(len(ids)))]+[('color',c,np.array([j for j,r in enumerate(rr)if r['color']==c]))for c in ['red','blue','pink']]+[('camera',c,np.array([j for j,r in enumerate(rr)if r['camera']==c]))for c in ['static','gripper']]+[('color_camera',c+'/'+cam,np.array([j for j,r in enumerate(rr)if r['color']==c and r['camera']==cam]))for c in ['red','blue','pink']for cam in ['static','gripper']]+[('root',root,np.array([j for j,r in enumerate(rr)if r['root_id']==root]))for root in sorted({r['root_id']for r in rr})]
    if split=='dev':
     groups.append(('alias','dev_geometry_alias8',np.array([j for j,i in enumerate(ids)if i in alias])))
     groups.extend([('difficulty','target_only',np.array([j for j,r in enumerate(rr)if r['color']==r['target_color']])),('difficulty','non_target',np.array([j for j,r in enumerate(rr)if r['color']!=r['target_color']])),('difficulty','target_closed_supported_negatives',np.array([j for j,r in enumerate(rr)if r['color']==r['target_color']and r['anchor']=='closed_on_support'and not r['y']]))])
    for kind,group,ii in groups:
     a=vec(y[ii],pr[ii]);v=vals(a);row=dict(arm=arm,seed=seed,split=split,group_type=kind,group=group,TP=int(a[0]),FP=int(a[1]),FN=int(a[2]),TN=int(a[3]),N=int(a[5]),positive_queries=int(sum(y[ii])),negative_queries=int(sum(~y[ii])),false_positive_rate=float(a[1]/(a[1]+a[3]))if a[1]+a[3]else None,**{k:finite(vv)for k,vv in v.items()});metrics.append(row)
     # Independent scalar reference (no classifier/encoder execution).
     sy=[bool(y[j])for j in ii];sp=[float(pr[j])for j in ii];tp=sum(t and p>=.5 for t,p in zip(sy,sp));fp=sum(not t and p>=.5 for t,p in zip(sy,sp));fn=sum(t and p<.5 for t,p in zip(sy,sp));tn=sum(not t and p<.5 for t,p in zip(sy,sp));assert [tp,fp,fn,tn]==[row[k]for k in ['TP','FP','FN','TN']]
     nll=sum(-math.log(max(1e-7,min(1-1e-7,p)))if t else-math.log1p(-max(1e-7,min(1-1e-7,p)))for t,p in zip(sy,sp))/len(ii);assert abs(nll-row['NLL'])<1e-10;verified+=1
    for j,(i,r)in enumerate(zip(ids,rr)):
     base={'arm':arm,'seed':seed,'split':split,'record_index':int(i),'record_id':r['record_id'],'root_id':r['root_id'],'anchor':r['anchor'],'camera':r['camera'],'color':r['color'],'y':int(y[j]),'probability':float(pr[j]),'correct':bool((pr[j]>=.5)==y[j])}
     if not base['correct']:errors.append(base)
     if int(i)in alias:alias_predictions.append(base)
    if split=='dev':
     for scope in ['all','target_only']:
      matrices[(arm,seed,scope)]=np.stack([vec([y[j]for j,r in enumerate(rr)if r['root_id']==root and(scope=='all'or r['color']==r['target_color'])],[pr[j]for j,r in enumerate(rr)if r['root_id']==root and(scope=='all'or r['color']==r['target_color'])])for root in roots])
 for(arm,seed,scope),matrix in matrices.items():
  point=vals(matrix.sum(0));samples=vals(weights@matrix)
  for m in METRICS:
   v=samples[m];g=v[np.isfinite(v)];lo,hi=np.quantile(g,[.025,.975]);cis.append(dict(arm=arm,seed=seed,scope=scope,metric=m,estimate=finite(point[m]),low95=float(lo),high95=float(hi),valid_replicates=len(g)))
 for seed in [0,1]:
  for scope in ['all','target_only']:
   for a,b in [('full','geometry_masked_RGB'),('full','RGB_masked_geometry'),('RGB_masked_geometry','geometry_masked_RGB')]:
    va,vb=vals(weights@matrices[(a,seed,scope)]),vals(weights@matrices[(b,seed,scope)]);pa,pb=vals(matrices[(a,seed,scope)].sum(0)),vals(matrices[(b,seed,scope)].sum(0))
    for m in METRICS:
     v=va[m]-vb[m];g=v[np.isfinite(v)];lo,hi=np.quantile(g,[.025,.975]);diffs.append(dict(comparison=a+'-minus-'+b,seed=seed,scope=scope,metric=m,estimate=finite(pa[m]-pb[m]),low95=float(lo),high95=float(hi),valid_replicates=len(g)))
 out={'utc':now(),'exploratory_seen_C':True,'threshold':.5,'metrics':metrics,'root_bootstrap_CI':cis,'paired_differences':diffs,'alias_predictions':alias_predictions,'errors':errors,'independent_scalar_metric_checks':verified,'new_fits':6,'unchanged_labels_and_eligibility':True,'bootstrap_original_file_sha256':sha(B/'results/bootstrap_draws.npz'),'alias_CI':'No separate CI from only2selected diagnostic roots; per-case and aggregate descriptive results only'};save(E/'results/evaluation.json',out);save(R/'evaluation.json',out)
 for name,data in [('metrics.csv',metrics),('root_bootstrap_CI.csv',cis),('paired_differences.csv',diffs),('alias_predictions.csv',alias_predictions),('errors.csv',errors)]:
  with(R/name).open('w')as f:w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
 for m in metrics:
  if m['split']=='dev'and m['group_type']in ['all','difficulty']:print(m)
 for m in diffs:
  if m['metric']=='balanced_accuracy':print(m)
if __name__=='__main__':main()
