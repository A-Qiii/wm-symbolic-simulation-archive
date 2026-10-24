from common import *
import csv
METRICS=['precision','recall','balanced_accuracy','F1','NLL','FP','false_positive_rate']
ALIAS={1101,1107,1113,1119,1510,1516,1522,1528}
def vector(y,p):
 y=np.asarray(y,bool);p=np.asarray(p,float);q=p>=.5;c=np.clip(p,1e-7,1-1e-7);return np.array([sum(y&q),sum(~y&q),sum(y&~q),sum(~y&~q),-sum(y*np.log(c)+(~y)*np.log1p(-c)),len(y)],float)
def values(a):
 tp,fp,fn,tn,loss,n=np.moveaxis(np.asarray(a,float),-1,0)
 def div(x,d):return np.divide(x,d,out=np.full_like(x,np.nan,dtype=float),where=d>0)
 return {'precision':div(tp,tp+fp),'recall':div(tp,tp+fn),'balanced_accuracy':(div(tp,tp+fn)+div(tn,tn+fp))/2,'F1':div(2*tp,2*tp+fp+fn),'NLL':div(loss,n),'FP':fp,'false_positive_rate':div(fp,fp+tn)}
def finite(v):return float(v)if np.isfinite(v)else None
def select(r,index,scope,color=None,camera=None,root=None):
 if color is not None and r['color']!=color:return False
 if camera is not None and r['camera']!=camera:return False
 if root is not None and r['root_id']!=root:return False
 target=r['color']==r['target_color']
 return scope=='all'or scope=='target'and target or scope=='supported_negative'and target and r['anchor']=='closed_on_support'and not r['y']or scope=='non_target'and not target or scope=='alias'and index in ALIAS
def interval(v):
 g=np.asarray(v)[np.isfinite(v)];return {'low95':float(np.quantile(g,.025))if len(g)else None,'high95':float(np.quantile(g,.975))if len(g)else None,'valid_replicates':len(g),'undefined_replicates':2000-len(g)}
def main():
 guard('score');assert not(E/'results/evaluation.json').exists();rows=read(B/'data/records.json');boot=np.load(B/'results/bootstrap_draws.npz');weights=boot['root_multiplicities'];roots=boot['root_ids'];metrics=[];matrices={};casepred=[];errors=[]
 for seed in [0,1]:
  for arm in ARMS:
   for split in ['train','dev']:
    z=np.load(E/f'results/predictions_{arm}_seed{seed}_{split}.npz');ids=z['record_indices'];pr=z['probability'];truth=read(B/f'data/supervision/{split}_targets.json');assert np.array_equal(ids,[r['record_index']for r in truth]);y=np.array([r['y']for r in truth],bool);rr=[rows[int(i)]for i in ids];assert all(r['eligible']and r['split']==split for r in rr);assert np.array_equal(y,[r['y']for r in rr])
    groups=[('all','all',None,None,None)]+[('color_'+c,'all',c,None,None)for c in ['red','blue','pink']]+[('camera_'+c,'all',None,c,None)for c in ['static','gripper']]+[('color_camera_'+c+'_'+cam,'all',c,cam,None)for c in ['red','blue','pink']for cam in ['static','gripper']]+[('root_'+root,'all',None,None,root)for root in sorted({r['root_id']for r in rr})]+[('target','target',None,None,None),('supported_negative','supported_negative',None,None,None),('non_target','non_target',None,None,None)]+[('target_color_'+c,'target',c,None,None)for c in ['red','blue','pink']]+[(scope+'_'+cam,scope,None,cam,None)for scope in ['target','supported_negative']for cam in ['static','gripper']]
    if split=='dev':groups.append(('alias8','alias',None,None,None))
    for group,scope,color,camera,root in groups:
     ii=np.array([j for j,(r,i)in enumerate(zip(rr,ids))if select(r,int(i),scope,color,camera,root)],int);a=vector(y[ii],pr[ii]);v=values(a);selected=[rr[j]for j in ii];metrics.append({'arm':arm,'seed':seed,'split':split,'group':group,'scope':scope,'color':color,'camera':camera,'root':root,'TP':int(a[0]),'FP':int(a[1]),'FN':int(a[2]),'TN':int(a[3]),'N':int(a[5]),'positive_queries':int(sum(y[ii])),'negative_queries':int(sum(~y[ii])),'root_count':len({r['root_id']for r in selected}),'positive_roots':len({r['root_id']for r in selected if r['y']}),'negative_roots':len({r['root_id']for r in selected if not r['y']}),**{k:finite(x)for k,x in v.items()if k!='FP'}})
    for j,(i,r)in enumerate(zip(ids,rr)):
     entry={'arm':arm,'seed':seed,'split':split,'record_index':int(i),'record_id':r['record_id'],'root_id':r['root_id'],'color':r['color'],'camera':r['camera'],'anchor':r['anchor'],'y':int(y[j]),'probability':float(pr[j]),'correct':bool((pr[j]>=.5)==y[j])}
     if not entry['correct']:errors.append(entry)
     if r['root_id']in ['relation36_C_24','relation36_C_34']and r['color']==r['target_color']:casepred.append(entry)
    if split=='dev':
     for scope in ['all','target','supported_negative']:
      matrices[(arm,seed,scope)]=np.stack([vector([y[j]for j,r in enumerate(rr)if r['root_id']==root and select(r,int(ids[j]),scope)],[pr[j]for j,r in enumerate(rr)if r['root_id']==root and select(r,int(ids[j]),scope)])for root in roots])
 cis=[];diffs=[]
 for(arm,seed,scope),a in matrices.items():
  point=values(a.sum(0));bootv=values(weights@a)
  for metric in METRICS:cis.append({'arm':arm,'seed':seed,'scope':scope,'metric':metric,'estimate':finite(point[metric]),**interval(bootv[metric])})
 for seed in [0,1]:
  for scope in ['all','target','supported_negative']:
   aa=matrices[('context_present',seed,scope)];bb=matrices[('context_zero',seed,scope)];av,bv=values(weights@aa),values(weights@bb);ap,bp=values(aa.sum(0)),values(bb.sum(0))
   for metric in METRICS:diffs.append({'comparison':'context_present-minus-context_zero','seed':seed,'scope':scope,'metric':metric,'estimate':finite(ap[metric]-bp[metric]),**interval(av[metric]-bv[metric])})
 rule=all(next(d for d in diffs if d['seed']==seed and d['scope']=='target'and d['metric']=='balanced_accuracy')['low95']>0 and next(d for d in diffs if d['seed']==seed and d['scope']=='supported_negative'and d['metric']=='false_positive_rate')['high95']<0 for seed in [0,1])
 out={'utc':now(),'exploratory_seen_C':True,'threshold':.5,'metrics':metrics,'root_bootstrap_CI':cis,'paired_differences':diffs,'cases':casepred,'errors':errors,'clear_primary_gain_rule_passed':bool(rule),'static_readout_tuning_closed':True,'bootstrap':'original2000 paired whole12Croot draws; undefined metrics null, count/FPR recomputed','new_fits':4,'new_data':0};save(E/'results/evaluation.json',out);save(R/'evaluation.json',out)
 for name,data in [('metrics.csv',metrics),('root_bootstrap_CI.csv',cis),('paired_differences.csv',diffs),('alias_neighbor_predictions.csv',casepred),('errors.csv',errors)]:
  with(R/name).open('w')as f:w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
 print('CLEAR_PRIMARY_GAIN',rule)
 for r in metrics:
  if r['group']in ['all','target','supported_negative']:print({k:r[k]for k in ['arm','seed','split','group','N','TP','FP','FN','TN','precision','recall','balanced_accuracy','NLL']})
 for d in diffs:
  if d['scope']=='target'and d['metric']=='balanced_accuracy'or d['scope']=='supported_negative'and d['metric']in ['FP','false_positive_rate']:print(d)
if __name__=='__main__':main()
