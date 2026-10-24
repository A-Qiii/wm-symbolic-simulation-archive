from common import *
import collections,csv
METRICS=['precision','recall','balanced_accuracy','F1','NLL']
def values(a):
 a=np.asarray(a,float);tp,fp,fn,tn,loss,n=np.moveaxis(a,-1,0)
 def ratio(x,d):return np.divide(x,d,out=np.full_like(x,np.nan,dtype=float),where=d>0)
 return {'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),'balanced_accuracy':(ratio(tp,tp+fn)+ratio(tn,tn+fp))/2,'F1':ratio(2*tp,2*tp+fp+fn),'NLL':ratio(loss,n)}
def vector(y,p):
 y=np.asarray(y,bool);p=np.asarray(p,float);pred=p>=.5;clipped=np.clip(p,1e-7,1-1e-7);return np.array([sum(y&pred),sum(~y&pred),sum(y&~pred),sum(~y&~pred),-np.sum(y*np.log(clipped)+(~y)*np.log1p(-clipped)),len(y)],float)
def finite(x):return float(x)if np.isfinite(x)else None
def main():
 install_guard([]);assert(E/'results/prediction.json').exists();assert not(E/'results/evaluation.json').exists();rows=read(E/'data/records.json');metrics=[];matrices={};errors=[];predmap={}
 roots=sorted({r['root_id']for r in rows if r['split']=='dev'});assert len(roots)==12;rng=np.random.default_rng(261019420);draws=rng.integers(0,12,size=(2000,12));weights=np.array([np.bincount(x,minlength=12)for x in draws]);save_raw_npz(E/'results/bootstrap_draws.npz',root_multiplicities=weights,root_ids=np.array(roots))
 for seed in [0,1]:
  for arm in ['object','pair','geometry']:
   for split in ['train','dev']:
    z=np.load(E/f'results/predictions_{arm}_seed{seed}_{split}.npz');ids=z['record_indices'];p=z['probability'];rs=[rows[int(i)]for i in ids];assert all(r['eligible']and r['split']==split for r in rs);truth=read(E/f'data/supervision/{split}_targets.json');assert np.array_equal(ids,[x['record_index']for x in truth]);y=np.array([r['y']for r in rs]);assert np.array_equal(y,[x['y']for x in truth]);predmap[(arm,seed,split)]={int(i):float(q)for i,q in zip(ids,p)}
    for i,q,rr in zip(ids,p,rs):
     if bool(q>=.5)!=rr['y']:errors.append({'arm':arm,'seed':seed,'split':split,'record_index':int(i),'record_id':rr['record_id'],'root_id':rr['root_id'],'color':rr['color'],'camera':rr['camera'],'anchor':rr['anchor'],'y':int(rr['y']),'probability':float(q),'error':'FN'if rr['y']else'FP'})
    groups=[('all','all',np.arange(len(rs)))]+[('color',c,np.array([i for i,r in enumerate(rs)if r['color']==c]))for c in COLORS]+[('camera',c,np.array([i for i,r in enumerate(rs)if r['camera']==c]))for c in ['static','gripper']]+[('color_camera',c+'/'+cam,np.array([i for i,r in enumerate(rs)if r['color']==c and r['camera']==cam]))for c in COLORS for cam in ['static','gripper']]+[('root',root,np.array([i for i,r in enumerate(rs)if r['root_id']==root]))for root in sorted({r['root_id']for r in rs})]
    for kind,group,ii in groups:
     selected=[rs[int(i)]for i in ii];a=vector(y[ii],p[ii]);v=values(a);full=[r for r in rows if r['split']==split and(kind=='all'or(kind=='color'and r['color']==group)or(kind=='camera'and r['camera']==group)or(kind=='root'and r['root_id']==group)or(kind=='color_camera'and r['color']+'/'+r['camera']==group))];phys=[r for r in full if r['physical_valid']];physpos=sum(r['y']for r in phys)
     metrics.append({'arm':arm,'seed':seed,'split':split,'group_type':kind,'group':group,'TP':int(a[0]),'FP':int(a[1]),'FN':int(a[2]),'TN':int(a[3]),'N':int(a[5]),'positive_queries':int(sum(y[ii])),'negative_queries':int(sum(~y[ii])),'positive_roots':len({r['root_id']for r in selected if r['y']}),'negative_roots':len({r['root_id']for r in selected if not r['y']}),'roots':len({r['root_id']for r in selected}),'physical_valid_queries':len(phys),'physical_valid_positives':physpos,'unobservable_queries':sum(not r['visually_eligible']for r in full),'physical_unknown_queries':sum(not r['physical_valid']for r in full),'eligible_coverage':len(selected)/len(phys)if phys else None,'full_positive_recall':int(a[0])/physpos if physpos else None,**{k:finite(x)for k,x in v.items()}})
    if split=='dev':matrices[(arm,seed)]=np.stack([vector([r['y']for r in rs if r['root_id']==root],[p[i]for i,r in enumerate(rs)if r['root_id']==root])for root in roots])
 cis=[];diffs=[]
 for (arm,seed),matrix in matrices.items():
  vv=values(weights@matrix);point=values(matrix.sum(0))
  for metric in METRICS:
   good=vv[metric][np.isfinite(vv[metric])];lo,hi=np.quantile(good,[.025,.975]);cis.append({'arm':arm,'seed':seed,'metric':metric,'estimate':finite(point[metric]),'low95':float(lo),'high95':float(hi),'valid_replicates':len(good),'root_clusters':12})
 for seed in [0,1]:
  for baseline in ['object','geometry']:
   pair=values(weights@matrices[('pair',seed)]);base=values(weights@matrices[(baseline,seed)]);pp=values(matrices[('pair',seed)].sum(0));bb=values(matrices[(baseline,seed)].sum(0))
   for metric in METRICS:
    diff=pair[metric]-base[metric];good=diff[np.isfinite(diff)];lo,hi=np.quantile(good,[.025,.975]);diffs.append({'comparison':'pair-minus-'+baseline,'seed':seed,'metric':metric,'estimate':finite(pp[metric]-bb[metric]),'low95':float(lo),'high95':float(hi),'valid_replicates':len(good),'higher_is_better':metric!='NLL'})
 conflicts=[]
 for split in ['train','dev']:
  z=np.load(E/f'data/features/{split}.npz');ids=z['record_indices']
  for kind in ['block_RGB_crop','object','pair','geometry']:
   groups=collections.defaultdict(list)
   for j,idx in enumerate(ids):key=str(z['block_crop_hash'][j])if kind=='block_RGB_crop'else hashlib.sha256(z['x_'+kind][j].tobytes()).hexdigest();groups[key].append(int(idx))
   bad=[{'input_hash':h,'record_indices':ii,'labels':[int(rows[i]['y'])for i in ii],'roots':sorted({rows[i]['root_id']for i in ii})}for h,ii in groups.items()if len({rows[i]['y']for i in ii})>1];conflicts.append({'split':split,'input':kind,'conflicting_exact_input_groups':len(bad),'queries_in_conflicts':sum(len(x['record_indices'])for x in bad),'examples':bad})
 result={'utc':now(),'threshold':.5,'metrics':metrics,'root_bootstrap_CI':cis,'paired_differences':diffs,'exact_input_conflicts':conflicts,'errors':errors,'bootstrap':'2000 entire12devroot paired draws, fixed seed261019420; pooled counts recomputed','dev_tuning':False,'test_D':False,'model_runs_complete':6};save(E/'results/evaluation.json',result);save(R/'evaluation.json',result)
 for name,data in [('metrics.csv',metrics),('root_bootstrap_CI.csv',cis),('paired_differences.csv',diffs),('errors.csv',errors)]:
  with(R/name).open('w')as f:w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
 print('EVALUATION_COMPLETE',flush=True)
 for r in metrics:
  if r['split']=='dev'and r['group_type']=='all':print(r,flush=True)
 for r in diffs:
  if r['metric']=='balanced_accuracy':print(r,flush=True)
if __name__=='__main__':main()
