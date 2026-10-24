from common import *
import csv
CAT=['red_location','blue_location','pink_location','drawer','slider']
BIN=['light','LED']
SLICES=[slice(0,8),slice(8,16),slice(16,24),slice(24,27),slice(27,30)]

def mean_ci(values,seed=261019900):
 v=np.asarray(values,float);v=v[np.isfinite(v)]
 if not len(v):return None,None,None
 rng=np.random.default_rng(seed);boot=v[rng.integers(len(v),size=(2000,len(v)))].mean(1)
 return float(v.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))

def root_values(root,v,valid):
 return {str(r):float(v[(root==r)&valid].mean())for r in sorted(set(root))if ((root==r)&valid).any()}

def rates(c):
 tp,fp,fn,tn=np.moveaxis(np.asarray(c,float),-1,0)
 def div(a,b):return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b>0)
 precision=div(tp,tp+fp);recall=div(tp,tp+fn);specificity=div(tn,tn+fp)
 return {'precision':precision,'recall':recall,'balanced_accuracy':(recall+specificity)/2,'FPR':div(fp,fp+tn)}

def score(a,meta):
 rows=[];rootresults=[];n=len(a['root_id']);root=a['root_id'];probs=a['prob'];cat=a['cat'];cv=a['cat_valid'].astype(bool);bv=a['binary_valid'].astype(bool)
 assert probs.shape==(n,32) and np.isfinite(probs).all() and np.all((probs>=0)&(probs<=1))
 predictions=np.stack([probs[:,s].argmax(-1)for s in SLICES],1);bn=probs[:,30:]>=.5
 for s in SLICES:assert np.allclose(probs[:,s].sum(-1),1,atol=1e-5)
 nll=np.stack([-np.log(np.clip(probs[:,s][np.arange(n),cat[:,j]],1e-9,1))for j,s in enumerate(SLICES)],1)
 facts=[]
 for j,name in enumerate(CAT):facts.append((name,(predictions[:,j]==cat[:,j]).astype(float),nll[:,j],cv[:,j],cat[:,j]!=a['base_cat'][:,j],a['base_valid'][:,j].astype(bool),None))
 for j,name in enumerate(BIN):
  y=a['binary'][:,j];p=probs[:,30+j];loss=-(y*np.log(np.clip(p,1e-9,1))+(1-y)*np.log(np.clip(1-p,1e-9,1)))
  facts.append((name,(bn[:,j]==y).astype(float),loss,bv[:,j],y!=a['base_binary'][:,j],a['base_binary_valid'][:,j].astype(bool),(bn[:,j],y.astype(bool))))
 for j,color in enumerate(['red','blue','pink']):
  for label,k in [('Holding',6),('In',3)]:
   y=cat[:,j]==k;p=probs[:,8*j+k];pred=p>=.5;loss=-(y*np.log(np.clip(p,1e-9,1))+(1-y)*np.log(np.clip(1-p,1e-9,1)))
   facts.append((label+'_'+color,(pred==y).astype(float),loss,cv[:,j],y!=(a['base_cat'][:,j]==k),a['base_valid'][:,j].astype(bool),(pred,y)))
 y=cat[:,3]==2;p=probs[:,26];pred=p>=.5;loss=-(y*np.log(np.clip(p,1e-9,1))+(1-y)*np.log(np.clip(1-p,1e-9,1)))
 facts.append(('drawer_open',(pred==y).astype(float),loss,cv[:,3],y!=(a['base_cat'][:,3]==2),a['base_valid'][:,3].astype(bool),(pred,y)))
 facts.extend([('primary_3objects_drawer',(predictions[:,:4]==cat[:,:4]).mean(1),nll[:,:4].mean(1),cv[:,:4].all(1),None,None,None),('all5_exact',(predictions==cat).all(1).astype(float),None,cv.all(1),None,None,None),('all7_exact',((predictions==cat).all(1)&(bn==a['binary']).all(1)).astype(float),None,cv.all(1)&bv.all(1),None,None,None)])
 for j in range(4):
  p=a['event_prob'][:,j];y=a['event_target'][:,j];loss=-(y*np.log(np.clip(p,1e-9,1))+(1-y)*np.log(np.clip(1-p,1e-9,1)))
  facts.append((['executor_success','event_red','event_blue','event_pink'][j],((p>=.5)==y).astype(float),loss,a['event_valid'][:,j].astype(bool)&np.isfinite(p),None,None,None))
 roles=['all']+sorted(set(a['role']))if meta['split']=='dev' else ['all']
 for role in roles:
  for mode in sorted(set(a['mode'])):
   base=(a['mode']==mode)&((a['role']==role)if role!='all' else np.ones(n,bool))
   strata={'all':np.ones(n,bool)}
   if mode not in ['probe','current']:
    strata.update(nonzero_control=a['controls']>0,zero_control=a['controls']==0,executor_success=a['status']=='success',executed_failure=a['status']=='executed_failure')
    if mode.startswith(('free','root_persistence')):strata['all_prefix_actuated']=a['all_prefix_actuated']
    for skill in sorted(set(a['skill'][base])):strata['skill:'+skill]=a['skill']==skill
   for stratum,sm in strata.items():
    sel=base&sm
    for fact,acc,loss,valid,changed,basevalid,binary in facts:
     changes=['all','changed','unchanged']if changed is not None and mode not in ['current','probe']and stratum=='all' else ['all']
     for change in changes:
      vv=sel&valid
      if change!='all':vv=vv&basevalid&((changed)if change=='changed' else ~changed)
      ident={**meta,'role':role,'mode':mode,'stratum':stratum,'change':change,'fact':fact};count=int(vv.sum());nr=len(set(root[vv]))
      for metric,vals in [('accuracy',acc),('NLL',loss)]:
       if vals is None:continue
       rv=root_values(root,vals,vv);mean,lo,hi=mean_ci(list(rv.values()))
       rows.append({**ident,'metric':metric,'n':count,'roots':nr,'value':mean,'lo95':lo,'hi95':hi,'micro_value':float(vals[vv].mean())if count else None,'estimand':'root_macro'})
       for r,v in rv.items():rootresults.append({**ident,'metric':metric,'root_id':r,'value':v})
      if binary is not None:
       pred,y=binary;groups=sorted(set(root[vv]));cs=np.array([[int(((root==r)&vv&mask).sum())for mask in [pred&y,pred&~y,~pred&y,~pred&~y]]for r in groups],float).reshape(-1,4)
       total=cs.sum(0);point=rates(total);rng=np.random.default_rng(261019900)
       boots=rates(cs[rng.integers(len(cs),size=(2000,len(cs)))].sum(1))if len(cs)else{k:np.array([])for k in point}
       for metric,v in point.items():
        boot=boots[metric];boot=boot[np.isfinite(boot)];vvv=float(v)if np.isfinite(v)else None
        rows.append({**ident,'metric':metric,'n':count,'roots':nr,'value':vvv,'lo95':float(np.quantile(boot,.025))if len(boot)else None,'hi95':float(np.quantile(boot,.975))if len(boot)else None,'micro_value':vvv,'estimand':'pooled_ratio_root_cluster_bootstrap','TP':int(total[0]),'FP':int(total[1]),'FN':int(total[2]),'TN':int(total[3]),'defined_bootstraps':len(boot)})
 return rows,rootresults

def writecsv(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
 guard('score');rows=[];rootrows=[]
 for seed in [0,1]:
  for arm in ARMS:
   for split in ['train','dev']:
    for prefix in ['predictions','probe_predictions']:
     a=dict(np.load(E/f'data/{prefix}_{arm}_seed{seed}_{split}.npz'));rr,roots=score(a,{'arm':arm,'seed':seed,'split':split});rows+=rr;rootrows+=roots
    print('SCORED',arm,seed,split,flush=True)
 writecsv(R/'all_metrics.csv',rows);writecsv(R/'root_metrics.csv',rootrows)
 # Identical root resampling for arm differences. Seeds are never pooled.
 keys=['seed','split','role','mode','stratum','change','fact','metric'];groups={}
 for r in rootrows:
  key=tuple(r[k]for k in keys);groups.setdefault(key,{}).setdefault(r['arm'],{})[r['root_id']]=r['value']
 paired=[]
 for key,g in groups.items():
  if not all(a in g for a in ARMS):continue
  assert set(g[ARMS[0]])==set(g[ARMS[1]])
  rv=[g['joint'][r]-g['route_current_only'][r]for r in sorted(g['joint'])];v,lo,hi=mean_ci(rv)
  paired.append({**dict(zip(keys,key)),'roots':len(rv),'joint_minus_control':v,'lo95':lo,'hi95':hi})
 writecsv(R/'paired_differences.csv',paired)
 primary=[r for r in rows if r['role']=='all'and r['stratum']=='all'and r['change']=='all'and r['fact']=='primary_3objects_drawer'and r['metric']=='NLL']
 primarydiff=[r for r in paired if r['role']=='all'and r['stratum']=='all'and r['change']=='all'and r['fact']=='primary_3objects_drawer'and r['metric']=='NLL']
 gap=[]
 for r in primary:
  if r['split']!='dev':continue
  t=next(v for v in primary if v['split']=='train'and all(v[k]==r[k]for k in ['arm','seed','mode']))
  gap.append({'arm':r['arm'],'seed':r['seed'],'mode':r['mode'],'train_NLL':t['value'],'seen_C_NLL':r['value'],'C_minus_train':r['value']-t['value']})
 save(E/'results/scores.json',{'utc':now(),'primary':primary,'paired_primary':primarydiff,'train_dev_gaps':gap,'metric_rows':len(rows),'paired_rows':len(paired),'root_rows':len(rootrows),'NLL_clip_floor':1e-9,'bootstrap_replicates':2000,'bootstrap_unit':'whole root, preserve branches','support_hardnegative_denominator':0,'support_FPR':None,'D_samples':0})
 print('SCORE_COMPLETE',flush=True)

if __name__=='__main__':main()
