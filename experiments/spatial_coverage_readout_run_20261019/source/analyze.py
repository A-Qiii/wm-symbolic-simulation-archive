from common import *
import csv,collections
METRICS=['table_macro_recall','table_left_recall','table_middle_recall','table_right_recall','target_recall','exact4','Holding_recall','Holding_FPR','holding_identity_positive','InDrawer_recall','InDrawer_FPR','drawer_accuracy']
def csvwrite(p,rows):
 if not rows:return
 with Path(p).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def counts(a,mask,rm):
 y=a['cat'][mask];valid=a['valid'][mask];p=a['prob'][mask];g=np.stack([p[:,j*8:j*8+8].argmax(1)for j in range(3)]+[p[:,24:].argmax(1)],1);n=len(y);v4=valid.all(1);out={'n':n,'valid4':int(v4.sum()),'exact4':int(((g==y).all(1)&v4).sum()),'drawer_n':int(valid[:,3].sum()),'drawer_correct':int(((g[:,3]==y[:,3])&valid[:,3]).sum()),'invalid_fields':int((~valid).sum())}
 for k in range(3):
  yes=(y[:,:3]==k)&valid[:,:3];out[f'table{k}_n']=int(yes.sum());out[f'table{k}_correct']=int((yes&(g[:,:3]==k)).sum())
 for name,k in [('Holding',6),('InDrawer',3)]:
  gt=y[:,:3]==k;pr=p[:,[k,k+8,k+16]]>=.5;vv=valid[:,:3]
  for key,q in [('TP',gt&pr),('FN',gt&~pr),('FP',~gt&pr),('TN',~gt&~pr)]:out[name+'_'+key]=int((q&vv).sum())
  if name=='Holding':
   positive=gt.any(1)&vv.all(1);out['holding_identity_n']=int(positive.sum());out['holding_identity_correct']=int((positive&(gt==pr).all(1)).sum())
 out['target_n']=0;out['target_correct']=0
 for j,index in enumerate(np.where(mask)[0]):
  r=rm[a['root_id'][index]]
  if 'gap_target_color'in r and a['step'][index]==0 and str(a['episode_id'][index]).endswith('_b0'):
   c=COLORS.index(r['gap_target_color'])
   if valid[j,c]:out['target_n']+=1;out['target_correct']+=int(g[j,c]==y[j,c])
 return out
def ratio(a,b):
 a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float));return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)
def metrics(c,support=(0,1,2)):
 rr=[ratio(c[f'table{k}_correct'],c[f'table{k}_n'])for k in range(3)];d={'table_macro_recall':np.mean(np.stack([rr[k]for k in support]),axis=0),'target_recall':ratio(c['target_correct'],c['target_n']),'exact4':ratio(c['exact4'],c['valid4']),'drawer_accuracy':ratio(c['drawer_correct'],c['drawer_n']),'holding_identity_positive':ratio(c['holding_identity_correct'],c['holding_identity_n'])}
 for name,k in [('left',0),('middle',1),('right',2)]:d['table_'+name+'_recall']=rr[k]
 for name in['Holding','InDrawer']:
  d[name+'_recall']=ratio(c[name+'_TP'],c[name+'_TP']+c[name+'_FN']);d[name+'_FPR']=ratio(c[name+'_FP'],c[name+'_FP']+c[name+'_TN'])
 return d
def num(x):return float(x)if np.isfinite(x)else None
def interval(x):
 x=np.asarray(x);x=x[np.isfinite(x)];return [float(np.quantile(x,.025)),float(np.quantile(x,.975)),len(x)]if len(x)else[None,None,0]
def main():
 sys.addaudithook(guard);assert(S/'results/compute_evaluation_complete.json').exists();rm={r['root_id']:r for r in read(S/'data/roots_manifest.json')+read(P/'data/core1200_v1/roots_manifest.json')};summary=[];rawcounts=[];rootscores=[];perexample=[];confusions=[]
 for seed in[0,1]:
  for split in['old_band','gap','externalC']:
   arrays={arm:dict(np.load(S/f'data/predictions_{arm}_seed{seed}_{split}.npz'))for arm in['control','coverage']};a=arrays['control'];b=arrays['coverage']
   for key in['root_id','episode_id','step','cat','valid','availability']:assert np.array_equal(a[key],b[key])
   for arm,aa in arrays.items():
    g=np.stack([aa['prob'][:,j*8:j*8+8].argmax(1)for j in range(3)]+[aa['prob'][:,24:].argmax(1)],1)
    for i in range(len(g)):perexample.append({'seed':seed,'split':split,'arm':arm,'root_id':aa['root_id'][i],'episode_id':aa['episode_id'][i],'boundary':int(aa['step'][i]),'truth':','.join(map(str,aa['cat'][i])),'prediction':','.join(map(str,g[i])),'valid':','.join(map(str,aa['valid'][i].astype(int))),'exact4':bool((g[i]==aa['cat'][i]).all())})
   for cohort in['h0','all_boundaries']:
    mask=(a['step']==0)&np.char.endswith(a['episode_id'],'_b0')if cohort=='h0'else np.ones(len(a['step']),bool);roots=sorted(set(a['root_id'][mask]));rng=np.random.default_rng(261019990+seed);ii=rng.integers(len(roots),size=(2000,len(roots)));w=np.zeros((2000,len(roots)));np.add.at(w,(np.repeat(np.arange(2000),len(roots)),ii.ravel()),1);values={};boots={}
    for arm,aa in arrays.items():
     countsper=[counts(aa,mask&(aa['root_id']==r),rm)for r in roots];keys=list(countsper[0]);mat=np.array([[c[k]for k in keys]for c in countsper]);total=dict(zip(keys,mat.sum(0)));support=[k for k in range(3)if total[f'table{k}_n']>0];values[arm]=metrics(total,support);boots[arm]=metrics(dict(zip(keys,(w@mat).T)),support);ident={'seed':seed,'split':split,'cohort':cohort,'arm':arm};rawcounts.append({**ident,**{k:int(v)for k,v in total.items()}})
     for r,c in zip(roots,countsper):rootscores.append({**ident,'root_id':r,**c})
     g=np.stack([aa['prob'][:,j*8:j*8+8].argmax(1)for j in range(3)],1)
     for j,c in enumerate(COLORS):
      mm=mask&aa['valid'][:,j];conf=np.zeros((8,8),int);np.add.at(conf,(aa['cat'][mm,j],g[mm,j]),1)
      for t in range(8):
       for pr in range(8):confusions.append({**ident,'color':c,'truth':t,'prediction':pr,'count':int(conf[t,pr])})
    for metric in METRICS:
     delta=values['coverage'][metric]-values['control'][metric];lo,hi,n=interval(boots['coverage'][metric]-boots['control'][metric]);summary.append({'seed':seed,'split':split,'cohort':cohort,'roots':len(roots),'metric':metric,'control':num(values['control'][metric]),'coverage':num(values['coverage'][metric]),'delta':num(delta),'lo95':lo,'hi95':hi,'valid_bootstrap':n})
 csvwrite(S/'results/paired_metrics.csv',summary);csvwrite(S/'results/counts.csv',rawcounts);csvwrite(S/'results/root_statistics.csv',rootscores);csvwrite(S/'results/per_example.csv',perexample);csvwrite(S/'results/confusions.csv',confusions)
 # Supervision-prior audit: paired root/branch/boundary counts, no balancing or post-hoc weighting.
 old=dict(np.load(O/'data/features_seed0_train.npz'));new=dict(np.load(S/'data/features_seed0_train.npz'));oldroots={r['root_id']:r for r in read(P/'data/core1200_v1/roots_manifest.json')};newroots={r['root_id']:r for r in read(S/'data/roots_manifest.json')};priors=[];tv=[]
 for field in range(4):
  a=np.bincount(old['cat'][:,field],minlength=8 if field<3 else 3);b=np.bincount(new['cat'][:,field],minlength=len(a));tv.append({'field':COLORS[field]if field<3 else'drawer','total_variation':float(np.abs(a/a.sum()-b/b.sum()).sum()/2),'old':a,'new':b})
 for arm,arr,roots in[('control',old,oldroots),('coverage',new,newroots)]:
  for e in 'AB':
   for kind in['closed_empty','open_empty','open_held']:
    mask=np.array([roots[r]['environment']==e and roots[r]['initial_state']==kind for r in arr['root_id']])
    for j,field in enumerate(COLORS+['drawer']):
     for k in range(8 if j<3 else 3):
      q=mask&(arr['cat'][:,j]==k);priors.append({'arm':arm,'scene':e,'initial_state':kind,'field':field,'class':k,'boundaries':int(q.sum()),'roots':len(set(arr['root_id'][q]))})
 csvwrite(S/'results/training_priors.csv',priors);save(S/'results/prior_audit.json',{'h0_match':all(x['h0_paired_match']for x in read(S/'results/h0_audit.json')if x['h0_paired_match']is not None),'paired_boundary_label_exact':int((old['cat']==new['cat']).all(1).sum()),'boundaries':600,'fields':tv,'interpretation':'Total effect of coverage plus geometric admission x-policy; downstream success/labels are mediators, not separately controlled.'})
 save(S/'results/analysis.json',{'utc':now(),'summary':summary,'bootstrap_replicates':2000,'unit':'root, paired across heads; seeds separate','macro_missing_class_rule':'Fix table classes present in each full reporting cohort (A/B normally3, C normally2); bootstrap draws missing any required class are undefined and valid counts reported','threshold':.5,'new_roots':20,'C_exploratory':True})
 print('ANALYSIS_COMPLETE',len(summary),flush=True)
if __name__=='__main__':main()
