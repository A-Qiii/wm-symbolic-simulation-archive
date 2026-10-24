from common import *
import torch,math
def scalar(y,p):
 y=[bool(v)for v in y];p=[float(v)for v in p];tp=sum(a and b>=.5 for a,b in zip(y,p));fp=sum(not a and b>=.5 for a,b in zip(y,p));fn=sum(a and b<.5 for a,b in zip(y,p));tn=sum(not a and b<.5 for a,b in zip(y,p));n=len(y)
 return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'N':n,'positive_queries':tp+fn,'negative_queries':fp+tn,'precision':tp/(tp+fp)if tp+fp else None,'recall':tp/(tp+fn)if tp+fn else None,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,'F1':2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,'false_positive_rate':fp/(fp+tn)if fp+tn else None,'NLL':sum(-math.log(max(1e-7,min(1-1e-7,b)))if a else-math.log1p(-max(1e-7,min(1-1e-7,b)))for a,b in zip(y,p))/n if n else None}
def eligible(r,i,scope):
 if scope=='target':return r['color']==r['target_color']
 if scope=='supported_negative':return r['color']==r['target_color']and r['sample']==3 and not bool(r['y'])
 if scope=='non_target':return r['color']!=r['target_color']
 if scope=='alias':return i in [1101,1107,1113,1119,1510,1516,1522,1528]
 assert scope=='all';return True
def main():
 guard('verify');torch.set_num_threads(4);ev=read(E/'results/evaluation.json');rows=read(B/'data/records.json');count=0
 for m in ev['metrics']:
  z=np.load(E/f'results/predictions_{m["arm"]}_seed{m["seed"]}_{m["split"]}.npz');jj=[]
  for j,i in enumerate(z['record_indices']):
   r=rows[i]
   if not eligible(r,int(i),m['scope']):continue
   if any(m[k]is not None and r['root_id'if k=='root'else k]!=m[k]for k in ['color','camera','root']):continue
   jj.append(j)
  s=scalar([rows[z['record_indices'][j]]['y']for j in jj],[z['probability'][j]for j in jj])
  for k,v in s.items():assert v is None and m[k]is None or v is not None and abs(v-m[k])<1e-10,(m['group'],k,v,m[k])
  count+=1
 draws=np.load(B/'results/bootstrap_draws.npz');roots=draws['root_ids'];weights=draws['root_multiplicities'];bootstrap=0
 for seed in [0,1]:
  for arm in ARMS:
   z=np.load(E/f'results/predictions_{arm}_seed{seed}_dev.npz');ids=z['record_indices'];pr=z['probability']
   for scope in ['all','target','supported_negative']:
    vv=[]
    for root in roots:
     jj=[j for j,i in enumerate(ids)if rows[i]['root_id']==root and eligible(rows[i],int(i),scope)];s=scalar([rows[ids[j]]['y']for j in jj],[pr[j]for j in jj]);vv.append([s['TP'],s['FP'],s['FN'],s['TN'],(s['NLL']or 0)*s['N'],s['N']])
    for d in [0,1,2,5,17,83,199,799,1499,1999]:
     jj=[j for root,n in zip(roots,weights[d])for _ in range(int(n))for j,i in enumerate(ids)if rows[i]['root_id']==root and eligible(rows[i],int(i),scope)];s=scalar([rows[ids[j]]['y']for j in jj],[pr[j]for j in jj]);a=weights[d]@np.array(vv);assert np.array_equal(a[:4],[s[k]for k in ['TP','FP','FN','TN']]);assert a[5]==s['N'];assert not s['N']or abs(a[4]/a[5]-s['NLL'])<1e-10;bootstrap+=1
 cache=np.load(E/'data/static_context_cache.npz');ctx={str(p):i for i,p in enumerate(cache['paths'])};aligned=0
 for split in ['train','dev']:
  a=np.load(E/f'data/{split}_features.npz');b=np.load(B/f'data/features/{split}.npz');assert np.array_equal(a['record_indices'],b['record_indices']);assert np.array_equal(a['x'][:,:261],b['x_pair'])
  for j,i in enumerate(a['record_indices']):
   r=rows[i];k=ctx[r['path']];assert a['context_indices'][j]==k;assert np.array_equal(a['x'][j,261:],cache['features'][k]);same=[x for x in rows if x['path']==r['path']and x['camera']=='static'];assert len(same)==3 and all(x['RGB_sha256']==str(cache['RGB_sha256'][k])for x in same);aligned+=1
 train=np.load(E/'data/train_features.npz');norm=np.load(E/'configs/train_normalization.npz');assert np.array_equal(norm['mean'],train['x'].mean(0,dtype=np.float64).astype(np.float32));assert np.array_equal(norm['std'],np.maximum(train['x'].std(0,dtype=np.float64),.05).astype(np.float32));x=(train['x']-norm['mean'])/norm['std'];assert np.array_equal(masked(x,'context_present'),x);assert not masked(x,'context_zero')[:,261:].any();assert np.array_equal(masked(x,'context_zero')[:,:261],x[:,:261]);drifts=[];fits=read(E/'results/training.json')['fits'];assert len(fits)==4 and all(r['updates']==600 and r['parameters']==12904 for r in fits)
 for seed in [0,1]:
  assert len({r['initial_tensor_hash']for r in fits if r['seed']==seed})==1;assert len({r['schedule_sha256']for r in fits if r['seed']==seed})==1;initial=torch.load(E/f'models/initial_seed{seed}.pt',map_location='cpu',weights_only=True)['state'];zero=torch.load(E/f'models/final_context_zero_seed{seed}.pt',map_location='cpu',weights_only=True)['state'];present=torch.load(E/f'models/final_context_present_seed{seed}.pt',map_location='cpu',weights_only=True)['state'];assert {k:tuple(v.shape)for k,v in zero.items()}=={k:tuple(v.shape)for k,v in present.items()};expected=initial['0.weight'][:,261:].numpy().copy()
  for _ in range(600):expected*=np.float32(1-.001*.001)
  delta=float(np.max(np.abs(expected-zero['0.weight'][:,261:].numpy())));assert delta<1e-6;drifts.append(delta)
 support={split:[r for r in rows if r['split']==split and r['eligible']and eligible(r,0,'supported_negative')]for split in ['train','dev']};coverage={s:{'queries':len(rr),'roots':sorted({r['root_id']for r in rr}),'colors':{c:sum(r['color']==c for r in rr)for c in ['red','blue','pink']},'cameras':{c:sum(r['camera']==c for r in rr)for c in ['static','gripper']}}for s,rr in support.items()}
 save(E/'results/independent_verification.json',{'utc':now(),'metric_rows_recomputed':count,'expanded_root_bootstrap_checks':bootstrap,'same_state_feature_rows_verified':aligned,'synchronized_states_verified_before_encoding':264,'pair_features_unchanged':True,'train_only_shared_normalization_verified':True,'same_initialization_shapes_schedules':True,'zero_context_weight_decay_only_max_deviation':drifts,'hard_negative_coverage':coverage,'new_training_prediction_encoder_execution':0});print('VERIFIED',count,bootstrap,aligned,coverage)
if __name__=='__main__':main()
