from common import *
import torch,math
def scalar(y,p):
 y=[bool(a)for a in y];p=[float(a)for a in p];tp=sum(a and b>=.5 for a,b in zip(y,p));fp=sum(not a and b>=.5 for a,b in zip(y,p));fn=sum(a and b<.5 for a,b in zip(y,p));tn=sum(not a and b<.5 for a,b in zip(y,p));n=len(y)
 return dict(TP=tp,FP=fp,FN=fn,TN=tn,N=n,positive_queries=tp+fn,negative_queries=fp+tn,precision=tp/(tp+fp)if tp+fp else None,recall=tp/(tp+fn)if tp+fn else None,balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,F1=2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,NLL=sum(-math.log(max(1e-7,min(1-1e-7,b)))if a else-math.log1p(-max(1e-7,min(1-1e-7,b)))for a,b in zip(y,p))/n if n else None,false_positive_rate=fp/(fp+tn)if fp+tn else None)
def main():
 guard('verify');torch.set_num_threads(4);ev=read(E/'results/evaluation.json');rows=read(B/'data/records.json');count=0;boots=0;draws=np.load(B/'results/bootstrap_draws.npz');alias={1101,1107,1113,1119,1510,1516,1522,1528}
 def include(r,i,k,g):
  return k=='all'or k in ['color','camera']and r[k]==g or k=='color_camera'and r['color']+'/'+r['camera']==g or k=='root'and r['root_id']==g or k=='alias'and i in alias or k=='difficulty'and (g=='target_only'and r['color']==r['target_color']or g=='non_target'and r['color']!=r['target_color']or g=='target_closed_supported_negatives'and r['color']==r['target_color']and r['anchor']=='closed_on_support'and not r['y'])
 for m in ev['metrics']:
  z=np.load(E/f'results/predictions_{m["arm"]}_seed{m["seed"]}_{m["split"]}.npz');sel=[j for j,i in enumerate(z['record_indices'])if include(rows[i],int(i),m['group_type'],m['group'])];v=scalar([rows[z['record_indices'][j]]['y']for j in sel],[z['probability'][j]for j in sel])
  for k,x in v.items():assert x is None and m[k]is None or x is not None and abs(x-m[k])<1e-10,(k,x,m[k])
  count+=1
 for seed in [0,1]:
  for arm in ARMS:
   z=np.load(E/f'results/predictions_{arm}_seed{seed}_dev.npz');ids=z['record_indices'];pr=z['probability'];rootids=draws['root_ids']
   for scope in ['all','target_only']:
    rootvalues=[]
    for root in rootids:
     jj=[j for j,i in enumerate(ids)if rows[i]['root_id']==root and(scope=='all'or rows[i]['color']==rows[i]['target_color'])];v=scalar([rows[ids[j]]['y']for j in jj],[pr[j]for j in jj]);rootvalues.append([v['TP'],v['FP'],v['FN'],v['TN'],v['NLL']*v['N'],v['N']])
    for draw in [0,1,2,5,17,83,199,799,1499,1999]:
     jj=[j for root,n in zip(rootids,draws['root_multiplicities'][draw])for _ in range(int(n))for j,i in enumerate(ids)if rows[i]['root_id']==root and(scope=='all'or rows[i]['color']==rows[i]['target_color'])];v=scalar([rows[ids[j]]['y']for j in jj],[pr[j]for j in jj]);a=draws['root_multiplicities'][draw]@np.array(rootvalues);assert np.array_equal(a[:4],[v[k]for k in ['TP','FP','FN','TN']]);assert abs(a[4]/a[5]-v['NLL'])<1e-10;boots+=1
 norm=np.load(E/'configs/train_normalization.npz');arr=np.load(B/'data/features/train.npz');assert np.array_equal(arr['x_pair'].mean(0,dtype=np.float64).astype(np.float32),norm['pair_mean']);assert np.array_equal(np.maximum(arr['x_pair'].std(0,dtype=np.float64),.05).astype(np.float32),norm['pair_std']);x=(arr['x_pair']-norm['pair_mean'])/norm['pair_std'];assert not masked(x,'geometry_masked_RGB')[:,:256].any();assert not masked(x,'RGB_masked_geometry')[:,256:].any();assert np.array_equal(masked(x,'full'),x)
 drift=[];fits=read(E/'results/training.json')['fits'];assert len(fits)==6 and all(f['updates']==600 and f['parameters']==8680 for f in fits)
 for seed in [0,1]:
  initial=torch.load(E/f'models/initial_pair_seed{seed}.pt',map_location='cpu',weights_only=True)['state'];assert len({f['initial_tensor_hash']for f in fits if f['seed']==seed})==1;assert len({f['schedule_sha256']for f in fits if f['seed']==seed})==1
  for arm in ARMS:
   state=torch.load(E/f'models/final_{arm}_seed{seed}.pt',map_location='cpu',weights_only=True)['state'];assert {k:tuple(v.shape)for k,v in state.items()}=={k:tuple(v.shape)for k,v in initial.items()}
   if arm=='full':assert tensor_hash(state)==tensor_hash(torch.load(B/f'models/final_pair_seed{seed}.pt',map_location='cpu',weights_only=True)['state'])
   else:
    sl=slice(0,256)if arm=='geometry_masked_RGB'else slice(256,261);expected=initial['0.weight'][:,sl].numpy().copy()
    for _ in range(600):expected*=np.float32(1-.001*.001)
    delta=float(np.max(np.abs(expected-state['0.weight'][:,sl].numpy())));assert delta<1e-6;drift.append({'arm':arm,'seed':seed,'excluded_columns_max_deviation_from_weight_decay_only':delta})
 for p in read(E/'results/prediction.json')['passes']:
  if p['arm']=='full':assert p['full_original_pair_predictions_bitwise_equal']
 save(E/'results/independent_verification.json',{'utc':now(),'metric_rows_independently_recomputed':count,'expanded_root_bootstrap_checks':boots,'all_normalizations_identical_train_only':True,'six_runs_same_shapes_initialization_per_seed_and_schedule600':True,'excluded_column_gradient_check':drift,'full_model_original_pair_bitwise_equivalent_weights_and_predictions':True,'new_training_prediction_in_verification':0});print('VERIFIED',count,boots,drift)
if __name__=='__main__':main()
