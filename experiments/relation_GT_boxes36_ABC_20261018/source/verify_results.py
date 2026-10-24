from common import *
import math
def calc(y,p):
 tp=sum(bool(a)and b>=.5 for a,b in zip(y,p));fp=sum(not a and b>=.5 for a,b in zip(y,p));fn=sum(bool(a)and b<.5 for a,b in zip(y,p));tn=sum(not a and b<.5 for a,b in zip(y,p));n=len(y)
 return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'N':n,'precision':tp/(tp+fp)if tp+fp else None,'recall':tp/(tp+fn)if tp+fn else None,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,'F1':2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,'NLL':sum(-math.log(max(1e-7,min(1-1e-7,float(b))))if a else-math.log1p(-max(1e-7,min(1-1e-7,float(b))))for a,b in zip(y,p))/n if n else None}
def main():
 install_guard([]);rows=read(E/'data/records.json');result=read(E/'results/evaluation.json');count=0
 for metric in result['metrics']:
  z=np.load(E/f'results/predictions_{metric["arm"]}_seed{metric["seed"]}_{metric["split"]}.npz');ii=z['record_indices'];pred=z['probability'];kind,group=metric['group_type'],metric['group'];sel=[j for j,i in enumerate(ii)if kind=='all'or(kind=='color'and rows[i]['color']==group)or(kind=='camera'and rows[i]['camera']==group)or(kind=='root'and rows[i]['root_id']==group)or(kind=='color_camera'and rows[i]['color']+'/'+rows[i]['camera']==group)];v=calc([rows[ii[j]]['y']for j in sel],[pred[j]for j in sel])
  for key,value in v.items():assert value is None and metric[key]is None or value is not None and abs(value-metric[key])<1e-10,(key,value,metric[key])
  count+=1
 boot=np.load(E/'results/bootstrap_draws.npz');weights=boot['root_multiplicities'];roots=boot['root_ids'];bootstrap_checks=0
 for arm in ['object','pair','geometry']:
  for seed in [0,1]:
   z=np.load(E/f'results/predictions_{arm}_seed{seed}_dev.npz');ii=z['record_indices'];pr=z['probability'];rootvectors=[]
   for root in roots:
    jj=[j for j,i in enumerate(ii)if rows[i]['root_id']==root];v=calc([rows[ii[j]]['y']for j in jj],[pr[j]for j in jj]);rootvectors.append([v['TP'],v['FP'],v['FN'],v['TN'],v['NLL']*v['N'],v['N']])
   for draw in [0,1,2,5,17,83,199,799,1499,1999]:
    jj=[j for root,n in zip(roots,weights[draw])for _ in range(int(n))for j,i in enumerate(ii)if rows[i]['root_id']==root];v=calc([rows[ii[j]]['y']for j in jj],[pr[j]for j in jj]);weighted=weights[draw]@np.array(rootvectors)
    for j,key in enumerate(['TP','FP','FN','TN']):assert weighted[j]==v[key]
    assert abs(weighted[4]/weighted[5]-v['NLL'])<1e-10;bootstrap_checks+=1
 manifest=read(E/'data/model_inputs.json');allowed={'record_index','split','path','key','width','height','RGB_sha256','object_bbox','gripper_bbox'};assert all(set(x)==allowed for x in manifest);index={r['record_index']:r for r in manifest};geo_checks=0
 for split in ['train','dev']:
  z=np.load(E/f'data/features/{split}.npz')
  for j,i in enumerate(z['record_indices']):
   r=index[int(i)];b=r['object_bbox'];g=r['gripper_bbox'];bw,bh=b[2]-b[0],b[3]-b[1];gw,gh=g[2]-g[0],g[3]-g[1];inter=max(0,min(b[2],g[2])-max(b[0],g[0]))*max(0,min(b[3],g[3])-max(b[1],g[1]));geo=np.array([((b[0]+b[2])-(g[0]+g[2]))/(2*r['width']),((b[1]+b[3])-(g[1]+g[3]))/(2*r['height']),math.log(bw/gw),math.log(bh/gh),inter/(bw*bh+gw*gh-inter)],np.float32);assert np.array_equal(geo,z['x_geometry'][j]);geo_checks+=1
  assert np.array_equal(z['x_pair'][:,:128],z['x_object'])and np.array_equal(z['x_pair'][:,-5:],z['x_geometry'])
 train=np.load(E/'data/features/train.npz');norm=np.load(E/'configs/train_normalization.npz');mu=train['x_pair'].mean(0,dtype=np.float64).astype(np.float32);sd=np.maximum(train['x_pair'].std(0,dtype=np.float64),.05).astype(np.float32);assert np.array_equal(mu,norm['pair_mean'])and np.array_equal(sd,norm['pair_std']);targets=read(E/'data/supervision/train_targets.json');fits=read(E/'results/training.json')['fits'];assert len(fits)==6 and all(x['updates']==600 for x in fits)
 for seed in [0,1]:
  assert len({x['shared_schedule_sha256']for x in fits if x['seed']==seed})==1;sc=np.load(E/f'configs/schedule_seed{seed}.npz');assert sc['indices'].shape==(600,32);assert np.array_equal(sc['record_indices'],train['record_indices'][sc['indices']]);assert all(rows[int(i)]['split']=='train'for i in sc['record_indices'].flat)
 save(E/'results/independent_verification.json',{'utc':now(),'metric_rows_recomputed':count,'expanded_root_bootstrap_checks':bootstrap_checks,'geometry_feature_rows_recomputed':geo_checks,'normalization_train_only_recomputed':True,'six_fits_exact600':True,'identical_schedules_within_each_seed':True,'input_manifest_has_only_RGB_box_metadata':True,'additional_training_or_prediction_passes':0});print('VERIFIED',count,bootstrap_checks,geo_checks,flush=True)
if __name__=='__main__':main()
