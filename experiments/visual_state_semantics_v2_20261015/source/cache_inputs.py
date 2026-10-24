from common import *
import sys,fcntl,importlib.util

def main():
 assert read(E/'results/data_audit.json')['infrastructure_passed'];assert not(E/'data/inputs.npz').exists();sys.path.insert(0,str(RGB/'source'));from cache import RGBEncoder
 spec=importlib.util.spec_from_file_location('prior_normalization',OLD/'source/common.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);enc=RGBEncoder();new=read(E/'data/windows.json');xs=[];ys=[];vs=[];metadata=[];(E/'data/features').mkdir(exist_ok=True)
 for i,row in enumerate(new):
  pp=E/'data/features'/(row['window_id']+'.npz')
  if pp.exists():feat=dict(np.load(pp))
  else:feat,_=enc.encode(row['public_frames']);np.savez_compressed(pp,**feat)
  x=mod.normalize(feat);assert all(np.isfinite(v).all()for v in x.values());xs.append(x);ys.append(row['targets_v2_1']['y']);vs.append(row['targets_v2_1']['v']);metadata.append({k:row[k]for k in ['window_id','root_id','episode_id','environment','split','condition','anchor','roles','public_frames']}|{'pool':'new_semantics','raw_feature_path':str(pp)})
  if i==127:save(E/'results/first128_input_audit.json',{'windows':128,'all_public_keys_only':True,'all_finite':True,'train_dev_root_overlap':len({r['root_id']for r in metadata if r['split']=='train'}&{r['root_id']for r in metadata if r['split']=='dev'}),'new_physics_labels_not_features':True})
  if i%50==0:print(json.dumps({'encoded':i+1,'total':len(new)}),flush=True)
 oldmeta=read(OLD/'data/public/windows.json');oldx=dict(np.load(OLD/'data/public/reader_inputs.npz'));oldt=dict(np.load(OLD/'data/_private/reader_targets.npz'))
 for i,row in enumerate(oldmeta):
  xs.append({k:v[i]for k,v in oldx.items()});ys.append(list(oldt['y'][i,:3])+[False]*6);vs.append(list(oldt['v'][i,:3])+[False]*6);metadata.append({k:row[k]for k in ['window_id','root_id','split','public_frames']}|{'pool':'legacy_holding_only','condition':row['condition'],'source':row['source'],'environment':row['root_id'][0]})
 inputs={k:np.stack([r[k]for r in xs])for k in xs[0]};np.savez_compressed(E/'data/inputs.npz',**inputs);np.savez_compressed(E/'data/targets.npz',y=np.array(ys,bool),v=np.array(vs,bool));save(E/'data/metadata.json',metadata);train={r['root_id']for r in metadata if r['split']=='train'};dev={r['root_id']for r in metadata if r['split']=='dev'};assert train.isdisjoint(dev)
 frozen=[E/'data/inputs.npz',E/'data/targets.npz',E/'data/metadata.json',E/'source/model.py',E/'source/train.py',E/'configs/protocol_v2_1.json',E/'source/labels_v2_1.py',CAND/'reader_seed1.pt',CAND/'normalization.json'];save(E/'configs/training_frozen.json',{'created_utc':now(),'new_windows':len(new),'legacy_Holding_only_windows':len(oldmeta),'train_roots':len(train),'dev_roots':len(dev),'no_D':True,'frozen_sha256':{str(p):sha(p)for p in frozen}});print('INPUTS_FROZEN',len(metadata),flush=True)
if __name__=='__main__':main()
