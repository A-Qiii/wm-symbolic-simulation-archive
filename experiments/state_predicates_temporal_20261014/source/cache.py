from pathlib import Path
import json,hashlib
import numpy as np
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_temporal_20261014';C=P/'experiments/state_predicates_causal_windows_20261014';O=P/'experiments/state_predicates_20261014'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not(E/'data/temporal.npz').exists();protocol=read(E/'configs/protocol.json')
 for p,h in protocol['frozen_sha256'].items():assert sha(p)==h,p
 a=read(E/'results/audit.json');by={q['sample_id']:q for q in a['recovered_samples']};meta=read(C/'data/metadata.json');stats=read(O/'data/normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rm=np.array(stats['robot_mean'],np.float32);rs=np.array(stats['robot_std'],np.float32)
 with np.load(C/'data/causal_train_dev.npz')as z:old={k:z[k].copy()for k in z.files}
 arrays={'y':old['y'],'v':old['v'],'splits':old['splits'],'roles':old['roles']};audit=[]
 for k in ['objects','context','robot']:
  arrays['short_history_'+k]=old[k];arrays['current_only_'+k]=np.repeat(old[k][:,-1:],4,axis=1)
 long={k:[]for k in ['objects','context','robot']};enc=FrozenVision(P)
 for i,r in enumerate(meta):
  paths=by[r['sample_id']]['temporal_inputs']['long_history'];reg=VisibleRegistry();f=enc.encode(paths,reg);fp=E/'data/features'/(r['sample_id']+'.npz');np.savez_compressed(fp,**f);tok=((f['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384);geo=f['geometry'].reshape(4,2,64,8);valid=f['valid'].reshape(4,2,64);ids=f['visible_registry_ids'].tolist();ob=np.zeros((4,3,2,390),np.float32)
  for j in range(3):
   uid=reg.colour_ids.get(j);assert uid in ids;slot=ids.index(uid);sl=slice(slot*4,slot*4+4);vv=valid[:,:,slot*4];ob[:,j,:,:384]=tok[:,:,sl].mean(2)*vv[:,:,None];ob[:,j,:,384:389]=geo[:,:,slot*4,:5];ob[:,j,:,389]=geo[:,:,sl,5].mean(2)
  grid=tok[:,:,48:].reshape(4,2,4,4,384).reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,-1);robot=[]
  for pp in paths:
   with np.load(pp)as z:robot.append(z['robot_obs'])
  new={'objects':ob.reshape(4,3,780),'context':grid,'robot':(np.stack(robot).astype(np.float32)-rm)/rs}
  diff={k:float(np.abs(v[-1]-old[k][i,-1]).max())for k,v in new.items()};assert max(diff.values())<1e-5,(r['sample_id'],diff)
  for k,v in new.items():long[k].append(v)
  # Audit labels/IDs only after public encoding; they never enter new[].
  lab=read(r['labels_private'])['frames'][-1];identity={c:reg.colour_ids[j]==lab['objects_audit'][c]['object_uid_evaluator_only']for j,c in enumerate(['red','blue','pink'])};assert all(identity.values())
  audit.append({'sample_id':r['sample_id'],'long_paths':paths,'feature_sha256':sha(fp),'identity':identity,'same_anchor_feature_max_errors':diff})
  if(i+1)%12==0:print('CACHED_LONG',i+1,flush=True)
 for k,v in long.items():arrays['long_history_'+k]=np.stack(v)
 np.savez_compressed(E/'data/temporal.npz',**arrays);save(E/'data/metadata.json',meta);save(E/'results/cache_audit.json',{'windows':54,'identity_checks':162,'same_anchor_max_error':max(max(r['same_anchor_feature_max_errors'].values())for r in audit),'only_public_inputs':True,'current_frame_repetition_explicit':True,'short_history_arrays_byte_equal':True,'details':audit});print('TEMPORAL_CACHE_COMPLETE')
if __name__=='__main__':main()
