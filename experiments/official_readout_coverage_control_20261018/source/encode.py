from coverage_common import *
import sys,torch,time,fcntl,shutil,subprocess
sys.path.insert(0,str(REP/'source'))
from features import Encoder
from representation import transform
from runtime import load
def main():
 assert not(E/'data/public/encoded.npz').exists();cfg=read(E/'configs/protocol.json');frozen=read(E/'configs/pre_encoding_frozen.json')
 for p,h in frozen['source_refs'].items():assert sha(p)==h,p
 rows=read(E/'data/public/states.json');parity=read(E/'data/public/parity_states.json');assert len(rows)==240
 for r in rows:assert sha(r['path'])==r['sha256']
 shutil.copy2(__file__,E/'source/encode.py');save(E/'configs/encoding_frozen.json',{'utc':now(),'source_sha256':sha(__file__),'batch_size':8,'parity':'existing8 A/B images; objects,scene,z max absolute difference<=1e-4; stop on mismatch','canonical_order':COLORS,'no_label_payload_access':True})
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(E/'data/private')or '/_private/'in str(p)or p.is_relative_to(EXP):raise RuntimeError('Encoder cannot access labels, physics or original experiment files')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);start=time.perf_counter();torch.cuda.reset_peak_memory_stats();enc=Encoder();fs=[]
 monitorfile=(E/'logs/encoding_gpu.csv').open('w');monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used','--format=csv,noheader,nounits','-lms','250'],stdout=monitorfile,stderr=subprocess.DEVNULL)
 try:
  # All encodings use the unchanged previous batch8 interface.
  pf=enc.encode([r['path']for r in parity])
  for i in range(0,len(rows),8):
   batch=rows[i:i+8];ff=enc.encode([r['path']for r in batch]);fs.extend(ff)
   for r,f in zip(batch,ff):np.savez_compressed(E/'data/public/features'/f"{r['index']:04d}.npz",**f)
   if(i+8)%80==0:print('OFFICIAL_ENCODED',i+8,'/240',flush=True)
  del enc;torch.cuda.empty_cache();norm=dict(np.load(REP/'data/public/normalization.npz',allow_pickle=False));model,query,stats=load(0,'canonical_matched');before=tensor_hash(model.state_dict())
  def encode_features(ff):
   arrays={k:np.stack([f[k]for f in ff]).astype(np.float32)for k in ff[0]};objects,scene=transform(arrays,norm);zs=[]
   with torch.inference_mode():
    for i in range(0,len(objects),128):zs.append(model.encode(torch.tensor(objects[i:i+128],device='cuda'),torch.tensor(scene[i:i+128],device='cuda')).cpu().numpy())
   return {'objects':objects,'scene':scene,'z':np.concatenate(zs)}
  check=encode_features(pf);expected=np.load(E/'data/public/parity_expected.npz',allow_pickle=False);diff={k:float(np.max(np.abs(check[k]-expected[k])))for k in check};save(E/'results/encoding_parity.json',{'differences':diff,'threshold':1e-4,'passed':all(v<=1e-4 for v in diff.values()),'states':[r['id']for r in parity]});assert all(v<=1e-4 for v in diff.values()),diff
  data=encode_features(fs);assert data['z'].shape==(240,4,64);assert tensor_hash(model.state_dict())==before;np.savez_compressed(E/'data/public/encoded.npz',**data,state_ids=np.array([r['state_id']for r in rows]),root_ids=np.array([r['root_id']for r in rows]))
  torch.cuda.synchronize();save(E/'results/encoding.json',{'utc':now(),'states':240,'canonical_order':COLORS,'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'model_state_before_after_hash':before,'representation_updates':0,'T_forward_calls':0,'private_payload_reads':0,'DINO_batch_size':8,'parity_passed':True,'public_observation_schema':['rgb_static','rgb_gripper','robot_obs']})
 finally:monitor.terminate();monitor.wait(timeout=10);monitorfile.close()
 print('FROZEN_ENCODING_COMPLETE',diff,flush=True)
if __name__=='__main__':main()
