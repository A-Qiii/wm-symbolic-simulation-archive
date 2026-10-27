"""Freeze DINO; feature extractor reads only the three allowed NPZ keys."""
from paths import GPU_LOCK
from paths import ROOT as B, RUNTIME as P
import os,json,fcntl,time,hashlib,subprocess,sys
import numpy as np
from pathlib import Path
def save(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2)+'\n')
from rgb_front import RGBFront

def main():
 version=os.environ.get('COLLECTION_VERSION','v1');mode=os.environ.get('COLLECT_MODE','fit');root=B/'data'/version
 assert(root/f'{mode}_complete.json').exists()
 subprocess.run([sys.executable,str(B/'src/audit_data.py'),mode],check=True,cwd=B)
 if mode=='test':assert(B/'configs/all_models_frozen.json').exists()
 out=B/'cache'/version/mode;out.mkdir(parents=True,exist_ok=True);net=RGBFront();rows=[];values=[];t0=time.time()
 try:
  for exp in ['state','transition']:
   for path in sorted((root/exp).glob('*/episode.json')):
    e=json.load(open(path))
    if (e['split']=='test')!=(mode=='test'):continue
    if mode=='fit':assert e['physical_valid']
    for boundary,(paths,state)in enumerate(zip(e['public_windows'],e['states'])):
     key=e['id']+f'__b{boundary}';dest=out/(key+'.npz')
     if dest.exists():features=np.load(dest)['features']
     else:
      features,_=net.extract(paths);np.savez_compressed(dest,features=features)
     row={'experiment':exp,'episode_id':e['id'],'group':e['group'],'split':e['split'],'boundary':boundary,'xyz':[state['objects'][n]['position']for n in ['large','small']],'shape':[[state['objects'][n]['aabb'][1][j]-state['objects'][n]['aabb'][0][j]for j in range(3)]for n in ['large','small']],'region':[state['objects'][n]['region']for n in ['large','small']],'held':[state['objects'][n]['held']for n in ['large','small']],'feature_file':str(dest)}
     values.append(features);rows.append(row)
     if len(rows)%100==0:save(B/'STATUS.json',{'stage':'cache_'+mode,'states':len(rows),'seconds':time.time()-t0});print('CACHE',len(rows),flush=True)
  np.savez_compressed(out/'all_features.npz',features=np.stack(values));save(out/'state_labels.json',rows);save(out/'complete.json',{'states':len(rows),'width':values[0].shape[-1],'input_keys':['rgb_static','rgb_gripper','robot_obs'],'extractor_sha256':hashlib.sha256((B/'src/rgb_front.py').read_bytes()).hexdigest(),'task_weights_used':False});print('CACHE_COMPLETE',mode,len(rows),flush=True)
 finally:net.close()

if __name__=='__main__':
 lock=GPU_LOCK.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:main()
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
