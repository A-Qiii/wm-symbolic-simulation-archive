"""Frozen inference on existing AB/C matched conditions only. No training."""
from audit import *
import torch,fcntl
sys.path.insert(0,str(OLD/'source'))
from features import Encoder
from representation import transform
from train_queries import load_query,normalize
from model import probs
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);cases=read(E/'results/extra_matched_place.json');assert all(r['environment']in'ABC'for r in cases);enc=Encoder();norm=dict(np.load(OLD/'data/public/normalization.npz'));fs=[]
 for r in cases:fs.append(enc.encode([r['path']]*8)[0])
 f={k:np.stack([x[k]for x in fs])for k in fs[0]};o,s=transform(f,norm);out=[];zs={}
 for kind in['generic','ae','interaction']:
  for seed in[0,1]:
   m,q,stats=load_query(kind,seed);m.eval();q.eval()
   with torch.no_grad():z=m.encode(torch.tensor(o,device='cuda'),torch.tensor(s,device='cuda'));p=probs(q(normalize(z,stats))).cpu().numpy()
   zs[f'{kind}_{seed}']=z.cpu().numpy()
   for i,r in enumerate(cases):out.append({'case_id':r['id'],'root_id':r['root_id'],'split':r['split'],'condition':r['condition'],'arm':kind,'seed':seed,'true_drawer_q':r['before_drawer_q'],'true_held':r['before_held'],'current_probabilities':p[i].tolist(),'drawer_correct':int(np.argmax(p[i,6:]))==(0 if r['before_drawer_q']<=.02 else 1 if r['before_drawer_q']>=.18 else 2),'holding_all_correct':bool(np.array_equal(p[i,:3]>=.5,np.array(r['before_held'])))})
 distances=[]
 for root in sorted({r['root_id']for r in cases}):
  op=next(i for i,r in enumerate(cases)if r['root_id']==root and r['condition']=='held_open');cl=next(i for i,r in enumerate(cases)if r['root_id']==root and r['condition']=='held_closed')
  with np.load(cases[op]['path'])as a,np.load(cases[cl]['path'])as b:rgbdiff={cam:float(np.abs(a[cam].astype(float)-b[cam].astype(float)).mean()/255)for cam in['rgb_static','rgb_gripper']}
  distances.append({'root_id':root,'split':cases[op]['split'],'public_RGB_MAE_0to1':rgbdiff,'normalized_scene_input_RMS':float(np.square(s[op]-s[cl]).mean()**.5),'normalized_object_input_RMS':float(np.square(o[op]-o[cl]).mean()**.5),'frozen_latent_RMS':{k:float(np.square(v[op]-v[cl]).mean()**.5)for k,v in zs.items()}})
 save(E/'results/frozen_AB_C_probe.json',out);save(E/'results/matched_input_distances.json',distances);save(E/'results/probe_audit.json',{'training_updates':0,'new_physical_calls':0,'D_inputs':0,'observations':len(cases),'models':6,'pretrained_weights_reused_read_only':True,'prior_cache_or_normalization_modified':False,'fixed_DINO_batch8':True});print('FROZEN_PROBE_COMPLETE',len(out),flush=True)
if __name__=='__main__':main()
