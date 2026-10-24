from common import *
import torch
from local_gap import features
def extract(split,seed):assert(S/f'cache/{split}_local_raw.npz').exists()
def make(split):
 eps=read(O/f'cache/{split}_metadata.json');arr=[];valid=[];start=time.time()
 for ei,e in enumerate(eps):
  xx=[]
  for item in e['features']:
   with np.load(item['public_inputs'][-1])as z:f={k:z[k]for k in z.files}
   x,_=features(f);xx.append(x);valid.append(x)
  x=np.stack(xx);arr.append(np.concatenate([x,np.repeat(x[-1:],7-len(x),axis=0)]))
  if(ei+1)%40==0:print('LOCAL_CACHE',split,ei+1,len(eps),round(time.time()-start,1),flush=True)
 x=np.stack(arr);np.savez_compressed(S/f'cache/{split}_local_raw.npz',context=x);save(S/f'cache/{split}_metadata.json',eps);v=np.stack(valid)
 save(S/f'results/missing_{split}.json',{'boundaries':len(v),'pair_missing':int((v[:,31]==0).sum()),'direction_valid_counts':v[:,[6,7,8,9,10,11,18,19,20,21,22,23]].sum(0).astype(int),'direction_opportunities':len(v),'no_GT_fill':True,'seconds':time.time()-start})
 if split=='train':
  mean=np.zeros(32,np.float32);std=np.ones(32,np.float32)
  for slot in [0,1]:
   for k in range(6):
    j=slot*12+k;mask=v[:,slot*12+6+k]>0
    if mask.any():mean[j]=v[mask,j].mean();std[j]=max(float(v[mask,j].std()),.001)
  mask=v[:,31]>0
  for j in range(27,31):mean[j]=v[mask,j].mean();std[j]=max(float(v[mask,j].std()),.001)
  save(S/'configs/local_normalization.json',{'source':'Only original training boundaries, no padded boundaries','mean':mean,'std':std})
def load(split,seed,device='cuda',arm='coarse'):
 data=dict(np.load(O/f'cache/{split}_seed{seed}.npz'));geo=dict(np.load(OLD/f'cache/{split}_geometry_raw.npz'));norm=read(S/'configs/original_geometry_normalization.json')['values'][f'geometry_seed{seed}'];obj=geo['objects'];n=norm['objects'];oo=(obj-np.array(n['mean'],np.float32))/np.array(n['std'],np.float32)
 for b in [0,8]:oo[...,b:b+8]*=obj[...,b+7:b+8]>0
 if arm=='coarse':
  raw=geo['context'];n=norm['context'];ctx=(raw-np.array(n['mean'],np.float32))/np.array(n['std'],np.float32)
  for b in range(0,32,4):ctx[...,b:b+4]*=raw[...,b+3:b+4]>0
 else:
  raw=np.load(S/f'cache/{split}_local_raw.npz')['context'];n=read(S/'configs/local_normalization.json');ctx=(raw-np.array(n['mean'],np.float32))/np.array(n['std'],np.float32)
  for slot in [0,1]:ctx[...,slot*12:slot*12+6]*=raw[...,slot*12+6:slot*12+12]
  ctx[...,24:]*=raw[...,31:32]
 data['objects']=np.concatenate([data['objects'],oo],-1);data['context']=np.concatenate([data['context'],ctx],-1)
 return{k:torch.from_numpy(v).to(device)for k,v in data.items()}
def batch(data,ids):return{k:v[ids]for k,v in data.items()}
if __name__=='__main__':
 guard()
 for split in ['train','development']:make(split)
