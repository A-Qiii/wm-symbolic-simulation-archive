from common import *
import torch
from wm_symbolic.public_geometry import visible_world_points,colour_masks

def feature(frame):
 obj=np.zeros((3,16),np.float32);ctx=np.zeros(32,np.float32);tcp=frame['robot_obs'][:3];reasons=collections.Counter()
 for vi,view in enumerate(['static','gripper']):
  need=['rgb_'+view,'depth_'+view,'seg_'+view,'calib_'+view+'_fov','calib_'+view+'_viewMatrix']
  if any(k not in frame for k in need):reasons['missing_camera_or_calibration']+=3;continue
  xyz=visible_world_points(frame,view);dep=frame['depth_'+view];seg=frame['seg_'+view];valid=np.isfinite(xyz).all(-1)&np.isfinite(dep)&(dep>.01)&(dep<5)&(seg>=0);masks=colour_masks(frame['rgb_'+view])
  for ci,color in enumerate(COLORS):
   candidates=[]
   for identifier in np.unique(seg[masks[color]&valid]):
    selection=masks[color]&valid&(seg==identifier);pts=xyz[selection]
    if len(pts)<6:continue
    extent=np.quantile(pts,.95,axis=0)-np.quantile(pts,.05,axis=0)
    if extent.max()>.13:continue
    candidates.append((len(pts),np.median(pts,axis=0)-tcp,extent))
   if candidates:
    n,point,extent=max(candidates,key=lambda x:x[0]);obj[ci,vi*8:(vi+1)*8]=np.r_[point,extent,np.log1p(n),1.]
   else:reasons['no_usable_visible_colour_region']+=1
  h,w=dep.shape
  for gy in range(2):
   for gx in range(2):
    sy=slice(gy*h//2,(gy+1)*h//2);sx=slice(gx*w//2,(gx+1)*w//2);vv=valid[sy,sx];j=vi*16+(gy*2+gx)*4
    if vv.any():ctx[j:j+4]=np.r_[np.median(xyz[sy,sx][vv],axis=0)-tcp,vv.mean()]
    else:reasons['empty_scene_cell']+=1
 assert np.isfinite(obj).all() and np.isfinite(ctx).all()
 return obj,ctx,reasons

def make_cache(split):
 source=C if split=='diagnosed_confirmation'else O;tag='confirm' if split=='diagnosed_confirmation'else split
 eps=read(source/f'cache/{tag}_metadata.json');memo={};objects=[];contexts=[];reasons=collections.Counter();masks=[];seen=0;start=time.time();provenance=[]
 for ei,e in enumerate(eps):
  oo=[];cc=[]
  for item in e['features']:
   p=item['public_inputs'][-1];assert p.endswith('.npz')and'private'not in p
   if p not in memo:
    with np.load(p)as z:f={k:z[k]for k in z.files}
    assert set(f).issuperset({'robot_obs','rgb_static'});memo[p]=feature(f);provenance.append({'path':p,'sha256':sha(p)})
   o,c,why=memo[p];oo.append(o);cc.append(c);reasons.update(why);masks.append(o[:,[7,15]]);seen+=1
  for dest,arr in [(objects,np.stack(oo)),(contexts,np.stack(cc))]:dest.append(np.concatenate([arr,np.repeat(arr[-1:],7-len(arr),axis=0)]))
  if(ei+1)%30==0:print('GEOMETRY_CACHE',split,ei+1,len(eps),round(time.time()-start,1),flush=True)
 np.savez_compressed(S/f'cache/{split}_geometry_raw.npz',objects=np.stack(objects),context=np.stack(contexts));save(S/f'cache/{split}_metadata.json',eps);save(S/f'cache/{split}_public_provenance.json',provenance)
 mask=np.stack(masks).astype(bool);save(S/f'results/geometry_missing_{split}.json',{'utc':now(),'boundaries':seen,'object_boundary_opportunities':seen*3,'unique_public_files':len(memo),'view_missing':{'static':int((~mask[...,0]).sum()),'gripper':int((~mask[...,1]).sum())},'both_views_missing':int((~mask.any(-1)).sum()),'at_least_one_view_missing':int((~mask.all(-1)).sum()),'reason_counts':dict(reasons),'no_GT_fill':True,'seconds':time.time()-start});print('GEOMETRY_DONE',split,round(time.time()-start,1),flush=True)

def extract(split,seed):
 assert(S/f'cache/{split}_geometry_raw.npz').exists(),split

def source_cache(split,seed):
 source=C if split=='diagnosed_confirmation'else O;tag='confirm' if split=='diagnosed_confirmation'else split
 return dict(np.load(source/f'cache/{tag}_seed{seed}.npz'))

def raw_inputs(split,seed,arm):
 if arm=='geometry':return dict(np.load(S/f'cache/{split}_geometry_raw.npz'))
 base=source_cache(split,seed);rng=np.random.default_rng(2610211000);po=rng.normal(size=(512,16)).astype(np.float32)/np.sqrt(512);pc=rng.normal(size=(512,32)).astype(np.float32)/np.sqrt(512)
 return {'objects':base['objects']@po,'context':base['context']@pc}

def fit_norm():
 eps=read(S/'cache/train_metadata.json');valid=np.array([[t<=e['length']for t in range(7)]for e in eps]);saved={}
 for seed in [0,1]:
  for arm in ARMS:
   r=raw_inputs('train',seed,arm);norm={}
   for name,x in r.items():
    xx=x[valid].reshape(-1,x.shape[-1]);mean=xx.mean(0);std=xx.std(0).clip(1e-3)
    if arm=='geometry':
     width=8 if name=='objects'else 4
     for b in range(0,x.shape[-1],width):
      use=xx[:,b+width-1]>0
      if use.any():mean[b:b+width-1]=xx[use,b:b+width-1].mean(0);std[b:b+width-1]=xx[use,b:b+width-1].std(0).clip(1e-3)
      mean[b+width-1]=0;std[b+width-1]=1
    norm[name]={'mean':mean.tolist(),'std':std.tolist()}
   saved[f'{arm}_seed{seed}']=norm
 save(S/'configs/input_normalization.json',{'source':'training boundaries only; padded boundaries excluded','values':saved})

def load(split,seed,device='cuda',arm='baseline'):
 data=source_cache(split,seed);raw=raw_inputs(split,seed,arm);norm=read(S/'configs/input_normalization.json')['values'][f'{arm}_seed{seed}']
 for name,x in raw.items():
  n=norm[name];out=(x-np.array(n['mean'],np.float32))/np.array(n['std'],np.float32)
  if arm=='geometry':
   width=8 if name=='objects'else 4
   for b in range(0,x.shape[-1],width):out[...,b:b+width]*=(x[...,b+width-1:b+width]>0)
  data[name]=np.concatenate([data[name],out.astype(np.float32)],-1)
 return {k:torch.from_numpy(v).to(device)for k,v in data.items()}
def batch(data,ids):return{k:v[ids]for k,v in data.items()}
if __name__=='__main__':
 guard()
 for split in ['train','development','historical_evaluation','diagnosed_confirmation']:budget();make_cache(split)
 fit_norm()
