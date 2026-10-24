"""All representation fitting uses AB public observations, no proposition labels."""
from common import *
import torch
def fit_pca(x,n):
 mean=x.mean(0);center=x-mean;cov=center.T@center/max(1,len(x)-1);values,vectors=np.linalg.eigh(cov);order=np.argsort(values)[::-1];v=vectors[:,order[:n]].copy();values=np.maximum(values[order],0)
 # Fix eigenvector signs to make exports stable.
 for j in range(v.shape[1]):
  if v[np.argmax(abs(v[:,j])),j]<0:v[:,j]*=-1
 return mean.astype(np.float32),v.astype(np.float32),float(values[:n].sum()/max(values.sum(),1e-12))
def current_arrays(rows):
 fs=[]
 for r in rows:
  with np.load(E/'data/public/features'/f"{r['id']}.npz")as z:fs.append({k:z[k].astype(np.float32)for k in z.files})
 return {k:np.stack([f[k]for f in fs])for k in fs[0]}
def transform(f,n):
 def project(v):return ((v-n['vision_mean'])/n['vision_std']-n['pca_mean'])@n['pca_basis']/n['pca_scale']
 loc=project(f['local_tokens']);g=f['geometry'];visible=g[:,:,:,0,4];loc*=visible[:,:,:,None,None]
 obj=np.concatenate([loc.transpose(0,2,1,3,4).reshape(len(loc),3,-1),g[:,:,:,0,:5].transpose(0,2,1,3).reshape(len(loc),3,10),g[:,:,:,:,5].mean(-1).transpose(0,2,1)],-1)
 scene=np.concatenate([project(f['global_grid']).reshape(len(loc),-1),(f['robot_history']-n['robot_mean'])/n['robot_std']],-1)
 # Geometry is scaled independently; no DINO channels swamp positions/visibility.
 obj[...,-12:]=(obj[...,-12:]-n['geometry_mean'])/n['geometry_std']
 return obj.astype(np.float32),scene.astype(np.float32)
def main():
 rows=read(E/'data/public/states.json');ab=np.array([r['split']=='train'for r in rows]);f=current_arrays(rows)
 tokens=np.concatenate([f['local_tokens'][ab].reshape(-1,384),f['global_grid'][ab].reshape(-1,384)])
 vm=tokens.mean(0);vs=np.maximum(tokens.std(0),.05);vv=(tokens-vm)/vs;pm,pv,ratio=fit_pca(vv,32);scale=np.maximum((vv@pv).std(0),.1)
 geo=np.concatenate([f['geometry'][ab,:,:,0,:5].transpose(0,2,1,3).reshape(-1,10),f['geometry'][ab,:,:,:,5].mean(-1).transpose(0,2,1).reshape(-1,2)],-1)
 n={'vision_mean':vm,'vision_std':vs,'pca_mean':pm,'pca_basis':pv,'pca_scale':scale,'robot_mean':f['robot_history'][ab].mean(0),'robot_std':np.maximum(f['robot_history'][ab].std(0),.01),'geometry_mean':geo.mean(0),'geometry_std':np.maximum(geo.std(0),.05)}
 obj,scene=transform(f,n);om,ov,oratio=fit_pca(obj[ab].reshape(-1,obj.shape[-1]),64);sm,sv,sratio=fit_pca(scene[ab],64)
 n.update(object_mean=om,object_basis=ov,scene_mean=sm,scene_basis=sv)
 generic=np.concatenate([((obj-om)@ov),((scene-sm)@sv)[:,None]],1);gm=generic[ab].mean((0,1));gs=np.maximum(generic[ab].std((0,1)),.1);n.update(generic_mean=gm,generic_std=gs);generic=(generic-gm)/gs
 np.savez_compressed(E/'data/public/normalization.npz',**n);np.savez_compressed(E/'data/public/inputs.npz',objects=obj,scene=scene,generic=generic.astype(np.float32))
 save(E/'results/normalization.json',{'fitted_split':'A/B only','train_states':int(ab.sum()),'dev_states':int((~ab).sum()),'DINO_channel_PCA32_variance_retained':ratio,'generic_object_PCA64_variance_retained':oratio,'generic_scene_PCA64_variance_retained':sratio,'object_shape':list(obj.shape),'scene_shape':list(scene.shape),'generic_shape':list(generic.shape),'no_semantic_labels_or_old_normalization':True})
 print(read(E/'results/normalization.json'),flush=True)
if __name__=='__main__':main()
