from pathlib import Path
import os,json,csv,hashlib,time
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
T=Path(__file__).resolve().parent.parent;P=T.parent.parent;O=T.parent/'observed_latent_tsne_20261022'
z=np.load(O/'data/observed_latents_seed0.npz');rows=list(csv.DictReader((O/'results/coordinates_seed0.csv').open()));r=[a for a in rows if a['object']=='blue'];assert len(r)==960 and all(int(a['point_id'])%3==1 for a in r)
x=np.concatenate([z['object_h'][1::3],z['context_c']],-1).astype(np.float64);assert x.shape==(960,256)
mean=x.mean(0);std=np.maximum(x.std(0),1e-8);p=PCA(n_components=50,svd_solver='full',whiten=False);xp=p.fit_transform((x-mean)/std)
params=json.loads((O/'configs/protocol.json').read_text())['tsne'];start=time.time();ts=TSNE(**params);xy=ts.fit_transform(xp);assert np.isfinite(xy).all()
np.savez_compressed(T/'results/blue_context_coordinates.npz',xy=xy,standardization_mean=mean,standardization_std=std,pca_components=p.components_,pca_mean=p.mean_,pca50=xp,pca_variance_ratio=p.explained_variance_ratio_)
out=[]
for i,a in enumerate(r):
 b={k:a[k]for k in ['point_id','model_seed','split','episode_id','root_id','step','true_In','true_Held','true_object_valid']};b['tsne_x']=float(xy[i,0]);b['tsne_y']=float(xy[i,1]);out.append(b)
with (T/'results/blue_context_coordinates.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
sources=[O/'data/observed_latents_seed0.npz',O/'results/coordinates_seed0.csv',O/'results/scene_coordinates_seed0.csv']
audit={'state':'blueh128 + same-boundary environmentc128, actual object_head input','points':960,'dimensions':256,'PCA_retained_variance':float(p.explained_variance_ratio_.sum()),'parameters':params,'KL_divergence':float(ts.kl_divergence_),'seconds':time.time()-start,'labels_in_fit':False,'model_inference_or_training':False,'inputs_sha256':{str(a):hashlib.sha256(a.read_bytes()).hexdigest()for a in sources}}
(T/'results/blue_projection.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
# Check the actual cached training boundaries, respecting each episode length.
checks=[]
for name,base,split in [('old228',P/'experiments/object_recurrent_state_run_20261019','train'),('new640',P/'experiments/data_expansion_training_20261021','train'),('development',P/'experiments/data_expansion_training_20261021','development'),('test',P/'experiments/data_expansion_training_20261021','combination_test')]:
 d=np.load(base/f'cache/{split}_seed0.npz');meta=json.loads((base/f'cache/{split}_metadata.json').read_text());n=hits=after=validn=0
 for i,e in enumerate(meta):
  length=e.get('length',len(e['rows']));cat=d['cat'][i,:length+1,:3];valid=d['cat_valid'][i,:length+1,:3].all(-1);v=((cat==3).sum(-1)>=2)&((cat==3)|(cat==6)).all(-1);n+=len(cat);hits+=int(v.sum());after+=int(v[1:].sum());validn+=int(valid.sum())
 checks.append({'cohort':name,'episodes':len(meta),'boundaries_including_root':n,'all_object_labels_valid':validn,'condition_hits':hits,'post_execution_hits':after,'condition':'at least two In and each of all three In or Held'})
(T/'results/combination_boundary_audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(audit,ensure_ascii=False));print(json.dumps(checks))
