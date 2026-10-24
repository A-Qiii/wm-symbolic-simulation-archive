from pathlib import Path
import os,json,csv,time
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,trustworthiness
T=Path(__file__).resolve().parent.parent;cfg=json.loads((T/'configs/protocol.json').read_text());amend={'change':'User requested four memories:3object panels reuse unchanged joint object projection;environment c128fit independently, drawer truth/error. No new model inference.','models':[0,1],'scene_boundaries_per_model':960,'labels_in_reduction':False,'same_fixed_TSNE_parameters':cfg['tsne'],'cross_object_scene_coordinates_comparable':False};(T/'configs/four_memory_panels.json').write_text(json.dumps(amend,ensure_ascii=False,indent=2)+'\n')
for seed in [0,1]:
 z=np.load(T/f'data/observed_latents_seed{seed}.npz');x=z['context_c'].astype(np.float64);assert x.shape==(960,128);mu=x.mean(0);std=np.maximum(x.std(0),1e-8);p=PCA(n_components=50,svd_solver='full',whiten=False);xp=p.fit_transform((x-mu)/std);ts=TSNE(**cfg['tsne']);start=time.time();xy=ts.fit_transform(xp);assert np.isfinite(xy).all();np.savez_compressed(T/f'results/scene_coordinates_seed{seed}.npz',xy=xy,standardization_mean=mu,standardization_std=std,pca_components=p.components_,pca_mean=p.mean_,pca50=xp,pca_variance_ratio=p.explained_variance_ratio_)
 with(T/f'data/samples_seed{seed}.csv').open()as f:objects=list(csv.DictReader(f))
 rows=[]
 for i,r in enumerate(objects[::3]):
  rr={k:r[k]for k in ['model_seed','split','episode_id','root_id','family_id','step','drawer_true','drawer_valid','drawer_pred','skill_just_executed','observed_update']};rr.update(point_id=i,latent_type='shared_context_observed_c128',drawer_error=int(r['drawer_true']!=r['drawer_pred']),tsne_x=float(xy[i,0]),tsne_y=float(xy[i,1]));rows.append(rr)
 with(T/f'results/scene_coordinates_seed{seed}.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 rec={'seed':seed,'latent':'contextc128only, directsceneheadinput','points':960,'PCA_retained_variance':float(p.explained_variance_ratio_.sum()),'parameters':cfg['tsne'],'seconds':time.time()-start,'KL_divergence':float(ts.kl_divergence_),'trustworthiness_k15':float(trustworthiness(xp,xy,n_neighbors=15)),'labels_in_fit':False,'independent_from_object_embedding':True};(T/f'results/scene_projection_seed{seed}.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');print('SCENE',json.dumps(rec),flush=True)
