from pathlib import Path
import os,sys,json,csv,time,hashlib,collections
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np,sklearn,scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,trustworthiness
from sklearn.neighbors import NearestNeighbors
T=Path(__file__).resolve().parent.parent
cfg=json.loads((T/'configs/protocol.json').read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def neigh(x,k=15):return NearestNeighbors(n_neighbors=k+1,n_jobs=4).fit(x).kneighbors(x,return_distance=False)[:,1:]
def fit(x,perp):
 params={**cfg['tsne'],'perplexity':perp};m=TSNE(**params);start=time.time();xy=m.fit_transform(x);assert xy.shape==(2880,2)and np.isfinite(xy).all();return xy,{'parameters':params,'seconds':time.time()-start,'KL_divergence':float(m.kl_divergence_),'n_iter':int(m.n_iter_),'learning_rate_effective':float(m.learning_rate_),'trustworthiness_k15':float(trustworthiness(x,xy,n_neighbors=15))}
records=[]
for seed in cfg['model_seeds']:
 if(T/f'results/projection_seed{seed}.json').exists():continue
 with np.load(T/f'data/observed_latents_seed{seed}.npz')as z:x=z['object_h'].astype(np.float64)
 assert x.shape==(2880,128);mean=x.mean(0);std=x.std(0);std=np.maximum(std,1e-8);scaled=(x-mean)/std;pca=PCA(n_components=50,svd_solver='full',whiten=False);xp=pca.fit_transform(scaled);xy,info=fit(xp,30.);arrays={'xy':xy,'standardization_mean':mean,'standardization_std':std,'pca_components':pca.components_,'pca_mean':pca.mean_,'pca50':xp,'pca_variance_ratio':pca.explained_variance_ratio_};rec={'seed':seed,'samples':2880,'latent_dim':128,'PCA_dim':50,'PCA_retained_variance':float(pca.explained_variance_ratio_.sum()),'standardized_joint_viz_only':True,'labels_used_in_PCA_TSNE':False,'main':info,'software':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__}}
 if seed==0:
  alt,ai=fit(xp,50.);arrays['alternative_perplexity50_xy']=alt;n1=neigh(xy);n2=neigh(alt);rec['stability']={**ai,'same_preprocessing':True,'mean_k15_neighbour_overlap_with_main':float(np.mean([len(set(a)&set(b))/15 for a,b in zip(n1,n2)])),'main_projection_unchanged':True,'no_prettiest_selection':True}
 np.savez_compressed(T/f'results/coordinates_seed{seed}.npz',**arrays)
 # Labels and metadata are joined only after both dimension-reduction fits.
 with(T/f'data/samples_seed{seed}.csv').open()as f:rows=list(csv.DictReader(f))
 assert len(rows)==len(xy)
 for i,r in enumerate(rows):
  assert int(r['point_id'])==i;r['tsne_x']=float(xy[i,0]);r['tsne_y']=float(xy[i,1])
  if seed==0:r['alternative_perplexity50_x']=float(alt[i,0]);r['alternative_perplexity50_y']=float(alt[i,1])
 with(T/f'results/coordinates_seed{seed}.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save(T/f'results/projection_seed{seed}.json',rec);print('PROJECTED',seed,json.dumps(rec),flush=True)
