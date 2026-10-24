from common import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn
def main():
 d=dict(np.load(E/'data/latents.npz'));c=read(E/'configs/protocol.json')['projection'];logical=np.concatenate([d['background'],np.stack([d['current'],d['predicted'],d['actual']],axis=1).reshape(-1,256)]);unique=[];ids=[];seen={}
 for row in logical:
  key=row.tobytes()
  if key not in seen:seen[key]=len(unique);unique.append(row)
  ids.append(seen[key])
 unique=np.stack(unique);scaled=(unique-d['mean'])/d['std'];pca=PCA(n_components=50,svd_solver='full');xp=pca.fit_transform(scaled);cfg={k:v for k,v in c.items()if k not in['PCA','PCA_solver','fit']};tsne=TSNE(**cfg);xy=tsne.fit_transform(xp);logicalxy=xy[np.array(ids)];pad=np.ptp(xy,axis=0)*.09;limits=np.stack([xy.min(0)-pad,xy.max(0)+pad]);np.savez_compressed(E/'results/coordinates.npz',unique_xy=xy,logical_xy=logicalxy,background_xy=logicalxy[:338],case_xy=logicalxy[338:].reshape(4,3,2),logical_to_unique=np.array(ids),axis_limits=limits,pca50=xp)
 save(E/'results/projection.json',{'logical_points':len(logical),'unique_points':len(unique),'exact_duplicates_aliased':len(logical)-len(unique),'one_joint_fit':True,'all_panels_same_axis_limits':limits,'PCA_retained_variance':float(pca.explained_variance_ratio_.sum()),'KL_divergence':float(tsne.kl_divergence_),'iterations':int(tsne.n_iter_),'sklearn_version':sklearn.__version__,'labels_used_in_fit':False,'seed':42,'perplexity':30,'learning_rate_effective':float(tsne.learning_rate_),'no_coordinate_jitter':True});print('ONE_UNION_FIT_COMPLETE',len(logical),len(unique),flush=True)
if __name__=='__main__':main()
