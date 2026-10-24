from common import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,trustworthiness
from sklearn.metrics import pairwise_distances
import sklearn,time,collections
def labels(r):
 y=r['truth']['y'];v=r['truth']['v'];h='未确定'if not all(v[:3])else'空手'if not any(y[:3])else'多物体'if sum(y[:3])>1 else['持红','持蓝','持粉'][int(np.argmax(y[:3]))]
 d='未确定'if not all(v[6:])else['关闭','打开','中间'][int(np.argmax(y[6:]))]
 return {'holding':h,'red_in':('未确定'if not v[3]else'在屉内'if y[3]else'不在屉内'),'blue_in':('未确定'if not v[4]else'在屉内'if y[4]else'不在屉内'),'pink_in':('未确定'if not v[5]else'在屉内'if y[5]else'不在屉内'),'drawer':d,'environment':r['environment'],'sampling_group':r['sampling_group']}
def nearest(x,k=10,roots=None):
 dd=pairwise_distances(x);np.fill_diagonal(dd,np.inf)
 if roots is not None:
  rr=np.array(roots);dd[rr[:,None]==rr[None,:]]=np.inf
 return np.argsort(dd,axis=1)[:,:k]
def run(x,cfg,perplexity):
 t=TSNE(**(cfg|{'perplexity':perplexity}));start=time.monotonic();xy=t.fit_transform(x);return xy,{'perplexity':perplexity,'KL_divergence':float(t.kl_divergence_),'iterations':int(t.n_iter_),'learning_rate_effective':float(t.learning_rate_),'seconds':time.monotonic()-start,'trustworthiness_k10':float(trustworthiness(x,xy,n_neighbors=10))}
def main():
 cfg=read(E/'configs/protocol.json');states=read(E/'data/main_states.json');ab=np.array([r['split']=='train'for r in states]);raw=dict(np.load(E/'data/main_latents.npz'));out={};logs={};scaled={};scales={};coords={}
 for kind in['interaction','ae']:
  x=raw[kind];mean=x[ab].mean(0);std=np.maximum(x[ab].std(0),1e-6);scaled[kind]=(x-mean)/std;scales[kind]=(mean,std);pca=PCA(n_components=50,svd_solver='full');xp=pca.fit_transform(scaled[kind]);xy,info=run(xp,cfg['tsne'],30);coords[kind]=xy;logs[kind]=info|{'PCA_retained_variance':float(pca.explained_variance_ratio_.sum()),'PCA_dimensions':50,'points':len(x)};out[kind+'_xy']=xy;out[kind+'_pca50']=xp;out[kind+'_mean']=mean;out[kind+'_std']=std
  if kind=='interaction':alt,ainfo=run(xp,cfg['tsne'],15);out['stability_xy']=alt;logs['stability']=ainfo|{'same_points_and_preprocessing':True};n1=nearest(xy);n2=nearest(alt);logs['stability']['mean_k10_neighbour_overlap']=float(np.mean([len(set(a)&set(b))/10 for a,b in zip(n1,n2)]))
 pair=read(E/'data/joint_pairs.json');jr=dict(np.load(E/'data/joint_latents.npz'));types=['actual','canonical_control','canonical_matched'];union=np.concatenate([jr[k]for k in types]);mean,std=scales['interaction'];ju=(union-mean)/std;jp=PCA(n_components=50,svd_solver='full');jx=jp.fit_transform(ju);jxy,jinfo=run(jx,cfg['tsne'],30);out['joint_xy']=jxy;out['joint_pca50']=jx;logs['joint']=jinfo|{'PCA_retained_variance':float(jp.explained_variance_ratio_.sum()),'points':len(union),'types':types,'one_shared_fit':True}
 # Original-space neighbours exclude same root; labels only evaluated after fitting.
 metrics=[];rootlist=[r['root_id']for r in states];labs=[labels(r)for r in states]
 for kind in['interaction','ae']:
  nn=nearest(scaled[kind],roots=rootlist)
  for field in['holding','red_in','blue_in','pink_in','drawer','environment']:
   vals=[r[field]for r in labs];agreements=[]
   for i,neighbours in enumerate(nn):
    if vals[i]=='未确定':continue
    valid=[j for j in neighbours if vals[j]!='未确定']
    if valid:agreements.append(np.mean([vals[j]==vals[i]for j in valid]))
   cnt=collections.Counter(v for v in vals if v!='未确定');base=sum((n/sum(cnt.values()))**2 for n in cnt.values());metrics.append({'model':kind,'field':field,'same_root_excluded':True,'k':10,'agreement':float(np.mean(agreements)),'class_frequency_random_baseline':base,'counts':dict(cnt)})
 np.savez_compressed(E/'results/coordinates.npz',**out);save(E/'results/projection.json',{'sklearn_version':sklearn.__version__,'fits':logs,'labels_in_PCA_or_TSNE':False,'model_parameter_updates':0,'chosen_prettiest_projection':False});save(E/'results/original_space_neighbours.json',metrics);print('PROJECTIONS_COMPLETE',logs,flush=True)
if __name__=='__main__':main()
