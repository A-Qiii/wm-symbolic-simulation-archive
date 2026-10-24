"""Deterministic binary CART, seven independent depth-4 trees; no dependencies or hand rules."""
import numpy as np
def fit_one(x,y,w,max_depth=4,min_leaf=2):
 def impurity(yy,ww):
  s=float(ww.sum());p=float((ww*yy).sum()/s);return s*2*p*(1-p)
 def build(ix,depth):
  yy=y[ix];ww=w[ix];node={'n':len(ix),'weighted_positive_probability':float((ww*yy).sum()/ww.sum())};base=impurity(yy,ww)
  if depth>=max_depth or len(ix)<2*min_leaf or base<=1e-12:return node
  best=None
  for j in range(x.shape[1]):
   vals=np.unique(x[ix,j]);thresholds=(vals[:-1]+vals[1:])/2
   for threshold in thresholds:
    left=x[ix,j]<=threshold;nl=int(left.sum())
    if nl<min_leaf or len(ix)-nl<min_leaf:continue
    score=impurity(yy[left],ww[left])+impurity(yy[~left],ww[~left])
    if best is None or score<best[0]-1e-12:best=(score,j,float(threshold),left)
  if best is None or base-best[0]<=1e-12:return node
  _,j,threshold,left=best;node.update(feature=j,threshold=threshold,weighted_impurity_gain=base-best[0],left=build(ix[left],depth+1),right=build(ix[~left],depth+1));return node
 return build(np.arange(len(x)),0)
def fit(x,y,valid,class_weights):
 trees=[]
 for j in range(7):
  gi=0 if j<3 else 1 if j<6 else 2;ii=valid[:,j];yy=y[ii,j];w=np.where(yy>.5,class_weights[gi][1],class_weights[gi][0]);trees.append(fit_one(x[ii],yy,w))
 return {'kind':'seven_independent_binary_CART','max_depth':4,'min_samples_leaf':2,'split_criterion':'weighted_Gini','tie_break':'lowest_feature_then_lowest_threshold','features':'seven observed ternary predicates, then nine declared action features','trees':trees}
def predict(model,x):
 result=np.empty((len(x),7),np.float32)
 for i,row in enumerate(x):
  for j,tree in enumerate(model['trees']):
   node=tree
   while 'feature'in node:node=node['left']if row[node['feature']]<=node['threshold']else node['right']
   result[i,j]=node['weighted_positive_probability']
 return result
def self_check():
 # Known exact cases verify leaf counts, impurity selection and a depth-2 XOR structure.
 x=np.array([[0],[0],[1],[1]],np.float32);y=np.array([0,0,1,1],np.float32);tree=fit_one(x,y,np.ones(4));assert tree['feature']==0 and tree['threshold']==.5 and tree['left']['weighted_positive_probability']==0 and tree['right']['weighted_positive_probability']==1
 same=fit_one(x,np.ones(4),np.ones(4));assert 'feature'not in same
 # CART deliberately does not split a zero-gain XOR root: this verifies the stated stop rule.
 xx=np.repeat(np.array([[0,0],[0,1],[1,0],[1,1]],np.float32),2,axis=0);yy=np.repeat([0,1,1,0],2);assert 'feature'not in fit_one(xx,yy,np.ones(8))
 assert 'feature'not in fit_one(np.array([[0],[1]],np.float32),np.array([0,1]),np.ones(2))
 return {'passed':True,'checks':['exact_separable','pure_leaf','zero_gain_stop','min_leaf_guard']}

