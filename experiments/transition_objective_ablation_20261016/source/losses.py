from config import *
import sys,torch
sys.path.insert(0,str(REP/'source'))
from model import feature_error
def batch_terms(m,data,sc,u,need_feature=True):
 o,s,z,b,a,act=data;ix=sc['single'][u];pr=m.transition(z[b[ix]],act[ix]);terms=[]
 def calc(pr,ix,weight):
  if need_feature:f=feature_error(m.decode(pr),o[a[ix]],s[a[ix]],(o[b[ix]],s[b[ix]])).mean()
  else:
   with torch.no_grad():f=feature_error(m.decode(pr),o[a[ix]],s[a[ix]],(o[b[ix]],s[b[ix]])).mean()
  return(weight,f,(pr-z[a[ix]]).square().mean())
 terms.append(calc(pr,ix,1.0))
 if(u+1)%4==0:
  seq=sc['multi'][u];cur=m.transition(z[b[seq[:,0]]],act[seq[:,0]])
  for depth in [1,2]:
   ni=seq[:,depth];cur=m.transition(cur,act[ni]);terms.append(calc(cur,ni,.25))
 return terms
def load_data():
 rows=read(PACK/'pairs.json');d=dict(np.load(PACK/'inputs.npz'));data=tuple(torch.tensor(v,device='cuda')for v in[d['objects'],d['scene'],d['z'],[r['before']for r in rows],[r['after']for r in rows],[r['action']for r in rows]])
 return rows,data
def mixed_loss(terms):return sum(w*(f+.25*z)for w,f,z in terms)
def rawZ(terms):return sum(w*z for w,f,z in terms)
