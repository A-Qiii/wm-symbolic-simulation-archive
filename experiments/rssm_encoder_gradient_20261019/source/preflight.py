from common import *
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch,fcntl
from wm_symbolic.world_model import WorldModel,loss_for_batch
from wm_symbolic.core_dataset import CoreEpisodes
from losses import losses,backward_routed,clip_groups
def flatten(gs):return torch.cat([x.detach().float().flatten()for x in gs])
def maxdiff(a,b):return float((flatten(a)-flatten(b)).abs().max())
def main():
 lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard('train');torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 norm=read(E/'configs/normalization.json');data=CoreEpisodes(P,'train',normalizer=norm);b=data.batch([0,1],'cuda');w=[torch.tensor(v,device='cuda')for v in norm['class_weights']]
 m=WorldModel(skill_count=6,event_count=3).cuda();m.load_state_dict(torch.load(E/'models/initial_seed0.pt',map_location='cuda',weights_only=True)['state']);e,r=partition(m);pars=list(m.parameters());records=[]
 for k in [0,3]:
  m.zero_grad(set_to_none=True);torch.manual_seed(1234);old,ot=loss_for_batch(m,b,k,w);gold=torch.autograd.grad(old,pars);oldv=float(old);oldterms={n:float(v)for n,v in ot.items()};gold={id(p):g.detach()for p,g in zip(pars,gold)}
  torch.manual_seed(1234);total,current,future,terms,enc=losses(m,b,k,w)
  assert float(total)==oldv and all(float(terms[n])==oldterms[n]for n in terms)
  assert torch.allclose(total,current+future,rtol=1e-6,atol=1e-6)
  gf=torch.autograd.grad(total,pars,retain_graph=True);gc=torch.autograd.grad(current,e,retain_graph=True);gefuture=torch.autograd.grad(future,e)
  assert maxdiff(gf,[gold[id(p)]for p in pars])==0
  currentnorm=float(flatten(gc).norm());futurenorm=float(flatten(gefuture).norm());assert currentnorm>0 and futurenorm>0
  byarm={}
  for arm in ARMS:
   m.zero_grad(set_to_none=True);torch.manual_seed(1234);t,c,f,tt,z=losses(m,b,k,w);route=backward_routed(m,t,c,z,arm,divisor=1)
   expected=[gold[id(p)]for p in e]if arm=='joint'else gc
   ed=maxdiff([p.grad for p in e],expected);rd=maxdiff([p.grad for p in r],[gold[id(p)]for p in r])
   assert ed<2e-6 and rd<2e-6,(arm,ed,rd)
   en,rn=clip_groups(m);byarm[arm]={'E':[p.grad.clone()for p in e],'R':[p.grad.clone()for p in r],'E_error':ed,'R_error':rd,'E_rawnorm':en,'R_rawnorm':rn}
  assert maxdiff(byarm[ARMS[0]]['R'],byarm[ARMS[1]]['R'])==0
  # Inject an arbitrary R magnitude: independent group clipping cannot rescale E.
  for p,g in zip(e,gc):p.grad=g.clone()
  for p in r:p.grad=gold[id(p)]*1000
  clip_groups(m);coupling=maxdiff([p.grad for p in e],byarm['route_current_only']['E']);assert coupling<2e-6
  records.append({'k':k,'original_scalar':oldv,'terms':oldterms,'scalar_exact':True,'full_grad_exact':True,'current_E_norm':currentnorm,'future_E_norm':futurenorm,'routed_errors':{a:{n:v for n,v in q.items()if n not in ['E','R']}for a,q in byarm.items()},'R_postclip_equal':True,'E_Rmagnitude_coupling_error':coupling})
  del gold,gf,gc,gefuture,byarm
 with torch.no_grad():
  a=m.action(b['skill'],b['arguments'],b['budget']);x=m.encode(b['tokens'],b['geometry'],b['visible'],b['robot']);s,_=m.initial(x[:,0],True);p,_=m.imagine(s,a[:,0],True)
  poison=b['tokens'].clone();poison[:,1:]+=100;x2=m.encode(poison,b['geometry'],b['visible'],b['robot']);s2,_=m.initial(x2[:,0],True);p2,_=m.imagine(s2,a[:,0],True);leak=float((p.vector()-p2.vector()).abs().max());assert leak==0
 save(E/'results/preflight.json',{'utc':now(),'passed':True,'records':records,'future_poison_prior_difference':leak,'optimizer_updates':0,'physics':0,'D_samples':0,'clipping':'separateE/R norm1 botharms; no shared premask norm','precision':'float32 exact forward/scalar/fullgradient reference; routed backward tolerance2e-6'})
 print('PREFLIGHT_PASS',json.dumps(records),flush=True);lock.close()
if __name__=='__main__':main()
