from common import *
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch,fcntl
from wm_symbolic.world_model import WorldModel,loss_for_batch
from fixed_dataset import CoreEpisodes
from losses import losses,backward_routed,clip_groups
def main():
 lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard('train');torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
 n=read(E/'configs/normalization.json');d=CoreEpisodes(P,'train',normalizer=n);b=d.batch([0,1],'cuda');w=[torch.tensor(x,device='cuda')for x in n['class_weights']];m=WorldModel(skill_count=6,event_count=3).cuda();m.load_state_dict(torch.load(E/'models/initial_seed0.pt',map_location='cuda',weights_only=True)['state']);e,r=partition(m);pars=list(m.parameters());out=[]
 torch.manual_seed(261019000)
 with torch.autocast('cuda',dtype=torch.bfloat16):old,ot=loss_for_batch(m,b,3,w)
 oldv=float(old.detach());oldterms={k:float(v.detach())for k,v in ot.items()};gold=torch.autograd.grad(old/2,pars);gold={id(p):g for p,g in zip(pars,gold)}
 torch.manual_seed(261019000)
 with torch.autocast('cuda',dtype=torch.bfloat16):t,c,f,terms,z=losses(m,b,3,w)
 assert float(t.detach())==oldv and all(float(terms[k].detach())==oldterms[k]for k in terms)
 gc=torch.autograd.grad(c/2,e)
 for arm in ARMS:
  m.zero_grad(set_to_none=True);torch.manual_seed(261019000)
  with torch.autocast('cuda',dtype=torch.bfloat16):t,c,f,terms,z=losses(m,b,3,w)
  backward_routed(m,t,c,z,arm,2)
  ed=max(float((p.grad-(gold[id(p)]if arm=='joint'else g)).abs().max())for p,g in zip(e,gc));rd=max(float((p.grad-gold[id(p)]).abs().max())for p in r)
  assert ed==0 and rd==0,(ed,rd)
  en,rn=clip_groups(m);out.append({'arm':arm,'E_max_error':ed,'R_max_error':rd,'E_norm':en,'R_norm':rn})
 save(E/'results/preflight_bf16.json',{'utc':now(),'passed':True,'scalar_exact':True,'records':out,'optimizer_updates':0});print('BF16_PREFLIGHT_PASS',out,flush=True)
if __name__=='__main__':main()
