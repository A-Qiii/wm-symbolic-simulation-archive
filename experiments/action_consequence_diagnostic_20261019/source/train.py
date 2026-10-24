from common import *
import torch,fcntl
from torch import nn
class Model(nn.Module):
 def __init__(self,dim):
  super().__init__();self.initial=nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Linear(128,128));self.step=nn.GRUCell(7,128);self.output=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,15))
 def rollout(self,initial,actions):
  h=self.initial(initial);out=[]
  for t in range(actions.shape[1]):h=self.step(actions[:,t],h);out.append(self.output(h))
  return torch.stack(out,1)
def prepare(split,arm,visual_seed=None):
 d=dataset(split);norm=np.load(S/'configs/physical_normalization.npz');physical=(d['physical']-norm['mean'])/norm['std'];base=d['initial_state'].copy()
 if visual_seed is not None:
  v=np.load(S/f'data/visual_seed{visual_seed}_{split}.npz')['current'];base[...,:9]=hard(v);base[...,9:]=1
 if arm=='relations':physical=np.zeros_like(physical)
 x=np.concatenate([base,physical],-1).astype(np.float32);return d,x
def loss(raw,d):
 valid=d['valid'][:,1:];target=d['y'][:,1:];bce=torch.nn.functional.binary_cross_entropy_with_logits(raw[...,:6],target[...,:6],reduction='none');held=(bce[...,:3]*valid[...,:3]).sum(-1)/valid[...,:3].sum(-1).clamp_min(1);inside=(bce[...,3:6]*valid[...,3:6]).sum(-1)/valid[...,3:6].sum(-1).clamp_min(1);door=torch.nn.functional.cross_entropy(raw[...,6:9].flatten(0,1),target[...,6:9].argmax(-1).flatten(),reduction='none').reshape(raw.shape[:2])*valid[...,6];status=torch.nn.functional.cross_entropy(raw[...,9:12].flatten(0,1),d['status'].flatten(),reduction='none').reshape(raw.shape[:2]);moved=torch.nn.functional.binary_cross_entropy_with_logits(raw[...,12:15],d['moved'],reduction='none');moved=(moved*d['moved_valid']).sum(-1)/d['moved_valid'].sum(-1).clamp_min(1);return ((held+inside+door)/3+.5*status+.25*moved).mean()
def outputs(m,x,actions):
 n=len(x);one=m.rollout(x[:,:4].reshape(n*4,-1),actions.reshape(n*4,1,7)).reshape(n,4,15);free=m.rollout(x[:,0],actions);return one,free
def probabilities(out):return torch.cat([out[...,:6].sigmoid(),out[...,6:9].softmax(-1),out[...,9:12].softmax(-1),out[...,12:15].sigmoid()],-1).cpu().numpy()
def main():
 sys.addaudithook(guard);assert not list((S/'models').glob('*.pt')),'No repeated fits';lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();verify_inputs();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');dim=read(S/'configs/physical_schema.json')['model_state_dimensions'];selection=[];updates=0
 for seed in [0,1]:
  schedule=np.load(S/f'configs/schedule_seed{seed}.npy')
  for arm in ['relations','physics_enriched']:
   raw,x=prepare('train',arm);vr,vx=prepare('validation',arm);data={k:torch.tensor(raw[k],device='cuda')for k in ['y','valid','status','moved','moved_valid','action']};val={k:torch.tensor(vr[k],device='cuda')for k in ['y','valid','status','moved','moved_valid','action']};x=torch.tensor(x,device='cuda');vx=torch.tensor(vx,device='cuda');torch.manual_seed(2610191800+seed);m=Model(dim).cuda();initial={k:v.detach().cpu().clone()for k,v in m.state_dict().items()};opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001);curve=[];candidates=[];begin=time.monotonic()
   for step,ii in enumerate(schedule,1):
    assert time.monotonic()-start<1200,'Frozen compute limit';ix=torch.tensor(ii,device='cuda');b={k:v[ix]for k,v in data.items()};one,free=outputs(m,x[ix],b['action']);objective=.5*loss(one,b)+.5*loss(free,b);assert torch.isfinite(objective);opt.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();updates+=1
    if step==1 or step%150==0:curve.append({'update':step,'loss':float(objective)});print('FIT',seed,arm,step,float(objective),flush=True)
    if step in [750,1500]:
     with torch.inference_mode():a,b=outputs(m,vx,val['action']);score=float(.5*loss(a,val)+.5*loss(b,val))
     path=S/f'models/{arm}_seed{seed}_step{step}.pt';torch.save({'state':m.state_dict(),'seed':seed,'arm':arm,'updates':step,'initial_state':initial},path);candidates.append({'update':step,'validation_objective':score,'path':str(path),'sha256':sha(path)})
   selected=min(candidates,key=lambda x:(x['validation_objective'],x['update']));receipt={'utc':now(),'arm':arm,'seed':seed,'updates':1500,'parameters':sum(p.numel()for p in m.parameters()),'seconds':time.monotonic()-begin,'curve':curve,'candidates':candidates,'selected':selected};save(S/f'results/train_{arm}_seed{seed}.json',receipt);selection.append(receipt);del m,opt,data,val,x,vx;torch.cuda.empty_cache()
 save(S/'configs/selected_models.json',{'utc':now(),'before_test_prediction':True,'fits':4,'updates':updates,'selection':[{'seed':r['seed'],'arm':r['arm'],**r['selected']}for r in selection]})
 for selected in read(S/'configs/selected_models.json')['selection']:
  seed=selected['seed'];arm=selected['arm'];m=Model(dim).cuda();m.load_state_dict(torch.load(selected['path'],weights_only=True,map_location='cuda')['state']);m.eval()
  for split in SPLITS:
   variants=['truth']+(['visual_initial']if arm=='relations'else[])
   for variant in variants:
    data,x=prepare(split,arm,seed if variant=='visual_initial'else None);xt=torch.tensor(x,device='cuda');at=torch.tensor(data['action'],device='cuda')
    with torch.inference_mode():one,free=outputs(m,xt,at);op=probabilities(one);fp=probabilities(free)
    np.savez_compressed(S/f'data/predictions_{arm}_{variant}_seed{seed}_{split}.npz',one_step=op,free=fp,root_id=data['root_id'],episode_id=data['episode_id']);del xt,at
  del m;torch.cuda.empty_cache()
 verify_inputs();save(S/'results/compute_complete.json',{'utc':now(),'seconds':time.monotonic()-start,'fits':4,'updates':updates,'candidates_saved':8,'new_RSSM_fits':0,'encoder_forwards':0,'new_collection':0,'C_D_samples':0,'uploads':0,'peak_cuda_GiB':torch.cuda.max_memory_allocated()/1024**3});print('COMPUTE_COMPLETE',updates,time.monotonic()-start,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(S/'results/FAILED.json',{'utc':now(),'error':repr(e),'no_automatic_budget_increase':True});raise
