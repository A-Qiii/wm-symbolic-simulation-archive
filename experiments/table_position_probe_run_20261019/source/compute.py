from common import *
import torch,fcntl
from torch import nn
def model(kind):return nn.Linear(1028,1)if kind=='linear'else nn.Sequential(nn.Linear(1028 if kind=='mlp'else 2,256),nn.SiLU(),nn.Linear(256,1 if kind=='mlp'else 3))
def transformed(raw,mu,sd):
 x=pack(raw);x[...,:1024]=(x[...,:1024]-mu)/sd;return x
def main():
 sys.addaudithook(guard);assert not(S/'configs/heads_frozen.json').exists();assert not list((S/'models').glob('*.pt')),'No refits';lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();verify_inputs();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');t=target('train');m=t['mask'];trained=[]
 counts=m.sum(-1);per=np.divide((t['q']*m).sum(-1),counts,out=np.zeros(len(counts)),where=counts>0);rootmeans=[per[(t['root_id']==r)&(counts>0)].mean()for r in sorted(set(t['root_id']))];constant=float(np.mean(rootmeans));save(S/'configs/constant_baseline.json',{'q':constant,'training_roots_only':True})
 for seed in [0,1]:
  raw=features(seed,'train');packed=pack(raw);eligible=packed[m,:1024];mu=eligible.mean(0,dtype=np.float64).astype(np.float32);sd=np.maximum(eligible.std(0,dtype=np.float64),1e-6).astype(np.float32);np.savez(S/f'configs/normalization_seed{seed}.npz',mean=mu,std=sd)
  tensor=torch.tensor(transformed(raw,mu,sd),device='cuda');q=torch.tensor(t['q'],device='cuda');mask=torch.tensor(m,device='cuda');cat=torch.tensor(t['cat'],device='cuda');geom=torch.tensor(((t['x'][...,None]-CUTS)/WIDTH).astype(np.float32),device='cuda');schedule=np.load(S/f'configs/schedule_seed{seed}.npy');den=mask.sum(-1).clamp_min(1);ok=mask.any(-1)
  for kind,offset in [('linear',900),('mlp',910),('geometry',920)]:
   torch.manual_seed(261019000+offset+seed);net=model(kind).cuda();parameters=sum(p.numel()for p in net.parameters());assert parameters=={'linear':1029,'mlp':263681,'geometry':1539}[kind];opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001);curve=[];fitstart=time.monotonic();initial={k:v.detach().cpu().clone()for k,v in net.state_dict().items()}
   for step,ii in enumerate(schedule,1):
    assert time.monotonic()-start<1800,'Compute time cap';ix=torch.tensor(ii,device='cuda');pred=net(geom[ix]if kind=='geometry'else tensor[ix]);lossobj=torch.nn.functional.cross_entropy(pred.flatten(0,1),cat[ix].flatten().clamp_max(2),reduction='none').reshape(-1,3)if kind=='geometry'else torch.nn.functional.smooth_l1_loss(pred[...,0],q[ix],reduction='none',beta=.005/.88);lossb=(lossobj*mask[ix]).sum(-1)/den[ix];loss=lossb[ok[ix]].mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
    if step==1 or step%150==0:curve.append({'update':step,'loss':float(loss)});print('FIT',seed,kind,step,float(loss),flush=True)
   path=S/f'models/{kind}_seed{seed}.pt';torch.save({'state':net.state_dict(),'kind':kind,'seed':seed,'updates':1500,'initial_state':initial},path);receipt={'utc':now(),'seed':seed,'kind':kind,'updates':1500,'parameters':parameters,'seconds':time.monotonic()-fitstart,'curve':curve,'checkpoint_sha256':sha(path),'schedule_sha256':sha(S/f'configs/schedule_seed{seed}.npy')};save(S/f'results/fit_{kind}_seed{seed}.json',receipt);trained.append(receipt);del net,opt;torch.cuda.empty_cache()
  del tensor,q,mask,cat,geom;torch.cuda.empty_cache()
 save(S/'configs/heads_frozen.json',{'utc':now(),'before_validation_prediction':True,'fits':6,'updates':9000,'sha256':{str(p):sha(p)for p in sorted((S/'models').glob('*.pt'))}})
 def predict(split):
  gt=target(split)
  for seed in [0,1]:
   stats=np.load(S/f'configs/normalization_seed{seed}.npz');raw=features(seed,split);inp=torch.tensor(transformed(raw,stats['mean'],stats['std']),device='cuda');geo=torch.tensor(((gt['x'][...,None]-CUTS)/WIDTH).astype(np.float32),device='cuda')
   for kind in ['linear','mlp','geometry']:
    net=model(kind).cuda();net.load_state_dict(torch.load(S/f'models/{kind}_seed{seed}.pt',map_location='cuda',weights_only=True)['state']);net.eval()
    with torch.inference_mode():out=net(geo if kind=='geometry'else inp).cpu().numpy()
    prefix='privileged_geometry'if kind=='geometry'else'visual';np.savez_compressed(S/f'data/{prefix}_{kind}_seed{seed}_{split}.npz',prediction=out,root_id=raw['root_id'],episode_id=raw['episode_id'],step=raw['step']);del net
   del inp,geo;torch.cuda.empty_cache()
 predict('validation');v=target('validation');selected=[]
 for seed in [0,1]:
  score={}
  for kind in ['linear','mlp']:
   pred=np.load(S/f'data/visual_{kind}_seed{seed}_validation.npz')['prediction'][...,0]*WIDTH+LO;values,_=root_values(abs(pred-v['x'])*1000,v['mask']&v['h0'][:,None],v['root_id']);score[kind]=float(values.mean())
  selected.append({'seed':seed,'validation_h0_root_MAE_mm':score,'selected':'linear'if score['linear']<=score['mlp']else'mlp'})
 save(S/'configs/development_selection.json',{'utc':now(),'before_evaluation_prediction':True,'criterion':'Final-step validation h0 root-weighted MAE, tie linear; no refit','selection':selected})
 for split in ['train','old_band','gap']:predict(split)
 assert verify_inputs()>0
 save(S/'results/compute_complete.json',{'utc':now(),'seconds':time.monotonic()-start,'peak_cuda_GiB':torch.cuda.max_memory_allocated()/1024**3,'fits':len(trained),'updates':sum(x['updates']for x in trained),'encoder_forwards':0,'new_simulations':0,'RSSM_updates':0,'C_D_data_access':0,'uploads':0});print('COMPUTE_COMPLETE',time.monotonic()-start,selected,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:save(S/'results/FAILED.json',{'utc':now(),'error':repr(exc),'partial_fits':len(list((S/'models').glob('*.pt'))),'no_retry_or_budget_expansion':True});raise
