from config import *
import torch,sys,argparse,fcntl
sys.path.insert(0,str(PRIOR/'source'))
from train_query import loss,th
from model import Query
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['control','empty_open'],required=True);ap.add_argument('--seed',type=int,choices=[0,1],required=True);a=ap.parse_args()
 for p,h in read(E/'configs/training_frozen.json')['sha256'].items():assert sha(p)==h,p
 cfg=read(E/'configs/protocol.json')['query_training'];train=E/'data/query_train'/f'{a.arm}.npz';devs=[E/'data/query_train'/f'{name}.npz'for name in ['old_C_selection','C_selection']];initial=cfg['initial_heads'][str(a.seed)]['path'];schedule=E/'configs'/f'schedule_seed{a.seed}.npz';allowed={str(p)for p in [train,*devs,schedule,Path(initial)]}
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(P/'experiments')and p.suffix in ['.json','.npz','.pt']and str(p)not in allowed and not p.is_relative_to(E/'models')and not p.is_relative_to(E/'results'):raise RuntimeError('Training forbidden data '+str(p))
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);tr=dict(np.load(train));dv=[dict(np.load(p))for p in devs];sc=np.load(schedule)[a.arm];assert sc.shape==(600,32);assert all(r.startswith(('A_','B_'))for r in tr['root_ids']);data={k:torch.tensor(tr[k],device='cuda')for k in ['z','y','v']};dd=[{k:torch.tensor(d[k],device='cuda')for k in ['z','y','v']}for d in dv];w=torch.tensor(cfg['class_weights'],device='cuda');q=Query().cuda();q.load_state_dict(torch.load(initial,weights_only=True,map_location='cuda')['state']);init=th(q);opt=torch.optim.AdamW(q.parameters(),lr=.001,weight_decay=.001);folder=E/'models'/f'{a.arm}_seed{a.seed}';folder.mkdir();logs=[];best=float('inf');selected=None
 def check(u,tloss):
  nonlocal best,selected
  scores=[]
  with torch.no_grad():
   for d,arr in zip(dv,dd):
    ls=loss(q(arr['z']),arr['y'],arr['v'],w).cpu().numpy();scores.append(float(np.mean([ls[d['root_ids']==r].mean()for r in sorted(set(d['root_ids']))])))
  score=sum(scores)/2;path=folder/f'update{u:04d}.pt';torch.save({'state':q.state_dict(),'update':u,'C_score':score},path);logs.append({'update':u,'score':score,'old_C':scores[0],'new_C_select':scores[1],'training_loss':tloss});print(a.arm,a.seed,u,score,flush=True)
  if score<best:best=score;selected=path
 check(0,None)
 for u in range(1,601):
  ix=sc[u-1];l=loss(q(data['z'][ix]),data['y'][ix],data['v'][ix],w).mean();opt.zero_grad();l.backward();torch.nn.utils.clip_grad_norm_(q.parameters(),1);opt.step()
  if u%150==0:check(u,float(l))
 save(folder/'selected.json',{'path':str(selected),'sha256':sha(selected),'C_score':best});save(E/'results'/f'train_{a.arm}_seed{a.seed}.json',{'updates':600,'draws':19200,'pool_size':len(tr['z']),'initial_hash':init,'initial_path':initial,'log':logs,'only_query_updated':True,'C_validation_or_D_access':False});print('TRAINING_DONE',a.arm,a.seed,flush=True)
if __name__=='__main__':main()
