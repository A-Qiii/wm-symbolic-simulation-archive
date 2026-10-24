from config import *
import torch,torch.nn.functional as F,sys,fcntl,argparse
sys.path.insert(0,str(REP/'source'))
from model import Query
def loss(logits,y,v,w):
 parts=[(F.binary_cross_entropy_with_logits(logits[:,lo:hi],y[:,lo:hi],pos_weight=w[lo:hi],reduction='none')*v[:,lo:hi]).sum(1)/v[:,lo:hi].sum(1).clamp(min=1)for lo,hi in [(0,3),(3,6)]];parts.append(F.cross_entropy(logits[:,6:],y[:,6:].argmax(1),weight=w[6:],reduction='none')*v[:,6:].all(1));return torch.stack(parts,-1).mean(-1)
def th(q):return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes()for k,t in sorted(q.state_dict().items()))).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['old_coverage','new_coverage'],required=True);ap.add_argument('--seed',type=int,choices=[0,1],required=True);a=ap.parse_args()
 for p,h in read(E/'configs/query_training_frozen.json')['sha256'].items():assert sha(p)==h,p
 cfg=read(E/'configs/query_inputs.json');train=E/'data/query_train'/f'{a.arm}.npz';devs=[E/'data/query_train'/f'{name}.npz'for name in ['old_C_selection','C_selection']];schedule=E/'configs'/f'query_schedule_{a.arm}_seed{a.seed}.npz';allowed={str(p)for p in [train,*devs,schedule]}
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(P/'experiments')and p.suffix in ['.json','.npz','.pt']and str(p)not in allowed and not p.is_relative_to(E/'models')and not p.is_relative_to(E/'results'):raise RuntimeError('Query trainer forbidden data: '+str(p))
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);tr=dict(np.load(train));dv=[dict(np.load(p))for p in devs];take=np.load(schedule)['indices'];assert take.shape==(1200,32);assert all(r.startswith(('A_','B_'))for r in tr['root_ids']);assert all(r.startswith('C_')for d in dv for r in d['root_ids'])
 data={k:torch.tensor(tr[k],device='cuda')for k in ['z','y','v']};dd=[{k:torch.tensor(d[k],device='cuda')for k in ['z','y','v']}for d in dv];w=torch.tensor(cfg['class_weights'],device='cuda');torch.manual_seed(3000+a.seed);q=Query().cuda();init=th(q);opt=torch.optim.AdamW(q.parameters(),lr=.001,weight_decay=.001);folder=E/'models'/f'query_{a.arm}_seed{a.seed}';folder.mkdir();best=float('inf');selected=None;logs=[]
 for u in range(1,1201):
  ix=take[u-1];l=loss(q(data['z'][ix]),data['y'][ix],data['v'][ix],w).mean();opt.zero_grad();l.backward();torch.nn.utils.clip_grad_norm_(q.parameters(),1);opt.step()
  if u%300==0:
   scores=[]
   with torch.no_grad():
    for d,arr in zip(dv,dd):
     ls=loss(q(arr['z']),arr['y'],arr['v'],w).cpu().numpy();scores.append(float(np.mean([ls[d['root_ids']==r].mean()for r in sorted(set(d['root_ids']))])))
   score=sum(scores)/2;path=folder/f'update{u:04d}.pt';torch.save({'state':q.state_dict(),'update':u,'selection_score':score},path);logs.append({'update':u,'score':score,'old_C_loss':scores[0],'new_C_select_loss':scores[1],'training_loss':float(l)});print(a.arm,a.seed,u,score,flush=True)
   if score<best:best=score;selected=path
 save(folder/'selected.json',{'path':str(selected),'sha256':sha(selected),'C_score':best,'selection':'minimum equal-weight old/new C-select root-macro masked loss, earlier tie'});save(E/'results'/f'train_query_{a.arm}_seed{a.seed}.json',{'updates':1200,'draws':38400,'distinct_states':len(set(take.flatten())),'pool_size':len(tr['z']),'initial_hash':init,'log':logs,'only_query_optimized':True,'test_or_validation_access':False});print('QUERY_DONE',a.arm,a.seed,flush=True)
if __name__=='__main__':main()
