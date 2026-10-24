from config import *
import sys,torch,torch.nn.functional as F,fcntl,argparse
sys.path.insert(0,str(REP/'source'))
from model import Query
def loss(logits,y,v,weight):
 parts=[]
 for lo,hi in [(0,3),(3,6)]:parts.append((F.binary_cross_entropy_with_logits(logits[:,lo:hi],y[:,lo:hi],pos_weight=weight[lo:hi],reduction='none')*v[:,lo:hi]).sum(1)/v[:,lo:hi].sum(1).clamp(min=1))
 parts.append(F.cross_entropy(logits[:,6:],y[:,6:].argmax(1),weight=weight[6:],reduction='none')*v[:,6:].all(1));return torch.stack(parts,-1).mean(-1)
def tensor_hash(q):return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes()for _,t in sorted(q.state_dict().items()))).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['original_coverage','intermediate_coverage'],required=True);ap.add_argument('--seed',type=int,choices=[0,1],required=True);a=ap.parse_args();cfg=read(E/'configs/head_protocol.json');schedule=E/'configs'/f'head_schedule_{a.arm}_seed{a.seed}.npz';train=E/'data/head_train'/f'{a.arm}.npz';dev=E/'data/head_train/C_selection.npz';allowed={str(p.resolve())for p in[train,dev,schedule]}
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   path=Path(args[0]).resolve();s=str(path)
   if path.is_relative_to(P/'experiments')and path.suffix in['.npz','.json','.pt']and s not in allowed and not path.is_relative_to(E/'models')and not path.is_relative_to(E/'results')and not path.is_relative_to(E/'configs'):
    raise RuntimeError('Query trainer blocks non-approved data path: '+s)
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);tr=dict(np.load(train));dv=dict(np.load(dev));take=np.load(schedule)['indices'];assert take.shape==(1200,32)and len(tr['z'])==144;assert all(x.startswith(('A_','B_'))for x in tr['root_ids']);assert all(x.startswith('C_')for x in dv['root_ids']);assert not(set(tr['root_ids'])&set(dv['root_ids']))
 arrays={k:torch.tensor(tr[k],device='cuda')for k in['z','y','v']};dd={k:torch.tensor(dv[k],device='cuda')for k in['z','y','v']};weight=torch.tensor(cfg['class_weights'],device='cuda');torch.manual_seed(3000+a.seed);q=Query().cuda();assert sum(p.numel()for p in q.parameters())==25473;initial_hash=tensor_hash(q);opt=torch.optim.AdamW(q.parameters(),lr=.001,weight_decay=.001);folder=E/'models'/f'{a.arm}_seed{a.seed}';folder.mkdir(exist_ok=False);best=float('inf');selected=None;logs=[]
 for u in range(1,1201):
  ix=take[u-1];ll=loss(q(arrays['z'][ix]),arrays['y'][ix],arrays['v'][ix],weight).mean();opt.zero_grad();ll.backward();torch.nn.utils.clip_grad_norm_(q.parameters(),1);opt.step()
  if u in cfg['checkpoints']:
   q.eval()
   with torch.no_grad():ls=loss(q(dd['z']),dd['y'],dd['v'],weight).cpu().numpy()
   score=float(np.mean([ls[dv['root_ids']==root].mean()for root in sorted(set(dv['root_ids']))]));path=folder/f'update{u:04d}.pt';torch.save({'state':q.state_dict(),'update':u,'C_selection_root_macro_loss':score,'world_model':'canonical_matched_seed0/update1000','normalization':'original_query_seed0'},path);logs.append({'update':u,'training_loss':float(ll),'C_selection_root_macro_loss':score});print(a.arm,a.seed,u,score,flush=True)
   if score<best:best=score;selected=path
   q.train()
 save(folder/'selected.json',{'path':str(selected),'sha256':sha(selected),'C_selection_root_macro_loss':best,'selection':'minimum fixed C-selection root-macro query loss; earlier tie'});save(E/'results'/f'head_training_{a.arm}_seed{a.seed}.json',{'initial_query_hash':initial_hash,'updates':1200,'sample_draws':38400,'distinct_training_states':144,'parameters':25473,'seed':a.seed,'world_model_loaded_or_updated':False,'normalization_changed':False,'C_validation_or_D_read':False,'log':logs});print('HEAD_TRAINING_DONE',a.arm,a.seed,flush=True)
if __name__=='__main__':main()
