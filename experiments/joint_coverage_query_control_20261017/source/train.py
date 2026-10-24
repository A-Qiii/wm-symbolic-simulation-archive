"""Only the original Query is updated, with matched600-step continuation arms."""
from config import *
import torch,sys,ast,fcntl
sys.path.insert(0,str(REP/'source'))
from model import Query

def loss_function():
 tree=ast.parse((PRIOR/'source/train_query.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='loss');ns={'torch':torch,'F':torch.nn.functional};exec(compile(ast.Module(body=[f],type_ignores=[]),'exact_frozen_query_loss','exec'),ns);return ns['loss']
def th(q):return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes()for _,t in sorted(q.state_dict().items()))).hexdigest()
def main():
 verify();assert not(E/'results/training.json').exists();cfg=read(E/'configs/protocol.json');initial={int(k):v['path']for k,v in cfg['initial_heads'].items()};allowed=set(initial.values())
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=str(args[0])
   if '/data/evaluation/'in p or str(COMBO)in p or str(PROBE)in p or str(EV)in p:raise RuntimeError('Query trainer cannot access evaluation data: '+p)
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);loss=loss_function();w=torch.tensor(cfg['class_weights'],device='cuda');logs=[];hashes={};inithashes={}
 for seed in [0,1]:
  schedule=np.load(E/'configs'/f'schedule_seed{seed}.npz')
  for arm in ['control','coverage']:
   tr=dict(np.load(E/'data/train'/f'{arm}.npz'));assert all(str(r).startswith(('A_','B_'))for r in tr['root_ids']);data={k:torch.tensor(tr[k],device='cuda')for k in ['z','y','v']};q=Query().cuda();q.load_state_dict(torch.load(initial[seed],weights_only=True,map_location='cuda')['state']);init=th(q)
   if seed in inithashes:assert init==inithashes[seed]
   inithashes[seed]=init;opt=torch.optim.AdamW(q.parameters(),lr=.001,weight_decay=.001);folder=E/'models'/f'{arm}_seed{seed}';folder.mkdir();torch.save({'state':q.state_dict(),'update':0,'initial_hash':init},folder/'update0000.pt');history=[];clipcount=0
   for u in range(1,601):
    ix=schedule[arm][u-1];q.train();ls=loss(q(data['z'][ix]),data['y'][ix],data['v'][ix],w).mean();opt.zero_grad();ls.backward();norm=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1));clipcount+=norm>1;opt.step()
    if u in [1,150,300,450,600]:history.append({'update':u,'training_loss':float(ls.detach()),'preclip_norm':norm})
   path=folder/'update0600.pt';torch.save({'state':q.state_dict(),'update':600,'initial_hash':init,'final_hash':th(q),'arm':arm,'seed':seed},path);hashes[str(path)]=sha(path);hashes[str(folder/'update0000.pt')]=sha(folder/'update0000.pt');logs.append({'seed':seed,'arm':arm,'updates':600,'draws':19200,'initial_hash':init,'final_hash':th(q),'clipped_updates':clipcount,'history':history});print('QUERY_CONTROL_TRAINED',arm,seed,history[-1],flush=True)
 save(E/'results/training.json',{'utc':now(),'runs':logs,'original_architecture_unchanged':True,'paired_initial_hashes_equal':True,'fixed_final_update':600,'earlystop_or_checkpoint_selection':False,'evaluation_data_used_for_training':False,'D_used_for_training_or_selection':False,'encoder_decoder_transition_updates':0,'new_physical_calls':0});save(E/'configs/final_heads_frozen.json',{'utc':now(),'before_evaluation':True,'sha256':hashes})
if __name__=='__main__':main()
