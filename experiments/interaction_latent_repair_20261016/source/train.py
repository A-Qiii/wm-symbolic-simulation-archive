from common import *
from runtime import *
from model import feature_error
import fcntl,time,argparse,sys
def frozen_hash(m):return tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--preflight',action='store_true');a=ap.parse_args();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=True
 access=[]
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=str(args[0]);
   if '/_private/'in p or '/data/D2/'in p or '/data/D_raw/'in p or '/data/closed_loop/'in p:raise RuntimeError('Training forbidden source '+p)
   if str(E)in p:access.append(p)
 sys.addaudithook(guard)
 if not a.preflight:
  for p,h in read(E/'configs/training_frozen.json')['sha256'].items():assert sha(p)==h,p
 rows=read(E/'data/public/pairs.json');data=dict(np.load(E/'data/public/inputs.npz'));o=torch.tensor(data['objects'],device='cuda');s=torch.tensor(data['scene'],device='cuda');b=torch.tensor([r['before']for r in rows],device='cuda');af=torch.tensor([r['after']for r in rows],device='cuda');act=torch.tensor([r['action']for r in rows],device='cuda');schedule=dict(np.load(E/'configs'/f'schedule_seed{a.seed}.npz'));m,q,stats=load(a.seed,initial=True);m.transition.requires_grad_(True);fhash=frozen_hash(m);qhash=tensor_hash(q.state_dict());init=tensor_hash(m.state_dict());cfg=read(E/'configs/protocol.json');start=time.monotonic()
 with torch.no_grad():z=m.encode(o,s).detach()
 def loss(u):
  ix=np.r_[schedule['base'][u],schedule['control_aux'if a.arm==ARMS[0]else'matched_aux'][u]];pred=m.transition(z[b[ix]],act[ix]);l=feature_error(m.decode(pred),o[af[ix]],s[af[ix]],(o[b[ix]],s[b[ix]])).mean()+.25*(pred-z[af[ix]]).square().mean()
  if(u+1)%4==0:
   seq=schedule['observed_chains'][u];cur=pred[:48]
   for depth in[1,2]:
    valid=np.where(seq[:,depth]>=0)[0]
    if not len(valid):break
    ni=seq[valid,depth];nxt=m.transition(cur[valid],act[ni]);l+=.25*(feature_error(m.decode(nxt),o[af[ni]],s[af[ni]],(o[b[ni]],s[b[ni]])).mean()+.25*(nxt-z[af[ni]]).square().mean());upd=cur.clone();upd[valid]=nxt;cur=upd
  return l
 # Full-loss backward including multi-step branch, without optimizer update.
 loss(3).backward();frozen_grad=[n for n,p in m.named_parameters()if not n.startswith('transition.')and p.grad is not None];tr=[n for n,p in m.named_parameters()if p.requires_grad];assert not frozen_grad and all(n.startswith('transition.')for n in tr);assert all(p.grad is None for p in q.parameters());assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in m.transition.parameters());assert frozen_hash(m)==fhash and tensor_hash(q.state_dict())==qhash
 pre={'passed':True,'arm':a.arm,'seed':a.seed,'initial_state_hash':init,'frozen_hash':fhash,'query_hash':qhash,'trainable':tr,'trainable_parameters':sum(p.numel()for p in m.parameters()if p.requires_grad),'frozen_gradients_none':True,'query_gradients_none':True,'optimizer_updates':0,'finite_transition_gradients':True};m.zero_grad(set_to_none=True)
 if a.preflight:save(E/'results'/f'gradient_preflight_{a.arm}_{a.seed}.json',pre);print('PREFLIGHT_PASS',pre,flush=True);return
 folder=E/'models'/f'{a.arm}_seed{a.seed}';folder.mkdir(exist_ok=False);opt=torch.optim.AdamW(m.transition.parameters(),lr=.0001,weight_decay=.0001);dev=np.array([i for i,r in enumerate(rows)if r['split']=='dev']);best=float('inf');selected=None;logs=[]
 def checkpoint(u,trainloss):
  nonlocal best,selected
  with torch.no_grad():pred=m.transition(z[b[dev]],act[dev]);errs=feature_error(m.decode(pred),o[af[dev]],s[af[dev]],(o[b[dev]],s[b[dev]])).mean(1).cpu().numpy()
  roots={root:float(errs[[j for j,i in enumerate(dev)if rows[i]['root_id']==root]].mean())for root in sorted({rows[i]['root_id']for i in dev})};oldroots={rows[i]['root_id']for i in dev if rows[i]['dataset_group']=='original'};newroots=set(roots)-oldroots;oldmean=float(np.mean([roots[r]for r in oldroots]));newmean=float(np.mean([roots[r]for r in newroots]));score=.5*(oldmean+newmean);assert frozen_hash(m)==fhash and tensor_hash(q.state_dict())==qhash
  path=folder/f'update{u:04d}.pt';torch.save({'state':m.state_dict(),'update':u,'score':score,'initial_hash':init,'frozen_hash':fhash,'seed':a.seed,'arm':a.arm},path);row={'update':u,'selection_score':score,'old_C35_root_mean':oldmean,'new_C2_root_mean':newmean,'all_C37_roots_equal_mean':float(np.mean(list(roots.values()))),'root_values':roots,'train_loss':trainloss,'frozen_hash_unchanged':True};logs.append(row);print(a.arm,a.seed,u,score,oldmean,newmean,flush=True)
  if score<best:best=score;selected=path
 checkpoint(0,None)
 for u in range(1000):
  opt.zero_grad(set_to_none=True);l=loss(u);assert torch.isfinite(l);l.backward();assert all(p.grad is None for n,p in m.named_parameters()if not n.startswith('transition.'));torch.nn.utils.clip_grad_norm_(m.transition.parameters(),5);opt.step()
  if(u+1)%250==0:checkpoint(u+1,float(l))
 save(folder/'selected.json',{'path':str(selected),'sha256':sha(selected),'selection':'fixed half old35/half new2 C root observation error; earlier tie','update':int(selected.stem[6:])});save(E/'results'/f'train_{a.arm}_{a.seed}.json',{'log':logs,'initial':pre,'final_frozen_hash':frozen_hash(m),'final_query_hash':tensor_hash(q.state_dict()),'seconds':time.monotonic()-start,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'updates':1000,'label_accesses':0,'file_accesses':sorted(set(access))});print('TRAIN_DONE',a.arm,a.seed,round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':main()
