from common import *
from model import Model,feature_error
import torch,fcntl,argparse,time,sys
from collections import defaultdict
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['generic','ae','interaction'],required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=read(E/'configs/protocol.json');freeze=read(E/'configs/training_frozen.json')
 for pp,h in freeze['sha256'].items():assert sha(pp)==h,pp
 access=[]
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   pp=str(args[0]);
   if '/_private/'in pp or '/candidate/'in pp or ('/experiments/'in pp and str(E)not in pp):raise RuntimeError('Stage1 forbidden source: '+pp)
   if str(E)in pp:access.append(pp)
 sys.addaudithook(guard)
 torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 rows=read(E/'data/public/states.json');pairs=read(E/'data/public/pairs.json');chains=read(E/'data/public/chains.json');norm=dict(np.load(E/'data/public/normalization.npz'))
 with np.load(E/'data/public/inputs.npz')as zz:obj=torch.tensor(zz['objects'],device='cuda');scene=torch.tensor(zz['scene'],device='cuda')
 b=torch.tensor([p['before']for p in pairs],device='cuda');aft=torch.tensor([p['after']for p in pairs],device='cuda');act=torch.tensor([p['action']for p in pairs],device='cuda')
 ab=[i for i,r in enumerate(rows)if r['split']=='train'];dev=np.array([i for i,p in enumerate(pairs)if p['split']=='dev']);abp=[i for i,p in enumerate(pairs)if p['split']=='train'];state_roots=defaultdict(list);pair_roots=defaultdict(list)
 for i in ab:state_roots[rows[i]['root_id']].append(i)
 for i in abp:pair_roots[pairs[i]['root_id']].append(i)
 state_roots=list(state_roots.values());pair_roots=list(pair_roots.values());rng=np.random.default_rng(160010+a.seed)
 def sample(groups,n):return np.array([g[rng.integers(len(g))]for g in[groups[j]for j in rng.integers(len(groups),size=n)]])
 m=Model(norm,'generic'if a.kind=='generic'else'learned').cuda();folder=E/'models'/f'{a.kind}_seed{a.seed}';folder.mkdir(exist_ok=False);log=[];start=time.monotonic();batch=cfg['batch'];updates=cfg['stage1_updates'];check=cfg['stage1_checkpoints']
 if a.kind=='ae':
  opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=.0001);best=float('inf');beststate=None
  devstates=[i for i,r in enumerate(rows)if r['split']=='dev']
  for u in range(1,updates+1):
   ix=sample(state_roots,batch);z=m.encode(obj[ix],scene[ix]);loss=feature_error(m.decode(z),obj[ix],scene[ix]).mean()+.01*torch.relu(.5-z.std(0)).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
   if u in check:
    with torch.no_grad():dz=m.encode(obj[devstates],scene[devstates]);score=float(feature_error(m.decode(dz),obj[devstates],scene[devstates]).mean())
    torch.save({'state':m.state_dict(),'update':u,'score':score},folder/f'ae_rep_{u:04d}.pt');log.append({'phase':'reconstruction','update':u,'dev_observation_loss':score});print(a.kind,a.seed,log[-1],flush=True)
    if score<best:best=score;beststate={k:v.detach().clone()for k,v in m.state_dict().items()}
  m.load_state_dict(beststate);m.freeze_representation()
 if a.kind=='generic':m.freeze_representation()
 opt=torch.optim.AdamW([p for p in m.parameters()if p.requires_grad],lr=cfg['lr'],weight_decay=.0001);best=float('inf');selected=None
 for u in range(1,updates+1):
  ix=sample(pair_roots,batch);z=m.encode(obj[b[ix]],scene[b[ix]]);pred=m.transition(z,act[ix]);z1=m.encode(obj[aft[ix]],scene[aft[ix]])
  predloss=feature_error(m.decode(pred),obj[aft[ix]],scene[aft[ix]],(obj[b[ix]],scene[b[ix]])).mean();match=(pred-z1.detach()).square().mean()
  loss=predloss+.25*match
  if a.kind=='interaction':
   si=sample(state_roots,batch);zs=m.encode(obj[si],scene[si]);loss+=feature_error(m.decode(zs),obj[si],scene[si]).mean()+.01*torch.relu(.5-zs.std(0)).mean()
  if u%4==0:
   # Chain links are validated observed endpoint identity, never concatenated independent pairs.
   seq=[chains[i]for i in ix];cur=pred
   for depth in [1,2]:
    valid=[j for j,c in enumerate(seq)if len(c)>depth]
    if not valid:break
    ni=np.array([int(seq[j][depth][1:])for j in valid]);pi=cur[valid];nxt=m.transition(pi,act[ni]);target=m.encode(obj[aft[ni]],scene[aft[ni]]).detach()
    loss+=.25*(feature_error(m.decode(nxt),obj[aft[ni]],scene[aft[ni]],(obj[b[ni]],scene[b[ni]])).mean()+.25*(nxt-target).square().mean())
    upd=cur.clone();upd[valid]=nxt;cur=upd
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
  if u in check:
   m.eval()
   with torch.no_grad():
    zz=m.encode(obj[b[dev]],scene[b[dev]]);pp=m.transition(zz,act[dev]);ee=feature_error(m.decode(pp),obj[aft[dev]],scene[aft[dev]],(obj[b[dev]],scene[b[dev]])).mean(1).cpu().numpy()
    vals=[]
    for root in sorted({pairs[i]['root_id']for i in dev}):vals.append(float(ee[[j for j,i in enumerate(dev)if pairs[i]['root_id']==root]].mean()))
    score=float(np.mean(vals));latentstd=float(zz.std(0).mean());recon=float(feature_error(m.decode(zz),obj[b[dev]],scene[b[dev]]).mean())
   cp=folder/f'update{u:04d}.pt';torch.save({'state':m.state_dict(),'kind':a.kind,'seed':a.seed,'update':u,'score':score},cp);log.append({'phase':'transition','update':u,'dev_root_mean_changed_weighted_observation_loss':score,'dev_current_reconstruction_loss':recon,'latent_mean_std':latentstd,'train_loss':float(loss)});print(a.kind,a.seed,log[-1],flush=True)
   if score<best:best=score;selected=cp
   m.train()
 save(folder/'selected.json',{'path':str(selected),'sha256':sha(selected),'selection':'minimum C root-macro changed-weighted fixed-observation prediction loss; earlier tie','no_proposition_labels':True});save(E/'results'/f'stage1_{a.kind}_seed{a.seed}.json',{'log':log,'selected':str(selected),'elapsed_seconds':time.monotonic()-start,'public_file_accesses':sorted(set(access)),'model_trainable_final':sum(p.numel()for p in m.parameters()if p.requires_grad),'all_parameters':sum(p.numel()for p in m.parameters())});print('DONE',a.kind,a.seed,flush=True)
if __name__=='__main__':main()
