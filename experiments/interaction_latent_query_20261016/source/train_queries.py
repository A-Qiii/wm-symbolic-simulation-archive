from common import *
from model import Model,Query,probs
import torch,torch.nn.functional as F,fcntl,argparse,time
def normalize(z,stats):return (z-stats['mean'])/stats['std']
def loss_by_state(logits,y,v,weight):
 out=[]
 for lo,hi in [(0,3),(3,6)]:
  l=F.binary_cross_entropy_with_logits(logits[:,lo:hi],y[:,lo:hi],pos_weight=weight[lo:hi],reduction='none')*v[:,lo:hi];out.append(l.sum(1)/v[:,lo:hi].sum(1).clamp(min=1))
 d=F.cross_entropy(logits[:,6:],y[:,6:].argmax(1),weight=weight[6:],reduction='none')*v[:,6:].all(1);out.append(d)
 return torch.stack(out,-1).mean(-1)
def load_stage1(kind,seed,device='cuda'):
 norm=dict(np.load(E/'data/public/normalization.npz'));sel=read(E/'models'/f'{kind}_seed{seed}'/'selected.json');assert sha(sel['path'])==sel['sha256'];cp=torch.load(sel['path'],map_location=device,weights_only=True)
 m=Model(norm,'generic'if kind=='generic'else'learned').to(device);m.load_state_dict(cp['state']);m.eval().requires_grad_(False);return m,sel
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['generic','ae','interaction'],required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
 cfg=read(E/'configs/protocol.json');budget=read(E/'configs/semantic_budget.json');rows=read(E/'data/public/states.json');label=read(E/'data/_private/budget_labels.json');ix=budget['train_state_indices'];dv=budget['development_selection_state_indices'];ab=[i for i,r in enumerate(rows)if r['split']=='train'];m,sel=load_stage1(a.kind,a.seed)
 with np.load(E/'data/public/inputs.npz')as d:o=torch.tensor(d['objects'],device='cuda');s=torch.tensor(d['scene'],device='cuda')
 with torch.no_grad():z=m.encode(o,s)
 mean=torch.cat([z[ab,:3].mean((0,1))[None].expand(3,-1),z[ab,3].mean(0)[None]],0);std=torch.cat([z[ab,:3].std((0,1))[None].expand(3,-1),z[ab,3].std(0)[None]],0).clamp(min=.1);stats={'mean':mean,'std':std};zn=normalize(z,stats)
 y=torch.zeros(len(rows),9,device='cuda');v=torch.zeros_like(y)
 # This run opens only the budgeted train and development selection labels.
 for i in ix+dv:y[i]=torch.tensor(label[str(i)]['y'],device='cuda');v[i]=torch.tensor(label[str(i)]['v'],device='cuda')
 weight=torch.ones(9,device='cuda')
 for lo,hi in[(0,3),(3,6)]:
  pos=(y[ix,lo:hi]*v[ix,lo:hi]).sum();neg=((1-y[ix,lo:hi])*v[ix,lo:hi]).sum();weight[lo:hi]=(neg/pos.clamp(min=1)).clamp(1,20)
 count=(y[ix,6:]*v[ix,6:]).sum(0);weight[6:]=count.sum()/(3*count.clamp(min=1));torch.manual_seed(3000+a.seed);q=Query().cuda();opt=torch.optim.AdamW(q.parameters(),lr=.001,weight_decay=.001);rng=np.random.default_rng(6000+a.seed);groups=[[i for i in ix if rows[i]['root_id']==r]for r in sorted({rows[i]['root_id']for i in ix})]
 folder=E/'models'/f'{a.kind}_seed{a.seed}'/'queries';folder.mkdir(exist_ok=False);best=float('inf');selected=None;log=[];start=time.monotonic()
 for u in range(1,cfg['stage2_updates']+1):
  take=[g[rng.integers(len(g))]for g in[groups[j]for j in rng.integers(len(groups),size=32)]];loss=loss_by_state(q(zn[take]),y[take],v[take],weight).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(q.parameters(),1);opt.step()
  if u in cfg['stage2_checkpoints']:
   q.eval()
   with torch.no_grad():losses=loss_by_state(q(zn[dv]),y[dv],v[dv],weight).cpu().numpy();pr=probs(q(zn[dv])).cpu().numpy()
   rootmeans=[float(losses[[j for j,i in enumerate(dv)if rows[i]['root_id']==root]].mean())for root in sorted({rows[i]['root_id']for i in dv})];score=float(np.mean(rootmeans));path=folder/f'update{u:04d}.pt';torch.save({'state':q.state_dict(),'normalization':stats,'update':u,'score':score,'representation_sha256':sel['sha256']},path);log.append({'update':u,'development_selection_loss':score,'training_loss':float(loss)});print(a.kind,a.seed,log[-1],flush=True)
   if score<best:best=score;selected=path
   q.train()
 assert sha(sel['path'])==sel['sha256'];assert all(p.grad is None for p in m.parameters());save(folder/'selected.json',{'path':str(selected),'sha256':sha(selected),'selection':'minimum root-macro C72 query loss; earlier tie','train_states':len(ix),'dev_selection_states':len(dv),'predicted_state_training':False});save(E/'results'/f'query_{a.kind}_seed{a.seed}.json',{'log':log,'selected':str(selected),'class_weights':weight.cpu().tolist(),'frozen_stage1_sha256':sel['sha256'],'semantic_head_parameters':sum(p.numel()for p in q.parameters()),'elapsed_seconds':time.monotonic()-start,'all_stage1_gradients_none':True});print('QUERY_DONE',a.kind,a.seed,flush=True)
def load_query(kind,seed,device='cuda'):
 m,sel=load_stage1(kind,seed,device);qs=read(E/'models'/f'{kind}_seed{seed}'/'queries/selected.json');assert sha(qs['path'])==qs['sha256'];cp=torch.load(qs['path'],map_location=device,weights_only=True);assert cp['representation_sha256']==sel['sha256'];q=Query().to(device);q.load_state_dict(cp['state']);q.eval().requires_grad_(False);return m,q,cp['normalization']
if __name__=='__main__':main()
