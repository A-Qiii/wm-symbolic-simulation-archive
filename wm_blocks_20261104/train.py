import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from common import *
from dataset import specifications,summary
from models import RGBState,HeldHead,ObjectDynamics,pack
import torch,numpy as np,math,argparse
from torch.nn import functional as F
def tensor(x,long=False):return torch.as_tensor(np.asarray(x),device='cuda',dtype=torch.long if long else torch.float32)
def subset(x,ix):return {k:v[ix]for k,v in x.items()}
def held_class(y):
 assert bool(((y==1).sum(-1)<=1).all());return torch.where((y>=0).all(-1),torch.where(y.max(-1).values>0,y.argmax(-1)+1,0),-1)
def ce(logits,y):
 loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-1,reduction='none');return loss.sum()/(y>=0).sum().clamp_min(1)
def labels_held(prob):
 full=torch.cat([(1-prob.sum(-1,keepdim=True)).clamp_min(0),prob],-1);return F.one_hot(full.argmax(-1),prob.shape[-1]+1)[...,1:].bool()
def state_loss(m,b):
 h,p,s,z=m(b['features']);mask=b['held']>=0
 lh=(F.binary_cross_entropy_with_logits(z,b['held'].clamp_min(0),pos_weight=tensor(3.),reduction='none')*mask).sum()/mask.sum().clamp_min(1)
 return F.smooth_l1_loss(p/m.sd,b['xyz']/m.sd)+F.smooth_l1_loss(s/.1,b['shape']/.1)+lh+.25*(ce(m.read_regions(p,s),b['region'])+ce(m.read_regions(b['xyz'],b['shape']),b['region']))
def state_score(m,b):
 with torch.no_grad():
  h,p,s,z=m(b['features']);r=m.read_regions(p,s).argmax(-1);held=z>0;valid=(b['region']>=0).all(-1)&(b['held']>=0).all(-1);ok=(r==b['region']).all(-1)&(held==b['held'].bool()).all(-1)
  return dict(loss=float(state_loss(m,b)),joint=int((ok&valid).sum())/max(1,int(valid.sum())),correct=int((ok&valid).sum()),total=int(valid.sum()),xyz_mae_m=float((p-b['xyz']).abs().mean()))
def freeze_weights(m):return {k:v.detach().cpu().clone()for k,v in m.state_dict().items()}
def optstep(m,opt,loss,step,budget):
 for g in opt.param_groups:g['lr']=3e-5+2.7e-4*(1+math.cos(math.pi*step/budget))/2
 opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
def head_loss(m,h,y,kind):
 logits=m(h)
 if kind=='ce':return ce(logits,held_class(y))
 mask=y>=0;return (F.binary_cross_entropy_with_logits(logits[:,1:],y.clamp_min(0),reduction='none')*mask).sum()/mask.sum().clamp_min(1)
def head_score(m,h,y,kind):
 with torch.no_grad():
  z=m(h);pred=labels_held(z.softmax(-1)[:,1:])if kind=='ce'else z[:,1:]>0;valid=(y>=0).all(-1);ok=(pred==y.bool()).all(-1)&valid
  return dict(loss=float(head_loss(m,h,y,kind)),correct=int(ok.sum()),total=int(valid.sum()),joint=float(ok.sum()/valid.sum()),multiple=int((pred.sum(-1)>1).sum()))
def build_data(rows,obs,eps,ids):
 lookup={(r['episode_id'],r['boundary']):i for i,r in enumerate(rows)};spec={e['id']:e for e in eps};arr={k:[]for k in ['observed','xyz','region','held','verb','obj','target','length']}
 for eid in ids:
  e=spec[eid];L=e['length'];ii=[lookup[(eid,min(t,L))]for t in range(5)]
  arr['observed'].append(obs[ii]);arr['length'].append(L)
  for k in ['xyz','region','held']:arr[k].append([rows[i][k]for i in ii])
  aa=[e['actions'][min(t,L-1)]for t in range(4)]
  arr['verb'].append([int(a[0]=='Place')for a in aa]);arr['obj'].append([NAMES.index(a[1])for a in aa]);arr['target'].append([{None:0,'A':1,'B':2,'large_top':3}[a[2]]for a in aa])
 return {k:tensor(v,k in ['region','verb','obj','target','length'])for k,v in arr.items()}
def crop(b,ids,rng):
 L=b['length'][ids].cpu().numpy();starts=np.array([rng.integers(n)for n in L]);ix=tensor(ids,True);times=tensor(np.minimum(starts[:,None]+np.arange(5),L[:,None]),True);ta=tensor(np.minimum(starts[:,None]+np.arange(4),L[:,None]-1),True);out={}
 for k in ['observed','xyz','region','held']:out[k]=b[k][ix[:,None],times]
 for k in ['verb','obj','target']:out[k]=b[k][ix[:,None],ta]
 out['length']=tensor(L-starts,True);return out,starts
def step_loss(s,b,t,est):
 valid=b['length']>t;p=s[...,:3];size=s[...,3:6];prob=s[...,6];lp=F.smooth_l1_loss(p/est.sd,b['xyz'][:,t+1]/est.sd,reduction='none').mean((-1,-2))
 reg=b['region'][:,t+1];lr=F.cross_entropy(est.read_regions(p,size).reshape(-1,3),reg.reshape(-1),ignore_index=-1,reduction='none').reshape(-1,3);lr=lr.sum(-1)/(reg>=0).sum(-1).clamp_min(1)
 hc=held_class(b['held'][:,t+1]);full=torch.cat([(1-prob.sum(-1,keepdim=True)).clamp_min(1e-7),prob.clamp_min(1e-7)],-1);lh=F.nll_loss(full.log(),hc,ignore_index=-1,reduction='none')
 return ((lp+.25*lr+.25*lh)*valid).sum()/valid.sum().clamp_min(1)
def dynamic_loss(net,b,est,score=False):
 ones=[];multis=[];root=[];priors=[]
 for t in range(4):
  s=net(b['observed'][:,t],b['verb'][:,t],b['obj'][:,t],b['target'][:,t],est.sd);ones.append(step_loss(s,b,t,est));priors.append(s)
 for start in range(4):
  s=b['observed'][:,start]
  for t in range(start,4):
   s=net(s,b['verb'][:,t],b['obj'][:,t],b['target'][:,t],est.sd);multis.append(step_loss(s,b,t,est))
   if start==0:root.append(s)
 loss=torch.stack(ones).mean()+.5*torch.stack(multis).mean()
 if not score:return loss
 counts={k:[0,0]for k in ['current','one','terminal']}
 def add(k,s,t,mask):
  valid=mask&(b['region'][:,t]>=0).all(-1)&(b['held'][:,t]>=0).all(-1);ok=(est.read_regions(s[...,:3],s[...,3:6]).argmax(-1)==b['region'][:,t]).all(-1)&(labels_held(s[...,6])==b['held'][:,t].bool()).all(-1);counts[k][0]+=int((ok&valid).sum());counts[k][1]+=int(valid.sum())
 for t in range(5):add('current',b['observed'][:,t],t,b['length']>=t)
 for t in range(4):add('one',priors[t],t+1,b['length']>t);add('terminal',root[t],t+1,b['length']==t+1)
 return dict(loss=float(loss),counts=counts,accuracy={k:a/max(1,n)for k,(a,n)in counts.items()})
def main():
 cfg=plan();rows=read(ROOT/'cache/fit/rows.json');feat=np.load(ROOT/'cache/fit/all_features.npz')['features'];eps=[summary(e)for e in specifications('fit')];good={e['id']for e in eps if e['physical_valid']and e['complete']and e['length']>0};paired_exclusions=[]
 # Keep fitting budgets matched if either controlled replacement has a physical
 # failure. Retain both raw attempts; record the valid counterpart exclusion.
 for group in {e['group']for e in eps if e['split']=='train'}:
  pair=[e for e in eps if e['group']==group and e['arm']in ['heldout','covered']]
  if pair and not all(e['id']in good for e in pair):
   for e in pair:
    if e['id']in good:paired_exclusions.append(e['id']);good.remove(e['id'])
 eps=[e for e in eps if e['id']in good]
 lengths=[[e['length']for e in eps if e['split']=='train'and e['arm']in ['shared',arm]]for arm in cfg['arms']];assert lengths[0]==lengths[1], 'Matched sampling lengths differ before training'
 ii=[i for i,r in enumerate(rows)if r['episode_id']in good and r['split']=='train'and r['arm']!='covered'];jj=[i for i,r in enumerate(rows)if r['episode_id']in good and r['split']=='development'];mu=feat[ii].reshape(-1,1567).mean(0);sd=feat[ii].reshape(-1,1567).std(0).clip(.001);np.savez_compressed(ROOT/'models/normalization.npz',mean=mu,std=sd)
 data={'features':tensor((feat-mu)/sd),'xyz':tensor([r['xyz']for r in rows]),'shape':tensor([r['shape']for r in rows]),'region':tensor([r['region']for r in rows],True),'held':tensor([r['held']for r in rows])};dev=subset(data,jj);records=[]
 save(ROOT/'reports/training_inputs.json',dict(training_visual_rows=len(ii),development_rows=len(jj),excluded_episodes=[e['id']for e in [summary(e)for e in specifications('fit')]if e['id']not in good],valid_counterparts_excluded_for_matching=paired_exclusions,task_weights_from_scratch=True))
 for seed in cfg['seeds']:
  sp=ROOT/f'models/state_seed{seed}.pt';torch.manual_seed(52000+seed);rng=np.random.default_rng(52000+seed);est=RGBState().cuda();st=time.monotonic()
  if not sp.exists():
   opt=torch.optim.AdamW(est.parameters(),lr=3e-4,weight_decay=1e-4);best=None;history=[]
   for step in range(1,cfg['state_updates']+1):
    b=subset(data,rng.choice(ii,64));optstep(est,opt,state_loss(est,b),step,cfg['state_updates'])
    if step%500==0:
     ev=state_score(est,dev);key=(ev['joint'],-ev['loss']);row=dict(step=step,development=ev,seconds=time.monotonic()-st);history.append(row)
     if best is None or key>best:best=key;selected=row;weights=freeze_weights(est)
     save(ROOT/f'logs/state_seed{seed}.json',history);status('state_training',seed=seed,**row)
   torch.save(dict(model=weights,selected=selected),sp);del opt
  cp=torch.load(sp,weights_only=False,map_location='cuda');est.load_state_dict(cp['model']);est.eval().requires_grad_(False)
  with torch.no_grad():h,p,size,z=est(data['features'])
  for kind in ['bce','ce']:
   hp=ROOT/f'models/held_{kind}_seed{seed}.pt';torch.manual_seed(53000+seed);head=HeldHead().cuda();rng=np.random.default_rng(53000+seed)
   if not hp.exists():
    opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4);best=None;history=[];draw=hashlib.sha256()
    for step in range(1,cfg['held_updates']+1):
     ix=rng.choice(ii,64);draw.update(ix.tobytes());optstep(head,opt,head_loss(head,h[ix],data['held'][ix],kind),step,cfg['held_updates'])
     if step%250==0:
      ev=head_score(head,h[jj],data['held'][jj],kind);row=dict(step=step,development=ev);history.append(row);key=(ev['joint'],-ev['loss'])
      if best is None or key>best:best=key;selected=row;weights=freeze_weights(head)
      save(ROOT/f'logs/held_{kind}_seed{seed}.json',history);status('held_calibration',seed=seed,kind=kind,**row)
    torch.save(dict(model=weights,selected=selected,sampling_sha256=draw.hexdigest()),hp);del opt
   head.load_state_dict(torch.load(hp,weights_only=False,map_location='cuda')['model']);head.eval().requires_grad_(False)
   if kind=='ce':
    with torch.no_grad():obs=pack(p,size,head.probabilities(h)).cpu().numpy()
   del head
  for arm in cfg['arms']:
   ids={s:[e['id']for e in eps if e['split']==s and (s=='development'or e['arm']in ['shared',arm])]for s in ['train','development']};dynamic={s:build_data(rows,obs,eps,ids[s])for s in ids}
   for kind in cfg['models']:
    dest=ROOT/f'models/{arm}_{kind}_seed{seed}.pt';recpath=dest.with_suffix('.json')
    if recpath.exists():records.append(read(recpath));continue
    torch.manual_seed(54000+seed);net=ObjectDynamics(kind).cuda();initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()for k,v in net.state_dict().items()if not k.startswith('effect.'))).hexdigest();opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(55000+seed);best=None;history=[];draw=hashlib.sha256();exposure=dict(prior_draws=0,new_draws=0,mid_start_draws=0,remaining_macro_steps=0);length_by_id={e['id']:e['length']for e in eps};st=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for step in range(1,cfg['dynamic_updates']+1):
     ix=rng.integers(len(ids['train']),size=cfg['batch']);b,starts=crop(dynamic['train'],ix,rng);draw.update(ix.tobytes()+starts.tobytes())
     for sample,start in zip(ix,starts):
      eid=ids['train'][int(sample)];exposure['prior_draws'if eid.startswith('prior_')else'new_draws']+=1;exposure['mid_start_draws']+=int(start>0);exposure['remaining_macro_steps']+=length_by_id[eid]-int(start)
     with torch.autocast('cuda',dtype=torch.bfloat16):loss=dynamic_loss(net,b,est)
     optstep(net,opt,loss,step,cfg['dynamic_updates'])
     if step%500==0:
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):ev=dynamic_loss(net,dynamic['development'],est,True)
      key=(min(ev['accuracy'].values()),-ev['loss']);row=dict(step=step,development=ev,seconds=time.monotonic()-st);history.append(row)
      if best is None or key>best:best=key;selected=row;weights=freeze_weights(net)
      save(ROOT/f'logs/{arm}_{kind}_seed{seed}.json',history);status('dynamics_training',arm=arm,kind=kind,seed=seed,**row)
    torch.save(dict(model=weights,selected=selected,kind=kind),dest)
    rec=dict(arm=arm,kind=kind,seed=seed,checkpoint=str(dest),sha256=sha(dest),state_sha256=sha(sp),held_sha256=sha(ROOT/f'models/held_ce_seed{seed}.pt'),selected=selected,updates=cfg['dynamic_updates'],sampling_sha256=draw.hexdigest(),shared_initialization_sha256=initial,train_episodes=len(ids['train']),train_transitions=sum(e['length']for e in eps if e['id']in ids['train']),development_episodes=len(ids['development']),parameters=sum(x.numel()for x in net.parameters()),actual_sampling_exposure=exposure,seconds=time.monotonic()-st,peak_cuda_bytes=torch.cuda.max_memory_allocated());save(recpath,rec);records.append(rec);del net,opt;torch.cuda.empty_cache()
   del dynamic
  del est,h,p,size,z,obs;torch.cuda.empty_cache()
 for seed in cfg['seeds']:
  rr=[r for r in records if r['seed']==seed];assert len({r['sampling_sha256']for r in rr})==1 and len({r['shared_initialization_sha256']for r in rr})==1,'Budget/sample mismatch'
 save(ROOT/'configs/models_frozen.json',dict(models=records,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),plan_sha256=sha(ROOT/'configs/plan.json'),source_hashes={p.name:sha(p)for p in Path(__file__).resolve().parent.glob('*.py')},test_evaluated=False));status('models_frozen',models=len(records),seeds=3)
if __name__=='__main__':gpu=lock();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True);main()
