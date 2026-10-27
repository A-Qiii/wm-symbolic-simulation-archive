from paths import GPU_LOCK
from paths import ROOT as B, RUNTIME as P
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,math,time,hashlib,fcntl
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from models import RGBState,Dynamics
def read(p):return json.load(open(p))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor(a,integer=False):return torch.tensor(np.asarray(a),device='cuda',dtype=torch.long if integer else torch.float32)
def subset(b,ix):return {k:v[ix]for k,v in b.items()}
def state_loss(m,b):
 h,p,s,z=m(b['features']);lp=F.smooth_l1_loss(p/m.sd,b['xyz']/m.sd);ls=F.smooth_l1_loss(s/.1,b['shape']/.1);lh=F.binary_cross_entropy_with_logits(z,b['held'],pos_weight=tensor(3.));lr=F.cross_entropy(m.read_regions(p).reshape(-1,3),b['region'].reshape(-1));lgt=F.cross_entropy(m.read_regions(b['xyz']).reshape(-1,3),b['region'].reshape(-1));return lp+ls+lh+.25*(lr+lgt)
def state_score(m,b):
 with torch.no_grad():
  h,p,s,z=m(b['features']);r=m.read_regions(p).argmax(-1);held=z.sigmoid()>=.5;ok=(r==b['region']).all(-1)&(held==b['held'].bool()).all(-1)
  return {'loss':float(state_loss(m,b)),'joint_correct':int(ok.sum()),'total':len(ok),'joint':float(ok.float().mean()),'per_object_region_correct':(r==b['region']).sum(0).tolist(),'per_object_held_correct':(held==b['held'].bool()).sum(0).tolist(),'XYZ_MAE_m':(p-b['xyz']).abs().mean(0).tolist()}
def crop(b,ids,rng):
 lengths=b['length'][ids].cpu().numpy();starts=np.array([rng.integers(int(n))for n in lengths]);out={};ix=tensor(ids,True);ts=tensor(np.minimum(starts[:,None]+np.arange(5),lengths[:,None]),True)
 for k in ['h','p','shape','z','xyz','region','held']:out[k]=b[k][ix[:,None],ts]
 ta=tensor(np.minimum(starts[:,None]+np.arange(4),lengths[:,None]-1),True)
 for k in ['verb','obj','target']:out[k]=b[k][ix[:,None],ta]
 out['length']=tensor(lengths-starts,True);return out,starts
def observed(b,t):return Dynamics.pack(b['h'][:,t],b['p'][:,t],b['shape'][:,t],b['z'][:,t])
def step_loss(state,b,t,est):
 valid=(b['length']>t).float();p=state[:,:2,128:131];lp=F.smooth_l1_loss(p/est.sd,b['xyz'][:,t+1]/est.sd,reduction='none').mean((-1,-2));lr=F.cross_entropy(est.read_regions(p).reshape(-1,3),b['region'][:,t+1].reshape(-1),reduction='none').reshape(-1,2).mean(-1);lh=F.binary_cross_entropy_with_logits(state[:,:2,134],b['held'][:,t+1],reduction='none').mean(-1);return ((lp+.25*lr+.25*lh)*valid).sum()/valid.sum().clamp_min(1)
def dynamic_loss(net,b,est,score=False):
 posts=[observed(b,t)for t in range(5)];ones=[];multi=[];priors=[];root=[]
 for t in range(4):
  s=net(posts[t],b['verb'][:,t],b['obj'][:,t],b['target'][:,t],est.sd);priors.append(s);ones.append(step_loss(s,b,t,est))
 for start in range(4):
  s=posts[start]
  for t in range(start,4):
   s=net(s,b['verb'][:,t],b['obj'][:,t],b['target'][:,t],est.sd);multi.append(step_loss(s,b,t,est))
   if start==0:root.append(s)
 loss=torch.stack(ones).mean()+.5*torch.stack(multi).mean()
 if not score:return loss
 counts={k:[0,0]for k in ['current','one','root_terminal']}
 def add(name,state,t,mask):
  r=est.read_regions(state[:,:2,128:131]).argmax(-1);z=state[:,:2,134].sigmoid()>=.5;ok=(r==b['region'][:,t]).all(-1)&(z==b['held'][:,t].bool()).all(-1);counts[name][0]+=int((ok&mask).sum());counts[name][1]+=int(mask.sum())
 for t,s in enumerate(posts):add('current',s,t,b['length']>=t)
 for t,s in enumerate(priors):add('one',s,t+1,b['length']>t)
 for t,s in enumerate(root):add('root_terminal',s,t+1,b['length']==t+1)
 return {'loss':float(loss),'counts':counts,'accuracy':{k:a/n for k,(a,n)in counts.items()}}
def get_dynamic_data(rows,preds,episodes,ids):
 sx={(x['episode_id'],x['boundary']):i for i,x in enumerate(rows)};lookup={e['id']:e for e in episodes};arr={k:[]for k in ['h','p','shape','z','xyz','region','held','verb','obj','target','length']}
 for eid in ids:
  ep=lookup[eid];L=len(ep['actions']);indices=[sx[(eid,min(t,L))]for t in range(5)]
  for k in ['h','p','shape','z']:arr[k].append(preds[k][indices])
  for k in ['xyz','region','held']:arr[k].append([rows[i][k]for i in indices])
  actions=[ep['actions'][min(t,L-1)]for t in range(4)];arr['verb'].append([0 if a[0]=='Pick'else 1 for a in actions]);arr['obj'].append([0 if a[1]=='large'else 1 for a in actions]);arr['target'].append([{None:0,'A':1,'B':2,'large_top':3}[a[2]]for a in actions]);arr['length'].append(L)
 return {k:tensor(v,k in ['region','verb','obj','target','length'])for k,v in arr.items()}

def main():
 version=os.environ.get('COLLECTION_VERSION','v1');cache=B/'cache'/version/'fit';assert(cache/'complete.json').exists();plan=read(B/'configs/experiment_plan.json');rows=read(cache/'state_labels.json');feat=np.load(cache/'all_features.npz')['features'];records=[];root=B/'models'/version
 frozen_source=root/'frozen_source';frozen_source.mkdir(parents=True,exist_ok=True)
 for source in [B/'src'/n for n in ['models.py','rgb_front.py','train.py','protocol.py']]+[B/'configs/experiment_plan.json']:
  dest=frozen_source/source.name
  if dest.exists():assert dest.read_bytes()==source.read_bytes(),'Training source changed; preserve old weights and use a new version'
  else:dest.write_bytes(source.read_bytes())
 for exp,spec in plan['experiments'].items():
  if os.environ.get('WM_BLOCKS_BASELINE_SCOPE','full')=='transition_holdout'and exp!='transition':continue
  for arm,split in spec['arms'].items():
   if os.environ.get('WM_BLOCKS_BASELINE_SCOPE','full')=='transition_holdout'and arm!='holdout':continue
   trainset=set(split['train']);devset=set(split['development']);active=[i for i,r in enumerate(rows)if r['episode_id']in trainset|devset];arm_rows=[rows[i]for i in active];arm_feat=feat[active];ii=[i for i,r in enumerate(arm_rows)if r['episode_id']in trainset];jj=[i for i,r in enumerate(arm_rows)if r['episode_id']in devset];mu=arm_feat[ii].reshape(-1,feat.shape[-1]).mean(0);sd=np.maximum(arm_feat[ii].reshape(-1,feat.shape[-1]).std(0),.001);x=((arm_feat-mu)/sd).astype(np.float32);base=root/exp/arm;base.mkdir(parents=True,exist_ok=True);np.savez_compressed(base/'feature_normalization.npz',mean=mu,std=sd);data={'features':tensor(x),'xyz':tensor([r['xyz']for r in arm_rows]),'shape':tensor([r['shape']for r in arm_rows]),'region':tensor([r['region']for r in arm_rows],True),'held':tensor([r['held']for r in arm_rows])};dev=subset(data,jj)
   for seed in plan['training_seeds']:
    statepath=base/f'state_seed{seed}.pt';torch.manual_seed(32000+seed);est=RGBState(feat.shape[-1]).cuda();rng=np.random.default_rng(32000+seed)
    if statepath.exists():est.load_state_dict(torch.load(statepath,weights_only=False)['model']);selected=torch.load(statepath,weights_only=False)['selected']
    else:
     opt=torch.optim.AdamW(est.parameters(),lr=3e-4,weight_decay=1e-4);best=None;hist=[];norms=[];losses=[];exposure=np.zeros(len(arm_rows),dtype=np.int64)
     for step in range(1,3001):
      ix=rng.choice(ii,64);np.add.at(exposure,ix,1);batch=subset(data,ix);lr=3e-5+2.7e-4*(1+math.cos(math.pi*step/3000))/2
      for pg in opt.param_groups:pg['lr']=lr
      opt.zero_grad(set_to_none=True);loss=state_loss(est,batch);assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(est.parameters(),1.);assert torch.isfinite(gn);norms.append(float(gn));losses.append(float(loss.detach()));opt.step()
      if step%250==0:
       ev=state_score(est,dev);ev['by_recipe']={name:state_score(est,subset(data,[i for i in jj if arm_rows[i]['episode_id'].split('__')[1]==name]))for name in sorted({arm_rows[i]['episode_id'].split('__')[1]for i in jj})};key=(int(ev['joint']>=.98),-ev['loss'])if ev['joint']>=.98 else(0,ev['joint'],-ev['loss']);row={'step':step,'development':ev,'eligible':ev['joint']>=.98,'optimization':{'mean_training_loss':float(np.mean(losses)),'mean_preclip_gradient_norm':float(np.mean(norms)),'clip_fraction':float(np.mean(np.array(norms)>1))}};hist.append(row);norms=[];losses=[]
       if best is None or key>best:best=key;selected=row;weights={k:v.detach().cpu().clone()for k,v in est.state_dict().items()};selected_exposure=exposure.copy()
       save(B/f'runs/{version}/{exp}_{arm}_state_seed{seed}.json',hist);save(B/'STATUS.json',{'stage':'state_training','experiment':exp,'arm':arm,'seed':seed,'step':step,'development':ev});print('STATE_DEV',exp,arm,seed,step,ev,flush=True)
     torch.save({'model':weights,'selected':selected,'seed':seed,'train_episodes':split['train'],'development_episodes':split['development'],'old_task_weights':False},statepath);save(statepath.with_suffix('.exposure.json'),{'full_budget_updates':3000,'selected_updates':selected['step'],'state_sample_exposures_full':exposure.tolist(),'state_sample_exposures_selected':selected_exposure.tolist(),'state_keys':[[r['episode_id'],r['boundary']]for r in arm_rows]});est.load_state_dict(weights);del opt
    est.eval().requires_grad_(False)
    fit_score=state_score(est,subset(data,ii));fit_score['by_recipe']={name:state_score(est,subset(data,[i for i in ii if arm_rows[i]['episode_id'].split('__')[1]==name]))for name in sorted({arm_rows[i]['episode_id'].split('__')[1]for i in ii})};save(statepath.with_suffix('.training_fit.json'),fit_score)
    if not selected['eligible']:
     save(B/'STATUS.json',{'stage':'development_state_failure','experiment':exp,'arm':arm,'seed':seed,'selected':selected,'test_not_evaluated':True});raise RuntimeError('Basic current-state development gate failed; diagnose before test')
    preds={k:[]for k in ['h','p','shape','z']}
    with torch.no_grad():
     for start in range(0,len(x),128):
      for k,v in zip(preds,est(data['features'][start:start+128])):preds[k].append(v.cpu().numpy())
    preds={k:np.concatenate(v)for k,v in preds.items()};dynamic={s:get_dynamic_data(arm_rows,preds,spec['episodes'],split[s])for s in ['train','development']};draw_hashes=[];init_hashes=[]
    for condition in plan['conditions']:
     path=base/f'{condition}_seed{seed}.pt'
     if path.exists():rec=read(path.with_suffix('.json'));records.append(rec);draw_hashes.append(rec['sampling_sha256']);init_hashes.append(rec['shared_initialization_sha256']);continue
     torch.manual_seed(33000+seed);net=Dynamics(condition).cuda();initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()for k,v in net.state_dict().items()if not k.startswith('motion.interaction.'))).hexdigest();init_hashes.append(initial);opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(34000+seed);best=None;hist=[];draw=hashlib.sha256();norms=[];losses=[];exposure=np.zeros(len(split['train']),dtype=np.int64)
     for step in range(1,3001):
      ids=rng.integers(len(split['train']),size=16);np.add.at(exposure,ids,1);batch,starts=crop(dynamic['train'],ids,rng);draw.update(ids.tobytes()+starts.tobytes());lr=3e-5+2.7e-4*(1+math.cos(math.pi*step/3000))/2
      for pg in opt.param_groups:pg['lr']=lr
      opt.zero_grad(set_to_none=True)
      with torch.autocast('cuda',dtype=torch.bfloat16):loss=dynamic_loss(net,batch,est)
      assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(gn);norms.append(float(gn));losses.append(float(loss.detach()));opt.step()
      if step%250==0:
       with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):ev=dynamic_loss(net,dynamic['development'],est,True)
       eligible=min(ev['accuracy'].values())>=.98;key=(1,-ev['loss'])if eligible else(0,min(ev['accuracy'].values()),-ev['loss']);row={'step':step,'development':ev,'eligible':eligible,'optimization':{'mean_training_loss':float(np.mean(losses)),'mean_preclip_gradient_norm':float(np.mean(norms)),'clip_fraction':float(np.mean(np.array(norms)>1))}};hist.append(row);norms=[];losses=[]
       if best is None or key>best:best=key;selected_dyn=row;weights={k:v.detach().cpu().clone()for k,v in net.state_dict().items()};selected_exposure=exposure.copy()
       save(B/f'runs/{version}/{exp}_{arm}_{condition}_seed{seed}.json',hist);save(B/'STATUS.json',{'stage':'dynamics_training','experiment':exp,'arm':arm,'condition':condition,'seed':seed,'step':step,'development':ev});print('DYNAMICS_DEV',exp,arm,condition,seed,step,ev,flush=True)
     torch.save({'model':weights,'condition':condition,'seed':seed,'selected':selected_dyn},path);net.load_state_dict(weights)
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):training_fit=dynamic_loss(net,dynamic['train'],est,True)
     rec={'experiment':exp,'arm':arm,'condition':condition,'seed':seed,'checkpoint':str(path),'sha256':sha(path),'state_checkpoint':str(statepath),'state_sha256':sha(statepath),'state_selected':selected,'selected':selected_dyn,'sampling_sha256':draw.hexdigest(),'shared_initialization_sha256':initial,'updates':3000,'train_episodes':360,'train_transitions':960,'parameters':sum(p.numel()for p in net.parameters()),'selected_training_fit':training_fit};save(path.with_suffix('.json'),rec);save(path.with_suffix('.exposure.json'),{'episode_ids':split['train'],'episode_exposures_full':exposure.tolist(),'episode_exposures_selected':selected_exposure.tolist(),'full_updates':3000,'selected_updates':selected_dyn['step']});records.append(rec);draw_hashes.append(draw.hexdigest());del opt,net;torch.cuda.empty_cache()
    assert draw_hashes[0]==draw_hashes[1]and init_hashes[0]==init_hashes[1]
    del dynamic,est;torch.cuda.empty_cache();save(B/'configs/model_progress.json',records)
   del data;torch.cuda.empty_cache()
 passed=all(r['selected']['eligible']for r in records);save(B/'configs/all_models_frozen.json',{'models':records,'development_gate_passed':passed,'complete':len(records)==(6 if os.environ.get('WM_BLOCKS_BASELINE_SCOPE','full')=='transition_holdout'else 24),'baseline_scope':os.environ.get('WM_BLOCKS_BASELINE_SCOPE','full'),'plan_sha256':sha(B/'configs/experiment_plan.json'),'all_task_weights_new':True,'test_not_seen':True});save(B/'STATUS.json',{'stage':'models_frozen'if passed else'development_dynamics_failure','models':len(records),'test_not_evaluated':True})
 if not passed:raise RuntimeError('Basic dynamics gate failed; test remains locked')

if __name__=='__main__':
 lock=GPU_LOCK.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
 try:main()
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
