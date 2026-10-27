from paths import GPU_LOCK
from paths import ROOT as B, RUNTIME as P
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,math,hashlib,fcntl,shutil
from pathlib import Path
import numpy as np,torch
from models import RGBState,Dynamics
from models_v2 import StateActorDynamics
from models_v3 import DiagonalStateDynamics
from train import read,save,tensor,get_dynamic_data,crop,dynamic_loss,observed
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def data(episodes,rows,features,est,norm):
 x=tensor((features-norm['mean'])/norm['std']);pred={k:[]for k in ['h','p','shape','z']}
 for i in range(0,len(x),128):
  for k,v in zip(pred,est(x[i:i+128])):pred[k].append(v.cpu().numpy())
 return get_dynamic_data(rows,{k:np.concatenate(v)for k,v in pred.items()},episodes,[e['id']for e in episodes])
@torch.no_grad()
def rollout(net,b,est,start=0):
 s=observed(b,start)
 for t in range(start,4):s=net(s,b['verb'][:,t],b['obj'][:,t],b['target'][:,t],est.sd)
 r=est.read_regions(s[:,:2,128:131]).argmax(-1);h=s[:,:2,134].sigmoid()>=.5;correct=(r==b['region'][:,4]).all(-1)&(h==b['held'][:,4].bool()).all(-1);bb=(r==torch.tensor([1,1],device=r.device)).all(-1)&~h.any(-1)
 return {'joint_correct':int(correct.sum()),'denominator':len(correct),'predicted_both_B':int(bb.sum()),'predicted_region':r.cpu().tolist()}
def main():
 plan=read(B/'configs/v3_plan.json');assert(B/'cache/v2/development_cross/complete.json').exists();original=read(B/'configs/experiment_plan.json')['experiments']['transition'];arm=original['arms']['holdout'];oldmodels=read(B/'configs/all_models_frozen.json')['models'];meta=read(B/'cache/v1/fit/state_labels.json');feat=np.load(B/'cache/v1/fit/all_features.npz')['features'];cache=B/'cache/v2/development_cross';cm=read(cache/'state_rows.json');cf=np.load(cache/'all_features.npz')['features'];cross_eps=[read(p)for p in sorted((B/'data/v2/development_cross').glob('*/episode.json'))];ce=[{'id':e['id'],'actions':[a['action']for a in e['actions']]}for e in cross_eps];lookup={e['id']:e for e in cross_eps};cr=[]
 for r in cm:
  s=lookup[r['episode']]['states'][r['boundary']];cr.append({'episode_id':r['episode'],'boundary':r['boundary'],'xyz':[s['objects'][n]['position']for n in ['large','small']],'shape':[[s['objects'][n]['aabb'][1][j]-s['objects'][n]['aabb'][0][j]for j in range(3)]for n in ['large','small']],'held':[s['objects'][n]['held']for n in ['large','small']],'region':[s['objects'][n]['region']for n in ['large','small']]})
 root=B/'models/v3';root.mkdir(exist_ok=True,parents=True);frozen=root/'frozen_source';frozen.mkdir(exist_ok=True)
 for src in [B/'src'/n for n in ['models.py','models_v2.py','models_v3.py','train.py','train_v3.py','rgb_front.py']]+[B/'configs/v3_plan.json']:
  dest=frozen/src.name
  if dest.exists():assert src.read_bytes()==dest.read_bytes()
  else:shutil.copy2(src,dest)
 records=[]
 for seed in plan['seeds']:
  old=next(r for r in oldmodels if r['experiment']=='transition'and r['arm']=='holdout'and r['condition']=='motion_matrix'and r['seed']==seed);assert sha(old['checkpoint'])==old['sha256'];normpath=Path(old['checkpoint']).parent/'feature_normalization.npz';norm=np.load(normpath);est=RGBState(1567).cuda();est.load_state_dict(torch.load(old['state_checkpoint'],map_location='cuda',weights_only=False)['model']);est.eval().requires_grad_(False);dynamic={}
  for split in ['train','development']:
   ids=set(arm[split]);ii=[i for i,r in enumerate(meta)if r['episode_id']in ids];eps=[next(e for e in original['episodes']if e['id']==eid)for eid in arm[split]];dynamic[split]=data(eps,[meta[i]for i in ii],feat[ii],est,norm)
  cross=data(ce,cr,cf,est,norm);load_ix=[i for i,eid in enumerate(arm['development'])if eid.endswith('__A_load_then_move')];load={k:v[load_ix]for k,v in dynamic['development'].items()}
  baseline=Dynamics('motion_matrix').cuda();baseline.load_state_dict(torch.load(old['checkpoint'],map_location='cuda',weights_only=False)['model']);baseline.eval();base={'cross_from_root':rollout(baseline,cross,est),'cross_after_pick':rollout(baseline,cross,est,1),'correct_load_after_pick':rollout(baseline,load,est,1)};save(B/f'reports/v3/diagnosis/development_baseline_seed{seed}.json',base);print('V1_DEV_CHALLENGE',seed,base,flush=True);del baseline
  dest=root/f'diagonal_state_matrix_seed{seed}.pt'
  if dest.exists():records.append(read(dest.with_suffix('.json')));continue
  torch.manual_seed(33000+seed);net=DiagonalStateDynamics().cuda();torch.manual_seed(33000+seed);reference=StateActorDynamics().cuda();a=net.state_dict();bb=reference.state_dict();assert all(torch.equal(v,bb[k])for k,v in a.items()if not k.startswith('motion.interaction.4.'));del reference
  opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(34000+seed);draw=hashlib.sha256();hist=[];best=None;exposure=np.zeros(360,dtype=np.int64)
  for step in range(1,3001):
   ids=rng.integers(360,size=16);np.add.at(exposure,ids,1);batch,starts=crop(dynamic['train'],ids,rng);draw.update(ids.tobytes()+starts.tobytes());lr=3e-5+2.7e-4*(1+math.cos(math.pi*step/3000))/2
   for pg in opt.param_groups:pg['lr']=lr
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=dynamic_loss(net,batch,est)
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
   if step%250==0:
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):ev=dynamic_loss(net,dynamic['development'],est,True)
    cross_score=rollout(net,cross,est,1);load_score=rollout(net,load,est,1);eligible=min(ev['accuracy'].values())>=.98;key=(1,-ev['loss'])if eligible else(0,min(ev['accuracy'].values()),-ev['loss']);row={'step':step,'development':ev,'eligible':eligible,'development_cross':cross_score,'development_correct_load':load_score,'challenge_used_for_gradient_or_checkpoint_selection':False};hist.append(row)
    if best is None or key>best:best=key;selected=row;weights={k:v.detach().cpu().clone()for k,v in net.state_dict().items()};selected_exposure=exposure.copy()
    save(B/f'runs/v3/seed{seed}.json',hist);save(B/'STATUS.json',{'version':'v3','stage':'diagonal_influence_training','seed':seed,'step':step,'development':ev,'cross':cross_score,'load':load_score});print('V3_DEV',seed,step,ev,cross_score,load_score,flush=True)
  assert draw.hexdigest()==old['sampling_sha256'],'Training samples/middle-start draws differ from v1'
  torch.save({'model':weights,'selected':selected,'seed':seed,'intervention':'actor unchanged from v2; learned diagonal influence replaces full matrix','pretrained_dynamics':False},dest);net.load_state_dict(weights);final_cross=rollout(net,cross,est,1);final_load=rollout(net,load,est,1);gate=selected['eligible']and final_cross['joint_correct']>=19 and final_load['joint_correct']>=19
  rec={'experiment':'transition','arm':'holdout','condition':'diagonal_state_matrix','seed':seed,'checkpoint':str(dest),'sha256':sha(dest),'state_checkpoint':old['state_checkpoint'],'state_sha256':old['state_sha256'],'normalization':str(normpath),'baseline':old,'selected':selected,'development_cross':final_cross,'development_correct_load':final_load,'development_gate_passed':gate,'updates':3000,'train_episodes':360,'train_transitions':960,'sampling_sha256':draw.hexdigest(),'parameters':sum(p.numel()for p in net.parameters()),'input':'same RGB/proprio learned states; no GT deployment input','challenge_is_development_only_not_training':True};save(dest.with_suffix('.json'),rec);save(dest.with_suffix('.exposure.json'),{'episode_ids':arm['train'],'full':exposure.tolist(),'selected':selected_exposure.tolist(),'selected_step':selected['step']});records.append(rec);del net,opt,est,dynamic,cross,load;torch.cuda.empty_cache()
 passed=all(r['development_gate_passed']for r in records);save(B/'configs/v3_models_frozen.json',{'models':records,'development_gate_passed':passed,'test_not_evaluated':True,'plan_sha256':sha(B/'configs/v3_plan.json')});save(B/'STATUS.json',{'version':'v3','stage':'models_frozen'if passed else'development_gate_failed','models':len(records),'test_not_evaluated':True})
 if not passed:raise RuntimeError('State-only actor did not pass development gate; final test remains locked')
if __name__=='__main__':
 lock=GPU_LOCK.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
 try:main()
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
