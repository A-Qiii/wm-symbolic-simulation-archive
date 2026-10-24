from pathlib import Path
import json, os, time, fcntl, hashlib, random
import numpy as np
import torch
import torch.nn.functional as F
from model import PredicateReader,targets

P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_20261014'
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 protocol=json.loads((E/'configs/protocol.json').read_text());assert protocol['frozen_before_training']
 assert not (E/'results/training_complete.json').exists()
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device='cuda'
 with np.load(E/'data/inputs_targets.npz') as z: ds={k:torch.tensor(z[k],device=device) for k in ['objects','context','robot','cat','valid']};splits=z['splits'].copy()
 tr=np.flatnonzero(splits=='train');de=np.flatnonzero(splits=='tune');y,valid=targets(ds['cat'],ds['valid'])
 starts=time.monotonic();records=[]
 for mode,seed in [(m,s) for s in protocol['main_seeds'] for m in ['matched_eight_class','direct_predicates']]+[('direct_predicates_leave_pink_labels_out',0)]:
  tag=mode+'_seed'+str(seed);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
  model=PredicateReader(mode).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);best=float('inf');best_epoch=None;log=[];exclude='leave_pink' in mode
  mask=valid.clone()
  if exclude:mask[:,[2,5]]=False
  model.train();begin=time.monotonic()
  for epoch in range(1,101):
   batch_losses=[]
   for batch in np.array_split(rng.permutation(tr),int(np.ceil(len(tr)/64))):
    batch=torch.tensor(batch,device=device);optimizer.zero_grad(set_to_none=True)
    o,d=model(ds['objects'][batch],ds['context'][batch],ds['robot'][batch])
    if mode=='matched_eight_class':
     oc=F.cross_entropy(o.reshape(-1,8),ds['cat'][batch,:3].reshape(-1),reduction='none').reshape(-1,3)
     dc=F.cross_entropy(d,ds['cat'][batch,3],reduction='none')[:,None];losses=torch.cat([oc,dc],1);vm=ds['valid'][batch];loss=(losses*vm).sum()/vm.sum()
    else:
     logits=torch.cat([o[:,:,0],o[:,:,1],d],-1);losses=F.binary_cross_entropy_with_logits(logits,y[batch],reduction='none');loss=(losses*mask[batch]).sum()/mask[batch].sum()
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();batch_losses.append(float(loss.detach()))
   if epoch==1 or epoch%5==0:
    model.eval()
    with torch.no_grad():pred=model.probabilities(ds['objects'][de],ds['context'][de],ds['robot'][de]);losses=F.binary_cross_entropy(pred.clamp(1e-7,1-1e-7),y[de],reduction='none');group=[]
    for cols in [[0,1,2],[3,4,5],[6]]:
     mm=mask[de][:,cols];group.append((losses[:,cols]*mm).sum()/mm.sum())
    score=float(torch.stack(group).mean());rec={'epoch':epoch,'train_loss':float(np.mean(batch_losses)),'dev_macro_BCE':score,'seconds':time.monotonic()-begin};log.append(rec)
    if score<best:
     best=score;best_epoch=epoch;torch.save({'model':model.state_dict(),'mode':mode,'seed':seed,'epoch':epoch,'dev_macro_BCE':score,'protocol_sha256':sha(E/'configs/protocol.json'),'architecture_parameters':sum(p.numel() for p in model.parameters()),'input_contract':protocol['input_contract'],'excluded_training_and_selection_labels':['held(pink)','in(pink,drawer)'] if exclude else []},E/'models'/f'{tag}.pt')
    save(E/'logs'/f'{tag}.json',log);save(E/'results/progress.json',{'stage':'training','mode':mode,'seed':seed,**rec});print(tag,epoch,'dev',round(score,5),'best',best_epoch,flush=True);model.train()
  ck=torch.load(E/'models'/f'{tag}.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model']);model.eval();ps=[]
  with torch.no_grad():
   for start in range(0,len(splits),128):ps.append(model.probabilities(ds['objects'][start:start+128],ds['context'][start:start+128],ds['robot'][start:start+128]).cpu().numpy())
  np.save(E/'results'/f'{tag}_probabilities.npy',np.concatenate(ps));records.append({'tag':tag,'seed':seed,'mode':mode,'selected_epoch':best_epoch,'dev_macro_BCE':best,'seconds':time.monotonic()-begin,'parameters':ck['architecture_parameters'],'checkpoint_sha256':sha(E/'models'/f'{tag}.pt')})
  save(E/'results/training_runs.json',records)
  del model,optimizer
 save(E/'results/training_complete.json',{'complete':True,'seconds':time.monotonic()-starts,'models':records,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'new_world_model_training':False,'controller_changed':False})
 print('TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
