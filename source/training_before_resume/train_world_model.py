from pathlib import Path
import argparse,json,time,math,random,fcntl,os,hashlib
import numpy as np
import torch
from wm_symbolic.world_model import WorldModel,loss_for_batch,symbol_nll,outcome_loss
from wm_symbolic.dataset import Episodes,batch_indices

P=Path('/home/robot/research/projects/wm_symbolic')
a=argparse.ArgumentParser();a.add_argument('--mode',choices=['probe','pilot','formal','one_step'],required=True);a.add_argument('--seed',type=int,default=0);a.add_argument('--updates',type=int);a.add_argument('--event-count',type=int,choices=[3,7],default=3);a.add_argument('--tag',default='');args=a.parse_args()
updates=args.updates or (1000 if args.mode=='probe' else 6000 if args.mode=='pilot' else 30000)
if args.mode=='pilot' and updates!=6000 or args.mode in ['formal','one_step'] and updates!=30000:raise ValueError('Protocol update budgets cannot be reduced')
out=P/'runs'/f'wm_{args.mode}{"_"+args.tag if args.tag else ""}_seed{args.seed}';out.mkdir(exist_ok=False)
lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');rng=np.random.default_rng(args.seed)
device='cuda';train=Episodes(P,'train',128 if args.mode=='probe' else None,event_count=args.event_count);stats=train.fit_stats();(out/'normalization_train_only.json').write_text(json.dumps(stats,indent=2))
if args.mode=='pilot' and train.transitions!=720:raise ValueError('Pilot requires exactly 720 train transitions')
if args.mode in ['formal','one_step'] and train.transitions!=4320:raise ValueError('Formal requires exactly 4320 train transitions')
dev=Episodes(P,'dev',normalizer=stats,event_count=args.event_count) if args.mode!='probe' else train
if args.mode=='pilot' and len(dev.root_ids)<10:raise ValueError('Pilot requires ten independent C dev roots')
if set(train.root_ids)&set(dev.root_ids) and args.mode!='probe':raise ValueError('Train/dev root overlap')
model=WorldModel(event_count=args.event_count).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=1e-4)
weights=[torch.tensor(x,device=device) for x in stats['class_weights']]
bf16=torch.cuda.is_bf16_supported();assert bf16,'This verified host should support BF16'
identity={'mode':args.mode,'seed':args.seed,'updates':updates,'train_transitions':train.transitions,'train_roots':train.root_ids,'dev_roots':dev.root_ids if args.mode!='probe' else [],
 'batch':8,'accumulation':2,'event_count':args.event_count,'tag':args.tag,'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,'device':torch.cuda.get_device_name(),
 'vision_backbone_frozen':True,'executor_loaded':False,'task_text_in_dynamics':False,'probe_only':args.mode=='probe',
 'source_sha256':{n:hashlib.sha256((P/'src/wm_symbolic'/n).read_bytes()).hexdigest() for n in ['world_model.py','dataset.py','vision_cache.py','state_labels.py']}}
(out/'identity.json').write_text(json.dumps(identity,indent=2))

def evaluate(data,limit=None):
 from wm_symbolic.evaluation import evaluate_data
 return evaluate_data(model,data,device,samples=8,limit=limit)


def interface_checks():
 model.eval();b=train.batch([0],device)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  e=model.encode(b['tokens'],b['geometry'],b['visible'],b['robot']);act=model.action(b['skill'],b['arguments'],b['budget']);root,_=model.initial(e[:,0],True);p,_=model.imagine(root,act[:,0],True)
  poisoned=b['tokens'].clone();poisoned[:,1:]=torch.randn_like(poisoned[:,1:])*100
  ep=model.encode(poisoned,b['geometry'],b['visible'],b['robot']);rp,_=model.initial(ep[:,0],True);pp,_=model.imagine(rp,act[:,0],True)
  isolation=float((p.vector()-pp.vector()).abs().max());assert isolation==0,'Future-input leakage'
  order=torch.randperm(b['tokens'].shape[-2],device=device);e_perm=model.encode(b['tokens'][:,:,order],b['geometry'][:,:,order],b['visible'][:,:,order],b['robot'])
  permutation=float((e-e_perm).abs().max());assert permutation<.03,permutation
  pointer_order=torch.tensor([2,0,1],device=device);s=model.symbols(root);sp=model.symbols(root,pointer_order);pointer_error=float((s['objects'][:,pointer_order]-sp['objects']).abs().max());assert pointer_error==0
  # Labels/outcomes are never arguments to the encoder or transition functions.
  result={'future_poison_prior_max_abs_difference':isolation,'token_permutation_encoder_max_abs_difference':permutation,'object_query_permutation_max_abs_difference':pointer_error,
   'root_split_disjoint':True,'normalization_source':'train only','future_feedback_in_action':False,'task_text_in_transition':False,'passed':True}
 (out/'input_isolation_checks.json').write_text(json.dumps(result,indent=2));model.train()

interface_checks();initial=evaluate(dev);(out/'initial_metrics.json').write_text(json.dumps(initial,indent=2));print('INITIAL',json.dumps({k:v for k,v in initial.items() if k!='counterexamples'}),flush=True)
best=float('inf');begin=time.monotonic();log=(out/'training.jsonl').open('w');batches=(out/'batch_indices.jsonl').open('w')
for update in range(1,updates+1):
 warm=min(update/1000,1);progress=max(0,(update-1000)/max(updates-1000,1));lr=3e-4*warm if update<=1000 else 3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*progress))
 for group in optimizer.param_groups:group['lr']=lr
 horizon=0 if update<=5000 or args.mode=='one_step' else 3 if update<=15000 else 5
 optimizer.zero_grad(set_to_none=True);parts={};used=[]
 for accumulation in range(2):
  indices=batch_indices(train,rng);used.append(indices.tolist());b=train.batch(indices,device)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms=loss_for_batch(model,b,horizon,weights)
  if not torch.isfinite(loss):
   torch.save({'update':update,'batch_indices':used,'batch':b,'model':model.state_dict()},out/'nonfinite_failure.pt');raise FloatingPointError('Nonfinite loss')
  (loss/2).backward()
  for key,value in terms.items():parts[key]=parts.get(key,0)+float(value.detach())/2
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm),norm;optimizer.step();batches.write(json.dumps({'update':update,'episodes':used})+'\n')
 if update==1 or update%25==0:
  row={'update':update,'lr':lr,'horizon':horizon,'seconds':time.monotonic()-begin,'grad_norm':float(norm),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),**parts};log.write(json.dumps(row)+'\n');log.flush();batches.flush();print('TRAIN',json.dumps(row),flush=True)
 interval=250 if args.mode=='probe' else 1000
 if update%interval==0 or update==updates:
  metric=evaluate(dev);metric.update(update=update,selection_split='train_probe' if args.mode=='probe' else 'dev');(out/f'metrics_{update:05d}.json').write_text(json.dumps(metric,indent=2))
  score=.5*(metric['free_3_nll']+metric['free_5_nll'])+metric['event_nll']
  checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'normalization':stats,'identity':identity,'update':update,'metrics':metric}
  if update%1000==0 or update==updates:torch.save(checkpoint,out/f'checkpoint_{update:05d}.pt')
  if score<best:best=score;torch.save(checkpoint,out/'best.pt')
  print('EVALUATION',update,json.dumps({k:v for k,v in metric.items() if k!='counterexamples'}),flush=True)
log.close();batches.close();final={'complete':True,'updates':updates,'elapsed_seconds':time.monotonic()-begin,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'best_selection_score':best,'final_metrics':metric,
 'scope':'128-transition training and interface probe, not independent effectiveness' if args.mode=='probe' else 'Guide training run; closed-loop acceptance evaluated separately'}
(out/'result.json').write_text(json.dumps(final,indent=2));print('TRAINING_COMPLETE',updates,flush=True)
