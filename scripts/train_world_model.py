from pathlib import Path
import argparse,json,time,math,random,fcntl,os,hashlib,shutil,datetime
import numpy as np
import torch
from wm_symbolic.world_model import WorldModel,loss_for_batch,symbol_nll,outcome_loss
from wm_symbolic.dataset import Episodes,batch_indices
from wm_symbolic.checkpointing import learning_rate,capture_rng,restore_rng,atomic_save,PauseRequest

P=Path('/home/robot/research/projects/wm_symbolic')
a=argparse.ArgumentParser();a.add_argument('--mode',choices=['probe','pilot','formal','one_step'],required=True);a.add_argument('--seed',type=int,default=0);a.add_argument('--updates',type=int);a.add_argument('--event-count',type=int,choices=[3,7],default=3);a.add_argument('--tag',default='');a.add_argument('--resume',type=Path);args=a.parse_args()
updates=args.updates or (1000 if args.mode=='probe' else 6000 if args.mode=='pilot' else 30000)
if args.mode=='pilot' and updates!=6000 or args.mode in ['formal','one_step'] and updates!=30000:raise ValueError('Protocol update budgets cannot be reduced')
if (P/'runs/PAUSED_BY_USER.json').exists():raise SystemExit('Paused by user; authorization and archival of the pause record required.')
if args.mode in ['formal','one_step'] and (P/'runs/FORMAL_TRAINING_HOLD.json').exists():raise SystemExit('Formal training held for the user-requested executor reliability review.')
resume=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
start=int(resume['update']) if resume else 0
if resume:
 for key,value in [('mode',args.mode),('seed',args.seed),('updates',updates),('event_count',args.event_count)]:
  if resume['identity'][key]!=value:raise ValueError(f'Resume identity mismatch: {key}')
 if not 0<start<updates:raise ValueError('Resume checkpoint is not an unfinished run')
 manifest_sha=hashlib.sha256((P/'data/data_manifest.json').read_bytes()).hexdigest()
 if resume.get('data_manifest_sha256',manifest_sha)!=manifest_sha:raise ValueError('Data manifest changed since checkpoint')
 for name,expected in resume['identity']['source_sha256'].items():
  if hashlib.sha256((P/'src/wm_symbolic'/name).read_bytes()).hexdigest()!=expected:raise ValueError(f'Model/data source changed: {name}')
out=P/'runs'/f'wm_{args.mode}{"_"+args.tag if args.tag else ""}_seed{args.seed}'
lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
out.mkdir(exist_ok=False)
torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');rng=np.random.default_rng(args.seed)
device='cuda';train=Episodes(P,'train',128 if args.mode=='probe' else None,normalizer=resume['normalization'] if resume else None,event_count=args.event_count);stats=resume['normalization'] if resume else train.fit_stats();(out/'normalization_train_only.json').write_text(json.dumps(stats,indent=2))
if args.mode=='pilot' and train.transitions!=720:raise ValueError('Pilot requires exactly 720 train transitions')
if args.mode in ['formal','one_step'] and train.transitions!=4320:raise ValueError('Formal requires exactly 4320 train transitions')
dev=Episodes(P,'dev',normalizer=stats,event_count=args.event_count) if args.mode!='probe' else train
if args.mode=='pilot' and len(dev.root_ids)<10:raise ValueError('Pilot requires ten independent C dev roots')
if set(train.root_ids)&set(dev.root_ids) and args.mode!='probe':raise ValueError('Train/dev root overlap')
if args.mode in ['formal','one_step']:
 gate=json.loads((P/'runs/stage1_gate.json').read_text())
 if not gate['entry_conditions_met']:raise ValueError('Pilot evidence gate not met')
 if len(train.root_ids)!=240 or len(dev.root_ids)!=30 or dev.transitions!=360:raise ValueError('Formal train/dev root and transition quotas not met')
 if sum(x.startswith('A_train_') for x in train.root_ids)!=120 or sum(x.startswith('B_train_') for x in train.root_ids)!=120:raise ValueError('Native A/B root quota differs')
 manifest=json.loads((P/'data/data_manifest.json').read_text())
 audit=json.loads((P/'runs/formal_data_audit.json').read_text())
 if not audit['passed'] or audit['manifest_sha256']!=hashlib.sha256((P/'data/data_manifest.json').read_bytes()).hexdigest():raise ValueError('Formal data audit missing or stale')
 expected={'train':(240,4320),'dev':(30,360),'calibration':(30,360),'test':(80,960)}
 for split,(roots,transitions) in expected.items():
  rows=[r for r in manifest['records'] if r['split']==split]
  if len(rows)!=transitions or len(set(r['root_id'] for r in rows))!=roots:raise ValueError('Formal quota differs: '+split)
 if manifest.get('event_names')!=['red_moved','blue_moved','pink_moved','led_was_off','led_was_on','light_was_off','light_was_on']:raise ValueError('Frozen v2 event interface required')

model=WorldModel(event_count=args.event_count).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=1e-4)
if resume:
 if train.root_ids!=resume['identity']['train_roots'] or train.transitions!=resume['identity']['train_transitions']:raise ValueError('Resume training roots/count changed')
 if args.mode!='probe' and dev.root_ids!=resume['identity']['dev_roots']:raise ValueError('Resume development roots changed')
 model.load_state_dict(resume['model'],strict=True);optimizer.load_state_dict(resume['optimizer'])
 if any(abs(g['lr']-learning_rate(start,updates))>1e-12 for g in optimizer.param_groups):raise ValueError('Checkpoint learning rate disagrees with saved update')
 if any(int(x['step'])!=start for x in optimizer.state.values() if 'step' in x):raise ValueError('Optimizer step mismatch')
weights=[torch.tensor(x,device=device) for x in stats['class_weights']]
bf16=torch.cuda.is_bf16_supported();assert bf16,'This verified host should support BF16'
identity={'mode':args.mode,'seed':args.seed,'updates':updates,'train_transitions':train.transitions,'train_roots':train.root_ids,'dev_roots':dev.root_ids if args.mode!='probe' else [],
 'batch':8,'accumulation':2,'event_count':args.event_count,'tag':args.tag,'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,'device':torch.cuda.get_device_name(),
 'vision_backbone_frozen':True,'executor_loaded':False,'task_text_in_dynamics':False,'probe_only':args.mode=='probe',
 'source_sha256':{n:hashlib.sha256((P/'src/wm_symbolic'/n).read_bytes()).hexdigest() for n in ['world_model.py','dataset.py','vision_cache.py','state_labels.py']}}
identity['resume_parent']=str(args.resume.resolve()) if args.resume else None
identity['resume_parent_sha256']=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None
identity['rng_continuity']='restored' if resume and 'rng' in resume else 'legacy_checkpoint_missing_rng' if resume else 'fresh_seed'
(out/'identity.json').write_text(json.dumps(identity,indent=2))
shutil.copy2(P/'data/data_manifest.json',out/'data_manifest.json')
snapshot=out/'source_snapshot';snapshot.mkdir()
for source in (P/'src/wm_symbolic').glob('*.py'):shutil.copy2(source,snapshot/source.name)
shutil.copy2(Path(__file__),snapshot/'train_world_model.py')
pause=PauseRequest()

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

interface_checks()
best=float('inf');metric=resume['metrics'] if resume else None
if resume:
 best_file=args.resume.parent/'best.pt'
 if best_file.exists():
  previous_best=torch.load(best_file,map_location='cpu',weights_only=False)
  if previous_best['update']<=start:
   v=previous_best['metrics'];best=.5*(v['free_3_nll']+v['free_5_nll'])+v['event_nll'];shutil.copy2(best_file,out/'best.pt')
 if 'best_selection_score' in resume:best=min(best,resume['best_selection_score'])
 if 'rng' in resume:restore_rng(resume['rng'],rng)
 else:
  # The batch sampler is an independent generator; its consumed calls can be replayed exactly.
  for _ in range(start*2):batch_indices(train,rng)
  fallback_seed=args.seed+1000003*start
  torch.manual_seed(fallback_seed);np.random.seed(fallback_seed%(2**32));random.seed(fallback_seed)
 receipt={'loaded_update':start,'next_update':start+1,'loaded_optimizer_lr':optimizer.param_groups[0]['lr'],
  'next_lr':learning_rate(start+1,updates),'normalization_reused':True,'rng_continuity':identity['rng_continuity'],
  'legacy_fallback_seed':fallback_seed if 'rng' not in resume else None,'batch_generator':'replayed original calls' if 'rng' not in resume else 'restored',
  'bitwise_identical_to_uninterrupted_run_claimed':False,'checkpoint':str(args.resume),'pid':os.getpid()}
 (out/'resume_validation.json').write_text(json.dumps(receipt,indent=2));print('RESUMED',json.dumps(receipt),flush=True)
else:
 initial=evaluate(dev);(out/'initial_metrics.json').write_text(json.dumps(initial,indent=2));print('INITIAL',json.dumps({k:v for k,v in initial.items() if k!='counterexamples'}),flush=True)
model.train();begin=time.monotonic();log=(out/'training.jsonl').open('w');batches=(out/'batch_indices.jsonl').open('w')

def checkpoint_at(update):
 return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'normalization':stats,'identity':identity,'update':update,'metrics':metric,
  'rng':capture_rng(rng),'scheduler':{'kind':'warmup_cosine','completed_update':update,'total_updates':updates,'next_lr':learning_rate(update+1,updates)},
  'best_selection_score':best,'data_manifest_sha256':hashlib.sha256((out/'data_manifest.json').read_bytes()).hexdigest(),'checkpoint_format':2}

def finish_pause(update):
 destination=out/f'checkpoint_pause_{update:05d}.pt';atomic_save(checkpoint_at(update),destination);log.flush();batches.flush()
 record={'status':'paused','pid':os.getpid(),'saved_update':update,'checkpoint':str(destination),'signal':pause.reason,
  'resume_requires_user_authorization':True,'complete':False,'paused_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (out/'pause.json').write_text(json.dumps(record,indent=2));(P/'runs/PAUSED_BY_USER.json').write_text(json.dumps(record,indent=2));print('PAUSED_SAFELY',json.dumps(record),flush=True)
 log.close();batches.close();raise SystemExit(0)

if pause.requested:finish_pause(start)
for update in range(start+1,updates+1):
 lr=learning_rate(update,updates)
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
 if update==start+1 or update%25==0:
  row={'update':update,'lr':lr,'horizon':horizon,'seconds':time.monotonic()-begin,'grad_norm':float(norm),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),**parts};log.write(json.dumps(row)+'\n');log.flush();batches.flush();print('TRAIN',json.dumps(row),flush=True)
 interval=250 if args.mode=='probe' else 1000
 if update%interval==0 or update==updates:
  metric=evaluate(dev);metric.update(update=update,selection_split='train_probe' if args.mode=='probe' else 'dev');(out/f'metrics_{update:05d}.json').write_text(json.dumps(metric,indent=2))
  score=.5*(metric['free_3_nll']+metric['free_5_nll'])+metric['event_nll']
  improved=score<best;best=min(best,score);checkpoint=checkpoint_at(update)
  if update%1000==0 or update==updates:atomic_save(checkpoint,out/f'checkpoint_{update:05d}.pt')
  if improved:atomic_save(checkpoint,out/'best.pt')
  print('EVALUATION',update,json.dumps({k:v for k,v in metric.items() if k!='counterexamples'}),flush=True)
 if pause.requested:finish_pause(update)
log.close();batches.close();final={'complete':True,'updates':updates,'resumed_from':start,'updates_this_segment':updates-start,'elapsed_seconds':time.monotonic()-begin,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'best_selection_score':best,'final_metrics':metric,
 'scope':'128-transition training and interface probe, not independent effectiveness' if args.mode=='probe' else 'Guide training run; closed-loop acceptance evaluated separately'}
(out/'result.json').write_text(json.dumps(final,indent=2));print('TRAINING_COMPLETE',updates,flush=True)
