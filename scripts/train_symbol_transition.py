"""Protocol-budget baseline. Perception frozen; dynamics uses 32 symbols only."""
from pathlib import Path
import argparse,json,time,hashlib,random,shutil,datetime
import numpy as np
import torch
from wm_symbolic.world_model import WorldModel
from wm_symbolic.symbol_transition import SymbolTransition
from wm_symbolic.symbol_dataset import SymbolEpisodes
from wm_symbolic.symbol_training import loss_for_symbols,evaluate_symbols
from wm_symbolic.dataset import batch_indices
from wm_symbolic.checkpointing import learning_rate,capture_rng,restore_rng,atomic_save,PauseRequest
from wm_symbolic.gpu_phase import gpu_phase

P=Path('/home/robot/research/projects/wm_symbolic');a=argparse.ArgumentParser();a.add_argument('--perception-checkpoint',type=Path,required=True);a.add_argument('--seed',type=int,default=0);a.add_argument('--tag',default='');a.add_argument('--resume',type=Path);args=a.parse_args()
if (P/'runs/PAUSED_BY_USER.json').exists():raise SystemExit('Paused by user')
if (P/'runs/FORMAL_TRAINING_HOLD.json').exists():raise SystemExit('Formal training held for executor reliability review')
sha=hashlib.sha256(args.perception_checkpoint.read_bytes()).hexdigest();ck=torch.load(args.perception_checkpoint,map_location='cpu',weights_only=False)
assert ck['identity']['mode']=='formal' and ck['identity']['seed']==args.seed
assert json.loads((P/'runs/stage1_gate.json').read_text())['entry_conditions_met']
stats=ck['normalization'];train=SymbolEpisodes(P,'train',sha,stats);dev=SymbolEpisodes(P,'dev',sha,stats)
assert train.transitions==4320 and dev.transitions==360 and len(train.root_ids)==240 and len(dev.root_ids)==30 and not set(train.root_ids)&set(dev.root_ids)
manifest_sha=hashlib.sha256((P/'data/data_manifest.json').read_bytes()).hexdigest();updates=30000
identity={'mode':'symbol_transition','seed':args.seed,'updates':updates,'event_count':7,'train_roots':train.root_ids,'dev_roots':dev.root_ids,'train_transitions':train.transitions,
 'perception_checkpoint':str(args.perception_checkpoint),'perception_checkpoint_sha256':sha,'dynamic_state_dimensions':32,'perception_frozen':True,'action_encoder_frozen':True,
 'batch':8,'accumulation':2,'loss':'L_sym + 0.5 L_out + 0.5 L_multi; L_sym has frozen-current and future terms; L_multi has symbol + 0.2 outcome; no latent KL or visual reconstruction in symbolic dynamics',
 'source_sha256':{name:hashlib.sha256((P/'src/wm_symbolic'/name).read_bytes()).hexdigest() for name in ['symbol_transition.py','symbol_dataset.py','symbol_training.py','dataset.py']}}
resume=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None;start=int(resume['update']) if resume else 0
if resume:
 for key in identity:
  if resume['identity'][key]!=identity[key]:raise ValueError('Resume identity mismatch '+key)
 assert resume['data_manifest_sha256']==manifest_sha and 0<start<updates
out=P/'runs'/f'symbol_transition{"_"+args.tag if args.tag else ""}_seed{args.seed}'
with gpu_phase(P,'train_symbol_transition'):
 out.mkdir(exist_ok=False);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed);rng=np.random.default_rng(args.seed);torch.set_num_threads(4)
 model=SymbolTransition();perception=WorldModel(event_count=7);perception.load_state_dict(ck['model']);model.initialize_frozen_action_encoder(perception);del perception,ck
 model.cuda();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,betas=(.9,.999),weight_decay=1e-4);weights=[torch.tensor(x,device='cuda') for x in stats['class_weights']]
 best=float('inf');metric=None
 if resume:
  model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer']);restore_rng(resume['rng'],rng);best=resume['best_selection_score'];metric=resume['metrics']
  assert all(int(s['step'])==start for s in optimizer.state.values() if 'step' in s)
  assert all(abs(g['lr']-learning_rate(start,updates))<1e-12 for g in optimizer.param_groups)
  previous=args.resume.parent/'best.pt'
  if previous.exists():shutil.copy2(previous,out/'best.pt')
 (out/'identity.json').write_text(json.dumps(identity,indent=2));(out/'normalization_train_only.json').write_text(json.dumps(stats,indent=2));shutil.copy2(P/'data/data_manifest.json',out/'data_manifest.json')
 snapshot=out/'source_snapshot';snapshot.mkdir()
 for name in identity['source_sha256']:shutil.copy2(P/'src/wm_symbolic'/name,snapshot/name)
 shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
 pause=PauseRequest();log=(out/'training.jsonl').open('w');batches=(out/'batch_indices.jsonl').open('w');begin=time.monotonic()
 def save(update,path):
  atomic_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'normalization':stats,'identity':identity,'update':update,'metrics':metric,'rng':capture_rng(rng),'best_selection_score':best,'data_manifest_sha256':manifest_sha,'checkpoint_format':2,
    'scheduler':{'kind':'warmup_cosine','completed_update':update,'total_updates':updates,'next_lr':learning_rate(update+1,updates)}},path)
 if not resume:
  initial=evaluate_symbols(model,dev);(out/'initial_metrics.json').write_text(json.dumps(initial,indent=2));print('INITIAL',json.dumps(initial),flush=True)
 model.train()
 for update in range(start+1,updates+1):
  lr=learning_rate(update,updates)
  for group in optimizer.param_groups:group['lr']=lr
  horizon=0 if update<=5000 else 3 if update<=15000 else 5;optimizer.zero_grad(set_to_none=True);parts={};used=[]
  for accumulation in range(2):
   indices=batch_indices(train,rng);used.append(indices.tolist());b=train.batch(indices,'cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms=loss_for_symbols(model,b,horizon,weights)
   if not torch.isfinite(loss):atomic_save({'update':update,'batch_indices':used,'model':model.state_dict()},out/'nonfinite_failure.pt');raise FloatingPointError('Nonfinite baseline loss')
   (loss/2).backward()
   for key,value in terms.items():parts[key]=parts.get(key,0)+float(value.detach())/2
  norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(norm);optimizer.step();batches.write(json.dumps({'update':update,'episodes':used})+'\n')
  if update==start+1 or update%25==0:
   row={'update':update,'lr':lr,'horizon':horizon,'seconds':time.monotonic()-begin,'grad_norm':float(norm),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),**parts};log.write(json.dumps(row)+'\n');log.flush();batches.flush();print('TRAIN',json.dumps(row),flush=True)
  if update%1000==0:
   metric=evaluate_symbols(model,dev);metric.update(update=update,selection_split='dev');(out/f'metrics_{update:05d}.json').write_text(json.dumps(metric,indent=2));score=.5*(metric['free_3_nll']+metric['free_5_nll'])+metric['event_nll'];improved=score<best;best=min(best,score)
   save(update,out/f'checkpoint_{update:05d}.pt')
   if improved:save(update,out/'best.pt')
   print('EVALUATION',update,json.dumps(metric),flush=True)
  if pause.requested:
   path=out/f'checkpoint_pause_{update:05d}.pt';save(update,path);record={'status':'paused','stage':'symbol_transition_training','checkpoint':str(path),'saved_update':update,'paused_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'resume_requires_user_authorization':True}
   (out/'pause.json').write_text(json.dumps(record,indent=2));(P/'runs/PAUSED_BY_USER.json').write_text(json.dumps(record,indent=2));log.close();batches.close();raise SystemExit(0)
 log.close();batches.close();result={'complete':True,'updates':updates,'resumed_from':start,'elapsed_seconds':time.monotonic()-begin,'final_metrics':metric,'best_selection_score':best,'scope':'Training and C development selection only; calibration and independent closed loop are separate'}
 (out/'result.json').write_text(json.dumps(result,indent=2));print('TRAINING_COMPLETE',updates,flush=True)
