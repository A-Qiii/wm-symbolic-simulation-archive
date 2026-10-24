from pathlib import Path
import argparse,json,time,os,fcntl,hashlib,random,shutil,signal,subprocess,sys
import numpy as np
import torch
from wm_symbolic.world_model import WorldModel,loss_for_batch
from wm_symbolic.dataset import batch_indices
from wm_symbolic.core_dataset import CoreEpisodes
from wm_symbolic.core_evaluation import evaluate
from wm_symbolic.checkpointing import learning_rate,capture_rng,restore_rng,atomic_save
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'data/core1200_v1';R=P/'runs/core1200_v1'
stop=False
def request(sig,frame):
 global stop
 stop=True
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def small(metrics):return {k:v for k,v in metrics.items() if k not in ['root_metrics','counterexamples']}
def main():
 a=argparse.ArgumentParser();a.add_argument('--mode',choices=['probe','main'],required=True);a.add_argument('--resume',type=Path);a.add_argument('--tag',default='');args=a.parse_args();updates=1000 if args.mode=='probe' else 6000
 if args.mode=='main' and (R/'ownership_label_correction.json').exists() and not (R/'training_probe_ownership_v2/result.json').exists():
  print('REPEAT_FRESH_PROBE_AFTER_OFFLINE_LABEL_CORRECTION',flush=True)
  subprocess.run([sys.executable,str(Path(__file__)),'--mode','probe','--tag','ownership_v2'],check=True)
 protocol=json.loads((R/'protocol.json').read_text());assert protocol['authorization']['human_reply']=='好的' and protocol['id']=='six_skill_core1200_v1'
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
 for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGUSR1]:signal.signal(sig,request)
 out=R/('training_'+args.mode+('_'+args.tag if args.tag else ''));out.mkdir(exist_ok=bool(args.resume));torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');rng=np.random.default_rng(0)
 ck=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
 train=CoreEpisodes(P,'train',128 if args.mode=='probe' else None,normalizer=ck['normalization'] if ck else None);stats=ck['normalization'] if ck else train.fit_stats()
 assert train.transitions==(128 if args.mode=='probe' else 720)
 dev=train if args.mode=='probe' else CoreEpisodes(P,'dev',normalizer=stats,dev_role='tune')
 if args.mode=='main':
  qa=json.loads((R/'full_train_dev_qa.json').read_text());assert qa['passed'] and qa['non_test_transitions_checked']==960
  assert len(train.root_ids)==60 and len(dev.root_ids)==12 and not(set(train.root_ids)&set(dev.root_ids))
  assert (R/'training_probe/result.json').exists()
 model=WorldModel(skill_count=6,event_count=3).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=1e-4)
 weights=[torch.tensor(x,device='cuda') for x in stats['class_weights']];start=0;best=float('inf');metric=None
 identity={'mode':'core1200_'+args.mode,'seed':0,'updates':updates,'skill_count':6,'event_count':3,'train_roots':train.root_ids,'dev_roots':[] if args.mode=='probe' else dev.root_ids,'train_transitions':train.transitions,'executor_loaded':False,'random_initialization':True,'old_checkpoint_loaded':False,'vision_frozen':True,'protocol_sha256':sha(R/'protocol.json'),'parameters':sum(p.numel() for p in model.parameters()),'source_sha256':{name:sha(P/'src/wm_symbolic'/name) for name in ['world_model.py','core_dataset.py','core_evaluation.py','vision_cache.py']}}
 identity['label_version']=json.loads((B/'data_manifest.json').read_text()).get('label_version','core_labels_v1')
 identity['training_labels_sha256']=hashlib.sha256('|'.join(sha(Path(r['labels_path'])) for e in train.episodes for r in e['rows']).encode()).hexdigest()
 if ck:
  assert ck['identity']==identity;model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng);start=ck['update'];best=ck['best_selection_score']
 else:
  (out/'identity.json').write_text(json.dumps(identity,indent=2));(out/'normalization_train_only.json').write_text(json.dumps(stats,indent=2));shutil.copy2(B/'data_manifest.json',out/'data_manifest_at_start.json');snap=out/'source_snapshot';snap.mkdir()
  for name in identity['source_sha256']:shutil.copy2(P/'src/wm_symbolic'/name,snap/name)
  shutil.copy2(Path(__file__),snap/Path(__file__).name)
  model.eval();b=train.batch([0], 'cuda')
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   encoded=model.encode(b['tokens'],b['geometry'],b['visible'],b['robot']);act=model.action(b['skill'],b['arguments'],b['budget']);root,_=model.initial(encoded[:,0],True);prior,_=model.imagine(root,act[:,0],True)
   poison=b['tokens'].clone();poison[:,1:]=torch.randn_like(poison[:,1:])*100;bad=model.encode(poison,b['geometry'],b['visible'],b['robot']);br,_=model.initial(bad[:,0],True);bp,_=model.imagine(br,act[:,0],True);err=float((prior.vector()-bp.vector()).abs().max());assert err==0
   order=torch.randperm(b['tokens'].shape[-2],device='cuda');perm=model.encode(b['tokens'][:,:,order],b['geometry'][:,:,order],b['visible'][:,:,order],b['robot']);perr=float((encoded-perm).abs().max());assert perr<.03
  (out/'input_isolation_checks.json').write_text(json.dumps({'passed':True,'future_poison_prior_difference':err,'token_permutation_error':perr,'GT_label_or_success_in_forward_arguments':False,'normalization_split':'train','old_checkpoint_loaded':False},indent=2))
  metric=evaluate(model,dev,limit=8 if args.mode=='probe' else None);(out/'initial_metrics.json').write_text(json.dumps(metric,indent=2));print('INITIAL',json.dumps(small(metric)),flush=True)
 model.train();begin=time.monotonic();log=(out/'training.jsonl').open('a');batches=(out/'batch_indices.jsonl').open('a')
 def checkpoint(update):return {'model':model.state_dict(),'optimizer':opt.state_dict(),'normalization':stats,'identity':identity,'update':update,'metrics':metric,'rng':capture_rng(rng),'best_selection_score':best,'checkpoint_format':2}
 for update in range(start+1,updates+1):
  for group in opt.param_groups:group['lr']=learning_rate(update,updates)
  k=0 if update<=5000 else 3;opt.zero_grad(set_to_none=True);terms_out={};used=[]
  for acc in range(2):
   ids=batch_indices(train,rng);used.append(ids.tolist());b=train.batch(ids,'cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms=loss_for_batch(model,b,k,weights)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
   (loss/2).backward()
   for key,value in terms.items():terms_out[key]=terms_out.get(key,0)+float(value.detach())/2
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();batches.write(json.dumps({'update':update,'episodes':used})+'\n')
  if update==1 or update%25==0:
   row={'update':update,'horizon':k,'lr':opt.param_groups[0]['lr'],'seconds':time.monotonic()-begin,'grad_norm':float(norm),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),**terms_out};log.write(json.dumps(row)+'\n');log.flush();batches.flush();(R/'training_progress.json').write_text(json.dumps({'mode':args.mode,'pid':os.getpid(),'updates':update,'target':updates,**row},indent=2));print('TRAIN',json.dumps(row),flush=True)
  if update%(250 if args.mode=='probe' else 1000)==0 or update==updates:
   metric=evaluate(model,dev,diagnostics=True);metric.update(update=update,selection_split='train_probe' if args.mode=='probe' else 'dev_tune');(out/f'metrics_{update:05d}.json').write_text(json.dumps(metric,indent=2));score=metric['free3_nll']+metric['event_nll'];improved=score<best;best=min(best,score)
   atomic_save(checkpoint(update),out/'latest.pt')
   if update%1000==0:atomic_save(checkpoint(update),out/f'checkpoint_{update:05d}.pt')
   if improved:atomic_save(checkpoint(update),out/'best.pt')
   print('EVALUATE',update,json.dumps(small(metric)),flush=True)
  if stop:
   atomic_save(checkpoint(update),out/'paused.pt');(out/'paused.json').write_text(json.dumps({'update':update,'safe':True,'checkpoint':str(out/'paused.pt')}));return
 (out/'result.json').write_text(json.dumps({'complete':True,'updates':updates,'elapsed_seconds_this_segment':time.monotonic()-begin,'best_selection_score':best,'final_metrics':metric,'initialization':'fresh random; frozen DINO only pretrained','scope':'128 train transitions smoke, not independent efficacy' if args.mode=='probe' else 'six-skill core first fit; test still sealed'},indent=2));print('TRAINING_COMPLETE',updates,flush=True)
if __name__=='__main__':main()
