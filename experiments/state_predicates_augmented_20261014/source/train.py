from pathlib import Path
import json,hashlib,fcntl,time,random
import numpy as np
import torch
import torch.nn.functional as F
from model import PredicateReader,targets
from metrics import calibrate
P=Path('/home/robot/research/projects/wm_symbolic');OLD=P/'experiments/state_predicates_20261014';E=P/'experiments/state_predicates_augmented_20261014'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 assert not (E/'results/model_threshold_freeze.json').exists()
 protocol=json.loads((E/'configs/protocol.json').read_text());assert sha(E/'source/model.py')==sha(OLD/'source/model.py')
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 with np.load(OLD/'data/inputs_targets.npz') as z:
  om=z['splits']=='train';original={k:z[k][om].copy() for k in ['objects','context','robot']};oy,ov=targets(torch.tensor(z['cat'][om]),torch.tensor(z['valid'][om]));oy=oy.numpy();ov=ov.numpy()
 with np.load(E/'data/supplement_train_dev.npz') as z:
  sm=z['splits']=='train';dm=z['splits']=='dev';combined={k:np.concatenate([original[k],z[k][sm]]) for k in original};ty=np.concatenate([oy,z['y'][sm]]);tv=np.concatenate([ov,z['train_valid'][sm]]);dev={k:torch.tensor(z[k][dm],device='cuda') for k in original};dy=z['y'][dm].copy();dv=z['train_valid'][dm].copy()
 meta=json.loads((E/'data/metadata_train_dev.json').read_text());devroots=np.array([r['root_id'] for r in meta if r['split']=='dev']);unique=sorted(set(devroots));assert len(unique)==6 and len(ty)==973 and len(dy)==37
 data={k:torch.tensor(v,device='cuda') for k,v in combined.items()};y=torch.tensor(ty,device='cuda');valid=torch.tensor(tv,device='cuda');logloss=torch.tensor(dy,device='cuda');devvalid=torch.tensor(dv,device='cuda')
 save(E/'configs/budget_matched_final_checkpoint.json',{'saved_before_training':True,'secondary_control':'Also retain the final update1500 checkpoint and its dev-calibrated thresholds. Evaluate after freeze as budget-matched secondary diagnostic, never use test to replace selected checkpoint.','all_selected_checkpoints_may_be_earlier_than1500':True})
 # Scope tests: changing supervision cannot affect a forward input tensor; known entity permutation is exact.
 torch.manual_seed(0);probe=PredicateReader().cuda().eval()
 with torch.no_grad():
  z={k:v[:2] for k,v in data.items()};p0=probe.probabilities(**z);order=[2,0,1];p1=probe.probabilities(z['objects'][:,:,order],z['context'],z['robot']);err=float((p1-p0[:,[2,0,1,5,3,4,6]]).abs().max())
 assert err<1e-6
 save(E/'results/input_checks.json',{'passed':True,'entity_permutation_max_error':err,'root_disjoint_train_dev':not(set(protocol['split']['train'])&set(protocol['split']['dev'])),'new_test_file_absent':not(E/'data/supplement_test.npz').exists(),'supplement_in_training_labels':int(tv[900:,3:6].sum()),'original_training_examples':900,'supplement_training_windows':73,'GT_or_future_forward_parameters':False})
 del probe
 runs=[];thresholds={};start=time.monotonic()
 def pred(m):
  m.eval()
  with torch.no_grad():return m.probabilities(**dev).cpu().numpy()
 def score(m):
  pp=pred(m);bce=-(dy*np.log(np.clip(pp,1e-7,1-1e-7))+(1-dy)*np.log(np.clip(1-pp,1e-7,1-1e-7)));vals=[]
  for cols in [[0,1,2],[6]]:
   vals.append(np.mean([(bce[devroots==r][:,cols]*dv[devroots==r][:,cols]).sum()/dv[devroots==r][:,cols].sum() for r in unique]))
  return float(np.mean(vals)),pp
 for seed in [0,1,2]:
  torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed);m=PredicateReader().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);best=float('inf');selected=None;order=rng.permutation(973).tolist();cursor=0;exposures=np.zeros(973,int);batches=[];log=[];begin=time.monotonic()
  for update in range(1,1501):
   while len(order)-cursor<60:order=order[cursor:]+rng.permutation(973).tolist();cursor=0
   indices=np.array(order[cursor:cursor+60]);cursor+=60;np.add.at(exposures,indices,1);batches.append(indices);ids=torch.tensor(indices,device='cuda');opt.zero_grad(set_to_none=True);m.train();o,d=m(data['objects'][ids],data['context'][ids],data['robot'][ids]);logits=torch.cat([o[:,:,0],o[:,:,1],d],-1);losses=F.binary_cross_entropy_with_logits(logits,y[ids],reduction='none');loss=(losses*valid[ids]).sum()/valid[ids].sum();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
   if update==1 or update%75==0:
    sc,pp=score(m);row={'update':update,'train_BCE':float(loss.detach()),'new_dev_root_macro_BCE':sc,'seconds':time.monotonic()-begin};log.append(row)
    if sc<best:
     best=sc;selected=update;torch.save({'model':m.state_dict(),'seed':seed,'mode':'direct_predicates','update':update,'new_dev_root_macro_BCE':sc,'protocol_sha256':sha(E/'configs/protocol.json'),'architecture_source_sha256':sha(E/'source/model.py'),'from_scratch':True},E/'models'/f'augmented_seed{seed}_selected.pt')
    save(E/'logs'/f'seed{seed}.json',log);save(E/'results/progress.json',{'stage':'training','seed':seed,**row});print('TRAIN',seed,update,round(sc,6),'selected',selected,flush=True)
  torch.save({'model':m.state_dict(),'seed':seed,'mode':'direct_predicates','update':1500,'new_dev_root_macro_BCE':sc,'protocol_sha256':sha(E/'configs/protocol.json'),'architecture_source_sha256':sha(E/'source/model.py'),'from_scratch':True},E/'models'/f'augmented_seed{seed}_final.pt')
  for tag,path in [(f'augmented_seed{seed}',E/'models'/f'augmented_seed{seed}_selected.pt'),(f'final_seed{seed}',E/'models'/f'augmented_seed{seed}_final.pt'),(f'old_seed{seed}',OLD/'models'/f'direct_predicates_seed{seed}.pt')]:
   ck=torch.load(path,map_location='cuda',weights_only=False);m.load_state_dict(ck['model']);pp=pred(m);np.save(E/'results'/f'{tag}_dev_probabilities.npy',pp);thresholds[tag]=calibrate(pp,dy,dv);thresholds[tag]['checkpoint_sha256']=sha(path);thresholds[tag]['checkpoint_path']=str(path)
  assert exposures.sum()==90000;np.save(E/'logs'/f'seed{seed}_batch_indices.npy',np.stack(batches).astype(np.int16))
  run={'seed':seed,'updates':1500,'selected_update':selected,'best_new_dev_score':best,'total_window_presentations':90000,'nominal_counted_exposures':int(exposures.sum()),'supplement_counted_exposures':int(exposures[900:].sum()),'old_counted_exposures':int(exposures[:900].sum()),'seconds':time.monotonic()-begin};runs.append(run);save(E/'results/training_runs.json',runs)
 save(E/'results/dev_calibration.json',thresholds)
 freeze={'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'all_training_complete':True,'model_and_thresholds_fixed_before_test':True,'new_test_feature_or_probability_file_exists':(E/'data/supplement_test.npz').exists(),'models':thresholds,'protocol_sha256':sha(E/'configs/protocol.json'),'dev_root_ids':unique,'total_seconds':time.monotonic()-start,'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'source_sha256':{str(f):sha(f) for f in (E/'source').glob('*.py')}};assert not freeze['new_test_feature_or_probability_file_exists'];save(E/'results/model_threshold_freeze.json',freeze);(E/'results/model_threshold_freeze.json').chmod(0o444);print('MODELS_THRESHOLDS_FROZEN',flush=True)
if __name__=='__main__':main()
