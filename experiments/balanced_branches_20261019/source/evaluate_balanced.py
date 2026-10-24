from study import *
from model import ObservationStateWorldModel
import evaluate as legacy_evaluate

def main():
 guard('evaluate');seal=read(U/'configs/models_frozen.json');lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');legacy_evaluate.S=U;refs=[];receipts=[]
 save(U/'configs/evaluation_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in [U/'configs/protocol.json',U/'source/evaluate_balanced.py',U/'source/analyze.py',R/'source/evaluate.py']},'before_new_predictions':True,'historical_evaluation_exploratory_only':True})
 oldrefs=read(C/'results/prediction_references.json')
 for seed in [0,1]:
  chosen=next(r for r in seal['selection']if r['seed']==seed);specs=[('updated',Path(chosen['path']))]
  if chosen['update']!=1000:specs.append(('updated_final1000',U/f'models/balanced_seed{seed}_step1000.pt'))
  for variant,path in specs:
   m=ObservationStateWorldModel(skill_count=6,event_count=3).cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval().requires_grad_(False);before=statehash(m)
   for split in ['train','development','evaluation']:receipts.append(legacy_evaluate.assess(m,NewEpisodes(split),variant,seed))
   assert before==statehash(m);del m;torch.cuda.empty_cache()
  for variant in ['control','control_final1000','updated','updated_final1000']:
   for split in ['train','development','evaluation']:
    if variant.startswith('control'):
     ref=next(r for r in oldrefs if r['variant']==variant.replace('control','updated')and r['seed']==seed and r['split']==split);assert sha(ref['path'])==ref['sha256'];path=Path(ref['path']);update=ref['selected_update']
    else:path=U/f'results/predictions_{"updated"if chosen["update"]==1000 else variant}_seed{seed}_{split}.npz';update=chosen['update']if variant=='updated'else 1000
    refs.append({'variant':variant,'seed':seed,'split':split,'path':str(path),'sha256':sha(path),'selected_update':update,'old_prediction_reused':variant.startswith('control'),'historical_evaluation_exploratory':split=='evaluation'})
 save(U/'results/prediction_references.json',refs);save(U/'results/inference_complete.json',{'utc':now(),'reports':receipts,'weights_unchanged':True,'new_training':False,'new_blind_evaluation':False});print('INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':main()
