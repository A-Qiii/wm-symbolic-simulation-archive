from study import *
import evaluate as legacy_evaluate
def main():
 guard('evaluate');seal=read(U/'configs/models_frozen.json');lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');legacy_evaluate.S=U;references=[];receipts=[]
 save(U/'configs/evaluation_frozen.json',{'utc':now(),'protocol_sha256':sha(U/'configs/protocol.json'),'legacy_evaluator_sha256':sha(R/'source/evaluate.py'),'local_evaluator_sha256':sha(U/'source/evaluate.py'),'analysis_sha256':sha(U/'source/analyze.py'),'before_new_model_predictions':True,'historical_evaluation_exploratory_only':True})
 for seed in [0,1]:
  chosen=next(r for r in seal['selection']if r['seed']==seed);specs=[('paired',Path(chosen['path']),['train','development','evaluation'])]
  if chosen['update']!=1000:specs.append(('paired_final1000',U/f'models/paired_seed{seed}_step1000.pt',['train','development','evaluation']))
  specs.append(('control_final1000',R/f'models/recovery_seed{seed}_step1000.pt',['train']))
  for variant,path,splits in specs:
   m=WorldModel(skill_count=6,event_count=3).cuda();cp=torch.load(path,map_location='cuda',weights_only=False);m.load_state_dict(cp['model']);m.eval().requires_grad_(False);before=statehash(m)
   for split in splits:receipts.append(legacy_evaluate.assess(m,NewEpisodes(split),variant,seed))
   assert statehash(m)==before;del m,cp;torch.cuda.empty_cache()
  for variant in ['control','control_final1000','paired','paired_final1000']:
   for split in ['train','development','evaluation']:
    if variant=='control':path=(I if split=='train'else R)/f'results/predictions_recovery_seed{seed}_{split}.npz'
    elif variant=='control_final1000':path=U/f'results/predictions_control_final1000_seed{seed}_train.npz'if split=='train'else R/f'results/predictions_recovery_final1000_seed{seed}_{split}.npz'
    else:path=U/f'results/predictions_{"paired"if chosen["update"]==1000 else variant}_seed{seed}_{split}.npz'
    assert path.exists(),path;references.append({'variant':variant,'seed':seed,'split':split,'status':'exploratory_reused'if split=='evaluation'else split,'path':str(path),'sha256':sha(path),'old_prediction_reused':not path.is_relative_to(U),'selected_update':chosen['update']if variant=='paired'else 500 if variant=='control'else 1000})
 save(U/'results/prediction_references.json',references);save(U/'results/inference_complete.json',{'utc':now(),'reports':receipts,'model_weights_unchanged':True,'new_blind_evaluation':False,'new_training_during_evaluation':False});print('INFERENCE_COMPLETE',flush=True)
if __name__=='__main__':main()
