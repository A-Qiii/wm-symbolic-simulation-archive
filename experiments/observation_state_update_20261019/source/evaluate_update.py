from study import *
from model import ObservationStateWorldModel,load_shared
import evaluate as legacy_evaluate

def main():
    guard('evaluate');seal=read(U/'configs/models_frozen.json');lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');legacy_evaluate.S=U;references=[];receipts=[];gradient_checks=[]
    save(U/'configs/evaluation_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in [U/'configs/protocol.json',U/'configs/evaluation_requirements.json',U/'source/evaluate_update.py',U/'source/analyze.py',U/'source/observation_control.py',R/'source/evaluate.py']},'before_new_model_predictions':True,'historical_evaluation_exploratory_only':True})
    old,new,_,norm=prepared();w=[torch.tensor(v,device='cuda')for v in norm['class_weights']]
    for seed in [0,1]:
        # Directly verify the zero matrix has a live gradient on the first scheduled real microbatch. No optimizer step.
        m=ObservationStateWorldModel(skill_count=6,event_count=3).cuda();load_shared(m,torch.load(CELL/f'models/joint_seed{seed}/latest.pt',map_location='cuda',weights_only=False)['model'])
        with np.load(C/f'configs/schedule_seed{seed}.npz')as z:examples=[old.get(int(i))for i in z['old'][0,0]]+[new.get(int(i))for i in z['new'][0,0]];rng=int(z['latent'][0,0])
        b={k:torch.stack([x[k]for x in examples]).cuda()for k in examples[0]};torch.manual_seed(rng)
        with torch.autocast('cuda',dtype=torch.bfloat16):total,cur,fut,terms,enc,terminal=losses(m,b,3,w);objective=total+terminal
        g=torch.autograd.grad(objective,m.observation_to_h.weight)[0];assert torch.isfinite(g).all() and g.norm()>0
        gradient_checks.append({'seed':seed,'zero_initialized_weight_gradient_norm':float(g.norm()),'nonzero_gradient_elements':int(torch.count_nonzero(g)),'optimizer_steps':0,'actual_first_scheduled_microbatch':True});del m,b,g,total,cur,fut,terms,enc,terminal,objective;torch.cuda.empty_cache()
        chosen=next(r for r in seal['selection']if r['seed']==seed);specs=[('updated',Path(chosen['path']))]
        if chosen['update']!=1000:specs.append(('updated_final1000',U/f'models/state_update_seed{seed}_step1000.pt'))
        for variant,path in specs:
            m=ObservationStateWorldModel(skill_count=6,event_count=3).cuda();cp=torch.load(path,map_location='cuda',weights_only=False);m.load_state_dict(cp['model']);m.eval().requires_grad_(False);before=statehash(m)
            for split in ['train','development','evaluation']:receipts.append(legacy_evaluate.assess(m,NewEpisodes(split),variant,seed))
            assert statehash(m)==before;del m,cp;torch.cuda.empty_cache()
        previous=read(C/f'results/train_paired_seed{seed}.json')['selected']['update']
        for variant in ['control','control_final1000','updated','updated_final1000']:
          for split in ['train','development','evaluation']:
            if variant.startswith('control'):base='paired'if variant=='control'or previous==1000 else'paired_final1000';path=C/f'results/predictions_{base}_seed{seed}_{split}.npz'
            else:base='updated'if chosen['update']==1000 else variant;path=U/f'results/predictions_{base}_seed{seed}_{split}.npz'
            references.append({'variant':variant,'seed':seed,'split':split,'path':str(path),'sha256':sha(path),'historical_evaluation_exploratory':split=='evaluation','old_prediction_reused':variant.startswith('control'),'selected_update':previous if variant=='control'else chosen['update']if variant=='updated'else 1000})
    save(U/'results/prediction_references.json',references);save(U/'results/gradient_start_check.json',gradient_checks);save(U/'results/inference_complete.json',{'utc':now(),'reports':receipts,'model_weights_unchanged':True,'new_blind_evaluation':False,'new_training_during_evaluation':False});print('INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':main()
