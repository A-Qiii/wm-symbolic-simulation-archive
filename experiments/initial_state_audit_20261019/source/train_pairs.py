from audit import *
import evaluate as prior_evaluator
def main():
 guard();lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');prior_evaluator.S=D;data=NewEpisodes('train');seal=read(S/'configs/models_frozen.json');reports=[]
 for seed in [0,1]:
  for arm in ['original','nominal','recovery']:
   path=CELL/f'models/joint_seed{seed}/final6000.pt'if arm=='original'else Path(next(r['path']for r in seal['selection']if r['seed']==seed and r['arm']==arm));m=WorldModel(skill_count=6,event_count=3).cuda();cp=torch.load(path,map_location='cuda',weights_only=False);m.load_state_dict(cp['model']);m.eval().requires_grad_(False);before=statehash(m);reports.append(prior_evaluator.assess(m,data,arm,seed));assert statehash(m)==before;del m,cp;torch.cuda.empty_cache()
 save(D/'results/train_pairs_complete.json',{'reports':reports,'same_previous_sealed_evaluator':str(S/'source/evaluate.py'),'evaluator_sha256':sha(S/'source/evaluate.py'),'new_training':0,'new_collection':0,'training_branch_visibility':'nominal:b0+b1; recovery:b0+b2. Matched b1/b2 pair has one untrained branch per model; do not call both branches training-exposed.'})
if __name__=='__main__':main()
