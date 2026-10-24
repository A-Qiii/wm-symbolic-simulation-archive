from common import *
import torch,fcntl
from dataset import NewEpisodes
from model import ObservationStateWorldModel
from evaluate import probabilities
def main():
 guard('verify_eval');verify_freeze();budget_check();seal=read(S/'configs/models_frozen.json');item=next(x for x in seal['selection']if x['arm']=='random'and x['seed']==0);assert sha(item['path'])==item['sha256'];lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True);data=NewEpisodes('train','random');b=data.batch([0],'cuda');m=ObservationStateWorldModel(skill_count=6,event_count=3).cuda();m.load_state_dict(torch.load(item['path'],map_location='cuda',weights_only=False)['model']);m.eval().requires_grad_(False)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  enc=m.encode(b['tokens'],b['geometry'],b['visible'],b['robot']).repeat_interleave(8,0);act=m.action(b['skill'],b['arguments'],b['budget']).repeat_interleave(8,0)
  torch.manual_seed(917);full=m.history(enc,act)[0]
  torch.manual_seed(917);prefix=m.history(enc[:,:3],act[:,:2])[0]
  assert torch.equal(full[2].h,prefix[2].h)and torch.equal(full[2].z,prefix[2].z)
  future=[]
  for state in[full[2],prefix[2]]:
   torch.manual_seed(918);rr=[]
   for t in range(2,6):state,_=m.imagine(state,act[:,t]);rr.append(probabilities(m,state))
   future.append(np.stack(rr))
  assert np.array_equal(*future)
 save(S/'results/no_future_observation_qa.json',{'utc':now(),'passed':True,'actual_training_role':data.episodes[0]['root_id'],'h2_full_history_equals_prefix_only':True,'four_imagined_successors_equal':True,'model_updates':0,'test_data_used':0,'note':'Prefix-only replay uses the same actual h0-h2 observations, no synthetic images. Suffix comparison resets its RNG identically.'});print('PREFIX_ONLY_QA_PASS',flush=True)
if __name__=='__main__':main()
