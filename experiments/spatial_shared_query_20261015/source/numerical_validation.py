"""Investigate CPU/GPU evaluation differences without changing the model."""
from common import *
from model import QueryRelationReader,decode
from train import load_inputs
import torch

def main():
 torch.set_num_threads(4);raw=load_inputs();x={k:torch.tensor(v)for k,v in raw.items()};meta=read(BASE/'data/metadata.json');out={}
 for arm in ['pooled','spatial']:
  cp=read(E/'configs'/arm/'reader_frozen.json');assert sha(cp['path'])==cp['sha256'];net=QueryRelationReader(arm).eval();net.load_state_dict(torch.load(cp['path'],weights_only=False,map_location='cpu')['model'])
  with torch.inference_mode():p=net.probabilities(**x).numpy()
  original=np.load(E/'results'/arm/'probabilities.npy');diff=np.abs(p-original);changed=np.argwhere(decode(p)!=decode(original));i,j=np.unravel_index(diff.argmax(),diff.shape)
  out[arm]={'rows':len(p),'max_absolute_difference':float(diff.max()),'mean_absolute_difference':float(diff.mean()),'maximum_at':{'window_id':meta[i]['window_id'],'fact':FACTS[j],'CPU_probability':float(p[i,j]),'saved_GPU_probability':float(original[i,j])},'decoded_fact_changes':changed.tolist(),'decoded_fact_changes_count':len(changed),'candidate_threshold_unchanged':0.5}
  np.save(E/'results'/arm/'cpu_validation_probabilities.npy',p)
  print(arm,json.dumps(out[arm]),flush=True)
 save(E/'results/numerical_validation.json',out)
 assert all(v['decoded_fact_changes_count']==0 for v in out.values())
if __name__=='__main__':main()
