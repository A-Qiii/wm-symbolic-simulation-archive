"""Historical frozen eight-class head: native causal RSSM history, contextual reference only."""
from pathlib import Path
import json,fcntl,time
import numpy as np
import torch
from wm_symbolic.world_model import WorldModel
from wm_symbolic.core_dataset import CoreEpisodes

P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_20261014'
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4);torch.manual_seed(0);torch.set_float32_matmul_precision('high')
 ck=torch.load(P/'runs/core1200_v1/training_main/best.pt',map_location='cpu',weights_only=False);m=WorldModel(skill_count=6,event_count=3).cuda().eval();m.load_state_dict(ck['model'])
 meta=json.loads((E/'data/metadata.json').read_text());lookup={(r['episode_id'],r['step']):r['index'] for r in meta};out=np.full((len(meta),7),np.nan,np.float32);start=time.monotonic()
 for split,role in [('dev','calibration'),('test',None)]:
  ds=CoreEpisodes(P,split,normalizer=ck['normalization'],dev_role=role)
  for i,e in enumerate(ds.episodes):
   b=ds.batch([i],'cuda')
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    encoded=m.encode(b['tokens'],b['geometry'],b['visible'],b['robot']).repeat(8,1,1);acts=m.action(b['skill'],b['arguments'],b['budget']).repeat(8,1,1);posts,_,_,_=m.history(encoded,acts)
    for step,s in enumerate(posts[:len(e['rows'])+1]):
     q=m.symbols(s);ob=(q['objects'].float()/1.69713068).softmax(-1).mean(0);dr=(q['drawer'].float()/1.69713068).softmax(-1).mean(0);pr=torch.cat([ob[:,6],ob[:,3],dr[2:3]])
     out[lookup[e['episode_id'],step]]=pr.cpu().numpy()
   if i%10==0:print('OLD_REFERENCE',split,i,flush=True)
 np.save(E/'results/frozen_old_native_history_probabilities.npy',out)
 (E/'results/old_reference_protocol.json').write_text(json.dumps({'role':'Historical context only. Not matched-input causal comparison: includes prior boundary/skill history and a much larger pretrained-on-AB RSSM. Main matched-eight vs direct uses identical four-frame inputs.','checkpoint_update':ck['update'],'temperature':1.69713068,'posterior_samples':8,'current_state_only':True,'future_frames_read_by_prediction':False,'seconds':time.monotonic()-start},indent=2))
if __name__=='__main__':main()
