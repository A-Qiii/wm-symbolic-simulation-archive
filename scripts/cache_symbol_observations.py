"""Freeze observed symbolic beliefs for the symbols-only dynamics baseline."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import Episodes
from wm_symbolic.gpu_phase import gpu_phase
from wm_symbolic.checkpointing import PauseRequest
from wm_symbolic.evaluation import mean_symbols
from wm_symbolic.symbol_transition import SymbolTransition
P=Path('/home/robot/research/projects/wm_symbolic');a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--split',choices=['train','dev','calibration','test'],required=True);args=a.parse_args()
checkpoint=Path(args.checkpoint);sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest();ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert ck['identity']['mode']=='formal';data=Episodes(P,args.split,normalizer=ck['normalization'],event_count=7)
assert data.transitions=={'train':4320,'dev':360,'calibration':360,'test':960}[args.split]
out=P/'data/features/symbol_beliefs'/sha/args.split;out.mkdir(parents=True,exist_ok=True);pause=PauseRequest();torch.set_num_threads(4)
model=WorldModel(event_count=7).eval();model.load_state_dict(ck['model']);items=[]
with gpu_phase(P,'cache_frozen_symbol_observations'),torch.inference_mode():
 model.cuda()
 for i,episode in enumerate(data.episodes):
  path=out/(episode['episode_id']+'.npz')
  if not path.exists():
   torch.manual_seed(8139+i);b=data.batch([i],'cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):
    e=model.encode(b['tokens'],b['geometry'],b['visible'],b['robot']).repeat_interleave(8,0)
    a=model.action(b['skill'],b['arguments'],b['budget']).repeat_interleave(8,0);posts,_,_,_=model.history(e,a)
    mean=[];sample=[]
    for state in posts:
     raw=model.symbols(state);mean.append(SymbolTransition.from_symbols(mean_symbols(raw,1,8)).h[0].float().cpu().numpy())
     sample.append(SymbolTransition.from_symbols(raw).h.float().cpu().numpy())
   temporary=path.with_suffix('.tmp.npz');np.savez(temporary,mean_logits=np.stack(mean),sample_logits=np.stack(sample));temporary.replace(path)
  items.append({'episode_id':episode['episode_id'],'root_id':episode['root_id'],'path':str(path)})
  print('SYMBOL_CACHE',episode['episode_id'],flush=True)
  if pause.requested:
   (P/'runs/PAUSED_BY_USER.json').write_text(json.dumps({'status':'paused','stage':'symbol_cache','split':args.split,'resume_requires_user_authorization':True},indent=2));raise SystemExit(0)
index={'perception_checkpoint':str(checkpoint),'perception_checkpoint_sha256':sha,'split':args.split,'items':items,
       'source':'Frozen full-model current posterior from public observation history; no GT labels are cached as dynamics inputs',
       'state_dimensions':32,'samples':8,'mean_logits_semantics':'Log mean categorical probabilities and logit mean independent probabilities'}
(out/'index.json').write_text(json.dumps(index,indent=2));print('SYMBOL_CACHE_COMPLETE',len(items),str(out/'index.json'))
