from common import *
import torch
from models import history
from mid_data import batch,crop
from inputs import cohort
from training import newmodel
def probabilities(m,state):
 p=m.symbols(state);return torch.cat([p['objects'].float().softmax(-1).flatten(1),p['drawer'].float().softmax(-1),p['slider'].float().softmax(-1),p['lights'].float().sigmoid()],-1).cpu().numpy()
def one(m,data,eps,source,arm,seed):
 entries=[(i,start)for i,e in enumerate(eps)for start in range(e['length'])];rows=[];started=time.time()
 for begin in range(0,len(entries),8):
  budget();part=entries[begin:begin+8];ids=[i for i,st in part];starts=[st for i,st in part];full=batch(data,ids);b=crop(full,starts);posts,priors=history(m,b['objects'],b['context'],b['availability'],b['verb'],b['pointer'],b['budget'])
  def add(mode,t,state):
   pp=probabilities(m,state)
   assert np.isfinite(pp).all()
   for j,(i,origin)in enumerate(part):
    e=eps[i];absolute=origin+t
    if absolute>e['length']:continue
    skill=e['rows'][absolute-1]['skill']if t else'initial_window';target=-1
    if t and skill.startswith('lift_'):target=COLORS.index(skill.split('_')[1])
    elif t and skill=='place_in_drawer':
     held=np.flatnonzero(b['cat'][j,t-1,:3].cpu().numpy()==6)
     if len(held)==1:target=int(held[0])
    rows.append({'source':source,'arm':arm,'seed':seed,'episode_id':e['episode_id'],'root_id':e['root_id'],'family_id':e['family_id'],'environment':e.get('environment',e['root_id'][0]),'state_kind':e.get('state_kind',-1),'branch':e['branch'],'branch_mode':e.get('branch_mode','historical_b'+str(e['branch'])),'origin':origin,'mode':mode,'step':absolute,'relative_step':t,'length':e['length'],'skill':skill,'target':target,'injected_miss_step':bool(t and e.get('branch_mode')=='gripper_miss'and absolute-1==e.get('fault_step')),'prob':pp[j],'cat':b['cat'][j,t].cpu().numpy(),'cat_valid':b['cat_valid'][j,t].cpu().numpy()})
  for t,state in enumerate(posts):add('current',t,state)
  for t,state in enumerate(priors):add('one_step',t+1,state)
  state=posts[0];add('free',0,state)
  for t in range(6):state,_=m.imagine(state,b['verb'][:,t],b['pointer'][:,t],b['budget'][:,t]);add('free',t+1,state)
  if source!='combination_test':
   # Only this initial recurrent state changes; observations/actions in prefix
   # are public and each history node is causally computed before its future.
   original_posts,_=history(m,full['objects'],full['context'],full['availability'],full['verb'],full['pointer'],full['budget'])
   state=torch.stack([original_posts[st][j]for j,st in enumerate(starts)]);add('warm_current',0,state);add('warm_free',0,state)
   for t in range(6):state,_=m.imagine(state,b['verb'][:,t],b['pointer'][:,t],b['budget'][:,t]);add('warm_free',t+1,state)
  if(begin//8+1)%20==0:print('EVAL',source,arm,seed,min(begin+8,len(entries)),len(entries),flush=True)
 out=S/f'results/predictions_{source}_{arm}_seed{seed}.npz';np.savez_compressed(out,**{k:np.array([r[k]for r in rows])for k in rows[0]})
 save(S/f'results/eval_{source}_{arm}_seed{seed}.json',{'utc':now(),'source':source,'arm':arm,'seed':seed,'episodes':len(eps),'origins':len(entries),'prediction_rows':len(rows),'seconds':time.time()-started,'sha256':sha(out)})
def phase(test=False):
 frozen=read(S/'configs/models_frozen.json')['sha256'];sources=['combination_test']if test else['old_development','development']
 if test:assert(S/'results/development_evaluation_complete.json').exists()
 for seed in [0,1]:
  for arm in ['old228','new868']:
   path=M/f'models/cold_mix_seed{seed}_step1500.pt'if arm=='old228'else S/f'models/new868_seed{seed}_step1500.pt';assert sha(path)==frozen[str(path)]
   m=newmodel(seed).cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval().requires_grad_(False);before=statehash(m)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for source in sources:
     if(S/f'results/eval_{source}_{arm}_seed{seed}.json').exists():continue
     data,eps=cohort(source,seed);one(m,data,eps,source,arm,seed);del data
   assert statehash(m)==before;del m;torch.cuda.empty_cache()
 save(S/f'results/{"test"if test else"development"}_evaluation_complete.json',{'utc':now(),'models':4,'weights_unchanged':True,'thresholds_unchanged':True,'checkpoint_selection':False})
