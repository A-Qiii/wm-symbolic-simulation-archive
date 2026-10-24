from common import *
import torch,fcntl
from models import history
from train import newmodel
from data import extract,load,batch
def probabilities(m,s):
 p=m.symbols(s);return torch.cat([p['objects'].float().softmax(-1).flatten(1),p['drawer'].float().softmax(-1),p['slider'].float().softmax(-1),p['lights'].float().sigmoid()],-1).cpu().numpy()
def assess(arm,seed,split):
 budget();data=load(split,seed,arm=arm);eps=read(S/f'cache/{split}_metadata.json');m=newmodel(arm,seed).cuda();path=S/f'models/{arm}_seed{seed}_step1500.pt';assert sha(path)==read(S/'configs/models_frozen.json')['sha256'][str(path)];m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval().requires_grad_(False);beforehash=statehash(m);rows=[]
 with torch.inference_mode():
  for start in range(0,len(eps),8):
   budget();ee=eps[start:start+8];b=batch(data,list(range(start,start+len(ee))));labs=[ [read(r['labels_path'])for r in e['rows']]for e in ee]
   with torch.autocast('cuda',dtype=torch.bfloat16):
    posts,priors=history(m,b['objects'],b['context'],b['availability'],b['verb'],b['pointer'],b['budget'])
    def add(mode,t,state,prev=None,detail=None):
     prob=probabilities(m,state);event=m.outcomes(prev,state,detail['action']).float().sigmoid().cpu().numpy()if detail is not None else np.full((len(ee),4),-1.);binding=detail['binding'].float().cpu().numpy()if detail is not None else np.full((len(ee),3),-1.)
     for j,e in enumerate(ee):
      if t>e['length']:continue
      l=labs[j][max(t-1,0)];r=e['rows'][max(t-1,0)];rows.append({'arm':arm,'seed':seed,'split':split,'kind':e['kind'],'root_id':e['root_id'],'family_id':e['family_id'],'episode_id':e['episode_id'],'branch':e['branch'],'held_color':e['held_color'],'target_color':e['target_color'],'valid_pair':e['valid_pair'],'mode':mode,'step':t,'prob':prob[j],'cat':b['cat'][j,t].cpu().numpy(),'cat_valid':b['cat_valid'][j,t].cpu().numpy(),'event_prob':event[j],'event_target':b['outcomes'][j,max(t-1,0)].cpu().numpy()if t else np.full(4,-1.),'events_valid':b['events_valid'][j,max(t-1,0)].cpu().numpy(),'binding':binding[j],'availability':b['availability'][j,t].cpu().numpy(),'skill':r['skill']if t else'root','status':('success'if l['success']else'executed_failure'if l['control_samples_including_start']>1 else'refused')if t else'root'})
    for t,p in enumerate(posts):add('current',t,p)
    for t,p in enumerate(priors):
     a=m.interface.action_features(b['verb'][:,t],b['pointer'][:,t],b['budget'][:,t]);nodes=posts[t]if arm=='object'else posts[t].reshape(-1,4,128);alpha=m.interface.binding(nodes,a,b['verb'][:,t],b['pointer'][:,t]);add('one_step',t+1,p,posts[t],{'action':a,'binding':alpha})
    free=posts[2]
    for t in range(2,6):
     prev=free;free,d=m.imagine(free,b['verb'][:,t],b['pointer'][:,t],b['budget'][:,t]);add('suffix',t+1,free,prev,d)
 assert statehash(m)==beforehash;np.savez_compressed(S/f'results/predictions_{arm}_seed{seed}_{split}.npz',**{k:np.array([r[k]for r in rows])for k in rows[0]});del m,data;torch.cuda.empty_cache();print('EVALUATED',arm,seed,split,len(eps),flush=True)
def main():
 guard();verify();assert(S/'configs/models_frozen.json').exists();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
 for seed in [0,1]:
  extract('historical_evaluation',seed)
  for arm in ARMS:
   for split in ['train','development','historical_evaluation','diagnosed_confirmation']:assess(arm,seed,split)
 save(S/'results/evaluation_complete.json',{'utc':now(),'models':4,'checkpoints_each':1,'samples_per_prediction':1,'future_observations_used':0,'CD_samples':0})
if __name__=='__main__':main()
