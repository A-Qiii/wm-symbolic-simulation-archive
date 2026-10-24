from common import *
import torch,fcntl
from wm_symbolic.world_model import WorldModel
from dataset import NewEpisodes
def probabilities(m,state):
 p=m.symbols(state)
 return torch.cat([p['objects'].float().softmax(-1).flatten(-2),p['drawer'].float().softmax(-1),p['slider'].float().softmax(-1),p['lights'].float().sigmoid()],-1).mean(0).cpu().numpy()
def arrays(path,rows):np.savez_compressed(path,**{k:np.array([r[k]for r in rows])for k in rows[0]})
def assess(m,data,variant,seed):
 records=[];latent=[];pair={};begin=time.monotonic()
 with torch.inference_mode():
  for ix,ep in enumerate(data.episodes):
   root=ep['root'];b=data.batch([ix],'cuda');torch.manual_seed(81831+int(root['seed']));labels=[ep['labels'][0]['before']]+[x['after']for x in ep['labels']]
   with torch.autocast('cuda',dtype=torch.bfloat16):
    enc=m.encode(b['tokens'],b['geometry'],b['visible'],b['robot']);actions=m.action(b['skill'],b['arguments'],b['budget']);ee=enc.repeat_interleave(8,0);aa=actions.repeat_interleave(8,0);posts,qs,priors,ps=m.history(ee,aa)
    def row(mode,t,state,base,event=None):
     j=max(0,t-1);l=ep['labels'][j];r=ep['rows'][j];records.append({'root_id':ep['root_id'],'episode_id':ep['episode_id'],'split':data.split,'branch':ep['branch'],'environment':root['environment'],'held_color':root['held_color'],'target_color':root['primary_color']if ep['branch']==0 else root['matched_color'],'mode':mode,'step':t,'base':base,'prob':probabilities(m,state),'cat':labels[t]['categorical'],'cat_valid':labels[t]['categorical_valid'],'controls':l['control_samples_including_start']-1 if t else -1,'status':('success'if l['success']else'executed_failure'if l['control_samples_including_start']>1 else'refused')if t else'root','skill':r['skill']if t else'root','success_prob':float(event)if event is not None else -1.,'success_target':float(l['success'])if t else -1.,'hidden_fault_action':ep['branch']==2 and t==2})
    for t in range(7):
     row('current',t,posts[t],t)
     latent.append({'root_id':ep['root_id'],'episode_id':ep['episode_id'],'step':t,'E':enc[0,t].float().cpu().numpy(),'h':posts[t].h.float().mean(0).cpu().numpy(),'q':qs[t].float().mean(0).cpu().numpy()})
    for t in range(6):
     event=m.outcome_logits(posts[t],aa[:,t],priors[t]).float().sigmoid()[:,0].mean().cpu();row('one_step',t+1,priors[t],t,event);row('persistence',t+1,posts[t],t)
    # No encoder or observe call inside this suffix: all three predictions use only h2 and requested actions.
    free=posts[2]
    for t in range(2,5):
     before=free;free,_=m.imagine(free,aa[:,t]);ev=m.outcome_logits(before,aa[:,t],free).float().sigmoid()[:,0].mean().cpu();row('suffix',t+1,free,2,ev)
    if ep['branch']in [1,2]:pair[ep['root_id'],ep['branch']]=[probabilities(m,posts[0]),probabilities(m,posts[1]),probabilities(m,priors[1]),enc[0,:2].float().cpu().numpy()]
  for rid in data.root_ids:
   assert all(np.array_equal(a,b)for a,b in zip(pair[rid,1],pair[rid,2])),('pre-fault paired RNG/input mismatch',rid)
 name=f'{variant}_seed{seed}_{data.split}';arrays(S/f'results/predictions_{name}.npz',records);arrays(S/f'results/latent_{name}.npz',latent);receipt={'utc':now(),'variant':variant,'seed':seed,'split':data.split,'roots':len(data.root_ids),'episodes':len(data.episodes),'prediction_rows':len(records),'seconds':time.monotonic()-begin,'paired_prefault_probabilities_exact':True,'suffix_future_observations_used':0,'samples':8};save(S/f'results/inference_{name}.json',receipt);print('EVALUATED',name,len(records),flush=True);return receipt
def main():
 guard('eval');verify_freeze();assert read(S/'results/audit_evaluation.json')['passed'];assert read(S/'results/families_evaluation.json')['passed'];seal=read(S/'configs/models_frozen.json');assert seal['utc']<read(S/'results/evaluation_collection_started.json')['utc'];lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True);reports=[]
 for seed in [0,1]:
  specs=[('original',CELL/f'models/joint_seed{seed}/final6000.pt')]
  for arm in ARMS:
   selected=next(x for x in seal['selection']if x['seed']==seed and x['arm']==arm);specs.append((arm,Path(selected['path'])));final=S/f'models/{arm}_seed{seed}_step1000.pt'
   if final!=Path(selected['path']):specs.append((arm+'_final1000',final))
  for variant,path in specs:
   expected=seal['sha256'].get(str(path));assert expected is None or sha(path)==expected;m=WorldModel(skill_count=6,event_count=3).cuda();cp=torch.load(path,map_location='cuda',weights_only=False);m.load_state_dict(cp['model']);m.eval().requires_grad_(False);before=statehash(m)
   for split in ['development','evaluation']:
    out=S/f'results/inference_{variant}_seed{seed}_{split}.json'
    if out.exists():reports.append(read(out));continue
    reports.append(assess(m,NewEpisodes(split),variant,seed))
   assert statehash(m)==before;del m,cp;torch.cuda.empty_cache()
 save(S/'results/inference_complete.json',{'utc':now(),'reports':reports,'weights_unchanged':True,'no_test_selection':True})
if __name__=='__main__':main()
