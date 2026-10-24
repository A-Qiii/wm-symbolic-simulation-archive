from common import *
import torch,fcntl,argparse
from model import ObservationStateWorldModel
from dataset import NewEpisodes
def probabilities(m,state):
 p=m.symbols(state);return torch.cat([p['objects'].float().softmax(-1).flatten(-2),p['drawer'].float().softmax(-1),p['slider'].float().softmax(-1),p['lights'].float().sigmoid()],-1).mean(0).cpu().numpy()
def assess(m,data,variant,seed,arm):
 records=[];pair={};begin=time.monotonic();audits={x['root_id']:x for x in read(S/'results/root_audits.json')}
 with torch.inference_mode():
  for ix,ep in enumerate(data.episodes):
   budget_check();root=ep['root'];b=data.batch([ix],'cuda');torch.manual_seed(81831+int(root['seed'])+COLORS.index(root['held_color'])*10+COLORS.index(root['matched_color']));labels=[ep['labels'][0]['before']]+[x['after']for x in ep['labels']]
   with torch.autocast('cuda',dtype=torch.bfloat16):
    enc=m.encode(b['tokens'],b['geometry'],b['visible'],b['robot']);actions=m.action(b['skill'],b['arguments'],b['budget']);ee=enc.repeat_interleave(8,0);aa=actions.repeat_interleave(8,0);posts,qs,priors,ps=m.history(ee,aa)
    def row(mode,t,state,base,event=None):
     j=max(0,t-1);l=ep['labels'][j];r=ep['rows'][j];records.append({'root_id':ep['root_id'],'family_id':root['family_id'],'episode_id':ep['episode_id'],'split':data.split,'branch':ep['branch'],'environment':root['environment'],'held_color':root['held_color'],'target_color':root['matched_color'],'mode':mode,'step':t,'base':base,'prob':probabilities(m,state),'cat':labels[t]['categorical'],'cat_valid':labels[t]['categorical_valid'],'controls':l['control_samples_including_start']-1 if t else -1,'status':('success'if l['success']else'executed_failure'if l['control_samples_including_start']>1 else'refused')if t else'root','skill':r['skill']if t else'root','event_prob':np.asarray(event)if event is not None else np.full(4,-1.),'event_target':np.array([float(l['success'])]+l['events'])if t else np.full(4,-1.),'hidden_fault_action':ep['branch']==2 and t==2,'valid_physical_pair':audits[root['root_id']]['valid_pair'],'in_arm_training':arm in root['arms']})
    for t in range(7):row('current',t,posts[t],t)
    for t in range(6):
     event=m.outcome_logits(posts[t],aa[:,t],priors[t]).float().sigmoid().mean(0).cpu().numpy();row('one_step',t+1,priors[t],t,event);row('persistence',t+1,posts[t],t)
    free=posts[2]
    for t in range(2,6):
     before=free;free,_=m.imagine(free,aa[:,t]);event=m.outcome_logits(before,aa[:,t],free).float().sigmoid().mean(0).cpu().numpy();row('suffix',t+1,free,2,event)
    pair[root['root_id'],ep['branch']]=[probabilities(m,posts[0]),probabilities(m,posts[1]),probabilities(m,priors[1]),enc[0,:2].float().cpu().numpy()]
  for rid in data.root_ids:assert all(np.array_equal(x,y)for x,y in zip(pair[rid,1],pair[rid,2])),rid
 name=f'{variant}_seed{seed}_{data.split}';np.savez_compressed(S/f'results/predictions_{name}.npz',**{k:np.array([r[k]for r in records])for k in records[0]});rec={'utc':now(),'variant':variant,'arm':arm,'seed':seed,'split':data.split,'families':len({e['family_id']for e in data.episodes}),'roles':len(data.root_ids),'episodes':len(data.episodes),'seconds':time.monotonic()-begin,'paired_prefault_exact':True,'future_observations_used':0,'samples':8};save(S/f'results/inference_{name}.json',rec);print('EVALUATED',name,rec['episodes'],flush=True);return rec
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--phase',choices=['train_dev','locked_test'],required=True);args=pa.parse_args();guard('eval');verify_freeze();budget_check();seal=read(S/'configs/models_frozen.json')
 if args.phase=='locked_test':assert seal['utc']<read(S/'results/test_collection_started.json')['utc'];assert read(S/'results/audit_locked_test.json')['passed']
 lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True);reports=[]
 for seed in [0,1]:
  for arm in ARMS:
   selected=next(x for x in seal['selection']if x['seed']==seed and x['arm']==arm);steps=[250,500,1000]if args.phase=='train_dev'else sorted({selected['update'],1000})
   for step in steps:
    path=S/f'models/{arm}_seed{seed}_step{step}.pt';assert sha(path)==seal['sha256'][str(path)];m=ObservationStateWorldModel(skill_count=6,event_count=3).cuda();cp=torch.load(path,map_location='cuda',weights_only=False);m.load_state_dict(cp['model']);m.eval().requires_grad_(False);before=statehash(m)
    for split in ['train','development']if args.phase=='train_dev'else['locked_test']:
     variant=f'{arm}_step{step}';out=S/f'results/inference_{variant}_seed{seed}_{split}.json'
     if out.exists():reports.append(read(out));continue
     reports.append(assess(m,NewEpisodes(split),variant,seed,arm))
    assert statehash(m)==before;del m,cp;torch.cuda.empty_cache()
 save(S/f'results/inference_{args.phase}_complete.json',{'utc':now(),'reports':reports,'weights_unchanged':True,'no_test_selection':True})
if __name__=='__main__':main()
