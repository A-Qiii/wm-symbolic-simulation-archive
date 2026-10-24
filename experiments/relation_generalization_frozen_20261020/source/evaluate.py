from common import *
import torch,fcntl
from models import ObjectAug
from features import load
def probabilities(m,s):
 p=m.symbols(s);return torch.cat([p['objects'].float().softmax(-1).flatten(1),p['drawer'].float().softmax(-1),p['slider'].float().softmax(-1),p['lights'].float().sigmoid()],-1)[0].cpu().numpy()
def action(skill):
 verb=0 if skill=='open_drawer'else 1 if skill=='close_drawer'else 3 if skill=='place_in_drawer'else 2;ptr=np.zeros(3,np.float32)
 if verb==2:ptr[COLORS.index(skill.split('_')[1])]=1
 return torch.tensor([verb],device='cuda'),torch.tensor(ptr[None],device='cuda'),torch.tensor([900.],device='cuda')
def main():
 guard();verify();budget();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True);eps=read(S/'cache/episodes.json');items=read(S/'cache/items.json')
 for seed in [0,1]:
  for arm in ARMS:
   budget();path=OLD/f'models/{arm}_seed{seed}_step1500.pt';assert sha(path)==read(S/'configs/models_frozen.json')['sha256'][str(path)];m=ObjectAug().cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval().requires_grad_(False);before=statehash(m);data=load(seed,arm);rows=[];prows=[]
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for e in eps:
     budget();labs=[read(r['labels_path'])for r in e['rows']];truths=[labs[0]['before']]+[l['after']for l in labs];meta={k:e[k]for k in ['episode_id','root_id','family_id','template','branch','held_color','target_color']};meta.update(seed=seed,arm=arm)
     def obs(prior,j):return m.observe(prior,data['objects'][j:j+1],data['context'][j:j+1],data['availability'][j:j+1])
     def add(mode,t,state):
      lab=truths[t];r=e['rows'][max(t-1,0)];result=read(Path(r['labels_path']).parent/'result_private.json')
      rows.append({**meta,'mode':mode,'step':t,'skill':r['skill']if t else'root','prob':probabilities(m,state),'cat':lab['categorical'],'cat_valid':lab['categorical_valid'],'status':result['status']if t else'root','task_success':bool(labs[t-1]['success'])if t else False})
     j=e['boundaries'][0];state=m.initial(data['objects'][j:j+1],data['context'][j:j+1],data['availability'][j:j+1]);posts=[state];priors=[];add('current',0,state)
     for t,r in enumerate(e['rows']):
      prior,_=m.imagine(state,*action(r['skill']));priors.append(prior);add('one_step',t+1,prior);state=obs(prior,e['boundaries'][t+1]);posts.append(state);add('current',t+1,state)
     for start in ([0,2]if e['template']=='recovery_seen6'else[0]):
      state=posts[start]
      for t in range(start,e['length']):state,_=m.imagine(state,*action(e['rows'][t]['skill']));add('free_h'+str(start),t+1,state)
     for j in e['probes']:
      item=items[j];t=item['action_index'];qdata=read(item['probe_labels_path']);control=item['completion_control'];q=next(q for q in qdata['probes']if q['completion_control']==control);state=obs(priors[t],j);window=[x['labels']for x in qdata['all_completion_controls']if control-4<=x['completion_control']<=control]
      stable_h=np.zeros(3,bool);stable_i=np.zeros(3,bool)
      if control>4 and len(window)==5:
       stable_h=np.array([[c==6 for c in x['categorical'][:3]]for x in window]).all(0);stable_i=np.array([[c==3 for c in x['categorical'][:3]]for x in window]).all(0);stable_i[COLORS.index(e['target_color'])]=False
      prows.append({**meta,'action_index':t,'step':t+1,'completion_control':control,'prob':probabilities(m,state),'cat':q['labels']['categorical'],'cat_valid':q['labels']['categorical_valid'],'continuous_Held':stable_h,'continuous_In':stable_i})
   assert statehash(m)==before
   for name,rr in [('predictions',rows),('probes',prows)]:
    assert rr;np.savez_compressed(S/f'results/{name}_{arm}_seed{seed}.npz',**{k:np.array([r[k]for r in rr])for k in rr[0]})
   save(S/f'results/evaluated_{arm}_seed{seed}.json',{'utc':now(),'episodes':len(eps),'predictions':len(rows),'probes':len(prows),'weights_unchanged':True,'checkpoint_sha256':sha(path)});del m,data;torch.cuda.empty_cache();print('EVALUATED',arm,seed,len(rows),len(prows),flush=True)
 save(S/'results/evaluation_complete.json',{'utc':now(),'models':4,'no_training':True,'threshold_changes':0,'future_observation_in_free_predictions':0,'probe_posteriors_not_fed_to_main_trajectory':True})
if __name__=='__main__':main()
