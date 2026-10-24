from common import *
import torch,fcntl
from models import history
from mid_data import load,batch,crop
from mid_train import newmodel
def probabilities(m,s):
 p=m.symbols(s);return torch.cat([p['objects'].float().softmax(-1).flatten(1),p['drawer'].float().softmax(-1),p['slider'].float().softmax(-1),p['lights'].float().sigmoid()],-1).cpu().numpy()
def write_arrays(path,rr):np.savez_compressed(path,**{k:np.array([r[k]for r in rr])for k in rr[0]})
def eval_development(m,seed,arm):
 data=load('development',seed);eps=read(O/'cache/development_metadata.json');entries=[(i,s)for i,e in enumerate(eps)for s in range(e['length'])];rows=[]
 for begin in range(0,len(entries),8):
  part=entries[begin:begin+8];ids=[i for i,s in part];starts=[s for i,s in part];b=crop(batch(data,ids),starts);posts,priors=history(m,b['objects'],b['context'],b['availability'],b['verb'],b['pointer'],b['budget'])
  def add(mode,t,state):
   pp=probabilities(m,state)
   for j,(i,origin)in enumerate(part):
    e=eps[i];absolute=origin+t
    if absolute>e['length']:continue
    rows.append({'source':'development','arm':arm,'seed':seed,'episode_id':e['episode_id'],'root_id':e['root_id'],'family_id':e['family_id'],'template':e['kind'],'branch':e['branch'],'origin':origin,'mode':mode,'step':absolute,'relative_step':t,'skill':e['rows'][absolute-1]['skill']if t else'initial_window','held_color':e['held_color'],'target_color':e['target_color'],'prob':pp[j],'cat':b['cat'][j,t].cpu().numpy(),'cat_valid':b['cat_valid'][j,t].cpu().numpy()})
  for t,s in enumerate(posts):add('current',t,s)
  for t,s in enumerate(priors):add('one_step',t+1,s)
  state=posts[0]
  for t in range(6):state,_=m.imagine(state,b['verb'][:,t],b['pointer'][:,t],b['budget'][:,t]);add('free',t+1,state)
  # Separate original h2-anchored prediction, after observing actual prefix.
  state=posts[2]
  for t in range(2,6):
   state,_=m.imagine(state,b['verb'][:,t],b['pointer'][:,t],b['budget'][:,t]);add('free_after2',t+1,state)
 write_arrays(S/f'results/development_{arm}_seed{seed}.npz',rows);print('DEV',arm,seed,len(rows),flush=True)
def general_data(root,seed):
 data=dict(np.load(root/f'cache/front_seed{seed}.npz'));raw=dict(np.load(root/'cache/geometry_raw.npz'));norm=read(S/'configs/input_normalization.json')['values'][f'geometry_seed{seed}']
 for name,x in raw.items():
  n=norm[name];out=(x-np.array(n['mean'],np.float32))/np.array(n['std'],np.float32);width=8 if name=='objects'else 4
  for j in range(0,x.shape[-1],width):out[...,j:j+width]*=x[...,j+width-1:j+width]>0
  data[name]=np.concatenate([data[name],out.astype(np.float32)],-1)
 return{k:torch.from_numpy(v).cuda()for k,v in data.items()}
def action(skill):
 v=0 if skill=='open_drawer'else 1 if skill=='close_drawer'else 3 if skill=='place_in_drawer'else 2;ptr=np.zeros(3,np.float32)
 if v==2:ptr[COLORS.index(skill.split('_')[1])]=1
 return torch.tensor([v],device='cuda'),torch.tensor(ptr[None],device='cuda'),torch.tensor([900.],device='cuda')
def eval_general(m,seed,arm,root,source):
 data=general_data(root,seed);eps=read(root/'cache/episodes.json');items=read(root/'cache/items.json');rows=[];prows=[]
 for e in eps:
  labs=[read(r['labels_path'])for r in e['rows']];truths=[labs[0]['before']]+[l['after']for l in labs];meta={k:e[k]for k in ['episode_id','root_id','family_id','template','branch','held_color','target_color']};meta.update(source=source,seed=seed,arm=arm,origin=0)
  def obs(prior,j):return m.observe(prior,data['objects'][j:j+1],data['context'][j:j+1],data['availability'][j:j+1])
  def add(mode,t,state):rows.append({**meta,'mode':mode,'step':t,'relative_step':t,'skill':e['rows'][t-1]['skill']if t else'initial_window','prob':probabilities(m,state)[0],'cat':truths[t]['categorical'],'cat_valid':truths[t]['categorical_valid']})
  j=e['boundaries'][0];state=m.initial(data['objects'][j:j+1],data['context'][j:j+1],data['availability'][j:j+1]);posts=[state];priors=[];add('current',0,state)
  for t,r in enumerate(e['rows']):
   prior,_=m.imagine(state,*action(r['skill']));priors.append(prior);add('one_step',t+1,prior);state=obs(prior,e['boundaries'][t+1]);posts.append(state);add('current',t+1,state)
  for start in ([0,2]if e['template']=='recovery_seen6'else[0]):
   state=posts[start]
   for t in range(start,e['length']):state,_=m.imagine(state,*action(e['rows'][t]['skill']));add('free'if start==0 else'free_after2',t+1,state)
  for j in e.get('probes',[]):
   it=items[j];t=it['action_index'];qd=read(it['probe_labels_path']);ctl=it['completion_control'];q=next(q for q in qd['probes']if q['completion_control']==ctl);window=[x['labels']for x in qd['all_completion_controls']if ctl-4<=x['completion_control']<=ctl];hh=np.zeros(3,bool);ii=np.zeros(3,bool)
   if ctl>4 and len(window)==5:
    hh=np.array([[c==6 for c in x['categorical'][:3]]for x in window]).all(0);ii=np.array([[c==3 for c in x['categorical'][:3]]for x in window]).all(0);ii[COLORS.index(e['target_color'])]=False
   prows.append({**meta,'action_index':t,'completion_control':ctl,'prob':probabilities(m,obs(priors[t],j))[0],'cat':q['labels']['categorical'],'cat_valid':q['labels']['categorical_valid'],'continuous_Held':hh,'continuous_In':ii})
 write_arrays(S/f'results/{source}_{arm}_seed{seed}.npz',rows)
 if prows:write_arrays(S/f'results/{source}_probes_{arm}_seed{seed}.npz',prows)
 print('GENERAL',source,arm,seed,len(rows),len(prows),flush=True)
def main():
 guard();verify();budget();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True);fresh='--fresh'in sys.argv
 for seed in [0,1]:
  for arm in ARMS:
   path=S/f'models/{arm}_seed{seed}_step1500.pt';assert sha(path)==read(S/'configs/models_frozen.json')['sha256'][str(path)];m=newmodel(seed).cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval().requires_grad_(False);before=statehash(m)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    if fresh:eval_general(m,seed,arm,S/'fresh_validation','independent')
    else:eval_development(m,seed,arm);eval_general(m,seed,arm,G,'exploratory')
   assert before==statehash(m);del m;torch.cuda.empty_cache();budget()
 save(S/f'results/{"independent"if fresh else"development"}_evaluation_complete.json',{'utc':now(),'models':6,'weights_unchanged':True,'thresholds_unchanged':True,'new_training_samples':0})
if __name__=='__main__':main()
