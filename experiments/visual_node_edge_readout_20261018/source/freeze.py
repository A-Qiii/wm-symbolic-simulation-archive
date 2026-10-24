from common import *
import torch,shutil

def main():
 assert not(E/'configs/training_frozen.json').exists();assert read(E/'results/unit_tests.json')['passed']
 rows=read(E/'data/public/states.json');labs=read(E/'data/private/labels.json');cohorts=read(E/'configs/cohorts.json');vec=np.load(E/'data/public/vectors.npz');raw=np.load(E/'data/public/raw_features.npz')
 y=np.array([r['y']for r in labs],np.float32);v=np.array([r['v']for r in labs],np.float32);ixs={k:np.array(ix,np.int64)for k,ix in cohorts.items()}
 for head in HEADS:
  ix=ixs['train_'+head];g=slice(0,3)if head=='Holding'else slice(3,6)
  d={'y':y[ix,g],'v':v[ix,g],'indices':ix,'state_ids':np.array([rows[i]['state_id']for i in ix]),'root_ids':np.array([rows[i]['root_id']for i in ix])}
  for arm,key in [('node','node'),('edge',head)]:
   x=vec[key][ix];mean=x.mean((0,1));std=np.maximum(x.std((0,1)),.05);np.savez_compressed(E/f'configs/norm_{head}_{arm}.npz',mean=mean,std=std);d[arm]=((x-mean)/std).astype(np.float32)
  assert np.array_equal(d['node'],d['edge'][...,:74]);np.savez_compressed(E/f'data/train/{head}.npz',**d)
  for seed in(0,1):
   for arm in ARMS:
    torch.manual_seed(17083000+100*HEADS.index(head)+seed);q=make(arm)
    torch.save({'state':q.state_dict(),'initial_hash':tensor_hash(q.state_dict())},E/f'models/initial_{head}_{arm}_seed{seed}.pt')
 for name,ix in ixs.items():
  if name.startswith('train_'):continue
  np.savez_compressed(E/f'data/evaluation/{name}.npz',node=vec['node'][ix],Holding=vec['Holding'][ix],In=vec['In'][ix],y=y[ix],v=v[ix],indices=ix,state_ids=np.array([rows[i]['state_id']for i in ix]),root_ids=np.array([rows[i]['root_id']for i in ix]),scene=np.array([rows[i]['scene']for i in ix]),visibility=raw['geometry'][ix,...,4])
 # Exact-RGB inconsistent known labels are evidence of single-image ambiguity.
 by={};conflicts=[]
 for i,r in enumerate(rows):by.setdefault(r['RGB_content_sha256'],[]).append(i)
 for hh,ii in by.items():
  if len(ii)<2:continue
  for j in range(6):
   known=[i for i in ii if v[i,j]]
   if len(set(y[known,j]))>1:conflicts.append({'RGB_sha256':hh,'fact':j,'indices':known,'labels':y[known,j].tolist()})
 save(E/'results/exact_RGB_label_conflicts.json',{'groups':conflicts,'count':len(conflicts),'meaning':'Conflicting known labels for bit-identical two RGBs are observational ambiguity; absence does not prove force/contact observable.'})
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 files=[p for sub in['configs','source','data/train','data/evaluation','models']for p in(E/sub).rglob('*')if p.is_file()]
 save(E/'configs/training_frozen.json',{'utc':now(),'before_first_optimizer_update':True,'sha256':{str(p):sha(p)for p in files},'fits':8,'updates_each':600,'same_node_features_in_both_arms':True,'parameter_count_each_head':6689})
 print('NODE_EDGE_TRAINING_FROZEN',flush=True)
if __name__=='__main__':main()

