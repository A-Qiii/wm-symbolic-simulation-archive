from common import *
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch,fcntl
from wm_symbolic.world_model import symbol_nll
def unpack(v):return {'objects':v[:,:24].reshape(-1,3,8),'drawer':v[:,24:27],'slider':v[:,27:30],'lights':v[:,30:]}
def probabilities(v):
 p=unpack(v);return torch.cat([p['objects'].softmax(-1).flatten(-2),p['drawer'].softmax(-1),p['slider'].softmax(-1),p['lights'].sigmoid()],-1)
def train():
 verify();guard('train');torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');norm=read(E/'configs/normalization.json');w=[torch.tensor(v,device='cuda')for v in norm['class_weights']];runs=[]
 for seed in [0,1]:
  for arm in ARMS:
   path=E/f'models/probe_{arm}_seed{seed}.pt';receipt=E/f'results/probe_train_{arm}_seed{seed}.json'
   if receipt.exists():runs.append(read(receipt));continue
   a=dict(np.load(E/f'data/features_{arm}_seed{seed}_train.npz'));roots=sorted(set(a['root_id']));assert len(roots)==60 and len(a['x'])==900
   rng=np.random.default_rng(261019600+seed);groups=[np.where(a['root_id']==r)[0]for r in roots];schedule=np.array([[g[rng.integers(len(g))]for g in[groups[j]for j in rng.integers(len(groups),size=32)]]for _ in range(600)])
   sp=E/f'data/probe_schedule_seed{seed}.npy'
   if sp.exists():assert np.array_equal(schedule,np.load(sp))
   else:np.save(sp,schedule)
   mu=a['x'].mean(0);sd=np.maximum(a['x'].std(0),1e-4);x=torch.tensor((a['x']-mu)/sd,device='cuda');targets={k:torch.tensor(a[k],device='cuda')for k in ['cat','cat_valid','binary','binary_valid']}
   torch.manual_seed(261019600+seed);m=torch.nn.Linear(512,32).cuda();initial=th(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);curve=[]
   for u,ix in enumerate(schedule,1):
    ii=torch.tensor(ix,device='cuda');pred=unpack(m(x[ii]));loss=symbol_nll(pred,targets['cat'][ii],targets['cat_valid'][ii],targets['binary'][ii],targets['binary_valid'][ii],w).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    if u%100==0:curve.append({'update':u,'loss':float(loss.detach())})
   torch.save({'state':m.state_dict(),'mean':torch.tensor(mu),'std':torch.tensor(sd),'arm':arm,'seed':seed,'updates':600},path);rec={'utc':now(),'arm':arm,'seed':seed,'updates':600,'parameters':16416,'initial_hash':initial,'final_hash':th(m.state_dict()),'path_sha256':sha(path),'schedule_sha256':sha(sp),'train_roots':60,'train_records':900,'C_samples_read':0,'curve':curve};save(receipt,rec);runs.append(rec);print('PROBE_FIT_COMPLETE',arm,seed,flush=True)
 save(E/'configs/probes_frozen.json',{'utc':now(),'before_dev_probe_prediction':True,'sha256':{str(p):sha(p)for p in(E/'models').glob('probe_*.pt')}});save(E/'results/probe_training.json',{'fits':runs})
def predict():
 guard('evaluate');torch.set_num_threads(4)
 for p,h in read(E/'configs/probes_frozen.json')['sha256'].items():assert sha(p)==h
 for seed in [0,1]:
  for arm in ARMS:
   ck=torch.load(E/f'models/probe_{arm}_seed{seed}.pt',map_location='cuda',weights_only=True);m=torch.nn.Linear(512,32).cuda();m.load_state_dict(ck['state']);m.eval()
   for split in ['train','dev']:
    a=dict(np.load(E/f'data/features_{arm}_seed{seed}_{split}.npz'));x=torch.tensor(a['x'],device='cuda')
    with torch.inference_mode():p=probabilities(m((x-ck['mean'])/ck['std'])).cpu().numpy()
    z={k:a[k]for k in ['root_id','episode_id','role','step','cat','cat_valid','binary','binary_valid']};n=len(p)
    z.update(prob=p,mode=np.array(['probe']*n),split=np.array([split]*n),base_cat=a['cat'],base_valid=a['cat_valid'],base_binary=a['binary'],base_binary_valid=a['binary_valid'],controls=np.full(n,-1),status=np.array(['probe_boundary']*n),skill=np.array(['probe']*n),all_prefix_actuated=np.zeros(n,bool),event_prob=np.full((n,4),np.nan),event_target=np.full((n,4),np.nan),event_valid=np.zeros((n,4),bool))
    np.savez_compressed(E/f'data/probe_predictions_{arm}_seed{seed}_{split}.npz',**z)
 print('PROBE_PREDICTIONS_COMPLETE',flush=True)
if __name__=='__main__':
 import sys
 lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 {'train':train,'predict':predict}[sys.argv[1]]()
