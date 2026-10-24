"""Four bounded CPU probes. No model selection; only training split labels accessible."""
from config import *
import torch,sys
from torch import nn

def features(z,kind):
 obj=z[:,:3];scene=np.repeat(z[:,3:4],3,axis=1);pool=np.repeat(obj.mean(1,keepdims=True),3,axis=1)
 return (obj if kind=='object'else np.concatenate([obj,scene,pool],axis=-1)).reshape(-1,64 if kind=='object'else 192).astype(np.float32)
def make(dim,kind):return nn.Linear(dim,1)if kind=='linear'else nn.Sequential(nn.Linear(dim,32),nn.SiLU(),nn.Linear(32,1))
def main():
 verify();cfg=read(E/'configs/protocol.json');assert not(E/'results/training.json').exists()
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=str(args[0])
   if '/data/validation/'in p or'/data/evaluator_D/'in p or str(COMBO)in p or str(REP)in p:raise RuntimeError('Probe training cannot access validation,D or original model data')
 sys.addaudithook(guard);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);data=dict(np.load(E/'data/train/states.npz'));rows=read(E/'data/train/objects.json');states=read(E/'data/train/states.json');root_ids=[s['root_id']for s in states for _ in range(3)];assert root_ids==[r['root_id']for r in rows];y=data['y'][:,3:6].reshape(-1).astype(np.float32);assert y.tolist()==[int(r['In'])for r in rows];weights=np.zeros(len(y),np.float32)
 for root in cfg['training_roots']:
  for label in [0,1]:
   ix=np.array([i for i,r in enumerate(rows)if r['root_id']==root and r['In']==bool(label)],int);assert len(ix)>0;weights[ix]=1/(len(cfg['training_roots'])*2*len(ix))
 assert np.isclose(weights.sum(),1);Y=torch.tensor(y);W=torch.tensor(weights);logs=[];hashes={}
 for feature in ['object','context']:
  x=features(data['z'],feature);mean=x.mean(0);std=np.maximum(x.std(0),.01);X=torch.tensor((x-mean)/std)
  np.savez_compressed(E/'models'/f'normalization_{feature}.npz',mean=mean,std=std)
  for arch in ['linear','mlp32']:
   torch.manual_seed(cfg['seed']);model=make(x.shape[1],arch);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);history=[]
   for update in range(1,501):
    logits=model(X).squeeze(-1);loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,Y,reduction='none')*W).sum();opt.zero_grad();loss.backward();opt.step()
    if update in [1,100,250,500]:history.append({'update':update,'train_weighted_loss':float(loss.detach())})
   name=f'{feature}_{arch}';path=E/'models'/f'{name}.pt';torch.save({'state':model.state_dict(),'feature':feature,'arch':arch,'dim':x.shape[1],'update':500,'seed':cfg['seed']},path);hashes[str(path)]=sha(path);logs.append({'name':name,'updates':500,'parameters':sum(p.numel()for p in model.parameters()),'history':history,'train_roots':len(set(root_ids)),'train_object_rows':len(rows),'model_device':'cpu'});print('PROBE_TRAINED',name,history[-1],flush=True)
  normpath=E/'models'/f'normalization_{feature}.npz';hashes[str(normpath)]=sha(normpath)
 save(E/'results/training.json',{'utc':now(),'runs':logs,'fit_count':4,'validation_reads_for_training':0,'D_reads_for_training':0,'world_model_or_encoder_updates':0,'original_query_updates':0,'early_stopping':False,'checkpoint_selection':'fixed500','train_only_normalization':True,'root_and_class_balanced_loss':True});save(E/'configs/probe_models_frozen.json',{'utc':now(),'before_validation_or_D_scoring':True,'sha256':hashes})
if __name__=='__main__':main()
