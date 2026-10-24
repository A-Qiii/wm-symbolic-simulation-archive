from common import *
from model import QueryRelationReader,parse_query
import torch,fcntl
from torch.nn import functional as F
def main():
 assert not(E/'results/training.json').exists();q=read(E/'configs/protocol.json');torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 mm=read(BASE/'data/public/windows.json');xx=dict(np.load(BASE/'data/public/reader_inputs.npz'));tt=dict(np.load(BASE/'data/_private/reader_targets.npz'));tr=np.array([r['split']=='train'for r in mm]);dv=~tr;assert set(r['root_id']for r in mm if r['split']=='train').isdisjoint(r['root_id']for r in mm if r['split']=='dev')
 x={k:torch.tensor(v,device='cuda')for k,v in xx.items()};y=torch.tensor(tt['y'].astype(np.float32),device='cuda');valid=torch.tensor(tt['v'],device='cuda');w=[]
 for cols in [[0,1,2],[3,4,5],[6]]:
  a=tt['y'][tr][:,cols];v=tt['v'][tr][:,cols];n=[int((v&(a==b)).sum())for b in [0,1]];w.append([sum(n)/(2*z)for z in n])
 weights=torch.tensor(w,device='cuda');net=QueryRelationReader().cuda();frozen_before={k:v.detach().cpu().clone()for k,v in net.named_parameters()if not v.requires_grad};opt=torch.optim.AdamW((v for v in net.parameters() if v.requires_grad),lr=.001,weight_decay=.001)
 def losses(z,yy,vv):
  raw=F.binary_cross_entropy_with_logits(z,yy,reduction='none');out=[]
  for g,cols in enumerate([[0,1,2],[3,4,5],[6]]):
   v0=vv[:,cols];out.append((raw[:,cols]*torch.where(yy[:,cols]>.5,weights[g,1],weights[g,0])*v0).sum(1)/v0.sum(1).clamp_min(1))
  return torch.stack(out,1).mean(1)
 roots=sorted({r['root_id']for r in mm if r['split']=='train'});ri={r:np.array([i for i,m in enumerate(mm)if m['root_id']==r])for r in roots};devroots=sorted({r['root_id']for r in mm if r['split']=='dev'});dr=[np.array([i for i,m in enumerate(mm)if m['root_id']==r])for r in devroots];rng=np.random.default_rng(170150);schedule=[];logs=[];checkpoints=[]
 for step in range(1,1201):
  ix=np.array([rng.choice(ri[roots[j]])for j in rng.integers(len(roots),size=32)]);schedule.append(ix);net.train();opt.zero_grad(set_to_none=True);z=net(**{k:v[ix]for k,v in x.items()});loss=losses(z,y[ix],valid[ix]).mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
  if step%100==0:logs.append({'update':step,'loss':float(loss)});print(json.dumps(logs[-1]),flush=True)
  if step in [300,600,900,1200]:
   net.eval()
   with torch.inference_mode():z=net(**x);lv=losses(z,y,valid);score=float(torch.stack([lv[r].mean()for r in dr]).mean());pro=z.sigmoid().cpu().numpy()
   cp=E/'models'/f'query_seed0_update{step:04d}.pt';torch.save({'model':net.state_dict(),'update':step,'seed':0},cp);record={'update':step,'C_root_macro_BCE':score,'path':str(cp),'sha256':sha(cp),'C':metrics(pro,tt['y'],tt['v'],np.flatnonzero(dv)),'AB':metrics(pro,tt['y'],tt['v'],np.flatnonzero(tr))};checkpoints.append(record);print(json.dumps({'checkpoint':step,'C_BCE':score,'C_all7':record['C']['all7'],'AB_all7':record['AB']['all7']}),flush=True)
 best=min(checkpoints,key=lambda r:(r['C_root_macro_BCE'],r['update']));net.load_state_dict(torch.load(best['path'],weights_only=False)['model']);net.eval()
 with torch.inference_mode():pro=net(**x).sigmoid().cpu().numpy();nodes=net.nodes(**{k:v[:2]for k,v in x.items()});q0=torch.tensor([parse_query('Holding(alias)',{'alias':2}),parse_query('In(alias,drawer)',{'alias':2}),parse_query('Open(drawer)',{})],device='cuda');answers=net.query(nodes,q0).sigmoid().cpu().numpy();assert np.allclose(answers,pro[:2][:,[2,5,6]],atol=1e-6)
 # Object permutation plus matching query permutation must preserve answers.
 perm=[2,0,1]
 with torch.inference_mode():pp=net(objects=x['objects'][:2,:,perm],context=x['context'][:2],robot=x['robot'][:2]).sigmoid().cpu().numpy()
 assert np.allclose(pp,pro[:2][:,perm+[j+3 for j in perm]+[6]],atol=1e-6)
 assert all(torch.equal(v.detach().cpu(),frozen_before[k])for k,v in net.named_parameters()if not v.requires_grad);np.save(E/'results/reader_probabilities.npy',pro);np.save(E/'logs/train_sample_indices.npy',np.stack(schedule));save(E/'results/training.json',{'seed':0,'updates':1200,'parameters':sum(v.numel()for v in net.parameters()),'trainable_parameters':sum(v.numel()for v in net.parameters()if v.requires_grad),'frozen_encoder_identical_before_after':True,'old_classification_heads_not_copied_or_called':True,'class_weights_AB_only':w,'checkpoints':checkpoints,'selected':best,'alias_query_and_permutation_checks':True,'loss_logs':logs,'no_test_used':True});save(E/'configs/reader_frozen.json',best);print('READER_FROZEN',best['path'],flush=True)
if __name__=='__main__':main()
