from common import *
from model import QueryRelationReader,decode,parse_query
import torch,fcntl,argparse
from torch.nn import functional as F
def metrics(p,y,v,ids):
 p,y,v=p[ids],y[ids],v[ids];d=decode(p);c=(d==y)&v;full=v.all(1);return {'windows':len(ids),'fully_valid9':int(full.sum()),'all9_correct_fully_valid':int(c[full].all(1).sum()),'valid_facts':int(v.sum()),'correct_valid_facts':int(c.sum()),'Holding_all3_correct':int(c[:,:3].all(1).sum()),'facts':{FACTS[j]:{'positive':int((y[:,j]&v[:,j]).sum()),'negative':int((~y[:,j]&v[:,j]).sum()),'TP':int(((d[:,j]==1)&y[:,j]&v[:,j]).sum()),'TN':int(((d[:,j]==0)&~y[:,j]&v[:,j]).sum()),'FP':int(((d[:,j]==1)&~y[:,j]&v[:,j]).sum()),'FN':int(((d[:,j]==0)&y[:,j]&v[:,j]).sum()),'UNKNOWN':int(((d[:,j]<0)&v[:,j]).sum())}for j in range(9)}}
def load_inputs():
 raw=dict(np.load(BASE/'data/inputs.npz'));raw.update(dict(np.load(E/'data/spatial_inputs.npz')));return raw
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['pooled','spatial'],required=True);a=ap.parse_args();variant=a.variant;assert not(E/'results'/variant/'training.json').exists()
 for path,h in read(E/'configs/training_frozen.json')['sha256'].items():assert sha(path)==h,path
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=read(BASE/'data/metadata.json');raw=load_inputs();t=dict(np.load(BASE/'data/targets.npz'));y0=t['y'];v0=t['v'];tr=np.array([r['split']=='train'for r in m]);new=np.array([r['pool']=='new_semantics'for r in m]);cnew=np.flatnonzero(~tr&new);cold=np.flatnonzero(~tr&~new);x={k:torch.tensor(v,device='cuda')for k,v in raw.items()};y=torch.tensor(y0.astype(np.float32),device='cuda');valid=torch.tensor(v0,device='cuda');weights=[]
 for cols in [[0,1,2],[3,4,5]]:
  aa=y0[tr][:,cols];vv=v0[tr][:,cols];counts=[int(((aa==k)&vv).sum())for k in[0,1]];weights.append([sum(counts)/(2*z)for z in counts])
 dc=np.bincount(y0[tr,6:].argmax(1)[v0[tr,6:].all(1)],minlength=3);dw=torch.tensor((dc.sum()/(3*dc)).astype(np.float32),device='cuda');bw=torch.tensor(weights,device='cuda')
 def losses(logits,yy,vv):
  bce=F.binary_cross_entropy_with_logits(logits[:,:6],yy[:,:6],reduction='none');loss=[];have=[]
  for g,cols in enumerate([[0,1,2],[3,4,5]]):
   val=vv[:,cols];loss.append((bce[:,cols]*torch.where(yy[:,cols]>.5,bw[g,1],bw[g,0])*val).sum(1)/val.sum(1).clamp_min(1));have.append(val.any(1))
  dv=vv[:,6:].all(1);loss.append(F.cross_entropy(logits[:,6:],yy[:,6:].argmax(1),weight=dw,reduction='none')*dv);have.append(dv);return torch.stack(loss,1).sum(1)/torch.stack(have,1).sum(1).clamp_min(1)
 net=QueryRelationReader(variant).cuda();frozen={n:p.detach().cpu().clone()for n,p in net.named_parameters()if not p.requires_grad};initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()for k,v in sorted(net.state_dict().items()))).hexdigest();net.eval()
 with torch.inference_mode():initial=net.probabilities(**x).cpu().numpy()
 reference=np.load(BASE/'results/probabilities.npy');maxdiff=float(np.abs(initial-reference).max());assert np.allclose(initial,reference,atol=1e-6),maxdiff
 save(E/'configs'/variant/'initial_checks.json',{'initial_model_state_sha256':initial_hash,'max_probability_difference_from_original':maxdiff,'same_initial_function_verified':True,'trainable_parameters':sum(p.numel()for p in net.parameters()if p.requires_grad),'frozen_parameters':sum(p.numel()for p in net.parameters()if not p.requires_grad)})
 opt=torch.optim.AdamW([p for p in net.parameters()if p.requires_grad],lr=.0005,weight_decay=.001);cr=sorted({m[i]['root_id']for i in cnew});cri=[np.array([i for i in cnew if m[i]['root_id']==r])for r in cr];oroot=sorted({m[i]['root_id']for i in cold});ori=[np.array([i for i in cold if m[i]['root_id']==r])for r in oroot];schedule=np.load(E/'data/train_schedule.npy');assert schedule.shape==(1200,32);assert all(m[i]['split']=='train'for row in schedule for i in row);log=[];checkpoints=[]
 for step,ix in enumerate(schedule,1):
  net.train();opt.zero_grad(set_to_none=True);z=net(**{k:v[ix]for k,v in x.items()});loss=losses(z,y[ix],valid[ix]).mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
  if step%100==0:log.append({'update':step,'loss':float(loss.detach())});print(json.dumps(log[-1]),flush=True)
  if step in[300,600,900,1200]:
   net.eval()
   with torch.inference_mode():z=net(**x);lv=losses(z,y,valid);score=float(torch.stack([lv[ii].mean()for ii in cri]).mean()+.2*torch.stack([lv[ii].mean()for ii in ori]).mean());pro=torch.cat([z[:,:6].sigmoid(),z[:,6:].softmax(-1)],-1).cpu().numpy()
   path=E/'models'/variant/f'query9_seed0_update{step:04d}.pt';torch.save({'model':net.state_dict(),'update':step,'seed':0,'variant':variant},path);r={'update':step,'C_selection_loss':score,'path':str(path),'sha256':sha(path),'C_new':metrics(pro,y0,v0,cnew),'C_legacy_Holding':metrics(pro,y0,v0,cold),'AB_new':metrics(pro,y0,v0,np.flatnonzero(tr&new))};checkpoints.append(r);print(json.dumps({'variant':variant,'checkpoint':step,'C_loss':score,'C_new_all9':r['C_new']['all9_correct_fully_valid'],'C_new_eligible9':r['C_new']['fully_valid9'],'C_legacy_H':r['C_legacy_Holding']['Holding_all3_correct']}),flush=True)
 best=min(checkpoints,key=lambda r:(r['C_selection_loss'],r['update']));net.load_state_dict(torch.load(best['path'],weights_only=False)['model']);net.eval()
 with torch.inference_mode():
  p=net.probabilities(**x).cpu().numpy();small={k:v[:5]for k,v in x.items()};nodes=net.nodes(**small);queries=torch.tensor([parse_query(q,{'alias':1})for q in ['Holding(alias)','In(alias,drawer)','Closed(drawer)','Open(drawer)','Intermediate(drawer)']],device='cuda');ans=net.query(nodes,queries).cpu().numpy();assert np.allclose(ans,p[:5][:,[1,4,6,7,8]],atol=1e-6);perm=[2,0,1];small['objects']=small['objects'][:,:,perm];small['local']=small['local'][:,:,perm];small['bbox']=small['bbox'][:,perm];pp=net.probabilities(**small).cpu().numpy();assert np.allclose(pp,p[:5][:,perm+[j+3 for j in perm]+[6,7,8]],atol=1e-6)
 assert all(torch.equal(v.detach().cpu(),frozen[n])for n,v in net.named_parameters()if not v.requires_grad);np.save(E/'results'/variant/'probabilities.npy',p);np.save(E/'logs'/variant/'train_indices.npy',schedule);save(E/'results'/variant/'training.json',{'variant':variant,'seed':0,'updates':1200,'trainable_parameters':sum(p.numel()for p in net.parameters()if p.requires_grad),'checkpoints':checkpoints,'selected':best,'binary_weights':weights,'drawer_class_counts_AB':dc,'frozen_encoder_unchanged':True,'alias_query_permutation_checks':True,'initial_function_same':True,'initial_state_sha256':initial_hash,'AB_only_training_indices_checked':True,'no_D':True,'loss_log':log});save(E/'configs'/variant/'reader_frozen.json',best);print('FIXED_SELECTION',variant,best['path'],flush=True)
if __name__=='__main__':main()
