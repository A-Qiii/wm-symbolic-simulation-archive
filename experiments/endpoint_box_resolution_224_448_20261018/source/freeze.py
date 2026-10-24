from common import *
from net import *
import importlib.util

def main():
 assert not(E/'configs/training_frozen.json').exists()
 check_manifest(E/'configs/pre_execution_frozen.json');check_manifest(BASE/'configs/delivery_manifest.json')
 rows=read(E/'data/public/images.json');anns=read(E/'data/annotations/final.json')['annotations']
 old_t=np.load(BASE/'data/features/tokens.npz')['tokens']
 cur_t=np.load(E/'data/features/r224.npz')['tokens']
 diff=float(np.max(np.abs(old_t-cur_t)))
 save(E/'results/reuse_feature_gate.json',{'utc':now(),'all40_tokens_bitwise_equal':bool(np.array_equal(old_t,cur_t)),'max_abs_difference':diff})
 assert np.array_equal(old_t,cur_t),'224 features not bitwise equal; stop before optimizer and explain necessary reproduction'
 spec=importlib.util.spec_from_file_location('reference_net',BASE/'source/net.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 checks={'all40_tokens_bitwise_equal':True,'all_split_arrays_identical':True,'old224_loss_gradient_decode_identical':True,'box_roundtrip':{},'unknown_gradients_zero':{},'background_mass_checks':[]}
 train=np.array([r['split']=='train'for r in rows]);assert train.sum()==16
 for res in(224,448):
  grid=res//14;n=grid*grid;t=np.load(E/f'data/features/r{res}.npz')['tokens'];mu=t[train].mean((0,1));sd=np.maximum(t[train].std((0,1)),.05);x=((t-mu)/sd).astype(np.float32);np.savez_compressed(E/f'configs/feature_normalization_r{res}.npz',mean=mu,std=sd)
  y=np.zeros((40,n,2),np.float32);v=np.zeros((40,2),np.float32);b=np.zeros((40,n,2,4),np.float32);roundtrips=0
  for r,a in zip(rows,anns):
   i=r['image_id'];assert a['image_id']==i
   for j,c in enumerate(a['classes']):
    v[i,j]=c['status']!='uncertain'
    for bb in c['boxes']:
     z=np.array(bb,np.float32)/[r['width'],r['height'],r['width'],r['height']];center=(z[:2]+z[2:])/2;cell=np.minimum((center*grid).astype(int),grid-1);p=cell[1]*grid+cell[0];assert y[i,p,j]==0,('positive collision',res,i,j);y[i,p,j]=1;b[i,p,j]=np.r_[center*grid-(cell+.5),z[2:]-z[:2]]
     assert iou(toboxes(torch.tensor(b[i,p,j]),torch.tensor(p),grid).numpy(),z)>.99999;roundtrips+=1
  assert roundtrips==91;checks['box_roundtrip'][res]=roundtrips
  for split in SPLITS:
   ix=np.array([r['image_id']for r in rows if r['split']==split]);out=E/('data/train'if split=='train'else'data/evaluation');out.mkdir(exist_ok=True)
   values={'x':x[ix],'y':y[ix],'v':v[ix],'b':b[ix],'image_ids':ix,'root_ids':np.array([rows[i]['root_id']for i in ix])}
   np.savez_compressed(out/f'r{res}_{split}.npz',**values)
   if res==224:
    ref=np.load(BASE/('data/train'if split=='train'else'data/evaluation')/f'{split}.npz')
    for k,val in values.items():assert np.array_equal(val,ref[k]),(split,k)
  raw=torch.randn(1,n,10,requires_grad=True);ls,_=loss(raw,torch.zeros(1,n,2),torch.zeros(1,2),torch.zeros(1,n,2,4),grid);ls.backward();assert float(ls.detach())==0 and torch.count_nonzero(raw.grad)==0;checks['unknown_gradients_zero'][res]=True
  # Constant negative logits give the same total negative focal weight at both grids.
  for positives in(0,1,2):
   checks['background_mass_checks'].append({'resolution':res,'positives':positives,'negative_weight':(256-positives)/(n-positives),'weighted_negative_count':(n-positives)*(256-positives)/(n-positives)})
  if res==224:
   # Include actual training labels and unequal visible-category masks in equivalence checks.
   torch.manual_seed(7721);q=make();qold=old.make();qold.load_state_dict(q.state_dict());xx=torch.tensor(x[:8]);yy=torch.tensor(y[:8]);vv=torch.tensor(v[:8]);bb=torch.tensor(b[:8]);r=q(xx);ro=qold(xx);r.retain_grad();ro.retain_grad();l,_=loss(r,yy,vv,bb,16);lo,_=old.loss(ro,yy,vv,bb);l.backward();lo.backward()
   assert torch.equal(l,lo)and torch.equal(r.grad,ro.grad)
   for p,po in zip(q.parameters(),qold.parameters()):assert torch.equal(p.grad,po.grad)
   for a in r.detach().numpy():assert decode(a,16)==old.decode(a)
 for seed in(0,1):
  schedule=np.load(E/f'configs/schedule_seed{seed}.npy');assert schedule.shape==(600,8)
  assert np.array_equal(schedule,np.load(BASE/f'configs/schedule_seed{seed}.npy'))
  assert np.array_equal(np.bincount(schedule.ravel(),minlength=16),np.full(16,300))
  for res in(224,448):assert sha(E/f'models/r{res}/initial_seed{seed}.pt')==sha(BASE/f'models/initial_seed{seed}.pt')
 assert len(set(r['root_id']for r in rows if r['split']=='train'))==4
 for split in SPLITS[1:]:assert not({r['root_id']for r in rows if r['split']=='train'}&{r['root_id']for r in rows if r['split']==split})
 assert sum(p.numel()for p in make().parameters())==25290
 # Check actual focal values and aggregate logit gradients, not just the count formula.
 mass=[]
 for grid in(16,32):
  n=grid*grid;raw=torch.full((2,n,10),.7,requires_grad=True);yy=torch.zeros(2,n,2);yy[0,0,0]=1;yy[1,:2,1]=1;vv=torch.tensor([[1.,0.],[1.,1.]]);bb=torch.zeros(2,n,2,4)
  ls,components=loss(raw,yy,vv,bb,grid);ls.backward();assert torch.count_nonzero(raw.grad[0,:,1])==0 and torch.count_nonzero(raw.grad[0,:,6:])==0
  grad=raw.grad[...,:2];mass.append([float(components[0]),float(grad[yy>0].sum()),float(grad[(yy==0)&vv[:,None,:].bool()].sum())])
 assert np.allclose(mass[0],mass[1],rtol=1e-6,atol=1e-7),mass
 checks['constant_logit_focal_and_gradient_mass_equal']=mass
 checks.update(utc=now(),passed=True,reuse224=True,new_fits_planned=2,parameters=25290,unknowns_are_not_negatives=True,same_samples_and_sampling=True,train_only_normalization=True)
 save(E/'results/preflight_and_reuse.json',checks)
 files=[p for sub in['configs','source','data','models']for p in(E/sub).rglob('*')if p.is_file()]
 save(E/'configs/training_frozen.json',{'utc':now(),'before_first_optimizer_update':True,'sha256':{str(p):sha(p)for p in files},'new_fits':2,'reused224_fits':2,'updates':600,'threshold':.5})
 print('STRICT224_REUSE_GATE_PASSED_NEW448_BUDGET_FROZEN',flush=True)
if __name__=='__main__':main()
