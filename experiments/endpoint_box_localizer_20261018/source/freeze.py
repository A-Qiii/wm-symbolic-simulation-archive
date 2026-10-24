from common import *
from net import *
import shutil

def main():
 assert not(E/'configs/training_frozen.json').exists();assert read(E/'results/annotation_quality_gate.json')['passed']
 rows=read(E/'data/public/images.json');anns=read(E/'data/annotations/final.json')['annotations'];t=np.load(E/'data/features/tokens.npz')['tokens'];train=np.array([r['split']=='train'for r in rows]);mu=t[train].mean((0,1));sd=np.maximum(t[train].std((0,1)),.05);x=((t-mu)/sd).astype(np.float32);np.savez_compressed(E/'configs/feature_normalization.npz',mean=mu,std=sd)
 y=np.zeros((40,256,2),np.float32);v=np.zeros((40,2),np.float32);b=np.zeros((40,256,2,4),np.float32);positive=[]
 for r,a in zip(rows,anns):
  i=r['image_id'];assert a['image_id']==i
  for j,c in enumerate(a['classes']):
   v[i,j]=c['status']!='uncertain'
   for bb in c['boxes']:
    z=np.array(bb,np.float32)/[r['width'],r['height'],r['width'],r['height']];center=(z[:2]+z[2:])/2;cell=np.minimum((center*16).astype(int),15);p=cell[1]*16+cell[0];assert y[i,p,j]==0,('positive collision',i,j,bb);y[i,p,j]=1;b[i,p,j]=np.r_[center*16-(cell+.5),z[2:]-z[:2]];positive.append({'image_id':i,'category':CATS[j],'patch':int(p)})
 for split in list(dict.fromkeys(r['split']for r in rows)):
  ix=np.array([r['image_id']for r in rows if r['split']==split]);out=E/('data/train'if split=='train'else'data/evaluation');out.mkdir(exist_ok=True);np.savez_compressed(out/f'{split}.npz',x=x[ix],y=y[ix],v=v[ix],b=b[ix],image_ids=ix,root_ids=np.array([rows[i]['root_id']for i in ix]))
 for seed in(0,1):
  rng=np.random.default_rng(181000+seed);schedule=np.concatenate([rng.permutation(16).reshape(2,8)for _ in range(300)]);np.save(E/f'configs/schedule_seed{seed}.npy',schedule);torch.manual_seed(181100+seed);q=make();torch.save({'state':q.state_dict(),'initial_hash':tensor_hash(q.state_dict())},E/f'models/initial_seed{seed}.pt')
 # Meaningful guards: unknown class creates no gradient; exact box encoding roundtrip; decode separate classes.
 raw=torch.randn(1,256,10,requires_grad=True);yy=torch.zeros(1,256,2);vv=torch.zeros(1,2);bb=torch.zeros(1,256,2,4);ls,_=loss(raw,yy,vv,bb);ls.backward();assert float(ls)==0 and torch.count_nonzero(raw.grad)==0
 for item in positive:
  i=item['image_id'];j=CATS.index(item['category']);p=item['patch'];decoded=toboxes(torch.tensor(b[i,p,j]),torch.tensor(p)).numpy();assert 0<=decoded.min()+1e-7 and decoded.max()<=1+1e-7
  target=next(z for z in anns[i]['classes'][j]['boxes']if iou(decoded,np.array(z)/[rows[i]['width'],rows[i]['height'],rows[i]['width'],rows[i]['height']])>.99999)
 assert sum(p.numel()for p in make().parameters())==25290
 for split in['AB_episode_heldout_development','C_seen_diagnostic','D_seen_diagnostic']:
  assert not(set(np.load(E/'data/train/train.npz')['root_ids'])&set(np.load(E/f'data/evaluation/{split}.npz')['root_ids']))
 save(E/'results/preflight.json',{'utc':now(),'passed':True,'positive_centers':len(positive),'no_center_collisions':True,'unknown_class_zero_gradient':True,'box_roundtrip_all':True,'parameters':25290,'train_only_normalization_images':16,'episode_disjoint':True})
 for p in Path(__file__).parent.glob('*.py'):
  if not(E/'source'/p.name).exists():shutil.copy2(p,E/'source'/p.name)
 files=[p for sub in['configs','source','data','models']for p in(E/sub).rglob('*')if p.is_file()];save(E/'configs/training_frozen.json',{'utc':now(),'before_first_optimizer_update':True,'sha256':{str(p):sha(p)for p in files},'fits':2,'updates':600,'threshold':.5});print('LOCALIZER_PREFLIGHT_PASSED_AND_TRAINING_FROZEN',len(positive),flush=True)
if __name__=='__main__':main()
