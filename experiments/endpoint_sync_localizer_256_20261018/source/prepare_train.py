from common import *
from net import *
import shutil
def main():
 assert read(E/'results/data_gate.json')['passed'];assert not(E/'configs/training_frozen.json').exists();rows=read(E/'data/public/images.json');ann=read(E/'data/annotation_private/labels.json');z=np.load(E/'data/features/train.npz');x=z['x'];ids=z['image_ids'];n=len(ids);mu=x.mean((0,1));sd=np.maximum(x.std((0,1)),.05);x=(x-mu)/sd;save_raw_npz(E/'configs/normalization.npz',mean=mu,std=sd);y=np.zeros((n,1024,3),np.float32);b=np.zeros((n,1024,3,4),np.float32);valid=np.zeros((n,3),np.float32);positive=0;double=0
 for index,i in enumerate(ids):
  r=rows[i];a=ann[i];assert r['split']=='train';valid[index]=[a['class_valid'][0],a['class_valid'][1],a['class_valid'][1]]
  for c in [0,1]:
   count={}
   boxes=sorted(a['boxes'][c],key=lambda box:((box[0]+box[2])/2,(box[1]+box[3])/2))
   for box in boxes:
    bb=np.array(box,np.float32)/[r['width'],r['height'],r['width'],r['height']];center=(bb[:2]+bb[2:])/2;cell=np.minimum((center*GRID).astype(int),GRID-1);patch=int(cell[1]*GRID+cell[0]);offset=count.get(patch,0);slot=0 if c==0 else 1+offset;assert slot<3 and y[index,patch,slot]==0;count[patch]=offset+1;double+=int(c==1 and offset==1);y[index,patch,slot]=1;b[index,patch,slot]=np.r_[center*GRID-(cell+.5),bb[2:]-bb[:2]];decoded=toboxes(torch.tensor(b[index,patch,slot]),torch.tensor(patch)).numpy();assert iou(decoded,bb)>.9999;positive+=1
 train=E/'data/train';train.mkdir();np.savez_compressed(train/'train.npz',x=x.astype(np.float32),y=y,b=b,v=valid,image_ids=ids,root_ids=np.array([rows[i]['root_id']for i in ids]));raw=torch.randn(1,1024,15,requires_grad=True);ls,_=loss(raw,torch.zeros(1,1024,3),torch.zeros(1,3),torch.zeros(1,1024,3,4));ls.backward();assert float(ls)==0 and torch.count_nonzero(raw.grad)==0
 # Decoder regression: two adjacent finger boxes survive, and >2 fingers are not capped.
 raw=torch.full((1024,15),-20.);z=raw.reshape(1024,3,5)
 for p in [100,101,300]:z[p,1]=torch.tensor([8.,0.,0.,-4.,-4.])
 assert len(decode(raw.numpy()))==3
 for seed in [0,1]:
  rng=np.random.default_rng(seed);schedule=np.concatenate([rng.permutation(n).reshape(-1,16)for _ in range(75)]);assert schedule.shape==(1200,16);np.save(E/f'configs/schedule_seed{seed}.npy',schedule);torch.manual_seed(seed);m=make();torch.save({'state':m.state_dict(),'initial_hash':tensor_hash(m.state_dict())},E/f'models/initial_seed{seed}.pt')
 save(E/'results/training_preflight.json',{'utc':now(),'passed':True,'train_images':n,'positive_centers':positive,'same_patch_two_finger_centers_preserved':double,'unknown_zero_gradient':True,'box_roundtrip':True,'decoder_adjacent_and_no_cardinality_cap':True,'parameters':sum(p.numel()for p in make().parameters()),'normalization_train_only':True})
 for p in Path(__file__).parent.glob('*.py'):
  dest=E/'source'/p.name
  if not dest.exists():shutil.copy2(p,dest)
  else:assert sha(p)==sha(dest)
 files=[f for sub in ['source','models']for f in (E/sub).rglob('*')if f.is_file()]+[train/'train.npz',E/'configs/protocol.json',E/'configs/normalization.npz']+list((E/'configs').glob('schedule_*.npy'));save(E/'configs/training_frozen.json',{'utc':now(),'before_first_optimizer_update':True,'sha256':{str(f):sha(f)for f in files}});print('TRAINING_FROZEN',flush=True)
if __name__=='__main__':main()
