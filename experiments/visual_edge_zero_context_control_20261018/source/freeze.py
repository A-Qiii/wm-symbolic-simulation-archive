from common import *
import shutil,torch

def main():
 assert not(E/'configs/training_frozen.json').exists()
 assert not(E/'results/training_attempts.json').exists()
 assert read(E/'results/control_preflight.json')['passed']
 for p,h in read(E/'configs/source_references.json')['sha256'].items():assert sha(p)==h,p
 for head in HEADS:
  d=np.load(E/f'data/train/{head}.npz');b=np.load(BASE/f'data/train/{head}.npz')
  assert d['x'].dtype==b['edge'].dtype==np.float32
  assert np.array_equal(d['x'][:,:,:74],b['edge'][:,:,:74]);assert np.count_nonzero(d['x'][:,:,74:])==0
  for key in ['y','v','indices','state_ids','root_ids']:assert np.array_equal(d[key],b[key])
  for path in(E/'data/evaluation').glob('*.npz'):
   assert not(set(d['root_ids'])&set(np.load(path)['root_ids'])),path
  for seed in(0,1):
   ini=torch.load(E/f'models/initial_{head}_seed{seed}.pt',weights_only=True,map_location='cpu');q=make();q.load_state_dict(ini['state']);assert tensor_hash(q.state_dict())==ini['initial_hash']
   assert sum(p.numel()for p in q.parameters())==6689
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 files=[p for sub in ['configs','source','data/train','data/evaluation','models']for p in(E/sub).rglob('*')if p.is_file()and'__pycache__'not in p.parts]
 save(E/'configs/training_frozen.json',{'utc':now(),'before_first_optimizer_update':True,'sha256':{str(p):sha(p)for p in files},'new_fits':4,'updates_each':600,'threshold':.5,'reference_retraining':False,'same_initialization_norm_schedule_loss_optimizer':True})
 print('MATCHED_CONTROL_FROZEN_BEFORE_FITS',len(files),flush=True)
if __name__=='__main__':main()
