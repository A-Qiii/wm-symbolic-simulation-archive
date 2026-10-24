from common import *
import shutil,torch
from wm_symbolic.world_model import WorldModel
from wm_symbolic.core_dataset import CoreEpisodes
from wm_symbolic.dataset import batch_indices
def main():
 assert not E.exists()and not R.exists()
 for d in ['source','configs','models','data','results','logs']:(E/d).mkdir(parents=True)
 (R/'figures').mkdir(parents=True)
 for f in Path(__file__).parent.glob('*'):
  if f.is_file():shutil.copy2(f,E/'source'/f.name)
 save(E/'configs/protocol.json',read(Path(__file__).parent/'protocol.json'))
 src=P/'runs/core1200_v1/training_main/normalization_train_only.json';norm=read(src);assert norm['fit_transitions']==720 and len(norm['fit_root_ids'])==60
 save(E/'configs/normalization.json',norm)
 tr=CoreEpisodes(P,'train',normalizer=norm);assert len(tr.episodes)==180 and tr.transitions==720
 for seed in [0,1]:
  torch.manual_seed(seed);m=WorldModel(skill_count=6,event_count=3);partition(m)
  torch.save({'state':m.state_dict(),'seed':seed,'hash':th(m.state_dict())},E/f'models/initial_seed{seed}.pt')
  rng=np.random.default_rng(seed);schedule=np.array([[batch_indices(tr,rng).tolist()for _ in range(2)]for u in range(6000)],np.int64)
  latent=np.arange(12000,dtype=np.int64).reshape(6000,2)+261019000+seed*20000
  np.savez_compressed(E/f'configs/schedule_seed{seed}.npz',indices=schedule,latent_seeds=latent)
 refs=[P/'src/wm_symbolic'/n for n in ['world_model.py','core_dataset.py','dataset.py','vision_cache.py','checkpointing.py','core_evaluation.py']]
 refs += [P/'scripts/core1200_train.py',P/'data/core1200_v1/data_manifest.json',P/'data/core1200_v1/roots_manifest.json',P/'data/core1200_v1/features/index.json',P/'runs/core1200_v1/protocol.json',src,P/'runs/core1200_v1/training_main/best.pt',P/'runs/core1200_v1/training_main/checkpoint_06000.pt']
 for f in refs[:6]:shutil.copy2(f,E/'source'/('reference_'+f.name))
 data=read(P/'data/core1200_v1/data_manifest.json')['records'];idx=read(P/'data/core1200_v1/features/index.json')['items']
 refs +=[Path(r['labels_path'])for r in data if r['split']in ['train','dev']]
 refs +=[Path(r['path'])for r in idx if r['split']in ['train','dev']]
 save(E/'configs/references.json',{'sha256':{str(p):sha(p)for p in refs},'data_noD':True,'references':len(refs)})
 save(E/'results/preparation.json',{'utc':now(),'initialization_only':True,'model_fits':0,'physics':0,'C_samples_read':0,'D_samples_read':0,'train_roots':60,'train_transitions':720})
 print('PREPARED',E,flush=True)
if __name__=='__main__':main()
