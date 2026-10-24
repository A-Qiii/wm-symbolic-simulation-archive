from common import *
from models import *
import sys,time
def main():
 assert not(E/'results/prediction.json').exists();freeze=read(E/'configs/models_frozen.json');assert freeze['fits']==6
 for p,h in freeze['sha256'].items():assert sha(p)==h
 assert sha(E/'configs/train_normalization.npz')==freeze['normalization_sha256']
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(os.fsdecode(args[0])).resolve()
   if p.is_relative_to(P/'experiments')and not p.is_relative_to(E):raise RuntimeError('Old experiments forbidden')
   if p.is_relative_to(E/'data/private')or p.is_relative_to(E/'data/supervision')or p==E/'data/records.json':raise RuntimeError('No annotations at prediction boundary')
 sys.addaudithook(guard);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);norm=np.load(E/'configs/train_normalization.npz');done=[];start=time.perf_counter()
 for seed in [0,1]:
  for arm in ['object','pair','geometry']:
   model=make(arm).eval();state=torch.load(E/f'models/final_{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);assert state['updates']==600;model.load_state_dict(state['state'])
   for split in ['train','dev']:
    with np.load(E/f'data/features/{split}.npz')as z:
     x=(z['x_'+arm]-norm[arm+'_mean'])/norm[arm+'_std']
     with torch.inference_mode():prob=model(torch.from_numpy(x)).squeeze(1).sigmoid().numpy()
     dest=E/f'results/predictions_{arm}_seed{seed}_{split}.npz';save_raw_npz(dest,record_indices=z['record_indices'],probability=prob);done.append({'arm':arm,'seed':seed,'split':split,'rows':len(prob),'sha256':sha(dest)})
 for p,h in freeze['sha256'].items():assert sha(p)==h
 save(E/'results/prediction.json',{'utc':now(),'seconds':time.perf_counter()-start,'passes':done,'private_or_label_reads':0,'model_hashes_unchanged':True,'threshold':.5});print('PREDICTION_COMPLETE',flush=True)
if __name__=='__main__':main()
