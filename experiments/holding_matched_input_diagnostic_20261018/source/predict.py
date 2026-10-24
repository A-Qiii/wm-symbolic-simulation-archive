from common import *
import torch
def main():
 assert not(E/'results/prediction.json').exists();guard('predict');torch.set_num_threads(4);torch.use_deterministic_algorithms(True);norm=np.load(E/'configs/train_normalization.npz');freeze=read(E/'configs/models_frozen.json')
 for p,h in freeze['sha256'].items():assert sha(p)==h
 runs=[]
 for seed in [0,1]:
  for arm in ARMS:
   model=make().eval();model.load_state_dict(torch.load(E/f'models/final_{arm}_seed{seed}.pt',map_location='cpu',weights_only=True)['state'])
   for split in ['train','dev']:
    z=np.load(B/f'data/features/{split}.npz');x=masked((z['x_pair']-norm['pair_mean'])/norm['pair_std'],arm)
    with torch.inference_mode():pr=model(torch.from_numpy(x)).squeeze(1).sigmoid().numpy()
    f=E/f'results/predictions_{arm}_seed{seed}_{split}.npz';npz(f,record_indices=z['record_indices'],probability=pr);equiv=None
    if arm=='full':equiv=np.array_equal(pr,np.load(B/f'results/predictions_pair_seed{seed}_{split}.npz')['probability'])
    runs.append({'arm':arm,'seed':seed,'split':split,'rows':len(pr),'sha256':sha(f),'full_original_pair_predictions_bitwise_equal':equiv})
 for p,h in freeze['sha256'].items():assert sha(p)==h
 save(E/'results/prediction.json',{'utc':now(),'passes':runs,'threshold':.5,'labels_private_reads':0});print('PREDICTION_COMPLETE',[(x['seed'],x['split'],x['full_original_pair_predictions_bitwise_equal'])for x in runs if x['arm']=='full'])
if __name__=='__main__':main()
