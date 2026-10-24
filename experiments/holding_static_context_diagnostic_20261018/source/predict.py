from common import *
import torch
def main():
 assert not(E/'results/prediction.json').exists();guard('predict');torch.set_num_threads(4);torch.use_deterministic_algorithms(True);norm=np.load(E/'configs/train_normalization.npz');freeze=read(E/'configs/models_frozen.json');assert sha(E/'configs/train_normalization.npz')==freeze['normalization_sha256'];runs=[]
 for p,h in freeze['sha256'].items():assert sha(p)==h
 for seed in [0,1]:
  for arm in ARMS:
   model=make().eval();model.load_state_dict(torch.load(E/f'models/final_{arm}_seed{seed}.pt',map_location='cpu',weights_only=True)['state'])
   for split in ['train','dev']:
    z=np.load(E/f'data/{split}_features.npz');x=masked((z['x']-norm['mean'])/norm['std'],arm)
    with torch.inference_mode():pr=model(torch.from_numpy(x)).squeeze(1).sigmoid().numpy()
    f=E/f'results/predictions_{arm}_seed{seed}_{split}.npz';npz(f,record_indices=z['record_indices'],probability=pr);runs.append({'arm':arm,'seed':seed,'split':split,'rows':len(pr),'sha256':sha(f)})
 for p,h in freeze['sha256'].items():assert sha(p)==h
 save(E/'results/prediction.json',{'utc':now(),'passes':runs,'threshold':.5,'private_and_label_reads':0});print('PREDICTION_COMPLETE',len(runs))
if __name__=='__main__':main()
