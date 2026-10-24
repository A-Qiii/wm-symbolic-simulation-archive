from common import *
import shutil,torch
def main():
 assert not E.exists()and not R.exists()
 old=read(B/'configs/delivery_manifest.json');assert sha(B/'configs/delivery_manifest.json')=='9d8c3926e5d3a50ccb52287e47648f3fcab6caaf1fc9927bf4dc761fcaee0590'
 for p,h in old['sha256'].items():assert sha(p)==h
 for n in ['configs','models','results','source','logs']: (E/n).mkdir(parents=True,exist_ok=True)
 (R/'figures').mkdir(parents=True)
 inputs={str(B/'configs/delivery_manifest.json'):sha(B/'configs/delivery_manifest.json')}
 for rel in ['configs/train_normalization.npz','configs/schedule_seed0.npz','configs/schedule_seed1.npz','models/initial_pair_seed0.pt','models/initial_pair_seed1.pt']:
  src=B/rel;dest=E/rel;shutil.copy2(src,dest);inputs[str(src)]=sha(src)
 for rel in ['data/features/train.npz','data/features/dev.npz','data/supervision/train_targets.json','data/supervision/dev_targets.json','data/records.json','results/bootstrap_draws.npz','source/train.py','source/predict.py','source/collect.py','source/labels_physical.py']:
  inputs[str(B/rel)]=sha(B/rel)
 initial=[]
 for seed in [0,1]:
  state=torch.load(E/f'models/initial_pair_seed{seed}.pt',map_location='cpu',weights_only=True)['state'];m=make();m.load_state_dict(state);assert sum(p.numel()for p in m.parameters())==8680
  initial.append({'seed':seed,'tensor_hash':tensor_hash(state),'shapes':{k:list(v.shape)for k,v in state.items()}})
 protocol={'utc':now(),'scope':'Exploratory matched-input diagnosis on already-seen C; no new collection, labels or data changes; not confirmatory',
 'frozen_dataset':str(B),'roots':{'train_AB':24,'dev_C':12},'query_rows':{'train':920,'dev':470},'label_and_eligibility':'All original labels, missingness, records and train/dev assignments unchanged',
 'arms':{'full':'all261 normalized features','geometry_masked_RGB':'normalized dimensions0:256 replaced by exact zero; keep dimensions256:261','RGB_masked_geometry':'keep normalized dimensions0:256; dimensions256:261 replaced by exact zero'},
 'architecture':'Linear261x33, SiLU, Linear33x1; all8680 parameter scalars and tensor shapes identical',
 'initialization':initial,'normalization':'exact original train-only pair_mean/std for all arms; mask after normalization; no per-arm refit',
 'training':{'seeds':[0,1],'torch_seeds':[261019210,261019211],'runs':6,'updates':600,'batch':32,'optimizer':'AdamW','lr':.001,'weight_decay':.001,'clip_grad_norm':1.,'BCE_pos_weight':4.75,'dtype':'float32','deterministic':True,'TF32':False,'schedule':'exact original saved600x32 indices shared by all arms within each seed','selection':'final600 only','early_stopping':False,'augmentation':False,'dev_tuning':False},
 'full_baseline':'Run full anew under the same matched code; compare tensor hashes and CPU predictions to original pair as reproducibility diagnostic, without retraining if different',
 'evaluation':{'threshold':.5,'metrics':['precision','recall','balanced_accuracy','F1','NLL'],'groups':['all','color','camera','color_camera','root','dev_geometry_alias8'],'paired_comparisons':['full-minus-geometry_masked_RGB','full-minus-RGB_masked_geometry','RGB_masked_geometry-minus-geometry_masked_RGB'],'bootstrap':'Reuse original2000 paired whole12Croot multiplicities(seed261019420); recompute pooled confusion/NLL; percentile95CI; undefined null; no multiple-comparison correction','seed_aggregation':'report both; do not select best'},
 'alias_diagnostic':{'roots':['relation36_C_24','relation36_C_34'],'colors':['red','blue'],'anchors':'inspect all8 existing anchors, focus s03 support vs s04 lifted and s05/s06 neighbors','evidence':['actual full static/wrist RGB and exact model crops','pixel and mask differences, cached feature differences','existing camera matrices and depth only as private diagnosis','per-control contacts including body/link/force/normal/gap and label recomputation at3 thresholds','quantized visible-mask boxes vs full pixels/context'], 'no_claim':'Identical boxes do not establish identical RGB or information-theoretic visual ambiguity; changed pixels alone do not prove reliable contact-state observability'},
 'forbidden':['new collection','new encoder features','label change','threshold tuning','best seed selection','WM','localization','SupCon','In experiment','D sample feature prediction score access']}
 save(E/'configs/protocol.json',protocol);save(R/'FROZEN_PROTOCOL.json',protocol);save(E/'configs/input_hashes.json',inputs)
 for src in Path(__file__).parent.glob('*.py'):shutil.copy2(src,E/'source'/src.name)
 freeze=[p for folder in ['configs','models','source']for p in(E/folder).rglob('*')if p.is_file()]
 save(E/'configs/pretraining_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in freeze},'original_manifest_files_verified':old['files'],'before_any_new_fit':True})
 print('PROTOCOL_FROZEN',sha(E/'configs/protocol.json'),'source_files',len(list((E/'source').glob('*.py'))))
if __name__=='__main__':main()
