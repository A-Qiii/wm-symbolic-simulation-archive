from common import *
import shutil,sys,platform,torch
def main():
 assert not(E/'configs/delivery_manifest.json').exists()
 oldchecks=[]
 refs=[(B/'configs/delivery_manifest.json','9d8c3926e5d3a50ccb52287e47648f3fcab6caaf1fc9927bf4dc761fcaee0590')]
 refs.extend((Path(x['path']),x['sha256'])for x in read(B/'configs/input_references.json')['old_manifests'])
 for mp,h in refs:
  assert sha(mp)==h;manifest=read(mp)
  for p,expected in manifest['sha256'].items():assert sha(p)==expected,p
  assert len(manifest['sha256'])==manifest['files'];assert all(not(Path(p).stat().st_mode&0o222)for p in manifest['sha256']);oldchecks.append({'manifest':str(mp),'manifest_sha256':h,'files_verified':manifest['files'],'read_only':True})
 for p,h in read(E/'configs/pretraining_frozen.json')['sha256'].items():assert sha(p)==h,p
 for p,h in read(E/'configs/input_hashes.json').items():assert sha(p)==h,p
 for p,h in read(E/'configs/models_frozen.json')['sha256'].items():assert sha(p)==h,p
 assert sha(E/'configs/protocol.json')==sha(R/'FROZEN_PROTOCOL.json')
 runs=read(E/'results/training_attempts.json');assert len(runs)==6 and all(x['status']=='completed'for x in runs)
 assert len(list((E/'models').glob('final_*.pt')))==6
 pred=read(E/'results/prediction.json');assert len(pred['passes'])==12
 for x in pred['passes']:assert sha(E/f'results/predictions_{x["arm"]}_seed{x["seed"]}_{x["split"]}.npz')==x['sha256']
 assert all(x['full_original_pair_predictions_bitwise_equal']for x in pred['passes']if x['arm']=='full')
 verify=read(E/'results/independent_verification.json');assert verify['metric_rows_independently_recomputed']==384 and verify['expanded_root_bootstrap_checks']==120
 ev=read(E/'results/evaluation.json');assert ev['threshold']==.5 and ev['new_fits']==6
 assert all(x['positive_queries']+x['negative_queries']==x['N']for x in ev['metrics'])
 alias=read(E/'results/alias_inspection.json');assert alias['physical_label_checks']==1490
 library=[];w=Path(__file__).parent.parent/'library_matched_holding_20261018'
 for name in ['01_matched_comparison.png','relation36_C_24_RGB_neighbors.png','relation36_C_34_RGB_neighbors.png']:
  c=read(w/(name+'.confirmed.json'));assert sha(c['upload_path'])==sha(R/'figures'/name)
  for attr in c['xattrs_applied']:assert os.getxattr(c['upload_path'],attr)
  library.append(c)
 save(R/'LIBRARY.json',library)
 shutil.copy2(__file__,E/'source/seal.py')
 for src in Path(__file__).parent.glob('*.py'):assert sha(src)==sha(E/'source'/src.name),src
 save(E/'configs/runtime.json',{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torch_CUDA_build':torch.version.cuda,'numpy':np.__version__,'single_local_GPU_fits':True,'gpu_lock':str(P/'runs/gpu.lock'),'cache_reused':'all original DINOv2 features; no encoder executed'})
 save(E/'results/final_verification.json',{'utc':now(),'passed':True,'original_frozen_experiments':oldchecks,'pretraining_protocol_source_initialization_unchanged':True,'input_data_labels_features_unchanged':True,'matched_final_models_unchanged':True,'prediction_hashes_verified':12,'new_fits':6,'updates_each':600,'new_collection':0,'new_encoder_executions':0,'D_samples_features_predictions_scores_read':False,'whole_world_model_guide_completed':False,'independent_verification':verify,'alias_contact_label_checks':1490,'figures_visually_reviewed':5,'library_files_confirmed':library})
 files=sorted(p for base in(E,R)for p in base.rglob('*')if p.is_file());assert all(not p.is_symlink()for p in files)
 manifest={'utc':now(),'scope':'Exploratory same-architecture three masked-input ablations, six fits on unchanged36root dataset; two alias RGB/contact/temporal diagnostics; posthoc target/support strata','files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files},'new_collection':0,'new_fits':6,'old_experiments_unchanged':True,'guide_complete':False}
 mp=E/'configs/delivery_manifest.json';save(mp,manifest)
 for p in files+[mp]:p.chmod(0o444)
 for p,h in manifest['sha256'].items():assert sha(p)==h and not(Path(p).stat().st_mode&0o222)
 receipt={'utc':now(),'experiment':str(E),'report':str(R/'REPORT.md'),'manifest':str(mp),'manifest_sha256':sha(mp),'files':manifest['files'],'bytes':manifest['bytes'],'read_only_verified':True,'library':library}
 save(Path(__file__).parent/'delivery_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
