from common import *
import shutil,sys,platform,torch
def main():
 assert not(E/'configs/delivery_manifest.json').exists()
 refs=[(B/'configs/delivery_manifest.json','9d8c3926e5d3a50ccb52287e47648f3fcab6caaf1fc9927bf4dc761fcaee0590'),(P/'experiments/holding_matched_input_diagnostic_20261018/configs/delivery_manifest.json','cbe4da01f7fcd965c29bf58a5b346646ec01e1f19e2ac5301cb95bdc5d199f25')]+[(Path(x['path']),x['sha256'])for x in read(B/'configs/input_references.json')['old_manifests']];old=[]
 for mp,h in refs:
  assert sha(mp)==h;manifest=read(mp)
  for p,expected in manifest['sha256'].items():assert sha(p)==expected,p
  assert len(manifest['sha256'])==manifest['files'];assert all(not(Path(p).stat().st_mode&0o222)for p in manifest['sha256']);old.append({'manifest':str(mp),'sha256':h,'files_verified':manifest['files'],'read_only':True})
 for file in ['configs/pretraining_frozen.json','configs/models_frozen.json']:
  for p,h in read(E/file)['sha256'].items():assert sha(p)==h,p
 for p,h in read(E/'configs/input_hashes.json').items():assert sha(p)==h,p
 assert sha(E/'configs/protocol.json')==sha(R/'FROZEN_PROTOCOL.json');assert sha(E/'configs/train_normalization.npz')==read(E/'configs/models_frozen.json')['normalization_sha256']
 f=read(E/'results/features.json');assert f['frames']==264 and f['encoder_unchanged']and f['new_images']==0
 for x in f['features']:assert sha(E/f'data/{x["split"]}_features.npz')==x['sha256']
 attempts=read(E/'results/training_attempts.json');assert len(attempts)==4 and all(x['status']=='completed'for x in attempts);assert len(list((E/'models').glob('final_*.pt')))==4
 pred=read(E/'results/prediction.json');assert len(pred['passes'])==8
 for x in pred['passes']:assert sha(E/f'results/predictions_{x["arm"]}_seed{x["seed"]}_{x["split"]}.npz')==x['sha256']
 verify=read(E/'results/independent_verification.json');assert(verify['metric_rows_recomputed'],verify['expanded_root_bootstrap_checks'],verify['same_state_feature_rows_verified'])==(324,120,1390)
 ev=read(E/'results/evaluation.json');assert ev['threshold']==.5 and ev['new_fits']==4 and ev['static_readout_tuning_closed']and not ev['clear_primary_gain_rule_passed'];assert all(r['positive_queries']+r['negative_queries']==r['N']for r in ev['metrics'])
 w=Path(__file__).parent.parent/'library_static_context_20261018';library=read(w/'confirmed.json');assert sha(library['upload_path'])==sha(R/'figures/01_static_context_diagnostic.png')
 for name in library['xattrs_applied']:assert os.getxattr(library['upload_path'],name)
 save(R/'LIBRARY.json',library);shutil.copy2(__file__,E/'source/seal.py')
 for src in Path(__file__).parent.glob('*.py'):assert sha(src)==sha(E/'source'/src.name)
 save(E/'configs/runtime.json',{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torch_CUDA_build':torch.version.cuda,'numpy':np.__version__,'GPU_order':'frozen encoder264existingframes then four600stepfits; exclusive samegpu.lock','offline_cache_reused':str(DINO)})
 save(E/'results/final_verification.json',{'utc':now(),'passed':True,'old_experiments':old,'old_files_verified_total':sum(x['files_verified']for x in old),'frozen_protocol_inputs_initializations_sources_unchanged':True,'frozen_models_predictions_features_unchanged':True,'new_frames':0,'encoded_existing_static_frames':264,'fits':4,'updates_per_fit':600,'prediction_passes':8,'D_samples_features_predictions_scores_read':False,'static_tuning_closed':True,'transition_collection_or_fitting_started':False,'independent_checks':verify,'figure_visually_reviewed':True,'library':library,'complete_guide_finished':False})
 files=sorted(p for base in(E,R)for p in base.rglob('*')if p.is_file());assert all(not p.is_symlink()for p in files)
 manifest={'utc':now(),'scope':'Final static readout diagnostic: same389-33-1 shape, context zero/present, two seeds, original36roots unchanged. No transitions.','files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files},'static_tuning_closed':True,'clear_primary_gain':False,'new_fits':4,'new_collection':0,'guide_complete':False}
 mp=E/'configs/delivery_manifest.json';save(mp,manifest)
 for p in files+[mp]:p.chmod(0o444)
 for p,h in manifest['sha256'].items():assert sha(p)==h and not(Path(p).stat().st_mode&0o222)
 receipt={'utc':now(),'experiment':str(E),'report':str(R/'REPORT.md'),'figure':str(R/'figures/01_static_context_diagnostic.png'),'next_evidence':str(R/'NEXT_TRANSITION_EVIDENCE.md'),'manifest':str(mp),'manifest_sha256':sha(mp),'files':manifest['files'],'bytes':manifest['bytes'],'read_only_verified':True,'library_file_id':library['library_file_id'],'file_id':library['file_id']};save(Path(__file__).parent/'delivery_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
