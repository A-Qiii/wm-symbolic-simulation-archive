from common import *
import shutil

def main():
 assert not E.exists()and not R.exists(),'Refuse to overwrite any previous run'
 check_manifest(BASE/'configs/delivery_manifest.json')
 for folder in ['configs','source','data/public/observations','data/annotations','models/r224','models/r448','results','logs']:(E/folder).mkdir(parents=True,exist_ok=True)
 (R/'figures').mkdir(parents=True)
 rows=read(BASE/'data/public/images.json')
 refs={str(BASE/'configs/delivery_manifest.json'):sha(BASE/'configs/delivery_manifest.json'),str(DINO/'model.safetensors'):sha(DINO/'model.safetensors')}
 for r in rows:
  src=Path(r['path']);dst=E/'data/public/observations'/src.name
  if not dst.exists():shutil.copy2(src,dst)
  refs[str(src)]=sha(src);r['previous_experiment_path']=str(src);r['path']=str(dst)
 save(E/'data/public/images.json',rows)
 for name in ['final.json','round1.json','reviewed.json','repeat.json','review_changes.json']:
  src=BASE/'data/annotations'/name;shutil.copy2(src,E/'data/annotations'/name);refs[str(src)]=sha(src)
 for seed in(0,1):
  shutil.copy2(BASE/f'configs/schedule_seed{seed}.npy',E/f'configs/schedule_seed{seed}.npy')
  for res in(224,448):shutil.copy2(BASE/f'models/initial_seed{seed}.pt',E/f'models/r{res}/initial_seed{seed}.pt')
  shutil.copy2(BASE/f'models/localizer_seed{seed}_update0600.pt',E/f'models/r224/localizer_seed{seed}_update0600.pt')
  for src in [BASE/f'configs/schedule_seed{seed}.npy',BASE/f'models/initial_seed{seed}.pt',BASE/f'models/localizer_seed{seed}_update0600.pt']:refs[str(src)]=sha(src)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 protocol={
  'utc':now(),'frozen_before_new_encoder_inference_or_optimizer_updates':True,
  'question':'Does increased frozen-DINO input sampling density improve the same small finger localization problem within the fixed budget?',
  'resolutions':[224,448],'grids':[16,32],'patches_per_image':[256,1024],
  'same_native_RGB':'static200x200,wrist84x84; interpolation adds no new sensor pixels',
  'images':40,'states':20,'known_boxes':91,'annotation_changes':0,
  'split':'Exact previous source-episode split:16train,16ABdevelopment,4Cseen,4Dseen; all historically viewed, no independent test',
  'seeds':[0,1],'updates_each':600,'batch_images':8,'visits_each_train_image':300,
  'head':'Shared384->64SiLU->10;25290parameters;same initial tensors; no scene/camera/XY/ROI inputs',
  'encoder':'Same cached frozen DINOv2-small, original RGB bilinear resized to chosen resolution, same ImageNet normalization and local pretrained weights',
  'features':'Train-only per-channel mean/std separately at each resolution, same reduction over images and patch positions; stdfloor.05',
  'loss':'Original focal alpha.25 gamma2 normalized by positive centers per image, plus5SmoothL1 and2GIoU, same positive boxes; no loss tuning',
  'negative_patch_weight':'At gridG, multiply each negative focal term in each image/class by (256 - positive_count)/(G*G - positive_count). Positives unchanged. Exact old arithmetic at G16. This equalizes nominal total negative mass despite4xlocations; actual hard-negative values may differ.',
  'regression_and_decoding_geometry':'Same center residual in cell units and normalized width/height; grid parameterized. Same3x3peak filter covers half the original-pixel width at448; this inherent grid effect is disclosed, not an isolated proof about DINO semantic resolution.',
  'optimizer':'AdamW lr.001 wd.0001 clip1; deterministic torch; final600; no early stopping, augmentation, threshold or epoch search',
  'baseline_reuse_gate':'Reuse224 final models only if all40 recached tokens exactly match, all normalized split tensors/targets exactly match, schedules/init identical, old-vs-new loss AND raw/parameter gradients plus decoded predictions match. If any gate fails, stop and explain before same-process224 reproduction; no silent fallback.',
  'primary_metrics':'AB development per category TP,FP,FN at score.5 and greedy1:1IoU.5; all-GT best IoU including missing prediction0. Report both seeds separately, no selecting favorable seed.',
  'secondary_metrics':'Train fit and AB static/wrist stratification; same uncertainty handling and visibility/reason breakdown; C/D already-seen diagnostics only; no not-visible examples, absence rejection unvalidated',
  'decode':'score>=.5,3x3maxpool peaks,classNMSIoU.3,cap1front/2fingers; no fallback',
  'cost':'Same batch8 paired frozen-encoder benchmark:1warmup+3timed forwards at each resolution; full40image cache cost; actual training loop plus1warmup/3timed fixed-weight forward/backward steps both grids without optimizer; report peakGPUmemory and cache bytes',
  'read_only_diagnostic':'Use saved model outputs at existing GT centers to count low score, off-box regression and nonpeak misses, no altered decoder or extra fit',
  'predeclared_example_ids':[0,1,16,17,26,27],
  'stopping':'Exactly two448 fits with600updates;224historical fits reused only under gate; no third resolution/network/extra labels/SupCon/relations/model adoption even if development improves',
  'limits':'Same assistant labels and reviewed sources; scale vs coverage vs head cannot be fully disentangled; no new independent accuracy claim'
 }
 save(E/'configs/protocol.json',protocol);save(E/'configs/source_references.json',{'sha256':refs})
 files=[p for p in E.rglob('*')if p.is_file()]
 save(E/'configs/pre_execution_frozen.json',{'utc':now(),'before_new_inference_and_training':True,'sha256':{str(p):sha(p)for p in files}})
 print('RESOLUTION_PROTOCOL_FROZEN',E,flush=True)
if __name__=='__main__':main()
