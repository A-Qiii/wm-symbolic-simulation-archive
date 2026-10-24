from common import *
import shutil,sys,platform,importlib.metadata,torch
def main():
 assert read(E/'results/final_verification.json')['passed'];assert not(E/'configs/delivery_manifest.json').exists()
 for p in Path(__file__).parent.glob('*.py'):
  dst=E/'source'/p.name
  if dst.exists():assert sha(p)==sha(dst),p
  else:shutil.copy2(p,dst)
 receipt=read(Path('/home/robot/Documents/Codex/2026-10-13/task-4/library-sync256-6quk3_yz/receipt.json'));assert receipt['library_file_id']=='libfile_5555cffa51888191859ac2061676f8ad';assert sha(receipt['local_path'])==sha(R/'figures/final_prediction_grid.png');save(E/'configs/library_delivery_receipt.json',receipt)
 save(E/'configs/training_runtime.json',{'utc':now(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'numpy':np.__version__,'transformers':importlib.metadata.version('transformers'),'DINO_cache':str(DINO),'DINO_weights_sha256':sha(DINO/'model.safetensors'),'trainable_parameters':25615,'training_GPU_peak_allocated_bytes':read(E/'results/checkpoints_frozen.json')['peak_GPU_bytes'],'features_GPU_peak_allocated_bytes':read(E/'results/features.json')['peak_GPU_bytes'],'note':'Runtime metadata collected at final sealing in the same unchanged independent world_model environment.'})
 commands='''# Executed commands and replay boundaries

These commands were run locally in order. Initialization and evaluation scripts intentionally reject an already populated experiment or scoring result. Do not rerun them over this sealed experiment. A future independent experiment requires a new name and a newly frozen protocol; D is already seen.

```bash
# W=/home/robot/Documents/Codex/2026-10-13/task-4
# P=/home/robot/research/projects/wm_symbolic
# E=$P/experiments/endpoint_sync_localizer_256_20261018

source "$P/world_model_env.sh"
CUDA_VISIBLE_DEVICES='' python "$W/sync_localizer_src/prepare.py"
source "$P/env.sh"
python "$E/source/collect.py" > "$E/logs/collection_attempt1.log" 2>&1

source "$P/world_model_env.sh"
CUDA_VISIBLE_DEVICES='' python "$E/source/gate.py"
# Source hashes frozen before extraction; code copied into E/source.
python "$E/source/features.py" > "$E/logs/features.log" 2>&1
CUDA_VISIBLE_DEVICES='' python "$W/sync_localizer_src/prepare_train.py"
python "$E/source/train.py" > "$E/logs/train.log" 2>&1

# Evaluation code boundary frozen after both final checkpoints, before metrics.
python "$E/source/predict.py" > "$E/logs/predict.log" 2>&1
CUDA_VISIBLE_DEVICES='' python "$E/source/score.py" > "$E/logs/score.log" 2>&1
CUDA_VISIBLE_DEVICES='' python "$E/source/diagnose.py"
CUDA_VISIBLE_DEVICES='' python "$W/sync_localizer_src/analyze.py"
CUDA_VISIBLE_DEVICES='' python "$W/sync_localizer_src/report.py"
CUDA_VISIBLE_DEVICES='' python "$W/sync_localizer_src/verify.py"
# report.py was rerun only to clarify line styles/log axis, using identical saved predictions.
```

Current original source copies, raw arrays, losses, logits, decoded outputs and matching records are sealed in E. Per-split `data/features/*.npz` are reusable RGB-only DINO tokens; `data/train/train.npz` also contains supervision and must never be used as a label-free inference container.

No simulator collection retry, new download, dependency installation, extra fit, threshold sweep, architecture search or closed-loop deployment occurred. The single oracle-assisted counterfactual did not alter model weights or official scores.
'''
 (R/'RUN_COMMANDS.md').write_text(commands)
 report=R/'REPORT.md';report.write_text(report.read_text()+f'\n## Library 交付\n\n原生 Library 文件：`CALVIN_256状态定位基线_留出预测对照.png`。\n\n- library_file_id：`{receipt["library_file_id"]}`\n- file_id：`{receipt["file_id"]}`\n- PNG SHA-256：`{sha(R/"figures/final_prediction_grid.png")}`\n\nLibrary 上传内容与本机预测叠图逐字节一致，返回的版本扩展属性已回写上传副本；未修改封存图像内容。\n')
 save(E/'results/status.json',{'utc':now(),'bounded_task_completed':True,'data_gate_passed':True,'fixed_baseline_reliable':False,'next_blocker':'low train and held-out recall; requires a separately authorized preregistered follow-up, not automatic continuation of this fixed experiment','fits_completed':2,'updates_each':1200,'final_holdout_used':True,'dataset_expansion':0,'extra_fits':0,'parameter_or_threshold_changes_after_freeze':0,'old_file_entries_unchanged':2417,'library_file_id':receipt['library_file_id'],'file_id':receipt['file_id'],'full_original_world_model_TS_LTL_guide_completed':False})
 files=sorted([p for root in [E,R]for p in root.rglob('*')if p.is_file()]);save(E/'configs/delivery_manifest.json',{'utc':now(),'files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}});print('SEALED',len(files),'files',sha(E/'configs/delivery_manifest.json'),flush=True)
if __name__=='__main__':main()
