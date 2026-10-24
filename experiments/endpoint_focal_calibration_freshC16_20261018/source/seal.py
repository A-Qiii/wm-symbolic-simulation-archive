from common import *
import shutil,platform,sys,torch

def main():
 assert read(E/'results/final_verification.json')['passed'];assert not(E/'configs/delivery_manifest.json').exists();install_guard(read(E/'configs/protocol.json')['old_inputs_allowlist']);assert all(sha(p)==h for p,h in read(E/'configs/input_frozen.json')['sha256'].items());assert all(sha(p)==h for p,h in read(E/'configs/precollection_frozen.json')['sha256'].items());assert all(sha(p)==h for p,h in read(E/'configs/evaluation_code_frozen.json')['sha256'].items())
 for f in Path(__file__).parent.glob('*'):
  if f.suffix not in ['.py','.md']:continue
  d=E/'source'/f.name
  if d.exists():assert sha(d)==sha(f),f
  else:shutil.copy2(f,d)
 receipt=read('/home/robot/Documents/Codex/2026-10-13/task-4/library_freshC16_20261018/receipt.json');assert receipt['library_file_id']=='libfile_d0b0d6efe8b08191b007d1e0055244b6';assert sha(receipt['local_path'])==sha(R/'figures/new_roots_operating_points.png');save(E/'configs/library_delivery_receipt.json',receipt)
 data=read(E/'results/evaluation_summary.json')['rows'];summary=[]
 for r in data:
  if r['workpoint']=='calibrated'and r['camera']=='all'and r['category']!='all':summary.append({'seed':r['seed'],'category':r['category'],'precision':r['precision'],'recall':r['recall'],'meets_0_90_point_estimate':r['precision']>=.9,'precision_CI95':r['CI95']['precision']})
 assert all(not q['meets_0_90_point_estimate']for q in summary);save(E/'results/localization_conclusion.json',{'utc':now(),'new_root_precision_checks':summary,'calibration_oldC_successes':4,'new_root_point_estimate_passes':0,'inference_scope':'same known C scene; unseen initialized roots and interactions; 16 clustered roots, not 256 independent samples','conclusion':'old-C calibration is not a precision guarantee on new roots; finger recall remains low; localization is not established as reliable input','both_fixed_and_calibrated_preserved':True,'detector_loss_model_threshold_search_closed':True,'deployment':False,'relation_experiment':'design only; GT-box correct-localization diagnostic must be separated from predicted-box end-to-end; block localization not yet provided by this head','new_training_runs':0,'new_root_collection_completed':16,'additional_collection_after_results':0,'old_D_holdout_samples_features_predictions_scores_read':0,'technical_index_preflight_note':'Four-state technical result index including per-scene label summaries was inspected only for registry schema, excluded from all calibration and validation.'})
 save(E/'configs/model_runtime.json',{'utc':now(),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'cuda_build':torch.version.cuda,'platform':platform.platform(),'old_final_heads_unchanged':True,'backbone_unchanged':True,'GPU_sequential':'collector completed, then features completed, then prediction completed; same GPU flock; scoring/report CPU','training_updates':0})
 commands=f'''# 实际执行命令

```bash
source {P}/world_model_env.sh
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/freshC16_src/prepare.py
source {P}/env.sh
python {E}/source/collect.py > {E}/logs/collect.log 2>&1
source {P}/world_model_env.sh
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/freshC16_src/gate.py
python {E}/source/features.py > {E}/logs/features.log 2>&1
python /home/robot/Documents/Codex/2026-10-13/task-4/freshC16_src/predict.py > {E}/logs/predict.log 2>&1
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/freshC16_src/evaluate.py > {E}/logs/evaluate.log 2>&1
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/freshC16_src/verify.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/freshC16_src/report.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/freshC16_src/seal.py
```

每个实验脚本执行一次，16根一次完成、无重试或补根。evaluate.py首次运行前另用Python保存configs/evaluation_code_frozen.json，对evaluate/pr_engine/common源码进行SHA冻结。新预测每种子一次；一次正式评分后，verify.py对已保存候选作独立数值复核，未产生新模型预测或重新选择阈值。

原focal两模型不训练、不选种子；两个工作点共享同一批冻结预测。原0.5保留；校准完成并封存后才生成新根。Library保存采用当前可调用上传接口；当前技能助手下载返回不可用，未复用之前下载的助手，按当前工具说明回写返回元数据。封存后不要覆盖运行此目录。
'''
 (R/'RUN_COMMANDS.md').write_text(commands)
 p=R/'REPORT.md';p.write_text(p.read_text()+f'\n## Library图件\n\n`CALVIN_新16根_固定与校准工作点.png` 已保存至Library：`{receipt["library_file_id"]}`，文件ID `{receipt["file_id"]}`。上传图件与本地报告图一致，SHA256 `{sha(R/"figures/new_roots_operating_points.png")}`。\n')
 save(E/'results/status.json',{'utc':now(),'bounded_localization_stage_completed':True,'new_roots':16,'new_states':128,'new_primary_images':256,'new_fits':0,'original_focal_seeds':[0,1],'two_workpoints_scored_once':True,'calibration_locked_before_new_root_generation':True,'localization_reliability_established':False,'new_class_precision_point_estimates_above_or_equal_0_90':0,'detector_search_closed':True,'relation_design_complete':True,'relation_experiment_trained':False,'main_guide_complete':False,'old_artifact_writes':0,'library_file_id':receipt['library_file_id'],'file_id':receipt['file_id']})
 files=sorted(p for root in [E,R]for p in root.rglob('*')if p.is_file());total=sum(p.stat().st_size for p in files);save(E/'configs/delivery_manifest.json',{'utc':now(),'files':len(files),'bytes':total,'sha256':{str(p):sha(p)for p in files}});print('FRESH_C16_SEALED',len(files),total,sha(E/'configs/delivery_manifest.json'),flush=True)
if __name__=='__main__':main()
