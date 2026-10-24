from common import *
import shutil,sys,platform,torch

def main():
 assert read(E/'results/final_verification.json')['passed'];assert not(E/'configs/delivery_manifest.json').exists();opened=install_guard(read(E/'configs/protocol.json')['allowlisted_old_inputs']);frozen=read(E/'configs/input_frozen.json')['sha256'];assert all(sha(p)==h for p,h in frozen.items());assert all(sha(p)==h for p,h in read(E/'configs/pre_run_frozen.json')['sha256'].items())
 for dest,src in read(E/'configs/copied_input_map.json').items():assert sha(dest)==sha(src)
 for f in Path(__file__).parent.glob('*'):
  if f.suffix not in ['.py','.md']:continue
  dest=E/'source'/f.name
  if dest.exists():assert sha(dest)==sha(f),f
  else:shutil.copy2(f,dest)
 rows=read(E/'results/three_way_comparison.json');assert len(rows)==108;selection=read(E/'results/selection.json');assert len(selection['checks'])==24
 def row(v,s,c):return next(q for q in rows if q['variant']==v and q['seed']==s and q['split']=='dev'and q['camera']=='all'and q['category']==c)
 for variant in ['BCE','mixed']:
  passed=True
  for seed in [0,1]:
   for cat in ['all']+CATS:passed &= row(variant,seed,cat)['AP50']+1e-12>=row('focal',seed,cat)['AP50']
   for k in ['precision05','recall05']:passed &= row(variant,seed,'all')[k]+1e-12>=row('focal',seed,'all')[k]
   passed &= row(variant,seed,CATS[0])['AP50']>row('focal',seed,CATS[0])['AP50']+1e-12
  assert bool(passed)==selection['eligible'][variant]
 assert selection['selected_configuration']=='focal'and not any(selection['eligible'].values())
 retained={'utc':now(),'configuration':'original focal gamma2 for all three slots','work_point':.5,'source':'training-before-frozen paired-seed selection rule','both_seeds':{str(s):{'path':str(BASE/f'models/final_seed{s}.pt'),'sha256':frozen[str(BASE/f'models/final_seed{s}.pt')]}for s in [0,1]},'normalization':{'path':str(BASE/'configs/normalization.npz'),'sha256':frozen[str(BASE/'configs/normalization.npz')]},'deployment':False,'meaning':'readonly research reference retained; no new candidate or production acceptance','loss_search_closed':True,'new_root_protocol_only':str(R/'FUTURE_HOLDOUT_PROTOCOL.md')};save(E/'configs/retained_configuration.json',retained)
 receipt=read('/home/robot/Documents/Codex/2026-10-13/task-4/library_mixed_20261018/receipt.json');assert receipt['library_file_id']=='libfile_22a5a1792cc88191acc00de1067ab013';assert sha(receipt['local_path'])==sha(R/'figures/three_way_PR_dev.png');save(E/'configs/library_delivery_receipt.json',receipt)
 save(E/'configs/runtime.json',{'utc':now(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'cuda_build':torch.version.cuda,'GPU_peak_allocated_bytes':read(E/'results/checkpoints_frozen.json')['peak_GPU_bytes'],'GPU_schedule':'single shared flock, sequential train then prediction; probes/eval/report CPU','new_fits':2,'updates_each':1200,'diagnostic_optimizer_updates':0})
 commands=['CUDA_VISIBLE_DEVICES=\'\' python /home/robot/Documents/Codex/2026-10-13/task-4/mixed_gamma_src/prepare.py']
 for name,log,cpu in [('train','train',False),('predict','predict',False),('evaluate','evaluate',True),('shared_diagnostics','shared_diagnostics',True),('merge_select','merge_select',True)]:commands.append(('CUDA_VISIBLE_DEVICES=\'\' 'if cpu else'')+f'python {E}/source/{name}.py > {E}/logs/{log}.log 2>&1')
 for name in ['score_diagnostics','report','verify','seal']:commands.append(f"CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/mixed_gamma_src/{name}.py")
 (R/'RUN_COMMANDS.md').write_text('# 实际执行命令\n\n```bash\nsource /home/robot/research/projects/wm_symbolic/world_model_env.sh\n'+'\n'.join(commands)+'\n```\n\n每个实验脚本执行一次；新增训练严格2次各1200步。模型冻结早于预测，预测冻结早于评分。CPU评估与无更新梯度诊断可以并行，GPU采集/训练/推理未并行。原配置和全BCE结果复用；没有新根采集、D评分、阈值选择或第四种损失。沙箱内第一次nvidia-smi无法访问驱动，正常授权的本机检查成功显示RTX5080 Laptop、96/16303MiB；没有修改驱动或环境。\n\n封存后不要覆盖运行此目录；任何后续工作必须独立目录，并继续保持本轮损失搜索关闭。\n')
 p=R/'REPORT.md';p.write_text(p.read_text()+f'\n## Library图件\n\n三方案C开发对照图已保存至Library：`{receipt["library_file_id"]}`，文件ID `{receipt["file_id"]}`。上传副本与本地报告图一致，SHA256 `{sha(R/"figures/three_way_PR_dev.png")}`。\n')
 save(E/'results/delivery_audit.json',{'utc':now(),'copied_old_files_bitwise_verified':len(read(E/'configs/copied_input_map.json')),'three_way_rows':len(rows),'selection_checks':24,'independent_selection_recomputed':True,'six_micro_AP_values_reconstructed_from_class_counts':True,'no_new_test':True,'D_sample_feature_prediction_score_reads':0,'old_input_files_unchanged':len(frozen),'old_files_opened':sorted(opened)})
 save(E/'results/status.json',{'utc':now(),'bounded_control_completed':True,'new_training_runs':2,'updates_each':1200,'loss_search_closed':True,'selected_configuration':'focal','work_point':.5,'both_seeds_retained':True,'deployment':False,'D_samples_features_predictions_metrics_reads':0,'new_root_collection_or_test':False,'old_artifact_writes':0,'library_file_id':receipt['library_file_id'],'file_id':receipt['file_id'],'next_scope':'object-state/relation readout preparation; new-root and threshold specification prepared only','main_guide_complete':False})
 files=sorted(p for root in [E,R]for p in root.rglob('*')if p.is_file());save(E/'configs/delivery_manifest.json',{'utc':now(),'files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}});print('MIXED_CONTROL_SEALED',len(files),sha(E/'configs/delivery_manifest.json'),flush=True)
if __name__=='__main__':main()
