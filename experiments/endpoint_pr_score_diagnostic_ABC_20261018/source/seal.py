from common import *
import shutil,sys,platform,torch
def main():
 assert read(E/'results/final_verification.json')['passed'];assert not(E/'configs/delivery_manifest.json').exists()
 for f in Path(__file__).parent.glob('*.py'):
  d=E/'source'/f.name
  if d.exists():assert sha(f)==sha(d)
  else:shutil.copy2(f,d)
 receipt=read(Path('/home/robot/Documents/Codex/2026-10-13/task-4/library-pr-abc-hrm4_n1h/receipt.json'));assert receipt['library_file_id']=='libfile_6f3d5e37a95481919ecd9f530512961b';assert sha(receipt['local_path'])==sha(R/'figures/PR_overview.png');save(E/'configs/library_delivery_receipt.json',receipt)
 save(E/'configs/runtime.json',{'utc':now(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'device':'CPU only, CUDA_VISIBLE_DEVICES empty','new_training_or_inference':0,'old_models_read_or_written':0})
 (R/'RUN_COMMANDS.md').write_text('''# Executed local commands

```bash
source /home/robot/research/projects/wm_symbolic/world_model_env.sh
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/pr_score_src/prepare.py
CUDA_VISIBLE_DEVICES='' python /home/robot/research/projects/wm_symbolic/experiments/endpoint_pr_score_diagnostic_ABC_20261018/source/diagnose.py > /home/robot/research/projects/wm_symbolic/experiments/endpoint_pr_score_diagnostic_ABC_20261018/logs/diagnostic.log 2>&1
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/pr_score_src/report.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/pr_score_src/verify.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/pr_score_src/seal.py
```

The first two scripts reject an existing initialized/completed experiment. They were run once, with no computation retry. The full candidate calculation took place only in `diagnose.py`; report generation and verification consume its new ABC-only outputs. No old D file or mixed-split aggregate was opened. The allowed 773 input files and read ledger are in configs/protocol.json and results/access_and_checks.json.

Do not rerun over the sealed directory. No model training, forward pass, checkpoint loading, rendering, package installation or network download was needed. The only external write was publishing the PR overview as a Library image and applying its returned version metadata to a separate upload copy.
''')
 p=R/'REPORT.md';p.write_text(p.read_text()+f'\n## Library 图件\n\n`CALVIN_AB_C完整候选PR诊断.png` 已保存为原生 Library 图件。\n\n- library_file_id：`{receipt["library_file_id"]}`\n- file_id：`{receipt["file_id"]}`\n- 本机图件 SHA-256：`{sha(R/"figures/PR_overview.png")}`\n\n上传副本与报告图件内容相同；扩展属性已按创建工具返回结果回写。\n')
 save(E/'results/status.json',{'utc':now(),'bounded_diagnostic_complete':True,'old_formal_05_results_unchanged':True,'D_file_reads':0,'training_updates':0,'model_forward_passes':0,'old_model_weight_reads':0,'candidate_calculation_runs':1,'curves':36,'fixed_threshold_points':288,'input_files_unchanged':773,'recommendation_not_executed':'single classification-objective contrast, keeping all other factors fixed; calibration alone cannot fix classwise C AP degradation','library_file_id':receipt['library_file_id'],'file_id':receipt['file_id']})
 files=sorted(p for root in [E,R]for p in root.rglob('*')if p.is_file());save(E/'configs/delivery_manifest.json',{'utc':now(),'files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}});print('PR_DIAGNOSTIC_SEALED',len(files),sha(E/'configs/delivery_manifest.json'),flush=True)
if __name__=='__main__':main()
