from common import *
import shutil,sys,platform,torch
def main():
 assert read(E/'results/final_verification.json')['passed'];assert not(E/'configs/delivery_manifest.json').exists();opened=install_guard(read(E/'configs/protocol.json')['allowlisted_old_inputs'])
 for f in Path(__file__).parent.glob('*.py'):
  d=E/'source'/f.name
  if d.exists():assert sha(f)==sha(d),f
  else:shutil.copy2(f,d)
 for d in (E/'data/baseline').glob('*'):
  if d.name.startswith('train_seed'):src=BASE/'results'/d.name
  elif d.suffix=='.npz':src=PRBASE/'data'/d.name
  else:src=PRBASE/'results'/d.name
  assert sha(d)==sha(src),(d,src)
 assert sha(E/'data/ABC_labels_metadata.json')==sha(PRBASE/'data/ABC_labels_metadata.json');receipt=read(Path('/home/robot/Documents/Codex/2026-10-13/task-4/library-gamma0-_y3o25x9/receipt.json'));assert receipt['library_file_id']=='libfile_d317fafd2164819183cbc5d47676820b';assert sha(receipt['local_path'])==sha(R/'figures/PR_dev_comparison.png');save(E/'configs/library_delivery_receipt.json',receipt)
 save(E/'configs/runtime.json',{'utc':now(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'cuda_build':torch.version.cuda,'training_GPU_peak_allocated_bytes':read(E/'results/checkpoints_frozen.json')['peak_GPU_bytes'],'GPU_scheduling':'single shared lock, sequential train then prediction; all scoring/probes/report CPU','new_training_runs':2,'extra_optimizer_updates_in_probes':0})
 (R/'RUN_COMMANDS.md').write_text('''# Executed local commands

```bash
source /home/robot/research/projects/wm_symbolic/world_model_env.sh
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/gamma0_src/prepare.py
python /home/robot/research/projects/wm_symbolic/experiments/endpoint_gamma0_control_ABC_20261018/source/train.py > /home/robot/research/projects/wm_symbolic/experiments/endpoint_gamma0_control_ABC_20261018/logs/train.log 2>&1
python /home/robot/research/projects/wm_symbolic/experiments/endpoint_gamma0_control_ABC_20261018/source/predict.py > /home/robot/research/projects/wm_symbolic/experiments/endpoint_gamma0_control_ABC_20261018/logs/predict.log 2>&1
CUDA_VISIBLE_DEVICES='' python /home/robot/research/projects/wm_symbolic/experiments/endpoint_gamma0_control_ABC_20261018/source/probe_gradients.py > /home/robot/research/projects/wm_symbolic/experiments/endpoint_gamma0_control_ABC_20261018/logs/gradient_probes.log 2>&1
CUDA_VISIBLE_DEVICES='' python /home/robot/research/projects/wm_symbolic/experiments/endpoint_gamma0_control_ABC_20261018/source/evaluate.py > /home/robot/research/projects/wm_symbolic/experiments/endpoint_gamma0_control_ABC_20261018/logs/evaluate.log 2>&1
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/gamma0_src/report.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/gamma0_src/verify.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/gamma0_src/seal.py
```

Training, prediction, gradient probes and evaluation each ran once. report.py was rerun only to label logarithmic gradient axes, using identical saved results. Two new optimizer runs completed 1200 updates each; the focal baseline was reused. No D sample, feature, prediction or score file was opened. Old seal manifests supplied checksum metadata only.

Do not rerun scripts over the sealed experiment. Initializers and completion guards prevent accidental duplicate runs. The next front-only modulation hypothesis is a separate coordination proposal, not an executed third run in this experiment. Existing user continuation authorization is quoted in NEXT_EXPERIMENT_PROPOSAL.md.
''')
 p=R/'REPORT.md';p.write_text(p.read_text()+f'\n## Library 图件\n\n`CALVIN_focal与同权重BCE_C对照.png` 已保存为原生 Library 图件。\n\n- library_file_id：`{receipt["library_file_id"]}`\n- file_id：`{receipt["file_id"]}`\n- PNG SHA-256：`{sha(R/"figures/PR_dev_comparison.png")}`\n\n上传副本与报告图件相同，返回版本扩展属性已回写上传副本。\n')
 save(E/'results/status.json',{'utc':now(),'bounded_control_completed':True,'new_training_runs':2,'updates_each':1200,'third_variant_runs':0,'overall_BCE_adopted':False,'C_micro_AP_delta_by_seed':[-.0028933091436837977,.05132719623175386],'C_front_AP_delta_by_seed':[.07096892578776515,.0687212035178042],'C_finger_AP_delta_by_seed':[-.026857816309572624,.04648586565703505],'D_samples_features_predictions_metrics_reads':0,'old_artifact_writes':0,'next_hypothesis':'front-only gamma0, fingers retain gamma2; separate 2x1200 budget, AB/C-only evaluation, not yet run','latest_user_continuation_authorization':'Sentinel_9dd2d7c1a5fc8191a759ea02da2c724a, 2026-10-18 14:03:27 UTC','next_proposal_path':str(R/'NEXT_EXPERIMENT_PROPOSAL.md'),'library_file_id':receipt['library_file_id'],'file_id':receipt['file_id'],'baseline_copies_bitwise_verified':True})
 files=sorted(p for root in [E,R]for p in root.rglob('*')if p.is_file());save(E/'configs/delivery_manifest.json',{'utc':now(),'files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}});print('GAMMA0_CONTROL_SEALED',len(files),sha(E/'configs/delivery_manifest.json'),flush=True)
if __name__=='__main__':main()
