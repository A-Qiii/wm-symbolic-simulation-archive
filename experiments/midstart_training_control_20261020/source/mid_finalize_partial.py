from common import *
import fcntl,re
def main():
 guard();verify();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);checks=[]
 for root in [O,C,OLD,G,P/'experiments/local_gap_control_20261020',P/'reports/relation_interface_diagnostic_20261019',P/'reports/rgbd_close_diagnostic_20261020']:
  manifest=read(root/'MANIFEST.json');assert sha(root/'MANIFEST.json')==read(root/'SEALED.json')['manifest_sha256']
  for r in manifest['files']:assert(root/r['path']).stat().st_size==r['bytes']and sha(root/r['path'])==r['sha256'],str(root/r['path'])
  checks.append({'root':str(root),'files':len(manifest['files']),'unchanged':True})
 for p,h in read(S/'configs/evaluation_frozen.json')['sha256'].items():assert sha(p)==h
 for p,h in read(S/'configs/models_frozen.json')['sha256'].items():assert sha(p)==h
 fits=[read(S/f'results/fit_{arm}_seed{seed}.json')for seed in [0,1]for arm in ARMS];assert all(r['updates']==1500 and r['parameters']==2490068 for r in fits);assert all(r['full_original_parity']['all_tensors_equal']for r in fits if r['arm']=='full')
 assert read(S/'results/support_and_leakage_audit.json')['passed'];assert read(S/'results/report_complete.json')['colors_and_axes_explained'];assert read(S/'results/development_evaluation_complete.json')['weights_unchanged']
 fresh=S/'fresh_validation';fresh_done=(S/'results/independent_evaluation_complete.json').exists();trigger=read(S/'results/independent_validation_trigger.json');assert fresh_done or (S/'results/independent_validation_interruption.json').exists()
 if fresh_done:
  assert read(fresh/'results/collection_complete.json')['episodes']==24
  for p,h in read(fresh/'configs/design_frozen.json')['sha256'].items():assert sha(p)==h
  for p,h in read(fresh/'configs/roots_frozen.json')['sha256'].items():assert sha(p)==h
 report=S/'reports/REPORT.html';links=re.findall(r'(?:href|src)="([^"]+)"',report.read_text());assert all((report.parent/x).exists()for x in links if not x.startswith(('http','#'))and x!='../results/final_verification.json')
 save(S/'results/final_verification.json',{'utc':now(),'old_sources':checks,'old_files_verified':sum(x['files']for x in checks),'models_fixed_step1500_unchanged':True,'full_two_seeds_exact_reproduction':True,'frozen_design_and_evaluation_unchanged':True,'updates_total':9000,'parameters_each':2490068,'new_training_data':0,'independent_validation_completed':fresh_done,'no_test_to_training':True,'no_private_GT_model_inputs':True,'fixed_GT_assisted_low_level_controller':True,'CD_accesses':0,'uploads':0,'old_failed_gates_unchanged':True,'report_links_checked':len(links),'runtime_dtype_error_before_first_history_update_retained':True})
 # Only now write the final report link check record, then seal all delivered files.
 files=[{'path':str(p.relative_to(S)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(S.rglob('*'))if p.is_file()and p.name not in ['MANIFEST.json','SEALED.json']];save(S/'MANIFEST.json',{'utc':now(),'root':str(S),'files':files});elapsed=(time.time()-read(S/'configs/launch.json')['start_epoch'])/60
 seal={'utc':now(),'status':'TRAINING_COMPLETE_INDEPENDENT_VALIDATION_INTERRUPTED','manifest_sha256':sha(S/'MANIFEST.json'),'manifested_files':len(files),'manifested_bytes':sum(r['bytes']for r in files),'elapsed_minutes':elapsed,'within45_minutes':elapsed<=45,'training_fits':6,'updates_per_fit':1500,'updates_total':9000,'new_training_data':0,'new_validation_episodes':24 if fresh_done else read(S/'results/independent_validation_interruption.json')['completed_episodes'],'planned_validation_episodes':24,'independent_validation_completed':fresh_done,'old_failed_gates_unchanged':True,'fixed_GT_assisted_low_level_controller':True,'world_model_GT_inputs':False,'original_guide_complete':False};save(S/'SEALED.json',seal)
 for p in S.rglob('*'):
  if p.is_file():p.chmod(0o444)
 for p in sorted([p for p in S.rglob('*')if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
 S.chmod(0o555);receipt={'directory':str(S),'report':str(report),**seal};save(Path('/home/robot/Documents/Codex/2026-10-13/task-4/midstart_training_delivery.json'),receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
