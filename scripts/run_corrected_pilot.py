"""Finish and retain the original pilot, then freeze the corrected label interface."""
from pathlib import Path
import time,json,subprocess,shutil,datetime,fcntl,os,hashlib
P=Path('/home/robot/research/projects/wm_symbolic');S=Path('/home/robot/Documents/Codex/2026-10-13/task-4');status=P/'runs/corrected_pilot_pipeline.json'
if (P/'runs/PAUSED_BY_USER.json').exists():
 raise SystemExit('Paused by user; explicit authorization required.')
lock=(P/'runs/corrected_pilot_pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(**x):status.write_text(json.dumps({'pid':os.getpid(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**x},indent=2))
save(status='waiting',step='retain_original_6000_updates')
while not (P/'runs/wm_pilot_seed0/result.json').exists():
 s=json.loads((P/'runs/pilot_pipeline.json').read_text())
 if s['status']=='blocked':save(status='blocked',reason='Original pilot failed',details=s);raise SystemExit(2)
 time.sleep(5)
original=json.loads((P/'runs/wm_pilot_seed0/result.json').read_text());assert original['updates']==6000
assert json.loads((P/'runs/label_protocol_regression.json').read_text())['passed']
save(status='preparing',step='freeze_label_interface_correction')
for n in ['world_model','dataset','evaluation','symbol_decoder','task_specs']:
 shutil.copy2(S/(n+'.py'),P/'src/wm_symbolic'/(n+'.py'))
shutil.copy2(S/'train_world_model.py',P/'scripts/train_world_model.py')
shutil.copy2(P/'data/data_manifest.json',P/'data/data_manifest_v1_frozen.json');shutil.copy2(P/'data/data_manifest_process_v2.json',P/'data/data_manifest.json')
record={'correction_count':1,'component':'Offline labeling and declared process-event interface','reasons':['Physical tabletop coordinate calibration used instead of placement-sampling rectangles','Device-state invariants include transient control-step events'],
 'evidence':['runs/label_geometry_audit.json','runs/device_event_audit.json','runs/label_protocol_regression.json'],
 'original_retained':'runs/wm_pilot_seed0 and data/data_manifest_v1_frozen.json','model_core_unchanged':['Frozen DINOv2','Object Transformer','RSSM 512 + 32x32','Shared symbolic head','Frozen HULC'],
 'outcome_head_events':7,'data_quotas_and_update_budgets_unchanged':True,'test_trajectories_used':False,'further_case_specific_corrections_allowed':False}
(P/'runs/correction_record.json').write_text(json.dumps(record,indent=2))
for mode in ['probe','pilot']:
 name=f'train_{mode}_protocol_v2';save(status='running',step=name)
 with (P/'logs'/f'{name}.log').open('w') as f:
  r=subprocess.run(['bash','-c','source "$1"; shift; exec python "$@"','pipeline',str(P/'world_model_env.sh'),str(P/'scripts/train_world_model.py'),'--mode',mode,'--event-count','7','--tag','protocol_v2','--seed','0'],stdout=f,stderr=subprocess.STDOUT,cwd=P)
 if r.returncode:save(status='blocked',step=name,returncode=r.returncode);raise SystemExit(r.returncode)
 if mode=='probe':
  metrics=json.loads((P/'runs/wm_probe_protocol_v2_seed0/result.json').read_text())['final_metrics']
  if metrics['current_accuracy']<.95 or metrics['prior_accuracy']<.95:save(status='blocked',step='corrected_128_probe_not_fit',metrics=metrics);raise SystemExit(2)
save(status='complete',step='corrected_pilot_ready_for_development_error_analysis')
