"""Offline C-only visible reference calibration; no executor invocation."""
from pathlib import Path
import json,numpy as np,hashlib
from wm_symbolic.public_geometry import drawer_front
P=Path('/home/robot/research/projects/wm_symbolic');m=json.loads((P/'data/data_manifest.json').read_text());rows=[r for r in m['records'] if r['split']=='dev'];seen=set();samples=[]
for row in rows:
 label=json.loads((Path(row['obs_after']).parent/'labels_process_v2.json').read_text())
 for key,side in [('obs_before','before'),('obs_after','after')]:
  path=row[key]
  if path in seen:continue
  seen.add(path)
  with np.load(path) as z:frame=dict(z)
  part=drawer_front(frame)
  samples.append({'path':path,'root_id':row['root_id'],'visual_front_y_m':None if part is None else part['front_y_m'],'offline_drawer_m':label[side]['continuous']['drawer_m'],'offline_class':label[side]['categorical'][3],'offline_valid':label[side]['categorical_valid'][3]})
closed=[x for x in samples if x['visual_front_y_m'] is not None and x['offline_valid'] and x['offline_class']==0]
assert len(closed)>=20 and len({x['root_id'] for x in closed})>=5
reference=float(np.median([x['visual_front_y_m'] for x in closed]));errors=[]
for x in samples:
 if x['visual_front_y_m'] is not None:errors.append(abs(reference-x['visual_front_y_m']-x['offline_drawer_m']))
config={'version':'visual_stop_candidate_v2','status':'frozen_candidate_requires_paired_validation_and_user_review','source':'C development visible RGB-D and anonymous instance masks; offline labels identify reference examples only','calibration_roots':sorted({x['root_id'] for x in samples}),
 'drawer_closed_front_y_m':reference,'drawer_open_min_m':.17,'drawer_closed_max_m':.02,'lift_min_rise_m':.05,'lift_max_TCP_distance_m':.10,'gripper_width_min_m':.012,'gripper_width_max_m':.07,'consecutive_checks':3,'check_every_controls':3,'minimum_controls':12,
 'allowed_live_inputs':['RGB','depth','visible anonymous masks','camera calibration','public robot observation'],'forbidden_live_inputs':['scene_obs','contacts','simulator poses','task checker success'],'policy_weights_unchanged':True,'no_recovery_actions_added':True}
out=P/'configs/executor_visual_stop_v2.json';assert not out.exists(),'Do not overwrite frozen candidate calibration';out.write_text(json.dumps(config,indent=2));report={'closed_reference_examples':len(closed),'closed_reference_roots':len({x['root_id'] for x in closed}),'visible_drawer_samples':len(errors),'absolute_visual_joint_estimate_error_m':{'median':float(np.median(errors)),'p95':float(np.quantile(errors,.95)),'max':max(errors)},'calibration_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'samples':samples,'scope':'Calibration evidence only; no completed-controller success claimed'};(P/'runs/visual_stop_geometry_calibration.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
