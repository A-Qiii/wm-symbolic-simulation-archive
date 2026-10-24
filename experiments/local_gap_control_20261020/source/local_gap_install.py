from pathlib import Path
import json,shutil,hashlib,datetime
P=Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/local_gap_control_20261020';OLD=P/'experiments/rgbd_observation_control_20261020';O=P/'experiments/object_recurrent_state_run_20261019';W=Path('/home/robot/Documents/Codex/2026-10-13/task-4')
for n in ['models.py','losses.py','losses_reference.py','train.py','evaluate.py']:
 t=(OLD/'source'/n).read_text()
 if n=='evaluate.py':t=t.replace("extract('historical_evaluation',seed)","pass # only original train/development in this bounded exploratory comparison").replace("['train','development','historical_evaluation','diagnosed_confirmation']","['train','development']")
 (S/'source'/n).write_text(t)
t=(OLD/'source/common.py').read_text().replace("S=P/'experiments/rgbd_observation_control_20261020'","S=P/'experiments/local_gap_control_20261020';OLD=P/'experiments/rgbd_observation_control_20261020'").replace("ARMS=['baseline','geometry']","ARMS=['coarse','local']");(S/'source/common.py').write_text(t)
for n in ['data_membership.json','episode_order.json','schedule_seed0.npz','schedule_seed1.npz']:shutil.copyfile(O/'configs'/n,S/'configs'/n)
shutil.copyfile(OLD/'configs/input_normalization.json',S/'configs/original_geometry_normalization.json')
for n in ['local_gap_view.py','local_gap_support_audit.py','local_gap_install.py']:shutil.copyfile(W/n,S/'source'/n)
decision={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'visual_review_completed':True,'decision':'PROCEED_WITH_FROZEN_VISIBLE_BOUNDARY_DESCRIPTOR','original_descriptor_unchanged':True,'evidence':'All156 role h5 states have visible pair and forward supports. Separate public-region audit finds external forward surface for upper object in108/108 train and48/48 development. Lower forward support is mostly container surface. Projection reviewed on fixed cases.','limitation':'Descriptor includes container and external visible boundaries, not guaranteed physical clearance or complete free space. It does not reconstruct hidden underside geometry. No invisible gaps filled with GT.','target_observability':'Upper-object external leading boundary visible; lower-object external queries often unavailable. Retain per-direction masks and report limitations.'};(S/'results/visibility_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
print('INSTALLED')
