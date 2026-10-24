from pathlib import Path
from html.parser import HTMLParser
from collections import Counter
import json,hashlib,datetime,shutil
import numpy as np
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic'); R=P/'runs/core1200_v1'; W=Path('/home/robot/Documents/Codex/2026-10-13/task-4'); B=P/'data/core1200_v1'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
colors=['red','blue','pink'];old=[];new=[]
for e in 'ABCD':
 names=list(OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml').objects.movable_objects)
 def ordered(f):
  with np.load(f) as z:lookup=dict(zip(names,z['scene_obs'][6:].reshape(3,6)))
  return np.stack([lookup['block_'+c] for c in colors])
 paths=list((P/'data/roots').glob(f'*/{e}_*/reset_evaluator_only.npz'))
 for v in ['v1','v2','v3']:paths+=list((P/f'runs/physical_oracle_{v}/roots').glob(f'{e}_*/reset_evaluator_only.npz'))
 old += [(e,ordered(f),str(f)) for f in paths]
 new += [(e,ordered(f),f.parent.name) for f in (B/'roots').glob(f'{e}_*/branch0_initial_private.npz')]
matches=[]
for i,(e,poses,name) in enumerate(new):
 for ee,other,source in old+new[:i]:
  if e!=ee:continue
  dist=np.linalg.norm(poses[:,:3]-other[:,:3],axis=1);yaw=np.abs((poses[:,5]-other[:,5]+np.pi/2)%np.pi-np.pi/2)
  if np.all(dist<=.012) or np.all(dist<=.02) and np.all(yaw<=.25):matches.append({'root':name,'source':source,'position_distances_m':dist.tolist()})
rootaudit={'actual_settled_roots_checked':len(new),'historical_candidates':len(old),'new_roots_checked_against_one_another':True,'rule':'same environment, all3 named positions within12mm OR all within20mm and all yaw modulo pi within0.25rad','matches':matches,'source':'Actual branch0_initial_private.npz after physical setup; read-only audit, no root selection or resampling after test.'}
write(R/'actual_root_exclusion_all100.json',rootaudit)
freeze=read(R/'evaluation/model_test_freeze.json');checks={name:sha(P/'src/wm_symbolic'/name)==value for name,value in freeze['evaluation_sources'].items()}
checks['checkpoint']=sha(Path(freeze['checkpoint']))==freeze['checkpoint_sha256']
checks['tasks']=sha(R/'evaluation/tasks.json')==freeze['planning_tasks_sha256']
checks['controller']=sha(P/'src/wm_symbolic/physical_oracle.py')=='0532b205828126e2b102d5c00dfa349d9534d870b7c077e797731b8fa08b973f'
checks['old_data_manifest']=sha(P/'data/data_manifest.json')=='3efd5c23f0da99da4048f70362ef4ade8846e046f32130a6740771f619a9968d'
checks['guide']=sha(P/'source/世界模型与TS-LTL符号规划_核心预实验方案.md')=='f4cb908e26db79176263d5002290ac4c94daf3025ce2523fc546eb896e2ad0d4'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  self.links.extend(v for k,v in attrs if k in ['href','src'])
htmlfile=P/'reports/core1200_final.html';parser=Links();parser.feed(htmlfile.read_text());missing=[v for v in parser.links if not (htmlfile.parent/v).exists()]
receipts=[]
for f in (W/'library_deliverables').glob('core1200_*receipt.json'):
 d=read(f);s=d['result']['structuredContent'];assert s['status']=='succeeded';receipts.append({k:s[k] for k in ['file_name','library_file_id','file_id','file_size_bytes']});shutil.copy2(f,R/f.name)
shutil.copy2(W/'library_deliverables/六技能核心实验_1200条结果.html',P/'reports/core1200_standalone.html')
validation={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_hashes_unchanged':checks,'actual_root_exclusion':rootaudit,'report_local_links':{'checked':len(parser.links),'missing':missing},'visual_review':{'three_figures_viewed':True,'new_video_final_frames_viewed':True,'findings':'Chinese text, axes, legends and panels legible; no clipping. Final values match JSON.','review_mode':'Agent viewed actual rendered PNGs; HTML local link validation, not a browser layout test.'},'library_deliverables':receipts,'regressions':{f.name:read(f)['passed'] for f in R.glob('*regression.json')}}
validation['passed']=all(checks.values()) and len(new)==100 and not matches and not missing and all(validation['regressions'].values())
write(R/'final_delivery_validation.json',validation);assert validation['passed'],validation
result=read(R/'RESULT.json');result.update(deliverables=receipts,delivery_validation='final_delivery_validation.json',remaining_blockers=['Planning efficacy not met: learned+LTL1/12; reference10/12.','Completion certification unavailable:8 calibration roots below unchanged10-root minimum.','Original full-guide scope not completed; further experiments need a new scope decision.']);write(R/'RESULT.json',result)
handoff=P/'runs/continuation_handoff.json';previous=R/'handoff_prior_to_final_delivery.json'
if not previous.exists():shutil.copy2(handoff,previous)
write(handoff,{'stage':'core1200_scope_complete_efficacy_not_met','updated_utc':validation['checked_utc'],'active_pipeline':None,'active_session':None,'authorized_scope':'2026-10-14T04:01:27Z user approved independent1200-transition first training and diagnostic evaluation with frozen v1. No broader expansion.','actual_core_transitions':1200,'core_roots':100,'environment_protocol':'AB train60roots/720, C dev20roots/240(12tune8calibration), D test20roots/240; v1 previously validatedABC, separateD controller audit provided.','main_training_complete':True,'main_training_updates':6000,'selected_checkpoint_update':1000,'selected_checkpoint':freeze['checkpoint'],'latest_report':str(htmlfile),'standalone_report':str(P/'reports/core1200_standalone.html'),'result':str(R/'RESULT.json'),'validation':str(R/'final_delivery_validation.json'),'D_metrics':{'current_fields':[1016,1500],'prior_fields':[822,1200],'prior_all5':[16,240],'prior_all7':[16,240],'changed':[179,185],'unchanged':[643,1015],'persistence_changed':[0,185],'wrong_action_changed':[0,185],'free4_fields':[207,300]},'closed_loop':result['closed_loop_physical_success'],'reference_executable_subset':result['closed_loop_reference_executable_subset'],'model_claim_counts':result['model_claim_counts'],'learned_unknown_stops':[11,12],'diagnosis':'Only root labels replaced with true labels, same latent/RNG; no plan/task recovery,0skills. No learned-model correction performed.','D_controller_audit':str(R/'evaluation/D_controller_audit.json'),'D_controller_dataset_status':{'success':180,'refused':49,'failed':11},'D_controller_side_effect':'1/240 slider movement>3mm, max146.85mm;9/240 non-target object>10mm, max28.24mm. Raw failures preserved.','completion_calibrated':False,'calibration_limitation':'8 reserved C roots below unchanged >=10-root minimum. Diagnostic fixed thresholds only.','selected_model_limitation':'Selected update1000 predates multi-step-loss stage;6000-update run finished but no multi-step-loss benefit claim.','original_full_guide_completed':False,'efficacy_passed':False,'old_pipeline_remains_paused':True,'old_main_data_transitions':4215,'old_manifest_unchanged':True,'controller_optimization_stopped_by_user':True,'controller_frozen_sha256':'0532b205828126e2b102d5c00dfa349d9534d870b7c077e797731b8fa08b973f','library_deliverables':receipts,'remaining_manual_work':[],'remaining_blockers':result['remaining_blockers'],'historical_handoff':str(previous),'next':'Report this completed first-round experiment and limitations; do not expand data, alter controller, or tune using D without a new user decision.'})
write(R/'post_pipeline_progress.json',{'phase':'complete','updated_utc':validation['checked_utc'],'complete_authorized_experiment':True,'efficacy_passed':False,'result':'runs/core1200_v1/RESULT.json','report':'reports/core1200_final.html','delivery_validation':'runs/core1200_v1/final_delivery_validation.json'})
print(json.dumps({'passed':validation['passed'],'hash_checks':checks,'root_matches':matches,'library_deliverables':receipts},ensure_ascii=False))
