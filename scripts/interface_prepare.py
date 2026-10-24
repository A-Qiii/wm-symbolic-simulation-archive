from pathlib import Path
import json,hashlib,datetime,shutil
import numpy as np
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/proposition_interface_20261014';B=P/'data/core1200_v1'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
R.mkdir(exist_ok=False)
for sub in ['snapshots','roots','evaluation','logs','media','figures']:(R/sub).mkdir()
names=['symbol_decoder.py','search.py','model_runtime.py','core_runtime.py','physical_oracle.py','world_model.py']
for name in names:shutil.copy2(P/'src/wm_symbolic'/name,R/'snapshots'/name)
for name in ['core1200_closed_loop.py','core1200_executor_service.py','core1200_tasks.py','core1200_calibrate.py']:shutil.copy2(P/'scripts'/name,R/'snapshots'/name)
roots=json.loads((B/'roots_manifest.json').read_text());dev=[]
for rid in ['A_core1200_00','B_core1200_01','C_core1200_00','C_core1200_01','C_core1200_02','C_core1200_03']:
 root=next(x for x in roots if x['root_id']==rid).copy();root['source_root_path']=str(B/'roots'/rid/'branch0_initial_private.npz');root['cohort']='ABC_development';dev.append(root)
cfg=OmegaConf.load(P/'vendor/calvin/calvin_env/conf/scene/calvin_scene_D.yaml');order=list(cfg.objects.movable_objects);lo,hi=np.array(cfg.surfaces.table);colors=['red','blue','pink'];excluded=[]
paths=list((P/'data/roots').glob('*/D_*/reset_evaluator_only.npz'))+list((B/'roots').glob('D_*/branch0_initial_private.npz'))
for version in ['v1','v2','v3']:paths+=list((P/f'runs/physical_oracle_{version}/roots').glob('D_*/reset_evaluator_only.npz'))
for f in paths:
 with np.load(f) as z:lookup=dict(zip(order,z['scene_obs'][6:].reshape(3,6)));excluded.append((np.stack([lookup['block_'+c] for c in colors]),str(f)))
test=[];rejections=[]
for i,kind in enumerate(['closed_empty','open_empty','open_held','closed_empty','open_empty','open_held']):
 seed=617000+i;rng=np.random.default_rng(seed);rejected=0
 while True:
  poses=np.zeros((3,6))
  for slot,j in enumerate(rng.permutation(3)):
   poses[j,:3]=[lo[0]+(hi[0]-lo[0])*(.16+.34*slot)+rng.uniform(-.014,.014),rng.uniform(-.14,-.05),.46];poses[j,5]=rng.uniform(-np.pi,np.pi)
  near=False
  for other,name in excluded:
   d=np.linalg.norm(poses[:,:3]-other[:,:3],axis=1);a=np.abs((poses[:,5]-other[:,5]+np.pi/2)%np.pi-np.pi/2)
   if np.all(d<=.012) or np.all(d<=.02) and np.all(a<=.25):near=True;break
  if not near:break
  rejected+=1
 rid=f'D_interface_{i:02d}';color=colors[i%3];root={'root_id':rid,'environment':'D','seed':seed,'initial_state':kind,'held_color':color if kind=='open_held' else None,'layout':{'block_'+c:poses[j].tolist() for j,c in enumerate(colors)},'setup_skills':([] if kind=='closed_empty' else ['open_drawer'])+([f'lift_{color}_block_table'] if kind=='open_held' else []),'cohort':'fresh_D_roots_seen_environment','source_root_path':str(R/'roots'/rid/'branch0_initial_private.npz')};test.append(root);excluded.append((poses,rid));rejections.append({'root_id':rid,'rejected_layouts':rejected})
for root in dev+test:
 d=R/'roots'/root['root_id'];d.mkdir();dump(d/'root.json',root);(d/'root.json').chmod(0o444)
dump(R/'roots_manifest.json',dev+test)
protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'One interface factor: proposition probability/UNKNOWN gating. No model training, controller adaptation, or changes to existing datasets.','old_guide_basis':{'line279':'Original guide explicitly says related facts UNKNOWN when multiple held or maximum category probability insufficient. This experiment revises that design assumption, not merely fixes an implementation contradiction.','line420':'Independent calibration chooses thresholds to target<=5% false completion claims and then coverage; insufficient evidence retains uncalibrated mark.','correction':'10 independent roots and30 claims were later engineering support rules, not requirements written in the original guide.'},'frozen_inputs':{'checkpoint':str(P/'runs/core1200_v1/training_main/best.pt'),'checkpoint_sha256':sha(P/'runs/core1200_v1/training_main/best.pt'),'controller_sha256':sha(P/'src/wm_symbolic/physical_oracle.py'),'data_manifest_sha256':sha(B/'data_manifest.json'),'source_hashes':{n:sha(P/'src/wm_symbolic'/n) for n in names}},'modes':['legacy_max_category_gate','proposition_margin_gate'],'shared_interface_safety':'Same goals, DFA constraints, all3 held facts and HandEmpty; each candidate also checks relevant pre-action facts for observability. False precondition values never prune actions or specify effects. All successors still come from RSSM. Shared complete fact scope makes paired baseline stricter than the historical goal/DFA-only baseline; historical results remain untouched.','probabilities':'Table=sum eight-class entries0:3; drawer=3; held=6. Original fine-class goals remain fine. Same function used in actual-posterior and imagined-successor decoding. Global multiple-held conflict and never-registered objects remain UNKNOWN.','threshold_plan':{'temperature':'retain previously C-calibrated1.69713068; no refit','event_temperature':'retain1.63376129','symbol_threshold_grid':[.8,.85,.9,.95],'event_threshold_grid':[.9,.95,.99],'selection':'Use only eight reserved C calibration roots; choose lowest threshold on grid satisfying <=5% empirical proposition assertion errors across both current and one-step predictions, and <=5% terminal-conjunction false claims when claims exist. If none qualifies keep0.8/0.9 diagnostic defaults and uncalibrated flag. Apply identical thresholds to both interface modes. No D-driven changes.','certification':'Report denominators, root coverage and uncertainty; empirical calibration is not statistical safety certification.'},'ABC_offline':'All60 train roots as descriptive in-sample AB diagnostics;12 C tuning roots development plus8 C independent calibration. No previous D metrics used for threshold selection.','ABC_closed_loop_tasks':6,'new_D_roots':6,'new_D_tasks':8,'methods_on_new_D':['legacy','proposition','reference'],'new_D_distribution':'Same CALVIN D environment, same separated-tabletop layout generator and frozen controller as prior study; new seeds/layouts/tasks. D environment and prior failure modes already observed, so this is fresh-root replication of a revised hypothesis, not unseen-environment generalization.','root_exclusion':'All historical same-environment roots and previous coreD actual roots plus new layouts;12mm position or20mm+0.25rad yaw rule. Also audit actual settled roots, keep failed setups.','root_exclusion_candidates':len(paths),'rejected_layouts':rejections,'limits':{'max_requested_skills':8,'search_beam':32,'horizon':8,'successor_queries_per_search':5000,'latent_samples':8,'reobserve_controls':4,'reobserve_limit':1},'physical_safety':'Keep original20mm moved-event criterion for forbidden objects. Preserve all continuous device/object displacement; report significant unrequested movement separately. Small brushing alone is not auto-failure; never hide large motion. New tasks also require slider_right to remain true.','test_freeze':'Save all roots and tasks before ABC physical/model validation; freeze code/threshold hashes afterABC before anynewD reset.'}
dump(R/'protocol.json',protocol);(R/'protocol.json').chmod(0o444)
dump(R/'progress.json',{'stage':'prepared','new_D_observed':False,'training_started':False});print(str(R))
