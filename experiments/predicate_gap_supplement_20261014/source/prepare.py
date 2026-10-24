from pathlib import Path
import json,hashlib,datetime
import numpy as np
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'data/predicate_gap_supplement_20261014';R=P/'experiments/predicate_gap_supplement_20261014';C=['red','blue','pink']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 assert not (R/'protocol.json').exists()
 B.mkdir(exist_ok=False)
 for p in [B/'episodes',B/'roots',R/'source',R/'logs',R/'figures',R/'results']:p.mkdir(parents=True,exist_ok=True)
 old=[]
 for path in [P/'data/core1200_v1/roots_manifest.json',P/'runs/proposition_interface_20261014/roots_manifest.json']:
  for r in json.loads(path.read_text()):
   if 'layout' in r:old.append((r['environment'],np.array([r['layout']['block_'+c] for c in C]),r['root_id']))
 # Include old actual reset root families, which were not stored in the above layout files.
 for e in 'ABCD':
  cfg=OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml');names=list(cfg.objects.movable_objects)
  paths=list((P/'data/roots').glob(f'*/{e}_*/reset_evaluator_only.npz'))
  paths += [q for v in ['v1','v2','v3'] for q in (P/f'runs/physical_oracle_{v}/roots').glob(f'{e}_*/reset_evaluator_only.npz')]
  for f in paths:
   with np.load(f) as z:a=z['scene_obs'][6:].reshape(3,6);lookup=dict(zip(names,a));old.append((e,np.array([lookup['block_'+c] for c in C]),str(f)))
 roots=[];exclusion=[]
 for ei,e in enumerate('ABCD'):
  cfg=OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml');lo,hi=np.array(cfg.surfaces.table)
  for ci,color in enumerate(C):
   for di,drawer in enumerate(['closed','open']):
    i=ci*2+di;seed=914000+ei*100+i;rng=np.random.default_rng(seed);rejected=0
    while True:
     poses=np.zeros((3,6));order=rng.permutation(3)
     for slot,j in enumerate(order):
      poses[j,:3]=[lo[0]+(hi[0]-lo[0])*(.14+.35*slot)+rng.uniform(-.012,.012),rng.uniform(-.125,-.055),.46]
      poses[j,5]=[-np.pi/3,0,np.pi/3][(ci+di+ei+j)%3]+rng.uniform(-.08,.08)
     near=[]
     for ee,other,name in old:
      if ee!=e:continue
      dist=np.linalg.norm(poses[:,:3]-other[:,:3],axis=1);yaw=np.abs((poses[:,5]-other[:,5]+np.pi/2)%np.pi-np.pi/2)
      if np.all(dist<=.012) or (np.all(dist<=.020) and np.all(yaw<=.25)):near.append(name)
     if not near:break
     rejected+=1
    rid=f'{e}_predicate_gap_{color}_{drawer}';root={'root_id':rid,'environment':e,'split':'train' if e in 'AB' else 'dev' if e=='C' else 'test','seed':seed,'target_color':color,'drawer_setting':drawer,'drawer_reset_m':0.0 if drawer=='closed' else .22,'layout':{'block_'+c:poses[j].tolist() for j,c in enumerate(C)},'near_probe_yaw_offset_rad':0.0 if (ci+di+ei)%2==0 else np.pi/2,'near_probe_translation_m':[0,-.055,.025],'follow_probe_translation_m':[.015 if (ci+di+ei)%2==0 else -.015,0,0],'trajectories':['grasp','near_closed_empty']}
    roots.append(root);old.append((e,poses,rid));exclusion.append({'root_id':rid,'rejected_layout_candidates':rejected})
 pilot=['A_predicate_gap_red_closed','A_predicate_gap_blue_open','B_predicate_gap_pink_closed','B_predicate_gap_red_open']
 protocol={'id':'predicate_gap_supplement_20261014','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'用户：你先补样本；只采集与质量验证，不训练、不改控制器、不规划、不上传','root_count':24,'trajectory_count':48,'roots_per_split':{'train':12,'dev':6,'test':6},'environments':{'train':['A','B'],'dev':['C'],'test':['D']},'pilot_roots':pilot,'root_split_frozen_before_execution':True,'selection_rule_frozen_before_test':True,'branches_same_group':True,'reset_rule':'Only initial reset sets block poses and drawer joint 0/.22m; objects start supported on table. No reset/teleport during motor-driven trajectories. Each branch has same seed/layout; settled scene/robot root checked <=1e-5.','grasp':'Exact existing PhysicalOracle.run(lift_COLOR_block_table), including its already frozen bounded retry; no added control optimization.','near_negative':'Fixed measurement trajectory using unchanged PhysicalOracle.safe_height/move/hold motor primitives: close far away; move to object offset (0,-55,+25)mm; hold; translate +/-15mm in X; hold. This is an explicitly logged collector probe, not a changed skill or grasp success.','planned_windows_per_root':['grasp/far_open_negative','grasp/near_open_negative','grasp/contact_supported_closed_negative_if_present','grasp/held_positive','near_closed_empty/far_closed_negative','near_closed_empty/near_closed_negative','near_closed_empty/near_closed_motion_negative'],'window_selection':'At most one 4-past-frame clip per named condition/trajectory, last qualifying stable four-frame window in its fixed phase. Missing conditions are retained as gaps, never replaced with synthetic or outcome-resampled roots.','near_definition_m':.075,'far_definition_m':.12,'physical_empty_gripper_max_width_m':.015,'held_definition':'Existing core label: bilateral force-bearing (>0.01N) contacts to descendant finger families and no non-robot external contact. Nearness/close command are never sufficient.','audit_only':'Held positives additionally need >=20mm TCP and object co-motion with <=15mm relative displacement drift, during continuous held phase. Near-empty negative motion needs >=8mm TCP displacement, object <=3mm displacement, no held contacts and persistent table support. Original labels are retained; disagreement stops collection for review.','negative_visibility':'A negative clip requires target visible in at least one camera in every past frame; invisible target is retained with audit gap, not accepted as held=false training evidence.','drawer_definition':'Original categories: closed<.025m, open>.16m, intermediate otherwise; exclude +/-5mm threshold ambiguity. Store exact state/boundary distances and visibility.','small_batch_gate':'4 train roots/8 trajectories. Require real held and near-closed negatives in >=2 roots each; physics guard/no label contradiction/root replay/input isolation pass; inspect actual images before continuing.','no_model_inference':True,'no_model_training':True,'no_upload':True,'source_sha256':{str(P/'src/wm_symbolic/physical_oracle.py'):sha(P/'src/wm_symbolic/physical_oracle.py'),str(P/'scripts/core1200_collect.py'):sha(P/'scripts/core1200_collect.py'),str(P/'src/wm_symbolic/calibrated_labels.py'):sha(P/'src/wm_symbolic/calibrated_labels.py'),str(P/'src/wm_symbolic/state_labels.py'):sha(P/'src/wm_symbolic/state_labels.py'),str(P/'src/wm_symbolic/core_ownership.py'):sha(P/'src/wm_symbolic/core_ownership.py')},'old_manifest_sha256':sha(P/'data/core1200_v1/data_manifest.json'),'old_model_sha256':sha(P/'runs/core1200_v1/training_main/best.pt'),'prior_reader_model_sha256':sha(P/'experiments/state_predicates_20261014/models/direct_predicates_seed0.pt')}
 save(B/'roots_manifest.json',roots);save(R/'protocol.json',protocol);save(R/'results/root_exclusion_plan.json',{'compared_old_plus_new_layouts':len(old),'new_roots':24,'rows':exclusion});save(B/'data_manifest.json',{'schema':'predicate_gap_supplement_v1','records':[],'trajectories':[]});save(R/'results/progress.json',{'stage':'plan_frozen','completed_trajectories':0})
 (R/'protocol.json').chmod(0o444);(B/'roots_manifest.json').chmod(0o444)
 print('PLAN_FROZEN',len(roots),len(roots)*2,flush=True)
if __name__=='__main__':main()
