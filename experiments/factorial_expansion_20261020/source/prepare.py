"""Run once before any simulator outcome; declares all factors and quotas."""
from common import *
import itertools
def main():
 assert not (S/'configs/design_frozen.json').exists()
 for d in ['configs','data','results','logs','reports','cache','figures']:(S/d).mkdir(parents=True,exist_ok=True)
 old=P/'experiments/relation_generalization_frozen_20261020'
 exclusion=read(old/'configs/AB_exclusion_inventory.json')['entries']
 for location in [old,P/'experiments/midstart_training_control_20261020/fresh_validation']:
  for f in read(location/'data/families_manifest.json'):
   exclusion.append({'environment':f['environment'],'poses':[f['layout']['block_'+c]for c in COLORS],'source':str(location)+':'+f['family_id']})
 save(S/'configs/AB_exclusion_inventory.json',{'entries':exclusion})
 inventory=[
  dict(source='original_model_train',trajectories=228,transitions=1128,roots=94,purpose='training;120core+108role; count once'),
  dict(source='original_development',trajectories=78,transitions=408,roots=34,purpose='development only'),
  dict(source='historical_evaluation',trajectories=126,transitions=696,roots=58,purpose='historical evaluation; not training'),
  dict(source='object_recurrent_confirm',trajectories=96,transitions=576,roots=48,purpose='diagnostic confirmation; not training'),
  dict(source='relation_generalization_frozen',trajectories=72,transitions=336,roots=36,purpose='explored independent evaluation; not training'),
  dict(source='midstart_crops',trajectories=0,transitions=0,roots=0,purpose='reused original228; no new experience'),
  dict(source='midstart_fresh_partial',trajectories=5,transitions=20,roots=5,purpose='unfinished independent validation; sealed, not scored or training')]
 save(S/'configs/data_inventory.json',inventory)
 permutations=list(itertools.permutations(COLORS));families=[]
 for split,n in [('train',160),('development',20),('combination_test',20)]:
  for i in range(n):
   e='AB'[i%2];idx=len(families)
   families.append(dict(family_id=f'{e}_factorial_{split}_{i:03d}',environment=e,split=split,index_in_split=i,
       geometry_seed=2610280000+idx,color_assignment_to_x_sorted_slots=permutations[idx%6],
       state_kinds=[(2*i+j)%5 for j in range(2)]if split!='combination_test'else[4,5],
       pilot=(i<4 if split=='train'else i<2)))
 save(S/'configs/family_proposal.json',families)
 protocol=dict(utc=now(),scope='Data expansion only, using existing fixed four verbs/six grounded actions; not full25-skill guide acceptance',
   counts=dict(new_episodes=800,new_requests=4800,train_episodes=640,development_episodes=80,combination_test_episodes=80,
     new_families=200,new_intended_roots=400,branches_per_root=2,steps_per_episode=6,pilot_episodes=32,
     combined_train_if_no_quarantine=868,combined_train_requests_if_no_quarantine=4968),
   initial_state_kinds={0:'all_table_empty_open_or_closed',1:'one_Held_open',2:'one_In_empty_open_or_closed',3:'one_In_plus_other_Held_open',4:'two_In_empty_open',5:'two_In_plus_third_Held_open_TEST_ONLY'},
   holdout='At least two objects In(drawer), and ALL THREE objects individually In(drawer) or Held. Includes twoIn+thirdHeld and threeIn. Audit all new control states and public boundary windows. Whole training/development family quarantined if encountered. No outcome replacement.',
   old_train_holdout='Original228 macro-boundary categorical cache audited: zero such combinations. Does not certify every historical low-level control.',
   splits='New family disjoint and assigned before geometry or motor outcomes. Test results sealed from model selection. No historical evaluation backfill. A/B native environments only.',
   geometry=dict(centers_A=[-.10,.06,.22],centers_B=[-.22,-.08,.08],x_jitter=.022,y_range=[-.145,-.065],yaw_range=[-float(np.pi),float(np.pi)],max_proposals_per_family=128,near_duplicate_m=.020),
   setup='Initial relation states established physically by unchanged controller, with full private setup controls. Setup failures retained with intended and actual state; no result-based replacement.',
   branches='Matched initial physical roots; fixed nominal versus gripper-open miss+retry when valid, otherwise fixed different order. Branches are distinct experiences but not independent roots.',
   failure='All successful, failed, refused, interrupted outcomes retained; no fake no-change target. Requested/applied disturbance stored privately, never an observation.',
   pilot_gate='32episodes/192requests, all physics guards;6commits each; same-root public equality; explicit public allowlist; no admitted train/dev heldout combination; feasible intended states across five factors. First50 storage, first128 record QA. Stop for genuine factor or provenance blocker.',
   budget=dict(max_seconds=21600,estimated_hours=[3,5],estimated_raw_GB=[12,18],min_free_GiB=30),
   training_updates=0,feature_cache='deferred until collection acceptance',controller_sha256=sha(P/'src/wm_symbolic/physical_oracle.py'),
   old_partial='Independent24 request deferred; its completed5 and interrupted6th preserved, not mixed into expansion',no_upload=True)
 save(S/'configs/protocol.json',protocol)
 save(S/'configs/launch.json',{'started_utc':now(),'started_epoch':time.time(),'deadline_epoch':time.time()+21600})
 m=P/'experiments/midstart_training_control_20261020';verified=0
 for f in read(m/'MANIFEST.json')['files']:
  assert sha(m/f['path'])==f['sha256'],f['path'];verified+=1
 save(S/'results/historical_seal_before.json',{'utc':now(),'files_verified':verified,'manifest_sha256':sha(m/'MANIFEST.json'),'changes':0})
 paths=list((S/'source').glob('*.py'))+[S/'configs'/f for f in ['protocol.json','family_proposal.json','AB_exclusion_inventory.json','data_inventory.json']]
 paths += [P/'src/wm_symbolic/physical_oracle.py',P/'src/wm_symbolic/env_adapter.py',P/'experiments/spatial_coverage_readout_run_20261019/source/preflight_feasible.py']
 save(S/'configs/design_frozen.json',{'utc':now(),'before_geometry_and_motor_outcomes':True,'sha256':{str(p):sha(p)for p in paths}})
 print('FROZEN',len(families),'families;800episodes;old files verified',verified,flush=True)
if __name__=='__main__':main()
