from pathlib import Path
import json,hashlib,time,datetime,shutil
P=Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/rgbd_observation_control_20261020';C=P/'experiments/object_recurrent_confirm_20261019';F=S/'fresh_test';W=Path('/home/robot/Documents/Codex/2026-10-13/task-4')
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
analysis=read(S/'results/analysis.json');assert analysis['supports_fresh_test_under_frozen_rule'],'Frozen development continuation rule failed'
remaining=read(S/'configs/launch.json')['deadline_epoch']-time.time();nf=8 if remaining>=21*60 else 4 if remaining>=12*60 else 0;assert nf,'Insufficient time for an intact fresh cohort'
assert not F.exists()
for d in ['source','configs','data','cache','results','logs','data/episodes']:(F/d).mkdir(parents=True,exist_ok=True)
for name in ['common.py','geometry.py','collect.py','cache.py','episode.py','data.py']:
 text=(C/'source'/name).read_text();text=text.replace("S=P/'experiments/object_recurrent_confirm_20261019'","S=P/'experiments/rgbd_observation_control_20261020/fresh_test'").replace("ARMS=['global','object']","ARMS=['baseline','geometry']")
 if name=='geometry.py':text=text.replace('len(accepted)==8','len(accepted)=='+str(nf)).replace('==48','=='+str(nf*6)).replace("'families':8","'families':"+str(nf)).replace("'role_versions':48","'role_versions':"+str(nf*6)).replace("'roles':48","'roles':"+str(nf*6))
 if name=='collect.py':text=text.replace('576',str(nf*72)).replace('==48','=='+str(nf*6)).replace("'role_pairs':48","'role_pairs':"+str(nf*6))
 if name=='data.py':text=text.replace('len(out)==96','len(out)=='+str(nf*12))
 (F/'source'/name).write_text(text)
families=read(C/'configs/family_role_proposal.json');families=[r for r in families if int(r['family_id'].rsplit('_',1)[1])<nf//2]
for i,r in enumerate(families):r['family_id']=r['family_id'].replace('relation_confirm','rgbd_independent');r['geometry_seed']=2610219000+i
assert len(families)==nf
inventory=read(C/'configs/AB_exclusion_inventory.json')
for r in read(C/'data/families_manifest.json'):inventory['entries'].append({'environment':r['environment'],'poses':[r['layout']['block_'+c]for c in ['red','blue','pink']],'source':r['family_id']})
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
save(F/'configs/family_role_proposal.json',families);save(F/'configs/AB_exclusion_inventory.json',inventory)
launch=read(S/'configs/launch.json');save(F/'configs/launch.json',{'deadline_epoch':launch['deadline_epoch'],'collection_deadline_epoch':launch['deadline_epoch']-180,'launch_epoch':time.time()})
plan={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'FROZEN_BEFORE_NEW_PHYSICS','families':nf,'environments':{'A':nf//2,'B':nf//2},'all_six_H_T_roles_per_family':True,'branches_per_role':2,'requested_transitions':nf*72,'expected_collection_minutes':15.37*nf/8,'expected_storage_bytes':int(1491562525*nf/8),'excluded_prior_layout_entries':len(inventory['entries']),'novelty':'Reject if all three positions within20mm under any color permutation of an old or newly accepted family; only physical geometry admission before task execution.','models':'All four fixed1500 checkpoints, no tuning on fresh data','data_policy':'No outcome-based replacement; retain failures. No C/D. Frozen original thresholds and metrics.','evaluation':'h0 Held, close one-step and h2-to-h6, Held/In retention, normal whole and joint3/4, paired family bootstrap intervals.','scope':'Independent AB geometry check of added observable information, not original-guide complete acceptance','trigger_development_targets':analysis['targets_improved_in_both_seeds'],'budget_remaining_seconds_at_freeze':remaining}
save(F/'configs/protocol.json',plan);shutil.copyfile(S/'configs/models_frozen.json',F/'configs/models_frozen.json')
(F/'source/extract_front.py').write_text("from common import *\nimport torch,fcntl\nfrom data import extract\nguard();verify();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)\nfor seed in [0,1]:extract('confirm',seed)\n")
files=list((F/'source').glob('*.py'))+list((F/'configs').glob('*.json'))+[S/'source/analyze.py',P/'src/wm_symbolic/physical_oracle.py',P/'src/wm_symbolic/public_geometry.py',P/'src/wm_symbolic/vision_cache.py']
hashes={str(p):sha(p)for p in files};hashes.update(read(S/'configs/models_frozen.json')['sha256']);save(F/'configs/design_frozen.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_any_new_physics':True,'sha256':hashes})
for p in files+[F/'configs/design_frozen.json']:
 if p.is_relative_to(F):p.chmod(0o444)
print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
