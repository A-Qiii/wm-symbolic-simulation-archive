from pathlib import Path
import json,hashlib,datetime,time,shutil,ast
P=Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/relation_generalization_frozen_20261020';C=P/'experiments/object_recurrent_confirm_20261019';OLD=P/'experiments/rgbd_observation_control_20261020';N=P/'experiments/local_gap_control_20261020';W=Path('/home/robot/Documents/Codex/2026-10-13/task-4')
read=lambda p:json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not S.exists()
for d in ['source','configs','data','data/episodes','data/features','cache','models','results','reports','logs']:(S/d).mkdir(parents=True,exist_ok=True)
start=datetime.datetime(2026,10,20,1,10,14,tzinfo=datetime.timezone.utc).timestamp()
save(S/'configs/launch.json',{'start_epoch':start,'deadline_epoch':start+45*60,'collection_deadline_epoch':start+38*60,'maximum_minutes':45})
# Start is 01:10:14 UTC, total cutoff 01:55:14; leave seven minutes for feature extraction/reporting.
common=(N/'source/common.py').read_text().replace("S=P/'experiments/local_gap_control_20261020'","S=P/'experiments/relation_generalization_frozen_20261020'").replace("ARMS=['coarse','local']","ARMS=['baseline','geometry']").replace("'training_deadline_epoch'if training else'deadline_epoch'","'deadline_epoch'")
(S/'source/common.py').write_text(common)
shutil.copyfile(OLD/'source/models.py',S/'source/models.py')
shutil.copyfile(OLD/'configs/input_normalization.json',S/'configs/input_normalization.json')
shutil.copyfile(OLD/'configs/models_frozen.json',S/'configs/models_frozen.json')
families=read(C/'configs/family_role_proposal.json');families=[r for r in families if int(r['family_id'].rsplit('_',1)[1])<2]
for i,r in enumerate(families):
 r['family_id']=r['family_id'].replace('relation_confirm','relation_generalization');r['geometry_seed']=2610239000+i;r['split']='independent';r['collect_union_roles']=[['red','blue'],['blue','pink'],['pink','red']]
save(S/'configs/family_role_proposal.json',families)
inventory=read(C/'configs/AB_exclusion_inventory.json')
for r in read(C/'data/families_manifest.json'):inventory['entries'].append({'environment':r['environment'],'poses':[r['layout']['block_'+c]for c in ['red','blue','pink']],'source':r['family_id']})
save(S/'configs/AB_exclusion_inventory.json',inventory)
train=read(P/'experiments/object_recurrent_state_run_20261019/cache/train_metadata.json');seqs=[tuple(r['skill']for r in e['rows'])for e in train];audit=[]
for c in ['red','blue','pink']:
 known=(f'lift_{c}_block_table','place_in_drawer','close_drawer','open_drawer');new=('close_drawer','open_drawer',f'lift_{c}_block_table','place_in_drawer')
 audit.append({'target':c,'known4':known,'known_exact_count':sum(s==known for s in seqs),'novel4':new,'new_exact_count':sum(s==new for s in seqs),'new_contiguous4_count':sum(any(s[i:i+4]==new for i in range(len(s)-3))for s in seqs),'same_action_multiset':sorted(known)==sorted(new)})
assert all(r['known_exact_count']>0 and r['new_contiguous4_count']==0 and r['same_action_multiset']for r in audit)
save(S/'results/action_sequence_audit.json',{'train_episodes':len(train),'exact_sequences':len(set(seqs)),'verb_vocabulary':['open_drawer','close_drawer','lift_named_table_object','place_in_drawer'],'hold_verb_supported':False,'named_sequence_comparison':audit,'newness_scope':'Novel full/contiguous four-action order relative to actual model training. Primitive actions and many shorter fragments are known. Initial H-in-drawer contexts may also differ from old four-step roots; matched new-family order comparison shares identical setup.'})
episode=(C/'source/episode.py').read_text().replace("S=P/'experiments/object_recurrent_confirm_20261019'","S=P/'experiments/relation_generalization_frozen_20261020'")
episode=episode.replace("('held_'+root['held_color']+'_private.npz')","(root['initial_state']+'_'+root['held_color']+'_private.npz')")
episode=episode.replace("fault_active=branch==2 and step==1","fault_active=branch==2 and step==root['fault_step']")
episode=episode.replace("actions=[];trace=[];startpos=","actions=[];trace=[];probes=[];verify_trace=[];verify_count=0;startpos=")
episode=episode.replace("def record(obs,act):\n    actions.append", "def record(obs,act):\n    nonlocal verify_count\n    actions.append")
needle="    contacts=[list(c) for c in p.getContactPoints(physicsClientId=env.env.cid) if c[9]>.01]"
insert="""    if skill.startswith('lift_') and act['phase']=='verify_stable_completion':
     verify_count+=1;lab=label(env);verify_trace.append({'completion_control':verify_count,'control_index':len(actions),'labels':lab})
     if verify_count in [4,8,12]:
      prefix='held_probe_'+str(verify_count);hist_save(s,prefix,history)
      probes.append({'completion_control':verify_count,'control_index':len(actions),'public_inputs':[str(s/f'{prefix}_{j}.npz')for j in range(4)],'labels':lab})
"""
assert needle in episode;episode=episode.replace(needle,insert+needle)
episode=episode.replace("dump(s/'labels_private.json',labels);", "dump(s/'held_probes_private.json',{'probes':probes,'all_completion_controls':verify_trace,'no_added_controls':True});dump(s/'labels_private.json',labels);")
(S/'source/episode.py').write_text(episode)
for name in ['geometry','collect','features','evaluate','score','report','finalize']:
 path=W/f'relgen_{name}.py';assert path.exists(),path;shutil.copyfile(path,S/f'source/{name}.py')
protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'FROZEN_BEFORE_NEW_GEOMETRY_OR_MODEL_RESULTS','time_limit_minutes':45,'cost_estimate':'72 episodes,336 task calls plus logged setups;12–18min collection; total30–45min; about1–1.5GB including features. No training/download/upload.','models':{'arms':['baseline same-capacity Object with frozen-feature random projections','geometry Object with public RGB-D coarse grid and per-object geometry'],'seeds':[0,1],'parameters_each':2490068,'checkpoint_update':1500,'weights_and_normalization':'From sealed RGB-D comparison; both arms already trained. geometry remains candidate, not winner. local_gap excluded.','global_baseline':'Not included: no comparable frozen same-input global model.'},'geometry':{'families':4,'A':2,'B':2,'role_pairs_per_family':3,'pairs':[['red','blue'],['blue','pink'],['pink','red']],'branches':2,'previous_geometry_excluded':len(inventory['entries']),'admission':'Deterministic up to128 candidates; geometry-only footprint/clearance/IK/reset; reject all three positions within20mm under any color permutation of old or accepted family. Freeze admitted roots before task rollouts. No outcome/model-based replacement.'},'templates':{'recovery_seen6':'open+lift H setup; P,L(T),P,L(T),P,C. Same trained order; new geometry. Branch2 overrides gripper-close on first requested L(T), then existing second L(T) can recover.','order_known4':'open+lift H+place H setup; L(T),P,C,O. Exact order seen in training.','order_novel4':'Same open/empty/H-in-drawer setup as order_known4; C,O,L(T),P. Same action multiset, unseen full and contiguous four-step training order. Branch2 overrides first L(T). No new skill.'},'prediction_starts':{'current':'h0 then each real observed boundary; no future observations','one_step':'Each actual previous posterior, predict next boundary without next RGB-D','free_h0':'From actual h0, whole 4 or6 action sequence with no future observations','free_h2':'Additional known recovery6 only: observe h0..h2 including grasp outcome, then predict remaining4 actions with no future observations'},'hold_probe':{'source':'Existing unmodified PhysicalOracle lift verify_stable_completion controls4,8,12; no new controllers/actions/physical holds. Public four-frame windows only.','model_usage':'Apply the same pre-lift action prior independently to each probe observation. No hold verb or hold-transition evaluation claimed. Probe posterior never fed into main trajectory.','positive_denominator':'Requested target truly Held continuously at every recorded control from probe4 through8 or8 through12. Each eligible interval counted; report positive frames, eligible runs, failures and absent/incomplete probes. Family/episode correlation explicit.','In_denominator':'Non-target H truly InDrawer throughout corresponding control interval; report retained relations separately.','additional_controls':0},'scoring':{'thresholds':'Held/In>=0.5; drawer argmax, unchanged','per_step':'All-three Held, all-three In, drawer, conjunction; invalid GT counted and shown, never silently dropped','whole':'Every observed boundary or every predicted boundary correct; report branch and template separately','semantic_checks':'Requested T Held on lift; T InDrawer and empty hand on placement; non-target H InDrawer; actual execution/refusal/success logged','close':'All close calls remain in full metrics, plus separate close status/three-class breakdown; no further close-feature search','retention':'Stable true Held/In at task boundaries and control probes; positive and prior-correct denominators explicit','family':'Per-family outcomes; paired family bootstrap2000 for geometry-minus-baseline, no acceptance gate invented','failure_policy':'No replacement or selective dropping. Partial collection reported as partial, preferably complete families; no test-to-training or threshold selection.'},'no_training':True,'no_private_model_inputs':True,'CD_accesses':0,'old_gates_unchanged':True}
save(S/'configs/protocol.json',protocol)
shutil.copyfile(W/'relgen_setup.py',S/'source/setup.py')
files=list((S/'source').glob('*.py'))+list((S/'configs').glob('*.json'))+[P/'src/wm_symbolic/physical_oracle.py',P/'src/wm_symbolic/public_geometry.py',P/'src/wm_symbolic/vision_cache.py']
files += [P/'src/wm_symbolic/world_model.py',P/'src/wm_symbolic/env_adapter.py',P/'src/wm_symbolic/calibrated_labels.py',P/'src/wm_symbolic/core_ownership.py',P/'scripts/evaluate_physical_oracle.py',P/'experiments/object_identity_readout_20261019/source/run.py',P/'experiments/spatial_coverage_readout_run_20261019/source/preflight_feasible.py',OLD/'source/data.py',P/'experiments/rssm_scaling_run_20261019/cells/roots40/configs/normalization.json'] + [P/f'experiments/rssm_scaling_run_20261019/cells/roots40/models/joint_seed{seed}/latest.pt' for seed in [0,1]]
hashes={str(p):sha(p)for p in files};hashes.update(read(S/'configs/models_frozen.json')['sha256']);save(S/'configs/design_frozen.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_any_new_geometry_or_rollout':True,'sha256':hashes})
for p in files+[S/'configs/design_frozen.json']:
 if p.is_relative_to(S):p.chmod(0o444)
print(json.dumps({'root':str(S),'deadline':datetime.datetime.fromtimestamp(read(S/'configs/launch.json')['deadline_epoch'],datetime.timezone.utc).isoformat(),'episodes':72,'task_calls':336},indent=2))
