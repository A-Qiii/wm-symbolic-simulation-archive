"""Freeze the separately authorized six-skill core study; no simulator execution."""
from pathlib import Path
import json, hashlib, datetime
import numpy as np
from omegaconf import OmegaConf

P=Path('/home/robot/research/projects/wm_symbolic'); B=P/'data/core1200_v1'; R=P/'runs/core1200_v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def main():
 B.mkdir(exist_ok=False); R.mkdir(exist_ok=False)
 skills=['open_drawer','close_drawer','lift_red_block_table','lift_blue_block_table','lift_pink_block_table','place_in_drawer']
 protocol={'id':'six_skill_core1200_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'authorization':{'human_reply':'好的','reply_utc':'2026-10-14T04:01:27Z','proposal':'采集约1200条数据并进行首轮训练','parent_thread':'01a11850-0ce1-73fb-9aa2-4e19130a1031','scope':'new independent six-skill dataset, random-initialized WM, independent predictive evaluation and small closed-loop study'},
 'controller':json.loads((P/'configs/controllers/selected_core_executor.json').read_text()),
 'counts':{'train':{'environments':'AB','roots':60,'transitions':720},'dev':{'environments':'C','roots':20,'transitions':240},'test':{'environments':'D','roots':20,'transitions':240}},
 'root_states':{'closed_empty':50,'open_empty':30,'open_held':20},'branches_per_root':3,'length':4,'skills':skills,
 'public_input':['four past/current dual-camera RGB frames','visible pixel segmentation masks without semantic lookup','robot proprioception','requested skill/static arguments','fixed 900-control budget'],
 'forbidden_input':['GT scene state/contact/positions','root state type/seed/split','controller status/reason','success label','future observation/outcome','task goal text in dynamics'],
 'root_reset':'Fresh simulator per branch; deterministic separated tabletop layout, 24 motor hold controls, physical open/lift setup when required; setup explicitly outside four requested transitions, saved and replayed. Compare true branch-start state, never teleport held objects.',
 'near_duplicate_rule':'Within same environment and named colors, reject all3 position distances <=12mm, or all<=20mm and all yaw differences modulo pi <=0.25rad. Exclude all existing roots and all81 controller roots, including unused v3. Apply among new roots too.',
 'counterfactuals':'Two planned missing-precondition calls per12 transitions; actual failures/refusals retained, no goal-based resampling. No implicit prerequisite completion.',
 'composition_holdout':'On held roots, second object follows cyclic color+1 in training and color+2 in dev/test. Those ordered two-object storage combinations are absent from train. Other branch structures shared to isolate novel scene/root and object-order effects.',
 'dev_roles':'C roots00-11 tune checkpoint; roots12-19 calibrate thresholds after selection. Test D withheld from training/model selection and run once after model+evaluation freeze.',
 'labels':'Actual calibrated locations and drawer/slider/light states, force-bearing contact ownership. Bilateral dynamic finger families for held. Invalid ambiguous facts masked. 3 moved events from full control trace. Success is official+stable controller goal, with task invariants scored separately; continuous incidental motion and original strict flags retained.',
 'training':{'fresh_random_initialization':True,'architecture':'unchanged frozen DINOv2-small +2-layer object Transformer + categorical RSSM + shared symbolic readout','updates':6000,'batch':8,'accumulate':2,'seed':0,'one_step_until':5000,'multi_horizon_after':3,'lr':0.0003,'loss':'unchanged world_model.loss_for_batch','probe_transitions':128,'probe_updates':1000,'probe_checkpoint_never_used_to_initialize_main':True},
 'evaluation':{'horizons':[1,2,3,4],'latent_samples':8,'checkpoint_score':'mean free3 symbol NLL + event NLL on12 dev tuning roots','baselines':['observed-state persistence','oracle-label persistence (diagnostic upper-bound perception)','shuffled requested action with unchanged current observation'],'report':['current state recognition','all facts and changing facts separately','object identity and holding/support','root-cluster uncertainty','action dependence','per-state-type and per-skill counts'],'test_once':True},
 'planning':'Only learned WM supplies action successors. Six-action search without hand-written dynamic precondition/effect mask. Task formulas impose goal/order/safety; no controller implicit prerequisites. Small varied initial-state/goal subset, same-controller fixed-script and GT-state goal-search diagnostics separately identified.',
 'gates':'First50 storage; first128 input isolation/training smoke. Complete collection QA before main fit. If inadequate predictions, diagnose first consequential error, at most one supported change; no unsolicited expansion to6000 transitions.',
 'old_data_manifest_sha256':sha(P/'data/data_manifest.json'),'old_main_pipeline_stays_paused':True,'original_full_guide_completed':False}
 write(R/'protocol.json',protocol); (R/'protocol.json').chmod(0o444)
 colors=['red','blue','pink']; roots=[]; excluded=[]; native={}
 for e in 'ABCD':
  cfg=OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml'); names=list(cfg.objects.movable_objects); native[e]=(cfg,names)
  candidates=list((P/'data/roots').glob(f'*/{e}_*/reset_evaluator_only.npz'))
  candidates += [q for version in ['v1','v2','v3'] for q in (P/f'runs/physical_oracle_{version}/roots').glob(f'{e}_*/reset_evaluator_only.npz')]
  for path in candidates:
   with np.load(path) as z:
    poses=z['scene_obs'][6:].reshape(3,6); lookup={n:v for n,v in zip(names,poses)}
    ordered=np.stack([lookup['block_'+c] for c in colors]); excluded.append((e,ordered,str(path)))
 types=['closed_empty','open_empty','open_held','closed_empty','open_empty','closed_empty','open_held','closed_empty','open_empty','closed_empty']
 audit=[]
 for ei,e in enumerate('ABCD'):
  cfg,names=native[e]; lo,hi=np.array(cfg.surfaces.table); count=30 if e in 'AB' else 20
  for i in range(count):
   seed=221000+ei*1000+i; rng=np.random.default_rng(seed); rejected=0
   while True:
    ordered=np.zeros((3,6)); order=rng.permutation(3)
    for slot,j in enumerate(order):
     ordered[j,:3]=[lo[0]+(hi[0]-lo[0])*(.16+.34*slot)+rng.uniform(-.014,.014),rng.uniform(-.14,-.05),.46]
     ordered[j,5]=rng.uniform(-np.pi,np.pi)
    near=[]
    for ee,other,path in excluded:
     if ee!=e:continue
     dist=np.linalg.norm(other[:,:3]-ordered[:,:3],axis=1); yaw=np.abs((other[:,5]-ordered[:,5]+np.pi/2)%np.pi-np.pi/2)
     if np.all(dist<=.012) or np.all(dist<=.020) and np.all(yaw<=.25):near.append(path)
    if not near:break
    rejected+=1
   rid=f'{e}_core1200_{i:02d}'; split='train' if e in 'AB' else 'dev' if e=='C' else 'test'; kind=types[i%10]; color=colors[i%3]
   other=colors[(i%3+(1 if split=='train' else 2))%3]; lift=lambda c:f'lift_{c}_block_table'
   if kind=='closed_empty':branches=[['open_drawer',lift(color),'place_in_drawer','close_drawer'],['place_in_drawer','open_drawer',lift(colors[(i+1)%3]),'place_in_drawer'],['open_drawer',lift(colors[(i+2)%3]),'close_drawer','place_in_drawer']]
   elif kind=='open_empty':branches=[[lift(color),'place_in_drawer','close_drawer','open_drawer'],['open_drawer',lift(colors[(i+1)%3]),'place_in_drawer','close_drawer'],[lift(colors[(i+2)%3]),'close_drawer','place_in_drawer','close_drawer']]
   else:branches=[['place_in_drawer','close_drawer','open_drawer',lift(other)],['close_drawer','place_in_drawer','close_drawer','open_drawer'],[lift(other),'place_in_drawer',lift(other),'place_in_drawer']]
   root={'root_id':rid,'environment':e,'split':split,'dev_role':'tune' if e=='C' and i<12 else 'calibration' if e=='C' else None,'seed':seed,'initial_state':kind,'held_color':color if kind=='open_held' else None,'layout':{'block_'+c:ordered[j].tolist() for j,c in enumerate(colors)},'branches':branches,'setup_skills':([] if kind=='closed_empty' else ['open_drawer'])+([lift(color)] if kind=='open_held' else [])}
   roots.append(root); excluded.append((e,ordered,rid));audit.append({'root_id':rid,'rejected_layout_candidates':rejected})
 write(B/'roots_manifest.json',roots);(B/'roots_manifest.json').chmod(0o444)
 write(R/'root_exclusion_audit.json',{'external_root_files':len(excluded)-100,'new_roots':100,'rows':audit,'selection_uses_outcomes':False})
 write(B/'data_manifest.json',{'schema':'core1200_v1','records':[],'real_transitions':0,'event_names':['red_moved','blue_moved','pink_moved']})
 write(R/'progress.json',{'stage':'protocol_frozen','transitions':0,'target':1200})
 h=json.loads((P/'runs/continuation_handoff.json').read_text());h.update(stage='core1200_protocol_frozen',active_pipeline='new separately authorized six-skill1200',latest_protocol=str(R/'protocol.json'),latest_progress=str(R/'progress.json'),next='First50 storage and128 training/input isolation checks; collect new roots only.');write(P/'runs/continuation_handoff.json',h)
 print(json.dumps({'roots':len(roots),'planned_transitions':1200,'excluded_root_files':len(excluded)-100,'protocol':str(R/'protocol.json')}))
if __name__=='__main__':main()
