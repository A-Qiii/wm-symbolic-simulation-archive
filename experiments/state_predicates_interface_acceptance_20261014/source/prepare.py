from pathlib import Path
import json,hashlib,datetime
import numpy as np
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_interface_acceptance_20261014';B=P/'data/predicate_interface_acceptance_20261014';PRE=P/'experiments/state_predicates_held_down_20261014';COLORS=['red','blue','pink']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not (E/'configs/protocol.json').exists() and not B.exists()
 for d in ['configs','source','data/features','results','figures','videos','logs']:(E/d).mkdir(parents=True,exist_ok=True)
 B.mkdir();(B/'episodes').mkdir()
 oldids=set();oldseeds=set();layouts=[];sources={};configs={e:OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml')for e in 'ABCD'}
 def visit(x,source):
  if isinstance(x,list):
   for q in x:visit(q,source)
  elif isinstance(x,dict):
   if isinstance(x.get('root_id'),str):oldids.add(x['root_id'])
   if isinstance(x.get('seed'),int):oldseeds.add(x['seed'])
   if isinstance(x.get('layout'),dict)and all('block_'+c in x['layout']for c in COLORS):
    v=np.array([x['layout']['block_'+c]for c in COLORS],float)
    if v.shape==(3,6):layouts.append((v,source))
   for v in x.values():
    if isinstance(v,(dict,list)):visit(v,source)
 for path in sorted(P.rglob('*roots*.json')):
  if 'vendor' in path.parts or E in path.parents or B in path.parents:continue
  sources[str(path)]=sha(path);visit(read(path),str(path))
 resetfiles=list((P/'data/roots').glob('*/*/reset_evaluator_only.npz'))
 for v in ['v1','v2','v3']:resetfiles+=list((P/f'runs/physical_oracle_{v}/roots').glob('*/reset_evaluator_only.npz'))
 for path in sorted(set(resetfiles)):
  env=path.parent.name[0]
  if env not in configs:continue
  with np.load(path)as z:poses=z['scene_obs'][6:].reshape(3,6)
  names=list(configs[env].objects.movable_objects);lookup=dict(zip(names,poses));layouts.append((np.stack([lookup['block_'+c]for c in COLORS]),str(path)));sources[str(path)]=sha(path);oldids.add(path.parent.name)
 unique={v.tobytes():(v,p)for v,p in layouts};layouts=list(unique.values());before=len(layouts);roots=[];audit=[]
 for j,c in enumerate(COLORS):
  for drawer in ['closed','open']:
   for rep in range(2):
    i=len(roots);seed=14142600+i;assert seed not in oldseeds;rng=np.random.default_rng(seed);rejects=0
    while True:
     v=np.zeros((3,6));order=rng.permutation(3)
     for slot,k in enumerate(order):v[k,:3]=[.35*(.16+.34*slot)+rng.uniform(-.014,.014),rng.uniform(-.14,-.05),.46];v[k,5]=rng.uniform(-np.pi,np.pi)
     distances=[];near=False
     for ov,src in layouts:
      d=np.linalg.norm(v[:,:3]-ov[:,:3],axis=1);yaw=np.abs((v[:,5]-ov[:,5]+np.pi/2)%np.pi-np.pi/2);distances.append(float(d.max()))
      if np.all(d<=.012)or(np.all(d<=.020)and np.all(yaw<=.25)):near=True;break
     if not near:break
     rejects+=1
    rid=f'D_interface_accept_{c}_{drawer}_{rep}';assert rid not in oldids
    r={'root_id':rid,'environment':'D','split':'independent_acceptance','seed':seed,'target_color':c,'drawer_setting':drawer,'drawer_reset_m':0.0 if drawer=='closed'else .22,'replicate':rep,'layout':{'block_'+co:v[k].tolist()for k,co in enumerate(COLORS)},'initialization':'Native D configuration with new table layout and declared initial drawer state; one initial reset only.','requests':['baseline_hold_96','lift_'+c+'_block_table','held_hold_96']+(['place_in_drawer','in_open_hold_96','close_drawer','in_closed_hold_96']if drawer=='open'else[])}
    roots.append(r);layouts.append((v,rid));audit.append({'root_id':rid,'layout_candidates_rejected':rejects,'minimum_max_named_object_distance_m_to_any_prior_layout':min(distances),'new_seed_and_id':True})
 old=read(PRE/'candidate/candidate_manifest.json');contract=read(PRE/'candidate/INPUT_CONTRACT.json');assert old['model']['seed']==1 and old['model']['thresholds']==[.5,.5,.5]
 fixed=[PRE/'candidate/candidate_manifest.json',PRE/'candidate/INPUT_CONTRACT.json']+[PRE/'candidate'/a['filename']for a in old['assets']]+[Path(contract['external_cached_backbone']['path'])/'model.safetensors']+[P/'src/wm_symbolic'/name for name in ['env_adapter.py','physical_oracle.py','core_ownership.py','calibrated_labels.py','vision_cache.py']]+[P/'scripts/core1200_collect.py',P/'scripts/evaluate_physical_oracle.py',E/'source/collect.py']
 frozen={str(p):sha(p)for p in fixed}
 conditions={
 'baseline':'Last frame after exactly 96 open-gripper motor holds from initial reset; no history padding.',
 'near_open':'Last descend_named_block frame with gripper command +1 during lift request, regardless of predicted or actual success.',
 'near_closed':'First descend_named_block frame with gripper command -1 during lift request. Contact/support truth is audited, not assumed.',
 'held_after_lift':'Final frame of requested lift, including failures. Condition achieved only if actual held at anchor.',
 'held_stationary':'Final frame after 96 unchanged closed-gripper hold actions following successful lift. Condition achieved iff all64 consecutive frames held and both TCP/object stay within 1mm of first position. No replacement window.',
 'held_downward':'Open roots only: anchor at first lower_into_drawer phase frame +63, provided the original phase lasts at least64 frames. Condition achieved iff all64 true-held, TCP and object net drop>=20mm, each consecutive z increment<=0.02mm AND total positive z increments<=0.10mm for each. Numerical tolerance disclosed. No window shifting; failed construction remains recorded.',
 'in_open':'Open roots only: final frame of 96 open-gripper holds after successful place; actual target in and drawer open required for intended condition.',
 'in_closed':'Open roots only: final frame of 96 open-gripper holds after successful close; actual target in and drawer closed required.',
 'natural_one_camera_occlusion':'Earliest t>=63 where target is >=4pixels in other camera at t, and for one camera current pixels<4 but max at the four input times>=20. Score regardless of current held truth; no synthetic occluders.',
 'natural_held_occlusion':'Same visibility criterion as natural_one_camera_occlusion, earliest eligible current true-held frame. Private truth selects this preregistered diagnostic stratum only, not inputs.',
 'natural_both_camera_occlusion':'Earliest t>=63 with target<4pixels in both cameras at t and max at any of the four input times>=20; score any physical truth. No synthetic occluders.'}
 protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'One finite independent state-interface acceptance of frozen seed1. No training, recalibration, repair, model switching, new root replacement, WM or planning execution. Local only.','roots':12,'root_balance':'3 target colours x2 declared drawer initial states x2 independent layouts. Root is statistical cluster, not each window or model output.','root_exclusion':'New ID and seed; near-duplicate rejection against all archived train/dev/test/controller root layouts across environments and among new roots: all3 named-object distances<=12mm, or all<=20mm and all yaw modulo pi differences<=0.25rad.','candidate':old['model'],'candidate_contract':contract,'conditions':conditions,'control':'Unmodified PhysicalOracle.run with existing internal bounded retry; extra holds use unmodified PhysicalOracle.hold(96, existing intended grip). No speed, gains, control limits or poses altered. Initial open drawer is a declared root state, not an intermediate completion teleport. Stop dependent place/close/holds on control failure; retain attempt and missing-condition reasons.','input_processing':{'history_offsets':[63,42,21,0],'fps':30,'span_seconds':2.1,'startup':'No prediction before t>=63; no padding.','registry':'Fresh VisibleRegistry per complete4-frame window, unchanged RGB colour registration and 2pixel ROI visibility rule. Missing entity stays existing zero feature; no GT identity substitution.','model':'Same frozen DINO weights, preprocessing, normalization, reader and thresholds. Only objects/context/robot tensors passed.','occlusion':'No new visibility-based abstention or label masking. Existing held-conflict UNKNOWN decoder only; report registered entity availability separately.','ground_truth':'Private labels at current t only; ambiguous physical category excluded per existing categorical_valid and counted. Visibility is not a ground-truth validity criterion. Future frames excluded.'},'denominators':{'windows':'One preregistered anchor per root-condition; report attempted roots, available windows, achieved condition, control/missing-history gaps, GT-invalid facts. Main condition table includes every anchored attempt even if intended physical condition failed; achieved-only table separately labelled.','facts':'All7 held(red/blue/pink), in(red/blue/pink,drawer), open(drawer). Current GT-valid facts scored; unavailable public entity still scored. Raw threshold and actual conflict-decoded known output both retained.','per_condition_root':'Root counts with >=1 FP, FN, UNKNOWN, invalid fact; denominator distinct roots with an available window, plus planned root count. Target truth-positive/negative root denominators separately reported.','all_facts':'Window passes only if all7 truth-valid and known-correct after conflict decoding. Root passes only if every available mandatory window passes and every planned mandatory condition is achieved. Missing/failed construction never counted as pass.','pooled':'Distinct roots cluster multiple windows; never label pooled window count independent. Natural occlusion anchors may duplicate other anchors; retain condition membership and report unique root-anchor count.'},'decision_rule':'Practical capability report, not 1% statistical certification. Recommend unqualified next-interface integration only when required physical coverage exists for each colour and no consequential known false assertions or recurring missed positive states appear. Localized visibility/condition gaps permit at most explicitly scoped research suggestions; list them without retuning. Do not claim general reliability from zero errors in a small sample. WM integration recommendation only; do not start WM.','failure_analysis':'First consequential error per root in chronological order, then group existing evidence by cause. No resampling, training, threshold adjustment, repeated acceptance or single-case repair on this set.','old_vs_new_sampling':'All acceptance windows use2.1s. Older973 rehearsal training windows and prior historical C used original short histories; acceptance does not retroactively change them.','frozen_sha256':frozen,'new_roots_sha256':None,'no_upload':True}
 save(B/'roots_manifest.json',roots);protocol['new_roots_sha256']=sha(B/'roots_manifest.json');save(E/'configs/protocol.json',protocol);save(E/'configs/root_exclusion_audit.json',{'before_collection':True,'prior_unique_layouts':before,'prior_root_ids':sorted(oldids),'prior_seeds':sorted(oldseeds),'source_sha256':sources,'rows':audit,'outcomes_or_predictions_used':False});save(B/'data_manifest.json',{'version':'independent_interface_acceptance_v1','records':[],'roots':[]});save(E/'results/progress.json',{'stage':'protocol_and_roots_frozen','collected_roots':0})
 for p in [B/'roots_manifest.json',E/'configs/protocol.json',E/'configs/root_exclusion_audit.json']:p.chmod(0o444)
 print(json.dumps({'frozen_roots':len(roots),'prior_unique_layouts_excluded':before,'prior_root_metadata_sources':len(sources),'protocol':str(E/'configs/protocol.json')},ensure_ascii=False))
if __name__=='__main__':main()
