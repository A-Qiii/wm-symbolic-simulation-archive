from experiment import *
import numpy as np,os,shutil
from omegaconf import OmegaConf
def near(a,b,pad=.0005):
 d=np.linalg.norm(a[:,:3]-b[:,:3],axis=1);yaw=np.abs((a[:,5]-b[:,5]+np.pi/2)%np.pi-np.pi/2);return bool(np.all(d<=.012+pad)or(np.all(d<=.020+pad)and np.all(yaw<=.25+pad*10)))
def main():
 assert not B.exists()and not(E/'configs/protocol.json').exists()
 assert sha(TRAIN/'delivery_manifest.json')=='daaaa75497eb09be78770ba7f08f5e11796a36cdfaa1f7948eabb04e13ba8018'
 from immutable import verify_sealed
 for d in ['configs','results','logs','figures','videos','data/features','data/public','data/predictions','data/_private']:(E/d).mkdir(parents=True,exist_ok=True)
 save(E/'results/immutable_before.json',{'created_utc':now(),'manifests':verify_sealed()});B.mkdir();(B/'episodes').mkdir();archive=read(P/'experiments/state_predicates_long_pairs_20261014/configs/root_exclusion.json');sources=dict(archive['sources_sha256']);prior=list(archive['prior_layouts']);ids=set();seeds=set()
 for path,h in sources.items():assert sha(path)==h,path
 def visit(x,src):
  if isinstance(x,list):
   for row in x:visit(row,src)
  elif isinstance(x,dict):
   if isinstance(x.get('root_id'),str):ids.add(x['root_id'])
   if isinstance(x.get('seed'),int):seeds.add(x['seed'])
   if isinstance(x.get('layout'),dict)and all('block_'+c in x['layout']for c in COLORS):
    a=np.array([x['layout']['block_'+c]for c in COLORS],float)
    if a.shape==(3,6):prior.append({'layout':a.tolist(),'source':src})
   for key,val in x.items():
    if key not in ['labels','label','predictions','results']and isinstance(val,(dict,list)):visit(val,src)
 current=[]
 for base in ['data','runs','experiments']:
  for root,dirs,files in os.walk(P/base):
   dirs[:]=[d for d in dirs if d not in ['episodes','features','models','figures','videos','__pycache__']and Path(root)/d not in [E,B]]
   for name in files:
    if name.endswith('.json')and'roots'in name:current.append(Path(root)/name)
 for path in sorted(current):sources[str(path)]=sha(path);visit(read(path),str(path))
 reset=list((P/'data/roots').glob('*/*/reset_evaluator_only.npz'))
 for ver in ['v1','v2','v3']:reset+=list((P/f'runs/physical_oracle_{ver}/roots').glob('*/reset_evaluator_only.npz'))
 assert all(str(p)in sources for p in reset),'new reset metadata not covered by immutable exclusion archive'
 unique={np.array(r['layout'],float).tobytes():r for r in prior};prior=list(unique.values());layouts=[np.array(r['layout'])for r in prior];cfg=OmegaConf.load(P/'vendor/calvin/calvin_env/conf/scene/calvin_scene_C.yaml');lo,hi=np.array(cfg.surfaces.table,float);roots=[]
 for c in COLORS:
  other=COLORS[(COLORS.index(c)+1)%3]
  for rep in [0,1]:
   seed=15193000+len(roots);rid=f'C_drawer_factor_{c}_{rep}';assert seed not in seeds and rid not in ids;rng=np.random.default_rng(seed);rejected=0
   while True:
    a=np.zeros((3,6));order=rng.permutation(3)
    for slot,k in enumerate(order):a[k,:3]=[lo[0]+(hi[0]-lo[0])*(.16+.34*slot)+rng.uniform(-.010,.010),rng.uniform(-.135,-.055),.46];a[k,5]=rng.uniform(-np.pi,np.pi)
    if not any(near(a,x)for x in layouts):break
    rejected+=1
   branches=[]
   for j,drawer in enumerate(['closed','open']):branches.append({'branch':j,'episode_id':rid+f'__b{j}','family':'lift','condition':'held_other_target_table_'+drawer,'role':'drawer_factor_C_diagnostic','drawer_reset_m':0. if drawer=='closed'else .22,'preparation_skills':['lift_'+other+'_block_table'],'declared_skill':'lift_'+c+'_block_table','declared_arguments':{'color':c},'expected_held_colour':other,'expected_drawer':drawer,'require_named_target_on_table':True,'expected_inside_colour':None})
   roots.append({'root_id':rid,'environment':'C','split':'dev','dev_role':'causal_diagnosis_not_acceptance','target_color':c,'other_color':other,'replicate':rep,'seed':seed,'layout':{'block_'+cc:a[i].tolist()for i,cc in enumerate(COLORS)},'branches':branches,'execution_order':[0,1]if rep==0 else[1,0],'new_independent_root':True,'rejected_layout_candidates':rejected});layouts.append(a)
 assert len(roots)==6 and all(sum(r['target_color']==c for r in roots)==2 for c in COLORS)
 D=P/'experiments/state_predicates_independent_D_20261015';AUD=P/'experiments/state_predicates_implementation_diagnostic_20261015';oldq=read(D/'configs/protocol.json');models=oldq['models']
 save(E/'configs/root_exclusion.json',{'created_utc':now(),'prior_unique_layouts':len(prior),'sources_sha256':sources,'prior_layouts':prior,'current_root_metadata_files':list(map(str,current)),'current_reset_metadata_files':len(reset),'only_root_geometry_used_for_sampling':True});save(B/'roots_manifest.json',roots)
 files=[Path(f)for f in oldq['frozen_sha256']]+[AUD/'NEXT_STEP_PROPOSAL.json',AUD/'delivery_manifest.json']+list((E/'source').glob('*.py'))
 q={'created_utc':now(),'scope':'Six new C roots, two reset drawer branches each, at most12 lift-target calls while legally holding the same other colour. Single physical factor diagnosis only. No training, newD, D re-inference/rescoring, multistep, upload, loss/architecture change or replacement roots.','models':models,'candidate_hash':sha(TRAIN/'candidate/transition.pt'),'representative':'candidate_seed0','parallel_seeds':'same frozen models recorded only; never selection','root_counts':{'C':6,'per_existing_colour_cycle':2},'maximum_diagnostic_calls':12,'pilot_roots':[roots[0]['root_id']],'pilot_gate':'Technical integrity only, no predictive score gate; both branches count within12.','one_factor':'Reset drawer joint0 or0.22m before any steps. Identical seed, all initial robot/object/other scene state. Same legal lift-other preparation and hold commands; no intermediate teleport. Order counterbalanced three closed-first and three open-first.','matching_tolerances':{'initial_non_drawer_max_abs':1e-5,'object_position_m':.001,'object_orientation_deg':.5,'robot_TCP_position_m':.001,'robot_TCP_orientation_deg':.5,'robot_arm_joint_deg':.5,'gripper_width_m':.001,'other_scene_joints_max_abs':1e-5,'gripper_command':'exact_equal','history':'All four pre-action frames at offsets[-63,-42,-21,0]; independent anchors may have different absolute frame numbers; all seven truth facts valid, first six equal and correct held-other, drawer opposite.','orientation':'Quaternion geodesic angle; joints wrapped to[-pi,pi].','causal_coverage':'A root is strictly covered only if both legal calls are collected and all frozen initial/history pose/fact matching gates pass. A mismatch is retained as uncovered for causal inference; legally feasible calls may still be measured descriptively, never dropped or replaced.'},'symbol_thresholds':[.5]*7,'UNKNOWN_rule':oldq['UNKNOWN_rule'],'input_version':oldq['input_version'],'macro_contract':oldq['macro_contract'],'prediction_timing':oldq['prediction_timing'],'GPU_queue':oldq['GPU_queue'],'fixed_colour_cycles':'heldblue->targetred,heldpink->targetblue,heldred->targetpink; no reverse combinations','comparison':'Report all7 current/after readouts and saved future prediction, heldtarget/heldother probabilities and UNKNOWN, actualrefusal/nochange, open-minus-closed differences byroot. Six roots are independent units;12 calls are not12 roots.','interpretation':['Only matched roots with both sides current and after readouts correct isolate successor mapping. Repeated closed-correct/open-wrong supports drawer context dependence within C; image and latent changes induced by the drawer are the intended treatment.','Both-wrong or both-right does not isolate drawer as the cause of old D failures. Unmatched preparations cannot support strict one-factor causal conclusions.','Describe all counts without posthoc acceptance threshold. A consistent directional pattern across independent roots and colours can motivate a separately authorized, bounded condition-combination training study; it is not authorization to train now.','Do not change latent bounds, losses, architecture or reverse-colour coverage in this diagnosis. Preserve old D failed68/84 and paired20/36.'],'training_updates':0,'D_neural_forward_calls':0,'roots_manifest_sha256':sha(B/'roots_manifest.json'),'exclusion_sha256':sha(E/'configs/root_exclusion.json'),'frozen_sha256':{str(p):sha(p)for p in files}}
 save(E/'configs/protocol.json',q);save(B/'data_manifest.json',{'version':'drawer_factor_C_online_v1','records':[],'windows':[],'private_pairs':[],'episodes':[]});save(E/'results/storage_before.json',{'free_bytes':shutil.disk_usage(B).free});save(E/'results/progress.json',{'phase':'frozen_before_C','roots':6,'planned_calls':12,'completed':0})
 for p in [E/'configs/protocol.json',E/'configs/root_exclusion.json',B/'roots_manifest.json']:p.chmod(0o444)
 print(json.dumps({'protocol_sha256':sha(E/'configs/protocol.json'),'candidate_sha256':q['candidate_hash'],'prior_layouts':len(prior),'source_files':len(sources),'C_roots':6,'max_calls':12,'pilot_root':roots[0]['root_id']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
