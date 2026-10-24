from experiment import *
import numpy as np,os,shutil
from omegaconf import OmegaConf
def near(a,b,pad=.0005):
 d=np.linalg.norm(a[:,:3]-b[:,:3],axis=1);yaw=np.abs((a[:,5]-b[:,5]+np.pi/2)%np.pi-np.pi/2);return bool(np.all(d<=.012+pad)or(np.all(d<=.020+pad)and np.all(yaw<=.25+pad*10)))
def main():
 assert not B.exists()and not(E/'configs/protocol.json').exists()
 assert sha(TRAIN/'delivery_manifest.json')=='daaaa75497eb09be78770ba7f08f5e11796a36cdfaa1f7948eabb04e13ba8018'
 for d in ['configs','results','logs','figures','videos','data/features','data/public','data/predictions','data/_private']:(E/d).mkdir(parents=True,exist_ok=True)
 B.mkdir();(B/'episodes').mkdir();archive=read(P/'experiments/state_predicates_long_pairs_20261014/configs/root_exclusion.json');sources=dict(archive['sources_sha256']);prior=list(archive['prior_layouts']);ids=set();seeds=set()
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
 unique={np.array(r['layout'],float).tobytes():r for r in prior};prior=list(unique.values());layouts=[np.array(r['layout'])for r in prior];cfg=OmegaConf.load(P/'vendor/calvin/calvin_env/conf/scene/calvin_scene_D.yaml');lo,hi=np.array(cfg.surfaces.table,float);roots=[]
 for c in COLORS:
  ci=COLORS.index(c);other=COLORS[(ci+1)%3];lift='lift_'+c+'_block_table';liftother='lift_'+other+'_block_table'
  for drawer in ['closed','open']:
   for rep in [0,1]:
    seed=15183000+len(roots);rid=f'D_independent_{c}_{drawer}_{rep}';assert seed not in seeds and rid not in ids;rng=np.random.default_rng(seed);rejected=0
    while True:
     a=np.zeros((3,6));order=rng.permutation(3)
     for slot,k in enumerate(order):a[k,:3]=[lo[0]+(hi[0]-lo[0])*(.16+.34*slot)+rng.uniform(-.010,.010),rng.uniform(-.135,-.055),.46];a[k,5]=rng.uniform(-np.pi,np.pi)
     if not any(near(a,x)for x in layouts):break
     rejected+=1
    openprep=[]if drawer=='open'else['open_drawer'];closeprep=[]if drawer=='closed'else['close_drawer']
    cases=[('open','empty_closed',closeprep,'open_drawer',None,'closed',False,None,'regular'),('lift','empty_target_table_open',openprep,lift,None,'open',True,None,'regular'),('place','held_open',openprep+[lift],'place_in_drawer',c,'open',False,None,'regular'),('close','empty_open_target_inside',openprep+[lift,'place_in_drawer'],'close_drawer',None,'open',False,c,'regular'),('close','held_open',openprep+[lift],'close_drawer',c,'open',False,None,'state_dependent'),('lift','held_other_target_table_open',openprep+[liftother],lift,other,'open',True,None,'state_dependent'),('place','held_closed',closeprep+[lift],'place_in_drawer',c,'closed',False,None,'state_dependent')]
    branches=[]
    for j,(fam,condition,prep,skill,held,draw,ontable,inside,role)in enumerate(cases):branches.append({'branch':j,'episode_id':rid+f'__b{j}','family':fam,'condition':condition,'role':role,'preparation_skills':prep,'declared_skill':skill,'declared_arguments':{'color':c}if fam=='lift'else{},'expected_held_colour':held,'expected_drawer':draw,'require_named_target_on_table':ontable,'expected_inside_colour':inside})
    roots.append({'root_id':rid,'environment':'D','split':'test','dev_role':None,'family':'mixed','target_color':c,'other_color':other,'initial_drawer':drawer,'drawer_reset_m':.22 if drawer=='open'else 0.,'replicate':rep,'seed':seed,'layout':{'block_'+cc:a[i].tolist()for i,cc in enumerate(COLORS)},'branches':branches,'paired_branches':{'close':[3,4],'lift':[1,5],'place':[2,6]},'new_independent_root':True,'rejected_layout_candidates':rejected});layouts.append(a)
 assert len(roots)==12 and all(sum(r['target_color']==c for r in roots)==4 for c in COLORS)and all(sum(r['initial_drawer']==d for r in roots)==6 for d in ['open','closed'])
 selection=read(TRAIN/'configs/model_selection_frozen.json');cand=read(TRAIN/'candidate/candidate.json');assert cand['run']=='aug132_seed0';models={'candidate_seed0':{'path':str(TRAIN/'candidate/transition.pt'),'sha256':sha(TRAIN/'candidate/transition.pt'),'role':'preselected_representative'}}
 for seed in [1,2]:
  cp=next(r for r in selection['runs']if r['run']==f'aug132_seed{seed}')['selected'];models[f'parallel_seed{seed}']=cp|{'role':'parallel_report_only_no_D_selection'}
 save(E/'configs/root_exclusion.json',{'created_utc':now(),'prior_unique_layouts':len(prior),'sources_sha256':sources,'prior_layouts':prior,'current_root_metadata_files':list(map(str,current)),'current_reset_metadata_files':len(reset),'no_prior_scores_or_labels_used_for_model_or_root_selection':True});save(B/'roots_manifest.json',roots)
 files=[TRAIN/'delivery_manifest.json',TRAIN/'candidate/candidate.json',TRAIN/'candidate/transition.pt',TRAIN/'configs/model_selection_frozen.json',TRAIN/'models/tree_aug132.json',TRAIN/'source/trees.py',TRAIN/'INDEPENDENT_VALIDATION_PROPOSAL.json',CAND/'reader_seed1.pt',CAND/'model.py',CAND/'metrics.py',CAND/'normalization.json',CAND/'INPUT_CONTRACT.json',RGB/'source/cache.py',RGB/'source/registry.py',COPY/'source/diagnostic.py',COPY/'configs/rules_frozen.json',OLD/'source/common.py',OLD/'source/transition.py',P/'scripts/core1200_collect.py',P/'scripts/evaluate_physical_oracle.py',Path(read(CAND/'INPUT_CONTRACT.json')['external_cached_backbone']['path'])/'model.safetensors']+[P/'src/wm_symbolic'/s for s in ['env_adapter.py','physical_oracle.py','core_ownership.py','calibrated_labels.py','state_labels.py']]+[Path(cp['path'])for cp in models.values()]+[E/'source'/name for name in ['experiment.py','prepare.py','worker.py','collect.py','validate_data.py','evaluate.py']]
 q={'created_utc':now(),'scope':'ONE frozen independent D single-step validation only; no training, fixes, replacementroots, multi-step model rollout, adaptive planning, closedloop or upload','models':models,'representative_frozen_before_D':'candidate_seed0','candidate_hash':sha(TRAIN/'candidate/transition.pt'),'symbol_thresholds':[.5]*7,'UNKNOWN_rule':'nonfinite/missing or multiple simultaneously positive held facts under unchanged reader rule; unknown counts wrong in strictall7','input_version':'RGB current-image registry v1 + original frozen DINOsmall + frozen normalization + genuine4frames at[t-63,t-42,t-21,t],2.1s','root_counts':{'D_test':12,'each_colour':4,'each_initial_drawer_position':6,'each_colour_by_initial_drawer':2},'maximum_calls_per_root':7,'maximum_diagnostic_calls':84,'root_unit':'12independent layouts; seven reset branches each are dependent,36family-pair contrasts are grouped within12roots','branch_contract':'Fixed7reset branches perroot, same initial scene/robot. Fourregular success conditions open/lift/place/close and3state-dependent calls. Preparation separately logged, unchanged legal primitives only. Any failed preparation/requiredcondition =>uncovered, no diagnostic forced or replacement. All primitive outcomes/posthold effects retained.','paired_branches':{'close':[3,4],'lift':[1,5],'place':[2,6]},'macro_contract':'unchanged PhysicalOracle.run budget900 then96real hold steps even onrefusal. Initial96+preparation_boundary96 holds outside score. No advancing physics while predictorworker runs.','prediction_timing':'Write action declaration, then blocking frozenworldmodel worker on current public4frames. Save prediction/hash/completionwallns before action_startwallns and frozen controller call. Afterreal96hold, encode actualnext window only for evaluation. No outcome,actualduration,phase,GTcoords,contacts or future available in beforepredict request.','GPU_queue':'One persistent prediction worker owns existinggpu.lock. Executor sends synchronous requests whilephysics paused; rendering/control and DINO/model calls never overlap. Existing Python3.10executor and3.11worldmodel envs unchanged. No other GPU training.','comparators':['parallel_seed1','parallel_seed2','fixed_AB132_depth4_symbolic_tree','frozen_original_AB_copy_effects','current_state_persistence','actual_next_observation_readout_reference'],'no_D_selection':True,'first_root_integrity_check':roots[0]['root_id'],'pilot_roots':[roots[0]['root_id']],'first_root_gate':'Technical timing/physics/hash/storage/coverage recording only, not predictive score or success quota. Samefixedroot counts toward84.','exclusion':'Audit allcurrent roots-metadata and savedreset metadata againstimmutable479layout archive, thenunion latest30longpair+18statecondition roots. Model and thresholds fixed. Samplingrejectall3namedcentres<=12.5mm OR all<=20.5mm and yawmodpi<=.255rad; actualreset checked12mm/20mm+.25rad.','evidence_standard':{'scope':'bounded fixed-controller one-step symbolic evidence only, never fullguide acceptance','coverage':'all84plannedcalls covered and seven endpoint facts valid, otherwise report incompletecoverage explicitly','candidate_symbol_gate':'eachheld/in/open positive recall>=.95,FPR<=.01;overallUNKNOWN<=.05;strictall7>=.95 and rootmacroall7>=.95','state_dependence_gate':'all36root-family pairedfull7 rate>=.95 and eachfamily>=11/12; target-only and full7 both reported','no_comparator_saturation_requirement':'Report all comparisons; no model selection or tuning if another comparator ties or wins','latent':'report MAE to actual frozen successorencoding and fractionof384predictedcoordinates perpair withabs>1; no clipping. No claim that sevenfacts prove multistep quality'},'public_prediction_fields_only':True,'training_updates':0,'root_exclusion_sources':len(sources),'roots_manifest_sha256':sha(B/'roots_manifest.json'),'exclusion_sha256':sha(E/'configs/root_exclusion.json'),'frozen_sha256':{str(p):sha(p)for p in files}}
 save(E/'configs/protocol.json',q);save(B/'data_manifest.json',{'version':'independent_D_online_v1','records':[],'windows':[],'private_pairs':[],'episodes':[]});save(E/'results/storage_before.json',{'free_bytes':shutil.disk_usage(B).free});save(E/'results/progress.json',{'phase':'frozen_before_D','roots':12,'planned_calls':84,'completed':0})
 for p in [E/'configs/protocol.json',E/'configs/root_exclusion.json',B/'roots_manifest.json']:p.chmod(0o444)
 print(json.dumps({'protocol_sha256':sha(E/'configs/protocol.json'),'candidate_sha256':q['candidate_hash'],'prior_layouts':len(prior),'source_files':len(sources),'D_roots':12,'max_calls':84,'pilot_root':roots[0]['root_id']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
