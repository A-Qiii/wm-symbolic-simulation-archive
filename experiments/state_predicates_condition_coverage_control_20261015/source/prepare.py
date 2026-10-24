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
 unique={np.array(r['layout'],float).tobytes():r for r in prior};prior=list(unique.values());layouts=[np.array(r['layout'])for r in prior];roots=[]
 for c in COLORS:
  other=COLORS[(COLORS.index(c)+1)%3]
  for rep,env in enumerate(['A','B']):
   cfg=OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{env}.yaml');lo,hi=np.array(cfg.surfaces.table,float)
   seed=15203000+len(roots);rid=f'{env}_coverage_{c}_{rep}';assert seed not in seeds and rid not in ids;rng=np.random.default_rng(seed);rejected=0
   while True:
    a=np.zeros((3,6));order=rng.permutation(3)
    for slot,k in enumerate(order):a[k,:3]=[lo[0]+(hi[0]-lo[0])*(.16+.34*slot)+rng.uniform(-.010,.010),rng.uniform(-.135,-.055),.46];a[k,5]=rng.uniform(-np.pi,np.pi)
    if not any(near(a,x)for x in layouts):break
    rejected+=1
   branches=[]
   for j,drawer in enumerate(['closed','open']):branches.append({'branch':j,'episode_id':rid+f'__b{j}','family':'lift','condition':'held_other_target_table_'+drawer,'role':'paired_condition_coverage_training','drawer_reset_m':0. if drawer=='closed'else .22,'preparation_skills':['lift_'+other+'_block_table'],'declared_skill':'lift_'+c+'_block_table','declared_arguments':{'color':c},'expected_held_colour':other,'expected_drawer':drawer,'require_named_target_on_table':True,'expected_inside_colour':None})
   roots.append({'root_id':rid,'environment':env,'split':'train','dev_role':None,'target_color':c,'other_color':other,'replicate':rep,'seed':seed,'layout':{'block_'+cc:a[i].tolist()for i,cc in enumerate(COLORS)},'branches':branches,'execution_order':[0,1]if rep==0 else[1,0],'new_independent_root':True,'rejected_layout_candidates':rejected});layouts.append(a)
 assert len(roots)==6 and all(sum(r['target_color']==c for r in roots)==2 for c in COLORS)
 C=P/'experiments/state_predicates_drawer_factor_C_20261015';cq=read(C/'configs/protocol.json');tq=read(TRAIN/'configs/protocol.json');models=cq['models'];save(E/'configs/root_exclusion.json',{'created_utc':now(),'prior_unique_layouts':len(prior),'sources_sha256':sources,'prior_layouts':prior,'current_root_metadata_files':list(map(str,current)),'current_reset_metadata_files':len(reset),'only_root_geometry_used_for_sampling':True});save(B/'roots_manifest.json',roots)
 for d in ['runs','models','data/training/public','data/training/_private']:(E/d).mkdir(parents=True,exist_ok=True)
 files=[Path(f)for f in cq['frozen_sha256']]+[C/'delivery_manifest.json',C/'NEXT_STEP_PROPOSAL.json',C/'results/call_details.json',C/'results/pose_matching.json',P/'data/predicate_drawer_factor_C_20261015/data_manifest.json',TRAIN/'data/public/inputs.npz',TRAIN/'data/_private/targets.npz',TRAIN/'data/public/pairs.json',TRAIN/'data/_private/evaluation_metadata.json',TRAIN/'results/predictions.npz',TRAIN/'results/latent_predictions.npz',TRAIN/'source/train.py',TRAIN/'configs/protocol.json',TRAIN/'configs/model_selection_frozen.json']+list((E/'source').glob('*.py'))
 q={k:cq[k]for k in ['models','candidate_hash','matching_tolerances','symbol_thresholds','UNKNOWN_rule','input_version','macro_contract','prediction_timing','GPU_queue','fixed_colour_cycles']}
 q.update(created_utc=now(),scope='One bounded matched condition-coverage training experiment. Six new A/B roots,12paired calls maximum; no newC/D, no originalD model inference or rescoring, no multi-step/loop/upload. No additional training after fixed evaluation.',root_counts={'A':3,'B':3,'each_existing_colour_cycle':2},maximum_diagnostic_calls=12,pilot_roots=[roots[0]['root_id']],pilot_gate='technical integrity only; firstroot counts within12; no outcome/accuracy-dependent resampling',one_factor='Each root uses same seeds/initial robot+objects and legal lift-other preparation; resetdrawer0/0.22 only. Paired motor rules/history/posthold unchanged. Training contrast replaces6newclosed bycorresponding6newopen,allother132examples identical.',collection_failure_policy='Retain every attempt and readout error. No replacement root. Exact138-per-arm design requires6completed strictly-matched pairs with valid target supervision; otherwise stop before training and report planned design incomplete, do not shrink budgets or select successes.',cohorts=['closed138','open138'],seeds=[0,1,2],updates_each=1500,total_optimizer_update_cap=9000,batch_size=16,example_presentations_each=24000,sampling='138local slots:0..131 commonoriginalAB132 in original order,132..137 newroots in frozen order, closed/open paired at identical index. For each seed, precompute one1500x16 complete-permutation stream with torch.Generator seed10000+seed; share exact indexfile acrossbotharms; no droppedtail or outcome sampling.',class_weights_common_to_both_cohorts=tq['class_weights_common_to_both_cohorts'],loss=tq['loss'],optimizer=tq['optimizer'],architecture=tq['architecture'],freeze=tq['freeze'],initialization='Fresh original initialization perseed; verify digest matches botharms and originalaug132 same-seed initialization. Do not finetune or addupdates to oldmodel.',selection={'checkpoints_saved':[300,600,900,1200,1500],'comparison_checkpoint':'fixed final1500 only','representative_per_arm':'seed0 fixed beforedata; all3seeds reported, no Cbased checkpoint/seedselection','C_usage':'Evaluation after all9000updates complete only. No tuning, earlystop, extension or selection fromC results.'},development_sets={'C_old':{'pairs':24,'roots':6,'source':'originaloldC success-chain','independent_test':False},'C_state':{'pairs':18,'roots':6,'source':'existingstate-dependentC','independent_test':False},'C_drawer':{'pairs':12,'roots':6,'source':'previousdrawer-factor C usedforproblem localization','independent_test':False},'combined':{'pairs':54,'roots':18,'aggregation':'separate groups first, then pooled and rootmacro; never independent generalization'}},development_criteria={'condition_effect':'All3same-seed open138 arms must improve C_drawer both-sides-full7 rootcount overclosed138; eachseed Cold andCstate all7 must not decrease vsclosed138 ororiginal132background. Report absolute andpaired differences even if thisconservative gate fails.','sufficient_for_future_independent_test':'All3open138 final1500 models must reach Cold24/24,Cstate18/18,Cdrawer12/12 and both Cstate/Cdrawer pairedfull7 6/6, with condition_effect true. Thisis a stringent developmentscreen, not guideacceptance.','readout':'Separatecurrent-readout errors, after-readout errors and both-correct cases; no dropping errors frommain denominator.','latent':'MAE objects/scene/all384 and rawSmoothL1 torealobservednext, fractioncoordinatesabs>1,>1.1,>2 andmaxabs. No clipping; symbolgain neverestablishes latentrepair.','failure':'Retainfirstimpactfulerror ofpredeclaredopen138seed0, oneoffline actual-next-latent replacement withsamehead fromsavedreference only. No corrective training.'},original132_background='Existing aug132seed0/1/2 final1500 cached predictions andlatents reused fororiginalC42 andpriorCdrawer12; not independent equal-N causalcontrast, never labelmoreupdates asdatagain.',no_new_D_or_C=True,no_D_model_forward=True,roots_manifest_sha256=sha(B/'roots_manifest.json'),exclusion_sha256=sha(E/'configs/root_exclusion.json'),frozen_sha256={str(p):sha(p)for p in files})
 save(E/'configs/protocol.json',q);save(B/'data_manifest.json',{'version':'condition_coverage_training_paired_online_v1','records':[],'windows':[],'private_pairs':[],'episodes':[]});save(E/'results/storage_before.json',{'free_bytes':shutil.disk_usage(B).free});save(E/'results/progress.json',{'phase':'frozen_before_AB','roots':6,'planned_calls':12,'completed':0})
 for p in [E/'configs/protocol.json',E/'configs/root_exclusion.json',B/'roots_manifest.json']:p.chmod(0o444)
 print(json.dumps({'protocol_sha256':sha(E/'configs/protocol.json'),'prior_layouts':len(prior),'source_files':len(sources),'AB_roots':6,'max_calls':12,'cohorts':['closed138','open138'],'updates_per_run':1500,'seeds':[0,1,2],'pilot_root':roots[0]['root_id']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
