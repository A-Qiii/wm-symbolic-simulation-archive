from config import *
import os,ast,shutil
from omegaconf import OmegaConf
assert not E.exists(),E
for p in ['source','configs','results','logs','data/raw','data/D5/public/features','data/D5/_private']:(E/p).mkdir(parents=True)
R.mkdir(parents=True,exist_ok=False)
for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,E/'source'/p.name)
oldseal=read(ABL/'configs/delivery_manifest.json')
for p,h in oldseal['sha256'].items():assert sha(p)==h,p
assert read(ABL/'results/gate.json')['passed']is False
layouts=[];seeds=set();rids=set();sources={}
def visit(x):
 if isinstance(x,list):
  for v in x:visit(v)
 elif isinstance(x,dict):
  if isinstance(x.get('layout'),dict)and all('block_'+c in x['layout']for c in COLORS):
   arr=np.asarray([x['layout']['block_'+c]for c in COLORS],float)
   if arr.shape==(3,6):layouts.append(arr);seeds.add(x.get('seed'));rids.add(x.get('root_id'))
  for k,v in x.items():
   if k not in ['labels','predictions','results']and isinstance(v,(list,dict)):visit(v)
for root in [P/'data',P/'experiments',P/'runs']:
 for folder,dirs,files in os.walk(root):
  dirs[:]=[d for d in dirs if d not in ['episodes','features','models','__pycache__']and Path(folder)/d!=E]
  for name in files:
   if name.endswith('.json')and'roots'in name:
    p=Path(folder)/name;sources[str(p)]=sha(p);visit(read(p))
prior=len(layouts);scene=P/'vendor/calvin/calvin_env/conf/scene/calvin_scene_D.yaml';cfg=OmegaConf.load(scene);lo,hi=np.array(cfg.surfaces.table);roots=[]
for color in COLORS:
 for rep in range(10):
  seed=16970000+len(roots);rid=f'D5_objective_{color}_{rep:02d}';assert seed not in seeds and rid not in rids;rng=np.random.default_rng(seed)
  for attempt in range(10000):
   arr=np.zeros((3,6));order=rng.permutation(3)
   for slot,j in enumerate(order):arr[j,:3]=[lo[0]+(hi[0]-lo[0])*(.14+.35*slot)+rng.uniform(-.012,.012),rng.uniform(lo[1]+.015,hi[1]-.015),.46];arr[j,5]=rng.uniform(-np.pi,np.pi)
   if not any(np.all(np.linalg.norm(arr[:,:3]-old[:,:3],axis=1)<.012)for old in layouts):break
  else:raise RuntimeError('Unique initialization generation exhausted')
  layouts.append(arr);roots.append({'root_id':rid,'environment':'D','split':'test_D5','seed':seed,'target':color,'replicate':rep,'layout':{'block_'+c:arr[j].tolist()for j,c in enumerate(COLORS)},'drawer_reset_m':0.,'branches':{'normal':['open_drawer',f'lift_{color}_block_table','place_in_drawer','close_drawer']},'generation_attempt':attempt+1})
order=np.random.default_rng(16971000).permutation(30);roots=[roots[i]for i in order]
for i,r in enumerate(roots):r['execution_order']=i
save(E/'configs/roots.json',roots);save(E/'configs/root_exclusion.json',{'previous_declared_layouts':prior,'prior_roots_file_sha256':sources,'new_root_count':30,'unique_seed_root_ids':True,'rule':'at leastone samecolor xyz difference>=12mm versus every prior/newdeclared layout; no outcome/model filtering','sampling':'Same D4 tabletop distribution, one seeded random configuration perroot;10targetroots percolor;fixed shuffled executionorder'})
models=read(ABL/'configs/choices_frozen.json');protocol={'frozen_utc':now(),'study':'Separate independent paired validation of frozen candidates; originalAB/Cgatefailure preserved and no formal promotion or adoption.',
 'models':models['selected'],'query_heads':models['query_heads'],'thresholds':{'binary':.5,'drawer':'argmax'},
 'initialization':{'environment':'sameCALVIN_D','roots':30,'per_target_color':10,'seeds':'16970000..16970029','execution_shuffle_seed':16971000,'distribution':'Same as D4:random object-slot permutation,x slots tablefraction.14/.49/.84 plusuniform±12mm,yuniform tableboundsinset15mm,z.46m,yawuniform[-pi,pi],closed drawer','table_lower':lo.tolist(),'table_upper':hi.tolist(),'reset_settle_controls':127,'new_environment_complexity':False},
 'collection':{'branches':['normal'],'skills':'open_drawer,lift_target_block_table,place_in_drawer,close_drawer','planned_roots':30,'max_invocations':120,'max_controls_per_skill':900,'posthold_controls':96,'internal_attempts_max':2,'external_retries':0,'replacement_roots':0,'stop_per_root':'stop future slots after first nonsuccess, retain observed failure boundary; unreached slots missing','infrastructure':'preserve partialroot; do not replay; stopcollection and report missingroots if infrastructureblocked','soft_wall_minutes':30,'public':'twoRGBcameras,currentrobot15 only','private':'physicalquaternions,contacts,labels,fullframevideoandtrace'},
 'inference':'Same actualtrajectory for4frozenmodels. Trueinitial recursive1..4 withno latent correction; trueprevious one-step; actualsuccessor readout. No networks trained or parameters changed. No oldD selection. No post-test choice of checkpoints or thresholds.',
 'primary':{'metric':'root with all9facts correct at all4skillboundaries','denominator':'complete observed4-boundary roots with valid9labels at every boundary; same eligible rootset allmodels','also_report':'n/30 observedconfirmed-all4correct lowerbound and(n+unscorable)/30 missingness upperbound;counts ofplanned,attempted,any-evaluable,complete,full9-evaluable roots; do not silently treatmissing assemanticwrong','physical_success':'native4call success count separately, never equate predictioncorrectness with physicaltask success'},
 'facts':{'perstage':'perfactTP/FN/FP/TN withvalidlabel masks andn/N;groupfullaccuracy;actual/one/recursive separate','positive_change_recall':'GTbefore0->after1,requiresvalidboth;predictedbefore0->after1 using actualbeforequery forone,previouspredictedquery forrecursive,actualbothforreadout','positive_change_FPR':'predictedonset on valid transitions withoutGTonset / no-GTonset count','also':'after-statehit amongGTonsets andordinary afterstatepositive recall/FPR, so onset-errors can be distinguished from previous-state readout errors'},
 'paired_uncertainty':{'unit':'root, never60independentroots from2seeds','methods':['gain/loss/tie root counts separatelyperseed','delta=n_gain/N-n_loss/N','colorstratifiedpairedrootbootstrappercentile95%CI,10000resamples,seed16972000','same rootdraws across bothqueryseeds;full9eligibleroots only forprimaryCI','95%Wilsoninterval for eachobservedrootproportion;descriptive smallN uncertainty'],'missing':'report coverage andbounds; if no eligible roots in acolor flag reducedscope; no synthetic fill','decision':'descriptive evidence forparent; no automatic gate,adoption,tuning or newcollection'},
 'auxiliary':'rawlatentMSE andfeatureerror only auxiliary. Optional interpolationdiagnostic mustnotdelayprimary and cannot be interpreted as realstates; omitted unless needed.',
 'finite_stop':'Finish afterthis singlefixedtest regardlessoutcome. Preserveoriginalgatefailure. No automatictraining,newDextension,thresholdchange or environmentexpansion.'}
save(E/'configs/protocol.json',protocol)
files=list((E/'source').glob('*.py'))+[E/'configs/protocol.json',E/'configs/roots.json',E/'configs/root_exclusion.json',P/'src/wm_symbolic/physical_oracle.py',P/'src/wm_symbolic/env_adapter.py',P/'scripts/evaluate_physical_oracle.py',REP/'source/labels_v2_1.py',REP/'source/features.py',REP/'source/representation.py',REP/'source/model.py',REP/'source/runtime.py',REP/'data/public/normalization.npz',REP/'models/canonical_matched_seed0/update1000.pt',REP/'models/canonical_matched_seed0/selected.json',REP/'models/query_seed0.pt',ABL/'results/gate.json',scene]+[Path(v['path'])for v in list(models['selected'].values())+list(models['query_heads'].values())]
save(E/'configs/before_collection_frozen.json',{'frozen_utc':now(),'data_seen':False,'original_gate_passed':False,'previous_ablation_sealed_files_verified':len(oldseal['sha256']),'sha256':{str(p):sha(p)for p in files}});print('D5_PROTOCOL_FROZEN roots30 maxcalls120 models4',flush=True)
