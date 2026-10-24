"""Design-only: parse existing files and generate proposed configurations. No simulator/model imports."""
from pathlib import Path
import ast,collections,datetime,hashlib,itertools,json,os,shutil,xml.etree.ElementTree as ET
import numpy as np
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic')
E=P/'experiments/combination_generalization_20261017'
D5=P/'experiments/independent_objective_D5_20261016'
ABL=P/'experiments/transition_objective_ablation_20261016'
REP=P/'experiments/interaction_latent_repair_20261016'
COLORS=['red','blue','pink'];BRANCHES=['normal','reverse_then_lift','place_then_other_lift']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else v.item()if isinstance(v,np.generic)else str(v))+'\n')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |'for r in rows])
def action(skill):
 fam='lift'if skill.startswith('lift_')else skill.split('_')[0];v=[0.0]*8;v[['open','lift','place','close'].index(fam)]=1.0
 if fam=='lift':v[4+COLORS.index(skill.split('_')[1])]=1.0
 v[7]=1.0;return v
def main():
 assert not E.exists(),E
 preservation={}
 for folder in [D5,ABL]:
  manifest=folder/'configs/delivery_manifest.json';m=read(manifest)
  for p,h in m['sha256'].items():assert sha(p)==h,p
  preservation[str(folder)]={'files_checked':len(m['sha256']),'manifest_sha256':sha(manifest),'mismatches':0}
 assert not read(ABL/'results/gate.json')['passed']
 d5=read(D5/'configs/protocol.json');assets={};assetfiles=[]
 for env in 'ABD':
  sf=P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{env}.yaml';cfg=OmegaConf.load(sf);base=P/'vendor/calvin/calvin_env/data';scale=float(cfg.global_scaling);objects={}
  for color in COLORS:
   urdf=base/cfg.objects.movable_objects['block_'+color].file;xml=ET.parse(urdf).getroot();box=xml.find('.//collision/geometry/box');assert box is not None
   dims=np.array(list(map(float,box.attrib['size'].split())))*scale;objects[color]={'urdf':str(urdf),'sha256':sha(urdf),'collision_box_m':dims.tolist()};assetfiles.append(urdf)
  table_urdf=base/cfg.objects.fixed_objects.table.file;xml=ET.parse(table_urdf).getroot();mesh=(table_urdf.parent/xml.find("link[@name='drawer_link']/collision/geometry/mesh").attrib['filename']).resolve()
  assets[env]={'scene_yaml':str(sf),'scene_sha256':sha(sf),'table_bounds':OmegaConf.to_container(cfg.surfaces.table),'global_scaling':scale,'objects':objects,'drawer_collision_mesh':str(mesh),'drawer_collision_sha256':sha(mesh),'table_urdf':str(table_urdf),'table_urdf_sha256':sha(table_urdf)};assetfiles.extend([sf,table_urdf,mesh])
 assert len({a['drawer_collision_sha256']for a in assets.values()})==1
 # Root manifests across all previous batches, including seed-only reset records.
 layouts=[];seeds=set();ids=set();reset_hashes=set();sources={};opaque=set()
 def visit(x):
  if isinstance(x,list):
   for v in x:visit(v)
  elif isinstance(x,dict):
   if 'root_id'in x:
    ids.add(str(x['root_id']))
    if isinstance(x.get('seed'),int):seeds.add(x['seed'])
    if x.get('reset_sha256'):reset_hashes.add(x['reset_sha256'])
    lay=x.get('layout')
    if isinstance(lay,dict)and all('block_'+c in lay for c in COLORS):
     arr=np.array([lay['block_'+c]for c in COLORS],float)
     if arr.shape==(3,6):layouts.append((str(x['root_id']),arr))
    else:opaque.add(str(x['root_id']))
   for k,v in x.items():
    if k not in ['labels','predictions','results']and isinstance(v,(dict,list)):visit(v)
 for base in [P/'data',P/'experiments',P/'runs']:
  for folder,dirs,files in os.walk(base):
   dirs[:]=[d for d in dirs if d not in ['episodes','features','models','__pycache__','raw']and Path(folder)/d!=E]
   for name in files:
    if name.endswith('.json')and 'roots'in name:
     p=Path(folder)/name;sources[str(p)]=sha(p);visit(read(p))
 nold=len(layouts);roots=[];perms=list(itertools.permutations(BRANCHES))
 for env_index,env in enumerate('ABD'):
  lo,hi=np.asarray(assets[env]['table_bounds']);local=0
  for color in COLORS:
   for rep in range(2):
    seed=17070000+env_index*100+local;rid=f'{env}_combination_{color}_{rep:02d}';assert seed not in seeds and rid not in ids;rng=np.random.default_rng(seed)
    for attempt in range(10000):
     arr=np.zeros((3,6));order=rng.permutation(3)
     for slot,j in enumerate(order):arr[j,:3]=[lo[0]+(hi[0]-lo[0])*(.14+.35*slot)+rng.uniform(-.012,.012),rng.uniform(lo[1]+.015,hi[1]-.015),.46];arr[j,5]=rng.uniform(-np.pi,np.pi)
     if not any(np.all(np.linalg.norm(arr[:,:3]-old[:,:3],axis=1)<.012)for _,old in layouts):break
    else:raise RuntimeError('Unique initial layout generation exhausted')
    nearest=min((float(np.max(np.linalg.norm(arr[:,:3]-old[:,:3],axis=1))),name)for name,old in layouts)
    other=COLORS[(COLORS.index(color)+1)%3]
    branches={'normal':['open_drawer',f'lift_{color}_block_table','place_in_drawer','close_drawer'],'reverse_then_lift':['open_drawer','close_drawer','open_drawer',f'lift_{color}_block_table'],'place_then_other_lift':['open_drawer',f'lift_{color}_block_table','place_in_drawer',f'lift_{other}_block_table']}
    roots.append({'root_id':rid,'environment':env,'split':'evaluation_only_known_training_scene'if env in 'AB'else'evaluation_only_existing_heldout_scene','seed':seed,'target':color,'other_target':other,'branch_actions':{k:[action(t)for t in v]for k,v in branches.items()},'replicate':rep,'layout':{'block_'+c:arr[j].tolist()for j,c in enumerate(COLORS)},'initial_state':'closed_drawer_empty_gripper_all_three_on_table','drawer_reset_m':0.0,'setup_skill_calls':0,'branches':branches,'branch_order':list(perms[local]),'generation_attempt':attempt+1,'nearest_declared_layout_max_samecolor_xyz_distance_m':nearest[0],'nearest_declared_layout_root':nearest[1]});layouts.append((rid,arr));local+=1
 for r in roots:
  assert len(r['branches'])==3 and all(len(x)==4 for x in r['branches'].values())
 order=np.random.default_rng(17071000).permutation(18);roots=[roots[i]for i in order]
 for i,r in enumerate(roots):r['root_execution_order']=i
 # Static feasibility certificates are design expectations, never evaluation labels or model dynamics.
 expected=[]
 for branch in BRANCHES:
  plan=roots[0]['branches'][branch];color=roots[0]['target'];drawer='Closed';held=None;inside=set();ontable=set(COLORS)
  for step,skill in enumerate(plan,1):
   before={'drawer':drawer,'holding':held,'inside':sorted(inside)}
   if skill=='open_drawer':assert held is None and drawer=='Closed';drawer='Open'
   elif skill=='close_drawer':assert held is None and drawer=='Open';drawer='Closed'
   elif skill.startswith('lift_'):
    target=skill.split('_')[1];assert held is None and target in ontable;held=target;ontable.remove(target)
   else:assert skill=='place_in_drawer'and held==color and drawer=='Open';inside.add(color);held=None
   expected.append({'branch':branch,'step':step,'skill':skill.replace(color,'TARGET').replace(COLORS[(COLORS.index(color)+1)%3],'OTHER'),'expected_before':before,'expected_after':{'drawer':drawer,'holding':held,'inside':sorted(inside)},'target_placeholder_color':color,'design_only_not_truth':True})
 pack=P/'experiments/intermediate_data_prediction_20261016/data/transition_train/pairs.json';training_rows=[r for r in read(pack)if r['split']=='train'];bindings=collections.Counter()
 for r in training_rows:
  v=action(r['skill']);assert r['action'][:7]==v[:7] and r['action'][7] in [0.,1.];bindings[(r['skill'],r['action'][7])]+=1
 for root in roots:
  rows=root['branch_actions']['place_then_other_lift'];assert rows[1][4:7]!=rows[3][4:7];assert rows[1][4+COLORS.index(root['target'])]==1 and rows[3][4+COLORS.index(root['other_target'])]==1
  assert tuple(root['branch_actions']['normal'][i]for i in range(3))==tuple(root['branch_actions']['place_then_other_lift'][i]for i in range(3))
 calibration=read(ABL/'results/calibration.json');training={}
 for arm in ['mixed','latent']:
  for seed in [0,1]:
   name=f'{arm}_seed{seed}';tr=read(ABL/'results'/f'train_{name}.json');training[name]={'training_updates_budget':tr['updates'],'single_draws':tr['single_draws'],'multi_starts':tr['multi_starts'],'selected_update':750 if arm=='mixed'else 1000,'max_gradient_norm_before_clip':max(x['gradient_norm_before_clip']for x in tr['batch_log']),'clipped_updates':sum(x['gradient_norm_before_clip']>5 for x in tr['batch_log'])}
 protocol={'status':'PROPOSED_AWAITING_PARENT_REVIEW_AND_EXECUTION_INSTRUCTION','created_utc':now(),'collection_authorized':False,'training_authorized':False,'design_only':True,
  'question':'Does the frozen scaled-latent candidate benefit alternative legal action compositions, including fourth-step true Holding, drawer reversal and target switch, beyond normal open-lift-place-close?',
  'models':d5['models'],'query_heads':d5['query_heads'],'thresholds':d5['thresholds'],'unchanged':'DINO,current-two-RGB encoder,object registry,normalizers,decoder,transition weights,query weights,canonical8D action encoding including final coordinate1 as D5; no model input GT/environment/branch/expected-state metadata.',
  'training_provenance':{'mixed':'F+0.25Z','latent':'alpha*Z','alpha':calibration['alpha'],'calibration':'initial AB-only64 batches incl16multistep; median full parameter gradient norms before clip; no C/D','mixed_grad_median':calibration['median_mixed_parameter_gradient'],'rawZ_grad_median':calibration['median_rawZ_parameter_gradient'],'optimizer':'AdamW lr1e-4 weight_decay1e-4 clip5; identical schedules and total1000update budget','selected_checkpoint':'same C selection rule, mixed750 vs latent1000; freeze existing choices','runs':training,'causal_limit':'Candidate package comparison includes loss composition, fixed gradient scaling, Adam optimization dynamics and C-selected update differences. Not isolated proof of removing F or Z alone; no new scaling ablation.'},
  'branches':{'normal':'open,lift(target_from_table),place_in_drawer,close','reverse_then_lift':'open,close,open,lift(target_from_table)','place_then_other_lift':'open,lift(target_from_table),place_in_drawer,lift(other_from_table); other fixedcycle red->blue->pink->red'},
  'roots':{'environments':['A','B','D'],'per_environment':6,'per_target_color_per_environment':2,'independent_roots':18,'episodes':54,'root_order_seed':17071000,'root_seed_ranges':['17070000..17070005','17070100..17070105','17070200..17070205'],'within_root_branch_order':'all six branch permutations used once perenvironment; eachbranch appears twice in eachorderposition','initial_distribution':'native scene tabletop range; three x slots14%/49%/84% plusuniform±12mm, y nativeboundsinset15mm, z.46m, yawuniform[-pi,pi]; samephysicalinitialcondition for all3branches','reset':'fresh native scene instance and same seed/layout/robot defaults for eachbranch;127settlecontrols; no preparation skill; beforefirstcall require identicalrobot_obs/scene_obs arrays andRGBhashes acrossbranches; namealignedphysicspositions/velocities and sign-equivalentquaternions within1e-8;contacts/supportstate/labels identical. Mismatchretainedandstop,noresetretry. Never copy D scene vector into A/B'},
  'split':'All54episodes and derivedframes/features evaluation-only; no new train or dev. A/B are known training scenes with fresh roots; D is existing heldout scene with fresh roots but a research history including D5. No claim of unseen environment or entirely unseen primitive coverage. Allbranches stay grouped byroot.',
  'budget':{'roots':18,'resets':54,'max_skill_invocations':216,'max_controls_per_skill':900,'posthold_controls':96,'reset_settle_controls':127,'max_control_steps':221994,'internal_attempts_max':2,'external_retries':0,'replacement_roots':0,'soft_wall_minutes':45,'storage_soft_ceiling_GiB':5,'stop':'Stop only currentbranch at first nonsuccess; retain observed failure/refusal and mark remaining slots unreached; next preregisteredbranch starts from independentreset. Infrastructure/initialization mismatch -> preservepartialbranch and stop for report, no replay. Never extend budget.'},
  'truth':{'version_proposal':'v2_1 semantics plus scene-native collision box dimensions, pendingreview; do not edit original labelsource','native_dimensions':assets,'reason':'OriginalSIZES is D-only: Ablue/Apink and Bred/Bblue/Bpink differ. Frozen models may carry earlier training-label limitations; this study does not retrain or relabel oldsealedresults.','holding':'unchangedbilateral force-bearing finger-family contacts and no external body support','inside':'native full oriented collisionbox within native drawer cavity, upward support path, no held; supporting ancestors also fit; frozen2mmmargin and3mmfloor tolerance','drawer':'Closedq<=.02;Openq>=.18;Intermediatebetween; ±.003m thresholdbands invalid','timing':'snapshotactualfullquaternions/contacts/drawer pose after everycall+96hold; failedcall stillscored; no truth from planned sequence/native successbit','uncertainty':'preserve existing validity masks; unknown neverfalse','privacy':'URDFdimensions and fullscenephysics evaluatoronly; no publicfeature/model access'},
  'metrics':{'primary':'full9correct at all4boundaries for eachbranch andseed, perenvironment n/N_completevalid; mixed-vs-latent paired gains/losses/ties on identicalcohort','coverage':'planned/attempted/anyvalid/complete/full9eligible/physicalall4success counts separate perenv+branch; n/6 and(n+unscorable)/6 missingnessbounds; failedbutcomplete included','paths':['actualsuccessorreadout','truepreviousstateone-step','trueinitialstate recursive1..4 withoutcorrection'],'perstage':'eachof9fact TP/FN/FP/TN, full9 andHolding/In/drawergroups; n/N masks; positiveonset recall/FPR andtrueonset-afterstatehit asD5','new_errors':'Report firstall9wrong boundary and cumulativeprefixcorrect; perstage latentwrong/mixedcorrect versus latentcorrect/mixedwrong. Also factwise correctbefore->wrongafter andrepairs, actual/one/recursive separately, bothstatesvalid. Never treatunevaluableaswrong.','critical':'reverse_then_lift h4 targetHoldingrecall andplace_then_other_lift h4 otherHoldingrecall amongactualpositives; bothfirst/secondobjectIn andHolding; percolorHoldingTP/FN/FP/TN andanyHoldingFP onactualfalse; normalh4negativecontrol; drawerclassconfusion andunexpectedInFP. Positiveandnegative coveragecounts explicit; ifnoactualpositiveexamples, limitation notsuccess.','readout_pairing':'For eachqueryseed actualreadout must beidenticalbetweenarms; partitionrecursiveerrors byactualreadoutcorrect/wrong to distinguishtransition/driftfromreadoutfailure.','crossbranch':'normal vsplace_then_other_lift share first3commands fromsamestate. Audit prefix physical/observational equality, but always useactuallyobservedGT. Compareh4conditionalonsharedcompletevalidprefix; ifresetorprefixmismatch retainandreport, do not force/synthesize equality.','uncertainty':'Root is resamplingunit, notbranch,frame,seed. Perbranchpairedbootstrap10000 seed17072000 stratifiedenv×target overeligible roots; samedraws bothseeds. EachperenvN<=6 isdescriptive. Crossbranchdifferences oncommoneligible roots withentire rootbranchvector resampled together; noper-branchindependent pooling. No newpass/failgate.'},
  'future_execution_checks':['obtain explicit parentexecution instruction includingnative-size evaluator amendment','verifythisproposalandallfrozenhashes','implementcollector adaptationandnative-size evaluator innewdirectory only; staticchecks andD-reductionparity beforedata','validateactualreset againstdeclaredlayout andpriorresethashes/initialstates; noresample afteroutcomes; stopifcollision/mismatch','serializecollection,featurecache,inferenceunderexistingGPUlock; nooptimizer','finishonce withfulltablesanddenominator-awareplots; noautomaticadoption,extraexperiment,orpost-testselection'],
  'read_only_findings':'Primitives supported underplannedpreconditions; newcompletecompositions notexecuted inthisdesign turn. Prior normal-chainA6/B6/D12 had closefailures1/0/1; D5had openfailure1 andclosefailure1. Cannot guarantee newcompositionsphysicalsuccess.',
  'prior_gate':'AB/C gate failure preserved; D5 remains independent positiveevidence, noformalmodelpromotion.'}
 for folder in ['configs','source','audit']:(E/folder).mkdir(parents=True,exist_ok=False)
 shutil.copy2(__file__,E/'source/build_protocol.py')
 save(E/'configs/protocol_proposed.json',protocol);save(E/'configs/roots_proposed.json',roots);save(E/'configs/expected_states_design_only.json',expected);save(E/'configs/native_geometry_proposed.json',assets)
 save(E/'audit/initialization_exclusion.json',{'prior_roots_files':sources,'prior_layout_records':nold,'prior_root_ids':len(ids),'prior_seeds':len(seeds),'seed_only_or_without_xyz_root_ids':sorted(opaque),'prior_reset_sha256':sorted(reset_hashes),'new_roots':18,'new_seeds_not_seen':True,'near_duplicate_rule':'reject if all3samecolorXYZdistances<12mm versusanyprior/newdeclaredlayout; applied acrossscenes conservatively','native_reset_still_pending':True,'scope_limit':'RootmetadatawithoutXYZ retainedbyid/seed/resethash. Actualphysicalreset duplicatecheckmust occurbeforefirstcall afterexecutionapproval; no native reset performednow.'})
 history=[]
 for fn in ['collection_ABC.json','collection_D3.json']:
  f=P/'experiments/intermediate_data_prediction_20261016/results'/fn
  for ep in read(f):
   if ep['branch']=='normal'and ep['environment']in 'ABD':history.append({'source':str(f),'root_id':ep['root_id'],'environment':ep['environment'],'slots':[{'skill':s['skill'],'status':s['status']}for s in ep['slots']]})
 save(E/'audit/controller_support.json',{'controller_sha256':sha(P/'src/wm_symbolic/physical_oracle.py'),'legality':'Allproposedcalls areexistingopen,close,tablelift,place; noholdingwhenopening/closing; tableliftalwaysstartswithtargetuntouchedontable','historical_normal_episodes':history,'static_expected_state_certificates':str(E/'configs/expected_states_design_only.json'),'new_sequence_physics_executed':False})
 save(E/'audit/action_binding.json',{'columns':['open','lift','place','close','red','blue','pink','posthold_controls/96'],'training_pairs_checked':len(training_rows),'training_source':str(pack),'training_source_sha256':sha(pack),'training_first7_match_skill_and_color':True,'final_coordinate':'1 for current96posthold asD5; historicaltrainingincludes0and1; neverHoldingtruth','bindings':[{'skill':k[0],'posthold_coordinate':k[1],'count':v}for k,v in sorted(bindings.items())],'proposed18root_target_switches_verified':True,'other_target_rule':'red->blue->pink->red','source_binding':str(REP/'source/common.py'),'model_forward_called':False})
 save(E/'audit/preservation.json',preservation)
 refs={str(p):sha(p)for p in assetfiles+[P/'src/wm_symbolic/physical_oracle.py',P/'src/wm_symbolic/env_adapter.py',REP/'source/labels_v2_1.py',REP/'source/features.py',REP/'source/model.py',REP/'source/runtime.py',REP/'source/common.py',pack,REP/'data/public/normalization.npz',REP/'models/query_seed0.pt',REP/'models/canonical_matched_seed0/update1000.pt',D5/'source/config.py',D5/'configs/encoding_and_scoring_frozen.json',ABL/'source/losses.py',ABL/'source/train.py',ABL/'results/calibration.json',ABL/'results/gate.json']}
 refs.update({v['path']:v['sha256']for v in list(d5['models'].values())+list(d5['query_heads'].values())})
 for p,h in refs.items():assert sha(p)==h,p
 save(E/'configs/frozen_references.json',{'design_time_utc':now(),'sha256':refs,'freeze_scope':'existingweightsqueryencoderthresholdcontroller; proposalnotyetapproved'})
 for env in 'ABD':
  rows=[r for r in roots if r['environment']==env];assert len(rows)==6 and collections.Counter(r['target']for r in rows)==dict.fromkeys(COLORS,2)
  assert len({tuple(r['branch_order'])for r in rows})==6
 assert 18*3*4==216
 oldD=np.array([[.07,.05,.05],[.05,.05,.05],[.1,.05,.05]])*.8
 assert np.allclose(np.array([assets['D']['objects'][c]['collision_box_m']for c in COLORS]),oldD)
 save(E/'audit/design_verification.json',{'passed':True,'utc':now(),'proposed_roots':18,'branches':54,'max_calls':216,'balanced_per_scene_and_color':True,'all6branch_orders_per_scene':True,'new_seed_and_manifest_layout_checks_passed':True,'static_legal_sequences':True,'originalDgeometry_equal':True,'all18target_switch_action_encodings_verified':True,'ABDdrawer_collision_mesh_identical':True,'native_AB_geometry_amendment_requires_review':True,'prior_sealed_files_unchanged':sum(x['files_checked']for x in preservation.values()),'simulator_instantiations':0,'skill_calls':0,'model_forward_calls':0,'training_updates':0,'approval_pending':True})
 # Concise reviewer document; expected states never become supervision.
 md=['本轮只完成协议设计与只读检查，尚未获得采集执行指令。计划使用冻结候选检验动作组合变化后的预测；不训练、不改控制器、不根据D5结果调参。',
 '**三条四步链及预期变化**',
 '所有分支从同一闭屉、空夹爪、三物体在桌面的新配置独立复位。下表为动作成功时的设计预期；c为首目标，d按红→蓝→粉→红预先固定，未列出的Holding/In预期为假，Intermediate预期为假。实际评分始终来自物理观测。',
 table(['组合','第1步','第2步','第3步','第4步','检验内容'],[
 ['正常对照','开：Open','抓c：Holding(c)=真','放：Holding(c)=假，In(c)=真','关：Closed，In(c)保持真','原四步链的同期基线'],
 ['反转后抓取','开：Open','关：Closed','再开：Open','抓桌面c：Holding(c)=真，Open保持真','反转后末步真实持物，检验是否仅偏向未持物'],
 ['放入后换目标','开：Open','抓桌面c：Holding(c)=真','放：Holding(c)=假，In(c)=真','抓桌面d：Holding(d)=真；Holding(c)=假，In(c)=真，In(d)=假，Open保持真','目标切换、原对象关系保持和末步持物敏感度']]),
 '正常链和换目标链前三步相同，第四步分别关屉和抓另一物体；反转链改变前段状态路径。没有持物开屉或抽屉内抓取。源码支持这些前提下的原子技能；这只是静态合法性，尚未执行新完整链。已有正常链A为23/24调用成功、B为24/24、D3为47/48，不能保证新链不失败。',
 '**动作目标绑定**',
 '8维顺序为[开,抓,放,关,红,蓝,粉,保持步数/96]。末维是保持时长，绝不是Holding真值；本轮保持96步故恒为1，与D5相同。放入动作不携带新目标，控制器释放当前唯一持物。抓取每步从实际技能名编码颜色，禁止把root首目标广播到整条链。',
 table(['c→d','第2步抓c','第4步抓d'],[[c+'→'+COLORS[(i+1)%3],str(action('lift_'+c+'_block_table')),str(action('lift_'+COLORS[(i+1)%3]+'_block_table'))]for i,c in enumerate(COLORS)]),
 f'已静态核对原训练包全部{len(training_rows)}条AB训练转移：前7维技能/颜色绑定一致；历史末维含0和1，后者已在原训练中存在。18个拟议root均核对第2/4步目标确实切换。实际执行后还须逐调用比较技能名、控制器目标和动作向量；不能仅凭本配置声称运行正确。',
 '**场景、配置与预算**',
 'A、B、D各6个新root，每色2个；18个独立root，每个复位运行3分支，共54段、最多216次技能调用。使用各场景原生桌面范围：x三个槽位14%/49%/84%加±12mm扰动，y边界内缩15mm，z=0.46m，yaw随机。18个实际布局、种子和固定顺序已写入配置；每场景六种分支顺序各用一次。',
 '每个分支以同场景、同seed、同显式布局和机器人默认状态创建独立原生环境，复位稳定127步后核验：robot_obs/scene_obs数组及双RGB哈希须相同；按物体名称对齐的位置/速度和等价四元数误差≤1e-8，接触支撑状态及有效标签相同。初态必须确为闭屉、空夹爪、三物体在桌面。若不一致，保留记录并停止报告，不重新复位找一致结果，也不把不同初态算成同根配对。原生复位尚未运行，此项是未来执行检查。',
 '每技能最多900控制步，保持96步，内部尝试至多2次；无额外准备技能、外部重试或替换配置。总控制步上限221,994，软时间45分钟、存储5 GiB；到限停止并报告缺失。一个分支首次失败/拒绝后保留末态并停止后续调用，其他预定分支仍独立复位运行；基础设施或初态核验失败则停止整轮并报告。预算不扩张。',
 '全部新数据仅评价。A/B是已知训练场景的新配置，D是既有留出场景的新配置，不能称全新环境泛化。一个root全部分支、帧和特征绑定；三个分支、两个模型种子不增加独立root数。新布局已排除历史声明布局近重复及旧root/seed；无XYZ的旧记录保留种子/复位哈希，实际复位还要核对，不宣称现在已运行过。',
 '**真实标签：需要审查的评分元数据适配**',
 '现有labels_v2_1.py把物体碰撞盒固定为D尺寸；A的blue/pink及B的red/blue/pink实际不同，直接套用会污染In标签。建议在新目录的评价器中按原生URDF读取尺寸×0.8，仅改变评分尺寸元数据，其他规则/容差和模型阈值保持冻结。D尺寸与原脚本一致，A/B/D抽屉碰撞网格SHA256相同。旧封存标签和结果不重写；旧模型可能承接既往标签限制，本轮不改训练。',
 table(['场景','red尺寸 m','blue尺寸 m','pink尺寸 m'],[[env,*[' × '.join(f'{v:.3f}'for v in assets[env]['objects'][c]['collision_box_m'])for c in COLORS]]for env in 'ABD']),
 '每次实际调用后保持96步，取真实物体姿态/四元数、夹指接触、承重接触和抽屉位移评分。Holding为双侧受力夹指接触且无外部支撑；In为完整有向盒位于屉内、有向上承重路径且未持物；Closed≤0.02m、Open≥0.18m，中间状态独立，阈值±0.003m区域保留无效掩码。未知不填假；计划预期、query和native成功位均不作真值。失败后的实际中间态仍评分。',
 '**评价和解释边界**',
 '每场景×分支×种子报告：每步九事实、四步全正确、完整/可评/物理成功数量，以及相对计划6条的缺失界限。真实后继读出、真实前态单步、真实初态递归分开。逐颜色列Holding假阳性/假阴性和召回，首目标与新目标的In、Holding都评价；配对列仅Z新增错误与修复、第一处全9错误及累计前缀正确。保留正变化召回/误报和有效标签分母。',
 '重点看两条新链最后抓取的真实阳性Holding召回及非目标Holding误报，并核对换目标后首物体In仍真、Holding已假。没有真实阳性时报告覆盖不足，不从计划补真值。正常链末步提供Holding阴性对照；不能只靠末步误报下降宣称泛化。实际后继读出在同种子两候选间须逐值相同，并把递归错误按真实读出正确/错误拆开。',
 '18根是统计单位，不是54段或108个独立样本。单分支以两候选共同完整可评root配对；跨分支以双方共同可评root比较。10,000次配对bootstrap（种子17072000）按场景×颜色分层，跨分支重采样整个root结果向量，两种子共用root抽样。每格最多6根，结果只作有限证据，缺失不算事实错误，不设新的自动通过门槛。',
 '冻结D5四检查点：混合F+0.25Z均update0750，仅潜状态αZ均update1000；α=3.341086352737259。α来自初始AB64批梯度范数中位数比0.6334645152/0.1895983666（16批含多步），未用C/D。AdamW lr=1e-4、wd=1e-4、clip=5、总1000步预算一致，原四次训练均未触发裁剪。共同C规则选出的更新数不同；本轮比较包括损失组成、固定梯度缩放、优化轨迹与C选择，不能完全归因于单一损失项。',
 '**审查后才能执行**',
 '请父线程审查具体配置及原生尺寸评分适配，再下达执行指令。本轮未启动仿真、采集、模型前向或训练。D5的2279文件及原消融70文件哈希无变化，原gate失败保留。获得执行指令后，只在新目录实现适配、核验并冻结代码，再运行既定预算；不追加试采样或调参。',
 table(['文件','内容'],[['[协议](configs/protocol_proposed.json)','完整配置，collection_authorized=false'],['[18根配置](configs/roots_proposed.json)','实际布局、种子、分支顺序、逐步8维动作'],['[设计预期](configs/expected_states_design_only.json)','逐步前后预期，禁止充当标签'],['[原生几何](configs/native_geometry_proposed.json)','场景URDF尺寸及资产哈希'],['[冻结引用](configs/frozen_references.json)','四模型、query、编码器、控制器及标签基准哈希'],['[动作绑定审计](audit/action_binding.json)','旧训练与当前目标颜色编码检查'],['[静态核查](audit/design_verification.json)','预算、排重、保留证据和零执行记录'],['[配置排重](audit/initialization_exclusion.json)','历史来源与仍待实际复位核对的范围']])]
 (E/'PROTOCOL.md').write_text('\n\n'.join(md)+'\n')
 files=sorted(p for p in E.rglob('*')if p.is_file());save(E/'configs/design_manifest.json',{'utc':now(),'status':'design_sealed_for_review_not_execution_authorization','sha256':{str(p):sha(p)for p in files}})
 print('DESIGN_SAVED',E,'roots',len(roots),'maxcalls',216,'old_layouts',nold,'prior_sealed_files',sum(x['files_checked']for x in preservation.values()),flush=True)
if __name__=='__main__':main()
