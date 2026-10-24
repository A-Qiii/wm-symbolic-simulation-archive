"""Declare a reviewable future protocol and unseen roots; zero collection/training."""
from audit import *
import os,datetime
from omegaconf import OmegaConf
COLORS=['red','blue','pink']
def main():
 cfgdir=E/'configs';cfgdir.mkdir(exist_ok=True);layouts=[];seeds=set();rootids=set();sources={}
 def visit(x):
  if isinstance(x,list):
   for v in x:visit(v)
  elif isinstance(x,dict):
   if isinstance(x.get('layout'),dict)and all('block_'+c in x['layout']for c in COLORS):
    layouts.append(np.array([x['layout']['block_'+c]for c in COLORS]));seeds.add(x.get('seed'));rootids.add(x.get('root_id'))
   for k,v in x.items():
    if k not in['labels','results','predictions']and isinstance(v,(dict,list)):visit(v)
 for base in[P/'data',P/'experiments',P/'runs']:
  for path,dirs,files in os.walk(base):
   dirs[:]=[d for d in dirs if d not in['episodes','features','models','__pycache__']and Path(path)/d!=E]
   for name in files:
    if name.endswith('.json')and'roots'in name:
     f=Path(path)/name;sources[str(f)]=sha(f);visit(read(f))
 scene=P/'vendor/calvin/calvin_env/conf/scene/calvin_scene_D.yaml';cfg=OmegaConf.load(scene);lo,hi=np.array(cfg.surfaces.table);prior=len(layouts);roots=[]
 for color in COLORS:
  for rep in range(4):
   seed=16961000+len(roots);rid=f'D2_place_repair_{color}_{rep}';assert seed not in seeds and rid not in rootids;rng=np.random.default_rng(seed)
   for attempt in range(10000):
    arr=np.zeros((3,6));order=rng.permutation(3)
    for slot,j in enumerate(order):arr[j,:3]=[lo[0]+(hi[0]-lo[0])*(.14+.35*slot)+rng.uniform(-.012,.012),rng.uniform(lo[1]+.015,hi[1]-.015),.46];arr[j,5]=rng.uniform(-np.pi,np.pi)
    if not any(np.all(np.linalg.norm(arr[:,:3]-old[:,:3],axis=1)<.012)for old in layouts):break
   else:raise RuntimeError('No unique layout found; no implicit relaxation')
   layouts.append(arr);roots.append({'root_id':rid,'environment':'D','split':'future_test_D2','seed':seed,'target':color,'replicate':rep,'drawer_reset_m':.22 if rep%2 else 0.,'layout':{'block_'+c:arr[j].tolist()for j,c in enumerate(COLORS)},'place_branches':['held_open','held_closed','empty_open','empty_closed'],'normal_reference_skills':([]if rep%2 else['open_drawer'])+[f'lift_{color}_block_table','place_in_drawer','close_drawer'],'closed_loop':{'arms':['canonical_control','canonical_matched'],'seed':0,'max_calls':6},'collected':False})
 save(cfgdir/'proposed_D2_roots.json',roots);save(cfgdir/'proposed_D2_exclusion.json',{'source_sha256':sources,'scene_sha256':sha(scene),'prior_layout_declarations':prior,'new_roots':12,'unique_seed_range':[16961000,16961011],'root_id_or_seed_collision':False,'minimum_separation_rule':'For each previous/new root, at least one same-colour block has initial xyz distance >=12mm. This is provenance screening, not proof of statistical independence.','observations_read_for_generation':False,'new_physical_calls':0})
 extra=read(E/'results/extra_matched_place.json');protocol={
 'status':'frozen proposal for independent acceptance; not approved execution and not an implemented trainer',
 'scope':'One controlled transition-only repair experiment in a new version. Does not complete or change original HULC guide criteria.',
 'rationale':'Old place intent target was not forwarded to executor; missing held_closed place rejection coverage exists in unused AB/C real data; current readout also has held_closed errors.',
 'action_contract':{'place':'place_in_drawer() without object parameter; action target bits 4:7 all zero for every place sample and search option','lift':'keep three named colour actions and their object target bits','goal':'In(requested colour) AND Closed unchanged','options_order':['open_drawer','close_drawer','lift_red_block_table','lift_blue_block_table','lift_pink_block_table','place_in_drawer'],'posthold':'preserve recorded 0/96 bit for old data; extra matched and online96; no GT-fill targets'},
 'seeds':[0,1],'initialization':'Each arm starts from the identical corresponding v1 interaction selected model and query head. Preserve entire original files read-only.',
 'freeze':['DINO','AB normalization and PCA','RGB object registry','object and scene encoders','all decoders','shared semantic query head and thresholds'],
 'trainable':'Only shared object/scene transition parameters; no new symbolic labels or feasibility classifier.',
 'arms':{'canonical_control':'Original AB888 pairs; auxiliary samples from original AB place pairs.','canonical_matched':'Same original AB888 pairs plus 12 pre-existing matched place pairs on4 additional AB roots; auxiliary samples only from these12. Common canonical interface in both arms.'},
 'additional_AB_pair_ids':[r['id']for r in extra if r['split']=='train'],
 'additional_C_pair_ids':[r['id']for r in extra if r['split']=='dev'],
 'optimizer':{'type':'AdamW','lr':0.0001,'weight_decay':0.0001,'clip_grad_norm':5,'updates':1000,'checkpoints':[0,250,500,750,1000]},
 'sampler':{'batch':64,'common_base':48,'auxiliary_place':16,'base':'Precompute same root-uniform AB indices and observed chains for each seed and both arms.','control_aux':'Root-uniform original AB place, then uniform pair in root.','matched_aux':'Uniform among4 AB roots, then among its3 collected branches; no condition values enter input/loss.','rng_seeds':[16962000,16962001],'save':'Persist complete pair-index/chain schedule before first optimizer update. No D/GT-dependent sampling.'},
 'loss':'Reuse v1 equal4-group changed-weighted after-observation loss plus0.25 stopgrad real-successor latent MSE. Every4 updates same common-base observed2/3-step terms at0.25 each. Current reconstruction and variance omitted as constants under frozen representation. No zero-delta target imposed on refusals; use recorded actual after including96 hold.',
 'selection':'Per arm/seed choose minimum0.5*(old C35-root-macro changed-weighted observation MSE)+0.5*(additional C2-root-macro same MSE), checkpoints0/250/500/750/1000; earlier checkpoint breaks exact tie. Never use D1, D2, semantic scores or closed-loop success for selection.',
 'development_reporting':'Report both terms, every condition, all current/actual-successor/predicted-successor semantic denominators, normal-call regressions and refusal false-positive rate. Current readout is frozen and known imperfect; no hidden head calibration.',
 'test':{'root_manifest':str(cfgdir/'proposed_D2_roots.json'),'root_count':12,'normal_reference_attempts_if_all_skills_reached':42,'matched_place_attempts':48,'preparation':'Each branch reset to same declared root. Real existing PhysicalOracle preparation; GT can verify/score preparation but never enter model or planner. Keep every failed preparation and root denominator, no replacement root. Preparation calls logged separately.','offline':'Both seeds on same stored true before/after windows; all90 planned calls stratified into success/refused/failed/unreached with exact denominators.','closed_loop':'Both arms seed0 only;12 matched roots each; max6 calls/rollout,24 rollouts,at most144 task calls. No stored reference sequence used by search.','planner':'Same beam32,horizon4,threshold0.8,reobserve each call,max6; options differ only by canonical place. Record selected action and actual native contract; no hard-coded preconditions/effects.','no_test_repair':'D1 used historical evidence only. D2 collect once after accepted implementation, no tuning/reselection/extra seeds/failure-specific retries after opening. Root-level paired reporting; do not treat branches/windows as independent.'},
 'acceptance_boundary':'Independent protocol acceptance and code/data/interface audit precede any optimizer update or D2 collection. No numerical original-guide threshold is changed. Report a failed repair as a failure; no guaranteed improvement or complete-project claim.',
 'known_limits':['Freezing encoder/head cannot repair current-state HeldClosed query errors.','The matched AB coverage adds only4 independent roots; C adds2 and lacks red.','This compares extra coverage under a common corrected interface; it does not isolate interface correction against v1 on the same new test.','Closed-loop seed1 is not scheduled; no claim of across-seed closed-loop reproducibility.'],
 'current_execution':{'training_updates':0,'new_physical_calls':0,'new_test_observations':0}}
 save(cfgdir/'proposed_protocol.json',protocol)
 extra_sources={str(P/'data/predicate_state_dependence_20261014/data_manifest.json'):sha(P/'data/predicate_state_dependence_20261014/data_manifest.json')}
 for r in extra:
  for key in['path','after_path']:extra_sources[r[key]]=sha(r[key])
  prep=P/'data/predicate_state_dependence_20261014/episodes'/r['episode_id']/'_private/preparation_condition.json';extra_sources[str(prep)]=sha(prep)
 save(cfgdir/'matched_source_freeze.json',{'sha256':extra_sources,'role':'source integrity only; private preparation remains evaluator-only, not model input'})
 freeze=[cfgdir/'proposed_protocol.json',cfgdir/'proposed_D2_roots.json',cfgdir/'proposed_D2_exclusion.json',cfgdir/'matched_source_freeze.json',E/'source/proposal.py',E/'results/extra_matched_place.json',OLD/'configs/final_before_D.json']
 save(cfgdir/'proposal_freeze.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'reviewable proposal only; no training or D2 collection authorized by this file','sha256':{str(p):sha(p)for p in freeze}})
 print('PROPOSAL_ONLY',len(roots),'new declared roots, zero collected observations',flush=True)
if __name__=='__main__':main()
