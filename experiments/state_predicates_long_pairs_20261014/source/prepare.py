from pathlib import Path
import json,hashlib,datetime,shutil
import numpy as np
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_long_pairs_20261014';B=P/'data/predicate_long_pairs_20261014';PRE=P/'experiments/state_predicates_held_down_20261014/candidate';C=['red','blue','pink']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not(E/'configs/protocol.json').exists()and not B.exists()
 for d in ['source','configs','data/features','results','figures','videos','logs']:(E/d).mkdir(parents=True,exist_ok=True)
 B.mkdir();(B/'episodes').mkdir();oldids=set();oldseeds=set();layouts=[];sources={};configs={e:OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml')for e in 'ABCD'}
 def visit(x,src):
  if isinstance(x,list):
   for v in x:visit(v,src)
  elif isinstance(x,dict):
   if isinstance(x.get('root_id'),str):oldids.add(x['root_id'])
   if isinstance(x.get('seed'),int):oldseeds.add(x['seed'])
   if isinstance(x.get('layout'),dict)and all('block_'+c in x['layout']for c in C):
    v=np.array([x['layout']['block_'+c]for c in C],float)
    if v.shape==(3,6):layouts.append((v,src))
   for v in x.values():
    if isinstance(v,(dict,list)):visit(v,src)
 for p in sorted(P.rglob('*roots*.json')):
  if 'vendor'in p.parts or E in p.parents or B in p.parents:continue
  sources[str(p)]=sha(p);visit(read(p),str(p))
 reset=list((P/'data/roots').glob('*/*/reset_evaluator_only.npz'))
 for v in ['v1','v2','v3']:reset+=list((P/f'runs/physical_oracle_{v}/roots').glob('*/reset_evaluator_only.npz'))
 for p in sorted(set(reset)):
  env=p.parent.name[0]
  if env not in configs:continue
  with np.load(p)as z:poses=z['scene_obs'][6:].reshape(3,6)
  lookup=dict(zip(list(configs[env].objects.movable_objects),poses));layouts.append((np.array([lookup['block_'+c]for c in C]),str(p)));sources[str(p)]=sha(p);oldids.add(p.parent.name)
 layouts=list({v.tobytes():(v,s)for v,s in layouts}.values());nold=len(layouts);prior=[{'layout':v.tolist(),'source':s}for v,s in layouts];roots=[];audit=[]
 for env in 'ABC':
  lo,hi=np.array(configs[env].surfaces.table,float)
  for ci,c in enumerate(C):
   for di,drawer in enumerate(['closed','open']):
    for rep in range(2 if env in 'AB' else 1):
     seed=14145200+len(roots);assert seed not in oldseeds;rng=np.random.default_rng(seed);reject=0
     while True:
      v=np.zeros((3,6));order=rng.permutation(3)
      for slot,k in enumerate(order):v[k,:3]=[lo[0]+(hi[0]-lo[0])*(.16+.34*slot)+rng.uniform(-.010,.010),rng.uniform(-.135,-.055),.46];v[k,5]=rng.uniform(-np.pi,np.pi)
      near=[]
      for ov,src in layouts:
       d=np.linalg.norm(v[:,:3]-ov[:,:3],axis=1);yaw=np.abs((v[:,5]-ov[:,5]+np.pi/2)%np.pi-np.pi/2)
       if np.all(d<=.0125)or(np.all(d<=.0205)and np.all(yaw<=.255)):near.append(src)
      if not near:break
      reject+=1
     rid=f'{env}_longpair_{c}_{drawer}_{rep}';assert rid not in oldids;skills=['open_drawer','lift_'+c+'_block_table','place_in_drawer','close_drawer']if drawer=='closed'else['lift_'+c+'_block_table','place_in_drawer','close_drawer','open_drawer'];role=None if env in 'AB'else('tune'if(ci+di)%2==0 else'calibration')
     roots.append({'root_id':rid,'environment':env,'split':'train'if env in 'AB'else'dev','dev_role':role,'target_color':c,'initial_drawer':drawer,'drawer_reset_m':0. if drawer=='closed'else .22,'replicate':rep,'seed':seed,'layout':{'block_'+co:v[k].tolist()for k,co in enumerate(C)},'planned_skills':skills,'new_independent_root':True});audit.append({'root_id':rid,'rejected_layout_candidates':reject,'old_id_seed_absent':True});layouts.append((v,rid))
 pilot=['A_longpair_red_closed_0','B_longpair_blue_open_0'];assert len(roots)==30 and sum(r['split']=='train'for r in roots)==24
 frozen=[P/'src/wm_symbolic'/s for s in ['env_adapter.py','physical_oracle.py','core_ownership.py','calibrated_labels.py','state_labels.py','vision_cache.py']]+[P/'scripts/core1200_collect.py',P/'scripts/evaluate_physical_oracle.py',PRE/'reader_seed1.pt',PRE/'model.py',PRE/'metrics.py',PRE/'normalization.json',PRE/'INPUT_CONTRACT.json',PRE/'candidate_manifest.json',Path(read(PRE/'INPUT_CONTRACT.json')['external_cached_backbone']['path'])/'model.safetensors']+[E/'source'/s for s in ['collect.py','validate_data.py','cache.py']]
 protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Collect fixed30newABCroots only, at most120 macro calls; freeze public visual encoder and candidate; no training, candidate inference, Dcollection or upload.','root_counts':{'train_A':12,'train_B':12,'dev_C':6},'balance':'EachA/B:3colours x2initialdrawerpositions x2layouts;C:3colours x2initialdrawerpositions. C tune/calibration3roots each, one percolour.','root_exclusion':'Use archived root metadata only to reject near duplicates; no old D observations/predictions/labels used. Generator rejects allnamedcentre distances<=12.5mm or all<=20.5mm with yaw modulo pi<=.255rad. Actual initialized layouts checked against12mm /20mm+.25rad rule.','pilot_roots':pilot,'maximum_macro_calls':120,'macro_contract':{'name':'frozen_skill_plus_posthold96_v1','initial_history':'One reset, then96existing hold controls with grip+1 outside all macros.','macro_start':'Current actual observation immediately before frozen PhysicalOracle.run(skill).','primitive':'Unchanged controller, budget900, existing max2 bounded attempts; no parameter changes.','post_hold':'Exactly96existing PhysicalOracle.hold controls INSIDE macro interval, retain public robot_obs[-1] grippercommand at primitive end. Endpoint afterthishold; nextmacro starts there.','macro_end':'After posthold96; before/after history uses exact[t-63,t-42,t-21,t].','action_conditioning':'Declared skill and arguments, primitivebudget900, posthold96. Actual duration, success, phase, contacts and target labels prohibited as predictive inputs.','failures':'Record failed/refused primitive and posthold state effects. Continue fixedremaining calls through unchanged controller precondition checks unless infrastructure/physicsguard error; no replacementroots or added attempts. Refusal remains a macro with96hold; do not call it a zero-duration selfloop.','waiting_effects':'Record TCP/allobject displacement, held losses at any wait frame, label changes and endpoint goals; never hide a drop behind earlier primitive success.','future_training_test':'Same exact wrapper and waiting convention must be used in training and later tests.'},'history':{'control_Hz':30,'physics_Hz':240,'physics_steps_per_control':8,'offsets':[63,42,21,0],'seconds':2.1,'time_verification':'Count actual pybullet.stepSimulation calls afterreset; frame f must equal8f physicssteps; no repeats or futureinputs; timestamps derive counted simulationsteps, wall time separate.'},'labels':{'order':['held(red)','held(blue)','held(pink)','in(red,drawer)','in(blue,drawer)','in(pink,drawer)','open(drawer)'],'all_entities':True,'held':'Unchanged bilateral descendantfinger force>.01N with no external force-bearing contact.','in':'Unchanged primary drawerownership (existing held override and supportlabelaudit), not generic geometriccontainment or whole-action completion.','open':'Unchanged categorical draweropen>.16 and valid outside frozen5mm boundaries at.025/.16.','validity':'Preserve existing physicalvalid mask for all7facts; visibility does not mask labels. No target-only mask.','storage':'Private full audit separate from public NPZ/actions; only7taskpredicates become supervision arrays. Tableleft/middle/right not a required modeling target.'},'features':{'backbone':'Frozen existingDINOv2-small; sameRGB/pad/ROI/visiblegeometry and normalization','registry':'FreshVisibleRegistry per4frame window, originalpixel-based registration. Missingentity stayszeros; no GT ID substitution.','shapes':{'objects':[4,3,780],'context':[4,3072],'robot':[4,15]},'candidate':'Seed1 reader unchanged; no readerforward in this task; only frozenvision featureextraction authorized.','queue':'Collection andDINO cache serialized byexistingGPU lock.'},'pilot_gate':'After2fixedtrainingroots, verify all8attempts accounted for; completecausal inputs for eachcompletedmacro, actualtick/time alignment, allentitylabels, hashes, no forbiddenphysicsmutation; failures retained. Projectremainingdisk from measuredbytes/frame timesworstcase remaining120*(900+96)+30*96 controls, require1.5xprojection+5GiB free. No controller-success quota used to discard failures.','frozen_sha256':{str(p):sha(p)for p in frozen},'no_training':True,'no_D_collection':True,'no_upload':True}
 save(B/'roots_manifest.json',roots);protocol['roots_manifest_sha256']=sha(B/'roots_manifest.json');save(E/'configs/protocol.json',protocol);save(E/'configs/root_exclusion.json',{'prior_unique_layouts':nold,'sources_sha256':sources,'prior_layouts':prior,'new_roots_audit':audit});save(B/'data_manifest.json',{'version':'long_task_predicate_pairs_v1','records':[],'roots':[]});save(E/'results/progress.json',{'stage':'frozen_before_collection','roots':0});save(E/'results/storage_before.json',{'free_bytes':shutil.disk_usage(B).free})
 for p in [E/'configs/protocol.json',E/'configs/root_exclusion.json',B/'roots_manifest.json']:p.chmod(0o444)
 print(json.dumps({'frozen_roots':30,'train':24,'dev':6,'max_macros':120,'pilot':pilot,'prior_layouts_excluded':nold},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
