from common import *
import csv,shutil
def main():
 assert not E.exists()and not R.exists()
 for d in ['configs','source','data/model_inputs','data/private','results','logs']:(E/d).mkdir(parents=True)
 (R/'figures').mkdir(parents=True)
 design=read(DESIGN/'configs/protocol.json');assert sha(DESIGN/'configs/protocol.json')=='761f74ce71f41a2a2a6c2e05e672e1fc1f536c5e56a3145c01b322f45d4d7698'
 assert sha(DESIGN/'configs/assignments.csv')=='72bab16eb2f08202026d5e9558c1c546676c01a0b5f0f980e74da6fe47f0b44e'
 shutil.copy2(DESIGN/'configs/assignments.csv',E/'configs/assignments.csv')
 registry=read(P/'experiments/holding_action_fork_feasibility12_20261018/configs/historical_exclusion.json')
 ids=set(registry['old_root_ids']);seeds=set(registry['old_seeds']);rgb=set(registry['old_RGB_array_hashes']);layouts={};metadata={}
 paths={Path(p)for p in registry['metadata_paths']}
 paths.update((P/'data').glob('*/roots_manifest.json'))
 for pattern in ['*/configs/protocol.json','*/configs/roots.json','*/configs/*exclusion*.json','*/configs/*root*manifest*.json']:
  paths.update((P/'experiments').glob(pattern))
 paths={p for p in paths if not p.is_relative_to(E)and not p.is_relative_to(DESIGN)}
 def walk(x):
  if isinstance(x,dict):
   if isinstance(x.get('root_id'),str):ids.add(x['root_id'])
   if isinstance(x.get('seed'),int):seeds.add(x['seed'])
   if isinstance(x.get('RGB_sha256'),str):rgb.add(x['RGB_sha256'])
   b=x.get('blocks')
   if isinstance(b,dict)and set(b)==set(COLORS)and all(isinstance(v,dict)and'initial_pos'in v and'initial_orn'in v for v in b.values()):layouts[encoded(b)]=b
   b=x.get('layout')
   if isinstance(b,dict)and all('block_'+c in b and isinstance(b['block_'+c],list)and len(b['block_'+c])==6 for c in COLORS):
    b={c:{'initial_pos':b['block_'+c][:3],'initial_orn':b['block_'+c][3:]}for c in COLORS};layouts[encoded(b)]=b
   for k,v in x.items():
    if isinstance(k,str)and k.startswith(str(P))and Path(k).name in ['roots_manifest.json','roots.json','root_exclusion.json']:paths.add(Path(k))
    walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 while paths:
  p=paths.pop()
  if str(p)in metadata:continue
  assert p.exists(),str(p)
  metadata[str(p)]=sha(p);walk(read(p))
 oldfiles={};sealed=[p for p in(P/'experiments').glob('*/configs/delivery_manifest.json')if p.parent.parent!=E]
 for m in sealed:
  q=read(m)
  if 'sha256'in q and isinstance(q['sha256'],dict):oldfiles.update(q['sha256']);oldfiles[str(m)]=sha(m)
 # Historical RGB extension reads only known A/B/C engineering archives, not D samples.
 abc_names=['relation_GT_boxes_holding32_20261018','acquisition_scene_surface_pilot8_20261018','relation_GT_boxes36_ABC_20261018','holding_action_fork_feasibility12_20261018','holding_action_fork_feasibility12_v2_20261018','holding_action_fork_feasibility12_v3_20261018','native_restore_semantics_root1_final_20261018']
 raw_read=0
 for n in abc_names:
  b=P/'experiments'/n
  for p in b.rglob('*.npz'):
   if any(s in p.parts for s in ['models','configs']):continue
   with np.load(p,allow_pickle=False)as z:
    for k in z.files:
     if k in ['rgb','rgb_static','rgb_gripper']:
      a=z[k]
      if a.ndim==3 and a.shape[-1]==3 and a.dtype==np.uint8:rgb.add(hashlib.sha256(a.tobytes()).hexdigest());raw_read+=1
 historical_count=len(layouts);known=list(layouts.values());roots=[]
 for a in csv.DictReader((E/'configs/assignments.csv').open()):
  seed=int(a['layout_seed']);rid=a['root_id'];assert rid not in ids and seed not in seeds
  rng=np.random.default_rng(seed);lo,hi=np.array(design['layout']['scene_table_bounds'][a['scene']]);blocks={}
  for c,fraction in zip(rng.permutation(COLORS),[.18,.50,.82]):blocks[str(c)]={'initial_pos':[float(lo[0]+fraction*(hi[0]-lo[0])+rng.uniform(-.012,.012)),float(rng.uniform(-.115,-.055)),.46],'initial_orn':[0.,0.,float(rng.uniform(-.2,.2))]}
  distances=[max(max(abs(np.array(blocks[c][key])-np.array(b[c][key])))for c in COLORS for key in ['initial_pos','initial_orn'])for b in known]
  assert min(distances,default=1)>.001,'registered_layout_within1mm_or1mrad:'+rid
  # Additional stricter position-only check, independent of yaw and scene.
  posdist=[max(max(abs(np.array(blocks[c]['initial_pos'])-np.array(b[c]['initial_pos'])))for c in COLORS)for b in known]
  assert min(posdist,default=1)>.001,'registered_positions_within1mm:'+rid
  known.append(blocks);r={**a,'seed':seed,'execution_rank':int(a['execution_rank']),'target_color':a['target_color_private'],'geometry':a['geometry_private'],'preparation':a['preparation_private'],'action':a['assigned_action_private'],'blocks':blocks,'drawer_initial_q_m':.005,'parent_root':None,'nearest_registered_max_position_gap_m':min(posdist),'initialization_signature':hashlib.sha256(encoded({'scene':a['scene'],'blocks':blocks,'drawer':.005}).encode()).hexdigest()};roots.append(r)
 assert len(roots)==54 and len({r['initialization_signature']for r in roots})==54
 old_verified=0
 for p,h in oldfiles.items():assert sha(p)==h,p;old_verified+=1
 protocol={'utc':now(),'design_sha256':sha(DESIGN/'configs/protocol.json'),'assignments_sha256':sha(E/'configs/assignments.csv'),'roots':roots,'root_execution_order':[r['root_id']for r in sorted(roots,key=lambda r:r['execution_rank'])],'gates':design['data_gate'],'authorization':'standing user instruction Sentinel_9dd2d7c1a5fc8191a759ea02da2c724a, delegated bounded local collection only; no training','budget':dict(design['budget'],saved_states_per_root_max=6,dual_camera_saved_states_max=324),'anchors':ANCHORS,'only_budget_amendment':'explicit current authorization permits at most6 savedstates/root324; endpoints unchanged','model_input_keys':sorted(PUBLIC_KEYS),'randomization':'reuse exact frozen assignments; no outcome-conditioned changes','prediction_not_individual_counterfactual':True,'training_authorized':False}
 save(E/'configs/protocol.json',protocol)
 save(E/'configs/historical_exclusion.json',{'old_root_ids':sorted(ids),'old_seeds':sorted(seeds),'old_RGB_array_hashes':sorted(rgb),'metadata_sha256':metadata,'unique_normalized_layouts':historical_count,'new_layouts':54,'layout_resampling':0,'new_vs_historical_and_new_pair_position_only_and_pose_checks_pass':True,'ABC_archived_RGB_arrays_read':raw_read,'D_access':'registration metadata only for exclusion; no D samples,features,predictions,scores'})
 save(E/'configs/historical_frozen.json',{'sha256':oldfiles,'verified_files':old_verified})
 for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,E/'source'/f.name)
 refs=[P/'src/wm_symbolic/physical_oracle.py',P/'src/wm_symbolic/physical_oracle_v2.py',P/'vendor/calvin/calvin_env/calvin_env/robot/robot.py',P/'vendor/calvin/calvin_env/calvin_env/robot/mixed_ik.py',P/'vendor/calvin/calvin_env/calvin_env/envs/play_table_env.py']
 save(E/'configs/input_references.json',{'sha256':{str(p):sha(p)for p in refs}})
 files=[p for d in ['source','configs']for p in(E/d).rglob('*')if p.is_file()]
 save(E/'configs/precollection_frozen.json',{'utc':now(),'before_any_new_physics':True,'sha256':{str(p):sha(p)for p in files}})
 print(encoded({'roots':54,'historical_layouts':historical_count,'metadata_files':len(metadata),'historical_RGB_hashes':len(rgb),'frozen_files_verified':old_verified,'protocol_sha256':sha(E/'configs/protocol.json')}))
if __name__=='__main__':main()
