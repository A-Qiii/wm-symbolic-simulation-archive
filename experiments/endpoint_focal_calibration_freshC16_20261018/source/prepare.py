from common import *
import shutil,csv

def main():
 assert not E.exists()
 for sub in ['configs','source','models','data/public','data/annotation_private','data/calibration','results','logs']:(E/sub).mkdir(parents=True,exist_ok=True)
 R.mkdir();(R/'figures').mkdir()
 manifests=sorted(set((P/'experiments').glob('*/configs/delivery_manifest.json'))|set((P/'experiments').glob('*/configs/data_frozen.json')))
 rootmeta=sorted(set((P/'experiments').glob('*/configs/roots.json'))|set((P/'experiments').glob('*/configs/root_exclusion.json'))|{BASE/'configs/protocol.json',BASE/'configs/historical_initialization_exclusion.json'})
 inputs=manifests+rootmeta+[BASE/'data/public/images.json',BASE/'configs/normalization.npz',BASE/'configs/mesh_semantics.json',PRBASE/'data/ABC_labels_metadata.json']
 for s in [0,1]:
  inputs += [BASE/f'models/final_seed{s}.pt',PRBASE/f'data/candidates_seed{s}_dev.npz']+[PRBASE/f'data/pr_seed{s}_dev_all_{c}.npz'for c in CATS]
 inputs=list(dict.fromkeys(inputs));opened=install_guard(inputs);hashes={str(p):sha(p)for p in inputs};seals={r:read(r/'configs/delivery_manifest.json')['sha256']for r in [BASE,PRBASE]}
 for p in inputs:
  for root,seal in seals.items():
   if p.is_relative_to(root)and p.name!='delivery_manifest.json':assert hashes[str(p)]==seal[str(p)],p
 # Calibration is completed and locked before any new root is generated/rendered.
 metadata=read(PRBASE/'data/ABC_labels_metadata.json');dev=[r for r in metadata if r['split']=='dev'];assert len(dev)==128 and len({r['root_id']for r in dev})==8
 cal=[]
 for seed in [0,1]:
  a=np.load(PRBASE/f'data/candidates_seed{seed}_dev.npz')
  for cid,cat in enumerate(CATS):
   curve=np.load(PRBASE/f'data/pr_seed{seed}_dev_all_{cat}.npz');n=sum(len(r['boxes'][cid])for r in dev if r['class_valid'][cid]);assert n in [100,254];good=np.where((curve['precision']>=.90)&((curve['TP']+curve['FP'])>0))[0]
   if len(good):
    best=max(good,key=lambda i:(float(curve['recall'][i]),float(curve['threshold'][i])));threshold=float(curve['threshold'][best]);select=(a['class_id']==cid)&(a['score']>=threshold);tp=int(a['tp'][select].sum());fp=int(select.sum())-tp;assert tp==int(curve['TP'][best])and fp==int(curve['FP'][best]);assert tp/(tp+fp)>=.90;assert all(float(curve['recall'][j])<=tp/n+1e-12 for j in good);status='success'
   else:threshold=None;select=np.zeros(len(a['score']),bool);tp=fp=0;status='failed_no_nonempty_prediction_reaches_precision_0.90'
   ids=set(map(int,a['image_id'][select]));tpids=set(map(int,a['image_id'][select&(a['tp']==1)]));cal.append({'seed':seed,'category':cat,'status':status,'threshold':threshold,'target_precision':.9,'GT':n,'TP':tp,'FP':fp,'FN':n-tp,'precision':tp/(tp+fp)if tp+fp else None,'recall':tp/n,'F1':2*tp/(n+tp+fp),'old_C_images':128,'old_C_roots':8,'valid_images':sum(r['class_valid'][cid]for r in dev),'valid_roots':len({r['root_id']for r in dev if r['class_valid'][cid]}),'selected_image_count':len(ids),'selected_root_count':len({r['root_id']for r in dev if r['image_id']in ids}),'TP_root_count':len({r['root_id']for r in dev if r['image_id']in tpids}),'candidate_unique_scores':len(curve['threshold'])})
 save(E/'configs/calibration_locked.json',{'utc':now(),'before_new_roots_generated':True,'rule':'For each seed and category, using pooled cameras and actual normal decoded old C predictions: maximize recall among existing score thresholds with nonempty predictions and precision >= .90; ties higher threshold. No envelope, interpolation, fallback, threshold grid or target relaxation.','old_C_is_calibration_not_validation':True,'calibration_failed_action':'no calibrated predictions for that class; threshold null, report failure; fixed0.5 still evaluated','rows':cal})
 for s in [0,1]:shutil.copy2(BASE/f'models/final_seed{s}.pt',E/f'models/final_seed{s}.pt')
 for n in ['normalization.npz','mesh_semantics.json']:shutil.copy2(BASE/'configs'/n,E/'configs'/n)
 shutil.copy2(PRBASE/'data/ABC_labels_metadata.json',E/'data/calibration/ABC_labels_metadata.json')
 for s in [0,1]:shutil.copy2(PRBASE/f'data/candidates_seed{s}_dev.npz',E/f'data/calibration/candidates_seed{s}_dev.npz')
 # Metadata-only historical registry: no old RGB, feature, prediction, D labels/scores opened.
 old_filehashes={};old_roots=set();old_seeds=set();initialsignatures=set();old_blocks=[]
 def walk(o):
  if isinstance(o,dict):
   for k,v in o.items():
    if k in ['root_id','root']and isinstance(v,str):old_roots.add(v)
    if k=='seed'and isinstance(v,int):old_seeds.add(v)
    if k=='initialization_signature'and isinstance(v,str):initialsignatures.add(v)
   if isinstance(o.get('blocks'),dict)and all(isinstance(b,dict)and 'initial_pos'in b and 'initial_orn'in b for b in o['blocks'].values()):old_blocks.append(o['blocks'])
   for v in o.values():walk(v)
  elif isinstance(o,list):
   for v in o:walk(v)
 for p in manifests:
  d=read(p)
  def gather(x):
   if isinstance(x,dict):
    for k,v in x.items():
     if isinstance(v,str)and len(v)==64 and all(c in '0123456789abcdef'for c in v)and isinstance(k,str)and k.startswith('/'):old_filehashes[k]=v
     else:gather(v)
   elif isinstance(x,list):
    for v in x:gather(v)
  gather(d)
 for p in rootmeta:walk(read(p))
 oldimages=read(BASE/'data/public/images.json');oldRGB={r['RGB_sha256']for r in oldimages};old_roots.update(r['root_id']for r in oldimages)
 roots=[]
 for k in range(16):
  seed=261018800+k;assert seed not in old_seeds;rng=np.random.default_rng(seed);colors=rng.permutation(['red','blue','pink']);blocks={}
  for color,x in zip(colors,[-.15,.075,.30]):blocks[str(color)]={'initial_pos':[float(x+rng.uniform(-.015,.015)),float(rng.uniform(-.115,-.045)),.46],'initial_orn':[0.,0.,float(rng.uniform(-.35,.35))]}
  drawer=float(np.clip([.005,.065,.15,.235][k//4]+rng.uniform(-.004,.004),0,.275));bases=[[-.10,-.22,.60],[.16,-.22,.60],[.04,-.16,.53],[-.14,-.26,.67],[.20,-.13,.66],[.02,-.29,.56],[.03,-.22,.60]];wps=[]
  for j,base in enumerate(bases):wps.append({'xyz':(np.array(base)+rng.uniform([-.025,-.018,-.018],[.025,.018,.018])).tolist(),'euler':[float(3.09+rng.uniform(-.12,.12)),float(rng.uniform(-.18,.18)),float(1.5+rng.uniform(-.45,.45))],'grip':1 if j%2==0 else-1,'control_periods':24,'physics_ticks_per_period':8})
  rid=f'freshC16_C_{k:02d}';sig=hashlib.sha256(json.dumps({'scene':'C','drawer':drawer,'blocks':blocks},sort_keys=True).encode()).hexdigest();assert rid not in old_roots and sig not in initialsignatures
  def near(b):return set(b)==set(blocks)and all(np.allclose(blocks[c]['initial_pos'],b[c]['initial_pos'],atol=.001,rtol=0)and np.allclose(blocks[c]['initial_orn'],b[c]['initial_orn'],atol=.001,rtol=0)for c in blocks)
  assert not any(near(b)for b in old_blocks);roots.append({'root_id':rid,'scene':'C','split':'fresh','seed':seed,'blocks':blocks,'drawer_initial_q_m':drawer,'waypoints':wps,'initialization_signature':sig})
 assert len({r['initialization_signature']for r in roots})==16
 save(E/'configs/historical_registry.json',{'metadata_files':[str(p)for p in manifests+rootmeta],'old_root_ids':sorted(old_roots),'old_seeds':sorted(old_seeds),'initialization_signatures':sorted(initialsignatures),'old_layouts_compared':len(old_blocks),'old_RGB_array_hashes':sorted(oldRGB),'old_file_hashes':old_filehashes,'scope':'available saved root registries and manifests, all sync256 primary RGB array hashes; technical 4-state raw/snapshot file hashes via manifest; not a claim about unregistered historical data','old_D_access':'only stored root/hash metadata for dedup; no old D sample arrays, features, predictions or metrics'})
 protocol={'utc':now(),'task':'Final bounded localization conclusion: fixed .5 versus old-C calibrated per-class threshold, original focal two seeds, no training','roots':roots,'budget':{'roots':16,'states_per_root':8,'max_states':128,'max_primary_images':256,'new_fits':0,'new_model_downloads':0,'replacement_roots':0,'retry_after_quality_failure':False},'claim':'same known C scene, unseen initialization/interaction configurations; not cross-scene generalization','root_generator':'same block/waypoint distributions as sync256; four drawer bands, four roots each; seeds261018800..815 fixed before collection','calibration':'configs/calibration_locked.json; supersedes earlier unexecuted global grid proposal; per-class per-seed pooled-camera precision>=.90 then max recall, ties higher threshold','models':{'original_focal':True,'seeds':[0,1],'workpoints':['fixed0.5','calibrated'],'no_seed_selection':True,'sha256':{str(E/f'models/final_seed{s}.pt'):sha(E/f'models/final_seed{s}.pt')for s in [0,1]},'normalization_sha256':sha(E/'configs/normalization.npz')},'collection':'same synchronous collector and geometry semantics, native reset plus seven fixed motor waypoints; preserve unreachable/collision/unknown states; no post-reset teleport or score-based selection','labels':'any unknown drawer pixel makes front class unknown; missing finger family makes finger class unknown; zero-visible never certified absent; only known-class background negatives','gates':{'min_boxes_each_class_camera':12,'min_positive_roots_each_class_camera':4,'min_handle_context_images_each_camera':8,'min_handle_context_roots_each_camera':4,'sync':'all snapshots and four same-tick render calls exact, primary/flag-off RGB-depth equal','dedup':'new root ids, seeds, layouts distinct; initial snapshot/full dual primary NPZ must not match old file hashes; individual RGB matches reported separately (duplicate view is not automatically duplicate root)','failure':'preserve failure, no replacement roots or score chasing; stop dependent scoring on invalid integrity/coverage'},'evaluation':{'one_new_prediction_pass_per_seed':True,'one_scoring_invocation':True,'AP':'full normal candidates AP50 invariant to operating threshold; report shared AP for both workpoints, no truncated AP substituted','fixed_work_point':.5,'calibrated_work_points':'per seed/class frozen old-C thresholds; no camera-specific thresholds','bootstrap':{'roots':16,'replicates':2000,'seed':261019,'paired_across_seeds_workpoints':True,'method':'resample entire roots with replacement; recompute TP/FP/FN and full AP using root multiplicities and score-tie groups; 95% percentile intervals','no_threshold_recalibration':True},'root_metrics':'all and class/camera AP50, P/R/F1 and unknown coverage; same candidate matches shared by workpoints'},'stop':'after128 states at most, conclude localization even if calibrated precision fails on new roots; no model/threshold/loss search; relation-readout design only','old_inputs_allowlist':[str(p)for p in inputs]}
 save(E/'configs/protocol.json',protocol);save(E/'configs/input_frozen.json',{'utc':now(),'sha256':hashes});save(E/'results/preparation.json',{'utc':now(),'calibration_complete':True,'new_roots_frozen':16,'new_root_id_seed_signature_intersections':0,'near_1mm_layout_reuse':False,'old_metadata_files':len(manifests+rootmeta),'old_files_opened':sorted(opened),'old_D_holdout_sample_feature_prediction_metric_reads':0,'note':'Preparation also inspected the existing four-state technical results index for registry schema; it contains per-scene technical label summaries, is not used for calibration or validation. No original sync256 D holdout samples or scores were opened.','free_disk_bytes':shutil.disk_usage(P).free})
 for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,E/'source'/f.name)
 files=[f for sub in ['configs','source','models']for f in(E/sub).rglob('*')if f.is_file()];save(E/'configs/precollection_frozen.json',{'utc':now(),'before_first_new_render':True,'sha256':{str(f):sha(f)for f in files}})
 print('CALIBRATION_AND_16_NEW_ROOTS_FROZEN',flush=True)
 for r in cal:print(r,flush=True)
if __name__=='__main__':main()
