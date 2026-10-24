from coverage_common import *
import shutil,gzip,torch,collections
def main():
 assert not E.exists()
 for sub in ['configs','source','data/train','data/evaluation','data/public/observations','data/public/features','data/private','models','results','logs']:(E/sub).mkdir(parents=True)
 R.mkdir(parents=True,exist_ok=False);refs={}
 def reg(p):p=Path(p);refs[str(p)]=sha(p);return p
 for root in [EXP,H,I]:
  mf=reg(root/'configs/delivery_manifest.json')
  for p,h in read(mf)['sha256'].items():assert sha(p)==h,p
 qnorm=dict(np.load(reg(W/'configs/query_normalization.npz'),allow_pickle=False));np.savez_compressed(E/'configs/query_normalization.npz',**qnorm)
 old={};rawsets={}
 for name in ['historical_AB_training','new_AB_train','new_AB_validation','old_C_selection','old_C_validation','C_selection','C_validation']:
  d=dict(np.load(reg(W/'data/evaluation'/f'{name}.npz'),allow_pickle=False));d['z']=(d['z']*qnorm['std']+qnorm['mean']).astype(np.float32);rawsets[name]=d
 old['Holding']={k:np.concatenate([rawsets[n][k]for n in['historical_AB_training','new_AB_train']])for k in rawsets['historical_AB_training']}
 old['Holding']['state_ids']=np.array(['oldpool:'+str(x)for x in rawsets['historical_AB_training']['state_ids']]+['combo:'+str(x)for x in rawsets['new_AB_train']['state_ids']])
 hdata=np.load(reg(H/'data/train/states.npz'),allow_pickle=False)
 for k in ['z','state_ids','root_ids']:assert np.array_equal(old['Holding'][k],hdata[k]),k
 assert np.array_equal(old['Holding']['y'][:,:3],hdata['y'])and np.array_equal(old['Holding']['v'][:,:3],hdata['v'])
 old['In']=rawsets['historical_AB_training'];assert len(old['Holding']['z'])==210 and len(old['In']['z'])==156
 for head,source in [('Holding',H/'configs/normalization_object.npz'),('In',I/'configs/object_normalization.npz')]:
  shutil.copy2(reg(source),E/'configs'/f'normalization_{head}.npz');norm=np.load(source);d=old[head];x=((d['z'][:,:3]-norm['mean'])/norm['std']).astype(np.float32)
  if head=='In':assert np.array_equal(x,np.load(reg(I/'data/train/control.npz'),allow_pickle=False)['x'])
  np.savez_compressed(E/'data/train'/f'old_{head}.npz',**d,x=x)
 # These old labels already use the same native geometry corrections in both arms.
 for p in [W/'audit/historical_label_changes.json',W/'audit/retention_C_label_changes.json',W/'audit/retention_C_unknown_In.json']:shutil.copy2(reg(p),E/'configs'/p.name)
 source=read(reg(EXP/'results/results.json'));new=[r for r in source if r['source_kind']=='new_official_frame'];assert len(new)==240
 new=sorted(new,key=lambda r:(r['split'],r['root_id'],r['frame_id']));public=[];private=[];align=[]
 for idx,row in enumerate(new):
  c=row['candidate'];assert c['final_step']==96 and c['fact_order']==[f'Holding({co})'for co in COLORS]+[f'In({co},drawer)'for co in COLORS]
  op=reg(c['observation_path']);assert sha(op)==c['observation_sha256'];s=np.load(op,allow_pickle=False);dest=E/'data/public/observations'/f'{idx:04d}.npz';np.savez_compressed(dest,**{k:s[k]for k in ['rgb_static','rgb_gripper','robot_obs']});check=np.load(dest)
  assert all(np.array_equal(check[k],s[k])for k in check.files)
  history=reg(c['physics_history_path']);assert sha(history)==c['physics_history_sha256']
  with gzip.open(history,'rt')as stream:
   for line in stream:last=json.loads(line)
  assert last['physics_step']==96 and np.array_equal(s['robot_obs'],last['state']['robot_obs'])and last['new_observation_sha256']==sha(op)
  assert c['y']==last['labels']['y'][:6]
  ids={'index':idx,'root_id':row['root_id'],'scene':row['scene'],'frame_id':row['frame_id'],'split':row['split'],'state_id':f"official:{row['root_id']}:{row['frame_id']}:step96"}
  public.append(ids|{'path':str(dest),'sha256':sha(dest),'source_observation_path':str(op),'source_observation_sha256':sha(op)})
  private.append(ids|{'y':c['y']+last['labels']['y'][6:],'v':c['v']+last['labels']['v'][6:],'candidate_hints':row['candidate_hints']})
  slot=['pink','blue','red']if row['scene']=='A'else COLORS;align.append(ids|{'official_scene_slot_names':slot,'canonical_names':COLORS,'canonical_from_scene_slot_indices':[slot.index(co)for co in COLORS],'label_fact_order':c['fact_order'],'image_identity_check':True,'proprio_same_time_check':True,'pixels_and_robot_unchanged':True,'model_object_order':'Frozen RGB registry centres[0,240,300] => red,blue,pink; no GT-dependent feature permutation'})
 save(E/'data/public/states.json',public);save(E/'data/private/labels.json',private);save(E/'results/image_label_alignment.json',align)
 official_train=[r for r in private if r['split']=='train_candidate'];official_val=[r for r in private if r['split']=='validation_candidate'];assert len(official_train)==len(official_val)==120
 for a in[official_train,official_val]:assert len({r['root_id']for r in a})==4
 assert {r['root_id']for r in official_train}.isdisjoint(r['root_id']for r in official_val)
 np.savez_compressed(E/'data/private/official_labels.npz',y=np.array([r['y']for r in private],np.float32),v=np.array([r['v']for r in private],np.float32),indices=np.arange(240),root_ids=np.array([r['root_id']for r in private]),state_ids=np.array([r['state_id']for r in private]),split=np.array([r['split']for r in private]))
 # Old actual retention sets; no predictions or outcomes are used in training schedules.
 for name in ['new_AB_validation','old_C_selection','old_C_validation','C_selection','C_validation']:np.savez_compressed(E/'data/evaluation'/f'{name}.npz',**rawsets[name])
 cc={k:np.concatenate([rawsets[n][k]for n in['old_C_selection','old_C_validation','C_selection','C_validation']])for k in rawsets['C_validation']};assert len(set(zip(cc['root_ids'],cc['state_ids'])))==len(cc['z']);np.savez_compressed(E/'data/evaluation/C_all_available.npz',**cc)
 pmeta=read(reg(PRIOR/'data/ABC/public/states.json'));cv=rawsets['C_validation'];ix=[]
 for i,sid in enumerate(cv['state_ids']):
  prefix,ind=str(sid).split(':');assert prefix=='new';s=pmeta[int(ind)];assert s['root_id']==cv['root_ids'][i]
  if any(tag.startswith('after_call:')for tag in s['tags']):ix.append(i)
 assert len(ix)==48;np.savez_compressed(E/'data/evaluation/C_endpoints48.npz',**{k:v[ix]for k,v in cv.items()})
 di=np.load(reg(D/'data/combo/public/inputs.npz'),allow_pickle=False);pairs=read(reg(D/'data/combo/public/pairs.json'));dl=read(reg(D/'data/combo/_private/labels.json'));assert len(pairs)==135
 np.savez_compressed(E/'data/evaluation/D12_seen_actual135.npz',z=di['z'][[p['after']for p in pairs]],y=np.array([dl[str(p['after'])]['y']for p in pairs],np.float32),v=np.array([dl[str(p['after'])]['v']for p in pairs],np.float32),root_ids=np.array([p['root_id']for p in pairs]),state_ids=np.array([f"D12pair:{p['id']}"for p in pairs]))
 # Eight AB observations for parity with the existing frozen encoder cache.
 combo_states=read(reg(COMBO/'data/combo/public/states.json'));ab=[(i,s)for i,s in enumerate(combo_states)if s['environment']in ['A','B']][:8];assert len(ab)==8
 ci=np.load(reg(COMBO/'data/combo/public/inputs.npz'),allow_pickle=False);save(E/'data/public/parity_states.json',[s|{'cache_index':i}for i,s in ab]);np.savez_compressed(E/'data/public/parity_expected.npz',**{k:ci[k][[i for i,s in ab]]for k in['objects','scene','z']})
 seeds=[0,1];rootids=sorted({r['root_id']for r in official_train});byroot={root:np.array([i for i,r in enumerate(official_train)if r['root_id']==root])for root in rootids};sampling=[]
 for seed in seeds:
  rng=np.random.default_rng(17082010+seed);official=np.empty((600,8),np.int64)
  for j,root in enumerate(rootids):official[:,2*j:2*j+2]=np.concatenate([rng.permutation(byroot[root])for _ in range(40)]).reshape(600,2)
  for row in official:rng.shuffle(row)
  assert np.bincount(official.ravel(),minlength=120).tolist()==[40]*120
  for head in HEADS:
   original=reg((H if head=='Holding'else I)/'configs'/f'schedule_seed{seed}.npz');arr=np.load(original,allow_pickle=False);oldsc=arr['indices'if head=='Holding'else'control'];assert oldsc.shape==(600,32)
   np.savez_compressed(E/'configs'/f'schedule_{head}_seed{seed}.npz',control=oldsc,common_old=oldsc[:,:24],official=official)
   initseed=(17078000 if head=='Holding'else 17075000)+seed;torch.manual_seed(initseed);q=make()
   if head=='In':initial=torch.load(reg(I/'models'/f'initial_seed{seed}.pt'),map_location='cpu',weights_only=True);q.load_state_dict(initial['state'])
   torch.save({'state':q.state_dict(),'initialization_seed':initseed,'state_hash':tensor_hash(q.state_dict())},E/'models'/f'initial_{head}_seed{seed}.pt')
   g=slice(0,3)if head=='Holding'else slice(3,6);ny=np.array([r['y']for r in official_train]);nv=np.array([r['v']for r in official_train])
   for arm in ARMS:
    yy=old[head]['y'][oldsc if arm=='control'else oldsc[:,:24]];vv=old[head]['v'][oldsc if arm=='control'else oldsc[:,:24]]
    if arm=='augmented':yy=np.concatenate([yy,ny[official]],1);vv=np.concatenate([vv,nv[official]],1)
    sampling.append({'head':head,'seed':seed,'arm':arm,'state_draws':19200,'old_draws':19200 if arm=='control'else 14400,'new_draws':0 if arm=='control'else 4800,'group_positive_draws_per_color':(yy[...,g]*vv[...,g]).sum((0,1)),'group_negative_draws_per_color':((1-yy[...,g])*vv[...,g]).sum((0,1)),'group_unknown_draws_per_color':(1-vv[...,g]).sum((0,1)),'same_information_budget':False})
 save(E/'results/sampling_information_budget.json',sampling)
 hp=read(reg(H/'configs/protocol.json'));ip=read(reg(I/'configs/protocol.json'));dc=read(reg(D/'configs/protocol.json'));queries={str(seed):dc['query_heads'][str(seed)]['original']for seed in seeds}
 for q in queries.values():reg(q['path'])
 files=['features.py','registry.py','representation.py','model.py','runtime.py','common.py']
 for f in files:reg(REP/'source'/f)
 for p in[REP/'data/public/normalization.npz',REP/'models/canonical_matched_seed0/selected.json',REP/'models/canonical_matched_seed0/update1000.pt',REP/'models/query_seed0.pt',Path('/home/robot/research/models/facebook--dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056/model.safetensors')]:reg(p)
 protocol={'utc':now(),'purpose':'Bounded data coverage contrast at unchanged frozen representation/transition; no formal adoption','heads':HEADS,'architecture':'shared own-object64 -> Linear32 -> SiLU -> Linear1,2113 parameters each; same recent independent head architecture','seeds':seeds,'arms':ARMS,'max_fits':8,'updates_each':600,'batch_states':32,'common_old_slots':24,'replacement_slots':8,'old_pools':{'Holding':{'states':210,'roots':36,'source':str(H/'data/train/states.npz'),'meaning':'156 native-corrected historical AB states plus54 full states from6 old combination training roots, identical recent Holding diagnostic pool'},'In':{'states':156,'roots':30,'source':str(I/'data/train/control.npz'),'meaning':'Recent object-In CONTROL baseline historical AB pool; prior6-dual-state coverage arm is not mixed into this new contrast'}},'old_normalization':'Copy corresponding existing Holding210/In156 training-only mean/std unchanged; no official/validation/C/D fitting','positive_weights':{'Holding':[hp['positive_weight']]*3,'In':ip['positive_weights']},'optimizer':'fresh AdamW lr.001 wd.001 global clip1; both arms on same CUDA device; no scheduler','loss':'same group masked positive-weighted BCE; mean valid objects per state then mean batch; unknown never treated as negative','initialization_seeds':{'Holding':[17078000,17078001],'In':[17075000,17075001]},'selection':'fixed final600,threshold.5,no search,no earlystop,no D-based selection','schedule':'Exact original head-specific control600x32 schedule retained. Both arms first24 identical. Control last8 original old draws; augmented last8 official training,exact2 per newepisode per update,each120 state exactly40 draws. Same official schedule for both heads.','official_training_states':120,'official_training_episodes':rootids,'official_validation_states':120,'official_validation_episodes':sorted({r['root_id']for r in official_val}),'old16_retests_excluded':True,'new_state_unknown_In_facts':sum(not v for r in private for v in r['v'][3:6]),'all_expansion_unknown6_note':'Two old-retest unknown facts excluded with those16states; four remaining unknown In facts stay masked in official validation.','official_labels':'Only step96 same-time new RGB/proprio, first6 candidate stability masks; drawer native same-time value/validity used only for evaluation','object_alignment':'Frozen RGB colour registry already outputs red,blue,pink; simulator slots A pink,blue,red map[2,1,0], B red,blue,pink map[0,1,2] for evaluator identity audit only; no GT encoder input','frozen_encoder':'Existing DINO batch8 + existing representation.transform/normalization + canonical_matched seed0 update1000 encode only','frozen_query_heads':queries,'frozen_drawer':'original formal Query entries6:9 unchanged per seed/input, argmax categorical decision','evaluation_cohorts':['official_validation','C_all_available','old_C_selection','old_C_validation','C_selection','C_validation','C_endpoints48','new_AB_validation','D12_seen_actual135'],'primary':'Perhead/perseed percolor FPR,FNR,positive/negative root counts; all3 exact; official same-annotation matched positive/negative correctness; paired combinedH+In+originaldrawer full9','secondary_gate':{'per_head_both_seeds':True,'official_validation':'mean of per-color (FPR+FNR)/2 must improve>=.01; all3 exact rate cannot decrease; each color FPR/FNR increase<=.05','retention':['C_all_available','C_validation','C_endpoints48','new_AB_validation'],'retention_limits':'all3 exact rate decrease<=.02; each defined per-color FPR/FNR increase<=.05; missing class is null, never zero','D':'previously seen descriptive only, excluded from gate','if_pass':'Only passed head reads already saved D12 mixed/latent one-step/recursive z, matched seed; no T forward or retraining. Full9 secondary only if both heads pass.','if_fail':'Stop secondary predicted-z reading for that head; no retuning or new fit'},'no_updates':['DINO','representation','all T','drawer query'],'no_new_collection_or_download':True}
 save(E/'configs/protocol.json',protocol);save(E/'configs/source_references.json',{'sha256':refs})
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 frozen={str(p):sha(p)for root in[E/'configs',E/'data',E/'models',E/'source']for p in root.rglob('*')if p.is_file()}
 save(E/'configs/pre_encoding_frozen.json',{'utc':now(),'before_new_image_feature_extraction_and_training':True,'sha256':frozen,'source_refs':refs})
 print('READOUT_DESIGN_FROZEN',len(public),'new images,120train/120validation,8fixedfits; old pools210/156',flush=True)
if __name__=='__main__':main()
