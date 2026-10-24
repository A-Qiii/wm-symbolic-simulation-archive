from common import *
import shutil,gzip

def main():
 assert not E.exists()
 for d in ['source','configs','data/public/observations','data/public/features','data/train','data/evaluation','data/private','models','results','logs']:(E/d).mkdir(parents=True)
 R.mkdir(parents=True);(R/'figures').mkdir();refs={}
 def reg(p):refs[str(p)]=sha(p);return p
 for root in [OLD,EXP]:
  for p,h in read(reg(root/'configs/delivery_manifest.json'))['sha256'].items():assert sha(p)==h,p
 cfg=read(reg(OLD/'configs/protocol.json'))
 oldmeta=read(reg(REP/'data/public/states.json'));newmeta=read(reg(PRIOR/'data/ABC/public/states.json'))
 combo={str(x['state_index']):x for x in read(reg(EX/'simultaneous_relation_probe_20261017/data/train/states.json'))}
 publicofficial={r['state_id']:r for r in read(reg(OLD/'data/public/states.json'))};pairs={p['id']:p for p in read(reg(D/'data/combo/public/pairs.json'))};ds=read(reg(D/'data/combo/public/states.json'))
 def resolve(sid,root):
  if sid.startswith('oldpool:'):sid=sid[8:]
  if sid.startswith('official:'):m=publicofficial[sid];assert m['root_id']==root;return m['path']
  typ,idx=sid.split(':')
  if typ=='combo':m=combo[idx];assert m['root_id']==root;return m['public_path']
  if typ=='D12pair':m=ds[pairs[idx]['after']]
  else:m=(oldmeta if typ=='old'else newmeta)[int(idx)]
  assert m['root_id']==root,(sid,m['root_id'],root);return m['path']
 rows=[];index={};cohorts={};labelrecords=[]
 def add(sid,root,path,y,v,role,extra=None):
  key=(root,sid)
  if key not in index:
   path=reg(path);raw=np.load(path,allow_pickle=False);rgb={k:raw[k].copy()for k in['rgb_static','rgb_gripper']};assert all(x.dtype==np.uint8 for x in rgb.values())
   i=len(rows);dest=E/'data/public/observations'/f'{i:05d}.npz';np.savez_compressed(dest,**rgb)
   h=hashlib.sha256(b''.join(k.encode()+str(rgb[k].shape).encode()+rgb[k].tobytes()for k in sorted(rgb))).hexdigest()
   rows.append({'index':i,'state_id':sid,'root_id':root,'scene':root[0]if not root.startswith('official')else root.split('_')[2],'path':str(dest),'sha256':sha(dest),'RGB_content_sha256':h,'source_path':str(path),'roles':[role]})
   labelrecords.append({'index':i,'state_id':sid,'root_id':root,'y':list(y),'v':list(v),'extra':extra or {}});index[key]=i
  else:
   i=index[key];assert np.array_equal(labelrecords[i]['y'],y)and np.array_equal(labelrecords[i]['v'],v);rows[i]['roles'].append(role)
  return i
 for head in HEADS:
  dd=np.load(reg(OLD/f'data/train/old_{head}.npz'));ii=[]
  for sid,root,y,v in zip(dd['state_ids'],dd['root_ids'],dd['y'],dd['v']):
   sid=str(sid);root=str(root);canonical=sid[8:]if sid.startswith('oldpool:')else sid
   ii.append(add(canonical,root,resolve(sid,root),y[:6],v[:6],'train_'+head))
  official=np.load(reg(OLD/'data/train/official_train.npz'))
  for sid,root,y,v in zip(official['state_ids'],official['root_ids'],official['y'],official['v']):ii.append(add(str(sid),str(root),resolve(str(sid),str(root)),y[:6],v[:6],'train_'+head))
  cohorts['train_'+head]=ii
  for seed in (0,1):shutil.copy2(reg(OLD/f'configs/schedule_{head}_seed{seed}.npz'),E/f'configs/schedule_{head}_seed{seed}.npz')
 for cohort in cfg['evaluation_cohorts']:
  d=np.load(reg(OLD/f'data/evaluation/{cohort}.npz'));ii=[]
  for sid,root,y,v in zip(d['state_ids'],d['root_ids'],d['y'],d['v']):
   sid=str(sid);root=str(root)
   path=combo[sid]['public_path']if cohort=='new_AB_validation'and sid in combo else None
   if cohort=='new_AB_validation':
    vm={str(x['state_index']):x for x in read(EX/'simultaneous_relation_probe_20261017/data/validation/states.json')};m=vm[sid];assert m['root_id']==root;path=m['public_path'];sid='combo_val:'+sid
   else:path=resolve(sid,root)
   ii.append(add(sid,root,path,y[:6],v[:6],cohort))
  cohorts[cohort]=ii
 old16=[r for r in read(reg(EXP/'results/results.json'))if r['source_kind']=='old_source_long_window_retest'];assert len(old16)==16;ii=[]
 for r in old16:
  c=r['candidate'];sid=f"official_retest:{r['root_id']}:{r['frame_id']}:step96"
  ii.append(add(sid,r['root_id'],c['observation_path'],c['y'],c['v'],'AB_heldout16',{'source_frame':r['frame_id'],'stable_window':[32,96],'earlier_physics_audited':True,'previous_readout_training':False}))
 cohorts['AB_heldout16']=ii
 trainroots={rows[i]['root_id']for h in HEADS for i in cohorts['train_'+h]}
 for name,ix in cohorts.items():
  if not name.startswith('train'):assert trainroots.isdisjoint(rows[i]['root_id']for i in ix),(name,'root leakage')
 for r in rows:r['train']=r['root_id']in trainroots
 save(E/'data/public/states.json',rows);save(E/'data/private/labels.json',labelrecords);save(E/'configs/cohorts.json',cohorts)
 shutil.copy2(reg(OLD/'data/private/labels.json'),E/'data/private/official_segments.json')
 protocol={'utc':now(),'scope':'fixed-scene pure-RGB node vs explicit visual-context edge perception feasibility; no world-model training',
 'object_identity':'unchanged current-RGB HSV largest connected component for red/blue/pink, two cameras independently; known closed object set, no temporal association or open-vocabulary claim',
 'input_audit':'Old scene1039 includes15 robot_obs; old64 object z receives scene context and is not pure vision. Old z,robot_obs,PCA and learned representation never used by candidate.',
 'pure_input':['rgb_static','rgb_gripper'],'prohibited_inputs':['robot_obs','scene_obs','simulation identities for localization','segmentation masks from simulator','contacts','3D poses','actions','labels'],
 'endpoints':ROIS,'endpoint_limitation':'fixed camera-role wrist and drawer image regions, not true instance segmentation or general gripper/drawer detection; image-derived 2D region geometry allowed',
 'backbone':'one shared frozen local-cache DINOv2-small for all images and endpoints; full-image patch self-attention means ROI descriptors contain global image context',
 'node':'74D = two camera object mean ROI descriptors projected to32 each + 2x5 image bbox/visibility; fixed visual descriptor retained, no relation-label clustering',
 'edge':'207D = same74 object +128 endpoint2x2 DINO/PCA32 grid +5 relative image geometry; Holding uses wrist region, In uses lower static drawer region',
 'fair_control':'same RGB and frozen backbone,labels,train roots,schedules,batch,updates,optimizer,weights andthreshold. node74->88->1 vs edge207->32->1 both6689parameters; width/input organization differs intentionally. Old robot-conditioned results not causal baseline.',
 'normalization':'single DINO PCA32 fitted only to union of training RGB tokens; each head train-pool statistics; no heldout/C/D fit',
 'heads':HEADS,'arms':ARMS,'seeds':[0,1],'max_fits':8,'updates_each':600,'batch_states':32,'old_pool_states':{'Holding':210,'In':156},'official_training_states':120,'positive_weights':cfg['positive_weights'],
 'schedule':'exact previous augmented24old+8official; identical within head/seed across arms, no sampling change',
 'optimizer':'AdamW lr.001 wd.001 clip1; masked positive-weighted BCE mean per-state valid objects then batch; no new auxiliary loss',
 'initialization':'seed17083000+100*head_index+seed, save initial weights separately per architecture before any training','selection':'fixed600,threshold.5,no hyperparameter search or earlystop; all runs retained',
 'evaluation':'per-color FP/FN and all3, root-level results, official same-segment positive/negative pairs, unseen six-fact label combinations relative to each training pool; model-only object/endpoint swap probes, no new physical counterfactual claims',
 'heldout':'AB_heldout16:4old official roots excluded from prior readout training, earlier physics-audited; only5Holding/7Inpositivefacts and2Inunknown, weak evidence. official_validation120/4roots previously readout-evaluated; C and AB54 retention; D repeatedly seen diagnostic only.',
 'observable_limit':'Single RGB cannot uniquely determine force-based stable contact/support in all cases. Preserve unknown masks; inspect exact-RGB conflicting labels and object visibility strata. Do not relabel failures or claim pure vision sufficient.',
 'no_collection_download_or_T_training':True,'no_formal_model_adoption':True,'SupCon_remains_paused':True}
 save(E/'configs/protocol.json',protocol)
 reg(REP/'source/registry.py');shutil.copy2(REP/'source/registry.py',E/'source/registry.py');shutil.copy2(REP/'source/registry.py',Path(__file__).parent/'registry.py')
 reg(DINO/'model.safetensors');save(E/'configs/source_references.json',{'sha256':refs})
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 frozen={str(p):sha(p)for sub in['source','configs','data/public']for p in(E/sub).rglob('*')if p.is_file()}
 save(E/'configs/pre_encoding_frozen.json',{'utc':now(),'before_encoding_or_training':True,'sha256':frozen})
 save(E/'results/input_audit.json',{'old_scene_robot_channels':15,'old_object_z_inherits_robot_context':True,'candidate_public_schema':['rgb_static','rgb_gripper'],'states_to_encode':len(rows),'unique_RGB_hashes':len({r['RGB_content_sha256']for r in rows}),'training_union_states':sum(r['train']for r in rows),'train_roots':sorted(trainroots),'cohorts':{k:{'states':len(ix),'roots':len({rows[i]['root_id']for i in ix})}for k,ix in cohorts.items()},'no_simulator_localization':True,'manual_fixed_camera_ROIs_disclosed':True})
 print('NODE_EDGE_PREPARED',len(rows),'states, train union',sum(r['train']for r in rows),'ABheldout16',flush=True)
if __name__=='__main__':main()
