from common import *
import yaml,collections
def main():
 sys.addaudithook(guard);assert not(S/'configs/protocol.json').exists();start=time.monotonic()
 for d in ['configs','data','models','results','figures','logs','cache']:(S/d).mkdir(parents=True,exist_ok=True)
 protocol={'utc':now(),'status':'FROZEN_BEFORE_ANY_FIT','plan':str(Q/'DESIGN.md'),'question':'Can fixed current-observation features decode table-local x and signed region-boundary distance?','feature_columns':['object512','global512','fixed_color_onehot3','public_availability1'],'gt_use':'targets, eligibility masks and offline scoring only; separate privileged_geometry heads deliberately receive signed GT distances','coordinate':'table-local x from calibrated physical mesh; x_world minus table origin, zero native rotation; q=(x+.44)/.88','cuts_m':CUTS.tolist(),'table_bounds_m':[-.44,.44],'valid_margin_m':.005,'fits':6,'updates_per_fit':1500,'total_updates':9000,'batch':64,'optimizer':{'name':'AdamW','lr':.0003,'weight_decay':.0001,'grad_clip':1.},'visual_architectures':{'linear':[1028,1],'mlp':[1028,256,1]},'geometry_architecture':[2,256,3],'visual_loss':'SmoothL1 beta=.005/.88; mean eligible objects per boundary then mean nonempty boundaries','geometry_loss':'Cross entropy on same eligible objects and same sample indices','input_normalization':'training eligible object mean/std for first1024 dims, std min1e-6; onehot and availability unchanged','initialization_seeds':{'linear':[261019900,261019901],'mlp':[261019910,261019911],'geometry':[261019920,261019921]},'selection':'Final update1500 only. Select linear vs MLP separately per E by lowest validation h0 root-weighted MAE, exact tie -> linear; retain/report both. No hyperparameter change or refit. Freeze selection before old_band/gap scoring.','primary':'h0 b0, root-weighted MAE mm, RMSE mm, fixed-region accuracy, error smaller than true boundary margin','secondary':'all boundaries root-cluster paired2000 bootstrap; per-region confusion and margin bins [5,10),[10,20),[20,40),[40,infinity) mm','old_classifier':'reuse saved coverage-trained original 8-way head predictions and old narrow-control predictions; full argmax of 8 categories, non-table output counts wrong','scope':'Exploratory reuse of historically seen A/B only; C/D forbidden, no new collection, no encoder forwards, no RSSM fits','compute_limit_seconds':1800,'configuration_changes_after_evaluation':0}
 save(S/'configs/protocol.json',protocol)
 protected={};manifests={base:read(base/'manifest.json')['sha256']for base in [R,O,Q]}
 def use(path):
  path=Path(path);h=sha(path)
  for base,m in manifests.items():
   if path.is_relative_to(base)and str(path.relative_to(base))in m:assert h==m[str(path.relative_to(base))],str(path)
  protected[str(path)]=h;return path
 for base in [R,O,Q]:use(base/'manifest.json')
 use(Q/'DESIGN.md');evidence=read(use(Q/'design_evidence.json'));recipe=read(use(R/'data/roots_manifest.json'));pair={r['root_id']:r.get('paired_old_root')for r in recipe if r.get('paired_old_root')}
 oldprotocol=read(use(O/'configs/protocol.json'));allrows=read(use(R/'data/data_manifest.json'))['records']+read(use(P/'data/core1200_v1/data_manifest.json'))['records'];rows={(r['episode_id'],int(r['step'])):r for r in allrows if r['root_id']in set(sum(evidence['root_ids_for_proposed_roles'].values(),[]))}
 indices=read(use(R/'data/features/index.json'))['items']+read(use(P/'data/core1200_v1/features/index.json'))['items'];idx={(r['episode_id'],int(r['step'])):r for r in indices if r['root_id']in set(sum(evidence['root_ids_for_proposed_roles'].values(),[]))}
 scenes={};mesh_checks={}
 for env in ['A','B']:
  cfg=yaml.safe_load(use(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{env}.yaml').read_text());tab=cfg['objects']['fixed_objects']['table'];assert np.allclose(tab['initial_orn'],0);mesh=(P/'vendor/calvin/calvin_env/data'/tab['file']).parent.parent/'meshes/base_link.obj';verts=np.array([[float(x)for x in l.split()[1:4]]for l in use(mesh).read_text().splitlines()if l.startswith('v ')])*cfg['global_scaling'];assert np.allclose([verts[:,0].min(),verts[:,0].max()],[-.44,.44]);scenes[env]=np.array(tab['initial_pos']);mesh_checks[env]={'origin':tab['initial_pos'],'rotation':tab['initial_orn'],'scale':cfg['global_scaling'],'bounds':[-.44,.44],'mesh_sha256':sha(mesh)}
 counts=[];members={};provenance=[];snapshots={};mismatch=0;maxdelta=0.;routing_records=0;invalid_total=0
 for split in SPLITS:
  base=source(split);fs=[]
  for seed in [0,1]:
   use(base/f'data/features_seed{seed}_{split}.npz');fs.append(features(seed,split));use(base/f'results/extraction_seed{seed}_{split}.json')
  a=fs[0];b=fs[1]
  for key in ['episode_id','root_id','step','cat','availability']:assert np.array_equal(a[key],b[key]),(split,key)
  for f in fs:
   for key in ['global','object','availability']:assert np.isfinite(f[key]).all()
   assert f['global'].shape==(len(a['cat']),512)and f['object'].shape==(len(a['cat']),3,512)
   assert np.all(f['object'][f['availability']==0]==0)
  members[split]=sorted(set(str(x)for x in a['root_id']));assert members[split]==evidence['root_ids_for_proposed_roles'][split]
  route=dict(np.load(use(base/f'data/routes_{split}.npz')));audit=read(use(base/f'results/routing_{split}.json'));assert audit['passed'];routeidx={(str(e),int(h)):i for i,(e,h)in enumerate(zip(route['episode_id'],route['step']))};auditidx={(x['episode_id'],int(x['h']),x['color']):x for x in audit['audit']}
  n=len(a['cat']);x=np.zeros((n,3),np.float64);valid=np.zeros((n,3),bool);table=np.zeros((n,3),bool);world=np.zeros((n,3,3));h0=np.array([int(h)==0 and str(e).endswith('_b0')for e,h in zip(a['episode_id'],a['step'])]);frames=[]
  for i,(eid,h,rid)in enumerate(zip(a['episode_id'],a['step'],a['root_id'])):
   eid=str(eid);h=int(h);rid=str(rid);assert rid[0]in'AB'and eid.startswith(rid+'_b');row=rows[eid,max(h-1,0)];labpath=use(row['labels_path']);label=read(labpath)['before'if h==0 else'after'];info=read(use(labpath.parent/'endpoint_info_private.json'))['before'if h==0 else'after'];item=idx[eid,h];frames.append(item['public_inputs'][-1]);assert np.array_equal(a['cat'][i],label['categorical'][:4]);ri=routeidx[eid,h];assert np.array_equal(route['availability'][ri],a['availability'][i]);assert np.array_equal(route['weights'][ri].sum(-1)/32,a['availability'][i]);registry=np.load(use(item['path']))['visible_registry_ids']
   if 'valid'in a:assert np.array_equal(a['valid'][i],label['categorical_valid'][:4])
   cal=label['table_calibration'];assert np.allclose(cal['local_x_bounds_m'],[-.44,.44])and np.allclose(cal['local_x_thirds_m'],CUTS);assert cal['mesh_sha256']==mesh_checks[rid[0]]['mesh_sha256']
   for j,c in enumerate(COLORS):
    objectlabel=label['continuous']['objects'][c];pos=np.asarray(objectlabel['position_m']);assert np.allclose(pos,info['scene_info']['movable_objects']['block_'+c]['current_pos'],atol=1e-12);world[i,j]=pos;table[i,j]=a['cat'][i,j]<3;valid[i,j]=bool(label['categorical_valid'][j]);rec=auditidx[eid,h,c];uid=info['scene_info']['movable_objects']['block_'+c]['uid'];chosen=rec.get('chosen_uid',rec.get('chosen_public_uid'));slot=int(route['slots'][ri,j]);assert chosen<0 or chosen==uid;assert slot<0 or int(registry[slot])==chosen;routing_records+=1
    if table[i,j]:
     local=np.asarray(objectlabel['table_local_position_m']);delta=float(np.abs(pos-scenes[rid[0]]-local).max());maxdelta=max(maxdelta,delta);assert delta<1e-10;x[i,j]=local[0];margin=float(np.min(abs(local[0]-CUTS)));assert valid[i,j]==(margin>.005);assert classes(local[0])==a['cat'][i,j]
   if h0[i]:
    reset=Path(row['obs_before']).parent.parent/'root_reset_private.npz';z=dict(np.load(use(reset)));hh=hashlib.sha256()
    for key in sorted(z):hh.update(key.encode());hh.update(z[key].tobytes())
    fingerprint=hh.hexdigest();assert fingerprint not in snapshots,('Duplicate initial physical state',rid,snapshots.get(fingerprint));snapshots[fingerprint]=rid
   provenance.append({'split':split,'index':i,'root_id':rid,'episode_id':eid,'boundary':h,'label_path':str(labpath),'label_side':'before'if h==0 else'after','public_frame':item['public_inputs'][-1]})
  mask=table&valid;invalid_total+=int((table&~valid).sum());np.savez_compressed(S/f'data/targets_{split}.npz',x=x,q=((x-LO)/WIDTH).astype(np.float32),mask=mask,table=table,valid=valid,cat=a['cat'][:,:3],root_id=a['root_id'],episode_id=a['episode_id'],step=a['step'],h0=h0,availability=a['availability'],public_frame=np.array(frames),world=world)
  counts.append({'split':split,'roots':len(members[split]),'boundaries':n,'all_object_records':n*3,'table_valid_objects':int(mask.sum()),'table_invalid_objects':int((table&~valid).sum()),'non_table_objects':int((~table).sum()),'h0_table_objects':int(mask[h0].sum()),'empty_training_boundaries':int((mask.sum(-1)==0).sum()),'eligible_unavailable_objects':int((mask&(a['availability']==0)).sum())})
 for s in SPLITS:
  for t in SPLITS:
   if s!=t:assert not set(members[s])&set(members[t])
 assert len(pair)==40 and not set(pair.values())&set(members['validation']);assert set(pair.values())==set(oldprotocol['members']['train']);assert set(members['validation']).isdisjoint(oldprotocol['members']['train'])
 for seed in [0,1]:
  schedule=np.load(use(O/f'configs/schedule_seed{seed}.npy'));assert schedule.shape==(1500,64)and schedule.min()>=0 and schedule.max()<600;np.save(S/f'configs/schedule_seed{seed}.npy',schedule)
  use(B/f'cells/roots40/models/joint_seed{seed}/final6000.pt')
  for split in ['old_band','gap']:
   for arm in ['control','coverage']:use(R/f'data/predictions_{arm}_seed{seed}_{split}.npz')
 for p in sorted((S/'source').glob('*.py')):protected[str(p)]=sha(p)
 for p in [P/'src/wm_symbolic/calibrated_labels.py',P/'src/wm_symbolic/world_model.py']:use(p)
 save(S/'configs/root_split.json',{'members':members,'paired_old_train_family':pair,'exact_initial_state_hashes':snapshots});save(S/'data/provenance.json',provenance);csvwrite(S/'results/data_counts.csv',counts)
 save(S/'results/preflight.json',{'utc':now(),'passed':True,'seconds':time.monotonic()-start,'counts':counts,'routing_records_checked':routing_records,'identity_errors':0,'table_coordinate_max_difference_m':maxdelta,'geometry_rule_mismatches':0,'table_invalid_objects':invalid_total,'duplicate_physical_roots':0,'mesh_checks':mesh_checks,'new_encoder_forwards':0,'new_fits':0,'C_D_data_access':0})
 for p in [S/'configs/protocol.json',S/'configs/root_split.json',S/'data/provenance.json']+list((S/'data').glob('targets_*.npz'))+list((S/'configs').glob('schedule_*.npy')):protected[str(p)]=sha(p)
 save(S/'configs/inputs_frozen.json',{'utc':now(),'before_any_fit':True,'files':len(protected),'sha256':protected});print('PREFLIGHT_PASS',len(protected),counts,flush=True)
if __name__=='__main__':main()
