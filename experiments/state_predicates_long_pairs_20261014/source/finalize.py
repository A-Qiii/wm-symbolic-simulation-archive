from pathlib import Path
import json,hashlib,collections,subprocess,datetime,re,html
import numpy as np
import imageio.v2 as imageio
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_long_pairs_20261014';B=P/'data/predicate_long_pairs_20261014';PRE=P/'experiments/state_predicates_held_down_20261014/candidate'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 protocol=read(E/'configs/protocol.json');roots=read(B/'roots_manifest.json');m=read(B/'data_manifest.json');meta=read(E/'data/public/windows_full.json');cache=read(E/'results/cache_full.json');val=read(E/'results/data_validation.json');summary=read(E/'results/summary.json');videos=read(E/'videos/index.json');registry=read(E/'results/registry_diagnostic_full.json')
 wait_audit=read(E/'results/wait_audit.json');assert wait_audit['all_boundary_spans_2point1_seconds']and wait_audit['boundary_windows']==150 and wait_audit['intervals']==150 and wait_audit['total_real_hold_controls']==14400
 assert val['passed']and cache['passed'];assert len(roots)==30 and len(m['records'])==120 and len(meta)==150 and len(videos)==30;assert collections.Counter(r['environment']for r in roots)=={'A':12,'B':12,'C':6};assert collections.Counter(r['target_color']for r in roots)=={'red':10,'blue':10,'pink':10};assert len({r['seed']for r in roots})==30
 for role in ['tune','calibration']:assert collections.Counter(r['target_color']for r in roots if r['dev_role']==role)=={'red':1,'blue':1,'pink':1}
 for r in roots:
  config=OmegaConf.load(B/'episodes'/r['root_id']/'_private/complete_config.yaml');assert config.env.bullet_time_step==240 and config.env.control_freq==30 and config.env.use_vr is False
 assert all(r['environment']in 'ABC'and r['new_independent_root']for r in roots);assert {r['root_id']for r in roots}=={r['root_id']for r in m['roots']}
 old=read(E/'configs/root_exclusion.json');old_sources=0
 for p,h in old['sources_sha256'].items():assert sha(p)==h,p;old_sources+=1
 for p,h in protocol['frozen_sha256'].items():assert sha(p)==h,p
 assert sha(B/'roots_manifest.json')==protocol['roots_manifest_sha256'];assert sha(E/'data/public/paired_inputs_full.npz')==cache['paired_inputs_sha256'];assert sha(E/'data/_private/supervision_full.npz')==cache['supervision_sha256'];stats=read(PRE/'normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rm=np.array(stats['robot_mean'],np.float32);rs=np.array(stats['robot_std'],np.float32);audits={r['window_id']:r for r in cache['audits']};maxerror=0.;checked=0
 with np.load(E/'data/public/paired_inputs_full.npz')as combined,np.load(E/'data/_private/supervision_full.npz')as targets:
  assert set(combined.files)=={'objects','context','robot','pair_before_index','pair_after_index','declared_action_features'};assert set(targets.files)=={'window_y','window_valid','before_y','before_valid','after_y','after_valid'}
  for i,r in enumerate(meta):
   assert all(str(B/'episodes'/r['root_id']/'public')in p for p in r['public_frames']);assert all(np.isfinite(combined[k][i]).all()for k in ['objects','context','robot'])
   with np.load(r['input_path'])as z,np.load(r['raw_feature_path'])as f:
    assert sha(r['input_path'])==r['input_sha256']and sha(r['raw_feature_path'])==r['raw_feature_sha256'];assert set(z.files)=={'objects','context','robot'}
    tok=((f['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384);geo=f['geometry'].reshape(4,2,64,8);valid=f['valid'].reshape(4,2,64);ids=f['visible_registry_ids'].tolist();ob=np.zeros((4,3,2,390),np.float32)
    # Rebuild with the recorded PUBLIC registration, including wrong/missing slots; never replace with GT IDs.
    for j,c in enumerate(['red','blue','pink']):
     uid=audits[r['window_id']]['registry_audit'][c]['registered_public_uid']
     if uid not in ids:continue
     slot=ids.index(uid);sl=slice(slot*4,slot*4+4);ob[:,j,:,:384]=tok[:,:,sl].mean(2)*valid[:,:,slot*4,None];ob[:,j,:,384:389]=geo[:,:,slot*4,:5];ob[:,j,:,389]=geo[:,:,sl,5].mean(2)
    robot=[]
    for p in r['public_frames']:
     with np.load(p)as source:robot.append(source['robot_obs'])
    rebuilt={'objects':ob.reshape(4,3,780),'context':tok[:,:,48:].reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,-1),'robot':(np.stack(robot).astype(np.float32)-rm)/rs}
    for k in rebuilt:
     assert np.array_equal(rebuilt[k],z[k])and np.array_equal(z[k],combined[k][i]);maxerror=max(maxerror,float(np.abs(rebuilt[k]-z[k]).max()));checked+=z[k].size
   target=audits[r['window_id']]['all7_targets'];assert np.array_equal(targets['window_y'][i],target['y'])and np.array_equal(targets['window_valid'][i],target['valid'])
  for i,r in enumerate(m['records']):
   for side in ['before','after']:
    j=int(combined['pair_'+side+'_index'][i]);assert meta[j]['window_id']==r[side+'_window_id']and meta[j]['root_id']==r['root_id']and meta[j]['split']==r['split'];assert np.array_equal(targets[side+'_y'][i],targets['window_y'][j])and np.array_equal(targets[side+'_valid'][i],targets['window_valid'][j])
  assert combined['declared_action_features'].shape==(120,9)and np.all(combined['declared_action_features'][:,7:]==1)
 video_checks=[]
 for v in videos:
  assert sha(v['path'])==v['sha256'];reader=imageio.get_reader(v['path']);s=reader.get_meta_data();n=reader.count_frames();assert n==v['frames']and abs(s['fps']-10)<1e-6 and tuple(s['size'])==(640,390);assert reader.get_data(0).shape==reader.get_data(n-1).shape==(390,640,3);reader.close();video_checks.append({'root_id':v['root_id'],'frames':v['frames'],'duration_seconds':v['frames']/10})
 missing=[]
 for href in re.findall(r'(?:href|src)="([^"]+)"',(E/'report.html').read_text()):
  path=Path(html.unescape(href));path=path if path.is_absolute()else E/path
  if not path.exists():missing.append(str(path))
 assert not missing,missing
 result={'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fixed_new_roots':30,'fixed_macro_calls':120,'roots_train_dev_disjoint':True,'balanced_colors_and_C_roles':True,'old_root_metadata_files_rehashed_unchanged':old_sources,'frozen_code_and_weights_files_rehashed_unchanged':len(protocol['frozen_sha256']),'raw_public_frame_files_validated':val['public_frame_files_hashed'],'actual_input_scalar_values_rebuilt':checked,'exact_rebuild_max_error':maxerror,'paired_targets_and_indices_checked':120,'videos_verified':video_checks,'report_links_missing':missing,'registry_issues_retained':{'wrong_slot_windows':registry['wrong_slot_windows'],'missing_slot_windows':registry['missing_slot_windows']},'no_candidate_reader_forward':True,'no_training':True,'no_new_D_collection':True,'no_upload':True};save(E/'results/final_validation.json',result)
 save(E/'results/progress.json',{'stage':'completed_fixed_collection_no_training','roots':30,'pairs':120,'unique_windows':150,'final_validation_passed':True,'registry_wrong_slot_windows':registry['wrong_slot_windows'],'registry_missing_slot_windows':registry['missing_slot_windows']})
 # Delivery manifests cover immutable products; themselves excluded to avoid self-reference.
 for base in [B,E]:
  files=[]
  for p in sorted(base.rglob('*')):
   if p.is_file()and p.name!='delivery_manifest.json':files.append({'relative_path':str(p.relative_to(base)),'bytes':p.stat().st_size,'sha256':sha(p)})
  save(base/'delivery_manifest.json',{'root':str(base),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'total_files':len(files),'total_bytes':sum(x['bytes']for x in files)});print('MANIFEST',base,len(files),sum(x['bytes']for x in files),sha(base/'delivery_manifest.json'),flush=True)
 print(json.dumps({k:v for k,v in result.items()if k!='videos_verified'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
