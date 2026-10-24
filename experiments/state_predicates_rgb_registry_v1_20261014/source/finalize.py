from pathlib import Path
import json,hashlib,datetime,re,html,tempfile
import numpy as np
from registry import extract
from cache import RGBEncoder
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_rgb_registry_v1_20261014';OLD=P/'experiments/state_predicates_long_pairs_20261014';B=P/'data/predicate_long_pairs_20261014';CAND=P/'experiments/state_predicates_held_down_20261014/candidate'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def verify_manifest(p):
 m=read(p);base=Path(m['root']);entries=m['files']if isinstance(m.get('files'),list)else m['entries'];total=0
 for r in entries:
  path=base/r.get('relative_path',r.get('path'));assert path.stat().st_size==r['bytes']and sha(path)==r['sha256'],path;total+=r['bytes']
 return {'manifest':str(p),'manifest_sha256':sha(p),'files_verified':len(entries),'bytes_verified':total,'opaque_integrity_hash_only_no_image_or_prediction_decoding':True}
def main():
 assert not(E/'delivery_manifest.json').exists();scope=read(E/'configs/scope.json');freeze=read(E/'configs/frozen_before_C.json');assert sha(E/'source/registry.py')==freeze['rule_sha256'];assert sha(E/'results/registration_train.json')==freeze['AB_results_sha256'];assert read(E/'results/registration_train.json')['registration_gate_passed']and read(E/'results/registration_dev.json')['registration_gate_passed']
 for p,h in scope['frozen_sha256'].items():assert sha(p)==h,p
 meta=read(E/'data/public/windows_v1.json');pairs=read(E/'data/public/pairs_v1.json');stats=read(CAND/'normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rm=np.array(stats['robot_mean'],np.float32);rs=np.array(stats['robot_std'],np.float32);checked=0;npublic=0;maxerr=0.;ids={r['window_id']:i for i,r in enumerate(meta)}
 with np.load(E/'data/public/paired_inputs_v1.npz')as final,np.load(E/'data/public/visibility_v1.npz')as visibility,np.load(OLD/'data/public/paired_inputs_full.npz')as old,np.load(E/'data/_private/supervision.npz')as targets:
  assert set(final.files)=={'objects','context','robot','pair_before_index','pair_after_index','declared_action_features'};assert np.array_equal(old['context'],final['context'])and np.array_equal(old['robot'],final['robot']);assert sha(E/'data/_private/supervision.npz')==sha(OLD/'data/_private/supervision_full.npz')
  for i,r in enumerate(meta):
   assert r['history_frame_indices']==[r['anchor_frame']-d for d in [63,42,21,0]]and len(set(r['history_frame_indices']))==4;assert np.allclose(r['simulation_times'],np.array(r['history_frame_indices'])/30,rtol=0,atol=1e-10);assert abs(r['simulation_times'][-1]-r['simulation_times'][0]-2.1)<1e-10
   with np.load(r['rgb_regions_path'])as regions,np.load(r['raw_RGB_feature_path'])as f:
    assert sha(r['rgb_regions_path'])==r['rgb_regions_sha256']and sha(r['raw_RGB_feature_path'])==r['raw_RGB_feature_sha256'];robot=[]
    for ti,(pp,src,h)in enumerate(zip(r['public_frames'],r['source_public_frames'],r['sanitized_sha256'])):
     assert str(E/'data/rgb')in pp and sha(pp)==h
     with np.load(pp)as z,np.load(src)as original:
      assert set(z.files)=={'rgb_static','rgb_gripper','robot_obs'}
      for k in z.files:assert np.array_equal(z[k],original[k])
      for cam in ['static','gripper']:
       actual,_=extract(z['rgb_'+cam]);assert np.array_equal(actual,regions[cam][ti])
      robot.append(z['robot_obs']);npublic+=1
    local=(f['local_tokens'].astype(np.float32)-vm)/vs;geo=f['geometry'];v=geo[:,:,:,0,4].astype(bool);objects=np.zeros((4,3,2,390),np.float32)
    for j in range(3):
     if not v[:,:,j].any():continue
     objects[:,j,:,:384]=local[:,:,j].mean(2)*v[:,:,j,None];objects[:,j,:,384:389]=geo[:,:,j,0,:5];objects[:,j,:,389]=geo[:,:,j,:,5].mean(2)
    grid=(f['global_grid'].astype(np.float32)-vm)/vs;rebuilt={'objects':objects.reshape(4,3,780),'context':grid.reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,3072),'robot':(np.stack(robot).astype(np.float32)-rm)/rs}
    for k,a in rebuilt.items():assert np.array_equal(a,final[k][i]);maxerr=max(maxerr,float(np.abs(a-final[k][i]).max()));checked+=a.size
    assert np.array_equal(v.transpose(0,2,1),visibility['visible'][i]);assert np.array_equal(~v.any((0,1)),visibility['missing_entire_window'][i]);assert np.array_equal(v.transpose(0,2,1),r['visible_per_frame_colour_camera'])
  assert len(meta)==150 and len(pairs)==120
  for j,r in enumerate(pairs):
   for side in ['before','after']:
    i=ids[r[side+'_window_id']];assert int(final['pair_'+side+'_index'][j])==int(old['pair_'+side+'_index'][j])==i and meta[i]['root_id']==r['root_id']and meta[i]['split']==r['split'];assert np.array_equal(targets[side+'_y'][j],targets['window_y'][i])and np.array_equal(targets[side+'_valid'][j],targets['window_valid'][i])
   assert np.array_equal(final['declared_action_features'][j],old['declared_action_features'][j])
 # Exercise the actual revised loader's allowlist before it can access a model.
 rejected=[]
 for forbidden in ['seg_static','scene_obs','labels_private','future_rgb']:
  with tempfile.NamedTemporaryFile(suffix='.npz',dir='/tmp')as tmp:
   np.savez_compressed(tmp.name,rgb_static=np.zeros((200,200,3),np.uint8),rgb_gripper=np.zeros((84,84,3),np.uint8),robot_obs=np.zeros(15,np.float32),**{forbidden:np.zeros(1)})
   try:RGBEncoder.__new__(RGBEncoder).encode([tmp.name]);raise RuntimeError('Forbidden member accepted: '+forbidden)
   except AssertionError as ex:assert 'Revised loader rejects'in str(ex);rejected.append(forbidden)
 diag=read(E/'results/reader_diagnostic.json');assert diag['training_updates']==0 and diag['frozen_reader_forward_batches']==2;assert sha(CAND/'reader_seed1.pt')==diag['checkpoint_sha256'];summary=read(E/'results/summary.json');assert summary['old_missing_windows']==summary['old_missing_truly_occluded']==29 and summary['old_missing_visible_misses']==0
 remaining=summary['remaining_errors'];assert len(remaining)==1 and remaining[0]['window_id']=='A_longpair_blue_open_1__t00096';changes={r['window_id']:r for r in read(E/'results/input_changes.json')};assert changes[remaining[0]['window_id']]['objects_max_abs_change']==0
 missing=[]
 for href in re.findall(r'(?:href|src)="([^"]+)"',(E/'report.html').read_text()):
  path=Path(html.unescape(href));path=path if path.is_absolute()else E/path
  if not path.exists():missing.append(str(path))
 assert not missing,missing
 # The original D acceptance is verified as opaque file hashes only, after all
 # rules and diagnostic choices have been frozen. No D result informs a rule.
 prior=[]
 for path in [OLD/'delivery_manifest.json',B/'delivery_manifest.json',P/'experiments/state_predicates_interface_acceptance_20261014/delivery_manifest.json',P/'data/predicate_interface_acceptance_20261014/delivery_manifest.json']:
  prior.append(verify_manifest(path));print('PRESERVED',path.parent.name,prior[-1]['files_verified'],flush=True)
 save(E/'results/prior_integrity.json',prior)
 result={'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rule_versions_attempted':1,'rule_unchanged_since_before_C':True,'sanitized_public_files_checked':npublic,'model_input_members':['rgb_static','rgb_gripper','robot_obs'],'detector_current_image_only_and_regions_recomputed':True,'causal_windows_checked':150,'paired_indices_actions_targets_checked':120,'input_scalar_values_rebuilt':checked,'max_abs_reconstruction_error':maxerr,'context_robot_byte_equal_legacy':True,'supervision_byte_equal_legacy':True,'actual_input_loader_rejected_members':rejected,'frozen_candidate_unchanged':True,'reader_forward_batches':2,'no_training':True,'no_new_D_collection':True,'old_D_used_only_for_opaque_integrity_hash_after_rule_freeze':True,'report_missing_links':missing,'prior_files_integrity_verified':sum(x['files_verified']for x in prior),'remaining_AB_readout_error_retained':remaining[0]['window_id'],'next_step_proposal_only':True};save(E/'results/final_validation.json',result)
 files=[]
 for p in sorted(E.rglob('*')):
  if p.is_file()and p.name!='delivery_manifest.json':files.append({'path':str(p.relative_to(E)),'bytes':p.stat().st_size,'sha256':sha(p)})
 save(E/'delivery_manifest.json',{'root':str(E),'algorithm':'SHA256','files':len(files),'bytes':sum(r['bytes']for r in files),'entries':files});print(json.dumps(result,ensure_ascii=False),flush=True);print('DELIVERY_MANIFEST',len(files),sum(r['bytes']for r in files),sha(E/'delivery_manifest.json'),flush=True)
if __name__=='__main__':main()
