from common import *
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry,WEIGHTS
E=P/'experiments/state_predicates_held_down_20261014';B=P/'data/predicate_held_down_replay_20261014';R=P/'experiments/state_predicates_causal_windows_20261014';COLORS=['red','blue','pink']
def main():
 assert not(E/'results/evaluation.json').exists()and not(E/'data/inputs.npz').exists();protocol=read(E/'configs/protocol.json');integration=read(E/'configs/integration_rule.json')
 for path,h in protocol['frozen_sha256'].items():assert sha(path)==h,path
 assert sha(WEIGHTS/'model.safetensors')==read(R/'results/feature_parity.json')['backbone_sha256'];manifest=read(B/'data_manifest.json');assert len(manifest['records'])==6 and all(r['verification']['within_predefined_tolerance']for r in manifest['episodes'])
 stats=read(O/'data/normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rm=np.array(stats['robot_mean'],np.float32);rs=np.array(stats['robot_std'],np.float32);enc=FrozenVision(P);old=read(O/'data/metadata.json')[0];probe=enc.encode(old['public_inputs'],VisibleRegistry())
 with np.load(old['feature_path'])as z:parity={k:float(np.max(np.abs(probe[k].astype(float)-z[k].astype(float))))for k in ['tokens','geometry','robot']}
 assert parity['tokens']<.02 and parity['geometry']==parity['robot']==0
 arrays={k:[]for k in ['objects','context','robot','y','v']};meta=[];aud=[]
 for r in manifest['records']:
  reg=VisibleRegistry();f=enc.encode(r['public_frames'],reg);fp=E/'data/features'/(r['sample_id']+'.npz');np.savez_compressed(fp,**f);tok=((f['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384);geo=f['geometry'].reshape(4,2,64,8);valid=f['valid'].reshape(4,2,64);ids=f['visible_registry_ids'].tolist();ob=np.zeros((4,3,2,390),np.float32)
  for j in range(3):
   uid=reg.colour_ids.get(j)
   if uid not in ids:continue
   slot=ids.index(uid);sl=slice(slot*4,slot*4+4);vv=valid[:,:,slot*4];ob[:,j,:,:384]=tok[:,:,sl].mean(2)*vv[:,:,None];ob[:,j,:,384:389]=geo[:,:,slot*4,:5];ob[:,j,:,389]=geo[:,:,sl,5].mean(2)
  grid=tok[:,:,48:].reshape(4,2,4,4,384).reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,-1);rob=[]
  for pp in r['public_frames']:
   with np.load(pp)as z:rob.append(z['robot_obs'])
  current={'objects':ob.reshape(4,3,780),'context':grid,'robot':(np.stack(rob).astype(np.float32)-rm)/rs}
  # Current labels and audit-only simulator IDs are accessed after public encoding.
  lab=read(r['labels_private']);last=lab['frames'][-1];cat=np.array(last['labels']['categorical'][:4]);y=np.r_[cat[:3]==6,cat[:3]==3,cat[3]==2].astype(np.float32);vis=np.array([all(max(t['objects_audit'][c]['visible_pixels'].values())>=4 for t in lab['frames'])for c in COLORS]);v=np.r_[np.array(last['labels']['categorical_valid'][:3])&vis,np.array(last['labels']['categorical_valid'][:3])&vis,last['labels']['categorical_valid'][3]].astype(bool);target=COLORS.index(r['target_color']);assert y[target]==1 and last['objects_audit'][r['target_color']]['held'];identity={c:reg.colour_ids.get(j)==last['objects_audit'][c]['object_uid_evaluator_only']for j,c in enumerate(COLORS)}
  for j,c in enumerate(COLORS):
   if vis[j]:assert identity[c],(r['sample_id'],c)
  for k,x in current.items():arrays[k].append(x)
  arrays['y'].append(y);arrays['v'].append(v);meta.append(r|{'target_index':target,'feature_path':str(fp),'feature_sha256':sha(fp),'current_category':cat.tolist(),'visible_all4_by_colour':vis.tolist(),'target_score_valid':bool(v[target]),'TCP_z_change_mm':float((rob[-1][2]-rob[0][2])*1000),'gripper_width_mm':float(rob[-1][6]*1000),'motion':lab['motion_audit_private']});aud.append({'sample_id':r['sample_id'],'entity_identity':identity,'visible_all4':vis.tolist()})
 data={k:np.stack(x)for k,x in arrays.items()};np.savez_compressed(E/'data/inputs.npz',**data);save(E/'data/metadata.json',meta);save(E/'results/cache_audit.json',{'windows':6,'parity_reference':old['feature_path'],'parity_max_errors':parity,'DINO_weights_sha256':sha(WEIGHTS/'model.safetensors'),'only_public_inputs':True,'entity_checks':aud})
 freeze={'created_utc':now(),'before_predictions':True,'metadata_sha256':sha(E/'data/metadata.json'),'arrays_sha256':sha(E/'data/inputs.npz'),'protocol_sha256':sha(E/'configs/protocol.json'),'integration_rule_sha256':sha(E/'configs/integration_rule.json'),'models':protocol['models'],'all_six_sources_retained':True,'eligible_target_windows':sum(r['target_score_valid']for r in meta),'samples':[{'sample_id':r['sample_id'],'root_id':r['root_id'],'anchor_frame':r['anchor_frame'],'history_frame_indices':r['history_frame_indices'],'public_frames':r['public_frames']}for r in meta]};save(E/'configs/evaluation_inputs_frozen.json',freeze);(E/'configs/evaluation_inputs_frozen.json').chmod(0o444)
 del enc;torch.cuda.empty_cache();import fcntl
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');out=[]
 for model in protocol['models']:
  assert sha(model['checkpoint'])==model['checkpoint_sha256'];m=loadmodel(model['checkpoint']);pred=infer(m,data);np.save(E/'results'/f'seed{model["seed"]}_predictions.npy',pred);km=known(pred,model['thresholds']);met=evaluate_predictions(pred,data,model['thresholds']);ii=np.arange(6);jj=np.array([r['target_index']for r in meta]);eligible=data['v'][ii,jj];row={'seed':model['seed'],'checkpoint':model['checkpoint'],'checkpoint_sha256':model['checkpoint_sha256'],'thresholds':model['thresholds'],'inference_batches':1,'metrics':met,'target_eligible':int(eligible.sum()),'target_raw_TP':int(((pred[ii,jj]>=.5)&eligible).sum()),'target_raw_FN':int(((pred[ii,jj]<.5)&eligible).sum()),'target_known_TP':int(((pred[ii,jj]>=.5)&km[ii,jj]&eligible).sum()),'target_known_FN':int(((pred[ii,jj]<.5)&km[ii,jj]&eligible).sum()),'target_UNKNOWN':int((~km[ii,jj]&eligible).sum()),'root_details':[]}
  for i,r in enumerate(meta):row['root_details'].append({'sample_id':r['sample_id'],'root_id':r['root_id'],'target_color':r['target_color'],'target_probability':float(pred[i,jj[i]]),'target_known':bool(km[i,jj[i]]),'target_eligible':bool(eligible[i]),'held_probabilities_RGB':pred[i,:3].tolist(),'all_probabilities':pred[i].tolist(),'all_truth':data['y'][i].tolist(),'all_valid':data['v'][i].tolist(),'all_known':km[i].tolist()})
  out.append(row);del m
 ref=next(r for r in out if r['seed']==integration['prospective_reference_seed']);candidate=ref['target_eligible']==ref['target_known_TP']==6 and all(ref['metrics'][g]['FP']+ref['metrics'][g]['FN']==0 for g in GROUPS)
 result={'scope':'SixpreviouslyselectedexistingCdevelopmentroots; no independenttest, training orrecalibration.','models':out,'prospective_reference_seed':integration['prospective_reference_seed'],'reference_meets_predeclared_limited_candidate_rule':candidate,'no_posthoc_seed_switch':True,'no_threshold_change':True,'new_training_updates':0,'D_or_test_sources':[]};save(E/'results/evaluation.json',result);print(json.dumps({'HELD_DOWN_FROZEN_EVALUATION_COMPLETE':True,'results':[{k:r[k]for k in ['seed','target_eligible','target_raw_TP','target_raw_FN','target_known_TP','target_known_FN','target_UNKNOWN']}for r in out],'prospective_reference_candidate':candidate},ensure_ascii=False))
if __name__=='__main__':main()
