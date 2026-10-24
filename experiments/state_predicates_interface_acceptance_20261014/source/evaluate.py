from common import *
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry,WEIGHTS
import fcntl
E=P/'experiments/state_predicates_interface_acceptance_20261014';B=P/'data/predicate_interface_acceptance_20261014';PRE=P/'experiments/state_predicates_held_down_20261014';COLORS=['red','blue','pink']
def root_metrics(pred,data,roots,taus):
 km=known(pred,taus);hat=pred>=.5;y=data['y'].astype(bool);v=data['v'];out={}
 for name,cols in GROUPS.items():
  rr=[]
  for rid in sorted(set(roots)):
   ii=np.where(np.array(roots)==rid)[0];pp=hat[np.ix_(ii,cols)];yy=y[np.ix_(ii,cols)];vv=v[np.ix_(ii,cols)];kk=km[np.ix_(ii,cols)];rr.append({'root_id':rid,'has_valid_positive':bool((yy&vv).any()),'has_valid_negative':bool((~yy&vv).any()),'raw_FP':bool((pp&~yy&vv).any()),'raw_FN':bool((~pp&yy&vv).any()),'known_FP':bool((pp&~yy&vv&kk).any()),'known_FN':bool((~pp&yy&vv&kk).any()),'UNKNOWN':bool((~kk&vv).any()),'GT_invalid':bool((~vv).any()),'all_group_facts_known_correct':bool(vv.all()and kk.all()and(pp==yy).all())})
  out[name]={'roots':len(rr),'positive_roots':sum(x['has_valid_positive']for x in rr),'negative_roots':sum(x['has_valid_negative']for x in rr),**{k:sum(x[k]for x in rr)for k in ['raw_FP','raw_FN','known_FP','known_FN','UNKNOWN','GT_invalid','all_group_facts_known_correct']},'details':rr}
 return out
def summarize_subset(indices,pred,data,meta):
 i=np.array(indices,dtype=int);sub={k:v[i]for k,v in data.items()};pr=pred[i];km=known(pr,[.5,.5,.5]);correct=((pr>=.5)==sub['y'].astype(bool));complete=(sub['v']&km&correct).all(1);return {'windows':len(i),'distinct_roots':len(set(meta[k]['root_id']for k in i)),'all7_known_correct_windows':int(complete.sum()),'all7_known_correct_window_rate':float(complete.mean())if len(i)else None,'GT_invalid_windows':int((~sub['v'].all(1)).sum()),'any_UNKNOWN_windows':int((~km&sub['v']).any(1).sum()),'metrics':evaluate_predictions(pr,sub,[.5,.5,.5]),'root_metrics':root_metrics(pr,sub,[meta[k]['root_id']for k in i],[.5,.5,.5])}
def main():
 assert not(E/'results/evaluation.json').exists()and not(E/'results/predictions.npy').exists()and not(E/'data/inputs.npz').exists()
 protocol=read(E/'configs/protocol.json');complete=read(E/'results/collection_complete.json');assert complete['roots']==12
 for path,h in protocol['frozen_sha256'].items():assert sha(path)==h,path
 manifest=read(B/'data_manifest.json');rows=manifest['records'];uniques={}
 for r in rows:
  key=r['unique_window_id']
  if key not in uniques:uniques[key]=r.copy()|{'conditions':[r['condition']],'condition_achieved':{r['condition']:r['intended_condition_achieved']}}
  else:uniques[key]['conditions'].append(r['condition']);uniques[key]['condition_achieved'][r['condition']]=r['intended_condition_achieved']
 meta=[];arrays={k:[]for k in ['objects','context','robot','y','v']};audit=[];stats=read(PRE/'candidate/normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rm=np.array(stats['robot_mean'],np.float32);rs=np.array(stats['robot_std'],np.float32);enc=FrozenVision(P)
 old=read(O/'data/metadata.json')[0];probe=enc.encode(old['public_inputs'],VisibleRegistry())
 with np.load(old['feature_path'])as z:parity={k:float(np.max(np.abs(probe[k].astype(float)-z[k].astype(float))))for k in ['tokens','geometry','robot']}
 assert parity['tokens']<.02 and parity['geometry']==parity['robot']==0
 for r in uniques.values():
  assert r['history_frame_indices']==[r['anchor_frame']-d for d in [63,42,21,0]]
  reg=VisibleRegistry();f=enc.encode(r['public_frames'],reg);fp=E/'data/features'/(r['unique_window_id']+'.npz');np.savez_compressed(fp,**f);tok=((f['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384);geo=f['geometry'].reshape(4,2,64,8);valid=f['valid'].reshape(4,2,64);ids=f['visible_registry_ids'].tolist();ob=np.zeros((4,3,2,390),np.float32)
  for j in range(3):
   uid=reg.colour_ids.get(j)
   if uid not in ids:continue
   slot=ids.index(uid);sl=slice(slot*4,slot*4+4);vv=valid[:,:,slot*4];ob[:,j,:,:384]=tok[:,:,sl].mean(2)*vv[:,:,None];ob[:,j,:,384:389]=geo[:,:,slot*4,:5];ob[:,j,:,389]=geo[:,:,sl,5].mean(2)
  grid=tok[:,:,48:].reshape(4,2,4,4,384).reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,-1);robot=[]
  for pp in r['public_frames']:
   with np.load(pp)as z:robot.append(z['robot_obs'])
  public={'objects':ob.reshape(4,3,780),'context':grid,'robot':(np.stack(robot).astype(np.float32)-rm)/rs}
  # Only after public encoding do private labels and identity audits enter scoring.
  lab=read(r['labels_private']);last=lab['frames'][-1];cat=np.array(last['labels']['categorical'][:4]);y=np.r_[cat[:3]==6,cat[:3]==3,cat[3]==2].astype(np.float32);cv=np.array(last['labels']['categorical_valid'][:4],bool);v=np.r_[cv[:3],cv[:3],cv[3]];vis=[[max(t['objects_audit'][c]['visible_pixels'].values())>=4 for c in COLORS]for t in lab['frames']];identity={c:{'registered_public_uid':reg.colour_ids.get(j),'private_uid':last['objects_audit'][c]['object_uid_evaluator_only'],'match':reg.colour_ids.get(j)==last['objects_audit'][c]['object_uid_evaluator_only']}for j,c in enumerate(COLORS)}
  for k,x in public.items():arrays[k].append(x)
  arrays['y'].append(y);arrays['v'].append(v);meta.append(r|{'index':len(meta),'target_index':COLORS.index(r['target_color']),'feature_path':str(fp),'feature_sha256':sha(fp),'current_category':cat.tolist(),'visible_each_frame_by_colour':vis,'public_registry_audit':identity,'current_pixels_by_colour':{c:last['objects_audit'][c]['visible_pixels']for c in COLORS}});audit.append({'window':r['unique_window_id'],'registry':identity,'visibility_does_not_mask_truth':True})
 data={k:np.stack(v)for k,v in arrays.items()};np.savez_compressed(E/'data/inputs.npz',**data);save(E/'data/metadata.json',meta);save(E/'results/cache_audit.json',{'unique_windows':len(meta),'condition_windows':len(rows),'parity':parity,'DINO_sha256':sha(WEIGHTS/'model.safetensors'),'public_only_encoding':True,'entity_checks':audit})
 frozen={'created_utc':now(),'before_predictions':True,'protocol_sha256':sha(E/'configs/protocol.json'),'root_manifest_sha256':sha(B/'roots_manifest.json'),'data_manifest_sha256':sha(B/'data_manifest.json'),'arrays_sha256':sha(E/'data/inputs.npz'),'metadata_sha256':sha(E/'data/metadata.json'),'candidate_checkpoint_sha256':sha(PRE/'candidate/reader_seed1.pt'),'evaluator_source_sha256':sha(Path(__file__)),'thresholds':[.5,.5,.5],'unique_windows':len(meta),'one_batch_for_all_unique_windows':True};save(E/'configs/evaluation_inputs_frozen.json',frozen);(E/'configs/evaluation_inputs_frozen.json').chmod(0o444)
 del enc;torch.cuda.empty_cache();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');model=loadmodel(PRE/'candidate/reader_seed1.pt');pred=infer(model,data);np.save(E/'results/predictions.npy',pred);del model
 km=known(pred,[.5,.5,.5]);hat=pred>=.5;details=[]
 from model import predicate_api
 for i,r in enumerate(meta):
  decoded=predicate_api(pred[i],[.5,.5,.5],COLORS);assert [x['value']!='UNKNOWN'for x in decoded]==km[i][[0,3,1,4,2,5,6]].tolist();truth=data['y'][i].astype(bool);v=data['v'][i];error=(hat[i]!=truth)&v;knownerror=error&km[i]
  details.append({'index':i,'window':r['unique_window_id'],'root_id':r['root_id'],'anchor_frame':r['anchor_frame'],'conditions':r['conditions'],'condition_achieved':r['condition_achieved'],'probabilities':pred[i].tolist(),'truth':truth.tolist(),'GT_valid':v.tolist(),'known':km[i].tolist(),'decoded_api':decoded,'raw_held_conflict':bool(hat[i,:3].sum()>1),'all7_known_correct':bool(v.all()and km[i].all()and not error.any()),'raw_error_columns':np.flatnonzero(error).tolist(),'known_error_columns':np.flatnonzero(knownerror).tolist(),'UNKNOWN_columns':np.flatnonzero(~km[i]&v).tolist(),'registry_audit':r['public_registry_audit'],'current_visibility':r['current_pixels_by_colour']})
 conds={}
 for condition in protocol['conditions']:
  ii=[i for i,r in enumerate(meta)if condition in r['conditions']];ach=[i for i in ii if meta[i]['condition_achieved'][condition]];planned=[s for s in manifest['roots']if condition in s['planned_conditions']];conds[condition]={'planned_roots':len(planned),'available':summarize_subset(ii,pred,data,meta),'achieved':summarize_subset(ach,pred,data,meta),'missing_roots':[s['root_id']for s in planned if not any(meta[i]['root_id']==s['root_id']for i in ii)],'not_achieved_roots':[meta[i]['root_id']for i in ii if i not in ach]}
 roots=[]
 for s in manifest['roots']:
  ii=[i for i,r in enumerate(meta)if r['root_id']==s['root_id']];mandatory=[c for c in s['planned_conditions']if not c.startswith('natural')];mr=[r for r in rows if r['root_id']==s['root_id']and r['condition']in mandatory];coverage=len(mr)==len(mandatory)and all(r['intended_condition_achieved']for r in mr);core=[i for i in ii if any(c in mandatory for c in meta[i]['conditions'])];bad=[details[i]for i in sorted(ii,key=lambda i:meta[i]['anchor_frame'])if details[i]['known_error_columns']or details[i]['UNKNOWN_columns']or not all(details[i]['GT_valid'])];roots.append({'root_id':s['root_id'],'mandatory_physical_coverage_complete':coverage,'mandatory_window_count':len(mr),'planned_mandatory_count':len(mandatory),'mandatory_all7_correct_available_windows':sum(details[i]['all7_known_correct']for i in core),'mandatory_all7_correct_root':bool(coverage and all(details[i]['all7_known_correct']for i in core)),'all_available_unique_windows_correct':all(details[i]['all7_known_correct']for i in ii),'first_consequential_error_or_UNKNOWN':bad[0]if bad else None,'controller_failures':s['controller_failures'],'collection_error':s['collection_error']})
 result={'created_utc':now(),'status':'single_frozen_independent_acceptance_evaluated','seed':1,'checkpoint_sha256':sha(PRE/'candidate/reader_seed1.pt'),'thresholds':[.5,.5,.5],'inference_batches':1,'new_training_updates':0,'unique_window_metrics':summarize_subset(list(range(len(meta))),pred,data,meta),'conditions':conds,'root_outcomes':roots,'details':details,'raw_held_conflict_windows':sum(r['raw_held_conflict']for r in details),'no_visibility_ground_truth_mask':True,'independent_roots':12,'condition_window_count':len(rows),'no_acceptance_retuning':True};save(E/'results/evaluation.json',result);print(json.dumps({'evaluated':True,'unique_windows':len(meta),'condition_windows':len(rows),'all7_correct_unique_windows':result['unique_window_metrics']['all7_known_correct_windows'],'all_mandatory_correct_roots':sum(r['mandatory_all7_correct_root']for r in roots),'held_conflicts':result['raw_held_conflict_windows'],'groups':result['unique_window_metrics']['metrics']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
