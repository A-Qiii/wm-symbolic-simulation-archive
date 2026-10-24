from common import *
from transition import LatentTransition
import fcntl,inspect
def basic(pr,y,v,k):
 n=int(v.sum());hat=pr>=.5;correct=(hat==y)&k;pos=int((v&y).sum());neg=n-pos;fp=int((v&hat&~y).sum());fn=int((v&~hat&y).sum());unknown=int((v&~k).sum())
 return {'facts':n,'positive':pos,'negative':neg,'raw_FP':fp,'raw_FN':fn,'raw_accuracy':float(((hat==y)&v).sum()/n)if n else None,'strict_known_correct':int((correct&v).sum()),'strict_accuracy':float((correct&v).sum()/n)if n else None,'UNKNOWN':unknown,'coverage':(n-unknown)/n if n else None,'positive_recall':(pos-fn)/pos if pos else None,'FPR':fp/neg if neg else None,'Brier':float(((pr-y)**2)[v].mean())if n else None}
def subset_metrics(prob,y,valid,idx,meta,masks):
 kk=known(prob,[.5,.5,.5]);hat=prob>=.5;full=(valid&kk&(hat==y)).all(1);roots=sorted({meta[i]['root_id']for i in idx});out={'pairs':len(idx),'roots':len(roots),'all7_known_correct_pairs':int(full[idx].sum()),'all7_known_correct_pair_rate':float(full[idx].mean())if len(idx)else None,'root_details':[],'mask_metrics':{},'predicate_groups':{}}
 for rid in roots:
  ri=np.array([i for i in idx if meta[i]['root_id']==rid]);entry={'root_id':rid,'pairs':len(ri),'all7_known_correct_pairs':int(full[ri].sum()),'all7_known_correct_pair_rate':float(full[ri].mean()),'all_pairs_correct':bool(full[ri].all()),'masks':{}}
  for name,mask in masks.items():entry['masks'][name]=basic(prob[ri],y[ri],valid[ri]&mask[ri],kk[ri])
  out['root_details'].append(entry)
 out['all_pairs_correct_roots']=sum(r['all_pairs_correct']for r in out['root_details']);out['root_macro_all7_pair_rate']=float(np.mean([r['all7_known_correct_pair_rate']for r in out['root_details']]))if roots else None
 for name,mask in masks.items():
  metric=basic(prob[idx],y[idx],valid[idx]&mask[idx],kk[idx]);rr=[r['masks'][name]for r in out['root_details']if r['masks'][name]['facts']];metric.update(roots_with_facts=len(rr),root_macro_strict_accuracy=float(np.mean([q['strict_accuracy']for q in rr]))if rr else None,root_macro_raw_accuracy=float(np.mean([q['raw_accuracy']for q in rr]))if rr else None);out['mask_metrics'][name]=metric
 for name,cols in GROUPS.items():out['predicate_groups'][name]=basic(prob[idx][:,cols],y[idx][:,cols],valid[idx][:,cols],kk[idx][:,cols])
 return out
def main():
 assert not(E/'results/evaluation.json').exists();protocol=read(E/'configs/protocol.json');selection=read(E/'configs/model_selection_frozen.json');assert selection['selected_before_final_C_evaluation']and selection['C_calibration_not_scored']and selection['total_optimizer_updates']==5400
 for p,h in protocol['frozen_sha256'].items():assert sha(p)==h,p
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');m=reader();digest=digest_model(m);meta=read(E/'data/_private/evaluation_metadata.json');windows={r['window_id']:r for r in read(DATA/'data/public/windows_v1.json')};pred={};latent={};checks=[]
 with np.load(E/'data/public/transition_inputs.npz')as z:public={k:torch.as_tensor(z[k],device='cuda')for k in z.files};assert set(public)=={'current_objects','current_scene','declared_actions'}
 with np.load(E/'data/_private/training_targets_and_audit.npz')as t:targets={k:t[k].copy()for k in t.files}
 pred['persistence']=targets['observed_current_probabilities'];pred['real_next_observation']=targets['observed_next_probabilities'];latent['persistence_objects']=public['current_objects'].cpu().numpy();latent['persistence_scene']=public['current_scene'].cpu().numpy()
 for run in selection['all_runs']:
  cp=run['selected'];assert sha(cp['path'])==cp['sha256'];net=LatentTransition(run['mode']).cuda();net.load_state_dict(torch.load(cp['path'],map_location='cuda',weights_only=False)['model']);net.eval().requires_grad_(False)
  with torch.inference_mode():
   hh,ss=net(**public);pp=logits(m,hh,ss).sigmoid();pred[run['run']]=pp.cpu().numpy();latent[run['run']+'_objects']=hh.cpu().numpy();latent[run['run']+'_scene']=ss.cpu().numpy();check={'run':run['run'],'forward_signature':list(inspect.signature(net.forward).parameters),'forbidden_extra_kwarg_rejected':False}
   try:net(**public,future_truth=torch.zeros(1,device='cuda'))
   except TypeError:check['forbidden_extra_kwarg_rejected']=True
   assert check['forbidden_extra_kwarg_rejected']
   altered=dict(public)
   if run['mode']=='action_only':
    altered['current_objects']=public['current_objects'].flip(0)+3.;altered['current_scene']=public['current_scene'].flip(0)-2.
   elif run['mode']=='state_only':altered['declared_actions']=public['declared_actions'].flip(0)+2.
   else:altered['declared_actions']=public['declared_actions'].flip(0)
   ah,ass=net(**altered);diff=float(max((ah-hh).abs().max(),(ass-ss).abs().max()));check['perturbation_max_latent_change']=diff
   if run['mode']!='full':assert diff==0.
   check['latent_MAE_AB_C']={}
   for split in ['train','dev']:
    ii=np.array([r['split']==split for r in meta]);h=latent[run['run']+'_objects'][ii];s=latent[run['run']+'_scene'][ii];check['latent_MAE_AB_C'][split]={'object_MAE':float(np.abs(h-targets['next_objects'][ii]).mean()),'scene_MAE':float(np.abs(s-targets['next_scene'][ii]).mean()),'predicted_coordinates_outside_GRU_range':float((np.abs(np.concatenate([h.reshape(len(h),-1),s],1))>1).mean())}
   checks.append(check)
  del net
 assert digest_model(m)==digest==selection['frozen_candidate_digest'];np.savez_compressed(E/'results/predictions.npz',**pred);np.savez_compressed(E/'results/latent_predictions.npz',**latent);save(E/'results/forward_contract_checks.json',checks)
 y=targets['after_y'].astype(bool);v=targets['after_valid'];yb=targets['before_y'].astype(bool);vb=targets['before_valid'];changed=(yb!=y)&vb;unchanged=(yb==y)&vb;target=np.zeros_like(v);targetobj=np.zeros_like(v);otherobj=np.zeros_like(v);drawer=np.zeros_like(v);drawer[:,6]=True
 for i,r in enumerate(meta):
  if r['skill']in ['open_drawer','close_drawer']:target[i,6]=True;otherobj[i,:6]=True
  else:
   color=r['declared_arguments']['color']if r['skill'].startswith('lift_')else r['intended_root_colour_evaluator_only'];j=COLORS.index(color);target[i,[j,j+3]]=True;targetobj[i,[j,j+3]]=True;otherobj[i,:6]=True;otherobj[i,[j,j+3]]=False
 masks={'all7':np.ones_like(v),'changed':changed,'unchanged':unchanged,'target_entity':target,'other_entity':~target,'target_object':targetobj,'other_object':otherobj,'drawer':drawer};currentgood=(vb&known(pred['persistence'],[.5,.5,.5])&((pred['persistence']>=.5)==yb)).all(1);missing=np.array([any(windows[r['before_window_id']]['missing_entire_window'])or any(windows[r['after_window_id']]['missing_entire_window'])for r in meta]);subsets={'all':np.ones(len(meta),bool),'current_reader_correct':currentgood,'current_reader_wrong_or_UNKNOWN':~currentgood,'missing_at_either_endpoint':missing};sets={'AB_all':np.array([r['split']=='train'for r in meta]),'C_all':np.array([r['split']=='dev'for r in meta]),'C_tune':np.array([r['dev_role']=='tune'for r in meta]),'C_calibration':np.array([r['dev_role']=='calibration'for r in meta])};results={};details={}
 for name,pp in pred.items():
  assert pp.shape==(120,7)and np.isfinite(pp).all();results[name]={};details[name]=[];kk=known(pp,[.5,.5,.5]);hat=pp>=.5
  for setname,sel in sets.items():results[name][setname]={sub:subset_metrics(pp,y,v,np.flatnonzero(sel&subsel),meta,masks)for sub,subsel in subsets.items()}
  for i,r in enumerate(meta):details[name].append({'pair_id':r['pair_id'],'root_id':r['root_id'],'split':r['split'],'dev_role':r['dev_role'],'skill':r['skill'],'probabilities':pp[i].tolist(),'truth':y[i].tolist(),'valid':v[i].tolist(),'raw_error_columns':np.flatnonzero((hat[i]!=y[i])&v[i]).tolist(),'UNKNOWN_columns':np.flatnonzero(~kk[i]&v[i]).tolist(),'all7_known_correct':bool((kk[i]&v[i]&(hat[i]==y[i])).all()),'changed_columns':np.flatnonzero(changed[i]&v[i]).tolist(),'target_entity_columns':np.flatnonzero(target[i]).tolist(),'current_reader_wrong_or_UNKNOWN':bool(not currentgood[i]),'missing_at_either_endpoint':bool(missing[i])})
 aggregate={}
 for mode in ['full','action_only','state_only']:
  names=[r['run']for r in selection['all_runs']if r['mode']==mode];aggregate[mode]={}
  for setname in sets:
   a={};mapping={'changed_root_macro_strict_accuracy':lambda x:x['mask_metrics']['changed']['root_macro_strict_accuracy'],'unchanged_root_macro_strict_accuracy':lambda x:x['mask_metrics']['unchanged']['root_macro_strict_accuracy'],'all7_pair_root_macro_rate':lambda x:x['root_macro_all7_pair_rate'],'target_entity_root_macro_strict_accuracy':lambda x:x['mask_metrics']['target_entity']['root_macro_strict_accuracy'],'other_entity_root_macro_strict_accuracy':lambda x:x['mask_metrics']['other_entity']['root_macro_strict_accuracy'],'UNKNOWN_facts':lambda x:x['mask_metrics']['all7']['UNKNOWN']}
   for key,fn in mapping.items():vals=[fn(results[n][setname]['all'])for n in names];a[key]={'values_in_seed_order':vals,'mean':float(np.mean(vals)),'std_across_seeds':float(np.std(vals,ddof=1))}
   aggregate[mode][setname]=a
 full=aggregate['full']['C_calibration'];baseline={}
 for mode in ['persistence','action_only','state_only']:
  if mode=='persistence':q=results[mode]['C_calibration']['all'];baseline[mode]={'changed':q['mask_metrics']['changed']['root_macro_strict_accuracy'],'unchanged':q['mask_metrics']['unchanged']['root_macro_strict_accuracy'],'all7':q['root_macro_all7_pair_rate']}
  else:q=aggregate[mode]['C_calibration'];baseline[mode]={'changed':q['changed_root_macro_strict_accuracy']['mean'],'unchanged':q['unchanged_root_macro_strict_accuracy']['mean'],'all7':q['all7_pair_root_macro_rate']['mean']}
 representative=selection['representative_run_per_mode']['full'];r=results[representative]['C_calibration']['all'];q=r['predicate_groups'];gate=all((x['positive_recall']is None or x['positive_recall']>=.95)and(x['FPR']is None or x['FPR']<=.01)for x in q.values())and r['mask_metrics']['all7']['UNKNOWN']/max(1,r['mask_metrics']['all7']['facts'])<=.05;gain_changed=all(full['changed_root_macro_strict_accuracy']['mean']>x['changed']+1e-12 for x in baseline.values());gain_all7=all(full['all7_pair_root_macro_rate']['mean']>x['all7']+1e-12 for x in baseline.values());unchanged_ok=full['unchanged_root_macro_strict_accuracy']['mean']+1e-12>=max(x['unchanged']for x in baseline.values());decision={'predeclared_criteria_applied':True,'representative_full_run':representative,'representative_C_calibration_predicate_gate':gate,'full_mean_changed_strictly_beats_each_simple_baseline':gain_changed,'full_mean_all7_strictly_beats_each_simple_baseline':gain_all7,'unchanged_mean_not_worse_than_strongest_baseline':unchanged_ok,'worth_one_new_independent_test':bool(gate and gain_changed and gain_all7 and unchanged_ok),'baseline_Ccal_values':baseline,'full_Ccal_values':full,'no_new_D_or_loop_executed':True,'no_additional_tuning_or_budget':True};save(E/'results/decision.json',decision)
 output={'created_utc':now(),'status':'bounded_one_step_development_evaluated_not_generalization_acceptance','selection_sha256':sha(E/'configs/model_selection_frozen.json'),'all120_pairs_retained':True,'subsets_are_diagnostics_only':True,'current_reader_wrong_pairs':[meta[i]['pair_id']for i in np.flatnonzero(~currentgood)],'set_pair_counts':{k:int(x.sum())for k,x in sets.items()},'masks_fact_counts_all120':{k:int((a&v).sum())for k,a in masks.items()},'results':results,'three_seed_aggregates':aggregate,'representatives':selection['representative_run_per_mode'],'decision':decision,'details':details};save(E/'results/evaluation.json',output);print(json.dumps({'sets':output['set_pair_counts'],'current_bad':output['current_reader_wrong_pairs'],'aggregates_C':{k:v['C_all']for k,v in aggregate.items()},'decision':decision},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
