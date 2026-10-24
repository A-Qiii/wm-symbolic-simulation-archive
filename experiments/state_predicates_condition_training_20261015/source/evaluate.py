from core import *
import torch,fcntl,csv,shutil
import trees
def csvsave(path,rows):
 with Path(path).open('w',newline='',encoding='utf-8-sig')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in r.items()}for r in rows)
def metric(p,y,v,ids,meta,masks):
 k=known(p);hat=p>=.5;cor=k&(hat==y)&v;full=cor.all(1);roots=sorted({meta[i]['root_id']for i in ids});out={'pairs':len(ids),'roots':len(roots),'all7':int(full[ids].sum()),'root_macro_all7':float(np.mean([full[[i for i in ids if meta[i]['root_id']==r]].mean()for r in roots]))if roots else None,'root_details':[],'facts':{}}
 def fact(ii,mm):
  vv=v[ii]&mm[ii];cc=cor[ii];kk=k[ii];pp=hat[ii];yy=y[ii];count=int(vv.sum());pos=int((vv&yy).sum());neg=count-pos;tp=int((vv&yy&pp).sum());fp=int((vv&~yy&pp).sum());return {'n':count,'strict_correct':int((vv&cc).sum()),'UNKNOWN':int((vv&~kk).sum()),'positive':pos,'negative':neg,'raw_TP':tp,'raw_FP':fp,'positive_recall':tp/pos if pos else None,'FPR':fp/neg if neg else None,'strict_accuracy':float((vv&cc).sum()/count)if count else None}
 for name,mask in masks.items():out['facts'][name]=fact(ids,mask)
 for name,cols in GROUPS.items():
  mm=np.zeros_like(v);mm[:,cols]=True;out['facts']['predicate_'+name]=fact(ids,mm)
 for r in roots:
  ii=[i for i in ids if meta[i]['root_id']==r];out['root_details'].append({'root_id':r,'pairs':len(ii),'all7':int(full[ii].sum()),'all_correct':bool(full[ii].all()),'facts':{name:fact(ii,mask)for name,mask in masks.items()}})
 out['all_correct_roots']=sum(r['all_correct']for r in out['root_details']);return out
def main():
 q=frozen();sel=read(E/'configs/model_selection_frozen.json');assert sel['all_training_complete']and sel['total_optimizer_updates']==9000;assert not(E/'results/predictions.npz').exists();z=dict(np.load(E/'data/public/inputs.npz'));public=read(E/'data/public/pairs.json');reader,logits,digest_model,_,_,Net=old_modules();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');m=reader();digest=digest_model(m);inputs={k:torch.as_tensor(z[k],device='cuda')for k in ['current_objects','current_scene','declared_actions']};pred={'persistence':z['observed_current_probabilities']};lat={};audit=[]
 # Only public current state/action tensors are used during predictor forward calls.
 with torch.inference_mode():
  for r in sel['runs']:
   reuse={}
   for which in ['selected','final1500']:
    name=r['run']+'_'+which;cp=r[which];assert sha(cp['path'])==cp['sha256']
    if cp['sha256']in reuse:
     old=reuse[cp['sha256']];pred[name]=pred[old].copy();lat[name+'_objects']=lat[old+'_objects'].copy();lat[name+'_scene']=lat[old+'_scene'].copy();continue
    net=Net('full').cuda().eval().requires_grad_(False);net.load_state_dict(torch.load(cp['path'],map_location='cuda',weights_only=False)['model']);h,s=net(**inputs);pred[name]=logits(m,h,s).sigmoid().cpu().numpy();lat[name+'_objects']=h.cpu().numpy();lat[name+'_scene']=s.cpu().numpy();reuse[cp['sha256']]=name;audit.append({'model':name,'checkpoint':cp,'input_keys':list(inputs)});del net
 assert digest_model(m)==digest==sel['reader_digest']
 for cohort in q['cohorts']:pred['tree_'+cohort]=trees.predict(read(E/'models'/('tree_'+cohort+'.json')),z['symbolic_inputs'])
 copy=loadmod('frozen_copy',COPY/'source/diagnostic.py');pred['copy_effect'],kk,ca=copy.predict(z['observed_current_probabilities'],public,read(COPY/'configs/rules_frozen.json')['rules']);assert np.array_equal(known(pred['copy_effect']),kk);save(E/'results/copy_input_audit.json',ca)
 # Reuse previous frozen predictions instead of running old models again.
 op=dict(np.load(OLD/'results/predictions.npz'));npred=dict(np.load(NEW/'results/predictions.npz'))
 for seed in [0,1,2]:pred[f'original600_seed{seed}']=np.concatenate([op[f'full_seed{seed}'],npred[f'full_seed{seed}']])
 np.savez_compressed(E/'results/predictors_before_truth.npz',**pred);save(E/'results/prediction_input_audit.json',{'created_utc':now(),'only_public_tensors_for_predictors':True,'observed_next_not_loaded_yet':True,'models':audit,'predictors_sha256':sha(E/'results/predictors_before_truth.npz'),'reader_digest_unchanged':digest})
 # Future observations/GT loaded only AFTER all predictor output is fixed.
 t=dict(np.load(E/'data/_private/targets.npz'));meta=read(E/'data/_private/evaluation_metadata.json');pred['real_next_observation']=t['observed_next_probabilities'];np.savez_compressed(E/'results/predictions.npz',**pred);np.savez_compressed(E/'results/latent_predictions.npz',**lat);y=t['after_y'].astype(bool);yb=t['before_y'].astype(bool);v=t['after_valid'];vb=t['before_valid'];beforegood=(known(pred['persistence'])&((pred['persistence']>=.5)==yb)&vb).all(1);aftergood=(known(pred['real_next_observation'])&((pred['real_next_observation']>=.5)==y)&v).all(1);N=len(meta);target=np.zeros_like(v)
 for i,r in enumerate(meta):
  if r['skill']in ['open_drawer','close_drawer']:target[i,6]=True
  else:j=COLORS.index(r['planned_colour_evaluation_only']);target[i,[j,j+3]]=True
 masks={'all7':np.ones_like(v),'changed':(yb!=y)&vb,'unchanged':(yb==y)&vb,'target_entity':target,'other_entity':~target};sets={}
 for sn,split,source in [('AB_old','train','old'),('AB_new','train','new'),('AB_all','train',None),('C_old','dev','old'),('C_new','dev','new'),('C_all','dev',None)]:sets[sn]=np.array([r['split']==split and(source is None or r['source']==source)for r in meta])
 for sn,role in [('C_tune','tune'),('C_calibration','calibration')]:sets[sn]=np.array([r['dev_role']==role for r in meta])
 subsets={'all':np.ones(N,bool),'before_readout_correct':beforegood,'before_readout_wrong_or_UNKNOWN':~beforegood,'both_observed_readouts_correct':beforegood&aftergood,'changed_macro':((yb!=y)&vb&v).any(1),'nochange_macro':~((yb!=y)&vb&v).any(1)}
 for status in ['success','refused']:subsets['execution:'+status]=np.array([r['primitive_status']==status for r in meta])
 for f in ['open','close','lift','place']:subsets['family:'+f]=np.array([r['family']==f for r in meta])
 for role in ['positive_reference','core_state_dependence','nochange_semantics']:subsets['role:'+role]=np.array([r['contrast_role']==role for r in meta])
 results={};rows=[];details=[];latmetrics={};errors=[]
 for name,p in pred.items():
  results[name]={};k=known(p);cor=k&((p>=.5)==y)&v;full=cor.all(1)
  for sn,ss in sets.items():
   results[name][sn]={}
   for sub,sm in subsets.items():
    r=metric(p,y,v,np.flatnonzero(ss&sm),meta,masks);results[name][sn][sub]=r
    for mask,mr in r['facts'].items():rows.append({'model':name,'set':sn,'subset':sub,'pairs':r['pairs'],'roots':r['roots'],'all7_correct_pairs':r['all7'],'root_macro_all7':r['root_macro_all7'],'all_correct_roots':r['all_correct_roots'],'mask':mask,**mr})
   bad=ss&~full;errors.append({'model':name,'set':sn,'wrong_pairs':int(bad.sum()),'wrong_with_before_readout_error':int((bad&~beforegood).sum()),'wrong_before_correct_after_readout_wrong':int((bad&beforegood&~aftergood).sum()),'wrong_both_observed_readouts_correct':int((bad&beforegood&aftergood).sum())})
  for i,r in enumerate(meta):details.append({'model':name,'pair_id':r['pair_id'],'root_id':r['root_id'],'source':r['source'],'split':r['split'],'dev_role':r['dev_role'],'skill':r['skill'],'condition':r['condition'],'status':r['primitive_status'],'all7_correct':bool(full[i]),'before_readout_correct':bool(beforegood[i]),'actual_after_readout_correct':bool(aftergood[i]),'prediction':[int(p[i,j]>=.5)if k[i,j]else'UNKNOWN'for j in range(7)],'probabilities':[float(x)if np.isfinite(x)else None for x in p[i]],'truth_after':y[i].astype(int).tolist(),'error_columns':np.flatnonzero(~cor[i]&v[i]).tolist(),'UNKNOWN_columns':np.flatnonzero(~k[i]&v[i]).tolist()})
  if name+'_objects'in lat:
   latmetrics[name]={}
   for sn,ss in sets.items():
    h,s=lat[name+'_objects'][ss],lat[name+'_scene'][ss];latmetrics[name][sn]={'object_MAE':float(np.abs(h-t['next_objects'][ss]).mean()),'scene_MAE':float(np.abs(s-t['next_scene'][ss]).mean()),'outside_GRU_range':float((np.abs(np.concatenate([h.reshape(len(h),-1),s],1))>1).mean())}
 contrasts=[];episodes={r['episode_id']:i for i,r in enumerate(meta)if r['source']=='new'}
 for r in read(NEW/'results/root_contrasts.json'):
  a,b=episodes[r['positive_episode']],episodes[r['core_episode']];cols=r['target_fact_columns'];models={};assert np.array_equal(yb[a,cols],yb[b,cols])and not np.array_equal(y[a,cols],y[b,cols])
  for name,p in pred.items():
   cor=known(p)&((p>=.5)==y)&v;models[name]={'both_target_correct':bool(cor[[a,b]][:,cols].all()),'both_all7_correct':bool(cor[[a,b]].all())}
  contrasts.append({k:r[k]for k in ['root_id','split','family','colour','positive_episode','core_episode','target_fact_columns']}|{'models':models})
 contrast_summary={}
 for name in pred:
  contrast_summary[name]={}
  for split,label in [('train','AB_new'),('dev','C_new')]:
   rr=[r for r in contrasts if r['split']==split];contrast_summary[name][label]={'roots':len(rr),'both_target_correct':sum(r['models'][name]['both_target_correct']for r in rr),'both_all7_correct':sum(r['models'][name]['both_all7_correct']for r in rr)}
 rep=sel['representatives']['aug132']+'_selected';repmetric=results[rep];repcontrast=contrast_summary[rep]['C_new'];representative_ok=repmetric['C_old']['all']['all7']==24 and repmetric['C_new']['all']['all7']==18 and repcontrast['both_target_correct']==6 and repcontrast['both_all7_correct']==6
 stability=all(results[f'aug132_seed{s}_selected']['C_old']['all']['all7']==24 and results[f'aug132_seed{s}_selected']['C_new']['all']['all7']>=17 and contrast_summary[f'aug132_seed{s}_selected']['C_new']['both_target_correct']==6 for s in [0,1,2])
 meanpairs={c:float(np.mean([contrast_summary[f'{c}_seed{s}_final1500']['C_new']['both_all7_correct']/6 for s in [0,1,2]]))for c in ['old96','aug132']};meanold={c:float(np.mean([results[f'{c}_seed{s}_final1500']['C_old']['all']['root_macro_all7']for s in [0,1,2]]))for c in ['old96','aug132']};data_benefit=meanpairs['aug132']>meanpairs['old96'] and meanold['aug132']>=meanold['old96']
 cal=repmetric['C_calibration']['all']['facts'];calgate=all((cal['predicate_'+g]['positive_recall']is None or cal['predicate_'+g]['positive_recall']>=.95)and(cal['predicate_'+g]['FPR']is None or cal['predicate_'+g]['FPR']<=.01)for g in GROUPS)and cal['all7']['UNKNOWN']/cal['all7']['n']<=.05
 tr=results['tree_aug132']['C_new']['all'];tc=contrast_summary['tree_aug132']['C_new'];neuraladvantage=repmetric['C_new']['all']['all7']>=tr['all7']and repcontrast['both_all7_correct']>=tc['both_all7_correct']and(repmetric['C_new']['all']['all7']>tr['all7']or repcontrast['both_all7_correct']>tc['both_all7_correct']);passed=representative_ok and stability and data_benefit and calgate
 decision={'created_utc':now(),'development_only_not_original_guide_acceptance':True,'representative':rep,'representative_correctness_pass':representative_ok,'three_seed_stability_pass':stability,'same1500_update_data_benefit_pass':data_benefit,'representative_Ccal_original_predicate_guard_pass':calgate,'final1500_Cnew_mean_both_all7_root_rate':meanpairs,'final1500_Cold_mean_root_macro_all7':meanold,'strict_neural_advantage_over_fixed_shallow_tree_demonstrated':neuraladvantage,'development_candidate_pass':passed,'no_new_training_after_evaluation':True,'no_D_or_loop':True};save(E/'results/decision.json',decision)
 readout={sn:{'pairs':int(ss.sum()),'before_all7':int(beforegood[ss].sum()),'after_all7':int(aftergood[ss].sum()),'before_UNKNOWN':int((~known(pred['persistence'])[ss]&vb[ss]).sum()),'after_UNKNOWN':int((~known(pred['real_next_observation'])[ss]&v[ss]).sum())}for sn,ss in sets.items()}
 output={'created_utc':now(),'results':results,'details':details,'latent_metrics':latmetrics,'readout':readout,'contrast_summary':contrast_summary,'decision':decision,'sets':{k:int(v.sum())for k,v in sets.items()},'predictions_sha256':sha(E/'results/predictions.npz'),'selection_sha256':sha(E/'configs/model_selection_frozen.json')};save(E/'results/evaluation.json',output);save(E/'results/root_contrasts.json',contrasts);csvsave(E/'results/all_metrics.csv',rows);csvsave(E/'results/all_pair_predictions.csv',details);csvsave(E/'results/error_partition.csv',errors);csvsave(E/'results/root_contrasts.csv',contrasts)
 first=next((r for r in details if r['model']==rep and r['split']=='dev'and not r['all7_correct']),None);save(E/'results/single_replacement_diagnostic.json',{'representative':rep,'first_development_error':first,'all_C_prediction_correct':repmetric['C_all']['all']['all7'],'replace_only_successor_latent_by_actual_next_encoding_same_head_correct':results['real_next_observation']['C_all']['all']['all7'],'C_pairs':42,'future_encoding_only_diagnostic_not_predictor_input':True,'interpretation':'Locates observed failure in successor prediction path when both observed endpoints read correctly; does not uniquely identify latent sufficiency versus transition mapping','corrections_or_additional_training':0})
 if passed:
  dst=E/'candidate';dst.mkdir(exist_ok=False);run=next(r for r in sel['runs']if r['run']==sel['representatives']['aug132']);shutil.copyfile(run['selected']['path'],dst/'transition.pt');save(dst/'candidate.json',{'run':run['run'],'checkpoint':run['selected'],'transition_sha256':sha(dst/'transition.pt'),'frozen_reader':str(CAND/'reader_seed1.pt'),'frozen_reader_sha256':sha(CAND/'reader_seed1.pt'),'scope':'one-step development candidate only','symbolic_tree_comparator':str(E/'models/tree_aug132.json'),'tree_sha256':sha(E/'models/tree_aug132.json'),'neural_advantage_demonstrated':neuraladvantage});save(E/'INDEPENDENT_VALIDATION_PROPOSAL.json',{'status':'proposal_only_not_executed','scope':'one future locked independent one-step validation, no closed loop or training','suggested_roots':9,'design':'3 declared families ×3 target colours, one entirely new independent root per cell; three fixed legal-preparation branches each =>maximum27 diagnostic calls; failures/uncovered retained, no resampling','data_partition':'new D roots only after future user authorization; freeze all model/checkpoint/tree choices and scoring before collection','models':['saved representative neural candidate','fixed augmented AB symbolic tree','frozen original copy-effect','persistence'],'same_contract':'two real cameras, original RGBv1 registry, genuine2.1s four-frame endpoints, fixed declared action and96posthold after every call including refusal; no outcome/future input','report':'all7/UNKNOWN, changed/unchanged, success/refusal, eachroot pairedbothsides, reader separation and latent error; report all failures','interpretation':'fixed controller interface semantics; if shallow tree ties, no evidence neural complexity is needed; neither result demonstrates multistep or full guide acceptance','no_execution_this_round':True})
 print(json.dumps({'sets':output['sets'],'readout':readout,'selected_C':{n:{s:results[n][s]['all']['all7']for s in ['C_old','C_new']}for n in pred if n.endswith('selected')or n.startswith('tree_')},'contrasts_C':{n:d['C_new']for n,d in contrast_summary.items()if n.endswith('selected')or n.startswith('tree_')},'decision':decision},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
