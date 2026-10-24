"""Frozen scorer: saved pre-action outputs only; no model inference, updates, or D selection."""
from experiment import *
import numpy as np,csv
GROUPS={'held':[0,1,2],'in':[3,4,5],'open':[6]}
def known(p):
 k=np.isfinite(p);conflict=(p[:,:3]>=.5).sum(1)>1;k[:,:3]&=~(conflict[:,None]&(p[:,:3]>=.5));return k
def csvsave(path,rows):
 if not rows:return
 with Path(path).open('w',newline='',encoding='utf-8-sig')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in r.items()}for r in rows)
def metric(p,y,v,ids,meta,masks):
 k=known(p);hat=p>=.5;cor=k&(hat==y)&v;full=cor.all(1);roots=sorted({meta[i]['root_id']for i in ids})
 def facts(mm):
  vv=v[ids]&mm[ids];kk=k[ids];pp=hat[ids];yy=y[ids];count=int(vv.sum());pos=int((vv&yy).sum());neg=count-pos;tp=int((vv&yy&pp&kk).sum());fp=int((vv&~yy&pp&kk).sum());return {'n':count,'strict_correct':int((vv&cor[ids]).sum()),'UNKNOWN':int((vv&~kk).sum()),'positive':pos,'negative':neg,'known_TP':tp,'known_FP':fp,'raw_TP':int((vv&yy&pp).sum()),'raw_FP':int((vv&~yy&pp).sum()),'positive_recall':tp/pos if pos else None,'FPR':fp/neg if neg else None,'strict_accuracy':float((vv&cor[ids]).sum()/count)if count else None}
 mm=dict(masks)
 for name,cols in GROUPS.items():a=np.zeros_like(v);a[:,cols]=True;mm['predicate_'+name]=a
 for j,name in enumerate(NAMES):a=np.zeros_like(v);a[:,j]=True;mm[name]=a
 rr=[{'root_id':r,'pairs':sum(meta[i]['root_id']==r for i in ids),'all7':int(full[[i for i in ids if meta[i]['root_id']==r]].sum()),'rate':float(full[[i for i in ids if meta[i]['root_id']==r]].mean())}for r in roots]
 return {'pairs':len(ids),'independent_roots':len(roots),'all7':int(full[ids].sum()),'strict_all7_rate':float(full[ids].mean())if len(ids)else None,'root_macro_all7':float(np.mean([r['rate']for r in rr]))if rr else None,'all_correct_roots':sum(r['all7']==r['pairs']for r in rr),'root_details':rr,'facts':{name:facts(mask)for name,mask in mm.items()}}
def main():
 q=frozen();validation=read(E/'results/data_validation.json');assert validation['passed'];m=read(B/'data_manifest.json');assert len(m['episodes'])==84;assert not(E/'results/evaluation.json').exists();roots=read(B/'roots_manifest.json');rmap={r['root_id']:r for r in roots};private={r['pair_id']:r for r in m['private_pairs']};meta=[];pred={};lat={};obsbefore=[];obsafter=[];nh=[];ns=[];beforey=[];aftery=[];beforev=[];afterv=[]
 for pub in m['records']:
  pr=private[pub['pair_id']];r=rmap[pub['root_id']];eid=pub['episode_id'];audit=read(pub['preaction_prediction_record_path']);assert sha(audit['predictions_path'])==audit['predictions_sha256'];arr=dict(np.load(audit['predictions_path']));aft=read(pub['after_observation_record_path']);bo=dict(np.load(audit['observed_path']));ao=dict(np.load(aft['observed_path']));obsbefore.append(bo['probability']);obsafter.append(ao['probability']);nh.append(ao['current_objects']);ns.append(ao['current_scene'])
  for name in ['candidate_seed0','parallel_seed1','parallel_seed2','tree','copy_effect','persistence']:pred.setdefault(name,[]).append(arr[name])
  for name in q['models']:
   for kind in ['objects','scene']:lat.setdefault(name+'_'+kind,[]).append(arr[name+'_'+kind])
  beforey.append(pr['before_targets']['y']);aftery.append(pr['after_targets']['y']);beforev.append(pr['before_targets']['valid']);afterv.append(pr['after_targets']['valid']);meta.append(pub|{'family':pr['family'],'condition':pr['condition'],'role':pr['contrast_role'],'primitive_status':pr['primitive']['status'],'outcome_group':pr['outcome_group'],'target_color':r['target_color'],'initial_drawer':r['initial_drawer'],'branch':int(eid.rsplit('__b',1)[1])})
 pred={k:np.concatenate(v)for k,v in pred.items()};lat={k:np.concatenate(v)for k,v in lat.items()};pred['real_next_observation']=np.concatenate(obsafter);before=np.concatenate(obsbefore);assert np.array_equal(before,pred['persistence']);yb=np.array(beforey,bool);y=np.array(aftery,bool);vb=np.array(beforev,bool);v=np.array(afterv,bool);nextobj=np.concatenate(nh);nextscene=np.concatenate(ns);N=len(meta)
 np.savez_compressed(E/'results/predictions.npz',**pred);np.savez_compressed(E/'results/latent_predictions.npz',**lat);np.savez_compressed(E/'data/_private/evaluation_arrays.npz',before_y=yb,after_y=y,before_valid=vb,after_valid=v,next_objects=nextobj,next_scene=nextscene);save(E/'data/_private/evaluation_metadata.json',meta)
 beforegood=(known(before)&((before>=.5)==yb)&vb).all(1);aftergood=(known(pred['real_next_observation'])&((pred['real_next_observation']>=.5)==y)&v).all(1);target=np.zeros_like(v)
 for i,r in enumerate(meta):
  cols=[6]if r['family']in ['open','close']else[COLORS.index(r['target_color']),3+COLORS.index(r['target_color'])];target[i,cols]=True
 masks={'all7':np.ones_like(v),'changed':(yb!=y)&vb,'unchanged':(yb==y)&vb,'target_entity':target,'other_entity':~target};sets={'all':np.ones(N,bool),'before_readout_correct':beforegood,'before_readout_wrong_or_UNKNOWN':~beforegood,'both_observed_readouts_correct':beforegood&aftergood,'changed_macro':((yb!=y)&vb&v).any(1),'nochange_macro':~((yb!=y)&vb&v).any(1)}
 for field,vals in [('primitive_status',['success','refused','failed','timeout']),('family',['open','lift','place','close']),('role',['regular','state_dependent']),('target_color',COLORS),('initial_drawer',['closed','open'])]:
  for value in vals:sets[field+':'+value]=np.array([r[field]==value for r in meta])
 results={};rows=[];details=[];latentmetrics={};partitions=[]
 for name,p in pred.items():
  k=known(p);cor=k&((p>=.5)==y)&v;full=cor.all(1);results[name]={sn:metric(p,y,v,np.flatnonzero(ss),meta,masks)for sn,ss in sets.items()}
  for sn,mm in results[name].items():
   for mask,mr in mm['facts'].items():rows.append({'model':name,'subset':sn,'pairs':mm['pairs'],'independent_roots':mm['independent_roots'],'all7_correct_pairs':mm['all7'],'root_macro_all7':mm['root_macro_all7'],'mask':mask,**mr})
  for i,r in enumerate(meta):details.append({'model':name,'pair_id':r['pair_id'],'episode_id':r['episode_id'],'root_id':r['root_id'],'family':r['family'],'colour':r['target_color'],'condition':r['condition'],'status':r['primitive_status'],'all7_correct':bool(full[i]),'before_readout_correct':bool(beforegood[i]),'actual_after_readout_correct':bool(aftergood[i]),'prediction':[int(p[i,j]>=.5)if k[i,j]else'UNKNOWN'for j in range(7)],'probabilities':[float(x)if np.isfinite(x)else None for x in p[i]],'truth_before':yb[i].astype(int).tolist(),'truth_after':y[i].astype(int).tolist(),'error_columns':np.flatnonzero(~cor[i]&v[i]).tolist(),'UNKNOWN_columns':np.flatnonzero(~k[i]&v[i]).tolist()})
  partitions.append({'model':name,'wrong_pairs':int((~full).sum()),'wrong_with_before_readout_error':int((~full&~beforegood).sum()),'wrong_before_correct_after_readout_wrong':int((~full&beforegood&~aftergood).sum()),'wrong_both_observed_readouts_correct':int((~full&beforegood&aftergood).sum())})
  if name in q['models']:
   h,s=lat[name+'_objects'],lat[name+'_scene'];coords=np.concatenate([h.reshape(N,-1),s],1);assert coords.shape==(N,384);latentmetrics[name]={}
   for sn,ss in sets.items():
    if not ss.any():continue
    per=np.abs(coords[ss])>1;latentmetrics[name][sn]={'pairs':int(ss.sum()),'object_MAE':float(np.abs(h[ss]-nextobj[ss]).mean()),'scene_MAE':float(np.abs(s[ss]-nextscene[ss]).mean()),'overall384_MAE':float(np.abs(coords[ss]-np.concatenate([nextobj[ss].reshape(int(ss.sum()),-1),nextscene[ss]],1)).mean()),'abs_gt1_coordinates':int(per.sum()),'coordinate_denominator':int(per.size),'abs_gt1_fraction':float(per.mean()),'max_abs_coordinate':float(np.abs(coords[ss]).max())}
 epi={r['episode_id']:i for i,r in enumerate(meta)};contrasts=[]
 for r in roots:
  ci=COLORS.index(r['target_color'])
  for family,branches in q['paired_branches'].items():
   ids=[epi.get(r['root_id']+f'__b{b}')for b in branches];cols=[6]if family=='close'else[ci]if family=='lift'else[ci,ci+3];covered=all(i is not None for i in ids);row={'root_id':r['root_id'],'family':family,'colour':r['target_color'],'initial_drawer':r['initial_drawer'],'branches':branches,'target_columns':cols,'covered':covered,'actual_same_target_before':None,'actual_different_target_after':None,'models':{}}
   if covered:
    a,b=ids;row['actual_same_target_before']=bool(np.array_equal(yb[a,cols],yb[b,cols]));row['actual_different_target_after']=bool(not np.array_equal(y[a,cols],y[b,cols]));row['all_endpoint_facts_valid']=bool(v[ids].all()and vb[ids].all())
    for name,p in pred.items():cor=known(p)&((p>=.5)==y)&v;row['models'][name]={'both_target_correct':bool(cor[ids][:,cols].all()),'both_all7_correct':bool(cor[ids].all())}
   contrasts.append(row)
 cs={}
 for name in pred:
  cs[name]={}
  for fam in ['all','close','lift','place']:
   rr=[r for r in contrasts if fam=='all'or r['family']==fam];covered=[r for r in rr if r['covered']];cs[name][fam]={'planned_root_family_contrasts':len(rr),'covered_contrasts':len(covered),'independent_roots':len(set(r['root_id']for r in rr)),'both_target_correct':sum(r['models'][name]['both_target_correct']for r in covered),'both_all7_correct':sum(r['models'][name]['both_all7_correct']for r in covered),'denominator_includes_uncovered':True}
 readout={'before':metric(before,yb,vb,np.arange(N),meta,{'all7':np.ones_like(vb)}),'after':metric(pred['real_next_observation'],y,v,np.arange(N),meta,{'all7':np.ones_like(v)}),'paired_both_readouts_correct':int((beforegood&aftergood).sum())};rep=results['candidate_seed0']['all'];facts=rep['facts'];coverage=N==84 and v.all()and vb.all();guard=all(facts['predicate_'+g]['positive_recall']is not None and facts['predicate_'+g]['positive_recall']>=.95 and facts['predicate_'+g]['FPR']is not None and facts['predicate_'+g]['FPR']<=.01 for g in GROUPS)and facts['all7']['UNKNOWN']/facts['all7']['n']<=.05;symbol=rep['strict_all7_rate']>=.95 and rep['root_macro_all7']>=.95;paired=cs['candidate_seed0']['all']['both_all7_correct']/36>=.95 and all(cs['candidate_seed0'][f]['both_all7_correct']>=11 for f in ['close','lift','place'])
 decision={'created_utc':now(),'representative':'candidate_seed0','candidate_sha256':q['candidate_hash'],'coverage_pass':bool(coverage),'predicate_guard_pass':bool(guard),'strict_all7_and_root_macro_pass':bool(symbol),'paired_state_dependence_pass':bool(paired),'bounded_one_step_evidence_supported':bool(coverage and guard and symbol and paired),'full_guide_acceptance':False,'multistep_evidence':False,'model_reselection':False,'corrections_or_retests':0,'training_updates':0};save(E/'results/decision.json',decision)
 output={'created_utc':now(),'candidate_frozen_before_D':True,'scoring_only_saved_preaction_predictions':True,'models':results,'readout':readout,'paired_summary':cs,'latent_metrics':latentmetrics,'error_partitions':partitions,'decision':decision,'metric_definitions':{'strict_all7':'all seven facts valid, known and correct; UNKNOWN is wrong','root_macro':'mean over independent roots of the fraction of correct covered single-step calls in that root; coverage gate additionally requires all84','paired':'both predefined branches correct, same root; 36root-family contrasts are NOT36 independent roots','latent':'MAE to real successor frozen encoding; abs>1 count divided by pairs*384 (3objects*96+scene96), no clipping','positive_recall':'known true-positive predictions / valid positive truths; also rawTP supplied','readout':'before and after observation encoding use respective real images; actual next is evaluation reference, never pre-action input'}}
 save(E/'results/evaluation.json',output);save(E/'results/pair_details.json',details);save(E/'results/root_contrasts.json',contrasts);csvsave(E/'results/all_metrics.csv',rows);csvsave(E/'results/all_pair_predictions.csv',details);csvsave(E/'results/error_partition.csv',partitions);csvsave(E/'results/root_contrasts.csv',contrasts)
 first=next((r for r in details if r['model']=='candidate_seed0'and not r['all7_correct']),None);save(E/'results/single_replacement_diagnostic.json',{'first_error_in_fixed_order':first,'candidate_all7':rep['all7'],'replace_only_successor_latent_by_actual_next_encoding_same_head_all7':readout['after']['all7'],'pairs':N,'diagnostic_only_no_new_inference_or_change':True,'interpretation':'If both observed endpoints are read correctly but predicted successor is wrong, failure lies in the learned successor prediction path. This replacement cannot distinguish latent insufficiency from transition mapping error.'});print(json.dumps({'decision':decision,'all7':{name:r['all']['all7']for name,r in results.items()},'paired':cs,'readout_before':readout['before']['all7'],'readout_after':readout['after']['all7']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
