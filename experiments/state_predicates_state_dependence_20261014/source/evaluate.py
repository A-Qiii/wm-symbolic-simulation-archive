from experiment import *
import sys,argparse,importlib.util,fcntl,csv,collections
import numpy as np
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def finite_json(x):
 if isinstance(x,dict):return {k:finite_json(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [finite_json(v)for v in x]
 if isinstance(x,float)and not np.isfinite(x):return None
 return x
def csvsave(p,rows):
 with Path(p).open('w',newline='',encoding='utf-8-sig')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in finite_json(r).items()}for r in rows)
def suppress_unknown_binary_metrics(x):
 if isinstance(x,dict):
  if x.get('UNKNOWN',0)>0:
   for k in ['raw_FP','raw_FN','raw_accuracy','positive_recall','FPR','root_macro_raw_accuracy']: 
    if k in x:x[k]=None
  for v in x.values():suppress_unknown_binary_metrics(v)
 elif isinstance(x,list):
  for v in x:suppress_unknown_binary_metrics(v)
def predict_once():
 import torch
 verify_frozen();assert read(E/'results/cache_full.json')['passed'];assert not(E/'results/predictions.npz').exists();assert not(E/'configs/inference_freeze.json').exists()
 selection=read(OLD/'configs/model_selection_frozen.json');runs=[r for r in selection['all_runs']if r['mode']=='full'];frozen={'created_utc':now(),'evaluation_source_sha256':sha(Path(__file__)),'inputs_sha256':sha(E/'data/public/inputs_full.npz'),'data_manifest_sha256':sha(B/'data_manifest.json'),'protocol_sha256':sha(E/'configs/protocol.json'),'selected_models':runs,'representative_run':'full_seed2','copy_rules_sha256':sha(COPY/'configs/rules_frozen.json'),'no_new_selection':True,'training_updates':0,'collection_complete_before_any_new_model_prediction':True};save(E/'configs/inference_freeze.json',frozen);(E/'configs/inference_freeze.json').chmod(0o444)
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');sys.path.insert(0,str(OLD/'source'));from common import reader,encode,logits,digest_model
 from transition import LatentTransition
 rulesmod=loadmod('frozen_copy_rules',COPY/'source/diagnostic.py');rule=read(COPY/'configs/rules_frozen.json')['rules'];m=reader();digest=digest_model(m);pairs=read(E/'data/public/pairs_full.json');z=dict(np.load(E/'data/public/inputs_full.npz'));inputs={k:torch.as_tensor(z[k],device='cuda')for k in ['objects','context','robot']};pred={};latents={};checks=[]
 with torch.inference_mode():
  h,s=encode(m,**inputs);prob=logits(m,h,s).sigmoid();direct=m.probabilities(**inputs);assert float((prob-direct).abs().max())==0;bi=torch.as_tensor(z['pair_before_index'],device='cuda');ai=torch.as_tensor(z['pair_after_index'],device='cuda');hc,sc=h[bi],s[bi];a=torch.as_tensor(z['declared_action_features'],device='cuda');pred['persistence']=prob[bi].cpu().numpy();pred['real_next_observation']=prob[ai].cpu().numpy();latents.update(current_objects=hc.cpu().numpy(),current_scene=sc.cpu().numpy(),actual_next_objects=h[ai].cpu().numpy(),actual_next_scene=s[ai].cpu().numpy())
  for r in runs:
   cp=r['selected'];assert sha(cp['path'])==cp['sha256'];net=LatentTransition('full').cuda().eval().requires_grad_(False);net.load_state_dict(torch.load(cp['path'],map_location='cuda',weights_only=False)['model']);hp,sp=net(hc,sc,a);pred[r['run']]=logits(m,hp,sp).sigmoid().cpu().numpy();latents[r['run']+'_objects']=hp.cpu().numpy();latents[r['run']+'_scene']=sp.cpu().numpy();checks.append({'run':r['run'],'checkpoint_sha256':sha(cp['path']),'input_keys':['current_objects','current_scene','declared_actions']});del net
 copy,known,ca=rulesmod.predict(pred['persistence'],pairs,rule);pred['copy_effect']=copy;assert digest_model(m)==digest
 np.savez_compressed(E/'results/predictions.npz',**pred);np.savez_compressed(E/'results/latents.npz',**latents);np.savez_compressed(E/'results/copy_known.npz',known=known);save(E/'results/copy_input_audit.json',ca);save(E/'results/inference.json',{'created_utc':now(),'passed':True,'model_runs':checks,'candidate_digest_unchanged':digest,'windows_encoded':len(h),'pairs':len(pairs),'no_private_targets_loaded_in_prediction_phase':True,'no_training':True,'no_new_C_model_selection':True,'prediction_sha256':sha(E/'results/predictions.npz')});print('FROZEN_PREDICTION_COMPLETE',len(pairs),flush=True)
def analyze():
 verify_frozen();inf=read(E/'results/inference.json');assert inf['passed'];assert sha(E/'results/predictions.npz')==inf['prediction_sha256'];sys.path.insert(0,str(OLD/'source'));oldmetric=loadmod('frozen_metric_implementation',OLD/'source/evaluate.py');known=oldmetric.known
 pred=dict(np.load(E/'results/predictions.npz'));latent=dict(np.load(E/'results/latents.npz'));t=dict(np.load(E/'data/_private/supervision_full.npz'));pairs=read(E/'data/public/pairs_full.json');manifest=read(B/'data_manifest.json');priv={r['pair_id']:r for r in manifest['private_pairs']};roots=read(B/'roots_manifest.json');rmap={r['root_id']:r for r in roots};y=t['after_y'].astype(bool);v=t['after_valid'];yb=t['before_y'].astype(bool);vb=t['before_valid'];n=len(pairs);beforegood=(known(pred['persistence'],[.5]*3)&((pred['persistence']>=.5)==yb)&vb).all(1);aftergood=(known(pred['real_next_observation'],[.5]*3)&((pred['real_next_observation']>=.5)==y)&v).all(1);target=np.zeros_like(v)
 for i,r in enumerate(pairs):
  if r['skill']in ['open_drawer','close_drawer']:target[i,6]=True
  else:
   c=r['declared_arguments']['color']if r['skill'].startswith('lift_')else rmap[r['root_id']]['target_color'];j=COLORS.index(c);target[i,[j,j+3]]=True
 masks={'all7':np.ones_like(v),'changed':(yb!=y)&vb,'unchanged':(yb==y)&vb,'target_entity':target,'other_entity':~target};sets={'AB':np.array([r['split']=='train'for r in pairs]),'C':np.array([r['split']=='dev'for r in pairs]),'C_tune':np.array([r['dev_role']=='tune'for r in pairs]),'C_calibration':np.array([r['dev_role']=='calibration'for r in pairs])};subsets={'all':np.ones(n,bool),'before_readout_correct':beforegood,'before_readout_wrong_or_UNKNOWN':vb.all(1)&~beforegood,'before_labels_uncertain':~vb.all(1),'both_observed_readouts_correct':beforegood&aftergood}
 for group in sorted({priv[r['pair_id']]['outcome_group']for r in pairs}):subsets['outcome:'+group]=np.array([priv[r['pair_id']]['outcome_group']==group for r in pairs])
 for fam in ['close','lift','place']:subsets['family:'+fam]=np.array([priv[r['pair_id']]['family']==fam for r in pairs])
 for role in ['positive_reference','core_state_dependence','nochange_semantics']:subsets['role:'+role]=np.array([priv[r['pair_id']]['contrast_role']==role for r in pairs])
 results={};rows=[];detail=[];rootrows=[];latmetrics={};currentstats={};attribution=[]
 for sn,sel in sets.items():
  currentstats[sn]={'pairs':int(sel.sum()),'before_all7_correct':int(beforegood[sel].sum()),'after_all7_correct':int(aftergood[sel].sum()),'before_labels_uncertain':int((~vb[sel].all(1)).sum()),'after_labels_uncertain':int((~v[sel].all(1)).sum()),'before_UNKNOWN':int((~known(pred['persistence'][sel],[.5]*3)&vb[sel]).sum()),'after_UNKNOWN':int((~known(pred['real_next_observation'][sel],[.5]*3)&v[sel]).sum())}
 for name,pp in pred.items():
  kk=known(pp,[.5]*3);correct=kk&((pp>=.5)==y)&v;full=correct.all(1);results[name]={}
  if name=='copy_effect':assert np.array_equal(kk,np.load(E/'results/copy_known.npz')['known'])
  for sn,sel in sets.items():
   results[name][sn]={}
   for sub,mask in subsets.items():
    out=oldmetric.subset_metrics(pp,y,v,np.flatnonzero(sel&mask),pairs,masks);suppress_unknown_binary_metrics(out);results[name][sn][sub]=out
    for mk,q in out['mask_metrics'].items():rows.append({'model':name,'set':sn,'subset':sub,'pairs':out['pairs'],'roots':out['roots'],'all7_known_correct_pairs':out['all7_known_correct_pairs'],'all_pairs_correct_roots':out['all_pairs_correct_roots'],'mask':mk,**q})
   err=sel&~full
   attribution.append({'model':name,'set':sn,'prediction_wrong_pairs':int(err.sum()),'wrong_with_before_reader_error':int((err&~beforegood&vb.all(1)).sum()),'wrong_before_correct_after_observed_reader_wrong':int((err&beforegood&~aftergood).sum()),'wrong_both_observed_readouts_correct':int((err&beforegood&aftergood).sum()),'before_label_uncertain':int((err&~vb.all(1)).sum()),'interpretation':'descriptive partition, not causal attribution'})
   for rd in results[name][sn]['all']['root_details']:rootrows.append({'model':name,'set':sn,'root_id':rd['root_id'],'pairs':rd['pairs'],'all7_correct':rd['all7_known_correct_pairs'],'all_pairs_correct':rd['all_pairs_correct'],'changed_correct':rd['masks']['changed']['strict_known_correct'],'changed_facts':rd['masks']['changed']['facts'],'unchanged_correct':rd['masks']['unchanged']['strict_known_correct'],'unchanged_facts':rd['masks']['unchanged']['facts'],'UNKNOWN':rd['masks']['all7']['UNKNOWN']})
  for i,r in enumerate(pairs):
   a=priv[r['pair_id']];detail.append({'model':name,'pair_id':r['pair_id'],'root_id':r['root_id'],'episode_id':r['episode_id'],'split':r['split'],'dev_role':r['dev_role'],'skill':r['skill'],'condition':a['condition'],'contrast_role':a['contrast_role'],'primitive_status':a['primitive']['status'],'primitive_reason':a['primitive'].get('reason'),'outcome_group':a['outcome_group'],'all7_correct':bool(full[i]),'before_readout_correct':bool(beforegood[i]),'actual_after_readout_correct':bool(aftergood[i]),'probabilities':[float(x)if np.isfinite(x)else None for x in pp[i]],'prediction_ternary':[int(pp[i,j]>=.5)if kk[i,j]else'UNKNOWN'for j in range(7)],'truth_after':y[i].astype(int).tolist(),'valid_after':v[i].tolist(),'error_columns':np.flatnonzero((~correct[i])&v[i]).tolist(),'UNKNOWN_columns':np.flatnonzero(~kk[i]&v[i]).tolist()})
  if name.startswith('full_'):
   latmetrics[name]={}
   for sn,sel in sets.items():
    hp=latent[name+'_objects'][sel];sp=latent[name+'_scene'][sel];latmetrics[name][sn]={'object_MAE':float(np.abs(hp-latent['actual_next_objects'][sel]).mean()),'scene_MAE':float(np.abs(sp-latent['actual_next_scene'][sel]).mean()),'outside_GRU_range':float((np.abs(np.concatenate([hp.reshape(len(hp),-1),sp],1))>1).mean())}
 # Same-root same-action positive/core paired discriminator, using actual outcomes.
 contrastrows=[];pairidx={r['episode_id']:i for i,r in enumerate(pairs)}
 for r in roots:
  aa=r['branches'][0]['episode_id'];bb=r['branches'][1]['episode_id'];base={'root_id':r['root_id'],'split':r['split'],'family':r['family'],'colour':r['target_color'],'positive_episode':aa,'core_episode':bb}
  if aa not in pairidx or bb not in pairidx:contrastrows.append(base|{'covered_both':False,'same_target_fact_before':None,'target_after_differs':None,'models':{}});continue
  ia,ib=pairidx[aa],pairidx[bb];j=COLORS.index(r['target_color']);cols=[6]if r['family']=='close'else[j]if r['family']=='lift'else[j,j+3];same=bool(np.array_equal(yb[ia,cols],yb[ib,cols]));different=bool(not np.array_equal(y[ia,cols],y[ib,cols]));valid=bool(vb[[ia,ib]][:,cols].all()and v[[ia,ib]][:,cols].all());models={}
  assert pairs[ia]['skill']==pairs[ib]['skill']and pairs[ia]['declared_arguments']==pairs[ib]['declared_arguments']
  for name,pp in pred.items():
   kk=known(pp,[.5]*3);cor=kk&((pp>=.5)==y)&v;models[name]={'both_full7_correct':bool(cor[[ia,ib]].all()),'both_target_effect_facts_correct':bool(cor[[ia,ib]][:,cols].all())}
  contrastrows.append(base|{'covered_both':True,'target_fact_columns':cols,'same_target_fact_before':same,'target_after_differs':different,'target_labels_valid':valid,'positive_actual_status':priv[pairs[ia]['pair_id']]['primitive']['status'],'core_actual_status':priv[pairs[ib]['pair_id']]['primitive']['status'],'models':models})
 save(E/'results/root_contrasts.json',contrastrows);csvsave(E/'results/root_contrasts.csv',contrastrows);csvsave(E/'results/all_metrics.csv',rows);csvsave(E/'results/all_root_metrics.csv',rootrows);csvsave(E/'results/all_pair_predictions.csv',detail);csvsave(E/'results/error_partition.csv',attribution)
 output={'created_utc':now(),'status':'frozen_pretraining_diagnosis_no_tuning','pairs':n,'planned_roots':18,'planned_calls_cap':54,'current_observation_readout':currentstats,'results':results,'latent_metrics':latmetrics,'details':detail,'root_contrast_count':sum(r['covered_both']and r['same_target_fact_before']and r['target_after_differs']and r.get('target_labels_valid',False)for r in contrastrows),'inference_sha256':sha(E/'results/inference.json')};save(E/'results/evaluation.json',finite_json(output))
 print(json.dumps({'readout':currentstats,'all7':{name:{sn:sets_['all']['all7_known_correct_pairs']for sn,sets_ in rr.items()}for name,rr in results.items()},'root_contrasts':output['root_contrast_count'],'error_partition':attribution},ensure_ascii=False),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['predict','analyze'],required=True);a=ap.parse_args();predict_once()if a.phase=='predict'else analyze()
