"""Independent arithmetic/interface audit; no neural forward, fitting or simulation."""
from experiment import *
import numpy as np
import collections, ast, inspect, sys

def known(p):
 k=np.isfinite(p)
 conflict=(p[:,:3]>=.5).sum(1)>1
 k[:,:3]&=~(conflict[:,None]&(p[:,:3]>=.5))
 return k

def main():
 q=verify_frozen();checks={};oldfiles=0
 for root,expected in [(OLD,'d15825f77cbf1310d45ab3a5ac15366bb4f734d8183ff3ab134457d17be9f16e'),(COPY,'08395e9b395c8ac546dffac207247fa325456eaf52d52d0bb78e1066b4f4716f')]:
  mf=root/'delivery_manifest.json';assert sha(mf)==expected
  for item in read(mf)['entries']:
   f=root/item['path'];assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256'];assert f.stat().st_mode&0o222==0;oldfiles+=1
 checks['old_training_and_copy_deliveries_unchanged_files']=oldfiles
 inf=read(E/'configs/inference_freeze.json');assert sha(E/'source/evaluate.py')==inf['evaluation_source_sha256'];assert sha(E/'data/public/inputs_full.npz')==inf['inputs_sha256'];assert sha(B/'data_manifest.json')==inf['data_manifest_sha256'];assert sha(E/'configs/protocol.json')==inf['protocol_sha256'];assert sha(COPY/'configs/rules_frozen.json')==inf['copy_rules_sha256']
 cache=read(E/'results/cache_full.json');assert sha(E/'data/_private/supervision_full.npz')==cache['supervision_sha256'];assert sha(E/'results/predictions.npz')==read(E/'results/inference.json')['prediction_sha256']
 roots=read(B/'roots_manifest.json');manifest=read(B/'data_manifest.json');episodes=manifest['episodes'];pairs=read(E/'data/public/pairs_full.json');windows=read(E/'data/public/windows_full.json');wm={w['window_id']:i for i,w in enumerate(windows)};rmap={r['root_id']:r for r in roots};priv={r['pair_id']:r for r in manifest['private_pairs']}
 assert len(roots)==18 and len(episodes)==54 and len(pairs)==54 and len(windows)==108
 assert collections.Counter(r['environment']for r in roots)==dict(A=6,B=6,C=6)
 for split,count in [('train',4),('dev',2)]:
  rr=[r for r in roots if r['split']==split];assert collections.Counter(r['target_color']for r in rr)=={c:count for c in COLORS};assert collections.Counter(r['family']for r in rr)=={f:count for f in ['close','lift','place']}
 assert len({r['root_id']for r in roots})==18 and len({r['seed']for r in roots})==18
 assert collections.Counter(r['root_id']for r in episodes)=={r['root_id']:3 for r in roots}
 for ep in episodes:
  r=rmap[ep['root_id']];br=r['branches'][ep['branch']];assert ep['split']==r['split']and ep['episode_id']==br['episode_id'];assert ep['covered'] and ep['collection_error']is None
  assert [a['skill']for a in ep['preparation_attempts']]==br['preparation_skills'];assert all(a['result']['status']=='success'for a in ep['preparation_attempts'])
 checks['root_split_and_finite_budget_verified']={'AB_roots':12,'C_roots':6,'AB_pairs':36,'C_pairs':18,'calls_each_root':3,'preparation_calls':sum(len(ep['preparation_attempts'])for ep in episodes),'uncovered':0}
 # Independent no-near-duplicate test against frozen exclusion metadata, no D observations.
 priors=[np.asarray(r['layout'])for r in read(E/'configs/root_exclusion.json')['prior_layouts']]
 for r in roots:
  a=np.array([r['layout']['block_'+c]for c in COLORS])
  for b in priors:
   xy=np.linalg.norm(a[:,:2]-b[:,:2],axis=1);yaw=np.abs((a[:,5]-b[:,5]+np.pi)%(2*np.pi)-np.pi)
   assert not(np.all(xy<.012)or(np.all(xy<.020)and np.all(yaw<.25)))
  priors.append(a)
 checks['layout_exclusion_metadata_only']=True
 z=dict(np.load(E/'data/public/inputs_full.npz'));t=dict(np.load(E/'data/_private/supervision_full.npz'));pred=dict(np.load(E/'results/predictions.npz'));ev=read(E/'results/evaluation.json')
 assert set(z)=={'objects','context','robot','pair_before_index','pair_after_index','declared_action_features'}
 oldpairs=read(OLD/'data/public/pairs.json');oldactions=np.load(OLD/'data/public/transition_inputs.npz')['declared_actions'];actionmap={r['skill']:oldactions[i]for i,r in enumerate(oldpairs)}
 for i,r in enumerate(pairs):
  assert np.array_equal(z['declared_action_features'][i],actionmap[r['skill']])
  for endpoint in ['before','after']:
   j=wm[r[endpoint+'_window_id']];assert z['pair_'+endpoint+'_index'][i]==j;assert np.array_equal(t[endpoint+'_y'][i],t['window_y'][j]);assert np.array_equal(t[endpoint+'_valid'][i],t['window_valid'][j]);assert np.array_equal(t[endpoint+'_y'][i],priv[r['pair_id']][endpoint+'_targets']['y'])
 checks['all_action_vectors_exact_old_semantics']=54
 stats=read(CAND/'normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rm=np.array(stats['robot_mean'],np.float32);rs=np.array(stats['robot_std'],np.float32)
 for i,w in enumerate(windows):
  fpath=Path(w['raw_features_path']);assert sha(fpath)==w['raw_features_sha256'];f=dict(np.load(fpath));audit=read(E/'data/_private'/(w['window_id']+'_cache.json'));assert [sha(p)for p in w['public_frames']]==audit['RGB_input_sha256'];assert audit['registry_sha256']==sha(RGB/'source/registry.py')
  g=f['geometry'];vis=g[:,:,:,0,4].astype(bool);tok=(f['local_tokens'].astype(np.float32)-vm)/vs;o=np.zeros((4,3,2,390),np.float32)
  for c in range(3):
   if not vis[:,:,c].any():continue
   o[:,c,:,:384]=tok[:,:,c].mean(2)*vis[:,:,c,None];o[:,c,:,384:389]=g[:,:,c,0,:5];o[:,c,:,389]=g[:,:,c,:,5].mean(2)
  gg=(f['global_grid'].astype(np.float32)-vm)/vs
  assert np.array_equal(o.reshape(4,3,780),z['objects'][i]);assert np.array_equal(gg.reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,3072),z['context'][i]);assert np.array_equal((f['robot_history']-rm)/rs,z['robot'][i])
 checks['frozen_normalization_recomputed_exact_windows']=108
 y=t['after_y'].astype(bool);yb=t['before_y'].astype(bool);v=t['after_valid'];vb=t['before_valid'];assert v.all()and vb.all();assert ((pred['persistence']>=.5)==yb).all()and known(pred['persistence']).all();assert ((pred['real_next_observation']>=.5)==y).all()and known(pred['real_next_observation']).all()
 beforegood=(known(pred['persistence'])&((pred['persistence']>=.5)==yb)&vb).all(1);aftergood=(known(pred['real_next_observation'])&((pred['real_next_observation']>=.5)==y)&v).all(1)
 target=np.zeros_like(v)
 for i,r in enumerate(pairs):
  if r['skill']in ['open_drawer','close_drawer']:target[i,6]=True
  else:
   c=r['declared_arguments']['color']if r['skill'].startswith('lift_')else rmap[r['root_id']]['target_color'];j=COLORS.index(c);target[i,[j,j+3]]=True
 masks={'all7':np.ones_like(v),'changed':(yb!=y)&vb,'unchanged':(yb==y)&vb,'target_entity':target,'other_entity':~target}
 sets={'AB':np.array([r['split']=='train'for r in pairs]),'C':np.array([r['split']=='dev'for r in pairs]),'C_tune':np.array([r['dev_role']=='tune'for r in pairs]),'C_calibration':np.array([r['dev_role']=='calibration'for r in pairs])}
 subsets={'all':np.ones(len(pairs),bool),'before_readout_correct':beforegood,'before_readout_wrong_or_UNKNOWN':vb.all(1)&~beforegood,'before_labels_uncertain':~vb.all(1),'both_observed_readouts_correct':beforegood&aftergood}
 for group in {a['outcome_group']for a in manifest['private_pairs']}:subsets['outcome:'+group]=np.array([priv[r['pair_id']]['outcome_group']==group for r in pairs])
 for f in ['close','lift','place']:subsets['family:'+f]=np.array([priv[r['pair_id']]['family']==f for r in pairs])
 for role in ['positive_reference','core_state_dependence','nochange_semantics']:subsets['role:'+role]=np.array([priv[r['pair_id']]['contrast_role']==role for r in pairs])
 counts=0
 for name,p in pred.items():
  k=known(p);cor=k&((p>=.5)==y)&v;full=cor.all(1)
  for sn,si in sets.items():
   for sub,sm in subsets.items():
    idx=si&sm;r=ev['results'][name][sn][sub];assert r['pairs']==int(idx.sum());assert r['all7_known_correct_pairs']==int(full[idx].sum())
    for mask,mm in masks.items():
     vv=v&mm&idx[:,None];q=r['mask_metrics'][mask];assert q['facts']==int(vv.sum());assert q['strict_known_correct']==int((vv&cor).sum());assert q['UNKNOWN']==int((vv&~k).sum());counts+=1
    for rd in r['root_details']:
     ii=idx&np.array([x['root_id']==rd['root_id']for x in pairs]);assert rd['pairs']==ii.sum();assert rd['all7_known_correct_pairs']==full[ii].sum();assert rd['all_pairs_correct']==bool(full[ii].all())
 checks['independently_recounted_model_set_subset_masks']=counts
 epid={r['episode_id']:i for i,r in enumerate(pairs)}
 for r in read(E/'results/root_contrasts.json'):
  a,b=epid[r['positive_episode']],epid[r['core_episode']];cols=r['target_fact_columns'];assert r['same_target_fact_before']==np.array_equal(yb[a,cols],yb[b,cols]);assert r['target_after_differs'] and not np.array_equal(y[a,cols],y[b,cols]);assert pairs[a]['skill']==pairs[b]['skill']and pairs[a]['declared_arguments']==pairs[b]['declared_arguments']
  for name,p in pred.items():
   cor=known(p)&((p>=.5)==y)&v;assert r['models'][name]['both_full7_correct']==bool(cor[[a,b]].all());assert r['models'][name]['both_target_effect_facts_correct']==bool(cor[[a,b]][:,cols].all())
 checks['matched_root_contrasts_independently_checked']=18
 # Inspect forward signature, no model construction or GPU call.
 tree=ast.parse((OLD/'source/transition.py').read_text());fn=next(x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name=='forward');assert [x.arg for x in fn.args.args]==['self','current_objects','current_scene','declared_actions']
 et=ast.parse((E/'source/evaluate.py').read_text());pf=next(x for x in et.body if isinstance(x,ast.FunctionDef)and x.name=='predict_once');literals=[x.value for x in ast.walk(pf)if isinstance(x,ast.Constant)and isinstance(x.value,str)];assert not any(x=='_private'or x.startswith('supervision')for x in literals)
 checks['no_private_targets_in_prediction_function']=True;checks['forward_interface_three_public_tensors']=True
 assert not list((E/'source').glob('train*'));assert read(E/'results/inference.json')['no_training'];checks['new_training_updates']=0
 first=next(r for r in ev['details']if r['model']=='full_seed2'and r['split']=='dev'and not r['all7_correct'])
 diag={'scope':'Offline diagnostic only; future observation is NEVER a prediction input. No new neural inference or correction.', 'single_replacement':'Replace predicted successor latent by actual next-observation encoded latent, retain exactly the same frozen symbol heads.', 'C_before_reader_correct':18,'C_actual_next_reader_correct':18,'full_seed2_before_replacement_all7':4,'after_replacement_all7':18,'first_error_in_order':first,'interpretation':'All 14 C transition-path errors remain when both actual observation endpoints read correctly. This localizes observed symbolic failure to successor prediction path; it does not uniquely prove latent representation sufficiency or a physical causal mechanism.', 'model_or_rule_corrections':0}
 save(E/'results/single_replacement_diagnostic.json',diag)
 result={'created_utc':now(),'passed':True,'checks':checks,'neural_forward_calls':0,'training_updates':0,'prior_data_validation_sha256':sha(E/'results/data_validation.json'),'prediction_sha256':sha(E/'results/predictions.npz')};save(E/'results/final_validation.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
