from core import *
import torch,collections,ast
def main():
 q=frozen();sel=read(E/'configs/model_selection_frozen.json');ev=read(E/'results/evaluation.json');assert sha(E/'configs/model_selection_frozen.json')==ev['selection_sha256'];assert sha(E/'results/predictions.npz')==ev['predictions_sha256'];z=dict(np.load(E/'data/public/inputs.npz'));t=dict(np.load(E/'data/_private/targets.npz'));meta=read(E/'data/_private/evaluation_metadata.json');p=dict(np.load(E/'results/predictions.npz'));lat=dict(np.load(E/'results/latent_predictions.npz'));checks={}
 for root in [OLD,NEW,COPY]:
  for item in read(root/'delivery_manifest.json')['entries']:
   path=root/item['path'];assert path.stat().st_mode&0o222==0 and sha(path)==item['sha256']
 checks['previous_deliveries_rehashed_unchanged']=sum(read(root/'delivery_manifest.json')['files']for root in [OLD,NEW,COPY])
 assert len(meta)==174 and len({r['pair_id']for r in meta})==174 and len({r['root_id']for r in meta})==48
 for source,split,n,rn in [('old','train',96,24),('new','train',36,12),('old','dev',24,6),('new','dev',18,6)]:
  rr=[r for r in meta if r['source']==source and r['split']==split];assert len(rr)==n and len({r['root_id']for r in rr})==rn
 for rid in {r['root_id']for r in meta}:assert len({(r['split'],r['dev_role'],r['source'])for r in meta if r['root_id']==rid})==1
 ox=np.load(OLD/'data/public/transition_inputs.npz');nx=np.load(NEW/'results/latents.npz');na=np.load(NEW/'data/public/inputs_full.npz')
 for key in ['current_objects','current_scene']:assert np.array_equal(z[key][:120],ox[key])and np.array_equal(z[key][120:],nx[key])
 assert np.array_equal(z['declared_actions'][:120],ox['declared_actions'])and np.array_equal(z['declared_actions'][120:],na['declared_action_features']);assert np.array_equal(z['symbolic_inputs'],np.concatenate([np.where(known(z['observed_current_probabilities']),z['observed_current_probabilities']>=.5,-1),z['declared_actions']],1))
 ot=np.load(OLD/'data/_private/training_targets_and_audit.npz');nt=np.load(NEW/'data/_private/supervision_full.npz')
 for key in ['before_y','before_valid','after_y','after_valid']:assert np.array_equal(t[key][:120],ot[key])and np.array_equal(t[key][120:],nt[key])
 assert np.array_equal(t['next_objects'][120:],nx['actual_next_objects'])and np.array_equal(t['next_scene'][120:],nx['actual_next_scene']);checks['merged_source_arrays_exact']=True
 train=np.array([i for i,r in enumerate(meta)if r['split']=='train']);weights=[]
 for cols in GROUPS.values():
  yy=t['after_y'][train][:,cols];vv=t['after_valid'][train][:,cols];nn=[int((vv&(yy==k)).sum())for k in [0,1]];weights.append([sum(nn)/(2*n)for n in nn])
 assert weights==q['class_weights_common_to_both_cohorts'];checks['common_AB_only_class_weights']=weights
 assert len(sel['runs'])==6 and sum(r['updates']for r in sel['runs'])==9000;totals=0;checkpoint_count=0
 for r in sel['runs']:
  schedule=np.load(E/'runs'/r['run']/'batch_schedule.npz');ii=schedule['global_training_indices'];schedule=schedule['local_indices'];assert schedule.shape==(1500,16)and ((schedule>=0)&(schedule<len(ii))).all();assert all(meta[i]['split']=='train'for i in ii)
  if r['cohort']=='old96':assert len(ii)==96 and all(meta[i]['source']=='old'for i in ii)
  else:assert len(ii)==132
  gen=torch.Generator().manual_seed(10000+r['seed']);pool=torch.empty(0,dtype=torch.long)
  for row in schedule:
   if len(pool)<16:pool=torch.cat([pool,torch.randperm(len(ii),generator=gen)])
   assert np.array_equal(row,pool[:16].numpy());pool=pool[16:]
  totals+=schedule.size;curve=[json.loads(line)for line in (E/'runs'/r['run']/'curve.jsonl').read_text().splitlines()];assert [a['update']for a in curve]==list(range(1,1501));assert all(np.isfinite([a['BCE'],a['latent_SmoothL1'],a['total_loss'],a['gradient_norm_before_clip']]).all()for a in curve);assert all(abs(a['BCE']+.05*a['latent_SmoothL1']-a['total_loss'])<1e-5 for a in curve)
  assert r['selected']==min(r['checkpoints'],key=lambda c:(c['tune_root_macro_BCE'],c['update']))
  for cp in r['checkpoints']:
   assert sha(cp['path'])==cp['sha256'];ck=torch.load(cp['path'],map_location='cpu',weights_only=False);assert ck['update']==cp['update']and ck['protocol_sha256']==sha(E/'configs/protocol.json');assert sum(v.numel()for v in ck['model'].values())==76224;assert all(int(a['step'])==cp['update']for a in ck['optimizer']['state'].values());checkpoint_count+=1
 for seed in [0,1,2]:assert len({r['initial_digest']for r in sel['runs']if r['seed']==seed})==1
 checks['training_exact_budget']={'runs':6,'updates':9000,'example_presentations':totals,'checkpoints':checkpoint_count,'same_initialization_per_seed':True,'no_C_in_batch_schedule':True,'all_batch_indices_reconstructed':True}
 # Check saved tree leaves against their AB support, without refitting or using C.
 treestats={}
 for cohort in ['old96','aug132']:
  model=read(E/'models'/('tree_'+cohort+'.json'));idx=np.array([i for i in train if cohort=='aug132'or meta[i]['source']=='old']);xx=z['symbolic_inputs'][idx];yy=t['after_y'][idx];vv=t['after_valid'][idx];nodes=leaves=mixed=0;maxdepth=0
  for col,tree in enumerate(model['trees']):
   gi=0 if col<3 else 1 if col<6 else 2
   def walk(node,ids,depth):
    nonlocal nodes,leaves,mixed,maxdepth
    nodes+=1;maxdepth=max(depth,maxdepth);assert len(ids)==node['n'];w=np.where(yy[ids,col]>.5,weights[gi][1],weights[gi][0]);prob=float((w*yy[ids,col]).sum()/w.sum());assert abs(prob-node['weighted_positive_probability'])<1e-12
    if 'feature'in node:
     assert depth<4;left=xx[ids,node['feature']]<=node['threshold'];assert left.sum()>=2 and (~left).sum()>=2;walk(node['left'],ids[left],depth+1);walk(node['right'],ids[~left],depth+1)
    else:leaves+=1;mixed+=int(np.unique(yy[ids,col]).size>1)
   walk(tree,np.flatnonzero(vv[:,col]),0)
  assert maxdepth<=4;treestats[cohort]={'nodes':nodes,'leaves':leaves,'mixed_training_label_leaves':mixed,'max_depth':maxdepth,'train_pairs':len(idx),'AB_support_only':True}
 checks['trees_fixed_depth_and_leaf_AB_support_verified']=treestats
 y=t['after_y'].astype(bool);yb=t['before_y'].astype(bool);v=t['after_valid'];vb=t['before_valid'];beforegood=(known(p['persistence'])&((p['persistence']>=.5)==yb)&vb).all(1);aftergood=(known(p['real_next_observation'])&((p['real_next_observation']>=.5)==y)&v).all(1);sets={}
 for sn,split,source in [('AB_old','train','old'),('AB_new','train','new'),('AB_all','train',None),('C_old','dev','old'),('C_new','dev','new'),('C_all','dev',None)]:sets[sn]=np.array([r['split']==split and(source is None or r['source']==source)for r in meta])
 for sn,role in [('C_tune','tune'),('C_calibration','calibration')]:sets[sn]=np.array([r['dev_role']==role for r in meta])
 subset={'all':np.ones(174,bool),'before_readout_correct':beforegood,'before_readout_wrong_or_UNKNOWN':~beforegood,'both_observed_readouts_correct':beforegood&aftergood,'changed_macro':((yb!=y)&vb&v).any(1),'nochange_macro':~((yb!=y)&vb&v).any(1)}
 for val in ['success','refused']:subset['execution:'+val]=np.array([r['primitive_status']==val for r in meta])
 for val in ['open','close','lift','place']:subset['family:'+val]=np.array([r['family']==val for r in meta])
 for val in ['positive_reference','core_state_dependence','nochange_semantics']:subset['role:'+val]=np.array([r['contrast_role']==val for r in meta])
 target=np.zeros_like(v)
 for i,r in enumerate(meta):
  if r['skill']in ['open_drawer','close_drawer']:target[i,6]=True
  else:j=COLORS.index(r['planned_colour_evaluation_only']);target[i,[j,j+3]]=True
 masks={'all7':np.ones_like(v),'changed':(yb!=y)&vb,'unchanged':(yb==y)&vb,'target_entity':target,'other_entity':~target};count=0
 for name,pr in p.items():
  k=known(pr);cor=k&((pr>=.5)==y)&v;full=cor.all(1)
  for sn,ss in sets.items():
   for sub,sm in subset.items():
    ii=ss&sm;r=ev['results'][name][sn][sub];assert r['all7']==int(full[ii].sum())and r['pairs']==int(ii.sum())
    for mask,mm in masks.items():
     vv=ii[:,None]&v&mm;f=r['facts'][mask];assert f['n']==int(vv.sum())and f['strict_correct']==int((vv&cor).sum())and f['UNKNOWN']==int((vv&~k).sum());count+=1
    for rd in r['root_details']:
     rr=ii&np.array([x['root_id']==rd['root_id']for x in meta]);assert rd['all7']==int(full[rr].sum())and rd['all_correct']==bool(full[rr].all())
 checks['independent_fact_mask_recounts']=count
 episodes={r['episode_id']:i for i,r in enumerate(meta)if r['source']=='new'}
 for r in read(E/'results/root_contrasts.json'):
  a,b=episodes[r['positive_episode']],episodes[r['core_episode']];cols=r['target_fact_columns'];assert meta[a]['skill']==meta[b]['skill']and meta[a]['declared_arguments']==meta[b]['declared_arguments'];assert np.array_equal(yb[a,cols],yb[b,cols])and not np.array_equal(y[a,cols],y[b,cols])
  for name,pr in p.items():
   cor=known(pr)&((pr>=.5)==y)&v;assert r['models'][name]['both_target_correct']==bool(cor[[a,b]][:,cols].all())and r['models'][name]['both_all7_correct']==bool(cor[[a,b]].all())
 for name,rr in ev['latent_metrics'].items():
  for sn,ss in sets.items():
   h,s=lat[name+'_objects'][ss],lat[name+'_scene'][ss];assert abs(rr[sn]['object_MAE']-float(np.abs(h-t['next_objects'][ss]).mean()))<1e-12;assert abs(rr[sn]['scene_MAE']-float(np.abs(s-t['next_scene'][ss]).mean()))<1e-12
 checks['root_pair_and_latent_metrics_recomputed']=True
 tree=ast.parse((OLD/'source/transition.py').read_text());fn=next(x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name=='forward');assert [a.arg for a in fn.args.args]==['self','current_objects','current_scene','declared_actions'];checks['neural_interface_unchanged']=True
 dec=ev['decision'];assert dec==read(E/'results/decision.json')and dec['development_candidate_pass'];assert all(ev['results'][f'aug132_seed{s}_selected']['C_old']['all']['all7']==24 and ev['results'][f'aug132_seed{s}_selected']['C_new']['all']['all7']==18 for s in [0,1,2]);assert all(ev['contrast_summary'][f'aug132_seed{s}_final1500']['C_new']['both_all7_correct']==6 and ev['contrast_summary'][f'old96_seed{s}_final1500']['C_new']['both_all7_correct']==0 for s in [0,1,2]);candidate=read(E/'candidate/candidate.json');assert sha(E/'candidate/transition.pt')==candidate['checkpoint']['sha256']==candidate['transition_sha256'];checks['candidate_is_exact_preselected_checkpoint']=True
 result={'created_utc':now(),'passed':True,'checks':checks,'new_transition_forward_calls':0,'new_training_updates':0,'protocol_sha256':sha(E/'configs/protocol.json'),'selection_sha256':sha(E/'configs/model_selection_frozen.json'),'predictions_sha256':sha(E/'results/predictions.npz')};save(E/'results/final_validation.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
