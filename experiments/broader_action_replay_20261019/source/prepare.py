from common import *
import ast,collections,yaml
def main():
 sys.addaudithook(guard);assert not(S/'configs/protocol.json').exists()
 for d in ['configs','data','results','figures','logs','cache']:(S/d).mkdir(parents=True,exist_ok=True)
 save(S/'configs/protocol.json',{'utc':now(),'frozen_before_new_forecasts':True,'goal':'Task-relevant held identity, drawer ownership, action consequences, extra false relations across broader existing A/B geometry','data':'All existing 60 A/B coverage roots, 180 four-action branches,720 transitions; no C/D','models':['original roots40 joint RSSM seed0/1 unchanged','original object head seed0/1','coverage object head seed0/1','previous selected accurate-relation dynamics seed0/1','previous selected physics-enriched dynamics seed0/1'],'compatibility':'RSSM retains its original symbol head. Object heads operate on current object/global frozen E features, are NOT inserted into RSSM. Their hard relations only feed the separate frozen small relation model as an explicit composed diagnostic.','new_forward':'Reuse cached global E, pad last boundary to7 exactly as original evaluator; original 8 posterior samples, seed81831,batch4,BF16,matmul high. Must exactly reproduce old test predictions before new RSSM replay. No DINO or spatial encoder forward.','initial_controls':'Same frozen relation model separately receives accurate relation state, native RSSM current hard state, original-object current hard state, coverage-object current hard state. One-step uses actual current observation; free uses only h0 and actions. Physical model receives true before state, original train-only normalization.','hard_decision':'Each held/in-drawer probability>=.5, drawer argmax. No thresholds fitted.','soft_mixed_definition':'Within hard-all-correct states, mark mixed if any truth-aligned binary confidence<.8 or true drawer probability<.8; retain actual probabilities and wrong-identity mass. This is descriptive, not a calibrated uncertainty detector.','cohorts':'Report train40 separately (seen by coverage head only; semantic templates paired to old training roots) and old_band10/gap10 (unseen in training by all compared models). All previously inspected; no fresh blind claim. Per-model root-level overlap CSV.','metrics':'Current held identity; post-place InDrawer true positives/false positives; one-step and free1..4 exact fields, full-prefix, changed fields and extra-positive relations; strata starting hard correct/confident vs mixed vs wrong. Root-weighted and A/B separated.','sufficiency':'Group real before relation+skill against old train successors/status; preserve new conflicts and examples. No hand-coded dynamics.','limits':{'new_fits':0,'new_collection':0,'C_D_samples':0,'uploads':0,'gpu_seconds':600,'round_minutes':30},'no_adaptive_tuning':True})
 protected={}
 manifests={R:read(R/'manifest.json')['sha256'],D:read(D/'manifest.json')['sha256'],O:read(O/'manifest.json')['sha256'],B:read(B/'configs/delivery_manifest.json')['sha256'],H:read(H/'manifest.json')['sha256']}
 def use(p):
  p=Path(p);v=sha(p)
  for root,old in manifests.items():
   if p.is_relative_to(root):
    k=str(p)if str(p)in old else str(p.relative_to(root));assert k in old,'Unsealed input '+str(p);assert old[k]==v,'Changed input '+str(p);break
  protected[str(p)]=v;return p
 roots=read(use(R/'data/roots_manifest.json'));rows=read(use(R/'data/data_manifest.json'))['records'];oldroots=read(use(D/'configs/roots.json'));overlap=[]
 for r in roots:
  for model in ['RSSM','old_object_head','coverage_object_head','accurate_relations','physics_enriched']:
   overlap.append({'root_id':r['root_id'],'scene':r['environment'],'split':r['split'],'model':model,'used_in_model_training':model=='coverage_object_head'and r['split']=='train','paired_old_root':r.get('paired_old_root'),'paired_old_root_was_training':r.get('paired_old_root')in oldroots['train'],'all_models_unseen_root':r['split']!='train','historically_inspected':True})
 csvwrite(S/'results/root_independence.csv',overlap)
 configs={env:yaml.safe_load((P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{env}.yaml').read_text())for env in ['A','B']}
 tree=ast.parse(use(D/'source/prepare.py').read_text());mainnode=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');physnode=next(n for n in mainnode.body if isinstance(n,ast.FunctionDef)and n.name=='physical');ns={'np':np,'COLORS':COLORS,'configs':configs};exec(compile(ast.Module(body=[physnode],type_ignores=[]),'<frozen physical features>','exec'),ns);physical=ns['physical'];counts=[];provenance=[]
 for split in SPLITS:
  by=collections.defaultdict(list)
  for r in rows:
   if r['split']==split:by[r['episode_id']].append(r)
  items=[]
  for eid,rr in sorted(by.items()):
   rr.sort(key=lambda r:r['step']);assert [r['step']for r in rr]==[0,1,2,3];labs=[];infos=[];actions=[];statuses=[];controls=[];frames=[]
   for t,r in enumerate(rr):
    lab=read(use(r['labels_path']));info=read(use(Path(r['labels_path']).parent/'endpoint_info_private.json'));res=read(use(Path(r['labels_path']).parent/'result_private.json'))
    if t==0:labs.append(lab['before']);infos.append(info['before']);frames.append(str(Path(r['obs_before']).parent.parent/'root_observation_3.npz'))
    else:assert labs[-1]['categorical']==lab['before']['categorical']
    labs.append(lab['after']);infos.append(info['after']);frames.append(str(Path(r['obs_after']).parent/'boundary_history_3.npz'));act=np.zeros(7,np.float32);act[SKILLS.index(r['skill'])]=1;act[6]=r['executor']['budget_steps']/900;actions.append(act);n=lab['control_samples_including_start']-1;controls.append(n);statuses.append(0 if lab['success']else 1 if n==0 else 2);provenance.append({'split':split,'root_id':r['root_id'],'episode_id':eid,'step':t,'skill':r['skill'],'labels_path':r['labels_path'],'private_info_path':str(Path(r['labels_path']).parent/'endpoint_info_private.json'),'result':res})
   cat=np.array([l['categorical'][:4]for l in labs]);valid=np.array([l['categorical_valid'][:4]for l in labs]);assert valid.all();y=rel_from_cat(cat).astype(np.float32);x=np.c_[y,np.ones((5,7),np.float32)];phys=np.stack([physical(i,rr[0]['environment'])[0]for i in infos]);items.append({'episode_id':eid,'root_id':rr[0]['root_id'],'scene':rr[0]['environment'],'cat':cat,'y':y,'initial_state':x,'physical':phys,'action':np.stack(actions),'status':np.array(statuses),'controls':np.array(controls),'public_frame':np.array(frames)})
  data={k:np.array([i[k]for i in items])for k in items[0]};np.savez_compressed(S/f'data/dataset_{split}.npz',**data);counts.append({'split':split,'roots':len(set(data['root_id'])),'episodes':len(items),'transitions':len(items)*4,**{s:int((data['status']==j).sum())for j,s in enumerate(STATUS)}})
  for seed in [0,1]:
   use(R/f'data/features_seed{seed}_{split}.npz')
   if split!='train':
    for arm in ['control','coverage']:use(R/f'data/predictions_{arm}_seed{seed}_{split}.npz')
 for seed in [0,1]:
  for p in [B/f'cells/roots40/models/joint_seed{seed}/final6000.pt',O/f'models/head_object_seed{seed}.pt',R/f'models/head_coverage_seed{seed}.pt',H/f'data/predictions_joint_seed{seed}_test.npz',O/f'data/features_seed{seed}_test.npz',O/f'data/predictions_object_seed{seed}_test.npz']:use(p)
 selected=read(use(D/'configs/selected_models.json'))
 for r in selected['selection']:
  use(r['path']);use(D/f'data/predictions_{r["arm"]}_truth_seed{r["seed"]}_test.npz')
 for p in [D/'source/train.py',D/'configs/physical_normalization.npz',D/'data/dataset_train.npz',D/'data/dataset_test.npz',D/'data/predictions_relations_truth_seed0_test.npz',D/'data/predictions_relations_truth_seed1_test.npz',H/'source/evaluate_with_h2.py',O/'source/common.py',B/'cells/roots40/configs/normalization.json']:use(p)
 save(S/'configs/selected_models.json',selected);save(S/'results/data_counts.json',counts);save(S/'data/provenance.json',provenance);save(S/'configs/roots.json',{s:[r['root_id']for r in roots if r['split']==s]for s in SPLITS})
 for p in [P/'src/wm_symbolic/world_model.py',P/'src/wm_symbolic/dataset.py']:use(p)
 for p in list((S/'source').glob('*.py'))+list((S/'configs').glob('*'))+list((S/'data').glob('*')):
  if p.is_file():protected[str(p)]=sha(p)
 save(S/'configs/inputs_frozen.json',{'utc':now(),'before_forwards':True,'files':len(protected),'sha256':protected});print('PREPARED',counts,len(protected),flush=True)
if __name__=='__main__':main()
