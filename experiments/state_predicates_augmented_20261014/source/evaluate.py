from pathlib import Path
import json,hashlib,fcntl,datetime,csv
import numpy as np
import torch
from model import PredicateReader,targets
from metrics import GROUPS,metric,known,summarize
P=Path('/home/robot/research/projects/wm_symbolic');OLD=P/'experiments/state_predicates_20261014';E=P/'experiments/state_predicates_augmented_20261014'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 assert not (E/'results/evaluation.json').exists(),'Final evaluation already completed; do not reselect or rerun without a reason'
 freeze=json.loads((E/'results/model_threshold_freeze.json').read_text());protocol=json.loads((E/'configs/protocol.json').read_text());assert not freeze['new_test_feature_or_probability_file_exists']
 for path,h in freeze['source_sha256'].items():assert sha(path)==h
 for path,h in protocol['frozen_sha256'].items():assert sha(path)==h
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 with np.load(E/'data/supplement_test.npz') as z:newdata={k:torch.tensor(z[k],device='cuda') for k in ['objects','context','robot']};ny=z['y'].copy();nv=z['eval_valid'].copy()
 nm=json.loads((E/'data/metadata_test.json').read_text());nr=np.array([r['root_id'] for r in nm]);conditions=np.array([r['condition'] for r in nm]);nt=np.array([r['target_index'] for r in nm]);nheld=ny[np.arange(len(ny)),nt]>0;closed=ny[:,6]==0
 with np.load(OLD/'data/inputs_targets.npz') as z:
  mask=z['splits']=='diagnostic_D';olddata={k:torch.tensor(z[k][mask],device='cuda') for k in newdata};yy,vv=targets(torch.tensor(z['cat'][mask]),torch.tensor(z['valid'][mask]));oy=yy.numpy();ov=vv.numpy();near=z['near_negative'][mask]
 om=[r for r in json.loads((OLD/'data/metadata.json').read_text()) if r['split']=='diagnostic_D'];orr=np.array([r['root_id'] for r in om]);groups={'true_held':nheld,'near_closed_negative':conditions=='near_closed_negative','near_closed_motion_negative':conditions=='near_closed_motion_negative','contact_supported_notheld':conditions=='contact_supported_close_command_negative','near_open_notheld':conditions=='near_open_negative','far_notheld':np.isin(conditions,['far_closed_negative','far_open_negative']),'held_and_drawer_closed':nheld&closed};probabilities={};results={};repro={}
 for tag,entry in freeze['models'].items():
  path=Path(entry['checkpoint_path']);assert sha(path)==entry['checkpoint_sha256'];ck=torch.load(path,map_location='cuda',weights_only=False);model=PredicateReader().cuda().eval();model.load_state_dict(ck['model'])
  outputs={}
  with torch.no_grad():
   for cohort,data in [('new_test',newdata),('old_D_regression',olddata)]:
    chunks=[]
    for i in range(0,len(data['objects']),128):chunks.append(model.probabilities(**{k:v[i:i+128] for k,v in data.items()}).cpu().numpy())
    outputs[cohort]=np.concatenate(chunks);np.save(E/'results'/f'{tag}_{cohort}_probabilities.npy',outputs[cohort])
  probabilities[tag]=outputs
  if tag.startswith('old_seed'):
   prior=np.load(OLD/'results'/f'direct_predicates_seed{tag[-1]}_probabilities.npy')[mask];error=float(np.max(np.abs(prior-outputs['old_D_regression'])));repro[tag]={'saved_old_probability_max_error':error,'same_threshold_point5_predictions':bool(np.array_equal(prior>=.5,outputs['old_D_regression']>=.5))};assert repro[tag]['same_threshold_point5_predictions']
  del model
 # Different thresholds are explicit controls; raw confusion never depends on them.
 for tag,outs in probabilities.items():
  variants=[(tag,freeze['models'][tag]['thresholds'])]
  if tag.startswith('old_seed'):variants.append((tag.replace('old_seed','old_original_seed'),[.5,.5,.5]))
  for name,taus in variants:
   pp=outs['new_test'];op=outs['old_D_regression'];nk=known(pp,taus);ok=known(op,taus);subsets={}
   for group,rows in groups.items():
    cols=[6] if group=='held_and_drawer_closed' else [0,1,2];subsets[group]={'roots':len(set(nr[rows])),'windows':int(rows.sum()),'metric':metric(pp[rows][:,cols],ny[rows][:,cols],nv[rows][:,cols],nk[rows][:,cols])}
   stress=nr!='D_predicate_gap_red_open';oldnear=ov.copy();oldnear[:,:3]&=near;oldnear[:,3:]=False;oldhc=(oy[:,:3].sum(1)>0)&(oy[:,6]==0)
   changed={}
   reference=probabilities['old_seed'+name[-1]]['old_D_regression']
   for g,cols in GROUPS.items():
    was=(reference[:,cols]>=.5)!=oy[:,cols].astype(bool);now=(op[:,cols]>=.5)!=oy[:,cols].astype(bool);valid=ov[:,cols];changed[g]={'fixed_old_errors':int((was&~now&valid).sum()),'new_errors_on_old_correct':int((~was&now&valid).sum())}
   results[name]={'thresholds':taus,'new_test':summarize(pp,ny,nv,taus),'new_test_buckets':subsets,'new_test_exclude_near_position_D_root':{'root_count':len(set(nr[stress])),'window_count':int(stress.sum()),'metrics':summarize(pp[stress],ny[stress],nv[stress],taus)},'old_D_regression':summarize(op,oy,ov,taus),'old_D_near_not_held':metric(op,oy,oldnear,ok),'old_D_held_closed_open':metric(op[oldhc,6:7],oy[oldhc,6:7],ov[oldhc,6:7],ok[oldhc,6:7]),'old_D_changed_errors':changed}
 # Individual examples and confidence, not pooled pseudo-independent seed counts.
 with (E/'results/predictions.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['model','cohort','root','sample','atom_column','truth','probability','raw_prediction','known'])
  for name,r in results.items():
   tag=name.replace('old_original_seed','old_seed');outs=probabilities[tag]
   for cohort,meta,y,v in [('new_test',nm,ny,nv),('old_D_regression',om,oy,ov)]:
    p=outs[cohort];k=known(p,r['thresholds'])
    for i,row in enumerate(meta):
     for j in np.flatnonzero(v[i]):w.writerow([name,cohort,row['root_id'],row.get('sample_id',row.get('episode_id')+'_'+str(row.get('step')) if 'episode_id' in row else ''),int(j),int(y[i,j]),float(p[i,j]),int(p[i,j]>=.5),bool(k[i,j])])
 criteria={}
 for seed in [0,1,2]:
  n=results[f'augmented_seed{seed}'];o=results[f'old_original_seed{seed}'];b=n['new_test_buckets'];reg={g:n['old_D_regression'][g]['accuracy']-o['old_D_regression'][g]['accuracy'] for g in GROUPS};criteria[str(seed)]={'new_primary_pass':b['held_and_drawer_closed']['metric']['FP']==0 and b['near_closed_negative']['metric']['FP']==0 and b['true_held']['metric']['TP']>=5,'supported_contact_FP':b['contact_supported_notheld']['metric']['FP'],'old_D_accuracy_change':reg,'old_D_no_gt2pp_drop':all(x>=-.02 for x in reg.values()),'held_positive_predictions':n['new_test']['held']['raw_positive_predictions'],'held_positive_recall':n['new_test']['held']['positive_recall'],'held_unknown':n['new_test']['held']['unknown']}
 # One paired root bootstrap per seed0 dataset, only for descriptive error-rate difference.
 rng=np.random.default_rng(20261014);ci={}
 for cohort,roots,y,v in [('new_test',nr,ny,nv),('old_D_regression',orr,oy,ov)]:
  a=probabilities['old_seed0'][cohort];b=probabilities['augmented_seed0'][cohort];unique=sorted(set(roots));dist={g:[] for g in GROUPS}
  for _ in range(2000):
   inds=np.concatenate([np.flatnonzero(roots==r) for r in rng.choice(unique,len(unique),replace=True)])
   for g,cols in GROUPS.items():
    vv=v[inds][:,cols];ea=((a[inds][:,cols]>=.5)!=y[inds][:,cols])*vv;eb=((b[inds][:,cols]>=.5)!=y[inds][:,cols])*vv;dist[g].append(float((eb.sum()-ea.sum())/vv.sum()))
  ci[cohort]={g:np.quantile(x,[.025,.975]).tolist() for g,x in dist.items()}
 report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':results,'preregistered_criteria':criteria,'new_test_roots':sorted(set(nr)),'new_test_windows':len(ny),'old_D_roots':len(set(orr)),'old_D_boundaries':len(oy),'old_checkpoint_reproduction':repro,'paired_root_bootstrap_95_interval_new_minus_old_error_rate_seed0':ci,'no_test_based_selection_or_repair':True,'raw_counts_independent_of_unknown_threshold':True,'all_models_and_thresholds_frozen_before_test':True}
 save(E/'results/evaluation.json',report);save(E/'results/progress.json',{'stage':'evaluation_complete','new_test_inference_run':True,'model_training_complete':True});print(json.dumps({'criteria':criteria,'new_test':{k:{g:{x:q[x] for x in ['n','positive','negative','FP','FN','unknown','known_error']} for g,q in v['new_test'].items()} for k,v in results.items() if 'final' not in k and ('original' in k or 'augmented' in k)},'old_regression':{k:{g:{x:q[x] for x in ['FP','FN','unknown','accuracy']} for g,q in v['old_D_regression'].items()} for k,v in results.items() if 'final' not in k and ('original' in k or 'augmented' in k)}},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
