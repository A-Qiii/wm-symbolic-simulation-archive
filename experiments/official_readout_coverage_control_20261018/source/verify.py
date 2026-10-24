from coverage_common import *
import csv,torch,collections,shutil
def export(name,rows):
 save(E/'results'/f'{name}.json',rows)
 if isinstance(rows,list)and rows:
  keys=list(dict.fromkeys(k for r in rows for k in r))
  with(R/f'{name}.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in r.items()}for r in rows)
def main():
 for mf in [E/'configs/pre_encoding_frozen.json',E/'configs/training_frozen.json',E/'configs/final_heads_frozen.json']:
  for p,h in read(mf)['sha256'].items():assert sha(p)==h,p
 for p,h in read(E/'configs/source_references.json')['sha256'].items():assert sha(p)==h,p
 cfg=read(E/'configs/protocol.json');train=read(E/'results/training.json');assert len(train['runs'])==8 and all(r['updates']==600 for r in train['runs']);attempts=read(E/'results/training_attempts.json');assert len(attempts)==8 and all(r['status']=='completed'for r in attempts)
 official=np.load(E/'data/train/official_train.npz',allow_pickle=False);valid=np.load(E/'data/evaluation/official_validation.npz',allow_pickle=False);assert len(official['z'])==len(valid['z'])==120 and len(set(official['root_ids']))==len(set(valid['root_ids']))==4 and set(official['root_ids']).isdisjoint(valid['root_ids']);assert not any(r.startswith(('C_','D_'))for r in official['root_ids']);assert official['v'][:,3:6].astype(bool).all();assert int((~valid['v'][:,3:6].astype(bool)).sum())==4
 schedule_checks=[]
 for head in HEADS:
  old=np.load(E/'data/train'/f'old_{head}.npz',allow_pickle=False);norm=np.load(E/'configs'/f'normalization_{head}.npz',allow_pickle=False);expected=np.load((H/'configs/normalization_object.npz')if head=='Holding'else(I/'configs/object_normalization.npz'),allow_pickle=False);assert all(np.array_equal(norm[k],expected[k])for k in norm.files)
  for p in(E/'data/evaluation').glob('*.npz'):assert set(old['root_ids']).isdisjoint(np.load(p,allow_pickle=False)['root_ids'])
  for seed in[0,1]:
   sc=np.load(E/'configs'/f'schedule_{head}_seed{seed}.npz',allow_pickle=False);oldsc=np.load((H if head=='Holding'else I)/'configs'/f'schedule_seed{seed}.npz',allow_pickle=False)['indices'if head=='Holding'else'control'];assert np.array_equal(sc['control'],oldsc)and np.array_equal(sc['common_old'],oldsc[:,:24]);assert np.bincount(sc['official'].ravel(),minlength=120).tolist()==[40]*120
   for row in sc['official']:assert sorted(collections.Counter(official['root_ids'][row]).values())==[2]*4
   rr=[r for r in train['runs']if r['head']==head and r['seed']==seed];assert len({r['initial_hash']for r in rr})==1
   schedule_checks.append({'head':head,'seed':seed,'old_schedule_exactly_preserved':True,'common24_equal':True,'each_official_state_draws':40,'each_new_episode_draws_per_update':2,'initial_hash_equal':True})
 export('schedule_checks',schedule_checks)
 rows=read(E/'results/actual_metrics.json');full=read(E/'results/full9_actual_metrics.json');deltas=[];confusion=[];regressions=[];recount=0
 for m in rows:
  d=np.load(E/'data/evaluation'/f"{m['cohort']}.npz",allow_pickle=False);p=np.load(E/'results/actual_probabilities'/f"{m['cohort']}_{m['arm']}_seed{m['seed']}.npz",allow_pickle=False);assert np.array_equal(d['state_ids'],p['state_ids']);g=slice(0,3)if m['head']=='Holding'else slice(3,6);pr=p['probability'][:,g]>=.5;y=d['y'][:,g].astype(bool);v=d['v'][:,g].astype(bool);assert int(((pr==y).all(1)&v.all(1)).sum())==m['all3_correct']and int(v.all(1).sum())==m['all3_valid']
  for j,c in enumerate(m['colors']):
   for k,cell in [('TP',pr[:,j]&y[:,j]),('FN',~pr[:,j]&y[:,j]),('FP',pr[:,j]&~y[:,j]),('TN',~pr[:,j]&~y[:,j])]:assert int((cell&v[:,j]).sum())==c[k];recount+=1
   confusion.append({k:m[k]for k in['head','seed','arm','cohort']}|c)
 for cohort in cfg['evaluation_cohorts']:
  d=np.load(E/'data/evaluation'/f'{cohort}.npz',allow_pickle=False)
  for seed in[0,1]:
   cp=np.load(E/'results/actual_probabilities'/f'{cohort}_control_seed{seed}.npz')['probability'];ap=np.load(E/'results/actual_probabilities'/f'{cohort}_augmented_seed{seed}.npz')['probability'];assert np.array_equal(cp[:,6:],ap[:,6:])
   for j in range(6):
    y=d['y'][:,j].astype(bool);v=d['v'][:,j].astype(bool);cb=cp[:,j]>=.5;ab=ap[:,j]>=.5
    for i in np.flatnonzero(v&(cb!=ab)):regressions.append({'cohort':cohort,'seed':seed,'head':'Holding'if j<3 else'In','color':COLORS[j%3],'root_id':str(d['root_ids'][i]),'state_id':str(d['state_ids'][i]),'truth':bool(y[i]),'control_probability':float(cp[i,j]),'augmented_probability':float(ap[i,j]),'effect':'regression'if ab[i]!=y[i]else'improvement','error_after':'FN'if y[i]and not ab[i]else'FP'if not y[i]and ab[i]else'none'})
   for head in HEADS:
    a=next(r for r in rows if r['cohort']==cohort and r['seed']==seed and r['head']==head and r['arm']=='augmented');c=next(r for r in rows if r['cohort']==cohort and r['seed']==seed and r['head']==head and r['arm']=='control');deltas.append({'head':head,'seed':seed,'cohort':cohort,'control_all3_correct':c['all3_correct'],'augmented_all3_correct':a['all3_correct'],'valid_states':a['all3_valid'],'correct_change':a['all3_correct']-c['all3_correct'],'control_balanced_error':c['mean_color_balanced_error'],'augmented_balanced_error':a['mean_color_balanced_error']})
 for r in full:
  d=np.load(E/'data/evaluation'/f"{r['cohort']}.npz");p=np.load(E/'results/actual_probabilities'/f"{r['cohort']}_{r['arm']}_seed{r['seed']}.npz")['probability'];pred=p>=.5;pred[:,6:]=False;pred[np.arange(len(pred)),6+p[:,6:].argmax(1)]=True;v=d['v'].all(1);assert int(((pred==d['y']).all(1)&v).sum())==r['all9_correct']and int(v.sum())==r['all9_valid']
 # Recount every official annotation/colour pair from saved predictions.
 paired=read(E/'results/official_matched_pairs.json');labels=read(E/'data/private/labels.json');meta={r['state_id']:r for r in labels};by=collections.defaultdict(list)
 for i,sid in enumerate(valid['state_ids']):
  for h in meta[str(sid)]['candidate_hints']:by[h['annotation_index']].append(i)
 for r in paired:
  j=COLORS.index(r['color'])+(0 if r['head']=='Holding'else 3);p=np.load(E/'results/actual_probabilities'/f"official_validation_{r['arm']}_seed{r['seed']}.npz")['probability'][:,j]>=.5
  for rec in r['records']:
   ix=np.array(sorted(set(by[rec['annotation_index']])));known=valid['v'][ix,j].astype(bool);assert rec['paired_correct']==bool((p[ix][known]==valid['y'][ix,j][known]).all())
 export('actual_changes',deltas);export('actual_confusion',confusion);export('decision_flips',regressions)
 for name in['full9_actual_metrics','official_matched_pairs','secondary_metrics','secondary_stage_metrics','sampling_information_budget']:export(name,read(E/'results'/f'{name}.json'))
 gate=read(E/'results/secondary_gate.json');assert [r['head']for r in gate['head_decisions']if r['eligible_for_saved_prediction_diagnostic']]==['In'];assert read(E/'results/secondary_audit.json')['T_forward_calls']==0
 export('verification',{'utc':now(),'passed':True,'fits':8,'updates_each':600,'same_initialization_per_pair':True,'old24_schedule_match':True,'official_each_state_draws':40,'no_cross_root_episode_leakage':True,'official_training_states':120,'official_validation_states':120,'old16_retests_used_in_main':0,'official_unknown_In_facts_masked':4,'excluded_old_retest_unknown_In_facts':2,'actual_confusion_cells_recounted':recount,'full9_metrics_recounted':len(full),'paired_metrics_recounted':len(paired),'D_gate_selection':False,'secondary_only_In':True,'new_T_forward_or_fit':False,'frozen_inputs_heads_sources_unchanged':True})
 shutil.copy2(__file__,E/'source/verify.py');print('READOUT_VERIFIED',recount,'confusion cells;8fixedfits',flush=True)
if __name__=='__main__':main()
