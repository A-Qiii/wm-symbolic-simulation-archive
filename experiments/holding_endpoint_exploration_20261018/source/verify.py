"""Independent count/hash verification; no model fitting or forward calls."""
from pathlib import Path
import json, hashlib, datetime, csv, shutil
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
E=P/'experiments/holding_endpoint_exploration_20261018';R=P/'reports/holding_endpoint_exploration_20261018'
D=P/'experiments/independent_identity_D12_20261017';H=P/'experiments/holding_readout_diagnostic_20261018'
def read(p):return json.loads(Path(p).read_text())
def save(name,data):
 (E/'results'/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 if isinstance(data,list)and data:
  keys=list(dict.fromkeys(k for row in data for k in row))
  with(R/f'{name}.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in row.items()}for row in data)
def main():
 hashes=0
 for p in [H/'configs/delivery_manifest.json',D/'configs/delivery_manifest.json',E/'configs/frozen.json']:
  for path,h in read(p)['sha256'].items():
   assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path;hashes+=1
 a=read(E/'results/actual_endpoints.json');pairs=read(D/'data/combo/public/pairs.json');labs=read(D/'data/combo/_private/labels.json');assert len(a)==len(pairs)==135
 for row,p in zip(a,pairs):
  for k in ['id','root_id','target','other_target','branch','slot','skill','before','after']:assert row[k]==p[k]
  assert row['truth']==labs[str(p['after'])]['y'][:3]
  assert row['all_probes_actual_correct']==all(all((v>=.5)==y for v,y in zip(pr,row['truth']))for pr in row['probe_probability'].values())
 colorrows=[];checks=0
 for summary in read(E/'results/actual_condition_metrics.json')+read(E/'results/actual_root_metrics.json'):
  rows=[r for r in a if(summary['condition']=='all'or r['condition']==summary['condition'])and('root_id'not in summary or r['root_id']==summary['root_id'])]
  assert len(rows)==summary['N'];key=f"{summary['form']}_seed{summary['seed']}";y=np.array([r['truth']for r in rows]);pr=np.array([r['probe_probability'][key]for r in rows])>=.5
  assert int((y==pr).all(1).sum())==summary['H3_correct']
  assert int(pr[~y.any(1)].any(1).sum())==summary['empty_FP']
  assert int((~y.any(1)).sum())==summary['empty_N']
  for j,col in enumerate(summary['colors']):
   vals={'TP':int((pr[:,j]&y[:,j]).sum()),'FN':int((~pr[:,j]&y[:,j]).sum()),'FP':int((pr[:,j]&~y[:,j]).sum()),'TN':int((~pr[:,j]&~y[:,j]).sum())}
   assert all(col[k]==v for k,v in vals.items());checks+=4
   if 'root_id'not in summary:
    colorrows.append({k:summary[k]for k in ['condition','form','seed','N','roots']}|col|{'FPR':col['FP']/(col['FP']+col['TN'])if col['FP']+col['TN']else None,'FNR':col['FN']/(col['FN']+col['TP'])if col['FN']+col['TP']else None})
 save('actual_color_metrics',colorrows)
 summaries=read(E/'results/actual_condition_metrics.json');elig=read(E/'results/condition_eligibility.json')
 for cond,value in elig.items():
  ss=[r for r in summaries if r['condition']==cond];assert len(ss)==4
  passes=[r['roots']>=6 and r['H3_correct']/r['N']>=.95 and all((c['FP']<=.05*(c['FP']+c['TN']))and(c['TP']+c['FN']==0 or(c['FN']<=.1*(c['TP']+c['FN'])and c['positive_roots']>=2))for c in r['colors'])for r in ss]
  assert all(passes)==value
 for s in read(E/'results/actual_agreement_subset.json'):
  rows=[r for r in a if s['condition']=='all'or r['condition']==s['condition']];assert len(rows)==s['all_N'];assert [r['id']for r in rows if r['all_probes_actual_correct']]==s['agreed_ids'];assert len(s['agreed_ids'])==s['agreed_N']
 errors=[]
 for r in a:
  for head,pr in r['probe_probability'].items():
   for j,c in enumerate(['red','blue','pink']):
    if(pr[j]>=.5)!=r['truth'][j]:errors.append({k:r[k]for k in ['id','root_id','condition','branch','slot','target','other_target']}|{'head':head,'color':c,'error':'FN'if r['truth'][j]else'FP','probability':pr[j]})
 save('actual_error_facts',errors)
 predicted=read(E/'results/predicted_readouts.json');assert len(predicted)==91*4*2
 assert all(elig[r['condition']]for r in predicted)
 orig={ (r['id'],r['seed'],r['world_model']):r for r in read(D/'results/predictions.json')if r['query_arm']=='original'}
 for r in predicted:
  o=orig[(r['id'],r['seed'],r['world_model'])]
  assert o[r['path']+'_probability'][:3]==r['original_probability'];assert o['actual_probability'][:3]==r['original_actual_probability']
  assert all((p>=.5)==v for p,v in zip(r['original_actual_probability'],r['truth']))
 for s in read(E/'results/prediction_comparison.json'):
  rr=[r for r in predicted if all(r[k]==s[k]for k in ['seed','world_model','path','condition'])];assert len(rr)==s['N']
  counts={k:0 for k in ['both_wrong','original_only_wrong','probe_only_wrong','both_correct','same_wrong_fact_cells']}
  for r in rr:
   er=[(p>=.5)!=y for p,y in zip(r['original_probability'],r['truth'])];ep=[(p>=.5)!=y for p,y in zip(r[s['form']+'_probability'],r['truth'])]
   counts['both_wrong'if any(er)and any(ep)else'original_only_wrong'if any(er)else'probe_only_wrong'if any(ep)else'both_correct']+=1
   counts['same_wrong_fact_cells']+=sum(x and y for x,y in zip(er,ep))
  assert all(s[k]==v for k,v in counts.items())
 prefix=read(E/'results/prefix_predictions.json');assert len(prefix)==88;assert all(all(r['valid'])and r['only_replaced_after_lift']for r in prefix)
 roots={r['root_id']for r in read(D/'results/root_contexts.json')if r['context_matched']};assert len(roots)==11
 for s in read(E/'results/prefix_metrics.json'):
  rr=[r for r in prefix if all(r[k]==s[k]for k in ['seed','world_model','step'])];assert {r['root_id']for r in rr}==roots
  p=np.array([r[s['path']+'_probability']for r in rr]);y=np.array([r['truth']for r in rr]);b=p>=.5;b[:,6:]=False;b[np.arange(len(b)),6+p[:,6:].argmax(1)]=True
  assert int((b[:,:3]==y[:,:3]).all(1).sum())==s['Holding_correct'];assert int((b==y).all(1).sum())==s['full9_correct']
  assert all(int((b[:,j]!=y[:,j]).sum())==s['wrong_facts_counts'][str(j)]for j in range(9))
 rawdiff=[];different_from_reset=[]
 for seed in [0,1]:
  for wm in ['mixed','latent']:
   z=np.load(E/'results'/f'prefix_{wm}_seed{seed}.npz');old=np.load(D/'results'/f'latents_{wm}_seed{seed}.npz');ids=old['pair_ids'].tolist()
   for i,rid in enumerate(z['root_ids'].tolist()):
    place=next(r for r in pairs if r['root_id']==rid and r['branch']=='normal'and r['slot']==2);close=next(r for r in pairs if r['root_id']==rid and r['branch']=='normal'and r['slot']==3)
    rawdiff.append(float(np.abs(z['after_place'][i]-old['one_step'][ids.index(place['id'])]).max()))
    different_from_reset.append(not np.array_equal(z['after_close'][i],old['one_step'][ids.index(close['id'])]))
 assert max(rawdiff)<1e-5;assert all(different_from_reset)
 save('verification',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'verified_hash_references':hashes,'actual_boundaries':135,'actual_confusion_cells':checks,'predicted_rows':len(predicted),'paired_comparison_groups':64,'prefix_rows':88,'prefix_conditions':24,'raw_place_vs_saved_one_step_maxdiff':max(rawdiff),'close_raw_differs_from_after_place_true_reset_all44':all(different_from_reset),'no_additional_model_forward':True,'source_review':'run.py initializes cur from actual after-lift once; step4 uses cur returned by place; no later true z insertion','global_attribution_stop_preserved':True})
 shutil.copy2(__file__,E/'source/verify.py');print('ENDPOINT_VERIFIED',checks,'confusion cells,',len(predicted),'predicted rows, 88 prefix rows')
if __name__=='__main__':main()
