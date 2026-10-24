"""Independent recount from saved predictions, plus ancestry and sampling checks."""
from config import *
import collections,torch
def classify(row):
 p=list(row);return [x>=.5 for x in p[:6]]+[j==max(range(3),key=lambda k:p[6+k])for j in range(3)]
def subset(name,y,v,dual):
 held=any(a and b for a,b in zip(y[:3],v[:3]));inside=any(a and b for a,b in zip(y[3:6],v[3:6]));return {'all':True,'dual':dual,'nondual':not dual,'holding_without_In':held and all(v[3:6])and not inside,'empty_hand':all(v[:3])and not held}[name]
def main():
 verify();cfg=read(E/'configs/protocol.json');seals=read(E/'configs/frozen.json')['prior_manifests'];priorcount=0
 for path,h in seals.items():
  assert sha(path)==h,path
  for p,expected in read(path)['sha256'].items():assert sha(p)==expected,p;priorcount+=1
 tr={a:dict(np.load(E/'data/train'/f'{a}.npz'))for a in ['control','coverage']};forbidden=set(cfg['root_split']['validation']+cfg['root_split']['diagnostic_D_seen']);assert set(tr['coverage']['root_ids']).isdisjoint(forbidden);assert set(tr['coverage']['root_ids'][156:])==set(cfg['root_split']['train'])
 for k in tr['control']:assert np.array_equal(tr['control'][k],tr['coverage'][k][:156])
 for b in read(E/'results/sampling_budget.json'):
  arm=b['arm'];s=np.load(E/'configs'/f"schedule_seed{b['seed']}.npz");assert np.array_equal(s['control'][:,:24],s['coverage'][:,:24]);ix=s[arm];assert ix.shape==(600,32);y=tr[arm]['y'][ix].astype(bool);v=tr[arm]['v'][ix].astype(bool)
  assert int(v.sum())==b['valid_fact_cells']and int((v&y).sum())==b['valid_positive_cells']and int((v&~y).sum())==b['valid_negative_cells'];assert np.bincount(s['coverage'][:,24:].ravel()-156).tolist()==[800]*6
  for key,values in [('per_fact_valid',v),('per_fact_positive',v&y),('per_fact_negative',v&~y)]:assert values.sum((0,1)).tolist()==b[key]
 for seed in [0,1]:
  initial=torch.load(cfg['initial_heads'][str(seed)]['path'],weights_only=True,map_location='cpu')['state']
  for arm in ['control','coverage']:
   saved=torch.load(E/'models'/f'{arm}_seed{seed}/update0000.pt',weights_only=True,map_location='cpu')['state'];assert set(saved)==set(initial);assert all(torch.equal(saved[k],initial[k])for k in saved)
   final=torch.load(E/'models'/f'{arm}_seed{seed}/update0600.pt',weights_only=True,map_location='cpu');assert final['update']==600 and set(final['state'])==set(initial);assert all(torch.isfinite(t).all()for t in final['state'].values())
 for p,h in read(E/'configs/final_heads_frozen.json')['sha256'].items():assert sha(p)==h,p
 training=read(E/'results/training.json');assert len(training['runs'])==4 and all(r['updates']==600 for r in training['runs']);assert training['encoder_decoder_transition_updates']==0
 data={f.stem:dict(np.load(f))for f in(E/'data/evaluation').glob('*.npz')};actual=read(E/'results/actual_predictions.json');ag=collections.defaultdict(list)
 for r in actual:
  d=data[r['cohort']];i=r['state_index'];assert str(d['root_ids'][i])==r['root_id']and str(d['state_ids'][i])==r['state_id'];assert np.isfinite(r['probability']).all()and np.all((np.array(r['probability'])>=0)&(np.array(r['probability'])<=1));ag[(r['cohort'],r['seed'],r['query_arm'])].append(r)
 assert len(actual)==6534 and len({(r['cohort'],r['seed'],r['query_arm'],r['state_index'])for r in actual})==6534
 allmet=read(E/'results/actual_metrics.json');factmet=read(E/'results/actual_fact_confusions.json');acount=0
 def records(m):
  for r in ag[(m['cohort'],m['seed'],m['query_arm'])]:
   d=data[r['cohort']];i=r['state_index'];y=d['y'][i].astype(bool).tolist();v=d['v'][i].astype(bool).tolist()
   if subset(m['subset'],y,v,r['dual_relation']):yield r,y,v,classify(r['probability'])
 for m in allmet:
  rr=list(records(m));assert len(rr)==m['states']and len({r[0]['root_id']for r in rr})==m['roots']
  for g,ix in [('full9',range(9)),('Holding',range(3)),('In',range(3,6)),('drawer',range(6,9))]:
   eligible=[r for r in rr if all(r[2][j]for j in ix)];ok=[r for r in eligible if all(r[1][j]==r[3][j]for j in ix)];assert len(eligible)==m[g+'_N']and len(ok)==m[g+'_correct'];acount+=1
 for m in factmet:
  j=FACTS.index(m['fact']);c=collections.Counter(('TP'if y[j]and p[j]else'FN'if y[j]else'FP'if p[j]else'TN')for _,y,v,p in records(m)if v[j]);assert all(c[k]==m[k]for k in ['TP','FN','FP','TN']);acount+=1
 traj=read(E/'results/trajectory_predictions.json');assert len(traj)==2592;labs=read(COMBO/'data/combo/_private/labels.json');tg=collections.defaultdict(list)
 for r in traj:tg[(r['split'],r['seed'],r['world_model'],r['query_arm'],r['branch'])].append(r)
 tcount=0
 for m in read(E/'results/trajectory_primary.json'):
  rr=tg[(m['split'],m['seed'],m['world_model'],m['query_arm'],m['branch'])];byroot=collections.defaultdict(list)
  for r in rr:byroot[r['root_id']].append(r)
  correct=0
  for root,rows in byroot.items():
   assert len(rows)==4;assert all(all(labs[str(r['after'])]['v'])for r in rows);correct+=all(classify(r[m['path']+'_probability'])==labs[str(r['after'])]['y']for r in rows)
  assert correct==m['correct']and len(byroot)==m['N']==6;tcount+=1
 for m in read(E/'results/trajectory_fact_confusions.json'):
  j=FACTS.index(m['fact']);rr=tg[(m['split'],m['seed'],m['world_model'],m['query_arm'],m['branch'])];c=collections.Counter()
  for r in rr:
   if r['slot']!=m['step']-1:continue
   lab=labs[str(r['after'])];y=lab['y'][j];p=classify(r[m['path']+'_probability'])[j]
   if lab['v'][j]:c['TP'if y and p else'FN'if y else'FP'if p else'TN']+=1
  assert all(c[k]==m[k]for k in ['TP','FN','FP','TN']);tcount+=1
 # Paired gains/losses must equal the difference in exact outcomes, independently by root.
 for m in read(E/'results/trajectory_paired.json'):
  dd=[int(r['coverage_correct'])-int(r['baseline_correct'])for r in m['per_root']];assert dd.count(1)==m['gains']and dd.count(-1)==m['losses']and abs(sum(dd)/len(dd)-m['difference'])<1e-12
 for m in read(E/'results/actual_paired.json'):
  rr=m['per_root'];assert sum(r['gains']for r in rr)==m['gains']and sum(r['losses']for r in rr)==m['losses'];assert all(r['coverage_correct']-r['base_correct']==r['gains']-r['losses']for r in rr)
 manifest=E/'configs/delivery_manifest.json'
 if manifest.exists():
  for p,h in read(manifest)['sha256'].items():assert sha(p)==h,p
 result={'utc':now(),'status':'passed','prior_sealed_file_checks':priorcount,'actual_metric_recounts':acount,'trajectory_metric_recounts':tcount,'four_runs_600_steps':True,'paired_initial_weights_equal':True,'common_old_slots_equal':True,'new_root_draws_each':800,'no_validation_D_training_roots':True,'no_old_source_model_result_changes':True,'all9_valid_mask_respected':True,'remaining_execution_blockers':[]};print('QUERY_CONTROL_VERIFICATION '+json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
