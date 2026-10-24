from common import *
import torch
def main():
 guard('eval');torch.set_num_threads(4);checks={};checks['frozen_input_files_unchanged']=verify_inputs();training=read(S/'results/training_complete.json');assert training['fits']==4 and training['continuation_updates']==3000;checks['four_fits_750_each']=True
 original=(B/'source/losses.py').read_text();assert (S/'source/original_losses.py').read_text()==original;expected=original.replace(" return total,current,future,terms,encoded"," lengths=mask.sum(1).long()\n terminal_mask=torch.stack([(lengths==t)&(lengths>0)for t in range(1,len(posts))],1).float()\n terminal_values=torch.stack([sym(posts[t],t)for t in range(1,len(posts))],1)\n terminal=.5*(terminal_values*terminal_mask).sum()/denom\n return total,current,future,terms,encoded,terminal");assert (S/'source/losses.py').read_text()==expected;checks['sole_loss_source_difference_terminal_measurement']=True
 source=(S/'source/train.py').read_text();assert "objective=total+terminal if arm=='terminal'else total" in source;assert "curr=current+terminal if arm=='terminal'else current" in source;checks['original_other_loss_weights_unchanged']=True
 for split,nroots,neps in [('train',40,120),('validation',10,30)]:
  eps=read(S/f'data/episodes_{split}.json');assert len(eps)==neps and len({e['root_id']for e in eps})==nroots;assert all(len(e['rows'])==4 for e in eps)
 tr=set(training['training_roots']);dv=set(training['selection_only_roots']);assert len(tr)==40 and len(dv)==10 and not tr&dv;checks['train_dev_root_disjoint']=True;checks['terminal_mask_real_h4_not_padding']='120training and30dev four-transition episodes; coefficient0.5/(batch*4), one real terminal per episode'
 frozen=read(S/'configs/models_frozen.json');assert len(frozen['sha256'])==12 and len(frozen['selection'])==4
 for path,h in frozen['sha256'].items():assert sha(path)==h,path
 changes=[]
 for seed in [0,1]:
  aa=np.load(S/f'configs/schedule_seed{seed}.npz');bb=np.load(CELL/f'configs/schedule_seed{seed}.npz');assert all(np.array_equal(aa[k],bb[k][:750])for k in aa.files);receipts=[read(S/f'results/train_{arm}_seed{seed}.json')for arm in ARMS];assert receipts[0]['initial_hash']==receipts[1]['initial_hash'];native=torch.load(CELL/f'models/joint_seed{seed}/final6000.pt',map_location='cpu',weights_only=True)['model']
  for arm,r in zip(ARMS,receipts):
   assert r['updates']==750 and r['original_loss_scalar_gradient_rng_exact'];assert [c['update']for c in r['candidates']]==[250,500,750];chosen=min(r['candidates'],key=lambda c:(c['dev_score'],c['update']));assert chosen==r['selected'];sealed=next(x for x in frozen['selection']if(x['seed'],x['arm'])==(seed,arm));assert sealed['path']==chosen['path'];new=torch.load(chosen['path'],map_location='cpu',weights_only=True)['model'];assert set(new)==set(native)
   for group,prefixes in [('encoder',['vision','geometry','robot_token','spatial','observation']),('dynamics',['skill','pointer','source','destination','executor','action_net','gru','prior']),('posterior',['posterior']),('shared_head',['object_query','object_symbol','scene_symbol'])]:
    names=[k for k in new if k.split('.')[0]in prefixes];changed=sum(not torch.equal(new[k],native[k])for k in names);diff=sum(float(((new[k]-native[k])**2).sum())for k in names);base=sum(float((native[k]**2).sum())for k in names);changes.append({'seed':seed,'arm':arm,'group':group,'changed_tensors':changed,'tensors':len(names),'relative_L2':(diff/max(base,1e-30))**.5})
 checks['paired_initial_models_schedules_and_budget']=True;checks['development_only_selection_exact']=True;checks['unchanged_model_architecture']=True;csvwrite(S/'results/weight_changes.csv',changes)
 parity=read(S/'results/baseline_parity.json');assert len(parity)==8 and all(x['all_arrays_exact']for x in parity);checks['native_previous_predictions_exact']=sum(x['original_rows_exact']for x in parity)
 rec=read(S/'data/analysis_records.json');rows=list(csv.DictReader((S/'results/metrics.csv').open()));counter=0
 for seed in [0,1]:
  for sp in ['validation','exploration']:
   for v in ['native','control','terminal','control_final750','terminal_final750']:
    rr=[r for r in rec if r['seed']==seed and r['variant']==v and r['mode']=='current'and r['target']>0 and(r['split']==sp or sp=='exploration'and r['split']in ['old_band','gap'])]
    target=next(x for x in rows if(x['seed'],x['split'],x['variant'],x['mode'],x['field'],x['cohort'])==(str(seed),sp,v,'current','all','all'));actual=correct(np.array([r['task_prob']for r in rr]),np.array([r['truth']for r in rr]))[:,-1];assert int(actual.sum())==int(float(target['count']))and len(rr)==int(target['n']);counter+=1
 checks['current_scores_independently_recomputed']=counter
 old=list(csv.DictReader((S/'results/numerical_attempt1/metrics.csv').open()));assert len(old)==len(rows)
 for a,b in zip(old,rows):
  assert {k:v for k,v in a.items()if k!='mean_nll'}=={k:v for k,v in b.items()if k!='mean_nll'}
  assert np.isfinite(float(b['mean_nll']))
 assert all(np.isfinite(r['nll'])for r in rec);checks['numerical_precision_fix_keeps_all_counts_and_intervals']=len(rows);checks['all_final_NLL_finite']=True

 for r in rec:
  if r['mode']in ['free_suffix','original_head_suffix']:assert 1<=r['origin']<r['target']<=4 and r['horizon']==r['target']-r['origin']
 evalsource=(S/'source/evaluate.py').read_text();assert "state,_=m.imagine(state,actions[:,target-1].repeat_interleave(sample,0));append('free_suffix'"in evalsource;assert "assert torch.equal(torch.cuda.get_rng_state(),before)"in evalsource;checks['suffix_no_future_observations_and_rng_restoration_static_audit']=True
 ef=read(S/'configs/evaluation_frozen.json');assert ef['model_selection_sha256']==sha(S/'configs/models_frozen.json');assert ef['protocol_sha256']==sha(S/'configs/protocol.json')
 for p,h in ef['files'].items():assert sha(p)==h
 checks['selection_frozen_before_exploration']=frozen['utc']<ef['utc'];assert checks['selection_frozen_before_exploration']
 for p,h in read(S/'configs/report_inputs.json')['sha256'].items():assert sha(p)==h
 report=read(S/'results/report_complete.json');assert all((S/'figures'/name).is_file()for name in report['figures']);assert(S/'REPORT.html').stat().st_size>100000;checks['report_figures']=len(report['figures'])
 save(S/'results/final_verification.json',{'utc':now(),'passed':True,'checks':checks,'new_verification_forwards':0,'limitations':['Only two seeds and historically explored20 roots','No failed-grasp or nonterminal-failure training examples','No real outgoing action after h4, so terminal-state future recovery untested','All original parameters continue; internal/readout component changes not uniquely attributable','No completed full-guide TS/LTL acceptance'],'no_collection_C_D_upload':True})
 files=sorted(p for p in S.rglob('*')if p.is_file()and not p.is_relative_to(S/'cache')and p.name!='manifest.json'and '__pycache__'not in str(p));manifest={str(p.relative_to(S)):sha(p)for p in files};save(S/'manifest.json',{'utc':now(),'status':'completed_sealed','files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':manifest});print('VERIFIED_SEALED',len(files),sha(S/'manifest.json'),flush=True)
if __name__=='__main__':main()
