from config import *
import time,collections,shutil,torch
def main():
 assert not E.exists();started=time.perf_counter()
 for d in ['configs','data/train','data/evaluation','source','results','logs','models']:(E/d).mkdir(parents=True)
 R.mkdir(parents=True);refs={}
 def reg(p):p=Path(p);refs[str(p)]=sha(p);return p
 for mf in [D/'configs/delivery_manifest.json',OBJ/'configs/delivery_manifest.json']:
  reg(mf)
  for p,h in read(mf)['sha256'].items():assert sha(p)==h,p
 # Check actual saved latents/boundaries/identity/action contracts, without model forward.
 inputs=np.load(reg(D/'data/combo/public/inputs.npz'));pairs=read(reg(D/'data/combo/public/pairs.json'));states=read(reg(D/'data/combo/public/states.json'));labs=read(reg(D/'data/combo/_private/labels.json'));prob=read(reg(D/'results/predictions.json'));rootmeta={r['root_id']:r for r in read(reg(D/'configs/roots.json'))};align=[]
 for p in pairs:
  assert np.array_equal(action(p['skill']),p['action']);assert states[p['before']]['root_id']==p['root_id']==states[p['after']]['root_id'];assert states[p['after']]['branch']==p['branch']
  if p['skill'].startswith('lift'):
   expected=p['other_target']if p['branch']in ['reverse_then_lift','place_then_other_lift']and p['slot']==3 else p['target'];assert p['skill'].split('_')[1]==expected
 for seed in [0,1]:
  for wm in ['mixed','latent']:
   lat=np.load(reg(D/'results'/f'latents_{wm}_seed{seed}.npz'));assert lat['pair_ids'].tolist()==[p['id']for p in pairs];assert np.array_equal(lat['actual'],inputs['z'][[p['after']for p in pairs]])
   for p in pairs:
    if p['slot']>0:prev=next(q for q in pairs if q['root_id']==p['root_id']and q['branch']==p['branch']and q['slot']==p['slot']-1);assert prev['after']==p['before']
   ix=[i for i,p in enumerate(pairs)if p['slot']==0];assert np.array_equal(lat['one_step'][ix],lat['recursive'][ix]);align.append({'seed':seed,'world_model':wm,'actual_latent_maxdiff':0,'first_step_one_recursive_equal':True,'boundaries':len(pairs)})
 for seed in [0,1]:
  for pid in [p['id']for p in pairs]:
   rr=[p for p in prob if p['seed']==seed and p['id']==pid];base=rr[0]['actual_probability']
   assert all(p['actual_probability'][:3]==base[:3]for p in rr)
   for wm,path in [(w,k)for w in ['mixed','latent']for k in METHODS]:
    q=[p for p in rr if p['world_model']==wm];assert all(p[path+'_probability'][:3]==q[0][path+'_probability'][:3]for p in q)
 for p in [REP/'source/model.py',REP/'source/registry.py',REP/'source/common.py',D/'source/predict.py',PRIOR/'source/prepare_transition.py',ABL/'source/train.py',ABL/'source/losses.py']:reg(p)
 save(E/'results/alignment.json',{'passed':True,'records':align,'color_order':COLORS,'query_Holding_indices':[0,1,2],'action_8':'open,lift,place,close,red,blue,pink,posthold/96; final component is NOT physical Holding','entity_alignment':'lift target checked against c/d and saved current-image color order; no ground-truth entity swap','actual_query_Holding_identical_across_heads_and_world_models':True,'recursive_no_observation_injection':True,'no_transition_forward':True})
 # Historical transition training: exact scheduled counts with physical-label provenance.
 pack=PRIOR/'data/transition_train';ps=read(reg(pack/'pairs.json'));ss=read(reg(pack/'states.json'));oldlabs=read(reg(REP/'data/_private/labels.json'));newlabs=read(reg(PRIOR/'data/ABC/_private/labels.json'));cache={}
 def label(i):
  s=ss[i]
  if s.get('dataset_group')=='new':return newlabs[str(int(s['id'][1:]))]
  if str(i)in oldlabs:return oldlabs[str(i)]
  src=Path(s.get('source',''));f=src.parent/'labels_ownership_v2_private.json'
  if f.exists()and src.name in ['initial_public.npz','final_public.npz']:
   if f not in cache:cache[f]=read(reg(f))
   l=cache[f]['before'if src.name=='initial_public.npz'else'after'];h=l['holding_derived'];v=l['categorical_valid'][:3];q=l['continuous']['drawer_m'];return {'y':h+[False]*3+[q<=.02,q>=.18,.02<q<.18],'v':v+[False]*3+[min(abs(q-.02),abs(q-.18))>.003]*3}
  return None
 state_labels=[label(i)for i in range(len(ss))];categories=[];prevs=collections.defaultdict(list)
 for i,p in enumerate(ps):prevs[(p['root_id'],p['after'])].append(i)
 for i,p in enumerate(ps):
  b,a=state_labels[p['before']],state_labels[p['after']];cats=['all'];valid=b is not None and a is not None and all(b['v'][:3])and all(a['v'][:3]);family='lift'if p['skill'].startswith('lift')else p['skill'].split('_')[0]
  if valid:
   cats+=['before_held'if any(b['y'][:3])else'before_empty','after_held'if any(a['y'][:3])else'after_empty']
   if family=='close'and not any(b['y'][:3])and b['v'][7]and b['y'][7]:
    cats+=['empty_open_to_close'];prior=[ps[j]for j in prevs[(p['root_id'],p['before'])]]
    if any(x['skill']=='open_drawer'for x in prior):cats+=['direct_open_then_empty_close']
    if any(x['skill']=='place_in_drawer'for x in prior):cats+=['place_then_empty_close']
   if family=='lift':cats+=['lift_held'if any(a['y'][:3])else'lift_empty']
   if family=='place':cats+=['place_released'if any(b['y'][:3])and not any(a['y'][:3])else'place_other']
  else:cats+=['Holding_label_unavailable_or_masked']
  categories.append(cats);assert np.array_equal(action(p['skill'],p['action'][7]),p['action'])
 coverage=[]
 for seed in [0,1]:
  sc=np.load(reg(ABL/'configs'/f'schedule_seed{seed}.npz'))
  for updates in [750,1000]:
   arrays={'single':sc['single'][:updates].ravel(),'multi_forward':sc['multi'][3:updates:4].ravel(),'multi_supervised':sc['multi'][3:updates:4,:,1:].ravel()}
   for kind,ix in arrays.items():
    counts=np.bincount(ix,minlength=len(ps));assert all(ps[i]['split']=='train'for i in np.flatnonzero(counts))
    for cat in sorted(set(c for cs in categories for c in cs)):
     for hold in ['all',0,1]:
      ids=[i for i,cs in enumerate(categories)if cat in cs and counts[i]>0 and(hold=='all'or ps[i]['action'][7]==hold)]
      coverage.append({'seed':seed,'updates_prefix':updates,'sampling_stream':kind,'category':cat,'posthold_contract':hold,'unique_transitions':len(ids),'independent_roots':len({ps[i]['root_id']for i in ids}),'actual_draws':int(counts[ids].sum()),'roots':sorted({ps[i]['root_id']for i in ids})})
 save(E/'results/historical_transition_coverage.json',coverage);save(E/'results/transition_label_provenance.json',{'labelled_states':sum(l is not None for l in state_labels),'total_states':len(ss),'unknown_not_filled':True,'core_ownership_files':len(cache),'meaning':'Counts refer to latest objective-ablation optimization stream;750 selected mixed,1000 selected latent; preceding checkpoint history not included.'})
 # Diagnostic training: all actual states from old AB training and six existing new AB training roots.
 qn=np.load(reg(WHOLE/'configs/query_normalization.npz'));datasets={}
 for name in ['historical_AB_training','new_AB_train','new_AB_validation','old_C_selection','old_C_validation','C_selection','C_validation','D_seen_diagnostic']:
  d=dict(np.load(reg(WHOLE/'data/evaluation'/f'{name}.npz')));datasets[name]=d;d['raw_z']=(d['z']*qn['std']+qn['mean']).astype(np.float32)
 train={k:np.concatenate([datasets[n][k]for n in ['historical_AB_training','new_AB_train']])for k in ['raw_z','y','v','root_ids','state_ids']};train['state_ids']=np.array(['oldpool:'+str(x)for x in datasets['historical_AB_training']['state_ids']]+['combo:'+str(x)for x in datasets['new_AB_train']['state_ids']]);assert len(train['raw_z'])==210 and len(set(train['root_ids']))==36
 for name,d in datasets.items():
  if name in ['historical_AB_training','new_AB_train']:continue
  assert set(train['root_ids']).isdisjoint(d['root_ids']);np.savez_compressed(E/'data/evaluation'/f'{name}.npz',z=d['raw_z'],y=d['y'][:,:3],v=d['v'][:,:3],root_ids=d['root_ids'],state_ids=d['state_ids'])
 np.savez_compressed(E/'data/train/states.npz',z=train['raw_z'],y=train['y'][:,:3],v=train['v'][:,:3],root_ids=train['root_ids'],state_ids=train['state_ids'])
 norms={}
 for form in FORMS:
  x=features(train['raw_z'],form);mean=x.reshape(-1,x.shape[-1]).mean(0);std=np.maximum(x.reshape(-1,x.shape[-1]).std(0),.01);norms[form]={'mean':mean,'std':std};np.savez_compressed(E/'configs'/f'normalization_{form}.npz',mean=mean,std=std)
 rootids=sorted(set(train['root_ids']));indices={r:np.flatnonzero(train['root_ids']==r)for r in rootids};newbudget=[]
 for seed in [0,1]:
  rng=np.random.default_rng(17078010+seed);sc=np.array([[rng.choice(indices[rootids[j]])for j in rng.integers(0,36,size=32)]for _ in range(600)]);np.savez_compressed(E/'configs'/f'schedule_seed{seed}.npz',indices=sc);yy=train['y'][sc.ravel(),:3];vv=train['v'][sc.ravel(),:3];newbudget.append({'seed':seed,'updates':600,'batch':32,'draws':19200,'roots':36,'held_positive_draws_per_color':(yy*vv).sum(0),'empty_state_draws':int((yy.sum(1)==0).sum()),'draws_by_root':{r:int((train['root_ids'][sc.ravel()]==r).sum())for r in rootids}})
 save(E/'results/new_training_sampling.json',newbudget)
 # Recount the selected formal Query's last 150 updates and its full 600-update run.
 qcoverage=[]
 oldpool=np.load(reg(FIX/'data/query_train/empty_open.npz'))
 for seed in [0,1]:
  sc=np.load(reg(FIX/'configs'/f'schedule_seed{seed}.npz'))['empty_open']
  for u in [150,600]:
   ix=sc[:u].ravel();yy=oldpool['y'][ix,:3];qcoverage.append({'seed':seed,'updates_prefix':u,'draws':len(ix),'roots':len(set(oldpool['root_ids'][ix])),'empty_draws':int((yy.sum(1)==0).sum()),'held_positive_draws_per_color':yy.sum(0),'selected_checkpoint':u==150,'note':'Only last continuation stage, excluding ancestors; query receives states, not action sequences.'})
 save(E/'results/historical_query_sampling.json',qcoverage)
 pos=float((train['y'][:,:3]*train['v'][:,:3]).sum());neg=float(((1-train['y'][:,:3])*train['v'][:,:3]).sum())
 protocol={'utc':now(),'purpose':'bounded independent Holding readout diagnostic, not a formal-model replacement','seeds':[0,1],'initialization_seeds':[17078000,17078001],'forms':{'object':'own contextualized object64 ->32SiLU->1 shared','context':'own64 + scene64 + mean_objects64 ->32SiLU->1 shared'},'updates':600,'batch':32,'optimizer':'AdamW lr0.001 wd0.001 clip1; fixed final600 only','positive_weight':neg/pos,'loss':'shared pooled valid-positive weighted BCE; mean valid objects per state then mean batch; root-balanced fixed schedule','training_states':210,'training_roots':36,'train_roots':rootids,'normalization':'training states only, per input feature dimension; std floor.01','threshold':.5,'no_inputs':['action','coordinates','simulation state','original query output','labels as input'],'no_updates':['encoder','world model','formal query'],'evaluation_order':'actual AB validation and C first; D actual descriptive only; saved predicted latents only if a form is reliable in both initializations','reliability_diagnostic_gate':{'not_a_project_acceptance_gate':True,'cohorts':['new_AB_validation','old_C_validation','C_validation'],'both_initializations_required_per_form':True,'each_cohort':{'all_three_Holding_accuracy_min':.95,'empty_state_any_FP_rate_max':.05,'each_color_FPR_max':.05,'each_color_FNR_max':.10,'each_color_positive_roots_min':2},'if_fail':'Stop prediction attribution for that input form; report failure, no retuning. C_selection descriptive, no selection. D not used in this eligibility gate.'},'max_fits':4,'postfit_no_new_training':True,'D':'existing diagnostic only; no selection','prefix_replacement':'not performed if one-step already shares the target Holding error; reuse saved one-step results','device':'CUDA serial, no batch change; performance measured separately from quality'}
 save(E/'configs/protocol.json',protocol);save(E/'configs/references.json',{'sha256':refs});save(E/'results/preparation_timing.json',{'seconds':time.perf_counter()-started,'includes':'hash validation, metadata alignment/coverage, reading saved z, normalization, schedules; no encoder/model inference','gpu_used':False});print('HOLDING_PREFLIGHT_READY',json.dumps({'train_states':210,'train_roots':36,'pos_weight':neg/pos,'alignment':'passed','coverage_rows':len(coverage)}),flush=True)
if __name__=='__main__':main()
