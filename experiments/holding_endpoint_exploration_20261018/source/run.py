"""Exploratory skill-endpoint appendix; frozen probes and one prefix intervention."""
import sys
sys.path.insert(0,'/home/robot/research/projects/wm_symbolic/experiments/holding_readout_diagnostic_20261018/source')
import config as old
from config import *
from model import make
import torch,importlib.util,fcntl,csv,shutil,time
H=E;HR=R;E=EX/'holding_endpoint_exploration_20261018';R=P/'reports/holding_endpoint_exploration_20261018'
def condition(p):
 if p['skill']=='open_drawer':return 'open'
 if p['skill']=='place_in_drawer':return 'place'
 if p['skill']=='close_drawer':return 'empty_close'if p['branch']=='reverse_then_lift'else'close_after_place'
 return 'lift_c'if p['slot']==1 else'lift_d_after_place'if p['branch']=='place_then_other_lift'else'lift_d_after_openclose'
def discrete(p):
 p=np.array(p);out=p>=.5;out[...,6:]=False;out.reshape(-1,9)[np.arange(p.reshape(-1,9).shape[0]),6+p.reshape(-1,9)[:,6:].argmax(1)]=True;return out
def metrics(prob,y,roots):
 pred=np.array(prob)>=.5;y=np.array(y).astype(bool);roots=np.array(roots);empty=~y.any(1);out={'N':len(y),'roots':len(set(roots)),'H3_correct':int((pred==y).all(1).sum()),'H3_accuracy':float((pred==y).all(1).mean()),'empty_N':int(empty.sum()),'empty_FP':int(pred[empty].any(1).sum()),'colors':[]}
 for j,c in enumerate(COLORS):
  pos=y[:,j];neg=~pos;out['colors'].append({'color':c,'TP':int((pred[:,j]&pos).sum()),'FN':int((~pred[:,j]&pos).sum()),'FP':int((pred[:,j]&neg).sum()),'TN':int((~pred[:,j]&neg).sum()),'positive_roots':len(set(roots[pos])),'negative_roots':len(set(roots[neg]))})
 return out
def reliable(m):return m['roots']>=6 and m['H3_accuracy']>=.95 and all((c['FP']/(c['FP']+c['TN'])<=.05 if c['FP']+c['TN']else True)and(c['FN']/(c['FN']+c['TP'])<=.1 and c['positive_roots']>=2 if c['FN']+c['TP']else True)for c in m['colors'])
def output(name,rows):
 save(E/'results'/f'{name}.json',rows)
 if isinstance(rows,list)and rows:
  keys=list(dict.fromkeys(k for r in rows for k in r))
  with(R/f'{name}.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in r.items()}for r in rows)
def main():
 assert not E.exists();old.verify()
 for root in [H,D]:
  for p,h in read(root/'configs/delivery_manifest.json')['sha256'].items():assert sha(p)==h,p
 for sub in ['source','configs','results','logs']:(E/sub).mkdir(parents=True)
 R.mkdir(parents=True);shutil.copy2(__file__,E/'source/run.py')
 cfg=read(D/'configs/protocol.json');refs={str(p):sha(p)for p in [H/'configs/delivery_manifest.json',H/'configs/final_heads_frozen.json',D/'configs/delivery_manifest.json',D/'results/predictions.json',D/'data/combo/public/inputs.npz',D/'data/combo/public/pairs.json',D/'data/combo/_private/labels.json',D/'results/root_contexts.json',REP/'source/model.py',WHOLE/'configs/query_normalization.npz']}
 for form in FORMS:
  refs[str(H/'configs'/f'normalization_{form}.npz')]=sha(H/'configs'/f'normalization_{form}.npz')
  for seed in [0,1]:p=H/'models'/f'{form}_seed{seed}_update0600.pt';refs[str(p)]=sha(p)
 for seed in [0,1]:
  p=cfg['query_heads'][str(seed)]['original']['path'];refs[p]=sha(p)
  for wm in ['mixed','latent']:
   p=cfg['models'][f'{wm}_seed{seed}']['path'];refs[p]=sha(p);p=D/'results'/f'latents_{wm}_seed{seed}.npz';refs[str(p)]=sha(p)
 protocol={'utc':now(),'scope':'EXPLORATORY SKILL ENDPOINTS ONLY; global actual reliability failure preserved','no_training_or_threshold_changes':True,'all_conditions':['open','empty_close','place','close_after_place','lift_c','lift_d_after_place','lift_d_after_openclose'],'report_all':True,'agreement_subset':'All four actual probe H3 equal physical truth; supplementary descriptive tags only, never substitutes for overall results','condition_eligibility':'At least6 roots; EVERY fixed probe H3>=.95, per-color FPR<=.05; for present positives FNR<=.1 and >=2 positive roots. No claim on absent-positive recall. Evaluate all fixed heads; no model selection.','predicted_readout':'Only eligible WHOLE conditions, matched seed of world-model and probe; both input forms, all rows, preserve disagreements with original query.','prefix':'Exactly once per fixed mixed/alphaZ model and seed: all11 complete normal roots; replace after-lift state with actual z, then predict place -> close with no further observation. Original Query all9; independent of global probe gate.','prefix_reproduction_tolerance':1e-5,'examples':'fixed D_identity_blue_red_00, seed0,alphaZ; no result-based example search','D':'previously viewed D diagnostic; not independent validation or model selection','training_updates':0,'physical_calls':0};save(E/'configs/protocol.json',protocol);refs[str(E/'configs/protocol.json')]=sha(E/'configs/protocol.json');refs[str(E/'source/run.py')]=sha(E/'source/run.py');save(E/'configs/frozen.json',{'utc':now(),'before_D_actual_probe_evaluation':True,'sha256':refs})
 pairs=read(D/'data/combo/public/pairs.json');z=np.load(D/'data/combo/public/inputs.npz')['z'];labs=read(D/'data/combo/_private/labels.json');oldpp=read(D/'results/predictions.json');pidx={p['id']:i for i,p in enumerate(pairs)};outs={};heads={};norms={};torch.set_num_threads(4)
 for form in FORMS:
  nm=np.load(H/'configs'/f'normalization_{form}.npz');norms[form]=dict(nm)
  for seed in [0,1]:
   q=make(form).eval().requires_grad_(False);q.load_state_dict(torch.load(H/'models'/f'{form}_seed{seed}_update0600.pt',map_location='cpu',weights_only=True)['state']);heads[(form,seed)]=q;x=((features(z[[p['after']for p in pairs]],form)-nm['mean'])/nm['std']).astype(np.float32)
   with torch.inference_mode():outs[(form,seed)]=q(torch.tensor(x)).squeeze(-1).sigmoid().numpy()
 actual=[]
 for i,p in enumerate(pairs):
  lab=labs[str(p['after'])];assert all(lab['v'][:3]);probs={f'{form}_seed{seed}':outs[(form,seed)][i].tolist()for form in FORMS for seed in [0,1]};agreed=all(np.array_equal(np.array(pr)>=.5,lab['y'][:3])for pr in probs.values());actual.append({**{k:p[k]for k in ['id','root_id','target','other_target','branch','slot','skill','before','after']},'condition':condition(p),'truth':lab['y'][:3],'all_probes_actual_correct':agreed,'probe_probability':probs})
 output('actual_endpoints',actual);summaries=[];roots=[];agreement=[];conditions=protocol['all_conditions']
 for cond in ['all']+conditions:
  rr=[r for r in actual if cond=='all'or r['condition']==cond];agreement.append({'condition':cond,'all_N':len(rr),'all_roots':len({r['root_id']for r in rr}),'agreed_N':sum(r['all_probes_actual_correct']for r in rr),'agreed_roots':len({r['root_id']for r in rr if r['all_probes_actual_correct']}),'agreed_ids':[r['id']for r in rr if r['all_probes_actual_correct']],'disagreed_ids':[r['id']for r in rr if not r['all_probes_actual_correct']]})
  for form in FORMS:
   for seed in [0,1]:
    key=f'{form}_seed{seed}';m=metrics([r['probe_probability'][key]for r in rr],[r['truth']for r in rr],[r['root_id']for r in rr]);summaries.append({'condition':cond,'form':form,'seed':seed,**m,'condition_pass_for_this_head':reliable(m)})
    for rid in sorted({r['root_id']for r in rr}):
     xx=[r for r in rr if r['root_id']==rid];roots.append({'condition':cond,'form':form,'seed':seed,'root_id':rid,**metrics([r['probe_probability'][key]for r in xx],[r['truth']for r in xx],[rid]*len(xx))})
 eligibility={c:all(m['condition_pass_for_this_head']for m in summaries if m['condition']==c)for c in conditions};output('actual_condition_metrics',summaries);output('actual_root_metrics',roots);output('actual_agreement_subset',agreement);save(E/'results/condition_eligibility.json',eligibility);print('ENDPOINT_ACTUAL',json.dumps(eligibility),flush=True)
 predicted=[]
 with torch.inference_mode():
  for seed in [0,1]:
   for wm in ['mixed','latent']:
    lat=np.load(D/'results'/f'latents_{wm}_seed{seed}.npz');assert lat['pair_ids'].tolist()==[p['id']for p in pairs];ix=[i for i,p in enumerate(pairs)if eligibility[condition(p)]]
    for path in ['one_step','recursive']:
     values={}
     for form in FORMS:
      nm=norms[form];x=((features(lat[path][ix],form)-nm['mean'])/nm['std']).astype(np.float32);values[form]=heads[(form,seed)](torch.tensor(x)).squeeze(-1).sigmoid().numpy()
     for j,i in enumerate(ix):
      p=pairs[i];orig=next(r for r in oldpp if r['id']==p['id']and r['seed']==seed and r['world_model']==wm and r['query_arm']=='original');predicted.append({'id':p['id'],'root_id':p['root_id'],'condition':condition(p),'branch':p['branch'],'slot':p['slot'],'seed':seed,'world_model':wm,'path':path,'truth':labs[str(p['after'])]['y'][:3],'original_actual_probability':orig['actual_probability'][:3],'original_probability':orig[path+'_probability'][:3],**{form+'_probability':values[form][j].tolist()for form in FORMS},'all_probes_actual_correct':actual[i]['all_probes_actual_correct']})
 output('predicted_readouts',predicted);comparison=[]
 for seed,wm,path,cond,form in [(s,w,p,c,f)for s in [0,1]for w in ['mixed','latent']for p in ['one_step','recursive']for c in conditions if eligibility[c]for f in FORMS]:
  rr=[r for r in predicted if r['seed']==seed and r['world_model']==wm and r['path']==path and r['condition']==cond];truth=np.array([r['truth']for r in rr]);a=np.array([r['original_probability']for r in rr])>=.5;b=np.array([r[form+'_probability']for r in rr])>=.5;ae=(a!=truth).any(1);be=(b!=truth).any(1);comparison.append({'seed':seed,'world_model':wm,'path':path,'condition':cond,'form':form,'N':len(rr),'roots':len({r['root_id']for r in rr}),'both_wrong':int((ae&be).sum()),'original_only_wrong':int((ae&~be).sum()),'probe_only_wrong':int((~ae&be).sum()),'both_correct':int((~ae&~be).sum()),'same_wrong_fact_cells':int(((a!=truth)&(b!=truth)).sum()),'probe_metrics':metrics([r[form+'_probability']for r in rr],truth,[r['root_id']for r in rr])})
 output('prediction_comparison',comparison)
 # One intervention, then two consecutive predictions. No observation replacement at step4.
 spec=importlib.util.spec_from_file_location('frozen_wm_model',REP/'source/model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);qn=np.load(WHOLE/'configs/query_normalization.npz');mean=torch.tensor(qn['mean'],device='cuda');std=torch.tensor(qn['std'],device='cuda');eligible_roots=sorted(r['root_id']for r in read(D/'results/root_contexts.json')if r['context_matched']);rr=[[next(p for p in pairs if p['root_id']==rid and p['branch']=='normal'and p['slot']==slot)for rid in eligible_roots]for slot in [2,3]];assert len(rr[0])==11;prefix=[];maxdiff=0
 with torch.inference_mode():
  for seed in [0,1]:
   q=mod.Query().cuda().eval().requires_grad_(False);q.load_state_dict(torch.load(cfg['query_heads'][str(seed)]['original']['path'],map_location='cuda',weights_only=True)['state'])
   for wm in ['mixed','latent']:
    allstate=torch.load(cfg['models'][f'{wm}_seed{seed}']['path'],map_location='cuda',weights_only=True)['state'];m=mod.Transition().cuda().eval().requires_grad_(False);m.load_state_dict({k.removeprefix('transition.'):v for k,v in allstate.items()if k.startswith('transition.')});cur=torch.tensor(z[[p['before']for p in rr[0]]],device='cuda');saved=[]
    for depth,rows in enumerate(rr):
     cur=m(cur,torch.tensor([p['action']for p in rows],device='cuda'));probs=mod.probs(q((cur-mean)/std)).cpu().numpy();saved.append(cur.cpu().numpy())
     for i,p in enumerate(rows):
      orig=next(r for r in oldpp if r['id']==p['id']and r['seed']==seed and r['world_model']==wm and r['query_arm']=='original');diff=float(np.max(np.abs(probs[i]-orig['one_step_probability'])))if depth==0 else None
      if diff is not None:assert diff<1e-5,diff;maxdiff=max(maxdiff,diff)
      prefix.append({'root_id':p['root_id'],'seed':seed,'world_model':wm,'step':depth+3,'skill':p['skill'],'target':p['target'],'truth':labs[str(p['after'])]['y'],'valid':labs[str(p['after'])]['v'],'actual_probability':orig['actual_probability'],'baseline_recursive_probability':orig['recursive_probability'],'replacement_probability':probs[i].tolist(),'matches_saved_one_step_maxdiff':diff,'only_replaced_after_lift':True})
    np.savez_compressed(E/'results'/f'prefix_{wm}_seed{seed}.npz',root_ids=np.array(eligible_roots),after_place=saved[0],after_close=saved[1])
 output('prefix_predictions',prefix);prefixmetrics=[]
 for seed,wm,step,path in [(s,w,t,p)for s in [0,1]for w in ['mixed','latent']for t in [3,4]for p in ['actual','baseline_recursive','replacement']]:
  rr2=[r for r in prefix if r['seed']==seed and r['world_model']==wm and r['step']==step];truth=np.array([r['truth']for r in rr2]);pred=discrete([r[path+'_probability']for r in rr2]);prefixmetrics.append({'seed':seed,'world_model':wm,'step':step,'path':path,'N':11,'Holding_correct':int((pred[:,:3]==truth[:,:3]).all(1).sum()),'full9_correct':int((pred==truth).all(1).sum()),'wrong_facts_counts':{str(j):int((pred[:,j]!=truth[:,j]).sum())for j in range(9)}})
 output('prefix_metrics',prefixmetrics);save(E/'results/run_audit.json',{'utc':now(),'probe_training_updates':0,'new_physical_calls':0,'original_model_parameter_updates':0,'actual_boundaries_reported':135,'withheld_unreached_slots':9,'prefix_roots':11,'prefix_model_conditions':4,'prefix_state_predictions':88,'prefix_replacements_per_root_model':1,'place_reproduces_saved_one_step_maxdiff':maxdiff,'global_gate_unchanged':True,'exploratory_D_no_model_selection':True});print('ENDPOINT_PREFIX_DONE',json.dumps(prefixmetrics),flush=True)
if __name__=='__main__':main()
