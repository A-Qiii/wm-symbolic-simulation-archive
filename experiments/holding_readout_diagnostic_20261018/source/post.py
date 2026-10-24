import sys
sys.path.insert(0,'/home/robot/research/projects/wm_symbolic/experiments/holding_readout_diagnostic_20261018/source')
from config import *
import torch,csv,collections,datetime,shutil
def main():
 verify();gate=read(E/'results/reliability_decision.json');assert not any(gate['eligible_forms'].values())
 save(E/'results/attribution_stop.json',{'utc':now(),'reason':'No input form reliable in both seeds on prespecified actual AB/C validation. C_validation has observed high no-Holding false positives; old_C validation also lacks enough held-color root coverage.','independent_readout_on_predicted_z':False,'independent_readout_on_D':False,'prefix_replacement_executed':False,'nearest_neighbor_executed':False,'extra_training_updates':0,'no_formal_model_changes':True,'original_saved_one_step_recursive_comparison_only':True})
 # Reuse original probabilities for the three specific target boundaries.
 pp=read(D/'results/predictions.json');pairs=read(D/'data/combo/public/pairs.json');labels=read(D/'data/combo/_private/labels.json');eligible={r['root_id']for r in read(D/'results/root_contexts.json')if r['context_matched']};rows=[]
 for seed,wm,(branch,step,name)in [(s,w,x)for s in [0,1]for w in ['mixed','latent']for x in [('reverse_then_lift',2,'empty_close'),('normal',3,'after_place'),('normal',4,'close_after_place')]]:
  rr=[r for r in pp if r['seed']==seed and r['world_model']==wm and r['query_arm']=='original'and r['root_id']in eligible and r['branch']==branch and r['slot']==step-1];assert len(rr)==11;truth=np.array([labels[str(r['after'])]['y'][:3]for r in rr]);assert not truth.any()
  wrong={k:np.array([[p>=.5 for p in r[k+'_probability'][:3]]for r in rr]).any(1)for k in METHODS}
  rows.append({'seed':seed,'world_model':wm,'boundary':name,'roots':11,**{k+'_Holding_FP_roots':int(v.sum())for k,v in wrong.items()},'one_and_recursive_wrong':int((wrong['one_step']&wrong['recursive']).sum()),'only_recursive_wrong':int((~wrong['one_step']&wrong['recursive']).sum()),'only_one_step_wrong':int((wrong['one_step']&~wrong['recursive']).sum())})
 save(E/'results/original_boundary_comparison.json',rows)
 # Describe C error locations without changing the frozen gate or running another head.
 states=read(PRIOR/'data/ABC/public/states.json');d=np.load(E/'data/evaluation/C_validation.npz');pred=read(E/'results/actual_predictions.json');strata=[]
 for form in FORMS:
  for seed in [0,1]:
   rr=sorted([r for r in pred if r['form']==form and r['seed']==seed and r['cohort']=='C_validation'],key=lambda r:r['index']);assert len(rr)==len(d['z']);groups=collections.defaultdict(list)
   for r in rr:
    i=r['index'];s=states[int(r['state_id'].split(':')[1])];assert s['root_id']==r['root_id'];y=d['y'][i].astype(bool);p=np.array(r['probability'])>=.5;after=any(t.startswith('after_call:')for t in s['tags']);key='skill_endpoint'if after else'intermediate_or_before_call';groups[key].append((i,y,p,s))
   for group,items in groups.items():strata.append({'form':form,'seed':seed,'group':group,'states':len(items),'roots':len({s['root_id']for i,y,p,s in items}),'all3_correct':sum(np.array_equal(y,p)for i,y,p,s in items),'no_Holding_states':sum(not y.any()for i,y,p,s in items),'no_Holding_any_FP':sum(not y.any()and p.any()for i,y,p,s in items),'FP_phases':dict(collections.Counter(s['phase']for i,y,p,s in items if not y.any()and p.any()))})
 save(E/'results/C_error_locations.json',strata)
 # Independent actual-prediction recount, root split and schedule invariants.
 actualchecks=0
 for m in read(E/'results/actual_metrics.json'):
  q=np.load(E/'data/evaluation'/f"{m['cohort']}.npz");rr=sorted([r for r in pred if r['form']==m['form']and r['seed']==m['seed']and r['cohort']==m['cohort']],key=lambda r:r['index']);pr=np.array([r['probability']for r in rr])>=.5;yy=q['y'].astype(bool);vv=q['v'].astype(bool);assert sum(vv.all(1)&(pr==yy).all(1))==m['all3_correct'];empty=vv.all(1)&~yy.any(1);assert sum(pr[empty].any(1))==m['empty_any_FP']
  for j,c in enumerate(m['colors']):
   for key,test in [('TP',pr[:,j]&yy[:,j]),('FN',~pr[:,j]&yy[:,j]),('FP',pr[:,j]&~yy[:,j]),('TN',~pr[:,j]&~yy[:,j])]:assert int((test&vv[:,j]).sum())==c[key];actualchecks+=1
 train=np.load(E/'data/train/states.npz');trroots=set(train['root_ids']);assert len(trroots)==36
 for f in(E/'data/evaluation').glob('*.npz'):assert trroots.isdisjoint(np.load(f)['root_ids'])
 for seed in [0,1]:
  sc=np.load(E/'configs'/f'schedule_seed{seed}.npz')['indices'];assert sc.shape==(600,32) and sc.min()>=0 and sc.max()<210
 runs=read(E/'results/training.json');assert len(runs['runs'])==4 and all(r['updates']==600 and r['batch_states']==32 for r in runs['runs'])
 # Historical performance is recovered from original records; absent fields remain absent.
 cfg=read(D/'configs/protocol.json');st=torch.load(cfg['models']['latent_seed0']['path'],map_location='cpu',weights_only=True)['state'];hist=[]
 for wm in ['mixed','latent']:
  for seed in [0,1]:
   p=ABL/'results'/f'train_{wm}_seed{seed}.json';r=read(p);hist.append({'experiment':'transition_objective_ablation','arm':wm,'seed':seed,'trainable_parameters':sum(v.numel()for k,v in st.items()if k.startswith('transition.')),'whole_learned_model_parameters':sum(v.numel()for v in st.values()),'batch':64,'multi_start_batch':48,'updates':r['updates'],'device':'CUDA from executed source','recorded_elapsed_seconds':r.get('elapsed_seconds'),'peak_GPU_memory':None,'GPU_utilization':None,'source':str(p)})
 for name,folder,params,dev in [('formal_query_last_continuation',FIX,25473,'CUDA'),('whole_query_coverage',WHOLE,25473,'CUDA'),('object_In_head',OBJ,2113,'CPU4threads')]:hist.append({'experiment':name,'trainable_parameters':params,'batch':32,'updates_each':600,'device':dev,'recorded_elapsed_seconds':None,'peak_GPU_memory':None,'GPU_utilization':None,'absence_note':'These original logs did not record elapsed/peak/utilization; no retrospective estimate from current idle GPU.'})
 save(E/'results/historical_performance.json',hist)
 samples=[]
 for line in(E/'logs/gpu_utilization.csv').read_text().splitlines():
  vals=[x.strip()for x in line.split(',')];samples.append({'host_local_timestamp':vals[0],'gpu_util_percent':int(vals[1]),'memory_controller_util_percent':int(vals[2]),'device_used_MiB':int(vals[3]),'device_total_MiB':int(vals[4])})
 save(E/'results/gpu_monitor_summary.json',{'samples':samples,'sample_count':len(samples),'gpu_util_range':[min(s['gpu_util_percent']for s in samples),max(s['gpu_util_percent']for s in samples)],'gpu_util_sample_mean':sum(s['gpu_util_percent']for s in samples)/len(samples),'device_used_MiB_range':[min(s['device_used_MiB']for s in samples),max(s['device_used_MiB']for s in samples)],'monitor_host_timezone_offset_hours':datetime.datetime.now().astimezone().utcoffset().total_seconds()/3600,'training_utc_times_in_training_json':True,'limits':'8 coarse device-level samples include setup and short gaps; do not infer sustained occupancy or per-run utilization. PyTorch peak includes its persistent workspaces, whereas nvidia-smi includes CUDA context and other device usage.'})
 save(E/'results/verification.json',{'utc':now(),'passed':True,'actual_confusion_cells_recounted':actualchecks,'train_validation_root_overlap':0,'four_runs_each600_batch32':True,'frozen_references_checked':len(read(E/'configs/frozen.json')['sha256']),'no_test_selection':True,'prediction_attribution_stopped_as_required':True})
 for name in ['actual_metrics','original_boundary_comparison','historical_transition_coverage','exact_sequence_coverage','historical_query_sampling','C_error_locations','historical_performance']:
  rr=read(E/'results'/f'{name}.json');keys=list(dict.fromkeys(k for r in rr for k in r))
  with(R/f'{name}.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in r.items()}for r in rr)
 shutil.copy2(__file__,E/'source/post.py');print('HOLDING_POST_READY',len(strata),'stratum rows',flush=True)
if __name__=='__main__':main()
