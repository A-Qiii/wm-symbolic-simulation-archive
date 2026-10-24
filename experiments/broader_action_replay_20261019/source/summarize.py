from common import *
import collections
def metric(rr,values):
 root=collections.defaultdict(list)
 for r,v in zip(rr,values):root[r['root_id']].append(float(v))
 return {'n':len(rr),'correct':sum(values),'roots':len(root),'root_mean':float(np.mean([np.mean(v)for v in root.values()]))if root else None}
def main():
 sys.addaudithook(guard);rows=read(S/'data/forecast_records.json');cur=read(S/'data/current_records.json');chains=read(S/'data/chains.json');summary=[];starts=[];outcomes=[]
 for scope,splits in [('unseen_all',['old_band','gap']),('new_head_training',['train'])]:
  for scene in ['all','A','B']:
   for seed in [0,1]:
    for model in ['RSSM','GT_relations','RSSM_state','old_head_state','coverage_head_state','GT_physics','GT_persistence']:
     for mode,h in [('one_step',1),('free',4)]:
      rr=[r for r in rows if r['split']in splits and r['seed']==seed and r['model']==model and r['mode']==mode and r['horizon']==h and(scene=='all'or r['scene']==scene)];summary.append({'scope':scope,'scene':scene,'seed':seed,'model':model,'mode':mode,**metric(rr,[r['all_correct']for r in rr]),'extra_binary':sum(r['extra_binary']for r in rr),'missing_binary':sum(r['missing_binary']for r in rr)})
     cc=[c for c in chains if c['split']in splits and c['seed']==seed and(scene=='all'or c['scene']==scene)];summary.append({'scope':scope,'scene':scene,'seed':seed,'model':model,'mode':'prefix4',**metric(cc,[c['models'][model]['prefix'][3]for c in cc]),'extra_binary':None,'missing_binary':None})
 for split in SPLITS:
  for seed in [0,1]:
   for mode in ['one_step','free']:
    for cls in ['correct_confident','correct_mixed','wrong']:
     rr=[r for r in rows if r['split']==split and r['seed']==seed and r['model']=='RSSM'and r['mode']==mode and (mode=='one_step'or r['horizon']==4)and r['initial_class']==cls]
     if rr:starts.append({'split':split,'seed':seed,'mode':mode,'initial_class':cls,**metric(rr,[r['all_correct']for r in rr])})
   for model in ['RSSM','GT_relations','RSSM_state','old_head_state','coverage_head_state','GT_physics']:
    for status in STATUS:
     rr=[r for r in rows if r['split']==split and r['seed']==seed and r['model']==model and r['mode']=='one_step'and r['actual_status']==status]
     if rr:outcomes.append({'split':split,'seed':seed,'model':model,'actual_status':status,'n':len(rr),'binary_correct':sum((r['success_probability']>=.5)==(status=='success')for r in rr),'predicted_success':sum(r['success_probability']>=.5 for r in rr),'three_class_correct':sum(r['predicted_status']==status for r in rr)if model!='RSSM'else None,'mean_success_probability':float(np.mean([r['success_probability']for r in rr]))})
 csvwrite(S/'results/summary.csv',summary);csvwrite(S/'results/native_initial_strata.csv',starts);csvwrite(S/'results/outcomes.csv',outcomes)
 mixed=[r for r in rows if r['split']!='train'and r['model']=='RSSM'and r['mode']=='one_step'and r['initial_class']=='correct_mixed'];save(S/'results/mixed_boundary_examples.json',mixed)
 # Same root paired comparison within each dynamics seed; current heads are separate interfaces.
 pairs=[]
 for seed in [0,1]:
  for baseline,alternative in [('RSSM_state','old_head_state'),('RSSM_state','coverage_head_state'),('old_head_state','coverage_head_state')]:
   for mode,h in [('one_step',1),('free',4)]:
    aa={(r['episode_id'],r['step']):r for r in rows if r['split']!='train'and(r['seed'],r['model'],r['mode'],r['horizon'])==(seed,baseline,mode,h)};bb={(r['episode_id'],r['step']):r for r in rows if r['split']!='train'and(r['seed'],r['model'],r['mode'],r['horizon'])==(seed,alternative,mode,h)};assert aa.keys()==bb.keys();by=collections.defaultdict(list)
    for k,r in aa.items():by[r['root_id']].append(int(bb[k]['all_correct'])-int(r['all_correct']))
    vals=np.array([np.mean(v)for v in by.values()]);rng=np.random.default_rng(2610192100);ci=np.quantile(vals[rng.integers(len(vals),size=(2000,len(vals)))].mean(1),[.025,.975]);pairs.append({'seed':seed,'baseline':baseline,'alternative':alternative,'mode':mode,'n':len(aa),'roots':len(by),'delta_pp':float(vals.mean()*100),'ci_low_pp':float(ci[0]*100),'ci_high_pp':float(ci[1]*100)})
 csvwrite(S/'results/paired_interface_differences.csv',pairs);save(S/'results/summary_complete.json',{'utc':now(),'summary_rows':len(summary),'mixed_examples':len(mixed),'new_forwards':0});print('SUMMARIZED',len(summary),flush=True)
if __name__=='__main__':main()
