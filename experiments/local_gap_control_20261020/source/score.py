from common import *
import ast
node=next(n for n in ast.parse((O/'source/analyze.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='analyze');exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged common relation metrics','exec'))
def main():
 guard();verify();roots=[];ret=[];first=[];scores=[];summary=[];case_rows=[]
 physical=read(P/'reports/rgbd_close_diagnostic_20261020/physical_records.json');phys={(r['split'],r['episode_id'],r['step']):r for r in physical}
 for f in sorted((S/'results').glob('predictions_*.npz')):
  z=dict(np.load(f));rr,ff,_,err=analyze(f);roots+=rr;ret+=ff;first+=err;arm=str(z['arm'][0]);seed=int(z['seed'][0]);split=str(z['split'][0]);ok=correct(rel(z['prob']),truth(z['cat']))
  for kind in sorted(set(z['kind'])):
   rs=[r for r in rr if r['kind']==kind];stable=[r for r in ff if r['kind']==kind and r['stable_trueIn']];summary.append({'arm':arm,'seed':seed,'split':split,'kind':str(kind),'roots':len(rs),'families':len({r['family_id']for r in rs}),**{k:float(np.mean([r[k]for r in rs]))for k in ['h0_held','h0_relation','h0_full','normal_whole','h2_pair','terminal']},'primary_joint3':float(np.mean([r.get('primary_joint3',0)for r in rs])),'primary_joint4':float(np.mean([r.get('primary_joint4',0)for r in rs])),'stable_In_opportunities':len(stable),'In_forget_count':sum(r['forgot']for r in stable),'In_forget_rate':sum(r['forgot']for r in stable)/max(1,len(stable))})
   for mode in ['one_step','suffix']:
    ix=np.flatnonzero((z['kind']==kind)&(z['skill']=='close_drawer')&(z['mode']==mode));cases=[]
    for j in ix:
     key=(split,str(z['episode_id'][j]),int(z['step'][j]));group=phys[key]['group']if key in phys else 'unclassified_fresh';pr=z['prob'][j,24:27];y=int(z['cat'][j,3]);cases.append({'arm':arm,'seed':seed,'split':split,'kind':str(kind),'mode':mode,'episode_id':str(z['episode_id'][j]),'family_id':str(z['family_id'][j]),'group':group,'true_class':y,'pred_class':int(pr.argmax()),'correct':bool(ok[j,2]),'all_relation_correct':bool(ok[j,3]),'success_Brier':float((z['event_prob'][j,0]-z['event_target'][j,0])**2)})
    case_rows+=cases
    for group in ['ALL']+sorted({r['group']for r in cases}):
     xs=cases if group=='ALL'else[r for r in cases if r['group']==group]
     if not xs:continue
     conf=np.zeros((3,3),int)
     for r in xs:conf[r['true_class'],r['pred_class']]+=1
     recall=[conf[c,c]/max(conf[c].sum(),1)for c in range(3)];scores.append({'arm':arm,'seed':seed,'split':split,'kind':str(kind),'mode':mode,'group':group,'n':len(xs),'families':len({r['family_id']for r in xs}),'correct':sum(r['correct']for r in xs),'accuracy':sum(r['correct']for r in xs)/len(xs),'macro_recall':float(np.mean([recall[c]for c in range(3)if conf[c].sum()])),'success_recall':recall[0],'jam_recall':recall[1],'limit_recall':recall[2],'all_relation_correct':sum(r['all_relation_correct']for r in xs),'confusion':conf.tolist(),'prediction_class_counts':dict(collections.Counter(r['pred_class']for r in xs)),'target_class_counts':dict(collections.Counter(r['true_class']for r in xs)),'majority_baseline_accuracy':sum(r['true_class']==1 for r in xs)/len(xs),'success_Brier':float(np.mean([r['success_Brier']for r in xs]))})
 csvwrite(S/'results/root_metrics.csv',roots);csvwrite(S/'results/retention_opportunities.csv',ret);csvwrite(S/'results/close_case_metrics.csv',case_rows);save(S/'results/first_errors.json',first);save(S/'results/summary.json',summary);save(S/'results/group_scores.json',scores)
 checks=[];deltas=[]
 for seed in [0,1]:
  for mode in ['one_step','suffix']:
   x={a:next(r for r in scores if r['split']=='development'and r['kind']=='role'and r['seed']==seed and r['arm']==a and r['mode']==mode and r['group']=='ALL')for a in ARMS}
   for k in ['macro_recall','success_recall']:
    diff=x['local'][k]-x['coarse'][k];checks.append({'seed':seed,'mode':mode,'metric':k,'difference':diff,'rule':'>0','passed':diff>0});deltas.append({'seed':seed,'mode':mode,'metric':k,'coarse':x['coarse'][k],'local':x['local'][k],'difference':diff})
   checks.append({'seed':seed,'mode':mode,'metric':'above_train_majority_baseline','local':x['local']['accuracy'],'rule':'>='+str(x['local']['majority_baseline_accuracy']),'passed':x['local']['accuracy']>=x['local']['majority_baseline_accuracy']})
  ss={a:next(r for r in summary if r['split']=='development'and r['kind']=='role'and r['seed']==seed and r['arm']==a)for a in ARMS}
  for k in ['h0_held','normal_whole','In_forget_rate']:
   diff=ss['local'][k]-ss['coarse'][k];passed=diff<=.02 if k=='In_forget_rate'else diff>=-.05;checks.append({'seed':seed,'metric':k,'difference':diff,'passed':passed})
 result={'utc':now(),'supports_independent_test':all(r['passed']for r in checks),'checks':checks,'close_deltas':deltas,'development_exploratory_used_to_choose_hypothesis':True,'old_continuation_gate_preserved_false':True,'no_posthoc_tuning':True,'summary':summary,'group_scores':scores};save(S/'results/analysis.json',result);print(json.dumps({'supports_independent_test':result['supports_independent_test'],'deltas':deltas,'checks':checks},indent=2),flush=True)
if __name__=='__main__':main()
