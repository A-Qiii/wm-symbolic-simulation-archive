"""Score saved query outputs and actual physics only; no new neural forward."""
from common import *
from symbolic import *
import csv
def main():
 assert read(E/'results/test_data_validation.json')['passed'];assert not(E/'results/test_evaluation.json').exists();m=read(B/'data_manifest.json');rules=read(E/'configs/rules_frozen.json')['operators'];windows=[];actions=[];rows=[];episodes=m['episodes'];decisions={}
 for ep in episodes:
  folder=B/'episodes'/ep['episode_id'];ww=read(folder/'_private/windows.json');windows+=ww;aa=read(folder/'_private/actions.json')if(folder/'_private/actions.json').exists()else[]
  for a in aa:actions.append(a|{'root_id':ep['root_id'],'episode_id':ep['episode_id'],'arm':ep['arm']})
  decisions[ep['episode_id']]=read(folder/'public/decisions.json')
 y=np.array([r['targets_evaluator_only']['y']for r in windows],bool);v=np.array([r['targets_evaluator_only']['v']for r in windows],bool);pred={name:np.array([r['inference'][key]for r in windows])for name,key in [('query_reader','probabilities'),('old_reader','old_reader_probabilities')]};metricsout={};details=[]
 for name,p in pred.items():
  co=known(p)&((p>=.5)==y)&v;metricsout[name]={}
  for arm in ['visual_learned_rules','GT_symbols_learned_rules']:
   for role in ['all','planning_endpoint','contact_probe','lowering_probe']:
    ids=np.array([i for i,r in enumerate(windows)if r['arm']==arm and(role=='all'or role in r['roles'])],int);mm=metrics(p,y,v,ids);rr=sorted({windows[i]['root_id']for i in ids});mm['independent_roots']=len(rr);mm['roots_all_scored_windows_correct']=sum(bool(co[[i for i in ids if windows[i]['root_id']==root]].all())for root in rr);metricsout[name][arm+':'+role]=mm
  for i,r in enumerate(windows):details.append({'model':name,'window_id':r['window_id'],'episode_id':r['episode_id'],'root_id':r['root_id'],'arm':r['arm'],'roles':r['roles'],'all7':bool(co[i].all()),'error_columns':np.flatnonzero(~co[i]&v[i]).tolist(),'UNKNOWN_columns':np.flatnonzero(~known(p)[i]&v[i]).tolist(),'probabilities':p[i].tolist(),'truth':y[i].astype(int).tolist(),'valid':v[i].tolist(),'public_frames':r['public_frames']})
 wm={w['window_id']:w for w in windows};effects=[]
 for a in actions:
  expected=a['primitive']['status']=='success';truth=np.array(a['after_targets']['y'],int);beforeGT=np.array(a['before_targets']['y'],int);beforevis=wm[a['before_window_id']]['inference']['decoded'];x=a['object'];results={}
  for label,rr,ss in [('learned_from_GT',rules,beforeGT),('learned_from_visual',rules,beforevis),('fixed_from_GT',fixed_baseline(),beforeGT),('fixed_from_visual',fixed_baseline(),beforevis)]:
   matched=[r for r in rr if r['family']==a['family']and applicable(r,ss,x)];nxt=apply(matched[0],ss,x)if matched else None;results[label]={'applicable':bool(matched),'applicability_correct':bool(matched)==expected,'all7_after_correct':bool(nxt is not None and np.array_equal(nxt,truth)),'predicted_after':list(nxt)if nxt is not None else None}
  effects.append({'episode_id':a['episode_id'],'root_id':a['root_id'],'arm':a['arm'],'step':a['step'],'family':a['family'],'object':x,'status':a['primitive']['status'],'truth_before':beforeGT.tolist(),'truth_after':truth.tolist(),'recorded_prediction':a['predicted_after'],'results':results})
 outcome={}
 for arm in ['visual_learned_rules','GT_symbols_learned_rules']:
  eps=[e for e in episodes if e['arm']==arm];ac=[e for e in effects if e['arm']==arm];success=[e for e in ac if e['status']=='success'];outcome[arm]={'assigned_roots':len(eps),'actual_goal_success':sum(e['actual_goal_success']for e in eps),'false_completion':sum(e['false_completion']for e in eps),'controller_calls':len(ac),'call_status_counts':dict(collections.Counter(e['status']for e in ac)),'stops':dict(collections.Counter(e['stop_reason']for e in eps)),'effect_comparison':{label:{'applicability_correct_on_attempts':sum(e['results'][label]['applicability_correct']for e in ac),'attempts':len(ac),'full7_after_on_real_success':sum(e['results'][label]['all7_after_correct']for e in success),'real_successes':len(success)}for label in ['learned_from_GT','learned_from_visual','fixed_from_GT','fixed_from_visual']}}
 first=None
 for ep in episodes:
  if ep['arm']!='visual_learned_rules'or ep['actual_goal_success']:continue
  d=next((x for x in details if x['model']=='query_reader'and x['episode_id']==ep['episode_id']and 'planning_endpoint'in x['roles']and not x['all7']),None)
  if d:
   decision=next((r for r in decisions[ep['episode_id']]if r['window_id']==d['window_id']),None);first={'episode':ep,'first_wrong_scored_planning_endpoint':d,'actual_decision':decision,'single_offline_GT_state_replacement_plan':search(d['truth'],ep['goal'],rules),'interpretation':'Same frozen learned rules; only current symbolic input replaced for an offline search diagnostic. Does not execute a correction or prove the unique perceptual mechanism.'};break
 save(E/'results/test_evaluation.json',{'created_utc':now(),'scope':'6 independent new D roots,2 dependent matched arms perroot; no D tuning.','perception':metricsout,'task_outcomes':outcome,'episodes':episodes,'first_failure_diagnostic':first,'total_distinct_window_records':len(windows),'predictions_before_actions_verified':True,'no_new_inference':True,'limits':'Actions are chosen by planners; rejected/unattempted action applicability lacks exhaustive counterfactual labels. No general OOD object-category or reliability claim.'});save(E/'results/test_window_details.json',details);save(E/'results/test_effect_details.json',effects);save(E/'results/test_windows.json',windows)
 with(E/'results/test_episode_table.csv').open('w',encoding='utf-8-sig',newline='')as f:
  keys=['root_id','arm','calls','actual_goal_success','reported_goal_success','false_completion','stop_reason','goal_text'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:e[k]for k in keys}for e in episodes)
 print(json.dumps({'outcomes':outcome,'perception':{n:{s:(r['all7'],r['windows'])for s,r in q.items()}for n,q in metricsout.items()}},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
