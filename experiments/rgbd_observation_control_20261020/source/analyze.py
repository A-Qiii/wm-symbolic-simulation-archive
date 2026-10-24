from common import *
import ast
node=next(n for n in ast.parse((O/'source/analyze.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='analyze');exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged relation metrics','exec'))
def main():
 guard();verify();rootrows=[];ret=[];event=[];first=[];summ=[];families=[];close=[];heldkeep=[]
 for path in sorted((S/'results').glob('predictions_*.npz')):
  z=dict(np.load(path));r,f,e,x=analyze(path);rootrows+=r;ret+=f;event+=e;first+=x;p=rel(z['prob']);y=truth(z['cat']);ok=correct(p,y);seed=int(z['seed'][0]);arm=str(z['arm'][0]);split=str(z['split'][0])
  for j in np.flatnonzero((z['skill']=='close_drawer')&(z['mode']!='current')):
   close.append({'arm':arm,'seed':seed,'split':split,'kind':str(z['kind'][j]),'family_id':str(z['family_id'][j]),'root_id':str(z['root_id'][j]),'branch':int(z['branch'][j]),'mode':str(z['mode'][j]),'drawer_correct':bool(ok[j,2]),'all_relation_correct':bool(ok[j,3]),'Held_correct':bool(ok[j,0]),'In_correct':bool(ok[j,1]),'drawer_target':int(z['cat'][j,3]),'drawer_pred':int(p[j,6:9].argmax()),'success_Brier':float((z['event_prob'][j,0]-z['event_target'][j,0])**2),'status':str(z['status'][j])})
  for eid in sorted(set(z['episode_id'])):
   ii=np.flatnonzero(z['episode_id']==eid);ix={(str(z['mode'][j]),int(z['step'][j])):j for j in ii}
   for t in range(1,7):
    if('current',t)not in ix:continue
    j=ix['current',t];pr=ix['one_step',t];prev=ix['current',t-1]
    for obj in range(3):
     if y[j,obj]and y[prev,obj]:heldkeep.append({'arm':arm,'seed':seed,'split':split,'kind':str(z['kind'][j]),'root_id':str(z['root_id'][j]),'step':t,'object':COLORS[obj],'prior_correct':bool(p[pr,obj]>=.5),'forgot':bool(p[pr,obj]>=.5 and p[j,obj]<.5)})
  for kind in sorted(set(z['kind'])):
   rr=[v for v in r if v['kind']==kind];ff=[v for v in f if v['kind']==kind and v['stable_trueIn']];hh=[v for v in heldkeep if v['arm']==arm and v['seed']==seed and v['split']==split and v['kind']==kind];cc=[v for v in close if v['arm']==arm and v['seed']==seed and v['split']==split and v['kind']==kind]
   keys=['h0_held','h0_relation','h0_full','h2_pair','normal_whole','normal_whole_full','terminal']+(['primary_joint3','primary_joint4','H_In_h2_post_correct_both']if kind=='role'else[])
   ss={'split':split,'kind':str(kind),'seed':seed,'arm':arm,'roots':len(rr),'families':len({v['family_id']for v in rr}),**{k:float(np.mean([v[k]for v in rr]))for k in keys},'stable_In_opportunities':len(ff),'In_forget_count':sum(v['forgot']for v in ff),'In_forget_rate':sum(v['forgot']for v in ff)/max(1,len(ff)),'stable_Held_opportunities':len(hh),'Held_forget_count':sum(v['forgot']for v in hh),'Held_forget_rate':sum(v['forgot']for v in hh)/max(1,len(hh))}
   for mode in ['one_step','suffix']:
    selected=[v for v in cc if v['mode']==mode];ss['close_'+mode+'_n']=len(selected)
    for metric in ['drawer_correct','all_relation_correct','success_Brier']:ss['close_'+mode+'_'+metric]=float(np.mean([v[metric]for v in selected]))if selected else None
   summ.append(ss)
   for fid in sorted({v['family_id']for v in rr}):
    fr=[v for v in rr if v['family_id']==fid];fc=[v for v in cc if v['family_id']==fid];row={'split':split,'kind':str(kind),'seed':seed,'arm':arm,'family_id':fid,**{k:float(np.mean([v[k]for v in fr]))for k in keys}}
    for mode in ['one_step','suffix']:
     for met in ['drawer_correct','all_relation_correct','success_Brier']:
      sel=[v[met]for v in fc if v['mode']==mode];row['close_'+mode+'_'+met]=float(np.mean(sel))if sel else None
    families.append(row)
 csvwrite(S/'results/root_metrics.csv',rootrows);csvwrite(S/'results/summary.csv',summ);csvwrite(S/'results/family_metrics.csv',families);csvwrite(S/'results/retention_opportunities.csv',ret);csvwrite(S/'results/Held_retention.csv',heldkeep);csvwrite(S/'results/close_metrics.csv',close);save(S/'results/first_errors.json',first)
 contrasts=[]
 for split in sorted({v['split']for v in families}):
  for kind in sorted({v['kind']for v in families if v['split']==split}):
   ff=[v for v in families if v['split']==split and v['kind']==kind]
   for met in ['h0_held','h0_relation','normal_whole','terminal','close_one_step_drawer_correct','close_suffix_drawer_correct','close_one_step_all_relation_correct','close_suffix_all_relation_correct','close_one_step_success_Brier']:
    ds=[]
    for seed in [0,1]:
     m={arm:{v['family_id']:v[met]for v in ff if v['seed']==seed and v['arm']==arm}for arm in ARMS};assert m['baseline'].keys()==m['geometry'].keys();ids=sorted(m['baseline']);arr=np.array([m['geometry'][i]-m['baseline'][i]for i in ids]);ds.append(arr)
     rng=np.random.default_rng(2610213311);bs=arr[rng.integers(len(arr),size=(2000,len(arr)))].mean(1);contrasts.append({'split':split,'kind':kind,'metric':met,'seed':str(seed),'difference':float(arr.mean()),'lo':float(np.quantile(bs,.025)),'hi':float(np.quantile(bs,.975)),'families':len(arr)})
    arr=np.mean(ds,0);rng=np.random.default_rng(2610213311);bs=arr[rng.integers(len(arr),size=(2000,len(arr)))].mean(1);contrasts.append({'split':split,'kind':kind,'metric':met,'seed':'mean','difference':float(arr.mean()),'lo':float(np.quantile(bs,.025)),'hi':float(np.quantile(bs,.975)),'families':len(arr)})
 csvwrite(S/'results/contrasts.csv',contrasts);checks=[];deltas={}
 for seed in [0,1]:
  r={v['arm']:v for v in summ if v['split']=='development'and v['kind']=='role'and v['seed']==seed};delta={k:r['geometry'][k]-r['baseline'][k]for k in ['h0_held','normal_whole','In_forget_rate','close_one_step_drawer_correct','close_suffix_drawer_correct']};deltas[seed]=delta
  for key,bound,positive in [('h0_held',-.05,True),('normal_whole',-.05,True),('In_forget_rate',.02,False)]:checks.append({'seed':seed,'criterion':key,'difference':delta[key],'passed':delta[key]>=bound if positive else delta[key]<=bound})
 improved=[k for k in ['h0_held','close_one_step_drawer_correct','close_suffix_drawer_correct']if all(deltas[s][k]>0 for s in [0,1])];support=bool(improved)and all(v['passed']for v in checks)
 result={'utc':now(),'supports_fresh_test_under_frozen_rule':support,'targets_improved_in_both_seeds':improved,'development_deltas':deltas,'nonregression_checks':checks,'summaries':summ,'contrasts':contrasts,'old_gates_unchanged_and_false':True,'historical_and_diagnosed_confirmation_exploratory_only':True,'not_guide_acceptance':True};save(S/'results/analysis.json',result);print(json.dumps({'supports_fresh_test':support,'improved':improved,'dev_deltas':deltas,'summary':[{k:r[k]for k in ['split','kind','seed','arm','roots','h0_held','normal_whole','terminal','close_one_step_drawer_correct','close_suffix_drawer_correct','In_forget_rate']}for r in summ if r['kind']=='role']},indent=2),flush=True)
if __name__=='__main__':main()
