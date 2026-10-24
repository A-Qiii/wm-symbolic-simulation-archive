from common import *
import subprocess,sys

def main():
 wins=read(BASE/'data/windows.json');roots={r['root_id']:r for r in read(B/'roots_manifest.json')};cases=[]
 # AB cases are fixed by state family and first stable manifest entry, not by correctness.
 for kind,condition in [('tabletop_closed','empty_closed'),('closed_inside','single_inside'),('empty_mid','empty_mid')]:
  choices=[w for w in wins if w['environment']=='A'and roots[w['root_id']]['color']=='pink'and w['condition']==condition and w['targets_v2_1']['v'][6]and (w['targets_v2_1']['y'][6]if kind!='empty_mid'else w['targets_v2_1']['y'][8])]
  w=choices[0];cases.append({'case':kind,'window_id':w['window_id'],'goal':'pink','truth9':w['targets_v2_1']['y']})
 w=next(w for w in wins if w['window_id']=='C_semantics_v2_fixtures_pink_0__empty_closed__t00127');cases.append({'case':'existing_C_false_completion','window_id':w['window_id'],'goal':'pink','truth9':w['targets_v2_1']['y']})
 outputs=[]
 for a in ['pooled','spatial']:
  for c in cases:
   p=subprocess.run([sys.executable,str(E/'source/query_to_logic.py'),'--variant',a,'--window-id',c['window_id'],'--query','In(obj_pink,drawer)','--goal-object',c['goal']],capture_output=True,text=True,check=True,timeout=60);o=json.loads(p.stdout);assert not o['GT_loaded']and o['physical_actions_executed']==0 and not o['supported_operators'];truth=bool(c['truth9'][5]and c['truth9'][6]);claimed=o['interface_status']=='perceived_goal_satisfied_unverified';outputs.append({'case':c['case'],'variant':a,'window_id':c['window_id'],'GT_goal_for_offline_evaluation_only':truth,'perceived_goal_claim':claimed,'false_goal_claim':claimed and not truth,'output':o})
 assert sum(r['false_goal_claim']for r in outputs)==1
 assert next(r for r in outputs if r['false_goal_claim'])['variant']=='spatial'
 for r in outputs:
  if r['case']in ['tabletop_closed','empty_mid']:assert r['output']['interface_status']=='no_plan_with_supported_learned_rules'
 save(E/'results/interface_regression.json',{'actual_CLI_runs':8,'cases':outputs,'preserved_false_positive_shown_not_hidden':True,'physical_skill_calls':0,'learned_rule_flags_unchanged':True})
 print('QUERY_TO_LOGIC_REGRESSION_PASSED_WITH_KNOWN_FALSE_GOAL_RETAINED',flush=True)
if __name__=='__main__':main()
