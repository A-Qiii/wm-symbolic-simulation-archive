"""Run frozen public observation -> typed query -> existing learned-rule search.
This never executes a skill. A zero-length plan is only a perceived goal claim.
"""
from common import *
import sys,subprocess,argparse,importlib.util

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['pooled','spatial'],default='pooled');ap.add_argument('--window-id',required=True);ap.add_argument('--query',default='In(obj_pink,drawer)');ap.add_argument('--goal-object',choices=COLORS,default='pink');a=ap.parse_args()
 contract=read(E/'configs/logic_interface_frozen.json')
 for path,h in contract['source_sha256'].items():assert sha(path)==h,path
 cp=read(SP/'configs'/a.variant/'reader_frozen.json');assert sha(cp['path'])==cp['sha256']
 cmd=[sys.executable,str(SP/'source/query.py'),'--variant',a.variant,'--window-id',a.window_id,'--query',a.query,'--goal-object',a.goal_object];res=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=60);obs=json.loads(res.stdout)
 spec=importlib.util.spec_from_file_location('preserved_learned_operator_search',BASE/'source/learn_actions.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 rules=read(BASE/'configs/learned_rules9.json');assert rules['AB_only'];state=[obs['decoded_state'][f]for f in FACTS];target=COLORS.index(a.goal_object);result=module.search(state,[(3+target,1),(6,1)],rules['operators'])
 interpreted='perceived_goal_satisfied_unverified'if result['status']=='solved'and result['plan']==[]else result['status']
 print(json.dumps({'variant':a.variant,'window_id':a.window_id,'typed_query':obs['query'],'query_probability':obs['probability'],'all_probabilities':obs['all_probabilities'],'visual_state9':obs['decoded_state'],'goal':f'In(obj_{a.goal_object},drawer) AND Closed(drawer)','goal_fact_columns':[3+target,6],'raw_existing_search_result':result,'interface_status':interpreted,'supported_operators':[x['id']for x in rules['operators']if x['supported']],'unsupported_operators':[x['id']for x in rules['operators']if not x['supported']],'GT_loaded':False,'physical_actions_executed':0,'candidate_promoted':False,'limits':'Finite typed queries and preserved learned-rule state search. Perceived goal is not physical verification. This is not an RSSM/TS-LTLf world-model acceptance.'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
