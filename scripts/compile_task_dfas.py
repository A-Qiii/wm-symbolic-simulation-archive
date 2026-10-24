from pathlib import Path
import json,argparse
from wm_symbolic.ltlf_adapter import compile_ltlf
from wm_symbolic.task_specs import tasks_for_root,parse_task_text
P=Path('/home/robot/research/projects/wm_symbolic');a=argparse.ArgumentParser();a.add_argument('--split',choices=['dev','calibration','test'],required=True);a.add_argument('--roots',type=int,required=True);args=a.parse_args()
out=P/'configs/tasks'/args.split;out.mkdir(parents=True,exist_ok=True);public=[];references=[]
for i in range(args.roots):
 tasks,refs=tasks_for_root('D' if args.split=='test' else 'C',args.split,i)
 for task in tasks:
  parsed=parse_task_text(task['text']);assert parsed['terminal_goals']==task['terminal_goals'] and parsed['ltlf']==task['ltlf'];task['parser_roundtrip_verified']=True
  d=compile_ltlf(task['ltlf'],P/'.tools/mona/usr/bin/mona');task['dfa']={'propositions':d.propositions,'initial':d.initial,'accepting':sorted(d.accepting),'transitions':d.transitions};(out/(task['task_id']+'.dot')).write_text(d.dot)
 public.extend(tasks);references.extend(refs)
(out/'task_specs.json').write_text(json.dumps(public,ensure_ascii=False,indent=2));(out/'references_evaluator_only.json').write_text(json.dumps(references,indent=2));print('COMPILED',len(public),'public tasks; references stored separately')
