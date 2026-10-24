from pathlib import Path
from itertools import product
import json
from wm_symbolic.legacy_dfa import reuse_compiled_automaton,provenance
P=Path('/home/robot/research/projects/wm_symbolic');tasks=json.loads((P/'configs/tasks/dev/task_specs.json').read_text());checks=0
for task in tasks:
 d=task['dfa'];a=reuse_compiled_automaton(d)
 for state in a.states:
  for values in product((False,True),repeat=len(d['propositions'])):
   labels=dict(zip(d['propositions'],values));expected=[dst for src,guard,dst in d['transitions'] if src==state and all(bit=='X' or (bit=='1')==v for bit,v in zip(guard,values))][0]
   result=a.advance((state,False,False,False,()),dict(labels,done=False));assert result[0]==expected
   finish=a.finish(result);assert finish[0]==result[0];assert finish[3]==(expected in a.accepting);checks+=1
result={'passed':True,'tasks':len(tasks),'valuation_and_finish_checks':checks,'provenance':provenance(),'scope':'Compiled-guard adapter regression only, not world-model task performance'}
(P/'runs/compiled_adapter_regression.json').write_text(json.dumps(result,indent=2));(P/'source/reused_planning.json').write_text(json.dumps(provenance(),indent=2));print(json.dumps(result))
