"""Expose the difference between ordered occurrences and first-completion order."""
from pathlib import Path
import json
from wm_symbolic.ltlf_adapter import compile_ltlf
P=Path('/home/robot/research/projects/wm_symbolic')
formulas={'guide_example':'F(a & X(F(b)))','strict_first_completion':'(!b U (a & !b & X(F(b))))'}
traces={'A_then_B_persist':[(False,False),(True,False),(True,True),(True,True)],
        'B_then_A_persist':[(False,False),(False,True),(True,True),(True,True)],
        'simultaneous_then_persist':[(False,False),(True,True),(True,True)]}
results={}
for name,formula in formulas.items():
 d=compile_ltlf(formula,P/'.tools/mona/usr/bin/mona')
 results[name]={key:d.accepts([dict(a=a,b=b) for a,b in values]) for key,values in traces.items()}
assert results['guide_example']=={'A_then_B_persist':True,'B_then_A_persist':True,'simultaneous_then_persist':True}
assert results['strict_first_completion']=={'A_then_B_persist':True,'B_then_A_persist':False,'simultaneous_then_persist':False}
record={'formulas':formulas,'traces':traces,'compiler_results':results,'project_task_formulas_changed':False,
        'issue':'Guide strong-next example enforces ordered state occurrences, but not first achievement order when completed goals persist.',
        'scope':'Formal-language regression fixture only; no simulated or fabricated experiment trajectory','requires_semantics_decision_before_changing_acceptance':True}
(P/'runs/order_semantics_audit.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
