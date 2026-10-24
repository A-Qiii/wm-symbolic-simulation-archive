from pathlib import Path
P=Path('/home/robot/research/projects/wm_symbolic');source=(P/'src/wm_symbolic/core_evaluation.py').read_text()
assert 'from collections import defaultdict' in source
source=source.replace('from collections import defaultdict','from collections import defaultdict\nfrom wm_symbolic.core_field_metrics import FieldMetrics',1)
needle='was=model.training;model.eval();s=defaultdict(float);'
assert source.count(needle)==1
source=source.replace(needle,'field_metrics=FieldMetrics();'+needle,1)
needle='  valid=b[\'cat_valid\'][:,t]&keep[:,None];'
assert source.count(needle)==1
source=source.replace(needle,"  field_metrics.add(name,pred,b,t,keep,previous,data.skills)\n"+needle,1)
needle=' model.train(was);return result'
assert source.count(needle)==1
source=source.replace(needle," result['detailed']=field_metrics.result()\n"+needle,1)
(P/'src/wm_symbolic/core_detailed_evaluation.py').write_text(source)
print('Detailed summaries added to the same evaluation forward calls; original active training evaluator unchanged.')
