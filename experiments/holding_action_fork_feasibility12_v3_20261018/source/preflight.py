"""Zero-physics comparator regression on immutable recorded data."""
from common import *
from collect import compare,contact_multiset,encoded
import ast

def main():
 old=P/'experiments/holding_action_fork_feasibility12_v2_20261018/qualification/private/runs'
 count=0
 for k in range(1,73):
  a=read(old/'H1_reference72'/f'c{k:02d}.json');b=read(old/'H2_after_full_lift72'/f'c{k:02d}.json')
  for key in ['python_state','history']:
   c=compare(a[key],b[key]);assert c['equal']and not c['raw_equal_before_contact_multiset'];count+=1
 # Deliberate mutations must remain detectable, including multiplicity and order.
 a=read(old/'H1_reference72/c01.json')['history'];b=copy.deepcopy(a);b[-1]['physical']['contacts'].append(copy.deepcopy(b[-1]['physical']['contacts'][0]));assert not compare(a,b)['equal']
 b=copy.deepcopy(a);b[-1]['physical']['contacts'][0][9]+=1e-12;assert not compare(a,b)['equal']
 b=copy.deepcopy(a);b.reverse();assert not compare(a,b)['equal']
 b=copy.deepcopy(a);b[-1]['physical']['objects']['block_red']['position'].reverse();assert not compare(a,b)['equal']
 for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
 result={'utc':now(),'physics_steps':0,'old_contact_order_false_positives_resolved':count,'duplicate_contact_still_detected':True,'numeric_change_1e_12_still_detected':True,'history_order_still_detected':True,'vector_order_still_detected':True,'no_simulator_state_mutation':True,'syntax_parsed':True}
 save(Path(__file__).parent/'preflight.json',result);print(result)

if __name__=='__main__':main()
