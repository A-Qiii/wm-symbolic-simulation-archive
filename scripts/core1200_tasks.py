"""Freeze goal/initial-state variation before D is observed; no learned edges here."""
from pathlib import Path
import json,hashlib,datetime
from wm_symbolic.ltlf_adapter import compile_ltlf
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/core1200_v1';out=R/'evaluation';out.mkdir(exist_ok=True)
rows=[];references=[]
def add(root,name,text,goals,reference,ordered=None,untouched=None):
 conjunction=' & '.join(n if v else '!'+n for n,v in goals.items());formula=f'F({conjunction})'
 if ordered:
  first,second=ordered
  # Other object must not be held before first storage and subsequent closure.
  formula+=f' & (!{second}_held U ({first}_drawer & X(!{second}_held U (drawer_closed & X F({second}_held & drawer_open)))))'
 if untouched:formula+=f' & G(!{untouched}_moved)'
 d=compile_ltlf(formula,P/'.tools/mona/usr/bin/mona');task_id=f'{root}_{name}'
 task={'task_id':task_id,'root_id':root,'text':text,'terminal_goals':goals,'ltlf':formula,'max_executed_skills':8,'dfa':{'propositions':d.propositions,'initial':d.initial,'accepting':sorted(d.accepting),'transitions':d.transitions}}
 rows.append(task);references.append({'task_id':task_id,'reference':reference,'role':'predeclared task/state-aware execution reference, independent of WM; no feedback or GT used to alter this sequence'})
 (out/(task_id+'.dot')).write_text(d.dot)
add('D_core1200_00','red_closed','把红块放入抽屉并关闭抽屉；不移动粉块。',{'red_drawer':True,'drawer_closed':True},['open_drawer','lift_red_block_table','place_in_drawer','close_drawer'],untouched='pink')
add('D_core1200_00','blue_open','把蓝块放入抽屉，结束时保持抽屉打开。',{'blue_drawer':True,'drawer_open':True},['open_drawer','lift_blue_block_table','place_in_drawer'])
add('D_core1200_01','pink_held_closed','握住粉块，并让抽屉处于关闭状态。',{'pink_held':True,'drawer_closed':True},['close_drawer','lift_pink_block_table'])
add('D_core1200_01','blue_closed','把蓝块放入抽屉并关闭。',{'blue_drawer':True,'drawer_closed':True},['lift_blue_block_table','place_in_drawer','close_drawer'])
add('D_core1200_02','held_pink_closed','把已经握住的粉块放入抽屉并关闭。',{'pink_drawer':True,'drawer_closed':True},['place_in_drawer','close_drawer'])
add('D_core1200_02','pink_then_blue','粉块留在抽屉内，最后握住蓝块并保持抽屉打开；不移动红块。',{'pink_drawer':True,'blue_held':True,'drawer_open':True},['place_in_drawer','lift_blue_block_table'],untouched='red')
add('D_core1200_03','red_close_then_blue','先把红块放进抽屉并在放入后关闭一次，再打开抽屉并握住蓝块；在红块放入且关闭前不得握住蓝块，不移动粉块。',{'red_drawer':True,'blue_held':True,'drawer_open':True},['open_drawer','lift_red_block_table','place_in_drawer','close_drawer','open_drawer','lift_blue_block_table'],ordered=('red','blue'),untouched='pink')
add('D_core1200_03','blue_close_then_pink','先把蓝块放进抽屉并在放入后关闭一次，再打开抽屉并握住粉块；在蓝块放入且关闭前不得握住粉块，不移动红块。',{'blue_drawer':True,'pink_held':True,'drawer_open':True},['open_drawer','lift_blue_block_table','place_in_drawer','close_drawer','open_drawer','lift_pink_block_table'],ordered=('blue','pink'),untouched='red')
add('D_core1200_04','close_empty','关闭抽屉并保持空手，不移动粉块。',{'drawer_closed':True,'hand_empty':True},['close_drawer'],untouched='pink')
add('D_core1200_04','pink_open','把粉块放入抽屉，结束时保持打开。',{'pink_drawer':True,'drawer_open':True},['lift_pink_block_table','place_in_drawer'])
add('D_core1200_06','held_red_open','把已经握住的红块放入抽屉，保持抽屉打开。',{'red_drawer':True,'drawer_open':True},['place_in_drawer'])
add('D_core1200_06','red_close_then_pink','先把持有的红块放入抽屉并关闭一次，再打开抽屉并握住粉块；此前不得握住粉块，不移动蓝块。',{'red_drawer':True,'pink_held':True,'drawer_open':True},['place_in_drawer','close_drawer','open_drawer','lift_pink_block_table'],ordered=('red','pink'),untouched='blue')
# Strict order regression uses valuations, not action-effect rules.
ordered=next(t for t in rows if t['task_id'].endswith('red_close_then_blue'))
from wm_symbolic.ltlf_adapter import CompiledDFA
dd=ordered['dfa'];d=CompiledDFA(tuple(dd['propositions']),dd['initial'],frozenset(dd['accepting']),tuple(dd['transitions']),'','','')
def valuation(**kw):return {n:bool(kw.get(n,False)) for n in d.propositions}
good=[valuation(drawer_closed=True),valuation(drawer_open=True),valuation(red_drawer=True,drawer_open=True),valuation(red_drawer=True,drawer_closed=True),valuation(red_drawer=True,drawer_open=True),valuation(red_drawer=True,drawer_open=True,blue_held=True)]
bad=[valuation(drawer_closed=True),valuation(blue_held=True),*good[2:]]
assert d.accepts(good) and not d.accepts(bad),'Strict first-achievement order not enforced'
for filename,data in [('tasks.json',rows),('references_evaluator_only.json',references)]:
 file=out/filename
 if file.exists():assert json.loads(file.read_text())==json.loads(json.dumps(data))
 else:file.write_text(json.dumps(data,indent=2,ensure_ascii=False));file.chmod(0o444)
protocol={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tasks':12,'roots':6,'reference_lengths':[len(r['reference']) for r in references],'maximum_requested_skills':8,'conditions':['learned_RSSM_LTLf','learned_RSSM_goal_only','predeclared_state_aware_reference'],'same_frozen_controller':True,'no_manual_transition_rules_in_learned_search':True,'strict_first_achievement_order_regression_passed':True,'D_observations_used_to_design_tasks':False,'calibration_note':'8 C roots reserved for temperature diagnostics. Keep original >=30 claims across>=10 roots completion-certification requirement;8 roots cannot establish certification. Any uncalibrated physical diagnostic is explicitly separate from certified task success.','diagnostic_threshold':.8,'diagnostic_event_threshold':.9,'test_evaluation_once':True}
(out/'planning_protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol,indent=2))
