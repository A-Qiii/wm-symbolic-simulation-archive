from pathlib import Path
import json,datetime
from wm_symbolic.ltlf_adapter import compile_ltlf
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/proposition_interface_20261014';rows=[];refs=[]
def add(root,name,text,goals,reference,untouched=None,ordered=None):
 form='F('+ ' & '.join(n if v else '!'+n for n,v in goals.items())+') & G(slider_right)'
 if untouched:form+=f' & G(!{untouched}_moved)'
 if ordered:
  first,second=ordered;form+=f' & (!{second}_held U ({first}_drawer & X(!{second}_held U (drawer_closed & X F({second}_held & drawer_open)))))'
 d=compile_ltlf(form,P/'.tools/mona/usr/bin/mona');tid=root+'_'+name
 rows.append({'task_id':tid,'root_id':root,'cohort':'fresh_D' if '_interface_' in root else 'ABC_development','text':text+'；保持滑门在右侧。','terminal_goals':goals,'ltlf':form,'max_executed_skills':8,'dfa':{'propositions':d.propositions,'initial':d.initial,'accepting':sorted(d.accepting),'transitions':d.transitions}});refs.append({'task_id':tid,'reference':reference})
def store(root,color,closed=True,initial_open=False,held=False,untouched=None):
 add(root,color+('_closed' if closed else '_open'),f'把{color}块放入抽屉，最终'+('关闭' if closed else '保持打开')+('，不移动'+untouched+'块' if untouched else ''),{color+'_drawer':True,'drawer_closed' if closed else 'drawer_open':True},([] if initial_open else ['open_drawer'])+([] if held else [f'lift_{color}_block_table'])+['place_in_drawer']+(['close_drawer'] if closed else []),untouched=untouched)
def hold_closed(root,color):add(root,color+'_held_closed',f'握住{color}块并关闭抽屉',{color+'_held':True,'drawer_closed':True},['close_drawer',f'lift_{color}_block_table'])
def order(root):add(root,'red_close_then_blue','先放入红块并关闭一次，再打开抽屉握住蓝块；此前不得握住蓝块，不移动粉块',{'red_drawer':True,'blue_held':True,'drawer_open':True},['open_drawer','lift_red_block_table','place_in_drawer','close_drawer','open_drawer','lift_blue_block_table'],untouched='pink',ordered=('red','blue'))
def two(root,second):add(root,'pink_then_'+second,'放入已持有的粉块，随后握住'+second+'块，保持抽屉打开',{'pink_drawer':True,second+'_held':True,'drawer_open':True},['place_in_drawer',f'lift_{second}_block_table'],untouched='red' if second=='blue' else 'blue')
store('A_core1200_00','red',untouched='pink');hold_closed('B_core1200_01','pink');store('C_core1200_00','blue',closed=False);store('C_core1200_01','red',initial_open=True);store('C_core1200_02','pink',initial_open=True,held=True);order('C_core1200_03')
store('D_interface_00','red',untouched='pink');store('D_interface_00','blue',closed=False);hold_closed('D_interface_01','pink');store('D_interface_02','pink',initial_open=True,held=True);two('D_interface_02','blue');order('D_interface_03');store('D_interface_04','pink',closed=False,initial_open=True);two('D_interface_05','red')
for name,value in [('tasks.json',rows),('references_evaluator_only.json',refs)]:
 f=R/name;assert not f.exists();f.write_text(json.dumps(value,ensure_ascii=False,indent=2));f.chmod(0o444)
print(json.dumps({'tasks':len(rows),'ABC':6,'newD':8,'newD_observed':False,'saved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
