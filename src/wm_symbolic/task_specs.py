"""Public goal/constraint templates and separate evaluator reference proposals."""
import re
COLORS=['red','blue','pink'];ZH={'red':'红','blue':'蓝','pink':'粉'};EN={v:k for k,v in ZH.items()}
def tasks_for_root(environment,split,index):
 root_id=f'{environment}_{split}_root{index:03d}';c=COLORS[index%3];d=COLORS[(index+1)%3];e=COLORS[(index+2)%3];family=index%4;tasks=[];references=[]
 for variant in range(2):
  target=c if not variant else d;forbidden=e
  if family==0:
   goals={target+'_drawer':True,'drawer_closed':True};formula=f'G(!{forbidden}_moved)'
   text=f'将{ZH[target]}块收进抽屉，结束时关上抽屉，全程不移动{ZH[forbidden]}块。'
   reference=['open_drawer',f'lift_{target}_block_table','place_in_drawer','close_drawer']
  elif family==1:
   first,second=(d,e) if not variant else (e,d);forbidden=c
   goals={first+'_drawer':True,second+'_slider_right':True,'drawer_closed':True};formula=f'G(!{forbidden}_moved)'
   text=f'将{ZH[first]}块放入抽屉、{ZH[second]}块放到滑门右侧区域，最后关闭抽屉，全程不移动{ZH[forbidden]}块。'
   reference=['open_drawer',f'lift_{first}_block_table','place_in_drawer','move_slider_left',f'lift_{second}_block_table','place_in_slider','close_drawer']
  elif family==2:
   first,second=d,e;goals={first+'_drawer':True,second+'_slider_right':True,'drawer_closed':True}
   earlier,later=(first+'_drawer',second+'_slider_right') if not variant else (second+'_slider_right',first+'_drawer')
   formula=f'F({earlier} & X(F({later}))) & G(!{c}_moved)'
   text=f'将{ZH[first]}块放入抽屉、{ZH[second]}块放入滑门右侧区域，最后关抽屉；严格先完成{earlier}再完成{later}，不移动{ZH[c]}块。'
   reference=['open_drawer',f'lift_{first}_block_table','place_in_drawer','move_slider_left',f'lift_{second}_block_table','place_in_slider','close_drawer'] if not variant else ['move_slider_left',f'lift_{second}_block_table','place_in_slider','open_drawer',f'lift_{first}_block_table','place_in_drawer','close_drawer']
  else:
   goals={target+'_drawer':True,'drawer_closed':True,'light_off':True};formula=f'G(led_on & !led_was_off) & F(({target}_drawer & drawer_closed) & X(F(light_off)))'
   text=f'收纳{ZH[target]}块并关闭抽屉，随后关闭灯泡，全程保持 LED 开启。'
   reference=[f'lift_{target}_block_table','place_in_drawer','close_drawer','turn_off_lightbulb']
  heldout=split=='test' and variant==1
  if heldout:
   extra=f'G(!{forbidden}_moved)' if family==3 else ('G(led_on & !led_was_off)' if index%2 else 'G(!led_on & !led_was_on)')
   formula=f'({formula}) & {extra}';text+=f' 附加全程约束：{extra}。'
  task_id=f'{root_id}_task{variant}';tasks.append({'task_id':task_id,'root_id':root_id,'family':family,'variant':variant,'text':text,'terminal_goals':goals,'ltlf':formula,'max_executed_skills':8,'minimum_reference_skills':len(reference),
   'heldout_goal_constraint_combination':heldout,
   'task_text_enters_dynamics':False,'reference_visible_to_planner':False})
  references.append({'task_id':task_id,'actions':reference,'verified_executable':False,'access':'evaluator_only'})
 return tasks,references

def parse_task_text(text):
 """Explicit bounded Chinese templates. Produces goals/LTL only, never actions."""
 extra=None
 if ' 附加全程约束：' in text:text,extra=text.split(' 附加全程约束：');extra=extra.removesuffix('。')
 if m:=re.fullmatch(r'将([红蓝粉])块收进抽屉，结束时关上抽屉，全程不移动([红蓝粉])块。',text):
  target,forbidden=map(EN.get,m.groups());goals={target+'_drawer':True,'drawer_closed':True};formula=f'G(!{forbidden}_moved)'
 elif m:=re.fullmatch(r'将([红蓝粉])块放入抽屉、([红蓝粉])块放到滑门右侧区域，最后关闭抽屉，全程不移动([红蓝粉])块。',text):
  first,second,forbidden=map(EN.get,m.groups());goals={first+'_drawer':True,second+'_slider_right':True,'drawer_closed':True};formula=f'G(!{forbidden}_moved)'
 elif m:=re.fullmatch(r'将([红蓝粉])块放入抽屉、([红蓝粉])块放入滑门右侧区域，最后关抽屉；严格先完成([a-z_]+)再完成([a-z_]+)，不移动([红蓝粉])块。',text):
  f,s,earlier,later,c=m.groups();first,second,forbidden=EN[f],EN[s],EN[c];goals={first+'_drawer':True,second+'_slider_right':True,'drawer_closed':True}
  if {earlier,later}!={first+'_drawer',second+'_slider_right'}:raise ValueError('Sequence targets inconsistent with terminal goals')
  formula=f'F({earlier} & X(F({later}))) & G(!{forbidden}_moved)'
 elif m:=re.fullmatch(r'收纳([红蓝粉])块并关闭抽屉，随后关闭灯泡，全程保持 LED 开启。',text):
  target=EN[m.group(1)];goals={target+'_drawer':True,'drawer_closed':True,'light_off':True};formula=f'G(led_on & !led_was_off) & F(({target}_drawer & drawer_closed) & X(F(light_off)))'
 else:raise ValueError('Task outside the four declared language templates')
 if extra:
  allowed={'G(led_on & !led_was_off)','G(!led_on & !led_was_on)'}|{f'G(!{c}_moved)' for c in COLORS}
  if extra not in allowed:raise ValueError('Unsupported extra constraint')
  formula=f'({formula}) & {extra}'
 return {'terminal_goals':goals,'ltlf':formula}
