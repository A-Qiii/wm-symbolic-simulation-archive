from common import COLORS
def evaluate_gates(summaries,integrity):
 usable=[];endpoint_hard=[]
 for r in summaries:
  c=r['target_color'];pre=next((x for x in r['captures']if x['anchor']=='t0'),None);post=next((x for x in r['captures']if x['anchor']=='t72'),None)
  for anchor in [pre,post]:
   if anchor and anchor['labels'][c]['supported_hard_negative']:endpoint_hard.append((r['root_id'],r['split'],c,anchor['anchor']))
  if r['status']=='completed72'and pre and post and pre['labels'][c]['eligible']and post['labels'][c]['eligible']:
   usable.append({**{k:r[k]for k in ['root_id','split','scene','target_color','geometry','preparation','assigned_action']},'pre':bool(pre['labels'][c]['Holding']),'post':bool(post['labels'][c]['Holding']),'pre_supported_hard_negative':pre['labels'][c]['supported_hard_negative'],'post_supported_hard_negative':post['labels'][c]['supported_hard_negative']})
 checks=[]
 def add(name,count,minimum,**context):checks.append({'gate':name,**context,'observed':count,'minimum':minimum,'passed':count>=minimum})
 add('integrity',int(integrity),1)
 for split,minimum in [('train',24),('dev',12)]:
  part=[r for r in usable if r['split']==split];add('usable_roots',len(part),minimum,split=split)
  for c in COLORS:
   for y in [False,True]:add('each_color_future_class',sum(r['target_color']==c and r['post']==y for r in part),2 if split=='train'else 1,split=split,color=c,post=y)
  for a,b in [(0,1),(1,0),(0,0),(1,1)]:add('transition_direction'if a!=b else'unchanged',sum(r['pre']==bool(a)and r['post']==bool(b)for r in part),2 if split=='train'else 1,split=split,pre=a,post=b)
  for action in ['hold_closed','lift_closed']:
   for y in [False,True]:add('same_action_class',sum(r['assigned_action']==action and r['post']==y for r in part),3 if split=='train'else 2,split=split,action=action,post=y)
  for c in COLORS:add('supported_hard_negative',len({rid for rid,sp,col,anchor in endpoint_hard if sp==split and col==c}),1,split=split,color=c)
 for a,b in [(0,1),(1,0)]:add('transition_color_diversity',len({r['target_color']for r in usable if r['pre']==bool(a)and r['post']==bool(b)}),2,pre=a,post=b)
 for action in ['hold_closed','lift_closed']:
  for y in [False,True]:add('same_action_color_diversity',len({r['target_color']for r in usable if r['assigned_action']==action and r['post']==y}),2,action=action,post=y)
 return {'passed':all(x['passed']for x in checks),'checks':checks,'usable_root_rows':usable,'supported_hard_negative_endpoint_rows':[{'root_id':a,'split':b,'color':c,'anchor':d}for a,b,c,d in endpoint_hard],'hard_negative_scope':'actual t0 or t72; each root once per color/split, including rejected actions with an actual t0; no failure-last or intermediate anchor counts','usable_scope':'completed safe72 with stable visible target pre and post; no false substitution for unknown','training_authorized':False,'four_baseline_comparison_justified':all(x['passed']for x in checks),'prediction_only_not_individual_counterfactual':True}
