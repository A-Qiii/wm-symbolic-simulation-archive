from common import *
import collections,itertools
ATOMS=['Holding(x)','In(x,drawer)','Closed(drawer)','Open(drawer)','Intermediate(drawer)']
def features(s,x):
 return {'Holding(x)':bool(s[x])if x is not None else False,'In(x,drawer)':bool(s[x+3])if x is not None else False,'Closed(drawer)':bool(s[6]),'Open(drawer)':bool(s[7]),'Intermediate(drawer)':bool(s[8]),'EmptyHand()':not any(s[:3]),'AnyIn()':any(s[3:6]),'TwoOrMoreIn()':sum(s[3:6])>=2}
def column(atom,x):return 6+ATOMS[2:].index(atom)if atom in ATOMS[2:]else x if atom=='Holding(x)'else x+3

def learn(rows):
 ops=[];audit=[]
 for family in ['open','lift','place','close']:
  rr=[r for r in rows if r['family']==family];groups=collections.defaultdict(list)
  for i,r in enumerate(rr):
   if r['native']['status']!='success':continue
   before=r['before_v2_1']['y'];after=r['after_v2_1']['y'];add=[];delete=[];unbound=[];x=r['object']
   for j in range(9):
    if before[j]==after[j]:continue
    if j>=6:atom=ATOMS[j-4]
    elif x is not None and j%3==x:atom=ATOMS[j//3]
    else:unbound.append(j);continue
    (add if after[j]else delete).append(atom)
   groups[(tuple(add),tuple(delete),tuple(unbound))].append(i)
  for idx,(sig,pos)in enumerate(groups.items()):
   neg=[i for i in range(len(rr))if i not in pos];ff={i:features(r['before_v2_1']['y'],r['object'])for i,r in enumerate(rr)};names=list(ff[pos[0]])if pos else[]
   if family in ['open','close']:names=[n for n in names if '(x)'not in n and n!='In(x,drawer)']
   common=[(n,ff[pos[0]][n])for n in names if all(ff[i][n]==ff[pos[0]][n]for i in pos)];best=None;chosen=None
   for size in range(len(common)+1):
    for candidate in itertools.combinations(common,size):
     tp=sum(all(ff[i][n]==v for n,v in candidate)for i in pos);tn=sum(not all(ff[i][n]==v for n,v in candidate)for i in neg);acc=(tp+tn)/len(rr);score=acc-.001*size;item={'preconditions':candidate,'accuracy':acc,'objective':score}
     if best is None or score>best['objective']:best=item
     if tp==len(pos)and tn==len(neg):chosen=item;break
    if chosen:break
   fit=chosen or best;op={'id':family+'_'+str(idx),'family':family,'preconditions':fit['preconditions'],'add':sig[0],'delete':sig[1],'in_sample_exact_fit':bool(chosen)and not sig[2],'negative_coverage':len(neg)>0,'supported':bool(chosen)and not sig[2]and len(neg)>0};ops.append(op);audit.append(op|{'positive_calls':len(pos),'negative_calls':len(neg),'real_failures':sum(rr[i]['native']['status']!='success'for i in neg),'other_effect_clusters':sum(rr[i]['native']['status']=='success'for i in neg),'no_exact_separator':chosen is None,'no_negative_coverage':len(neg)==0,'unbound_effect_columns':sig[2],'objective':fit['objective'],'accuracy':fit['accuracy'],'positive_episode_steps':[(rr[i]['episode_id'],rr[i]['step'])for i in pos],'negative_episode_steps':[(rr[i]['episode_id'],rr[i]['step'])for i in neg]})
 return ops,audit

def applicable(op,s,x):return op['supported']and all(features(s,x)[n]==v for n,v in op['preconditions'])
def successor(op,s,x):
 if not applicable(op,s,x):return None
 s=list(s)
 for atom in op['delete']:s[column(atom,x)]=0
 for atom in op['add']:s[column(atom,x)]=1
 return tuple(s)
def search(s,goal,ops,max_depth=8):
 s=tuple(map(int,s))
 if -1 in s:return {'status':'unknown_state','plan':None}
 q=collections.deque([(s,[])]);seen={s}
 while q:
  cur,path=q.popleft()
  if all(cur[j]==value for j,value in goal):return {'status':'solved','plan':path}
  if len(path)>=max_depth:continue
  for op in ops:
   for x in range(3)if op['family']in['lift','place']else[None]:
    n=successor(op,cur,x)
    if n is not None and n not in seen:seen.add(n);q.append((n,path+[{'operator_id':op['id'],'object':x,'predicted9':n}]))
 return {'status':'no_plan_with_supported_learned_rules','plan':None,'states':len(seen)}

def main():
 calls=read(BASE/'data/calls.json');eligible=[r for r in calls if all(r['before_v2_1']['v'])and all(r['after_v2_1']['v'])];excluded=[{'episode_id':r['episode_id'],'step':r['step'],'before_invalid':[j for j,v in enumerate(r['before_v2_1']['v'])if not v],'after_invalid':[j for j,v in enumerate(r['after_v2_1']['v'])if not v]}for r in calls if r not in eligible];ops,audit=learn([r for r in eligible if r['split']=='train']);scores=[]
 for split in ['train','dev']:
  for r in eligible:
   if r['split']!=split:continue
   s=r['before_v2_1']['y'];x=r['object'];matches=[o for o in ops if o['family']==r['family']and applicable(o,s,x)];pred=successor(matches[0],s,x)if matches else None;scores.append({'split':split,'episode_id':r['episode_id'],'step':r['step'],'family':r['family'],'native_status':r['native']['status'],'predicted_applicable':bool(matches),'applicability_vs_native_correct':bool(matches)==(r['native']['status']=='success'),'after9_correct':pred==tuple(r['after_v2_1']['y'])if pred is not None else False})
 save(E/'configs/learned_rules9.json',{'operators':ops,'AB_only':True,'audit':audit,'candidate_preconditions':'Existing9 facts plus EmptyHand,AnyIn,TwoOrMoreIn (fixed derived predicates); no colour identity and no hidden geometry in learner; unsupported clusters not available for search.','GT_supervision_only_not_visual_inference':True});save(E/'results/rule_learning9.json',{'raw_calls':len(calls),'eligible_complete_before_after9':len(eligible),'excluded_but_retained':excluded,'audit':audit,'scores':scores,'initial_closed_goal_plan':search([0,0,0,0,0,0,1,0,0],[(3,1),(6,1)],ops),'failure_effects_not_forced_into_success':True,'no_negative_coverage_blocks_deployment':True,'raw_unconstrained_fit_preserved_in_results_rule_learning9_initial_json':True});print(json.dumps({'operators':ops,'eligible':len(eligible),'excluded':len(excluded)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
