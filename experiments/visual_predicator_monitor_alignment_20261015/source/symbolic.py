"""Learned lifted operator execution and finite BFS; manual rules only as named baseline."""
import itertools,collections
import numpy as np
FEATURES=['Holding(x)','In(x,drawer)','Open(drawer)','EmptyHand()','OtherHeld(x)']
def binding(row,before):
 if row['family']=='lift':return ['red','blue','pink'].index(row['declared_arguments']['color'])
 if row['family']=='place':
  held=np.flatnonzero(np.asarray(before)[:3]==1);return int(held[0])if len(held)==1 else None
 return None
def features(state,x):
 a=np.asarray(state);h=a[:3];known=bool((h>=0).all());empty=bool((h==0).all())if known else None
 other=None if not known else bool(any(h[j]==1 for j in range(3)if j!=x))
 return {'Holding(x)':False if x is None else(bool(a[x])if a[x]>=0 else None),'In(x,drawer)':False if x is None else(bool(a[3+x])if a[3+x]>=0 else None),'Open(drawer)':bool(a[6])if a[6]>=0 else None,'EmptyHand()':empty,'OtherHeld(x)':other}
def applicable(operator,state,x):
 if not operator.get('supported',True):return False
 f=features(state,x);return all(f[name]is not None and f[name]==value for name,value in operator['preconditions'])
def col(atom,x):
 if atom=='Open(drawer)':return 6
 if x is None:raise ValueError('Unbound effect argument')
 return x if atom=='Holding(x)'else 3+x if atom=='In(x,drawer)'else (_ for _ in []).throw(ValueError(atom))
def apply(operator,state,x):
 if not applicable(operator,state,x):return None
 n=list(state)
 for atom in operator['delete']:n[col(atom,x)]=0
 for atom in operator['add']:n[col(atom,x)]=1
 return tuple(n)
def grounded(operators):
 for op in operators:
  for x in range(3)if op['family']in ['lift','place']else[None]:yield op,x
def goal_satisfied(state,goal):return all(state[int(j)]==int(value)for j,value in goal)
def search(state,goal,operators,depth=8,expansions=512):
 state=tuple(int(z)for z in state)
 if -1 in state:return {'status':'unknown_state','plan':None,'expanded':0}
 queue=collections.deque([(state,[])]);seen={state};count=0
 while queue and count<expansions:
  s,path=queue.popleft();count+=1
  if goal_satisfied(s,goal):return {'status':'solved','plan':path,'expanded':count}
  if len(path)>=depth:continue
  for op,x in grounded(operators):
   nxt=apply(op,s,x)
   if nxt is not None and nxt not in seen:
    seen.add(nxt);queue.append((nxt,path+[{'operator_id':op['id'],'family':op['family'],'object':x,'predicted_state':list(nxt)}]))
 return {'status':'no_plan','plan':None,'expanded':count}
def fixed_baseline():
 return [dict(id=f,family=f,preconditions=p,add=a,delete=d,supported=True)for f,p,a,d in [
 ('open',[['Open(drawer)',False],['EmptyHand()',True]],['Open(drawer)'],[]),
 ('lift',[['EmptyHand()',True],['In(x,drawer)',False]],['Holding(x)'],[]),
 ('place',[['Holding(x)',True],['Open(drawer)',True]],['In(x,drawer)'],['Holding(x)']),
 ('close',[['EmptyHand()',True],['Open(drawer)',True]],[],['Open(drawer)'])]]
def signature(before,after,x):
 add=[];delete=[];unsupported=[]
 for j in range(7):
  if before[j]==after[j]:continue
  if j==6:name='Open(drawer)'
  elif x is not None and j%3==x:name='Holding(x)'if j<3 else'In(x,drawer)'
  else:unsupported.append(j);continue
  (add if after[j]else delete).append(name)
 return tuple(sorted(add)),tuple(sorted(delete)),tuple(unsupported)
def learn(rows,before,after):
 operators=[];audit=[]
 for family in ['open','lift','place','close']:
  ids=[i for i,r in enumerate(rows)if r['family']==family];successful=[i for i in ids if rows[i]['primitive_status']=='success'];clusters=collections.defaultdict(list)
  for i in successful:clusters[signature(before[i],after[i],binding(rows[i],before[i]))].append(i)
  for ci,(sig,positive)in enumerate(sorted(clusters.items())):
   neg=[i for i in ids if i not in positive];names=FEATURES if family in ['lift','place']else['Open(drawer)','EmptyHand()'];f={i:features(before[i],binding(rows[i],before[i]))for i in ids};common=[]
   for name in names:
    vals=[f[i][name]for i in positive]
    if vals and vals[0]is not None and all(v==vals[0]for v in vals):common.append([name,vals[0]])
   choices=[]
   for size in range(len(common)+1):
    for combo in itertools.combinations(common,size):
     answers={i:all(f[i][n]is not None and f[i][n]==v for n,v in combo)for i in ids};tp=sum(answers[i]for i in positive);tn=sum(not answers[i]for i in neg);acc=(tp+tn)/len(ids);choices.append({'preconditions':list(combo),'accuracy':acc,'objective':acc-.001*size,'exact':tp==len(positive)and tn==len(neg)})
    if any(r['exact']for r in choices):break
   exact=[r for r in choices if r['exact']];chosen=max(exact if exact else choices,key=lambda r:r['objective']);supported=bool(exact)and not sig[2]
   op={'id':family+'_'+str(ci),'family':family,'preconditions':chosen['preconditions'],'add':list(sig[0]),'delete':list(sig[1]),'supported':supported};operators.append(op)
   audit.append(op|{'positive_transitions':len(positive),'negative_transitions':len(neg),'negative_real_failures':sum(rows[i]['primitive_status']!='success'for i in neg),'negative_other_effect_clusters':sum(rows[i]['primitive_status']=='success'for i in neg),'positive_intersection':common,'objective':chosen['objective'],'train_accuracy':chosen['accuracy'],'no_exact_separator':not bool(exact),'unbound_effect_columns':list(sig[2]),'positive_pair_ids':[rows[i]['pair_id']for i in positive],'negative_pair_ids':[rows[i]['pair_id']for i in neg],'no_negative_coverage':not bool(neg)})
 return operators,audit
