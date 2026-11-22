"""Finite candidate programs, same bank for both learned models."""
import itertools
NAMES=['large','small','small2']
def bank(max_actions=6):
 pairs=[[['Pick',n,None],['Place',n,t]]for n,t in [('large','A'),('large','B'),('small','A'),('small','B'),('small','large_top'),('small2','A'),('small2','B'),('small2','large_top')]]
 return [sum((pairs[i]for i in indices),[])for length in range(1,max_actions//2+1)for indices in itertools.product(range(8),repeat=length)]
def goal_matches(s,g):return all(s['region'][int(i)]==v for i,v in g['regions'].items())and not any(s['held'])
def protected(s,g):return all(s['region'][int(i)]==v for i,v in g['always'].items())
def legal(s,a):
 i=NAMES.index(a[1])
 if a[0]=='Pick':return not any(s['held'])
 if s['held']!=[j==i for j in range(3)]:return False
 # Preserve the prior experiment's cross-region carrier-transfer interface;
 # same-region carrier no-ops were not included in the fitted skill data.
 if i==0:return s['region'][0]in [0,1]and s['region'][0]!={'A':0,'B':1}[a[2]]
 return True
def choose(rt,state,prefix,goal,max_actions=6):
 cache={():state};syms={():rt.symbols(state)};options=[]
 for i,program in enumerate(bank(max_actions)):
  if program[:len(prefix)]!=prefix or len(program)<=len(prefix):continue
  key=();pred=[];ok=protected(syms[key],goal);reason=None
  for a in program[len(prefix):]:
   if not ok:reason='protected_region';break
   if not legal(syms[key],a):ok=False;reason='skill_precondition';break
   nk=key+(tuple(a),)
   if nk not in cache:cache[nk]=rt.predict(cache[key],a);syms[nk]=rt.symbols(cache[nk])
   key=nk;pred.append(dict(action=a,**syms[key]));ok=ok and protected(syms[key],goal)
  options.append(dict(index=i,program=program,predictions=pred,passes=ok and goal_matches(syms[key],goal),legal=ok,reason=reason,remaining=len(program)-len(prefix)))
 good=[o for o in options if o['passes']];return (min(good,key=lambda o:(o['remaining'],o['index']))if good else None),options
