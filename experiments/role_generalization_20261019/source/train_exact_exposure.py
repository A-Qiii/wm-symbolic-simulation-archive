"""Implement the already frozen equal role-frequency rule for the 8000-draw remainder."""
from common import *
import ast
base=S/'source/train.py';source=base.read_text()
old="while len(cycles)<8000:cycles.extend(rng.permutation(72).tolist())"
new="""while len(cycles)<7992:cycles.extend(rng.permutation(72).tolist())
  # 111 complete passes give7992. Match the remaining8 actual role identities across arms,
  # one draw in each of8 different families and4 draws per branch. No outcome is inspected.
  shared={}
  for ix,(a,b)in enumerate(zip(new['random'].episodes,new['balanced'].episodes)):
   if a['root_id']==b['root_id']:shared.setdefault(a['family_id'],[]).append(ix)
  eligible=sorted(shared);assert len(eligible)>=8;chosen=rng.choice(eligible,size=8,replace=False);branches=rng.permutation([1]*4+[2]*4)
  for fid,branch in zip(chosen,branches):
   options=[ix for ix in shared[fid]if new['random'].episodes[ix]['branch']==branch];cycles.append(int(rng.choice(options)))
  for arm in ARMS:
   counts=collections.Counter((new[arm].episodes[i]['root']['held_color'],new[arm].episodes[i]['root']['matched_color'],new[arm].episodes[i]['branch'])for i in cycles)
   if arm=='random':reference=counts
   else:assert counts==reference
  family_counts=collections.Counter(new['random'].episodes[i]['family_id']for i in cycles);assert max(family_counts.values())-min(family_counts.values())<=1"""
assert source.count(old)==1;source=source.replace(old,new);ast.parse(source)
save(S/'configs/training_implementation.json',{'utc':now(),'base_source_sha256':sha(base),'compiled_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'reason':'Enforce frozen equal global role+branch exposure and per-family counts within1 despite8000 not divisible by72.111 full shuffled passes plus8 identity-matched draws in8different families; no data outcome or model score used.','updates_changed':False,'loss_or_model_changed':False})
exec(compile(source,str(base)+'<exact exposure remainder>','exec'),{'__name__':'__main__'})
