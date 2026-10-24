"""Offline support ownership, never imported by model or learned search."""
from copy import deepcopy

def drawer_ownership(label,info):
 result=deepcopy(label);scene=info['scene_info'];objects=scene['movable_objects'];table=scene['fixed_objects']['table'];tid=table['uid'];drawer=table['links']['drawer_link']
 byid={o['uid']:(name,o) for name,o in objects.items()}
 def supporting(uid,visited):
  if uid in visited:return []
  name,obj=byid[uid];paths=[]
  for c in obj['contacts']:
   if c[9]<=.01:continue
   if c[1]==uid:other,link,normal=c[2],c[4],c[7]
   elif c[2]==uid:other,link,normal=c[1],c[3],[-x for x in c[7]]
   else:continue
   # Only upward load-bearing support, not side brushing or downward contact.
   if normal[2]<=.5:continue
   if other==tid and link==drawer:paths.append([name,'drawer'])
   elif other in byid:
    for rest in supporting(other,visited|{uid}):paths.append([name]+rest)
  unique=[]
  for path in paths:
   if path not in unique:unique.append(path)
  return unique
 for j,color in enumerate(['red','blue','pink']):
  # A held object cannot acquire drawer ownership from a nearby side contact.
  if result['categorical'][j]!=7:continue
  name='block_'+color;paths=supporting(objects[name]['uid'],set())
  if paths:
   result['categorical'][j]=3;result['categorical_valid'][j]=True
   result['continuous']['objects'][color]['drawer_support_paths']=paths
 result['ownership_label_version']='core_ownership_v2'
 result['holding_derived']=[bool(ok and value==6) for value,ok in zip(result['categorical'][:3],result['categorical_valid'][:3])]
 result['hand_empty_derived']=not any(result['holding_derived']) if all(result['categorical_valid'][:3]) else None
 return result
