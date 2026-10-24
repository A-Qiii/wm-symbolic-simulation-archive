from common import *
import collections
sys.addaudithook(guard)
roots=read(S/'data/roots_manifest.json');old={r['root_id']:r for r in read(P/'data/core1200_v1/roots_manifest.json')};checks=[]
for r in roots:
 if r['split']=='train':
  t=old[r['paired_old_root']]
  for key in['environment','initial_state','held_color','branches','setup_skills']:assert r[key]==t[key],(r['root_id'],key)
  for c in COLORS:
   x=r['layout']['block_'+c];y=t['layout']['block_'+c];assert x[1:]==y[1:];assert cls(x[0])==cls(y[0])
  for c in COLORS:
   for d in COLORS:
    if c!=d and cls(t['layout']['block_'+c][0])==cls(t['layout']['block_'+d][0]):assert(r['layout']['block_'+c][0]<r['layout']['block_'+d][0])==(t['layout']['block_'+c][0]<t['layout']['block_'+d][0])
  checks.append({'root_id':r['root_id'],'paired_old_root':r['paired_old_root'],'only_x_recipe_changed':True})
 else:
  if r['split']=='gap':
   c=r['gap_target_color'];x=r['layout']['block_'+c][0];k=r['gap_target_class'];assert c!=r['held_color'];assert(.087206<x<.141667)if k==1 else(.151667<x<.249974);assert r['environment']=='A'or k==1
for split,counts in [('train',{'closed_empty':20,'open_empty':12,'open_held':8}),('old_band',{'closed_empty':5,'open_empty':3,'open_held':2}),('gap',{'closed_empty':5,'open_empty':3,'open_held':2})]:
 assert dict(collections.Counter(r['initial_state']for r in roots if r['split']==split))==counts
assert dict(collections.Counter((r['environment'],r['gap_target_class'])for r in roots if r['split']=='gap'))=={('A',2):3,('A',1):2,('B',1):5}
assignment=read(S/'configs/joint_quantile_assignment.json');groups=collections.defaultdict(list)
rm={r['root_id']:r for r in roots}
for a in assignment:
 r=rm[a['root']];groups[r['environment'],a['color'],cls(r['layout']['block_'+a['color']][0])].append(a['rank'])
for key,ranks in groups.items():assert sorted(ranks)==list(range(len(ranks))),key
save(S/'results/recipe_invariants.json',{'utc':now(),'passed':True,'train_pairs':40,'only_designed_recipe_change':'x','fixed_y_yaw_color_class_initial_setup_actions':True,'within_class_order_preserved':True,'quantile_ranks_each_used_once':True,'evaluation_state_quotas_exact':True,'A_gap_targets_right3_middle2':True,'B_gap_targets_middle5_no_right':True,'rows':checks})
print('RECIPE_INVARIANTS_PASS')
