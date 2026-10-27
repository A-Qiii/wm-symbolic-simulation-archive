"""Lock layout splits, recipes and budgets before new-data collection."""
import hashlib,json
import numpy as np
from block_env import B,save
pick=lambda n:['Pick',n,None]
place=lambda n,t:['Place',n,t]
COMMON={
 'A_loaded_forward':{'stacked':True,'large_region':0,'actions':[pick('large'),place('large','B')]},
 'B_empty_reverse':{'stacked':False,'large_region':1,'actions':[pick('large'),place('large','A')]},
 'A_empty_forward':{'stacked':False,'large_region':0,'actions':[pick('large'),place('large','B')]},
 'A_remove_then_move':{'stacked':True,'large_region':0,'actions':[pick('small'),place('small','A'),pick('large'),place('large','B')]},
 'A_load_then_move':{'stacked':False,'large_region':0,'actions':[pick('small'),place('small','large_top'),pick('large'),place('large','B')]},
 'A_smallB_releaseB':{'stacked':False,'large_region':0,'small_region_B':True,'actions':[pick('small'),place('small','B')]}}
POSITIVE={
 'state':{'B_smallB_releaseB':{'stacked':False,'large_region':1,'small_region_B':True,'actions':[pick('small'),place('small','B')]}},
 'transition':{'B_loaded_reverse':{'stacked':True,'large_region':1,'actions':[pick('large'),place('large','A')]}}}
TEST={
 'state':{'paired_A_smallHeld':{'stacked':False,'large_region':0,'small_region_B':True,'actions':[pick('small'),place('small','B')]},'paired_B_smallHeld':{'stacked':False,'large_region':1,'small_region_B':True,'actions':[pick('small'),place('small','B')]}},
 'transition':{'paired_B_loaded_reverse':{'stacked':True,'large_region':1,'actions':[pick('large'),place('large','A')]},'paired_B_empty_reverse':{'stacked':False,'large_region':1,'actions':[pick('large'),place('large','A')]}}}

def layout(group,recipe):
 r=np.random.default_rng(group['layout_seed']);dx=float(r.uniform(-.008,.008));dy=float(r.uniform(-.004,.004));yaw=float(r.uniform(-.06,.06));reg=recipe['large_region'];xy=[(-.08 if reg==0 else .18)+dx,-.14+dy]
 g={'seed':group['layout_seed'],'group_id':group['id'],'large_xy':xy,'stacked':recipe['stacked'],'yaw':yaw,'small_relative_xy':[float(r.uniform(.027,.033)),float(r.uniform(-.002,.002))]}
 if recipe.get('small_region_B'):g['small_xy']=[.32+dx,-.14+dy]
 return g

def main():
 target=B/'configs/experiment_plan.json'
 if target.exists():raise FileExistsError('Plan already locked: '+str(target))
 experiments={}
 for exp,base in [('state',240000000),('transition',250000000)]:
  rng=np.random.default_rng(base);order=rng.permutation(100).tolist();split={i:('train'if j<60 else'development'if j<80 else'test')for j,i in enumerate(order)};rank={s:{i:j for j,i in enumerate(sorted(k for k,v in split.items()if v==s))}for s in ['train','development','test']};groups=[{'id':f'{exp}_{i:03d}','layout_seed':base+1000+i,'split':split[i],'covered_positive':rank[split[i]][i]%2==1}for i in range(100)];episodes=[];arms={arm:{s:[]for s in ['train','development','test']}for arm in ['holdout','covered']}
  for g in groups:
   recipes=TEST[exp]if g['split']=='test'else{**COMMON,**POSITIVE[exp]}
   for name,recipe in recipes.items():
    eid=g['id']+'__'+name;episodes.append({'id':eid,'group':g['id'],'split':g['split'],'recipe':name,'layout':layout(g,recipe),'actions':recipe['actions']})
    for arm in arms:
     replace='A_smallB_releaseB'if exp=='state'else'B_empty_reverse'
     include=g['split']=='test'or(name in COMMON and not(arm=='covered'and g['covered_positive']and name==replace))or(arm=='covered'and g['covered_positive']and name in POSITIVE[exp])
     if include:arms[arm][g['split']].append(eid)
  group_index={g['id']:g for g in groups};replace='A_smallB_releaseB'if exp=='state'else'B_empty_reverse';positive=next(iter(POSITIVE[exp]))
  for s in ['train','development']:
   ordered=[]
   for eid in arms['holdout'][s]:
    gid,name=eid.split('__');ordered.append(gid+'__'+positive if group_index[gid]['covered_positive']and name==replace else eid)
   assert set(ordered)==set(arms['covered'][s]);arms['covered'][s]=ordered
  for arm in arms:
   assert len(arms[arm]['train'])==360 and len(arms[arm]['development'])==120 and len(arms[arm]['test'])==40
   selected=set(arms[arm]['train']);assert sum(len(x['actions'])for x in episodes if x['id']in selected)==960
  experiments[exp]={'groups':groups,'episodes':episodes,'arms':arms,'holdout_definition':('No large=B AND small=Held at any training/development observation, target, four-frame input or trajectory history'if exp=='state'else'No loaded large B→A transition in training/development; loaded large B terminal states remain allowed'),'positive_control':('In 30/60 training groups and 10/20 development groups, replace A_smallB_releaseB by B_smallB_releaseB; same small actor/action/target counts, retain 30 A examples and add 30 B examples.'if exp=='state'else'In 30/60 training groups and 10/20 development groups, replace empty B→A by loaded B→A; same large actor, direction, destination and action counts, retain 30 empty reverse examples and add 30 loaded reverse examples.'),'sample_budget':'360 train episodes / 960 transitions per arm; 120 development episodes; same updates.'}
 plan={'version':'blocks_01','groups_per_experiment':100,'group_splits':{'train':60,'development':20,'test':20},'training_seeds':[0,1,2],'state_updates_per_seed':3000,'dynamics_updates_per_condition_seed':3000,'dynamics_batch_size':16,'state_batch_size':64,'optim':'AdamW lr3e-4 cosine to3e-5; weight_decay1e-4; gradient_clip1','conditions':['direct_relative','motion_matrix'],'random_initialization':'All task/state/readout/transition weights new; only frozen pretrained DINO reused','model_input_allowlist':['rgb_static','rgb_gripper','robot_obs'],'geometry_supervision':'training-only XYZ/size, region and physical Held; same in both structures; no support/contact/on labels','state_holdout_model_scope':'Four-frame current visual state estimator, no GRU history; tests joint visual configuration and feedback non-target preservation. Four-frame image windows still checked against holdout.','test_protocol':'No final test labels/predictions inspected for tuning; collect/evaluate once after engineering and development freeze; exposed engineering seeds never final test','development_selection':'Lowest development loss among models with current/one-step/root-end joint >=.98; otherwise best minimum of these then loss, marked basic ability failure','planning':'Finite candidates evaluated by model successors; no predicted feasible program => stop; skill/physics failure => stop; no fallback','experiments':experiments}
 save(target,plan);sha=hashlib.sha256(target.read_bytes()).hexdigest();save(B/'configs/experiment_plan.sha256.json',{'sha256':sha,'file':'configs/experiment_plan.json'});print('LOCKED',sha)

if __name__=='__main__':main()
