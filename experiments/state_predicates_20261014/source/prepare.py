from pathlib import Path
import json, hashlib, collections, datetime
import numpy as np
from wm_symbolic.vision_cache import VisibleRegistry

P=Path('/home/robot/research/projects/wm_symbolic')
E=P/'experiments/state_predicates_20261014'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def main():
 for d in ['source','configs','data','models','results','figures','logs']: (E/d).mkdir(parents=True,exist_ok=True)
 assert not (E/'configs/protocol.json').exists(),'Do not overwrite frozen experiment'
 m=json.loads((P/'data/core1200_v1/data_manifest.json').read_text())
 idx=json.loads((P/'data/core1200_v1/features/index.json').read_text())['items']; cache={(r['episode_id'],r['step']):r for r in idx}
 by={}
 for r in m['records']: by.setdefault(r['episode_id'],[]).append(r)
 objects=[]; contexts=[]; robots=[]; cats=[]; masks=[]; metadata=[]; near=[]; counts={}; roots={}
 oldstats=json.loads((P/'runs/core1200_v1/training_main/normalization_train_only.json').read_text());vm=np.array(oldstats['vision_mean'],np.float32);vs=np.array(oldstats['vision_std'],np.float32)
 fingerprints={}; source_images={}; source_features={}
 for eid,rows in sorted(by.items()):
  rows.sort(key=lambda r:r['step']);split='train' if rows[0]['split']=='train' else ('diagnostic_D' if rows[0]['split']=='test' else rows[0]['dev_role'])
  labels=[json.loads(Path(rows[0]['labels_path']).read_text())['before']]+[json.loads(Path(r['labels_path']).read_text())['after'] for r in rows]
  reg=VisibleRegistry();roots.setdefault(split,set()).add(rows[0]['root_id']);counts.setdefault(split,np.zeros((4,8),int))
  for step,lab in enumerate(labels):
   item=cache[eid,step];frames=[]
   for f in item['public_inputs']:
    with np.load(f) as z: frames.append({k:z[k] for k in ['rgb_static','rgb_gripper','seg_static','seg_gripper','robot_obs']})
   reg.update(frames)
   with np.load(item['path']) as z:
    tok=((z['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384);geo=z['geometry'].reshape(4,2,64,8);valid=z['valid'].reshape(4,2,64);ids=z['visible_registry_ids'].tolist()
   ob=np.zeros((4,3,2,390),np.float32)
   for j in range(3):
    uid=reg.colour_ids.get(j);slot=ids.index(uid) if uid in ids else None
    if slot is None: continue
    sl=slice(slot*4,slot*4+4);v=valid[:,:,slot*4]
    ob[:,j,:,:384]=tok[:,:,sl,:].mean(2)*v[:,:,None]
    ob[:,j,:,384:389]=geo[:,:,slot*4,:5]
    ob[:,j,:,389]=geo[:,:,sl,5].mean(2)
   grid=tok[:,:,48:,:].reshape(4,2,4,4,384).reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,2,4,384)
   rb=np.stack([f['robot_obs'] for f in frames]).astype(np.float32)
   objects.append(ob.reshape(4,3,-1));contexts.append(grid.reshape(4,-1));robots.append(rb);cats.append(lab['categorical'][:4]);masks.append(lab['categorical_valid'][:4])
   near.append([lab['continuous']['objects'][c]['tcp_distance_m']<.075 and lab['categorical'][j]!=6 for j,c in enumerate(['red','blue','pink'])])
   for j,(c,ok) in enumerate(zip(cats[-1],masks[-1])):
    if ok: counts[split][j,c]+=1
   image_hash=hashlib.sha256(b''.join(f[k].tobytes() for f in frames for k in ['rgb_static','rgb_gripper'])).hexdigest()
   fingerprints.setdefault(image_hash,set()).add(split)
   metadata.append({'index':len(metadata),'episode_id':eid,'root_id':rows[0]['root_id'],'split':split,'step':step,'environment':rows[0]['environment'],'category':cats[-1],'valid':masks[-1],'feature_path':item['path'],'public_inputs':item['public_inputs'],'frame_hash':image_hash,'object_positions_for_evaluation_only':[lab['continuous']['objects'][c]['position_m'] for c in ['red','blue','pink']],'drawer_m_for_evaluation_only':lab['continuous']['drawer_m']})
   source_features[item['path']]=sha(item['path'])
  if len(objects)%150==0:print('PREPARED',len(objects),flush=True)
 splits=np.array([r['split'] for r in metadata]);train=splits=='train';r=np.stack(robots);mean=r[train].mean((0,1));std=np.maximum(r[train].std((0,1)),1e-3)
 cross={h:list(ss) for h,ss in fingerprints.items() if len(ss)>1};assert not cross
 assert all(not a&b for i,a in enumerate(roots.values()) for b in list(roots.values())[i+1:])
 arr={'objects':np.stack(objects),'context':np.stack(contexts),'robot':(r-mean)/std,'cat':np.array(cats,np.int64),'valid':np.array(masks,bool),'near_negative':np.array(near,bool),'splits':splits}
 np.savez_compressed(E/'data/inputs_targets.npz',**arr);save(E/'data/metadata.json',metadata)
 save(E/'data/normalization.json',{'vision_stats_source':'AB training only, unchanged existing normalizer','vision_mean':vm.tolist(),'vision_std':vs.tolist(),'robot_mean':mean.tolist(),'robot_std':std.tolist(),'fit_root_ids':sorted(roots['train'])})
 audit={'rows':len(metadata),'counts':{s:x.tolist() for s,x in counts.items()},'roots':{s:sorted(x) for s,x in roots.items()},'exact_image_cross_split_duplicates':cross,'source_feature_sha256':source_features,'held_near_negative_counts':{s:int(arr['near_negative'][splits==s].sum()) for s in roots},'segmentation':'Existing simulator visible pixel masks; no hidden object positions or raw numeric object IDs input to model','history':'Exactly four frames at/before each boundary. No later boundary, skill action, success, or GT enters forward.'}
 save(E/'data/audit.json',audit)
 protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Current-state recognition diagnostic only; no future-state learning, control changes, or LTL success claim','frozen_before_training':True,'split':{s:sorted(x) for s,x in roots.items()},'diagnostic_D_status':'Previously observed D, independent of training but not a new blind test','models':['matched_eight_class','direct_predicates','direct_predicates_leave_pink_labels_out'],'main_seeds':[0,1,2],'extension_seed':0,'epochs':100,'batch_size':64,'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.001},'selection':'Minimum unweighted mean three-predicate BCE on C tune; checkpoint every 5 epochs','calibration':'C calibration only; per-predicate tau grid .5,.6,.7,.8,.85,.9,.95,.975,.99,.995; largest coverage subject to known assertion error<=5%, at least20 known assertions; held additionally FPR<=1% over all valid negative cases; report achieved coverage and diagnostic uncertainty, not certification','success_criteria':{'absolute':'D diagnostic each predicate balanced accuracy>=.90, held raw FPR<=.02, calibrated known error<=.05 and coverage>=.70','relative':'Direct improves macro balanced accuracy >=.02 OR known coverage >=.10 versus matched eight-class at known error<=.05; held FPR does not increase by >.01','limitations':'Criteria are preregistered diagnostic gates, not evidence of statistically guaranteed success or novel-object generalization'},'entity_interface':'Entity handle selects visible ROI; no learned color index. Shared object encoder, shared temporal GRU and shared held/in heads. Drawer relation target is only fixed drawer; new containers and predicates untested. Color registration currently only red/blue/pink.','input_contract':{'objects':'N x4frames xKentities x(2views*(384DINO+6visible_geometry))','context':'N x4frames x(2views*2x2coarseDINO)','robot':'N x4frames x15 proprioception','prohibited':['true object coordinates','contact states','category labels','future RGB','action success','skill name','environment ID']},'loss':'Unweighted cross entropy eight categories + three drawer categories versus per-predicate BCE, same trunk/input/order/seeds. No label smoothing or class resampling.','extension':'Exclude every pink held/in or eight-class label in auxiliary direct training; context may still show pink. Evaluate known appearance with unlabelled entity only, not unseen new object.','confusion_gap':'No near-gripper-not-held negatives in AB boundary data; only 2 in C tune and2 in D; no targeted mid-action confusion collection in this bounded diagnostic. Drawer intermediate class absent.','fixed_sha256':{str(P/'src/wm_symbolic/physical_oracle.py'):sha(P/'src/wm_symbolic/physical_oracle.py'),str(P/'runs/core1200_v1/training_main/best.pt'):sha(P/'runs/core1200_v1/training_main/best.pt'),str(P/'data/core1200_v1/data_manifest.json'):sha(P/'data/core1200_v1/data_manifest.json')},'dataset_sha256':sha(E/'data/inputs_targets.npz'),'no_new_collection':True,'no_upload_authorized':True}
 save(E/'configs/protocol.json',protocol);(E/'configs/protocol.json').chmod(0o444)
 print('FROZEN',json.dumps({'boundaries':len(metadata),'roots':{s:len(x) for s,x in roots.items()},'cross_split_image_duplicates':len(cross)}),flush=True)
if __name__=='__main__':main()
