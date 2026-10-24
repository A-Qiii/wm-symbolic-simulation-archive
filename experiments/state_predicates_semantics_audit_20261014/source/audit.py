"""CPU-only audit of saved states, tensors and predictions. No model forward."""
from pathlib import Path
import ast,json,hashlib,gzip,itertools,xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_semantics_audit_20261014'
O=P/'experiments/state_predicates_20261014';S=P/'experiments/state_predicates_augmented_20261014';T=P/'experiments/state_predicates_temporal_20261014';A=P/'experiments/state_predicates_interface_acceptance_20261014'
C=['red','blue','pink'];HASH={};CACHE={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);b=p.read_bytes();HASH[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(gzip.decompress(b)if p.suffix=='.gz'else b)
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def load(p):
 p=Path(p);HASH[str(p)]=sha(p)
 with np.load(p)as z:return {k:z[k]for k in z.files}
def targets(l):
 c=np.array(l['categorical'][:4]);v=np.array(l['categorical_valid'][:4],bool);return np.r_[c[:3]==6,c[:3]==3,c[3]==2],np.r_[v[:3],v[:3],v[3]]
def native(env):
 cfgp=P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{env}.yaml';cfg=yaml.safe_load(cfgp.read_text());HASH[str(cfgp)]=sha(cfgp);base=P/'vendor/calvin/calvin_env/data';tp=base/cfg['objects']['fixed_objects']['table']['file'];HASH[str(tp)]=sha(tp);xml=ET.parse(tp).getroot();j=xml.find("joint[@name='base__drawer']");sc=cfg['global_scaling'];ori=j.find('origin');R=Rotation.from_euler('xyz',np.fromstring(ori.attrib['rpy'],sep=' ')).as_matrix();xyz=np.fromstring(ori.attrib['xyz'],sep=' ')*sc;axis=np.fromstring(j.find('axis').attrib['xyz'],sep=' ');meshes={}
 for kind in ['visual','collision']:
  v=xml.find("link[@name='drawer_link']/"+kind);mp=(tp.parent/v.find('geometry/mesh').attrib['filename']).resolve();HASH[str(mp)]=sha(mp);vv=np.array([[float(x)for x in ln.split()[1:4]]for ln in mp.read_text().splitlines()if ln.startswith('v ')]);meshes[kind]=vv
 for corner in itertools.product([-.225,.225],[.025],[-.05,.2]):assert np.linalg.norm(meshes['visual']-corner,axis=1).min()<1e-6
 blocks={}
 for c in C:
  bp=base/cfg['objects']['movable_objects']['block_'+c]['file'];HASH[str(bp)]=sha(bp);bx=ET.parse(bp).getroot().find('link/collision/geometry/box');blocks[c]=np.fromstring(bx.attrib['size'],sep=' ')*sc
 return {'R':R,'xyz':xyz,'axis':axis,'scale':sc,'meshes':meshes,'blocks':blocks}
NATIVE={e:native(e)for e in 'ABCD'}
def geometry(env,q,pos,c,orn=None):
 n=NATIVE[env];R=n['R'];origin=n['xyz']+R@(n['axis']*q);local=R.T@(np.array(pos)-origin)/n['scale'];rim=(n['meshes']['visual']*n['scale']@R.T+origin)[:,2].max();coll=(n['meshes']['collision']*n['scale']@R.T+origin)[:,2].max();g={'center_in_opening_XY':bool(abs(local[0])<=.225 and -.05<=local[2]<=.2),'center_local_m':local.tolist(),'center_world_m':list(pos),'visual_rim_world_z_m':float(rim),'collision_full_mesh_top_world_z_m':float(coll),'center_above_visual_rim_mm':float((pos[2]-rim)*1000)}
 if orn is not None:
  rot=Rotation.from_quat(orn)if len(orn)==4 else Rotation.from_euler('xyz',orn);corners=np.array(list(itertools.product(*[[-s/2,s/2]for s in n['blocks'][c]])));world=rot.apply(corners)+pos;loc=(world-origin)@R/n['scale'];fullxy=bool((np.abs(loc[:,0])<=.225).all()and((loc[:,2]>=-.05)&(loc[:,2]<=.2)).all());g.update(world_corners_m=world.tolist(),object_min_world_z_m=float(world[:,2].min()),object_max_world_z_m=float(world[:,2].max()),bottom_above_visual_rim_mm=float((world[:,2].min()-rim)*1000),bottom_above_collision_top_mm=float((world[:,2].min()-coll)*1000),whole_object_XY_inside=fullxy,whole_object_in_visual_cavity=bool(fullxy and((loc[:,1]>=.025)&(loc[:,1]<=.1)).all()),object_above_visual_cavity=bool((loc[:,1]>.1).all()))
 return g
def state(r,source):
 if source=='core':
  folder=Path(r['public_inputs'][-1]).parent;side='after'
  if r['step']==0:folder=sorted(folder.glob('[0-9][0-9]_*'))[0];side='before'
  info=read(folder/'endpoint_info_private.json')[side];lab=read(folder/'labels_ownership_v2_private.json')[side];obs=info['scene_info']['movable_objects'];return {'labels':lab,'position':{c:obs['block_'+c]['current_pos']for c in C},'orientation':{c:obs['block_'+c]['current_orn']for c in C},'uids':{c:obs['block_'+c]['uid']for c in C},'contacts':{c:obs['block_'+c]['contacts']for c in C},'tcp':info['robot_info']['tcp_pos'],'table':info['scene_info']['fixed_objects']['table']}
 t=read(r['labels_private'])['frames'][-1];orient={}
 if 'object_pose_state_private'in t:orient={c:t['object_pose_state_private'][c][3:]for c in C}
 elif source=='accept':orient={c:np.array(t['scene_obs_private'][6:]).reshape(3,6)[i,3:].tolist()for i,c in enumerate(C)}
 return {'labels':t['labels'],'position':{c:t['objects_audit'][c]['position_m']for c in C},'orientation':orient,'uids':{c:t['objects_audit'][c]['object_uid_evaluator_only']for c in C},'contacts':{c:t['contacts_private'].get(c,t['contacts_private'].get('block_'+c,[]))for c in C},'tcp':t['robot_obs'][:3],'private_frame':t,'table':{'uid':5,'links':{'drawer_link':3}}}
def bbox(frame,uid):
 seg=frame['seg_gripper'];mask=(seg>=0)&((seg&((1<<24)-1))==uid);yy,xx=np.where(mask);return {'pixels':len(xx),'box':([int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]if len(xx)else None)}
def projection(fr,st):
 b=bbox(fr,st['uids']['blue']);p=bbox(fr,st['uids']['pink']);gap=None
 if b['box']and p['box']:
  x,y=b['box'],p['box'];gap=float(np.hypot(max(x[0]-y[2],y[0]-x[2],0),max(x[1]-y[3],y[1]-x[3],0)))
 return {'blue':b,'pink':p,'gap_px':gap,'covisible':b['pixels']>=4 and p['pixels']>=4,'close':bool(b['pixels']>=4 and p['pixels']>=4 and gap<=5),'center_distance_3D_mm':float(np.linalg.norm(np.array(st['position']['blue'])-st['position']['pink'])*1000),'pink_minus_blue_z_mm':float((st['position']['pink'][2]-st['position']['blue'][2])*1000)}
def upward_drawer(st,c,visited=None):
 visited=set()if visited is None else visited
 if c in visited:return False
 uid=st['uids'][c];by={v:k for k,v in st['uids'].items()};table=st['table']
 for cc in st['contacts'][c]:
  if cc[9]<=.01:continue
  other,link,n=(cc[2],cc[4],cc[7])if cc[1]==uid else(cc[1],cc[3],[-x for x in cc[7]])
  if n[2]<=.5:continue
  if other==table['uid']and link==table['links']['drawer_link']:return True
  if other in by and upward_drawer(st,by[other],visited|{c}):return True
 return False
def registry_cls():
 p=P/'src/wm_symbolic/vision_cache.py';tree=ast.parse(p.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='VisibleRegistry');ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns);return ns['VisibleRegistry']
def reconstruct(f,reg,frames,stats):
 vm,vs,rm,rs=[np.array(stats[k],np.float32)for k in ['vision_mean','vision_std','robot_mean','robot_std']];tok=((f['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384);geo=f['geometry'].reshape(4,2,64,8);valid=f['valid'].reshape(4,2,64);ids=f['visible_registry_ids'].tolist();ob=np.zeros((4,3,2,390),np.float32);geomerr=0.
 for j in range(3):
  uid=reg.colour_ids.get(j)
  if uid not in ids:continue
  slot=ids.index(uid);sl=slice(slot*4,slot*4+4);vv=valid[:,:,slot*4];ob[:,j,:,:384]=tok[:,:,sl].mean(2)*vv[:,:,None];ob[:,j,:,384:389]=geo[:,:,slot*4,:5];ob[:,j,:,389]=geo[:,:,sl,5].mean(2)
  for t,fr in enumerate(frames):
   for ci,cam in enumerate(['static','gripper']):
    seg=reg.grouped(fr['seg_'+cam]);yy,xx=np.where(seg==uid);h,w=seg.shape;top=(224-h)//2;left=(224-w)//2;v=len(xx)>=2;box=[(xx.min()+left)/224,(yy.min()+top)/224,(xx.max()+left+1)/224,(yy.max()+top+1)/224,1]if v else[0,0,1/224,1/224,0];geomerr=max(geomerr,float(np.abs(geo[t,ci,slot*4,:5]-box).max()))
 grid=tok[:,:,48:].reshape(4,2,4,4,384).reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,-1);rb=(np.stack([x['robot_obs']for x in frames]).astype(np.float32)-rm)/rs
 return {'objects':ob.reshape(4,3,780),'context':grid,'robot':rb},geomerr
def long_motion(r):
 tp=P/'data/predicate_causal_replay_20261014/episodes'/r['trajectory_id']/'trace_private.json.gz'
 if str(tp)not in CACHE:CACHE[str(tp)]=read(tp)
 tr=CACHE[str(tp)];a=r['anchor_frame'];win=tr[a-63:a+1];rp=np.array([t['robot_obs'][:3]for t in win]);c=r['target_color'];op=np.array([t['objects_audit'][c]['position_m']for t in win]);held=[t['labels']['holding_derived'][C.index(c)]for t in win];rm=float(np.linalg.norm(rp-rp[0],axis=1).max()*1000);om=float(np.linalg.norm(op-op[0],axis=1).max()*1000);return {'held_all64':all(held),'TCP_max_displacement_mm':rm,'object_max_displacement_mm':om,'stationary64':bool(all(held)and rm<=1 and om<=1)}
def metric(p,y,v):
 hat=p>=.5;known=np.ones_like(hat);conf=hat[:,:3].sum(1)>1;known[:,:3]&=~(conf[:,None]&hat[:,:3]);err=(hat!=y)&v
 return {'valid_facts':int(v.sum()),'raw_FP':int((hat&~y&v).sum()),'raw_FN':int((~hat&y&v).sum()),'known_FP':int((hat&~y&v&known).sum()),'known_FN':int((~hat&y&v&known).sum()),'UNKNOWN':int((~known&v).sum()),'error_windows':int(err.any(1).sum()),'conflict_windows':int(conf.sum())},err,known
def main():
 assert not(E/'results/audit.json').exists();protocol=read(E/'configs/protocol.json')
 for p,h in protocol['frozen_sha256'].items():assert sha(p)==h,p
 regclass=registry_cls();stats=read(O/'data/normalization.json');longpaths={r['sample_id']:r['long_paths']for r in read(T/'results/cache_audit.json')['details']};rows=[];audit={'windows':0,'identity_compared':0,'identity_missing':0,'identity_mismatch':[],'max_tensor_error':{},'max_ROI_box_error':0.,'label_mismatch':[],'long_metadata_path_mismatch':0,'unknown_public_keys':[]};domains={};sources=[('core',O/'data/metadata.json',O/'data/inputs_targets.npz'),('supplement',S/'data/metadata_train_dev.json',S/'data/supplement_train_dev.npz'),('long',T/'data/metadata.json',T/'data/temporal.npz')]
 for source,mp,ap in sources:
  meta=read(mp);arr=load(ap);reg=None;previous=None
  for i,r in enumerate(meta):
   if source=='core'and r['split']=='diagnostic_D':continue
   st=state(r,source);lab=st['labels'];y,v=targets(lab);split='train'if r['split']=='train'else'dev';sid=r.get('sample_id',r.get('episode_id','')+f"__boundary{r.get('step',0)}");paths=r.get('public_inputs',r.get('public_frames'));fp=r['feature_path'];group=r.get('episode_id',r.get('trajectory_id'))
   if source=='long':
    paths=longpaths[sid];fp=str(T/'data/features'/(sid+'.npz'));audit['long_metadata_path_mismatch']+=paths!=r['public_frames'];reg=regclass()
   elif group!=previous:reg=regclass()
   previous=group;frames=[load(p)for p in paths];reg.update(frames);f=load(fp);actual,ge=reconstruct(f,reg,frames,stats);audit['max_ROI_box_error']=max(audit['max_ROI_box_error'],ge);audit['windows']+=1
   for fr in frames:
    bad=[k for k in fr if k not in {'rgb_static','rgb_gripper','depth_static','depth_gripper','robot_obs','seg_static','seg_gripper'}and not k.startswith('calib_')]
    audit['unknown_public_keys']+=bad
   for k,a in actual.items():
    ref=arr[('long_history_'if source=='long'else'')+k][i];audit['max_tensor_error'][k]=max(audit['max_tensor_error'].get(k,0.),float(np.abs(a-ref).max()))
   for j,c in enumerate(C):
    if j not in reg.colour_ids:audit['identity_missing']+=1
    else:
     audit['identity_compared']+=1
     if reg.colour_ids[j]!=st['uids'][c]:audit['identity_mismatch'].append([sid,c,reg.colour_ids[j],st['uids'][c]])
   mask=v if source=='core'else arr['eval_valid'if source=='supplement'else'v'][i];yy=y if source=='core'else arr['y'][i].astype(bool)
   if source=='core':
    refc=arr['cat'][i];refv=arr['valid'][i]
    if not(np.array_equal(refc,lab['categorical'][:4])and np.array_equal(refv,lab['categorical_valid'][:4])):audit['label_mismatch'].append(sid)
   elif not np.array_equal(yy,y):audit['label_mismatch'].append(sid)
   cat=lab['categorical'][:4];q=lab['continuous']['drawer_m'];heldgeo={c:geometry(r['environment'],q,st['position'][c],c,st['orientation'].get(c))for j,c in enumerate(C)if y[j]and v[j]};over=[c for c,g in heldgeo.items()if q>.16 and g['center_in_opening_XY']];pink=bool(y[2]and v[2]and not y[1]and v[1]and cat[1]in [0,1,2,3,4,5]);proj=projection(frames[-1],st)if pink else None;support=[c for j,c in enumerate(C)if cat[j]==3 and v[j]and not upward_drawer(st,c)];domain='old_train'if source=='core'and split=='train'else('old_tune'if source=='core'and r['split']=='tune'else('old_cal'if source=='core'else('new_train'if split=='train'else'new_dev')))
   if source=='long':domain='long_train'if split=='train'else r['dev_role']
   row={'id':sid,'source':source,'split':split,'domain':domain,'root':r['root_id'],'environment':r['environment'],'category':cat,'physical_y':y.tolist(),'physical_valid':v.tolist(),'loss_eval_mask':mask.tolist(),'target':r.get('target_color'),'input_span_seconds':2.1 if source=='long'else .1,'actual_public_paths':paths,'actual_feature_path':fp,'held_over_open_drawer':over,'held_geometry':heldgeo,'pink_held_blue_supported':pink,'projection':proj,'drawer_labels_without_upward_support_path':support,'long_motion':long_motion(r)if source=='long'else None}
   rows.append(row);domains.setdefault(domain,[]).append(len(rows)-1)
   if len(rows)%150==0:print('AUDITED_INPUTS',len(rows),flush=True)
  del arr
 counts=[]
 for split,source in itertools.product(['train','dev'],['core','supplement','long']):
  rr=[r for r in rows if r['split']==split and r['source']==source];pp=[r for r in rr if r['pink_held_blue_supported']];gg=[(r,c)for r in rr for c in r['held_over_open_drawer']];close=[r for r in pp if r['projection']['close']];counts.append({'split':split,'source':source,'windows':len(rr),'roots':len(set(r['root']for r in rr)),'span_seconds':rr[0]['input_span_seconds'],'held_over_open_drawer_windows':len(set(r['id']for r,c in gg)),'held_over_open_drawer_by_colour':dict(Counter(c for r,c in gg)),'held_over_open_drawer_center_height_above_rim_mm':[min(g['center_above_visual_rim_mm']for r,c in gg for g in [r['held_geometry'][c]]),max(g['center_above_visual_rim_mm']for r,c in gg for g in [r['held_geometry'][c]])]if gg else None,'pink_held_blue_supported':len(pp),'pink_held_blue_supported_roots':len(set(r['root']for r in pp)),'pink_blue_covisible':sum(r['projection']['covisible']for r in pp),'pink_blue_close':len(close),'pink_blue_close_roots':len(set(r['root']for r in close)),'pink_blue_close_blue_supervised':sum(r['loss_eval_mask'][1]for r in close),'all_blue_held_supervised':sum(r['loss_eval_mask'][1]for r in rr),'all_loss_mask_counts':np.array([r['loss_eval_mask']for r in rr]).sum(0).tolist(),'physical_valid_counts':np.array([r['physical_valid']for r in rr]).sum(0).tolist(),'stationary64':sum(r['long_motion']['stationary64']for r in rr)if source=='long'else None,'drawer_labels_without_upward_support_path':sum(len(r['drawer_labels_without_upward_support_path'])for r in rr)})
 scores=[];errors=[]
 for seed,domain in itertools.product([0,1,2],['old_tune','old_cal','new_dev','fresh_tune','fresh_cal']):
  rr=[rows[i]for i in domains[domain]];ppath=T/f'results/long_history_seed{seed}_fixed600_{domain}.npy';HASH[str(ppath)]=sha(ppath);pr=np.load(ppath);y=np.array([r['physical_y']for r in rr],bool);v=np.array([r['physical_valid']for r in rr],bool);mask=np.array([r['loss_eval_mask']for r in rr],bool);full,er,known=metric(pr,y,v);original,_,_=metric(pr,y,mask);scores.append({'seed':seed,'domain':domain,'windows':len(rr),'original_masked':original,'posthoc_full_physical_valid':full})
  for i,j in zip(*np.where(er)):
   errors.append({'seed':seed,'domain':domain,'id':rr[i]['id'],'root':rr[i]['root'],'column':int(j),'predicate':(['held_'+c for c in C]+['in_'+c for c in C]+['open_drawer'])[j],'truth':bool(y[i,j]),'probability':float(pr[i,j]),'included_original_mask':bool(mask[i,j]),'decoded_known':bool(known[i,j]),'all_probabilities':pr[i].tolist(),'pink_held_blue_supported':rr[i]['pink_held_blue_supported'],'projection':rr[i]['projection'],'span_seconds':rr[i]['input_span_seconds']})
 acceptmeta=read(A/'data/metadata.json');ev=read(A/'results/evaluation.json');cases=[]
 for rid in protocol['accepted_test_cases']:
  r=next(r for r in acceptmeta if r['unique_window_id']==rid);st=state(r,'accept');q=st['labels']['continuous']['drawer_m'];g={c:geometry('D',q,st['position'][c],c,st['orientation'][c])for c in C};frames=[load(p)for p in r['public_frames']];case={'id':rid,'frame_indices':r['history_frame_indices'],'geometry':g,'projection_blue_pink':projection(frames[-1],st),'current_labels':st['labels'],'physical_audit':st['private_frame']['objects_audit'],'prediction':next(x for x in ev['details']if x['window']==rid),'motion':r['motion'],'image_paths':r['public_frames']};cases.append(case)
 trainroots={r['root']for r in rows if r['split']=='train'};devroots={r['root']for r in rows if r['split']=='dev'};assert not trainroots&devroots;assert not audit['identity_mismatch']and not audit['label_mismatch']and not audit['unknown_public_keys'];assert max(audit['max_tensor_error'].values())<1e-5 and audit['max_ROI_box_error']<1e-6
 result={'scope':protocol['scope'],'input_audit':audit,'coverage':counts,'C_saved_prediction_scores':scores,'C_full_label_raw_errors':errors,'two_acceptance_cases':cases,'split_audit':{'train_roots':len(trainroots),'dev_roots':len(devroots),'intersection':[],'test_used_for_coverage':False},'limitations':['No new forward pass, no training-pool prediction files: training prediction-error rate is unknown.','Current physical states in short windows do not prove full2.1s stationary coverage.','Full raw A/B traces not exhaustively mined in this audit; coverage table counts actual presented model windows.','Repeated boundaries within one root are correlated; window count is not independent root count.','C full-label scores are posthoc diagnostics and do not replace frozen original masked metrics.'],'source_sha256':HASH}
 save(E/'results/audit.json',result);save(E/'results/window_inventory.json',rows);print(json.dumps({'complete':True,'inputs':audit,'coverage':counts,'seed1_C_errors':[r for r in errors if r['seed']==1]},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
