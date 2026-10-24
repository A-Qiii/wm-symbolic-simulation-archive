from pathlib import Path
import json,gzip,hashlib,re,datetime
from collections import Counter
import numpy as np
from wm_symbolic.vision_cache import VisibleRegistry
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_temporal_20261014';C=P/'experiments/state_predicates_causal_windows_20261014';B=P/'data/predicate_causal_replay_20261014'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def stats(a):
 a=np.asarray(a,float);return {'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max())}
def main():
 assert not(E/'configs/protocol.json').exists(), 'Audit must finish before protocol freeze'
 metadata=read(C/'data/metadata.json');roots=read(B/'roots_manifest.json');samples=[];trajectories=[]
 for root in roots:
  ep=B/'episodes'/(root['root_id']+'__grasp');cfg=(ep/'complete_config.yaml').read_text();physics=float(re.search(r'bullet_time_step: ([0-9.]+)',cfg)[1]);control=float(re.search(r'control_freq: ([0-9.]+)',cfg)[1]);assert physics==240 and control==30 and re.search(r'use_vr: false',cfg)
  reg=VisibleRegistry();frames=[];last=None;duplicates=0
  for pp in sorted((ep/'public').glob('*.npz')):
   with np.load(pp)as z:f={k:z[k]for k in z.files}
   reg.update([f]);h=hashlib.sha256(f['rgb_static'].tobytes()+f['rgb_gripper'].tobytes()).hexdigest();duplicates+=h==last;last=h
   j=['red','blue','pink'].index(root['target_color']);uid=reg.colour_ids.get(j);a={'frame':int(pp.stem),'RGB_sha256':h,'robot':f['robot_obs'].tolist(),'views':{}}
   for cam in ['static','gripper']:
    mask=VisibleRegistry.grouped(f['seg_'+cam])==uid;y,x=np.where(mask);a['views'][cam]={'pixels':len(x),'centroid_xy':[float(x.mean()),float(y.mean())]if len(x)else None,'median_depth_m':float(np.median(f['depth_'+cam][mask]))if len(x)else None}
   frames.append(a)
  trajectories.append({'root_id':root['root_id'],'split':root['split'],'physics_hz':physics,'action_repeat':int(physics//control),'control_hz':control,'frames':len(frames),'identical_adjacent_dual_RGB_pairs':duplicates,'consecutive_pairs':len(frames)-1})
  for r in metadata:
   if r['root_id']!=root['root_id']:continue
   t=r['anchor_frame'];q={'sample_id':r['sample_id'],'root_id':r['root_id'],'split':r['split'],'dev_role':r['dev_role'],'condition':r['condition'],'anchor':t,'target_color':r['target_color'],'public_motion':{},'available_history_frames':t+1}
   for length in [4,16,32,64]:
    ff=frames[t-length+1:t+1];rr=np.array([f['robot']for f in ff]);q['public_motion'][str(length)]={'span_seconds':(length-1)/control,'unique_dual_RGB':len(set(f['RGB_sha256']for f in ff)),'adjacent_equal_dual_RGB':sum(a['RGB_sha256']==b['RGB_sha256']for a,b in zip(ff,ff[1:])),'tcp_net_mm':float(np.linalg.norm(rr[-1,:3]-rr[0,:3])*1000),'tcp_path_mm':float(np.linalg.norm(np.diff(rr[:,:3],axis=0),axis=1).sum()*1000),'gripper_width_range_mm':float(np.ptp(rr[:,6])*1000),'all_frames_target_visible':all(max(x['pixels']for x in f['views'].values())>=4 for f in ff),'views':{}}
    for cam in ['static','gripper']:
     va=ff[0]['views'][cam];vb=ff[-1]['views'][cam];q['public_motion'][str(length)]['views'][cam]={'centroid_net_pixels':float(np.linalg.norm(np.array(vb['centroid_xy'])-va['centroid_xy']))if va['centroid_xy']and vb['centroid_xy']else None,'median_depth_change_mm':(vb['median_depth_m']-va['median_depth_m'])*1000 if va['median_depth_m']is not None and vb['median_depth_m']is not None else None}
   q['temporal_inputs']={arm:[str(ep/'public'/f'{t-d:04d}.npz')for d in offsets]for arm,offsets in {'current_only':[0,0,0,0],'short_history':[3,2,1,0],'long_history':[63,42,21,0]}.items()}
   samples.append(q)
 # Private facts are appended only after all public motion measurements.
 for q in samples:
  r=next(r for r in metadata if r['sample_id']==q['sample_id']);lab=read(r['labels_private']);q['current_held_for_scoring']=lab['frames'][-1]['objects_audit'][r['target_color']]['held_contact_audit'];q['physical_motion_audit_for_explanation_only']=lab['motion_audit_private']
 hist=[]
 for name,file in [('original','state_predicates_20261014/data/metadata.json'),('supplement','state_predicates_augmented_20261014/data/metadata_train_dev.json')]:
  for r in read(P/'experiments'/file):
   if r['split']not in ['train','tune','calibration','dev']:continue
   paths=r.get('public_inputs',r.get('public_frames'));hashes=[];robot=[]
   for pp in paths:
    with np.load(pp)as z:hashes.append(hashlib.sha256(z['rgb_static'].tobytes()+z['rgb_gripper'].tobytes()).hexdigest());robot.append(z['robot_obs'])
   hist.append({'source':name,'root_id':r['root_id'],'split':r['split'],'sample_id':r.get('sample_id',str(r.get('episode_id'))+'_'+str(r.get('step'))),'unique_dual_RGB':len(set(hashes)),'same_adjacent_dual_RGB':sum(a==b for a,b in zip(hashes,hashes[1:])),'tcp_net_mm':float(np.linalg.norm(np.array(robot[-1])[:3]-np.array(robot[0])[:3])*1000),'root_snapshot':name=='original'and r['step']==0})
 agg=[]
 for split in ['train','dev']:
  for condition in ['ordinary_negative','hard_negative','held_positive']:
   rows=[q for q in samples if q['split']==split and q['condition']==condition]
   for length in ['4','16','32','64']:
    agg.append({'split':split,'condition':condition,'frames':int(length),'roots':len(rows),'tcp_net_mm':stats([q['public_motion'][length]['tcp_net_mm']for q in rows]),'static_centroid_net_pixels':stats([q['public_motion'][length]['views']['static']['centroid_net_pixels']for q in rows]),'RGB_unique':stats([q['public_motion'][length]['unique_dual_RGB']for q in rows]),'all_visible_count':sum(q['public_motion'][length]['all_frames_target_visible']for q in rows)})
 summary={'sampling':{'physics_hz':240,'substeps_per_action':8,'recorded_control_hz':30,'elapsed_simulated_seconds_per_frame':1/30,'four_frame_span_seconds':.1,'long_sparse_four_frame_span_seconds':2.1,'time_basis':'Simulator action stepping, not wall-clock render/training runtime. Original root histories follow24settling controls and contain real consecutive frames; fallback padding is not exercised.'},'full_trajectories':trajectories,'recovered_samples':samples,'aggregates':agg,'historical_windows':hist,'historical_summary':{'windows':len(hist),'root_snapshots_after_settling':sum(r['root_snapshot']for r in hist),'synthetic_initial_padding':0,'padding_evidence':'core1200_collect executes24env.steps before root histories; deque already has4true frames. hist_save has a padding fallback but it is not exercised here.','all_dual_RGB_equal':sum(r['unique_dual_RGB']==1 for r in hist),'unique_RGB_counts':dict(Counter(r['unique_dual_RGB']for r in hist))},'fairness':{'recovered_training_windows':36,'recovered_dev_windows':18,'missing_64frame_history':sum(q['available_history_frames']<64 for q in samples),'all_64frames_visible':sum(q['public_motion']['64']['all_frames_target_visible']for q in samples),'new_roots_needed':False,'historical_long_context_unavailable':'Original full RGB/depth/mask intermediate frames were not exported. Historical900+73 training windows must remain identical 4-frame rehearsal in every arm; they are not part of the temporal treatment. No synthetic history or unequal exclusions.','scope':'A targeted temporal evidence intervention on the same recovered36train/18dev anchors with identical historical rehearsal, not a whole-corpus1/4/64frame benchmark.','limitation':'Recovered data contain no in-positive examples; identical historical rehearsal is needed to retain support for in and broad state coverage.'}}
 save(E/'results/audit.json',summary);print(json.dumps({'audit_complete':True,'sampling':summary['sampling'],'historical_summary':summary['historical_summary'],'fairness':summary['fairness'],'train_motion':[a for a in agg if a['split']=='train'and a['frames']in[4,64]]},ensure_ascii=False))
if __name__=='__main__':main()
