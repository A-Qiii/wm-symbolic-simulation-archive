from pathlib import Path
import json,gzip,hashlib,sys
from collections import Counter
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_countermotion_20261014';sys.path.insert(0,str(P/'experiments/trajectory_boundary_mining_20261014/source'))
from mine_trajectories import families,state
C=['red','blue','pink'];HASH={}
def read(p,gz=False):
 p=Path(p);b=p.read_bytes();HASH[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(gzip.decompress(b)if gz else b)
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not(E/'results/existing_held_motion_audit.json').exists();protocol=read(E/'configs/protocol.json');fam=families();core=P/'data/core1200_v1';records=[r for r in read(core/'data_manifest.json')['records']if r['environment']=='C'];sup=P/'data/predicate_gap_supplement_20261014';sr=[r for r in read(sup/'roots_manifest.json')if r['environment']=='C'];recovered=P/'data/predicate_causal_replay_20261014';rr=[r for r in read(recovered/'roots_manifest.json')if r['environment']=='C'];tasks=[('core',r,Path(r['obs_before']).parent)for r in records]+[('supplement',r,sup/'episodes'/(r['root_id']+'__'+branch))for r in sr for branch in r['trajectories']]+[('full_recovered',r,recovered/'episodes'/(r['root_id']+'__grasp'))for r in rr];summaries=[];candidates=[];selected=[];mismatch=[]
 for ti,(source,r,folder)in enumerate(tasks):
  t=read(folder/'trace_private.json.gz',True);available={};held={c:[]for c in C};positions={c:[]for c in C};uids={};visible={c:[]for c in C}
  if source=='core':
   info=read(folder/'endpoint_info_private.json')['before'];rb=info['robot_info'];robot=[list(rb['tcp_pos'])+list(rb['tcp_orn'])+[rb['gripper_opening_width']]+list(rb['arm_joint_states'])+[rb['gripper_action']]]+[x['robot_obs']for x in t];N=len(robot);labs=read(folder/'labels_ownership_v2_private.json')
   for c in C:
    ob=info['scene_info']['movable_objects']['block_'+c];uid=ob['uid'];uids[c]=uid;positions[c]=[ob['current_pos']]+[x['objects']['block_'+c][:3]for x in t];held[c]=[state(ob['current_pos'],ob['contacts'],uid,rb['uid'],robot[0],fam)['held']]+[state(x['objects']['block_'+c],x['contacts'],uid,rb['uid'],x['robot_obs'],fam)['held']for x in t]
    for j,side in [(0,'before'),(-1,'after')]:
     if held[c][j]!=labs[side]['holding_derived'][C.index(c)]:mismatch.append((str(folder),c,side))
   available[0]=str(folder/'initial_public.npz')
   for k in range(4):available[N-4+k]=str(folder/f'boundary_history_{k}.npz')
  else:
   robot=[x['robot_obs']for x in t];N=len(t)
   for c in C:
    held[c]=[x['objects_audit'][c]['held_contact_audit']for x in t];positions[c]=[x['objects_audit'][c]['position_m']for x in t];uids[c]=t[-1]['objects_audit'][c]['object_uid_evaluator_only'];visible[c]=[x['objects_audit'][c]['visible_pixels']for x in t]
    for i,x in enumerate(t):
     if held[c][i]!=x['labels']['holding_derived'][C.index(c)]:mismatch.append((str(folder),c,i))
   if source=='full_recovered':available={i:str(folder/'public'/f'{i:04d}.npz')for i in range(N)}
   else:
    for q in read(folder/'samples_manifest.json'):
     for i,pp in zip(q['history_frame_indices'],q['public_frames']):available[i]=pp
  robot=np.array(robot);found=Counter();selected_here=set()
  for c in C:
   obj=np.array(positions[c]);h=np.array(held[c],bool);prefix=np.r_[0,np.cumsum(h)];allheld=np.where(prefix[64:]-prefix[:-64]==64)[0]+63
   for end in allheld:
    start=end-63;rp=robot[start:end+1,:3];op=obj[start:end+1];dz=float(rp[-1,2]-rp[0,2]);odz=float(op[-1,2]-op[0,2]);drift=float(np.linalg.norm((op-rp)-(op-rp)[0],axis=1).max());rmax=float(np.linalg.norm(rp-rp[0],axis=1).max());omax=float(np.linalg.norm(op-op[0],axis=1).max());conditions=[]
    if dz<=-.02 and odz<=-.02 and drift<=.015:conditions.append('held_down')
    if rmax<=.001 and omax<=.001:conditions.append('held_stationary')
    for condition in conditions:
     inds=[int(end)-d for d in [63,42,21,0]];missing=[i for i in inds if i not in available];complete=not missing;vis=None
     if complete:
      pixels=[]
      for i in inds:
       with np.load(available[i])as z:pixels.append(max(int((((z['seg_'+cam]&((1<<24)-1))==uids[c])&(z['seg_'+cam]>=0)).sum())for cam in ['static','gripper']))
      vis=all(n>=4 for n in pixels)
     q={'source':source,'root_id':r['root_id'],'trajectory':str(folder),'condition':condition,'colour':c,'anchor':int(end),'indices':inds,'held_all64':True,'TCP_z_change_mm':dz*1000,'object_z_change_mm':odz*1000,'relative_drift_mm':drift*1000,'TCP_max_move_mm':rmax*1000,'object_max_move_mm':omax*1000,'exact_public_complete':complete,'visible_all4':vis,'missing_indices':missing,'public_paths':[available[i]for i in inds]if complete else[]};candidates.append(q);found[condition]+=1
     key=(c,condition)
     if complete and vis and key not in selected_here:selected.append(q);selected_here.add(key)
  summaries.append({'source':source,'root_id':r['root_id'],'trajectory':str(folder),'frames':N,'available_public_frames':len(available),'held_motion_candidates':dict(found)})
  if(ti+1)%60==0:print('AUDITED_C',ti+1,'of',len(tasks),flush=True)
 assert not mismatch
 counts={}
 for condition in ['held_down','held_stationary']:
  rows=[q for q in candidates if q['condition']==condition];counts[condition]={'candidate_windows':len(rows),'roots':len(set(q['root_id']for q in rows)),'trajectories':len(set(q['trajectory']for q in rows)),'complete_exact_public_windows':sum(q['exact_public_complete']for q in rows),'selected_windows':sum(q['condition']==condition for q in selected),'examples':rows[:3]}
 result={'frozen_rule':protocol['additional_existing_data_audit'],'trajectory_count':len(tasks),'root_count':len(set(r['root_id']for _,r,_ in tasks)),'frames_scanned':sum(q['frames']for q in summaries),'counts':counts,'trajectories':summaries,'candidates':candidates,'selected':selected,'physical_label_mismatches':mismatch,'source_sha256':HASH,'no_D_read':True,'no_new_collection_or_replay':True,'limitation':'Withinrecordedtrajectorysegments; exactsynchronizedpublicRGBD/mask/proprioisrequired. LossyMP4andprivateGTarenot substitutes. Absenceofaqualifyinglongwindowdoesnotprovethephysicalstateisabsent.'};save(E/'results/existing_held_motion_audit.json',result);save(E/'configs/additional_windows_frozen.json',{'selection_before_predictions':True,'windows':selected,'counts':counts});(E/'configs/additional_windows_frozen.json').chmod(0o444);print(json.dumps({'C_AUDIT_COMPLETE':True,'trajectories':len(tasks),'counts':{k:{kk:vv for kk,vv in v.items()if kk!='examples'}for k,v in counts.items()}},ensure_ascii=False))
if __name__=='__main__':main()
