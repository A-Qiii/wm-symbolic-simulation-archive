from audit import *
def held(contacts,uid,robot,families):
 links=set();external=0
 for c in contacts:
  if c[9]<=.01 or uid not in c[1:3]:continue
  other,link=(c[2],c[4])if c[1]==uid else(c[1],c[3])
  if other==robot:links.add(link)
  else:external+=1
 return bool(all(links&f for f in families)and not external)
def main():
 assert not(E/'results/raw_place_coverage.json').exists();proto={'scope':'Read saved A/B/C core place_in_drawer traces only; count current physical held-over-opening states and exact2.1s public input availability. Not exhaustive other-action raw-log coverage. Separate already-existing6C held-down windows.','rule':'Same contact-based held; drawer>0.16; centre in native opening XY. Four causal inputs at[-63,-42,-21,0]control ticks. No minimumduration or appearance bin based on D. No new model forward or simulator.'}
 if (E/'configs/raw_place_scope.json').exists():assert read(E/'configs/raw_place_scope.json')==proto
 else:save(E/'configs/raw_place_scope.json',proto);(E/'configs/raw_place_scope.json').chmod(0o444)
 families=[set(x)for x in read(P/'experiments/trajectory_boundary_mining_20261014/configs/audit_protocol.json')['finger_families']];records=[r for r in read(P/'data/core1200_v1/data_manifest.json')['records']if r['environment']in 'ABC'and r['skill']=='place_in_drawer'];rows=[]
 for i,r in enumerate(records):
  folder=Path(r['obs_before']).parent;tr=read(folder/'trace_private.json.gz');info=read(folder/'endpoint_info_private.json')['before'];uids={c:info['scene_info']['movable_objects']['block_'+c]['uid']for c in C};n=NATIVE[r['environment']];R=n['R'];available={0,len(tr)-3,len(tr)-2,len(tr)-1,len(tr)};cc=[]
  for c in C:
   found=[]
   for ti,t in enumerate(tr,1):
    if t['scene_obs'][1]<=.16 or not held(t['contacts'],uids[c],info['robot_info']['uid'],families):continue
    pos=np.array(t['objects']['block_'+c][:3]);q=t['scene_obs'][1];origin=n['xyz']+R@(n['axis']*q);loc=R.T@(pos-origin)/n['scale']
    if abs(loc[0])<=.225 and -.05<=loc[2]<=.2:found.append((ti,float(pos[2]),all(ti-d in available for d in [63,42,21,0])))
   if found:cc.append({'colour':c,'frames':len(found),'first_frame':found[0][0],'last_frame':found[-1][0],'center_z_min_m':min(x[1]for x in found),'center_z_max_m':max(x[1]for x in found),'complete_exact_public_long_windows':sum(x[2]for x in found)})
  rows.append({'root':r['root_id'],'environment':r['environment'],'trajectory':str(folder),'frames_scanned':len(tr),'held_over_drawer':cc})
  if(i+1)%60==0:print('AUDITED_SAVED_PLACE',i+1,len(records),flush=True)
 summary=[]
 for split,envs in [('train','AB'),('dev','C')]:
  rr=[x for x in rows if x['environment']in envs];q=[(r,c)for r in rr for c in r['held_over_drawer']];summary.append({'split':split,'trajectories':len(rr),'frames_scanned':sum(r['frames_scanned']for r in rr),'roots':len(set(r['root']for r in rr)),'held_over_drawer_frames':sum(c['frames']for r,c in q),'held_over_drawer_roots':len(set(r['root']for r,c in q)),'held_over_drawer_trajectories':len(set(r['trajectory']for r,c in q)),'complete_exact_public_long_windows':sum(c['complete_exact_public_long_windows']for r,c in q),'by_colour':{col:{'frames':sum(c['frames']for r,c in q if c['colour']==col),'roots':len(set(r['root']for r,c in q if c['colour']==col)),'center_z_range_m':[min(c['center_z_min_m']for r,c in q if c['colour']==col),max(c['center_z_max_m']for r,c in q if c['colour']==col)]}for col in C}})
 extra=[]
 for r in read(P/'experiments/state_predicates_held_down_20261014/data/metadata.json'):
  st=read(r['labels_private'])['frames'][-1];c=r['target_color'];g=geometry('C',st['labels']['continuous']['drawer_m'],st['objects_audit'][c]['position_m'],c,st.get('object_pose_state_private',{}).get(c,[None]*6)[3:]if 'object_pose_state_private'in st else None);extra.append({'id':r['sample_id'],'root':r['root_id'],'geometry':g,'motion':r['motion'],'status':'existingCdiagnostic_not_seed1_training_or_originaldevscoring'})
 save(E/'results/raw_place_coverage.json',{'protocol':proto,'summary':summary,'trajectories':rows,'existing_C_held_down6':extra,'source_sha256':HASH});print(json.dumps({'summary':summary,'C_extra':extra},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
