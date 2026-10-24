"""Offline independent replay of saved contact/geometry labels and accounting."""
from pathlib import Path
import json,hashlib,csv,datetime,sys,copy,xml.etree.ElementTree as ET,shutil
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/official_state_resimulation_20261018';R=P/'reports/official_state_resimulation_20261018';OLD=P/'experiments/official_AB_state_pilot_20261018';EV=P/'experiments/combination_generalization_20261017/evaluator_review_v1'
sys.path.insert(0,str(EV/'source'));from native_evaluator import NativeEvaluator
COLORS=['red','blue','pink']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(name,data):
 (E/'results'/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda v:v.item()if isinstance(v,np.generic)else v.tolist()if isinstance(v,np.ndarray)else str(v))+'\n')
 if isinstance(data,list)and data:
  keys=list(dict.fromkeys(k for r in data for k in r))
  with(R/f'{name}.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in r.items()}for r in data)
def topology():
 path=P/'vendor/calvin/calvin_env/data/franka_panda/panda_longer_finger.urdf';root=ET.parse(path).getroot();joints=root.findall('joint');children={}
 for j in joints:children.setdefault(j.find('parent').get('link'),[]).append(j)
 base=next(l.get('name')for l in root.findall('link')if l.get('name')not in {j.find('child').get('link')for j in joints});ordered=[]
 def dfs(link):
  for j in children.get(link,[]):ordered.append(j);dfs(j.find('child').get('link'))
 dfs(base);assert len(ordered)==16
 index={j.get('name'):i for i,j in enumerate(ordered)};assert [index[n]for n in ['panda_finger_joint1','panda_finger_joint2','tcp_joint']]==[9,11,15]
 fam=[]
 for name in ['panda_finger_joint1','panda_finger_joint2']:
  j=ordered[index[name]];links={j.find('child').get('link')};family={index[name]}
  for _ in range(len(ordered)):
   for k,jj in enumerate(ordered):
    if jj.find('parent').get('link')in links:family.add(k);links.add(jj.find('child').get('link'))
  fam.append(sorted(family))
 assert fam==[[9,10],[11,12]]
 save('robot_topology_reference',{'source':str(path),'sha256':sha(path),'method':'URDF depth-first child-link order, crosschecked with fixed native gripper IDs9/11 and TCP15; no new physical initialization','joint_indices':index,'finger_families':fam});return fam
def main():
 rows=read(E/'results/results.json');attempts=read(E/'results/attempts.json');budget=read(E/'results/run_budget.json');assert len(rows)==len(attempts)==128 and len({(r['root_id'],r['frame_id'])for r in rows})==128;assert all(r['status']=='completed'for r in attempts);assert budget['known_completed_physics_ticks']==1024 and budget['retry_count']==0 and budget['stop']is None
 for p,h in read(E/'configs/frozen.json')['sha256'].items():assert sha(p)==h,p
 for p,h in read(OLD/'configs/delivery_manifest.json')['sha256'].items():assert sha(p)==h,p
 families=topology();ev={s:NativeEvaluator(s)for s in ['A','B']};facts=[];stepmetrics=[];allmotor=[];physical_checks=0;obschecks=0
 for row in rows:
  folder=Path(row['folder']);source=np.load(row['source_path'],allow_pickle=False);assert sha(row['source_path'])==row['source_sha256'];assert row['scene']in['A','B'];audit=read(folder/'control_restore_audit.json');commands=audit['reset_and_command_motor_calls'];assert len(commands)==18;robotuid=commands[0]['bodyUniqueId'];grip=int(source['robot_obs'][-1]);target=.04 if grip==1 else 0.;force=2. if grip==1 else 200.
  reset={c['jointIndex']:c for c in commands if c['phase']=='reset'};hold={c['jointIndex']:c for c in commands if c['phase']=='hold_command'};assert len(reset)==len(hold)==9
  for j in range(7):assert reset[j]['targetPosition']==source['robot_obs'][7+j]
  for j in [9,11]:assert reset[j]['targetPosition']==source['robot_obs'][6]/2 and hold[j]['targetPosition']==target and hold[j]['force']==force
  assert np.array_equal(audit['fixed_absolute_command'][0],source['robot_obs'][:3])and np.array_equal(audit['fixed_absolute_command'][1],source['robot_obs'][3:6])and audit['fixed_absolute_command'][2]==grip
  assert read(folder/'integration_ledger.json')['completed_physics_ticks']==8
  allmotor.append({k:row[k]for k in ['root_id','scene','frame_id','sample_roles']}|row['motor_audit'])
  recomputed=[];confs=[]
  for step in range(9):
   record=read(folder/f'physics_step{step:02d}.json');snap=record['physical'];held=[];forces=[]
   for c in COLORS:
    uid=snap['objects']['block_'+c]['uid'];f=[0.,0.];ext=0.
    for contact in snap['contacts']:
     if uid not in[contact[1],contact[2]]:continue
     body,link=(contact[2],contact[4])if contact[1]==uid else(contact[1],contact[3]);value=float(contact[9])
     if body==robotuid:
      for k,family in enumerate(families):
       if link in family:f[k]=max(f[k],value)
     else:ext=max(ext,value)
    if min(f)>.01 and ext<=.01:held.append('block_'+c)
    forces.append((f,ext))
   assert set(snap['held'])==set(held);changed=copy.deepcopy(snap);changed['held']=held;lab=ev[row['scene']].evaluate(changed);assert lab['y']==record['labels']['y']and lab['v']==record['labels']['v'];physical_checks+=1
   confidence=[]
   for j,c in enumerate(COLORS):
    f,ext=forces[j];hc=(min(f)>=.015 and ext<=.005)if lab['y'][j]else(ext>=.015 or min(f)<=.005);assert hc==record['confidence'][c]['Holding_confident'];confidence.append(bool(hc))
   for j,c in enumerate(COLORS):
    g=lab['objects'][c];ic=bool(lab['v'][j+3]and(g['outside']or confidence[j])and(not lab['y'][j+3]or g['certain_contained_load_bearing_paths']));assert ic==record['confidence'][c]['In_confident'];confidence.append(ic)
   recomputed.append(lab);confs.append(confidence)
   if step in[1,4,8]:
    q=Path(record['new_observation_path']);assert q.is_relative_to(E/'data')and sha(q)==record['new_observation_sha256'];obs=np.load(q,allow_pickle=False);assert np.array_equal(obs['robot_obs'],record['state']['robot_obs']);assert 'scene_obs'not in obs.files and 'contacts'not in obs.files;assert obs['rgb_static'].shape==(200,200,3)and obs['rgb_gripper'].shape==(84,84,3);obschecks+=1
  cand=read(folder/'candidate_state_label.json');assert cand==row['candidate']and cand['new_physics_step']==8 and cand['observation_path']==str(folder/'observation_step08.npz')
  for j in range(6):
   value=recomputed[8]['y'][j];valid=all(lab['y'][j]==value and lab['v'][j]and cf[j]for lab,cf in zip(recomputed[4:9],confs[4:9]));assert cand['y'][j]==value and cand['v'][j]==valid
   facts.append({k:row[k]for k in ['root_id','scene','frame_id','sample_roles']}|{'fact':cand['fact_order'][j],'group':'Holding'if j<3 else'In','color':COLORS[j%3],'y':value,'valid':valid,'classification':'positive'if valid and value else'negative'if valid else'unknown','reason':cand['validity_reason'][j],'new_observation_path':cand['observation_path'],'old_RGB_used_for_label':False})
 save('candidate_facts',facts);save('control_audit_summary',allmotor)
 summary=[]
 for rid in ['all']+sorted({r['root_id']for r in rows}):
  for role in ['all','continuous_restoration','uniform_coverage']:
   for group in ['Holding','In']:
    ff=[r for r in facts if(rid=='all'or r['root_id']==rid)and(role=='all'or role in r['sample_roles'])and r['group']==group];rr=[r for r in rows if(rid=='all'or r['root_id']==rid)and(role=='all'or role in r['sample_roles'])]
    summary.append({'root_id':rid,'role':role,'group':group,'initializations':len(rr),'independent_episodes':len({r['root_id']for r in rr}),'fact_cells':len(ff),'positive':sum(f['classification']=='positive'for f in ff),'negative':sum(f['classification']=='negative'for f in ff),'unknown':sum(f['classification']=='unknown'for f in ff),'positive_states':len({(f['root_id'],f['frame_id'])for f in ff if f['classification']=='positive'}),'unknown_states':len({(f['root_id'],f['frame_id'])for f in ff if f['classification']=='unknown'})})
 save('episode_summary',summary)
 color=[]
 for group in ['Holding','In']:
  for c in COLORS:
   ff=[r for r in facts if r['group']==group and r['color']==c];pp=[f for f in ff if f['classification']=='positive'];color.append({'group':group,'color':c,'positive':len(pp),'positive_episodes':len({f['root_id']for f in pp}),'negative':sum(f['classification']=='negative'for f in ff),'unknown':sum(f['classification']=='unknown'for f in ff),'positive_episode_ids':sorted({f['root_id']for f in pp})})
 save('color_coverage',color)
 for step in[1,4,8]:
  for rid in ['all']+sorted({r['root_id']for r in rows}):
   rr=[r for r in rows if rid=='all'or r['root_id']==rid];m={'step':step,'root_id':rid,'initializations':len(rr),'max_object_drift_mm':1000*max(d['translation_m']for r in rr for d in r['steps'][step]['object_drift_from_initial'].values()),'max_object_rotation_deg':180/np.pi*max(d['rotation_rad']for r in rr for d in r['steps'][step]['object_drift_from_initial'].values()),'max_TCP_from_recorded_mm':1000*max(r['steps'][step]['tcp_position_from_recorded_m']for r in rr),'max_TCP_from_initialized_mm':1000*max(r['steps'][step]['tcp_position_from_initialized_m']for r in rr),'mean_static_MAE_0_255':float(np.mean([r['steps'][step]['new_RGB_vs_original_MAE_0_255']['rgb_static']for r in rr])),'mean_wrist_MAE_0_255':float(np.mean([r['steps'][step]['new_RGB_vs_original_MAE_0_255']['rgb_gripper']for r in rr]))}
   for g,sl in [('Holding',slice(0,3)),('In',slice(3,6))]:m[g+'_raw_positive_states']=sum(any(r['steps'][step]['labels']['y'][sl])for r in rr);m[g+'_raw_positive_facts']=sum(sum(r['steps'][step]['labels']['y'][sl])for r in rr)
   stepmetrics.append(m)
 save('checkpoint_metrics',stepmetrics)
 old=read(OLD/'results/restoration_refined_masks.json');index={(r['root_id'],r['frame_id']):r for r in rows};pairs=[]
 for oldrow in old:
  for j,c in enumerate(COLORS):
   if not oldrow['restored_t1']['y'][3+j]:continue
   r=index[(oldrow['root_id'],oldrow['frame_id'])];f=next(f for f in facts if f['root_id']==r['root_id']and f['frame_id']==r['frame_id']and f['fact']==f'In({c},drawer)');pairs.append({'root_id':r['root_id'],'frame_id':r['frame_id'],'color':c,'old_restore_candidate':True,'old_original_GT_still_unknown':True,'new_raw_steps_1_4_8':[r['steps'][s]['labels']['y'][3+j]for s in[1,4,8]],'new_step8_classification':f['classification'],'new_observation_path':f['new_observation_path']})
 assert len({(r['root_id'],r['frame_id'])for r in pairs})==50;save('old_In_candidate_pairs',pairs)
 save('verification',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'initializations':128,'source_episodes':4,'no_retries':True,'physics_ticks':1024,'complete_physics_records_recomputed':physical_checks,'new_observation_records_checked':obschecks,'candidate_fact_cells_checked':len(facts),'motor_calls_checked':128*18,'old_In_candidate_frames':50,'old_In_candidate_facts':len(pairs),'all_old_In_candidate_facts_stable_new_positive':all(r['new_step8_classification']=='positive'for r in pairs),'prior_pilot_seal_unchanged':True,'no_extra_physics_or_model_forward_in_verification':True});shutil.copy2(__file__,E/'source/verify.py');print('RESIM_VERIFIED',physical_checks,'physics snapshots;',obschecks,'new observations;',len(pairs),'old In candidate facts',flush=True)
if __name__=='__main__':main()
