"""Freeze metadata-only source sampling, episode split and long-window rules."""
from pathlib import Path
import json,hashlib,collections,shutil
import numpy as np
import official_ab_fetch as F
from official_expansion_fetch import E,R,OLD,read,write
SHORT=F.P/'experiments/official_state_resimulation_20261018'
SEED=170819
def key(tag,value):return hashlib.sha256(f'{SEED}|{tag}|{value}'.encode()).hexdigest()
def main():
 assert not(E/'configs/frozen_selection.json').exists()
 seg=read(E/'results/task_segments_A_B.json');previous=read(OLD/'configs/combined_frame_selection.json')['roots'];oldroots={r['root_id']for r in previous};roots={s['root_id']for s in seg};eligible={};new=[];splits={r:'train_candidate'for r in oldroots};inputs=[]
 for scene in ['A','B']:
  good=[]
  for rid in sorted(roots-oldroots):
   ss=[s for s in seg if s['root_id']==rid and s['scene']==scene and s['interval_inclusive'][1]-s['interval_inclusive'][0]>=12];counts=collections.Counter(s['task']for s in ss)
   if all(counts[f'lift_{c}_block_{loc}']>=1 for c in['red','blue','pink']for loc in['table','drawer'])and counts['place_in_drawer']>=2:good.append(rid)
  eligible[scene]=sorted(good);chosen=sorted(good,key=lambda r:key('episode',r))[:4];assert len(chosen)==4
  splitorder=sorted(chosen,key=lambda r:key('split',r))
  for rid in chosen:
   splits[rid]='validation_candidate'if rid in splitorder[:2]else'train_candidate';new.append(rid)
 for rid in sorted(new):
  ss=[s for s in seg if s['root_id']==rid and s['interval_inclusive'][1]-s['interval_inclusive'][0]>=12];chosen=[]
  for color in ['red','blue','pink']:
   for location in ['table','drawer']:
    candidates=[s for s in ss if s['task']==f'lift_{color}_block_{location}'];s=min(candidates,key=lambda s:key('segment',s['annotation_index']));chosen.append((s,[0,1/3,2/3,1]))
  for s in sorted([s for s in ss if s['task']=='place_in_drawer'],key=lambda s:key('segment',s['annotation_index']))[:2]:chosen.append((s,[0,.5,1]))
  frames={}
  for s,phases in chosen:
   start,end=s['interval_inclusive']
   for phase in phases:
    fid=start+int(np.floor((end-start)*phase));hint={k:s[k]for k in['annotation_index','task','annotation','interval_inclusive']};hint['phase_fraction']=phase
    if fid not in frames:frames[fid]={'root_id':rid,'scene':s['scene'],'official_episode_index':s['official_episode_index'],'official_episode_inclusive':s['official_episode_inclusive'],'frame_id':fid,'split':splits[rid],'source_kind':'new_official_frame','candidate_hints':[],'sample_roles':['metadata_stratified'],'source_path':str(E/'data/frames'/rid/f'episode_{fid:07d}.npz')}
    frames[fid]['candidate_hints'].append(hint)
  assert len(frames)<=30;inputs.extend(frames.values())
 # Four fixed descriptive retests per old episode, picked from already sealed outcomes.
 oldrows=read(SHORT/'results/results.json');retest=[]
 for rid in sorted(oldroots):
  rr=sorted([r for r in oldrows if r['root_id']==rid],key=lambda r:r['frame_id']);chosen=[]
  predicates=[('known_Holding_positive',lambda r:any(y and v for y,v in zip(r['candidate']['y'][:3],r['candidate']['v'][:3]))),('known_In_positive',lambda r:any(y and v for y,v in zip(r['candidate']['y'][3:],r['candidate']['v'][3:]))),('known_all_Holding_In_negative',lambda r:all(r['candidate']['v'])and not any(r['candidate']['y'])),('known_unknown_fact',lambda r:not all(r['candidate']['v']))]
  for role,predicate in predicates:
   available=[r for r in rr if r['frame_id']not in[x['frame_id']for x in chosen]and predicate(r)]
   if not available:available=[r for r in rr if r['frame_id']not in[x['frame_id']for x in chosen]];role+='__fallback_first_remaining_source_frame'
   r=available[0];item={k:r[k]for k in['root_id','scene','official_episode_index','official_episode_inclusive','frame_id','source_path','source_sha256']};item.update(split=splits[rid],source_kind='old_source_long_window_retest',candidate_hints=[],sample_roles=[role],old_step8_candidate=r['candidate']);chosen.append(item)
  retest.extend(chosen)
 assert len(retest)==16;inputs.extend(retest);assert len(inputs)<=256
 # Reuse the retained central directory. Do not download it again.
 F.E=OLD;ids={r['frame_id']for r in inputs if r['source_kind']=='new_official_frame'};members=[]
 for e in F.entries():
  n=Path(e['name']).name
  if '/training/'in e['name']and n.startswith('episode_')and n.endswith('.npz'):
   fid=int(n[8:-4])
   if fid in ids:members.append(e|{'frame_id':fid})
 assert len(members)==len(ids);used=read(E/'results/network_ledger.json')['body_bytes_read'];cost=sum(m['compressed_size']+30+65535 for m in members);assert used+cost<=F.BUDGET
 protocol={'seed':SEED,'selection':'Hash rank seed|episode selects4 eligible new episodes per scene. Eligibility requires each color lift_table and lift_drawer interval plus2 place_in_drawer intervals. Rank seed|segment chooses1 of each color/location and2 place intervals. Lift phase fractions0,1/3,2/3,1; place0,1/2,1; floor to frame, deduplicate without replacement. No outcome or model prediction is read for new sources.','episode_split':'Old4 stay train_candidate because inspected. New4 per scene: two smallest seed|split hashes validation_candidate, remainingtwo train_candidate. All original and derived records inherit root split; no frame leakage. C/D absent. No training authorized.','old_retest':'4 per original episode: first distinct source frame in prior H-positive, In-positive, all-six-negative, unknown classes; missing category uses first remaining frame. Uses sealed prior labels for descriptive change analysis only, not new episode selection.','max_runs':256,'new_frame_count':len(ids),'old_retest_count':16,'run_count':len(inputs),'max_physics_steps_per_run':96,'physics_dt_seconds':1/240,'image_checkpoints':[32,64,96],'stable_window_inclusive':[32,96],'final_candidate_step':96,'controller':'Same recorded absolute TCP xyz/euler and robot_obs[-1] reissued through native apply_action before steps1,9,...89;12 control periods,8 native physics ticks each; scene.step every8. No action trajectory replay. Joint/motor targets audited.','initialization':'fresh native world each source once; restore recorded state and zero unrecorded movable-object velocities; no env.reset implicit tick','strict_labels':'Frozen native Holding and In unchanged; same per-tick confidence .005/.015N and certain support normal .55 as previous resimulation','stable_labels':'Every tick32..96 must have same value as96, valid native label and unambiguous per-tick force evidence. Otherwise unknown.','force_stability':'Holding positive additionally requires coefficient of variation(population std/mean)<=0.25 for each finger-family TOTAL positive normal force across all65 ticks. In positive additionally requires at least one same certain contained object-to-drawer path at every tick, and CV<=0.25 of total certain upward force on each body-to-body edge across all65 ticks. Certain edges use force>=.015N,nz>=.55,gap<=.003. No force-variation constraint for negatives beyond all-tick confident negative evidence.','supplementary_table_state':'Diagnostic only: native table base link -1 supports object with certain upward contact at all65ticks and final Holding/In both reliable false; not a new learned predicate. Empty hand requires all3 Holding reliable false.','basic_coverage_check':'Descriptive target: per split and color Holding positive and negative each represented by>=2 distinct episodes; separately report In positive/negative and tabletop-negative episodes without claiming formal acceptance. No adaptive replenishment.','drift':'Record all object translations/rotations and TCP pose drift each step; report regardless of outcome. No drift-based posthoc filtering.','prohibited_after_init':['state teleport/reset','new constraint','changeDynamics','external force/torque'],'failure_policy':'Each frozen initialization once. Technical failure saves partial state and stops whole run without retry; physical failure/unstable/unknown is recorded and next frozen initialization continues.','max_total_ticks':len(inputs)*96,'formal_training':False,'original_RGB_labels_unchanged':True,'network_cumulative_cap_bytes':F.BUDGET,'network_used_before_new_frames':used,'compressed_new_frame_bytes':sum(m['compressed_size']for m in members),'conservative_remaining_download_bound':cost,'projected_cumulative_upper_bytes':used+cost}
 write(E/'configs/eligible_episodes.json',eligible);write(E/'configs/episode_split.json',{'source_episodes':12,'splits':splits,'old_source_frames_preserved':128,'new_source_episodes':sorted(new),'no_C_D':True});write(E/'configs/inputs.json',sorted(inputs,key=lambda r:(r['scene'],r['root_id'],r['frame_id'])));write(E/'configs/frame_members.json',members);write(E/'configs/protocol.json',protocol)
 shutil.copy2(__file__,E/'source'/Path(__file__).name)
 refs=[E/'configs'/f for f in['eligible_episodes.json','episode_split.json','inputs.json','frame_members.json','protocol.json']]+[E/'source'/Path(__file__).name,E/'source/official_language_safe_parse.py',E/'results/task_segments_A_B.json',E/'results/metadata_safe_summary.json',OLD/'configs/delivery_manifest.json',SHORT/'configs/delivery_manifest.json']
 write(E/'configs/frozen_selection.json',{'utc':F.now(),'before_new_frame_download_and_simulation':True,'sha256':{str(p):F.sha(p)for p in refs}})
 (R/'FEASIBILITY.md').write_text('# 元数据与划分可行性（采集前）\n\n安全解析17870条语言标注；A/B共12204条，跨episode/scene区间为0。构造器调用数0，嵌入数组未使用。新候选A有15、B有13个episode同时含三色桌面/屉内抓取和至少两段放入抽屉任务。\n\n固定选A/B各4个新episode，原4个保留训练候选；每scene的新episode各2训练候选、2验证候选，总8训练候选/4验证候选。所有帧与派生图按episode同属一个分割。语言任务仅用于分层候选，不作为GT。\n\n每个新episode最多30源帧：三色×桌面/抽屉各1任务段×4时点，另2段放入抽屉×3时点；重复帧去重不补。原4episode各4旧源初态单独长窗口复测，不增加独立来源数。总运行'+str(len(inputs))+'，新增源帧'+str(len(ids))+'。\n\n下载累计277.76MiB；新增帧压缩总量'+f"{protocol['compressed_new_frame_bytes']/2**20:.2f}"+'MiB，含最保守ZIP头预留的累计上界'+f"{protocol['projected_cumulative_upper_bytes']/2**20:.2f}"+'MiB，小于512MiB。冻结后不改选样补正例。\n\n96步共400ms，图像检查点32/64/96，稳定窗口32–96跨266.7ms。窗口每步谓词/有效性/力阈值均须通过；正例另外要求双指总力或固定承重路径各边总力的CV≤0.25。未知保留，物理失败不重试，技术失败保存后停止。\n')
 print('SELECTION_FROZEN',len(inputs),'runs;',len(ids),'new frames; split',splits,'cumulative_upper_MiB',(used+cost)/2**20,flush=True)
if __name__=='__main__':main()
