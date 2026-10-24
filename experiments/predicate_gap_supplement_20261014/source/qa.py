from pathlib import Path
import argparse,json,hashlib,gzip,collections,subprocess
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'data/predicate_gap_supplement_20261014';R=P/'experiments/predicate_gap_supplement_20261014'
C=['red','blue','pink'];NAMES={'far_open_negative':'远离／张开／未拿住','near_open_negative':'靠近／张开／未拿住','contact_supported_close_command_negative':'接触且受支撑／闭合指令／未拿住','held_positive':'真实拿住','far_closed_negative':'远离／闭合／未拿住','near_closed_negative':'靠近／闭合／未拿住','near_closed_motion_negative':'移动后仍靠近／闭合／未拿住'}
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['pilot','final'],required=True);args=a.parse_args();m=json.loads((B/'data_manifest.json').read_text());protocol=json.loads((R/'protocol.json').read_text());issues=[];root_splits={};traj_splits={};window_splits={};frame_splits={};counts=collections.Counter();comb=collections.Counter();targets=collections.Counter();nearroots=set();heldroots=set();label_counts=collections.Counter();image_records=[];videos=0;neg_visibility=0;anchors=[]
 for t in m['trajectories']:
  if t['label_issues']:issues+=t['label_issues']
  if not t['physics_guard_passed']:issues.append([t['trajectory_id'],'physics_guard'])
  if max(t['root_replay'].values())>1e-5:issues.append([t['trajectory_id'],'root_replay'])
  if t['controller_sha256']!=protocol['source_sha256'][str(P/'src/wm_symbolic/physical_oracle.py')]:issues.append([t['trajectory_id'],'controller_changed'])
  for path in t['public_video_files']:
   reader=imageio.get_reader(path);count=sum(1 for frame in reader);reader.close()
   if count!=t['frames']:issues.append([path,'frame_count_mismatch',count,t['frames']])
   videos+=1
 for r in m['records']:
  root_splits.setdefault(r['root_id'],set()).add(r['split']);traj_splits.setdefault(r['trajectory_id'],set()).add(r['split']);window_splits.setdefault(r['window_rgb_sha256'],set()).add(r['split'])
  for h in r['frame_rgb_sha256']:frame_splits.setdefault(h,set()).add(r['split'])
  lab=json.loads(Path(r['labels_private']).read_text());frames=lab['frames'];j=C.index(r['target_color']);last=frames[-1];near=last['objects_audit'][r['target_color']]['tcp_distance_m']<.075;drawer=last['labels']['categorical'][3];isheld=last['objects_audit'][r['target_color']]['held_contact_audit'];counts[(r['split'],r['condition'])]+=1;comb[(r['split'],r['target_color'],isheld,drawer)]+=1;targets[(r['split'],'positive' if isheld else 'negative')]+=1
  if r['condition']=='held_positive':heldroots.add(r['root_id'])
  if r['condition']=='near_closed_negative':nearroots.add(r['root_id'])
  for ci,color in enumerate(C):
   ok=last['labels']['categorical_valid'][ci];held=last['labels']['holding_derived'][ci];visible=any(n>=4 for n in last['objects_audit'][color]['visible_pixels'].values())
   label_counts[(r['split'],color,'held_positive' if held and ok else 'held_negative_visible' if ok and visible else 'not_visible_or_invalid')]+=1
  for ix,path in enumerate(r['public_frames']):
   with np.load(path) as z:
    forbidden=set(z.files)-{'rgb_static','rgb_gripper','depth_static','depth_gripper','robot_obs','seg_static','seg_gripper'}-{k for k in z.files if k.startswith('calib_')}
    if forbidden:issues.append([path,'private_fields_in_public',sorted(forbidden)])
    if any(not np.isfinite(z[k]).all() for k in ['depth_static','depth_gripper','robot_obs']):issues.append([path,'nonfinite'])
    h=hashlib.sha256(z['rgb_static'].tobytes()+z['rgb_gripper'].tobytes()).hexdigest()
    if h!=r['frame_rgb_sha256'][ix]:issues.append([path,'image_hash'])
   t=frames[ix];obj=t['objects_audit'][r['target_color']]
   if not isheld:
    if not any(v>=4 for v in obj['visible_pixels'].values()):issues.append([r['sample_id'],'invisible_target_as_negative'])
    else:neg_visibility+=1
   if obj['held_contact_audit']!=t['labels']['holding_derived'][j]:issues.append([r['sample_id'],'label_contact_conflict'])
  if isheld and not r['motion_verified']:issues.append([r['sample_id'],'unverified_held_follow'])
  if r['condition'].startswith('near_closed') and not r['motion_verified']:issues.append([r['sample_id'],'unverified_negative_motion'])
  anchors.append({'sample_id':r['sample_id'],'root':r['root_id'],'condition':r['condition'],'split':r['split'],'target':r['target_color'],'held':isheld,'drawer_category':drawer,'tcp_distance_mm':1000*last['objects_audit'][r['target_color']]['tcp_distance_m'],'width_mm':1000*last['gripper_width_m'],'finger_bilateral':last['objects_audit'][r['target_color']]['bilateral_finger_force'],'support_contacts':last['objects_audit'][r['target_color']]['external_support_contacts'],'visible':last['objects_audit'][r['target_color']]['visible_pixels'],'drawer_m':last['labels']['continuous']['drawer_m']})
 for name,lookup in [('root',root_splits),('trajectory',traj_splits),('window',window_splits),('frame',frame_splits)]:
  for key,ss in lookup.items():
   if len(ss)>1:issues.append(['cross_split_'+name,key,list(ss)])
 hashes_ok={path:sha(path)==h for path,h in protocol['source_sha256'].items()};hashes_ok['old_manifest']=sha(P/'data/core1200_v1/data_manifest.json')==protocol['old_manifest_sha256'];hashes_ok['old_world_model']=sha(P/'runs/core1200_v1/training_main/best.pt')==protocol['old_model_sha256'];hashes_ok['old_reader']=sha(P/'experiments/state_predicates_20261014/models/direct_predicates_seed0.pt')==protocol['prior_reader_model_sha256']
 if not all(hashes_ok.values()):issues.append(['frozen_artifacts_changed',hashes_ok])
 # Visible contact sheet: balanced actual positive/near-negative pairs, only already collected rows.
 font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
 examples=[]
 for color in C:
  for cond in ['held_positive','near_closed_negative']:
   pool=[r for r in m['records'] if r['target_color']==color and r['condition']==cond]
   closed=[r for r in pool if r['planned_drawer']=='closed'];choice=(closed or pool)
   if choice:examples.append(choice[0])
 fig,axes=plt.subplots(len(examples),3,figsize=(12,3.1*len(examples)),gridspec_kw={'width_ratios':[1,1,1.4]},layout='constrained',squeeze=False)
 for k,r in enumerate(examples):
  z=np.load(r['public_frames'][-1]);lab=json.loads(Path(r['labels_private']).read_text());a=next(x for x in anchors if x['sample_id']==r['sample_id'])
  for c,cam in enumerate(['static','gripper']):axes[k,c].imshow(z['rgb_'+cam]);axes[k,c].axis('off');axes[k,c].set_title('静态相机' if c==0 else '夹爪相机',fontsize=11)
  ax=axes[k,2];ax.axis('off');ax.text(0,.95,f"{r['target_color']}：{NAMES[r['condition']]}",fontsize=13,color='#28784B' if a['held'] else '#A85A17',va='top',fontweight='bold');text=f"{r['root_id']}\n{r['split']} · 实际 drawer={a['drawer_m']:.4f}m\n\nheld 真值：{a['held']}\n双侧受力接触：{a['finger_bilateral']}\n外部支撑接触数：{a['support_contacts']}\n距夹爪：{a['tcp_distance_mm']:.1f}mm\n夹爪开度：{a['width_mm']:.1f}mm\n真实运动核验：{r['motion_verified']}";ax.text(0,.80,text,fontsize=10.7,va='top',linespacing=1.5)
 fig.suptitle('补采真实图像：绿色为真实拿住，棕色为闭合夹爪靠近但未拿住；不是模型预测',fontsize=13);fig.savefig(R/'figures'/f'{args.phase}_contact_sheet.png',dpi=135);plt.close(fig)
 report={'phase':args.phase,'technical_pass':not issues,'approved_for_completion':False,'visual_review':'pending' if args.phase=='pilot' else 'pending_final','root_count':len(root_splits),'trajectory_count':len(m['trajectories']),'window_count':len(m['records']),'history_frames_per_window':4,'held_positive_roots':len(heldroots),'near_closed_negative_roots':len(nearroots),'counts':[{'split':k[0],'condition':k[1],'windows':v} for k,v in sorted(counts.items())],'target_polarity':[{'split':k[0],'polarity':k[1],'windows':v} for k,v in sorted(targets.items())],'state_combinations':[{'split':k[0],'color':k[1],'held':k[2],'drawer_category':k[3],'windows':v} for k,v in sorted(comb.items())],'all_entity_visibility_and_labels':[{'split':k[0],'color':k[1],'status':k[2],'count':v} for k,v in sorted(label_counts.items())],'cross_split_root_or_trajectory_or_rgb_duplicates':any(isinstance(x,list) and str(x[0]).startswith('cross_split') for x in issues),'unique_window_rgb_hashes':len(window_splits),'unique_frame_rgb_hashes':len(frame_splits),'target_negative_visible_history_frames':neg_visibility,'videos_fully_decoded_frame_count_checked':videos,'issues':issues,'preserved_source_and_old_model_hashes':hashes_ok,'pilot_minimum_families_pass':len(heldroots)>=2 and len(nearroots)>=2,'gaps':[{'trajectory':t['trajectory_id'],**gap} for t in m['trajectories'] for gap in t['gaps']],'no_training_or_model_evaluation':True,'no_upload':True}
 save(R/'results'/f'{args.phase}_qa.json',report);save(R/'results'/f'{args.phase}_anchors.json',anchors);save(R/'results'/f'{args.phase}_examples.json',examples)
 print(json.dumps({k:report[k] for k in ['technical_pass','root_count','trajectory_count','window_count','held_positive_roots','near_closed_negative_roots','issues','gaps']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
