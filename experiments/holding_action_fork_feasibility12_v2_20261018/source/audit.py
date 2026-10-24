"""Read-only raw-data audit; never reruns physics or changes the frozen gate."""
from pathlib import Path
import json, hashlib, datetime, csv, collections, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

P=Path('/home/robot/research/projects/wm_symbolic')
NAME='holding_action_fork_feasibility12_v2_20261018'
E=P/'experiments'/NAME; R=P/'reports'/NAME; Q=E/'qualification'; D=Q/'private/runs'
OLD=P/'experiments/holding_action_fork_feasibility12_20261018'
O=OLD/'data/private/fork12_A_00/branches/fork12_A_00_aligned_lift'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def canonical(x):
 """Only reorder complete contact tuples; never change or drop a contact value."""
 if isinstance(x,dict):
  return {k:(sorted([canonical(a)for a in v],key=lambda a:json.dumps(a,sort_keys=True))if k=='contacts'else canonical(v))for k,v in x.items()}
 if isinstance(x,list):return [canonical(a)for a in x]
 return x
def write_csv(path,rows):
 with path.open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def max_abs(x,y):return float(np.max(np.abs(np.asarray(x,float)-np.asarray(y,float))))
def quat_angle_deg(x,y):
 x=np.asarray(x);y=np.asarray(y);x=x/np.linalg.norm(x);y=y/np.linalg.norm(y)
 return float(np.degrees(4*np.arctan2(min(np.linalg.norm(x-y),np.linalg.norm(x+y)),max(np.linalg.norm(x-y),np.linalg.norm(x+y)))))

def main():
 assert not (E/'results/attempts.json').exists()
 frozen=read(E/'configs/frozen.json');inputs=read(E/'configs/input_references.json')
 for p,h in {**frozen['sha256'],**inputs['sha256']}.items():assert sha(p)==h,p
 protocol=read(E/'configs/protocol.json');assert protocol['roots']==read(OLD/'configs/protocol.json')['roots']
 gate=read(Q/'results/QUALIFICATION.json');checks=read(Q/'results/full_history_checks.json');summary=read(Q/'results/execution_summary.json')
 assert not gate['passed'] and summary['execution_controls']==216 and summary['actual_ticks']==1730
 oldtrace=[json.loads(l)for l in (O/'control_trace_private.jsonl').read_text().splitlines()]
 assert len(oldtrace)==72
 runs={name:[read(D/name/f'c{k:02d}.json')for k in range(73)]for name in ['H1_reference72','L1_original_full72','H2_after_full_lift72']}
 h1,l1,h2=[runs[n]for n in runs]
 contact_rows=[];traj=[]
 for k in range(1,73):
  a,b=h1[k],h2[k]
  contact_rows.append({'control':k,'physical_exact':a['physical']==b['physical'],'python_raw_exact':a['python_state']==b['python_state'],'history_raw_exact':a['history']==b['history'],'python_contact_multiset_exact':canonical(a['python_state'])==canonical(b['python_state']),'history_contact_multiset_exact':canonical(a['history'])==canonical(b['history']),'action_exact':a['action']==b['action'],'cutoffs_exact':a['cutoffs']==b['cutoffs']})
  old,new=oldtrace[k-1],l1[k]
  pa=old['physical']['objects']['block_red'];pb=new['truth']['objects']['block_red']
  row={'control':k,'seconds':k/30,'issued_command_exact':all(old['action'][key]==new['action'][key]for key in ['target_xyz','target_quaternion','gripper']),'cutoffs_exact':old['holding_at_cutoffs']==new['cutoffs'],'old_red_holding':old['holding_at_cutoffs'][1][0],'replay_red_holding':new['cutoffs'][1][0],'red_position_gap_mm':float(np.linalg.norm(np.asarray(pa['position'])-pb['position'])*1000),'red_orientation_gap_deg':quat_angle_deg(pa['quaternion'],pb['quaternion']),'red_linear_velocity_max_abs_gap_m_s':max_abs(pa['linear_velocity'],pb['linear_velocity']),'red_angular_velocity_max_abs_gap_rad_s':max_abs(pa['angular_velocity'],pb['angular_velocity']),'old_red_z_m':pa['position'][2],'replay_red_z_m':pb['position'][2],'same_hold_red_position_gap_mm':float(np.linalg.norm(np.asarray(a['truth']['objects']['block_red']['position'])-b['truth']['objects']['block_red']['position'])*1000),'max_robot_penetration_m':new['max_robot_penetration_m'],'non_target_robot_contacts':len(new['non_target_robot_contacts'])}
  for color in ['blue','pink']:
   row[color+'_position_gap_mm']=float(np.linalg.norm(np.asarray(old['physical']['objects']['block_'+color]['position'])-new['truth']['objects']['block_'+color]['position'])*1000)
  traj.append(row)
 assert all(r['physical_exact']and r['python_contact_multiset_exact']and r['history_contact_multiset_exact']and r['action_exact']and r['cutoffs_exact']for r in contact_rows)
 assert all(r['issued_command_exact']and r['cutoffs_exact']for r in traj)
 write_csv(R/'same_hold_contact_order_audit.csv',contact_rows);write_csv(R/'original_lift_replay_per_control.csv',traj)
 imgs=[];anchors=[]
 for k in [0,12,24,36,60,72]:
  for cam in ['static','gripper']:
   with np.load(O/f's{k:02d}'/f'native_{cam}.npz',allow_pickle=False)as a,np.load(D/'L1_original_full72'/f'c{k:02d}_images'/f'native_{cam}.npz',allow_pickle=False)as b:
    for field in ['rgb','depth_buffer','mask']:
     x,y=a[field],b[field];diff=x!=y;px=np.any(diff,axis=2)if x.ndim==3 else diff
     imgs.append({'control':k,'camera':cam,'field':field,'height':x.shape[0],'width':x.shape[1],'exact':bool(np.array_equal(x,y)),'different_values':int(diff.sum()),'changed_pixels':int(px.sum()),'total_pixels':int(px.size),'changed_pixels_percent':float(100*px.mean()),'max_abs_value_difference':float(np.max(np.abs(x.astype(float)-y.astype(float))))})
  a=read(O/f's{k:02d}/physical_full.json');b=l1[k]['physical']; ar=a['bodies'][0];br=b['bodies'][0]
  anchors.append({'control':k,'arm_position_max_gap_rad':max(abs(ar['joint_states'][j][0]-br['joint_states'][j][0])for j in range(7)),'arm_velocity_max_gap_rad_s':max(abs(ar['joint_states'][j][1]-br['joint_states'][j][1])for j in range(7)),'arm_applied_torque_max_gap_Nm':max(abs(ar['joint_states'][j][3]-br['joint_states'][j][3])for j in range(7)),'finger_position_max_gap_m':max(abs(ar['joint_states'][j][0]-br['joint_states'][j][0])for j in [9,11]),'finger_velocity_max_gap_m_s':max(abs(ar['joint_states'][j][1]-br['joint_states'][j][1])for j in [9,11]),'gear_applied_force_max_gap':max_abs(a['constraints'][0]['applied_forces'],b['constraints'][0]['applied_forces']),'prior_output_caveat':k==0})
 write_csv(R/'original_lift_replay_images.csv',imgs);write_csv(R/'original_lift_replay_physics_anchors.csv',anchors)
 annotation=[]
 for name,states in runs.items():
  for k in [0,12,72]:
   state=states[k];hist=state['history'];cuts=state['cutoffs'];contacts=state['physical']['contacts']
   stable=all(all(h['holding_at_cutoffs'][t][0]==cuts[1][0]for t in range(3))and all(sum(v)<=1 for v in h['holding_at_cutoffs'])for h in hist)
   support=[c for c in contacts if {c[1],c[2]}=={4,5}and c[9]>.01]
   for cam in ['static','gripper']:
    with np.load(D/name/f'c{k:02d}_images'/f'native_{cam}.npz',allow_pickle=False)as z:seg=z['mask'];body=seg&((1<<24)-1);link=(seg>>24)-1
    obj=int(((seg>=0)&(body==4)).sum());fingers=[int(((seg>=0)&(body==0)&np.isin(link,f)).sum())for f in [[9,10],[11,12]]]
    annotation.append({'run':name,'control':k,'camera':cam,'target_color':'red','Holding':cuts[1][0],'Holding_three_cutoff_12control_stable':stable,'external_table_contact_above_0_01N':bool(support),'object_pixels':obj,'left_family_pixels':fingers[0],'right_family_pixels':fingers[1],'view_eligible':obj>=4 and all(x>=4 for x in fingers),'dataset_qualification':False})
 write_csv(R/'qualification_annotations_not_dataset.csv',annotation)
 grouped=[]
 for group in ['H1_reference72__H2_after_full_lift72','H1_reference72__L1_original_full72','original_full_lift']:
  selected=[c for c in checks if c['comparison'].startswith(group+'/')]
  fields=collections.defaultdict(list)
  for c in selected:
   name=c['comparison'].split('/')[-1]
   if name.startswith('first_tick'):name='first_control_physics_tick'
   fields[name].append(c)
  for field,items in fields.items():grouped.append({'group':group,'field':field,'comparisons':len(items),'passed':sum(c['equal']for c in items),'failed':sum(not c['equal']for c in items)})
 write_csv(R/'frozen_gate_counts.csv',grouped)
 # Extra offline c00 image and first-tick checks do not amend the frozen decision.
 extra=[]
 for other in ['L1_original_full72','H2_after_full_lift72']:
  for cam in ['static','gripper']:
   with np.load(D/'H1_reference72/c00_images'/f'native_{cam}.npz',allow_pickle=False)as a,np.load(D/other/'c00_images'/f'native_{cam}.npz',allow_pickle=False)as b:
    for field in ['rgb','depth_buffer','mask']:extra.append({'against':'H1_reference72','other':other,'control':0,'camera':cam,'field':field,'exact':bool(np.array_equal(a[field],b[field]))})
 assert all(c['exact']for c in extra)
 prior=['relation_GT_boxes_holding32_20261018','acquisition_scene_surface_pilot8_20261018','relation_GT_boxes36_ABC_20261018','holding_matched_input_diagnostic_20261018','holding_static_context_diagnostic_20261018','holding_action_fork_feasibility12_20261018','native_restore_semantics_root1_20261018','native_restore_semantics_root1_fix01_20261018','native_restore_semantics_root1_final_20261018']
 oldfiles={};prior_manifest=[]
 for name in prior:
  mp=P/'experiments'/name/'configs/delivery_manifest.json';m=read(mp);oldfiles.update(m['sha256']);oldfiles[str(mp)]=sha(mp);prior_manifest.append({'name':name,'path':str(mp),'sha256':sha(mp),'listed_files':len(m['sha256'])})
 for p,h in oldfiles.items():assert sha(p)==h,p
 guide=P/'source/世界模型与TS-LTL符号规划_核心预实验方案.md'
 assert sha(guide)=='f4cb908e26db79176263d5002290ac4c94daf3025ce2523fc546eb896e2ad0d4'
 assert not any((E/'data/public').rglob('*.npz'))
 audit={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_protocol_sha256':sha(E/'configs/protocol.json'),'frozen_gate':gate,'frozen_code_and_inputs_unchanged':True,'old_manifest_files_verified':len(oldfiles),'prior_manifests':prior_manifest,'guide_sha256':sha(guide),'guide_complete':False,'raw_failure_decomposition':{'same_hold_python_history_contact_order_only':144,'original_lift_physical_anchors':5,'original_lift_image_arrays':29},'posthoc_contact_multiset_audit':{'method':'sort complete contacts lists only, no values removed or rounded; raw evidence/gate unchanged','python_exact_after_sort':72,'history_exact_after_sort':72,'controls':72},'same_client_full72_evidence':{'H1_H2_physical_controls_exact':72,'H1_H2_fullfirstcontrol_ticks_exact':8,'H1_H2_image_arrays_exact':438,'H1_L1_commonprefix_physical_controls_exact':12,'H1_L1_fullfirstcontrol_ticks_exact':8,'H1_L1_image_arrays_exact':78,'fresh_motor_gear_contact_values_included':True,'start0_image_extra_checks':extra},'original_lift_replay':{'issued_commands_exact':72,'controls':72,'all_color_three_cutoff_labels_exact':72,'first_observed_red_position_difference_control':next(r['control']for r in traj if r['red_position_gap_mm']>0),'first_control_red_position_gap_mm':traj[0]['red_position_gap_mm'],'max_red_position_gap_mm':max(r['red_position_gap_mm']for r in traj),'max_gap_control':max(traj,key=lambda r:r['red_position_gap_mm'])['control'],'last_red_position_gap_mm':traj[-1]['red_position_gap_mm'],'max_red_orientation_gap_deg':max(r['red_orientation_gap_deg']for r in traj),'original_control1_physics_substeps_unrecorded':True,'all5_poststep_physics_anchors_differ':True,'poststep_RGB_arrays_differ':10,'poststep_depth_arrays_differ':10,'poststep_mask_arrays_differ':9,'causal_location':'difference is already present in common prefix control1, before lift-specific command13; not evidence full72 duration caused divergence','unresolved_cause':'archived reconstructed trajectory differs from original online trajectory; accessible start state and Python/commands are equal, but hidden solver/contact/controller process state and instrumentation cannot be distinguished from existing records; no isolated causal intervention performed','labels_do_not_rescue_strict_gate':True},'execution':summary,'pilot':{'status':'NOT_STARTED_QUALIFICATION_FAILED','same12_planned_root_ids_as_original':True,'planned_roots':12,'planned_target_colors':{'red':4,'blue':4,'pink':4},'planned_scenes':{'A':4,'B':4,'C':4},'planned_conditions':24,'planned_branches':48,'planned_saved_states_max':288,'actual_roots_attempted':0,'actual_conditions_attempted':0,'actual_branches':0,'actual_saved_states':0,'actual_new_distinct_layouts':0,'action_dependence':'NOT_ASSESSED','state_dependence':'NOT_ASSESSED','negative_coverage_claim':False,'train_dev_proposal':False},'qualification_counts':{'existing_roots':1,'root_id':'fork12_A_00','scene':'A','target_color':'red','runs_hold':2,'runs_lift':1,'controls':216,'execution_ticks':1728,'initialization_ticks':2,'total_ticks':1730,'saved_dual_camera_capture_states':len(list(Q.rglob('integrity.json'))),'saved_native_camera_npz':len(list(Q.rglob('native_*.npz'))),'physics_control_traces':216,'first_control_substep_traces':24,'no_dataset_use':True},'safety':{'max_robot_penetration_m':max(x['max_robot_penetration_m']for items in runs.values()for x in items[1:]),'unrequested_robot_contact_samples':sum(len(x['non_target_robot_contacts'])for items in runs.values()for x in items[1:]),'no_new_software_no_D_no_fits':True},'corrective_physics_runs':0,'gate_unchanged':True}
 save(R/'AUDIT.json',audit)
 audit['qualification_annotation_rows']=annotation
 save(R/'AUDIT.json',audit)
 save(E/'results/NOT_STARTED.json',{'utc':audit['utc'],'status':audit['pilot']['status'],'reason':'Frozen full72 qualification failed. No pilot roots, geometry conditions, branches, training, D access, replacements or additional physics after qualification. Posthoc contact sorting clarifies 144 bookkeeping failures but does not erase 34 independent original-replay failures.','coverage':'not assessed, not a negative coverage result','qualification':str(Q/'results/QUALIFICATION.json'),'audit':str(R/'AUDIT.json')})
 # Figure: static scientific evidence. Percent pixels uses any changed RGB channel.
 font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');matplotlib.rcParams['font.family']=font.get_name();matplotlib.rcParams['axes.unicode_minus']=False
 plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10})
 fig,ax=plt.subplots(2,2,figsize=(14,10),layout='constrained');blue='#2166ac';orange='#c65d14';purple='#7a3db0';teal='#007f7d'
 x=[0]+[r['control']for r in traj]
 ax[0,0].plot(x,[0]+[r['red_position_gap_mm']for r in traj],color=orange,lw=2,label='原始lift vs 文件恢复后重放lift')
 ax[0,0].plot(x,[0]+[r['same_hold_red_position_gap_mm']for r in traj],color=blue,lw=2,label='同进程H1 vs 完整lift后的H2')
 ax[0,0].axvspan(0,12,color='#dddddd',alpha=.6,label='预定共同hold前缀（1–12步）');ax[0,0].set(title='A  红色目标块的位置差异已出现在共同前缀',xlabel='动作控制步（每步8个物理tick）',ylabel='同一控制步的三维位置距离（mm）');ax[0,0].legend(fontsize=8);ax[0,0].grid(alpha=.2)
 for cam,color,label in [('static',purple,'固定相机'),('gripper',teal,'夹爪相机')]:
  rows=[r for r in imgs if r['camera']==cam and r['field']=='rgb'];ax[0,1].plot([r['control']for r in rows],[r['changed_pixels_percent']for r in rows],'o-',color=color,label=label)
 ax[0,1].set(title='B  原始lift与重放lift的RGB差异',xlabel='原实验保存图像的控制步',ylabel='至少一个RGB通道变化的像素占比（%）');ax[0,1].legend();ax[0,1].grid(alpha=.2)
 labels=['相同hold：完整物理状态','相同hold：原始Python/历史','相同hold：仅接触顺序排序后*','原始lift vs 重放：物理锚点'];yes=[72,0,144,0];no=[0,144,0,5];n=np.arange(4)
 ax[1,0].barh(n,yes,color=blue,label='完全相同');ax[1,0].barh(n,no,left=yes,color=orange,label='不完全相同');ax[1,0].set_yticks(n,labels);ax[1,0].invert_yaxis();ax[1,0].set_xlim(0,174)
 for i,(a,b)in enumerate(zip(yes,no)):ax[1,0].text(a+b+2,i,f'{a}/{a+b}相同',va='center',fontsize=9)
 ax[1,0].set(title='C  记录顺序问题与物理重放问题须分别解释',xlabel='比较次数（不同检查组的分母各自独立）');ax[1,0].legend(loc='lower right',fontsize=9)
 ax[1,1].axis('off');ax[1,1].text(0,1,'冻结结果：987项检查，178项未通过\n\n144项：接触点返回顺序造成的Python/历史记录差异\n  *只排序完整接触元组后144/144相同；原判定保留。\n34项：原始lift重放仍不一致\n  5个物理锚点 + 10 RGB + 10深度 + 9分割数组。\n\n实际：既有1根布局，3段×72步，共1730物理tick。\n12布局试采集：0根、0分支、0数据集状态；未启动。\n\n意义：同进程完整历史后的hold可复现；\n原始记录跨进程恢复后的逐步重放尚未获证。\n不能据此评估动作依赖、状态依赖或继续训练。',va='top',linespacing=1.5,fontsize=11)
 fig.suptitle('完整72步恢复核验：保留失败门槛，停止后续试采集',fontsize=17)
 fig.savefig(R/'full72_qualification.png',dpi=180);fig.savefig(R/'full72_qualification.pdf');plt.close(fig)
 print(json.dumps({'audit':str(R/'AUDIT.json'),'old_files':len(oldfiles),'first_mm':traj[0]['red_position_gap_mm'],'max_mm':audit['original_lift_replay']['max_red_position_gap_mm'],'max_control':audit['original_lift_replay']['max_gap_control'],'last_mm':traj[-1]['red_position_gap_mm'],'qualification_counts':audit['qualification_counts'],'safety':audit['safety']},ensure_ascii=False))

if __name__=='__main__':main()
