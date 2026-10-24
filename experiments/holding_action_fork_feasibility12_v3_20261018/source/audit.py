"""Offline audit of the stopped symmetric live-snapshot qualification."""
from pathlib import Path
import json,hashlib,datetime,csv,collections,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

P=Path('/home/robot/research/projects/wm_symbolic');NAME='holding_action_fork_feasibility12_v3_20261018'
E=P/'experiments'/NAME;Q=E/'qualification';R=P/'reports'/NAME;D=Q/'data/private/fork12_A_00/branches'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def enc(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def canon(x):
 if isinstance(x,dict):return {k:(sorted(v,key=enc)if k=='contacts'else canon(v))for k,v in x.items()}
 if isinstance(x,list):return [canon(v)for v in x]
 return x
def equal(a,b):return enc(canon(a))==enc(canon(b))
def csvwrite(p,rows):
 with p.open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def robot(phy):return next(x for x in phy['bodies']if x['uid']==0)
def red(phy):return next(x for x in phy['bodies']if x['uid']==4)
def metrics(a,b):
 ar,br=robot(a),robot(b);ao,bo=red(a),red(b)
 return {'red_position_gap_mm':float(np.linalg.norm(np.asarray(ao['base_pose'][0])-bo['base_pose'][0])*1000),'red_linear_velocity_max_gap_m_s':float(np.max(np.abs(np.asarray(ao['base_velocity'][0])-bo['base_velocity'][0]))),'arm_position_max_gap_rad':max(abs(ar['joint_states'][j][0]-br['joint_states'][j][0])for j in range(7)),'arm_applied_torque_max_gap_Nm':max(abs(ar['joint_states'][j][3]-br['joint_states'][j][3])for j in range(7)),'left_finger_position_gap_um':abs(ar['joint_states'][9][0]-br['joint_states'][9][0])*1e6,'right_finger_position_gap_um':abs(ar['joint_states'][11][0]-br['joint_states'][11][0])*1e6,'gear_force_max_gap':float(np.max(np.abs(np.asarray(a['constraints'][0]['applied_forces'])-b['constraints'][0]['applied_forces'])))}

def main():
 for cfg in ['frozen.json','input_references.json']:
  for p,h in read(E/'configs'/cfg)['sha256'].items():assert sha(p)==h,p
 gate=read(Q/'results/QUALIFICATION.json');summary=read(Q/'results/execution_summary.json');assert not gate['passed']
 assert summary['branch_controls']==145 and summary['preparation_controls']==220 and summary['actual_ticks']==2922
 assert not(E/'results/attempts.json').exists()and not any((E/'data/public').iterdir())
 branches={name:read(D/name/'branch_result.json')for name in ['H1','L1','L2']};assert not(D/'H2').exists()
 checks={name:read(D/name/'fidelity_checks.json')for name in branches};flat=[{'branch':name,**x}for name,items in checks.items()for x in items]
 assert len(flat)==17 and sum(not x['equal']for x in flat)==2
 sub={name:read(D/name/'first_control_8physics_ticks_private.json')for name in branches};rows=[]
 for pair in [('H1','L1'),('L1','L2')]:
  for i,(a,b)in enumerate(zip(sub[pair[0]],sub[pair[1]]),1):
   rows.append({'reference':pair[0],'actual':pair[1],'physics_tick':i,'milliseconds_since_control_start':i*1000/240,'full_physical_exact':equal(a['physical_full'],b['physical_full']),'python_exact':equal(a['python_state'],b['python_state']),'motor_commands_exact':equal(a['python_state']['motors'],b['python_state']['motors']),'cutoffs_exact':equal(a['cutoffs'],b['cutoffs']),**metrics(a['physical_full'],b['physical_full'])})
 csvwrite(R/'first_control_physics_ticks.csv',rows)
 assert all(x['full_physical_exact']for x in rows if x['reference']=='H1')
 assert all(not x['full_physical_exact']and x['motor_commands_exact']and x['cutoffs_exact']for x in rows if x['reference']=='L1')
 assert rows[8]['python_exact']
 fields=[]
 for field in sub['L1'][0]['python_state']:
  a=sub['L1'][0]['python_state'][field];b=sub['L2'][0]['python_state'][field];fields.append({'field':field,'exact_first_physics_tick':equal(a,b)})
 assert all(x['exact_first_physics_tick']for x in fields);csvwrite(R/'first_tick_controller_fields.csv',fields)
 arrays=[]
 for item in flat:
  obj=item.get('capture',item.get('first_tick_capture'))
  for x in obj['images']:arrays.append({'branch':item['branch'],'where':item['where'],**x})
 csvwrite(R/'all_compared_image_arrays.csv',arrays);assert len(arrays)==102 and sum(not x['equal']for x in arrays)==11
 image_stats=[]
 for anchor,folder in [('start0','s00'),('first_physics_tick','first_physics_tick_images'),('control1_end','s01')]:
  for cam in ['static','gripper']:
   with np.load(D/'L1'/folder/f'native_{cam}.npz',allow_pickle=False)as a,np.load(D/'L2'/folder/f'native_{cam}.npz',allow_pickle=False)as b:
    for key in ['rgb','depth_buffer','mask']:
     diff=a[key]!=b[key];pixels=np.any(diff,axis=2)if diff.ndim==3 else diff;image_stats.append({'anchor':anchor,'camera':cam,'field':key,'exact':bool(not diff.any()),'unequal_values':int(diff.sum()),'unequal_pixels':int(pixels.sum()),'total_pixels':int(pixels.size),'unequal_pixels_percent':float(100*pixels.mean())})
 csvwrite(R/'L1_L2_images.csv',image_stats)
 starts=[]
 for name,b in branches.items():
  c=b['restore'];starts.append({'branch':name,'core_exact':c['physical_core']['equal'],'Python_exact':c['python']['equal'],'motor_reissue_has_no_physical_effect':c['motor_reissue_physical_effect']['equal'],'full_with_prior_outputs_exact':c['physical']['equal'],'prior_output_differences':c['prior_reported_forces']['different_scalars_or_structures'],'status':b['status'],'branch_controls':b['executed_controls'],'saved_control_states':b['saved_states']})
 assert all(x['core_exact']and x['Python_exact']and x['motor_reissue_has_no_physical_effect']for x in starts);csvwrite(R/'start_and_run_counts.csv',starts)
 # Independently recheck common12 from complete persisted control records/images.
 common=[]
 for k in range(1,13):
  a=read(D/'H1'/f'c{k:02d}.json');b=read(D/'L1'/f'c{k:02d}.json');row={'control':k,'full_record_exact':equal(a,b)}
  for cam in ['static','gripper']:
   with np.load(D/'H1'/f's{k:02d}'/f'native_{cam}.npz',allow_pickle=False)as x,np.load(D/'L1'/f's{k:02d}'/f'native_{cam}.npz',allow_pickle=False)as y:
    for key in ['rgb','depth_buffer','mask']:row[cam+'_'+key+'_exact']=bool(np.array_equal(x[key],y[key]))
  assert all(v for key,v in row.items()if key!='control');common.append(row)
 csvwrite(R/'H1_L1_common12.csv',common)
 # Actual command inputs at L2 control1 exactly match L1, including IK motor output.
 a=read(D/'L1/c01.json');b=read(D/'L2/c01.json');assert equal(a['action'],b['action'])
 motors=sub['L1'][0]['python_state']['motors'];actual=[v for v in motors.values()if v['kwargs'].get('physicsClientId',0)==0];assert len(actual)==13
 known=[]
 for name in branches:
  for k in [0,12,72]:
   f=Q/'data/public'/f'{name}_s{k:02d}.npz'
   if not f.exists():continue
   truth=read(D/name/f's{k:02d}/truth_private.json');known.append({'branch':name,'control':k,'target_Holding':truth['cutoffs'][1][0],'target_physical_stable':truth['stable'][0],'completed_program':branches[name]['status']=='completed','dataset_eligible':False})
 csvwrite(R/'diagnostic_labels_not_dataset.csv',known)
 prior=['relation_GT_boxes_holding32_20261018','acquisition_scene_surface_pilot8_20261018','relation_GT_boxes36_ABC_20261018','holding_matched_input_diagnostic_20261018','holding_static_context_diagnostic_20261018','holding_action_fork_feasibility12_20261018','native_restore_semantics_root1_20261018','native_restore_semantics_root1_fix01_20261018','native_restore_semantics_root1_final_20261018','holding_action_fork_feasibility12_v2_20261018'];oldfiles={};manifests=[]
 for n in prior:
  f=P/'experiments'/n/'configs/delivery_manifest.json';m=read(f);oldfiles.update(m['sha256']);oldfiles[str(f)]=sha(f);manifests.append({'name':n,'sha256':sha(f)})
 for p,h in oldfiles.items():assert sha(p)==h,p
 guide=P/'source/世界模型与TS-LTL符号规划_核心预实验方案.md';assert sha(guide)=='f4cb908e26db79176263d5002290ac4c94daf3025ce2523fc546eb896e2ad0d4'
 n_npz=0
 for f in Q.rglob('native_*.npz'):
  with np.load(f,allow_pickle=False)as z:assert set(z.files)=={'rgb','depth_buffer','mask'}and z['rgb'].dtype==np.uint8 and np.isfinite(z['depth_buffer']).all()
  n_npz+=1
 assert n_npz==302
 source=(E/'source/collect.py').read_text();assert 'restoreState(fileName'not in source and source.count('restoreState(stateId=')==1
 audit={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'QUALIFICATION_FAILED_PILOT_NOT_STARTED','estimand':read(E/'configs/protocol.json')['estimand'],'protocol_sha256':sha(E/'configs/protocol.json'),'original_gate_unchanged':True,'preflight':read(E/'configs/preflight.json'),'execution':summary,'runs':starts,'qualification_composite_checks':{'attempted':17,'passed':15,'failed':2,'note':'one control-end comparison and one eight-substep/first-image comparison fail; these are not independent experiments'},'image_array_checks':{'attempted':102,'passed':91,'failed':11},'H1_L1_common12_verified':True,'H1_L1_first8_substeps_verified':True,'L1_L2_starts':{'core_Python_cutoffs_dualimages_exact':True,'previous_step_readouts_separately_preserved':True},'first_impactful_error':{'where':'first actual physics tick of L2 control1 compared with L1 control1','time_ms':1000/240,'same_semantic_action':True,'same_first_tick_python_all_fields':True,'same_13_motor_commands_including_computed_IK_targets':True,'red_position_gap_mm':rows[8]['red_position_gap_mm'],'red_position_gap_control_end_mm':rows[-1]['red_position_gap_mm'],'fresh_arm_torque_max_gap_Nm':rows[8]['arm_applied_torque_max_gap_Nm'],'not_contact_permutation':True,'not_only_stale_force_readback':True,'causal_boundary':'recorded restored core and control/motor inputs match, first physics response differs; unobserved internal dynamics state is not localized by this run','no_claim_of_specific_solver_cause':True},'H1_H2_full72':'NOT_REACHED','L1_L2_full72':'FAILED_AT_CONTROL1','new_physics_corrections':0,'pilot':{'planned_roots':12,'planned_color_counts':{'red':4,'blue':4,'pink':4},'planned_conditions':24,'planned_branches_max':48,'planned_saved_states_max':288,'roots_attempted':0,'conditions_attempted':0,'branches':0,'saved_states':0,'action_dependence':'NOT_ASSESSED','state_dependence':'NOT_ASSESSED','negative_coverage_claim':False,'training':0},'saved_qualification_only':{'control_anchor_states':148,'extra_first_tick_states':3,'total_capture_states':151,'native_camera_npz':302,'control_records':145,'physics_substep_records':24,'no_training_use':True},'old_unique_files_verified':len(oldfiles),'old_manifests':manifests,'guide_complete':False,'all_frozen_hashes_unchanged':True,'no_D_no_training_no_downloads_no_parameter_changes':True,'source_has_no_physics_file_reload':True,'safety':{'max_branch_robot_penetration_m':max(b['safety']['max_robot_penetration_m']for b in branches.values()),'non_target_robot_contact_samples':sum(b['safety']['unrequested_robot_contact_samples']for b in branches.values())}}
 save(R/'AUDIT.json',audit);save(E/'results/NOT_STARTED.json',{'utc':audit['utc'],'status':audit['status'],'reason':'Live symmetric qualification failed at L2 first control; H2 and pilot not run. No additional physics permitted in this frozen attempt.','coverage':'not assessed','audit':str(R/'AUDIT.json')})
 # All panels explain actual observed quantities; no projected unrun trajectories.
 font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');matplotlib.rcParams['font.family']=font.get_name();matplotlib.rcParams['axes.unicode_minus']=False;plt.rcParams.update({'font.size':10,'axes.titlesize':12})
 fig,ax=plt.subplots(2,2,figsize=(14,9),layout='constrained');blue='#2166ac';orange='#c65d14';teal='#007f7d';purple='#7a3db0'
 for ref,label,color in [('H1','H1 vs L1：相同前缀',blue),('L1','L1 vs L2：重复lift',orange)]:
  rs=[r for r in rows if r['reference']==ref];ax[0,0].plot([0]+[r['physics_tick']for r in rs],[0]+[r['red_position_gap_mm']*1000 for r in rs],'o-',color=color,label=label)
 ax[0,0].set(title='A  起点相同，首次物理推进后出现差异',xlabel='首个控制步内的物理tick（每tick 4.167ms）',ylabel='红色目标块三维位置差（μm）');ax[0,0].legend();ax[0,0].grid(alpha=.2)
 rs=rows[8:];ax[0,1].plot([r['physics_tick']for r in rs],[r['arm_applied_torque_max_gap_Nm']for r in rs],'o-',color=orange,label='L1 vs L2：7臂关节中的最大值');ax[0,1].set(title='B  首tick后更新的电机力矩也不同',xlabel='首个控制步内的物理tick',ylabel='已更新电机力矩的最大绝对差（N·m）');ax[0,1].legend(fontsize=9);ax[0,1].grid(alpha=.2)
 anchors=['start0','first_physics_tick','control1_end'];x=np.arange(3)
 for j,(cam,color,label)in enumerate([('static',purple,'固定相机'),('gripper',teal,'夹爪相机')]):
  vals=[next(r['unequal_pixels_percent']for r in image_stats if r['anchor']==a and r['camera']==cam and r['field']=='rgb')for a in anchors];ax[1,0].bar(x+(j-.5)*.32,vals,width=.32,color=color,label=label)
 ax[1,0].set_xticks(x,['恢复起点','首物理tick','第1控制步末端']);ax[1,0].set(title='C  同一程序L1与L2的RGB比较',ylabel='至少一个RGB通道不同的像素占比（%）');ax[1,0].legend();ax[1,0].grid(axis='y',alpha=.2)
 ax[1,1].axis('off');ax[1,1].text(0,1,'资格结果：未通过，立即停止\n\nH1：72/72步完成\nL1：72/72步完成；与H1共同12步通过\nL2：第1步发现差异，停止于1/72步\nH2：未执行\n\n3次实时快照恢复均通过起点检查。\nL1/L2首tick的Python状态与13条电机命令相同。\n变化已包含位置与更新力值，无法由接触顺序解释。\n\n实际：220准备步 + 145动作步，2922物理tick。\n后续12布局采集：0布局、0分支、0状态。\n动作依赖与状态依赖：尚未评估。',va='top',fontsize=11,linespacing=1.5)
 fig.suptitle('同进程实时快照的对称恢复核验：首个物理tick未能精确重放',fontsize=16);fig.savefig(R/'symmetric_live_restore.png',dpi=180);fig.savefig(R/'symmetric_live_restore.pdf');plt.close(fig)
 print(json.dumps({'audit':str(R/'AUDIT.json'),'old_unique_files':len(oldfiles),'first_tick':rows[8],'last_tick':rows[-1],'images':image_stats,'safety':audit['safety']},ensure_ascii=False))

if __name__=='__main__':main()
