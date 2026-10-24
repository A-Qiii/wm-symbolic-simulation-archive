from common import *
import copy,csv,collections,shutil,xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def differences(a,b,path=''):
 out=[]
 if isinstance(a,dict)and isinstance(b,dict):
  assert set(a)==set(b)
  for k in sorted(a):out+=differences(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,list)and isinstance(b,list):
  if len(a)!=len(b):return [{'path':path,'reference':len(a),'actual':len(b),'delta':None}]
  for k,(x,y)in enumerate(zip(a,b)):out+=differences(x,y,path+'/'+str(k))
 elif canonical(a)!=canonical(b):out.append({'path':path,'reference':a,'actual':b,'delta':abs(float(a)-float(b))if isinstance(a,(float,int))and isinstance(b,(float,int))else None})
 return out
def write_csv(path,rows):
 with Path(path).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def angular_matrix(q):
 x,y,z,w=q
 return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],[2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],[2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])
def main():
 D=E/'private/runs';summary=read(E/'results/execution_summary.json');assert summary['runs_completed']==6 and summary['execution_controls']==144 and summary['stop']is None
 specs=read(E/'configs/protocol.json')['runs'];pairs=[('H1_reference','H2_after_same_H',24),('H1_reference','H3_after_L1',24),('H1_reference','H4_after_L2',24),('L1_short_lift','L2_short_lift',24),('H1_reference','L1_short_lift',12),('H1_reference','L2_short_lift',12)]
 states={s['name']:[read(D/s['name']/f'c{k:02d}.json')for k in range(25)]for s in specs};metric=[];firstticks=[];images=[];start=[];rawdiff=[]
 def compare_images(a,b,tag):
  for cam in ['static','gripper']:
   with np.load(a/f'native_{cam}.npz',allow_pickle=False)as x,np.load(b/f'native_{cam}.npz',allow_pickle=False)as y:
    for field in ['rgb','depth_buffer','mask']:
     images.append({'comparison':tag,'camera':cam,'field':field,'equal':np.array_equal(x[field],y[field]),'different_values':int(np.sum(x[field]!=y[field]))})
 for a,b,n in pairs:
  for k in range(n+1):
   x,y=states[a][k],states[b][k];diff=differences(x['physical'],y['physical']);rawdiff+=[{'comparison':a+'__'+b,'control':k,**r}for r in diff];record={'reference':a,'compared':b,'control':k,'different_full_physical_fields':len(diff),'max_base_xyz_difference_m':max(abs(xb['base_pose'][0][j]-yb['base_pose'][0][j])for xb,yb in zip(x['physical']['bodies'],y['physical']['bodies'])for j in range(3)),'both_finger_full_joint_states_exact':all(canonical(x['physical']['bodies'][0]['joint_states'][j])==canonical(y['physical']['bodies'][0]['joint_states'][j])for j in[9,11]),'contacts_exact':canonical(x['physical']['contacts'])==canonical(y['physical']['contacts']),'cutoff_labels_exact':canonical(x['cutoffs'])==canonical(y['cutoffs']),'history_exact':canonical(x['history'])==canonical(y['history']),'python_exact':canonical(x['python_state'])==canonical(y['python_state'])}
   if k:
    record['actions_exact']=canonical(x['action'])==canonical(y['action']);metric.append(record)
   else:start.append(record)
   compare_images(D/a/f'c{k:02d}_images',D/b/f'c{k:02d}_images',f'{a}__{b}/c{k:02d}')
  for tick in range(1,9):
   x=read(D/a/'first_control_ticks'/f'tick{tick:02d}.json');y=read(D/b/'first_control_ticks'/f'tick{tick:02d}.json');dd=differences(x['physical'],y['physical']);firstticks.append({'reference':a,'compared':b,'physics_tick':tick,'different_full_physical_fields':len(dd),'cutoff_labels_exact':x['cutoffs']==y['cutoffs']})
  compare_images(D/a/'first_physics_tick_images',D/b/'first_physics_tick_images',f'{a}__{b}/first_physics_tick')
 assert len(metric)==120 and len(firstticks)==48 and len(images)==792
 assert all(r['different_full_physical_fields']==0 and all(r[k]for k in ['both_finger_full_joint_states_exact','contacts_exact','cutoff_labels_exact','history_exact','python_exact','actions_exact'])for r in metric)
 assert all(r['different_full_physical_fields']==0 and r['cutoff_labels_exact']for r in firstticks);assert all(r['equal']for r in images)
 write_csv(R/'control_comparisons.csv',metric);write_csv(R/'first_physics_ticks.csv',firstticks);write_csv(R/'image_array_comparisons.csv',images);write_csv(R/'pre_action_comparisons.csv',start);save(R/'retained_pre_action_differences.json',rawdiff)
 full_ticks=[]
 for s in specs:
  name=s['name'];a=states[name][0]['physical'];ref=states['H1_reference'][0]['physical'];full_ticks.append({'run':name,'tick':0,'max_arm_torque_difference_Nm':max(abs(a['bodies'][0]['joint_states'][j][3]-ref['bodies'][0]['joint_states'][j][3])for j in range(7)),'max_finger_motor_force_difference_N':max(abs(a['bodies'][0]['joint_states'][j][3]-ref['bodies'][0]['joint_states'][j][3])for j in [9,11]),'constraint_readback_difference':abs(a['constraints'][0]['applied_forces'][0]-ref['constraints'][0]['applied_forces'][0])})
  for tick in range(1,9):
   a=read(D/name/'first_control_ticks'/f'tick{tick:02d}.json')['physical'];ref=read(D/'H1_reference/first_control_ticks'/f'tick{tick:02d}.json')['physical'];full_ticks.append({'run':name,'tick':tick,'max_arm_torque_difference_Nm':max(abs(a['bodies'][0]['joint_states'][j][3]-ref['bodies'][0]['joint_states'][j][3])for j in range(7)),'max_finger_motor_force_difference_N':max(abs(a['bodies'][0]['joint_states'][j][3]-ref['bodies'][0]['joint_states'][j][3])for j in[9,11]),'constraint_readback_difference':abs(a['constraints'][0]['applied_forces'][0]-ref['constraints'][0]['applied_forces'][0])})
 write_csv(R/'readback_refresh_by_physics_tick.csv',full_ticks)
 # Independent annotation audit, all original fixed cutoffs retained.
 labelrows=[]
 for spec in specs:
  name=spec['name']
  for k,x in enumerate(states[name]):
   phy=x['physical'];cs=phy['contacts'];holding=[]
   for threshold in[.005,.01,.015]:
    colors=[]
    for color,uid in [('red',4),('blue',3),('pink',2)]:
     rel=[c for c in cs if uid in[c[1],c[2]]and c[9]>threshold];fingerlinks=set();external=False
     for c in rel:
      other=c[2]if c[1]==uid else c[1];link=c[4]if c[1]==uid else c[3]
      if other==0:fingerlinks.add(link)
      else:external=True
     colors.append(bool(fingerlinks&{9,10}and fingerlinks&{11,12}and not external))
    holding.append(colors)
   assert holding==x['cutoffs'];stable=all(all(h['holding_at_cutoffs'][t][0]==holding[1][0]for t in range(3))for h in x['history']);support=[c for c in cs if c[1:5]==[4,5,-1,-1]and c[9]>.01 and c[7][2]>.5];labelrows.append({'run':name,'control':k,'red_Holding':holding[1][0],'red_12control_stable':stable,'red_supported':bool(support),'table_upward_force_N':sum(c[9]*c[7][2]for c in support),'max_robot_penetration_m':x.get('max_robot_penetration_m',None)})
 write_csv(R/'labels_per_control.csv',labelrows)
 # Read-only causal-stage inspection of previous preparation; native box geometry.
 original=[json.loads(line)for line in(OLDROOT/'preparation_trace_private.jsonl').read_text().splitlines()];block_urdf=P/'vendor/calvin/calvin_env/data/blocks/block_red_middle.urdf';xml=ET.parse(block_urdf);size=np.fromstring(xml.find('.//collision/geometry/box').attrib['size'],sep=' ')*.8;mass=float(xml.find('.//inertial/mass').attrib['value']);base=original[199]['physical']['objects']['block_red'];z0=base['position'][2];tabletop=float(np.median([c[6][2]for c in original[199]['physical']['contacts']if c[1:5]==[4,5,-1,-1]and c[9]>.01]));preparation=[]
 for row in original:
  if row['logical_control']<190:continue
  ob=row['physical']['objects']['block_red'];rot=angular_matrix(ob['quaternion']);bottom=ob['position'][2]-float(np.abs(rot[2])@(size/2));forces={'table':0.,'leftpad':0.,'rightpad':0.};fric=0.;normal=0.;dist=[]
  for c in row['physical']['contacts']:
   if 4 not in[c[1],c[2]]:continue
   sg=1 if c[1]==4 else-1;other=c[2]if sg==1 else c[1];link=c[4]if sg==1 else c[3];nz=sg*c[9]*c[7][2];fz=sg*(c[10]*c[11][2]+c[12]*c[13][2]);group='table'if other==5 and link==-1 else'leftpad'if other==0 and link==10 else'rightpad'if other==0 and link==12 else None
   if group:forces[group]+=nz+fz
   if other==0:fric+=fz;normal+=nz
   if other==5 and link==-1:dist.append(c[8])
  preparation.append({'control':row['logical_control'],'stage':row['action']['phase'],'close_control':max(0,row['logical_control']-200),'center_z_m':ob['position'][2],'center_rise_since_close_start_mm':1000*(ob['position'][2]-z0),'native_box_bottom_z_m':bottom,'box_bottom_clearance_above_table_mm':1000*(bottom-tabletop),'table_vertical_contact_force_N':forces['table'],'leftpad_vertical_contact_force_N':forces['leftpad'],'rightpad_vertical_contact_force_N':forces['rightpad'],'pad_friction_vertical_sum_N':fric,'pad_normal_vertical_sum_N':normal,'Holding':'block_red'in row['physical']['held'],'contact_table_min_distance_m':min(dist)if dist else None,'command_z_m':row['action']['target_xyz'][2]})
 write_csv(R/'original_preparation_stage_analysis.csv',preparation);last=preparation[-1];first=next(r for r in preparation if r['Holding']);geom={'native_box_dimensions_m':size.tolist(),'native_mass_kg':mass,'tabletop_z_m_from_contact_points':tabletop,'before_close_center_z_m':z0,'first_Holding_control':first['control'],'last':last,'planned_grasp_z_m':read(OLDROOT/'aligned_plan_private.json')['condition_waypoints'][2][0][2],'close_hold_actual_TCP_z_m':last['command_z_m'],'closure_command_tracks_actual_TCP_not_previous_requested_grasp':True};save(R/'supported_start_geometry.json',geom)
 oldnames=['relation_GT_boxes_holding32_20261018','acquisition_scene_surface_pilot8_20261018','relation_GT_boxes36_ABC_20261018','holding_matched_input_diagnostic_20261018','holding_static_context_diagnostic_20261018','holding_action_fork_feasibility12_20261018','native_restore_semantics_root1_20261018','native_restore_semantics_root1_fix01_20261018'];verified={}
 for name in oldnames:
  mp=P/'experiments'/name/'configs/delivery_manifest.json';m=read(mp)
  for p,h in m['sha256'].items():assert sha(p)==h,p;verified[p]=h
  verified[str(mp)]=sha(mp)
 for p,h in read(E/'configs/frozen.json')['sha256'].items():assert sha(p)==h,p
 for p,h in read(E/'configs/input_hashes.json')['sha256'].items():assert sha(p)==h,p
 audit={'utc':now(),'runs':6,'prefix_controls_total':144,'execution_ticks':1152,'initialization_ticks_successful_attempt':2,'initialization_ticks_preserved_failed_attempts':4,'all_attempts_total_ticks':1158,'new_layouts':0,'fits':0,'full_control_state_comparisons':len(metric),'first_control_physics_tick_comparisons':len(firstticks),'equal_image_array_comparisons':len(images),'archive_equal_image_arrays':6,'same_time_render_audit_observations':157,'all_full_physical_fields_equal_after_first_tick':True,'all_compared_python_history_actions_labels_equal':True,'labels_independently_recomputed':len(labelrows),'whole_trajectory_numerical_tolerance':0,'first_divergence_after_first_tick':None,'old_files_verified':len(verified),'old_experiments':oldnames,'frozen_source_unchanged':True,'native_fork_design':'salvageable_for_this_configuration_and_bounded_existing_root_validation; not general promotion','full_original72step_history_replayed':False,'supported_start_was_already_Holding':True,'guide_complete':False}
 save(R/'AUDIT.json',audit);save(E/'results/postcollection_audit.json',audit)
 # Publication-style plot; all axes and colors explained in report and legend.
 fig,axs=plt.subplots(2,3,figsize=(18,10),layout='constrained');colors={'H2_after_same_H':'#1976b6','H3_after_L1':'#e88727','H4_after_L2':'#2d9b74'};names={'H2_after_same_H':'重复hold后恢复','H3_after_L1':'第一次短lift后恢复','H4_after_L2':'第二次短lift后恢复'}
 for ax,key,title,ylabel in zip(axs[0],['max_arm_torque_difference_Nm','max_finger_motor_force_difference_N','constraint_readback_difference'],['A 机械臂力矩输出刷新','B 双指电机力输出刷新','C 夹爪gear约束输出刷新'],['与H1最大绝对差 / N·m','与H1最大绝对差 / N','与H1绝对差 / 原生impulse÷dt']):
  for name,color in colors.items():
   rr=[r for r in full_ticks if r['run']==name];ax.plot([r['tick']for r in rr],[r[key]for r in rr],marker='o',markersize=4,color=color,label=names[name])
  ax.set_title(title);ax.set_xlabel('首次动作后的物理tick（0＝动作前）');ax.set_ylabel(ylabel);ax.set_xticks(range(9));ax.grid(alpha=.2)
 axs[0,0].legend(fontsize=9)
 ax=axs[1,0]
 for a,b,n in pairs:
  rr=[r for r in metric if r['reference']==a and r['compared']==b];ax.plot([r['control']for r in rr],[r['max_base_xyz_difference_m']for r in rr],'.-',alpha=.65,color='#697780',label=b+' vs '+a)
 ax.set_ylim(-1e-12,1e-12);ax.set_xlabel('控制步（每步8物理tick）');ax.set_ylabel('全部body位置最大绝对差 / m');ax.set_title('D 所有比较重合于0；完整物理字段也完全相同');ax.grid(alpha=.2);ax.text(.03,.93,'灰线：6组配对完全重合\n120个控制步、48个物理tick配对\n792组图像数组配对：差异均为0',transform=ax.transAxes,va='top')
 pr=[r for r in preparation if r['control']>=200];xx=[r['close_control']for r in pr];ax=axs[1,1]
 for key,color,label in [('table_vertical_contact_force_N','#1976b6','桌面'),('leftpad_vertical_contact_force_N','#e88727','左指垫link10'),('rightpad_vertical_contact_force_N','#2d9b74','右指垫link12')]:ax.plot(xx,[r[key]for r in pr],'.-',color=color,label=label)
 ax.axhline(9.8,color='gray',linestyle='--',label='目标重力9.8N');ax.axvline(2,color='#8b518e',linestyle=':',label='首次Holding');ax.set_xlabel('闭合控制步（0＝闭合前）');ax.set_ylabel('作用于红块的接触力竖直分量 / N');ax.set_title('E 原始准备：支撑从桌面转到指垫');ax.legend(fontsize=9);ax.grid(alpha=.2)
 ax=axs[1,2];ax.plot(xx,[r['center_rise_since_close_start_mm']for r in pr],'.-',color='#1976b6',label='块中心相对闭合前上移');ax.plot(xx,[r['box_bottom_clearance_above_table_mm']for r in pr],'.-',color='#e88727',label='原生盒体最低点离桌面');ax.axvline(2,color='#8b518e',linestyle=':',label='首次Holding');ax.axhline(0,color='gray',linewidth=.8);ax.set_xlabel('闭合控制步（0＝闭合前）');ax.set_ylabel('竖直位移或间隙 / mm');ax.set_title('F 不是标签改动：闭合阶段已发生亚毫米抬升');ax.legend(fontsize=9);ax.grid(alpha=.2)
 fig.suptitle('恢复语义：历史力读数在首个物理tick刷新，受控轨迹精确复现\n已有首根｜6段×24控制步｜未训练、未新增布局；原失败试采保持不变',fontsize=17);fig.savefig(R/'figures/01_restore_semantics_and_start.png',dpi=160);fig.savefig(R/'figures/01_restore_semantics_and_start.pdf');plt.close(fig)
 shutil.copy2(__file__,E/'source/analyze.py');print(json.dumps({'audit':audit,'geometry':geom,'readbacks_at_zero':[r for r in full_ticks if r['tick']==0]},ensure_ascii=False))
if __name__=='__main__':main()
