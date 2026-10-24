from common import *
import csv,copy,collections,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def diff(a,b,path=''):
 out=[]
 if isinstance(a,dict) and isinstance(b,dict):
  assert set(a)==set(b)
  for k in sorted(a):out+=diff(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,list) and isinstance(b,list):
  assert len(a)==len(b)
  for k,(x,y)in enumerate(zip(a,b)):out+=diff(x,y,path+'/'+str(k))
 elif canonical(a)!=canonical(b):out.append({'path':path,'reference':a,'restored':b,'absolute_difference':abs(a-b)if isinstance(a,(float,int))else None})
 return out
def csvwrite(path,rows):
 with path.open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 d=E/'data/private/fork12_A_00';before=read(d/'aligned_action_start_checkpoint_private.json')['physical'];restored=read(d/'aligned_hold_start_restore_actual_private.json');last=read(d/'branches/fork12_A_00_aligned_lift/s72/physical_full.json');differences=diff(before,restored)
 assert len(differences)==8
 for r in differences:
  def at(x,p):
   for key in p.strip('/').split('/'):x=x[int(key)]if isinstance(x,list)else x[key]
   return x
  r['last_lift_value']=at(last,r['path']);r['restored_equals_last_lift']=canonical(r['restored'])==canonical(r['last_lift_value'])
 assert all(r['restored_equals_last_lift']for r in differences)
 csvwrite(R/'restore_differences.csv',differences)
 pc=read(d/'aligned_hold_start_restore_comparison.json');assert pc['python']['equal'];assert read(d/'aligned_base_restore_comparison.json')['physical']['equal']
 allowed={f'/bodies/0/joint_states/{j}/3'for j in range(7)}|{'/constraints/0/applied_forces/0'};assert {r['path']for r in differences}==allowed
 rows=read(E/'results/records.json');branches=read(E/'results/branches.json');summary=read(E/'results/collection_summary.json');assert summary['roots_attempted']==1 and summary['branches_attempted']==2 and summary['states_saved']==7 and not summary['action_differences_interpretable'];assert len(rows)==42 and not any(b['fork_valid']for b in branches)
 historic=set(read(E/'configs/historical_exclusion.json')['old_RGB_array_hashes']);camera_checks=[];state_checks=[];root_hashes={}
 for bid in [b['branch_id']for b in branches]:
  for sd in sorted((d/'branches'/bid).glob('s*')):
   step=int(sd.name[1:]);pub=E/'data/public'/f'{bid}_s{step:02d}.npz';integrity=read(sd/'integrity.json');assert all(integrity[k]for k in ['same_tick_both_cameras','render_physics_unchanged','baseline_RGB_depth_equal'])
   with np.load(pub,allow_pickle=False)as z:
    assert set(z.files)=={'rgb_static','rgb_gripper'}
    for cam in ['static','gripper']:
     with np.load(sd/f'native_{cam}.npz',allow_pickle=False)as n:
      assert np.array_equal(z['rgb_'+cam],n['rgb']);h=hashlib.sha256(n['rgb'].tobytes()).hexdigest();assert h not in historic
      matches=[r for r in rows if r['branch_id']==bid and r['control_index']==step and r['camera']==cam];assert len(matches)==3 and all(r['RGB_sha256']==h for r in matches)
      root_hashes.setdefault(h,set()).add('fork12_A_00');camera_checks.append({'branch':bid,'control':step,'camera':cam,'public_equals_native':True,'historical_duplicate':False,'sha256':h})
   truth=read(sd/'truth_private.json');assert len(truth['last12_controls'])==12
   for j,color in enumerate(COLORS):
    target=[r for r in rows if r['branch_id']==bid and r['control_index']==step and r['color']==color];assert len(target)==2
    actual=bool(truth['cutoffs'][1][j]);stable=all(all(h['holding_at_cutoffs'][k][j]==actual for k in range(3))and all(sum(x)<=1 for x in h['holding_at_cutoffs'])for h in truth['last12_controls']);assert all(r['Holding']==actual and r['physical_valid']==stable and r['unknown']==(not(stable and r['visible']))for r in target)
   state_checks.append({'branch':bid,'control':step,'same_tick':True,'physical_unchanged_by_render':True,'three_cutoff_12control_stability_recomputed':True})
 assert all(len(v)==1 for v in root_hashes.values())
 image_compare=[]
 for cam in ['static','gripper']:
  p=d/'branches/fork12_A_00_aligned_lift/s00'/f'native_{cam}.npz';q=d/'branches/fork12_A_00_aligned_hold/s00'/f'native_{cam}.npz'
  with np.load(p,allow_pickle=False)as a,np.load(q,allow_pickle=False)as b:
   for key in ['rgb','depth_buffer','mask']:assert np.array_equal(a[key],b[key]);image_compare.append({'camera':cam,'field':key,'unequal_values':int(np.sum(a[key]!=b[key]))})
 traces=[]
 for p in sorted(d.rglob('*.jsonl')):traces += [json.loads(line)for line in p.read_text().splitlines()]
 assert len(traces)==292;assert sorted(t['actual_root_control_count']for t in traces)==list(range(1,293));assert sum(t['branch']=='fork12_A_00_aligned_hold'for t in traces)==0
 assert all(t['tick']==2+8*t['actual_root_control_count']for t in traces)
 act=read(E/'data/public/fork12_A_00_aligned_lift_actions.json');assert len(act['action_vectors_xyz_quaternion_grip'])==act['completed_controls']==72
 commands=np.array(act['action_vectors_xyz_quaternion_grip']);assert np.array_equal(commands[:12],np.repeat(commands[:1],12,axis=0));assert np.max(np.abs(commands[60:,:3]-commands[59,:3]))==0
 all_original={}
 names=['relation_GT_boxes_holding32_20261018','acquisition_scene_surface_pilot8_20261018','relation_GT_boxes36_ABC_20261018','holding_matched_input_diagnostic_20261018','holding_static_context_diagnostic_20261018']
 for name in names:
  manifest=P/'experiments'/name/'configs/delivery_manifest.json';m=read(manifest)
  for path,h in m['sha256'].items():assert sha(path)==h,path;all_original[path]=h
  all_original[str(manifest)]=sha(manifest)
 for p,h in read(E/'configs/precollection_frozen.json')['sha256'].items():assert sha(p)==h,p
 for p,h in read(E/'configs/input_references.json')['sha256'].items():assert sha(p)==h,p
 audit={'utc':now(),'independent_restore_difference_count':8,'all_differences_equal_last_lift_readback':True,'all_other_full_physical_fields_exact':True,'individual_finger_joint_states_exact':True,'python_and_motor_commands_exact':pc['python']['equal'],'pre_action_image_fields_bitwise_equal':image_compare,'state_audits':len(state_checks),'camera_audits':len(camera_checks),'annotation_queries_recomputed':len(rows),'real_controls':len(traces),'execution_ticks':len(traces)*8,'initialization_ticks':2,'total_actual_ticks':len(traces)*8+2,'valid_sibling_pairs':0,'paired_common_prefix_controls':0,'historical_RGB_overlap':0,'all_pilot_lineage':'engineering_feasibility_only','old_files_hash_verified':len(all_original),'old_manifests':names,'frozen_source_unchanged':True,'fits':0,'first50_storage':'not reached:7states','128check':'not reached:7states; no training','guide_complete':False}
 save(R/'AUDIT.json',audit);save(E/'results/postcollection_audit.json',audit);csvwrite(R/'camera_audit.csv',camera_checks);csvwrite(R/'state_audit.csv',state_checks)
 ledger=[]
 for root in read(E/'configs/protocol.json')['roots']:
  for condition in root['condition_order']:
   for action in root['action_order']:
    b=next((b for b in branches if b['root_id']==root['root_id']and b['condition']==condition and b['action']==action),None);ledger.append({'root_id':root['root_id'],'scene':root['scene'],'target_color':root['target_color'],'condition':condition,'action':action,'status':b['status']if b else'not_attempted_after_stop','executed_controls':b['executed_controls']if b else 0,'saved_states':b['saved_states']if b else 0,'valid_pair':False})
 csvwrite(R/'planned_and_actual_branches.csv',ledger)
 target=[{k:r[k]for k in ['branch_id','control_index','camera','Holding','physical_valid','table_supported','table_upward_normal_force_N','visible','unknown']}for r in rows if r['color']==r['target_color']];csvwrite(R/'target_state_annotations.csv',target)
 fig=plt.figure(figsize=(15,12),layout='constrained');gs=fig.add_gridspec(3,3,height_ratios=[1,1,1.1]);blue='#1976b6';orange='#e88727'
 for row,cam in enumerate(['static','gripper']):
  with np.load(d/'branches/fork12_A_00_aligned_lift/s00'/f'native_{cam}.npz')as z:a=z['rgb'].copy()
  with np.load(d/'branches/fork12_A_00_aligned_hold/s00'/f'native_{cam}.npz')as z:b=z['rgb'].copy()
  for col,(im,title)in enumerate([(a,'检查点：lift执行前'),(b,'恢复后：hold执行前'),(np.abs(a.astype(int)-b.astype(int)).astype(np.uint8),'绝对像素差：全黑表示0')]):
   ax=fig.add_subplot(gs[row,col]);ax.imshow(im);ax.set_title(('静态相机'if cam=='static'else'腕部相机')+'｜'+title);ax.set_xlabel('图像列 / pixel');ax.set_ylabel('图像行 / pixel')
 ax=fig.add_subplot(gs[2,:2]);x=np.arange(7);a=[before['bodies'][0]['joint_states'][j][3]for j in x];b=[restored['bodies'][0]['joint_states'][j][3]for j in x];c=[last['bodies'][0]['joint_states'][j][3]for j in x]
 ax.plot(x,a,'o-',color=blue,label='检查点读数');ax.plot(x,b,'s--',color=orange,label='restore后的读数');ax.plot(x,c,'x',color='black',markersize=9,label='lift末端读数（与橙色完全重合）');ax.set_xticks(x);ax.set_xlabel('机械臂转动关节的原生索引');ax.set_ylabel('getJointState 已施加力矩 / N·m');ax.set_title('7个关节读数未恢复；最大差值2.00749 N·m');ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9)
 ax=fig.add_subplot(gs[2,2]);a=before['constraints'][0]['applied_forces'][0];b=restored['constraints'][0]['applied_forces'][0];ax.bar([0,1,2],[a,b,b],color=[blue,orange,'#777777']);ax.set_xticks([0,1,2],['检查点','恢复后','lift末端']);ax.set_ylabel('getConstraintState 原生返回量');ax.set_title('夹爪gear约束读数同样保留末端值');ax.grid(axis='y',alpha=.2);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
 fig.suptitle('原生检查点分叉门槛失败：图像相同，8个力相关读数不同\n仅1个布局；hold未执行，不能比较动作效果',fontsize=17)
 fig.savefig(R/'figures/01_native_restore_gate_failure.png',dpi=160);fig.savefig(R/'figures/01_native_restore_gate_failure.pdf');plt.close(fig)
 sizepub=sum(p.stat().st_size for p in(E/'data/public').rglob('*')if p.is_file());sizeprivate=sum(p.stat().st_size for p in(E/'data/private').rglob('*')if p.is_file())
 report=f'''# 原生状态分叉可执行性试采：按门槛停止

第一个新布局在恢复第二分支起点时未通过预先冻结的完整状态一致性门槛。已停止全部剩余采集；有效lift/hold同起点配对为0，训练为0。此结果不支持动作条件转移预测、状态作用或世界模型完成的结论。

## 设计、实际工作量与原因

| 元素 | 冻结设计 | 实际结果及含义 |
| --- | --- | --- |
| 独立布局 | 新根12个，ABC各4，红/蓝/粉各4 | 仅fork12_A_00（A、红）进入执行；其余11根未运行 |
| 起始条件 | aligned与固定localY偏移12mm；保留安全拒绝 | 仅aligned准备完成；offset未尝试；未筛选结果或更换候选 |
| 分支 | 同检查点lift/hold，各72控制步、2.4秒 | lift72步完成；hold在动作前中止，执行0步 |
| 同动作验证 | 两分支前12步命令相同，逐步全物理比较 | 只执行了lift前缀；配对重放0步，不能宣称确定性通过 |
| 保存上限 | 48分支×6状态，最多288状态 | 2分支已创建；lift6状态＋恢复后的hold起点1状态，共7状态、14张主相机图像 |
| 总物理执行 | 单卡串行，条件最多900控制步 | 292控制步＝2336执行tick，另有环境初始化/复位2tick，共2338tick；控制步包含24初始稳定、196抓取准备、72lift；无重试 |
| 标签与模型输入 | 真值仅标注/评分；双相机同步 | 公开npz仅RGB，动作文件仅真实命令；物理、mask与检查点均私有；没有训练或特征编码 |
| 存储 | 首50状态检查、128训练/泄漏检查 | 只到7状态，两个规模检查均未达到；实际公开数据{sizepub:,}字节，私有证据{sizeprivate:,}字节，不外推为转移成本 |

开始之前已冻结协议和采集源码，协议SHA256：`{sha(E/'configs/protocol.json')}`。初始布局、种子、条件顺序、动作顺序和停止条件均未事后更改。本次所有根及后代仅属engineering可执行性试采，不能作为训练/开发/独立测试；未读取D样本、特征、预测或评分。历史排除清单包含旧登记元数据，不能将此描述为从未读取任何D相关元数据。

## 首个导致停止的差异

原生PyBullet3.2.7 `saveState/restoreState`在同一执行client使用，同时恢复Python控制器、IK、RNG、场景逻辑和相机状态，并重新施加同一已保存电机命令。检查点在原始抓取准备后保存；先真实执行lift，再恢复并检查hold起点，没有推进额外物理步。

| 检查对象 | 恢复检查结果 | 解释 |
| --- | --- | --- |
| 7个body的姿态/速度、23个joint的位置/速度/反力，含两侧手指9/11及固定接触pad10/12 | 完全一致 | 未使用汇总夹爪宽度替代两侧状态 |
| 接触11条、link状态、动力学/视觉配置、引擎参数、约束配置 | 完全一致 | 逐项规范序列化比较，接触只排序、不放宽数值容差 |
| Python/电机命令/历史 | 完全一致 | 保存的可观察控制器记忆已恢复 |
| 静态与腕部RGB、depth、mask，共6数组 | 每个元素相同 | 两相机同tick；渲染前后物理不变；mask开启/关闭时RGB与depth一致 |
| 机械臂关节0–6已施加力矩 | 7个值不同，最大绝对差{max(r['absolute_difference']for r in differences[:7]):.8f} N·m | 恢复值逐项等于lift末端值，而非检查点值 |
| gear夹爪约束的已施加力返回值 | 1个值不同，绝对差{differences[-1]['absolute_difference']:.10f}（原生返回单位） | 检查点−0.0000855996，恢复后−0.0021397955；恢复值等于lift末端值 |

独立离线重算恰好发现以上8处差异，其余全物理记录一致。因此，首先破坏的是冻结的完整状态判据，并非图像不同或左右手指位置恢复错误。证据支持“力相关API读数未回到检查点”；**它不能区分仅上一步诊断读数未刷新，还是内部求解状态也影响下一步动力学**。原生求解器缓存无法直接完整观测，而且停止规则禁止继续推进hold，因此不能声称已经证明动力学非确定性。初始base的立即恢复通过，也不能替代经历动作后的恢复验证。

这一限制没有通过忽略这8项、增加预热步或放宽容差来绕过；未作第二次物理试验或修正。公开目录中的原始图像只作为失败审计证据，不是已通过门槛的转移数据。

## 图表如何阅读

![原生恢复门槛对照](figures/01_native_restore_gate_failure.png)

上两行分别为静态相机与腕部相机，列依次是检查点图像、恢复后图像及RGB逐元素绝对差；横纵轴为像素列/行，图像中红色积木是本根目标。差图全黑表示每个RGB通道差为0，不能据此推断完整动力学状态相同。depth和mask也逐元素相同，见AUDIT.json。

左下图横轴是机械臂关节原生索引0–6，纵轴是已施加力矩N·m。蓝实线/圆点表示检查点，橙虚线/方点表示恢复后，黑叉表示lift末端；橙点与黑叉重合。连线仅便于跨关节对照，不表示时间轨迹。右下柱状图比较同一gear约束的API返回量，蓝/橙/灰分别是检查点/恢复后/lift末端；保留原生数值，不混入机械臂力矩坐标。两图显示读数差异来自恢复前后，而非两个动作的效果比较。

## 起始标签和安全证据

本根aligned准备结束时，红色目标已经满足三阈值、12步稳定的Holding，且无桌面向上支撑接触；不是预先假定的“在桌面夹住但未抬起”状态。lift保存的全部6状态均仍为Holding，恢复后的hold起点也为Holding。因此，即使暂时忽略恢复失败，本次也没有可用的支撑→悬持配对证据，不能用它证明动作或状态必要性。没有把固定条件名当成真值标签。

准备过程最大机器人穿透1.41315mm，lift阶段0.01558mm；未记录非目标机器人接触，未触发设备/无关物体位移拒绝。控制命令与完整接触轨迹已保留。这些是一个根的过程检查，不是执行器晋级或多技能成功率。原始日志末尾出现环境析构重复disconnect告警，发生在停止结果写出及显式关闭之后，不是本次门槛失败原因。

## 审计和后续边界

离线重算7状态、14相机图像、42对象-视角标签；检查292条实际控制记录和2336执行tick（另有初始化/复位2tick），没有hold执行。历史RGB重合0；相同根内兄弟起点图像重复是设计要求。冻结采集源码/输入引用hash未变；五个先前实验的{len(all_original):,}个文件（含清单）hash复核通过。CSV列出全部48个计划分支，未运行项明确标记，而非补零成功。

当前阻塞是严格原生分叉有效性未建立。本轮按停止规则关闭，不提出或启动训练/开发数据扩采。若继续研究，应先由协调线程决定是否另行冻结一个仅验证恢复语义的方案，区分“诊断读数未刷新”与“后继动力学受影响”，或验证全新client原生复位与确定性历史重放；这不授权更换模型/主数据/验收标准，不自动放宽本轮门槛，也不应继续静态读出调参。

原指南要求的720→约6000转移、多技能族、1/3/5步、世界模型实际供给TS/LTLf和独立闭环评估仍未完成。本次试采没有取代这些要求。

文件：FROZEN_PROTOCOL.json；AUDIT.json；restore_differences.csv；planned_and_actual_branches.csv；target_state_annotations.csv；camera_audit.csv；state_audit.csv；PNG/PDF图。原始实验目录：`{E}`。
'''
 (R/'REPORT.md').write_text(report)
 shutil.copy2(__file__,E/'source/audit_report.py')
 print(json.dumps({'audit':audit,'report':str(R/'REPORT.md'),'public_bytes':sizepub,'private_bytes':sizeprivate},ensure_ascii=False))
if __name__=='__main__':main()
