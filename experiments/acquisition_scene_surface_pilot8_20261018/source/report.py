from common import *
import csv,shutil,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle,Patch
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False,'savefig.dpi':180})
def table(headers,rs):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(x)for x in r)+' |'for r in rs)+'\n'
def main():
 a=read(E/'results/existing_audit.json');b=read(E/'results/pilot_audit.json');proto=read(E/'configs/protocol.json');old=a['original_full_protocol'];gate=a['original_data_gate'];summary=read(E/'results/collection_summary.json');native=a['native_table_surfaces'];oldblocks=a['block_surface_audit'];newblocks=b['initial_state_audit'];replays=a['geometry_replays'];oldroots=a['root_audit'];roots=b['roots'];trace=b['control_trace_statistics'];coverage=b['coverage'];assert b['decision']['next_independent_collection_design_justified']
 fig,axs=plt.subplots(1,3,figsize=(15,6));fig.suptitle('采样配置缺陷：固定世界横坐标没有随 A/B/C 桌面范围变化',fontsize=18,y=.98)
 for ax,scene in zip(axs,'ABC'):
  lo,hi=np.array(native[scene]);ax.add_patch(Rectangle(lo[:2],hi[0]-lo[0],hi[1]-lo[1],facecolor='#e7e9ec',edgecolor='#454545',lw=2));rr=[x for x in oldblocks if x['scene']==scene];nn=[x for x in newblocks if x['scene']==scene]
  for outside,marker,color,label in [(False,'o','#6c8cad','原32根：范围内'),(True,'x','#c76d2c','原32根：范围外')]:
   rs=[r for r in rr if r['outside_native_table_xy']==outside];ax.scatter([r['requested_x_m']for r in rs],[r['requested_y_m']for r in rs],marker=marker,c=color,s=46,alpha=.8,label=label)
  ax.scatter([r['requested_x_m']for r in nn],[r['requested_y_m']for r in nn],marker='D',c='#17856c',edgecolor='white',s=55,label='独立8根试采：修正后');ax.set_xlim(-.38,.38);ax.set_ylim(-.17,-.005);ax.set_xlabel('世界 x（米）');ax.set_ylabel('世界 y（米）');ax.set_title(f'{scene} 场景：旧范围外 {sum(r["outside_native_table_xy"]for r in rr)}/{len(rr)}\n新 {sum(not r["requested_inside_native_surface"]for r in nn)}/{len(nn)}');ax.grid(alpha=.15)
 h,l=axs[0].get_legend_handles_labels();fig.legend([Patch(facecolor='#e7e9ec',edgecolor='#454545')]+h,['官方声明的 table 采样区']+l,loc='lower center',ncol=4,bbox_to_anchor=(.5,.11),fontsize=10);fig.text(.5,.04,'点表示采集前指定的方块中心，不是模型预测。灰区是官方采样支持范围，不是物理可达性的全部边界。\n旧96个方块与新24个方块来自不同初态；这不是同根随机对照，不能用来估计成功率增益。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.22,1,.90]);fig.savefig(R/'figures/01_scene_sampling_cause.png');fig.savefig(R/'figures/01_scene_sampling_cause.pdf');plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(14,10));fig.suptitle('一次修正后的 8 根工程试采：7 根通过抓取安全检查，1 根保留拒绝',fontsize=18,y=.98)
 ax=axs[0,0];xx=np.arange(3);pos=[next(x for x in coverage if x['scope']=='all_actual_outcomes'and x['group_type']=='color'and x['group']==c)['positive_roots']for c in COLORS];attempt=[sum(r['target_color']==c for r in roots)for c in COLORS];ax.bar(xx,pos,color='#17856c',label='可靠 Holding 正例独立根');ax.bar(xx,np.array(attempt)-pos,bottom=pos,color='#cb8d4c',label='已执行初态，未取得正例');ax.set_xticks(xx,['红色','蓝色','粉色']);ax.set_ylim(0,4);ax.set_ylabel('本次该颜色为抓取目标的独立根数');ax.set_title('A  三色均有正例，但这不是原门槛验收')
 for i,(n,total)in enumerate(zip(pos,attempt)):ax.text(i,total+.08,f'{n}/{total}',ha='center',fontsize=15)
 ax.legend(fontsize=9,loc='upper right')
 ax=axs[0,1];rr=[next(x for x in coverage if x['scope']=='all_actual_outcomes'and x['group_type']=='color'and x['group']==c)for c in COLORS];bottom=np.zeros(3)
 for key,label,color in [('positive_queries','可用正例','#17856c'),('negative_queries','可用负例','#6c8cad'),('unobservable','端点不可见，弃权','#b9bec4')]:
  vals=[r[key]for r in rr];ax.bar(xx,vals,bottom=bottom,color=color,label=label)
  for x,v,base in zip(xx,vals,bottom):ax.text(x,base+v/2,str(v),ha='center',va='center')
  bottom+=vals
 ax.set_xticks(xx,['红色','蓝色','粉色']);ax.set_ylabel('每色120查询（60状态 × 2相机）');ax.set_ylim(0,153);ax.set_title('B  正负均按同期承重接触与稳定性标注');ax.legend(fontsize=9,loc='upper right')
 ax=axs[1,0];ro=read(E/'results/replays/relation32_A_00.json')['planning'];rn=read(E/'data/private/engineering8_C_06/frozen_controller_result.json')['planning_private'];x=np.arange(1,17);ax.plot(x,[q['distance_m']*1000 for q in ro],marker='o',color='#6c8cad',label='旧 A_00：红块');ax.plot(x,[q['distance_m']*1000 for q in rn],marker='s',color='#cb8d4c',label='新 C_06：蓝块');ax.axhline(12,ls='--',color='#333',label='未改动的12mm判据');ax.set_xticks([1,4,8,12,16]);ax.set_xlabel('冻结的16个候选编号');ax.set_ylabel('候选首次被拒时的几何间距（mm）');ax.set_ylim(8.5,12.6);ax.set_title('C  几何拒绝仍存在，未放松安全间隙');ax.legend(fontsize=9,loc='lower left')
 ax=axs[1,1];m=np.zeros((8,3))
 for i,r in enumerate(roots):
  m[i]=[r['initial_non_base_table_contacts']==0,r['safety_checked_grasp_success'],r['positive_queries_all_colors']>0]
 ax.imshow(m,cmap=matplotlib.colors.ListedColormap(['#cb8d4c','#17856c']),vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(8),[r['root_id'].replace('engineering8_','')+' '+{'red':'红','blue':'蓝','pink':'粉'}[r['target_color']]for r in roots]);ax.set_xticks(range(3),['初态无设备接触','抓取安全检查通过','有稳定Holding正例']);ax.set_title('D  每根保留；绿色满足，橙色不满足')
 for i in range(8):
  for j in range(3):ax.text(j,i,'是'if m[i,j]else'否',ha='center',va='center',color='white',fontsize=10)
 fig.text(.5,.028,'全部为工程开发数据：8初态、60同期状态、120主图、360查询；没有模型训练或泛化评估。\n原32根与原门槛仍然失败。新试采不补入原划分，7/8只是这8次真实运行的描述。',ha='center',fontsize=11);fig.tight_layout(rect=[0,.10,1,.94]);fig.savefig(R/'figures/02_engineering_pilot_results.png');fig.savefig(R/'figures/02_engineering_pilot_results.pdf');plt.close(fig)
 # All original gate requirements and root lineage are included, not only failures.
 gt=old['data_gates'];gate_rows=[]
 for key,label in [('train_per_color','训练集每颜色'),('dev_test_per_color','开发/测试各自每颜色'),('each_camera_train','训练集每相机'),('each_camera_dev_test','开发/测试各自每相机')]:
  z=gt[key];gate_rows.append([label,z['positive_records'],z['negative_records'],z['positive_roots'],z.get('negative_roots','未另设')])
 lineages=[]
 for r in oldroots:lineages.append([r['root_id'].replace('relation32_',''),r['split'],r['target_color'],r['controller_status'],r['reason']or'—',r['eligible_positive_queries']])
 pilot_rows=[]
 for c in COLORS:
  z=next(x for x in coverage if x['scope']=='all_actual_outcomes'and x['group_type']=='color'and x['group']==c);z2=next(x for x in coverage if x['scope']=='safety_checked_successful_grasp_roots'and x['group_type']=='color'and x['group']==c);pilot_rows.append([c,z['positive_roots'],z['positive_queries'],z['negative_roots'],z['negative_queries'],z['unobservable'],z2['negative_queries']])
 grasp=[x for x in trace if x['stage']=='frozen_grasp'];maxpenetration=max(x['max_robot_penetration_m']for x in grasp)*1000;maxdrift=max(x['max_unrequested_object_displacement_m']for x in grasp)*1000;postcontacts=[x for x in trace if x['stage']=='post_grasp_anchors'and x['non_target_robot_contact_samples']]
 report=f'''# 采集阻塞诊断与独立8根工程试采

已完成一次有证据的采样修正：把固定世界横坐标改为各场景官方桌面范围内的相对位置。**原32根数据、原失败结论和全部门槛未动；新8根只属于工程开发，未合并、未训练。** 新试采7根抓取通过沿用的物理安全阈值，1根继续被12mm间隙判据拒绝。三色均形成可靠 Holding 正例及负例，支持制定下一份独立冻结的采集方案，但不构成正式模型评估或控制器全面验收。

## 找到的原因及证据强度

**采样配置存在明确场景不匹配。** 原采集器对A/B/C都使用世界 x=[−0.15,0.075,0.30]，而原生声明的 table x 范围分别为 A=[−0.20,0.35]、B=[−0.35,0.15]、C=[0.00,0.35] 米（各自 y=[−0.15,−0.03] 米）。所以B右槽、C左槽超出各自的桌面采样支持区。原96个指定方块中23个位于范围外，涉及23根。官方采样范围不是所有物理可达表面的边界，越界不等于必定失败；但用它们作为可靠桌面抓取正例来源缺少依据。

原6个监测停止根（B_09、B_14、C_24、C_27、C_28、C_29）在开始抓取前均已出现范围外方块对 table link0（button_link）的接触。保留的初态接触力和方块速度表明一些方块尚在按钮/底面夹挤中运动；这不是关系读出错误。比如B_09粉块与按钮的已存接触力约1974.9 N，C_24蓝块有约524.9 N的按钮接触。这里是仿真接触记录，不是实际机器人可承受力的声明。全部坐标、初态位移/速度、力及对应根见 original_block_surface_audit.csv 和 existing_audit.json。

**首个红根 A_00 的失败另属保守几何拒绝。** 它的目标在官方桌面范围内；16候选均在下降阶段对 slide_link（table link2）间隙小于12mm时提前拒绝，当时未发出抓取控制。原记录距离9.690–11.942mm都仍大于零，因此这是前瞻安全裕量判据，不是已发生的碰撞。后续路径是否会碰撞不能由首次拒绝点判定。本轮没有通过缩小间隙或移除碰撞体来制造正例。

**未发现坐标/几何复制错误。** 对旧A_00失败、A_03成功、B_09监测失败、C_24监测失败的四个已保存初态做纯几何重放；每个比较24个碰撞形状及原生/查询世界的关节链接坐标和AABB。链接坐标差均为0，最大AABB差5.6×10⁻¹⁷m；64个候选的离散决定和数值（容差10⁻⁶）逐项复现。重放使用独立DIRECT审计世界，没有相机渲染、没有恢复后物理推进，也没有重新采集这四根；恢复的接触力不被用作标签。完整记录在 results/replays/。

12mm不是被证明的数学最小距离，但有明确历史工程依据：此前相同公开初态的v1执行使滑门移动9.466mm，而保持该间隙的v2轨迹滑门移动0、官方抓取成功。又核对了旧A/B/C各一条成功的闭抽屉抓取轨迹及C_acceptance_v2_00失败例，成功轨迹的逐步控制/接触/位姿文件仍在。历史v2确实是 **134/135可行请求通过、并未全面晋级**；完成文件不代表全部验收成功。本轮也未修改这一历史状态。

## 原颜色缺口与完整门槛（结论保持失败）

训练每色各6个目标根。红色：2成功、3无合格几何候选、1非目标设备变化；蓝色：3成功、3无合格几何候选；粉色：4成功、1几何候选拒绝、1设备变化。故原训练可靠正例根仍是红2、蓝3、粉4。原开发红1/蓝2/粉2；原测试红1/蓝2/粉1。对具体失败根的记录见文末完整根表。

{table(['原覆盖门槛适用组','最低正查询','最低负查询','最低正例独立根','最低负例独立根'],gate_rows)}

查询为“一个颜色×一台相机×一个状态”，不等于独立样本。物理标签必须通过原双指接触、无外部支撑、12控制步稳定、0.005/0.01/0.015N一致性检查；可见条件仍为目标方块及每侧指尖家族分别≥4像素。任何失败门槛停止拟合，不放宽、不补入其他划分。还要求同tick四次渲染、前后状态/接触一致、flags开关RGB/深度逐位一致、根与种子去重，以及不出现旧数据/跨划分RGB重复。未设的阈值不能在事后当成原门槛。

原32根：**训练A9/B9共18，开发C6，测试C8**。根的所有状态/相机/颜色保持同划分。根种子261019100…261019131与完整初态、参数、全部原门槛存于 configs/original_gate_and_lineage.json。原清单SHA256仍为 `4823571c19297461d26903e6d804ff5b2bcebe999d9170eb89c7bcd2a030975c`。本轮读取了原train/dev/test失败证据，且重放包含原test的C_24，因此这些旧根现在已参与方法开发，不能以后重新包装为未见测试。

## 唯一修正与先验冻结

新采集器版本 `scene_relative_x_collector_v2_engineering_pilot` 将横向槽位改成 `lo_x + f*(hi_x−lo_x)`，f=[0.18,0.50,0.82]，这组比例来自已有控制器测试生成器。保持原x抖动±0.012m、y∈[−0.115,−0.055]m、z=0.46m、yaw±0.20rad、颜色置换、抽屉初始开度、控制器/候选/12mm与2mm间隙、动作锚点以及标签全部不变。只增加私有逐控制步轨迹用于解释失败，不改变控制决策。

8新初态在任何试采渲染前冻结：A2（红/蓝），B3（红/蓝/粉），C3（红/蓝/粉），种子261019300…261019307；最多64同期状态、每根900控制步、无替换根、无额外候选调参。全部根标记engineering_pilot，无train/dev/test划分，也不填补原32根缺口。它们是在看过旧失败后设计的工程初态，不是独立泛化测试。公开NPZ仅RGB；隐藏状态、接触、原生mask和动作只在控制与私有标注侧，没有任何模型输入或训练。

预先规定：修正后8根所有初始物体都在原生桌面采样区，且初态不与按钮/开关/滑门/抽屉有>0.01N接触；每色至少1个沿用安全阈值检查的稳定Holding正例根和负例，正例在两相机均可见，全部同期/物理守卫通过，才支持写下一份采集设计。这个工程条件**不替代原每色4正例根门槛**。

## 8根实际结果

8根全部尝试一次，7根保存8状态，C_06保存4状态；共 **60状态、120主图、240相机调用、360查询、{summary['controls']:,}控制步**。8根24个初始方块全部在各自声明范围内，初态非底面设备接触为0。初态最大位移0.0412mm、最大速度0.227mm/s；未按结果删根。

{table(['颜色','稳定正例根','可用正查询','负例根（含失败根）','可用负查询（含失败根）','端点缺失查询','仅7成功根的负查询'],pilot_rows)}

所有54个可用正查询均出自7个抓取安全检查通过根；失败根C_06的23条可用负查询仍保留。可用合计314，端点不可见46，物理不确定0。没有将不可见查询记为负例，也没有将失败根整根删除。按相机和每根的完整计数见 pilot_Holding_coverage.csv。

抓取执行窗口的安全检查沿用原阈值：控制器成功；非目标物体位移<10mm；非抽屉设备状态变化不超过0.003；机械手穿透<10mm。本次7个成功抓取窗口实际最大非目标位移 **{maxdrift:.3f}mm**、机械手穿透 **{maxpenetration:.3f}mm**，非抽屉设备变化为0，非目标机械手接触记录为0。这里未重新运行旧完整官方任务/技能链验收，不能称控制器全面通过。

C_06蓝块仍在桌面有效范围内，但16候选对slide_link的首次拒绝距离9.179–11.892mm，均不足12mm，0个抓取控制步后保留失败。这说明域内也存在固定姿态候选的保守拒绝，横坐标修正没有消除所有可行性限制。本轮没有第二次修正、没有再换根试到成功。

采集锚点中固定的抓取后平移/释放也保留完整轨迹：B_04和失败C_06分别出现23、19个非目标接触样本，C_06后续锚点的非目标位移最大2.724mm；这些后处理采集动作不能宣传为全程无接触。它们没有被隐藏，也没有用于把失败改写成成功。所有阶段在 pilot_control_safety.csv 分列；上段安全成功仅指预先定义的冻结抓取窗口。

同期核验与重算完成：60次当前标签、360个原生框、2,160次历史接触阈值、3,313个控制步的Holding真值。原生/公开RGB一致，前后完整状态一致，flags开关RGB/深度一致。没有旧RGB哈希重复。初态、稳定性和完整接触轨迹可供复核。未采集/评估In正例，未训练关系模型、定位器、世界模型或SupCon。

## 下一步是否有依据

**有依据制定下一份独立冻结采集方案**：确定的场景不匹配已改正，8根中初始按钮接触消失，三色均形成安全检查通过的稳定Holding正/负证据。但不同种子/布局的旧32根与新8根不是同根随机对照，7/8不能当成改善幅度、泛化成功率或模型性能。仍有保守几何拒绝，应作为预期失败保留。

下一份方案应继续使用修正后的场景相对横坐标、同一控制器和同一标签/可见性定义；在生成任何新根前冻结根总数、场景/颜色配额、原完整覆盖门槛、训练/开发/测试根级划分和预算停止条件。若预计独立正例不足，应在采集前明确足够的总预算，而不是采完补成功根。所有已用于诊断的旧根和本8根都排除在新的未见测试之外，保留历史失败谱和边界案例，不以本轮观察挑更容易的布局/颜色。

仅制定新方案有依据；**新的正式采集没有授权或启动**。本轮8根预算已用完，原32根红/蓝不足4正例根的失败原样保留，不能将本轮3/2/2根加进去“补过门槛”。整个世界模型/TS-LTL指南仍未完成。

## 图表解释

1. figures/01_scene_sampling_cause.png：每幅对应A/B/C；横/纵轴是世界x/y米。灰色矩形为官方table采样区；蓝圆是原32根区内中心，橙叉是原区外中心，绿菱形是独立8根的新指定中心，颜色不表示方块颜色。它解释生成器没有随场景移动的问题，不表示模型性能或同根因果对照。
2. figures/02_engineering_pilot_results.png：A按方块颜色给出可靠正例独立根/指定目标根，绿为有正例、橙为未得到正例；B按颜色堆叠可用正（绿）、负（蓝）、不可见（灰）查询；C的横轴为16固定候选编号，纵轴为首次拒绝几何距离mm，两条线分别旧A_00和新C_06，黑虚线是未变的12mm；D逐根显示初态无设备接触、抓取安全检查、有稳定正例三项，绿满足、橙不满足。这些均为工程数据记录，非精确率/召回率。

## 原32根完整结果与划分

{table(['原根（前缀relation32_）','原划分','目标色','冻结状态','原因','可用正查询'],lineages)}

## 文件和复现

实验 `{E}`；报告 `{R}`。源码 source/，冻结参数 configs/，同期RGB data/public/，原生mask/真值/逐控制步轨迹 data/private/，审计 results/，日志 logs/。原实验与报告均只读，末尾再次校验完整原清单和控制器源文件。运行顺序见 RUN_COMMANDS.md；四个历史重放不消耗新试采初态，也没有物理执行。采集GPU串行上锁，审计和图表在已有环境CPU运行，没有软件或模型下载。

本次实际数据目录{summary['data_bytes']:,}字节。模型拟合、视觉特征编码和模型预测均为0。全部初态结果、失败及后续接触记录保留，不为改善数字筛选数据。
'''
 (R/'REPORT.md').write_text(report)
 (R/'RUN_COMMANDS.md').write_text(f'''# 实际执行顺序
```bash
source {P}/env.sh
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/acquisition_pilot_src/audit_existing.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/acquisition_pilot_src/prepare.py
python -u {E}/source/collect.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/acquisition_pilot_src/audit_pilot.py
source {P}/world_model_env.sh
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/acquisition_pilot_src/report.py
```
采集只运行一次。prepare、collect和audit_existing含防覆盖断言，不能重跑覆盖封存目录。历史几何重放结果保存在results/replays；新采集8根为全新工程初态，不作为未见测试。
''')
 for name in ['audit_pilot.py','report.py']:shutil.copy2(Path(__file__).parent/name,E/'source'/name)
 shutil.copy2(Path(__file__).parent/'audit_existing.log',E/'logs/audit_existing.log')
 outcome={'utc':now(),'scope':'bounded8configuration engineering acquisition pilot only','old_outcome':'failed gate unchanged, train red2 blue3 pink4 positive roots','cause':'scene-agnostic fixed world-x slots are outside native table sampling support for B/C; distinct conservative geometric rejection remains','correction':'only scene-relative x-slot generation; controller, margins and labels unchanged','pilot_roots':8,'pilot_states':60,'safety_checked_grasp_successes':7,'failures':1,'positive_roots_by_color':{'red':3,'blue':2,'pink':2},'new_collection_design_justified':True,'new_collection_started':False,'training_updates':0,'geometry_replays':4,'guide_complete':False,'report':str(R/'REPORT.md')};save(E/'results/ENGINEERING_PILOT_COMPLETE.json',outcome);save(R/'OUTCOME.json',outcome)
 print('REPORT_WRITTEN',R/'REPORT.md',flush=True)
if __name__=='__main__':main()
