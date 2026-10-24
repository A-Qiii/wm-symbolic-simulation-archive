"""Reporting only: consumes sealed-source pilot records; never runs physics."""
from pathlib import Path
import json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

P=Path('/home/robot/research/projects/wm_symbolic')
E=P/'experiments/official_state_resimulation_20261018'
R=P/'reports/official_state_resimulation_20261018'
OLD=P/'experiments/official_AB_state_pilot_20261018'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafbfc',
                     'axes.facecolor':'#fafbfc','savefig.facecolor':'#fafbfc'})
COLORS=['red','blue','pink'];CN={'red':'红色','blue':'蓝色','pink':'粉色'}
def read(p): return json.loads(Path(p).read_text())
def short(s): return s.replace('official_ABC_','').replace('_episode_',' / ')
def table(header,rows):
 return '| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+''.join('| '+' | '.join(str(v)for v in row)+' |\n'for row in rows)
def main():
 rows=read(E/'results/results.json');facts=read(E/'results/candidate_facts.json')
 summary=read(E/'results/episode_summary.json');coverage=read(E/'results/color_coverage.json')
 metrics=[r for r in read(E/'results/checkpoint_metrics.json')if r['root_id']=='all']
 verification=read(E/'results/verification.json');assert verification['passed']
 roots=sorted({r['root_id']for r in rows});figdir=R/'figures';figdir.mkdir(exist_ok=True)
 styles=[('positive','可靠为真','#21956f'),('negative','可靠为假','#3879ad'),('unknown','未知','#adb3bb')]
 fig,axes=plt.subplots(1,2,figsize=(12.8,5.5))
 for ax,group in zip(axes,['Holding','In']):
  data=[next(r for r in summary if r['root_id']==rid and r['role']=='all'and r['group']==group)for rid in roots]
  bottom=np.zeros(4)
  for key,label,color in styles:
   values=np.array([r[key]for r in data]);ax.bar(np.arange(4),values,bottom=bottom,color=color,label=label,width=.62)
   for i,v in enumerate(values):
    if v>=5:ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white'if key!='unknown'else'#283340',fontsize=11)
    elif v:ax.annotate(str(v),(i,bottom[i]+v/2),xytext=(20,0),textcoords='offset points',va='center',fontsize=10,arrowprops={'arrowstyle':'-','color':'#555'})
   bottom+=values
  ax.set(xticks=np.arange(4),xticklabels=[short(r)for r in roots],ylim=(0,105),ylabel='事实格数量（32帧 × 3种颜色）',title=group+'：第8步候选，逐步检查第4–8步')
  ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.07),frameon=False)
 fig.suptitle('4个来源episode：可靠负例是“谓词为假”，不是预测错误',fontsize=15,y=.98)
 fig.text(.5,.025,'横轴是来源episode，每个episode共32次初始化；三色柱段表示标签状态，不表示物体颜色。帧数不是独立轨迹数。',ha='center',fontsize=10)
 fig.tight_layout(rect=(0,.14,1,.93));fig.savefig(figdir/'episode_fact_coverage.png',dpi=170);plt.close(fig)

 selected=[]
 for group,classification in [('Holding','positive'),('In','positive'),('In','unknown')]:
  f=sorted([f for f in facts if f['group']==group and f['classification']==classification],key=lambda f:(f['root_id'],f['frame_id'],f['color']))[0]
  r=next(r for r in rows if r['root_id']==f['root_id']and r['frame_id']==f['frame_id'])
  selected.append((f,r))
 selection=[];fig,axes=plt.subplots(3,4,figsize=(11.8,9.3))
 titles=['原官方静态相机\n仅图像参考','新第8步静态相机\n配对同刻新标签','原官方腕部相机\n仅图像参考','新第8步腕部相机\n配对同刻新标签']
 for i,(f,r)in enumerate(selected):
  original=np.load(r['source_path'],allow_pickle=False);new=np.load(r['candidate']['observation_path'],allow_pickle=False)
  image_arrays=[original['rgb_static'],new['rgb_static'],original['rgb_gripper'],new['rgb_gripper']]
  for j,a in enumerate(image_arrays):
   axes[i,j].imshow(a,interpolation='nearest');axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
   for sp in axes[i,j].spines.values():sp.set_visible(False)
   if i==0:axes[i,j].set_title(titles[j],fontsize=10)
  cls={'positive':'可靠为真','unknown':'未知'}[f['classification']]
  axes[i,0].set_ylabel(f"{short(f['root_id'])}\n帧 {f['frame_id']}\n{f['fact']}\n新状态：{cls}",fontsize=10)
  selection.append({'selection_rule':'first lexicographic (root_id, frame_id, color) within each requested class; descriptive only',**f})
 fig.suptitle('同一来源初始化：原图与33.3 ms后新图',fontsize=15,y=.98)
 fig.text(.5,.025,'行依次展示Holding正例、In正例、In未知；腕部图原生84×84，静态图200×200。\n各列只做显示缩放，无裁剪或重标旧图；小像素差不能恢复原始历史接触力。',ha='center',fontsize=10)
 fig.tight_layout(rect=(.02,.075,1,.93));fig.savefig(figdir/'original_new_examples.png',dpi=180);plt.close(fig)
 (E/'results/illustration_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')

 x=[r['step']for r in metrics];fig,axes=plt.subplots(1,3,figsize=(14.1,4.8))
 for key,label,color in [('max_object_drift_mm','物体最大位移','#cf6a32'),('max_TCP_from_initialized_mm','TCP最大位移','#286da8')]:
  y=[r[key]for r in metrics];axes[0].plot(x,y,'o-',label=label,color=color,lw=2)
  axes[0].annotate(f'{y[-1]:.2f}',(8,y[-1]),xytext=(-6,8),textcoords='offset points',ha='right',color=color)
 axes[0].set(ylabel='相对本次初始化的位移（mm）',title='位置变化：全128次初始化的最大值')
 axes[0].legend(frameon=False,fontsize=10)
 y=[r['max_object_rotation_deg']for r in metrics];axes[1].plot(x,y,'o-',color='#8154a3',label='物体最大转角',lw=2)
 axes[1].annotate(f'{y[-1]:.2f}°',(8,y[-1]),xytext=(-6,8),textcoords='offset points',ha='right',color='#8154a3')
 axes[1].set(ylabel='相对本次初始化的转角（度）',title='旋转变化：全物体、全初始化最大值');axes[1].legend(frameon=False,fontsize=10)
 for key,label,color in [('mean_static_MAE_0_255','静态相机平均MAE','#218b70'),('mean_wrist_MAE_0_255','腕部相机平均MAE','#bd5289')]:
  axes[2].plot(x,[r[key]for r in metrics],'o-',label=label,color=color,lw=2)
 axes[2].set(ylabel='相对原官方图的平均像素绝对误差（0–255）',title='图像差异：全帧/通道均值，再跨128帧平均');axes[2].legend(frameon=False,fontsize=10)
 for ax in axes:
  ax.set(xticks=x,xticklabels=['1\n4.2 ms','4\n16.7 ms','8\n33.3 ms'],xlabel='固定物理步 / 仿真时间');ax.set_ylim(bottom=0);ax.grid(alpha=.18)
 fig.suptitle('新状态发生了真实位姿变化，图像相近不表示物理状态相同',fontsize=15,y=.98)
 fig.text(.5,.02,'点是固定检查时刻；连线只帮助读图，不是连续采样估计。不同曲线/时刻的最大值可能来自不同物体或初始化。',ha='center',fontsize=10)
 fig.tight_layout(rect=(0,.07,1,.93));fig.savefig(figdir/'pose_and_image_drift.png',dpi=170);plt.close(fig)

 episode_rows=[]
 for rid in roots:
  h=next(r for r in summary if r['root_id']==rid and r['role']=='all'and r['group']=='Holding')
  n=next(r for r in summary if r['root_id']==rid and r['role']=='all'and r['group']=='In')
  episode_rows.append([short(rid),32,*[h[k]for k in ['positive','negative','unknown']],*[n[k]for k in ['positive','negative','unknown']]])
 episode_rows.append(['合计（4个episode）',128,39,345,0,60,302,22])
 color_rows=[[c['group'],CN[c['color']],c['positive'],c['negative'],c['unknown'],c['positive_episodes'],', '.join(short(s)for s in c['positive_episode_ids'])]for c in coverage]
 role_rows=[]
 for role in ['continuous_restoration','uniform_coverage']:
  for g in ['Holding','In']:
   t=next(r for r in summary if r['root_id']=='all'and r['role']==role and r['group']==g)
   role_rows.append(['连续16帧/episode'if role=='continuous_restoration'else'均匀锚点及相邻帧/episode',g,t['initializations'],t['positive'],t['negative'],t['unknown']])
 drift_rows=[[m['step'],f"{m['step']/240*1000:.2f}",f"{m['max_object_drift_mm']:.3f}",f"{m['max_object_rotation_deg']:.3f}",f"{m['max_TCP_from_initialized_mm']:.3f}",f"{m['max_TCP_from_recorded_mm']:.3f}",f"{m['mean_static_MAE_0_255']:.3f}",f"{m['mean_wrist_MAE_0_255']:.3f}"]for m in metrics]
 maxarm=max(r['motor_audit']['max_arm_target_change_rad']for r in rows)
 budget=read(E/'results/run_budget.json')
 text=f'''# 官方状态初始化的短时重仿真：128次受限验证

本轮完成4个官方A/B来源episode的128次初始化；每次只运行8个物理步，零重试、零新下载、零训练。**新渲染图与同刻新物理状态配对后，得到Holding可靠正事实39个、In可靠正事实60个；22个In事实保留未知。** 这证明该小样本中可以建立新观测与严格物理标签的配对路径。它没有恢复旧官方图像的历史接触力，也没有完成正式数据集、模型训练或TS-LTL闭环验收。

“可靠”在本文仅指预先固定的第4–8步检查均通过：该窗口跨度16.7 ms，总运行33.3 ms。它不表示长期保持、技能成功或独立场景泛化。原50帧In候选所含60个逐色事实，在本轮**新**第8步状态中全部通过；原官方图对应的历史正标签仍保持未知。

## 控制恢复：为何这轮出现了Holding正例

原生`robot.reset`按观测夹爪宽度恢复手指位置及电机目标；仅修改`gripper_action`变量不会发出开闭命令。本轮按冻结协议，初始化后调用一次未修改的原生`apply_action`，目标为记录的TCP位置、朝向及`robot_obs[-1]`夹爪命令，不重放数据集的下一动作。每次记录并核查9项reset电机调用及9项原生控制调用，共2304项。

{table(['控制项','初始化reset','一次原生保持命令','含义'],[
['手臂7关节','记录关节角为电机目标','原生IK使TCP以记录位姿为目标','两阶段目标最大差 '+f'{maxarm:.6f} rad'],
['手指位置目标','观测总宽度/2','闭合每指0 m；张开每指0.04 m','位置目标不等于实际夹爪宽度'],
['手指电机最大力','200','闭合200；张开2','原生电机力上限，不是测得的接触力'],
['命令覆盖','128个初态','77次闭合、51次张开','39个Holding正状态全部在闭合命令下']])}

之前的单步宽度保持重建未出现Holding正例；本轮完整原生命令后出现39个新Holding正状态。这个对比涉及夹爪驱动、手臂IK目标及观察时刻，**不是仅改变夹爪命令的隔离因果实验**，不能用来给旧图补正标签。

## 固定规则与数据隔离

- 每个初态新建原生世界，按官方状态恢复一次位置；原文件没有的物体速度显式设为零。初始化后禁止传送、改变动力学、新抓取约束及外力/力矩。
- 每次固定8步，步长1/240秒；第0–8步保存完整物理记录，第1/4/8步保存双RGB、深度和proprio。候选始终取第8步，不逐帧选最好时刻。
- 保持既有严格Holding/In定义。Holding为双指接触且无外部承重支持；In须完整OBB满足抽屉几何界限并具有容器内承重路径，同时不是Held。原定义中的边界未知继续保留。
- 可靠性掩码要求第4、5、6、7、8步的真值、原生有效性和力证据均一致明确。Holding正例双指族接触均≥0.015 N、外部接触≤0.005 N；负例须外部支持≥0.015 N或至少一个指族≤0.005 N。In正例还须承重路径各边通过既有确定性阈值（力≥0.015 N、向上法向≥0.55）。这些是更保守的有效性筛选，不改原谓词真值。
- 新标签只引用本次新图路径与SHA256；公共观测NPZ没有`scene_obs`或`contacts`。完整接触/真值留在单独评分记录。新数据标记为`official-state-initialized resimulation`；固定控制器转移训练和正式训练均未启用。

这是短时再仿真的**状态标签候选**；相邻官方帧分别初始化，不构成已执行技能转移。384份检查点观测也不能算作384条独立样本轨迹。

## 按episode与颜色统计

下表每组谓词有384个事实格（128状态×3颜色）；正/负/未知互斥。正事实数不等于正状态数：Holding正状态39个，In正状态50个、包含60个逐色正事实。

{table(['来源episode','初始化','H真','H假','H未知','In真','In假','In未知'],episode_rows)}

![按来源episode的事实状态](figures/episode_fact_coverage.png)

图中横轴为来源episode，纵轴为事实格数。绿色表示可靠为真、蓝色可靠为假、灰色未知；这些是标签状态，蓝色负例不是预测错误，也不表示蓝色物体。每柱96格，不据此计算独立episode的置信区间。

{table(['谓词','物体颜色','真','假','未知','含正例episode数','正例来源'],color_rows)}

粉色Holding只有B / 0052中的313804、313805两帧，二者相邻，**仅1个独立来源episode**。22个In未知中20个为A / 0146红物体、2个为B / 0052粉物体；均保留边界不明确标志，没有为补齐正例调整初态、阈值或步数。

两种原有采样规则也分别保留，不互相替换：

{table(['原有采样组成','谓词','初始化','真','假','未知'],role_rows)}

## 图像示例与未知原因

![原始图与新仿真图对照](figures/original_new_examples.png)

每行是同一个来源初始化，列依次为原静态图、新第8步静态图、原腕部图、新第8步腕部图。物体红/蓝/粉色是场景外观；新状态的标签写在行标注中，不赋给原图。为便于复核，三类各取按episode、帧号、颜色字典序的首个案例；只作解释，没有据此选择训练数据。

{table(['新状态案例','第8步证据','结论与原因'],[
['A / 0109，1244236，蓝Holding','双指族最大法向力188.442 / 188.425 N；外部接触0','新状态双指夹持且无外部承重，固定窗口均通过'],
['A / 0109，1235281，红In','最小侧壁/上沿余量17.654 mm；存在确定的红物体→抽屉承重路径','几何与支撑同时通过，固定窗口均通过'],
['A / 0146，1793606，红In','最小侧壁/上沿余量1.650 mm，严格内部要求≥2 mm','几何处于未知带；有支撑也不能改判内部']])}

接触力来自新仿真接触记录，不是电机最大力字段；原图中“看起来夹住/放入”不单独构成严格标签证据。

## 位姿变化：新图相近，但不是原始状态复制品

{table(['步','时间ms','物体最大位移mm','物体最大转角°','TCP相对初始化最大mm','TCP相对原记录最大mm','静态MAE','腕部MAE'],drift_rows)}

![物理位姿和图像差异](figures/pose_and_image_drift.png)

左图橙线是三物体/128次初始化的最大位移，蓝线是TCP相对本次初始化的最大位移；中图紫线为物体最大转角；右图绿线/粉线是静态/腕部新图相对原图的像素MAE。横轴为固定步数及时间；位移单位mm、转角单位度、MAE在0–255像素尺度。连线帮助阅读检查点，不表示同一对象的连续轨迹。

第8步最大物体位移6.122 mm、转角5.245°，不能称为零漂移。图像MAE对整幅图求平均，背景会稀释局部物体变化；较小MAE不等于物理等价。未按漂移大小挑选样本，也没有追加稳定步数或修补个例。

## 验证、预算与交付

运行耗时{budget['elapsed_seconds']:.3f}秒，共128次初始化、1024物理步、0重试、0新下载字节、0训练更新、0模型前向。离线复核重新从保存接触计算Holding，并用冻结的原生几何评估器重算1152份物理记录；检查384份新观测哈希/同时刻proprio、768个候选事实格、2304项电机调用，全部通过。离线复核没有新仿真。原AB试验1271文件的封存哈希以及本轮预先冻结引用均未改变。

{table(['交付','内容及用途'],[
['[冻结协议](../../experiments/official_state_resimulation_20261018/configs/protocol.json)','步数、可靠性窗口、禁止动作与数据用途'],
['[独立验证](../../experiments/official_state_resimulation_20261018/results/verification.json)','核查项目与通过结果'],
['[运行预算](../../experiments/official_state_resimulation_20261018/results/run_budget.json)','实际初始化、物理步、重试和下载/训练数量'],
['[逐事实明细](candidate_facts.csv)','768格，源episode/帧、新图路径、真值/有效性及原因'],
['[逐episode统计](episode_summary.csv)','按来源及原有采样规则分组'],
['[逐颜色覆盖](color_coverage.csv)','正例数、正例来源episode数及ID'],
['[50帧旧In候选逐色对照](old_In_candidate_pairs.csv)','60个旧候选事实与新状态判定；旧历史GT仍未知'],
['[控制恢复明细](control_audit_summary.csv)','每次命令、目标、最大力、手臂目标变化'],
['[检查点指标](checkpoint_metrics.csv)','每个episode/时刻的位姿、图像差异与原始谓词计数'],
['[一次性执行源码](../../experiments/official_state_resimulation_20261018/source/run.py)','拒绝覆盖既有目录，不可当作追加采样命令'],
['[离线复核源码](../../experiments/official_state_resimulation_20261018/source/verify.py)','需原生executor环境；不重新运行物理'],
['[图表报告源码](../../experiments/official_state_resimulation_20261018/source/report.py)','只读取既有记录，无训练或仿真'],
['[交付哈希清单](../../experiments/official_state_resimulation_20261018/configs/delivery_manifest.json)','覆盖本轮数据、源码、报告及日志，清单自身除外']])}

数据路径为`experiments/official_state_resimulation_20261018/data/<root>/<frame>/`：保留初态参考、控制审计、8步积分台账、9份物理快照、3份新观测以及唯一第8步候选。重现命令和环境路径见[运行说明](RUN_COMMANDS.md)。

## 本轮可以与不能推出什么

可以确认：已下载的这4个官方A/B episode可作为新物理世界的初始化，按记录TCP和原生夹爪命令进行固定短时演化后，得到有接触证据的新图/新标签候选。原50帧In候选在新状态中全部有严格正事实，Holding也出现了有限正例。

尚不能确认：旧官方图的历史物理标签、长期抓持稳定性、颜色/episode覆盖充分、固定技能转移质量、正式模型改进或TS-LTL闭环成功。只有4个来源episode且粉色Holding只有1个来源，不能把相关帧当独立覆盖。**本次有界验证已结束；未开展额外采样、训练或正式数据采用，原指南的完整验收仍未完成。**
'''
 (R/'REPORT.md').write_text(text)
 (R/'RUN_COMMANDS.md').write_text('''# 运行与复核说明

以下是本轮已执行路径；数据已封存，不要再次执行一次性采样脚本。所有工作在本机完成，GPU阶段用项目锁串行。

```bash
source /home/robot/research/projects/wm_symbolic/env.sh
python /home/robot/Documents/Codex/2026-10-13/task-4/official_state_resimulation.py
python /home/robot/Documents/Codex/2026-10-13/task-4/resimulation_verify.py
source /home/robot/research/projects/wm_symbolic/world_model_env.sh
python /home/robot/Documents/Codex/2026-10-13/task-4/resimulation_report.py
```

执行/复核使用独立Python3.10原生仿真环境；图表使用Python3.11世界模型环境的NumPy/Matplotlib，不调用模型或CUDA。源码副本保存在本轮`source/`；`verify.py`会写复核JSON/CSV，`report.py`会写报告，因此如需再次复核应先复制到新审计目录，不能改动已封存交付。

一次性执行脚本带既有目录拒绝覆盖、128次/1024步预算、每个初态一次的台账和禁止初始化后干预的守卫。技术失败会保存记录并停止；本轮没有失败或重试。报告脚本仅展示已存数据。
''')
 shutil.copy2(__file__,E/'source/report.py')
 print('REPORT_CREATED',R/'REPORT.md',flush=True)
if __name__=='__main__':main()
