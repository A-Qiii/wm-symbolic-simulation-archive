"""CPU scientific figures and actual camera videos for state-semantic development."""
from common import *
import sys,gzip,collections,subprocess,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
R=P/'reports/state_semantics_v2_20261015';FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(FONT);plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':11,'axes.unicode_minus':False})

def figsave(fig,name):fig.savefig(R/'figures'/name,dpi=160,bbox_inches='tight');plt.close(fig)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |'for row in rows)+'\n'
def frame(ax,path,title):
 z=np.load(path);ax.imshow(z['rgb_static']);ax.axis('off');ax.set_title(title,fontsize=10);sub=ax.inset_axes([.7,0,.3,.3]);sub.imshow(z['rgb_gripper']);sub.axis('off')

def main():
 for n in ['figures','videos','tables','source']:(R/n).mkdir(parents=True,exist_ok=True)
 d=read(E/'results/development_evaluation.json');audit=read(E/'results/data_audit.json');train=read(E/'results/training.json');capacity=read(E/'results/placement_capacity_audit.json');alias=read(E/'results/close_state_aliasing.json');rules=read(E/'results/rule_learning9.json');windows=read(E/'data/windows.json');metadata=read(E/'data/metadata.json');pro=np.load(E/'results/probabilities.npy');ix={r['window_id']:i for i,r in enumerate(metadata)};episodes=read(B/'manifest.json')['episodes'];lookup={r['episode_id']:r for r in episodes}
 # Operational thresholds, not learned decision boundaries.
 fig,ax=plt.subplots(figsize=(12,3));ax.broken_barh([(0,2)],(.8,.6),facecolors='#4b79b0');ax.broken_barh([(2,16)],(.8,.6),facecolors='#d3a649');ax.broken_barh([(18,5)],(.8,.6),facecolors='#278776');ax.broken_barh([(0,16)],(0,.4),facecolors='#b9bdc3');ax.broken_barh([(16,7)],(0,.4),facecolors='#78a599')
 for x in [2,18]:ax.axvspan(x-.3,x+.3,color='#666666',alpha=.25);ax.axvline(x,color='#333',ls='--',lw=1)
 ax.text(1,1.1,'Closed',ha='center',color='white');ax.text(10,1.1,'Intermediate',ha='center');ax.text(20.5,1.1,'Open',ha='center',color='white');ax.text(8,.2,'旧 ¬Open：不能等同 Closed',ha='center');ax.text(19.5,.2,'旧 Open',ha='center');ax.set_yticks([.2,1.1],['旧标签表达','新版三状态']);ax.set_ylim(-.15,1.6);ax.set_xlim(0,23);ax.set_xlabel('实际抽屉关节开度 q（cm）');ax.set_title('关闭与非打开分开：半开受阻必须保留为中间状态');fig.text(.5,-.05,'蓝：关严；金：中间；绿：充分打开；灰色竖带为预先固定的±0.3cm监督不确定区。该图表示标签定义，不是模型拟合曲线。',ha='center',fontsize=10);figsave(fig,'drawer_semantics.png')
 # Show retained genuine physical counterexamples from C, using actual realized labels.
 selected=[]
 for condition in ['single_inside','outside_front_supported','stacked']:
  pool=[r for r in windows if r['split']=='dev'and r['condition']==condition and r['anchor']==127];selected.append(pool[0])
 selected.append(next(w for w in windows if w['split']=='dev' and not all(w['targets_v2_1']['v'][3:6])))
 fig,axs=plt.subplots(1,4,figsize=(14,4.8))
 for ax,w,title in zip(axs,selected,['完整在内并承重','外侧接触反例','间接支撑但突出','边界例保留不确定']):
  target=(next(j for j in range(3) if not w['targets_v2_1']['v'][3+j]) if not all(w['targets_v2_1']['v'][3:6]) else COLORS.index(next(r['color']for r in read(B/'roots_manifest.json')if r['root_id']==w['root_id'])));c=COLORS[target];lab=w['targets_v2_1'];obj=lab['objects'][c];old=w['old_labels']['categorical'][target]==3;truth=str(bool(lab['y'][3+target]))if lab['v'][3+target]else'不确定';frame(ax,w['public_frames'][-1],title);ax.text(.5,-.04,f"目标{c}；旧In={old}\n新In={truth}\n边界余量{obj['minimum_side_rim_clearance_m']*1000:.1f}mm",ha='center',va='top',transform=ax.transAxes,fontsize=10)
 fig.text(.5,-.16,'所有画面均为C的真实控制帧，右下为腕部相机；文字中的GT只用于离线标签审核，不输入视觉模型。负余量表示至少一个角点越界。',ha='center',fontsize=10);figsave(fig,'physical_counterexamples.png')
 fig,ax=plt.subplots(figsize=(14,5));xx=np.arange(len(capacity));vals1=[x['branches']['stacked']['end_q']*100 for x in capacity];vals2=[x['branches']['separated']['end_q']*100 for x in capacity];ax.plot(xx,vals1,'o-',color='#b75643',label='同物块堆叠');ax.plot(xx,vals2,'s-',color='#278776',label='同物块并排');ax.axhline(2,color='#333',ls='--',label='Closed阈值2cm');ax.set_xticks(xx,[x['root_id'].replace('_semantics_v2_fixtures','')for x in capacity],rotation=65,ha='right');ax.set_ylabel('固定Close执行后开度（cm）');ax.set_title('相同执行器与物块：摆放方式对能否关严的影响');ax.legend();ax.grid(axis='y',alpha=.2);fig.text(.5,-.23,'每个横轴项是一组共同根的两次复位对照，红色堆叠、绿色并排。折线仅连接同类条件，横轴不是时间；未增大推力或修改控制器。',ha='center',fontsize=10);figsave(fig,'placement_vs_closure.png')
 modelnames=[('new_query9','新版九事实'),('previous_query7_with_notOpen_closed','前版共享七事实\n按旧语义迁移'),('legacy7_with_notOpen_closed','旧冻结读出\n按旧语义迁移')];fig,axs=plt.subplots(1,3,figsize=(14,4.5));cols=['#278776','#c28c50','#858e99']
 for j,(key,name)in enumerate(modelnames):
  r=d['metrics'][key]['dev:new_semantics'];h=d['metrics'][key]['dev:legacy_holding_only'];vals=[r['all9_correct_fully_valid']/max(1,r['fully_valid9']),r['facts']['Closed(drawer)']['FP'],r['facts']['In(red,drawer)']['FP']+r['facts']['In(blue,drawer)']['FP']+r['facts']['In(pink,drawer)']['FP']]
  for k,v in enumerate(vals):axs[k].bar(j,v,color=cols[j]);axs[k].text(j,v+.02 if k==0 else v+.5,(f"{r['all9_correct_fully_valid']}/{r['fully_valid9']}"if k==0 else str(v)),ha='center',fontsize=10)
 for ax,title,ylabel in zip(axs,['C完整九事实正确','C未关闭却报Closed','C不在内却报In'],['正确率（仅9项均确定窗口）','错误窗口数','错误对象事实数']):ax.set_xticks(range(3),[name for _,name in modelnames]);ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.15)
 axs[0].set_ylim(0,1.13);fig.text(.5,-.08,'三者使用同一批公开图像与新版标签。两个旧模型无法表示Intermediate，Closed沿用旧错误取反方式作迁移对照；不声称模型数据或表达能力公平。',ha='center',fontsize=10);figsave(fig,'development_readout_comparison.png')
 # A matched C pair shows the physical interference evidence and model output, not online control.
 pair=next((r for r in capacity if r['split']=='dev'and r['stack_obstructed_separated_closed']),next(r for r in capacity if r['split']=='dev'));selected_pair=[]
 for case in ['stacked','separated']:
  eid=pair['branches'][case]['episode_id'];selected_pair.append(max([w for w in windows if w['episode_id']==eid],key=lambda w:w['anchor']))
 fig,axs=plt.subplots(1,2,figsize=(9,5))
 for ax,w,case in zip(axs,selected_pair,['堆叠','并排']):
  p=pro[ix[w['window_id']]];lab=w['targets_v2_1'];frame(ax,w['public_frames'][-1],f"{case}：真实q={lab['drawer_q_m']*100:.2f}cm");ax.text(.5,-.05,f"GT {lab['drawer_class']}\n模型 P(Closed/Open/Mid)\n{p[6]:.3f} / {p[7]:.3f} / {p[8]:.3f}",transform=ax.transAxes,ha='center',va='top')
 fig.text(.5,-.15,'同根、同物块，改变初始摆放并分别真实执行固定Close。模型只离线读取公开观测；图中文字GT为评估对照，未用于补答案。',ha='center',fontsize=10);figsave(fig,'matched_C_endpoints.png');save(R/'tables/illustrated_pair.json',pair)
 # All per-class metrics and exact gate result remain external references and a concise standalone copy.
 save(R/'tables/development_gate.json',d);save(R/'tables/placement_capacity_audit.json',capacity);save(R/'tables/close_state_aliasing.json',alias);save(R/'tables/data_audit.json',audit);save(R/'tables/rule_learning9.json',rules)
 pr=[]
 for split,z in audit['splits'].items():pr.append([split,z['roots'],z['windows'],z['valid9'],z['positive_valid'],z['invalid_by_fact']])
 mr=[]
 for key,name in modelnames:
  z=d['metrics'][key]['dev:new_semantics'];h=d['metrics'][key]['dev:legacy_holding_only'];mr.append([name.replace('\n',' '),f"{z['all9_correct_fully_valid']}/{z['fully_valid9']}",z['facts']['Closed(drawer)']['FP'],sum(z['facts'][f]['FP']for f in FACTS[3:6]),f"{h['Holding_all3_correct']}/84"])
 gate=table(['预先冻结门槛','结果'],[[k,'通过'if v else'未通过']for k,v in d['gate_checks'].items()]);n_pairs=sum(r['stack_obstructed_separated_closed']for r in capacity);unsupported=[r['family']for r in rules['audit']if not r['supported']]
 text=f'''# 状态语义第二版：AB/C审计、一次训练与开发结论

本轮没有新增D验收，先完成状态定义和开发验证。预先冻结门槛决定：**{'允许再评估一次独立验收的可行性'if d['new_independent_acceptance_permitted_by_frozen_gate']else'不进入新的独立验收'}**。开发结果与未通过原因见下表。所有失败、原标签和首次构造失败均保留，固定执行器、推力、碰撞和动力学未修改。

上一轮“6/6”仅代表旧形式目标In∧¬Open；严格物块完整在内且抽屉关至2cm内是5/6，粉色双物体场景仍开11.24cm。GT诊断组也不应把半开报告为真正关闭。该更正已在[上一轮报告]({P}/reports/visual_predicator_minimal_20261015/REPORT.md)明确记录，本轮不覆盖任何旧分数。

## 先定义状态，再训练视觉查询

新版九事实为Holding×3、In×3、Closed、Open、Intermediate。Closed不是Open的取反；每个都由共享查询模型给出。完整[操作性规则与容差]({E}/configs/LABEL_RULES.md)和采集前哈希保存在configs。

![抽屉状态定义](figures/drawer_semantics.png)

横轴是真实抽屉开度cm；蓝色Closed≤2cm、金色Intermediate、绿色Open≥18cm。两阈值附近±0.3cm为监督不确定区，单独计数。模型输入没有开度GT；它从双相机RGB与本体状态识别三个互斥状态。

In要求真实八角点完整位于抽屉内腔，具有向上承重路径且非Held；支持它的可移动物体也须完整容纳。原生抽屉link局部范围x[-.18,.18]、y[.02,.08]、z[-.04,.16]米，局部y是竖直方向。侧壁／上沿有符号余量≥2mm确定在内，≤−2mm确定在外，中间为不确定；底面允许3mm碰撞数值容差。承重边要求力>0.01N、世界向上法向z>0.5、间距≤3mm。完整强路径（每边≥0.015N且法向≥0.55）可明确认证，弱平行接触不会推翻它。所有物体位姿和接触仅作监督／评分，不给视觉端。

![实际反例](figures/physical_counterexamples.png)

四格是C真实相机帧（右下腕部相机）；GT文字只用于审核。突出边沿但经另一物块支撑不再被算作完整In；几何边界样本没有删除，而是报告确定性掩码。悬空夹持样本来自真实Lift／Place下降过程中，非合成渲染。

A首16回合审计暴露两点：支撑掩码把已有强支撑路径也被弱平行接触置为不确定；外侧前沿初次构造会滑落。只在A上修正实现为v2.1，另加outside_front_supported复位分支；原v2与原滑落尝试保留。v2.1及模型训练设计在C结果查看前冻结。原始完整物理trace可重算标签，v1、v2、v2.1差异单独留档，未把改标签当新模型得分。

## 数据与因果执行检查

本轮 {audit['episodes']} 个真实回合、27个根组、{audit['calls']}次固定技能调用、{audit['windows']}个因果四帧窗口。AB18根组训练，C9根组开发；同根的各复位条件不是独立根，也不是一次物理转移。每帧保存完整接触力／法向／body与link、物块位姿、抽屉变换与关节量，实际240Hz物理、30Hz控制。首批约50窗口存储测量、128窗口输入／划分检查均保留。

{table(['划分','根组','窗口','九项都确定','各事实有效正例数','各事实不确定数'],pr)}

旧312窗口只复用Holding监督，In及三种抽屉监督均掩掉；它们没有被缺证据地重标。新模型的输入仍是公开RGB的冻结DINO特征、颜色对象区域与15维本体状态；drawer/gripper仍为场景类型锚点，不能称精确分割节点。没有开放词汇类别、没有遮挡记忆模块。

## 关闭受阻与摆放的对照

![摆放对照](figures/placement_vs_closure.png)

{len(capacity)}个共同根各以堆叠／并排复位，同样物块和固定Close执行器；其中{n_pairs}组表现为堆叠未关严、并排关严。图中红色堆叠、绿色并排，横轴为不同根组、纵轴为最终开度；折线不是时间。接触审核进一步保存抽屉中物块与非drawer固定link碰撞的首次时刻、峰值力、抽屉开度和穿透量。没有靠增加力或移动GT状态强行完成。

![C对照端点](figures/matched_C_endpoints.png)

这是摆放／阻塞限制的物理证据；不能把所有失败都归咎于视觉。完整边界以内的并排物块能容纳，并不表示固定Place技能会选择并排位置：它仍往同一中心放，可能形成突出堆叠。当前不改控制器，失败必须显示未Closed。

## 一次共享九事实查询训练

冻结既有对象／场景编码器和GRU，丢弃旧分类头，训练五种关系类型的共享查询头。参数 {train['trainable_parameters']}，seed0、1200更新；300/600/900/1200检查点全部保留，按预先指定C根均值损失＋0.2旧C Holding损失选择，选中{train['selected']['update']}更新。每批32，16个旧AB Holding样本与16个新AB样本，分别按根均匀抽样。Holding与In采用AB逆频率BCE，抽屉三分类采用AB逆频率CE，三组等权；旧样本缺失标签不参与对应损失。编码器权重前后逐项相同。

![开发读出对比](figures/development_readout_comparison.png)

{table(['读出','C确定窗口完整九事实','Closed误报数','In误报事实数','旧C Holding完整三事实'],mr)}

以上是开发集成绩，不是盲测。两个旧模型不能输出Intermediate，表中将其旧¬Open当Closed的错误迁移方式明确展示，用来揭示语义差异，不能解释为架构公平竞赛。所有有效事实均评分，边界窗口的有效率与未确定事实数同时公布。

{gate}

未通过的门槛不降低，未追加训练种子、超参数搜索或新D试验。实际反例逐类的FP、覆盖根数、各对象TP/TN替代计数和错误帧保存在results/development_evaluation.json与development_errors.json。

## 学习动作规则与状态是否足够

成功调用只提取真实九事实增删效果；失败／拒绝为同技能负例，绝不伪造成功效果。AB按技能和参数角色聚类，正前态求文字交集，再找最短精确区分合取；允许否定、固定派生EmptyHand/AnyIn/TwoOrMoreIn，不能按物体颜色记忆失败。不确定转移保留但不能强制当确定训练证据。{rules['eligible_complete_before_after9']}/{rules['raw_calls']}调用前后九事实都确定。

未获支持的技能簇：{', '.join(unsupported)or'无'}。九事实虽能区分“真的Closed”和“只是不Open”，但不必然足够预测Close能否成功。已把Holding数、In数和抽屉三状态相同、物体身份可置换的前态分组；若既有Close成功又有受阻，则不能从这些状态发明确定的失败前提。见tables/close_state_aliasing.json和rule_learning9.json。无精确分隔的规则标记unsupported，搜索不可调用，不用手写后继兜底。

本轮采集动作由冻结脚本预先安排，模型推理发生在保存的公开窗口上；因此不能称为新版视觉规划闭环已验收。查询CLI可以检查Closed/Open/Intermediate及Holding/In，学习规则搜索会如实返回不支持或无计划，不偷偷用GT替换。

## 结论与交付边界

已完成操作性语义、带完整证据的AB/C反例数据、一次新查询训练、迁移对照、受阻物理审计以及保留失败的开发门槛。新D仍为0。继续提高链路可靠性需要由证据决定：是视觉读出仍不足，还是状态缺少突出／阻挡信息，或固定Place无法选择可容纳位置。当前不借增大推力、降低成功定义或补GT答案消除失败。

[运行说明](README.md) · [图表和真实视频](gallery.html)。所有技术配置与数据在`{E}`和`{B}`。旧报告、旧模型、旧数据保留，未宣称原完整世界模型方案完成。
'''
 (R/'REPORT.md').write_text(text);save(R/'tables/report_summary.json',{'development_gate':d['gate_checks'],'new_D_runs':0,'real_episodes':audit['episodes'],'root_groups':27,'windows':audit['windows'],'skill_calls':audit['calls'],'training_updates':1200,'paired_roots':len(capacity),'stack_failed_separated_closed':n_pairs,'selected_model':train['selected']['path']});print('REPORT_WRITTEN',str(R),flush=True)
if __name__=='__main__':main()
