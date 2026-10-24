from common import *
import csv,html,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(FONT);plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':11,'axes.unicode_minus':False})
LABELS={'tabletop_closed':'桌面目标＋关闭','contained_closed':'完整收纳＋关闭','falling_now':'当前无支撑下降','fallen_below_table':'跌至桌面下方','protruding_supported':'承重但边界突出','near_boundary_supported':'距边界不足5mm'}
def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |'for row in rows)+'\n'
def figsave(fig,name):fig.savefig(R/'figures'/name,dpi=160,bbox_inches='tight');plt.close(fig)
def camera(ax,path,title):
 z=np.load(path);ax.imshow(z['rgb_static']);ax.set_title(title,fontsize=11);ax.axis('off');sub=ax.inset_axes([.70,0,.3,.3]);sub.imshow(z['rgb_gripper']);sub.axis('off')
def main():
 cov=read(E/'results/coverage.json');bind=read(E/'results/query_binding.json');physical=read(E/'data/_private/AB_object_state_audit.json');meta=read(BASE/'data/metadata.json');ix={r['window_id']:i for i,r in enumerate(meta)};wins={w['window_id']:w for w in read(BASE/'data/windows.json')};probs={a:np.load(SP/'results'/a/'probabilities.npy')for a in ['pooled','spatial']};protocol=read(E/'configs/protocol.json');decision=read(E/'configs/decision.json');cp=read(E/'results/existing_C_checkpoint_trace.json');interface=read(E/'results/interface_regression.json');typed=cov['coverage']['intended_target'];allq=cov['coverage']['all_object_queries'];rows=[]
 for key in LABELS:
  v=typed[key]['all_colors'];rows.append([LABELS[key],v['independent_roots'],v['object_windows'],f"{v['In_positive']} / {v['In_negative']} / {v['In_invalid']}",v['sampler_unique_windows_seen']])
 allrows=[]
 for c in COLORS:
  n=allq['tabletop_closed']['by_color'][c];p=allq['contained_closed']['by_color'][c];allrows.append([c,f"{n['object_windows']} / {n['independent_roots']}",f"{p['object_windows']} / {p['independent_roots']}",f"{p['object_windows']} : {n['object_windows']}"])
 # Count readout mistakes on the audited AB categories, without changing saved probabilities.
 category_metrics=[]
 for scope in ['intended_target','all_object_queries']:
  for key in LABELS:
   rr=[r for r in physical if r['categories'][key]and(scope=='all_object_queries'or r['is_intended'])and r['In_valid']]
   for arm in ['pooled','spatial']:
    fp=fn=0;values=[]
    for r in rr:
     p=float(probs[arm][ix[r['window_id']],3+COLORS.index(r['object'])]);values.append(p);fp+=int(p>=.5 and not r['In_truth']);fn+=int(p<.5 and r['In_truth'])
    category_metrics.append({'scope':scope,'category':key,'variant':arm,'valid_object_windows':len(rr),'In_FP':fp,'In_FN':fn,'min_probability':min(values)if values else None,'max_probability':max(values)if values else None})
 save(E/'results/coverage_readout.json',category_metrics)
 with (R/'tables/AB_category_readout.csv').open('w',newline='')as f:
  wr=csv.DictWriter(f,fieldnames=list(category_metrics[0]));wr.writeheader();wr.writerows(category_metrics)
 cats=list(LABELS)[:5];xx=np.arange(len(cats));fig,axs=plt.subplots(1,2,figsize=(13,5))
 rr=[typed[k]['all_colors']for k in cats];axs[0].bar(xx,[r['independent_roots']for r in rr],color='#488e98');axs[0].set_ylim(0,21);axs[0].set_title('独立根覆盖：不按帧数放大');axs[0].set_ylabel('根数（AB共18根）')
 for i,r in enumerate(rr):axs[0].text(i,r['independent_roots']+.3,str(r['independent_roots']),ha='center')
 pos=np.array([r['In_positive']for r in rr]);neg=np.array([r['In_negative']for r in rr]);bad=np.array([r['In_invalid']for r in rr]);axs[1].bar(xx,pos,color='#488e98',label='In真');axs[1].bar(xx,neg,bottom=pos,color='#d28e50',label='In假');axs[1].bar(xx,bad,bottom=pos+neg,color='#a2a8b2',label='标签不确定');axs[1].set_title('指定目标的有效正负组成');axs[1].set_ylim(0,100);axs[1].set_ylabel('目标窗口数（相关观测）');axs[1].legend()
 for i,r in enumerate(rr):axs[1].text(i,r['object_windows']+1,str(r['object_windows']),ha='center')
 for ax in axs:ax.set_xticks(xx,[LABELS[k].replace('＋','\n')for k in cats],rotation=20,ha='right');ax.grid(axis='y',alpha=.15)
 fig.text(.5,-.10,'左图青柱为独立根数；右图青／橙／灰为In真／假／不确定的目标窗口数。根会跨状态重复出现，不能相加为新案例。\n“收纳关闭”按真值定义为正例；全部关闭状态的公平正负组成另表列出。下降中的0窗口不表示原轨迹没有发生跌落。',ha='center',fontsize=10);figsave(fig,'AB_root_and_label_coverage.png')
 fig,axs=plt.subplots(1,2,figsize=(12,4.6));querylabels=['红','蓝','粉','均不在内','多个在内']
 for ax,arm in zip(axs,['pooled','spatial']):
  a=np.array(bind[arm]['object_identity_confusion']);im=ax.imshow(a,vmin=0,vmax=49,cmap='Blues',aspect='auto');ax.set_xticks(range(5),querylabels,rotation=15);ax.set_yticks(range(3),['红','蓝','粉']);ax.set_xlabel('预测In对象');ax.set_ylabel('唯一真实In对象');ax.set_title('池化对照'if arm=='pooled'else'空间保留')
  for i in range(3):
   for j in range(5):ax.text(j,i,str(a[i,j]),ha='center',va='center',color='white'if a[i,j]>25 else'#222')
 fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.02,pad=.03,label='AB窗口数：越深越多');fig.text(.5,-.09,'135个AB窗口恰有一个真实In对象；两组各134个完全选对，另1个多报红块（对应不同堆叠根），没有把真对象换成另一对象。\n这些是训练窗口统计；不是独立泛化结果。置换／别名通过仅排除已检查的索引连接错误，不能排除学到颜色或背景捷径。',ha='center',fontsize=10);figsave(fig,'AB_object_binding_confusion.png')
 fig,ax=plt.subplots(figsize=(9,4.6));colors={'pooled':'#488e98','spatial':'#6268a3'}
 for arm in ['pooled','spatial']:
  for frame,style in [('00063','-'),('00127','--')]:
   r=sorted([r for r in cp if r['variant']==arm and r['window_id'].endswith('__empty_closed__t'+frame)],key=lambda x:x['update']);ax.plot([x['update']for x in r],[x['probabilities']['In(pink,drawer)']for x in r],style+'o',color=colors[arm],label=('池化'if arm=='pooled'else'空间')+' 帧'+frame)
 ax.axhline(.5,color='#a85747',linestyle=':',label='冻结0.5读出阈值');ax.axvline(600,color='#aaa',linestyle=':',label='两组预定选点600');ax.set(ylim=(0,1),xticks=[300,600,900,1200],xlabel='已保存检查点的训练更新数',ylabel='In(粉块,drawer)概率',title='同一C根的既有误报随检查点变化');ax.legend(fontsize=9,ncol=2);ax.grid(alpha=.15)
 fig.text(.5,-.08,'青＝池化，紫＝空间；实／虚线是同根的两帧，真实In均为假。使用冻结权重CPU只读重算。\n不是新训练、不是种子方差，不能据此换检查点；原600步分数与门槛保持原样。',ha='center',fontsize=10);figsave(fig,'existing_error_checkpoint_trace.png')
 example=[]
 for case,variant in [('tabletop_closed','spatial'),('closed_inside','spatial'),('existing_C_false_completion','spatial')]:example.append(next(r for r in interface['cases']if r['case']==case and r['variant']==variant))
 fig,axs=plt.subplots(1,3,figsize=(13,4.8));titles=['AB：桌面负例','AB：正常收纳正例','已有C：桌面误报']
 for ax,r,title in zip(axs,example,titles):
  w=wins[r['window_id']];camera(ax,w['public_frames'][-1],title);o=r['output'];truth=r['GT_goal_for_offline_evaluation_only'];ax.text(.5,-.03,f"In粉={o['all_probabilities']['In(pink,drawer)']:.4f}；Closed={o['all_probabilities']['Closed(drawer)']:.4f}\nGT共同目标={truth}\n视觉报告目标={r['perceived_goal_claim']}",ha='center',va='top',transform=ax.transAxes,fontsize=10)
 fig.text(.5,-.08,'同一冻结空间网络实际查询；大图静态相机，右下腕部相机。前两例来自同一AB根的不同复位条件，不算两个独立根。\n共同目标＝In粉 AND Closed；GT仅离线评分。右图粉块在桌上，接口如实暴露虚假完成，未执行技能或宣称物理成功。',ha='center',fontsize=10);figsave(fig,'query_to_logic_examples.png')
 hyp=[['类别缺失或漏标','目标桌面关闭18根48窗口、全部被采样；420窗口标签重算一致','不支持“没有这种负例”或已发现漏标；细粒度布局／背景覆盖是否足够仍未确定'],['显式对象绑定错误','840视图颜色槽位、缓存、标签列一致；648AB窗口的6种置换和别名通过；切换目标不改神经概率','未发现已检查的索引、名称或目标列连接错误；不能排除学习到的背景／颜色捷径'],['训练随机性','仅seed0；同权重初始化和批次，已有检查点的C概率会跨过0.5','观察到训练过程依赖，无法估计种子方差或宣布随机性为唯一原因'],['“下降中”样本缺口','AB选定窗口0个下降中，已有落地12根24窗口','真实缺口，但既有蓝块误报帧也已落地，粉块误报帧在桌上，不能据此解释这些错误']]
 interface_rows=[]
 for r in interface['cases']:interface_rows.append([r['variant'],r['case'],r['GT_goal_for_offline_evaluation_only'],r['output']['interface_status'],r['false_goal_claim']])
 text=f'''# AB覆盖与对象查询绑定：限定核查

**未发现明确漏标或对象绑定实现错误；本轮停止继续模型调参，新增采集与训练均为0。** 指定目标“在桌上且抽屉关闭”的负例已覆盖全部18个AB根，不能把空间版同一C根的两个共同目标误报归结为这种类别缺失。细粒度外观覆盖、学习捷径和优化过程的贡献尚不可分辨；只有一个种子，不能宣称已证明训练随机性。

[首版至当前结果索引](RESULTS_INDEX.md) · [运行接口](README.md) · [图表浏览](gallery.html)

## 独立案例和正负覆盖

AB新语义数据为18个根、420个窗口；18个根的种子及初始布局哈希均不同。一个根下的不同复位条件、相邻帧、三个对象查询均保留为相关观测，不增加独立根数。旧AB228窗口只有Holding监督，未冒充In或抽屉覆盖。C本轮仅查看此前已记录的六个空间版错误窗口，没有复制C初态到训练。

![根数与标签](figures/AB_root_and_label_coverage.png)

{table(['指定目标的状态','独立根','目标窗口','In正 / 负 / 不确定','实际参与训练的窗口'],rows)}

每种颜色的指定目标在“桌面关闭”中各有6根16窗口；“收纳关闭”也各有6根，红／蓝／粉分别10／14／11窗口。18个根都同时提供了正、负关闭状态，因此这些根数不可相加。全部关闭状态的指定目标正负为35:48（正例42.2%）；两类实际采样出现1623和2513次，重复次数不等于独立数据量。

网络对每帧同时查询三对象，故还必须检查非指定目标。全部关闭状态共249个对象事实，正负52:197（正例20.9%）；每个颜色都有桌面负例，粉块并非漏掉监督：

{table(['被查询对象','桌面关闭负例：窗口 / 根','收纳关闭正例：窗口 / 根','关闭时In正 : 负'],allrows)}

上述正负来自冻结物理定义。桌面对象要求桌面基座向上承重、物体底面位于0.44±0.005米且未被持有；收纳要求完整几何包含、合格承重路径、未被持有。所有420个AB窗口从原始接触／姿态记录重新计算，标签值和有效掩码均与训练目标一致，没有改写旧标签。两组在197个桌面关闭负例、52个收纳关闭正例上的In判断都正确；这是训练覆盖检查，不能据此证明C泛化充分。

下降中的定义是当前未持有、无向上外部支撑且垂直速度小于−0.02米/秒，不能拿已落地帧代替。原AB连续轨迹发生过跌落，但入选训练窗口中的24个目标状态都已在桌面下方，当前下降窗口为0。本轮没有从同一轨迹补相邻帧再声称新增独立案例。

进一步复核此前蓝块`outside_front__t00063`误报：其底面约−0.016毫米、顶面39.991毫米，垂直速度约1.3e−11米/秒且有地面承重，已经落地。旧报告“滑落过程”描述不够精确，应理解为跌落事件来源，不能解释为该帧仍在空中。原结果保留，本报告补充澄清。

## 对象是否绑定错了

AB的840个当前相机视图重新提取红／蓝／粉区域，2520个对象颜色槽位与缓存边框完全一致。空间输入的局部patch、全局格子及bbox从既有缓存重新组装，差值均为0；420行训练标签顺序也一致。没有下载或重编码DINO。

两组在全部648个AB窗口（新420＋旧Holding228）上测试全部6种对象排列：对象历史、局部patch和bbox一起排列后，输出严格按同一排列变化，最大差为0。别名与查询顺序的最大差为5.96e−8，非法实体／关系参数被拒绝。程序显式目标只参与最后选择In对象列，不作为神经网络输入；同一画面分别指定红、蓝、粉目标的6次实际CLI查询，完整概率完全相同，目标列选择均正确。

![对象混淆](figures/AB_object_binding_confusion.png)

在135个AB“恰有一个对象In”的窗口中，两组各134个完全选择正确；各余1个在堆叠端点额外报了红块，实际蓝块仍被正确识别。两组出错根不同，没有把唯一真对象直接换成另一个对象。该结果支持没有显式槽位互换，但不排除模型利用颜色、背景或全局场景特征形成语义混淆。本接口仍是已知红／蓝／粉三实体与五种关系的有限模板，不是开放词汇对象识别。

## 能否判定这两次共同目标误报的原因

{table(['待检验解释','本轮证据','可以下的结论'],hyp)}

![既有检查点](figures/existing_error_checkpoint_trace.png)

空间版同一C粉块桌面根，在300步概率约0.10、600步约0.506–0.508、900步约0.19–0.20、1200步约0.89；池化版在900与1200步也出现高概率误报。两帧来自同一根，不是两次独立失败。这说明误报随拟合过程变化，不能只归结为“空间结构把目标绑错”；更不能据此切换检查点或抬阈值。图中使用CPU复核概率，旧GPU评分及600步选择结果原样保留。

没有建立“覆盖不足是唯一原因”的证据，因此不满足本次有条件启动补充AB/C采集的依据。单独补下降状态也不能验证当前桌面误报的原因。本轮固定停止，既不新增种子，也不延长训练。

## 当前可运行的查询到逻辑接口

新`query_to_logic.py`实际加载冻结网络，从保存的公开双相机特征／本体状态输出九事实，再把目标对象的In和Closed送入原有已学规则搜索。神经推理与搜索均不读取GT；下面的真值只在外部回归评分使用。原规则文件和全部unsupported标记有哈希保护，未重学或补写人工动态前提／效果。

![接口例子](figures/query_to_logic_examples.png)

{table(['模型','既有输入案例','GT共同目标','接口返回','是否虚报目标'],interface_rows)}

8次接口命令均实际执行，物理技能调用为0。没有可用算子时如实返回`no_plan_with_supported_learned_rules`；若视觉状态声称目标已满足，原搜索会返回空计划，本包装明确标为`perceived_goal_satisfied_unverified`。空间版C错误因此仍会产生一次视觉虚假完成，未被接口隐藏，也没有被称为物理成功。

当前已具备可运行的对象关系查询与逻辑目标接口；感知候选未通过门槛，动作失败覆盖不足及Close状态混淆仍使已学算子不可部署。这里不是RSSM供给TS后继或完整LTLf闭环验收。

## 为什么下一步需要更大范围的选择

若继续，应先决定主要目标是减少虚假完成、提高跨布局／环境泛化，还是推进动作状态充分性与失败覆盖。这三者对应不同的数据、读出／校准和规划状态改动，当前一组C错误无法决定哪一条值得投入。需要预先固定独立根的场景变化、正负与边界配额、必要种子比较及验收预算，再开展新的对照；不能持续拿既有C误报选阈值或加邻帧来追零错误。

原空间对照仍为202/209与203/209完整九事实，41个越界反例误报4与1，共同目标误报0与2，全部旧分数保留。本轮新训练0、新根0、新D0；原指南完整世界模型与规划项目仍未完成。
'''
 (R/'REPORT.md').write_text(text)
 readme=f'''# 当前接口与只读复核

[核查结论](REPORT.md) · [首版至当前索引](RESULTS_INDEX.md) · [图表](gallery.html)

下面命令运行冻结空间版，故意展示已知C误报如何进入逻辑目标判断，不执行动作：

```bash
source {P}/world_model_env.sh
python {E}/source/query_to_logic.py --variant spatial \\
  --window-id C_semantics_v2_fixtures_pink_0__empty_closed__t00127 \\
  --query 'In(obj_pink,drawer)' --goal-object pink
```

将`--variant`改为`pooled`可作同帧对照。可查询Holding(obj_red)、In(obj_blue,drawer)、Closed(drawer)、Open(drawer)、Intermediate(drawer)。实体别名仅在已声明三对象映射内有效；这是开发模型，不是已验收部署候选。

输出含九事实概率／离散值、目标对应列、原搜索结果、可用／不可用算子和物理调用数。`perceived_goal_satisfied_unverified`表示视觉模型称目标为真，不能当成真实成功。查询与搜索不读标签或私有仿真状态。

|位置|用途|
|---|---|
|`{E}/configs/protocol.json`|本轮限定核查口径|
|`{E}/configs/decision.json`|停止追加训练／采集的证据判断|
|`{E}/source/physical_audit.py`|原始AB物理记录重算标签和状态覆盖；需env.sh的SciPy|
|`{E}/source/binding_audit.py`|AB颜色缓存、标签列、别名／置换与既有错误检查；用world_model_env.sh|
|`{E}/source/query_to_logic.py`|当前可运行公开特征→共享查询→原规则逻辑接口|
|`{E}/results/interface_regression.json`|实际8次接口回归，包含保留的错误完成|
|`{E}/data/_private`|评分用物理审计数据；不进入查询推理|
|`{E}/results`|覆盖、绑定、既有检查点解释、接口输出|

只读哈希复核：`python3 {E}/source/verify_delivery.py`。原模型训练脚本有防覆盖检查；本轮不重跑训练。报告中的图表由保存数据生成，没有新增仿真画面。
''';(R/'README.md').write_text(readme)
 index=f'''# 对象关系查询项目：首版至当前结果索引

最后整理：2026-10-16。各行的语义、数据和根不同，不能串成一条同口径准确率曲线，也不能把开发C分数当独立D。所有旧评分与失败均保留。

|版本／工作|实际结果|关键限制与入口|
|---|---|---|
|查询最小实现首版|视觉＋学习规则的严格“完整入内且关闭”3/6根|[完整历史报告]({P}/reports/visual_predicator_minimal_20261015/REPORT.md)；旧七事实语义，6根与GT符号组配对|
|单项监控对齐|视觉组严格目标1/3根；未新增训练|同一历史报告，任务对象和训练口径不同，不与后续直接比率排名|
|冻结编码器迁移＋新共享查询头|视觉组严格目标5/6根；旧形式物理目标6/6不能称完全关闭|[视频／图表]({P}/reports/visual_predicator_minimal_20261015/gallery.html)；[原可运行查询]({P}/reports/visual_predicator_minimal_20261015/README.md)|
|九事实语义重建|完整包含＋承重路径，Closed/Open/Intermediate分开；新630窗口、27根、108技能调用|[语义报告]({P}/reports/state_semantics_v2_20261015/REPORT.md)；AB18根、C9根，C是开发集|
|九事实时序网络|C完整九事实165/209；41个越界反例全部误报|同一语义报告；未晋升，未新D|
|一次硬对比损失修正|仍165/209，越界反例仍41；出现Holding退步|[单次修正记录]({P}/reports/state_semantics_v2_20261015/tables/single_repair_comparison.json)，未继续无限修正|
|几何／可见性／线性探针诊断|空间patch探针C In有3FP、2FN；9次额外Close均关闭且物块均掉落|[诊断报告]({P}/reports/containment_diagnostic_20261015/REPORT.md)与[视频]({P}/reports/containment_diagnostic_20261015/gallery.html)；线性探针不是共享神经查询模型，也不是部署验收|
|同容量池化共享查询|C完整九事实202/209；原41反例余4FP；共同目标0FP|[空间对照报告]({P}/reports/spatial_shared_query_20261015/REPORT.md)；1200更新，固定选600，未过门槛|
|空间保留共享查询|C完整九事实203/209；原41反例余1FP；共同目标2FP，来自同一根|同一空间对照报告；全部In仍4FP、1FN；未晋升、未新D|
|当前AB覆盖／绑定核查|18根均有目标桌面关闭负例和收纳关闭正例；标签、颜色槽位及查询绑定检查通过|[本轮报告](REPORT.md)；当前下降窗口缺口不能解释已落地／桌面误报；新增训练与采集均0|
|当前查询→逻辑接口|8次实际离线接口回归，错误完成如实暴露；物理动作0|[运行命令](README.md)，原unsupported标记保持，未完成世界模型/LTLf验收|

先前基础证据另列，避免误归给当前查询网络：

- [早期执行器复核]({P}/reports/executor_review_v2/index.html)：冻结执行器与视觉终止监控的历史配对，不是当前共享查询模型成绩。
- [早期采集画面]({P}/reports/collection_preview_20261013/collected_data_preview_zh.html)。
- [已有正常动作与状态读出复核]({P}/reports/state_readout_review_20261015/REPORT.md)：旧候选在旧独立D的正常动作后果47/48；不能归给本轮空间版，更不能等同当前完整九事实或长序列规划。

当前所有工作保存在本机`{P}`，原robot_state与旧实验只读保留。原附件的完整CALVIN/HULC、RSSM世界模型供给TS后继、LTLf搜索和闭环独立验收仍未完成；这些局部结果不替代原指南验收。
''';(R/'RESULTS_INDEX.md').write_text(index)
 for name in ['coverage.json','feature_label_binding.json','query_binding.json','existing_C_checkpoint_trace.json','goal_binding_actual_queries.json','interface_regression.json','coverage_readout.json','existing_C_error_physics.json']:shutil.copy2(E/'results'/name,R/'tables'/name)
 shutil.copy2(E/'configs/decision.json',R/'tables/decision.json')
 pages=[('AB_root_and_label_coverage.png','AB根数与正负覆盖'),('AB_object_binding_confusion.png','对象绑定混淆'),('existing_error_checkpoint_trace.png','既有误报随检查点变化'),('query_to_logic_examples.png','当前查询到逻辑接口')]
 content='<!doctype html><meta charset="utf-8"><title>AB覆盖与查询绑定</title><style>body{max-width:1250px;margin:35px auto;font:17px system-ui;background:#f8f8f6;color:#23303a}img{width:100%;background:white}section{margin:32px 0}a{color:#266977}</style><h1>AB覆盖与查询绑定：限定核查</h1><p>未新增采集或训练；图表均来自保留记录。<a href="REPORT.md">报告</a> · <a href="RESULTS_INDEX.md">历次结果索引</a> · <a href="README.md">运行接口</a></p>'
 for file,title in pages:content+=f'<section><h2>{html.escape(title)}</h2><img src="figures/{file}" alt="{html.escape(title)}"></section>'
 (R/'gallery.html').write_text(content)
 print('COVERAGE_REPORT_AND_RESULTS_INDEX_GENERATED',flush=True)
if __name__=='__main__':main()
