from common import *
import shutil,csv,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False,'savefig.dpi':180})
ARMS=['object','pair','geometry'];NAMES={'object':'单对象 RGB','pair':'对象对 RGB＋二维几何','geometry':'仅二维几何'};COLORS_PLOT={'object':'#688bac','pair':'#188f79','geometry':'#cf8b38'}
def table(h,rs):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+'\n'.join('| '+' | '.join(str(x)for x in r)+' |'for r in rs)+'\n'
def main():
 install_guard([]);result=read(E/'results/evaluation.json');audit=read(E/'results/data_audit.json');gate=read(E/'results/data_gate.json');protocol=read(E/'configs/protocol.json');collection=read(E/'results/collection_summary.json');features=read(E/'results/features.json');training=read(E/'results/training.json');rows=read(E/'data/records.json');verify=read(E/'results/independent_verification.json');assert gate['passed']and len(training['fits'])==6
 def metric(arm,seed,kind='all',group='all',split='dev'):return next(r for r in result['metrics']if r['arm']==arm and r['seed']==seed and r['split']==split and r['group_type']==kind and r['group']==group)
 def ci(arm,seed,key):return next(r for r in result['root_bootstrap_CI']if r['arm']==arm and r['seed']==seed and r['metric']==key)
 fig,axs=plt.subplots(2,2,figsize=(14,10));fig.suptitle('正确原生框下的 Holding 读出：二维几何对照优于对象对视觉输入',fontsize=18,y=.985)
 for ax,key,title in zip(axs.flat[:3],['precision','recall','balanced_accuracy'],['A  精确率：预测为持有时，有多少是真的','B  召回率：真实持有中识别了多少','C  平衡准确率：正、负两类召回率的平均']):
  for i,arm in enumerate(ARMS):
   for seed,offset,marker in [(0,-.09,'o'),(1,.09,'s')]:
    v=ci(arm,seed,key);x=i+offset;ax.vlines(x,v['low95']*100,v['high95']*100,color=COLORS_PLOT[arm],lw=2);ax.plot([x-.025,x+.025],[v['low95']*100]*2,color=COLORS_PLOT[arm]);ax.plot([x-.025,x+.025],[v['high95']*100]*2,color=COLORS_PLOT[arm]);ax.scatter(x,v['estimate']*100,color=COLORS_PLOT[arm],marker=marker,s=65,zorder=3);ax.text(x,v['estimate']*100+1.5,f'{v["estimate"]*100:.1f}',ha='center',fontsize=9)
  ax.set_xticks(range(3),['单对象RGB','对象对RGB＋几何','仅几何']);ax.set_ylim(45,107);ax.set_ylabel('百分比（越高越好）');ax.set_title(title,fontsize=12);ax.grid(axis='y',alpha=.2)
 ax=axs[1,1];diff=[r for r in result['paired_differences']if r['metric']=='balanced_accuracy'];labels=[]
 for j,d in enumerate(diff):
  y=3-j;color='#688bac'if d['comparison']=='pair-minus-object'else'#cf8b38';ax.hlines(y,d['low95']*100,d['high95']*100,color=color,lw=3);ax.scatter(d['estimate']*100,y,color=color,marker='o'if d['seed']==0 else's',s=65,zorder=3);labels.append(f'对象对 − '+('单对象'if d['comparison']=='pair-minus-object'else'仅几何')+f' / 种子{d["seed"]}')
 ax.axvline(0,color='#333',ls='--');ax.set_yticks([3,2,1,0],labels,fontsize=10);ax.set_xlabel('配对平衡准确率差（百分点）');ax.set_title('D  相同根重采样：高于0才支持对象对更好',fontsize=12);ax.grid(axis='x',alpha=.2)
 fig.text(.5,.035,'开发集：C场景12个新根，470可用查询＝60正＋410负；固定阈值0.5。圆点为种子0，方点为种子1。\n线段为2,000次整根配对bootstrap的95%区间，不把连续帧/双相机当独立样本。蓝/绿/橙分别为单对象/对象对/仅几何。\n这是正确可见框条件诊断，没有证明额外视觉关系收益；没有最终测试、D数据或闭环部署。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.12,1,.95]);fig.savefig(R/'figures/01_readout_comparison.png');fig.savefig(R/'figures/01_readout_comparison.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(14,6));labels=[NAMES[a]+f' s{s}'for a in ARMS for s in [0,1]];groups=[('color','red'),('color','blue'),('color','pink'),('camera','static'),('camera','gripper')]
 for ax,key,title in zip(axs,['precision','recall'],['精确率（%）','召回率（%）']):
  mat=np.array([[metric(a,s,k,g)[key]*100 for k,g in groups]for a in ARMS for s in [0,1]]);im=ax.imshow(mat,vmin=40,vmax=100,cmap='YlGnBu',aspect='auto');ax.set_yticks(range(6),labels,fontsize=9);ax.set_xticks(range(5),['红色','蓝色','粉色','固定相机','腕部相机'],rotation=15);ax.set_title(title)
  for i in range(6):
   for j in range(5):ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',color='white'if mat[i,j]>78 else'black',fontsize=10)
  fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
 fig.suptitle('按颜色与相机检查误差：高召回仍伴随误报',fontsize=18,y=.98);fig.text(.5,.035,'每行是一个输入设置/种子，颜色越深表示数值越高；所有数值使用相同开发集合和0.5阈值。\n颜色组和相机组是两种重叠分组，不能相加当成独立数据；每色20正例，固定/腕部各30正例。完整分母、负例数与根数见metrics.csv。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.15,1,.92]);fig.savefig(R/'figures/02_color_camera.png');plt.close(fig)
 pred={}
 for arm in ARMS:
  for seed in [0,1]:
   z=np.load(E/f'results/predictions_{arm}_seed{seed}_dev.npz');pred[(arm,seed)]={int(i):float(p)for i,p in zip(z['record_indices'],z['probability'])}
 first=next(x for x in result['errors']if x['arm']=='pair'and x['seed']==0 and x['split']=='dev');conf=next(x for x in result['exact_input_conflicts']if x['split']=='dev'and x['input']=='geometry')['examples'][0];negative=next(i for i in conf['record_indices']if not rows[i]['y']);positive=next(i for i in conf['record_indices']if rows[i]['y']);cases=[first['record_index'],negative,positive];diagnostics=[]
 fig,axs=plt.subplots(1,3,figsize=(14,7.5));titles=['首个对象对错误：张爪接近','相同几何：桌面支撑下闭爪','相同几何：抬升承重']
 for ax,i,title in zip(axs,cases,titles):
  r=rows[i];d=E/'data/private'/r['root_id']/f's{r["sample"]:02d}';s=read(d/'physical_truth.json')['physical'];mapping=read(d.parent/'body_link_mapping.json');uid=s['objects']['block_'+r['color']]['uid'];finger=[0.,0.];external=0.
  for c in s['contacts']:
   if c[9]<=.01:continue
   if c[1]==uid:other,link=c[2],c[4]
   elif c[2]==uid:other,link=c[1],c[3]
   else:continue
   if other==mapping['robot_uid']:
    for j,family in enumerate(mapping['finger_families']):
     if link in family:finger[j]+=c[9]
   else:external+=c[9]
  with np.load(r['path'])as z:ax.imshow(z[r['key']],interpolation='nearest')
  for field,color in [('object_bbox','#00ffff'),('gripper_bbox','#ffb52e')]:
   bb=r[field];ax.add_patch(Rectangle(bb[:2],bb[2]-bb[0],bb[3]-bb[1],fill=False,edgecolor=color,lw=2))
  ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title+f'\n真实Holding={int(r["y"])}；'+('固定相机'if r['camera']=='static'else'腕部相机'),fontsize=12);ax.set_xlabel('\n'.join([f'{NAMES[a]}：{pred[(a,0)][i]:.3f} / {pred[(a,1)][i]:.3f}'for a in ARMS])+f'\n其他物体支撑接触法向力和：{external:.2f} N',fontsize=10);diagnostics.append({'record_index':i,'record_id':r['record_id'],'truth':r['y'],'finger_force_sums_N':finger,'external_contact_normal_force_sum_N':external,'predictions':{a:[pred[(a,s)][i]for s in [0,1]]for a in ARMS},'object_bbox':r['object_bbox'],'gripper_bbox':r['gripper_bbox']})
 fig.suptitle('高分不等于完整关系：相同二维框可以对应不同承重状态',fontsize=18,y=.98);fig.text(.5,.025,'青框为方块；橙框为两侧指尖可见像素并集。数值为预测概率（种子0 / 1），≥0.5即判为Holding。\n中、右两图的五维几何输入完全相同，真实标签相反，因此几何模型无法同时判对；接触力只用于私有评分，未输入模型。\n左图是按开发记录顺序的首个对象对误报。这里展示真实错误，没有追加训练、改阈值或挑选种子。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.16,1,.92]);fig.savefig(R/'figures/03_first_error_and_geometry_alias.png');plt.close(fig);save(E/'results/error_diagnostic.json',{'first_error':first,'cases':diagnostics,'geometry_conflict_group':conf,'interpretation':'Same geometry with opposite verified physical truth proves 2D geometry is not sufficient. First pair false positive is corrected by existing object and geometry controls, but differing heads prevent a causal claim that visual content itself is harmful. No extra fit or tuning.'})
 overall=[]
 for arm in ARMS:
  for seed in [0,1]:
   m=metric(arm,seed);v=ci(arm,seed,'balanced_accuracy');overall.append([NAMES[arm],seed,f'{m["precision"]*100:.2f}',f'{m["recall"]*100:.2f}',f'{m["balanced_accuracy"]*100:.2f}',f'[{v["low95"]*100:.2f},{v["high95"]*100:.2f}]',f'{m["NLL"]:.4f}',f'{m["TP"]}/{m["FP"]}/{m["FN"]}/{m["TN"]}'])
 cov=[]
 for split in ['train','dev']:
  for c in COLORS:
   x=next(r for r in audit['coverage']if r['split']==split and r['scope']=='all_actual_outcomes'and r['group_type']=='color'and r['group']==c);cov.append([split,c,x['positive_queries'],x['negative_queries'],x['positive_roots'],x['negative_roots'],x['unobservable']])
 gate_rows=[]
 for key,label in [('train_per_color','训练每颜色'),('dev_per_color','开发每颜色'),('each_camera_train','训练每相机'),('each_camera_dev','开发每相机')]:
  x=protocol['data_gates'][key];gate_rows.append([label,x['positive_records'],x['negative_records'],x['positive_roots'],x.get('negative_roots','未另设')])
 differences=[r for r in result['paired_differences']if r['metric']=='balanced_accuracy'];difftable=[[r['comparison'],r['seed'],f'{r["estimate"]*100:+.2f}',f'[{r["low95"]*100:+.2f},{r["high95"]*100:+.2f}]']for r in differences]
 group_table=[]
 for arm in ARMS:
  for seed in [0,1]:
   for kind,g in [('color','red'),('color','blue'),('color','pink'),('camera','static'),('camera','gripper')]:
    x=metric(arm,seed,kind,g);group_table.append([NAMES[arm],seed,g,x['positive_queries'],x['negative_queries'],f'{x["positive_roots"]}/{x["negative_roots"]}',f'{x["precision"]*100:.2f}',f'{x["recall"]*100:.2f}',f'{x["balanced_accuracy"]*100:.2f}'])
 report=f'''# 36新根：正确原生框条件下的Holding读出

**新数据覆盖门槛通过，六次预注册拟合已完成。对象对输入提高了召回率和平衡准确率，但仅二维框几何的对照更好，因此本实验不支持“额外视觉关系学习收益”的结论。** 这也不意味着视觉信息本身有害：三种头的输入维数、隐藏宽度不同，参数量接近但有效函数空间并不相同。几何信息本身亦不充分：有完全相同二维框而真值相反的稳定状态。

本次是已知A/B/C场景中的新初态配置证据。C开发集只用于事先冻结的数据门槛与最终固定评估，无开发调参；没有最终测试集。D样本、特征、预测和评分未打开。正确原生可见框是明确的oracle条件，不能声称视觉端到端、完整关系状态识别或闭环任务成功。

## 冻结方案与实际数据

采集前冻结protocol和全部6个模型初始化：36根精确尝试，训练A12/B12（各场景每色4目标根，训练每色共8），开发C12（每色4目标根）；每根最多8状态，总上限288，控制上限900，原控制器的至多两次尝试不改。种子261019500…261019535，所有状态/颜色/相机随根同划分。完整根列表及决定规则见 FROZEN_PROTOCOL.json。

采样沿用已验证的场景相对x比例0.18/0.50/0.82与既有抖动、y/z/yaw、抽屉开度和动作锚点；12mm/2mm几何间隙、物理监测、标签和观测条件不变。所有根只执行一次，无替换根、无筛到成功。实际30根成功抓取，6根几何候选拒绝（训练3、开发3），共 **264状态、528主图、1,584条查询**。失败根和没有到达的阶段如实保留。

原失败32根与8根工程试采只作为已封存工程证据，没有并入新数据。前者训练正例红2/蓝3/粉4、门槛失败的结论保持不变；后者不再被称为未见测试。新根通过已有ID/种子、初始化、1mm邻近布局及历史文件/RGB哈希排除。原封存文件最终完整复核，验证记录见results/final_verification.json（交付封存前生成）。历史去重只覆盖已登记的历史元数据，不能覆盖未登记数据。

{table(['划分','颜色','可用正查询','可用负查询','正例独立根','负例独立根','端点缺失查询'],cov)}

原可靠性与正负覆盖门槛均未放松：

{table(['范围','最低正查询','最低负查询','最低正例根','最低负例根'],gate_rows)}

训练正例根每色7≥4，开发每色3≥1；相机覆盖门槛也全部通过。完整门槛结果data_gate.json中passed=true、failures=[]。这里是读出实验可行性门槛，不是原世界模型指南的最终验收标准。

Holding真值仍为双指尖家族承重接触>0.01N、无其他物体支撑，最近12个真实控制步及0.005/0.01/0.015N三个阈值一致且无多持有矛盾。可见条件为方块及每侧指尖家族各≥4像素；不可见弃权，不当负例。1,584物理查询全部通过稳定性，194因端点不可见而排除：训练160、开发34。可用训练920＝160正＋760负，开发470＝60正＋410负。所有物理正例均通过可见条件，因此本批全物理分母正例召回与可用集合召回相同；这不证明未来不可见正例不存在。

三台？本实验始终只有**两台相机**：固定和腕部。每个阶段在同一物理tick取两张主图并做两次flags关闭对照；渲染前后完整状态/接触相同，RGB/深度逐位相等。审计重算264次真值、1,584个原生框、9,504次历史接触阈值和14,344个控制步的Holding。没有旧RGB或跨划分重复，详情data_audit.json。状态、力和原生mask只供控制/私有标注。

## 相同预算下的三个输入条件

{table(['设置','实际模型输入','头与参数量','说明'],[[NAMES['object'],'方块紧框RGB编码128维','128→67→1，8,711','框内可能有背景/遮挡者，不是实例mask抠图'],[NAMES['pair'],'相同方块128＋两指尖并集紧框128＋相对几何5','261→33→1，8,680','指尖并集的矩形包含两指之间区域'],[NAMES['geometry'],'仅相对二维框几何5','5→1244→1，8,709','不读取任何RGB；用来检查框几何是否已解释收益']])}

所有头都是一层SiLU隐藏层，最多差31参数（约0.36%）。同初始化种子与标准Linear规则，但输入维度/宽度/fan-in不同，不能声称初始权重完全相同或有效容量完全等价。没有试其他结构挑结果。

两种视觉条件共用缓存冻结DINOv2-small：每个紧框RGB独立双线性缩至224，ImageNet归一化，patch均值384维，再用事先保存、数据无关的正交投影得到128维；不输入全景tokens。编码器hash保持 `3a697b872089933d345fd71087af86e85ff31886cf1e2654763dc1884ef2af27`，编码1,917个唯一裁剪，GPU峰值约{features['peak_GPU_bytes']/1024**2:.1f}MiB。相对几何是方块中心减指尖并集中心（按图像宽高归一）、框宽高比的对数、IoU，共5维。没有3D坐标、接触、robot_obs、动作、阶段、根ID、颜色名或真值作为特征。

训练仅用920条可用训练记录的均值/总体标准差（下限0.05）归一化；对象/几何维度统计与对象对对应维度完全相同。每种子共用同一抽样序列：均匀选训练根，再选根内查询；batch32，每头600更新，AdamW lr0.001、weight_decay0.001、clip1，BCE正权重760/160=4.75。两种子0/1，共**6次×600更新**；float32、确定性、TF32关闭，无增强/AMP/早停。只用最终600步权重、阈值固定0.5，不调参、不挑种子。训练进程禁止读取开发标签/特征和私有标注；预测进程只读锁定模型与数值特征；评分进程之后才读取真值。

## 开发集结果

精确率TP/(TP+FP)，召回TP/(TP+FN)，平衡准确率=(正例召回+负例召回)/2。分母为0则记null，不偷偷填0；NLL为未加权二元负对数似然，概率裁剪到[10⁻⁷,1−10⁻⁷]。以下全部为同一470可用查询，12开发根中9有正例、12有负例；34不可见查询单独报告。

{table(['输入设置','种子','精确率%','召回率%','平衡准确率%','平衡准确率95%区间','NLL','TP/FP/FN/TN'],overall)}

对象对相对单对象在两个种子上提高召回和平衡准确率，但精确率/NLL并非每个种子都更好；不能概括成所有指标均提升。几何单独输入在两种子上的精确率、召回、平衡准确率与NLL都优于对象对。尤其几何精确率仍仅77.9–78.9%，高平衡准确率不等于没有Holding误报。

不确定性用12整个开发根配对重采样2,000次，同一抽样用于所有设置和种子，重新计算汇总混淆计数与NLL；报告95%百分位区间，不把双相机/连续状态当独立样本。每项2,000次均有定义。没有多重比较校正；12根的开发结果不能替代新的最终测试。

{table(['配对比较','种子','平衡准确率差（百分点）','根级95%区间'],difftable)}

这些差异支持“对象对比单对象更能识别本批持有状态”，但**不支持额外RGB视觉信息优于二维几何**。几何头宽得多，即使总参数接近也会影响函数表达，不能进一步推断视觉信息本身造成了性能下降。

## 颜色、相机、根覆盖及误差

{table(['设置','种子','分组','正查询','负查询','正/负根','精确率%','召回率%','平衡准确率%'],group_table)}

完整逐根、颜色×相机交叉分组、各指标分母、弃权覆盖以及训练结果均在metrics.csv；逐条错误在errors.csv，逐根采集状态和安全检查在root_outcomes.csv/control_safety.csv。分类错误不会改变该根采集结果或真值。

按开发记录顺序，第一个对象对种子0错误是 `relation36_C_24_s02_static_red`：张爪接近红块，真实Holding=0，对象对给0.660（种子1为0.772）；同记录单对象0.433/0.380，几何0.152/0.208，均判负。接触标注已重新核验，端点可见。现有对照改变了输入设置及头宽度，因此只能定位为本次对象对读出的误报，不能单凭这个例子给出某个视觉特征的因果解释。本轮未做追加修正。

更明确的表示限制来自完全相同的几何输入：C_24红块腕部视角在桌面支撑下闭爪（Holding=0）与抬升/稳定/平移（Holding=1）具有相同方块框[34,5,51,37]和指尖并集框[14,1,70,27]，五维几何逐位相同。C_34蓝块还有同类冲突。开发共2组8查询，训练也有2组10查询。既定几何模型面对相同输入必须给相同预测，因此不能同时判对这些相反真值；它不是Holding的充分判据。

几何种子0全部16个误报来自桌面支撑闭爪12个、释放近桌面4个；种子1再增加1个张爪接近误报。对象对两种子也各有9个闭爪支撑误报。所有条件都保留，不为了提高分数删除歧义样本。精确裁剪RGB/视觉特征没有发现相反标签的逐位完全相同输入，但这不证明所有承重关系都能从静态RGB唯一识别。错误证据图及私有力摘要见figures/03_first_error_and_geometry_alias.png与results/error_diagnostic.json。

## 验证、边界与交付

独立实现复核了360行指标；另将60次根重采样展开成实际重复记录，核对计数与NLL；逐条重算1,390行二维几何，核对训练专用归一化、三组共享抽样序列、每次恰600步。没有为验证重新训练或重新预测，结果见independent_verification.json。六个最终模型先锁定，再各对train/dev执行一次预测，模型hash随后保持不变。

本轮没有In正例训练、定位器优化、世界模型、SupCon或闭环部署。高分使用正确对象身份和可见框，且来自固定桌面抓取/释放分布；相近框与Holding高度相关，又有已证实的同框异真值，不能把读出头接入规划后宣称解决关系状态。整个6000转移/世界模型实际供给TS与LTLf搜索/4–8技能闭环指南尚未完成。

目录 `{E}`；报告 `{R}`。configs保存采集前完整协议、历史排除、随机投影、初始权重摘要、训练归一化/抽样序列与冻结清单；data/public仅RGB，data/private为接触/状态/mask/轨迹；data/model_inputs.json只含RGB/框元数据，supervision分开train/dev目标；models含6初始与6最终模型；results含逐次预测、评分和完整性证据。首50与128状态存储记录保留；最终数据原始目录约{collection['data_bytes']/1024**2:.1f}MiB，特征缓存另计。运行命令见RUN_COMMANDS.md。

图1颜色蓝/绿/橙分别单对象/对象对/几何；圆/方为种子0/1，纵轴百分比，线段为整根95%区间；右下是配对平衡准确率差，0虚线区分差异方向。图2各行设置与种子、列为颜色或相机，深色数值更高；两套分组重叠，不能相加。图3青框方块、橙框两指尖可见像素并集，概率按种子0/1给出，力仅作私有评分说明。所有图都包含样本条件与实际含义。
'''
 report=report.replace('三台？本实验始终只有**两台相机**','本实验只有**两台相机**')
 (R/'REPORT.md').write_text(report)
 (R/'RUN_COMMANDS.md').write_text(f'''# 实际运行顺序
```bash
source {P}/world_model_env.sh
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/relation36_src/prepare.py
source {P}/env.sh
python -u {E}/source/collect.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/relation36_src/audit.py
source {P}/world_model_env.sh
python -u /home/robot/Documents/Codex/2026-10-13/task-4/relation36_src/features.py
python -u /home/robot/Documents/Codex/2026-10-13/task-4/relation36_src/train.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/relation36_src/predict.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/relation36_src/score.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/relation36_src/verify_results.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/relation36_src/report.py
```
采集、编码、训练串行使用GPU锁；CPU评分/报告不占用CUDA。prepare/collect/features/train/predict/score均防覆盖或防重复运行。没有额外采样、拟合或调阈值。旧封存目录只读。
''')
 for f in Path(__file__).parent.glob('*.py'):
  dst=E/'source'/f.name
  if not dst.exists():shutil.copy2(f,dst)
  else:assert sha(dst)==sha(f),'Frozen source mismatch: '+f.name
 save(R/'OUTCOME.json',{'utc':now(),'roots':36,'states':264,'gate_passed':True,'fits':6,'updates_per_fit':600,'dev_eligible_queries':470,'dev_positive_negative':[60,410],'main_finding':'Pair improves recall/BA over object, but geometry-only exceeds pair in both seeds. No extra visual relation-learning benefit established. Exact geometry aliases with opposite Holding truth remain.','old_cohorts_merged':False,'D_samples_read':False,'guide_complete':False})
 print('REPORT_COMPLETE',R/'REPORT.md',flush=True)
if __name__=='__main__':main()
