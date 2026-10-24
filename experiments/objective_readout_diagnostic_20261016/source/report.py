from config import *
import csv,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'font.size':10.5,'axes.unicode_minus':False,'svg.fonttype':'none'})
S=read(E/'results/latent_summary.json');G=read(E/'results/gradient_records.json');CP=read(E/'results/checkpoint_pairs.json');CS=read(E/'results/checkpoint_summary.json');scale=read(E/'results/scale_audit.json');cases=read(E/'data/cases.json');samples=read(E/'configs/samples.json');D=dict(np.load(E/'data/latent_probes.npz'));end=list(csv.DictReader((R/'latent_endpoints.csv').open()));steps=[0,1,4,8,16,32]
def get(name,origin,obj,split,step=32,branch='all',cohort='all'):
 return next(x for x in S if x['model']==name and x['origin']==origin and x['objective']==obj and x['split']==split and x['step']==step and x['branch']==branch and x['cohort']==cohort)
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |',*['| '+' | '.join(map(str,r))+' |'for r in rows]])
colors={'mixed':'#2476B5','latent_only':'#DC8730'};labels={'mixed':'混合目标 F+0.25Z','latent_only':'仅 latent 项 0.25Z'}
fig,axs=plt.subplots(2,3,figsize=(14,8.5))
for row,split in enumerate(['AB_train','C_development']):
 for obj in colors:
  rr=[get('actual_seed0','actual',obj,split,st)for st in steps]
  for col,key in enumerate(['mixed_objective_mean','RMS_to_target_mean']):axs[row,col].plot(steps,[x[key]for x in rr],color=colors[obj],marker='o',lw=2,label=labels[obj])
  for seed in [0,1]:
   rr=[get(f'actual_seed{seed}','actual',obj,split,st)for st in steps];axs[row,2].plot(steps,[x['correct_to_wrong']for x in rr],color=colors[obj],ls='-'if seed==0 else'--',marker='o'if seed==0 else's',lw=2,label=labels[obj]+f' / 查询种子{seed}')
 for col in range(3):
  ax=axs[row,col];ax.grid(alpha=.2);ax.set_xticks([0,8,16,24,32]);ax.set_xlabel('独立 latent 的梯度更新步数');ax.set_title(('AB训练：144边界'if row==0 else'C开发：96边界')+'\n'+['共同混合目标值','距真实后继 latent 的平均RMS','起初全9正确 → 变错的边界数'][col])
 axs[row,0].set_ylabel('F+0.25Z，越低越好');axs[row,1].set_ylabel('原始 latent 坐标RMS');axs[row,2].set_ylabel('边界个数');axs[row,2].set_yticks(range(0,8));axs[row,2].set_ylim(-.3,7.3)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False)
fig.suptitle('真实后继 latent 并非当前混合目标的驻点；目标下降可伴随读出退化',fontsize=15,y=.98)
fig.text(.5,.025,'全部边界均纳入；前两列的 latent 路径与查询头无关，两个种子完全相同。\n右列实线圆点为查询种子0、虚线方点为种子1；橙线保持0。每条链的边界相关，非独立样本。',ha='center',fontsize=10)
fig.subplots_adjust(top=.81,bottom=.15,hspace=.5,wspace=.32)
for ext in ['png','svg']:fig.savefig(R/f'true_latent_objective_conflict.{ext}',dpi=160)
plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,8))
for row,origin in enumerate(['one_step','recursive']):
 for col,split in enumerate(['AB_train','C_development']):
  ax=axs[row,col];names=['span3_seed0','span4_seed0','span3_seed1','span4_seed1'];base=[];mix=[];lat=[];den=144 if split=='AB_train'else 96
  for n in names:
   a=get(n,origin,'mixed',split);b=get(n,origin,'latent_only',split);base.append(a['baseline_correct']);mix.append(a['correct']);lat.append(b['correct'])
  x=np.arange(4)
  for off,vals,c,label in [(-.25,base,'#8A929B','起点'),(0,mix,colors['mixed'],'混合目标32步'),(.25,lat,colors['latent_only'],'仅latent项32步')]:
   ax.bar(x+off,np.array(vals)/den*100,.24,color=c,label=label)
   for xx,v in zip(x+off,vals):ax.text(xx,v/den*100+1,str(v),ha='center',fontsize=8)
  ax.set_xticks(x,['三步/种子0','四步/种子0','三步/种子1','四步/种子1']);ax.set_ylim(0,110);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.2);ax.set_ylabel('全9事实正确边界比例（%）');ax.set_title(('单步预测起点'if row==0 else'四步递归预测起点')+' · '+('AB训练 N=144'if col==0 else'C开发 N=96'))
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False);fig.suptitle('反向证据：从已有预测点出发，混合目标通常修正更多错误',fontsize=15,y=.99)
fig.text(.5,.025,'柱上数字为正确边界数，N为每个模型使用的同一批边界；模型、种子重复使用这些数据。\n固定32步不能视为等有效优化量：F梯度约为0.25Z梯度的10—13倍，纯latent距离仅缩至起点的0.778倍。',ha='center',fontsize=10)
fig.subplots_adjust(top=.82,bottom=.16,hspace=.42,wspace=.24)
for ext in ['png','svg']:fig.savefig(R/f'predicted_latent_counterevidence.{ext}',dpi=160)
plt.close(fig)

out=[]
def add(x):out.append(x)
add('# 冻结目标与读出一致性：有限机制诊断\n\n2026-10-16，本机完成。**建议值得做一次预先固定的“仅latent目标 vs 当前混合目标”受控网络训练消融，但当前证据不支持直接改正式损失。** 本阶段只优化每样本临时latent，没有训练网络、没有新增仿真、没有修改正式权重或查询头，也没有开始后续实验。')
add('支持机制的证据是：全部240个真实后继latent上的F梯度非零，且32步混合目标均下降；原先读出全9事实正确的边界，每个查询种子都有4个AB、6个C变错。反向证据是：从已有预测latent出发，混合目标通常比仅latent项修正更多错误；此外，既有网络检查点中，F和Z同时下降仍有语义翻转。因此不能把全部失败归为F项，更不能据此断言latent不含语义信息。')
add('C是反复使用的开发数据，未构成新泛化检验。本次全部轨迹与模型来自已有CALVIN PhysicalOracle对照研究；不代表原HULC/RSSM/TS-LTL方案完成。所有latent优化点都只是无约束向量，**不保证对应真实可实现的机器人状态**。')
add('## 采样与有限预算\n\n没有按标签或预测错误抽样：纳入既有60条轨迹的全部240个技能边界，36个不同根。AB旧普通链12根×4=48边界，新AB12根×普通/持物关屉两分支×4=96边界，合计AB144；C选点6根+C核验6根，各两分支×4，合计C96。根之间才是较高层的分组单位，同一根不同分支和技能边界相关。')
add(table(['项目','固定内容','原因'],[
['模型','前阶段正式选择的四个250更新模型、两个固定读出头；编码器/解码器完全冻结','保持网络与读出不变'],
['真实起点','240个真实后继latent×2查询头=480案例','直接检验真实点是不是混合目标驻点'],
['预测起点','四模型×240边界×单步/初态递归=1920案例','包含正常正确、Holding/In误报、抽屉错误'],
['目标','F+0.25Z 与 0.25Z','保留Z系数，隔离移除F；没有把Z暗中放大4倍'],
['更新','独立latent SGD步长4，32步；保存0/1/4/8/16/32步','训练前封存，不用C选参数或选最佳步'],
['梯度','按每样本目标求和，不在batch再平均；不裁剪、不线搜索','每样本实际步长不随批量变化'],
['标签','采样及所有自动微分阶段禁止读私有标签，只在评分阶段读取','GT不进入反传；查询Jacobian仅作无标签诊断'],
['旧检查点','两臂×两种子×0/250/500/750/1000，共20个','对同一样本比较相邻检查点，禁止重新选模']]))
coverage=[]
for split in ['AB_train','C_development']:
 for origin in ['one_step','recursive']:
  rr=[r for r in S if r['origin']==origin and r['objective']=='mixed'and r['split']==split and r['branch']=='all'and r['step']==0]
  sums={k:sum(r['N_boundaries']for r in rr if r['cohort']==k)for k in ['all','baseline_correct','Holding_false_positive','In_false_positive','drawer_error']}
  coverage.append([split,origin,*[sums[k]for k in ['all','baseline_correct','Holding_false_positive','In_false_positive','drawer_error']]])
add(table(['集合','预测起点','模型×边界数','初始全9正确','Holding误报','In误报','抽屉错'],coverage))
add('上表在四个模型上重复评估相同边界，分母不能当作独立环境样本；错误组可重叠。全部真实状态分组、漏报、技能位置、物体颜色、普通/持物关屉分支已写入 `latent_by_state_and_error.csv`，包含8196个固定分组/步数记录。')
add('## 目标尺度核对\n\n令真实前后特征为x前、x后，真实后继latent为z*。当前训练目标为：\n\n`L(z)=F(decoder(z), x后; x前)+0.25×mean256((z-z*)²)`\n\nF把对象视觉768维、对象几何36维、场景视觉1024维、机器人15维分别求均方均值，再四组等权平均；每坐标乘固定实际变化权重`1+4×min(|x后-x前|,1)`。目标不是所有1843个特征简单混成一个均值。所有特征标准化/PCA来自既有AB拟合并冻结。')
add(table(['尺度','实测/实现','含义'],[
['原始真实latent RMS',f'{scale["latent_raw_RMS"]:.4f}','Z使用原始坐标，并未按查询标准差归一化'],
['query标准差 min/median/max',' / '.join(f'{scale["query_normalization_std"][k]:.4f}'for k in ['min','median','max']),'只有读出前执行(z-mean)/std'],
['Z每坐标平方项系数','0.25/256 = 0.0009765625','Z梯度是0.5(z-z*)/256'],
['单纯Z的32步距离比例',f'{scale["latent_only_analytic_distance_ratio_step32"]:.6f}','解析式(1-4×0.5/256)^32；不代表收敛'],
['F各组单坐标基础系数','0.25/768、0.25/36、0.25/1024、0.25/15，再乘1—5','几何/机器人每坐标权重较大，但需连同decoder Jacobian看梯度']]))
grow=[]
for origin in ['actual','one_step','recursive']:
 for split in ['AB_train','C_development']:
  rr=[g for g in G if g['origin']==origin and g['split']==split and(origin!='actual'or g['seed']==0)]
  grow.append([origin,split,f'{np.median([g["gradF_RMS"]for g in rr]):.6f}',f'{np.median([g["grad025Z_RMS"]for g in rr]):.6f}','Z梯度为0'if origin=='actual'else f'{np.median([g["grad_norm_ratio_F_to_025Z"]for g in rr]):.2f}','未定义'if origin=='actual'else f'{np.median([g["grad_cosine_F_Z"]for g in rr]):.3f}'])
add(table(['起点','集合','F梯度RMS中位数','0.25Z梯度RMS中位数','范数比中位数','F/Z梯度余弦中位数'],grow))
add('预测点上F梯度约为0.25Z的10—13倍，而且两者梯度余弦中位数为正：多数局部方向并非互相对抗。因此“混合目标32步修正更多”可能包含更强有效步幅，不能直接解释为F方向更优；“F主导”也不能直接上升为结构性冲突。这里确证的是**当前具体归一化、decoder和0.25系数下**存在局部目标与读出不一致，尚未区分重构偏差、尺度与读出边界几何的贡献。各特征组梯度RMS、query标准化范围与输入标准化范围保留在 `gradient_directions.csv` 和 `scale_audit.json`。')
add('## 真实后继latent会被目标推向哪里\n\n在z=z*处，Z及其梯度严格为0，但全部240点的F梯度RMS大于预定阈值1e-8。因此这些真实点不是当前无约束混合目标的驻点。这是可直接计算的局部事实；不表示真实状态流形上的受约束最优也相同。')
rows=[]
for seed in [0,1]:
 for split in ['AB_train','C_development']:
  a=get(f'actual_seed{seed}','actual','mixed',split);b=get(f'actual_seed{seed}','actual','latent_only',split);st=get(f'actual_seed{seed}','actual','mixed',split,0)
  rows.append([seed,split,a['N_boundaries'],f'{a["baseline_correct"]}→{a["correct"]}',a['correct_to_wrong'],a['wrong_to_correct'],f'{st["objective_mean"]:.4f}→{a["objective_mean"]:.4f}',f'{a["RMS_to_target_mean"]:.4f}',b['correct']])
add(table(['查询种子','集合','边界N','混合目标全9正确','正确→错','错→正确','混合目标均值','距真latent平均RMS','仅Z最终全9正确'],rows))
add('240个真实点的共同混合目标均值从0.435592降至0.267698，其中终点F为0.255827、Z为0.047486；距真实latent平均RMS为0.209205，按查询std归一化后为0.249900。两查询头不参与目标，因此两个种子的优化路径完全相同，仅读出边界不同。种子0读出236/240→226/240；种子1为230/240→223/240，后者同时有3个原错变对，不能只报净损失而隐去10个新错误。')
add('![真实latent目标与语义](true_latent_objective_conflict.png)\n\n图1：蓝色是优化混合目标，橙色是只优化0.25Z。横轴为固定梯度步数。左列对两臂都评估同一个F+0.25Z，纵轴是均值；中列是离开真实latent的平均原始坐标RMS；右列统计最初全9正确后来变错的边界数，实线圆点与虚线方点分别为查询种子0/1。两行是AB训练和C开发，分母分别144/96。橙线停在真实点，蓝线目标下降且发生语义退化，直接支持所检验的局部机制。')
state_rows=[]
for seed in [0,1]:
 for split in ['AB_train','C_development']:
  for branch in ['normal','held_closed']:
   a=get(f'actual_seed{seed}','actual','mixed',split,branch=branch);state_rows.append([seed,split,branch,a['N_boundaries'],a['baseline_correct'],a['correct_to_wrong'],a['wrong_to_correct']])
add(table(['种子','集合','分支','N','起初全9正确','正确→错','错→正确'],state_rows))
add('AB每种子新增4个错误分布为3个普通边界、1个持物关屉边界；C各新增6个中5个在持物关屉分支。它不是只在普通链末步发生的机制。按truth Holding、In、drawer状态的全部分组已随CSV交付，空组没有伪造补齐。')
eg=[]
for seed,root,branch,step in [(0,'A_longpair_blue_closed_1','normal',2),(1,'B_longpair_red_closed_0','normal',2),(0,'A_intermediate_v3_red_0','held_closed',1),(0,'C_intermediate_v3_pink_0','normal',3)]:
 r=next(r for r in end if r['origin']=='actual'and r['objective']=='mixed'and int(r['seed'])==seed and r['root_id']==root and r['branch']==branch and int(r['global_step'])==step);bad=[g for g in ['Holding','In','drawer']if r[g+'_final_correct']=='False'];g=bad[0];eg.append([seed,root+f'/{branch}/h{step}',g,f'{float(r[g+"_initial_margin"]):.3f}→{float(r[g+"_final_margin"]):.3f}',f'{float(r["objective_initial"]):.4f}→{float(r["objective_final"]):.4f}'])
add(table(['种子','示例边界','出错组','signed logit margin','混合目标'],eg))
add('示例用于解释全量计数，不承担样本选择。二元命题margin定义为真值符号乘原始logit；drawer为真实类logit减最强竞争类logit。正值在正确侧，负值在错误侧，**不是校准置信度**。两个种子各10个新增错误中，各有9个的对应错误组起始最小margin沿−∇F的一阶方向已下降；另外1个在有限非线性路径后跨界。计算query JVP时没有输入GT，标签仅在事后给方向加符号。')
add('## 预测起点的反向证据\n\n同一个起始预测latent送入两个固定目标，没有重新调用转移或真实环境。下表每行是四模型重复评估的汇总，AB为576次模型×边界、C为384次；每个模型和种子的原始结果在图与CSV中保留，不作独立样本统计。')
rows=[]
for origin in ['one_step','recursive']:
 for split in ['AB_train','C_development']:
  a=[get(n,origin,'mixed',split)for n in ['span3_seed0','span4_seed0','span3_seed1','span4_seed1']];b=[get(n,origin,'latent_only',split)for n in ['span3_seed0','span4_seed0','span3_seed1','span4_seed1']]
  total=lambda rr,k:sum(x[k]for x in rr)
  rows.append([origin,split,total(a,'N_boundaries'),total(a,'baseline_correct'),total(a,'correct'),f'{total(a,"correct_to_wrong")} / {total(a,"wrong_to_correct")}',total(b,'correct'),f'{total(b,"correct_to_wrong")} / {total(b,"wrong_to_correct")}'])
add(table(['起点','集合','模型×边界N','起点正确','混合终点正确','混合 正→错/错→正','仅Z终点正确','仅Z 正→错/错→正'],rows))
add('单步预测起点上，混合目标共修正127个原错，同时引入22个新错；仅Z修正62个，没有新错。递归预测起点上，混合修正191个，仅Z修正73个，两者都未引入新错。这些计数直接反驳“F总会伤害语义”或“去掉F必然更好”。单步的22个新错同时伴随F与Z下降，说明甚至更接近真实latent也不必在整段路径上保持查询语义。固定32步只移动到有限点，不能解释为优化收敛后的比较。')
add('![预测latent对照](predicted_latent_counterevidence.png)\n\n图2：横轴为前阶段的三步/四步训练模型和种子，灰柱是起点，蓝柱为混合目标32步，橙柱为仅latent项32步；纵轴为全9事实正确比例，柱上写正确边界个数。左/右列为AB/C，上/下行为单步/递归预测起点。每格分母固定为144或96。图的意义是展示反向证据，同时保留各模型结果；因梯度幅度不同，不能用柱高直接证明目标本身或训练效率孰优。')
add('## 既有网络检查点是否也发生不一致\n\n在20个冻结检查点上，对同一240边界分别做单步与初态递归推理，再配对相邻250更新，得到7680个配对。只观察已有轨迹，没有新训练。下表统计两个误差都下降的配对；列出的翻转依然可能受其他训练样本、递归误差与读出非线性影响，不能作为F单项因果证据。')
rows=[]
for seed in [0,1]:
 for origin in ['one_step','recursive']:
  for split in ['AB_train','C_development']:
   r=next(x for x in CS if x['seed']==seed and x['origin']==origin and x['split']==split and x['branch']=='all'and x['cohort']=='F_and_Z_down');rows.append([seed,origin,split,r['N_pairs'],r['correct_before'],r['correct_to_wrong'],r['wrong_to_correct'],f'{r["Holding_margin_delta_mean"]:.3f}',f'{r["In_margin_delta_mean"]:.3f}',f'{r["drawer_margin_delta_mean"]:.3f}'])
add(table(['种子','路径','集合','F/Z都降配对N','之前全9正确','正→错','错→正','Holding margin均变','In margin均变','drawer margin均变'],rows))
add('全部检查点配对中，混合目标下降5129对：3048对起初全9正确，其中22对变错，另有78对原错变对。F与Z都下降4355对：2450对起初全9正确，其中15对变错，另有71对原错变对；15个恶化配对涉及9个根。总体改善多于退化，但误差与语义并非逐样本单调一致。完整配对同时记录三个组的前后signed margin，可在 `checkpoint_pairs.csv` 复核；这些相关配对不能当成数千个独立试验。')
add('## 判断与下一步建议\n\n**值得做一次小型受控目标消融，尚不值得直接替换正式损失。** 支持消融的理由不是平均分的小幅变化，而是训练域真实latent上已有可复现反例：F梯度把本已读出正确的点推离真实编码，目标降低后跨过查询边界，两个固定查询头都观察到4个AB退化边界。只优化Z的解析对照完全保持真实点，排除了“只要运行优化就会漂移”的实现解释。')
add('仍需保留的替代解释是：F梯度明显更强；固定decoder重构偏差与query边界不一致；query标准化与Z的原始坐标度量不同；预测latent上的路径跨界并不一定来自F单项。F与Z平均梯度同向、且混合目标常修正更多错误，因此当前证据没有给出“仅Z网络训练一定获胜”的方向性结论，也没有证明latent缺少信息。')
add('若父线程决定开展消融，应在训练前固定Z系数、数据/起点/预算与共同选点规则，并同步报告梯度量级和优化进度，避免把删除F造成的更新幅度变化解释成纯目标效果。本阶段不提出更多参数扫描，不加入命题监督，也不自动执行该训练。')
add('## 复核与文件\n\n实验目录：`'+str(E)+'`；报告目录：`'+str(R)+'`。')
add(table(['文件','内容'],[
['configs/protocol.json / frozen.json / samples.json','运行前协议、全部边界及36项输入/源码/模型哈希'],['data/latent_probes.npz / cases.json','2400起点、两目标6快照、初末latent、目标梯度与query JVP'],['data/checkpoints.npz / checkpoint_cases.json','20个已有检查点的9600次边界推理'],['latent_endpoints.csv / gradient_directions.csv','4800目标终点和2400初始梯度记录，含原始signed margin'],['latent_by_state_and_error.csv','全体、AB/C、分支、真状态、技能、颜色、初始错误分组'],['checkpoint_pairs.csv / checkpoint_summary.csv','7680个逐样本相邻检查点配对及汇总'],['results/scale_audit.json / run_audit.json','尺度、梯度规范、冻结状态、无标签微分审计'],['results/verification.json / configs/delivery_manifest.json','独立数值核对与最终SHA256封存']]))
add('独立复核通过：此前103个封存文件与本阶段36项输入哈希一致；所有网络参数冻结；两目标起点完全相同；真实latent仅Z更新严格不动；仅Z解析更新与实测最大绝对误差2.38e-6；F分组梯度和与总梯度最大差5.96e-8（float32舍入）；全部记录的混合目标非增；4800终点语义、1080个主汇总组、7680个检查点配对重新计算一致。两个查询种子的真实latent路径逐元素一致，进一步验证查询没有进入优化目标。')
add('只读复验命令：\n\n```bash\nsource /home/robot/research/projects/wm_symbolic/world_model_env.sh\npython /home/robot/research/projects/wm_symbolic/experiments/objective_readout_diagnostic_20261016/source/verify.py\n```\n\n本次有限诊断已结束；没有环境或传输阻塞，正式模型和损失保持不变，下一步训练决策交回父线程。')
(R/'REPORT.md').write_text('\n\n'.join(out)+'\n');print('REPORT_WRITTEN',flush=True)
