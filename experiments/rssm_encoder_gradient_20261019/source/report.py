from common import *
import csv,base64,html,unicodedata,re
os.environ['MPLCONFIGDIR']=str(E/'logs/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':10,'pdf.fonttype':3})
COLORS={'joint':'#2166AC','route_current_only':'#D6604D'}
NAMES={'joint':'未来梯度进入编码器','route_current_only':'编码器仅接收当前损失'}
MODES={'current':'当前状态','one_step':'真实前态一步','free1':'根起点一步','free3':'根起点三步','free4':'根起点四步','probe':'冻结编码器线性查询','persistence':'保持当前状态','root_persistence1':'保持根状态一步','root_persistence3':'保持根状态三步','root_persistence4':'保持根状态四步','wrong_action':'错误动作'}
def fmt(x,d=3):return '未定义'if x is None else f'{x:.{d}f}'
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |'for r in rows)+'\n'
def getmetric(rows,arm,seed,mode,split='dev',role='all',stratum='all',change='all',fact='primary_3objects_drawer',metric='NLL'):
 criteria={'arm':arm,'seed':seed,'mode':mode,'split':split,'role':role,'stratum':stratum,'change':change,'fact':fact,'metric':metric}
 return next(r for r in rows if all(r[k]==v for k,v in criteria.items()))
def wrap(t,width=85):
 out=[]
 for original in t.split('\n'):
  line='';n=0
  for c in re.findall(r'[A-Za-z0-9_.:/+−–×-]+|.',original):
   w=sum(2 if unicodedata.east_asian_width(ch)in'WF'else 1 for ch in c)
   if n+w>width:out.append(line);line='';n=0
   line+=c;n+=w
  out.append(line)
 return '\n'.join(out)
def savefig(fig,name,caption):
 fig.text(.07,.025,wrap(caption,148),va='bottom',fontsize=9)
 fig.savefig(R/f'figures/{name}.png',dpi=170,bbox_inches='tight');fig.savefig(R/f'figures/{name}.pdf',bbox_inches='tight')
 return fig
def report():
 guard('report');score=read(E/'results/scores.json');scope=read(E/'results/scope_design.json');rows=[]
 with(R/'all_metrics.csv').open()as f:
  for r in csv.DictReader(f):
   for k in ['seed','n','roots']:r[k]=int(r[k])
   for k in ['value','lo95','hi95','micro_value']:r[k]=float(r[k])if r[k]else None
   rows.append(r)
 pairs=score['paired_primary'];coll=[read(E/f'results/collapse_{a}_seed{s}_dev.json')for s in [0,1]for a in ARMS]
 fits=[read(E/f'results/train_{a}_seed{s}.json')for s in [0,1]for a in ARMS]
 training=read(E/'results/training_attempts.json');figs=[]
 # Main results: same roots and same facts; seeds separate, no seed selection.
 fig,axs=plt.subplots(2,2,figsize=(12,9));modes=['current','one_step','free3','probe']
 for seed in [0,1]:
  ax=axs[0,seed]
  for j,arm in enumerate(ARMS):
   rr=[getmetric(rows,arm,seed,m)for m in modes];y=np.array([r['value']for r in rr]);x=np.arange(4)+(j-.5)*.19
   ax.errorbar(x,y,yerr=[y-[r['lo95']for r in rr],[r['hi95']for r in rr]-y],fmt='o',capsize=3,color=COLORS[arm],label=NAMES[arm])
  ax.set_xticks(range(4),['当前','一步','三步自由','线性查询']);ax.set_ylabel('三物体＋抽屉 NLL（越低越好）');ax.set_title(f'开发集20根 · 种子{seed}');ax.grid(alpha=.18);ax.legend(fontsize=8)
  ax=axs[1,seed]
  for j,mode in enumerate(['free3','probe']):
   r=next(v for v in pairs if v['seed']==seed and v['split']=='dev'and v['mode']==mode)
   ax.errorbar(r['joint_minus_control'],j,xerr=[[r['joint_minus_control']-r['lo95']],[r['hi95']-r['joint_minus_control']]],fmt='o',capsize=5,color='#2166AC')
  ax.axvline(0,color='grey',ls='--');ax.set_yticks([0,1],['三步自由预测','冻结编码器线性查询']);ax.set_xlabel('配对NLL差：未来梯度组 − 仅当前组\n负值表示未来梯度组误差较低');ax.grid(alpha=.18)
 fig.suptitle('直接未来梯度是否改善预测与可查询表示？',fontsize=16);fig.tight_layout(rect=(0,.13,1,.94))
 cap1='蓝：未来损失直接更新编码器；橙：编码器仅接收当前损失，其余模块仍学习未来。上图点为完整根等权均值，误差线为2000次整根重采样95%区间。下图在相同根上先求两组差再重采样。种子分开报告；C已经用于开发，这些区间不代表独立迁移能力。'
 figs.append(savefig(fig,'01_representation_and_prediction',cap1))
 # Generalization gap and explicit unchanged/changed facts.
 fig,axs=plt.subplots(2,2,figsize=(12,9))
 for seed in [0,1]:
  ax=axs[0,seed]
  for j,arm in enumerate(ARMS):
   for k,mode in enumerate(['free3','probe']):
    yy=[getmetric(rows,arm,seed,mode,split=s)['value']for s in ['train','dev']];xx=np.array([0,1])+k*1.6+(j-.5)*.07
    ax.plot(xx,yy,'o-',color=COLORS[arm],label=NAMES[arm]if k==0 else None)
  ax.set_xticks([0,1,1.6,2.6],['训练','开发','训练','开发']);ax.set_title(f'种子{seed}：左为三步预测，右为线性查询');ax.set_ylabel('相同四字段根平均NLL');ax.grid(alpha=.18);ax.legend(fontsize=8)
  ax=axs[1,seed];names=['red_location','blue_location','pink_location','drawer','slider']
  for j,arm in enumerate(ARMS):
   for change,ls in [('changed','-'),('unchanged','--')]:
    rr=[getmetric(rows,arm,seed,'one_step',fact=n,metric='accuracy',change=change)for n in names]
    ax.plot(range(5),[r['value']if r['value']is not None else np.nan for r in rr],'o'+ls,color=COLORS[arm],label=NAMES[arm]+('·改变'if change=='changed'else'·不变'))
  ax.set_xticks(range(5),['红物体','蓝物体','粉物体','抽屉','滑门']);ax.set_ylim(-.03,1.03);ax.set_ylabel('一步分类准确率（根平均）');ax.legend(fontsize=7);ax.grid(alpha=.18)
 fig.suptitle('数据规模之外：训练拟合、开发差距与不变事实',fontsize=16);fig.tight_layout(rect=(0,.13,1,.94))
 cap2='上排连线只连接同一模型在训练60根与已看过的开发20根上的误差，不是学习曲线，也没有新增规模训练。下排实线是实际改变字段、虚线是不变字段；红/蓝/粉表示固定物体身份。样本按实际前后标签分类，不按请求意图；滑门改变样本极少，完整分母见CSV。'
 figs.append(savefig(fig,'02_gap_and_variable_facts',cap2))
 # Baselines and collapse evidence.
 fig,axs=plt.subplots(2,2,figsize=(12,9))
 for seed in [0,1]:
  ax=axs[0,seed];methods=['one_step','persistence','wrong_action','free3','root_persistence3']
  for j,arm in enumerate(ARMS):
   rr=[getmetric(rows,arm,seed,m)for m in methods];ax.bar(np.arange(5)+(j-.5)*.34,[r['value']for r in rr],width=.34,color=COLORS[arm],label=NAMES[arm])
  ax.set_xticks(range(5),['一步','保持前态','错误动作','三步','保持根态']);ax.set_ylabel('三物体＋抽屉 NLL');ax.set_title(f'种子{seed}：同终点对照');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.18)
 ax=axs[1,0]
 for j,arm in enumerate(ARMS):
  cc=[next(c for c in coll if c['arm']==arm and c['seed']==s)for s in [0,1]];ax.bar(np.arange(2)+(j-.5)*.34,[c['E_effective_rank']for c in cc],width=.34,color=COLORS[arm],label=NAMES[arm])
 ax.set_xticks([0,1],['种子0','种子1']);ax.set_ylabel('编码器输出协方差有效秩');ax.set_title('有效秩描述分布，不等于语义质量');ax.legend(fontsize=8)
 ax=axs[1,1]
 for j,arm in enumerate(ARMS):
  cc=[next(c for c in coll if c['arm']==arm and c['seed']==s)for s in [0,1]];ax.bar(np.arange(2)+(j-.5)*.34,[c['posterior_entropy_gap_mean_nats']for c in cc],width=.34,color=COLORS[arm],label=NAMES[arm])
 ax.set_xticks([0,1],['种子0','种子1']);ax.set_ylabel('后验边际熵 − 平均条件熵（nat）');ax.set_title('跨状态后验分布变化；32变量平均');ax.legend(fontsize=8)
 fig.suptitle('动作条件与坍塌诊断：避免仅凭方差或KL下结论',fontsize=16);fig.tight_layout(rect=(0,.13,1,.94))
 cap3='一步保持使用真实前态的模型读出；三步保持始终复制同一个根状态读出。错动作固定置换技能与静态参数，仍对原真实终点评分，不是物理反事实。下排来自真实观测编码及确定性后验历史，不把随机采样噪声算成状态多样性；还需结合线性查询、当前误差和动作敏感性判断。'
 figs.append(savefig(fig,'03_baselines_and_collapse',cap3))
 # Mechanism interpretation retains separate outcomes rather than a post-hoc gate.
 primarylines=[]
 for mode in ['free3','probe']:
  rr=[next(v for v in pairs if v['seed']==s and v['split']=='dev'and v['mode']==mode)for s in [0,1]]
  line=MODES[mode]+'：'+'；'.join(f"种子{r['seed']}差值{fmt(r['joint_minus_control'])}，95%区间[{fmt(r['lo95'])}, {fmt(r['hi95'])}]"for r in rr)+'。'
  if all(r['hi95']<0 for r in rr):line+='两种子均支持本数据范围内误差降低。'
  elif all(r['lo95']>0 for r in rr):line+='两种子均显示本数据范围内误差升高。'
  else:line+='没有出现两种子区间均位于有利方向的结果；保留差异和不确定性，不追加调参。'
  primarylines.append(line)
 summary='\n\n'.join(primarylines)
 generalization_note='三步预测的训练NLL仅0.005–0.066，开发NLL为2.478–3.841，差距很大；这与过拟合或A/B到C的环境分布差异一致，本轮不能唯一分辨两者。较低平均查询误差也不等于完整状态更可靠：线性探针种子0五字段全对反而从22/300降至11/300，种子1从0/300升至18/300。'
 summary+='\n\n'+generalization_note
 probe_better=all(next(v for v in pairs if v['seed']==s and v['split']=='dev'and v['mode']=='probe')['hi95']<0 for s in [0,1])
 interpretation=('线性探针为已知标签可读性提供了支持，但不等于获得新关系或开放词汇。'if probe_better else'本轮不能据此确认共享视觉编码器的可查询信息稳定改善。即使后继预测变好，也可能来自其他模块与编码器共同适应，不能单凭后继分数断言表示更好。')
 keyrows=[]
 for seed in [0,1]:
  for mode in ['current','one_step','free3','probe']:
   rr=[getmetric(rows,a,seed,mode)for a in ARMS];delta=next(v for v in pairs if v['seed']==seed and v['split']=='dev'and v['mode']==mode)
   keyrows.append([seed,MODES[mode],rr[0]['n'],rr[0]['roots'],fmt(rr[0]['value']),fmt(rr[1]['value']),f"{fmt(delta['joint_minus_control'])} [{fmt(delta['lo95'])}, {fmt(delta['hi95'])}]"])
 gaprows=[[g['seed'],NAMES[g['arm']],MODES[g['mode']],fmt(g['train_NLL']),fmt(g['seen_C_NLL']),fmt(g['C_minus_train'])]for g in score['train_dev_gaps']if g['mode']in['current','one_step','free3','probe']]
 exactrows=[]
 for seed in [0,1]:
  for mode in ['current','free3','probe']:
   for arm in ARMS:
    rr=[getmetric(rows,arm,seed,mode,fact=f,metric='accuracy')for f in ['primary_3objects_drawer','all5_exact','all7_exact']]
    exactrows.append([seed,MODES[mode],NAMES[arm],f"{rr[0]['value']*100:.1f}%",f"{round(rr[1]['micro_value']*rr[1]['n'])}/{rr[1]['n']}",f"{round(rr[2]['micro_value']*rr[2]['n'])}/{rr[2]['n']}"])
 collapsed=[[c['seed'],NAMES[c['arm']],fmt(c['E_std_mean']),fmt(c['E_effective_rank'],1),fmt(c['posterior_conditional_entropy_mean_nats']),fmt(c['posterior_marginal_entropy_mean_nats']),fmt(c['posterior_entropy_gap_mean_nats']),fmt(c['deterministic_h_std_mean']),fmt(c['action_object_TV_mean'])]for c in coll]
 role_rows=[]
 for seed in [0,1]:
  for arm in ARMS:
   for role in sorted(set(r['role']for r in rows if r['split']=='dev'and r['role']!='all')):
    r=getmetric(rows,arm,seed,'free3',role=role);role_rows.append([seed,NAMES[arm],role,r['roots'],r['n'],fmt(r['value'])])
 status_rows=[]
 for seed in [0,1]:
  for arm in ARMS:
   for mode,st in [('one_step','zero_control'),('one_step','nonzero_control'),('one_step','executed_failure'),('free3','all_prefix_actuated')]:
    r=getmetric(rows,arm,seed,mode,stratum=st);status_rows.append([seed,NAMES[arm],MODES[mode],st,r['n'],r['roots'],fmt(r['value'])])
 scheme=scope['metadata_only_example'];scale=table(['未来训练规模','完整根','真实转移','6000更新等效遍历次数'],[[i+1,s['roots'],s['transitions'],fmt(s['fixed_updates6000_epoch_equivalent'],1)]for i,s in enumerate(scope['nested_root_design'])])
 trainingtable=table(['种子','条件','更新','主体秒数','峰值GPU GiB','C选点'],[[r['seed'],NAMES[r['arm']],r['updates'],fmt(r['elapsed_seconds_this_segment'],1),fmt(r['peak_GPU_bytes']/2**30,2),'无']for r in fits])
 md=f'''# RSSM编码器直接未来梯度对照

完成时间：{now()}。本机四次固定预算训练、四个固定线性探针和A/B/C评估已完成。报告保留两种子，不按结果选模型；未打开D样本、未采新数据。

{summary}

{interpretation}

## 问题、对照与适用范围

核心问题是从视觉和交互学可复用状态、预测动作如何改变状态，并让下游能查询这些状态。本次只检验原监督RSSM中未来损失直接更新编码器这一训练因素。它与旧无命题标签交互预训练相关，但不是相同实验；不能从跨研究分数推断架构优劣。完整旧新核对见[旧实验与本次对照](旧实验与本次对照_20261019.md)。

|元素|本次固定内容|作用与边界|
|---|---|---|
|数据|A/B 720转移、60根、180分支；C240转移、20根、60分支|每根3分支、每分支4请求；分支和帧不能冒充独立根|
|输入|原四帧双相机冻结DINO、可见mask辅助ROI、本体感觉|不是纯RGB对象发现；GT标签仅作损失/评分，不作推理输入|
|架构|原11,513,563参数对象Transformer/RSSM；E2,080,768，其他9,432,795|架构、前向计算规则、损失定义在两臂相同|
|唯一主对照|蓝组E收总损失梯度；橙组E仅收当前重构与当前符号损失梯度|其他参数两臂都收总梯度；后者并非完全不受动力学间接影响|
|预算|2种子×2臂×6000；相同初始、batch、潜变量随机起点、优化器、日程|前5000单步，末1000加入三步；只取最终6000，不用C选点|
|裁剪|两臂均对E与其余参数分别norm1|相对旧全局裁剪的共同改变，排除其余梯度大小直接缩放E的耦合|
|探针|冻结E，单层512→32，16,416参数，各600步|相同配对初始及根均衡采样；各自仅以训练E均值/方差标准化；检验已知标签可读性|
|统计|完整根等权NLL/准确率；2000次整根bootstrap；种子分别报|二值精确率/召回/BA为汇总混淆计数比值并按根重采样；无分母留空|

在同一权重、批次和随机状态的FP32及实际BF16检查中，原前向损失与本次相等，其他参数梯度相等，控制组E梯度恰为当前损失梯度；放大其余梯度不改变E裁剪。预检优化器更新0。训练源码、数据、初始状态在拟合前冻结，评估/探针源码在产生模型预测前另行冻结。

## 相同事实、相同根的主要结果

NLL是预测对真实类别的负对数概率，越低越好。主指标对三个物体位置/归属和抽屉四字段等权，再按完整根等权；不让几乎不变的灯掩盖对象错误。三步只编码真实根状态，再按真实连续请求递推，中途不输入真实观测。完整状态准确率另列CSV。

{table(['种子','读出路径','边界记录','根','未来梯度组NLL','仅当前组NLL','配对差及95%区间'],keyrows)}

![主要结果](figures/01_representation_and_prediction.png)

{cap1}

以下是预先声明的分类/完整状态指标，作为NLL的补充而非更换主指标。四字段平均准确率不等于完整状态正确；五字段加入滑门，七字段再加入两灯。探针种子0完整状态退化必须与平均标签收益同时保留。

{table(['种子','路径','条件','四字段平均准确率','五字段全对n/N','七字段全对n/N'],exactrows)}

在三分支根观测记录（20个独立根、60份根记录）上，原RSSM当前五字段全对：未来梯度组种子0为1/60、种子1为4/60；仅当前组分别0/60、1/60。说明大量错误在尚未自由递推前就存在，不能全部归给多步累积。一阶预测中未变化的三个物体位置/归属也仅约29.5%–58.4%正确，动作造成变化的字段则多数较高。这里只定位已有预测路径的短板，不据此唯一归因于某个模块，也未做新的规划闭环或修补。

## 训练—开发差距与覆盖

{table(['种子','条件','路径','训练NLL','已看过C NLL','C减训练'],gaprows)}

![训练开发差距](figures/02_gap_and_variable_facts.png)

{cap2}

C原12根选模子集和8根校准子集均已在后续研究中使用，不重新命名为盲测。下表分别保留原角色，不能把小子集最好的数字选作结论。

{table(['种子','条件','原C角色','根','三步边界记录','NLL'],role_rows)}

## 真实执行、拒绝与不变事实

请求不等于机器人实际执行。A/B720请求中600非零、120零控制；C240中199非零、41零控制。根起点三次请求共有A/B180、C60条分支，其中三次都实际执行的仅A/B60、C19条。所有拒绝和执行失败保留；19条子集不能代替全60条主分母。

{table(['种子','条件','路径','条件组','记录','根','NLL'],status_rows)}

各物体位置、Holding、In、抽屉、滑门与两灯的改变/不变、每技能、二值混淆、完整五/七字段、事件概率均见[all_metrics.csv](all_metrics.csv)。灯无正例/变化、滑门变化极少，不把其高准确率解释为动作理解。接触支撑硬负例的必要条件在当前端态记录中未出现，分母0，FPR未定义，不报告成0错误率。

## 坍塌与动作依赖

![基线与坍塌](figures/03_baselines_and_collapse.png)

{cap3}

{table(['种子','条件','E平均标准差','E有效秩','后验条件熵','后验边际熵','熵差','确定性h标准差','动作对象TV'],collapsed)}

实际E有效秩仅约1.7–3.2，尽管输出512维且有非零方差，变化仍集中在少数方向。后验分布和动作响应并未变成常数，但这不足以排除部分信息缺失或表示退化。

后验熵按32个离散变量平均，单位nat；熵差刻画后验分布跨记录的变化，不能当作语义互信息。TV为正确/错误动作后物体类别分布的总变差，越大只表示更依赖动作，不保证正确。E方差、有效秩、非零h或KL接近1nat自由下限，单独都不能排除部分坍塌；应与主任务和线性查询一起解释。

## 数据规模与下一轮设计（未执行）

现有A/B和C各有22种四请求顺序，C没有训练未见顺序。因此本轮没有提供未见组合迁移证据，也不能从单一720规模推断增加数据一定改善。已保存以下完整根嵌套名单；每层同时保持A/B及初始关屉空手、开屉空手、开屉持物的原比例，C20根固定。

{scale}

每次更新抽16条完整分支，上表等效遍历次数=6000×16/(3×根数)，属于抽样次数口径。统一6000步固定的是计算预算，小数据会被重复更多次；若另做等遍历次数对照，应明确让更新数随数据量变化，两种研究不能混成一条“数据越多越好”的曲线。未来两臂应继续同种子配对，不因小集合偶然好看选择种子。

只靠现有轨迹可以构造动作顺序的探索性整根留出。例如，预先按元数据字典序选出的三请求组合“{' → '.join(scheme['sequence'])}”：从A/B移除所有含该组合的{len(scheme['removed_train_roots'])}根及其全部兄弟分支，剩{scheme['remaining_train_roots']}训练根，六技能和三初态均仍有覆盖；C中{len(scheme['C_roots_containing_sequence'])}根包含该组合。所有候选和名单见scope_design.json。此设计没有泄漏兄弟轨迹，但初态/动作分布仍随留出改变，且数据已被看过，只能称探索性重新划分。留出请求顺序不等于留出物理关系或状态—动作条件；若要检验后两者，需要另行定义和核验覆盖。

六技能与32个固定输出限制了可表达的范围，增加同类样本不会自动增加新物体、关系或技能词汇。真正测试复用，需要预声明新对象/关系/动作组合与独立根；五步以上还需真实连续轨迹，当前四步不能拼接冒充。无需把每个局部谓词修到完美再研究机制，但必须量化错误如何影响后继和下游使用。

## 训练与交付审计

{trainingtable}

训练进程按单卡锁串行运行，四条均6000更新，四个probe各600更新；无新增主模型拟合、数据采集、D评估、闭环实验或162根草案执行。旧数据、原模型和封存报告保持只读。旧实验的无标签方案比较、冻结E的转移修复与本次监督RSSM梯度对照回答相关但不同的问题；本轮只允许两臂内部的训练因素比较，不能宣称完整指南或通用规划问题完成。

源配置和权重：[实验目录](../../experiments/rssm_encoder_gradient_20261019)。[根级原始统计](root_metrics.csv)、[配对差](paired_differences.csv)、[全部指标](all_metrics.csv)。所有数字来自保存的预测，不从训练loss猜测性能；核验和最终文件清单在实验results/verification.json与configs/delivery_manifest.json。
'''
 (R/'REPORT.md').write_text(md)
 # Standalone readable HTML generated without external assets or scripts.
 blocks=[];in_table=False
 for line in md.splitlines():
  if line.startswith('|'):
   if set(line.replace('|','').replace(' ',''))=={'-'}:continue
   if not in_table:blocks.append('<table>');in_table=True
   blocks.append('<tr>'+''.join('<td>'+html.escape(t.strip())+'</td>'for t in line.strip('|').split('|'))+'</tr>');continue
  if in_table:blocks.append('</table>');in_table=False
  if line.startswith('!['):
   path=line.split('](')[1][:-1];data=base64.b64encode((R/path).read_bytes()).decode();blocks.append('<img src="data:image/png;base64,'+data+'">')
  elif line.startswith('# '):blocks.append('<h1>'+html.escape(line[2:])+'</h1>')
  elif line.startswith('## '):blocks.append('<h2>'+html.escape(line[3:])+'</h2>')
  elif line.strip():blocks.append('<p>'+html.escape(line)+'</p>')
 (R/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>RSSM梯度对照</title><style>body{font:16px/1.7 sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#203246}table{border-collapse:collapse;width:100%;font-size:14px}td{border:1px solid #ccd5dc;padding:7px}tr:first-child{background:#edf3f7;font-weight:bold}img{max-width:100%}h2{margin-top:2em}</style>'+''.join(blocks))
 # A short PDF with editable-vector plots; full detail remains in Markdown/CSV.
 with PdfPages(R/'RSSM编码器梯度对照_20261019.pdf')as pdf:
  page=plt.figure(figsize=(8.27,11.69));page.text(.07,.95,'RSSM编码器梯度对照',fontsize=20,va='top')
  txt='\n\n'.join([summary,interpretation,'比较范围：原监督RSSM、A/B60训练根720转移，已看过C20开发根240转移。两种子、两条件各6000更新，只取终点。冻结编码器后，同预算600步线性查询。','蓝组允许未来损失直接更新编码器；橙组仅当前损失更新编码器。两臂其他模块仍学习未来。当前/未来符号监督、KL、事件及多步目标均保留；这不是无语义标签预训练。','旧无标签AE/交互实验出现后继和小闭环收益，但计算与batch不严格匹配，当前读出未稳定优于AE；后续修复冻结编码器，不能回答本次梯度问题。跨研究架构/输入/监督不同，只能作本轮两臂内部比较。','统计按完整根重采样2000次；两种子分别报告。开发C不是新测试，未读取D或新增仿真。所有拒绝/失败保留，没有调整阈值或因结果增加拟合。','后续页蓝/橙含义固定；点、线、柱、轴和误差线均在图内解释。详细全字段分母、配置、归属与配对差见同目录REPORT.md和CSV。'])
  page.text(.07,.89,wrap(txt,88),fontsize=min(11,680/(len(wrap(txt,88).splitlines())*1.7)),va='top',linespacing=1.7);pdf.savefig(page);plt.close(page)
  for fig in figs:pdf.savefig(fig)
  page=plt.figure(figsize=(8.27,11.69));page.text(.07,.95,'规模、复用与证据边界',fontsize=19,va='top')
  txt='\n\n'.join(['更多边界不等于更多独立经验：训练720转移来自60根，开发240转移来自20根，每根三个分支。C的22种四请求顺序全部在A/B出现，当前不构成未见组合验证。','下一轮只读设计：完整根嵌套20/40/60，转移240/480/720，固定同一C20根，保持A/B和三初态比例。统一6000更新固定计算但不固定重复暴露；另做等遍历次数才可分开讨论，不能混成一个结论。',f"整根动作组合留出可行示例：移除所有含指定三请求组合的{len(scheme['removed_train_roots'])}个训练根，保留{scheme['remaining_train_roots']}根；所有兄弟分支跟随根，不能拼接。六技能及三初态仍存在，但分布比例改变、数据已经看过，因此仅是探索性设计。",'固定六技能和32输出不会因重复采样变成开放词汇。可复用性需要新关系、新对象、动作组合及独立根的单独验证。当前没有五步数据，不拼接四步轨迹。','三次请求都实际执行的C分支仅19/60；零控制41/240全部保留。当前没有接触支撑硬负例分母，FPR未定义，不能写成0。','方差、有效秩、后验熵或KL自由下限均不能独自证明有效状态表示。需要同一批根上的查询和预测表现；当前读出、一步预测、根起点递推是不同误差路径。','本轮完成固定预算比较，不追加修补单例、不选择更好种子。它不代表原HULC、约6000转移、多技能、TS/LTLf完整指南完成，也不替代闭环验证。'])
  page.text(.07,.89,wrap(txt,88),fontsize=min(11,680/(len(wrap(txt,88).splitlines())*1.8)),va='top',linespacing=1.8);pdf.savefig(page);plt.close(page)
 for fig in figs:plt.close(fig)
 save(E/'results/report_receipt.json',{'utc':now(),'summary':primarylines,'interpretation':interpretation,'figures':3,'PDF_pages':5,'report_files':['REPORT.md','REPORT.html','RSSM编码器梯度对照_20261019.pdf'],'new_fits':0})
 print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':report()
