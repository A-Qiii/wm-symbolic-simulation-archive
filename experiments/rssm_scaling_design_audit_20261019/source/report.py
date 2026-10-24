from audit import *
os.environ['MPLCONFIGDIR']=str(X/'results/matplotlib')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.backends.backend_pdf import PdfPages
import re,unicodedata
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'pdf.fonttype':3,'font.size':10})
NAMES={'joint':'未来梯度进入编码器','route_current_only':'编码器仅当前损失'}
COLORS={'joint':'#2166AC','route_current_only':'#D6604D'}
def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(str(x)for x in row)+' |'for row in rows)+'\n'
def fmt(x,n=3):return '未定义'if x is None or x==''else f'{float(x):.{n}f}'
def wrap(t,width=90):
 out=[]
 for line in t.split('\n'):
  cur='';n=0
  for tok in re.findall(r'[A-Za-z0-9_.:/+−–×-]+|.',line):
   w=sum(2 if unicodedata.east_asian_width(c)in'WF'else 1 for c in tok)
   if n+w>width:out.append(cur);cur='';n=0
   cur+=tok;n+=w
  out.append(cur)
 return '\n'.join(out)
def main():
 guard();proposal=read(X/'configs/proposed_protocol.json');feas=read(X/'results/design_feasibility.json');old=read(OLD/'results/scores.json');rows=list(csv.DictReader((R/'saved_prediction_strata.csv').open()));rel=list(csv.DictReader((R/'relation_confusions.csv').open()));pairs=list(csv.DictReader((R/'saved_prediction_paired.csv').open()))
 def metric(seed,arm,mode,fact,st='all',split='dev',kind='accuracy',change='all'):
  return next(r for r in rows if r['seed']==str(seed)and r['arm']==arm and r['mode']==mode and r['fact']==fact and r['stratum']==st and r['split']==split and r['metric']==kind and r['change']==change)
 def count(r):return f"{round(float(r['micro'])*int(r['records']))}/{r['records']}"
 geom=[]
 for seed in [0,1]:
  for arm in ARMS:
   a=dict(np.load(OLD/f'data/predictions_{arm}_seed{seed}_dev.npz'))
   for mode in ['current','free3']:
    m=a['mode']==mode;p=a['prob'][m];y=a['cat'][m,:3];yp=p[:,:24].reshape(-1,3,8).argmax(-1);wrong=yp!=y;tablewrong=wrong&(y<3)&(yp<3)
    geom.append({'seed':seed,'arm':arm,'mode':mode,'object_class_errors':int(wrong.sum()),'within_table_left_center_right_errors':int(tablewrong.sum()),'other_object_class_errors':int(wrong.sum()-tablewrong.sum()),'fraction_tablebins':float(tablewrong.sum()/wrong.sum())})
 writecsv(R/'object_error_types.csv',geom)
 addendum={'utc':now(),'status':'PROPOSED_NOT_EXECUTED','root_memberships_unchanged':True,'main_fit_budget_unchanged':8,'known_old_evidence_motivation':'CurrentfourfieldNLL includes coarse tablebins and isnot relation9coherence. Savedpredictions show differing patterns; predeclare both before any newfit.','co_primary_readouts':['samefourfieldNLL for probe andfree3, retain historical definition','relation9_exact for probe andfree3 atsamefixedthresholds, reported alongsideNLL, no scalarcombinedscore'],'secondary':['all5/all7exact','Holding/In precision, recall, BA and multiple-holding violations','changed/unchanged fields','zero-control/nonzero/executedfailure','current,one-step,free1/free3/free4','Cinitialheld4roots vsnonheld16roots'],'future_interpretation':'A lowerNLL without coherentrelation/endpoint benefit is limited averageprobability gain, not generalizedsharedstate competence. No perfection gate or posthocthreshold changes.','C_caveat':'Cchanges scene andheld-object→second-object order. CminusABis externaldistributiongap, not pure scene causal effect. Report raw scores, each initialtype, and predeclared0.5closed+0.3openempty+0.2openheld standardization for bothABtest andC; descriptive only, no selecting favorable subgroup.','batch_pairing':'Within eachsize/seed, exactsame materializedbatches andlatent RNG starts forbothgradientarms. Across sizes usepaired initialization/trainingseeds; samplecontents necessarilydiffer, not claimedidentical.','test_timing':'Freeze all8finalmodels and8probes before ABtest/Cpredictions; validation neverselectscheckpoint/threshold; technicalfailure repair canresume fixedbudget but no extrafits.','new_fits_executed':0}
 save(X/'configs/proposed_measurement_addendum.json',addendum)
 fig,axs=plt.subplots(2,2,figsize=(12,9))
 for seed in [0,1]:
  for j,arm in enumerate(ARMS):
   rr=[metric(seed,arm,'probe',f)for f in ['primary4_mean_accuracy','relation9_exact','all5_exact']];v=np.array([float(r['value'])for r in rr]);lo=np.array([float(r['lo95'])for r in rr]);hi=np.array([float(r['hi95'])for r in rr]);axs[0,seed].bar(np.arange(3)+(j-.5)*.33,v,width=.33,color=COLORS[arm],label=NAMES[arm],yerr=[v-lo,hi-v],capsize=3)
   rr=[metric(seed,arm,'free3','relation9_exact',st)for st in ['initial_not_held','initial_open_held']];v=[float(r['value'])for r in rr];xx=np.arange(2)+(j-.5)*.33;axs[1,seed].bar(xx,v,width=.33,color=COLORS[arm],label=NAMES[arm])
   for x,y,r in zip(xx,v,rr):axs[1,seed].text(x,y+.025,count(r),ha='center',fontsize=10)
  axs[0,seed].set_xticks(range(3),['四字段平均准确率','九项关系全对','五字段全对']);axs[0,seed].set_title(f'冻结E线性查询 · 种子{seed} · C20根/300边界');axs[0,seed].set_ylim(0,1.1);axs[0,seed].set_ylabel('正确比例（越高越好）');axs[0,seed].legend(fontsize=8)
  axs[1,seed].set_xticks(range(2),['非持物初态\n16根/48条终点','持物初态\n4根/12条终点']);axs[1,seed].set_title(f'原RSSM三步终点九关系 · 种子{seed}');axs[1,seed].set_ylim(0,1.17);axs[1,seed].set_ylabel('九项关系同时正确比例');axs[1,seed].legend(fontsize=8,loc='upper right')
  for ax in axs[:,seed]:ax.grid(axis='y',alpha=.2)
 fig.suptitle('平均查询收益与关系一致性不是同一个结果',fontsize=16);fig.tight_layout(rect=(0,.16,1,.94))
 caption='蓝：未来梯度进入编码器；橙：编码器仅当前损失。上图同一C20根、300边界，误差线是2000次整根重采样95%区间。四字段=三物体位置/归属＋抽屉；九关系=3Holding＋3In＋抽屉3分类对应事实；五字段再含滑门。下图是三请求后的终点，不代表三个中间状态全对。C已经用于开发，持物初态还含不同对象顺序，不能把组差唯一归因于场景或组合。'
 fig.text(.06,.025,wrap(caption,150),va='bottom',fontsize=9);fig.savefig(R/'figures/coherence_and_initial_state.png',dpi=170,bbox_inches='tight');fig.savefig(R/'figures/coherence_and_initial_state.pdf',bbox_inches='tight')
 coverage=feas['coverage'];covtable=table(['划分','根','分支/转移','关空/开空/开持','非零/零控制','三次均执行分支'],[[c['group'],c['roots'],f"{c['episodes']}/{c['transitions']}",'/'.join(str(c['initial_states'][k])for k in ['closed_empty','open_empty','open_held']),f"{c['nonzero']}/{c['zero_control']}",c['rootstart3_all_actuated']]for c in coverage])
 cohrows=[]
 for seed in [0,1]:
  for mode in ['probe','free3']:
   for arm in ARMS:
    r=metric(seed,arm,mode,'relation9_exact');f=metric(seed,arm,mode,'all5_exact');mult=metric(seed,arm,mode,'multiple_holding');cohrows.append([seed,NAMES[arm],mode,count(r),count(f),count(mult)])
 reject=[]
 for seed in [0,1]:
  for arm in ARMS:
   reject.append([seed,NAMES[arm]]+[fmt(metric(seed,arm,'one_step','primary4_mean_accuracy',s,kind='NLL')['value'])for s in ['zero_control','nonzero_control','executed_failure']])
 relations=[]
 for seed in [0,1]:
  for arm in ARMS:
   for name in ['Holding','In']:
    v=next(v for v in rel if v['seed']==str(seed)and v['arm']==arm and v['split']=='dev'and v['mode']=='probe'and v['stratum']=='all'and v['relation']==name)
    relations.append([seed,NAMES[arm],name,fmt(v['precision']),fmt(v['recall']),fmt(v['BA']),f"{v['TP']}/{v['FP']}/{v['FN']}/{v['TN']}"])
 fields=[]
 for seed in [0,1]:
  for arm in ARMS:
   for name in ['red_location','blue_location','pink_location','drawer','slider']:
    ch=metric(seed,arm,'one_step',name,change='changed');un=metric(seed,arm,'one_step',name,change='unchanged');fields.append([seed,NAMES[arm],name,f"{fmt(ch['value'])} / {ch['records']}",f"{fmt(un['value'])} / {un['records']}"])
 gaps=[]
 for seed in [0,1]:
  for arm in ARMS:
   for mode in ['current','probe','free3']:
    a=metric(seed,arm,mode,'primary4_mean_accuracy',split='train',kind='NLL');b=metric(seed,arm,mode,'primary4_mean_accuracy',split='dev',kind='NLL');c=metric(seed,arm,mode,'relation9_exact',split='train');d=metric(seed,arm,mode,'relation9_exact',split='dev');gaps.append([seed,NAMES[arm],mode,fmt(a['value']),fmt(b['value']),count(c),count(d)])
 testids='、'.join(proposal['members']['test10']);validids='、'.join(proposal['members']['validation10'])
 md=f'''# 规模与泛化：指标澄清、保存预测审计和最小下一轮设计

{now()}。本轮新增训练0、模型推理0、采集0、D样本读取0；旧135文件封存保持原位。新根名单只按一次固定SHA256排序及环境/初态配额决定，未按模型表现挑根、换盐或重抽。下面的下一轮方案尚未执行。

**当前证据更具体的解释：平均类别概率改善，并未稳定变成关系组合同时正确。大部分完整位置字段错误是台面左/中/右混淆；原RSSM的三步九关系在非持物C初态47/48全对，但持物初态仅2–4/12全对。C同时存在场景和初始关系/对象顺序差异，不能把总差距都解释成数据量不足或纯场景变化。**

## 上一轮引用的误差究竟是什么

|口径|严格定义|实际分母与不包含的内容|
|---|---|---|
|查询误差|冻结E输出512维，用同预算600步Linear512→32预测；取三物体各8类位置/归属＋抽屉3类，四字段**无类别权重**NLL平均|C300边界记录=20根×3分支×5边界，1200个有效字段；A/B900边界、3600字段。不是九关系错误率，也不是语言查询|
|三步误差|仅由真实根观测初始化RSSM，沿该分支真实三次请求递推，不读中途观测；8条潜轨迹的类别概率先平均，再取四字段NLL|C60终点、240字段、20根；三次均非零执行仅19/60，主分母保留拒绝。A/B180终点、720字段、60根。不是三步前缀全对率|
|四字段NLL|每字段 −ln(max(p真实类别,1e−9))；先四字段平均，再根内记录平均，最后根等权|单位nat，越低越好；训练与开发均由保存预测按此同一口径评分，不是训练loss曲线|
|所报两组差|未来梯度组NLL减仅当前组；相同根先求差，再2000次整根bootstrap|负值有利；种子0/1分开。区间只描述这批根的抽样变化，不包含全部训练随机性或新分布不确定性|
|五字段全对|三物体8分类＋抽屉3分类＋滑门3分类均argmax正确，全部标签有效|它包含台面左中右等粗位置，不只是关系；C查询300或三步60记录|
|七字段全对|五字段再加两盏灯的0.5二值分类均正确|两灯几乎固定，不能用其正确证明动态能力|
|本次九关系审计|Holding(red/blue/pink)取物体类别6概率≥0.5；In取类别3概率≥0.5；抽屉3类argmax对应三个互斥事实，要求全部正确|九个布尔事实并非独立；与五字段指标不可互换。它也不是完整开放世界状态|

物体8类为台面左/中/右、抽屉、两个滑门区、持物、其他（0–7）；抽屉为闭/中间/开（0–2）。模型输出32个logit不等于32个事实：3×8＋3＋3＋2实际对应7字段。原训练及probe训练用原类别权重和7字段损失；上述评价四字段NLL不加这些训练权重。

“改变事实”在一步中按真实前后标签是否改变判定；三步中指终点相对根的**净改变**，不代表中间从未变化。所有有效字段计分，不按模型是否预测对挑选。三步仍是请求尺度，零控制拒绝也占一个请求。保存文件没有free2，因此本次不伪造完整1–3步前缀正确率。

## 保存预测揭示了什么

![平均与一致性](figures/coherence_and_initial_state.png)

{caption}

{table(['种子','条件','路径','九关系全对','五字段全对','预测同时持多个物体'],cohrows)}

线性probe九关系全对的配对差：种子0 −1.0百分点〔−6.7,+4.0〕，种子1 +2.0百分点〔−1.0,+5.3〕；原RSSM三步九关系差分别+1.7〔−5.0,+10.0〕和−3.3〔−10.0,0.0〕。这些是旧结果事后审计，不能用它们替换已封存的主指标或调阈值；它们说明四字段平均概率收益未稳定传递到完整关系组合。

多持物是跨物体的冲突，probe在C可达52–76/300；同物体Holding与In共享同一8类softmax概率预算，两者概率之和不能超过1；这种输出结构限制不能算成学会了物理规律。原RSSM完整读出还使用历史/动态隐状态，不能把其较高关系分数直接归功于单独的E或这个线性probe。

三步物体分类错误中，台面左中右之间混淆为seed0未来组57/74、控制58/79，seed1未来组53/73、控制70/85。即约73%–82%的三步物体分类错误属于这类粗位置混淆。故低五字段全对不能直接翻译为“Holding/In都不可读”；反过来九关系正确也不能保证完整位置或过程状态正确。

### 训练与开发的差距

{table(['种子','条件','路径','A/B训练NLL','C开发NLL','A/B九关系全对','C九关系全对'],gaps)}

原RSSM在A/B训练集的当前与三步九关系均全对，C仍有缺口；训练NLL也明显低于C。线性probe与完整RSSM的差距则提示其输入和读出能力不同。这里的A/B全部是旧模型的训练根，不能把此表当成本轮预留根泛化结果；它也不能区分过拟合、场景变化和上下文组合变化各自的因果份额。

### 改变与不变字段

下表为C一步根平均分类准确率/有效字段数；不是所有字段微平均。静态灯不在此表，滑门改变分母仅1。

{table(['种子','条件','字段','改变：准确率/N','不变：准确率/N'],fields)}

### 拒绝与执行分组

{table(['种子','条件','零控制41条NLL','非零199条NLL','其中执行失败5条NLL'],reject)}

零控制组并未消失；非零组同样有明显错误。执行失败只有5条，不能以其波动做总体结论。“success”沿用执行器记录，不自动等同于每个目标事实满足。完整分组根数、置信区间、初态分层见saved_prediction_strata.csv。

### 关系类型

下表汇总C300个probe边界×3颜色=900个有效二值答案；P是精确率，R是召回率，BA=(召回＋特异度)/2，零分母留空。混淆计数聚合在答案层面，与根平均准确率不是同一估计量。

{table(['种子','条件','关系','P','R','BA','TP/FP/FN/TN'],relations)}

Holding的收益不是两种子一致；In召回有所上升但假阳性多。应把平均概率、单关系判别和整组一致性分开看，不用一个平均分代替全部研究目标。

## 为什么旧20/40/60方案不能直接执行

旧方案只能在固定C20根上作规模比较，没有预留A/B根。若再从这60个根抽A/B测试，60根训练臂已见过全部测试，构成泄漏；旧60根训练模型不能进入新共同测试比较。它只保留为历史参考。

还需补充旧“C22种请求顺序都在AB出现”的边界：只看动作字符串确实如此，但初始持物颜色→后来抓取颜色在AB是红→蓝、蓝→粉、粉→红，C是红→粉、蓝→红、粉→蓝。源证据是core1200_prepare.py及roots_manifest.json。即C具有场景与上下文对象顺序的联合变化；新方案以A/B新根作主评估，C仅作外部分布诊断，并分开其4个持物根和16个非持物根。

## 一套最小且无根泄漏的方案（等待执行决策）

{covtable}

20根完全包含于40根；两档均A/B各半，关屉空手/开屉空手/开屉持物比例为50/30/20%。每根3个四请求分支全部跟随根，全部6技能覆盖。公共20根已有22种四请求顺序，固定A/B测试和验证的18种均在其中：这轮主要检验同技能/已见请求结构下的**新物理根泛化**，不宣称新任务组合迁移。

成员由固定盐`{SALT}`与root_id的SHA256在“环境×初态”内排序决定。每环境测试配额关空3/开空1/开持1，验证2/2/1，余下20训练根，再按5/3/2选10根作小档。一次排序后名单固定，未根据性能换名单。跨集合无同根、无双RGB＋本体感觉逐数组重复，也无同环境三物体位置均≤12mm的匹配；这不是对所有可能近重复的完备认证。

固定A/B测试10根：{testids}。

固定A/B验证10根：{validids}。

完整20/40/C名单见[root_memberships.csv](root_memberships.csv)。测试初始持物仅蓝/粉2根，没有初始持红根；但全部轨迹中三色Holding正例13/13/14、In正例13/23/24均存在。验证缺初始持粉，主测试滑门无变化，失败调用仅2条。必须保留这些覆盖限制，不换根追求好看结果。

### 所有拟合与使用边界

1. 从相同种子的随机初始权重开始，不载入旧训练模型。冻结DINO原始缓存可复用；旧60根归一化、类别权重、已训练E/R/query一律不能进入新拟合。
2. 输入归一化及类别权重只在公共20训练根重算，再在全部规模和两臂固定，避免40根通过预处理改变额外因素。probe也只用同一公共20根的300标注边界、同600×32抽样和同初始化；各模型probe标准化只拟合这同300边界。这样规模差异不会混入query标签预算翻倍。
3. 两档各2种子×2梯度条件，共8条主训练，每条6000更新，前5000单步、末1000三步，原batch/优化器/学习率/分组裁剪不变，只取6000终点。每个规模内两臂严格共享batch及潜变量随机起点；规模之间训练种子配对，但样本内容随根集合改变。
4. 固定验证10根仅报告运行诊断，不选检查点、不改阈值。全部8个主模型及8个probe冻结后才统一评分A/B测试10根和C20根；不据测试结果选择规模、种子、子集或继续修补。已见旧数据全部明确标为探索性重新划分，“新根”仅指本轮优化器未见，并非全新盲测。
5. 同时报四字段NLL、九关系全对、五/七字段全对及关系混淆；每种子计算同一测试根的40减20配对差，再比较两梯度条件的改善差。不将它们加权成一个事后总分，也不设“局部谓词必须完美”门槛。

### 有界预算与能回答的不确定性

建议上限：**8条主训练×6000=48,000更新；8个probe×600=4,800更新**。按已完成四条的真实耗时估计，主训练约{feas['remaining_main_fit_budget_minutes']:.1f}分钟；连同评估、统计与核验预留约3小时本机串行时间。GPU峰值参考约1.2GiB，不因剩余显存扩大batch或额外运行。无新采集、D、闭环、局部谓词修补或新增架构。此轮只提出预算，实际启动0条。

固定6000更新使两档都抽96,000条完整分支：20根60分支约1600次等效遍历，40根120分支约800次。它回答“**相同训练计算下，增加独立经验是否改善同场景未见根的共享状态及转移**”，不回答等暴露次数条件。等暴露需另外预声明20根3000与40根6000等安排；旧3000检查点尚无多步项，不能拿来冒充该对照。本建议预算不包含等暴露实验。

只有20→40一个规模差，最多给出这一区间的配对变化，不能拟合或宣称scaling law、不能外推6000转移、更不能由两点保证单调性。只有一套嵌套根名单和两个训练种子，也不足以估计更换样本子集的全部不确定性。若A/B改善而C仍差，支持这批已见任务结构内的新根受益、外部变化仍未解决；若仅NLL改善而九关系/终点不改善，则平均概率收益尚未转成连贯状态；若两者都未改善，不能凭两点断言更多独立经验永远无效。

本轮仍是带符号监督的经验规模比较，不是无命题标签交互学习。固定六技能、三色身份和32输出限制了表示范围。预先固定同时报告原始分数、各初态分数和按初态50/30/20%标准化的分数；标准化后的C减A/B差距仍只是外部分布诊断，不是纯场景变化的因果效应。

## 核验与交付

原135文件封存复验保持SHA256不变；本轮只读取旧预测和A/B/C元数据/标签，未实例化训练网络。新源、固定根名单、方案及测量补充在新实验configs/source；分层CSV、关系混淆、根当前到三步终点配对和物体错误类型在本报告目录。方案未启动，不要求通过旧C表现再决定抽哪些根。
'''
 (R/'REPORT.md').write_text(md)
 page1='\n\n'.join(['这轮做了什么：只读保存预测与根元数据；新增训练、模型推理、采集均为0。旧模型与协议保持封存。','上一轮“查询误差”：冻结E的线性头，三物体各8类位置/归属＋抽屉3类的四字段无权重NLL。不是九关系错误率。C300边界/20根；三步误差为根起点连续三次请求后的60终点/20根，8条潜轨迹概率平均后取负对数。','五字段全对包含台面左中右位置及滑门；七字段再含灯。九关系是3Holding、3In与抽屉三类对应事实。三步终点正确不代表三个中间状态都正确；保存数据没有free2，不编造前缀准确率。','新方案：A/B60根固定分成训练池40、验证10、测试10；训练池中嵌套20。两档均A/B各半、三初态50/30/20%。所有兄弟分支随根；名单一次哈希排序，不按性能选根。旧60根模型见过全部新A/B测试，只作历史参考。','训练和probe预处理、类别权重及probe300标注边界均只用公共20根并固定。八条主模型/八个probe全部冻结后才评估共同A/B测试和C。','预算尚未执行：2规模×2种子×2梯度条件=8条，各6000更新；probe8条各600。主训练实测外推约145.5分钟，含分析核验预留约3小时单卡串行。没有新采集、D或额外架构。','6000更新固定计算，却使20/40根重复暴露约1600/800次。这是等计算的独立经验比较，不是等暴露；只有一个20→40差值，不声称规模定律或外推规律。'])
 page3='\n\n'.join(['当前分解：平均概率收益没有稳定转化成九关系组合。probe九关系差seed0 −1.0百分点、seed1 +2.0百分点；配对区间均跨0。其多持物冲突达52–76/300，不能把平均NLL降低视作全状态连贯。','三步物体类别错误中约73%–82%只是台面左中右混淆。因此低五字段全对不等于Holding/In全都读不出；反之关系正确也不代表完整位置状态正确。','C非持物初态：四模型三步九关系均47/48全对（16根）。C初始持物：仅2–4/12全对（4根）。这仍不是规划闭环成功，也不是自由滚动整个前缀可靠。','C不只是场景变了：A/B初始持物→第二对象为红→蓝、蓝→粉、粉→红；C为反向关系。C22种纯请求字符串虽都见过，带初始关系的组合却不同。C总差距不能唯一归于场景或数据量。','新A/B测试10根中全部六技能和三色关系正例都有，但初始持物只有蓝/粉2根，滑门无变化，执行失败仅2条。新20根已涵盖22种请求顺序，此比较针对同任务结构下未参与本轮拟合的物理根，不能称未见任务组合迁移。','若扩大根数使A/B新根的查询、九关系完整性与三步终点一起改善，才支持该计算预算下更多独立经验帮助可复用状态与转移。若只改善NLL，需保留收益边界；若只改善A/B而C不改善，不能掩盖外部变化缺口。','全部旧数据都已被观察，新的根划分仅相对本轮优化器隔离，仍是探索性。名单、完整分母、区间和配置见REPORT.md及CSV。下一轮没有开始，等待对这一明确预算和设计的执行决策。'])
 with PdfPages(R/'规模与泛化_只读审计与下一轮方案_20261019.pdf')as pdf:
  for title,txt in [('指标定义与最小规模方案',page1),('figure',None),('当前证据与能回答的问题',page3)]:
   if title=='figure':pdf.savefig(fig);continue
   page=plt.figure(figsize=(8.27,11.69));page.text(.07,.95,title,fontsize=19,va='top');wrapped=wrap(txt,88);page.text(.07,.89,wrapped,fontsize=min(11,680/(len(wrapped.splitlines())*1.65)),va='top',linespacing=1.65);pdf.savefig(page);plt.close(page)
 plt.close(fig)
 save(X/'results/report_receipt.json',{'utc':now(),'PDF_pages':3,'new_fits':0,'new_model_inference':0,'new_collection':0,'proposed_main_fits':8,'proposed_total_main_updates':48000,'proposed_probe_fits':8,'proposed_total_probe_updates':4800,'same_compute_only':True,'no_scaling_law_claim':True,'summary':'Averagequery NLL gain doesnot consistently improve relation9coherence. CurrentC combines scene andheld-object-order shifts; nextproposal holdsoutABroots and fixesprobe labels/normalization oncommontrain20.'})
 print('REPORT_COMPLETE_NO_FITS')
if __name__=='__main__':main()
