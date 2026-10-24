from common import *
import shutil,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=fm.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
def tab(headers,data):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in data)+'\n'
def pct(v):return '—'if v is None else f'{v*100:.2f}'
LABEL={'context_zero':'上下文置零','context_present':'保留上下文'};COL={'context_zero':'#7a7f87','context_present':'#216fa8'}
def main():
 guard('report');ev=read(E/'results/evaluation.json');v=read(E/'results/independent_verification.json');m={(r['arm'],r['seed'],r['split'],r['group']):r for r in ev['metrics']};cis={(c['arm'],c['seed'],c['scope'],c['metric']):c for c in ev['root_bootstrap_CI']};diff={(d['seed'],d['scope'],d['metric']):d for d in ev['paired_differences']};cases=ev['cases'];fig,axs=plt.subplots(2,2,figsize=(15,10));ax=axs[0,0]
 for arm in ARMS:
  for seed in [0,1]:
   for k,split in enumerate(['train','dev']):
    x=seed*2+k+(-.10 if arm=='context_zero'else .10);r=m[arm,seed,split,'target'];c=cis[arm,seed,'target','balanced_accuracy']if split=='dev'else None;err=[[100*(r['balanced_accuracy']-c['low95'])],[100*(c['high95']-r['balanced_accuracy'])]]if c else None;ax.errorbar(x,100*r['balanced_accuracy'],yerr=err,fmt='o'if arm=='context_zero'else's',color=COL[arm],capsize=4,ms=7);ax.annotate(pct(r['balanced_accuracy']),(x,100*r['balanced_accuracy']),xytext=(0,-16 if arm=='context_zero' else 8),textcoords='offset points',ha='center',fontsize=9)
 ax.set_xticks(range(4),['训练·种子0','开发·种子0','训练·种子1','开发·种子1']);ax.set_ylim(70,107);ax.set_ylabel('目标查询平衡准确率（%）');ax.set_title('全部目标：训练353条，开发166条');ax.grid(axis='y',alpha=.2)
 ax=axs[0,1]
 for arm in ARMS:
  for seed in [0,1]:
   for k,split in enumerate(['train','dev']):
    x=seed*2+k+(-.10 if arm=='context_zero'else .10);r=m[arm,seed,split,'supported_negative'];c=cis[arm,seed,'supported_negative','false_positive_rate']if split=='dev'else None;err=[[100*(r['false_positive_rate']-c['low95'])],[100*(c['high95']-r['false_positive_rate'])]]if c else None;ax.errorbar(x,100*r['false_positive_rate'],yerr=err,fmt='o'if arm=='context_zero'else's',color=COL[arm],capsize=4,ms=7);ax.annotate(f"{r['FP']}/{r['N']}",(x,100*r['false_positive_rate']),xytext=(0,9 if arm=='context_zero'else-16),textcoords='offset points',ha='center',fontsize=10)
 ax.set_xticks(range(4),['训练·种子0','开发·种子0','训练·种子1','开发·种子1']);ax.set_ylim(-12,112);ax.set_ylabel('桌面支撑负例误报率（%）');ax.set_title('闭爪且桌面支撑：标注数字为误报条数/负例数');ax.grid(axis='y',alpha=.2)
 ax=axs[1,0]
 for seed in [0,1]:
  d=diff[seed,'target','balanced_accuracy'];ax.errorbar(d['estimate']*100,seed,xerr=[[100*(d['estimate']-d['low95'])],[100*(d['high95']-d['estimate'])]],fmt='o'if seed==0 else's',color='#216fa8',capsize=5,ms=8);ax.annotate(f"{100*d['estimate']:+.2f} [{100*d['low95']:+.2f}, {100*d['high95']:+.2f}]",(100*d['estimate'],seed),xytext=(0,14),textcoords='offset points',ha='center',fontsize=10)
 ax.set_yticks([0,1],['种子0','种子1']);ax.set_ylim(-.5,1.6);ax.axvline(0,color='gray',ls='--');ax.set_xlabel('保留上下文−置零：目标BA变化（百分点）');ax.set_title('配对根级区间仍跨零，未达到冻结的主要增益标准');ax.grid(axis='x',alpha=.2)
 ax=axs[1,1];ident=[('relation36_C_24','static'),('relation36_C_24','gripper'),('relation36_C_34','static'),('relation36_C_34','gripper')];conditions=[('context_zero',0),('context_present',0),('context_zero',1),('context_present',1)];probs=np.array([[next(c['probability']for c in cases if c['root_id']==root and c['camera']==cam and c['anchor']=='closed_on_support'and c['arm']==arm and c['seed']==seed)for arm,seed in conditions]for root,cam in ident]);im=ax.imshow(probs,vmin=0,vmax=1,cmap='RdBu_r',aspect='auto');ax.set_xticks(range(4),['置零·种子0','保留·种子0','置零·种子1','保留·种子1']);ax.set_yticks(range(4),['C24 固定','C24 腕部','C34 固定','C34 腕部']);ax.set_title('代表性支撑负例（全部真值0）：Holding预测概率')
 for i in range(4):
  for j in range(4):ax.text(j,i,f'{probs[i,j]:.3f}',ha='center',va='center',color='white'if abs(probs[i,j]-.5)>.3 else'black',fontsize=10)
 cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.03);cb.set_label('概率；红色>0.5为误报，蓝色<0.5为正确负例')
 fig.suptitle('同刻固定相机全图上下文：减少部分误报，支撑关系仍未可靠泛化',fontsize=17);fig.text(.05,.025,'灰圆：上下文置零；蓝方：保留上下文。两组均389→33→1，冻结编码器，原批次，每次600步，阈值0.5。\n开发区间按完整12根配对重采样2,000次；支撑误报率有1次零分母，使用1,999个有效重采样。训练点仅为拟合值，无置信区间。\n正确原生框＋同步多视角上下文，已见C上的探索性诊断；不与单腕部方法作等传感器预算比较，不表示已经显式学会支撑。',fontsize=10);fig.tight_layout(rect=[0,.12,1,.94]);fig.savefig(R/'figures/01_static_context_diagnostic.png',dpi=150);fig.savefig(R/'figures/01_static_context_diagnostic.pdf');plt.close(fig)
 overall=tab(['条件','种子','开发P%','开发R%','开发BA%','开发NLL','TP/FP/FN/TN'],[[LABEL[a],s,pct(m[a,s,'dev','all']['precision']),pct(m[a,s,'dev','all']['recall']),pct(m[a,s,'dev','all']['balanced_accuracy']),f"{m[a,s,'dev','all']['NLL']:.4f}",'/'.join(str(m[a,s,'dev','all'][k])for k in ['TP','FP','FN','TN'])]for s in [0,1]for a in ARMS])
 primary=tab(['条件','种子','目标P%','目标R%','目标BA%','目标FP/106','支撑负例FP/12','训练支撑FP/8'],[[LABEL[a],s,pct(m[a,s,'dev','target']['precision']),pct(m[a,s,'dev','target']['recall']),pct(m[a,s,'dev','target']['balanced_accuracy']),m[a,s,'dev','target']['FP'],m[a,s,'dev','supported_negative']['FP'],m[a,s,'train','supported_negative']['FP']]for s in [0,1]for a in ARMS])
 pairtable=tab(['种子','范围/指标','保留−置零','配对根级95%区间','有效重采样'],[[s,scope+'/'+metric,f"{diff[s,scope,metric]['estimate']*scale:+.4f}",f"[{diff[s,scope,metric]['low95']*scale:+.4f}, {diff[s,scope,metric]['high95']*scale:+.4f}]",diff[s,scope,metric]['valid_replicates']]for s in [0,1]for scope,metric,scale in [('target','balanced_accuracy',100),('supported_negative','FP',1),('supported_negative','false_positive_rate',100),('all','precision',100),('all','NLL',1)]])
 subgroup=tab(['条件/种子','颜色或相机','正/负查询','P%','R%','BA%'],[[LABEL[a]+'/'+str(s),g.replace('color_','').replace('camera_',''),f"{m[a,s,'dev',g]['positive_queries']}/{m[a,s,'dev',g]['negative_queries']}",pct(m[a,s,'dev',g]['precision']),pct(m[a,s,'dev',g]['recall']),pct(m[a,s,'dev',g]['balanced_accuracy'])]for a in ARMS for s in [0,1]for g in ['color_red','color_blue','color_pink','camera_static','camera_gripper']])
 train_table=tab(['条件','种子','训练总体BA%','开发总体BA%','训练目标BA%','开发目标BA%','训练NLL','开发NLL'],[[LABEL[a],s,pct(m[a,s,'train','all']['balanced_accuracy']),pct(m[a,s,'dev','all']['balanced_accuracy']),pct(m[a,s,'train','target']['balanced_accuracy']),pct(m[a,s,'dev','target']['balanced_accuracy']),f"{m[a,s,'train','all']['NLL']:.4f}",f"{m[a,s,'dev','all']['NLL']:.4f}"]for s in [0,1]for a in ARMS])
 case_table=tab(['支撑负例，真值均0','置零s0','保留s0','置零s1','保留s1'],[[root+'/'+cam]+[f'{p:.6f}'for p in line]for(root,cam),line in zip(ident,probs)])
 future='''# 后续受控动作条件转移试采：需要的证据（本轮未启动）

这是进入后续诊断前应冻结的证据要求，不是已执行方案，也不替代原指南的720→约6000转移、多技能族与世界模型实际供给TS/LTLf要求。没有创建转移数据或训练模型。

| 需要证明什么 | 受控对照或核验 | 为什么需要 |
| --- | --- | --- |
| 动作导致后继差异 | 同一原生初态/同一历史的lift与hold分支，起始图像与接触一致，按根及全部后代同划分 | 防止不同初态被当作动作效果 |
| 起始状态也不可缺少 | 包含相同动作在不同接触/夹持起态下的不同实际后继；平衡动作与结果，保存拒绝和失败 | 只从受支撑成功夹持状态比较lift/hold时，action-only即可解题，不能证明世界模型理解状态 |
| 真实物理执行且预算可控 | 固定已核验的执行器版本与安全边界；原生复位加确定性重放，或预先批准的仅分支起点检查点恢复；禁止执行中瞬移、强制附着/接触修改 | 执行器尚不能因本轮静态分数而获得晋级；重放一致性要先验证 |
| 视觉状态与动作共同带来预测增益 | persistence、仅视觉状态、仅动作、视觉状态＋动作的相同结构/预算消融；保存所有种子和训练/开发差异 | 排除不变性、动作标签或初态分布捷径 |
| 确实预测未来，未读取未来 | 模型输入只含起点图像/允许历史和真实动作标识；后继图像/接触只作监督与评分；同步相机并冻结特征边界 | 防止把后继观测泄漏当成预测 |
| 不以大量容易负例掩盖变化错误 | 单独报告目标对象、支撑消失/出现、保持不变、动作失败和弃权；用独立根的配对区间 | 本轮全量负例中大量为非目标物体，困难负例覆盖很稀少 |
| 物理真值可靠但不冒充视觉输入 | 接触来源、三阈值与稳定窗口核验；GT仅标签/评分，初始读出误差与转移预测误差分开诊断 | 不能用当前不可靠硬Holding当作正确起点，也不能手写动态前提/效果替代学习 |
| 结果能推广且遵守原指南 | 先冻结新根配额、分支继承/哈希排除、预算与停止条件；训练A/B、开发C，D继续封存直至正式测试阶段；沿用原指南规模与验收要求 | 既有C已反复观察，不能再次称独立确认集，针对lift/hold的小试不等于多技能世界模型完成 |

静态读出优化到此关闭。上面列出的是后续行动所需证据；本次没有为了获得更高静态分数再选择模型、阈值、负例或种子。\n'''
 (R/'NEXT_TRANSITION_EVIDENCE.md').write_text(future)
 report=f'''# 最后一次静态读出诊断：同刻固定相机全图上下文

**增加上下文减少了部分误报、改善总体精确率和NLL，但目标BA与桌面支撑误报的主要配对区间没有排除零，C24腕部/C34错误仍在。因此未达到事先冻结的“明确主要增益”规则，静态读出优化到此关闭。** 这不是“图像没有信息”的结论，也不表示模型已显式学会物理支撑。

本轮只使用原36根：训练A/B24根、开发C12根，264已有状态、920训练/470开发可用查询。原数据、标签、失败、不可见弃权和根划分全部不变；无采集、渲染、定位、SupCon、In、世界模型/转移训练或闭环；D样本/特征/预测/评分未读。C已用于前几轮，全部结果为探索性机制诊断。

## 冻结方案与输入

协议在特征编码和任何新拟合之前冻结，SHA `41eb1d7fde7ae371e2680ce8b5c741f8301175251de228f250aed4732a5636b9`。完整协议见FROZEN_PROTOCOL.json。两组统一**389→33→1、SiLU、12,904参数**：原261维配对RGB＋二维框几何不动，追加128维同状态固定相机完整RGB特征。保留组使用全部维度，置零组在共同归一化后将最后128维精确置零。隐藏宽度33只选一次，无搜索。

264张原始全图均已核对与腕部同tick、渲染前后状态一致；编码器只读取RGB，不读取动作、阶段、状态、接触、根ID、tick或标签。复用本机缓存DINOv2-small，冻结权重；双线性224缩放、ImageNet标准化、patch均值384维、原数据无关384×128投影，约1.1秒编码，未下载新权重。编码器张量hash前后保持`3a697b872089933d345fd71087af86e85ff31886cf1e2654763dc1884ef2af27`。同一状态所有颜色和两相机查询共享同一全图向量。

**这是正确原生框条件下的同步多视角上下文诊断。** 对固定相机查询是增加同视角全景，对腕部查询是增加第二相机；不可与单腕部方法作等传感器预算比较。每次全图mean-patch编码没有按目标选取支撑区域，图像中的背景、布局和姿态均可能影响输出；没有显式支撑结构或因果注意力证据。

所有389维统计只由920训练查询估计（总体std下限0.05），两组共用，前261统计逐位等于原pair统计。相同种子0/1初始化、形状、原600×32批次序列；AdamW lr0.001、weight_decay0.001、BCE正权重4.75、clip1、float32确定性CUDA、TF32关闭。共4次新拟合×600步，仅最终权重，阈值0.5，无硬负例过采样、开发调参/筛种子。新389输入的fan-in初始化与旧261模型不同，所以本轮只在匹配的两个条件内作因果消融比较，不拿旧分数直接作基线。置零自然减少可利用的自由度；相同张量形状不等于相同有效函数空间。

## 主要诊断：目标和桌面支撑

全部目标开发166查询＝60正＋106负。桌面支撑负例取原目标颜色、closed_on_support锚点、原y=0，12查询来自6个根；不是所有闭爪锚点。相同定义训练仅8查询来自4根。纯负集合P在有误报时为0、无阳性预测时未定义；R/BA始终未定义，所以主要用FP及FPR。

{primary}

预先冻结的“明确主要增益”规则要求：两个种子的目标BA配对95%下界均>0，且支撑负例FPR配对95%上界均<0。这是本轮解释规则，不是替换原指南验收标准。本次不通过，不能因总体指标改善或个别案例成功而更改规则。

沿用原2,000次完整12开发根的配对重采样；按根重复全部状态/相机/查询后重算计数和指标，避免伪独立。支撑FPR有1次抽样没有支撑负例，记未定义；其余1,999次计算区间，不填0。FP计数2,000次均定义，区间对应固定抽12根时计数波动；各次负例分母可不同。下面BA/P/FPR单位为百分点，FP为条数，NLL为原值。

{pairtable}

支撑误报从9/12降为5/12和7/12，点估计改善，但区间上界达到0；目标BA提高2.94/2.36个百分点，区间跨0。这个结果应称**部分收益、主要证据不足**，不能称没有任何作用，也不能称支撑判别已解决。

## 总体结果与训练—开发差异

{overall}

总体P提高18.01/8.67个百分点、NLL降低0.0990/0.0916，两项配对区间均不跨0；总体BA提高1.12/0.98个百分点，区间跨0。种子0多1个漏报，种子1召回不变，不能用精确率提升掩盖召回代价。

{train_table}

四个模型训练分类均100%，训练8条支撑负例全部正确，开发仍有明显误报。这说明并非600步内连训练数据都未拟合，限制在泛化/覆盖或表示使用，而不是应继续调大训练预算的直接证据。

**具体覆盖缺口：**训练支撑负例只有4根，红4查询、粉4查询、蓝0；开发为6根，三色各4查询。原总体正负门槛没有保证“每色目标闭爪桌面支撑”这一困难交叉条件。它为C34蓝块误判提供了一个可检查的分布差异，但本轮没有进行控制该因素的因果实验，不能认定它是唯一原因，也不追加样本补洞。所有样本/标签保持原状。

开发总体470查询中304是非目标负例。完整逐根、目标×颜色/相机、支撑×相机与训练结果在metrics.csv；下面颜色和相机是重叠分组，不能相加。

{subgroup}

## C24/C34是否得到解决

{case_table}

C24固定相机原已判对，加入上下文后更低；其腕部支撑负例仍给0.950/0.986。C34固定支撑负例仍给0.597/0.799，腕部仍0.944/0.984，均超过0.5。对应抬升正例都仍为正。因此两组腕部同框8查询仍是6TP＋2FP，未解决代表性错误；更没有以两个例子代替全部目标/支撑分层统计。完整两根全部已有邻居、两相机、两种子概率在alias_neighbor_predictions.csv。

上一轮实际像素与接触证据仍有效：同框并非同RGB，完整静态画面中可见抬升/桌面参照，标签与接触稳定核验一致。此次全图上下文改善了一些错误，但global mean-patch特征可能反映布局、背景或姿态相关性；无法据此宣称显式学会支撑。模型对某些同刻两相机查询仍给出冲突判定。原物理Holding可作为监督/评分目标，但此头不能作确定性规划输入。

## 关闭静态优化，保留后续转移证据要求

不再换宽度、调阈值、挑种子、做硬负例重采样或追加静态数据。当前明确限制是：罕见目标支撑负例覆盖不足、训练完全拟合而开发错误持续，以及全景均值特征未稳定利用支持关系线索。没有证明完整静态RGB本身不可辨；也没有证明只需再调一次就能可靠。

后续动作条件转移试采必须先提供：同初态不同动作的真实分支；同动作不同状态的结果差异，防止action-only捷径；persistence/仅状态/仅动作/状态＋动作匹配对照；起点/后继严格隔离、稳定物理真值；困难目标变化与失败分层、独立根配对区间和新根防泄漏。细节与原因见NEXT_TRANSITION_EVIDENCE.md。**本轮只整理要求，没有采集或训练转移，也没有替代原指南720→约6000转移及多技能规划验收。**

## 复核与交付

独立标量复算{v['metric_rows_recomputed']}行指标，展开复核{v['expanded_root_bootstrap_checks']}次根级抽样；逐条核对1,390查询的上下文路径、同状态RGBhash和缓存行，原261特征逐位不变；两种子置零列只有weight decay、偏差为0。模型/数据/输入边界与训练专用归一化均通过。封存前再核验原36根、匹配消融及更早32/8根的全部已交付文件，记录在final_verification.json。

图中灰圆为上下文置零、蓝方为保留上下文；上左训练/开发目标BA，上右支撑负例误报率并标FP/分母；下左目标BA配对变化与0虚线，下右四个真值0案例的预测概率（红色高概率表示误报）。开发线段为整根95%区间，训练点仅为拟合值。源代码、冻结协议、2初始化/4最终模型、原始预测、CSV和报告均分目录保存；原数据按路径只读复用。
'''
 (R/'REPORT.md').write_text(report)
 save(R/'OUTCOME.json',{'utc':now(),'clear_primary_gain':False,'static_readout_tuning_closed':True,'partial_evidence':'overall precision and NLL improve; targetBA CIs cross0, supported-negative error reductions CIs touch0; C24 wrist/C34 not resolved','next_transition_pilot':'evidence requirements only; no data or fitting','new_images':0,'new_fits':4,'guide_complete':False})
 (R/'RUN_COMMANDS.md').write_text('source /home/robot/research/projects/wm_symbolic/world_model_env.sh\n\n运行顺序：prepare.py（CPU）→features.py（GPU独占锁，264原图）→train.py（同GPU，四次拟合）→predict.py（CPU）→score.py→verify.py→report.py→seal.py。源码目录：'+str(E/'source')+'。只编码和拟合一次，没有新采集或重渲染。\n')
 for src in Path(__file__).parent.glob('*.py'):
  dst=E/'source'/src.name
  if dst.exists():assert sha(src)==sha(dst)
  else:shutil.copy2(src,dst)
 print('REPORT_READY',R/'REPORT.md')
if __name__=='__main__':main()
