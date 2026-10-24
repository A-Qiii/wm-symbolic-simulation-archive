from pathlib import Path
import json,os,html,csv,datetime,collections
from zoneinfo import ZoneInfo
B=Path('/home/robot/research/projects/wm_symbolic/experiments/expansion_mechanism_analysis_20261021');P=B.parent.parent;S=B.parent/'data_expansion_training_20261021';M=B.parent/'midstart_training_control_20261020';os.environ['MPLCONFIGDIR']=str(B/'cache/matplotlib')
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
rd=lambda n:json.loads((B/f'results/{n}.json').read_text())
def tab(head,rows):return'<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in r)+'</tr>'for r in rows)+'</table></div>'
def f(n,d):return f'{n}/{d} ({100*n/d:.2f}%)'
chain=rd('conditional_chain');boot=rd('grouped_bootstrap');paired=rd('paired_strata');recovery=rd('recovery_windows');drawer=rd('drawer_mechanism_summary');support=rd('terminal_vs_continuation_support');grad=rd('gradient_clip_audit');horiz=rd('same_root_horizons');curve=rd('training_convergence_summary');classbal=rd('drawer_training_class_balance');nt=rd('non_target_action_summary');target=rd('target_contrast_summary');phys=rd('physical_outcomes');obs=rd('observation_matched_gains');superv=rd('supervision_path_counts');cols={'old228':'#7b8794','new868':'#1864ab'}
def c(arm,se,mode='free'):return next(r for r in chain if(r['source'],r['arm'],r['seed'],r['cohort'],r['mode'])==('development',arm,se,'all',mode))
fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for se,ax in enumerate(axes):
 for arm in ['old228','new868']:
  for mode,marker,ls in [('free','o','-'),('observed','s','--')]:
   r=c(arm,se,mode);ax.plot([1,2,3],[100*r[k]for k in ['p1','p2_given1','p3_given12']],color=cols[arm],marker=marker,linestyle=ls,label=arm+' '+mode)
 ax.set_title(f'Seed{se}, same 320 development windows');ax.set_xticks([1,2,3],['P(C1)','P(C2 | C1)','P(C3 | C1,C2)']);ax.set_ylim(60,100);ax.grid(alpha=.2)
axes[0].set_ylabel('Conditional joint success (%)');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(B/'figures/conditional_chain.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4));rows=[]
for se in [0,1]:
 for metric in ['D_equal','D_marginal']:
  row=next(r for r in boot if(r['seed'],r['arm'],r['cluster'],r['metric'])==(se,'new868','family_id',metric));rows.append(row)
for j,r in enumerate(rows):
 color=['#1864ab','#d17b23'][r['seed']];lo,hi,pt=[100*r[k]for k in ['CI95_low','CI95_high','estimate']];ax.errorbar(pt,j,xerr=[[pt-lo],[hi-pt]],fmt='o',capsize=4,color=color)
ax.axvline(0,color='black',ls='--',lw=1);ax.set_yticks(range(4),['Seed0: joint - p1^3','Seed0: joint - p1*p2*p3','Seed1: joint - p1^3','Seed1: joint - p1*p2*p3']);ax.set_xlabel('Difference (percentage points); 95% family bootstrap CI');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(B/'figures/independence_difference_ci.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));x=np.arange(2)
for off,group,col in [(-.18,'old228','#7b8794'),(.18,'new640','#1864ab')]:
 vv=[next(r['nonterminal_boundaries']for r in support if r['group']==group and r['drawer']==d)for d in [0,1]];ax.bar(x+off,vv,.36,color=col,label='Old228 nonterminal'if group=='old228'else'Added640 nonterminal')
 for i,v in enumerate(vv):ax.text(x[i]+off,v+7,str(v),ha='center')
ax.set_xticks(x,['Two In, no Held, drawer closed','Two In, no Held, drawer middle']);ax.set_ylabel('Nonterminal boundary occurrences (#)');ax.set_ylim(0,400);ax.legend();fig.tight_layout();fig.savefig(B/'figures/continuation_coverage.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
for se,ax in enumerate(axes):
 for arm in ['old228','new868']:
  for typ,ls,m in [('endpoint_correct','-','o'),('all0_to_step_correct','--','s')]:
   rr=sorted([r for r in horiz if(r['source'],r['arm'],r['seed'])==('development',arm,se)],key=lambda r:r['step']);ax.plot([r['step']for r in rr],[100*r[typ]/80 for r in rr],color=cols[arm],ls=ls,marker=m,label=arm+' '+('endpoint'if typ=='endpoint_correct'else'all prefix'))
 ax.set_title(f'Seed{se}: fixed 80 root episodes');ax.set_xticks(range(7));ax.set_xlabel('Requested skills from same root');ax.grid(alpha=.2);ax.set_ylim(0,105)
axes[0].set_ylabel('Joint correctness (%)');fig.legend(*axes[1].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=9);fig.tight_layout(rect=[0,.09,1,1]);fig.savefig(B/'figures/fixed_root_horizons.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4))
for arm,root,name in [('old228',M,'cold_mix'),('new868',S,'new868')]:
 for se in [0,1]:
  with(root/f'logs/{name}_seed{se}.csv').open()as file:rr=list(csv.DictReader(file))
  for ax,k in zip(axes,['total','multi3','terminal_from_h2']):
   vals=np.array([float(r[k])for r in rr]);ax.plot(np.arange(100,1501),np.convolve(vals,np.ones(100)/100,mode='valid'),color=cols[arm],ls=['-','--'][se],label=f'{arm} seed{se}');ax.set_title(k);ax.set_xlabel('Optimizer updates');ax.set_ylabel('Trailing 100-update mean loss (log)');ax.set_yscale('log');ax.grid(alpha=.2)
axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(B/'figures/training_curves.png',dpi=160);plt.close(fig)
# Main compact data tables; complete CSVs preserve all strata.
chainrows=[]
for arm in ['old228','new868']:
 for se in [0,1]:
  r=c(arm,se);chainrows.append([arm,se,f(r['step1_correct'],320),f(r['step12_correct'],r['step1_correct']),f(r['step123_correct'],r['step12_correct']),f(r['step123_correct'],320)])
cirows=[]
for se in [0,1]:
 r=c('new868',se);a=next(x for x in boot if(x['seed'],x['arm'],x['cluster'],x['metric'])==(se,'new868','family_id','D_equal'));m=next(x for x in boot if(x['seed'],x['arm'],x['cluster'],x['metric'])==(se,'new868','family_id','D_marginal'));cirows.append([se,f'{100*r["p1_cubed"]:.2f}%',f'{100*r["conditional_product"]:.2f}%',f'{100*a["estimate"]:+.2f} [{100*a["CI95_low"]:+.2f}, {100*a["CI95_high"]:+.2f}]',f'{100*m["estimate"]:+.2f} [{100*m["CI95_low"]:+.2f}, {100*m["CI95_high"]:+.2f}]'])
comp=[]
for se in [0,1]:
 for met in ['step1','endpoint','whole','observed_whole']:
  r=next(r for r in paired if(r['source'],r['seed'],r['metric'],r['factor'])==('development',se,met,'all'));comp.append([se,met,f(r['old_correct'],320),f(r['new_correct'],320),r['gains'],r['losses']])
state=[]
for se in [0,1]:
 for level in ['In2_Held0_drawer0','In2_Held0_drawer1','In2_Held0_drawer2','In1_Held1_drawer2']:
  r=next(r for r in paired if(r['source'],r['seed'],r['metric'],r['factor'],r['level'])==('development',se,'whole','initial_relation',level));state.append([se,level,f(r['old_correct'],r['windows']),f(r['new_correct'],r['windows']),r['gains'],r['losses']])
mechanism=[]
for se in [0,1]:
 for cond in ['non_drawer_true_stable','close_drawer_to_middle','close_drawer_to_closed','open_drawer_to_open']:
  a=next(r for r in drawer if(r['source'],r['arm'],r['seed'],r['condition'])==('development','old228',se,cond));b=next(r for r in drawer if(r['source'],r['arm'],r['seed'],r['condition'])==('development','new868',se,cond));mechanism.append([se,cond,f(a['free_drawer_correct'],a['window_transition_forecasts']),f(b['free_drawer_correct'],b['window_transition_forecasts']),f(a['observed_drawer_correct'],a['window_transition_forecasts']),f(b['observed_drawer_correct'],b['window_transition_forecasts'])])
rootrows=[]
for se in [0,1]:
 for arm in ['old228','new868']:
  r=sorted([r for r in horiz if(r['source'],r['seed'],r['arm'])==('development',se,arm)],key=lambda r:r['step']);rootrows.append([se,arm,'末端']+[f"{x['endpoint_correct']}/80"for x in r[1:]]);rootrows.append([se,arm,'0至该步全对']+[f"{x['all0_to_step_correct']}/80"for x in r[1:]])
now=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
htmltext=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>扩采究竟改善了什么：机制与统计溯源</title><style>body{{font:15px/1.75 system-ui,sans-serif;max-width:1180px;margin:36px auto;padding:0 24px;color:#172b3d}}h1{{font-size:28px}}h2{{margin-top:36px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd8e2;padding:7px;text-align:left}}th{{background:#eef4f8}}img{{width:100%}}.scroll{{overflow:auto}}.lead{{padding:16px;background:#edf5fb}}code{{overflow-wrap:anywhere}}</style>
<h1>扩采究竟改善了什么：从统计关系到机制溯源</h1><p>北京时间{now}。本轮只读已有预测、训练源码、采样表和控制日志；无训练、模型前向推理、扩采、调参或原文件改写。开发集是诊断主线，留出只描述。所有新增分组是事后分析，不冒充原验收指标。</p>
<p class="lead"><strong>当前结论：</strong>扩采最明确修复的是<strong>状态成为后续动作起点时的覆盖缺口</strong>，尤其两物体已In、抽屉关闭/半开的继续执行。它显著提高了同一开发窗口的整段预测，并非只提高一个末态分数。剩余错误则混合了未输入的外加干预、真实关闭受阻的多结果、以及自由隐状态没有稳定复用已学条件关系。现有证据不足以把它们统一归成“网络太小”“学习率不对”或“缺根起点六步监督”；这些必须分别核对。</p>
<h2>1．89.1%的三次方与66%：原来比较的不是同一批起点</h2>
<p>89.0625%来自每种子全部480个一步起点（80条×6个非终端起点）；三步全程66.25%来自320个窗口（80条×4个起点）。前者包含只剩1–2步的起点，且两个种子不能合成双倍独立数据。额外160个末尾一步起点分别命中146/160=91.25%、144/160=90.00%，比保留的三步窗口更容易，直接把原一步参照抬高了。将每个种子限制在<strong>完全相同的320个三步窗口</strong>，一步正确率变成286/320=89.375%与279/320=87.1875%。定义C1/C2/C3为对应未来末端所有Held3、In3和抽屉分类联合正确；暂不要求起点正确。</p>
{tab(['模型','种子','P(C1)','P(C2|C1)','P(C3|C1,C2)','三项乘积＝整段正确'],chainrows)}
<p>这三个条件概率的乘积是概率链式法则，不要求独立。新种子0是286/320×240/286×213/240=213/320；种子1是279/320×238/279×211/238=211/320。第二步条件成功率下降，第三步又回升，<strong>不是随预测步数单调恶化</strong>。旧模型对应链条更弱；扩采提高了多个环节，并非仅第一步。</p>
<img src="../figures/conditional_chain.png"><p>横轴为上述条件成功概率，纵轴百分比。灰/蓝是旧228/新868训练模型；实线圆点为自由展开，虚线方块为逐步看到实际公共观测后再预测下一步；每个面板固定一个训练种子。后续条件概率在各自此前成功的窗口内计算，风险集合会改变，不能把曲线变化当成同一批样本难度恒定。</p>
<h3>差4.4个百分点究竟能否接受</h3>
<p>主参照定义为D＝三步全程率−同窗口一步率³。独立且每步同精度只是参照假设，不是已知真模型。每次重采样都同时重算一步率、其三次方和整段率。主分析按20个共同布局族整组配对Bootstrap 10,000次，保留每族两个根、配对分支和重叠窗口；另按40个共同物理根重采样作敏感性分析。两个种子分别算，区间只表示这些数据布局的采样波动，不包含重新训练的方差。</p>
{tab(['种子','匹配一步率³','观察整段率','D及95%族区间，百分点','整段−三步各自边际率乘积及95%区间'],cirows)}
<img src="../figures/independence_difference_ci.png"><p>点是新模型每个种子的差值，横线为95%布局族配对Bootstrap区间，黑虚线为零差。蓝/橙分别种子0/1。p1³假定每步精度等于同窗口第1步；p1×p2×p3只假定三个实际边际事件独立。后者差值点估计为+2.28/+2.03个百分点，说明当前点估计不是额外负相关损失，而更接近少量难例聚集；区间也跨零，不能据此证明独立。新模型三个步对(1,2)/(1,3)/(2,3)的二元正确事件相关系数φ，种子0约0.108/0.050/0.168，种子1约0.067/0.073/0.164，属于弱正相关的点估计；不是负相关造成的额外整段损失，也不代表已经确认了相关性的具体因果来源。</p>
<p>原先约4.4个百分点只是0.890625³−0.6625的粗比较。匹配后两种子的差是−4.83、−0.34个百分点，主区间分别[−13.14,+3.22]、[−6.30,+5.53]；按40根的区间约[−13.04,+2.71]、[−6.27,+4.97]，判断一致。<strong>现有数据既不能确认存在额外系统性误差放大，也不能证明差异已经小到可忽略。</strong>区间跨零不等于等效；如果将来容差是5个百分点，这两个区间均不足以证明差距落在±5内。当前尚未指定实际容差。</p>
<p>三步率开立方只是量级换算：两个种子的每步等效率约87.31%、87.04%，比各自匹配的一步率低约2.06、0.15个百分点。它不说明真实错误均匀分布。规划可接受性还取决于要求成功率、禁止关系的违规代价、是否逐步重观测；本联合指标既不是LTL满足率，也不是闭环任务成功率，不能用“统计不显著”替代规划验收。</p>
<h2>2．第二步为什么更难，末态为什么又能恢复</h2>
<p>新模型自由展开第二步，在第一步正确后分别新增46/286、41/279个失败。种子0其中21个、种子1其中22个是关闭动作后的抽屉分类（包括真实停在半开、真实关闭却预测半开、实际仍打开）；各有15个是抓取后Held关系错误，另有少量无关动作下的抽屉漂移及混合错误。第2步并不是第1步难度与动作组成的简单复制。实际边际正确率为种子0的89.38%→82.50%→87.19%，种子1的87.19%→84.38%→86.88%；三者边际乘积为64.29%/63.91%，并不比观察整段率高。</p>
<p>逐步真实观测可以在保存预测中做<strong>严格匹配</strong>：取同episode、同origin的one_step相对第1–3步，起点与请求相同，第1步概率逐值等于自由展开。第2/3步只在前一步后收到实际公共观测，不使用GT隐状态。新模型三步全程由213/320、211/320变成236/320、222/320；旧模型为140/320、143/320→164/320、162/320。可见反馈有帮助，但提升并不代表所有无反馈错误都是数值误差传播；反馈也揭示了之前的真实执行结果和干预后果。</p>
<p>新模型第3步末端279/320、278/320高于整段213/320、211/320，差额66、67个窗口恰好由三个模式构成：种子0的011=24、101=38、001=4；种子1的011=30、101=34、001=3。这里1/0分别表示该步联合对/错，忽略起点。包含起点的旧统计中种子1差额68还包含1个起点错但未来三步全对的窗口。</p>
<p>这些恢复有具体动作原因：新种子0/1的66/67个未来错后恢复窗口中，最后一次错误为Held的有38/38个，随后重复抓取使真实/预测重新一致；最后一次错为抽屉的有28/29个，其中24/25个经open_drawer恢复。打开动作把两条此前不同的状态轨迹重新推向同一个“打开”状态；重试抓取也可能抵消未知张爪干预。它不是模型发现自己错后主动纠正，更不意味着中间关系未错。另有少数恢复来自放置、关闭或其他动作。</p>
<h2>3．扩采的主要收益：终点状态变成可继续使用的条件</h2>
{tab(['种子','相同320窗口的口径','旧228','新868','新修正旧错','新增错误'],comp)}
<p>新开发三步全程净增73/68个窗口，不只是末端净增55/47；逐步输入实际观测时全程净增72/60，说明收益并非单纯减少自由递归漂移。最有解释力的落点在初始关系，而不是一个总平均数。</p>
{tab(['种子','窗口起点真实条件','旧全程','新全程','修正','新增错误'],state)}
<p>两物体In且无Held的三个抽屉条件合计102个窗口：种子0全程13/102→68/102，种子1为18/102→70/102，分别贡献55/73和52/68的全程净增，约76%；这组没有新增损失。这是净收益落点，不能把它当作“这批特定训练样本造成了76%提升”的独立因果份额。</p>
<img src="../figures/continuation_coverage.png"><p>横轴固定两物体In、无Held，区分抽屉关闭/半开；纵轴是原始经历中非终端状态出现次数，灰为旧228，蓝为新增640。旧训练都为0，新增加275/341。它衡量可作为后续动作前态的覆盖，不是像素是否曾出现过。</p>
<p><strong>重要区别：</strong>旧训练并非完全没见过这两类状态——它们分别有31/57次出现，但全部在经历终点；它们不会成为动作前态，也不会被非终端中途裁剪选成训练初始状态。新数据新增275/341次非终端出现，两个种子相应得到436/452次关闭起点、542/554次半开起点的冷启动训练抽样。旧数据相应抽样均为0。这样既补了冷初始化，也补了“识别后继续执行”的转移支持。</p>
<p>这一解释与预测行为一致：旧模型在544个本应保持抽屉类别不变的非抽屉动作预测中，各错60次，而且60次全预测成打开；即便喂入真实观测，仍各错60次。其中56次落在真实关闭或半开且两物体In、无Held条件，另4次为一物体In且另一物体Held的半开条件。新模型该项提高484/544→522/544、530/544；真实观测前缀下进一步达到543/544、544/544。原来的“打开”偏置并非只是累积误差，而是对缺少继续执行支持的条件外推。网络结构没变而此项显著改善，反对“该结构根本表示不了保持关系”的强结论。</p>
<p>这证明新布局上的已覆盖状态条件泛化有所提高，也改善了跨动作复用：真实稳定的非目标In关系在close_drawer后前后均正确从385/405、388/405升到两个种子均405/405；在place后182/192、177/192→192/192；三种抓取后也普遍改善。样本是对象—窗口间隔，会重叠，不是独立405个经历。不能把这种已覆盖条件泛化直接等同于任意新组合的系统性组合能力。并且短窗口收益不能外推到根起点六步：同一新开发原报告的根六步非目标In前后均正确反而从564/575、572/575降到552/575、556/575。这里575是根链物理间隔，前述405等是重叠三步窗口中的预测间隔，分母不同不可相减；它们共同说明新模型更会处理被观测或短程可达的条件，尚未保证持续自由状态也能复用同样的保持关系。</p>
<h2>4．剩余抽屉错误：实际关闭困难与无关动作漂移不是一回事</h2>
{tab(['种子','真实动作—结果条件','旧自由预测抽屉正确','新自由预测抽屉正确','旧真实前缀','新真实前缀'],mechanism)}
<p>对于真实关闭请求后停在半开的63次窗口转移预测（32个实际执行转移，可能共享根而非32个独立布局），扩采只把抽屉识别37→41、24→25；远小于非抽屉保持的改进。新模型错误的22/38次中，20/36次把半开预测为关闭；平均P(关闭)约0.658/0.685，P(半开)约0.260/0.279，绝非都差一点阈值。</p>
<p>追到原控制结果：32个物理转移中30个在horizontal_push_drawer阶段以unreachable_or_obstructed失败，末段TCP误差约0.061–0.150米；2个因requires_empty_gripper拒绝执行。真实末端开度0.0615–0.1153米，距最近分类阈值至少0.0365米；不是在0.025/0.16米边界附近抖动，也没有声明张爪干预。这说明主要错例是<strong>把实际推动受阻后停在半开的结果预测成名义关闭成功</strong>。日志只能确认“无法到达或受阻”，无法凭这个统一原因码再判定每次究竟是碰撞、可达性还是其他具体物理因素。两条半开拒绝例并未造成新模型这里的错误；新模型全部22/38个半开分类错都来自实际推动失败。作为对照，新开发10次持物时关闭、因requires_empty_gripper而拒绝的条件，旧抽屉预测8/10、7/10，新模型10/10、9/10（真实前缀也相同）。所以它并非一概忽略动作执行前条件，主要难点更集中于仍要实际执行、但无法完成推动的细分结果。</p>
<p>数据总量增加没有保证这个困难结果获得更多优化：按不同原始执行记录计，旧228中真实关闭停半开的转移57条，新增640又有293条；但固定抽样预算下，实际有效曝光从6316/6246降至4431/4589。新的输入/布局更广，单例重复更少；同时沿用旧类别权重，关闭/半开/打开分别1.974/1/0.670。半开分支在先验一步目标中只有约7%，关闭类得到约1.97倍的交叉熵代价权重。这与“难结果曝光不足、代价偏向关闭”解释相容，<strong>尚未经过权重或训练预算单变量对照，不能认定它已被因果证实</strong>。</p>
<p>另一类确有共享状态更新的迹象：非抽屉动作中，新模型分别10/3次从前态抽屉预测正确变为错误，多发生于真实关闭不变而预测漂成半开。旧模型对应45/37次，说明扩采减少了这种问题。相同物理转移从相距3步预测与在动作前冷初始化预测配对，160个条件中新模型分别15/10次出现“3步前预测抽屉错、临近一步对”，反向为0/4次；但两个起点之间真实观测信息不同，不能把全部差异归咎于GRU数值不稳。</p>
<h2>5．网络有没有学会条件依赖，还是只记住动作模板</h2>
<p>不能说模型完全学成了“抓取必定Held”。相同新开发预测中，对目标真实Table→Held的103次条件，新模型目标Held/In正确102/103、99/103；对目标已经In而再次桌面抓取、真实In→In的104次条件，正确91/104、94/104。使用实际前缀则后者98/104、104/104。它确实能利用状态，但某些自由状态没有携带足以复用这一条件的信息；旧模型对应In→In自由正确90/104、82/104，改善并不在每个种子都明显。</p>
<p>更接近控制对照的现成案例来自旧角色开发：同一布局族、同一个请求颜色、同样其余对象符号状态及抽屉状态，比较目标在桌面且实际抓起，与目标已In而实际保持In。24个配对中，旧模型冷起点同时预测对两种结果22/24、23/24，新模型23/24、23/24。模型具有表达这种条件依赖的能力。可是在同一批无干预角色根链的新种子0，自由展开到重复抓取时22/24条把真实In→In预测为Held，真实公共前缀的单步能修正这24条根链的首次错误。这不是一个抽屉专属问题，而是<strong>当前符号读出正确，不保证自由隐状态对后续动作与观测隐状态等价</strong>。这些配对只控制粗粒度符号条件，不是完全相同像素或连续物理状态的因果实验。</p>
<p>新开发和组合留出里没有找到同时满足“同族同动作、其余对象符号完全相同、抽屉相同”的Table/In严格配对；因此不硬造配对，只报告较宽状态分层。现有数据也不足以把模板位置相关性与隐状态分布差异完全拆开。它们需要在控制状态、动作、序列位置后比较，不能仅看某个绝对步骤错误高就宣布记住了模板。</p>
<p>网络源码提供了可检验的解释通路：三个128维对象节点与一个128维共享上下文；每个动作都会更新全部对象GRU与context_prior，共享上下文同时进入对象读出与抽屉/滑门/灯读出，未设“无关动作自动保持”硬门。place目标还从隐状态软绑定。这样的耦合允许非目标或抽屉漂移；但扩采后跨动作保持明显改善，不能把“允许干扰”直接变成“已证明架构有缺陷”。还有一个确定的结构约束：action_features只取pointer.sum；lift目标身份经alpha进入对象更新，而context_prior直接只接对象状态均值和动作特征，不直接接目标身份。相同隐状态与预算下，不同颜色lift可有相同下一步场景上下文。它可能限制某些目标相关场景效应，但当前主要关闭受阻动作本就没有目标指针，不能把它直接列为现有主因。没有保存隐状态、绑定权重、逐模块梯度归因或容量/冻结编码器消融，无法判定是共享更新、表示容量、输入丢失细节还是优化未充分约束造成具体错例。</p>
<h2>6．训练实现审计：哪些是事实，哪些只是候选原因</h2>
<p>视觉DINO与旧数据训练得到的学习编码器均冻结；新增数据只训练下游2,490,068参数的对象循环模型及几何适配。输入包含因果四帧双相机特征、可见几何、公共机器人观测与请求动作，不含未来真实执行结果、私有接触状态或实际张爪覆盖标记。因此宏动作背后的执行多结果可能需要输入中已有但尚未提取的细节，也可能部分不可预测；不能仅从当前错误判断缺的是数据还是网络。</p>
<p>训练没有一个可概括为“50%真实观测喂入”的概率。50%指完整/中途裁剪起点。主history路径在<strong>每个有效边界都输入实际公共观测</strong>；另有独立不接收中间观测的展开损失：</p>
{tab(['损失分支','总目标中的系数','实际观测与递归条件'],[['current',1,'每个边界后验，已看到该边界公共观测'],['prior',0.5,'从每步实际观测后验预测下一步，递归长度1'],['multi3',0.5,'从每个实际后验起点，自由1/2/3步；三种长度在同一均值内'],['terminal_from_h2',0.5,'从裁剪后posts[2]出发，只监督剩余经历最终边界；有效最长4步'],['outcome',0.25+'×(po+mo)' if False else '0.25×po + 0.25×mo','实际前缀单步及自由1–3步的成功/事件头；并非状态预测一定成功']])}
<p>每种子同样24,000次片段抽样，当前观测有效边界113,834/113,839；有效先验一步89,834/89,839；multi1/2/3目标机会分别约89.8k/65.8k/45.0k。terminal有效目标17,535/17,539，其中只有6,000次长度4，且并非根起点自由6步。没有用未来标签初始化循环状态，也没有scheduled sampling或模型滚动时把GT符号喂回。这里计数是有效目标机会，不是最终梯度贡献；各分支分别归一化，类别权重与有效通道数还会改变贡献。</p>
<p>由此可定位一个<strong>长程训练—评估条件差异</strong>：根起点4–6步中间状态并未被同根自由路径直接逐步监督，terminal只要求从另一观测起点展开后的终点对，允许途中错后恢复。这与角色根链重复抓取错误相容。但三步已经在multi3覆盖内；前述统计也没有证明三步存在额外系统性放大，所以不能把“没有根六步项”预先选为所有错误的主因。</p>
<p>符号损失是三个完整8分类对象、抽屉、滑门和两个灯共7个有效通道的加权交叉熵/BCE平均，并非直接优化Held/In/抽屉七项联合全对；没有显式状态等价或非目标保持正则。抽屉仍只有共享损失中的一个通道。新增先验一步半开目标次数6532/6634左右，远低于打开、关闭；类别权重没有随扩采更新。修改权重会改变另一个因素，当前对照固定它是设计事实，不是代码错误。</p>
<img src="../figures/training_curves.png"><p>横轴真实优化更新，纵轴为对数刻度的各损失最近100次均值；灰/蓝旧/新，实线/虚线种子0/1。三个面板是总损失、多步符号、从h2终端损失。新旧抽样内容不同，绝对损失不能直接当成同分布拟合优劣，更不能当验证集曲线。</p>
{tab(['模型','种子','末100总损失','末100multi3','末100终端','末500总损失每100步线性斜率'],[[r['arm'],r['seed'],f"{r['last100_total']:.4f}",f"{r['last100_mult3']:.4f}",f"{r['last100_terminal']:.4f}",f"{r['last500_total_linear_slope_per100updates']:+.4f}"]for r in curve])}
<p>AdamW学习率固定3e-4、weight_decay=1e-4、clip=1、BF16，均未改。新训练末500步的69/75次梯度被裁剪，旧训练各16次；损失仍下降，不能宣称868数据已充分收敛。旧role实际有效曝光约53.8k→7.9k，旧core仍约36k不变；同更新次数公平控制了这次数据对照的优化预算，却没有控制每个旧条件的重复量。没有共同训练探针逐检查点的损失、开发学习曲线、更多更新、学习率/裁剪/权重/容量消融，所以“预算不足”“损失权重欠平衡”和“表征不足”的贡献还无法分别定量。有限预算是有证据的候选解释；特定学习率错误或模型太小则证据更弱。</p>
<h2>7．四、五步有没有回升：必须固定同一批根</h2>
<img src="../figures/fixed_root_horizons.png"><p>本图始终用新开发相同80条根经历，横轴从根请求的技能数0–6，纵轴联合正确比例。灰/蓝为旧/新，实线圆点只看该步末端，虚线方块要求起点至该步始终正确，每个面板一个种子。全程曲线不能回升；末端可以恢复，不能解释为长程可靠性提高。</p>
{tab(['种子','模型','口径','1','2','3','4','5','6'],rootrows)}
<p>新模型根起点第3→4→5步末端为69→67→52、70→67→48，没有四、五步高于三步的现象；第6步又回到61/62。全程则分别42→37→25→23、43→39→26→26，未恢复。之前混合起点的某些5/6步曲线较高，既受剩余长度筛选影响，也受动作后状态重新汇合影响，不能拿不同分母判断误差随长度自然减小。</p>
<h2>8．大局判断：不能把收益和残余错误归给同一个模糊原因</h2>
<p><strong>最强证据指向的数据机制：</strong>旧数据把多物体状态放在终点，而新数据使其成为可裁剪、可继续执行的前态；扩采由此提高了状态条件依赖及跨动作关系保持。它不是单一抽屉补丁，反映了轨迹数据“有终态标签”与“有该状态的后继支持”之间的差别。但训练预算分配使某些困难实际结果重复量下降，数据完整度必须按状态—动作—结果及其继续执行机会核算，不能只数868条或6000次总转移。</p>
<p><strong>对剩余错误最有支持的解释：</strong>关闭宏动作具有受阻、拒绝、成功等不同实际结果，模型仍偏向训练代价和常见结果支持的名义端点；一部分执行信息根本未输入（配对张爪覆盖已证明不可识别），另一部分可能在冻结特征或循环表示中未被充分利用。对角色长链，冷观测状态能区分In/Table而自由状态复用失败，提示需要检查表征是否对后续转移充分，以及训练是否真正约束了这些自由状态，而非只让当下符号读出正确。</p>
<p><strong>证据强度排序：</strong>（1）继续执行支持缺口被补齐，是扩采收益最直接的解释；（2）困难结果的有效曝光、固定类别权重与固定优化预算，以及输入中的执行结果不确定性，是残余近程错例的优先解释；（3）自由/观测隐状态条件不等价与训练展开分布差异，是长链候选机制；（4）共享更新耦合、冻结表示/容量和具体优化超参数仍不能区分，尚无证据宣称结构整体坏了或某个参数就是主因。第（2）的多个因素目前不能再强行排唯一名次。</p>
<p>相反，分类边界噪声与0.5解码阈值不是当前主要解释：上述关闭困难离阈值至少36.5毫米；新开发各960次匹配未来预测的131/133次联合错中，仅1/1次属于“对象argmax关系对但0.5阈值不对”。这些只是诊断，未改变正式解码。没有重新校准或在留出上挑阈值。</p>
<p>因此，下一轮若继续，应围绕<strong>条件支持是否充分、实际曝光是否足够，以及自由状态能否支持后继转移</strong>做预先限定的整体对照；当前材料不能替用户选择只改drawer头、盲目加参数、只延长根六步损失或继续堆数据中的任何一个。首先需要一个控制抽样分配与计算预算的可比较诊断，才能把数据收益、优化不足和自由状态训练目标的作用拆开。本轮没有启动这些实验。</p>
<h2>留出描述与审计入口</h2><p>组合留出未来三步全程从46/320、57/320变成75/320、85/320；严格末端组合子集从42/183、53/183变成67/183、77/183。它说明点估计改善，但依旧不足以证明任意组合复用；根六步全程仅2/80、4/80。留出错误的后验读出也不都正确，不能照搬开发集解释。没有使用这些结果改模型、选择方案或调参。</p>
<ul><li><a href="../configs/protocol.json">分析定义</a>；<a href="../configs/uncertainty_protocol.json">配对分组Bootstrap定义</a>；<a href="../results/conditional_chain.csv">完整条件概率</a>；<a href="../results/grouped_bootstrap.csv">差值与置信区间</a>；<a href="../results/paired_model_bootstrap.csv">扩采配对提升区间</a></li><li><a href="../results/matched_windows.csv">同窗口逐条预测</a>；<a href="../results/paired_strata.csv">扩采按条件/动作/关系得失</a>；<a href="../results/recovery_windows.csv">错后恢复动作与真实/预测状态</a>；<a href="../results/physical_outcomes.csv">原控制状态与连续开度</a></li><li><a href="../results/drawer_forecasts.csv">抽屉逐次预测/输入反馈对照</a>；<a href="../results/non_target_action_summary.csv">跨动作非目标保持</a>；<a href="../results/within_family_state_contrast_pairs.csv">同族同动作状态对照</a>；<a href="../results/same_transition_horizon_pairs.csv">同物理转移不同预测起点</a></li><li><a href="../results/terminal_vs_continuation_support.csv">终点与继续执行覆盖</a>；<a href="../results/training_exposure_contexts.csv">状态×动作×结果实际曝光</a>；<a href="../results/drawer_training_class_balance.csv">类别与固定权重</a>；<a href="../results/supervision_path_counts.csv">真实观测/展开机会计数</a>；<a href="../results/training_convergence_summary.csv">收敛日志统计</a></li><li><a href="../../data_expansion_training_20261021/source/models.py">冻结网络源码</a>；<a href="../../data_expansion_training_20261021/source/losses.py">冻结损失源码</a>；<a href="../results/final_verification.json">最终对账与原文件保护核验</a></li></ul></html>'''
# Avoid unsupported exact seed1 middle count in prose; use audited values.
mid=next(r['valid_target_occurrences']for r in classbal if(r['seed'],r['arm'],r['branch'],r['drawer_class'])==(1,'new868','prior1',1));htmltext=htmltext.replace('6532/6634左右',f'6532/{mid}')
(B/'reports/REPORT.html').write_text(htmltext)
summary={'generated_Asia_Shanghai':now,'status':'completed_existing_data_mechanism_analysis','report':str(B/'reports/REPORT.html'),'main_update':'Cannot infer systematic error amplification from66%vs89%; matching and groupCIs change conclusion. Main expansion gains are continuation-state support, not an assumed long-horizon training failure.','no_training':True,'no_model_inference':True,'no_collection':True,'newdev_matched_step1_old':[[262,320],[249,320]],'newdev_matched_step1_new':[[286,320],[279,320]],'newdev_whole3_old':[[140,320],[143,320]],'newdev_whole3_new':[[213,320],[211,320]],'D_equal_pp_seed0_seed1':[-4.829272460937517,-.33947448730466956],'D_equal_family_CI_pp_seed0_seed1':[[-13.138729858398424,3.22028320312499],[-6.304321289062498,5.5255859375]],'no_practical_acceptance_tolerance_specified':True,'main_gain_condition':'In2Held0:terminal-only closed/middle inolddata ->275/341new nonterminal boundaries;55/73 and52/68netwhole3gainlocatedinallIn2Held0conditions','residual_close_middle':'32physical transitions:30controllerpushfailed+2emptygripperrefused;min36.5mmfromlabelthreshold. Effectiveclose-to-middle draws decline6316/6246 ->4431/4589despite moreuniqueexamples.','causal_limits':['No isolated data-condition intervention','No optimizer/weights/capacity/frontend/latent-alignment ablation','No repeated-checkpoint validationcurve','Same-root observedfeedbackchangesinformation, notpure numericalreset','Holdout descriptiveonly'],'remaining_blockers':[]};(B/'reports/SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print('REPORT_WRITTEN',mid)
