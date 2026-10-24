from pathlib import Path
import json,html,datetime,os,collections
from zoneinfo import ZoneInfo
A=Path('/home/robot/research/projects/wm_symbolic/experiments/prediction_error_analysis_20261021');S=A.parent/'data_expansion_training_20261021';os.environ['MPLCONFIGDIR']=str(A/'cache/matplotlib')
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
rd=lambda n:json.loads((A/f'results/{n}.json').read_text())
def table(head,rows):return'<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table>'
def frac(n,d):return f'{n}/{d} ({100*n/d:.1f}%)'if d else'—'
counts=rd('three_step_counting');strict=rd('strict_combination_three_step');agg=rd('six_step_first_error_summary');budget=rd('sampling_exposure');pairs=rd('paired_same_sample');chains=rd('six_step_chains');details=rd('first_errors_detailed');diag=rd('observation_prefix_error_partition');mech=rd('mechanism_summary');bounds=rd('external_information_bound');cond=rd('training_conditional_exposure');observed=rd('observed_comparison');gainchains=rd('six_step_paired_chains')
sources=['old_development','development','combination_test'];names={'old_development':'旧开发','development':'新开发','combination_test':'组合留出（仅描述）'};en={'old_development':'Old development (78 episodes)','development':'New development (80 episodes)','combination_test':'Combination holdout (80 episodes)'};colors={'old228':'#7b8794','new868':'#1864ab'}
fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
for ax,so in zip(axes,sources):
 for ai,arm in enumerate(['old228','new868']):
  means=[]
  for j,(num,den)in enumerate([('endpoint_correct','endpoint_eligible'),('future_interval_correct','future_interval_eligible'),('including_initial_interval_correct','including_initial_interval_eligible')]):
   rr=[next(r for r in counts if(r['source'],r['arm'],r['seed'])==(so,arm,se))for se in [0,1]];vals=[100*r[num]/r[den]for r in rr];means.append(np.mean(vals))
   for se,val in enumerate(vals):ax.scatter(j+(ai-.5)*.3,val,c='black',marker=['o','^'][se],s=23,zorder=3)
  ax.bar(np.arange(3)+(ai-.5)*.3,means,.3,color=colors[arm])
 ax.set_xticks(range(3),['Endpoint t3','All t1..t3','All t0..t3'],rotation=15);ax.set_ylim(0,105);ax.set_title(en[so],fontsize=11);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Joint correctness (%)');fig.legend([Patch(color=colors[a])for a in colors]+[Line2D([],[],marker=m,color='black',linestyle='None')for m in ['o','^']],['Old228','Expanded868','Seed0','Seed1'],loc='lower center',ncol=4);fig.tight_layout(rect=[0,.09,1,1]);fig.savefig(A/'figures/endpoint_vs_interval.png',dpi=150);plt.close(fig)
cats=['initial_first','target_only_first','non_target_stable_only_first','drawer_only_first','mixed_first','non_target_GT_change_only_first','no_error'];catlabels=['Initial recognition','Target only','Stable non-target only','Drawer only','Mixed components','Non-target GT changed','No error'];catcolors=['#8c8c8c','#e68632','#3985b8','#8463af','#cfad30','#b36363','#38a08c']
fig,axes=plt.subplots(1,3,figsize=(15,4.7),sharey=True)
for ax,so in zip(axes,sources):
 rr=[next(r for r in agg if(r['source'],r['arm'],r['seed'],r['cohort'])==(so,arm,se,'all'))for arm in ['old228','new868']for se in [0,1]];bottom=np.zeros(4)
 for cat,col in zip(cats,catcolors):v=np.array([r[cat]for r in rr]);ax.bar(range(4),v,bottom=bottom,color=col);bottom+=v
 for i,r in enumerate(rr):ax.text(i,r['episodes']+1,str(r['episodes']),ha='center',fontsize=9)
 ax.set_xticks(range(4),['Old s0','Old s1','New s0','New s1']);ax.set_title('Old development (48 six-skill episodes)'if so=='old_development'else en[so],fontsize=11);ax.set_ylim(0,90);ax.grid(axis='y',alpha=.15)
axes[0].set_ylabel('Six-skill episodes (#)');fig.legend([Patch(color=c)for c in catcolors],catlabels,loc='lower center',ncol=4,fontsize=9);fig.tight_layout(rect=[0,.16,1,1]);fig.savefig(A/'figures/six_step_first_error.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.4));rr=[(arm,se)for se in [0,1]for arm in ['old228','new868']];bottom=np.zeros(4)
for group,col,label in [('old_core','#a2aab3','Old core120'),('old_role','#c57928','Old role108'),('new640','#1864ab','New640')]:
 vv=np.array([next(r['effective_transitions']for r in budget if(r['seed'],r['arm'],r['group'])==(se,arm,group))for arm,se in rr]);ax.bar(range(4),vv,bottom=bottom,color=col,label=label)
 for i,v in enumerate(vv):
  if v:ax.text(i,bottom[i]+v/2,f'{v:,}',ha='center',va='center',fontsize=9,color='white'if group!='old_core'else'black')
 bottom+=vv
ax.set_xticks(range(4),['Seed0 Old228','Seed0 Expanded868','Seed1 Old228','Seed1 Expanded868']);ax.set_ylabel('Effective training transition exposures (#)');ax.set_ylim(0,105000);ax.grid(axis='y',alpha=.15);ax.legend(loc='upper center',ncol=3,fontsize=9);fig.tight_layout();fig.savefig(A/'figures/sampling_exposure.png',dpi=150);plt.close(fig)
bound=rd('six_step_boundaries');fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
for ax,so in zip(axes,sources):
 for arm in ['old228','new868']:
  for se in [0,1]:
   z=[r for r in bound if(r['source'],r['arm'],r['seed'])==(so,arm,se)and r['no_declared_intervention']];eid=sorted({r['episode_id']for r in z});idx={(r['episode_id'],r['step']):r for r in z};yy=[100*sum(all(idx[e,k]['correct']for k in range(t+1))for e in eid)/len(eid)for t in range(7)];ax.plot(range(7),yy,color=colors[arm],marker=['o','^'][se],linestyle=['-','--'][se],label=f'{arm} seed{se}')
 ax.set_title(('Old development (six-skill subset)'if so=='old_development'else en[so])+f'\nNo declared intervention: n={len(eid)}',fontsize=10);ax.set_xticks(range(7));ax.set_xlabel('Prefix end boundary (t0..t6)');ax.set_ylim(0,105);ax.grid(alpha=.2)
axes[0].set_ylabel('All boundaries correct through t (%)');fig.legend([Line2D([],[],color=colors[a],marker=['o','^'][se],linestyle=['-','--'][se])for a in colors for se in [0,1]],['Old seed0','Old seed1','New seed0','New seed1'],loc='lower center',ncol=4);fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(A/'figures/no_intervention_prefix_survival.png',dpi=150);plt.close(fig)
countrows=[]
for so in sources:
 for arm in ['old228','new868']:
  for se in [0,1]:
   r=next(r for r in counts if(r['source'],r['arm'],r['seed'])==(so,arm,se));countrows.append([names[so],arm,se,frac(r['endpoint_correct'],r['endpoint_eligible']),frac(r['future_interval_correct'],r['future_interval_eligible']),frac(r['including_initial_interval_correct'],r['including_initial_interval_eligible']),r['endpoint_correct_after_earlier_error']])
firstrows=[[names[r['source']],r['arm'],r['seed'],r['episodes']]+[r[k]for k in cats]for r in agg if r['cohort']=='all']
nofaultrows=[[names[r['source']],r['arm'],r['seed'],r['episodes'],r['initial_first'],r['transition_first'],r['drawer_component_first'],r['non_target_stable_component_first'],r['target_component_first'],r['whole_correct'],r['final_correct']]for r in agg if r['cohort']=='no_declared_intervention']
gainrows=[[names[r['source']],r['seed'],r['horizon'],r['metric'],r['eligible'],r['both_correct'],r['gain'],r['loss'],r['both_wrong'],r['new_minus_old_correct']]for r in pairs if r['factor']=='all'and r['horizon']in [3,6]and r['metric']in ['endpoint','including_initial_interval']]
# Three representative development cases, chosen by fixed lexicographic rule within specified categories.
cases=[]
for source,condition in [('old_development',lambda r:r['target_was_In_predicted_Held']),('development',lambda r:'drawer'in r['components']and r['first_error_skill'].startswith('lift_')),('development',lambda r:'drawer'in r['components']and r['GT_after'][3]==1 and r['first_error_skill']=='close_drawer')]:
 rr=sorted([r for r in details if r['source']==source and r['arm']=='new868'and r['seed']==0 and r['no_declared_intervention']and condition(r)],key=lambda r:(r['episode_id'],r['first_error_step']))
 if rr:
  r=rr[0];cases.append([names[source],r['episode_id'],r['first_error_step'],r['first_error_skill'],r['GT_previous'],r['GT_after'],r['prediction_after_objects']+[r['prediction_after_drawer']],r['one_step_objects']+[r['one_step_drawer']],r['components']])
casefile=[{'selection':'Lexicographically first new868 seed0 no-declared-intervention case in each specified development category','row':r}for r in cases];(A/'results/representative_development_cases.json').write_text(json.dumps(casefile,ensure_ascii=False,indent=2)+'\n')
external_timing=[]
for so in sources:
 for arm in ['old228','new868']:
  for se in [0,1]:
   rr=[r for r in chains if(r['source'],r['arm'],r['seed'])==(so,arm,se)and not r['no_declared_intervention']]
   external_timing.append([names[so],arm,se,len(rr),sum(r['first_error_step']>=0 and r['first_error_step']<min(r['fault_steps'])for r in rr),sum(r['at_declared_fault']for r in rr),sum(r['after_declared_fault']for r in rr),sum(r['whole_correct']for r in rr)])
main='最有证据的主因是无新观测的循环状态转移不稳：模型有时不能维持或利用执行前关系，因而在无关动作后改变抽屉状态，或把已在抽屉内目标的再次桌面抓取预测为Held。初始识别错误和未输入的执行干预各自存在，但不足以解释开发集剩余长链失败。'
next_direction='下一轮只做一次自由展开监督起点对照：使用现有数据、相同模型与固定采样计划，把现有terminal_from_h2的末态监督与从根连续展开的末态监督比较；其余损失、初始化和阈值固定，事先规定并报告真实展开计算预算。只用开发集的无声明干预样本，主看整段正确率和首错位置，不按留出失败调参。本轮未执行该训练。'
now=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>逐样本预测：指标口径、首错与训练分配</title><style>body{{font:15px/1.7 system-ui,sans-serif;max-width:1260px;margin:34px auto;padding:0 24px;color:#1b2938}}h1{{font-size:28px}}h2{{margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ccd8e2;padding:7px;text-align:left}}th{{background:#eef4f8}}img{{width:100%}}.note{{background:#eff5fa;padding:15px}}code{{overflow-wrap:anywhere}}.scroll{{overflow:auto}}</style>
<h1>逐样本预测：指标口径、首错与训练分配</h1><p>北京时间{now}。本轮只读已有概率、标签、采样表与执行日志；没有模型推理、训练、扩采、阈值调整或重新选择检查点。原封存实验未改写。</p>
<p class="note">{main}</p><p>对旧场景的部分末态退步，训练分配变化是明确混杂，却不是已被单独验证的原因。旧角色转移曝光下降约85%，但同类“抓取已In目标”的总曝光反而增加，且旧角色单步预测未同步下降。当前证据更支持自由展开下的条件状态转移脆弱，而非“完全没学过该转移”或普遍旧知识遗忘。无法据现有两次配对训练确定哪个参数或损失项造成它。</p>
<h2>三步指标到底数了什么</h2><p>此前free_h3是<strong>第3步末端联合关系正确率</strong>。每个起点只检查t3：三个对象的Held与In共6个二元关系，加一个抽屉三分类，全部正确才计1。它不要求t0、t1、t2正确，不评分滑门、灯、技能成功或LTL公式；也不是实体闭环任务成功率。</p><p>每条六步经历可从绝对边界0、1、2、3各预测3步，所以80条×4=320个起点—目标对。旧开发含30条四步、48条六步，因此30×2+48×4=252。它们有共享经历、根与布局族，不能当作320或252个独立物理样本。标签有效掩码按3个对象与抽屉同时有效筛选；本轮所有这些三步窗口和六步根链均无无效标签，整段/末端的分母可直接比较。</p>
<img src="../figures/endpoint_vs_interval.png"><p>灰柱是旧228训练模型，蓝柱是扩充868模型，柱高为两个种子的比例均值，黑圆/三角分别种子0/1。横轴依次只看t3、要求未来t1至t3全对、要求起点t0至t3全对；纵轴为联合正确比例。每个面板是独立数据范围。末端可以在前面出错后恢复，因此末端柱高不能解读成整段可靠性。</p>
{table(['范围','训练模型','种子','只看第3步末端','未来1至3步全对','起点0至3步全对','末端对但此前错的起点数'],countrows)}
<p>新开发“末端”两种子均值87.0%，未来三步全对为66.25%，包含起点为66.09%；旧模型相应71.09%、44.22%、43.91%。改善仍存在，但“87%整段三步正确”是不成立的说法。组合留出所有起点的新模型未来三步全对仅75/320、85/320，包含起点为75/320、84/320。</p>
<h3>严格留出组合子集只是末端标签筛选</h3>{table(['模型','种子','第3步末端','未来1至3步全对','起点0至3步全对'],[[r['arm'],r['seed'],frac(r['endpoint_correct'],r['endpoint_eligible']),frac(r['future_interval_correct'],r['future_interval_eligible']),frac(r['including_initial_interval_correct'],r['including_initial_interval_eligible'])]for r in strict])}<p>分母183是320个起点—目标对中，第3步真实目标状态满足“至少2个In且三个对象各自In或Held”的子集，不是183条经历，也不要求前面每步都处于该组合。上述子集的全程数字仍按同一183个目标条件重算。本节留出数字仅描述，未用于修改模型或建议中的具体参数。</p>
<h2>六步：末端正确与从未偏离是两件事</h2><p>本节仅看根起点0、长度确为6的经历：旧开发48条角色经历，新开发80条，留出80条。旧开发30条四步core不能补齐为六步。首次偏离按边界0至6扫描；若边界0已错，记初始识别；否则比较首错动作前后真实关系并结合请求对象。抓取目标按请求颜色，放置目标按动作前真实唯一Held定义；无唯一Held时不虚构目标。这个真实目标只用于离线归因，未提供给模型。</p>
<img src="../figures/six_step_first_error.png"><p>纵轴是经历数，横轴旧/新模型各两个种子；旧开发柱高48，其余80。灰=初始识别；橙=仅目标关系错；蓝=仅真实稳定非目标关系错；紫=仅抽屉错；黄=同时涉及多个错误成分；暗红=仅非目标真实关系发生改变但模型没预测对；绿=0至6步全对。每条链只落入一个颜色，所有颜色相加等于分母。暗红不能称作遗忘，黄色也不能任意归给某一成分。</p>
{table(['范围','模型','种子','六步条数','初始','仅目标','仅非目标保持','仅抽屉','混合','仅非目标真实改变','全程正确'],firstrows)}
<p>末态与全程结论可能反向：旧角色种子0末端19/48→24/48，但全程5/48→0/48；种子1末端24/48→15/48，但全程1/48→4/48。新开发末端61/80、62/80，全程仅23/80、26/80；留出末端6/80、12/80，全程2/80、4/80。只看最终边界会掩盖中途错后恢复，也不能据一个末态退步宣称整个旧场景普遍遗忘。</p>
<h2>外加干预与模型连续预测误差</h2><p>逐条检查了1,368个技能转移的执行日志。历史角色经历的24条分支在第2个技能施加张爪覆盖，但原预测文件injected_miss_step对它们始终为False——旧元数据未携带新扩采分支标记。补充分析以控制日志requested_gripper与applied_gripper及declared_execution_disturbance为准，只补归因，不更改任何原记录或评分。新开发有24条声明干预分支，留出20条。</p>
{table(['范围','六步经历','干预配对根数','输入同而干预后联合GT不同的配对','由这些配对得到的全程全对上限'],[[names[so],next(r['six_step_episodes']for r in bounds if r['source']==so),next(r['declared_intervention_pairs']for r in bounds if r['source']==so),next(r['identical_input_opposite_GT_pairs']for r in bounds if r['source']==so),str(next(r['maximum_whole_chain_correct_due_to_these_pairs']for r in bounds if r['source']==so))+'/'+str(next(r['six_step_episodes']for r in bounds if r['source']==so))]for so in sources])}
<p>在旧开发24对、新开发23对、留出20对中，模型可见的因果视觉/几何特征前缀和请求动作完全相同，干预后的真实联合状态不同；四个冻结模型对应预测也逐值完全相同。因此确定性预测器不能在同一对的两条分支上都命中那个边界。新开发另一对虽执行干预，但未改变该边界联合标签，未硬算成不可识别配对。上限只针对所有边界均正确，不能套用到第6步末端，因为后续重试可使真实状态重新一致。</p>
{table(['范围','模型','种子','有声明干预链数','首错在干预前','首错在干预当步','首错在干预后','全程正确'],external_timing)}<p>此表只按时间位置划分；不能把‘当步首错’全部自动归因给外加干预，也不能把后续错误都自动视作传播。前述完全相同输入、不同GT的配对才提供了具体的信息不可识别证据。未来有干预分支在中途起点若已经看到干预后的观测，不能继续用‘当时不知道干预’解释新的初始化错误。</p><p>外部干预不是全部解释。下表直接排除整个经历内任何声明干预，而不是只排除首错恰好发生的干预步。剩余样本仍有普通执行失败、拒绝和部分可观测性，不能自动称为无随机性的理想执行。</p>
{table(['范围','模型','种子','无声明干预条数','初始首错','转移首错','涉及抽屉','涉及非目标保持','涉及目标','全程正确','末端正确'],nofaultrows)}
<p>“涉及”列保留混合成分，会重叠，不能与初始/转移列相加。新开发新模型的56条无声明干预经历中，两个种子初始均0错，首次转移错33、31条，其中抽屉成分30、23条。抽屉首错中，真实抽屉一直不变但预测自行改变的有10、5条；真实执行后停在半开、模型却预测其他状态的有14、15条。它既包含状态保持问题，也包含实际执行结果预测问题，不是所有close_drawer后错误都等同于非目标遗忘。</p>
<img src="../figures/no_intervention_prefix_survival.png"><p>仅无声明干预的根链：旧开发24条、新开发56条、留出60条。横轴是边界0至6，纵轴要求此前所有边界都正确的比例；曲线只能不升。灰/蓝为旧/新，实线圆点/虚线三角为种子0/1。它把首次偏离展示出来，不会因后面偶然恢复而重新计为全程成功；这是离线预测可靠性，不是机器人闭环成功率。</p>
<h2>真实观测前缀能否纠正首次偏离</h2><p>对每个自由展开的首错边界，读取已保存的两种预测：one_step从此前真实公共观测前缀预测下一步；current已看到当前真实图像后的后验读出。没有新推理、没有GT隐状态初始化。它能说明真实观测信息是否有帮助，不能单独证明是某个GRU门、状态编码器或物理可观测性造成错误。</p>
{table(['范围','模型','种子','无干预首错在转移的链','单步对且当前对','单步对但当前错','单步错但当前对','单步与当前都错'],[[names[r['source']],r['arm'],r['seed'],r['free_first_transition_errors'],r['one_correct_current_correct'],r['one_correct_current_wrong'],r['one_wrong_current_correct'],r['one_wrong_current_wrong']]for r in diag if r['cohort']=='no_declared_intervention'])}
<p><strong>开发集诊断：</strong>新开发两个种子分别22/33、18/31个自由首错，在读取实际公共前缀后的单步预测中变正确；其余11/33、13/31个在单步仍错，但看到真实当前帧后都能读对。旧开发无干预角色链的新种子0，24个自由首错在观测前缀单步下全部正确；其中22条首错是“目标真实In→In，但自由预测Held”。这比只看末态更直接地支持连续预测状态的条件利用不稳。</p>
<p><strong>留出描述：</strong>排除干预后，两个新模型首错中的当前后验仍有16/47、21/55条读错；非目标真实关系改变的错误也很多，不能把开发集的“看见后都能读对”推广到留出。留出只描述，不作为选择修正方法或调参依据。</p>
{table(['开发范围','经历ID','首次错边界','请求','真实前[三对象,抽屉]','真实后','自由预测后','真实前缀单步后','首错成分'],cases)}<p>三个案例均按预定错误类别中经历ID字典序取第一个，而非按可视效果选择。对象顺序红、蓝、粉；类别0/1/2为桌面左/中/右，3为In，6为Held；抽屉0关闭、1半开、2打开。表中对象预测为完整8类分布的argmax，用于辅助读例；正式联合指标始终使用Held/In概率各0.5阈值，不能拿argmax例表代替正式计数。</p>
<h2>同样本上究竟得到了什么、丢了什么</h2>
{table(['范围','种子','步数','口径','有效起点','两者都对','新对旧错','旧对新错','两者都错','新减旧'],gainrows)}<p>endpoint只看末端，including_initial_interval要求起点与中间各步全对。所有四格均在完全相同经历、起点、请求与真实标签上配对。新开发3步末端的净增55、47来自修正70、59与新增错误15、12；留出相应净增18、20来自修正49、42与新增错误31、22。旧角色第6步种子1净减9是新增错误11与修正2的差，而种子0净增5是修正18与新增错误13的差。两个种子不是两批独立物理数据。</p>
<h2>训练分配变化能解释旧场景退步吗</h2><img src="../figures/sampling_exposure.png"><p>横轴是两个种子各旧228/新868训练条件；纵轴为实际进入有效序列的训练转移次数，重复曝光会重复计数。灰=旧四步core，橙=旧六步role，蓝=新增640经历。每个种子总高完全相同，旧core也逐样本、裁剪起点保持一致；旧role的大部分曝光被新增数据分摊，不能把“1500更新相同”理解成旧数据训练量相同。</p>
{table(['种子','训练条件','数据组','片段抽样数','片段占比','有效转移曝光','转移占比','经历覆盖'],[[r['seed'],r['arm'],r['group'],r['episode_draws'],f"{100*r['draw_share']:.2f}%",r['effective_transitions'],f"{100*r['transition_share']:.2f}%",r['unique_episodes']]for r in budget])}
<p>旧数据合计片段占比从100%降到57.35%/57.33%，有效转移占比从100%降到48.87%/48.80%；旧role片段50%→7.35%/7.33%，转移59.91%/59.92%→8.78%/8.72%。每条旧role平均抽样111.1次→16.34/16.30次，108条仍全部覆盖，不能把覆盖到等同于训练量相同。旧core三步末端两模型两种子均60/60，部分回退集中在role长程，确与分配变化相容。</p>
{table(['训练情境','种子','旧模型总曝光','新模型总曝光'],[[typ,se,sum(r['transitions']for r in cond if r['context']==typ and r['seed']==se and r['arm']=='old228'),sum(r['transitions']for r in cond if r['context']==typ and r['seed']==se and r['arm']=='new868')]for typ in ['lift_target_In','place_single_Held','place_no_single_Held','close_ends_middle','lift_true_drawer_stable']for se in [0,1]])}
<p>训练情境按真实标签离线分组：lift_target_In为请求桌面抓取时目标已In；place_single_Held为放置前恰好一个Held；place_no_single_Held为空手或无唯一Held；close_ends_middle为关闭请求后真实停在半开；lift_true_drawer_stable为抓取请求前后真实抽屉状态不变。这些组可重叠，不能相加。特别是lift_target_In总曝光4669/4635→6237/6174，即使细分到In2、Held0也4558/4533→5433/5388，不能说这类关系完全未见或总曝光减少。</p>
<p>另一个反证是旧role沿真实前缀单步从247/288、245/288变成250/288、246/288，并未随末态同步下降；排除干预边界后243/264、243/264→249/264、245/264。因而旧末态回退更像训练分配变化下既有自由展开脆弱性的表现，而非当前关系识别或单步机制普遍失效。现有实验没有单独改变采样权重，不能证明“85%稀释造成了9条回退”，也不能把抽象状态相同的样本当作相同隐状态或完整历史。</p>
<h2>一个主因与一个下一步方向</h2><p><strong>主因：</strong>{main}这是对可观察失败机制的定位，不是对具体代码错误或某个模块的因果定罪。对旧场景部分回退，采样稀释是未消除的混杂，不能单独归因。</p>
<p><strong>唯一建议方向：</strong>{next_direction}现有losses.py中multi3最长连续展开3步，terminal项从已看过实际前两步观测的posts[2]出发；六步完整经历对应最多4步连续终端展开，未直接训练根起点自由6步末态。这个训练—评估条件差异是源码事实，是否修正它就能改善仍是待检验假设，不能提前宣称成功。</p>
<h2>范围与证据</h2><p>本分析是冻结结果后的追因，新增整段指标、错误分组和训练情境不是原先预注册主指标。开发集用于诊断，留出结果只描述；原80条留出没有被重新选择、补样或用来拟合。原A/B内部组合划分不能当作C/D泛化或原始指南全项目验收。</p>
<ul><li><a href="../configs/protocol.json">补充分析定义</a>；<a href="../results/three_step_counting.csv">三步口径逐模型计数</a>；<a href="../results/three_step_windows.csv">逐起点三步记录</a>；<a href="../results/strict_combination_three_step.csv">严格组合子集</a></li><li><a href="../results/six_step_chains.csv">逐六步链首错与成分</a>；<a href="../results/first_errors_detailed.csv">首错前后状态及预测</a>；<a href="../results/paired_same_sample.csv">同样本得失</a>；<a href="../results/observation_prefix_error_partition.csv">观测前缀四格诊断</a></li><li><a href="../results/execution_log_audit.csv">1,368次转移的干预日志核对</a>；<a href="../results/external_ambiguity_pairs.csv">输入完全相同而GT不同的配对证据</a>；<a href="../results/sampling_exposure.csv">真实训练分配</a>；<a href="../results/training_joint_context_exposure.csv">状态×动作训练曝光</a></li><li><a href="../results/verification.json">原5,732文件只读且哈希未变</a>；<a href="../results/counting_verification.json">原指标对账与分母复核</a>；<a href="../../data_expansion_training_20261021/source/losses.py">冻结训练损失源码</a>；<a href="../../data_expansion_training_20261021/reports/REPORT.html">原训练报告</a></li></ul></html>'''
(A/'reports/REPORT.html').write_text(content)
summary={'generated_Asia_Shanghai':now,'status':'completed_readonly_analysis','report':str(A/'reports/REPORT.html'),'no_training':True,'no_model_inference':True,'no_collection':True,'main_cause':main,'one_next_direction_not_executed':next_direction,'three_step_newdev_endpoint_seed0_seed1':[[279,320],[278,320]],'three_step_newdev_all_future_seed0_seed1':[[213,320],[211,320]],'three_step_newdev_including_initial_seed0_seed1':[[213,320],[210,320]],'six_step_newdev_no_intervention_first_transition_errors_seed0_seed1':[[33,56],[31,56]],'six_step_newdev_no_intervention_drawer_first_components_seed0_seed1':[[30,33],[23,31]],'six_step_test_whole_seed0_seed1':[[2,80],[4,80]],'remaining_blockers':[]};(A/'reports/SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False))
