from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import html,re,base64,ast,collections
LABEL={'visual_rssm':'视觉RSSM','GT_relations':'准确关系→小模型','GT_physics':'准确关系＋物理→小模型','visual_initial_same_relation_model':'视觉关系→同一小模型','GT_persistence':'准确初态保持不变'}
EN={'visual_rssm':'Visual RSSM','GT_relations':'GT relations','GT_physics':'GT + physics','visual_initial_same_relation_model':'Visual state / same GT model','GT_persistence':'GT persistence'}
COL={'visual_rssm':'#3269b0','GT_relations':'#008c78','GT_physics':'#d58b26','visual_initial_same_relation_model':'#b94d76','GT_persistence':'#929292'}
def rows(name):return list(csv.DictReader((S/'results'/name).open()))
def table(head,data):return '\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(str(x)for x in r)+'|'for r in data])
def relation(p):
 x=hard(np.asarray(p));held=[COLORS[i]for i in range(3)if x[i]];inside=[COLORS[i]for i in range(3)if x[3+i]];return 'held='+('/'.join(held)or'empty')+'; in='+('/'.join(inside)or'none')+'; drawer='+['closed','partial','open'][int(x[6:9].argmax())]
def main():
 sys.addaudithook(guard);met=rows('relation_metrics.csv');cur=rows('current_metrics.csv');pref=rows('prefix_metrics.csv');out=rows('outcome_metrics.csv');motion=rows('movement_metrics.csv');pos=rows('positive_relations.csv');cases=read(S/'results/cases.json');provenance=read(S/'data/provenance.json');provl={(r['episode_id'],r['t']):r for r in provenance};figs=[]
 def metric(seed,model,mode,h,field='all_relations',stratum='all',status='all',subset='all'):
  return next((r for r in met if(r['split'],r['seed'],r['model'],r['mode'],r['horizon'],r['field'],r['stratum'],r['actual_status'],r['subset'])==('test',str(seed),model,mode,str(h),field,stratum,status,subset)),None)
 def value(r):return '—'if r is None else f'{int(r["correct"])}/{int(r["n"])}；{100*float(r["root_accuracy"]):.1f}%'
 def finish(fig,name):fig.tight_layout(pad=2);fig.savefig(S/'figures'/name,dpi=160,bbox_inches='tight');plt.close(fig);figs.append(name)
 fig,ax=plt.subplots(1,2,figsize=(11,4))
 for seed in [0,1]:
  labels=[];values=[];notes=[]
  for st in ['initial_all_correct','initial_all_wrong']:
   r=metric(seed,'visual_rssm','one_step',1,stratum=st);labels.append('Initial relations correct'if st.endswith('correct')else'Initial relations wrong');values.append(100*float(r['root_accuracy'])if r else np.nan);notes.append(f'{r["correct"]}/{r["n"]}\n{r["roots"]} roots'if r else'no samples')
  ax[seed].bar([0,1],values,color=['#008c78','#b94d76']);ax[seed].set(xticks=[0,1],xticklabels=labels,ylim=(0,115),ylabel='One-step task-relation exact (%)',title=f'Visual RSSM seed {seed}')
  for i,v,note in zip(range(2),values,notes):ax[seed].text(i,v+2,note,ha='center',fontsize=10)
 finish(fig,'initial_state_strata.png')
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for seed in [0,1]:
  for model in LABEL:
   endpoint=[100*float(metric(seed,model,'free',h)['root_accuracy'])for h in range(1,5)];chain=[100*float(next(r for r in pref if(r['split'],r['seed'],r['model'],r['stratum'],r['horizon'])==('test',str(seed),model,'all',str(h)))['root_accuracy'])for h in range(1,5)]
   axes[0,seed].plot(range(1,5),endpoint,'o-',label=EN[model],color=COL[model]);axes[1,seed].plot(range(1,5),chain,'o-',label=EN[model],color=COL[model])
  for row in range(2):axes[row,seed].set(xticks=range(1,5),ylim=(-3,105),xlabel='Open-loop action horizon',ylabel='Root-weighted exact (%)',title=f'seed {seed}: '+('endpoint only'if row==0 else'all prefix boundaries correct'));axes[row,seed].legend(fontsize=8,loc='lower left')
 finish(fig,'open_loop_and_prefix.png')
 fig,axes=plt.subplots(2,2,figsize=(11,7))
 for seed in [0,1]:
  for row,(mode,h)in enumerate([('one_step',1),('free',4)]):
   models=list(LABEL);a=np.array([[float(metric(seed,m,mode,h,field=f)['root_accuracy'])for f in FIELDS]for m in models]);ax=axes[row,seed];ax.imshow(a,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto');ax.set(xticks=range(4),xticklabels=['Held ID','In drawer','Drawer','All'],yticks=range(5),yticklabels=[EN[m]for m in models],title=f'seed {seed}, {mode} h={h}')
   for y in range(5):
    for x in range(4):ax.text(x,y,f'{a[y,x]*100:.1f}',ha='center',va='center',color='white'if a[y,x]>.7 else'black',fontsize=9)
 finish(fig,'relation_breakdown.png')
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for seed in [0,1]:
  for k,model in enumerate(['visual_rssm','GT_relations']):
   rr=[next(r for r in out if(r['split'],r['seed'],r['model'],r['mode'],r['actual_status'])==('test',str(seed),model,'one_step',st))for st in STATUS];axes[seed].bar(np.arange(3)+(k-.5)*.32,[float(r['success_probability_mean'])for r in rr],width=.32,label=EN[model],color=COL[model])
  axes[seed].set(xticks=range(3),xticklabels=['Success (98)','Refused (20)','Failed (2)'],ylim=(0,1.06),ylabel='Mean predicted P(success)',title=f'seed {seed}, actual execution strata');axes[seed].legend(fontsize=8)
 finish(fig,'actual_execution_outcomes.png')
 for index,c in enumerate(cases):
  d=dataset('test');e=c['episode_index'];fig,axes=plt.subplots(2,5,figsize=(16,7),gridspec_kw={'height_ratios':[1.4,1]})
  for t in range(5):
   f=np.load(str(d['public_frame'][e,t]));axes[0,t].imshow(f['rgb_static']);axes[0,t].axis('off');axes[0,t].set_title('Initial observation'if t==0 else c['skills'][t-1]+'\n'+c['actual_statuses'][t-1],fontsize=9)
   ps=np.array([c['truth'][t],c['current_visual'][t],c['current_visual'][0]if t==0 else c['models']['visual_rssm']['free_prob'][t-1],c['truth'][0]if t==0 else c['models']['GT_relations']['free_prob'][t-1],c['current_visual'][0]if t==0 else c['models']['visual_initial_same_relation_model']['free_prob'][t-1]]);state=hard(ps);a=np.c_[state[:,:6],state[:,6:9].argmax(-1)/2];axes[1,t].imshow(a,cmap='Blues',vmin=0,vmax=1,aspect='auto');axes[1,t].set(xticks=range(7),xticklabels=['H-R','H-B','H-P','I-R','I-B','I-P','Door'],yticks=range(5),yticklabels=['GT','Visual current','RSSM free','GT-state model','Visual-state model']if t==0 else[],xlabel=f'boundary {t}')
   for y in range(5):
    for x in range(7):axes[1,t].text(x,y,['C','P','O'][int(state[y,6:9].argmax())]if x==6 else str(int(a[y,x])),ha='center',va='center',fontsize=8,color='white'if a[y,x]>.7 else'black')
  fig.suptitle(c['episode_id']+f' | seed {c["seed"]} | '+c['case_type'],fontsize=14);name=f'chain_{index:02d}.png';c['figure']=name;finish(fig,name)
 save(S/'results/cases_with_figures.json',cases)
 inv=read(S/'results/baseline_inventory.json');suff=read(S/'results/state_sufficiency.json');coverage=read(S/'results/heldout_sufficiency.json');counts=read(S/'results/data_counts.json');compute=read(S/'results/compute_complete.json');sel=read(S/'configs/selected_models.json')
 md=['# 动作后果诊断：状态识别与变化预测','本轮回到任务主线：持有谁、谁进入抽屉、抽屉是否开合，以及动作实际成功/拒绝和跨对象移动。所有数据和模型留在本机；无新采集、无视觉编码器前向、无RSSM重训、无C/D样本读取、无上传。','## 结论及证据','在当前历史A/B评估范围，准确关系输入的小模型两个种子均达到一步120/120、四步每个边界全对30/30。视觉RSSM为一步110/120、116/120，四步整段全对27/30、28/30。理想初态保持不变仅一步20/120、四步0/30，说明结果确实涉及动作后果，而不是靠状态不变取得。','主要失败集中在持有对象绑定：A_core1200_22的实际蓝块被高置信地读成红块，然后被传递为“红块在抽屉”。但也有初态硬关系正确、非目标红块的错误概率逐步升高并越过阈值的情况。因此当前证据支持优先检查任务对象绑定与转移读出的连续一致性，不支持把全部问题称为感知错误或动力学学不会。','错误初态分层只有1个独立根，样本很少；95%根bootstrap在1根时会退化为点，不能据此宣称总体因果比例或高精度区间。两个种子也不能当作额外独立根。所有历史评估仅为探索性复用。','## 原有基线盘点',table(['候选','为什么复用或不复用'],[[x['candidate'],x['finding']]for x in inv]),'已有SymbolTransition输入视觉符号概率，旧copy-effect诊断只是成功动作的标签归纳，不能替代准确状态输入、含拒绝/失败的对照。本轮复用全部RSSM预测，只新训练4个小模型，每个1500更新，共6000更新。','## 数据及状态是否充分',table(['划分','根','4步分支','转移','成功','拒绝/0控制','执行后失败'],[[r[k]for k in ['split','roots','episodes','transitions','success','refused','executed_failure']]for r in counts]),'相同根的3个分支和所有边界保持同组。训练40根、开发10根、历史评估10根。训练集同“任务关系＋动作”的27组，没有发现不同的关系后果/执行状态；更细的物体类别＋动作115组也没有冲突。这是有限采样下未发现歧义，不是证明粗状态在任意几何和任务中充分。',table(['队列','关系＋动作组','冲突组','转移中训练见过的关系＋动作','训练见过的动作序列'],[[r['split'],r['groups'],r['conflicting_groups'],f'{r["transitions_with_train_seen_relation_action"]}/{r["transitions"]}',f'{r["action_sequences_seen_in_train"]}/{r["episodes"]}']for r in coverage]),'评估120/120转移的关系—动作组合及30/30动作序列都在训练见过。这限制了100%成绩的解释：成功体现当前任务状态接口在该采样分布中的可学习性，不能外推为新技能组合或真实物理规律泛化。','物理增强输入另含96维实际记录：物体位置/四元数/线速度/角速度，机器人TCP/关节/夹爪，门与开关状态、接触力摘要和场景几何。只有动作前值入模，归一化只使用训练集t0–t3。机器人关节速度、控制器和碰撞求解器内部状态并未完整保存，因此称“物理增强”，不冒称充分Markov状态。','## 对照如何防止未来泄漏','普通准确关系输入为持有3位、入抽屉3位、抽屉三分类及有效性标志；物理增强模型使用相同结构和参数量，只把原先为零的物理槽填入动作前记录。动作仅为6技能one-hot及计划预算，不输入实际控制步、成功、失败原因或未来图像。','一步预测从每个真实动作前状态重新初始化；自由多步只给t0状态和已知动作序列，之后不再读取真实中间状态。损失组合为一步与四步各一半，含关系、三类执行结果、三个移动事件。开发集只在750/1500两个预定检查点间选模型；四组均选1500步，随后冻结再评价历史测试。','GT小模型是确定性128维GRU，视觉RSSM是原随机潜变量模型，结构、状态表示、训练规模不同。因此两者分数差不能直接当“感知的纯因果贡献”。另外把同一个GT关系模型的输入替换为视觉硬关系，保持权重和动作不变，用于测量该模型对初态错误的敏感性；这也不是对原RSSM或真实环境的因果干预。','## 初态分层后的动作预测','“初态正确”指当前可评分的持有身份、抽屉归属和开合同时正确，不保证所有潜变量或几何信息正确。一步使用紧邻动作前的真实观察读出，四步使用最初观察读出。下表n/N为原始计数，百分比为先根内、后根间平均，二者可能不同。']
 data=[]
 for seed in [0,1]:
  for mode,h in [('one_step',1),('free',4)]:
   for st in ['all','initial_all_correct','initial_all_wrong']:
    r=metric(seed,'visual_rssm',mode,h,stratum=st);data.append([seed,mode+' '+str(h),st,'—'if not r else r['roots'],value(r)])
 md+=[table(['种子','预测','初态分层','独立根','全部任务关系正确'],data),'![初态分层](figures/initial_state_strata.png)','横轴区分初态关系正确/错误，纵轴为下一步全部任务关系正确率；绿色为正确初态，粉色为错误初态。标注为正确数/转移数及根数，错误组只有1根，图不是因果贡献百分比。','## 持有、归属、开合分别看']
 data=[]
 for seed in [0,1]:
  for model in LABEL:
   for mode,h in [('one_step',1),('free',4)]:data.append([seed,LABEL[model],mode+' '+str(h)]+[value(metric(seed,model,mode,h,field=f))for f in FIELDS])
 md+=[table(['种子','模型','预测','持有身份','抽屉归属集合','开合状态','全部关系'],data),'![关系分解](figures/relation_breakdown.png)','横轴是关系组，纵轴是模型，数字为根平均准确率百分数；蓝色越深越高，各面板色阶相同。精确持有身份和归属集合要求三个对象均正确，避免只判断空手或只平均大量负例。']
 data=[]
 for r in pos:
  if r['split']=='test'and r['model']in ['visual_rssm','GT_relations']and (r['mode']=='one_step'or r['horizon']=='4'):data.append([r['seed'],LABEL[r['model']],r['mode']+' '+r['horizon'],r['held_identity_correct']+'/'+r['held_positive_n'],r['in_drawer_true_positive']+'/'+r['in_drawer_positive_n'],r['in_drawer_false_positive']+'/'+r['in_drawer_negative_n'],r['drawer_open_true_positive']+'/'+r['drawer_open_positive_n']])
 md+=[table(['种子','模型','预测','真持有时身份正确','入屉正例召回','入屉负例误报','真开屉召回'],data),'## 多步累计与变化子集','![多步预测](figures/open_loop_and_prefix.png)','横轴为不读取新观察的动作步数。上排仅看该步终点，下排要求此前每个边界都正确；蓝色原RSSM、绿色准确关系小模型、橙色物理增强、粉色同小模型使用视觉初态、灰色保持初态不变。每种子30分支但只有10个独立根，不能把终点恢复正确掩盖中间错误。']
 data=[]
 for seed in [0,1]:
  for model in ['visual_rssm','GT_relations','GT_physics','GT_persistence']:
   data.append([seed,LABEL[model]]+[value(metric(seed,model,'one_step',1,field=f,subset='changed'))for f in FIELDS])
 md+=[table(['种子','模型','持有发生变化','入屉归属发生变化','开合发生变化','任一关系变化'],data),'## 执行成功与关系正确不是同一件事','实际评估有98成功、20拒绝、2执行后失败。两个RSSM的二分类成功判断在这些记录上均正确，但仍有持有/入屉对象错误，说明高技能成功概率不能代替任务对象正确性。原RSSM没有单独的“拒绝 vs 执行失败”输出，本轮不为它编造三分类分数；原自由多步缓存也未保存成功概率，因此成功预测比较限于一步。']
 data=[]
 for r in out:
  if r['split']=='test'and r['mode']=='one_step'and r['model']in ['visual_rssm','GT_relations','GT_physics']:data.append([r['seed'],LABEL[r['model']],r['actual_status'],r['n'],r['predicted_success_count'],f'{float(r["root_Brier"]):.6f}',r['three_class_correct']or'未定义'])
 md+=[table(['种子','模型','实际结果','n','预测成功数','根平均Brier','三分类正确数'],data),'![执行结果](figures/actual_execution_outcomes.png)','横轴为实际执行结果和样本数，纵轴是预测成功概率均值；蓝色RSSM、绿色准确关系模型。拒绝和执行失败时应低，不能将低成功概率解释成模型明确知道拒绝原因。','## 跨对象影响','移动事件来自已有轨迹中物体最大位移超过2厘米的标签，与“最终是否在抽屉”不同。抓取动作的直接对象由已声明颜色确定，放置动作的直接对象仅在评分时按真实动作前唯一持物确定；开/关抽屉的直接对象是设备，因此所有方块都属于其他对象。这些目标绑定只用于分组，不是模型输入。']
 data=[]
 for r in motion:
  if r['split']=='test'and r['mode']=='one_step'and r['role']=='other_object':data.append([r['seed'],LABEL[r['model']],r['true_positive']+'/'+r['positive'],r['false_positive']+'/'+r['negative'],r['n']])
 md+=[table(['种子','模型','其他对象移动召回','其他对象不移动误报','物体—动作记录'],data),'原RSSM种子0在22个其他对象移动正例中检出18个、276个负例误报6个；种子1与准确关系模型均为22/22、0/276。它提示任务关系与跨对象影响需分别验收；这里仍是固定执行器数据中的条件预测，不是独立操纵某个对象后的因果效应估计。','## 代表性错误链与实际失败']
 explanations={'initial_error_propagation':'第一个有影响的错误在动作之前：实际持有blue，视觉读出却以约0.983认为持有red。放入抽屉以后，原模型持续预测red入屉，而真实是blue。准确关系小模型用真实初态可全程正确；同一小模型只换成视觉初态也产生错误链，支持对象绑定错误向后继传播的机制。该替换只改变预测器输入，没有改真实轨迹，也未直接修改原RSSM潜状态。','aligned_initial_dynamics_error':'初始硬判定为持有pink，正好正确；但red持有概率仍约0.412。到第4步开屉，原RSSM预测red在屉内概率升至约0.788，实际只有pink在内。首错是额外red归属被误报；准确关系模型和同模型视觉硬初态都正确。证据定位到原视觉潜状态/多步读出的残留身份混淆，不能称作已经排除所有感知因素的纯动力学错误。','refusal_or_failure_chain':'先在空手、关屉时请求放置，实际0控制步拒绝，关系保持不变。后续开屉、抓blue、放置blue成功。原模型正确保留了拒绝后的状态；无需人工给搜索器写“拒绝必自环”规则，这里只是检查记录与学习预测是否一致。','successful_canonical_chain':'此固定序列最终满足hand_empty、red在屉内、drawer_closed。原RSSM与准确关系模型都正确预测。第一步开屉因已开而被记为0步拒绝，仍不妨碍后续任务完成；动作是否拒绝和最终任务是否成功必须分开。该例只重放已有动作，不是新运行的规划闭环。','actual_physical_failure':'这是已有轨迹的真实执行失败，而非分类器评分失败：最后一次放置执行了311个控制步，控制器报告failed_goal_and_retry_precondition_absent。此前blue已在屉内，机器人又持有pink；动作后pink已释放；ownership_v2标签按pink→blue→drawer支撑链把pink和blue都记为入屉，而执行器和官方成功检查仍为false。源码也证实：执行器的in_drawer要求物体直接接触抽屉且接触力大于0.01，ownership_v2允许向上的间接支撑链。这是任务关系标签与执行器完成条件的语义差别，不能写成“pink没有入屉”，也不能仅凭这一例断言通用碰撞原因。准确关系对照能预测此记录，不等于已经学会通用碰撞物理。该分支与前面A22案例同根，不能重复计作新的独立证据。'}
 for c in cases:
  md+=[f'案例 {c["episode_id"]}，种子{c["seed"]}：{c["case_type"]}。',explanations[c['case_type']],'!['+c['case_type']+'](figures/'+c['figure']+')','上排按边界0–4展示真实静态RGB；动作名和success/refused/failed说明真实执行记录。下排H-R/H-B/H-P是红/蓝/粉持有位，I-R/I-B/I-P是入屉位，Door的C/P/O表示关/半开/开；深蓝为真或开，浅色为假或关。行分别为GT、真实观察读出、原RSSM自由预测、准确初态小模型、视觉初态同小模型。颜色表示状态值，不表示对象像素颜色。边界0的GT模型行只显示已提供初态，不是预测成绩。']
  data=[]
  for t in range(5):data.append([t,'起点'if t==0 else c['skills'][t-1],'—'if t==0 else c['actual_statuses'][t-1],relation(c['truth'][t]),relation(c['current_visual'][t]),'—'if t==0 else relation(c['models']['visual_rssm']['free_prob'][t-1])])
  md.append(table(['边界','此前动作','实际结果','真实关系','当前视觉关系','原RSSM自由预测'],data))
  if c['case_type']=='actual_physical_failure':
   for t,st in enumerate(c['actual_statuses']):
    if st=='executed_failure':r=provl[c['episode_id'],t];md.append(f'原始证据：{r["private_info_path"]}；{r["labels_path"]}；结果原因 {r["result"].get("reason")}，真实控制步 {r["controls"]}。')
 md+=['## 本轮之后回到用户重点','核心目标仍是从视觉与操作经历学习任务状态、动作后果和跨对象影响，形成可解释规划接口；不继续左中右定位或换色支线。','下一项选择：复用这次已经学到的关系转移对照，在现有更宽几何分布的A/B转移上检查任务关系接口和状态充分性，并将已有物体身份读出与RSSM读出按同根配对。先做缓存/记录重放，不新增采集或训练；重点看持有对象绑定是否修复，以及相同关系/动作在更宽几何中是否出现真实后果冲突。若出现冲突，再据此扩充任务状态或动力学监督；不因当前100%就把GT关系当作线上输入。','选择依据是：当前狭窄数据的准确关系动力学已可学，盲目扩大RSSM训练缺少针对性；主要错误仍是对象绑定传播与少量多步混淆，同时当前关系—动作支持较重复，必须检查更宽已有数据才能判断接口是否够用。新比较也必须单独登记并限时，不能把此次历史探索成绩当作独立泛化。','## 技术交付与限制',f'4次小训练、6000更新、8个预定检查点，实际计算{compute["seconds"]:.2f}秒、PyTorch峰值{compute["peak_cuda_GiB"]:.3f}GiB。时间不含代码、统计与报告制作。两个输入版本结构和初始权重配对，但它们与原RSSM不同；完整配置、源码、目标、模型、预测文件与指标CSV分目录保存。','本轮数据执行器仍是已有PhysicalOracle V1工程控制器，不是重新完成HULC固定执行器验收。没有生成TS搜索成功率或声称运行新LTLf闭环，原指南完整验收仍未完成。','统计文件：relation_metrics.csv按初态、动作结果和变化子集分层；current_metrics.csv当前读出；positive_relations.csv正例召回；outcome_metrics.csv执行结果；movement_metrics.csv跨对象移动；prefix_metrics.csv累计链正确性；initial_state_substitution.csv同小模型初态替换配对区间。原始预测与案例索引均保留。']
 text='\n\n'.join(md)+'\n';(S/'REPORT.md').write_text(text);source=(P/'experiments/object_identity_readout_20261019/source/report.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='render_markdown');ns={'re':re,'html':html};exec(compile(ast.Module(body=[node],type_ignores=[]),'<render>','exec'),ns);body=ns['render_markdown'](text)
 for name in figs:body=body.replace('figures/'+name,'data:image/png;base64,'+base64.b64encode((S/'figures'/name).read_bytes()).decode())
 (S/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>动作后果诊断</title><style>body{max-width:1280px;margin:36px auto;font:16px/1.7 system-ui;color:#203047;padding:0 22px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{border:1px solid #ced8df;padding:6px;overflow-wrap:anywhere}th{background:#e9f3f3}img{max-width:100%}figure{margin:24px 0}p{overflow-wrap:anywhere}h1,h2{color:#08766b}</style>'+body+'</html>');save(S/'results/next_action_decision.json',{'utc':now(),'chosen':'Existing broader A/B geometry replay and cached object-identity vs RSSM task-state comparison through learned relation consequences','why':['GT relation dynamics solved current observed support','Initial object binding errors propagate','Current relation-action/sequence support may be repetitive'],'scope':'Separate bounded follow-up; no automatic new fitting/collection','not_chosen':['table-left-middle-right optimization','color swap tests','unfocused RSSM scaling']});save(S/'results/report_complete.json',{'utc':now(),'figures':len(figs),'cases':len(cases),'independent_case_roots':len(set(c['root_id']for c in cases)),'new_fits':0,'new_model_forwards':0,'C_D_samples':0});print('REPORT_COMPLETE',len(figs),len(cases),flush=True)
if __name__=='__main__':main()
