"""Report-only comparison of the two preserved, completed training runs."""
from common import *
import collections,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
R=P/'reports/state_semantics_v2_20261015';EC=P/'experiments/visual_state_semantics_v2_hardcontrast_20261015'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(FONT)
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':11,'axes.unicode_minus':False})
def table(h,rows):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |'for r in rows)+'\n'
def figsave(fig,name):fig.savefig(R/'figures'/name,dpi=160,bbox_inches='tight');plt.close(fig)
def main():
 d0=read(E/'results/development_evaluation.json');d1=read(EC/'results/development_evaluation.json');t0=read(E/'results/training.json');t1=read(EC/'results/training.json');m=read(E/'data/metadata.json');w=read(E/'data/windows.json');wi={r['window_id']:r for r in w};p0=np.load(E/'results/probabilities.npy');p1=np.load(EC/'results/probabilities.npy');pairs=read(EC/'configs/AB_pairs.json');audit=read(E/'results/AB_frozen_representation_audit.json')
 assert t0['updates']==t1['updates']==1200 and t1['base_sampler_identical_to_initial'] and not d1['new_independent_acceptance_permitted_by_frozen_gate']
 rows=[];pr=[]
 for tag,d in [('初版九事实',d0),('唯一修正：AB配对损失',d1)]:
  z=d['metrics']['new_query9']['dev:new_semantics'];a=d['metrics']['new_query9']['train:new_semantics'];h=d['metrics']['new_query9']['dev:legacy_holding_only']
  rows.append([tag,f"{a['all9_correct_fully_valid']}/{a['fully_valid9']}",f"{z['all9_correct_fully_valid']}/{z['fully_valid9']}（78.95%）",z['facts']['Closed(drawer)']['FP'],sum(z['facts'][f]['FP']for f in FACTS[3:6]),sum(z['facts'][f]['FN']for f in FACTS[3:6]),f"{z['Holding_all3_correct']}/210",f"{h['Holding_all3_correct']}/84"])
 for r in pairs:
  i,j,k=r['positive_index'],r['negative_index'],r['fact_column'];x={**r,'before_probabilities':[float(p0[i,k]),float(p0[j,k])],'after_probabilities':[float(p1[i,k]),float(p1[j,k])]}
  for name,arr in [('before',p0),('after',p1)]:
   v=np.clip(arr[[i,j],k],1e-7,1-1e-7);x[name+'_logit_margin']=float((np.log(v/(1-v)))[0]-(np.log(v/(1-v)))[1]);x[name+'_both_correct']=bool(v[0]>=.5 and v[1]<.5)
  pr.append(x)
 stats={'pairs':len(pr),'before_both_correct':sum(r['before_both_correct']for r in pr),'after_both_correct':sum(r['after_both_correct']for r in pr),'before_median_margin':float(np.median([r['before_logit_margin']for r in pr])),'after_median_margin':float(np.median([r['after_logit_margin']for r in pr])),'base_sampler_identical':True,'training_updates_each':1200,'total_training_updates':2400,'C_all9_before':165,'C_all9_after':165,'C_denominator':209,'C_counterexample_outside_FP_before':41,'C_counterexample_outside_FP_after':41,'D_runs':0,'promotion':False,'pairs_detail':pr}
 save(R/'tables/single_repair_comparison.json',stats);save(R/'tables/development_gate_after_repair.json',d1);save(R/'tables/AB_frozen_representation_audit.json',audit)
 fig,axs=plt.subplots(1,2,figsize=(13,4.5));xx=np.arange(1,len(pr)+1)
 for key,colour,label in [('before','#8793a2','初版'),('after','#277f87','唯一修正')]:
  axs[0].plot(xx,[r[key+'_logit_margin']for r in pr],'o-',color=colour,label=label,markersize=4)
 axs[0].axhline(1,color='#9b6b3e',ls='--',label='配对目标间隔1');axs[0].set(xlabel='AB同根、相反In标签的配对序号',ylabel='logit(在内正例) − logit(越界负例)',title='训练配对分离改善，不能当作C通过');axs[0].legend(fontsize=9)
 xx=np.arange(2);axs[1].bar(xx-.18,[69,41],.36,color='#8793a2',label='初版');axs[1].bar(xx+.18,[59,41],.36,color='#277f87',label='唯一修正');axs[1].set_xticks(xx,['AB：88个越界承重事实','C：41个越界承重事实']);axs[1].set(ylabel='误报In的事实数（越低越好）',title='同一阈值0.5：开发误报没有下降');axs[1].legend()
 for i,(a,b)in enumerate([(69,59),(41,41)]):axs[1].text(i-.18,a+1,str(a),ha='center');axs[1].text(i+.18,b+1,str(b),ha='center')
 for ax in axs:ax.grid(axis='y',alpha=.15)
 fig.text(.5,-.07,'灰：初版；青：只加AB配对损失。左图横轴为配对编号、连线不是时间；右图两个集合不能按总数当作同规模比赛。所有指标来自已保存模型。',ha='center',fontsize=10);figsave(fig,'single_repair_comparison.png')
 # Current and earlier readouts on identical observations, with semantic mismatch disclosed.
 conditions=[('new_query9',d0,'初版九事实','#277f87'),('new_query9',d1,'唯一修正','#5561a3'),('previous_query7_with_notOpen_closed',d0,'前版七事实\n旧语义迁移','#c28c50'),('legacy7_with_notOpen_closed',d0,'旧冻结读出\n旧语义迁移','#858e99')]
 fig,axs=plt.subplots(1,3,figsize=(15,5))
 for j,(key,d,label,col)in enumerate(conditions):
  z=d['metrics'][key]['dev:new_semantics'];vals=[z['all9_correct_fully_valid']/z['fully_valid9'],z['facts']['Closed(drawer)']['FP'],sum(z['facts'][f]['FP']for f in FACTS[3:6])]
  for k,v in enumerate(vals):axs[k].bar(j,v,color=col);axs[k].text(j,v+(.015 if k==0 else 1),f"{z['all9_correct_fully_valid']}/{z['fully_valid9']}"if k==0 else str(v),ha='center',fontsize=9)
 for ax,title,ylabel in zip(axs,['C完整九事实正确','C未关闭却报Closed','C不在内却报In'],['正确率（仅9项均确定窗口）','错误窗口数','错误对象事实数']):ax.set_xticks(range(4),[r[2]for r in conditions]);ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.15)
 axs[0].set_ylim(0,1.08);fig.text(.5,-.06,'同一公开输入、新版GT。两旧模型没有Intermediate输出，以旧¬Open作Closed仅展示错误迁移的后果；训练数据和表达能力不同，不能当公平架构竞赛。',ha='center',fontsize=10);figsave(fig,'development_readout_comparison.png')
 # Show concrete retained errors; images are not edited and no GT feeds inference.
 chosen=[]
 for condition in ['outside_front_supported','stacked']:
  chosen.append(next((i,r)for i,r in enumerate(m)if r['split']=='dev'and r['pool']=='new_semantics'and r['condition']==condition and r['anchor']==127 and any(wi[r['window_id']]['targets_v2_1']['v'][j]and not wi[r['window_id']]['targets_v2_1']['y'][j]and p1[i,j]>=.5 for j in range(3,6))))
 fig,axs=plt.subplots(1,2,figsize=(10,5.7));errorrows=[]
 for ax,(i,r)in zip(axs,chosen):
  win=wi[r['window_id']];l=win['targets_v2_1'];j=next(j for j in range(3,6)if l['v'][j]and not l['y'][j]and p1[i,j]>=.5);obj=COLORS[j-3];z=np.load(win['public_frames'][-1]);ax.imshow(z['rgb_static']);ax.axis('off');sub=ax.inset_axes([.67,0,.33,.33]);sub.imshow(z['rgb_gripper']);sub.axis('off');ax.set_title('外沿接触'if r['condition']=='outside_front_supported'else'堆叠突出');ax.text(.5,-.03,f"{FACTS[j]}：新GT=False\n初版P={p0[i,j]:.4f}；修正P={p1[i,j]:.4f}\n完整容纳余量={l['objects'][obj]['minimum_side_rim_clearance_m']*1000:.1f}mm",ha='center',va='top',transform=ax.transAxes)
  errorrows.append({'window_id':r['window_id'],'fact':FACTS[j],'GT':False,'initial_probability':float(p0[i,j]),'repair_probability':float(p1[i,j]),'public_frame':win['public_frames'][-1]})
 fig.text(.5,-.08,'两图为真实C相机帧，右下腕部相机；负余量说明至少一个角点越界。概率≥0.5被报作In，因此两版都误判。GT仅用于此处离线评分。',ha='center',fontsize=10);figsave(fig,'remaining_In_errors.png');save(R/'tables/illustrated_In_errors.json',errorrows)
 aliases=read(E/'results/close_state_aliasing.json')['exact9_ambiguous_groups'];fig,ax=plt.subplots(figsize=(10,4));labels=[]
 for i,g in enumerate(aliases):
  a=g['outcome_counts'].get('Closed',0);b=g['outcome_counts'].get('Intermediate',0);ax.bar(i,a,color='#277f87',label='实际关严'if i==0 else None);ax.bar(i,b,bottom=a,color='#bd6b50',label='实际仍为中间状态'if i==0 else None);labels.append(('AB'if g['split']=='train'else'C')+'：仅'+COLORS[g['exact9'][3:6].index(1)]+'在内');ax.text(i,a/2,str(a),ha='center',color='white');ax.text(i,a+b/2,str(b),ha='center',color='white')
 ax.set_xticks(range(len(aliases)),labels);ax.set(ylabel='真实Close调用数',title='完全相同的命名九事实前态，仍有不同关闭结果');ax.legend(loc='upper right');fig.text(.5,-.06,'各组均为三个Holding=False、Open=True、Closed/Mid=False，且只有标题所示物块In=True；颜色表示真实结果。不同物理布局被九事实压成同一状态。',ha='center',fontsize=10);figsave(fig,'exact_state_ambiguity.png')
 text=(R/'REPORT.md').read_text().replace('AB/C审计、一次训练与开发结论','AB/C审计、一次有界修正与开发结论')
 text=text.replace('本轮没有新增D验收，先完成状态定义和开发验证。','本机已实际完成27根组、171回合、108次固定技能调用、630个新窗口，以及两版各1200步训练（共2400步）。唯一的配对损失修正未提高C完整状态正确数：仍为165/209。两版Closed均0误报，但越界承重反例仍41/41误报In。本轮没有新增D验收。')
 text=text.replace('未通过的门槛不降低，未追加训练种子、超参数搜索或新D试验。','未通过的门槛不降低；在下面明确记录唯一一次、同种子的配对损失修正。没有追加超参数搜索或新D试验。')
 text=text.replace('已把Holding数、In数和抽屉三状态相同、物体身份可置换的前态分组；若既有Close成功又有受阻，则不能从这些状态发明确定的失败前提。','除了数量汇总，也严格按完全相同的九个命名事实分组，仍发现Close结果冲突；不能从这些状态发明确定的失败前提。')
 text=text.replace('一次新查询训练、迁移对照','初版查询训练与一次配对损失修正、迁移对照')
 section=f'''## 唯一一次读出修正：训练对比分开了，C仍未通过

初版在AB的越界承重反例也错69/88。为查明是否是冻结表示完全相同导致无法分离，仅在AB找出24组同根、同物体、In标签相反的配对。冻结查询输入没有一对完全相同，也没有差异小于1e−5的一对；最大分量差范围0.099–1.823。因此不能宣称编码器已经丢失了全部几何信息。这为一次加强配对区分的目标函数检查提供依据，并不证明唯一神经原因。

唯一改变是给原损失加`0.25 × mean(softplus(1 − (正例In logit − 负例In logit)))`，每步8对；仍是同一冻结编码器、同一共享头、seed0、1200更新、优化器、基础32样本批次及C检查点选择标准。基础采样索引逐项验证相同；配对用独立随机流，全部来自AB。初版选600步，修正版按同一C损失规则选1200步。没有选用D，没有改阈值或验收门槛，也没有第二次读出修正。

{table(['版本','AB完整九事实','C完整九事实','C Closed误报','C In误报','C In漏报','新C Holding完整三项','旧C Holding完整三项'],rows)}

![唯一修正的收益与边界](figures/single_repair_comparison.png)

24个训练配对中，两端都按0.5正确分类的配对从{stats['before_both_correct']}增加到{stats['after_both_correct']}，正负logit差中位数从{stats['before_median_margin']:.2f}变为{stats['after_median_margin']:.2f}。AB越界承重误报69/88降至59/88；C仍41/41。修正版解决一个In漏报，却新增两个新C Holding误报；完整窗口正确数恰好仍165/209并不表示错误完全相同。两版都不晋升为验收候选，旧模型与本轮初版均保留。

![仍未修复的C In错误](figures/remaining_In_errors.png)

外侧承重、间接支撑但突出两类C误报分别41/41、24/24；后者是前者的子集，不能加成65个独立错误。悬空夹持反例两版均0/3误报；C有1个几何边界事实保留不确定，210窗口中209个九项均确定，并非通过屏蔽越界例提高分数。固定预算内这次修正只证明训练配对区分改善，没有证明跨环境的完整容纳读出已可靠。

## 规则部署门槛与物理状态的独立限制

![相同九事实的不同结果](figures/exact_state_ambiguity.png)

例如AB中，三个Holding都假、仅红色In为真、Open为真的**完全同一个九事实前态**，Close出现7次关严、3次中间状态；仅蓝色In为真时为6次关严、4次中间状态。该冲突在正确GT符号上就存在，替换视觉答案不能消除。18个同根摆放对照有12组堆叠受阻而并排关严；另外6组堆叠也关严，部分过程中物体滑动／重排，不应概括为“堆叠一定失败”。接触峰值是模拟器实际记录的量，不是增加执行器推力；大峰值本身也不能作为失败判据。

本轮新标签数据中Open、Lift、Place在AB分别只有6次成功、0次失败。最短合取初拟合因此会给出空前提，甚至允许空手且抽屉关闭时直接Place。这个初拟合和无效计划保存在`learned_rules9_initial.json`／`rule_learning9_initial.json`，不作为可用算子。部署门槛要求观察到负例且可精确区分、效果角色可绑定；三类技能因此被标为unsupported，Close也因冲突／额外物体效果不获支持。该门槛只禁止缺证据的规则，没有手写前提、效果或GT后继。旧七事实失败数据缺少新版In所需完整接触证据，未强行迁移。

后续边界明确：本轮不继续加训练种子、调阈值、采集D或改控制器。新的有界实验需要先解决AB/C视觉完整容纳判别和新版动作负例覆盖；若仍以确定符号后继预测Close，还需要单独设计并验证可观测的突出／阻挡状态，或另行决定怎样处理非确定后继。调整固定Place落点属于执行器变化，不能混入本轮成绩。上述是下一步条件，不是已实施或已验收能力。

'''
 text=text.replace('## 结论与交付边界',section+'## 结论与交付边界');(R/'REPORT.md').write_text(text)
 summary=read(R/'tables/report_summary.json');summary.update({'training_updates':2400,'training_runs':2,'unique_reader_repair':1,'initial_model':t0['selected']['path'],'repair_model':t1['selected']['path'],'selected_model':None,'promoted_for_acceptance':False,'development_gate_after_repair':d1['gate_checks']});save(R/'tables/report_summary.json',summary)
 # Preserve exact source/config snapshots as portable evidence; raw data remain in project.
 for name,ep in [('initial',E),('repair',EC)]:
  for folder in ['source','configs']:
   dest=R/'source'/name/folder;dest.mkdir(parents=True,exist_ok=True)
   for f in (ep/folder).iterdir():
    if f.is_file():shutil.copy2(f,dest/f.name)
 print(json.dumps({'comparison':stats|{'pairs_detail':'saved separately'},'report':str(R/'REPORT.md')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
