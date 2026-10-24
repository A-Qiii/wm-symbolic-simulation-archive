from common import *
import shutil,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=FONT.get_name();plt.rcParams['axes.unicode_minus']=False
VARIANTS=['focal','BCE','mixed'];COLORS={'focal':'#2466b5','BCE':'#d77a29','mixed':'#278575'};LABEL={'focal':'原focal γ2/2','BCE':'全BCE γ0/0','mixed':'混合 γ0/2'};CAT={'all':'总体 micro','drawer_front':'抽屉前板含把手','gripper_finger':'两根手指及指尖'}
def get(rows,v,s,split,cam,cat):return next(q for q in rows if q['variant']==v and q['seed']==s and q['split']==split and q['camera']==cam and q['category']==cat)
def draw(ax,rows,seed,split,cam,cat):
 for v,folder in [('focal','baseline'),('BCE','BCE'),('mixed','evaluation')]:
  q=get(rows,v,seed,split,cam,cat);z=np.load(E/f'data/{folder}/pr_seed{seed}_{split}_{cam}_{cat}.npz');ax.plot(np.r_[0.,z['recall']],np.r_[1.,z['precision']],color=COLORS[v],lw=1.1,label=f'{LABEL[v]} AP={q["AP50"]:.3f}');ax.step(z['recall'],z['precision_envelope'],where='pre',color=COLORS[v],ls=':',lw=.6,alpha=.6);ax.scatter(q['recall05'],q['precision05'],color=COLORS[v],s=35,edgecolor='white',linewidth=.5,zorder=4)
 ax.set(xlabel='召回率 TP / 全部有效GT',ylabel='精确率 TP / (TP+FP)',xlim=(0,1.01),ylim=(0,1.025));ax.legend(fontsize=8,loc='lower left');ax.grid(alpha=.17)
def plots(rows,sd,gd):
 for split in ['dev','train']:
  fig,axes=plt.subplots(2,3,figsize=(15,9))
  for seed in [0,1]:
   for j,cat in enumerate(['all']+CATS):draw(axes[seed,j],rows,seed,split,'all',cat);axes[seed,j].set_title(f'种子{seed}｜{CAT[cat]}')
  fig.suptitle(('C开发集'if split=='dev'else'A/B训练集')+'｜三种分类损失，同初始化、日程与1200步预算',fontsize=16,y=.995);fig.text(.5,.951,'蓝=原focal，橙=全BCE，绿=前板BCE＋手指focal；γ顺序为前板/手指。每个点来自正常解码。',ha='center',fontsize=10);fig.text(.5,.013,'实线=实际PR；点线=AP精确率包络；圆点=原0.5工作点。IoU≥0.5匹配、全GT分母；总体是混合排序micro AP，不是分类AP均值。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(R/f'figures/three_way_PR_{split}.png',dpi=160);plt.close(fig)
 for seed in [0,1]:
  fig,axes=plt.subplots(2,3,figsize=(15,9))
  for i,cam in enumerate(['static','gripper']):
   for j,cat in enumerate(['all']+CATS):draw(axes[i,j],rows,seed,'dev',cam,cat);axes[i,j].set_title(f'{"静态相机"if cam=="static"else"腕部相机"}｜{CAT[cat]}')
  fig.suptitle(f'C开发集：种子{seed}分相机PR；颜色、实线/点线/圆点与总览相同',fontsize=15);fig.text(.5,.012,'相机与类别仅用于分层评分；网络输入仍只有同一冻结RGB特征。未知图像类别屏蔽，漏检目标保留在GT分母。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.035,1,.955]);fig.savefig(R/f'figures/three_way_C_camera_seed{seed}.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(13,5.8))
 for seed,ax in enumerate(axes):
  for j,v in enumerate(VARIANTS):
   q=[next(x for x in sd if x['variant']==v and x['seed']==seed and x['category']==c)for c in CATS];med=[x['TP_score_quartiles'][1]for x in q];err=np.array([[m-x['TP_score_quartiles'][0]for m,x in zip(med,q)],[x['TP_score_quartiles'][2]-m for m,x in zip(med,q)]]);ax.errorbar(np.arange(2)+(j-1)*.14,med,yerr=err,fmt='o',capsize=4,color=COLORS[v],label=LABEL[v])
  ax.axhline(.5,color='#555555',ls='--',lw=1,label='固定阈值0.5');ax.set(xticks=[0,1],xticklabels=['前板','手指'],ylim=(0,.75),ylabel='普通匹配TP的预测分数',title=f'种子{seed}：点=中位数，误差线=四分位区间');ax.grid(axis='y',alpha=.16);ax.legend(fontsize=8,loc='upper left')
 fig.suptitle('C开发集分数分布：混合方案前板与手指的分数尺度拉开',fontsize=16);fig.text(.5,.02,'仅描述各模型已匹配TP，匹配集合可能不同；不是GT中心oracle或置信区间。未修改任何分数、阈值、排序或主指标。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(R/'figures/score_comparability.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(14,5.7))
 for j,v in enumerate(VARIANTS):
  q=[next(x for x in gd['initial_probes']if x['variant']==v and x['seed']==s and x['batch']=='root_balanced')for s in [0,1]];axes[0].bar(np.arange(2)+(j-1)*.23,[x['total_gradient_norm']['shared']for x in q],.23,color=COLORS[v],label=LABEL[v])
 axes[0].set(xticks=[0,1],xticklabels=['种子0','种子1'],ylabel='共享层总梯度L2范数',title='相同初始权重、固定16根批次；裁剪前');axes[0].legend(fontsize=8)
 for j,v in enumerate(['BCE','mixed']):
  q=[next(x for x in gd['final_parameter_distances']if x['variant']==v and x['seed']==s)for s in [0,1]]
  for k,(group,name)in enumerate([('shared','共享层'),('finger_output','手指出口')]):axes[1].bar(np.arange(2)+(j*2+k-1.5)*.17,[x['relative_distance'][group]for x in q],.17,color=COLORS[v],alpha=1 if k==0 else .5,label=f'{LABEL[v]} {name}')
 axes[1].set(xticks=[0,1],xticklabels=['种子0','种子1'],ylabel='与原focal最终参数的L2距离 / 原参数L2范数',title='训练后参数变化；深色=共享层，浅色=手指出口');axes[1].legend(fontsize=8)
 for ax in axes:ax.grid(axis='y',alpha=.17)
 fig.suptitle('共享层耦合证据：手指直接梯度不变，不等于手指模型不变',fontsize=15);fig.text(.5,.015,'左为零优化更新诊断；右为各最终权重比较。两种子都显示共享层及手指出口改变，但不能据此把最终AP变化全部归因于共享层。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(R/'figures/shared_layer_diagnostics.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(15,5))
 for v,folder in [('focal',E/'data/baseline'),('BCE',E/'data/BCE'),('mixed',E/'results')]:
  for seed,style in [(0,'-'),(1,'--')]:
   h=read(folder/f'train_seed{seed}.json')['history'];h=[q for q in h if q['update']==1 or q['update']%100==0];values=[[q['total']for q in h],[q['focal']if v=='focal'else q['classification']for q in h],[5*q['smoothL1']+2*q['GIoU']for q in h]]
   for ax,vals in zip(axes,values):ax.plot([q['update']for q in h],vals,color=COLORS[v],ls=style,label=f'{v} S{seed}')
 for ax,title in zip(axes,['总损失','分类项（目标不同，不能直接排名）','相同公式的加权回归项']):ax.set(title=title,xlabel='优化更新次数',ylabel='损失（对数轴）',yscale='log');ax.legend(fontsize=7);ax.grid(alpha=.16)
 fig.suptitle('训练过程：蓝=focal，橙=BCE，绿=混合；实线=S0，虚线=S1',fontsize=15);fig.text(.5,.015,'三者显示相同13个记录位置；新混合保存全部1200步。自然尺度变化未通过学习率、权重或裁剪补偿。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(R/'figures/training_losses.png',dpi=160);plt.close(fig)
def main():
 install_guard([]);rows=read(E/'results/three_way_comparison.json');selection=read(E/'results/selection.json');sd=read(E/'results/score_comparability.json')['rows'];gd=read(E/'results/shared_layer_diagnostics.json');plots(rows,sd,gd)
 lines=['# 类别分开gamma：最后一次有界损失对照','', '**按训练前冻结的双种子规则，保留原focal配置及0.5工作点；不采用全BCE或混合方案，不部署任何新模型。** 混合方案C总体micro AP50在S0/S1下降9.23/5.00个百分点，原0.5总体精确率也都下降。两类分开的结果不能直接拼成共享模型的结果。','', '这是一项依据已见C结果提出的探索方案。C改善或下降只用于开发判断，不能作为新的独立验证；本轮没有读取D样本/特征/预测/评分，也没有采集或评分任何新根。主指南的完整世界模型与规划实验尚未完成。','', '## 变量、预算与事先选择规则','', '|元素|本轮设置|原因|','|---|---|---|','|唯一训练变化|前板槽0 gamma=0；手指槽1/2 gamma=2|检验前板既有开发收益能否在共享头内保留；不是拼接两个模型|','|分类权重/归一化|正例0.25、负例0.75；每图除以全部正中心数至少1，再批均值|原样保留，未同时调混合权重或损失尺度|','|输入/头|同一冻结DINO缓存与归一化；共享384→64→15头，25615参数|不增加GT、相机或根信息输入|','|回归/解码|5×SmoothL1＋2×GIoU；同类NMS IoU>0.5；匹配IoU≥0.5|坐标、正中心、有效mask与未知处理未变|','|训练预算|种子0/1各1200步、batch16、同初始权重与原batch日程；AdamW lr0.001/wd0.0001、clip1|两次新fit已全部完成，无第三次或延长|','|原比较组|已封存原focal、全BCE的AB/C结果|未重新训练或重做全量推理|','', '训练前规则：候选在两个种子上，C总体、前板、手指AP均不低于原focal，前板AP严格提高，且原0.5工作点总体精确率与召回率都不低于原focal；否则保留原方案。浮点比较仅允许1e−12数值容差。若两候选均通过，预先优先混合方案；没有平均掉失败种子或挑最好单seed。相机分层和训练指标用于诊断，不事后新增选择条件。全BCE结果当时已知，规则是在混合训练前冻结，不声称针对全部历史方案的前瞻预注册。','', '该规则只是“可进入新根验证”的保守相对筛选，不是生产或主实验验收门槛。原focal被保留也不表示当前0.5召回率足够。','', '## 三方案C结果','', '|种子|对象|原focal AP50|全BCE AP50|混合AP50|混合−focal（百分点）|','|---|---|---:|---:|---:|---:|']
 for s in [0,1]:
  for c in ['all']+CATS:
   q=[get(rows,v,s,'dev','all',c)for v in VARIANTS];lines.append(f'|{s}|{CAT[c]}|{q[0]["AP50"]:.4f}|{q[1]["AP50"]:.4f}|{q[2]["AP50"]:.4f}|{100*(q[2]["AP50"]-q[0]["AP50"]):+.2f}|')
 lines+=['','![三方案C PR](figures/three_way_PR_dev.png)','', 'AP50是正常完整候选在IoU=0.5下的全点精确率包络积分，所有有效GT都在分母中。总体micro把两类分数合并排序，不是分类AP均值；不是COCO多IoU mAP，也不是GT中心oracle。','', '|种子|方案|总体0.5精确率|总体0.5召回率|F1|TP/FP/FN|','|---|---|---:|---:|---:|---|']
 for s in [0,1]:
  for v in VARIANTS:
   q=get(rows,v,s,'dev','all','all');lines.append(f'|{s}|{v}|{q["precision05"]:.4f}|{q["recall05"]:.4f}|{q["F1_05"]:.4f}|{q["TP05"]}/{q["FP05"]}/{q["FN05"]}|')
 lines+=['','混合的总体F1与召回率虽略升，精确率下降且micro AP下降；预先规则因此不通过，不能事后改用较有利的F1来挑模型。前板0.5召回率从S0/S1的0.18/0.15降到0.16/0.11，即前板AP上升也不等于固定工作点更好。','', '所有36个划分×相机×类别×种子条件、三方案共108行AP与0.5统计保存在 `three_way_comparison.csv`；当前混合另保留108个预列0.5/0.3/0.1点，这些低阈值只描述，不选择或使用。','', '![S0分相机](figures/three_way_C_camera_seed0.png)','', '![S1分相机](figures/three_way_C_camera_seed1.png)','', '![AB训练PR](figures/three_way_PR_train.png)','', '## 为什么两类AP不能拼成总体AP','', 'S1混合前板与手指AP分别提高4.97和3.99个百分点，总体micro AP却下降5.00个百分点。这并非算术矛盾：分类AP只衡量各类内部排序，总体micro还取决于跨类别分数排序。已从两类PR阈值增量独立重建全部6条C总体曲线的AP，与保存值一致。','', '![分数尺度](figures/score_comparability.png)','', '|种子|方案|前板TP分数中位数|手指TP分数中位数|每个前板TP之前的手指FP平均数|','|---|---|---:|---:|---:|']
 for s in [0,1]:
  for v in VARIANTS:
   a=next(q for q in sd if q['variant']==v and q['seed']==s and q['category']==CATS[0]);b=next(q for q in sd if q['variant']==v and q['seed']==s and q['category']==CATS[1]);lines.append(f'|{s}|{v}|{a["TP_score_quartiles"][1]:.4f}|{b["TP_score_quartiles"][1]:.4f}|{a["opposite_class_FP_strictly_above_each_TP_mean"]:.2f}|')
 lines+=['','混合方案前板TP分数明显下移，而手指TP分数接近原focal；更多手指FP可能在前板TP之前进入总体排序。上表直接统计现有正常候选，不改分数。各模型TP集合可能不同，因此这属于分数排序的描述性证据，不能把全部AP差异定量归因于一个因素；没有用GT重标分数或借此新调阈值。','', '## 共享隐藏层与手指影响','', '![共享层证据](figures/shared_layer_diagnostics.png)','', '在同一初始权重、同一冻结训练批次上，混合与focal的手指直接梯度及回归梯度逐位一致；但前板目标改变了共享层梯度。全头clip1还会改变所有梯度的共同缩放。最终混合的共享层和手指出口参数均与原focal不同，手指C AP在S0下降2.59个百分点、S1上升3.99个百分点，未获得双种子稳定保护。','', '|种子|固定16根批次中，仅改前板造成的共享层梯度差L2|混合最终共享层参数相对距离|混合最终手指出口参数相对距离|','|---|---:|---:|---:|']
 for s in [0,1]:
  a=next(q for q in gd['same_initial_pairs']if q['seed']==s and q['batch']=='root_balanced');b=next(q for q in gd['final_parameter_distances']if q['seed']==s and q['variant']=='mixed');lines.append(f'|{s}|{a["changed_front_gradient_norm_by_group"]["shared"]:.4f}|{b["relative_distance"]["shared"]:.4f}|{b["relative_distance"]["finger_output"]:.4f}|')
 lines+=['','相对距离=参数差L2/原focal参数L2。初始探测含两个固定批次×两种子×三目标，共12项，优化更新为0；稀少槽强调批次不当作总体平均。最终层间变化与训练共同演化；本轮没有冻结/交换共享层，因此不能声称已隔离证明“共享层单独导致最终手指AP变化”。','', '## 训练和覆盖','', '![训练曲线](figures/training_losses.png)','', '|种子|实际更新|前板/手指默认/碰撞槽正曝光|含碰撞正例batch数|裁剪更新数|非有限梯度|','|---|---:|---|---:|---:|---:|']
 for s in [0,1]:
  q=read(E/f'results/train_seed{s}.json');lines.append(f'|{s}|{q["updates"]}|{q["positive_exposures_by_slot"]}|{q["batches_containing_collision_positive"]}|{q["clipped_updates"]}|{q["nonfinite_gradients"]}|')
 lines+=['','每张AB训练图出现75次；碰撞槽仍只有11图/9根的11个正中心，实际825次曝光未增加。两个混合训练的梯度均有限，1200步均触发原clip1。原focal无历史梯度日志，不能据此断言混合“比原方案更经常裁剪”。','', '## 选择结果、停止条件与后续','', f'按冻结规则输出：`{selection["selected_configuration"]}`。全BCE与混合均不满足双种子要求；失败条件逐项保存于 `results/selection.json`。保留原两个seed与0.5，未部署、未挑seed，亦不把保留参照当成已通过应用验收。','', '损失选择到此结束，不追加gamma、alpha或混合权重搜索。后续应转回原指南对象状态与关系读取的标签、接口及未知传播；定位器不可靠输出必须显式未知，不能当成对象不存在或关系为假。','', '新根与阈值规范见 `FUTURE_HOLDOUT_PROTOCOL.md`：准备一次16个新C根的最小定位验证，冻结配置/0.5后再生成根；以根为独立单位，报告配对不确定性。仅准备规范，没有生成根清单、采集新数据、读取新测试或开启阈值选择。原4状态技术样本和旧D均不能冒充新测试。','', '## 复核与文件','', '- 100个选定旧输入先后核对封存哈希；训练只可读原train.npz，推理只可读原train/dev RGB特征，模型和预测在评分前分别冻结。旧文件不写，D样本、特征、预测、评分不读。旧清单中的哈希元数据仅用于完整性核验。','- `configs/pre_run_frozen.json`含开训前协议、选择规则、代码、初始化和日程哈希；`results/checkpoints_frozen.json`和`predictions_frozen.json`记录完成时间与内容哈希。','- 正常解码NMS前缀与参考实现核对；完整PR/AP、GT分母、一对一匹配、未知mask、固定工作点与无更新诊断均由验证脚本检查。','- 新训练仅2次，各1200步；GPU持共享独占锁，训练和预测串行，评估/诊断/绘图CPU运行。没有下载、安装、主环境改动或应用重启。','- 后补的分数诊断和报告只读取已保存结果；没有修改冻结训练或候选规则。具体执行命令见 `RUN_COMMANDS.md`。']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n');shutil.copy2(Path(__file__).parent/'future_holdout.md',R/'FUTURE_HOLDOUT_PROTOCOL.md');save(E/'results/report_assets.json',{'utc':now(),'figures':[str(p)for p in(R/'figures').glob('*.png')],'selected_configuration':selection['selected_configuration'],'new_test_collected':False,'loss_search_closed':True});print('MIXED_REPORT_READY',flush=True)
if __name__=='__main__':main()
