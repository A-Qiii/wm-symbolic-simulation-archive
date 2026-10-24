from coverage_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import shutil,collections
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafbfc','axes.facecolor':'#fafbfc','savefig.facecolor':'#fafbfc'})
def table(h,rs):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(str(v)for v in r)+' |\n'for r in rs)
def main():
 assert read(E/'results/verification.json')['passed'];actual=read(E/'results/actual_metrics.json');changes=read(E/'results/actual_changes.json');pairs=read(E/'results/official_matched_pairs.json');full=read(E/'results/full9_actual_metrics.json');secondary=read(E/'results/secondary_metrics.json');sampling=read(E/'results/sampling_information_budget.json');training=read(E/'results/training.json');encoding=read(E/'results/encoding.json');cfg=read(E/'configs/protocol.json');figdir=R/'figures';figdir.mkdir(exist_ok=True)
 def get(head,seed,arm,cohort):return next(r for r in actual if(r['head'],r['seed'],r['arm'],r['cohort'])==(head,seed,arm,cohort))
 fig,axes=plt.subplots(1,4,figsize=(14.5,4.5),sharey=True)
 for ax,(head,rate)in zip(axes,[('Holding','FPR'),('Holding','FNR'),('In','FPR'),('In','FNR')]):
  for seed,marker,ls in[(0,'o','-'),(1,'s','--')]:
   for arm,color,label in [('control','#357ba3','旧覆盖'),('augmented','#d57a32','加官方覆盖')]:
    a=get(head,seed,arm,'official_validation');ax.plot(range(3),[c[rate]*100 for c in a['colors']],marker+ls,color=color,label=label+f' / seed{seed}',alpha=.85)
  ax.set(xticks=range(3),xticklabels=['红','蓝','粉'],title=head+' '+('误报率'if rate=='FPR'else'漏检率'),xlabel='物体颜色');ax.grid(alpha=.16);ax.set_ylim(0,35)
 axes[0].set_ylabel('错误率（%）');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,.08),frameon=False)
 fig.suptitle('官方4验证episode：逐色误报和漏检，固定600更新',fontsize=15,y=.98)
 fig.text(.5,.025,'误报率=FP/可靠负事实，漏检率=FN/可靠正事实；蓝线为旧覆盖，橙线为加官方覆盖，圆/方点对应seed0/1。线只连接类别以便阅读。',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.17,1,.92));fig.savefig(figdir/'official_color_errors.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11.6,4.8),sharey=True)
 cohorts=['C_validation','C_endpoints48','new_AB_validation','D12_seen_actual135'];names=['旧C验证273','C端点48','已见AB54','已见D12\n135边界']
 for ax,head in zip(axes,HEADS):
  for seed,color in[(0,'#357ba3'),(1,'#d57a32')]:
   rr=[next(r for r in changes if r['head']==head and r['seed']==seed and r['cohort']==c)for c in cohorts];vals=[100*r['correct_change']/r['valid_states']for r in rr];xs=np.arange(4)+(-.17 if seed==0 else .17);ax.bar(xs,vals,width=.30,color=color,label=f'seed{seed}')
   for x,y in zip(xs,vals):ax.text(x,y+(.12 if y>=0 else-.12),f'{y:+.1f}',ha='center',va='bottom'if y>=0 else'top',fontsize=9)
  ax.axhline(0,color='#333',lw=.8);ax.set(xticks=range(4),xticklabels=names,title=head);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.set_ylim(-4,7);ax.legend(frameon=False)
 axes[0].set_ylabel('整组三色正确率变化（百分点）');fig.suptitle('保留能力：加官方覆盖 − 旧覆盖；负值表示退化',fontsize=15,y=.98)
 fig.text(.5,.025,'柱色表示随机种子，两臂同初始化/共同24旧槽。C端点48是C273的子集，不是独立重复；D已反复查看，只作描述，不参与门槛。',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.075,1,.92));fig.savefig(figdir/'retention_changes.png',dpi=170);plt.close(fig)
 flips=read(E/'results/decision_flips.json');bad=sorted([r for r in flips if r['cohort']=='official_validation'and r['head']=='Holding'and r['color']=='red'and r['effect']=='regression'and r['error_after']=='FN'],key=lambda r:(r['root_id'],r['state_id'],r['seed']));example=bad[0];pub=next(r for r in read(E/'data/public/states.json')if r['state_id']==example['state_id']);image=np.load(pub['path']);er=[r for r in bad if r['state_id']==example['state_id']];assert len(er)==2
 fig,axes=plt.subplots(1,3,figsize=(12.4,4.5),gridspec_kw={'width_ratios':[1,1,1.25]})
 for ax,key,label in zip(axes[:2],['rgb_static','rgb_gripper'],['同一第96步静态图','同一第96步腕部图']):
  ax.imshow(image[key],interpolation='nearest');ax.set(xticks=[],yticks=[],title=label)
  for s in ax.spines.values():s.set_visible(False)
 for seed,r in enumerate(er):
  axes[2].bar(seed-.17,r['control_probability'],width=.30,color='#357ba3',label='旧覆盖'if seed==0 else None);axes[2].bar(seed+.17,r['augmented_probability'],width=.30,color='#d57a32',label='加官方覆盖'if seed==0 else None)
 axes[2].axhline(.5,color='#666',ls='--',label='固定0.5阈值');axes[2].set(xticks=[0,1],xticklabels=['seed0','seed1'],ylim=(0,1),ylabel='Holding(red)分数',title='真值为真：两seed新增漏检');handles,labels=axes[2].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=9)
 fig.suptitle(pub['root_id'].replace('official_ABC_','')+' / 帧'+str(pub['frame_id'])+'：输入与GT相同，读出决策改变',fontsize=14,y=.98)
 fig.text(.5,.025,'示例只作错误解释，不用于重新采样或调参。左图200×200、腕部84×84；颜色为物体外观。图表蓝/橙是两训练臂，不是物体颜色。',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.15,1,.91));fig.savefig(figdir/'Holding_red_regression.png',dpi=170);plt.close(fig);save(E/'results/report_example.json',{'rule':'first lexicographic official-validation red Holding newly-false-negative state, both seeds shown, descriptive only','source':pub,'decision_flips':er})
 primary=[]
 for cohort,title in [('official_validation','官方新验证'),('C_all_available','全部既有C'),('C_validation','既有C验证273'),('C_endpoints48','C技能端点48'),('new_AB_validation','已见AB54'),('D12_seen_actual135','已见D12actual135')]:
  for head in HEADS:
   vals=[]
   for seed in[0,1]:
    c=get(head,seed,'control',cohort);a=get(head,seed,'augmented',cohort);vals.append(f"{c['all3_correct']}→{a['all3_correct']} / {a['all3_valid']}")
   primary.append([title,head,*vals])
 conf=[]
 for head in HEADS:
  for seed in[0,1]:
   c=get(head,seed,'control','official_validation');a=get(head,seed,'augmented','official_validation')
   for cc,aa in zip(c['colors'],a['colors']):conf.append([head,seed,cc['color'],cc['TP']+cc['FN'],cc['TN']+cc['FP'],f"{cc['FP']}→{aa['FP']}",f"{cc['FN']}→{aa['FN']}"])
 pairtable=[]
 for head in HEADS:
  for seed in[0,1]:
   for color in COLORS:
    c=next(r for r in pairs if(r['head'],r['seed'],r['arm'],r['color'])==(head,seed,'control',color));a=next(r for r in pairs if(r['head'],r['seed'],r['arm'],r['color'])==(head,seed,'augmented',color));pairtable.append([head,seed,color,f"{c['paired_correct']}→{a['paired_correct']}",a['eligible_segment_color_pairs']])
 nine=[]
 for seed in[0,1]:
  c=next(r for r in full if(r['seed'],r['arm'],r['cohort'])==(seed,'control','official_validation'));a=next(r for r in full if(r['seed'],r['arm'],r['cohort'])==(seed,'augmented','official_validation'));nine.append([seed,f"{c['all9_correct']}→{a['all9_correct']} / {a['all9_valid']}",f"{a['drawer3_correct']} / {a['drawer3_valid']}（两臂相同）"])
 secs=[]
 for seed in[0,1]:
  for wm in['mixed','latent']:
   for method in['one_step','recursive']:
    c=next(r for r in secondary if(r['seed'],r['world_model'],r['method'],r['arm'])==(seed,wm,method,'control'));a=next(r for r in secondary if(r['seed'],r['world_model'],r['method'],r['arm'])==(seed,wm,method,'augmented'));secs.append([seed,wm,method,f"{c['all3_correct']}→{a['all3_correct']} / 135",sum(x['FP']for x in c['colors']),sum(x['FP']for x in a['colors'])])
 sb=[]
 for r in sampling:sb.append([r['head'],r['seed'],r['arm'],r['old_draws'],r['new_draws'],'/'.join(str(int(x))for x in r['group_positive_draws_per_color']),'/' .join(str(int(x))for x in r['group_negative_draws_per_color']),'/' .join(str(int(x))for x in r['group_unknown_draws_per_color'])])
 (R/'REPORT.md').write_text(f'''# 固定读出头的数据覆盖对照：主结果与退化

完成2头×2seed×2臂共8次训练，每次固定600更新。**新增官方派生覆盖提高了官方验证的整体读取，但Holding红色漏检增加，In在旧C有小幅退化。** 本轮没有改表征、T、drawer或阈值，没有追加训练。Holding按预先冻结的逐色限制停止预测z诊断；只有In通过保留容差后读取了原已保存预测z。没有正式模型替换。文中的sigmoid输出是正例加权BCE训练所得分数，并未校准为概率。

## 主要actual结果

下表为整组三色都正确的状态数，写作“旧覆盖→加官方覆盖 / 全组三标签有效状态”。两seed分别展示，不挑最好seed。全部既有C为771状态；In其中754状态三色全有效。C验证273和端点48包含在既有C中，不能相加当独立样本。

{table(['actual集合','头','seed0','seed1'],primary)}

官方验证Holding整组改善伴随红色漏检从1/17增到2/17（两seed均如此），增加5.88个百分点；这超过训练前冻结的逐色恶化上限5个百分点，不能用总体正确率掩盖。In在C273从261/273降到257/273（下降1.47个百分点，两seed）；其C48端点保持47/48，下降仍在预设2个百分点保留容差内。Holding在已见D12也从132/130降到129/128；D只作描述，不参与门槛。

{table(['头','seed','颜色','可靠正事实','可靠负事实','误报FP：旧→加','漏检FN：旧→加'],conf)}

![官方逐色错误率](figures/official_color_errors.png)

图中横轴为物体颜色，纵轴错误率%。FPR分母为可靠负事实、FNR分母为可靠正事实；蓝/橙表示旧覆盖/加官方覆盖，圆/方和实/虚线表示seed0/1。连线只是连接类别，不代表连续颜色尺度。所有官方验证色类的正/负事实均来自4个episode；帧相关，不把120状态当120独立episode。

![保留能力变化](figures/retention_changes.png)

纵轴为加覆盖减旧覆盖的三色整组正确率变化，正值改善、负值退化；柱色为两seed。端点48是273集合的子集。完整旧C各子集和每根结果保存在[actual明细](../../experiments/official_readout_coverage_control_20261018/results/actual_metrics.json)，未知类不填成负例，缺少某色正类时FNR保持null。

## 同任务段的正负配对与全9

同段配对以“语言标注段×物体颜色”为单位：只有该段的采样状态中同时有可靠正与可靠负才纳入；该段该色全部已知正负都判对才记配对正确。语言只提供段身份，GT仍是新物理标签。不同段可能包含相同源帧，配对数不是独立轨迹数。

{table(['头','seed','颜色','配对正确：旧→加','有效段×色数'],pairtable)}

Holding seed1的红配对也从4/10降至3/10，进一步说明总体改善并不覆盖所有颜色。In配对红/蓝改善、粉维持。完整段ID、根、正负状态数见[配对表](official_matched_pairs.csv)。

组合全9采用同seed、同臂的Holding头与In头，drawer三个输出直接复制原冻结Query，按原argmax判类别：

{table(['seed','官方全9正确：旧→加','原drawer正确'],nine)}

官方120状态中4个In未知、13个drawer边界无效，全9有效分母为103；drawer自身有效分母107。全9只在全部标签有效时评分，不把未知算负。原drawer没有训练或校准，输出已逐数组验证两臂完全一致。

## 可追溯的Holding退化示例

{table(['来源状态','seed','旧覆盖分数','加覆盖分数','真实标签'],[[pub['state_id'],r['seed'],f"{r['control_probability']:.4f}",f"{r['augmented_probability']:.4f}",'Holding(red)=真']for r in er])}

![同一输入的红Holding决策退化](figures/Holding_red_regression.png)

这是按来源/状态字典序找到的首个官方验证红Holding新增漏检，两个seed都出现。两臂输入z、物理标签、初始权重、归一化、共同24槽及优化器相同；本对照改变的是最后8槽的数据。可以把这次读出决策差异归于该受控训练数据替换，不能进一步声称某个图像细节或T导致错误。未改变阈值、补采此例或对单例修补。

补充的只读错误与训练覆盖审阅见[错误审阅](ERROR_REVIEW.md)。其中明确区分新增漏判和两臂共同漏判，并给出原图/新图、物理稳定性、旧C/D代表图和可量化覆盖边界。

## 数据、旧池和固定预算

- 主新增集合只用240个新源状态：官方派生A/B训练120状态/4episode，验证120状态/4episode。均为step96新RGB/proprio和同时刻新标签。旧16复测不入主训练/验证；扩充原有6个In未知中2个随旧复测排除，剩余4个在官方验证按mask排除。
- Holding旧池完全复制最近独立Holding诊断的210状态/36根：156个原生几何修正历史AB状态，加6个旧组合训练根的全部54状态。In旧池复制最近对象In实验的**原control池156状态/30根**；此前加6个双关系状态的另一coverage臂没有混入。两头旧池不同，因此两头效果不是等旧信息量比较。
- 两臂均使用各头原训练统计归一化（Holding210、In156），没有用官方新增、验证、C或D拟合统计。旧原生几何修正y/v两臂完全相同；旧C无完整几何的In仍未知。
- 结构均为共享对象token64→32→SiLU→1，2113参数。保持原masked加权BCE、AdamW lr0.001/wd0.001、全局clip1；Holding正权8.5454545，In各色5.1428571。每state先对有效物体平均，再对batch平均。
- 每head/seed的control完整复制该头旧600×32采样日程，augmented的前24槽逐项相同，后8槽替换为官方训练状态。每批新增4episode各2样本，整个训练120状态各出现40次。两臂同初始化，固定终点600、阈值0.5；未搜超参或选择D结果。

{table(['头','seed','臂','旧状态抽取','新状态抽取','有效正抽取 红/蓝/粉','有效负抽取 红/蓝/粉','未知抽取 红/蓝/粉'],sb)}

这里的数量是训练抽取次数，不是独立样本数。两臂更新数/批大小相同，但有效正负信息预算不同，不能称为完全等信息实验。

冻结编码严格复用原DINO、RGB颜色登记器、变换/PCA与交互编码器。登记器按红/蓝/粉输出；原仿真A槽粉/蓝/红映射[2,1,0]，B槽红/蓝/粉映射[0,1,2]仅用于标签身份核验。没有把scene_obs、接触、坐标或标签输入编码器。8个旧AB输入的对象/场景特征与缓存最大差0，z最大差7.15×10⁻⁷；编码模型前后参数哈希相同。

## 门槛与只读In的次级诊断

本轮在训练前冻结的诊断门槛为：两seed官方验证逐色平均平衡错误率改善至少1个百分点、整组正确率不下降、每色FPR/FNR恶化不超过5个百分点；C全集、C273、C48、AB54的整组正确率下降不超过2个百分点且每色FPR/FNR恶化不超过5个百分点。它是本次是否允许次级诊断的操作条件，不是原指南验收标准或显著性检验。

Holding因官方红漏检增加而不通过，未读其保存预测z，也未做全9次级诊断。In通过；以下仅将固定In头读取相同的已保存D12预测latent，**没有重新运行T，也没有产生新独立测试**。D此前已反复查看，不用于选头、选seed或阈值。

{table(['seed','原T标识','保存预测方式','In3正确：旧→加','旧FP总数','加覆盖FP总数'],secs)}

这些次级输出的FN均为0、self-held物体In误报均为0；其余误报仍保留于[次级逐色表](secondary_metrics.csv)和[逐阶段表](secondary_stage_metrics.csv)。改进说明读出头会影响同一预测z的符号判读，不能据此宣称T动力学已重新通过验证或闭环成功。

## 执行与验证证据

8次训练循环合计{sum(r['seconds']for r in training['runs']):.3f}秒，单次{min(r['seconds']for r in training['runs']):.3f}–{max(r['seconds']for r in training['runs']):.3f}秒，最大PyTorch分配{max(r['peak_allocated_bytes']for r in training['runs'])/2**20:.2f}MiB；240新状态特征提取及8旧状态一致性检查{encoding['seconds']:.3f}秒、峰值分配{encoding['peak_allocated_bytes']/2**20:.2f}MiB。CUDA上下文/缓存和nvidia-smi设备总占用口径不同。250ms轻量采样保存在日志中，短训练不能据此推断持续GPU利用率。

离线复核：8次同初始化配对及共同24槽、每新状态40次、原归一化一致、训练/验证根隔离、4个主实验未知mask、864个actual混淆格、36组全9结果与24组配对统计全部通过。C/D未进入训练或选点，T/编码器/drawer更新为0，未追加采集或网络下载。

{table(['证据','说明'],[
['[冻结方案](../../experiments/official_readout_coverage_control_20261018/configs/protocol.json)','来源、旧池、权重、采样、终点和次级门槛'],
['[actual逐色混淆](actual_confusion.csv)','逐头/seed/集合的FP/FN及分母'],
['[actual变化](actual_changes.csv)','完整改善与退化，不挑seed'],
['[全9结果](full9_actual_metrics.csv)','冻结drawer组合与有效分母'],
['[决策翻转明细](decision_flips.csv)','每个新增错误和修正错误的状态ID与分数'],
['[信息预算](sampling_information_budget.csv)','有效正负/未知抽取数量'],
['[复核结果](../../experiments/official_readout_coverage_control_20261018/results/verification.json)','实际执行的检查'],
['[训练记录](../../experiments/official_readout_coverage_control_20261018/results/training.json)','8run固定终点、耗时、显存和loss'],
['[运行命令](RUN_COMMANDS.md)','本机环境与脚本入口'],
['[交付封存](../../experiments/official_readout_coverage_control_20261018/configs/delivery_manifest.json)','源码、数据、模型、结果、图表与日志哈希']])}

本轮止于有界对照与证据交付。结果支持覆盖数据可改善部分读取，同时明确暴露Holding红漏检和In旧C退化；没有全局可靠性、正式采用或世界模型/TS-LTL完成的结论。全部8终点均保存，不自动追加调参。
''')
 (R/'RUN_COMMANDS.md').write_text('''# 本轮已执行顺序

```bash
source /home/robot/research/projects/wm_symbolic/world_model_env.sh
python /home/robot/Documents/Codex/2026-10-13/task-4/readout_cov_src/prepare.py
python /home/robot/Documents/Codex/2026-10-13/task-4/readout_cov_src/encode.py
python /home/robot/Documents/Codex/2026-10-13/task-4/readout_cov_src/freeze_train.py
python /home/robot/Documents/Codex/2026-10-13/task-4/readout_cov_src/train.py
python /home/robot/Documents/Codex/2026-10-13/task-4/readout_cov_src/evaluate.py
python /home/robot/Documents/Codex/2026-10-13/task-4/readout_cov_src/secondary.py
python /home/robot/Documents/Codex/2026-10-13/task-4/readout_cov_src/verify.py
python /home/robot/Documents/Codex/2026-10-13/task-4/readout_cov_src/report.py
```

仅encode和train使用GPU，两阶段串行持有项目GPU锁；评价/复核/绘图使用CPU。一次性训练拒绝已有attempts台账，不能再次运行凑结果。复核和报告也写文件，封存后重新审计应改用新目录。源码副本、配置、全部模型终点均在本轮独立实验目录，旧实验未修改。
''');shutil.copy2(__file__,E/'source/report.py');print('READOUT_REPORT_CREATED',flush=True)
if __name__=='__main__':main()
