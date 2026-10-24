from common import *
import csv,shutil,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fm.fontManager.addfont(font);plt.rcParams['font.family']=fm.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
LABEL={'full':'RGB＋几何','geometry_masked_RGB':'仅几何（RGB置零）','RGB_masked_geometry':'仅RGB（几何置零）'}
COL={'full':'#2463aa','geometry_masked_RGB':'#da841a','RGB_masked_geometry':'#218269'}
def table(headers,data):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n'for row in data)+'\n'
def pct(x):return '—'if x is None else f'{100*x:.2f}'
def main():
 guard('report');ev=read(E/'results/evaluation.json');audit=read(E/'results/alias_inspection.json');rows=read(B/'data/records.json');verify=read(E/'results/independent_verification.json');m={(x['arm'],x['seed'],x['group']):x for x in ev['metrics']if x['split']=='dev'};ci={(x['arm'],x['seed'],x['scope'],x['metric']):x for x in ev['root_bootstrap_CI']}
 fig,axs=plt.subplots(2,3,figsize=(15,9))
 for ri,scope in enumerate(['all','target_only']):
  for col,(metric,name)in enumerate([('precision','正例精确率'),('recall','正例召回率'),('balanced_accuracy','平衡准确率')]):
   ax=axs[ri,col]
   for ai,arm in enumerate(ARMS):
    for seed in [0,1]:
     c=ci[(arm,seed,scope,metric)];x=ai+(-.09 if seed==0 else .09);ax.errorbar(x,c['estimate']*100,yerr=[[100*(c['estimate']-c['low95'])],[100*(c['high95']-c['estimate'])]],color=COL[arm],fmt='o'if seed==0 else 's',capsize=4,ms=7);ax.annotate(f'{100*c["estimate"]:.1f}',(x,c['estimate']*100),xytext=(0,-15 if seed==0 else 8),textcoords='offset points',ha='center',fontsize=9)
   ax.set_xticks(range(3),['RGB＋几何','仅几何\nRGB置零','仅RGB\n几何置零']);ax.set_ylim(25,105);ax.set_ylabel(name+'（%）');ax.grid(axis='y',alpha=.22);ax.set_title(('全部470查询'if scope=='all'else'目标物体166查询（事后分层）')+' · '+name)
 fig.suptitle('相同261→33→1结构、相同初始化与600步预算：图像减少误报，仍有漏报',fontsize=17);fig.text(.06,.025,'蓝／橙／绿：全输入／RGB置零／几何置零；圆／方：种子0／1；线段：按完整12根配对重采样2,000次的95%区间。\n全部查询含60正例、410负例；目标分层含60正例、106负例。其余304条非目标查询全为负例。C已用于先前分析，本图为探索性诊断。',fontsize=10);fig.tight_layout(rect=[0,.10,1,.94]);fig.savefig(R/'figures/01_matched_comparison.png',dpi=150);fig.savefig(R/'figures/01_matched_comparison.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(15,5.7))
 for ax,(metric,name,scale)in zip(axs,[('precision','精确率变化（百分点）',100),('balanced_accuracy','平衡准确率变化（百分点）',100),('NLL','NLL变化（越负越好）',1)]):
  for j,(scope,seed)in enumerate([('all',0),('all',1),('target_only',0),('target_only',1)]):
   d=next(x for x in ev['paired_differences']if x['comparison']=='full-minus-geometry_masked_RGB'and x['seed']==seed and x['scope']==scope and x['metric']==metric);ax.errorbar(scale*d['estimate'],j,xerr=[[scale*(d['estimate']-d['low95'])],[scale*(d['high95']-d['estimate'])]],fmt='o'if seed==0 else's',color='#2463aa'if scope=='all'else'#218269',capsize=5);ax.text(.02,.98-j*.08,f'{scale*d["estimate"]:+.2f} [{scale*d["low95"]:+.2f}, {scale*d["high95"]:+.2f}]',transform=ax.transAxes,va='top',fontsize=9)
  ax.set_yticks(range(4),['全部 · 种子0','全部 · 种子1','目标 · 种子0','目标 · 种子1']);ax.set_ylim(-.5,4.8);ax.axvline(0,color='gray',ls='--');ax.grid(axis='x',alpha=.2);ax.set_xlabel(name)
 fig.suptitle('加入RGB的配对变化：全输入减去RGB置零',fontsize=17);fig.text(.04,.03,'蓝：全部470查询；绿：目标物体166查询（事后分层）。点与线：差值及整根95%区间；0虚线表示无差异。\n精确率与NLL均改善；总体平衡准确率区间跨零。目标分层改善更大，仍属于已见C上的探索性结果。未作多重比较校正。',fontsize=10);fig.tight_layout(rect=[0,.15,1,.93]);fig.savefig(R/'figures/02_paired_RGB_effect.png',dpi=150);fig.savefig(R/'figures/02_paired_RGB_effect.pdf');plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(14,8));statepred=[]
 for row,root in enumerate(audit['roots']):
  trace=root['trace'];t=np.array([x['control']for x in trace]);ext=np.array([x['external_vertical_normal_force_sum_N']for x in trace]);holding=np.array([int(x['holding'])for x in trace]);z=np.array([x['z_m']for x in trace]);axs[row,0].plot(t,ext,color='#da841a',label='桌面等外部接触的向上法向力');axs[row,0].set_ylabel('外部向上法向力（N）');axs[row,0].set_title(root['root_id']+' · 接触来源');axs[row,1].plot(t,(z-z[0])*1000,color='#2463aa',label='方块相对初态高度');right=axs[row,1].twinx();right.step(t,holding,color='#218269',where='post',label='原Holding真值');right.set_ylim(-.1,1.25);right.set_yticks([0,1]);right.set_ylabel('Holding真值',color='#218269');axs[row,1].set_ylabel('相对初态高度（mm）');axs[row,1].set_title(root['root_id']+' · 高度与标签')
  for st in root['states']:
   if st['sample']in [3,4,7]:
    for ax in axs[row]:ax.axvline(st['control'],color='gray',ls=':',alpha=.6);ax.text(st['control'],.95,f's{st["sample"]:02d}',transform=ax.get_xaxis_transform(),ha='center',va='top')
   for cam in ['static','gripper']:
    idx=st['image_metadata'][cam]['record_index'];rr=rows[idx]
    for arm in ARMS:
     for seed in [0,1]:
      p=np.load(E/f'results/predictions_{arm}_seed{seed}_dev.npz');where=np.where(p['record_indices']==idx)[0]
      statepred.append({'root_id':root['root_id'],'sample':st['sample'],'anchor':st['anchor'],'camera':cam,'record_index':idx,'y':int(st['holding']),'eligible':rr['eligible'],'arm':arm,'seed':seed,'probability':float(p['probability'][where[0]])if len(where)else None})
  for ax in axs[row]:ax.set_xlabel('真实控制步（每步8个物理tick）');ax.grid(alpha=.2)
 fig.suptitle('已有逐控制步接触记录：桌面支撑消失后标签改变，随后抬升并稳定',fontsize=16);fig.text(.04,.015,'橙：外部接触的向上法向力（不含切向摩擦）；蓝：方块高度；绿阶梯：原Holding标签。灰点线：已保存的闭爪s03、抬升s04、释放s07图像时刻。\n只使用原始记录，没有中间RGB重渲染。两根在闭爪图像后下一控制步离开桌面；s04在其后73控制步，标签已稳定。力与3D高度仅用于诊断，不进入任何模型。',fontsize=10);fig.tight_layout(rect=[0,.10,1,.94]);fig.savefig(R/'figures/03_contact_timeline.png',dpi=150);fig.savefig(R/'figures/03_contact_timeline.pdf');plt.close(fig)
 with(R/'alias_all_neighbor_predictions.csv').open('w')as f:w=csv.DictWriter(f,list(statepred[0]));w.writeheader();w.writerows(statepred)
 overall=table(['输入条件','种子','P%','R%','BA%','NLL','TP/FP/FN/TN'],[[LABEL[a],s,pct(m[a,s,'all']['precision']),pct(m[a,s,'all']['recall']),pct(m[a,s,'all']['balanced_accuracy']),f"{m[a,s,'all']['NLL']:.4f}",'/'.join(str(m[a,s,'all'][k])for k in ['TP','FP','FN','TN'])]for a in ARMS for s in [0,1]])
 target=table(['输入条件','种子','目标P%','目标R%','目标BA%','目标FP/106','支撑负例FP/12','非目标FP/304'],[[LABEL[a],s,pct(m[a,s,'target_only']['precision']),pct(m[a,s,'target_only']['recall']),pct(m[a,s,'target_only']['balanced_accuracy']),m[a,s,'target_only']['FP'],m[a,s,'target_closed_supported_negatives']['FP'],m[a,s,'non_target']['FP']]for a in ARMS for s in [0,1]])
 paired=table(['范围','种子','指标','全输入−仅几何（RGB置零）','整根95%区间'],[[('全部'if d['scope']=='all'else'目标分层'),d['seed'],d['metric'],f"{d['estimate']*(1 if d['metric']=='NLL' else 100):+.4f}",f"[{d['low95']*(1 if d['metric']=='NLL' else 100):+.4f}, {d['high95']*(1 if d['metric']=='NLL' else 100):+.4f}]"]for d in ev['paired_differences']if d['comparison']=='full-minus-geometry_masked_RGB'and d['metric']in ['precision','recall','balanced_accuracy','NLL']])
 aliasstats=[]
 for root in audit['roots']:
  c=root['comparisons'][0];a,b=root['states'][3:5];aliasstats.append([root['root_id'],f"{a['external_normal_force_sum_N']:.2f}→0",f"{a['object_position'][2]:.4f}→{b['object_position'][2]:.4f}",f"{100*c['wrist_full_RGB']['changed_pixel_fraction']:.2f}%",f"{100*c['block_crop']['changed_pixel_fraction']:.2f}%",f"{100*c['finger_union_crop']['changed_pixel_fraction']:.2f}%",f"{c['visible_mask_IoU']:.4f}"])
 alias_table=table(['闭爪s03→抬升s04','外部向上法向力N','方块中心z米','腕部RGB变化像素','方块裁剪变化像素','手指并集裁剪变化像素','方块mask IoU'],aliasstats)
 txt=f'''# 相同结构输入消融与Holding同框样本诊断

**固定结构和训练条件后，图像特征提供了可测量的额外预测证据：减少误报、提高精确率并降低NLL；总体平衡准确率尚无稳定的正向差值。困难目标分层中的改善更明显，但闭爪桌面支撑仍未可靠识别。两组同框异真值不是同RGB异真值：已有静态图像中有可区分的支撑/抬升线索。**

这是原36根数据上的探索性诊断，C已经在上一轮使用。没有新采集、没有新编码特征、没有改变标签/可见性/划分/阈值；未读取D样本/特征/预测/评分，未做WM、定位器、SupCon或In实验。原36根及更早32根/8根试采保持封存。

## 完全相同的结构和执行条件

采集前已冻结的数据仍为24个A/B训练根、12个C开发根；训练920查询（160正/760负），开发470查询（60正/410负）。原194条不可见查询不进入三个对照；它们没有被改成负例，也没有重做选择。

本轮方案在任何新拟合之前冻结，SHA `a6afafd6836e40ba81d21c73c2745ad2ebcbca46c703fbb8c244f065e54cf22d`。完整协议见FROZEN_PROTOCOL.json。三个头全部Linear(261,33)→SiLU→Linear(33,1)，**8,680个参数、相同张量形状和每种子逐位相同初始化**。沿用原训练集pair_mean/std，先归一化再置零：全输入保留261维；仅几何将前256视觉维置零；仅RGB将末5几何维置零。未重新估计任一条件的归一化。RGB仍由原生正确框选取紧裁剪，因此“仅RGB”并不意味着不依赖GT框。

各条件使用逐位相同的原600×32训练索引；两种子0/1，AdamW lr0.001、weight_decay0.001、BCE正权重4.75、梯度clip1、float32和确定性CUDA、TF32关闭。每次恰600步，只取最后权重；阈值0.5，无早停、调参、种子筛选。共6次新拟合。原全模型未直接替代此次运行：重新运行的两个全输入权重以及train/dev预测均**逐位复现**原pair结果。

置零维度对应的首层权重只有weight decay变化，独立验证与600次衰减的偏差为0。隐藏宽度、fan-in初始化、优化器、张量大小的旧混杂已控制；输入置零自然使可利用的自由度减少，不能宣称所有条件有效函数空间相同。尤其本轮33隐藏单元几何条件不是上一轮5→1244→1的宽几何头；两者回答不同问题，不可把上一轮较高几何分数抹去。

## 固定阈值结果与不确定性

{overall}

P=TP/(TP+FP)，R=TP/(TP+FN)，BA=(正例召回+负例召回)/2；NLL为未加权二元负对数似然，概率裁剪[10⁻⁷,1−10⁻⁷]。分母为0记null。所有概率/错例在CSV，所有种子保留。

全输入相对几何置零RGB条件，FP从54/62降到20/18，增加5/4个漏报；精确率提高20.70/26.50个百分点，NLL降低0.1229/0.1502；两者整根95%差值区间均不跨0。BA差为−0.02/+2.03个百分点，两区间均跨0。可以说**图像在这个结构、预算和分布下提供额外证据**，不能说整体BA已显著提高或已学会可靠的物理支撑判断。完整RGB条件相对仅RGB的总体BA提高1.44个百分点，但区间下界为0，目标分层差值区间跨0。

采用原冻结的2,000次完整12根配对bootstrap，所有条件共用相同根重数，重算汇总混淆计数/NLL；不把同根状态、双相机或不同物体当独立样本。下表除NLL外均为百分点。12根、两个种子、已见C和未校正多重比较限制了推广性。

{paired}

## 困难目标与桌面支撑分层

开发470条中304条是非目标方块且全为负例，占410条负例的74.1%。原全量BA因此容易被大量非目标负例抬高。目标分层只有166条（60正/106负）；它不是新测试集。

该分层由协调线程在看到整体结果后要求追加，已单独保存posthoc_strata_addendum.json，明确标为**事后探索性分析**；没有据此改训练或阈值。目标=原root的目标颜色，不作为模型特征。困难支撑集合=目标颜色、原closed_on_support锚点且原y=0，共12条；不能用全部闭爪锚点替代，因为其中另有已脱离支撑的正例。

{target}

目标BA中全输入相对几何置零RGB提高10.93/12.70个百分点，根级95%区间分别[3.37,17.32]/[5.83,18.56]。这说明图像增益并非仅来自更容易的非目标负例。但12条真实桌面支撑负例中，几何条件12/12误报，全输入和仅RGB均9/12误报。该纯负集合的召回/BA没有定义，报告FP及FPR（100%或75%），不填虚假的BA。

## 两组同框样本：逐像素、接触来源和时间邻居

{alias_table}

只比较原保存图像，未生成、补帧或重新渲染。上述变化像素指任一RGB通道值不同；差异不自动等于可泛化语义线索。腕部方块裁剪为C24的17×32和C34的16×20像素。原5维几何逐位相同，但原256维视觉特征并不相同，L2距离分别14.16/12.20，余弦相似度0.859/0.919。mask IoU也小于1；框来自可见mask的整数极值，独立重算完全一致，未发现取框bug。

**物理标签来源。** 两根s03均有双指家族接触，同时与table uid5/base link−1有约78.58/71.21N的向上法向支撑，远高于0.005/0.01/0.015N三个阈值；s04起外部接触为0、双指接触仍在，原Holding变为1。接触法向方向、body/link、点和间隙已逐条保存；这两组外部接触确是向上桌面支撑，并非无关的侧向擦碰。s03轻微穿透约0.007–0.711mm是原物理求解接触记录，不是标签索引错位。一般实现使用“任何超过阈值的非机器人接触”作为排除条件，比自然语言的“已夹住”更严格。

原12控制步稳定窗口与三个力阈值全部复算一致；两根共{audit['physical_label_checks']}次物理标签核对。s03控制248/247后下一步249/248离开桌面，已有s04图像在321/320步，中心高度从约0.460升至0.595m；之后稳定和平移保持正例，回落接触桌面后变负。没有中间控制步RGB，不能凭记录制造未保存的视觉证据。两相机在每锚点同tick、前后完整状态相同，公开RGB与私有原生渲染逐位相同，未发现同步/接触阈值/标签缺陷。

**量化与表示省略。** 腕部相机随夹爪共同抬升，方块相对手指位置近似不变，整数边界进一步合并微小轮廓变化。C24投影方块中心从(41.877,21.875)到(41.884,21.869)像素；C34从(41.887,22.070)到(41.825,22.067)。因此同框主要反映共同运动下的相对几何不变和栅格化信息损失，不证明传感器看到了相同状态。

**真实图像已有何种线索。** 两个完整固定相机视图可见方块由桌面附近移动到上方，桌面/其他方块提供参照；完整腕部视图可见桌面、其他方块的比例及背景变化。模型的手指并集裁剪也保留了棕色桌面与浅色背景的变化，方块裁剪主要是单色面及少量边缘。因此不能说当前RGB裁剪完全没有线索，也不能把相同框当成信息论视觉歧义。人工检查这些明确抬升的对照可以区分场景；本轮没有训练全景模型，**尚未证明自动系统能从单张完整RGB稳定判断所有支撑状态**，更不能从像素差异推出接触力可被直接测得。

当前每次预测只用单相机的两个独立紧裁剪及相对框几何。它不接收整张图、桌面支撑区域的完整上下文、跨相机融合或时间历史；独立224缩放还丢失绝对像素尺寸与位置。跨时刻的静态画面线索存在，但当前表示/训练没有可靠利用。两根的8条同几何查询（2负6正）在全部6模型中都预测为正：P75%、R100%、BA50%；图像没有解决这两处错误，且视觉模型对支撑负例更过度自信。逐邻居双相机概率见alias_all_neighbor_predictions.csv。

固定相机的现有裁剪读出也提供了一个直接对照：C24的s03负例在全输入模型中概率0.093/0.048，s04正例0.974/0.927，两个种子均已区分；同一物理状态的腕部读出却都接近1。C34固定相机s03仍给0.937/0.926而误报，s04给0.999/0.999。可见已有单帧视觉输入确实能解决其中一例，但对视角和物体不稳；这既反对“视觉必然不可辨”的结论，也不足以证明完整支撑判别已经解决。

## Holding标签是否适合视觉任务

结论分开看：**原标签在这两组样本中是物理上自洽的“由双指承担、无外部接触”目标，保留它是合理的；不是标签错误导致冲突。** 如果规划中的Holding意指能随夹爪携带、已经脱离桌面，这个定义比仅看闭爪更有用；若原意只是“夹紧”，则名称语义更窄，应解释而不能悄悄改标签。本次未改变定义。

但“方块和两指可见”只证明端点存在，**不证明接触/承重关系对当前单帧裁剪充分可观测**。接触力是隐藏物理量，图像提供几何和场景线索；这两例恰有可见线索，不能据此宣布整个视觉任务不可辨。更准确的判断是：标签可以作为物理任务监督/评分，但当前裁剪输入不适合被当作已充分观测的确定性Holding符号来源。需要在原指南允许的视觉上下文/时间状态估计中验证或表达不确定性；本轮没有添加模型、改标签、接入规划或启动分支采集。

## 复核与文件

独立标量实现复核{verify['metric_rows_independently_recomputed']}行指标，展开复核{verify['expanded_root_bootstrap_checks']}次整根重采样；训练专用归一化、相同初始化/形状/批次、置零列仅衰减、全输入逐位复现全部通过。初次汇总负查询数出现整数按位取反的显示错误，已在评分层改为bool并独立复核；分类计数/P/R/BA/NLL/区间本来正确，训练、预测和数据未重复或改变。该记录在posthoc_strata_addendum.json。

协议与完整技术配置在configs，6最终模型在models，逐次预测/指标/接触诊断/复核在results，source保存实际源码。原数据通过路径引用和SHA校验复用，不复制修改。交付时再次核验原36根全部3,255个文件和更早封存清单，final_verification.json与delivery_manifest.json记录封存结果。

图1蓝/橙/绿为全输入/仅几何/仅RGB；圆/方为种子0/1，纵轴百分比、线段整根95%区间，分别显示全量和目标分层。图2展示全输入减仅几何的配对变化，0虚线表示无差异，NLL越负越好。图3橙线外部向上法向力、蓝线高度、绿阶梯原Holding，横轴真实控制步。两张RGB邻居图按原时间顺序显示固定相机、腕部、方块裁剪、手指并集裁剪；青框方块、橙框手指，所有图都有条件、分母与限制说明。
'''
 (R/'REPORT.md').write_text(txt)
 save(R/'OUTCOME.json',{'utc':now(),'matched_architecture_RGB_evidence':'precision and NLL improve in both seeds, overall BA pairedCI crosses0; posthoc target-only BA improves with pairedCI above0','holding_label_assessment':'physically consistent; no label defect found; geometry alias is not RGB ambiguity; current crop inputs do not establish sufficient observability; full static context has qualitative cues but no full-frame classifier tested','remaining_alias_errors':'all6 models false-positive both supported alias negatives','new_collection':0,'new_fits':6,'no_new_experiment_started':True,'entire_guide_complete':False})
 (R/'RUN_COMMANDS.md').write_text('运行环境：source /home/robot/research/projects/wm_symbolic/world_model_env.sh\n\n本机顺序运行prepare.py → train.py（单GPU独占gpu.lock）→ predict.py（CPU）→ score.py → inspect_alias.py → verify.py → report.py → seal.py。除train外均CUDA_VISIBLE_DEVICES空。无采集/渲染/编码器运行。score仅用固定预测重汇总一次以加入事后难例分层并修正负查询计数类型。\n\n源码目录：'+str(E/'source')+'\n')
 for src in Path(__file__).parent.glob('*.py'):
  dest=E/'source'/src.name
  if dest.exists():assert sha(dest)==sha(src),src
  else:shutil.copy2(src,dest)
 print('REPORT_WRITTEN',R/'REPORT.md')
if __name__=='__main__':main()
