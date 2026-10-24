from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=FONT.get_name();plt.rcParams['axes.unicode_minus']=False
COLORS=['#2466b5','#d77a29'];CN={'all':'两类合并 micro','drawer_front':'抽屉前板含把手','gripper_finger':'两根手指＋指尖'}
def get(rows,seed,split,cam,cat):return next(r for r in rows if r['seed']==seed and r['split']==split and r['camera']==cam and r['category']==cat)
def draw_pr(ax,seed,split,cam,cat,base,new,bp,npnt):
 for index,(name,data,points,folder)in enumerate([('focal γ=2',base,bp,E/'data/baseline'),('同权重 BCE γ=0',new,npnt,E/'data/evaluation')]):
  q=get(data,seed,split,cam,cat);z=np.load(folder/f'pr_seed{seed}_{split}_{cam}_{cat}.npz');ax.plot(np.r_[0.,z['recall']],np.r_[1.,z['precision']],color=COLORS[index],lw=1.4,label=f'{name} AP={q["AP50_all_points"]:.3f}');ax.step(z['recall'],z['precision_envelope'],where='pre',color=COLORS[index],ls=':',lw=.7,alpha=.8)
  for threshold,marker in [(.5,'o'),(.3,'^'),(.1,'s')]:
   p=next(v for v in points if v['seed']==seed and v['split']==split and v['camera']==cam and v['category']==cat and v['threshold']==threshold)
   if p['precision']is not None:ax.scatter(p['recall'],p['precision'],color=COLORS[index],marker=marker,s=35,edgecolors='white',linewidths=.5,zorder=4)
 ax.set(xlim=(0,1.01),ylim=(0,1.025),xlabel='召回率：TP / 全部有效 GT',ylabel='精确率：TP / (TP+FP)');ax.grid(alpha=.18);ax.legend(fontsize=8,loc='lower left')
def curves(base,new,bp,npnt):
 for split in ['train','dev']:
  fig,axes=plt.subplots(2,3,figsize=(15,9))
  for seed in [0,1]:
   for col,cat in enumerate(['all']+CATS):draw_pr(axes[seed,col],seed,split,'all',cat,base,new,bp,npnt);axes[seed,col].set_title(f'种子 {seed}｜{CN[cat]}')
  fig.suptitle(('训练 A/B'if split=='train'else'开发 C')+'｜只取消 focal 调制，完整候选 PR 对照',fontsize=17,y=.995);fig.text(.5,.955,'蓝色=原 focal γ=2，橙色=同权重 BCE γ=0；两模型同缓存、初始化、batch日程、1200步与NMS。',ha='center',fontsize=10);fig.text(.5,.012,'实线=实际 PR，点线=AP 精确率包络；圆点=原0.5工作点，三角=描述性0.3，方块=描述性0.1。匹配IoU≥0.5；全GT分母。未选择新阈值。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(R/'figures'/f'PR_{split}_comparison.png',dpi=160);plt.close(fig)
 for seed in [0,1]:
  fig,axes=plt.subplots(4,3,figsize=(15,16))
  for row,(split,cam)in enumerate([(s,c)for s in ['train','dev']for c in ['static','gripper']]):
   for col,cat in enumerate(['all']+CATS):draw_pr(axes[row,col],seed,split,cam,cat,base,new,bp,npnt);axes[row,col].set_title(f'{split} / {"静态"if cam=="static"else"腕部"} / {CN[cat]}',fontsize=11)
  fig.suptitle(f'种子 {seed}｜分相机完整候选 PR 对照',fontsize=17);fig.text(.5,.01,'轴、颜色、线型和标记与总览一致；未知图像类别屏蔽，其余所有GT含未检出目标均在分母中。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.03,1,.97]);fig.savefig(R/'figures'/f'PR_by_camera_seed{seed}.png',dpi=150);plt.close(fig)
def training_plots():
 fig,axes=plt.subplots(1,3,figsize=(15,4.9))
 for seed,style in [(0,'-'),(1,'--')]:
  for variant,color in [('focal',COLORS[0]),('BCE',COLORS[1])]:
   d=read(E/f'data/baseline/train_seed{seed}.json')if variant=='focal'else read(E/f'results/train_seed{seed}.json');h=d['history'];h=[r for r in h if r['update']==1 or r['update']%100==0];steps=[r['update']for r in h];vals=[[r['total']for r in h],[r['focal']if variant=='focal'else r['classification']for r in h],[5*r['smoothL1']+2*r['GIoU']for r in h]]
   for ax,v in zip(axes,vals):ax.plot(steps,v,color=color,ls=style,marker='.',label=f'{variant} S{seed}')
 for ax,title in zip(axes,['训练批总损失','分类项（目标不同，值不直接排名）','相同公式的加权回归项']):ax.set(xlabel='优化更新次数',ylabel='损失（对数轴）',yscale='log',title=title);ax.grid(alpha=.18);ax.legend(fontsize=8)
 fig.suptitle('固定预算训练记录：蓝=focal，橙=BCE；实线=S0，虚线=S1',fontsize=15);fig.text(.5,.015,'图示与旧基线相同的13个更新点；新BCE的1200步完整日志另存。回归项=5×SmoothL1+2×GIoU。损失尺度变化属于取消调制的结果。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(R/'figures/training_losses.png',dpi=160);plt.close(fig)
 probes=read(E/'results/gradient_probes.json')['rows'];fig,axes=plt.subplots(1,3,figsize=(16,5.2))
 for seed in [0,1]:
  h=read(E/f'results/train_seed{seed}.json')['history'];x=np.array([r['update']for r in h]);g=np.array([r['preclip_global_gradient_norm']for r in h]);axes[0].plot(x,g,color=COLORS[seed],alpha=.18,lw=.6);med=np.array([np.median(g[max(0,i-24):min(len(g),i+26)])for i in range(len(g))]);axes[0].plot(x,med,color=COLORS[seed],lw=1.5,label=f'BCE S{seed}：50步滑动中位数')
 axes[0].axhline(1,color='#333333',ls='--',label='原裁剪上限1');axes[0].set(yscale='log',xlabel='更新次数',ylabel='裁剪前全头梯度 L2 范数（对数轴）',title='仅新BCE有逐步梯度历史');axes[0].legend(fontsize=7);axes[0].grid(alpha=.16)
 for ax,stage in zip(axes[1:],['shared_initial','final']):
  labels=[];cc=[];rr=[]
  for seed in [0,1]:
   for gamma in [2,0]:
    checkpoint=stage if stage=='shared_initial'else('focal_final'if gamma==2 else'BCE_final');q=next(q for q in probes if q['seed']==seed and q['checkpoint']==checkpoint and q['gamma']==gamma and q['batch']=='root_balanced');labels.append(f'S{seed} γ{gamma}');cc.append(q['classification_gradient_norm']);rr.append(q['weighted_regression_gradient_norm'])
  x=np.arange(4);ax.bar(x-.17,cc,.34,color='#bb4a4a',label='分类梯度');ax.bar(x+.17,rr,.34,color='#388998',label='加权回归梯度');ax.set(xticks=x,xticklabels=labels,yscale='log',ylabel='全头梯度 L2 范数（对数轴）',title='同一初始权重：只改损失'if stage=='shared_initial'else'各自最终权重：无更新探测');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.16)
 fig.suptitle('梯度诊断：原基线没有历史梯度日志；柱图是固定批次重新求导，零优化更新',fontsize=14);fig.text(.5,.015,'左：浅线为每步值，粗线为50步中位数；蓝=S0、橙=S1。中/右：红=分类、青=加权回归，使用预先固定的16根均衡训练批次。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.055,1,.93]);fig.savefig(R/'figures/gradient_diagnostics.png',dpi=160);plt.close(fig)
def fmt(v):return '—'if v is None else f'{v:.4f}'
def main():
 install_guard([]);base=read(E/'data/baseline/summary.json')['rows'];new=read(E/'results/evaluation_summary.json')['rows'];bp=read(E/'data/baseline/fixed_threshold_points.json');npnt=read(E/'results/fixed_threshold_points.json');comp=read(E/'results/comparison.json');directions=read(E/'results/seed_consistency.json');probes=read(E/'results/gradient_probes.json')['rows'];curves(base,new,bp,npnt);training_plots();pre=read(E/'results/preflight.json');train=[read(E/f'results/train_seed{s}.json')for s in [0,1]]
 lines=['# 仅取消 focal 调制：gamma=0 同权重 BCE 对照','', '**本轮没有形成总体一致的提升，不能整体采用 BCE。** C 的总体 AP50 在种子0/1上分别变化 −0.29 / +5.13 个百分点；前板 AP50 两种子一致提升 +7.10 / +6.87 个百分点，手指为 −2.69 / +4.65 个百分点。训练总体 AP50 两种子均下降。','', '这是当前固定数据、结构、初始化及1200更新预算下的对照，不证明任何损失普遍更优。两个种子全部报告；没有凭较好种子替换模型，也没有部署。','', '## 唯一实验变量与可比性','', '|项目|原 focal|gamma=0 对照|为什么保持或改变|','|---|---|---|---|','|正例分类权重|0.25×(1−p)²|0.25|仅移除调制项|','|负例分类权重|0.75×p²|0.75|类别权重不变，不是无权重BCE|','|分类归一化|每图 patch/槽加总，除以 max(该图正中心数,1)，再取批均值|完全相同|未同时改变正负样本预算或损失缩放|','|回归|5×SmoothL1 + 2×GIoU|完全相同|相同正中心、框目标与代码|','|RGB输入与结构|冻结DINOv2-small，448输入，32×32 patch，共享384→64→15头|同一缓存、归一化与25615参数头|无GT、深度、根号或相机标签网络输入|','|训练|种子0/1，各1200步，batch16，AdamW lr0.001/wd0.0001，clip1|原初始权重和1200×16日程逐字节复用|每张训练图仍出现75次；无增强、AMP或新数据|','|解码与评分|同类NMS IoU>0.5；完整候选；一对一IoU≥0.5|完全相同|未改NMS、slot、有效mask或阈值；GT中心不参与检测评分|','', f'开训前核验 {pre["old_sealed_selected_inputs_verified"]} 个旧输入与封存哈希一致；运行时 Torch/Numpy 版本一致。源码核验表明 `net.py` 相对旧版本只有权重表达式一处改变。回归项逐位相等、gamma0等于所述加权BCE、未知类别零梯度等检查通过。','', '取消调制会自然改变分类损失/梯度的尺度与样本间相对贡献。该后果属于本次因素，没有补偿性调整损失系数、学习率、裁剪或训练时长。','', '## 完整候选 AP50：先看两个种子的方向','', 'AP50 为 IoU=0.5 下全点插值精确率包络积分，所有有效GT包括从未匹配目标都在分母中。总体为两类 micro 汇总；不是COCO多IoU mAP，也不是GT中心oracle。C仅开发诊断，不是本轮未见测试。','', '|种子|划分|类别|原focal AP50|BCE AP50|变化（百分点）|','|---|---|---|---:|---:|---:|']
 for q in comp:
  if q['camera']=='all':lines.append(f'|{q["seed"]}|{q["split"]}|{q["category"]}|{q["focal_AP50"]:.4f}|{q["BCE_AP50"]:.4f}|{100*q["delta_AP50"]:+.2f}|')
 lines+=['','![C PR对照](figures/PR_dev_comparison.png)','', '![训练PR对照](figures/PR_train_comparison.png)','', 'C 前板的提升也在每个相机、每个种子上同向；C 手指在种子0两个相机均下降，在种子1两个相机均上升。两个种子不足以作统计显著性声明，尤其根内状态相关。','', '|种子|C相机|类别|focal AP50|BCE AP50|变化（百分点）|','|---|---|---|---:|---:|---:|']
 for q in comp:
  if q['split']=='dev'and q['camera']!='all':lines.append(f'|{q["seed"]}|{q["camera"]}|{q["category"]}|{q["focal_AP50"]:.4f}|{q["BCE_AP50"]:.4f}|{100*q["delta_AP50"]:+.2f}|')
 lines+=['','全部训练/开发、分类别/相机的36对结果在 `comparison.csv`；原始PR阈值点保留于实验数据目录。','', '![种子0分相机](figures/PR_by_camera_seed0.png)','', '![种子1分相机](figures/PR_by_camera_seed1.png)','', '## 原0.5工作点保持不变','', '|种子|划分|类别|focal 精确率/召回率|BCE 精确率/召回率|focal TP/FP/FN|BCE TP/FP/FN|','|---|---|---|---|---|---|---|']
 for q in comp:
  if q['camera']=='all':lines.append(f'|{q["seed"]}|{q["split"]}|{q["category"]}|{fmt(q["focal_precision05"])}/{fmt(q["focal_recall05"])}|{fmt(q["BCE_precision05"])}/{fmt(q["BCE_recall05"])}|{q["focal_TP05"]}/{q["focal_FP05"]}/{q["focal_FN05"]}|{q["BCE_TP05"]}/{q["BCE_FP05"]}/{q["BCE_FN05"]}|')
 lines+=['','原focal的正式分数文件未改。图中0.3/0.1仅为与前次诊断一致的预列描述性标记，没有优化、选择或采用新阈值。','', '## 损失、梯度与优化后果','', '![训练损失](figures/training_losses.png)','', '旧focal仅保存13个批次损失快照，未保存历史梯度；新BCE保存全部1200步。两者分类目标不同，不能因某方损失数值较小就判更优。加权回归项公式相同，可观察其训练过程。','', '![梯度诊断](figures/gradient_diagnostics.png)','', '下表是在预先固定的16根均衡训练批次、同一初始权重上重算的无更新梯度。它隔离了损失变更的即时影响，不是额外训练。','', '|种子|调制gamma|分类梯度范数|加权回归梯度范数|总梯度范数|clip1隐含乘数|','|---|---:|---:|---:|---:|---:|']
 for q in probes:
  if q['checkpoint']=='shared_initial'and q['batch']=='root_balanced':lines.append(f'|{q["seed"]}|{q["gamma"]}|{q["classification_gradient_norm"]:.4f}|{q["weighted_regression_gradient_norm"]:.4f}|{q["total_preclip_gradient_norm"]:.4f}|{q["implied_clip_multiplier"]:.5f}|')
 lines+=['','相同初始批次中，BCE 分类梯度约为 focal 的34.1倍（S0）/27.6倍（S1），而回归梯度不变；裁剪后的整体更新方向/尺度因此受到影响。这是观测到的优化机制差异，不能单独证明它造成全部最终AP变化。各自最终模型与稀少槽强调批次的完整梯度探测在 `gradient_probes.json`；所有探测都验证参数哈希未变。','', '|新运行|更新数|梯度非有限次数|发生clip的更新|裁剪前梯度最小/中位/最大|','|---|---:|---:|---:|---|']
 for q in train:
  g=[r['preclip_global_gradient_norm']for r in q['history']];lines.append(f'|BCE S{q["seed"]}|{q["updates"]}|{q["nonfinite_gradients"]}|{q["clipped_updates"]}|{min(g):.3f}/{np.median(g):.3f}/{max(g):.3f}|')
 lines+=['','没有用这份事后梯度结果回头修改裁剪阈值、学习率或分类权重。基线没有逐步梯度日志，不能声称新运行“比旧运行更经常裁剪”；只能比较同一固定批次上的探测。','', '## 稀少碰撞槽覆盖保持一致','', '训练标签仍为前板槽190个正中心、手指默认槽499个、同patch第二手指槽11个。第二手指槽覆盖11图/9根，不是固定的左指或右指；其正例只占510个手指中心的2.16%。','', '|种子|前板正曝光|默认手指正曝光|碰撞槽正曝光|含碰撞正例的batch|每张图重复次数|','|---|---:|---:|---:|---:|---:|']
 for q in train:lines.append(f'|{q["seed"]}|{q["positive_exposures_by_slot"][0]}|{q["positive_exposures_by_slot"][1]}|{q["positive_exposures_by_slot"][2]}|{q["batches_containing_collision_positive"]}|75|')
 lines+=['','这些实际曝光与原保存日程预先计算的数值一致。没有增加稀少槽样本权重或重采样。原已识别的训练8图/C2图两指框IoU>0.5与NMS冲突仍保留，本轮没有修改它。','', '## 结论与后续边界','', '本轮只能支持“前板在当前条件下取消调制带来双种子同向的开发AP提升；手指和总体未形成一致收益”。不能整体采用BCE，不能将种子1的较好结果当作定论，更不能宣称损失普遍优越。原focal模型与本轮两模型均保留。','', '最新持续推进授权允许另行提出清楚的有界实验；本轮自身仍严格只新增两个1200步运行。具体下一步建议已写入 `NEXT_EXPERIMENT_PROPOSAL.md`：只对前板取消调制，手指保留原focal，以检验是否能保留前板一致收益而不扰动手指；它仍需作为独立协议运行，本轮尚未启动该方案。','', '## 文件、访问与复核','', f'- 本轮实验：`{E}`；报告：`{R}`。','- 原训练数据、RGB特征缓存、初始化、batch日程和选择的旧模型/基线曲线先做封存哈希核验；模型纯RGB特征输入。真值只进入监督、评分或无更新梯度探测。','- 训练进程只能打开旧 train.npz；推理进程只能打开旧 train/dev 特征；新预测先保存并哈希，再读取ABC标签评分。旧D样本、特征、预测和评分文件均未打开；旧封存清单中的哈希元数据仅用于选择输入的完整性核验。','- 新两个训练run均完成1200步；没有额外fit、延长、阈值搜索、超参搜索、数据扩增、下载、安装或部署。GPU串行持锁。','- 基线复用其已封存ABC正常候选AP/PR；未重新训练或重算其全量预测。旧基线最终权重仅用于无更新梯度探测，哈希保持不变。','- 运行命令、访问账本和最终验证分别见 `RUN_COMMANDS.md`、实验 `results/*frozen.json` / `gradient_probes.json` / `final_verification.json`。']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n')
 proposal='''# 下一有界实验提案：仅前板取消 focal 调制

本文件是协调用的具体提案，不是已执行结果。当前 gamma0 对照已结束，新增训练预算已用完；没有在该对照内追加第三方案。

证据：C 前板 AP50 在两个种子分别提高 7.10 / 6.87 个百分点，且静态/腕部相机四个配对方向都向上；手指两种子方向不一致，总体也不一致。

假设：在同一共享头中，仅前板改为 gamma=0、手指保留 gamma=2，可能保留前板收益并避免对手指分类目标的直接改动。共享隐藏层会产生耦合，因此不能把两个独立结果简单拼接或假定一定成功。

唯一变量：相对原focal，只有前板slot0的调制因子从原值变为1；手指slot1/2的alpha、gamma、有效mask、正中心与归一化均原样。完整BCE作为已有参考，不再重训。

预算：最多两个新run，种子0/1从原保存初始权重开始，各1200更新；batch16及原日程、缓存特征、回归损失、学习率、裁剪、分辨率、NMS等全部固定。无新采集、阈值选择或架构改变。结束即报告，不能自动扩大这一提案的预算。

评价边界：仅训练A/B与开发C；D不读。主看C整体与两类AP50、分相机AP，以及原0.5工作点P/R。报告两seed所有配对变化，前板的既有开发收益属于选择该假设的依据，故C结果只算开发诊断，不算新的未见验证。GT中心oracle不代替正常候选评分。

结果判读：要同时查看两种子的前板收益是否保留、手指是否受共享层影响以及总体方向；不采用单seed最好结果，不因一项提升而隐去其他类别退步。当前不能承诺成功，也不预设损失普遍优劣。

持续推进授权来源（原样转交）：
<transcript_evidence>
用户2026-10-18 14:03:27 UTC，消息 Sentinel_9dd2d7c1a5fc8191a759ea02da2c724a：还是老样子，如果有明确的实验方向就继续，把能做的工作做完
</transcript_evidence>
此授权不替代新的安全权限，也不允许绕过工具拒绝。
'''
 (R/'NEXT_EXPERIMENT_PROPOSAL.md').write_text(proposal);save(E/'results/report_assets.json',{'utc':now(),'figures':[str(p)for p in(R/'figures').glob('*.png')],'overall_adoption':False,'next_proposal_executed':False});print('GAMMA0_REPORT_READY',R,flush=True)
if __name__=='__main__':main()
