from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=FONT.get_name();plt.rcParams['axes.unicode_minus']=False
COLORS=['#2466b5','#d47424'];NAMES={'all':'两类合并（micro）','drawer_front':'抽屉前板含把手','gripper_finger':'两根手指＋指尖'}
def get(summary,seed,split,cam='all',cat='all'):return next(q for q in summary if q['seed']==seed and q['split']==split and q['camera']==cam and q['category']==cat)
def pt(points,seed,split,t,cam='all',cat='all'):return next(q for q in points if q['seed']==seed and q['split']==split and q['camera']==cam and q['category']==cat and q['threshold']==t)
def draw_pr(ax,summary,points,split,cam,cat):
 for seed in [0,1]:
  q=get(summary,seed,split,cam,cat);d=np.load(E/f'data/pr_seed{seed}_{split}_{cam}_{cat}.npz');x=np.r_[0.,d['recall']];y=np.r_[1.,d['precision']];ax.plot(x,y,color=COLORS[seed],lw=1.35,label=f'S{seed} AP50={q["AP50_all_points"]:.3f}');ax.step(d['recall'],d['precision_envelope'],where='pre',color=COLORS[seed],lw=.7,ls=':',alpha=.8)
  for threshold,marker in [(.5,'o'),(.3,'^'),(.1,'s')]:
   p=pt(points,seed,split,threshold,cam,cat);ax.scatter(p['recall'],p['precision'],c=COLORS[seed],marker=marker,s=35,zorder=4,edgecolors='white',linewidths=.5)
 ax.set(xlim=(0,1.01),ylim=(0,1.025),xlabel='召回率：TP / 全部有效 GT',ylabel='精确率：TP / (TP+FP)');ax.grid(alpha=.18);ax.legend(fontsize=8,loc='lower left')
def pr_figures(summary,points):
 fig,axes=plt.subplots(2,3,figsize=(15,9))
 for ri,split in enumerate(['train','dev']):
  for ci,cat in enumerate(['all']+CATS):draw_pr(axes[ri,ci],summary,points,split,'all',cat);axes[ri,ci].set_title(('训练 A/B'if split=='train'else'开发 C')+'｜'+NAMES[cat],fontsize=12)
 fig.suptitle('完整候选 PR：训练分数偏保守，开发集还存在排序退化',fontsize=18,y=.995);fig.text(.5,.955,'两种子独立展示；固定同类 NMS IoU>0.5，一对一匹配 IoU≥0.5；无 GT 中心筛选、无候选数量上限。',ha='center',fontsize=10);fig.text(.5,.018,'蓝=S0，橙=S1；实线=实际 PR，点线=AP 的精确率包络；圆点=原阈值0.5，三角=0.3，方块=0.1。所有低阈值点仅为已见数据描述，不是新正式成绩。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(R/'figures/PR_overview.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(4,3,figsize=(15,16))
 for ri,(split,cam)in enumerate([(s,c)for s in ['train','dev']for c in ['static','gripper']]):
  for ci,cat in enumerate(['all']+CATS):draw_pr(axes[ri,ci],summary,points,split,cam,cat);axes[ri,ci].set_title(f'{split} / {"静态"if cam=="static"else"腕部"} / {NAMES[cat]}',fontsize=11)
 fig.suptitle('分相机完整候选 PR / AP50',fontsize=18,y=.995);fig.text(.5,.01,'轴、颜色、线型与总览一致。各面板分母是该划分/相机/类别的全部有效 GT；无法正确匹配的目标仍计入分母。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.03,1,.98]);fig.savefig(R/'figures/PR_by_camera.png',dpi=150);plt.close(fig)
def hist_panel(ax,q):
 centers=(np.array(q['hist_bins'][:-1])+q['hist_bins'][1:])/2;den=q['GT'];ax.bar(centers-.011,np.array(q['TP_hist_counts'])/den,width=.022,color='#279760',label='TP 候选 / 全部 GT');ax.bar(centers+.011,np.array(q['FP_hist_counts'])/den,width=.022,color='#d84e4b',alpha=.8,label='FP 候选 / 全部 GT');ax.bar(1.08,q['never_matched_GT']/den,width=.065,color='#828995',label='始终未匹配 GT / 全部 GT');ax.axvline(.5,color='#363c43',ls='--',lw=1);ax.set(yscale='log',ylim=(1/max(den,1)*.6,200),xlim=(-.02,1.145),xticks=[0,.25,.5,.75,1,1.08],xticklabels=['0','.25','.5','.75','1','未匹配'],xlabel='候选分数（每格 0.05） / 独立未匹配栏',ylabel='每个有效 GT 对应数量（对数轴）');ax.grid(axis='y',alpha=.15);ax.text(.02,.95,f'GT={den}；始终未匹配={q["never_matched_GT"]}\n全分数 TP={q["TP_at_all_scores"]}，FP={q["kept_candidates"]-q["TP_at_all_scores"]}',transform=ax.transAxes,va='top',fontsize=8)
def hist_figures(summary):
 fig,axes=plt.subplots(2,2,figsize=(14,9.2))
 for row,split in enumerate(['train','dev']):
  for seed in [0,1]:hist_panel(axes[row,seed],get(summary,seed,split));axes[row,seed].set_title(f'{"训练 A/B"if split=="train"else"开发 C"}｜种子 {seed}｜两类合并')
 h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.959),ncol=3,fontsize=10);fig.suptitle('候选分数分布：绿/红为完整正常解码后的正确/错误检出',fontsize=17,y=.995);fig.text(.5,.012,'每根柱都除以全部有效 GT。绿色各柱＋灰色未匹配栏之和=1；红色表示额外误检/GT，不要求和为1。虚线是原阈值0.5；未使用 GT 中心筛选候选。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.91]);fig.savefig(R/'figures/score_distributions.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(19,9))
 for row,split in enumerate(['train','dev']):
  for col,(cat,seed)in enumerate([(c,s)for c in CATS for s in [0,1]]):hist_panel(axes[row,col],get(summary,seed,split,cat=cat));axes[row,col].set_title(f'{split} / S{seed} / {NAMES[cat]}',fontsize=10)
 fig.suptitle('分类别候选分数分布（与总览相同定义和全 GT 分母）',fontsize=17);fig.text(.5,.01,'绿色 TP，红色 FP，灰色始终未匹配目标；虚线为原阈值0.5。阈值越低会引入更多 TP，也会引入更多 FP。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.03,1,.96]);fig.savefig(R/'figures/score_distributions_by_class.png',dpi=150);plt.close(fig)
def conflict_figure(audit):
 examples=audit['conflicts'];fig,axes=plt.subplots(2,5,figsize=(15,7));assert len(examples)==10
 for ax,q in zip(axes.ravel(),examples):
  boxes=np.array(q['boxes'],float)/[q['width'],q['height'],q['width'],q['height']];lo=boxes[:,:2].min(0)-.015;hi=boxes[:,2:].max(0)+.015
  for b,color in zip(boxes,['#00a7b5','#f19a20']):ax.add_patch(Rectangle(b[:2],b[2]-b[0],b[3]-b[1],edgecolor=color,fill=False,lw=2))
  ax.set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]),aspect='equal',title=f'{q["split"]} 图{q["image_id"]} / {q["camera"]}\n两 GT 框 IoU={q["IoU"]:.3f}',xlabel='归一化 x',ylabel='归一化 y');ax.tick_params(labelsize=7);ax.grid(alpha=.15)
 fig.suptitle('真实两指可见框与同类 NMS 的冲突：训练 8 图，开发 2 图',fontsize=16);fig.text(.5,.015,'青/橙分别是同一张图的两个真实手指可见框（不是预测，不叠 RGB）。原 NMS 对同类框 IoU>0.5 即抑制；即使两框精确，也不能同时保留。此处未改变 NMS。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(R/'figures/GT_overlap_NMS_conflicts.png',dpi=160);plt.close(fig)
def main():
 summary=read(E/'results/summary.json')['rows'];points=read(E/'results/fixed_threshold_points.json');audit=read(E/'results/slot_and_GT_overlap.json');oracle=read(E/'results/oracle_GT_center_train_only.json');pr_figures(summary,points);hist_figures(summary);conflict_figure(audit)
 # Macro AP is separate from pooled-candidate micro AP.
 macro=[{'seed':seed,'split':split,'camera':cam,'mean_of_two_class_AP50':sum(get(summary,seed,split,cam,cat)['AP50_all_points']for cat in CATS)/2,'micro_AP50':get(summary,seed,split,cam)['AP50_all_points']}for seed in [0,1]for split in ['train','dev']for cam in ['all','static','gripper']];save(E/'results/macro_vs_micro_AP.json',macro)
 lines=['# A/B 与 C 保存预测的完整 PR/AP、分数尺度诊断','', '**结论：训练端的低 F1(0.5) 主要不能解释成“表征或回归没学会”；高 AP 与低单点召回说明工作点过于保守。开发 C 另有明显的正常候选排序/定位联合质量下降，统一降阈值不能同时恢复高召回和高精度。**','', '本轮只读已保存 A/B 训练及 C 开发输出，没有新训练、模型推理、D 文件读取、校准拟合或正式阈值修改。旧报告和旧模型没有改写。下述低阈值结果全是已见数据的描述性诊断，不能称作未见性能或部署结果。','', '## 三种不同问题','', '|问题|证据|可支持的结论|不能推出的结论|','|---|---|---|---|','|固定 0.5 工作点|训练召回仅 31.43% / 36.71%，但完整候选 micro AP50 为 94.60% / 93.71%|训练候选整体有较好排序；大量正确候选分数低于 0.5|不能仅凭这个 F1 宣称表征未学会|','|降低工作点|描述性阈值 0.3：训练召回 84.14% / 88.29%，精确率 92.90% / 89.70%|低阈值能在训练端大幅恢复召回，保留较高精确率|这不是正式阈值选择，也没有检验新根泛化|','|C 的候选质量|micro AP50 降至 69.58% / 66.63%；0.3 时精确率仅 67.61% / 65.29%|开发候选的 TP/FP 分数重叠更重；单纯分数平移不能恢复类内 AP|不能据此锁定某一种损失就是原因，AP 同时受回归与 NMS 影响|','', '![PR总览](figures/PR_overview.png)','', '## 计算口径','', '从每张保存输出的 1024 patch × 3 槽开始，按原公式解码中心/宽高并裁剪到 [0,1]；只排除非正面积框，不设置分数下限、不做 GT 中心选择、不做局部峰值筛选、不设候选数上限。前板槽 0；手指槽 1/2 共同进入同类贪心 NMS（严格 IoU>0.5）。','', '候选按分数降序；同分按图号、patch、slot 固定顺序。一对一匹配同图同类未匹配 GT，取最大 IoU（并列沿用原评分规则），IoU≥0.5 为 TP，其余为 FP。未知图像类别整体屏蔽。召回分母始终包含该分层全部有效 GT，包含完整候选中从未被正确匹配的目标。','', '**AP50 定义：**单一 IoU=0.5 的全点插值 AP。每个不同分数阈值包含整个同分组，构造精确率包络 `p_interp(r)=max_{r′≥r}p(r′)`，对各召回增量求 `Σ Δr × p_interp(r)`；无法覆盖的召回范围贡献零。不是 11 点或 101 点近似，也不是 COCO 多 IoU mAP。','', '总体为把两类候选和 GT 汇总后的 micro AP；两类 AP 的算术平均另存 `macro_vs_micro_AP.json`，两者不可混用。根内状态相关，未把帧视为独立样本进行统计显著性检验。全文“恢复幅度大”是描述，不是显著性结论。','', '全分数 NMS 的任一分数前缀与先按该分数过滤再 NMS 等价，因为低分候选不能反向抑制高分候选。所有有效图像类别已以独立的原标量 NMS 实现核对 0.5 前缀；A/B 和 C 的原 0.5 总 TP/FP/FN 完全复现。','', '## 每类、每相机 AP50 与无法检出的目标','', '|种子|划分|相机|类别|GT 分母|AP50|全候选最大召回|始终未匹配 GT|','|---|---|---|---|---:|---:|---:|---:|']
 for q in summary:lines.append(f'|{q["seed"]}|{q["split"]}|{q["camera"]}|{q["category"]}|{q["GT"]}|{q["AP50_all_points"]:.4f}|{q["recall_ceiling_with_full_normal_decode"]:.4f}|{q["never_matched_GT"]}|')
 lines+=['','这里“全候选最大召回”是原解码器降至全分数后的实际一对一匹配结果，同时有大量 FP，不是可用工作点，也不是 oracle 中心指标。训练仍有 14/9 个 GT 永远未匹配；C 有 12/16 个。这些都保留在 AP/召回和分数图的分母中。','', '![分相机PR](figures/PR_by_camera.png)','', '## 固定描述性阈值对比：不挑最佳开发点','', '|种子|划分|分数阈值|TP|FP|FN|精确率|召回率|F1|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for q in points:
  if q['camera']=='all'and q['category']=='all'and q['threshold']in[.5,.4,.3,.2,.1]:lines.append(f'|{q["seed"]}|{q["split"]}|{q["threshold"]:.1f}|{q["TP"]}|{q["FP"]}|{q["FN"]}|{q["precision"]:.4f}|{q["recall"]:.4f}|{q["F1"]:.4f}|')
 lines+=['','0.5 行是原正式成绩，保持不变。其他行仅用于观察取舍；更低 0.05/0.01/0.0 以及所有分类/相机分层见 `threshold_points.csv`。没有根据 C 选择新阈值。','', '在“精确率至少 90%”这个事先列明的描述性条件下，训练全体最大召回为 89.71% / 86.43%，而 C 只有 25.71% / 10.17%。这些是已见曲线上的边界点，不是新的泛化保证。前板和手指不能混为一谈：C 前板 AP50 87.37% / 88.24%，手指仅 65.59% / 62.99%；类内排序落差不能被全局单调校准修复。','', '## 正误检分数分布','', '绿色 TP 和红色 FP 都来自完整正常 NMS＋匹配后的候选，不是只查看 GT 中心。每个分数区间数量除以全部有效 GT；绿色各柱之和加灰色“始终未匹配”栏等于 1。红色是额外 FP/GT，不要求加总为 1。对数纵轴用于同时展示低分背景候选与少量高分候选。','', '![分数分布](figures/score_distributions.png)','', '![分类分数分布](figures/score_distributions_by_class.png)','', '## 与 oracle 中心诊断严格分开','', '|种子|训练真实中心|该中心指定槽的框 IoU≥0.5|所有中心分数<0.5|正确回归且分数<0.5|','|---|---:|---:|---:|---:|']
 for q in oracle:lines.append(f'|{q["seed"]}|{q["GT_centers"]}|{q["assigned_slot_box_IoU_ge05"]}|{q["assigned_slot_score_lt05"]}|{q["adequate_box_and_score_lt05"]}|')
 lines+=['','470/433 是“回归已达 IoU≥0.5 且分数低于 0.5”的子集；所有真实中心低分数量实际上是 475/436。693/700 只说明在已知目标中心处能回归出好框，不含搜索错误中心、误检排序和 NMS，因此绝不能作为正常检测召回或 AP。','', '## 第二手指槽与 NMS 冲突','', '|训练输出槽|正中心|覆盖图|覆盖根|有效图×patch 槽位|','|---|---:|---:|---:|---:|']
 for slot,label in [('0','前板槽0'),('1','手指默认槽1'),('2','同patch第二手指槽2')]:
  q=audit['slots'][slot];lines.append(f'|{label}|{q["positive_centers"]}|{q["positive_images"]}|{q["positive_roots"]}|{q["valid_image_patch_positions"]}|')
 lines+=['','槽2只在同一 patch 出现两个手指中心时使用，不是固定表示左指或右指。其正例 11 个，分布于 11 图、9 根，占 510 个手指正中心的 2.16%；相对 261,120 个有效图×patch 槽位仅约 0.0042%。这是明确稀少性，不能单独证明本轮漏检都由它造成。','', '训练有 8 图、C 有 2 图的两个真实手指可见框 IoU>0.5。即使两个 GT 框都被完美预测，原同类 NMS 也会抑制其中一个。这揭示规则与重叠实例目标的冲突，但不会自动解释所有 FN；实际预测框有误差，未对它们进行替代 NMS 实验。','', '![GT框与NMS冲突](figures/GT_overlap_NMS_conflicts.png)','', '## 下一步只改变一个因素','', '|候选下一步|本轮支持与限制|建议|','|---|---|---|','|训练集内单调分数校准|适合纠正训练工作点偏保守；不改类内排序/AP。现有 A/B 都参与过头训练，其校准分数仍属拟合内诊断；C 的手指 AP 落差仍在|可作工作点解释，但不应当作开发排序问题的修复，也不宣称当前0.3是选好的阈值|','|仅改分类目标的固定对照|训练排序已较好，C 正误候选仍重叠；最直接研究分类分数对“可正确定位”的排序。现有证据没有证明具体哪一种损失一定有效|**若目标是继续可靠定位，优先此单因素对照**：冻结 DINO、头结构、回归损失、数据、初始化种子、1200步预算、优化器、NMS及原阈值；只换预先明确的一种分类目标，用 AP50 与固定工作点指标一起比较|','|仅延长原训练|原训练 AP 已约0.94，GT中心回归已693/700；只有最终快照，未测AP随训练时间的变化|缺少“延长就能修复C排序”的证据，不作为当前首选；也不能断言延长绝无用处|','', '建议的“分类目标”是一项待另行预注册的实验因素，本轮没有更换损失，也没有据此启动训练。若采取该对照，不同时做校准、延长步数、更改 NMS 或增设槽位；这些都应保持独立。C 已被反复描述性使用，后续 C 结果仍是开发诊断；本轮未触碰 D，不能据此更新任何 D 结论。','', 'AP 下滑是检测候选的排序/定位联合证据，可能包含框质量、域差异与 NMS 的影响；“优先分类目标对照”是基于现有结果的实验选择，不是已证实的病因。','', '## 文件与访问边界','', f'- 旧实验目录只读：`{BASE}`。只允许 773 个明确列出的 A/B/C 文件；不读取全划分混合标签、混合预测或旧指标文件。',f'- 新独立实验：`{E}`；报告：`{R}`。','- `data/candidates_seed*_train/dev.npz` 保存完整 NMS 后候选、TP/FP、匹配与分数；`data/pr_*.npz` 保存所有阈值点。`ap_summary.csv` 与 `threshold_points.csv` 可直接复查。','- `results/access_and_checks.json` 保存允许路径访问账本与检查；`configs/input_frozen.json` 保存原输入哈希，结束时全部复核相同。`configs/protocol.json` 说明口径。','- AP 漏检分母、同分组、精确率包络经解析例子检查；分数图 TP＋未匹配分母检查通过；原 0.5 NMS 前缀及 AB/C 总计数回归检查通过。','- 本轮没有下载、安装、GPU计算、渲染、训练、模型推理或模型权重读取。旧模型与正式0.5成绩原样保留。']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n');save(E/'results/report_assets.json',{'utc':now(),'figures':[str(p)for p in(R/'figures').glob('*.png')],'old_evaluation_unchanged':True,'next_step_recommendation':'single classification-objective contrast for reliability; calibration explains train operating point but cannot repair observed classwise C AP loss; no run started'});print('PR_REPORT_READY',R,flush=True)
if __name__=='__main__':main()
