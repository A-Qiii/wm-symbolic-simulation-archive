from common import *
import shutil,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
COL={'fixed0.5':'#2466b5','calibrated':'#d77a29'};CN={'all':'总体 micro','drawer_front':'抽屉前板含把手','gripper_finger':'两根手指及指尖'}
def get(rows,s,cam,c,op):return next(q for q in rows if q['seed']==s and q['camera']==cam and q['category']==c and q['workpoint']==op)
def intervaltext(m,ci):return f'{100*m:.1f}% [{100*ci["lo"]:.1f}, {100*ci["hi"]:.1f}]'
def figures(rows,roots,cal,coverage):
 fig,axes=plt.subplots(2,3,figsize=(15,9))
 for s in [0,1]:
  for j,c in enumerate(['all']+CATS):
   ax=axes[s,j]
   for k,op in enumerate(['fixed0.5','calibrated']):
    q=get(rows,s,'all',c,op);keys=['precision','recall','F1'];values=[q[m]for m in keys];xx=np.arange(3)+(k-.5)*.32;lo=[q['CI95'][m]['lo']for m in keys];hi=[q['CI95'][m]['hi']for m in keys];ax.bar(xx,values,.32,color=COL[op],label='固定0.5'if k==0 else'旧C分类别校准');ax.vlines(xx,lo,hi,color='#333333',lw=1);ax.hlines(lo,xx-.04,xx+.04,color='#333333',lw=1);ax.hlines(hi,xx-.04,xx+.04,color='#333333',lw=1)
   ax.hlines(.9,-.36,.36,color='#aa3030',ls='--',lw=1.4,label='精确率目标0.90');ax.set(xticks=[0,1,2],xticklabels=['精确率P','召回率R','F1'],ylim=(0,1.05),ylabel='比例（0—1）',title=f'S{s}｜{CN[c]}｜完整AP50={q["AP50"]:.3f}');ax.grid(axis='y',alpha=.17);ax.legend(fontsize=8,loc='upper right')
 fig.suptitle('16个新C根：锁定原focal，对比固定与开发校准工作点',fontsize=17,y=.995);fig.text(.5,.951,'蓝=固定0.5；橙=旧C每类精确率≥0.90时召回率最大的阈值。两种子分别校准；新根从未参与阈值选择。',ha='center',fontsize=10);fig.text(.5,.014,'误差线=2000次整根配对bootstrap的95%百分位区间；不是逐帧独立样本。红线仅约束精确率P。AP使用全候选，两个工作点共用。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(R/'figures/new_roots_operating_points.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(15,8.8))
 for s in [0,1]:
  for j,c in enumerate(['all']+CATS):
   ax=axes[s,j];z=np.load(E/f'data/evaluation/pr_seed{s}_all_{c}.npz');ax.plot(np.r_[0,z['recall']],np.r_[1,z['precision']],color='#6b6b6b',lw=1,label='统一分数阈值PR');ax.step(z['recall'],z['precision_envelope'],where='pre',color='#999999',ls=':',lw=.6)
   for op,mark in [('fixed0.5','o'),('calibrated','^')]:q=get(rows,s,'all',c,op);ax.scatter(q['recall'],q['precision'],color=COL[op],marker=mark,s=60,label='固定0.5'if op=='fixed0.5'else'分类别校准',zorder=4)
   ax.axhline(.9,color='#aa3030',ls='--',lw=.7);ax.set(xlabel='召回率：TP/全部有效GT',ylabel='精确率：TP/(TP+FP)',xlim=(0,1.01),ylim=(0,1.025),title=f'S{s}｜{CN[c]}');ax.grid(alpha=.17);ax.legend(fontsize=8,loc='lower left')
 fig.suptitle('正常完整候选PR与两种工作点：模型、分数和NMS均未改变',fontsize=16);fig.text(.5,.015,'灰实线=PR、灰点线=AP包络；蓝圆点=0.5、橙三角=校准。总体分类别校准点未必位于统一阈值PR线上，不能据此重算或替代完整AP。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.045,1,.95]);fig.savefig(R/'figures/new_roots_PR.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(16,8.8))
 for s in [0,1]:
  for j,c in enumerate(['all']+CATS):
   ax=axes[s,j]
   for op in ['fixed0.5','calibrated']:
    q=sorted([r for r in roots if r['seed']==s and r['camera']=='all'and r['category']==c and r['workpoint']==op],key=lambda r:r['root_id']);ax.plot(range(16),[r['recall']for r in q],marker='o',ms=3,color=COL[op],lw=1,label='固定0.5'if op=='fixed0.5'else'分类别校准')
   ax.set(xticks=range(16),xticklabels=[f'{i:02d}'for i in range(16)],xlabel='新根编号（同根8状态、双相机）',ylabel='该根有效GT召回率',ylim=(0,1.02),title=f'S{s}｜{CN[c]}');ax.grid(alpha=.17);ax.legend(fontsize=8)
 fig.suptitle('逐根召回率：展示全部16根，连线仅辅助比较编号，不表示时间趋势',fontsize=15);fig.text(.5,.015,'蓝=固定0.5；橙=校准。各根分母采用其有效GT；未知图像类别不当negative，根级样本量详见root_metrics.csv。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.045,1,.95]);fig.savefig(R/'figures/root_recall_distribution.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,5.7));x=np.arange(4);known=[q['known_images']for q in coverage];unknown=[q['unknown_images']for q in coverage];ax.bar(x,known,color='#359278',label='有效类别图像');ax.bar(x,unknown,bottom=known,color='#b7bdc7',label='UNKNOWN：整个图像类别不评分');ax.set(xticks=x,xticklabels=[('静态'if q['camera']=='static'else'腕部')+' / '+('前板'if q['category']=='drawer_front'else'手指')for q in coverage],ylabel='图像数量（每相机128）',ylim=(0,145));ax.legend();ax.grid(axis='y',alpha=.17)
 for i,q in enumerate(coverage):ax.text(i,135,f'有效框{q["usable_positive_boxes"]} / {q["positive_roots"]}根',ha='center',fontsize=10);ax.text(i,known[i]/2,str(known[i]),ha='center',color='white');ax.text(i,known[i]+unknown[i]/2,str(unknown[i]),ha='center',fontsize=9)
 fig.suptitle('新根有效覆盖：49个前板图像类别、2个手指图像类别保持UNKNOWN',fontsize=14);fig.text(.5,.02,'绿=可评分；灰=严格排除。手指每个有效图像有2个GT框。零像素不推断不存在；本轮没有整图“确实不存在”的负例。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(R/'figures/unknown_coverage.png',dpi=160);plt.close(fig)
def main():
 install_guard([]);rows=read(E/'results/evaluation_summary.json')['rows'];roots=read(E/'results/root_metrics.json');cal=read(E/'configs/calibration_locked.json')['rows'];coverage=read(E/'results/coverage.json');figures(rows,roots,cal,coverage);lines=['# 定位阶段结论：固定0.5与旧C校准工作点的新根验证','', '**开发校准在旧C达到目标，但未在本次新根的四个类别/种子点估计上维持90%精确率。定位阶段到此封存，不再调模型或阈值。** 前板召回明显增加，手指高精确率工作点仍只有约11%—12%召回；当前定位不能作为已可靠的关系读取基础。区间跨过90%意味着样本有限，不能证明总体精确率必然低于90%，也不能据此宣称目标已验证。','', '测试为16个此前未用的C场景根，128状态/256主图。同一C场景已见，结论只针对同场景未见初始化/交互配置，不是跨场景泛化。旧C只用于校准，新根没有参与阈值/模型选择；两个原focal种子都保留，无重训、无挑seed。','', '## 校准规则与锁定值','', '最新授权明确采用每个种子、每个部件类别分别校准，两个相机合并；候选来自旧C正常完整解码预测的已有分数。在非空预测且实际precision≥0.90的候选中最大化recall，等召回取更高阈值。没有用PR包络、插值或GT中心oracle；无可行非空候选时标校准失败，不放宽目标。此规则替代上轮尚未执行的全局网格提案。','', '|种子|类别|锁定阈值|旧C P/R|TP/FP/FN|有效GT/有效图像/根|有TP根|','|---|---|---:|---|---|---|---:|']
 for q in cal:lines.append(f'|{q["seed"]}|{q["category"]}|{q["threshold"]:.9f}|{q["precision"]:.4f}/{q["recall"]:.4f}|{q["TP"]}/{q["FP"]}/{q["FN"]}|{q["GT"]}/{q["valid_images"]}/{q["valid_roots"]}|{q["TP_root_count"]}|')
 lines+=['','旧C共有8根、64状态、128图；四项校准均有非空可行预测，详细候选数及预测根覆盖见 `configs/calibration_locked.json`。阈值锁定早于新根生成与渲染；在新根结果后未修改任何阈值。','', '## 新根结果：固定与校准并列','', '![工作点比较](figures/new_roots_operating_points.png)','', '|种子|类别|工作点|精确率P（95%根bootstrap区间）|召回率R（95%区间）|F1（95%区间）|TP/FP/FN|','|---|---|---|---|---|---|---|']
 for s in [0,1]:
  for c in ['all']+CATS:
   for op in ['fixed0.5','calibrated']:
    q=get(rows,s,'all',c,op);lines.append(f'|{s}|{CN[c]}|{op}|{intervaltext(q["precision"],q["CI95"]["precision"])}|{intervaltext(q["recall"],q["CI95"]["recall"])}|{intervaltext(q["F1"],q["CI95"]["F1"])}|{q["TP"]}/{q["FP"]}/{q["FN"]}|')
 lines+=['','四项校准新根精确率分别为前板S0 88.82%、S1 86.75%；手指S0 87.88%、S1 89.71%。即使部分非常接近目标，也不放宽90%规则。前板减少漏检同时增加FP；手指通过提高阈值减少FP，也漏掉更多真目标。总体F1的变化在S0/S1不同，不能只报较好种子。','', '![新根PR](figures/new_roots_PR.png)','', '完整AP50来自所有正常候选、一对一IoU≥0.5匹配和全有效GT分母。两个工作点仅改变接受哪些预测，不改变模型分数，所以完整AP50相同；不把阈值截断后的面积冒充完整AP。分类别校准的总体P/R点不一定在统一阈值PR线上。','', '|种子|相机|类别|AP50（95%根bootstrap区间）|GT|','|---|---|---|---|---:|']
 for q in rows:
  if q['workpoint']=='fixed0.5':lines.append(f'|{q["seed"]}|{q["camera"]}|{q["category"]}|{intervaltext(q["AP50"],q["AP50_CI95"])}|{q["GT"]}|')
 lines+=['','所有类别/相机/工作点的P/R/F1、计数、AP和区间保存在 `metrics.csv`，所有16根分布保存在 `root_metrics.csv`。','', '## 根级不确定性与有效覆盖','', '![全部根召回](figures/root_recall_distribution.png)','', '按整根重采样2000次、固定seed=261019；同一重采样用于两个种子、两工作点及所有分层，报告95%百分位区间。重复根被视为独立副本，按其倍数重算累计TP/FP、GT及完整AP，不能平均图像AP。保存的配对差值区间见 `results/paired_bootstrap.json`。两个训练种子不是独立统计样本。','', '区间仅描述这16个根的经验分布，不包含阈值重新校准或更换场景的不确定性，也不提供有限样本的精确率保证。前板固定0.5在这批样本FP为0，其经验bootstrap P区间退化为[1,1]，这不是未来零误报保证。多个分层未进行显著性筛选或多重比较优胜选择。','', '![未知覆盖](figures/unknown_coverage.png)','', '|相机|类别|有效/未知图像|有效框|有效根|原因|','|---|---|---|---:|---:|---|']
 for q in coverage:lines.append(f'|{q["camera"]}|{q["category"]}|{q["known_images"]}/{q["unknown_images"]}|{q["usable_positive_boxes"]}|{q["positive_roots"]}|{q["unknown_reasons"]}|')
 lines+=['','任一未知抽屉像素使整幅前板类别UNKNOWN；任一手指族零可见像素使手指类别UNKNOWN。有效GT合计715（前板207、手指508）。未知不作为负例，零像素不推断对象不存在。有效类别内背景patch规则未变；静态/腕部把手困难负例上下文分别覆盖126/128图、均16根。本轮没有可认证整图不存在负例，因此不能声称已验证物体不存在识别。','', '## 采集、去重和访问边界','', '16根使用预冻结seed261018800—261018815；保留原块布局与运动路点采样范围，4个抽屉初始带各4根。每根原生复位后7条固定电机路点，最多8状态；没有按到达、碰撞、可见性或预测分数筛选。所有16根一次完成，无替换根、无重试。','', '512次相机调用包含256张主图及256张同刻去分割标志复核图；128个前后完整物理快照相同，RGB/depth逐位一致，几何mask/可见框一致性与最低覆盖门槛通过。采集原始/标注约78.0MB，首50状态实测存储另存；后续冻结DINO特征复用本机缓存权重，不下载模型。','', '已登记旧根ID、种子、初始化签名及近1mm布局无交集；16个新初态快照互异。与79240个历史文件哈希及既有512张主RGB数组哈希比较，新初态、原始主捕获、双相机文件、单图RGB均无交集，新根内部也无重复双图。历史4状态通过其已存初始化/文件哈希排除，没有把技术样本当新测试。去重覆盖已登记历史，不声称覆盖未登记数据。','', '旧D定位留出的样本数组、特征、预测、评分均未打开；只使用既存根/哈希元数据去重。准备阶段为了解技术索引格式，读取了原4状态技术结果索引，其中含各场景标签摘要；该索引未参与校准或验证。原4状态及旧D都不是本轮测试数据。','', '训练/推理均无GT输入。新根特征进程拒绝读取私有真值；两种子每个仅一次新根推理，原始预测先封存再评分。评分脚本运行一次，随后对保存候选做独立数值核验；没有根据核验或结果改变模型、阈值或样本。','', '## 定位结论与关系主线','', '原0.5缺少召回；开发校准改善前板召回，但新根精确率目标未获保持，手指更严格阈值下召回依旧很低。当前结果支持“定位仍是不可靠输入”，不能称已完成可靠对象关系感知，也不能把AP较高等同于某个工作点可用。阶段预算用尽后封存，不继续检测器搜索。','', '下一步设计见 `RELATION_READOUT_DESIGN.md`：用仿真可见真值框作为明确的正确定位条件，隔离Holding等关系读取，同时与预测框端到端结果严格分开。同一读出器与工作点下可做一次仅定位替换诊断；真值框不能输入运行时规划，更不能把关系标签写回预测。','', '当前定位器只有前板/手指，缺积木定位，因此该设计明确禁止拿真值积木框参与的结果叫全端到端。新C16也未保证持物正例或连续接触真值，不能擅自当关系训练集。本轮仅写设计，没有关系模型训练、新采集或新标签制造。完整世界模型与TS-LTL主指南尚未完成。','', '## 交付索引','', '- `REPORT.md`、`metrics.csv`、`root_metrics.csv`、`coverage.csv`、`figures/`。','- 实验 `configs/calibration_locked.json`、`protocol.json`、`precollection_frozen.json` 固定校准、根与预算；`results/deduplication.json`、`coverage.json`、`evaluation_summary.json`、`paired_bootstrap.json`、`final_verification.json` 记录证据。','- 运行命令与封存哈希见 `RUN_COMMANDS.md` 和实验 `configs/delivery_manifest.json`。']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n');shutil.copy2(Path(__file__).parent/'RELATION_READOUT_DESIGN.md',R/'RELATION_READOUT_DESIGN.md');save(E/'results/report_assets.json',{'utc':now(),'figures':[str(f)for f in(R/'figures').glob('*.png')],'all_four_new_class_precision_point_estimates_below_0_90':all(q['precision']<.9 for q in rows if q['workpoint']=='calibrated'and q['camera']=='all'and q['category']!='all'),'detector_search_closed':True,'relation_design_only':True});print('FRESH_C16_REPORT_READY',flush=True)
if __name__=='__main__':main()
