from config import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
import csv
fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
COL={'actual':'#2471a3','one_step':'#d68910','recursive':'#b03a2e'}
NAMES={'actual':'真实后继编码后读出','one_step':'真实前态的一步预测','recursive':'仅初始观测的递归预测'}
HEADS=['original_coverage_seed0','intermediate_coverage_seed0','original_coverage_seed1','intermediate_coverage_seed1']
HN={'original_fixed_head':'原冻结头','original_coverage_seed0':'原覆盖 · 种子0','intermediate_coverage_seed0':'补充覆盖 · 种子0','original_coverage_seed1':'原覆盖 · 种子1','intermediate_coverage_seed1':'补充覆盖 · 种子1'}
def main():
 part=read(E/'results/part1_metrics.json');ev=read(E/'results/head_evaluation.json');heads=ev['heads'];coverage=read(E/'results/head_data_coverage.json');freeze=read(E/'configs/training_frozen.json')
 fig,axs=plt.subplots(2,2,figsize=(14,10));h=np.arange(1,5)
 for col,(cohort,title)in enumerate([('normal','正常完整链：固定5根'),('refusal','含拒绝的调用记录：固定10根')]):
  rows=part[cohort];N=rows[0]['N'];ax=axs[0,col]
  for m,marker in [('actual','o'),('one_step','s'),('recursive','^')]:
   vals=[r[m+'_all9']for r in rows];ax.plot(h,np.array(vals)/N*100,marker=marker,color=COL[m],label=NAMES[m],lw=2,ms=7)
  ax.set(title=title,xlabel='已调用的技能步（一次完整调用，包括拒绝）',ylabel='九命题全对的状态比例（%）',xticks=h,ylim=(-5,110));ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9,loc='upper right'if col==1 else'lower left')
  ax=axs[1,col];xs=np.arange(8);bottom=np.zeros(8)
  cats=[('both_right','真实与预测均对','#479c80'),('actual_right_prediction_wrong','真实读对、预测错','#e6a33a'),('both_wrong','真实与预测均错','#bd625b'),('actual_wrong_prediction_right','真实读错、预测对','#9380b2')]
  for key,label,color in cats:
   vals=np.array([r[m+'_paired_facts'][key]for r in rows for m in ['one_step','recursive']]);ax.bar(xs,vals,bottom=bottom,label=label,color=color,width=.75);bottom+=vals
  ax.set(xticks=xs,xticklabels=[f'{i}步\n{t}'for i in h for t in ['一步','递归']],ylabel=f'逐命题配对数（每柱合计{N*9}项）',ylim=(0,N*9*1.24));ax.legend(fontsize=8,ncol=2,loc='upper center');ax.grid(axis='y',alpha=.2)
 fig.suptitle('固定编码器、转移和原查询头：真实读出与两种预测逐项配对',fontsize=16,y=.98)
 fig.text(.04,.02,'上排：同一批根、同一实际动作；三条线分别显示真实后继读出、逐步重置真实前态、不中途纠正的递归。\n下排：每个对象命题独立计数，四种颜色相加等于全部已知命题；橙色是预测路径额外失配，不能单独证明纯动力学误差。',fontsize=10)
 fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(R/'paired_prediction.png',dpi=170);fig.savefig(R/'paired_prediction.svg');plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(15,5.3),gridspec_kw={'width_ratios':[1,1.25,1.25]});colors=['#687c9a','#2c9c95'];ax=axs[0]
 for j,arm in enumerate(['original_coverage','intermediate_coverage']):
  vals=[heads[f'{arm}_seed{s}']['C_validation']['overall']['full9_correct']for s in[0,1]];bars=ax.bar(np.arange(2)+(j-.5)*.32,vals,width=.32,color=colors[j],label=['原覆盖','补充覆盖'][j]);ax.bar_label(bars)
 ax.set(xticks=[0,1],xticklabels=['种子0','种子1'],ylabel='九命题全对状态数 / 110',ylim=(0,110),title='完整九命题：结果随种子变化');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
 for s in[0,1]:
  ax=axs[s+1]
  for j,arm in enumerate(['original_coverage','intermediate_coverage']):
   g=heads[f'{arm}_seed{s}']['C_validation']['overall']['groups'];vals=[g[k]['wrong_states']for k in ['Holding','In','drawer']];bars=ax.bar(np.arange(3)+(j-.5)*.32,vals,width=.32,color=colors[j],label=['原覆盖','补充覆盖'][j]);ax.bar_label(bars)
  ax.set(xticks=[0,1,2],xticklabels=['持物\nN=116','入屉\nN=110','抽屉\nN=116'],ylabel='组内至少一项错误的状态数（越少越好）',ylim=(0,39),title=f'种子{s}：持物改善，入屉有取舍');ax.grid(axis='y',alpha=.2)
 fig.suptitle('C根级隔离验证：同一冻结世界模型，只重训查询头',fontsize=16,y=.99)
 fig.text(.035,.025,'灰蓝=原覆盖；青绿=补充覆盖。两臂各144个训练状态、1200次更新、38400次抽样；新增记录仅6项标签有效。\n左图仅计110个完整标签状态（5根）；右图按各组有效标签计数。两个种子不是置信区间，不据此选择最终部署头。',fontsize=10)
 fig.tight_layout(rect=[0,.13,1,.93]);fig.savefig(R/'query_coverage.png',dpi=170);fig.savefig(R/'query_coverage.svg');plt.close(fig)
 lines=['# 预测路径与查询头的两项限定诊断','','完成时间：'+now(),'','## 结论','','真实中间状态可以读出部分关键信息，但原查询头并非稳定可读：正常链在第2、3步各只有3/5条真实后继九命题全对。每步重置真实前态仍不能全部预测正确；不中途纠正的递归在第2、3步进一步丢失完整状态。含拒绝记录在第一步已发生大量失配，不能把它们都解释成累积误差。','','只改变查询头的覆盖与抽样后，两个种子的持物错误都减少；九命题整体结果一升一降，入屉错误存在取舍。现有证据不支持“补充查询训练已解决预测问题”，也不支持“所有中间状态不可表示”。编码器、转移和规划器均未修改，没有新的仿真或任务成功率评测。','','## 一、相同真实转移的配对诊断','','固定 canonical_matched seed0/update1000、原 query_seed0、原阈值。一步预测每步编码真实前态；递归只编码初始态；真实后继编码是单独读出对照。一步单位为一次完整技能调用，拒绝也计一步。','','![两类轨迹的全九命题正确率和逐命题配对](paired_prediction.png)','','图的颜色、轴与含义在图内说明。上排分母固定为正常5根、含拒绝10根；两个来源有相同复位根，不能合成15个独立根。下排四种颜色为同一真实后继、同一命题上的配对分类，不使用两条总体准确率相减作归因。','', '| 记录组 | 技能步 | 真实后继读对 | 真实前态一步 | 递归 | 一步对/递归错 | 一步错/递归对 |','|---|---:|---:|---:|---:|---:|---:|']
 for cohort,title in [('normal','正常'),('refusal','含拒绝')]:
  for r in part[cohort]:lines.append(f'| {title} | {r["horizon"]} | {r["actual_all9"]}/{r["N"]} | {r["one_step_all9"]}/{r["N"]} | {r["recursive_all9"]}/{r["N"]} | {r["one_step_right_recursive_wrong"]} | {r["one_step_wrong_recursive_right"]} |')
 lines+=['','正常链最早有影响的失配集中在第2步（提起物体）：一步路径的持物/入屉错误状态为2/1，递归为3/4；到第3步（放入）递归5/5条仍在持物命题上出错。含拒绝组的第一步不一定是拒绝动作：此时真实读出已有8/10条未全对，一步和递归另有19/90项命题出现“真实读对、预测错”。第2、3步该配对数在一步路径为17/90、20/90，在递归路径为30/90、35/90。说明真实读出和预测路径都存在失配，但不能凭此把全部差异归为纯动力学问题。','','原空间潜状态MSE保留在[完整配对表](part1_prediction_pairing.md)。正常第4步递归MSE较大（0.2886对0.1386），却比一步路径多读对一条（5/5对4/5）；它是256维表示的平方误差，不能替代语义正确性。','','已核对实际start/end帧、相邻边界、日志动作和96帧稳定等待。输入为同边界双相机RGB及15维本体观测，使用单帧、没有未来窗口。89个唯一真实状态、85个已观测转移、765条逐命题配对；23个未命中缓存的旧观测只做了冻结DINO编码。真实后继与递归概率重现旧封存结果最大差异1.32×10⁻⁶。失败、未到达及原三步链都保留，未补造第四步。','','## 二、等预算查询头覆盖对照','','在看到本次C验证结果前，冻结了模型、协议、数据来源哈希、根级划分、抽样表和训练代码。世界模型固定为同一个seed0；此处两个种子只控制查询头初始化与抽样。仍用原25473参数Query结构、原归一化、原类别权重、原0.5/argmax阈值与掩码损失。两臂同种子初始权重哈希相同。','','| 项目 | 原覆盖 | 补充中间覆盖 | 原因/限制 |','|---|---|---|---|','| 不同A/B训练状态 | 144（18根） | 144（22根） | 128个共享；16个固定替换 |','| 完整九项标签状态 | 142 | 126 | 其余标签按原validity掩码保留 |','| 每次更新抽样 | 20个共享状态+12个原池状态 | 相同20个+持物开屉/放入后/持物关屉各4个 | 都按根均衡，再在根内抽状态 |','| 更新/总抽样/种子 | 1200 / 38400 / 0、1 | 相同 | 有效标签数量不同，完整记录在冻结协议 |','| C选择 | 106状态、5根 | 同一批 | 最小根平均掩码损失；4个候选点 |','| C验证 | 116状态、6个其他根 | 同一批 | 110状态完整九项（5根），6状态仅六项（1根） |','','所有四次训练都完成1200次更新，并均由C选择集选中第600次。未根据C验证或D2结果换检查点。C在此前模型实验中被使用过，因此这是一项新查询头的根级隔离验证，不是全新世界模型泛化测试。','','![两种子查询头覆盖对照](query_coverage.png)','','| 查询头 | 完整九项全对 | 完整根平均 | 持物错/116 | 入屉错/110 | 抽屉错/116 |','|---|---:|---:|---:|---:|---:|']
 for name in ['original_fixed_head']+HEADS:
  m=heads[name]['C_validation']['overall'];lines.append(f'| {HN[name]} | {m["full9_correct"]}/110 | {m["full9_root_macro"]:.1%} | {m["groups"]["Holding"]["wrong_states"]} | {m["groups"]["In"]["wrong_states"]} | {m["groups"]["drawer"]["wrong_states"]} |')
 lines+=['','完整根平均：先在各完整标签根内求准确率，再平均5根；不是把6个部分标签状态当全九项正确。组错误为该组至少一项错，不同组可重叠。','','| C验证子类 | 原覆盖 seed0 → 补充 | 原覆盖 seed1 → 补充 | 可解释范围 |','|---|---|---|---|','| 持物且开屉，完整九项 | 3/4 → 3/4 | 3/4 → 2/4 | 2根；未一致改善 |','| 放入后，完整九项 | 1/2 → 2/2 | 2/2 → 2/2 | 2根，样本很少 |','| 持物且关屉，仅六项 | 0/2 → 2/2 | 1/2 → 2/2 | 仅1根；三项In未知，不能报九项准确率 |','| 其他状态，完整九项 | 63/104 → 74/104 | 65/104 → 62/104 | 收益与退化并存 |','','逐状态配对：seed0有15个完整状态由错变对、3个由对变错；seed1为8个由错变对、12个由对变错。补充臂在两个种子中都降低持物误报；但入屉组出现4和9个漏报事实，原覆盖两种子均为0。补充覆盖未解决持物/入屉解耦，且不能只报告持物收益。','','额外16个A/B状态来自4个匹配根，其物理记录缺少姿态，不能按原v2.1定义可靠重标In。未知仍未知，没有把缺失标签写成false。两臂更新数和抽样数相同，但有效事实监督约34.5万对31.9万；这项对照识别的是“新增根与中间状态覆盖、抽样比例、部分标签”整体处理效果，不能把改善单独归因于过采样。','','## 三、D2只作既有记录诊断回测','','以下四步序列均对应同一批5条正常链或10条含拒绝记录，值为全九项正确的条数。它们已被反复查看，不是独立测试，不用于选择查询头。只复用本次第一部分的冻结潜状态，没有新采集。','', '| 查询头 | 正常：真实读出 /5 | 正常：一步 /5 | 正常：递归 /5 | 含拒绝：一步 /10 | 含拒绝：递归 /10 |','|---|---|---|---|---|---|']
 for name in ['original_fixed_head']+HEADS:
  d=heads[name]['D2_diagnostic'];seq=lambda c,m:'、'.join(str(r[m+'_all9'])for r in d[c]);lines.append('| '+HN[name]+' | '+' | '.join([seq('normal',m)for m in ['actual','one_step','recursive']]+[seq('refusal',m)for m in ['one_step','recursive']])+' |')
 lines+=['','补充覆盖使部分真实后继和拒绝组一步读出改善，但正常递归第3步仍是0/5；seed0补充头在正常第一步反而从5/5退到2/5，而真实后继读出仍5/5。仅以真实潜状态训练的查询头，对预测潜状态不一定保持兼容。这支持保留预测路径失配问题；尚不能定位为编码器、转移和查询阈值中的单一原因。','','## 文件与复核','','- [逐命题765行配对](part1_paired_propositions.csv)；[第一部分详细分组与原空间误差](part1_prediction_pairing.md)。','- [C验证逐状态配对汇总](C_validation_paired_states.csv)；全量逐状态概率、标签与掩码保存在实验results目录。','- 实验目录：`'+str(E)+'`。协议、来源和训练前哈希在configs；只新增的查询头权重在models；训练与评估日志在logs。','- `configs/training_frozen.json`记录训练前全部输入与源码哈希；`heads_frozen_before_validation.json`记录C验证前的四个选择结果。','- `results/verification.json`为最终独立重算与旧封存完整性检查；`configs/delivery_manifest.json`封存本次交付。','','本轮到此结束，不新增调参、规划器修改或任务成功率评测。当前数据不足以验证持物关屉的完整九命题，亦不足以证明跨种子的总体改善。']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n')
 with(R/'C_validation_summary.csv').open('w')as f:
  w=csv.writer(f);w.writerow(['head','full9_correct','full9_N','full9_root_macro','Holding_wrong_N116','In_wrong_N110','drawer_wrong_N116'])
  for name in ['original_fixed_head']+HEADS:
   m=heads[name]['C_validation']['overall'];w.writerow([name,m['full9_correct'],m['full9_N'],m['full9_root_macro']]+[m['groups'][k]['wrong_states']for k in ['Holding','In','drawer']])
 print('REPORT_AND_TWO_FIGURES_READY',flush=True)
if __name__=='__main__':main()
