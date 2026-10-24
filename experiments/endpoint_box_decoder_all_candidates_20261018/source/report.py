from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import shutil,collections

CN={'drawer_front':'前板','gripper_finger':'夹指','train':'A/B训练','AB_episode_heldout_development':'A/B留出开发','C_seen_diagnostic':'C已见诊断','D_seen_diagnostic':'D已见诊断'}
def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in rows)
def main():
 check_manifest(E/'configs/evaluation_frozen.json');assert read(E/'results/verification.json')['passed']
 e=read(E/'results/evaluation.json');rows=read(E/'data/images.json');anns=read(E/'data/annotations/final.json')['annotations'];audit=read(E/'results/GT_candidate_audit.json');traces=read(E/'results/candidate_trace.json');evidence=read(E/'results/design_issue_evidence.json')
 def group(res,seed,split,cat,variant):return next(x for x in e['groups']if x['dimensions']=='split/category'and x['resolution']==res and x['seed']==seed and x['split']==split and x['category']==cat and x['variant']==variant)
 def cell(x):return f"{x['TP']}/{x['FP']}/{x['FN']}；{x['mean_best_IoU_including_misses']:.3f}"
 fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':fp.get_name(),'font.size':11,'axes.unicode_minus':False})
 fig,axes=plt.subplots(3,2,figsize=(10,13));examples=read(E/'configs/protocol.json')['preselected_figures']
 descriptions=['训练：相邻两根夹指','开发：数量截断丢掉真候选','开发：没有合格候选，修正无效']
 for row,(case,desc)in enumerate(zip(examples,descriptions)):
  i=case['image_id'];r=rows[i]
  with np.load(r['path'])as z:rgb=z[r['key']]
  for col,variant in enumerate(['original','corrected']):
   rec=next(x for x in e['records']if x['resolution']==case['resolution']and x['seed']==case['seed']and x['image_id']==i and x['category']=='gripper_finger'and x['variant']==variant)
   ax=axes[row,col];ax.imshow(rgb,interpolation='nearest');ax.set_axis_off()
   for b in anns[i]['classes'][1]['boxes']:
    x0,y0,x1,y1=b;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#22b6ed',linewidth=2.0,linestyle='--'))
   matched={m['prediction']for m in rec['matches']}
   for pi,p in enumerate(rec['predictions']):
    x0,y0,x1,y1=np.array(p['box'])*[r['width'],r['height'],r['width'],r['height']];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#14b65b'if pi in matched else'#e94444',linewidth=2.0))
   ax.set_xlim(-.5,r['width']-.5);ax.set_ylim(r['height']-.5,-.5)
   ax.set_title(f"{desc}\n{'原版'if variant=='original'else'修正版'}：命中{rec['TP']}/{rec['GT_count']}，误检{rec['FP']}（{case['resolution']}，种子{case['seed']}，id{i}）",fontsize=11)
 fig.suptitle('解码能找回丢掉的候选，但不能补出模型缺少的候选',fontsize=17,y=.985)
 fig.text(.5,.015,'蓝虚线=冻结RGB标注；绿实线=一对一命中；红实线=误检。只显示夹指，所有保留夹指预测均画出。\n左右使用完全相同图像和模型输出；固定分数≥0.5、NMS IoU>0.3。无绿/红框表示未输出。',ha='center',fontsize=11)
 fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(R/'figures/decoder_examples.png',dpi=150);plt.close(fig)
 bysplit=[]
 for split in SPLITS:
  rr=[]
  for res in[224,448]:
   for seed in[0,1]:
    for cat in CATS:rr.append([res,seed,CN[cat],cell(group(res,seed,split,cat,'original')),cell(group(res,seed,split,cat,'corrected'))])
  bysplit.append('### '+CN[split]+'\n\n'+table(['输入','种子','部件','原版 TP/FP/FN；IoU','修正版 TP/FP/FN；IoU'],rr))
 adjacent=[[x['seed'],x['image_id'],x['patch'],f"{x['score']:.3f}",f"{x['IoU']:.3f}",'局部峰移除→保留并命中']for x in evidence['adjacent_training_centers']]
 center_cases=[]
 for a in audit:
  q=a['GT_center_qualified_candidate']
  if a['resolution']==448 and a['seed']==0 and a['split']==SPLITS[1]and q and not a['original_matched']:
   center_cases.append([a['image_id'],q['patch'],f"{q['score']:.3f}",f"{q['IoU']:.3f}",'恢复命中'if a['corrected_matched']else'仍被正常NMS移除'])
 cause=read(E/'results/candidate_cause_summary.json');causerows=[]
 for x in cause:
  if x['split']==SPLITS[1]:
   cc=x['corrected_miss_causes'];causerows.append([x['resolution'],x['seed'],CN[x['category']],x['GT_boxes']-x['corrected_TP'],cc.get('no_qualified_candidate',0),cc.get('NMS_removed_all_qualified',0),cc.get('one_to_one_matching_competition',0)])
 unknown=[]
 for res in[224,448]:
  for seed in[0,1]:
   for c in CATS:
    rr=[x for x in e['records']if x['resolution']==res and x['seed']==seed and x['category']==c and x['status']=='uncertain']
    vals=[]
    for v in['original','corrected']:
     z=[x for x in rr if x['variant']==v];vals.append(f"{sum(x['pred_count']>0 for x in z)}/{len(z)}图；{sum(x['pred_count']for x in z)}框")
    unknown.append([res,seed,CN[c],*vals])
 q=evidence['reported_cap_candidate'];nmsexample=next(a for a in audit if a['resolution']==448 and a['seed']==0 and a['image_id']==30 and a['category']=='gripper_finger'and a['GT_index']==0);nq=nmsexample['qualified_candidates'][0];block=next(x for x in traces if x['resolution']==448 and x['seed']==0 and x['image_id']==30 and x['category']=='gripper_finger'and x['patch']==nq['corrected']['blocking_patch']);cand=next(x for x in traces if x['resolution']==448 and x['seed']==0 and x['image_id']==30 and x['category']=='gripper_finger'and x['patch']==499)
 report=f'''# 保存候选的解码修正验证

本轮**确认并修正了两个可复现的设计问题，但泛化仍未通过**。224训练夹指命中从26/30、27/30提升到29/30、30/30；448开发种子0从3/29提升到6/29，同时误检9→10。448开发剩余23/24个夹指漏检中，22/24个没有符合固定分数和IoU要求的候选。原始结果全部保留，修正版仅诊断，未部署。

## 范围与唯一修正版

仅CPU读取上轮16个保存输出文件，共160组图像×模型配置，204,800个patch类别位置，3457个分数达到0.5的候选。没有加载模型、重新提特征、推理或训练；两个分辨率、两个种子、40图、标注和来源划分完全不变。

| 步骤 | 原版 | 本轮修正版 | 原因 |
| --- | --- | --- | --- |
| 候选分数 | score≥0.5 | 不变 | 不调阈值弥补模型输出 |
| 同类局部峰 | 3×3网格内只保留局部最大 | 取消 | 相邻真实夹指不是重复实例，单中心监督与局部互斥会冲突 |
| 框解码 | 原sigmoid/tanh参数化，裁剪至图像范围，去无效框 | 不变 | 几何与输入完全相同 |
| 框NMS | 按分数降序、相同分数按patch编号；同类IoU>0.3抑制 | 不变，处理全部候选 | NMS实现独立复核通过；不把GT用于去重 |
| 数量上限 | 最多1个前板、2根夹指后停止 | 两类都不设数量上限 | 错框可能先占名额；所有额外保留框如未匹配都计FP |

前一轮224去峰诊断仍保留1/2框上限，因此不覆盖本版本。这里同时去掉两个筛选条件，是一个预先明确的组合修正版；每项机械去除原因由候选轨迹说明，没有另做多个变体或选择最好结果。原版输出与指标精确重现后，修正版预测先冻结，随后才读取标注评分。

## 确定的设计问题

**相邻正中心被当成互斥峰。** 224训练id0、2、6中，两根不同真实夹指的中心落在相邻格。以下较低分的候选已超过0.5且框正确，原版却在NMS前移除；每种子恰好恢复3根。

{table(['种子','图像','候选patch','分数','对其GT的IoU','轨迹'],adjacent)}

448开发种子0还有三个GT中心正确候选被局部峰移除：

{table(['图像','候选patch','分数','IoU','修正后'],center_cases)}

**提前截断会让错框占掉真框名额。** 448开发id27种子0，patch132的score={q['score']:.3f}、IoU=0.702，原版在保留两个框后根本未处理它；修正版保留它。该图从1命中/1误检/1漏检变为2命中/1误检/0漏检，原来的错框仍计误检，并没有用GT把它删掉。

**NMS实现没有发现错误。** 320组图像类别的NMS结果由另一套标量IoU实现独立复现。id30的patch499仍被patch500抑制，两框IoU={iou(cand['box'],block['box']):.3f}>0.3，符合固定规则；这说明贪心NMS和分数排序也可能留下错框、移除较正确候选，是规则与评分的局限，不是代码错误。本轮没有进一步改NMS。

![解码修正叠框](figures/decoder_examples.png)

上图左原版、右修正版；三行分别是训练相邻夹指、开发截断案例、开发仍无合格候选的案例，均在本轮解码前指定。蓝虚线是冻结RGB标注，绿实线是一对一命中，红实线是误检。所有保留夹指框均画出，前板未在这张简图显示，但在全量评分中保留。左右原图尺度完全相同；坐标原点在左上，向右/下递增。

## 全量结果：逐集合、分辨率、种子原样报告

每格为**TP/FP/FN；所有GT的最佳预测IoU均值**。一对一匹配阈值0.5；无输出时IoU=0，有未达匹配阈值的框仍计其实际IoU。该IoU统计可能因多输出错误框而增大，必须同时看FP。训练/开发/C/D不合并，两个种子不挑选。未知类别不当负例；C/D历史已见，只作诊断。

{chr(10).join(bysplit)}

新增误检已全部计入：224开发夹指FP由5/6变8/7；224 C夹指由2/2变3/3；448训练种子1由0变1；448开发种子0由9变10。某些额外框落在同一目标附近但边界不合格，也按FP计数。没有把重复或错位框用GT事后剔除。

## 解码之后，为什么仍不可靠

“无合格候选”指遍历全部保存的同类候选后，没有同时满足score≥0.5、有效框且与该GT IoU≥0.5的候选。GT仅用于这个事后原因分类，未参与解码、NMS或输出筛选；它不是另一种检测成绩。

{table(['输入','种子','部件','修正后漏检','无合格候选','合格候选全被NMS移除','一对一竞争'],causerows)}

448开发夹指：种子0剩23个漏检，其中22个无合格候选，1个因正常NMS；种子1剩24个漏检，全部无合格候选。因此解码修正没有解决泛化。静态小夹指修正后仅2/13、1/13，腕部大夹指仍仅4/16、4/16；问题不限于patch尺度。

训练也要分情况看：224修正后29/30、30/30，原先部分“训练未拟合”其实是解码造成；448训练仍25/30、28/30，原分数不足的漏检并未因此消失。不要把“修复训练解码”写成“学习到了可靠视觉语义”。

## 未知与标注限制

40图中，前板的20张腕部图全部uncertain，夹指有4张静态图uncertain。下表只统计未知区域的输出覆盖，不作TP/FP评价：

{table(['输入','种子','类别','原版有输出图/未知图；框数','修正版有输出图/未知图；框数'],unknown)}

完全不可见not_visible实例仍为0，缺失拒识无法评价。40图具有可追溯RGB、来源划分和标注复核记录；同助手复标不等于独立准确率，不能称为独立人工金标准。相邻帧和多个候选也不增加独立源episode数量。

## 阶段结论与后续路线

详见[简明阶段结论与最低需求](STAGE_SUMMARY.md)。已完成的是40图可追溯标注、轻量定位基线、分辨率与解码诊断。确定问题是相邻同类中心互斥和提前数量截断；NMS执行正确。仍未知的是有限训练覆盖、冻结表征、中心监督/评分、边界标注分别造成多少留出退化。当前不能用这批已看开发图继续证明泛化。

后续应先把上游定位作为独立基础模块验证，停止在这40图上反复创造检测头；下游关系主线需要可靠的两端点和独立来源，不能靠人工动态前提/效果或GT位置掩盖定位错误。本轮没有开启任何后续实验。

## 可核验证据

- [冻结协议](../../experiments/{NAME}/configs/protocol.json)、[预测先于评分冻结](../../experiments/{NAME}/configs/outputs_frozen_before_scoring.json)。
- [3457候选轨迹](../../experiments/{NAME}/results/candidate_trace.json)、[确定设计问题](../../experiments/{NAME}/results/design_issue_evidence.json)、[逐GT候选审计](GT_candidate_audit.csv)。
- [全部逐图指标](all_image_metrics.csv)、[分组指标](all_group_metrics.csv)、[漏检原因](candidate_cause_summary.csv)。
- [独立核验](../../experiments/{NAME}/results/verification.json)：320组NMS、1344个TP/FP/FN单元、728个跨版本GT评分；旧五个封存逐文件哈希不变。
- [运行说明](RUN_COMMANDS.md)、[封存清单](../../experiments/{NAME}/configs/delivery_manifest.json)。

本轮模型前向、特征提取、优化器步数均为0。结果只用于诊断，未采用部署；主指南的世界模型、TS后继与闭环验收不能由这40图替代。
'''
 (R/'REPORT.md').write_text(report)
 summary=f'''# 当前做到哪里，下一步需要什么

**这一支线已完成40图可追溯RGB标注、轻量定位基线、224/448分辨率对照和保存候选解码诊断；泛化未通过。** 修正版能找回被规则误删的框，但无法补出模型没有给出的合格候选。

| 已确定 | 证据与意义 |
| --- | --- |
| 相邻真实夹指不应互斥 | 224每种子3个正确候选被3×3局部峰移除；取消后训练命中29/30、30/30 |
| 数量先验不能直接等同于“取最高分两个框” | 448开发id27先保留的错框占了名额；取消截断找回正确夹指，错框仍计FP |
| NMS代码运行正确 | 独立实现复现320组图像类别结果；固定规则仍有错误抑制的实例 |
| 解码修正不足以解决泛化 | 448开发仅6/29、5/29命中；23/24个剩余漏检中22/24个没有合格候选 |

尚未确定：样本来源覆盖、冻结DINO表示、中心监督与评分、部件边界标注各占多少原因。提高分辨率没有稳定改善；不能据此断言任何一个因素无效或已解决。腕部前板全未知、没有可靠不可见负例、同助手标注未经独立准确率验证，这些限制仍在。

![简洁叠框](figures/decoder_examples.png)

蓝虚线=冻结RGB标注；绿实线=命中；红实线=误检。左原版、右修正版。第一行找回相邻夹指；第二行找回被截断候选但仍有错框；第三行两版都没有输出。图只显示夹指，未隐藏任何保留夹指预测。

## 最合理的后续路线

先完成**上游对象/部件定位模块**，再继续**下游对象对关系与世界模型研究**。修复已证实的解码逻辑应作为后续工具的基本要求；不能一边用不可靠端点，一边把关系错误归因于对象对表示或RSSM。

| 环节 | 最低数据或工具需求 | 完成后才能说明什么 |
| --- | --- | --- |
| 标签审核 | 能显示原始双RGB、逐实例框/掩码、可见/遮挡/未知状态的本机标注与审核工具；对部件定义和用于评分的标注作独立复核 | 明确抽屉前板/把手、夹指的边界；当前同助手复标只能说明重复一致性 |
| 定位数据覆盖 | 保留源episode/初始化谱系及RGB哈希；在必须评分的类别×相机中覆盖可见、部分遮挡、可靠不可见和混淆负例；新开发/测试根不得落入历次已看清单 | 能分辨误检、漏检和缺失拒识，而不是把uncertain当负例。腕部不能确认前板就继续标未知，不把内腔改名凑标签 |
| 定位工具链 | 选择一套标准实例检测或分割训练/评估流程，任务适配后固定；复用本机单卡、缓存、候选导出与一对一评价器，不再自造多套小头轮流试开发集 | 获得可独立验收的视觉端点。标准工具也不保证夹指/抽屉开箱可用；本轮未安装、下载或选择新模型 |
| 评价设计 | 预先冻结来源划分、一次预算和验收标准；正式A/B训练、C开发、D测试遵循主指南，历次已见C/D只保留诊断角色 | 新来源的可靠定位，而非已看40图上的改进。每个统计格有样本只是能计算分母，不能据此保证充分性 |
| 下游关系 | 使用预测端点及可见内容的视觉特征；对象身份与关系标签可追溯，GT仅标签/评分；把定位失败与关系读出失败分开计数 | 比较对象对/节点表示，继而检验共享符号读出与学习转移提供TS后继。不能用人工前提/效果替代世界模型 |

这里的“最低”指必要覆盖与工具能力，不是承诺某个小样本数即可通过。现有16张训练图、4个训练源episode不支持估算可靠样本量；后续应在新的来源和预先设计的标注批次上确定，不能继续用当前开发分数决定补多少图。主指南既定规模和验收不因这条定位支线缩小。

还需区分部件定位和关系证据：**前板框不等于容器内部空间**，无法单独证明In；**夹指框不等于物体已被持有**，Holding还需物体与夹指的相对状态。后续关系研究应先明确两端点及必要视觉区域，再用预测信息验证；不能把GT框或仿真坐标当作正式推理输入。

本轮到此封存：不自动启动新检测头、第三分辨率、阈值搜索或训练。下一步的准备重点是独立标签审核、未见来源和固定定位工具链；原有关系/世界模型主线保留，待端点证据充分后接回。

[完整逐集合/分辨率/种子结果](REPORT.md) · [逐候选证据](../../experiments/{NAME}/results/candidate_trace.json) · [封存清单](../../experiments/{NAME}/configs/delivery_manifest.json)
'''
 (R/'STAGE_SUMMARY.md').write_text(summary)
 (R/'RUN_COMMANDS.md').write_text(f'''# 已执行过程

本轮所有操作为CPU保存输出的后处理；执行时显式设置CUDA_VISIBLE_DEVICES为空。原始图仅在报告叠框时读取，解码阶段不读标签或场景信息；候选输出先冻结再评分。

```bash
source /home/robot/research/projects/wm_symbolic/world_model_env.sh
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/decoder_diag_src/prepare.py
CUDA_VISIBLE_DEVICES='' python /home/robot/research/projects/wm_symbolic/experiments/{NAME}/source/decode.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/decoder_diag_src/evaluate.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/decoder_diag_src/verify.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/decoder_diag_src/report.py
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/decoder_diag_src/seal.py
```

source副本与日志随结果保存。封存后只核对哈希，不在原目录重跑写入脚本。未加载模型权重、未重新提取特征、未运行模型前向或优化器；无GPU采集/训练进程。原版和唯一修正版均以相同score0.5、NMS0.3、匹配IoU0.5评价；没有按GT删预测或搜索更多解码版本。
''')
 shutil.copy2(__file__,E/'source/report.py');print('DECODER_REPORT_AND_STAGE_SUMMARY_READY',flush=True)
if __name__=='__main__':main()
