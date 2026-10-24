from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle,Patch
import shutil,csv,collections

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fp=FontProperties(fname=FONT)
plt.rcParams.update({'font.family':fp.get_name(),'font.size':11,'axes.unicode_minus':False,'savefig.dpi':150})
COLORS={224:'#3478b9',448:'#de7b27'}
PART={'drawer_front':'#11a66b','gripper_finger':'#d42ac2'}
CN={'drawer_front':'抽屉前板','gripper_finger':'夹指','train':'A/B训练','AB_episode_heldout_development':'A/B源episode留出开发','C_seen_diagnostic':'C已见诊断','D_seen_diagnostic':'D已见诊断'}

def main():
 assert read(E/'results/verification.json')['passed'];check_manifest(E/'configs/evaluation_frozen.json')
 evaluation=read(E/'results/evaluation.json');groups=evaluation['groups'];rows=read(E/'data/public/images.json');anns=read(E/'data/annotations/final.json')['annotations'];pred={(x['resolution'],x['seed'],x['image_id']):x['predictions']for x in read(E/'results/decoded_predictions.json')}
 def g(res,seed,split,cat,camera=None):
  dims='split/category'if camera is None else'split/camera/category'
  return next(x for x in groups if x['resolution']==res and x['seed']==seed and x['split']==split and x['category']==cat and x['dimensions']==dims and(camera is None or x['camera']==camera))
 def cell(x):return f"{x['TP']}/{x['FP']}/{x['FN']}；{x['mean_best_IoU_including_misses']:.3f}"
 def table(head,rr):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in rr)
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for row,split in enumerate(SPLITS[:2]):
  for col,cat in enumerate(CATS):
   ax=axes[row,col]
   for i,(res,seed)in enumerate([(224,0),(224,1),(448,0),(448,1)]):
    x=g(res,seed,split,cat);ax.bar(i,x['recall'],color=COLORS[res],edgecolor='white',hatch='//'if seed else None,width=.65);ax.text(i,x['recall']+.025,f"{x['TP']}/{x['GT_boxes']}",ha='center',fontsize=11)
   ax.set_xticks(range(4),['224\n种子0','224\n种子1','448\n种子0','448\n种子1']);ax.set_ylim(0,1.15);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('已知可见框召回率（TP/GT）');ax.set_title(f'{CN[split]} · {CN[cat]}');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.suptitle('输入分辨率对照：训练能定位大部分框，留出仍大量漏检',fontsize=17,y=.99);fig.text(.5,.015,'蓝=224；橙=448。柱上为命中/标注框；固定score≥0.5、IoU≥0.5。两种子分别报告，未筛选。',ha='center',fontsize=11);fig.tight_layout(rect=[0,.045,1,.95]);fig.savefig(R/'figures/recall_train_development.png');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for ax,split in zip(axes,SPLITS[:2]):
  for i,(camera,seed)in enumerate([(0,0),(0,1),(1,0),(1,1)]):
   for j,res in enumerate([224,448]):
    x=g(res,seed,split,'gripper_finger',camera);pos=i+(j-.5)*.35;ax.bar(pos,x['recall'],width=.32,color=COLORS[res]);ax.text(pos,x['recall']+.025,f"{x['TP']}/{x['GT_boxes']}",ha='center',fontsize=9)
  ax.set_xticks(range(4),['静态\n种子0','静态\n种子1','腕部\n种子0','腕部\n种子1']);ax.set_ylim(0,1.16);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('夹指召回率（TP/GT）');ax.set_title(CN[split]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.legend(handles=[Patch(color=COLORS[r],label=str(r))for r in[224,448]],loc='upper center',ncol=2,bbox_to_anchor=(.5,1));fig.text(.5,.015,'静态夹指约1个224网格patch宽；腕部约5个。更大的腕部目标在留出来源仍失败。',ha='center');fig.tight_layout(rect=[0,.055,1,.91]);fig.savefig(R/'figures/finger_by_camera.png');plt.close(fig)
 for camera,ids in [(0,[0,16,26]),(1,[1,17,27])]:
  fig,axes=plt.subplots(3,5,figsize=(16,10))
  for rr,i in enumerate(ids):
   r=rows[i]
   with np.load(r['path'])as z:rgb=z[r['key']]
   for col,key in enumerate([None,(224,0),(448,0),(224,1),(448,1)]):
    ax=axes[rr,col];ax.imshow(rgb,interpolation='nearest');ax.set_axis_off()
    for a in anns[i]['classes']:
     for box in a['boxes']:
      x0,y0,x1,y1=box;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=PART[a['category']],linestyle='--',linewidth=1.3))
    if key:
     for p in pred[key[0],key[1],i]:
      x0,y0,x1,y1=np.array(p['box'])*[r['width'],r['height'],r['width'],r['height']];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=PART[p['category']],linewidth=1.6))
    ax.set_xlim(-.5,r['width']-.5);ax.set_ylim(r['height']-.5,-.5)
    ax.set_title((f'id{i:02d} '+('训练'if r['split']=='train'else'开发')+'\n仅标注')if key is None else f'{key[0]} · 种子{key[1]}',fontsize=11)
  fig.suptitle(('静态'if camera==0 else'腕部')+'预定示例：相同RGB、相同标注、固定阈值输出',fontsize=16,y=.99)
  fig.text(.5,.018,'绿=前板及把手；粉=逐根夹指。虚线=标注；实线=预测；无线=无输出。腕部前板均未知，其预测不评分。',ha='center',fontsize=11)
  fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(R/f'figures/examples_{"static"if camera==0 else"wrist"}.png');plt.close(fig)

 cost=read(E/'results/features.json')['runs'];headcost=read(E/'results/head_compute_benchmark.json')['runs'];train=read(E/'results/training.json')['runs'];oldtrain=read(BASE/'results/training.json')['runs'];audit=read(E/'results/GT_center_audit.json');strata=read(E/'results/stratified_diagnostic.json')
 mainrows=[[CN[cat],res,cell(g(res,0,SPLITS[1],cat)),cell(g(res,1,SPLITS[1],cat))]for cat in CATS for res in(224,448)]
 trainrows=[[CN[cat],res,cell(g(res,0,'train',cat)),cell(g(res,1,'train',cat))]for cat in CATS for res in(224,448)]
 camrows=[[CN[split],('静态'if camera==0 else'腕部'),res,cell(g(res,0,split,'gripper_finger',camera)),cell(g(res,1,split,'gripper_finger',camera))]for split in SPLITS[:2]for camera in(0,1)for res in(224,448)]
 cdrows=[[CN[split],CN[cat],res,cell(g(res,0,split,cat)),cell(g(res,1,split,cat))]for split in SPLITS[2:]for cat in CATS for res in(224,448)]
 smallrows=[]
 for split in SPLITS[:2]:
  for res in(224,448):
   vals=[next(x for x in strata if x['resolution']==res and x['seed']==s and x['split']==split and x['camera']==0 and x['category']=='gripper_finger'and x['size']=='short_side_less_than1patch224')for s in(0,1)]
   smallrows.append([CN[split],res,*[f"{x['TP']}/{x['GT_boxes']}；{x['mean_best_IoU']:.3f}"for x in vals]])
 missrows=[]
 for res in(224,448):
  for seed in(0,1):
   for split in SPLITS[:2]:
    z=[x for x in audit if x['resolution']==res and x['seed']==seed and x['split']==split and x['category']=='gripper_finger'and not x['matched']];counts=collections.Counter(x['first_center_gate_failure']for x in z)
    missrows.append([res,seed,CN[split],len(z),counts['center_score_below_.5'],counts['center_box_IoU_below_.5'],counts['center_not_peak'],counts['other_decoder_or_competition']])
 costrows=[]
 for r,h,training in zip(cost,headcost,[oldtrain,train]):
  costrows.append([r['resolution'],r['shape'][1],f"{r['raw_token_bytes']/2**20:.2f} / {r['compressed_cache_bytes']/2**20:.2f}",f"{r['encoder_batch8_median_seconds']*1000:.2f}",f"{r['cache_loop_peak_allocated_bytes']/2**20:.2f}",f"{h['median_seconds']*1000:.2f}",f"{h['peak_allocated_bytes']/2**20:.2f}",f"{training[0]['seconds']:.3f} / {training[1]['seconds']:.3f}"])
 unknown=[]
 for res in(224,448):
  for cat in CATS:
   values=[]
   for seed in(0,1):
    zz=[r for r in evaluation['records']if r['resolution']==res and r['seed']==seed and r['category']==cat and r['status']=='uncertain'];values.append(f"{sum(r['pred_count']>0 for r in zz)}/{len(zz)}")
   unknown.append([res,CN[cat],*values])
 report=f'''# 224与448冻结DINO输入分辨率对照

**提高到448未解决夹指定位问题。** A/B留出开发中，静态夹指仍仅0/13、1/13命中，腕部夹指仅3/16、4/16；整体夹指两种子的变化方向不一致。训练静态定位改善，但腕部训练有所退化，不能把更密网格称为定位验收通过。两次448拟合均按600步结束；未调阈值、扩样、增加网络或进入关系阶段。

## 对照如何保持可比

| 元素 | 本轮处理 | 原因与解释边界 |
| --- | --- | --- |
| 图像、标注、来源 | 完全相同40图、91已知框；16训练/16 A/B开发/4 C/4 D；训练4源episode与开发4源episode隔离 | 不新增标签，不修正已冻结框；这些来源历次已见，不是独立测试 |
| 唯一目标因素 | 同一冻结DINO输入224或448，16×16或32×32网格 | 原始静态200×200、腕部84×84不变；升采样没有增加传感器细节 |
| 224复用 | 全40图重算特征逐值一致；各划分归一化输入、目标、loss、输出/参数梯度、初值与采样核验通过；预测及指标再次精确重现 | 复用旧两种子最终权重，本轮没有重复224优化器运行；复用不是拿不同设置当基线 |
| 448拟合 | 同一25,290参数共享头、两种子初始化、每批8图、600步、每训练图300次访问 | 输入patch总量4倍，但图像访问数及正框数不增加；归一化仍只用训练图，规则相同、统计值随分辨率计算 |
| 负patch权重 | 每图每类负项乘(256−正中心数)/(当前patch数−正中心数)，正项保持；224精确走旧算式 | 防止背景总权重因patch增加4倍；常数logit时focal值与梯度总质量相等。真实困难负例分布仍可改变 |
| 回归与解码 | 同一中心格残差参数化、宽高归一化；score≥0.5、3×3峰、NMS IoU0.3、最多1前板/2指 | 网格改密也使中心残差和3×3峰对应的原图范围减半；这是本对照的网格效应，不能单独归因于DINO语义表示 |
| 评价 | A/B开发逐部件TP/FP/FN及所有GT最佳IoU为主；训练拟合、相机分层为辅；C/D只诊断 | 固定IoU≥0.5一对一匹配；没有输出时IoU=0，未达0.5的输出仍计入其实际IoU；未挑种子或checkpoint |

训练沿用AdamW lr0.001、wd0.0001、clip1；损失focal(alpha0.25,gamma2)+5 SmoothL1+2 GIoU。SmoothL1中心偏移仍以格为单位，宽高以图像比例为单位；正框池和每批归一化沿用旧实现。每类负项折算在运行前写入协议，未根据结果调整。

## 主结果：A/B源episode留出开发

表格每格为**TP/FP/FN；全部GT最佳IoU均值**。TP是命中、FP是误检、FN是漏检。前板8个已知框，夹指29个；未知不计负例。

{table(['部件','输入','种子0：TP/FP/FN；IoU','种子1：TP/FP/FN；IoU'],mainrows)}

448夹指种子0由5命中降为3、误检5升为9；种子1由2升为5、误检6降为1。两者不能合并成“已经改善”的判断。前板命中数4/8与3/8不变，仅IoU小幅变化。

![训练与开发召回](figures/recall_train_development.png)

蓝色224、橙色448，斜纹为种子1；横轴为输入/种子，纵轴为固定规则下召回率。柱上TP/GT展示真实分母。上排训练、下排开发；没有误差棒或显著性推断，两个种子不代替独立数据来源。

## 小夹指是否得到解决，腕部大夹指是否仍失败

静态夹指在224网格下短边中位数约1patch，腕部约5patch；448把两者的网格尺度加倍。以下分别报告两个相机，避免整体数字掩盖尺寸差异。

{table(['集合','相机','输入','种子0：TP/FP/FN；IoU','种子1：TP/FP/FN；IoU'],camrows)}

![夹指按相机分解](figures/finger_by_camera.png)

左图训练、右图开发；蓝224、橙448。横轴区分相机和种子，纵轴为夹指召回率，柱上TP/GT。开发静态0/13→0/13及0/13→1/13，说明本预算下的小目标问题未解决；开发腕部5/16→3/16及2/16→4/16，说明问题也不限于小尺度。

沿用上轮的尺度定义，单列**静态短边小于一个224 patch**的框：训练7个、开发5个，以下只是尺寸诊断，没有改变模型阈值或主指标。

{table(['集合','输入','种子0：TP/GT；IoU','种子1：TP/GT；IoU'],smallrows)}

## 训练本身学得如何

{table(['部件','输入','种子0：TP/FP/FN；IoU','种子1：TP/FP/FN；IoU'],trainrows)}

前板训练两种分辨率均8/8。448训练夹指为25/30、28/30，并非完全拟合；静态从10/14、11/14提高到12/14、13/14，腕部从16/16、16/16降至13/16、15/16。不能只展示静态改善。

保存输出的GT中心审计显示：448训练夹指回归框的平均IoU为0.959/0.962，最小值0.904/0.882；但5/2个漏检的中心分数都低于0.5（约0.423–0.499），且它们都是局部峰。这里使用GT中心仅为定位失败环节，不是推理时使用真实位置。**回归在训练标签上已较好拟合，检测输出仍有中心分数不足；同时开发远低于训练，存在明显留出退化。** 不能称全部是欠拟合，也不能称训练已经完美而只剩泛化。

下表把每个漏检在其监督中心按“分数→回归IoU→局部峰”的顺序归类；它是约定的诊断优先级，不是唯一因果分解。错误中心附近也可能产生其他预测，完整逐GT数据保留。

{table(['输入','种子','集合','夹指漏检','中心分数<0.5','中心框IoU<0.5','非局部峰','其他竞争'],missrows)}

## 预先指定的可视化样本

示例固定为训练id0/1、首个开发状态id16/17及开发id26/27，选择发生在新推理前；不按结果挑图。像素坐标以左上角为原点，向右/下递增。绿框是前板及把手，粉框是逐根可见夹指；虚线为旧冻结标注，实线为模型预测。左列仅标注，之后依次224/448种子0、224/448种子1。无实线表示未输出；腕部前板未知，其预测既不是正确框，也不作为FP。

按冻结顺序的第一个开发状态A0134/1540629（id16/17），448两种子补上了静态前板，但静态边缘夹指仍未输出；腕部均只命中两指之一，种子0还多一个误检。影响下游的最早错误仍在定位环节，不能因为这个状态的前板恢复就认为端点已齐备。

![静态固定示例](figures/examples_static.png)

![腕部固定示例](figures/examples_wrist.png)

## 计算与存储成本

同一RTX5080 Laptop、batch8、同一进程冻结编码器，先1次热身再固定3次同步测量，报告中位数。编码峰值是PyTorch allocated值，非整机显存。头的成本测量仅固定初始权重的前向/反向，无优化器更新；实际训练只新增两次448。

{table(['输入','patch/图','40图特征MiB（原始/压缩）','编码batch8 ms','编码峰值MiB','头前后向batch8 ms','头峰值MiB','实际600步秒（种子0/1）'],costrows)}

编码单批耗时比约{cost[1]['encoder_batch8_median_seconds']/cost[0]['encoder_batch8_median_seconds']:.2f}倍，编码峰值约{cost[1]['cache_loop_peak_allocated_bytes']/cost[0]['cache_loop_peak_allocated_bytes']:.2f}倍；特征元素数量4倍。40图缓存循环（不含模型加载和压缩）为{cost[0]['cache_loop_seconds_excluding_model_load_and_compression']:.3f}/{cost[1]['cache_loop_seconds_excluding_model_load_and_compression']:.3f}秒。头很小，毫秒级测量受启动/CPU调度/频率影响，三次采样不足以宣称448训练更快；224实际训练时间来自原封存运行，不能视为同一时刻的公平测速。448实际训练峰值为{train[0]['peak_allocated_bytes']/2**20:.2f}MiB。完整原始计时见[编码成本](../../experiments/{NAME}/results/features.json)与[头成本](../../experiments/{NAME}/results/head_compute_benchmark.json)。

## C/D与未知区域：仍只作诊断

{table(['集合','部件','输入','种子0：TP/FP/FN；IoU','种子1：TP/FP/FN；IoU'],cdrows)}

C/D各只有4图，历史已见，不能当独立泛化测试。全数据前板20张腕部图均未知，夹指4张静态图未知；未知上的输出覆盖如下，不是误检率。

{table(['输入','未知类别','种子0：有预测/未知图','种子1：有预测/未知图'],unknown)}

完全不可见not_visible实例仍为0，所以缺失拒识仍无法评价。标注来自同助手RGB复核，重复一致性不等于独立准确率；没有用未知标签补负例。

## 判断与最小下一步证据

本次有界对照只支持：**保持标注、预算和规则，提高输入到448不足以解决可用定位；不能认定分辨率是唯一或主要瓶颈。** 也不能排除更密网格在其他预算下有作用，本轮未做这个实验。训练静态改善包含更密中心格/更小原图峰邻域的效应，不能当作DINO语义能力单独改善。

最小已有证据是448训练仍漏的id4/8/9/13夹指：回归边界与标签很接近，但中心分类置信不足；较大腕部也出现同样现象，开发则更严重。这把下一步问题缩小到“在已确认标签下，当前单中心监督/分类表示能否稳定给出足够分数”，而不是继续增加输入尺寸。

如继续研究，先独立复核这些失败框、部件定义和单中心目标是否合理；确认后，最多设计一次单独改变中心监督方式的固定对照，保持数据、网络、分辨率和解码不变。当前证据尚不能保证这种改动有效，也不能据此自动开启训练或调低阈值。本轮未执行该建议、未选定更优分辨率部署；进入关系读取前，仍缺可靠定位与不可见/独立来源评价证据。

## 交付与核验

独立重算320个类别图像记录中的672个TP/FP/FN单元、364个跨分辨率/种子GT评分；224所有预测及指标精确重现，旧四个交付逐文件哈希不变。两次448各600步，无优化器重试，无新数据/模型下载、DINO/T/SupCon更新或第三分辨率。

- [运行前协议](../../experiments/{NAME}/configs/protocol.json)、[严格224复用与等权检查](../../experiments/{NAME}/results/preflight_and_reuse.json)。
- [训练记录](../../experiments/{NAME}/results/training.json)、[独立核验](../../experiments/{NAME}/results/verification.json)、[全量评估](../../experiments/{NAME}/results/evaluation.json)。
- [逐图CSV](localization_by_image.csv)、[分组CSV](localization_by_group.csv)、[逐GT配对变化](paired_GT_comparison.csv)、[尺度分层](stratified_diagnostic.csv)、[GT中心审计](GT_center_audit.csv)。
- [运行说明](RUN_COMMANDS.md)、[封存清单](../../experiments/{NAME}/configs/delivery_manifest.json)。

这是本轮定位对照完成，不是主指南的世界模型、TS后继或闭环成功。
'''
 (R/'REPORT.md').write_text(report)
 (R/'RUN_COMMANDS.md').write_text(f'''# 本轮运行记录

协议冻结前代码暂存于本机任务的resolution_box_src，prepare.py创建独立目录并复制源码。prepare/features/freeze/train/evaluate依次执行，之后verify/report/seal只处理保存输出。环境为本机独立world_model_env.sh，不涉及robot_state、ROS或单词本应用。

```bash
source /home/robot/research/projects/wm_symbolic/world_model_env.sh
# 下列主脚本已依次执行；封存后禁止在原目录重跑写结果的程序。
python /home/robot/Documents/Codex/2026-10-13/task-4/resolution_box_src/prepare.py
python /home/robot/research/projects/wm_symbolic/experiments/{NAME}/source/features.py
python /home/robot/research/projects/wm_symbolic/experiments/{NAME}/source/freeze.py
python /home/robot/research/projects/wm_symbolic/experiments/{NAME}/source/train.py
python /home/robot/research/projects/wm_symbolic/experiments/{NAME}/source/evaluate.py
python /home/robot/Documents/Codex/2026-10-13/task-4/resolution_box_src/verify.py
python /home/robot/Documents/Codex/2026-10-13/task-4/resolution_box_src/report.py
python /home/robot/Documents/Codex/2026-10-13/task-4/resolution_box_src/seal.py
```

特征提取与训练分别持有同一runs/gpu.lock，串行使用GPU。RGB编码器禁止读取标注；训练禁止访问评估、原RGB和标注原件。每种子采样600批×8图，对每张训练图访问300次。448只进行2次优化器拟合；224复用已核验旧权重。GPU基准测量无优化器；不构成额外模型拟合。全部日志随实验封存。

归一化在各自分辨率的16张训练图及其patch上拟合，其他集合仅应用。未知类别整图忽略；没有完全不可见负例，不能把空框未知导出成负例。单中心目标、解码和阈值均在新推理前冻结。

评估后GT中心审计和尺寸分层只读取保存输出；不替换推理、不修改解码、不选种子。没有第三输入尺寸、额外网络或追加训练。后续建议未执行。
''')
 shutil.copy2(__file__,E/'source/report.py')
 print('RESOLUTION_REPORT_AND_FIGURES_READY',flush=True)
if __name__=='__main__':main()
