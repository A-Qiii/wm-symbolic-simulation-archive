from common import *
import shutil,os,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from PIL import Image
CAT_CN={'drawer_front':'抽屉前板及把手','gripper_finger':'夹指'}
STATUS={'visible':'可见','partial':'部分可见','uncertain':'未知','not_visible':'不可见'}
ACTION={'correct_visible_extent':'修正坐标','exclude_unresolved_identity':'排除：身份未决','exclude_unresolved_extent':'排除：范围未决','retain_unknown':'保留未知','include_identifiable_visible_parts':'新纳入可识别部件','retain_original':'保留原框'}
def rel(p):return os.path.relpath(p,R)
def link(p,label):return f'[{label}]({rel(p)})'
def table(headers,rows):return '\n'.join(['| '+' | '.join(map(str,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v).replace('|','/')for v in r)+' |'for r in rows])
def fmt(x):return '—'if x is None else f'{x:.3f}'
def main():
 check_manifest(E/'configs/labels_frozen.json');check_manifest(E/'configs/sensitivity_frozen.json');assert read(E/'results/verification.json')['passed'];assert not(R/'REPORT.md').exists()
 d=read(E/'results/sensitivity.json');ann={x['image_id']:x for x in read(E/'data/labels/adjudicated_eval_v1.json')['annotations']};meta={x['image_id']:x for x in read(E/'data/images.json')};ledger=read(E/'data/labels/decision_ledger.json')['decisions'];den=read(E/'data/labels/denominator_changes.json');pred=read(E/'results/saved_predictions_selected8.json')
 fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':10})
 def raw(i):return np.asarray(Image.open(E/f'data/observations/id{i:02d}.png'))
 fig,axs=plt.subplots(4,2,figsize=(10,18))
 for ax,i in zip(axs.flat,IDS):
  r=meta[i];ax.imshow(raw(i),interpolation='nearest');notes=[]
  for ci,c in enumerate(ann[i]['classes']):
   for b in c['boxes']:
    x,y,x1,y1=b;ax.add_patch(Rectangle((x,y),x1-x,y1-y,fill=False,color=['#19a266','#a42bbe'][ci],linewidth=1.6))
   if c['status']=='uncertain':notes.append(CAT_CN[c['category']]+'：unknown，不计分')
  if notes:ax.text(.02,.98,'\n'.join(notes),transform=ax.transAxes,va='top',fontsize=8,color='#a35200',bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
  ax.set_title(f'id{i:02d} | {r["scene"]} | {"静态"if r["camera"]==0 else"腕部"} | {r["split"]}',fontsize=10);ax.set_xlabel('x 原图像素');ax.set_ylabel('y 原图像素');ax.set_xlim(-.5,r['width']-.5);ax.set_ylim(r['height']-.5,-.5)
 fig.suptitle('冻结后的8图评估标签（没有模型预测）\n绿框＝可见前板连同把手；紫框＝各根夹指；橙字＝未决类别整图忽略',fontsize=13);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(R/'figures/adjudicated_labels8.png',dpi=150);plt.close(fig)
 old={x['image_id']:x for x in read(E/'data/labels/original_selected.json')['annotations']};ind={x['image_id']:x for x in read(E/'data/review/independent_annotations.json')['records']}
 cases=[(0,'gripper_finger',(25,40,100,75),'三处灰白候选中，哪两处是夹指？'),(8,'gripper_finger',(45,105,108,143),'水平白段是否整段属于夹指？'),(1,'drawer_front',(0,48,84,84),'把手上方，哪些棕色像素属于前板？'),(9,'drawer_front',(0,40,84,84),'斜边在右下裁切处的前板范围？')]
 fig,axs=plt.subplots(2,2,figsize=(14,10))
 for ax,(i,c,crop,q)in zip(axs.flat,cases):
  ax.imshow(raw(i),interpolation='nearest');ob=next(x['boxes']for x in old[i]['classes']if x['category']==c);ib=[ind[i]['drawer']['bbox_xyxy']]if c=='drawer_front'else[x['bbox_xyxy']for x in ind[i]['fingers']]
  for boxes,col,ls in[(ob,'#e08013','--'),(ib,'#007caa',':')]:
   for x,y,x1,y1 in boxes:ax.add_patch(Rectangle((x,y),x1-x,y1-y,fill=False,color=col,linestyle=ls,linewidth=2))
  x,y,x1,y1=crop;ax.set_xlim(x-.5,x1-.5);ax.set_ylim(y1-.5,y-.5);ax.set_xlabel('x 原图像素');ax.set_ylabel('y 原图像素');ax.set_title(f'id{i:02d}：{q}',fontsize=11)
 fig.legend(handles=[Line2D([0],[0],color='#e08013',ls='--',label='原标注候选'),Line2D([0],[0],color='#007caa',ls=':',label='独立复核候选')],loc='lower center',ncol=2);fig.suptitle('需要外部部件辨认的4图：两种线型只区分来源，不表示正确与错误\n当前均为unknown；原图裁切、最近邻显示，不补像素、不显示模型输出',fontsize=13);fig.tight_layout(rect=(0,.04,1,.93));fig.savefig(R/'figures/arbitration_cases.png',dpi=160);plt.close(fig)
 def group(res,seed,variant,version,support,split='selected8_all',cat='all_categories',camera='all'):
  return next(g for g in d['groups']if all(g[k]==v for k,v in dict(resolution=res,seed=seed,variant=variant,label_version=version,support=support,split=split,category=cat,camera=camera).items()))
 configs=[(224,0),(224,1),(448,0),(448,1)];fig,axs=plt.subplots(2,2,figsize=(13,8));xs=np.arange(4)
 for row,support in enumerate(['all_known_within_selected8','common_known_class_images']):
  for col,metric in enumerate(['TP','FP']):
   ax=axs[row,col]
   for k,(v,color,name)in enumerate([('original','#8a939c','原标签'),('adjudicated','#129b8a','冻结新标签')]):
    vals=[group(r,s,'corrected',v,support)[metric]for r,s in configs];bars=ax.bar(xs+(k-.5)*.34,vals,.34,color=color,label=name);ax.bar_label(bars,padding=2)
   ax.set_xticks(xs,[f'{r}/s{s}'for r,s in configs]);ax.set_ylim(0,20 if metric=='FP'else 20);ax.set_ylabel(('匹配框数 TP'if metric=='TP'else'未匹配输出数 FP'));ax.set_xlabel('输入分辨率 / 训练种子；均为同一批保存的预测');ax.set_title(('各版全部已知：11→12个类别图，标签框均18个'if row==0 else'共同已知：两版均9个类别图、14个标签框')+'\n'+('TP：越高表示更多参考框被匹配'if metric=='TP'else'FP：只在该版已知类别图中计数'),fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axs[0,0].legend();fig.suptitle('仅8图的标注敏感性：固定保存的去峰值/去数量上限解码输出\n灰＝原标签；绿＝新标签。上排受纳入/排除项影响，下排固定共同计分范围。',fontsize=13);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(R/'figures/label_sensitivity8.png',dpi=160);plt.close(fig)
 frozen=read(E/'configs/labels_frozen.json');report=[]
 report.append('# 8图标注裁决与保存预测的敏感性\n')
 report.append('本轮已完成限定的8图复核：7个“图像—类别”组合修正可见范围，3个组合从unknown纳入，2个保留原框，4个仍为unknown。新版本先冻结，再读取已保存的预测计分；没有训练、模型前向、特征提取、重新解码、下载或新采集。原标注、独立复核文件和全部旧实验均保留。')
 report.append('这些标签是带逐项像素证据的助手裁决，**不是独立人工金标准**。裁决者已有项目历史结果上下文，不能声称盲审；本轮冻结前未读取模型输出文件。原始像素与冻结定义约束了坐标判断，分数没有参与选框。')
 report.append('## 范围与冻结顺序\n')
 report.append(table(['来源','图像ID','原标签已知类别图 / 框数','新标签已知类别图 / 框数','能说明什么'],[['A/B训练','0、1、8、9','6 / 10','4 / 6','仅已拟合图像的标签敏感性'],['A/B开发','无','0 / 0','0 / 0','不能重估开发成绩'],['C已见诊断','32、33','2 / 3','4 / 6','仅这两图，非独立泛化测试'],['D已见诊断','38、39','3 / 5','4 / 6','仅这两图，非独立泛化测试'],['合计','8图、4状态、每状态双相机','11 / 18','12 / 18','同为18框，但计分对象不同']]))
 report.append('选择规则沿用独立复核：每个A/B/C/D场景选字典序最小根的最早状态、双相机；不是随机样本。静态原图200×200，腕部84×84。每图两个待评类别，所以总共16个类别图；known仅指本次可计分，不能理解为真值绝对正确。')
 report.append(f'标签冻结时间：`{frozen["frozen_utc"]}`；新标签SHA256：`{sha(E/"data/labels/adjudicated_eval_v1.json")}`。'+link(E/'configs/protocol.json','预先冻结的规则')+'、'+link(E/'configs/labels_frozen.json','标签冻结清单')+'、'+link(E/'results/verification.json','核验结果')+'可追溯。')
 report.append('## 部件含义与裁决\n')
 report.append(table(['部件','框的含义','排除项与原因'],[['抽屉前板及把手','同一可移动前板和连接把手的可见外接矩形','不把腔底、侧板、固定柜面作为目标；斜部件的外接框可几何上跨过背景'],['夹指','掌部远端每根可识别机械夹指，各一框、无左右身份','不含掌部、腕部或被夹物；无法确定其中一根时整类unknown，避免未标真物体被计FP'],['partial','已识别但被遮挡/画框裁切的可见范围','不要求完整部件，也不补全隐藏长度'],['unknown','身份或可见边界不足以可靠确定','不计TP/FP/FN；不等价于not_visible或负例']]))
 report.append('坐标为原图整数 `[x0,y0,x1,y1]`，右下端不包含。边缘抗锯齿/近端连接仍可能有约1–2像素误差；本次仅做外接框裁决，没有分割真值。'+link(R/'DECISIONS.md','16项逐项坐标、证据与剩余疑点')+'列出两版坐标；'+link(E/'data/labels/decision_ledger.json','机器可读决策账本')+'保留完整说明。')
 report.append('![冻结后的8图标签](figures/adjudicated_labels8.png)\n图中绿框表示抽屉前板连同把手，紫框表示夹指；橙字说明该类unknown并整图忽略。横纵轴是原图像素，y向下。每对图来自同一状态的静态/腕部相机；仅显示标签，无模型输出。它说明哪些实际像素被纳入评估，不能证明预测准确。')
 report.append('明确的修正例：id38右夹指从约y84开始，原框从y94起截掉可见上段，并向粉色物体延伸；新框沿灰条改为`[73,84,84,107]`。id1两根上缘裁切夹指的旧框明显包含左右背景；新框收紧到可见灰条。id32两根窄指条和id33/id39连接把手的前板均可从本图识别，因此新纳入。没有把独立标注整套照抄，也没有通过多数表决处理id0/id8的部件身份问题。')
 report.append('## 分母变化\n')
 report.append(table(['变化','图像—类别','已知类别图变化','参考框变化','原因'],[['移出计分','id0、id8：夹指','−2','−4','部件身份或掌部/指体分界未决'],['新纳入','id32：夹指','+1','+2','窄指体与掌部连接、可见轮廓可辨'],['新纳入','id33、id39：前板','+2','+2','前板与把手及腔底边界可辨'],['仍为unknown','id1、id9：前板','0','0','把手可见，但前板/桌沿或侧板范围未决'],['共同计分范围','4个静态前板＋5个夹指类别图','9固定','14固定','两版共同known，用于隔离坐标变化影响']]))
 report.append('完整逐图分母见'+link(R/'denominator_changes.csv','denominator_changes.csv')+'。总框数18→18只是净数相抵；训练框10→6、C框3→6、D框5→6，不能把两版总召回率直接解释成性能升降。共同范围两版都保留14框，才能比较相同实例集合下的坐标变化。')
 report.append('## 固定预测的两种比较\n')
 report.append('所有表均用同一批旧模型输出，输入分辨率224/448、种子0/1；没有重训。原解码=原有局部峰值及每类数量上限，修正解码=此前已保存的去峰值/去数量上限版本；本轮没有再执行解码。预测分数门槛0.5、已保存NMS IoU门槛0.3，计分按分数顺序一对一IoU≥0.5；已知类别图中所有未匹配输出都计FP。TP/FP/FN依次为匹配数/误检数/漏检数。unknown输出保留记录但不计FP。')
 rows=[]
 for var in['original','corrected']:
  for res,seed in configs:
   a=group(res,seed,var,'original','all_known_within_selected8');b=group(res,seed,var,'adjudicated','all_known_within_selected8');co=group(res,seed,var,'original','common_known_class_images');cn=group(res,seed,var,'adjudicated','common_known_class_images')
   trip=lambda g:f'{g["TP"]}/{g["FP"]}/{g["FN"]}'
   rows.append(['原解码'if var=='original'else'修正解码',f'{res}/s{seed}',trip(a),trip(b),trip(co),trip(cn)])
 report.append(table(['保存的解码版本','分辨率/种子','旧：全部known，18框','新：全部known，18框','旧：共同范围，14框','新：共同范围，14框'],rows))
 report.append('![8图敏感性](figures/label_sensitivity8.png)\n图使用修正解码的保存结果。灰柱=原标签，绿柱=新标签；横轴是分辨率/种子，纵轴是框数。左列TP、右列FP；上排使用各版全部known，下排固定9个共同类别图/14框。FN等于对应参考框数减TP。上排变化混合了纳入/排除与坐标影响；下排显示仅224/s1因id38夹指坐标修正多匹配1框，其余TP/FP数量不变。种子不是独立图像样本，图不用于选最佳配置。')
 report.append('共同范围中唯一跨越匹配门槛的变化发生在id38夹指、224/s1：两种解码均由0TP/2FP/2FN变为1TP/1FP/1FN，平均最佳IoU由0.316到0.529。这是同一预测对不同参考边界的评分变化，不能证明模型已得到改进。其他共同范围的整数计数未变，不等于坐标或IoU完全未变。')
 report.append('新纳入腕部前板暴露了原先不计分的输出。例如修正解码448/s0在id33前板产生12个FP、id39产生5个FP；两图各漏1个参考框。224/s0与s1在id33的最佳IoU分别为0.500264和0.504508，刚超过0.5，1–2像素的标注差异就可能改变匹配；标签已冻结，未据此再改框。因此这些小样本TP不宜作为稳健定位证据。id0、id8夹指现在unknown，旧报告涉及它们的“恢复真框”描述只能视为依赖原标签的诊断，不能再作为已验证部件身份。')
 report.append('各split/类别/相机、两种解码、两版标签、两种计分范围的全部结果见'+link(R/'all_group_metrics.csv','分组表')+'；逐图输出和匹配见'+link(R/'all_image_metrics.csv','逐图表')+'。以下仅列新标签、修正解码的分层计数，避免总数掩盖类别变化。')
 rr=[]
 for res,seed in configs:
  row=[f'{res}/s{seed}']
  for sp in['train','C_seen_diagnostic','D_seen_diagnostic']:
   for cat in CATS:
    g=group(res,seed,'corrected','adjudicated','all_known_within_selected8',sp,cat);row.append(f'{g["TP"]}/{g["FP"]}/{g["FN"]} (n={g["GT_boxes"]})')
  rr.append(row)
 report.append(table(['配置','训练前板','训练夹指','C前板','C夹指','D前板','D夹指'],rr))
 report.append('这里的n是参考框数，不是图像数。训练项只有4图且原权重拟合过旧标签；C/D也是此前见过的诊断根。A/B开发列不存在：选集为0图，不能把0计数当作模型失败或开发分数。')
 report.append('## 仍需外部辨认的具体图\n')
 report.append('![4个未决问题](figures/arbitration_cases.png)\n橙色虚线=原候选，蓝色点线=独立复核候选；颜色只表示来源，不表示对错。横纵轴为原图像素，最近邻裁切放大。id1/id9的原版本unknown，所以只有独立候选框。这些图没有模型输出；需要熟悉夹具/抽屉部件的人指认，而不是再增加助手投票。')
 questions=read(E/'data/labels/external_arbitration_needed.json');report.append(table(['具体图','未决类别','需要确认的事实'],[[q['image_id'],CAT_CN[q['category']],q['question']]for q in questions]))
 report.append('本次有限交付不依赖用户立即回答：四项继续unknown。若要把它们纳入后续训练/评估，才需要用户或领域人员针对上述像素作明确指认，并另存后续标签版本。尤其应统一“掌部远端夹指的连接点”和“前板含把手但排除桌沿/侧板”的示例；当前定义尚未支持所有视角的一致身份判断。')
 report.append('## 证据边界与停止位置\n')
 report.append('本轮支持的结论是：旧标签存在具体坐标截短、过宽以及unknown过保守的情况；少数部件身份争议仍未解决；保存预测的评分对标签与计分覆盖敏感。它不支持全40图标签准确率、模型泛化改善、448优于224、世界模型/规划链通过，亦不能证明新标签是人工真值。**其余32图尚未完成独立标注审计，所有全数据旧分数继续标记为“标签审计未完成、暂定”**，原报告原封保留，本报告仅作补充。')
 report.append('核验完成：512条逐图/类别/配置/标签/范围记录，其中328条可计分记录由另一套NumPy IoU矩阵实现复核；128条旧标签子集记录与旧保存评估逐项完全一致；1440条分组行的计数和分母核对通过。6个旧实验封存清单共2034个文件哈希不变；新标签冻结哈希复查通过。此处“另一套实现”是代码复核，不是另一位标注者。')
 report.append('本轮已停止新实验。没有安装、训练、下载、模型前向或新数据；旧robot_state、ROS、单词本应用未改动。执行没有权限阻塞；后续可靠评估的阻碍是未决部件定义与未完成的其余标签审计。原指南整体仍未完成，本轮不能作为其验收。')
 report.append('复现与文件清单见'+link(R/'RUN_COMMANDS.md','运行说明')+'、'+link(E/'configs/delivery_manifest.json','封存清单')+'。')
 (R/'REPORT.md').write_text('\n\n'.join(report)+'\n')
 dec=['# 逐项裁决账本\n','仅8图；依据原始RGB和冻结定义，在读取保存预测前冻结。没有多数表决；不把模型输出当标注依据。坐标右下端不包含。\n']
 for x in ledger:
  i=x['image_id'];o=x['original'];n=x['adjudicated'];dec.append(f'## id{i:02d} / {CAT_CN[x["category"]]}\n')
  dec.append(table(['项目','值'],[['处理',ACTION[x['action']]],['原状态 / 框',STATUS[o['status']]+' / '+str(o['boxes'])],['新状态 / 框',STATUS[n['status']]+' / '+str(n['boxes'])],['像素证据',x['pixel_evidence']],['剩余疑点',x['remaining_uncertainty']or'未发现足以阻止此范围计分的身份争议；仍非人工金标准。']]))
  dec.append(f'[原图网格裁切](figures/pixels_id{i:02d}.png) · [原图与两套标注对照](figures/review_id{i:02d}.png)\n')
 (R/'DECISIONS.md').write_text('\n\n'.join(dec)+'\n')
 (R/'STAGE_SUMMARY.md').write_text('8图裁决已冻结并重算保存预测的敏感性。7个类别图修正坐标、3个新纳入、2个保留原框、4个unknown。旧/新标签都18框，但known类别图11→12，共同范围固定9类别图/14框。共同范围仅224/s1的id38夹指多匹配1框，其余整数计数不变；新纳入腕部前板后可见先前未计分的FP。4个具体像素问题留待用户/领域人员辨认，不再助手投票。无训练/前向/特征提取/重解码/新数据。其余32图审计未完成，所有全数据旧分数暂定，不能外推8图。6个旧封存实验2034文件哈希不变。本轮停止；原指南整体未完成。\n')
 commands='''# 运行记录

项目环境仍为独立 `world_model_env.sh`。本轮CPU，未调用Torch或GPU。

```bash
source /home/robot/research/projects/wm_symbolic/world_model_env.sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
# 以下脚本均具有不覆盖已有结果的断言；封存后不要在同一目录重跑。
python adjudicate8_src/prepare.py
python adjudicate8_src/adjudicate.py
python adjudicate8_src/score.py
python adjudicate8_src/verify.py
python adjudicate8_src/report.py
python adjudicate8_src/seal.py
```

实际工作目录为 `/home/robot/Documents/Codex/2026-10-13/task-4`；写项目新实验目录的命令获窄范围沙箱授权。源码副本在实验目录 `source/`。首次准备脚本的一个 `else10` 排版语法错误在执行任何任务逻辑前被Python拒绝，修正为 `else 10` 后首次准备成功；原错误日志保留，不涉及标签改动或模型尝试。

复核顺序是准备原图与旧标注、查看全部8图及网格裁切、写裁决与冻结哈希、读取保存预测的8图子集、评分、独立数值核验、生成报告并封存。原解码预测与修正解码预测都来自已封存JSON，不重新调用解码器。重复核验只读执行，不产生新模型。
'''
 (R/'RUN_COMMANDS.md').write_text(commands)
 shutil.copy2(__file__,E/'source/report.py')
 print('EIGHT_IMAGE_REPORT_READY',str(R/'REPORT.md'))
if __name__=='__main__':main()
