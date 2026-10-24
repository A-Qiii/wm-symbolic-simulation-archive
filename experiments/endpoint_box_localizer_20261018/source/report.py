from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import shutil,csv,collections
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':9})
SPLITS=['train','AB_episode_heldout_development','C_seen_diagnostic','D_seen_diagnostic'];SN=['A/B训练','A/B源episode留出开发','C已见诊断','D已见诊断']
def table(cols,rows):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in rows)+'\n'
def main():
 assert read(E/'results/verification.json')['passed'];rows=read(E/'data/public/images.json');anns=read(E/'data/annotations/final.json')['annotations'];ev=read(E/'results/evaluation.json');G={(r['seed'],r['split'],r['category']):r for r in ev['groups']if r['dimensions']=='split/category'};pred={(r['seed'],r['image_id']):r['predictions']for r in read(E/'results/decoded_predictions.json')};cols=['#00b977','#d82daa']
 for page in range(5):
  fig,axs=plt.subplots(4,2,figsize=(10,15))
  for ax,r in zip(axs.flat,rows[page*8:(page+1)*8]):
   ax.imshow(np.load(r['path'])[r['key']],interpolation='nearest');status=[]
   for cl,col in zip(anns[r['image_id']]['classes'],cols):
    status.append(('前板'if cl['category']=='drawer_front'else'夹指')+':'+cl['status'])
    for b in cl['boxes']:x0,y0,x1,y1=b;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=col,lw=1.3))
   ax.set(title=f"id{r['image_id']:02d} {r['scene']} {'静态'if r['camera']==0 else'腕部'} | "+' / '.join(status),xticks=[],yticks=[])
  fig.suptitle(f'最终复核标注 {page+1}/5：绿色=活动前板及把手；粉色=逐根可见夹指',fontsize=12);fig.text(.5,.008,'仅可见边界；partial表示该类别至少一处被遮挡或出框。uncertain不画框、训练不作负例。本助手逐图标注，不是独立人工金标准。',ha='center',fontsize=8);fig.tight_layout(rect=(0,.025,1,.975));fig.savefig(R/f'figures/final_annotations_page_{page+1}.png',dpi=140);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(10,4.5));x=np.arange(4)
 for ax,cat,title in zip(axs,CATS,['抽屉前板','夹指']):
  for seed,color in [(0,'#377eb8'),(1,'#ef942c')]:
   vals=[100*G[seed,s,cat]['recall']for s in SPLITS];offset=-.17 if seed==0 else .17;ax.bar(x+offset,vals,.32,color=color,label=f'种子{seed}')
   for j,v in enumerate(vals):r=G[seed,SPLITS[j],cat];ax.text(j+offset,v+1,f"{r['TP']}/{r['GT_boxes']}",ha='center',fontsize=8)
  ax.set(title=title,xticks=x,xticklabels=['训练','A/B留出','C已见','D已见'],ylim=(0,113),ylabel='IoU≥0.5的一对一框召回率（%）');ax.legend();ax.grid(axis='y',alpha=.2)
 fig.text(.5,.015,'蓝/橙为两个固定种子；柱上为成功匹配框数/已知可见框数。未知前板未计入分母，另报覆盖；C/D各仅2根，均已见，不是独立测试。',ha='center',fontsize=8);fig.tight_layout(rect=(0,.07,1,1));fig.savefig(R/'figures/localization_recall.png',dpi=150);plt.close(fig)
 chosen=[0,16,17,26];fig,axs=plt.subplots(4,3,figsize=(11,13))
 for ri,i in enumerate(chosen):
  r=rows[i];rgb=np.load(r['path'])[r['key']]
  for j in range(3):
   ax=axs[ri,j];ax.imshow(rgb,interpolation='nearest');
   for ci,cl in enumerate(anns[i]['classes']):
    for b in cl['boxes']:x0,y0,x1,y1=b;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=cols[ci],lw=1,ls='--'))
   if j>0:
    for p in pred[j-1,i]:
     b=np.array(p['box'])*[r['width'],r['height'],r['width'],r['height']];x0,y0,x1,y1=b;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=cols[CATS.index(p['category'])],lw=2));ax.text(x0,y0,f"{p['score']:.2f}",color='black',fontsize=8,bbox={'fc':'white','alpha':.8,'pad':.3})
   ax.set(title=f"id{i} "+('复核标注'if j==0 else f'种子{j-1} 固定阈值预测'),xticks=[],yticks=[])
 fig.suptitle('已冻结输出的成功与失败定位：虚线标注，实线预测',fontsize=13);fig.text(.5,.012,'颜色区分类别：绿前板、粉夹指。缺实线表示未输出该目标；数值是未校准分数。id0是训练邻近夹指筛选案例；id16/17为首个留出漏检状态；id26为已复核开发例。',ha='center',fontsize=8);fig.tight_layout(rect=(0,.04,1,.97));fig.savefig(R/'figures/prediction_examples.png',dpi=140);plt.close(fig)
 counts=[]
 for split in SPLITS:
  for cam in(0,1):
   rr=[r for r in rows if r['split']==split and r['camera']==cam]
   for ci,c in enumerate(CATS):
    ct=collections.Counter(anns[r['image_id']]['classes'][ci]['status']for r in rr);counts.append({'split':split,'camera':cam,'category':c,'images':len(rr),'boxes':sum(len(anns[r['image_id']]['classes'][ci]['boxes'])for r in rr),**{k:ct[k]for k in ['visible','partial','not_visible','uncertain']}})
 with(R/'annotation_coverage.csv').open('w')as f:wr=csv.DictWriter(f,fieldnames=list(counts[0]));wr.writeheader();wr.writerows(counts)
 quality=read(E/'results/annotation_quality_gate.json');train=read(E/'results/training.json');audit=read(E/'results/failure_audit.json');ver=read(E/'results/verification.json')
 txt='''# 小规模RGB框标注与冻结DINO定位基线

本轮已完成标注协议、40张RGB的逐图标注与复核，以及一个定位基线的两个固定种子训练。结果不足以进入紧致裁剪的对象对关系阶段：A/B源episode留出开发集的前板召回率只有50%/37.5%，夹指17.2%/6.9%；腕部前板标注仍不可靠，完全不可见负例也缺失。未采用模型、未追加训练或改阈值。

## 目标边界与标注含义

'''
 txt+=table(['目标／状态','本轮定义','为何如此／限制'],[
 ['drawer_front','活动抽屉的可见前板及附着把手；不包含内腔底板、侧板、固定柜体或柜门','明确是部件定位，不是活动抽屉整体。把手单点不能替代容器结构，前板成功也不能直接证明In'],
 ['gripper_finger','每根可辨认机械夹指单独框，最多两根；只标主掌部以下的可见指体/指尖','排除腕部、机械臂及被夹物；无物理左右指身份，不声称跨相机实例关联已解决'],
 ['visible','该类别可见部件边界可辨认','紧致RGB整数xyxy框；不是仿真投影或隐藏部分补全'],
 ['partial','该类别至少一处可见部件被遮挡或越过图像边界','记录occluded、out_of_frame等原因；框只包可见部分，类别图像级状态不等于每根指都同样遮挡'],
 ['not_visible','经RGB审核能确认无可见目标部件','协议保留此类，但本样本没有可靠实例；不可见误检率无法估计'],
 ['uncertain','无法可靠辨认类别或界定可见边界','不画框、该类别整图训练/评分掩码为未知；不把未知当负例或用固定ROI回填']])
 txt+='标注者为本助手：实际查看全部40图，再查看叠框页，纠正id10/id26夹指坐标录入错误；保留首轮及修正记录。预定8图在不显示旧框的原图页上重复标注，属于同一助手重复一致性，不是独立双人盲标或人工金标准。\n\n'
 txt+=table(['冻结质量检查','结果','意义'],[['状态重复一致率',f"{quality['status_agreement']:.4f}（15/16）",'唯一分歧为id30夹指visible/partial；由RGB复核修正可见性，门槛保留修正前结果'],['20个重复框IoU',f"中位数{quality['median_box_IoU']:.3f}；P10 {quality['p10_box_IoU']:.3f}",'超过预定0.70/0.50，仅支持自身标注可复现'],['训练有效标注','前板8框、夹指30框，各覆盖4个源episode','训练阶段按类别掩码未知；未把自动预测当标签'],['训练未知比例','前板50%；夹指6.25%','达到而未放宽预定上限；腕部前板能力未验证']])
 txt+='[最终标注JSON](../../experiments/endpoint_box_localizer_20261018/data/annotations/final.json)、[首轮](../../experiments/endpoint_box_localizer_20261018/data/annotations/round1.json)、[坐标修正](../../experiments/endpoint_box_localizer_20261018/data/annotations/review_changes.json)、[重复标注](../../experiments/endpoint_box_localizer_20261018/data/annotations/repeat.json)全部保留，可与原图逐项检查。\n\n'
 txt+='最终叠框页：[1](figures/final_annotations_page_1.png)、[2](figures/final_annotations_page_2.png)、[3](figures/final_annotations_page_3.png)、[4](figures/final_annotations_page_4.png)、[5](figures/final_annotations_page_5.png)。绿色为前板加把手，粉色为逐根夹指；页面明示未知与遮挡。原始无框图和网格也在同一figures目录。\n\n'
 txt+='## 先划分和冻结，再训练\n\n'
 txt+=table(['集合','来源','本轮图像','角色'],[
 ['A/B训练','A0125、A0137、B0025、B0032；每episode按帧排序取第8/23个既有状态','8状态×2相机=16图','训练定位头及计算特征归一化'],
 ['A/B留出开发','A0134、A0141、B0053、B0063；同一预定取样规则','8状态×2相机=16图','当前定位训练未用；历次已查看，不能称独立测试'],
 ['C已见诊断','C_intermediate_v3_blue_2、C_intermediate_v3_pink_3，各取既有记录中点','2状态×2相机=4图','历史根诊断；不推断独立源episode统计'],
 ['D已见诊断','D_identity_blue_pink_01、D_identity_blue_pink_00，各取既有记录中点','2状态×2相机=4图','反复使用来源，绝非新的独立测试']])
 txt+='A/B按官方源episode隔离，跨集合根ID和精确RGB内容均无重合。20状态40图、样本索引、定义、质量门槛、训练预算、解码与评价方法均在本轮标注和训练前冻结；未因验证结果补图、改样本或标签。标注复核中的两处坐标和一处可见性修正均发生在训练前。该小预算仅服务本轮定位诊断，没有替代主指南的数据规模和完整验收。\n\n'
 txt+='## 模型与评价口径\n\n'
 txt+=table(['元素','配置','作用／限制'],[
 ['共享视觉编码器','已缓存DINOv2-small全程冻结；当前RGB缩放224，输出16×16×384','定位阶段使用全图上下文；不是局部裁剪特征。未读取robot_obs、仿真分割、真实坐标或任务谓词'],
 ['定位头','每patch共享384→64 SiLU→10，25,290参数；2类别中心分数和每类4个框参数','无显式XY、场景、相机输入或固定搜索ROI；DINO位置/全图上下文仍可能携带布局先验'],
 ['监督','人工RGB框中心与可见边界；未知类别整图忽略','框只作训练目标和评分，不进入模型特征'],
 ['预算','一个结构，种子0/1各600更新，batch8，训练16图随机排列300轮','AdamW lr0.001、wd0.0001、clip1；无增强、无早停、只取最终权重；DINO和T均不训练'],
 ['损失','中心focal（alpha0.25、gamma2）+5×框SmoothL1+2×GIoU loss','未搜索损失权重或阈值'],
 ['固定解码','分数≥0.5、3×3局部峰、同类NMS IoU0.3；最多1前板/2夹指','无框时返回未定位，不回填常量框；分数未校准'],
 ['评价','IoU≥0.5的按分数贪心一对一匹配；TP、FP、FN和全部GT最佳IoU','未输出时IoU记0；均值包含漏检，不只统计成功框。未知另报预测覆盖，不按负例算FP']])
 txt+='## 主结果：训练拟合不代表定位泛化\n\n表内TP/FP/FN均以框为单位；IoU均值包含未输出的0，不以成功样本筛选分母。\n\n'
 trs=[]
 for split,name in zip(SPLITS,SN):
  for cat,title in zip(CATS,['前板','夹指']):
   rr=[G[s,split,cat]for s in(0,1)];trs.append([name,title,rr[0]['GT_boxes']]+[f"{r['TP']}/{r['FP']}/{r['FN']}; IoU {r['mean_best_IoU_including_misses']:.3f}"for r in rr])
 txt+=table(['集合','部件','已知GT框','种子0 TP/FP/FN；均值IoU','种子1 TP/FP/FN；均值IoU'],trs)
 txt+='![定位召回率](figures/localization_recall.png)\n\n蓝/橙表示种子0/1；纵轴为已知可见框的召回率，柱上给出TP/GT分母。前板训练8/8并未延续到开发；C/D样本极少且已见，不能据此估计广泛跨场景泛化。前板FP为0伴随大量漏检和未知样本，不能理解为安全可靠。\n\n'
 txt+='**A/B开发按场景、相机与部件分解**：TP/FP/FN仍为框数；“—”表示无已知标签，绝不是零错误。\n\n'
 trs=[]
 for scene in['A','B']:
  for cam in(0,1):
   for cat,title in zip(CATS,['前板','夹指']):
    rr=[next(r for r in ev['groups']if r['dimensions']=='split/scene/camera/category'and r['split']=='AB_episode_heldout_development'and r['seed']==s and r['scene']==scene and r['camera']==cam and r['category']==cat)for s in(0,1)];trs.append([scene,'静态'if cam==0 else'腕部',title,f"{rr[0]['known_class_images']}/{rr[0]['unknown_class_images']}"]+['—'if not r['known_class_images']else f"{r['TP']}/{r['FP']}/{r['FN']}"for r in rr])
 txt+=table(['场景','相机','部件','已知/未知图数','种子0','种子1'],trs)
 txt+='完整C/D分相机、可见性及遮挡原因的统计在[分组CSV](localization_by_group.csv)；[逐图CSV](localization_by_image.csv)保留每项预测和缺失，未把相关帧当作独立样本计算显著性。\n\n'
 txt+='## 覆盖缺口与第一个影响结果的错误\n\n全40图中，前板20图已知、20图不确定（全部腕部），夹指36图已知、4图不确定；共91个已知框。没有经核验的完全不可见负例，所以不能报告“目标不在画面时的误检率”。未知前板上仍输出框的图数为种子0的20/20、种子1的12/20；这些不是经过评分的FP，也不是正确定位，显示目前不能把模型输出当成可用端点。夹指未知图分别有3/4、2/4输出。\n\n'
 txt+='按预定图像顺序，A/B开发首个失败状态是A0134/1540629（id16静态、id17腕部）：静态前板和边缘可见夹指未输出，腕部两指也未输出。这里尚未调用任何关系头，因此错误首先发生在定位环节，不能归咎于In/Holding读出。\n\n'
 txt+='![预测示例](figures/prediction_examples.png)\n\n虚线为复核标注，实线为预测；绿前板、粉夹指。左列只显示标注，中/右列是两个种子固定阈值输出。id0展示训练中的邻近夹指漏检；id16/17是首个留出失败状态；id26是预先标注并复核过的开发例。没有实线即未输出，未知前板的预测不可当正确框。\n\n'
 txt+='对保存输出的只读诊断发现两个不同问题。训练id0的夹指中心分数0.750、该点回归框IoU0.973，但被3×3局部峰筛选删除。只替换这一筛选步骤、保持分数阈值/NMS/数量上限不变，训练夹指TP从26/27增加到29/30；A/B开发TP仍是5/2，种子0还多1个FP。因此这个解码问题不能解释开发集的主要退化，替换结果仅作诊断，未用于主结果或模型采用。\n\n'
 txt+='对A/B开发漏检在GT中心的审计：前板4/5个漏检全部中心分数低于0.5；夹指24/27个漏检中21/25个中心分数低于0.5，其余包括框偏移与峰筛选。中心审计使用GT仅定位分析位置，不是替代推理；它也不能单独证明数据量、覆盖、特征分辨率或模型容量中的哪一项是根因。没有继续调阈值、补样或重训。\n\n'
 txt+='[单项解码诊断CSV](decode_diagnostic.csv)与[逐GT失败审计](failure_by_GT.csv)保留证据。下一步不能直接进入对象对关系：需要先由复核者确认标注定义/边界，明确腕部前板是否有足够可见证据，并补齐可靠不可见与困难负例的定位验证设计；这些扩展本轮未执行。真正独立评价仍须使用历次未训练、未查看、未推理的新源episode/初始化谱系，先冻结排除清单；当前所有集合均不能改称独立测试。\n\n'
 txt+='## 可复核交付\n\n'
 txt+=f"两个定位训练均实际完成600更新，循环合计{sum(r['seconds']for r in train['runs']):.3f}秒；仅小头训练，特征缓存另计。独立复核{ver['class_image_records']}个类别图像记录、{ver['independently_checked_TP_FP_FN_cells']}个TP/FP/FN单元和182个跨种子GT框评分。归一化只使用16张训练图，未知梯度屏蔽、框编码还原、中心无碰撞及来源隔离检查通过。旧封存实验逐文件哈希不变。\n\n"
 txt+='- [冻结协议](../../experiments/endpoint_box_localizer_20261018/configs/protocol.json)、[图像来源与划分](../../experiments/endpoint_box_localizer_20261018/data/public/images.json)、[质量门槛结果](../../experiments/endpoint_box_localizer_20261018/results/annotation_quality_gate.json)。\n- [最终标注](../../experiments/endpoint_box_localizer_20261018/data/annotations/final.json)、[标注覆盖CSV](annotation_coverage.csv)。\n- [训练记录](../../experiments/endpoint_box_localizer_20261018/results/training.json)、[验证记录](../../experiments/endpoint_box_localizer_20261018/results/verification.json)、[评估全量JSON](../../experiments/endpoint_box_localizer_20261018/results/evaluation.json)。\n- [运行与复核说明](RUN_COMMANDS.md)、[交付清单](../../experiments/endpoint_box_localizer_20261018/configs/delivery_manifest.json)。\n\n没有恢复SupCon，没有下载、采集、部署或替换正式模型。当前结果是定位诊断失败与可复核标注，不是主指南世界模型或闭环成功。\n'
 (R/'REPORT.md').write_text(txt)
 (R/'RUN_COMMANDS.md').write_text('''# 已执行命令

在独立world_model环境中，依次运行新实验source中的prepare.py、sheets.py、annotate_round1.py、overlays.py、review.py、quality.py、features.py、freeze.py、train.py、evaluate.py、diagnose.py、verify.py、report.py。最后seal.py封存。最初脚本由本任务endpoint_box_src暂存目录启动，训练和评估使用冻结source副本。原实验只读。

训练仅两次，种子0/1各600更新；train.py拒绝已有training_attempts.json，禁止训练时访问评估数据及标注原件。图像编码器仅接受双RGB代理数据，禁止编码时读标注。source_references、pre_annotation_frozen、annotations_frozen、training_frozen和final_heads_frozen记录阶段哈希。

所有主结果使用固定score0.5、3×3局部峰、NMS0.3和每类数量上限。diagnose.py唯一额外项是对保存输出去掉局部峰的只读单项诊断，没有运行新模型或替换主结果。没有优化器/模型重试。日志在实验logs目录；冻结检查中的Tensor转标量警告发生于未知类零梯度检查，不是训练失败。

封存后不要在原目录重跑写结果的脚本；只读核验交付清单及已有输出。最终标注是助手RGB审阅结果，需复核者检查，不应标称独立人工金标准。原始数据未改，坐标是原图RGB像素xyxy上界不含；未知类不能按COCO空框误当负例。
''')
 shutil.copy2(__file__,E/'source/report.py');print('BOX_LOCALIZER_REPORT_AND_FINAL_ANNOTATIONS_READY',flush=True)
if __name__=='__main__':main()
