from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=FONT.get_name();plt.rcParams['axes.unicode_minus']=False
COLORS={'drawer_front':'#00bcd4','gripper_finger':'#ff8f00'}
def metric(allm,seed,split,cam='all',cat='all'):return next(x for x in allm if x['seed']==seed and x['split']==split and x['camera']==cam and x['category']==cat)
def box(ax,b,color,style='-',lw=1.8):ax.add_patch(Rectangle((b[0]-.5,b[1]-.5),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=color,linewidth=lw,linestyle=style))
def grid(split,rows,anns,preds,matches):
 ix=[r['image_id']for r in rows if r['split']==split and r['root_id'].endswith(('_00','_07'))and r['sample']in[0,4]]
 # Train has A and B. Show A roots 0/7, preserve the same eight-view layout.
 if split=='train':ix=[i for i in ix if rows[i]['scene']=='A']
 assert len(ix)==8
 fig,axes=plt.subplots(8,3,figsize=(13.5,24.5));fig.patch.set_facecolor('#f7f8fa');name={'train':'训练 A（固定样例）','dev':'开发 C','final':'最终留出 D'}[split]
 fig.suptitle(f'{name}｜同步真值与 RGB-only 定位预测',fontsize=20,y=.991)
 fig.text(.5,.971,'列：同步真值 / 固定种子 0 / 固定种子 1；青色＝前板含把手，橙色＝手指＋指尖；实线 GT，虚线预测。',ha='center',fontsize=10)
 fig.text(.5,.957,'原始 RGB 最近邻显示；横纵轴为原图像素（左上为原点）。未知类别不画 GT 框、不计分；黄色点为未知语义像素。',ha='center',fontsize=9)
 for rowindex,i in enumerate(ix):
  r=rows[i];a=anns[i];rgb=np.load(r['path'])[r['key']];w,h=r['width'],r['height'];short=f'{r["scene"]}_{r["root_id"][-2:]} / 状态 {r["sample"]} / '+('静态'if r['camera']=='static'else'腕部');unknown=[('前板'if c==0 else'手指')for c in [0,1]if not a['class_valid'][c]]
  for col in range(3):
   ax=axes[rowindex,col];ax.imshow(rgb,interpolation='nearest');ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5);ax.set_xticks([0,w-1]);ax.set_yticks([0,h-1]);ax.tick_params(labelsize=7);ax.set_aspect('equal')
   if col==0:
    for c,cat in enumerate(CATS):
     for b in a['boxes'][c]:box(ax,b,COLORS[cat])
    sem=np.load(a['semantic_file'])['unknown_drawer_pixels'];yy,xx=np.where(sem)
    if len(xx):ax.scatter(xx,yy,s=15,c='#ffeb3b',marker='x',linewidths=.7)
    title=short+'\nGT'+('；未知：'+','.join(unknown)if unknown else'；两类有效')
   else:
    seed=col-1
    for p in preds[seed][i]['predictions']:
     b=np.array(p['box'])*[w,h,w,h];box(ax,b,COLORS[p['category']],'--');ax.text(b[0],max(0,b[1]-1),f'{p["score"]:.2f}',color=COLORS[p['category']],fontsize=6,backgroundcolor='#101010')
    q=[m for m in matches if m['seed']==seed and m['image_id']==i];tp=sum(m['TP']for m in q);fp=sum(m['FP']for m in q);fn=sum(m['FN']for m in q);title=f'种子 {seed}｜TP {tp} / FP {fp} / FN {fn}'+('\n未知类别预测显示但不评分'if unknown else'\n固定阈值 0.5；IoU ≥ 0.5 匹配')
   ax.set_title(title,fontsize=9,pad=6)
 fig.text(.5,.008,'样例规则在采集前固定：根 00/07，状态 0/4，双相机；未按得分挑选。未知不等于物体不存在。预测没有使用 GT 框或深度。',ha='center',fontsize=9)
 fig.subplots_adjust(top=.935,bottom=.025,left=.05,right=.98,wspace=.18,hspace=.45);path=R/'figures'/f'{split}_prediction_grid.png';fig.savefig(path,dpi=150);plt.close(fig);return path
def charts(metrics,coverage):
 fig,axes=plt.subplots(1,3,figsize=(15.5,4.9));colors=['#2764b7','#d5702a'];names=['训练 A/B','开发 C','留出 D'];x=np.arange(3)
 for seed in [0,1]:
  val=[metric(metrics,seed,s)['F1']for s in ['train','dev','final']];axes[0].bar(x+(seed-.5)*.32,val,.32,color=colors[seed],label=f'种子 {seed}')
  for xx,v in zip(x+(seed-.5)*.32,val):axes[0].text(xx,v+.02,f'{v:.3f}',ha='center',fontsize=8)
  tr=read(E/f'results/train_seed{seed}.json')['history'];axes[1].plot([a['update']for a in tr],[a['total']for a in tr],color=colors[seed],marker='.',label=f'种子 {seed}')
 for ax in axes[:2]:ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
 axes[0].set(xticks=x,xticklabels=names,ylim=(0,1),ylabel='检测 F1（两类合并）',title='训练集也未充分拟合');axes[1].set(xlabel='优化更新次数',ylabel='当前训练批总损失（对数轴）',title='固定 1,200 步，无最佳检查点选择',yscale='log')
 for j,cat in enumerate(CATS):
  for seed in [0,1]:
   m=metric(metrics,seed,'final',cat=cat);pos=j*3+seed;axes[2].bar(pos,m['TP'],color='#2f9e60',label='TP 正确检出'if pos==0 else None);axes[2].bar(pos,m['FN'],bottom=m['TP'],color='#c9ced5',label='FN 漏检'if pos==0 else None);axes[2].bar(pos,m['FP'],bottom=m['TP']+m['FN'],color='#d94848',label='FP 误检'if pos==0 else None)
 axes[2].set(xticks=[0,1,3,4],xticklabels=['前板 S0','前板 S1','手指 S0','手指 S1'],ylabel='框数',title='留出集：漏检占主导');axes[2].legend(fontsize=8);fig.suptitle('固定 RGB-only 定位基线｜全量正式评分（未知类别已屏蔽）',fontsize=15);fig.text(.5,.01,'左图：同一固定解码器，F1=2TP/(2TP+FP+FN)。中图：训练批损失，不是开发集表现。右图：TP+FN 为有效 GT，FP 额外叠加。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(R/'figures/metrics_summary.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(12.5,6.7));ys=np.arange(len(coverage));known=[q['known_images']for q in coverage];unknown=[q['unknown_images']for q in coverage];ax.barh(ys,known,color='#2b8a78',label='有效图像类别：可监督和评分');ax.barh(ys,unknown,left=known,color='#bfc5ce',label='未知图像类别：屏蔽');labels=[]
 for y,q in enumerate(coverage):
  labels.append(f'{q["split"]} / {"静态"if q["camera"]=="static"else"腕部"} / {"前板"if q["category"]=="drawer_front"else"手指"}');ax.text(q['images']+1,y,f'{q["usable_positive_boxes"]} 框 / {q["positive_roots"]} 根',va='center',fontsize=8)
 ax.set(yticks=ys,yticklabels=labels,xlabel='图像类别数（同图两类分别统计）',xlim=(0,170),title='数据门槛通过，但未知标签仍完整保留');ax.invert_yaxis();ax.legend(loc='lower right',fontsize=9);ax.grid(axis='x',alpha=.2);fig.text(.5,.012,'右侧数字＝可用正框数 / 覆盖的独立根数；手指每个有效视图有两框。没有经证实的整图缺席负例；把手背景提供局部负监督。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(R/'figures/coverage.png',dpi=160);plt.close(fig)
def main():
 (R/'figures').mkdir(exist_ok=True);rows=read(E/'data/public/images.json');anns=read(E/'data/annotation_private/labels.json');preds=[read(E/f'data/predictions/decoded_seed{s}.json')for s in [0,1]];matches=read(E/'results/matches.json');metrics=read(E/'results/metrics.json')['metrics'];coverage=read(E/'results/coverage.json');paths=[grid(s,rows,anns,preds,matches)for s in ['train','dev','final']];charts(metrics,coverage)
 collect=read(E/'results/collection_summary.json');feat=read(E/'results/features.json');trains=[read(E/f'results/train_seed{s}.json')for s in [0,1]];diag=read(E/'results/single_counterfactual.json');audit=read(E/'results/confidence_geometry_audit.json');protocol=read(E/'configs/protocol.json');exclusion=read(E/'configs/historical_initialization_exclusion.json');pre=read(E/'results/training_preflight.json')
 lines=['# 同步 GT 数据与 RGB-only 定位基线：固定预算结果','', '本轮完成 32 个独立初始化根、256 个状态、512 张双相机主图，以及两个固定种子的训练和一次性评分。数据与输入隔离检查通过，但检测器未达到可靠定位：训练集也有大量漏检，因此不能把问题只归因于留出泛化。没有增加数据、延长训练、调阈值或改架构。','', '## 正式结果','', '以下每一项都来自实际 RGB-only 预测。置信度阈值固定 0.5，匹配 IoU ≥ 0.5；未知图像类别不进入分母。IoU 均值只计算 TP，不能代替召回率。','', '|种子|划分|类别|TP|FP|FN|精确率|召回率|F1|匹配 TP 均值 IoU|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for m in metrics:
  if m['camera']=='all':lines.append(f'|{m["seed"]}|{m["split"]}|{m["category"]}|{m["TP"]}|{m["FP"]}|{m["FN"]}|{m["precision"]:.3f}|{m["recall"]:.3f}|{m["F1"]:.3f}|{m["matched_TP_mean_IoU"]:.3f}|')
 lines+=['','分相机的 TP/FP/FN/IoU 在 `metrics.csv`；逐根结果在实验 `results/root_metrics.json`。两种子不是两套数据；同根的 8 个状态相关，因此没有把帧当成独立样本计算置信区间。','', '![正式指标](figures/metrics_summary.png)','', '## 数据、元素与原因','', '|元素|本轮实际做法|原因与边界|','|---|---|---|','|初始化根|A/B 各 8 根训练；C 8 根开发；D 8 根最终留出|根级隔离，不能跨划分共享轨迹状态|',f'|新初始化|32 个不同物块布局/朝向、抽屉初始开度与固定电机目标序列；与 {exclusion["checked_saved_configs"]} 份历史保存配置做 1 mm/0.001 rad 复用排查|不是把随机种子称作新场景；A/B/C/D 场景资源均已知|','|状态采样|原生复位后 7 段固定电机目标，每段 24 控制周期 × 8 物理步；每根 8 个状态|跳过中间图像渲染，动力学调用与原生 step 一致；目标未到位、碰撞状态也保留；不是技能成功率实验|','|双相机同步|每状态两张带对象/链接 mask 的主图，加两张同状态 flag-off 完整性复核图|全状态前后快照一致、主图/复核 RGB 与深度逐位一致；512 张复核图不是新增样本|','|抽屉前板|前板外面及边缘、附属把手；不含内侧板面、腔体、底板、侧板|沿用已确认的三角面语义；深度一致阈值仍为 0.5 mm|','|前板未知|任何原生抽屉像素语义不确定，整张图的前板类别屏蔽|129 个前板视图保持未知，未放松规则|','|手指|每根机械手指和其指尖组成一个可见框；运行时链接名称确定|3 张静态图仅观察到一侧手指，整类屏蔽；没有把零像素判为不存在|','|负监督|有效图像类别内非正中心 patch 为背景，另统计无手指像素/正中心的把手 patch|把手不是第三类；不是已证实的整图缺席负例，因此尚未检验目标缺席识别|','|可见框|mask 的半开区间 xyxy 包围框|不推测被遮挡部分；触边不自动等于已证实出画|','', '![标签覆盖](figures/coverage.png)','', '|划分|相机|类别|有效图/总图|未知图|可用正框|正框根数|把手负背景图|','|---|---|---|---:|---:|---:|---:|---:|']
 for q in coverage:lines.append(f'|{q["split"]}|{q["camera"]}|{q["category"]}|{q["known_images"]}/{q["images"]}|{q["unknown_images"]}|{q["usable_positive_boxes"]}|{q["positive_roots"]}|{q["hardnegative_handle_images"]}|')
 lines+=['','门槛在采集前冻结：每类、每相机训练至少 32 个有效正框/6 根，开发与留出至少 12 框/4 根；手指把手负背景训练至少 16 图/6 根，开发与留出至少 8 图/4 根。均通过。未知比例如实报告，不通过修改有效性阈值增加样本。最终静态前板只有 5 根具备有效标签，结论范围应据此理解。','', '## 固定训练与隔离','', f'- 冻结 DINOv2-small，本机已有缓存；448×448 RGB，32×32 patches。共享 384→64→15 MLP，{pre["parameters"]:,} 个可训练参数，前板 1 槽、手指 2 槽。同 patch 的 {pre["same_patch_two_finger_centers_preserved"]} 对手指中心均保留。','- RGB 输入双线性缩放及 ImageNet 归一化；patch 特征均值/标准差只由 256 张训练图计算。模型没有相机、根、位置、深度、mask 或 GT 框输入。','- focal α=0.25、γ=2；正框加 5×SmoothL1 和 2×GIoU。AdamW lr=0.001、wd=0.0001，batch=16，clip=1；种子 0/1 各 1200 步，无增强。','- 解码读取全部 patch/槽中 score≥0.5 候选；同类贪心 NMS IoU>0.5；没有局部峰值筛选或候选数上限。','- 只保留最终检查点。两模型冻结后才推理开发/留出；预测先保存并哈希，再读标签评分。最终留出每种子一次推理、一次共同评分。C 开发结果只作报告，没有用于模型选择。','- 编码器读取私有标签、训练读取非训练数据、推理读取标签容器均有文件访问守卫。预检验证未知类别零梯度、框编码往返、相邻同类候选与无数量上限。','', '## 第一个有影响的错误及一次诊断','',f'按固定顺序选 seed 0 留出第一个错误：`{diag.get("root_id")}`，状态 {diag.get("sample")}，{diag.get("camera")}，{diag.get("category")}。原候选框与 GT 的 IoU 为 {diag["intervention"].get("existing_box_IoU_to_missed_GT",0):.3f}，置信度 {diag["intervention"].get("previous_score",0):.6f}，未过 0.5。仅把该候选 logit 替换为 20，保持框、其他候选及全局阈值不变，局部 TP/FP/FN 从 0/0/1 变为 1/0/0。','', '这说明该单例的首个阻断点是候选置信度筛选。它不证明整体只需降低阈值，也不证明更长训练一定有效。诊断用到 GT 选择候选，单独保存，未混入正式 RGB-only 评分；只做这一项替换，无模型修正。', '', '|种子|划分|类别|FN|没有合格几何候选|合格几何候选均低于阈值|存在高分合格候选仍漏检|FP 中心落在已知把手|','|---|---|---|---:|---:|---:|---:|---:|']
 for a in audit['rows']:lines.append(f'|{a["seed"]}|{a["split"]}|{a["category"]}|{a["FN"]}|{a["FN_with_no_raw_box_IoU_at_least_05"]}|{a["FN_with_adequate_geometry_but_all_scores_below_05"]}|{a["FN_with_adequate_above_threshold_raw_candidate"]}|{a["FP_center_on_known_handle"]}|')
 lines+=['','这张分解表只是对已冻结候选的观察性审计，不是换阈值后的检测率。高分合格候选仍漏检可能涉及 NMS 或一对一匹配竞争；没有追加干预。把手统计仅看 FP 框中心，不能概括全部误检。','', '## 实际预测图','', '每组固定展示根 00/07、状态 0/4、两相机。训练图展示 A；开发 C；最终留出 D。颜色和坐标解释均写在图中。预测无框时保持空白，不补画 GT。','', '![留出预测](figures/final_prediction_grid.png)','', '![训练预测](figures/train_prediction_grid.png)','', '![开发预测](figures/dev_prediction_grid.png)','', '## 资源、文件与可复核范围','', f'- 本机 RTX 5080 Laptop 16 GB / Ubuntu 24.04，独立执行器与世界模型环境。采集 {collect["seconds"]:.2f} 秒、特征提取 {feat["seconds"]:.2f} 秒；训练种子 0/1 为 {trains[0]["seconds"]:.2f}/{trains[1]["seconds"]:.2f} 秒（各阶段计时口径见 JSON）。',f'- 数据原始 RGB/同步深度/对象链接 mask/语义有效 mask/快照/相机矩阵等 {collect["raw_and_annotation_bytes"]:,} 字节；首 50 状态测得每状态约 {read(E/"results/storage_first50.json")["bytes_per_state_excluding_root_metadata"]:,.0f} 字节（含复核图，不含根共享元数据）。128 状态时另有存储检查。',f'- 实验目录：`{E}`；报告目录：`{R}`。`configs/protocol.json` 是预先冻结协议；`configs/precollection_frozen.json`、`data_frozen.json`、`training_frozen.json`、`results/checkpoints_frozen.json`、`predictions_frozen.json` 记录先后边界。','- `data/public/` 只含 RGB；`annotation_private/` 含全部监督/评估证据；`features/` 缓存冻结特征；`models/` 为两个最终头；`results/` 为原始计数与诊断；`logs/` 保留全过程。执行命令见 `RUN_COMMANDS.md`。','- 采集失败 0 次、重试 0 次；256 个状态均保留。旧实验封存检查见 `results/final_verification.json`。','', '**结论边界：**同步标签数据生成与 RGB 输入隔离可复用；这次固定定位基线失败于低召回，尚不能作为可靠闭环感知输入。本轮不是整份世界模型/TS-LTL 预实验的完成证明，也没有部署、SupCon 或闭环技能测试。下一次若研究训练目标、置信度校准或读出容量，须另行预注册；当前 D 已被评分，不再能作为未见最终测试集。']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n');save(E/'results/report_figures.json',{'utc':now(),'files':[str(p)for p in paths]+[str(p)for p in(R/'figures').glob('*.png')if p not in paths],'selection_frozen_before_collection':True});print('REPORT_READY',R,flush=True)
if __name__=='__main__':main()
