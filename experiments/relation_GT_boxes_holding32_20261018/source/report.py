from common import *
import csv,collections,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle,Patch
from matplotlib.colors import ListedColormap
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'savefig.dpi':180})

def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(x)for x in r)+' |'for r in rows)+'\n'
def csvout(name,rows):
 with(R/name).open('w')as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 install_guard([]);protocol=read(E/'configs/protocol.json');audit=read(E/'results/data_audit.json');assert not audit['passed'];rows=read(E/'data/records.json');roots=read(E/'results/root_summaries.json');specs=protocol['roots'];summary=read(E/'results/collection_summary.json');inventory=read(E/'results/existing_data_inventory.json');strata=audit['strata'];storage=read(E/'results/storage_first50.json');checks=read(E/'results/integrity.json');frozen=read(E/'configs/precollection_frozen.json');assert all(sha(p)==h for p,h in frozen['sha256'].items())
 def metric(split,kind,group,scope='eligible',rel='Holding'):return next(x for x in strata if x['split']==split and x['group_type']==kind and x['group']==group and x['scope']==scope and x['relation']==rel)
 counters=[]
 for c in COLORS:
  a=metric('train','color',c);p=metric('train','color',c,'physical');counters.append({'color':c,'eligible_positive_roots':a['positive_roots'],'perfect_visibility_upper_bound_positive_roots':p['positive_roots'],'frozen_required_positive_roots':4,'visibility_can_close_root_gap':p['positive_roots']>=4})
 save(E/'results/single_component_diagnostic.json',{'replacement':'Remove only visual eligibility filtering, retain same roots, recorded physical labels and frozen root threshold. This is evaluator-only ceiling accounting, not changed training data.','rows':counters,'conclusion':'Red/blue root shortage persists with perfect endpoint visibility; no localization replacement can create missing positive roots. No controller/threshold/layout correction executed.'});csvout('visibility_upper_bound.csv',counters)
 rootrows=[]
 for sp,rr in zip(specs,roots):
  rs=[r for r in rows if r['root_id']==sp['root_id']];result=read(E/'data/private'/sp['root_id']/'frozen_controller_result.json');planning=result.get('planning_private',[]);first=None
  if result['status']!='success':
   retries=[e for e in result.get('events',[])if e['phase']=='bounded_symmetric_grasp_retry'];first=retries[0]['end_step']if retries else result['controls']
  rootrows.append({'root_id':sp['root_id'],'split':sp['split'],'scene':sp['scene'],'target_color':sp['target_color'],'saved_states':rr['states'],'controller_status':result['status'],'first_failure_reason':result['reason'],'first_failure_control':first,'planned_candidates':len(planning),'valid_candidates':sum(p['valid']for p in planning),'controller_controls':result['controls'],'whole_root_controls':rr['controls'],'eligible_positive_queries':sum(r['eligible']and r['y']for r in rs),'eligible_negative_queries':sum(r['eligible']and not r['y']for r in rs),'unobservable_queries':sum(not r['visually_eligible']for r in rs),'missing_anchors':','.join(rr['missing_anchors'])})
 csvout('root_outcomes.csv',rootrows)
 unavailable=[{'arm':arm,'seed':seed,'status':'not_run_data_gate_failed','trained_updates':0,'precision':None,'recall':None,'F1':None,'balanced_accuracy':None,'NLL':None,'reason':'train red positive roots2<4; blue3<4'}for arm in ['object','pair']for seed in [0,1]];save(E/'results/model_metrics.json',unavailable);csvout('model_metrics.csv',unavailable)
 first=read(E/'data/private/relation32_A_00/frozen_controller_result.json');save(E/'results/first_consequential_failure.json',{'root_id':'relation32_A_00','split':'train','target':'red','controller_controls':first['controls'],'reason':first['reason'],'candidate_count':len(first['planning_private']),'all_rejected_at_descent_stage':all(p.get('stage')==2 for p in first['planning_private']),'distance_mm':[1000*p['distance_m']for p in first['planning_private']],'required_margin_mm':12,'interpretation':'all16 query-world candidates rejected before execution. These are geometric clearance predictions, not observed execution-world collisions. This removes one intended red positive root; the broader shortage is 2/6 red and3/6 blue target roots yielding stable positives. No causal repair attempted.'})
 fig,axs=plt.subplots(2,2,figsize=(14,10.2));fig.suptitle('正确原生框的关系读出：数据门槛未通过，训练未启动',fontsize=20,y=.985)
 ax=axs[0,0];val=[metric('train','color',c)['positive_roots']for c in COLORS];ax.bar(['红色','蓝色','粉色'],val,color=['#d45552','#3787ba','#bd6dab']);ax.axhline(4,color='#333',ls='--',label='冻结门槛：每色至少 4 个正例根');ax.set_ylim(0,6.5);ax.set_ylabel('训练集中有 Holding 正例的独立根数');ax.set_title('A  独立覆盖：红 2、蓝 3，不足 4');ax.legend(fontsize=9,loc='upper left')
 for i,v in enumerate(val):ax.text(i,v+.12,str(v),ha='center',fontsize=16)
 ax=axs[0,1];splits=['train','dev','test'];counts=[metric(s,'all','all')for s in splits];xx=np.arange(3);bottom=np.zeros(3)
 for key,label,color in [('positive_records','可用正例','#219b83'),('negative_records','可用负例','#6296bc'),('unknown_records','端点不可见，弃权','#b8bec6')]:
  vals=np.array([r[key]for r in counts]);ax.bar(xx,vals,bottom=bottom,label=label,color=color)
  for x,v,b in zip(xx,vals,bottom):
   if v:ax.text(x,b+v/2,str(v),ha='center',va='center',fontsize=10)
  bottom+=vals
 ax.set_xticks(xx,['训练 A/B','开发 C','测试 C']);ax.set_ylabel('查询条数（颜色 × 相机 × 状态）');ax.set_title('B  重复视角和连续状态不算独立根');ax.legend(fontsize=9)
 ax=axs[1,0];cams=['static','gripper'];known=[sum(metric(s,'camera',c)['known_records']for s in splits)for c in cams];missing=[600-x for x in known];ax.barh([1,0],known,color='#6296bc',label='可用查询');ax.barh([1,0],missing,left=known,color='#b8bec6',label='方块可见像素不足 4，弃权');ax.set_yticks([1,0],['固定相机','腕部相机']);ax.set_xlim(0,620);ax.set_xlabel('每台相机 600 条物理有效查询');ax.set_title('C  可见性缺口主要在腕部相机')
 for y,n,m in zip([1,0],known,missing):ax.text(n/2,y,f'{n}/600 可用',ha='center',va='center');ax.text(610,y,str(m)+' 缺失',ha='right',va='center',fontsize=9)
 ax.legend(fontsize=9,loc='lower left')
 ax=axs[1,1];labels=['完成冻结抓取','几何候选全部拒绝','非目标状态变化监测停止'];vals=[18,8,6];ax.barh([2,1,0],vals,color=['#219b83','#d79247','#a86d8a']);ax.set_yticks([2,1,0],labels);ax.set_xlim(0,22);ax.set_xlabel('独立根数，共 32 根');ax.set_title('D  首个影响发生在正例采集阶段')
 for y,n in zip([2,1,0],vals):ax.text(n+.3,y,str(n),va='center')
 fig.text(.5,.035,'原生 mask 仅给出正确可见框；接触力真值只用于私有评分。可见不等于能从 RGB 判定承重。\n已核验 200 状态 / 400 主图 / 1,200 查询。全部 Holding 正例可见，完美可见也无法补足红、蓝正例根。',ha='center',fontsize=11)
 fig.tight_layout(rect=[0,.09,1,.95]);fig.savefig(R/'figures/01_data_gate.png');fig.savefig(R/'figures/01_data_gate.pdf');plt.close(fig)
 mat=np.full((32,8),-1);missing_marks=[]
 for i,sp in enumerate(specs):
  for j in range(8):
   rr=[r for r in rows if r['root_id']==sp['root_id']and r['sample']==j and r['color']==sp['target_color']]
   if rr:
    assert len(rr)==2;mat[i,j]=int(rr[0]['y'])
    if not all(r['visually_eligible']for r in rr):missing_marks.append((j,i))
 fig,ax=plt.subplots(figsize=(12,13));ax.imshow(mat,vmin=-1,vmax=1,cmap=ListedColormap(['#e1e4e8','#729bbe','#219b83']),aspect='auto',interpolation='nearest');ax.set_yticks(np.arange(32),[r['root_id'].replace('relation32_','')+' '+{'red':'红','blue':'蓝','pink':'粉'}[r['target_color']]for r in specs],fontsize=9);ax.set_xticks(range(8),['初始稳定','目标上方','张爪下降','支撑上闭爪','抬升','运行后稳定','闭爪平移','桌上释放'],rotation=20);ax.set_xlabel('事先冻结的 8 个动作阶段；空缺阶段不补采');ax.set_ylabel('独立根：A/B 训练 18 根，C 开发 6 根，C 测试 8 根');ax.axhline(17.5,color='black',lw=2);ax.axhline(23.5,color='black',lw=2)
 if missing_marks:ax.scatter(*zip(*missing_marks),s=34,marker='x',color='#9b4900',linewidths=1.5,label='至少一台相机目标方块 <4 像素')
 ax.set_title('每根目标方块的严格 Holding 真值与缺失状态\n失败根全部保留；一行内部的多帧不增加独立根数',pad=20,fontsize=16);legend=[Patch(color='#e1e4e8',label='阶段未到达，缺失'),Patch(color='#729bbe',label='物理 Holding=0'),Patch(color='#219b83',label='物理 Holding=1')];h,l=ax.get_legend_handles_labels();ax.legend(handles=legend+h,loc='upper center',bbox_to_anchor=(.5,-.09),ncol=2,fontsize=10);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(R/'figures/02_root_state_matrix.png');plt.close(fig)
 # Fixed demonstration: first successful red root, closed-on-support vs lifted.
 fig,axs=plt.subplots(2,3,figsize=(13,8.7));cases=[('relation32_A_03',3,'red'),('relation32_A_03',4,'red'),('relation32_A_00',0,'red')]
 for col,(root,sample,color)in enumerate(cases):
  for row,cam in enumerate(cams):
   ax=axs[row,col];r=next(x for x in rows if x['root_id']==root and x['sample']==sample and x['color']==color and x['camera']==cam)
   with np.load(r['path'])as z:ax.imshow(z[r['key']],interpolation='nearest')
   for field,cc in [('object_bbox','#00ffff'),('gripper_bbox','#ffb52e')]:
    b=r[field]
    if b:ax.add_patch(Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,edgecolor=cc,lw=2))
   ax.set_xticks([]);ax.set_yticks([]);ax.set_title(['桌上闭爪','抬升后','首个几何拒绝根：初始'][col]+f'\nHolding={int(r["y"])}，端点可见={r["visually_eligible"]}',fontsize=11)
   if col==0:ax.set_ylabel('固定相机'if row==0 else'腕部相机',fontsize=12)
 fig.suptitle('实际 RGB 与原生可见框；标签来自同期物理真值',fontsize=18,y=.99);fig.text(.5,.035,'青框：目标方块；橙框：两侧指尖家族的可见像素并集（不是整个机械手）。\n前两列同属 A_03 红块：闭爪仍受桌面支撑时为负，抬升承重后为正。末列 A_00 的全部 16 个几何候选被拒绝，未执行抓取。\n这些是数据说明例，不能证明视觉模型可辨识关系；没有训练、预测或模型正确率。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.13,1,.94]);fig.savefig(R/'figures/03_native_box_evidence.png');plt.close(fig)
 colorrows=[]
 for split in splits:
  for c in COLORS:
   r=metric(split,'color',c);colorrows.append([split,c,r['positive_records'],r['negative_records'],r['unknown_records'],r['positive_roots'],r['negative_roots']])
 firstdist=[1000*q['distance_m']for q in first['planning_private']]
 text=f'''# 正确原生框条件下的关系读出：本轮未达到训练门槛

冻结的 32 个独立根已全部执行，保存 **200 个实际状态、400 张双相机主图、1,200 条方块关系查询**。训练红块正例只有 **2 根**、蓝块 **3 根**，低于事先规定的每色 4 根；粉块为 4 根。按预注册规则，本轮 **0 次拟合、0 次视觉编码、0 次关系预测**。四个初始模型只是事先保存的随机初始化，不是训练模型。不能据此声称对象对优于单对象，也不能给出精确率、召回率或平衡准确率。

## 现有数据为什么不够

旧语义数据有 630 个物理标签窗口和 2,520 个引用 RGB 文件，但这些文件均未保存原生 mask。近期两个定位批次有 320 状态、640 主图、原生 mask，却没有同期承重接触记录。不能把不同运行的标签和图像拼成同期证据，也不能从闭爪或物体中心位置推断 Holding/In。当前严格交集为零。完整审计见 existing_data_inventory.json。

## 本次实际采集与真值

只使用本机已有 CALVIN、EGL 和冻结 PhysicalOracleV2 candidate0，动作通过原生 IK/电机执行；物理查询在私有评估侧。执行世界不重置物体、不附着、不改碰撞、不加外力。独立 query client 的位置重置只用于候选几何检查。采集适配器保留原生每控制步的 8 次物理推进和 scene.step 顺序，略去中间 RGB 渲染；它不是新控制策略。每根最多 900 控制步，本次总 {summary['controls']:,} 控制步、{summary['physics_ticks']:,} 物理 tick（包含初始化）。

每根固定最多 8 个锚点：初始稳定、目标上方、张爪下降、支撑上闭爪、抬升、运行后稳定、闭爪平移、桌上释放。控制器原有至多两次尝试未改。32 根没有补根、替换根或重采；14 个失败根各只有 4 个实际锚点，所以总数是 200 而非 256。失败后的三个固定电机动作仍保留实际标签，未当成抓取成功。抬升后的平移/释放是预先固定的采集动作，不算控制器验收成功。

Holding 保持既有定义：同一方块同时对左右指尖家族有大于 0.01 N 的接触，且没有其他物体的支撑接触。有效性额外要求最近 12 个真实控制步、0.005/0.01/0.015 N 三个力阈值都一致，且不存在多个被持有物体的矛盾。当前 1,200 查询均通过物理有效性检查。此处真值描述接触关系，不从 RGB 反推。

两相机同一 tick 获取 RGB、深度和原生 body/link mask；重新渲染 flags=0 的 RGB/深度逐位相同，渲染前后完整位姿/关节/速度/接触不变。独立审计重算 **200 次原生标签、1,200 个框、7,200 次历史接触阈值检查**。所有检查通过；旧数据哈希交集、跨划分 RGB 重复和本批 RGB 重复均为零。历史去重仅使用已有登记元数据与哈希，不能覆盖未登记历史数据。

可见条件是目标方块以及左右各指尖家族分别至少 4 像素。原生 mask 给出的是紧贴可见像素的框，不是投影三维框或物体中心。229 条查询因方块不足 4 像素而弃权；没有把不可见当成负例。所有 Holding 正例在两相机均通过此必要条件，但**端点可见不保证承重接触可由 RGB 唯一确定**。

## 数据分布与停止原因

一条查询 = 一个状态 × 一台相机 × 一个颜色方块；连续状态、同根颜色及双相机高度相关，不能当成独立根。根级划分已在采集前冻结：A9/B9 共18训练根，C6开发根，C8测试根；所有状态、颜色和相机随根绑定。C 场景资产在历史实验已知，只能谈新初态和交互配置。开发/测试标签本次仅用于覆盖与完整性审计，无训练选择；若以后改变方案再使用这些根，应披露本次已查看其覆盖。

{table(['划分','颜色','可用正查询','可用负查询','端点缺失查询','正例独立根','负例独立根'],colorrows)}

冻结门槛的全部项目中仅两项失败：训练红色正例根 2/4，缺 2；训练蓝色 3/4，缺 1。其余每色正/负条数、负例根、各相机正/负覆盖及开发/测试最低覆盖均通过。门槛是本次可行性条件，不是原指南验收标准。没有因为只差独立根就放宽门槛，或拿 C 数据补训练。

固定相机可用 **597/600** 查询，腕部相机 **374/600**；区别来自方块可见性。腕部有 226 条缺失，固定相机有 3 条。可用总计 **971**，其中正 **118**、负 **853**。这些是数据计数，不是模型 TP/TN。

## 第一个有影响的错误与单项诊断

按冻结根顺序，第一个影响正例采集的是训练根 `relation32_A_00`（红色）。其 16 个候选在私有 query world 的下降阶段都不满足几何间隙要求；已记录的候选距离范围 **{min(firstdist):.3f}–{max(firstdist):.3f} mm**，规定至少 12 mm，因此控制器在 **0 个执行控制步**时返回 `no_collision_free_grasp_candidate`。这是几何判据拒绝，不能表述为真实执行碰撞。候选详细记录见 data/private/relation32_A_00/frozen_controller_result.json。

全批共 18 根控制器抓取成功、8 根无合格几何候选、4 根触发非目标设备变化监测、2 根触发其他方块变化监测。后两类只能确认冻结监测触发；未保存每个中间控制状态的完整链条，不能进一步断言具体碰撞/物理漂移机制。`root_outcomes.csv` 保留每根首次失败控制步、原因、实际锚点与正负覆盖。

只做一个离线单项替换诊断：将“端点可见筛选”替换为“完美可见上限”，其他根和物理标签不动。训练红/蓝/粉的正例根仍是 **2/3/4**。所以定位或可见性替换不能补足本轮红蓝正例独立根；第一个有效瓶颈在正例数据形成，而非尚未训练的关系读出头。该诊断只用于评估计数，未改变模型输入或训练集合。没有执行控制器、布局、阈值或网络修正。

## In 的审计结论

当前冻结 v2.1 In 真值要求整个物体 OBB 位于抽屉腔体内并存在指向抽屉的承重支撑路径，且未被持有；边界、支撑力/法向不确定性遵循旧冻结实现。本轮只覆盖桌面抓取/释放，**1,200 查询全为 In=0，0 正例、0 正例根**。也未建立可见腔体区域的观测条件。因此 In 不训练、不报告准确率。抽屉前板或把手框不会代替内部容纳证据。

## 已冻结但未执行的公平比较

{table(['要素','单对象','对象对','原因/限制'],[['输入','方块 RGB crop','同一方块 + 两指尖并集 RGB crop + 5 项相对二维框几何','不输入3D位置、接触、robot_obs、动作、阶段、根或颜色名'],['视觉编码','冻结 DINOv2-small，各紧框独立224×224，patch均值384','相同编码器及相同流程','使用已有模型缓存，禁止全景上下文隐藏输入'],['投影','固定随机正交384→128','相同投影','数据无关，非标签/PCA拟合'],['读出头','128→67→1，8,711参数','261→33→1，8,680参数','差31参数约0.36%；宽度/输入维度不同，不称完全同结构'],['初始化','种子0/1，同一标准Linear初始化规则','同规则、同种子','形状与fan-in不同，数值不可能完全相同，初始权重已存'],['预算','各600更新、batch32','相同','同根均匀抽样序列，AdamW lr.001 wd.001，最终checkpoint'],['标签/阈值','同一可用集合，训练集类别权重','同一集合','仅训练统计做归一化，固定0.5，无调参/挑种子'],['本轮实际','0更新、未编码','0更新、未编码','数据门槛失败，全部性能指标N/A']])}

按颜色/相机/根的精确率、召回率、F1、平衡准确率、NLL 和两臂配对根 bootstrap 都**没有执行**；`model_metrics.csv` 的四行明确为 `not_run_data_gate_failed`，数值空缺而非 0。也未做预测框替换、SupCon、世界模型、闭环规划或进一步定位训练。正确 GT 框条件本来就是条件诊断，不能称视觉端到端结果。

## 图表怎么读

1. `figures/01_data_gate.png`：左上横轴方块颜色、纵轴训练正例独立根，黑虚线为事先门槛4；红/蓝/粉柱只代表对象颜色。右上按训练/开发/测试统计查询，绿为可用物理正例、蓝为可用物理负例、灰为不可见弃权。左下比较同样600查询下两相机的可用数。右下统计冻结控制器18/8/6类运行结果。意义是数据不足阻止公平比较，不是关系模型失败。
2. `figures/02_root_state_matrix.png`：每行一根，每列一个固定锚点；灰缺阶段、蓝Holding=0、绿Holding=1，棕色叉表示至少一个相机目标方块不足4像素。黑横线分开训练、开发、测试。目标颜色写在行名；一行多绿格仍只是一根。
3. `figures/03_native_box_evidence.png`：实际同批RGB；青框为方块，橙框为两侧指尖可见像素并集。两行是固定/腕部相机；前两列是首个成功红色根A_03的桌上闭爪与抬升状态，第三列是首个几何拒绝根A_00的初态。图中文字的Holding是真值，不是预测；不从示例推断模型能力。

## 文件、存储与复现

实验目录 `{E}`；报告目录 `{R}`。`source/` 为代码，`configs/` 为预采集冻结计划、历史排除、投影、初始权重摘要及数据哈希，`data/public/` 只有双相机RGB数组，`data/private/` 是接触/状态/mask/映射/原生框与稳定性审计，`data/records.json` 是评估用记录，含标签和根元数据，不能直接作为模型输入。`models/` 只有初始化，`results/` 为审计与停止记录，`logs/` 为命令日志。

首50状态实测平均 **{storage['bytes_per_state']/1024:.1f} KiB/状态**（不含共享配置和根索引），相同格式256状态预计 {storage['projected_256_state_bytes']/1024**2:.1f} MiB。最终数据目录 {summary['data_bytes']:,} 字节。首128状态也保存存储审计；标签/泄漏完整审计在全批200状态完成，**未执行128状态训练检查**，不能据此声称完成指南的训练/验收阶段。

执行顺序：预采集 prepare.py → executor 环境 collect.py → executor 环境 audit.py → world_model 环境 CPU report.py。第一次 audit.py 用 world_model 环境因缺SciPy在读取数据前退出；随后使用已有带SciPy的executor环境成功重算，无依赖安装。原始失败日志保留。运行命令见 RUN_COMMANDS.md。采集GPU锁保证串行；审计/报告为CPU，旧实验和原定位器文件只读。

本轮已经耗尽32根上限，剩余56个理论状态槽来自14个未到达阶段，不能用补根或伪造阶段填满。已完成有证据的有界数据可行性检验；**关系模型比较与整个世界模型/TS-LTL指南均未完成**。若后续要增加独立根、改变布局/执行器或降低覆盖门槛，需要明确改变本轮已封存方案；本轮不自行实施。
'''
 (R/'REPORT.md').write_text(text)
 commands=f'''# 实际运行命令

```bash
source {P}/world_model_env.sh
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/relation_boxes_src/prepare.py
source {P}/env.sh
python -u {E}/source/collect.py > {E}/logs/collect.log 2>&1
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/relation_boxes_src/audit.py > {E}/logs/audit_executor.log 2>&1
source {P}/world_model_env.sh
CUDA_VISIBLE_DEVICES='' python /home/robot/Documents/Codex/2026-10-13/task-4/relation_boxes_src/report.py
```

prepare/collect 包含防重复断言，封存后不可直接重跑覆盖。本次采集一次，无补根。audit/report 可从同批已存数据复核，但封存文件须保持只读。初次错误 audit 命令与 audit.log 一并保留：world_model 环境缺SciPy，转用已有executor环境后通过。
''';(R/'RUN_COMMANDS.md').write_text(commands)
 for name in ['audit.py','report.py']:shutil.copy2(Path(__file__).parent/name,E/'source'/name)
 stop={'utc':now(),'outcome':'bounded_collection_complete_relation_fits_blocked_by_frozen_data_gate','roots_attempted':32,'saved_states':200,'training_updates':0,'feature_passes':0,'model_prediction_passes':0,'new_downloads':0,'replacement_roots':0,'controller_corrections':0,'first_failure':'training red root A_00 has zero valid query-world grasp candidates','blocking_gates':audit['failures'],'In_positive_records':0,'available_evidence':str(R/'REPORT.md'),'guide_complete':False};save(E/'results/STOP_DATA_GATE.json',stop);save(R/'OUTCOME.json',stop)
 print('REPORT_WRITTEN',str(R/'REPORT.md'),flush=True)
if __name__=='__main__':main()
