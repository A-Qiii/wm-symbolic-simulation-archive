from common import *
import collections,csv,html,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(FONT);plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':11,'axes.unicode_minus':False})
MODELS=['original_temporal','pooled','spatial'];NAMES=['原冻结时序网络','同容量池化对照','空间保留网络'];COLORS_PLOT=['#929da7','#488e98','#6268a3'];SHORT=['H红','H蓝','H粉','In红','In蓝','In粉','Closed','Open','Mid']
def table(h,rr):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |'for r in rr)+'\n'
def figsave(f,n):f.savefig(R/'figures'/n,dpi=160,bbox_inches='tight');plt.close(f)
def camera(ax,path,title):
 z=np.load(path);ax.imshow(z['rgb_static']);ax.set_title(title,fontsize=10);ax.axis('off');sub=ax.inset_axes([.70,0,.3,.3]);sub.imshow(z['rgb_gripper']);sub.axis('off')
def main():
 ev=read(E/'results/evaluation.json');d=ev['metrics'];decision=read(E/'configs/candidate_decision.json');train={a:read(E/'results'/a/'training.json')for a in ['pooled','spatial']};meta=read(BASE/'data/metadata.json');index={r['window_id']:i for i,r in enumerate(meta)};wins={r['window_id']:r for r in read(BASE/'data/windows.json')};probs={'original_temporal':np.load(BASE/'results/probabilities.npy')}|{a:np.load(E/'results'/a/'probabilities.npy')for a in ['pooled','spatial']};errors=read(E/'results/errors.json');original_errors=[r for r in errors if r['model']=='original_temporal'];spatial_errors=[r for r in errors if r['model']=='spatial'];shutil.copy2(E/'results/evaluation.json',R/'tables/evaluation.json');shutil.copy2(E/'results/errors.json',R/'tables/errors.json');shutil.copy2(E/'configs/candidate_decision.json',R/'tables/candidate_decision.json')
 mainrows=[];factrows=[]
 for name,label in zip(MODELS,NAMES):
  z=d[name]['C:new'];g=z['joint_goal_all_objects'];h=d[name]['C:legacy_H'];mainrows.append([label,f"{z['all9_correct_fully_valid']}/{z['fully_valid9']}",f"{z['root_all_valid_facts_correct']}/9",f"{z['correct_valid_facts']}/{z['valid_facts']}",f"{d[name]['C_counterexamples']['outside_supported']['FP']}/41",f"{g['FP']}/{g['FN']}",f"{h['Holding_all3_correct']}/84"])
  for fact,zf in z['facts'].items():factrows.append({'model':name,'fact':fact,**zf})
 with (R/'tables/per_fact_confusion.csv').open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=list(factrows[0]));writer.writeheader();writer.writerows(factrows)
 fig,axs=plt.subplots(1,3,figsize=(14,4.7))
 for i,name in enumerate(MODELS):
  z=d[name]['C:new'];vals=[z['all9_correct_fully_valid']/z['fully_valid9'],d[name]['C_counterexamples']['outside_supported']['FP'],z['joint_goal_all_objects']['FP']]
  for j,v in enumerate(vals):axs[j].bar(i,v,color=COLORS_PLOT[i]);axs[j].text(i,v+(.015 if j==0 else .25),f"{z['all9_correct_fully_valid']}/{z['fully_valid9']}"if j==0 else str(v),ha='center')
 for ax,title,ylabel in zip(axs,['完整九事实：略有提高','越界In误报：明显减少','共同目标误报：空间版退步'],['正确率（209个确定窗口）','误报事实数 / 41','误报目标数 / 606个负目标']):ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xticks(range(3),['原时序','池化对照','空间保留']);ax.grid(axis='y',alpha=.15)
 axs[0].set_ylim(0,1.1);axs[2].set_ylim(0,3);fig.text(.5,-.06,'灰、青、紫分别为原网络、池化对照、空间网络。共同目标为Closed且物块仍在内；两个新增误报来自同根的两个时间窗，不能算两次独立失败。',ha='center',fontsize=10);figsave(fig,'performance_and_goal_regression.png')
 fig,axs=plt.subplots(1,2,figsize=(15,4.5))
 for ax,kind,title in zip(axs,['FP','FN'],['各事实误报（真值假，却报真）','各事实漏报（真值真，却报假）']):
  a=np.array([[d[name]['C:new']['facts'][fact][kind]for fact in FACTS]for name in MODELS]);im=ax.imshow(a,cmap='YlOrBr',vmin=0,vmax=max(1,int(a.max())),aspect='auto');ax.set_yticks(range(3),['原时序','池化','空间']);ax.set_xticks(range(9),SHORT,rotation=40,ha='right');ax.set_title(title)
  for i in range(3):
   for j in range(9):ax.text(j,i,str(a[i,j]),ha='center',va='center',color='white'if a[i,j]>a.max()*.65 and a.max()>1 else'#222')
  fig.colorbar(im,ax=ax,fraction=.035,label='错误数')
 fig.text(.5,-.12,'H=Holding；红／蓝／粉为对象；Mid=Intermediate。每列正负例数不同，表中同时给出TP/TN/FP/FN；两幅色条独立，不能按颜色深度跨图比较。',ha='center',fontsize=10);figsave(fig,'per_relation_confusions.png')
 fig,axs=plt.subplots(1,2,figsize=(12,4.5))
 for arm,color,label in zip(['pooled','spatial'],COLORS_PLOT[1:],NAMES[1:]):
  cp=train[arm]['checkpoints'];xx=[r['update']for r in cp];axs[0].plot(xx,[r['C_selection_loss']for r in cp],'o-',color=color,label=label);axs[1].plot(xx,[r['C_new']['all9_correct_fully_valid']for r in cp],'o-',color=color,label=label);best=train[arm]['selected'];axs[0].scatter([best['update']],[best['C_selection_loss']],marker='*',s=160,color=color,edgecolors='black',zorder=4)
 axs[0].set(ylabel='预先固定的C根平均选择损失',title='按损失选择，两组均选600步');axs[1].set(ylabel='C九事实全部正确窗口数 / 209',title='继续训练没有持续改善')
 for ax in axs:ax.set_xlabel('本轮新增梯度更新步数');ax.legend(fontsize=9);ax.grid(alpha=.15);ax.set_xticks([300,600,900,1200])
 fig.text(.5,-.06,'青=池化，紫=空间；星号是按预定损失选中的检查点。C用于开发选点，已看过且不是盲测；没有看到误报后换检查点或延长训练。',ha='center',fontsize=10);figsave(fig,'fixed_budget_checkpoints.png')
 examples=[('C_semantics_v2_fixtures_red_0__outside_front_supported__t00127',3,'外沿反例：改正'),('C_semantics_v2_fixtures_blue_1__stacked__t00127',4,'堆叠反例：改正'),('C_semantics_v2_fixtures_pink_0__empty_closed__t00127',5,'正常负例：新增误报'),('C_semantics_v2_fixtures_pink_0__stacked__t00601',5,'受阻端点：仍误报')];fig,axs=plt.subplots(1,4,figsize=(15,5.2));exampledata=[]
 for ax,(wid,j,title)in zip(axs,examples):
  w=wins[wid];i=index[wid];label=w['targets_v2_1'];camera(ax,w['public_frames'][-1],title);pp=[float(probs[n][i,j])for n in MODELS];ax.text(.5,-.04,f"{FACTS[j]} GT={bool(label['y'][j])}\n原 / 池化 / 空间概率\n{pp[0]:.4f} / {pp[1]:.4f} / {pp[2]:.4f}",ha='center',va='top',transform=ax.transAxes,fontsize=10);exampledata.append({'window_id':wid,'fact':FACTS[j],'truth':bool(label['y'][j]),'probabilities':dict(zip(MODELS,pp))})
 fig.text(.5,-.09,'同一仿真观测RGB帧给三种网络，右下为腕部相机；GT只用于图中文字评分。In阈值始终0.5；第三格的粉块在桌面上，空间网络却略高于阈值。',ha='center',fontsize=10);figsave(fig,'same_frame_before_after.png');save(R/'tables/representative_readouts.json',exampledata)
 unique={}
 for r in spatial_errors:unique[(r['root_id'],r['condition'],tuple(r['error_columns']))]=r
 fig,axs=plt.subplots(2,3,figsize=(13,9))
 for ax,r in zip(axs.ravel(),unique.values()):
  j=r['error_columns'][0];camera(ax,r['public_frames'][-1],f"{r['condition']}\n{FACTS[j]} GT={bool(r['truth'][j])}，P={r['probabilities'][j]:.4f}")
 for ax in axs.ravel()[len(unique):]:ax.axis('off')
 fig.text(.5,.02,'空间版本6个错误窗口归并为5个条件／对象案例；空抽屉两窗口来自同一根。最后一格Open被报Intermediate。\n大图／右下图为静态／腕部相机；GT为评分真值，P为空间版概率；In阈值0.5，抽屉取三态最大概率。',ha='center',fontsize=10);figsave(fig,'remaining_spatial_errors.png')
 fig,axs=plt.subplots(1,2,figsize=(15,4.3))
 for ax,wid,title in zip(axs,[examples[0][0],examples[2][0]],['外沿反例：完整读出变化','关闭空抽屉：共同目标退步']):
  i=index[wid];w=wins[wid];matrix=np.array([w['targets_v2_1']['y']]+[probs[n][i].tolist()for n in MODELS],float);im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_yticks(range(4),['GT','原时序','池化','空间']);ax.set_xticks(range(9),SHORT,rotation=35,ha='right');ax.set_title(title)
  for yy in range(4):
   for xx in range(9):ax.text(xx,yy,f'{matrix[yy,xx]:.2f}',ha='center',va='center',color='white'if matrix[yy,xx]<.65 else'#222',fontsize=9)
 fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.015,pad=.025,label='概率：紫低、黄高；GT行为0/1');fig.text(.5,-.12,'H=Holding，Mid=Intermediate；红／蓝／粉为对象。Holding和In按0.5读出，抽屉三态联合softmax取最大。\n右图空间版同时高概率Closed和约0.506的In粉，因此错误宣布该共同目标成立；GT仅用于离线评分。',ha='center',fontsize=10);figsave(fig,'full_query_readout.png')
 conf=[]
 for fact in FACTS:
  z=d['original_temporal']['C:new']['facts'][fact];conf.append([fact,f"{z['positive']}/{z['negative']}"]+[' / '.join(str(d[n]['C:new']['facts'][fact][k])for k in ['TP','TN','FP','FN'])for n in MODELS])
 guardrows=[]
 for k in d['spatial']['core_gates']|d['spatial']['normal_class_and_goal_guards']:
  guardrows.append([k]+['通过'if(d[a]['core_gates']|d[a]['normal_class_and_goal_guards'])[k]else'未通过'for a in ['pooled','spatial']])
 goalrows=[]
 for name,label in zip(MODELS,NAMES):
  z=d[name]['C:new'];gg=z['joint_goal_all_objects'];tg=z['joint_goal_intended_object'];goalrows.append([label,f"{z['families']['goal_related_In_and_Closed']['exact_correct_windows']}/209",f"{gg['TP']}/{gg['positive']}",f"{gg['TN']}/{gg['negative']}",f"{gg['FP']}/{gg['FN']}",f"{tg['TP']+tg['TN']}/{tg['valid']}"])
 params=train['spatial']['trainable_parameters'];summary={'shared_network_training_runs':2,'updates_each':1200,'total_updates':2400,'seeds':[0],'trainable_parameters_each':params,'AB_new_roots':18,'C_new_roots':9,'legacy_AB_roots':48,'legacy_C_roots':18,'C_already_development':True,'new_D':0,'candidates':decision['candidates'],'figures':6,'planning_operator_support_unchanged':True,'summary_table':mainrows};save(R/'tables/summary.json',summary)
 text=f'''# 保留空间信息的共享查询网络：固定预算开发验证

本轮完成两组共享查询网络，各seed0、1200步，共2400次新增更新。**没有模型通过开发门槛，没有晋升候选，没有新增D或规划执行。** 空间版本完整九事实为203/209，越界反例误报从41降到1，但新出现2个“关闭且粉块在内”的共同目标误报。不能用总体分数提高掩盖这个退步。

![主结果](figures/performance_and_goal_regression.png)

{table(['网络','C完整九事实','整根全部有效事实正确','C有效事实正确','越界反例误报','共同目标FP/FN','旧C Holding全三项'],mainrows)}

C新数据为9个根、210窗口，其中209窗口九项标签都确定；AB为18根、420窗口。旧数据仅保留Holding监督：AB48根228窗口、C18根84窗口。合计942输入窗口；旧In和抽屉标签没有迁移。根级划分不重叠，同根多帧和复位条件不是独立样本。C已经用于前轮诊断及本轮检查点选择，所有分数都是开发结果。

## 预先冻结的对照

两组结构、{params}个可训练参数、初始权重、基础批次、优化器和损失完全一致。原冻结时序编码器662784参数不变；每个对象和场景保留96维历史表示，并加入96维当前视觉分支。所有5种关系共享同一个查询头：对象／锚点特征及差值、乘积、16维关系类型进入784→128→64→1网络，抽屉三态联合softmax。不是每个谓词各训一个分类器，也不是上一轮线性探针。

|元素|池化对照|空间保留版本|为什么这样比较|
|---|---|---|---|
|对象区域|2×2 patch先求均值，再重复为4槽|保留原2×2的4槽|只改变区域内空间布局保留程度|
|场景区域|4×4格降成2×2，再重复回4×4槽|保留原4×4格|保持输入大小与参数容量一致|
|DINO通道投影|共享384→32|相同|已有缓存直接输入，不下载或重编码骨干|
|当前对象／场景分支|分别映射为96维|相同|让查询能访问缓存，绕过旧时序压缩入口|
|旧时序信息|冻结对象／场景96维|相同|保留Holding等依赖历史的线索|
|查询读出|同一共享类型查询头|相同|所有共同事实统一训练、统一评分|

两组均从旧九事实选定模型初始化，把新增空间输入列的权重置零，其余头权重复制；训练前942窗口的输出与原网络最大差为0，初始状态哈希相同。缺失对象ROI按当前RGB可见性置零，不填GT位置。drawer／gripper仍是带类型的场景锚点，不声称有精确抽屉实例或三维边界。

每批32：16新AB、16旧AB Holding，分别先均匀选根再选窗口，1200批索引逐项相同。Holding和In使用AB逆频率BCE，抽屉用AB逆频率三分类CE，按存在的关系组等权。没有只针对41反例的损失。AdamW学习率0.0005、权重衰减0.001、梯度裁剪1；仅一个种子，不追加搜索。

![固定预算和选点](figures/fixed_budget_checkpoints.png)

300/600/900/1200检查点全部保留，按预定“新C根平均损失＋0.2旧C Holding根平均损失”选点，两组均选600。不能因600的共同目标误报再改选其他点。原冻结网络只是只读参考，没有第三次训练。

## 全关系、正常类与共同目标

![关系混淆](figures/per_relation_confusions.png)

下表每格依次是 **TP / TN / FP / FN**，H为Holding；UNKNOWN本轮各关系均为0。完整机器可读表在tables/per_fact_confusion.csv。

{table(['事实','C有效正/负例','原时序 TP/TN/FP/FN','池化 TP/TN/FP/FN','空间 TP/TN/FP/FN'],conf)}

两新网络新C Holding均210/210，旧C Holding均84/84；抽屉三态均209/210，Closed均0误报0漏报。池化版In有4误报2漏报，空间版有4误报1漏报。空间版的4个误报中只有1个属于原41个越界承重反例，另外是跌落过程中的蓝块和桌面粉块两个窗口。因此41反例明显改善，不等于所有正常负例都可靠。

“共同目标”是每个对象的In(object,drawer)与Closed同时为真，不能用¬Open代替Closed。下面也单独检查In×3与Closed四个目标相关事实，而非只看逻辑与的结果。

{table(['网络','目标相关四事实完全正确','共同目标TP/正例','共同目标TN/负例','共同目标FP/FN','原任务指定对象目标正确'],goalrows)}

共有629个确定的对象共同目标，其中23真、606假；大量打开状态下，无论In读对与否，共同目标都为假。因此原网络共同目标100%不代表它具有正确的中间符号状态，更不代表规划链路验收成功。空间版的两个新增目标误报来自同一根的两个窗口，实际粉块仍在桌上。

{table(['训练前固定检查','池化','空间'],guardrows)}

核心感知门槛沿用上一轮；本轮在训练前还增加正常类和共同目标保护项：Holding至少209/210、抽屉三态至少208/210、In正例漏报最多2、共同目标0误报。两个网络均未通过核心反例门槛，空间版也没通过共同目标0误报。没有降低门槛或调整0.5阈值。

## 同帧读出变化与剩余错误

![同帧读出](figures/same_frame_before_after.png)

空间版相对原模型纠正43个有效事实，同时退步4个有效事实；池化版纠正40、退步2。两组新增缓存分支后都明显改善；相对原参考同时发生了结构扩展和继续训练，不能把全部收益因果归于直接路径。空间分辨率在同容量对照上仅多一个完整正确窗口，又出现共同目标误报，不能据此声称细patch全面优于池化。

![完整状态读出](figures/full_query_readout.png)

![剩余错误](figures/remaining_spatial_errors.png)

空间版剩余6个错误窗口来自4个根：蓝块完全在内的一次漏报；蓝块滑落过程的一次误报；粉块堆叠受阻端点仍被高概率报In；关闭空抽屉且粉块在桌面时两个概率约0.506–0.508的In误报；以及粉块链中Open被报Intermediate（原网络也错）。图中文字GT仅供离线对照，未送入查询网络。

桌面粉块的误报接近0.5，但不能据C结果抬高阈值以抹掉它；受阻粉块的In概率约0.999，也不是简单调整一个阈值就能解释全部错误。训练后半段C结果没有持续改善，当前证据不支持自动增加更新次数。

## 交付与下一步边界

这次实验验证了可直接运行的共享查询接口，支持Holding、In、Closed、Open、Intermediate及实体别名，训练后的置换／别名一致性和冻结编码器检查均通过；它不等于已通过验收的部署候选。旧规划算子unsupported标记保持原样，本轮没有执行新的搜索或技能链。

继续有限优化有依据：带缓存直达分支的两组读出均有改善，剩余错误集中在局部边界、跌落／受阻端点和正常关闭场景。若继续，先在AB核实这些类别的覆盖、区分可见性与语义，再预先固定一次数据或读出／校准改动；不能用这几个C错误直接选阈值或不断加训练。若后续感知门槛通过，才能设计一次冻结的新根感知验收，检查完整事实与共同目标；规划验收还需先补动作失败覆盖和状态充分性，本轮不授权或运行新D。

本轮只完成已授权的感知开发对照；原指南要求的完整CALVIN/HULC、RSSM世界模型供给TS后继与LTLf闭环验收仍未完成。

[运行脚本与目录](README.md) · [图表浏览](gallery.html)。前轮真实掉落／关闭视频保留在[诊断图库]({P}/reports/containment_diagnostic_20261015/gallery.html)。旧模型、旧标签、旧结果都未覆盖。
''';(R/'REPORT.md').write_text(text)
 commands=[]
 for variant in ['pooled','spatial']:
  for wid,q,target in [(examples[0][0],'In(obj_red,drawer)','red'),(examples[2][0],'In(obj_pink,drawer)','pink'),(examples[2][0],'Closed(drawer)','pink'),(examples[3][0],'Intermediate(drawer)','pink'),('C_semantics_v2_chain_pink_0__chain__t01170','Holding(obj_pink)','pink'),('C_semantics_v2_chain_pink_0__chain__t01170','Open(drawer)','pink')]:commands.append({'variant':variant,'window_id':wid,'query':q,'goal_object':target})
 save(R/'tables/query_commands.json',commands)
 (R/'README.md').write_text(f'''# 共享查询网络开发实验

[完整结论](REPORT.md) · [图表](gallery.html)

两组各1200更新已完成；固定规则均选600步，但都未通过感知门槛。检查点为可复核实验结果，没有晋升部署或执行新D。

```bash
source {P}/world_model_env.sh
python {E}/source/query.py --variant spatial \\
  --window-id C_semantics_v2_fixtures_pink_0__empty_closed__t00127 \\
  --query 'In(obj_pink,drawer)' --goal-object pink
```

该例特意展示模型仍存在的错误共同目标；它只返回保存公开观测的神经读出，不调用GT或执行规划。`--variant pooled`可运行同帧对照；其他查询为Holding(obj_red)、Closed(drawer)、Open(drawer)、Intermediate(drawer)。

|目录／脚本|内容和用途|
|---|---|
|`{E}/configs/protocol.json`|训练前架构、预算、种子、选择标准、根划分和门槛|
|`prepare.py`|仅复用942已有特征缓存，生成空间输入和共同采样序列|
|`model.py`|同容量池化／空间网络，所有关系共享查询头|
|`train.py --variant pooled或spatial`|两次固定训练；已有结果防覆盖，不应重复运行|
|`evaluate.py`|全部事实、关系混淆、共同目标、根级分数、门槛和回归|
|`query.py`|CPU实际推理CLI，只加载公开特征及已训练权重|
|`report.py`、`seal.py`|图表、实际查询复核、旧文件完整性和交付清单|
|`{E}/models/pooled`、`models/spatial`|各4个检查点，均保留|
|`{E}/logs`|完整训练与评估日志、两组实际批次索引|
|`{E}/results`|预测概率、每个检查点指标、所有错误与目标判断|
|`{BASE}/data`|原942窗口元数据／输入；targets只供训练和独立评估|

训练使用本机单GPU串行，共2400更新；无新图像采集、无付费API、无新模型下载。报告中的线性探针仅为前轮诊断背景，本轮结果来自247569可训练参数的共享神经查询网络。

只读复核：`python {E}/source/verify_delivery.py`。
''')
 figs=''.join(f'<article><a href="figures/{f.name}"><img src="figures/{f.name}"></a></article>'for f in sorted((R/'figures').glob('*.png')))
 (R/'gallery.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>空间共享查询：开发对照</title><style>body{max-width:1150px;margin:32px auto;padding:0 20px;background:#f3f5f7;color:#193b48;font:17px/1.6 sans-serif}article{padding:20px;background:white;border-radius:10px;margin:24px 0}img{width:100%;height:auto}a{color:#176d83}</style><h1>空间信息保留：共享查询网络对照</h1><p>两组各1200更新；203/209完整状态不等于通过门槛。空间版新增2个共同目标误报，未晋升候选。</p><p><a href="REPORT.md">完整报告</a> · <a href="README.md">运行命令</a></p>'+figs+'</html>');print('SHARED_QUERY_REPORT_GENERATED',flush=True)
if __name__=='__main__':main()
