from common import *
import csv,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
COHORTS=['official_validation','AB_heldout16','C_validation','C_all_available','new_AB_validation','D12_seen_actual135']
NAMES=['官方验证120','A/B训练留出16','C验证273','C全部可用771','A/B既有组合54','D既有诊断135']

def table(cols,rows):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in rows)+'\n'
def main():
 assert read(E/'results/verification.json')['passed'];ms=read(E/'results/actual_metrics.json');M={(r['cohort'],r['head'],r['seed'],r['arm']):r for r in ms};parts=read(E/'results/matched_error_decomposition.json');old=read(E/'results/error_decomposition.json')
 def get(c,h,s,a):return M[c,h,s,a]
 def pair(c,h,s):
  z=get(c,h,s,'zero_context');f=get(c,h,s,'full_context');return f"{z['all3_correct']}→{f['all3_correct']}/{f['all3_valid']}"
 fig,axs=plt.subplots(1,2,figsize=(12,5.4));xx=np.arange(len(COHORTS));seedcolors=['#377eb8','#f08c28']
 for ax,h in zip(axs,HEADS):
  for seed,col in enumerate(seedcolors):
   delta=[100*(get(c,h,seed,'full_context')['all3_accuracy']-get(c,h,seed,'zero_context')['all3_accuracy'])for c in COHORTS];yy=xx+(-.16 if seed==0 else .16);ax.barh(yy,delta,height=.28,color=col,label=f'种子{seed}')
   for y,v in zip(yy,delta):ax.text(v+(.25 if v>=0 else-.25),y,f'{v:+.1f}',ha='left'if v>=0 else'right',va='center',fontsize=9)
  ax.axvline(0,color='#555555',lw=.8);ax.set(yticks=xx,yticklabels=NAMES,title=h,xlabel='完整上下文 − 归一化后置零对照（百分点）',xlim=(-26,18));ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.legend(loc='lower left')
 fig.suptitle('严格同结构对照：额外上下文的收益依赖数据来源',fontsize=14);fig.text(.5,.015,'每个状态须三色都判对；仅计三色标签全有效状态。蓝/橙为两个固定种子，右侧为完整上下文改善，左侧为退化。各组存在相关帧，C全部与C验证重叠。',ha='center',fontsize=9);fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(R/'figures/matched_control_changes.png',dpi=160);plt.close(fig)
 # Re-render the already selected evidence only, without repeating diagnosis or selecting new examples.
 examples=read(E/'results/ROI_review_sources.json')['examples'];raw=np.load(BASE/'data/public/raw_features.npz');fig,axes=plt.subplots(2,4,figsize=(12,8.2));cols=['#df3434','#315ae5','#c832b6']
 for j,r in enumerate(examples):
  z=np.load(r['path'])
  for cam,key in enumerate(['rgb_static','rgb_gripper']):
   im=z[key];h,w=im.shape[:2];ax=axes[cam,j];ax.imshow(im,interpolation='nearest');ax.set(xticks=[],yticks=[],title=r['scene']+(' 静态抽屉区域'if cam==0 else' 腕部完整视野'))
   for k in range(3):
    g=raw['geometry'][r['index'],cam,k]
    if g[4]:ax.add_patch(Rectangle((g[0]*w,g[1]*h),(g[2]-g[0])*w,(g[3]-g[1])*h,fill=False,ec=cols[k],lw=1.2))
   box=BOX if cam==0 else np.array([0.,0.,1.,1.]);ax.add_patch(Rectangle((box[0]*w,box[1]*h),(box[2]-box[0])*w-1,(box[3]-box[1])*h-1,fill=False,ec='#00c4c9',ls='--',lw=2))
 fig.suptitle('相同固定区域跨场景审阅：抽屉、桌面、机械臂和背景共同进入特征',fontsize=13);fig.text(.5,.02,'青色虚框为固定ROI；红/蓝/粉实框为当前RGB检测框。A/B取训练源，C取原新增In误报，D取既有诊断。无GT投影或实例分割。',ha='center',fontsize=9);fig.subplots_adjust(left=.025,right=.985,bottom=.08,top=.91,wspace=.1,hspace=.22);fig.savefig(R/'figures/ABCD_fixed_ROI_review.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,5));labels=['双相机均检测到','仅静态检测到'];x=np.arange(2)
 for seed in(0,1):
  for arm in['zero_context','full_context']:
   offset=(-.3 if seed==0 else .1)+(0 if arm=='zero_context'else .2);vals=[]
   for level in ['both','static_only']:
    r=next(r for r in parts if r['cohort']=='C_validation'and r['head']=='In'and r['seed']==seed and r['arm']==arm and r['dimension']=='visibility'and r['level']==level);vals.append(100*r['FP']/(r['FP']+r['TN']))
   axes[0].bar(x+offset,vals,width=.18,color=seedcolors[seed],alpha=.4 if arm=='zero_context'else 1,label=f'种子{seed} '+('置零'if arm=='zero_context'else'完整'))
 axes[0].set(xticks=x,xticklabels=labels,ylabel='真值为不在抽屉的对象事实中：误报比例（%）',title='C273误报集中在仅静态检测到的对象',ylim=(0,50));axes[0].legend(fontsize=8,ncol=2);axes[0].grid(axis='y',alpha=.18)
 bars=[(0,'zero_context'),(0,'full_context'),(1,'zero_context'),(1,'full_context')];bottom=np.zeros(4)
 for j,c in enumerate(COLORS):
  vals=[get('C_validation','In',s,a)['colors'][j]['FP']for s,a in bars];axes[1].bar(np.arange(4),vals,bottom=bottom,color=cols[j],label=['红对象','蓝对象','粉对象'][j]);bottom+=vals
 for i,b in enumerate(bottom):axes[1].text(i,b+1,str(int(b)),ha='center')
 axes[1].set(xticks=range(4),xticklabels=['种子0\n置零','种子0\n完整','种子1\n置零','种子1\n完整'],ylabel='In误报对象事实数',title='三种颜色均退化，不是单一颜色问题',ylim=(0,145));axes[1].legend();axes[1].grid(axis='y',alpha=.18)
 fig.text(.5,.015,'左：蓝/橙表示种子，浅/深表示置零/完整；两个可见组的负例分母为554、205。另25个双相机未检出的已知事实均为正例，因此不画FPR。右：堆叠色代表对象颜色。',ha='center',fontsize=9);fig.tight_layout(rect=(0,.07,1,1));fig.savefig(R/'figures/C_false_positive_decomposition.png',dpi=160);plt.close(fig)
 flat=[]
 for r in ms:
  for c in r['colors']:flat.append({k:r[k]for k in['cohort','head','seed','arm','states','roots','all3_valid','all3_correct','all3_accuracy','mean_balanced_error']}|c)
 with(R/'matched_metrics.csv').open('w')as f:wr=csv.DictWriter(f,fieldnames=list(flat[0]));wr.writeheader();wr.writerows(flat)
 text='''# 严格同结构零上下文对照与跨场景退化诊断

完整上下文分支暂不采用。相同结构、初始权重、归一化、训练采样和优化下，它仍改善官方验证集的 In，却大幅损害C；原节点/边对照中的隐层宽度差异不足以解释这一退化。本次只完成感知读出诊断，没有证明通用关系图、世界模型或闭环规划成功。

这是看到原实验结果后单独追加的控制实验。原8次训练及报告保持封存；新增4次，每次600更新，两谓词×两种子。完整上下文4个头直接复用原封存权重，40组输出逐位复现，不重新训练或择优。C阈值、权重、ROI均未调整，SupCon继续暂停，未新增数据或调用T。

## 控制的元素与原因

'''
 text+=table(['元素','严格控制','含义与限制'],[
 ['输入与结构','两臂均207→32→SiLU→1；每头6689参数','排除原74→88与207→32隐层宽度不同的混淆'],
 ['零上下文臂','原训练归一化后，将74:207精确置零，训练和推理一致；0:74逐位相同','联合移除端点128维和相对几何5维；不能区分两者各自贡献'],
 ['完整上下文臂','复用原完整边头及冻结分数','不增加参考模型重试或选种子'],
 ['初始化、采样','同一初始张量；每更新24旧状态+8官方状态；同一600×32索引','每个种子独立配对，仅2个种子不能估计稳定统计区间'],
 ['损失与优化','相同掩码加权BCE；AdamW lr=.001、wd=.001、clip=1','类别权重、标签和未知掩码原样保留'],
 ['图像节点','冻结DINO双相机局部patch经训练集PCA；RGB颜色框','全图DINO注意力已引入上下文，置零臂不能称完全无上下文'],
 ['端点','In为固定静态像素区域；Holding为整个腕部视野','没有自动抽屉/夹爪实例分割，不构成通用实例关系图'],
 ['检测与缺失','HSV仅识别已知红/蓝/粉对象；未检测到则该相机局部描述和框为零','遮挡、颜色阈值失配与出视野未区分；未检测到不等于物理不存在'],
 ['机器人信息','本候选的输入链仅RGB，未复用旧含robot_history的对象z','旧网络“对象特征”曾含机器人状态，本次冻结视觉分支与其分开'],
 ['预算','新增4次固定600更新；无C调参、无DINO/T/SupCon训练','额外通道的联合数值干预不等于真实场景因果干预']])
 text+='## 主结果\n\n数字为“置零→完整/三色标签全有效的状态数”，一个状态须三种颜色全对。相关帧不是独立试验；C全部可用包含C验证，不能相加。\n\n'
 text+=table(['数据组','Holding种子0','Holding种子1','In种子0','In种子1'],[[n]+[pair(c,h,s)for h in HEADS for s in(0,1)]for c,n in zip(COHORTS,NAMES)])
 text+='![同结构对照](figures/matched_control_changes.png)\n\n图中横轴为完整减置零的三色全对率百分点，蓝/橙是种子0/1；向右改善、向左退化。Holding方向不一致；In官方验证收益伴随C的17.2/21.6个百分点下降。图不显示统计显著性。\n\n'
 text+='A/B留出16只表示训练根级隔离：此前做过物理审计；官方验证、C和D也均已查看。D135为反复使用的既有诊断，不能声称独立测试。官方验证In有效116/120、A/B留出In有效14/16、C全部In有效754/771；其余未知保留掩码，没有改标签。A/B54是既有组合保持测试，并非新组合独立成功。\n\n'
 text+='## In假阳性/假阴性：保留收益背后的代价\n\nFP=真值为不在抽屉却预测在；FN=真值在抽屉却预测不在。每格为“置零FP/FN→完整FP/FN”。\n\n'
 trs=[]
 for c in ['official_validation','AB_heldout16','C_validation','C_all_available']:
  for s in(0,1):
   z=get(c,'In',s,'zero_context');f=get(c,'In',s,'full_context');trs.append([c,s]+[f"{z['colors'][j]['FP']}/{z['colors'][j]['FN']}→{f['colors'][j]['FP']}/{f['colors'][j]['FN']}"for j in range(3)])
 text+=table(['数据组','种子','红','蓝','粉'],trs)
 text+='官方验证In全对数提高，但红对象新增5/4个FN。A/B16全对数保持9/14，平均颜色平衡错误率却从0.1294升至0.1857，两种子相同；因此不能只凭全对数宣称收益。\n\n'
 text+='## C退化分解：来源、可见性与固定区域\n\n'
 for dimension,title in [('origin','C全部可用：来源分解'),('visibility','C验证273：检测可见性'),('target_bbox_drawer_ROI','C验证273：目标框与固定区域的重叠')]:
  co='C_all_available'if dimension=='origin'else'C_validation';rr=[r for r in parts if r['cohort']==co and r['head']=='In'and r['dimension']==dimension];levels=list(dict.fromkeys(r['level']for r in rr));trs=[]
  for level in levels:
   def lookup(s,a):return next(r for r in rr if r['seed']==s and r['arm']==a and r['level']==level)
   z=lookup(0,'zero_context');tr=[level,f"{z['TP']+z['FN']}/{z['FP']+z['TN']}"]
   for s in(0,1):
    a=lookup(s,'zero_context');b=lookup(s,'full_context');tr.append(f"{a['FP']}/{a['FN']}→{b['FP']}/{b['FN']}")
   trs.append(tr)
  text+=f'**{title}**\n\n'+table(['分组','正例/负例事实数','种子0 FP/FN','种子1 FP/FN'],trs)
 text+='**C验证273：按采集根分解**\n\n'
 roots=[r['level']for r in parts if r['cohort']=='C_validation'and r['head']=='In'and r['dimension']=='root'and r['seed']==0 and r['arm']=='zero_context'];trs=[]
 for root in roots:
  tr=[root]
  for seed in(0,1):
   zz=next(r for r in parts if r['cohort']=='C_validation'and r['head']=='In'and r['dimension']=='root'and r['seed']==seed and r['arm']=='zero_context'and r['level']==root);ff=next(r for r in parts if r['cohort']=='C_validation'and r['head']=='In'and r['dimension']=='root'and r['seed']==seed and r['arm']=='full_context'and r['level']==root);tr.append(f"{zz['FP']}→{ff['FP']}")
  trs.append(tr)
 text+=table(['根ID','种子0误报事实','种子1误报事实'],trs)
 text+='两个种子在6个C验证采集根均增加误报，不能归为单根异常；各根内的多帧仍相关。可见性分组的both/static_only/neither分别为双相机均检测到/仅静态检测到/双相机均未检测到。ROI分组outside/partial/fully分别为目标框与固定区域完全不重叠/部分重叠/完全位于区域内，not_detected_static为静态未检测到。\n\n'
 text+='C273误报从9/7增至111/126。仅静态检测到的205个负例中，FP从两种子均1增至90/91（43.9%/44.4%）；双相机均检测到的554个负例中从8/6增至21/35。双相机都未检测到的25个已知事实均为正例，各臂都判对，因此本组新增误报不能由缺失全零输入解释。三种颜色均退化。C全部可用的旧语义来源与intermediate_v3表现不同，必须保留来源分层，不能把771状态当同质样本。\n\n'
 text+='![C误报分解](figures/C_false_positive_decomposition.png)\n\n左图纵轴是负例误报率，蓝/橙区分种子、浅/深区分置零/完整；右图纵轴为误报事实数，堆叠红/蓝/粉代表对象颜色。一个状态可能贡献多个颜色错误，事实数与状态全对数不可混用。\n\n'
 text+='![固定区域跨场景审阅](figures/ABCD_fixed_ROI_review.png)\n\n青色虚框为各场景同一归一化ROI；红/蓝/粉实框来自当前RGB检测。上排是静态视野，下排是腕部；A/B示例来自训练源，C来自原种子0新增In误报，D来自既有诊断。静态区域覆盖抽屉，同时包含桌面、机械臂和背景；腕部整个视野也含大量木纹/机械臂。示例C蓝对象在桌面、位于固定ROI边缘，已被静态检测到却被完整分支预测为In。只是审阅示例，不是跨域crop质量的随机抽样估计。\n\n'
 ex=[r for r in read(E/'results/decision_flips.json')if r['cohort']=='C_validation'and r['head']=='In'and r['state_id']=='new:828'and r['color']=='blue']
 text+='可追溯例：state_id=new:828，root_id=C_intermediate_v3_blue_2，蓝对象In真值为假。原报告按根/状态/颜色排序选取该例，本轮沿用；不是重新挑选最差样本，也不是闭环时间轴上的首个错误。两种子的“置零→完整”分数分别为'+ '、'.join(f"{r['zero_score']:.4f}→{r['full_score']:.4f}"for r in ex)+'，固定阈值0.5下都由正确转为误报。分数未作概率校准。\n\n'
 text+='目标框与ROI重叠仅是图像几何代理：partial_ROI的大量负例被误报，与固定区域混入桌面一致；它不是抽屉分割IoU，也不证明框内物体在抽屉中。A/B/C/D场景位置大致固定而纹理和机械臂遮挡变化，当前设计带有固定场景先验。没有GT投影或仿真身份进入定位，也没有独立抽屉实例检测，因此不能将本结果表述为通用对象关系图验证。\n\n'
 text+='## 可确认的原因与仍未分离的原因\n\n1. 本次同结构、同初始化干预确认：在当前训练数据和目标下，提供端点视觉与相对几何的联合通道导致这组C误报大幅增加，原对照的隐藏宽度差异不是充分解释。\n2. 原实验对模型输入的只读置换已显示端点与对象通道都会影响输出；本次不重复置换或挑选新样本。它证明特征被使用，不能证明学到了物理关系。\n3. 固定区域、纹理域差异和可见性分层支持上下文泛化失效的解释；没有分别置零端点/几何或更换背景的物理反事实，本轮不能单独判定“背景捷径”是唯一原因。In新增端点来自静态相机；腕部图像仍进入对象节点，两者须区分。\n4. Holding的端点是整个腕部视野，可能编码背景、姿态和采集来源。原端点置换改变预测，但本次Holding收益不一致；缺少同一状态只替换腕部背景的控制，不能据此确认它使用了夹爪关系，也不能把腕部域捷径当已证实事实。\n5. 按训练均值/标准差计算，In端点特征的均值偏移RMS为官方验证0.222、C273 0.467、D135 0.514。D偏移更大却没有相同规模的In退化，故单个漂移指标不能解释性能。完整表在feature_domain_drift.csv；不同分块的维数与方差地板不同，不用其大小排序因果贡献。\n\n'
 text+='原官方验证存在24个双相机都未检测到、标签已知的In事实，其中4正20负；原节点和本次置零表示将其压成同一输入。原表示碰撞审计发现这24个事实至少有4个不可同时正确，A/B16相应9个事实（3正6负）至少有3个。完整上下文能区分一部分，但也引入FN。此为局部表示丢失信息的证据，不能升级为原始双RGB物理不可区分的证明。单帧RGB也不能保证稳定接触/受力/支撑；保留既有标签掩码，不为模型错误改标。\n\n'
 text+='## 同片段正负对与证据边界\n\n同一官方标注片段、同一颜色中同时出现已知正负例，须两类全部判对才算成功。片段和相邻帧相关，不作独立显著性推断。\n\n'
 pp=read(E/'results/matched_pairs.json');prs=[]
 for h in HEADS:
  for s in(0,1):
   row=[h,s]
   for c in COLORS:
    z=next(r for r in pp if r['head']==h and r['seed']==s and r['arm']=='zero_context'and r['color']==c);f=next(r for r in pp if r['head']==h and r['seed']==s and r['arm']=='full_context'and r['color']==c);row.append(f"{z['correct']}→{f['correct']}/{f['eligible']}")
   prs.append(row)
 text+=table(['谓词','种子','红','蓝','粉'],prs)
 text+='In同片段对几乎没有提升，官方全对数收益不能直接解释为关系判别能力增强。所有本轮数据已被查看，阈值固定0.5；未提供新独立episode结论，没有采用边模型、没有重训世界模型、没有推进闭环成功声明。下一步模型/数据/ROI选择不由本报告自动触发。\n\n'
 ver=read(E/'results/verification.json');train=read(E/'results/training.json');text+='## 验证、文件与运行证据\n\n'
 text+=f"新增4次训练均实际执行600更新；训练循环合计{sum(r['seconds']for r in train['runs']):.3f}秒（不含环境启动、I/O和审计），GPU为{train['device']}。使用共享GPU锁，验证数据训练期间被文件访问守卫禁止。独立复算{ver['independently_recounted_confusion_cells']}个混淆矩阵单元、{ver['all3_groups']}组三色指标、40组六事实指标；40组参考分数逐位一致；归一化、初始权重、索引、标签与掩码一致性及训练/评估根级隔离检查通过。原封存清单哈希逐文件校验不变。\n\n"
 text+='- [逐色指标CSV](matched_metrics.csv)：含分母、FP/FN、FPR/FNR和正负来源根数。\n- [严格对照分层CSV](matched_error_decomposition.csv)：按颜色、根、来源、可见性、ROI重叠及交叉分组。\n- [原节点/边只读分层CSV](error_decomposition.csv)与[特征漂移CSV](feature_domain_drift.csv)。\n- [决策翻转CSV](decision_flips.csv)、[六事实指标CSV](combined6_metrics.csv)。\n- [原封存报告](../visual_node_edge_readout_20261018/REPORT.md)。\n- [运行与复核说明](RUN_COMMANDS.md)。\n- [验证记录](../../experiments/visual_edge_zero_context_control_20261018/results/verification.json)、[训练记录](../../experiments/visual_edge_zero_context_control_20261018/results/training.json)、[冻结协议](../../experiments/visual_edge_zero_context_control_20261018/configs/protocol.json)。\n- [交付哈希清单](../../experiments/visual_edge_zero_context_control_20261018/configs/delivery_manifest.json)。\n'
 (R/'REPORT.md').write_text(text)
 (R/'RUN_COMMANDS.md').write_text('''# 已执行命令与复核

运行前 source /home/robot/research/projects/wm_symbolic/world_model_env.sh。脚本为新增实验source目录中的prepare.py、diagnose.py、freeze.py、train.py、evaluate.py、verify.py、report.py、seal.py，按此顺序执行；原实验目录未写入。prepare/diagnose从本任务暂存源码启动，之后使用冻结副本。report只对原已选ROI图重排版，没有重复选择样本或训练。

训练脚本拒绝已有training_attempts.json，评估拒绝已有actual_metrics.json；不要在封存目录重跑会写入结果的命令。若需复核，只读校验delivery_manifest.json中的SHA256以及已有scores与标签计数。完整终端日志在实验logs目录。新增4次训练均600更新，无重试、无种子挑选；原8次结果在原封存目录保留。

训练配置：相同207→32→SiLU→1，batch32（24旧+8官方），AdamW lr0.001/weight_decay0.001，梯度范数clip1，相同掩码BCE。训练归一化后74:207精确置零，0:74保留。Holding正例权重8.545454545，In5.142857075，对三色相同。两个固定种子均从原完整边的初始张量开始。评估固定0.5；不作阈值或ROI搜索。所有新模型保存在实验models目录，未替换正式模型。
''')
 shutil.copy2(__file__,E/'source/report.py');print('MATCHED_CONTROL_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
