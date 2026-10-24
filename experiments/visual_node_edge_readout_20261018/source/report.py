from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager
import shutil
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafbfc','axes.facecolor':'#fafbfc','savefig.facecolor':'#fafbfc'})
def table(h,rows):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in rows)
def main():
 assert read(E/'results/verification.json')['passed'];rows=read(E/'data/public/states.json');raw=np.load(E/'data/public/raw_features.npz');actual=read(E/'results/actual_metrics.json');changes=read(E/'results/changes.json');pairs=read(E/'results/matched_pairs.json');combined=read(E/'results/combined6_metrics.json');swaps=read(E/'results/endpoint_object_swaps.json');train=read(E/'results/training.json');enc=read(E/'results/encoding.json');flips=read(E/'results/decision_flips.json');figdir=R/'figures'
 def get(c,h,s,a):return next(x for x in actual if(x['cohort'],x['head'],x['seed'],x['arm'])==(c,h,s,a))
 colors=['#e03a32','#305be3','#c731b9']
 def show(ax,index,cam,target=None,endpoint=None):
  im=np.load(rows[index]['path'])[['rgb_static','rgb_gripper'][cam]];h,w=im.shape[:2];ax.imshow(im,interpolation='nearest');ax.set(xticks=[],yticks=[])
  for j in(range(3)if target is None else[target]):
   g=raw['geometry'][index,cam,j]
   if g[4]:ax.add_patch(Rectangle((g[0]*w,g[1]*h),(g[2]-g[0])*w,(g[3]-g[1])*h,fill=False,ec=colors[j],lw=1.8))
  if endpoint and ROIS[endpoint]['camera']==cam:
   b=ROIS[endpoint]['box'];ax.add_patch(Rectangle((b[0]*w,b[1]*h),(b[2]-b[0])*w-1,(b[3]-b[1])*h-1,fill=False,ec='#00c6cf',lw=2.3,ls='--'))
  for sp in ax.spines.values():sp.set_visible(False)
  return raw['geometry'][index,cam,:,4]
 idx=next(r['index']for r in rows if r['train']and r['state_id'].startswith('official:'))
 fig,axes=plt.subplots(1,3,figsize=(12,4.5),gridspec_kw={'width_ratios':[1,1,1.45]})
 show(axes[0],idx,0,endpoint='In');axes[0].set_title('静态RGB：物体框与抽屉区域')
 show(axes[1],idx,1,endpoint='Holding');axes[1].set_title('腕部RGB：物体框与夹爪视野')
 axes[2].axis('off');axes[2].text(.02,.92,'同一个冻结DINO骨干',weight='bold',fontsize=13)
 axes[2].text(.02,.80,'对象节点：两相机对象视觉 + 像素框\n74维，保留冻结视觉描述',fontsize=11)
 axes[2].text(.02,.56,'关系边：对象节点 + 相关端点视觉\n+ 图像相对位置，共207维\nHolding → 腕部夹爪视野\nIn → 静态抽屉区域',fontsize=11)
 axes[2].text(.02,.24,'单节点74→88→1 / 关系边207→32→1\n每头均6689参数，跨三种颜色共享\n不输入robot_obs、接触或仿真坐标',fontsize=10)
 fig.suptitle('纯RGB最小路径：已知颜色对象与明确的视觉关系端点',fontsize=15)
 fig.text(.5,.025,'红/蓝/粉实框来自当前RGB颜色区域；青色虚框是固定相机区域先验，既非仿真分割，也不是自动夹爪/抽屉实例检测。',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(figdir/'visual_inputs_and_architecture.png',dpi=165);plt.close(fig)
 cohorts=['official_validation','AB_heldout16','C_validation','C_endpoints48','new_AB_validation','D12_seen_actual135'];names=['官方AB\n120状态/4根','另保留AB\n16状态/4根','旧C验证\n273状态','C端点48','已见AB54','已见D135']
 fig,axes=plt.subplots(1,2,figsize=(13,4.7),sharey=True)
 for ax,head in zip(axes,HEADS):
  for seed,col in[(0,'#357ba3'),(1,'#d57a32')]:
   rr=[next(x for x in changes if(x['cohort'],x['head'],x['seed'])==(c,head,seed))for c in cohorts];vals=[100*x['change']/x['valid']for x in rr];xs=np.arange(len(rr))+(-.18 if seed==0 else .18);ax.bar(xs,vals,width=.32,color=col,label=f'seed{seed}')
   for x,y in zip(xs,vals):ax.text(x,y+(.3 if y>=0 else-.4),f'{y:+.1f}',ha='center',va='bottom'if y>=0 else'top',fontsize=8)
  ax.axhline(0,color='#444',lw=.8);ax.set(xticks=np.arange(6),xticklabels=names,title=head,ylim=(-25,17));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False)
 axes[0].set_ylabel('关系边 − 单节点：三色全对率变化（百分点）');fig.suptitle('相同RGB与参数预算下：In改善伴随明显旧C退化',fontsize=15)
 fig.text(.5,.025,'柱色只表示随机种子；0线为无变化，负值退化。未知标签不算负；实际有效分母见报告。C48包含在C273中，D是反复使用的诊断。',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.08,1,.93));fig.savefig(figdir/'performance_changes.png',dpi=165);plt.close(fig)
 requests=[('official_validation','In','improvement',False,'官方AB：误报修正'),('official_validation','In','regression',True,'官方AB：新增漏检'),('C_validation','In','regression',False,'旧C：新增误报'),('AB_heldout16','In','regression',True,'另保留AB：新增漏检')];examples=[]
 fig,axes=plt.subplots(4,3,figsize=(11,11.8),gridspec_kw={'width_ratios':[1,1,1.25]})
 for row,(cohort,head,effect,truth,title)in enumerate(requests):
  r=sorted([x for x in flips if(x['cohort'],x['head'],x['effect'],x['truth'],x['seed'])==(cohort,head,effect,truth,0)],key=lambda x:(x['root_id'],x['state_id'],x['color']))[0];index=r['index'];j=COLORS.index(r['color']);examples.append(r)
  for cam in(0,1):
   vis=show(axes[row,cam],index,cam,target=j,endpoint=head);axes[row,cam].set_title(title+' / '+('静态'if cam==0 else'腕部')+('（目标未检出）'if not vis[j]else''),fontsize=9)
  d=np.load(E/f'data/evaluation/{cohort}.npz');i=np.where(d['indices']==index)[0][0];off=3+j
  for seed in(0,1):
   p0=np.load(E/f'results/scores/{cohort}_node_seed{seed}.npz')['scores'][i,off];p1=np.load(E/f'results/scores/{cohort}_edge_seed{seed}.npz')['scores'][i,off]
   axes[row,2].bar(seed-.17,p0,width=.30,color='#357ba3',label='单节点'if seed==0 else None);axes[row,2].bar(seed+.17,p1,width=.30,color='#d57a32',label='关系边'if seed==0 else None)
  axes[row,2].axhline(.5,color='#555',ls='--');axes[row,2].set(ylim=(0,1),xticks=[0,1],xticklabels=['seed0','seed1'],ylabel='未校准分数',title=f"In({r['color']})，GT={r['truth']}")
 handles,labs=axes[0,2].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False)
 fig.suptitle('固定规则选取的改善与退化：不挑最好seed，不修补案例',fontsize=14)
 fig.text(.5,.02,'实框为目标颜色检测，青色虚框为抽屉区域；无实框表示当前RGB未检出目标。蓝/橙柱是读取方式，灰虚线是原0.5阈值。',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.09,1,.96));fig.savefig(figdir/'error_examples.png',dpi=145);plt.close(fig);save(E/'results/report_examples.json',{'selection':'seed0 each specified cohort/head/effect/truth, first lexicographic root/state/color; both seeds displayed','examples':examples})
 mainrows=[]
 for c,label in [('official_validation','官方AB验证（已见）'),('AB_heldout16','额外保留AB4根'),('C_validation','旧C验证273'),('C_all_available','全部既有C771'),('C_endpoints48','C端点48'),('new_AB_validation','已见AB54'),('D12_seen_actual135','已见D135')]:
  for h in HEADS:
   vals=[]
   for s in(0,1):
    a=get(c,h,s,'node');b=get(c,h,s,'edge');vals.append(f"{a['all3_correct']}→{b['all3_correct']} / {a['all3_valid']}")
   mainrows.append([label,h,*vals])
 conf=[]
 for c,label in [('official_validation','官方AB120'),('AB_heldout16','保留AB16'),('C_validation','旧C273')]:
  for s in(0,1):
   a=get(c,'In',s,'node');b=get(c,'In',s,'edge')
   for aa,bb in zip(a['colors'],b['colors']):conf.append([label,s,aa['color'],aa['TP']+aa['FN'],aa['TN']+aa['FP'],f"{aa['FP']}→{bb['FP']}",f"{aa['FN']}→{bb['FN']}"])
 pp=[]
 for h in HEADS:
  for s in(0,1):
   for color in COLORS:
    a=next(x for x in pairs if(x['head'],x['seed'],x['arm'],x['color'])==(h,s,'node',color));b=next(x for x in pairs if(x['head'],x['seed'],x['arm'],x['color'])==(h,s,'edge',color));pp.append([h,s,color,f"{a['correct']}→{b['correct']} / {a['eligible']}"])
 six=[]
 for c in ['official_validation','AB_heldout16','C_validation']:
  for s in (0,1):
   a=next(r for r in combined if(r['cohort'],r['seed'],r['arm'])==(c,s,'node'));b=next(r for r in combined if(r['cohort'],r['seed'],r['arm'])==(c,s,'edge'));six.append([c,s,f"{a['correct6']}→{b['correct6']} / {a['valid6']}",f"{a['novel6_correct']}→{b['novel6_correct']} / {a['novel6_valid']}"])
 swaprows=[[r['cohort'],r['head'],r['seed'],r['swap'],f"{r['mean_absolute_score_change']:.3f}",f"{r['threshold_flips']}/{r['facts']}"]for r in swaps if r['cohort']in ['official_validation','C_validation']]
 (R/'REPORT.md').write_text(f'''# 共享视觉编码器、对象节点与显式关系边：有限可行性结果

已完成纯RGB路径、同输入参数量匹配对照及8次固定600更新。**关系边改善了官方AB的In读取，但明显损害旧C，且额外4个AB来源episode没有一致收益。当前结果不支持替换正式模型，也没有证明纯视觉足够。** 不恢复SupCon，不训练世界模型，不追加数据或调参。

## 本次实现与输入审查

旧表征的scene1039包含15维robot_obs，旧对象64维z又通过context接收scene，因此旧z不是纯视觉。本轮重新读取仅有两张RGB的代理文件，从本机缓存的同一个冻结DINOv2-small提取特征；没有使用旧z、旧PCA、旧归一化或旧语义编码器。保存的公开代理严格只含rgb_static、rgb_gripper，编码时禁止读取训练标签和其他实验。

{table(['元素','实际来源与作用','限制'],[
['对象对应','当前RGB的固定HSV红/蓝/粉阈值，每色最大连通区域；两相机分别定位','已知三色集合，没有跨帧追踪、通用实例检测或开放词汇识别'],
['对象节点74维','各相机对象ROI的冻结DINO均值/PCA32各一份，加两套像素框与可见性','DINO是整幅图自注意力，节点ROI描述并非完全没有全局上下文；未检出时该相机对象特征置0'],
['Holding关系端点','固定腕部相机完整视野的2×2视觉网格','这是夹爪视野上下文，不是精确的指尖分割'],
['In关系端点','静态图归一化框[0.10,0.52,0.98,1.00]的2×2视觉网格','根据训练RGB选择的固定抽屉区域；不是自动抽屉检测，可能含桌面或遮挡'],
['关系边207维','目标对象74维 + 端点视觉128维 + 图像相对位置/尺寸/可见性5维','真正额外读取第二个视觉端点；未加入动作、颜色ID、机器人状态或3D仿真坐标'],
['监督与评分','原原生几何/接触标签及未知mask；仿真颜色槽位只用于标签身份','接触、稳定力、仿真分割和物体坐标没有用于候选推理或视觉定位']])}

![实际输入与结构](figures/visual_inputs_and_architecture.png)

节点是冻结视觉信息的载体，关系分支对“目标物体—指定端点”组合评分；并非把两个单物体头重命名。本轮没有消息传递网络、额外视觉骨干、对比压缩或世界模型联合训练。Holding与In各自跨三色共享一个头。共享参数的颜色置换等变性已测试；共享不等于已证明跨颜色泛化。

## 公平对照与证据规模

两臂使用同一RGB、DINO/PCA、标签、根级划分、600更新日程、batch32、AdamW、正权重和0.5阈值。单节点74→88→1与关系边207→32→1各6689参数。网络宽度和输入组织不同，是本轮明确的设计差异；匹配参数量不等于完全相同函数空间。两架构按相同固定seed生成各自初始化，不能称为逐参数相同初始化。8个终点全部保留，没有选最好seed。

Holding训练为旧210状态/36根 + 官方120状态/4根；In为旧156状态/30根 + 同一官方120状态。沿用各头原augmented的每批24旧+8官方日程，两臂逐项相同。共享视觉PCA只在训练并集330张RGB对应的8588个可见对象/端点token上拟合，保留方差约{enc['PCA32_variance_fraction']*100:.1f}%；每头归一化只用对应训练池。

额外保留AB16来自A0109/A0146/B0016/B0052四个源episode，未进入上一轮读取训练；这些帧此前已做物理审阅，不能说完全未见。它只有5个Holding正事实、7个In正事实和2个In未知，红/粉Holding各只有1正事实。官方AB120来自另4根，但已经用于上一轮读取评估，也不是新独立测试。C、AB54与D均按历史用途披露，D只作反复使用的诊断。本轮所有根与训练根隔离，未通过验证结果调整任何参数。

## 实际观测读取结果

表内“单节点→关系边 / 有效状态数”要求三色全部判对。未知不填负，状态帧不是独立episode。

{table(['集合','谓词','seed0','seed1'],mainrows)}

![读取变化](figures/performance_changes.png)

官方AB的In主要减少蓝/粉误报，却引入红漏检；旧C退化主要是可见物体的In误报大量增加。额外AB16的In整组仍9/14，两seed均从零漏检变为红1次、粉1次漏检；平均逐色平衡错误率从0.129升至0.186。因此不能只根据官方AB总分选择关系边。

{table(['集合','seed','颜色','正事实','负事实','FP 单→边','FN 单→边'],conf)}

Holding在官方AB的蓝漏检减少，但红/粉不一致；旧C和D没有整体优势。所有Holding逐色混淆及每根结果见[混淆表](actual_confusion.csv)与[完整指标](../../experiments/visual_node_edge_readout_20261018/results/actual_metrics.json)。

同任务段配对要求一个标注段、同色同时有可靠正负，且全部已知状态都判对：

{table(['谓词','seed','颜色','同段全对 单→边 / 有效段'],pp)}

In蓝配对seed0从4/6降到3/6，其他In配对没有改善；所以逐帧In总分提升也没有转化为稳定的同段区分优势。段内帧相关，配对数不是独立试验数。

## 未见状态组合与端点是否真正参与

以训练并集出现过的完整六事实真值模式为参照，共17种已知模式；仅全部标签有效的状态参与组合统计。这是符号状态组合，不是未见物体种类，也不是新的动作链。

{table(['集合','seed','全6正确 单→边','未见6事实组合正确 单→边'],six)}

官方AB有11个未见组合状态/4根；额外AB只有1个/1根。AB54和C在这个定义下没有未见组合，不能把它们称为新组合泛化证据。关系边没有在11个未见组合上优于节点：seed0从10降至9、seed1保持10。分母过小，不能得出稳健的组合泛化结论。

以下只替换已保存关系输入中的端点或目标特征：同场景内循环置换，保持另一部分固定。分数改变说明模型确实使用两端信息；这是网络内部输入依赖检查，不是物理上可实现的反事实，也不把置换后的旧GT当新世界真值。

{table(['集合','头','seed','替换部分','平均分数绝对变化','阈值翻转/事实'],swaprows)}

在旧C，In对端点置换更敏感（平均变化约0.088/0.094），与端点上下文依赖一致；但这不能单独证明场景背景捷径是退化原因。没有因这个现象调整ROI或补训单例。

## 不可观测性、表示缺失与错误案例

未检出不等于物理不存在。官方AB中有24个已知In事实的物体在两相机都未被颜色登记器检出，其中4正、20负；单节点输入全部归为同一个零视觉向量，确定性的共享分类器至少会错4个事实。额外AB相应9事实含3正、6负，至少错3个。该下界来自本表示的信息丢失，不是原始RGB天然不可区分的证明；关系端点加入上下文后没有出现同样的精确输入冲突。统计规范化了浮点正零/负零，原始审计日志保留。

原始双RGB完全相同却标签冲突的检查没有发现实例，但这不证明所有力、稳定夹持和支撑都可从单帧看出。被遮挡的对象、近接触边界、短时力学稳定性都不能仅凭网络错误断言标签错或网络差。本轮保持原真值与未知mask，分别保存[可见性分层](visibility_metrics.csv)及[表示冲突审计](representation_ambiguity.csv)。没有对困难样本改标签。

![改善与退化案例](figures/error_examples.png)

{table(['选择类别','来源状态','目标','seed0分数 单→边','GT'],[[requests[i][4],r['state_id'],r['color'],f"{r['node_score']:.3f}→{r['edge_score']:.3f}",r['truth']]for i,r in enumerate(examples)])}

案例按seed0、指定集合/改善或退化类别、根/状态/颜色字典序首例选取，两个seed都展示。蓝/橙柱区分模型，实框是目标颜色，青色虚框为抽屉区域，灰线为原阈值。输出是加权BCE训练得到的未校准分数。首个旧C新增误报为new:828的蓝In，分数0.023→0.973；模型错误可追溯，但本轮没有做声称能确定真实视觉因果的单例修补。

## 验证、开销与交付边界

1426个状态对应1356份不同RGB，只共享一次冻结DINO编码；特征提取{enc['feature_seconds']:.2f}秒，含训练PCA共{enc['total_with_PCA_seconds']:.2f}秒，峰值PyTorch分配{enc['peak_allocated_bytes']/2**20:.2f}MiB。8次训练循环合计{sum(r['seconds']for r in train['runs']):.3f}秒，峰值分配{max(r['peak_allocated_bytes']for r in train['runs'])/2**20:.2f}MiB。短运行受启动/缓存影响，不把秒级差异当可靠速度排名；nvidia-smi采样日志另存。

4项必要单元测试通过：纯RGB接口拒绝额外状态字段、颜色对应、缺失对象不补全、端点独立影响与颜色置换等变性/参数量。首次要求逐bit相同的置换测试出现7.45e-9舍入差，保留失败日志后以float32容差1e-7绝对/1e-6相对复核，未改模型。独立复算960个混淆格和80组整组三色指标通过；输入文件/冻结源码/模型哈希、根隔离和两臂共同采样均通过。DINO与T更新0，SupCon训练仍0。

现有数据足以完成这次最小实现与暴露跨场景退化；不足以认证纯视觉普遍足够或新组合稳定泛化。更强结论需要更多独立来源episode与覆盖遮挡/接触边界的证据，不能用当前4根16状态重复试验代替。本轮没有自动采集、下载、修改正式模型或继续扫描。

{table(['交付','内容'],[
['[协议](../../experiments/visual_node_edge_readout_20261018/configs/protocol.json)','输入、固定ROI、对照、划分、预算与已知限制'],
['[输入清单](../../experiments/visual_node_edge_readout_20261018/data/public/states.json)','每张RGB代理、原来源、角色和哈希'],
['[全部决策翻转](decision_flips.csv)','改善与退化逐例分数'],
['[训练日志](../../experiments/visual_node_edge_readout_20261018/results/training.json)','全部8终点与训练损失/耗时/显存'],
['[验证记录](../../experiments/visual_node_edge_readout_20261018/results/verification.json)','实际已执行检查'],
['[运行命令](RUN_COMMANDS.md)','本机环境及入口'],
['[封存清单](../../experiments/visual_node_edge_readout_20261018/configs/delivery_manifest.json)','源码、配置、RGB代理、特征、模型、评估与图表逐文件哈希']])}
''')
 (R/'RUN_COMMANDS.md').write_text('''# 本机已执行入口

```bash
source /home/robot/research/projects/wm_symbolic/world_model_env.sh
python /home/robot/Documents/Codex/2026-10-13/task-4/node_edge_src/prepare.py
python /home/robot/Documents/Codex/2026-10-13/task-4/node_edge_src/encode.py
python /home/robot/Documents/Codex/2026-10-13/task-4/node_edge_src/test_visual.py
python /home/robot/Documents/Codex/2026-10-13/task-4/node_edge_src/freeze.py
python /home/robot/Documents/Codex/2026-10-13/task-4/node_edge_src/train.py
python /home/robot/Documents/Codex/2026-10-13/task-4/node_edge_src/evaluate.py
python /home/robot/Documents/Codex/2026-10-13/task-4/node_edge_src/verify.py
python /home/robot/Documents/Codex/2026-10-13/task-4/node_edge_src/ambiguity.py
python /home/robot/Documents/Codex/2026-10-13/task-4/node_edge_src/report.py
```

编码与训练串行持有项目GPU锁。prepare/encode/freeze/train/evaluate有一次性输出保护，不应重跑覆盖结果；报告、复核也会写文件，封存后另建审计目录。首次单元测试浮点舍入与首次碰撞统计正负零问题的日志保留，均未触发模型重训。完整代码副本位于本轮实验source目录。
''')
 shutil.copy2(__file__,E/'source/report.py');print('NODE_EDGE_REPORT_CREATED',flush=True)
if __name__=='__main__':main()
