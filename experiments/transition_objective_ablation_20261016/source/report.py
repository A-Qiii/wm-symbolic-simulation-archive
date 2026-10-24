from config import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(font);plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none'})
M=read(E/'results/metrics.json');C=read(E/'results/cumulative.json');PP=read(E/'results/predictions.json');LAB=read(DIAG/'data/_private/labels.json');NAMES=['mixed_seed0','latent_seed0','mixed_seed1','latent_seed1'];T={n:read(E/'results'/f'train_{n}.json')for n in NAMES};CAL=read(E/'results/calibration.json');CHOICE=read(E/'configs/choices_frozen.json')['selected']
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |',*['| '+' | '.join(map(str,r))+' |'for r in rows]])
def metric(n,g,branch='normal',path='full4'):return sorted([r for r in M if r['condition']==n and r['group']==g and r['branch']==branch and r['path']==path and r['color']=='all'],key=lambda r:r['global_step'])
def cum(n,g,branch='normal',method='recursive'):return next(r for r in C if r['condition']==n and r['group']==g and r['branch']==branch and r['method']==method and r['color']=='all')
def selected(n):return next(r for r in T[n]['log']if CHOICE[n]['path'].endswith(f'update{r["update"]:04d}.pt'))
def vec(a):return ' / '.join(map(str,a))
def csvout(path,rows):
 with path.open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
flat=[]
for r in M:
 for method in ['actual','one_step','recursive']:
  x=r[method];flat.append({k:r[k]for k in ['condition','group','branch','path','global_step','local_depth','color','roots']}|{'method':method,'full9_correct':x['full9_correct'],'full9_N':x['full9_N'],**{g+'_correct':x['groups'][g]['correct']for g in ['Holding','In','drawer']},'raw_feature':x['feature_mean'],'raw_latent_MSE':x['raw_MSE']})
csvout(R/'step_summary.csv',flat);csvout(R/'cumulative_summary.csv',[{k:v for k,v in r.items()if k!='prefix_correct'}|{f'prefix_h{i+1}':v for i,v in enumerate(r['prefix_correct'])}for r in C])
csvout(R/'checkpoint_raw_metrics.csv',[{'condition':n,'update':r['update'],'selected':CHOICE[n]['path'].endswith(f'update{r["update"]:04d}.pt'),'group':g,**{k:v for k,v in d.items()if not isinstance(v,list)},**{f'rawF_h{i+1}':v for i,v in enumerate(d['recursive_feature'])},**{f'rawZ_h{i+1}':v for i,v in enumerate(d['recursive_rawZ'])},'C_single_rawF':r['C_single_feature'],'C_single_rawZ':r['C_single_rawZ'],'AB_single_rawF':r['AB_single_scheduleweighted_feature'],'AB_single_rawZ':r['AB_single_scheduleweighted_rawZ']}for n,t in T.items()for r in t['log']for g,d in r['same_metric_AB_C'].items()])
lookup={(r['condition'],r['root_id'],r['branch'],r['path'],r['global_step']):r for r in PP};pair=[]
def margin(logits,j,y):
 if j<6:return logits[j]*(1 if y else-1)
 return (logits[j]-max(logits[k]for k in range(6,9)if k!=j))*(1 if y else-1)
for r in PP:
 if not r['condition'].startswith('mixed'):continue
 z=lookup[(r['condition'].replace('mixed','latent'),r['root_id'],r['branch'],r['path'],r['global_step'])];lab=LAB[str(r['after'])]
 for j,f in enumerate(FACTS):
  row={k:r[k]for k in ['group','root_id','color','branch','path','global_step','skill']}|{'seed':int(r['condition'][-1]),'fact':f,'truth':lab['y'][j],'valid':lab['v'][j]}
  for name,rr,method in [('actual',r,'actual'),('mixed_one',r,'one_step'),('latent_one',z,'one_step'),('mixed_recursive',r,'recursive'),('latent_recursive',z,'recursive')]:row.update({name+'_prediction':bool(discrete(rr[method+'_probability'])[j]),name+'_signed_logit_margin':float(margin(rr[method+'_logits'],j,lab['y'][j]))})
  pair.append(row)
csvout(R/'objective_proposition_pairs.csv',pair)
first=[]
for n in NAMES:
 for root,branch in sorted({(r['root_id'],r['branch'])for r in PP}):
  rr=sorted([r for r in PP if r['condition']==n and r['root_id']==root and r['branch']==branch and r['path']=='full4'],key=lambda r:r['global_step'])
  for r in rr:
   lab=LAB[str(r['after'])];bad=discrete(r['recursive_probability'])!=np.asarray(lab['y'])
   if bad.any():
    first.append({'condition':n,'group':r['group'],'root_id':root,'branch':branch,'first_wrong_step':r['global_step'],'actual_full9_correct':bool(np.array_equal(discrete(r['actual_probability']),lab['y'])),'wrong_facts':';'.join(FACTS[j]for j in np.where(bad)[0]),'signed_logit_margins':';'.join(f'{margin(r["recursive_logits"],j,lab["y"][j]):.6f}'for j in np.where(bad)[0])});break
csvout(R/'first_wrong_boundary.csv',first)
colors={'mixed':'#2878B5','latent':'#DC8428'};labels={'mixed':'混合目标 F+0.25Z','latent':'仅Z目标 αZ（α固定）'}
fig,axs=plt.subplots(2,2,figsize=(12,8.4),sharex=True,sharey=True)
for seed in [0,1]:
 for col in [0,1]:
  ax=axs[seed,col];base=metric(f'mixed_seed{seed}','C_validate');actual=[r['actual']['full9_correct']for r in base]if col==0 else cum(f'mixed_seed{seed}','C_validate',method='actual')['prefix_correct'];ax.plot(range(1,5),actual,'s--',color='#747B83',label='真实状态读出（两臂相同）',lw=1.5)
  for arm in ['mixed','latent']:
   n=f'{arm}_seed{seed}';vals=[r['recursive']['full9_correct']for r in metric(n,'C_validate')]if col==0 else cum(n,'C_validate')['prefix_correct'];ax.plot(range(1,5),vals,color=colors[arm],marker='o'if arm=='mixed'else'D',ls='-'if arm=='mixed'else'--',lw=2,label=labels[arm])
  ax.set_title(f'种子{seed} · '+('当前边界全9正确'if col==0 else'截至当前边界全部正确'));ax.set_xticks(range(1,5),['1 开屉','2 抓起','3 放置','4 关屉']);ax.set_ylim(-.2,6.4);ax.set_yticks(range(7));ax.set_ylabel('正确根数 / 6');ax.set_xlabel('技能边界');ax.grid(axis='y',alpha=.2)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False,fontsize=10)
fig.suptitle('固定三步监督，只改变转移目标：后段出现收益，但种子0放置后退化',y=.98,fontsize=14)
fig.text(.5,.025,'同6条C核验普通链；混合目标选750更新，仅Z选1000更新，均由共同C多步原始Z选点。\n预定要求两个种子的放置后、关屉后与整链均增加至少1根；种子0放置后5→4，因此未晋级。',ha='center',fontsize=10)
fig.subplots_adjust(top=.83,bottom=.14,hspace=.4,wspace=.2)
for ext in ['png','svg']:fig.savefig(R/f'C_step_and_prefix.{ext}',dpi=160)
plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
for seed in [0,1]:
 for col,key in enumerate(['common_multistep_Z','common_multistep_F']):
  ax=axs[seed,col]
  for arm in ['mixed','latent']:
   n=f'{arm}_seed{seed}';ls=T[n]['log'];v=[r['same_metric_AB_C']['C_select'][key]for r in ls];ax.plot([r['update']for r in ls],v,color=colors[arm],marker='o'if arm=='mixed'else'D',ls='-'if arm=='mixed'else'--',label=labels[arm]);b=selected(n);ax.scatter([b['update']],[b['same_metric_AB_C']['C_select'][key]],s=140,facecolors='none',edgecolors=colors[arm],lw=2,zorder=5)
  ax.set_title(f'种子{seed} · '+('共同C多步原始Z（选点）'if col==0 else'同口径C多步原始F（诊断）'));ax.set_xlabel('参数更新次数');ax.set_ylabel('原始latent MSE，越低越好'if col==0 else'原始特征误差，越低越好');ax.set_xticks([0,250,500,750,1000]);ax.grid(alpha=.2)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
fig.suptitle('只比较共同原始指标；不跨臂比较缩放后的训练loss',y=.99,fontsize=14)
fig.text(.5,.02,'同6条C选点普通链，从真实初态递归4步；h2/h3/h4权重0.25/0.50/0.25。\n空心圈标出实际采用检查点；两列单位不同，F不参与选点。未用C核验或D重选。',ha='center',fontsize=10)
fig.subplots_adjust(top=.83,bottom=.15,hspace=.45,wspace=.25)
for ext in ['png','svg']:fig.savefig(R/f'C_raw_metric_curves.{ext}',dpi=160)
plt.close(fig)

out=[]
def add(x):out.append(x)
add('# 固定三步监督下的转移目标消融\n\n2026-10-16，本机完成。**预定双种子晋级门槛未通过：种子0放置后全9正确由5/6降至4/6。** 仅Z目标降低了两个种子的C多步latent MSE，并改善关屉后与整链预测，但未实现所要求的全部后段一致改善。本轮已停止，不追加调参，不采新D，不替换正式模型。')
add('这次实验把“目标与读出局部不一致”的机制推进到网络训练验证：仅改变转移目标，固定三步监督跨度、数据、初始transition、encoder/decoder与修正query。结果支持“向真实后继latent拟合能改善部分预测”，尚不支持“仅Z目标更可靠到可正式采用”。以下C数据已反复用于开发；没有新增独立泛化证据。轨迹属于既有CALVIN PhysicalOracle对照研究，本次不代表原HULC/RSSM/TS-LTL全方案完成。')
add('## 预定门槛与原始计数\n\n箭头均为混合目标→仅Z；每个边界的全9正确要求Holding三命题、In三命题和抽屉三分类全部正确。“整链”指四个边界预测全对，不代表实际任务执行成功。')
rows=[]
for seed in [0,1]:
 a=f'mixed_seed{seed}';b=f'latent_seed{seed}';aa=metric(a,'C_validate');bb=metric(b,'C_validate');rows.append([seed,f'{CHOICE[a]["C_score"]:.6f}→{CHOICE[b]["C_score"]:.6f}',f'{aa[2]["recursive"]["full9_correct"]}/6→{bb[2]["recursive"]["full9_correct"]}/6',f'{aa[3]["recursive"]["full9_correct"]}/6→{bb[3]["recursive"]["full9_correct"]}/6',f'{cum(a,"C_validate")["all4_correct"]}/6→{cum(b,"C_validate")["all4_correct"]}/6','未通过'if seed==0 else'通过'])
add(table(['种子','C选点多步原始Z ↓','C核验放置后','C核验关屉后','C核验整链','单种子门槛'],rows))
add('门槛在训练前固定：两个种子各自的C核验放置后、关屉后、四边界全对数都须至少增加1/6；前两步递归与每个单步预测最多损失1/6，旧/新AB各12根的整链最多损失1根，真实状态读出必须逐概率相同。种子0仅放置后条件失败，其余条件通过；种子1全部通过。总门槛要求两种子都通过，故失败。C选点Z只用于共同选点及诊断，未添加事后门槛；没有利用收益修改规则。')
add('![C逐步与整链](C_step_and_prefix.png)\n\n图1：蓝实线是混合目标，橙虚线是固定缩放后的仅Z目标，灰方点虚线是真实状态读出控制。横轴四个技能边界，纵轴6根中正确根数；左列独立计当前边界，右列要求截至该步始终全对。上/下行分别为种子0/1。橙线的末步与整链上升，同时种子0放置后下降，直接显示收益与退化并存。')
add('## 对照和校准\n\n每种子两臂完全复用原1000×64单步调度：每批48个旧AB、12个新普通、4个新持物关屉样本。多步固定为原三步基线：每4次更新一次、48条起点，12个新AB根的[开/抓/放]与[抓/放/关]窗口各重复2次；局部深度2、3各乘0.25。两臂初始transition相同，网络结构、感知、归一化、decoder和对应种子的修正query全部冻结。失败调用和严格In=false的状态保留，没有成功筛选或标签反传。')
add(table(['项','混合目标','仅Z目标','作用'],[
['每个监督目标','F+0.25Z','αZ，α=3.341086352737259','Z是原始4×64latent MSE，无query标准化'],
['单步/多步系数','1；0.25、0.25','相同','保持抽样、跨度与位置权重'],
['优化器','AdamW lr1e-4，weight_decay1e-4，clip5','相同','仅transition可训练'],
['预算','两种子各1000更新','相同','合计4000更新，不追加'],
['选点','共同C多步原始Z，最低且同分取早','相同','候选0/250/500/750/1000'],
['采集与采用','无新增数据/仿真','无新增数据/仿真','候选实验，正式模型不变']]))
add('梯度校准在任何网络更新前进行，只取已冻结种子0/1调度的前32个批次，各32、共64，其中16个包含既定多步目标；每个索引都验证为AB训练样本。对同一初始transition计算完整参数梯度、在裁剪前取L2范数。两个种子共用一个系数：')
add('`α = median(||∇θ(F+0.25Z)||) / median(||∇θ Z||) = 0.6334645152 / 0.1895983666 = 3.3410863527`')
add('因此校准批次上的缩放后仅Z梯度范数中位数也为0.6334645152。系数及64条原始记录先封存，之后不自适应、不看C调整。这里的F、Z均包括该批次固定单步/多步权重。相对于保留0.25Z系数，当前α约大13.36倍，这是显式梯度尺度控制，不是隐含学习率扫描。')
rows=[]
for n,t in T.items():
 g=[r['gradient_norm_before_clip']for r in t['batch_log']];rows.append([n,f'{np.median(g):.6f}',f'{max(g):.6f}',sum(v>5 for v in g),f'{t["elapsed_seconds"]:.3f}s'])
add(table(['运行','训练期间梯度范数中位数','最大值','发生裁剪次数/1000','运行计时'],rows))
add('初始中位范数相同不意味着Adam优化等价：梯度方向、逐参数一二阶矩、epsilon、权重衰减与后续梯度分布仍不同；本轮没有触发clip5。缩放后的loss不能跨臂比较，报告只比较原始Z和原始F。计时包含缓存latent上的训练、检查点评估与保存，不含早期特征缓存/仿真，亦不作为效率胜负依据。每臂单次运行均处理64000单步样本和12000多步起点、100000次转移状态评估；仅Z无需通过decoder反传，算力并非完全等价。')
add('## 共同选点和同口径训练诊断\n\n同6条C选点普通链从真实初态递归4步，以原始Z在h2/h3/h4按0.25/0.50/0.25加权选最低点。C核验6根及D均未进入选择。旧单步C指标仍按旧37根、新6根分别根平均再各半汇总，只作诊断。')
rows=[]
for n in NAMES:
 r=selected(n);rows.append([n,r['update'],f'{r["score"]:.6f}',f'{r["same_metric_AB_C"]["C_select"]["common_multistep_F"]:.6f}',f'{r["C_single_rawZ"]:.6f}',f'{r["C_single_feature"]:.6f}'])
add(table(['运行','选点更新','C多步Z ↓','同口径C多步F ↓','C单步Z ↓','C单步F ↓'],rows))
add('混合目标两个种子均选750更新，仅Z均选1000更新。这是同一规则得到的不同选点；并非按C核验挑选。独立复核确认本轮混合臂的全部10个候选检查点参数与上一阶段三步基线逐元素完全相同，最大差0；变化的是共同选点规则，不是基线训练实现。')
add('![共同原始指标](C_raw_metric_curves.png)\n\n图2：横轴是参数更新次数，蓝/橙分别为混合/仅Z目标；左列纵轴是原始多步Z并用于选点，右列纵轴是同口径F仅供诊断，两列单位不同。空心圈是正式选点。两种子的Z下降约24%，但F未同步一致变好；这一现象说明更接近真实latent与更好重构并非同一目标。')
rows=[]
for n in NAMES:
 r=selected(n)
 for g in ['old_AB','new_AB','C_select']:
  d=r['same_metric_AB_C'][g];rows.append([n,g,d['roots'],f'{d["common_multistep_Z"]:.6f}',f'{d["common_multistep_F"]:.6f}'])
add(table(['运行','集合','根N','同口径多步Z ↓','同口径多步F ↓'],rows))
add('AB表与C使用完全相同的初态递归及h2/h3/h4权重。完整检查点逐步F/Z、AB精确单步采样频率加权指标及C单步指标见 `checkpoint_raw_metrics.csv`；逐更新原始F/Z与梯度范数见各 `train_*.json`，避免将不同批次的训练loss曲线误认作同一固定集合。')
add('## 普通链逐阶段原始n/N\n\n每串顺序为开屉/抓起/放置/关屉；所有数字是对应N根中的全9正确根数。真实读出输入真实后态；单步预测每次输入真实前态；初态递归不在中途注入真实状态。')
rows=[]
for g in ['old_AB','new_AB','C_select','C_validate']:
 for n in NAMES:
  rr=metric(n,g);rows.append([g,n,rr[0]['roots'],vec([r['actual']['full9_correct']for r in rr]),vec([r['one_step']['full9_correct']for r in rr]),vec([r['recursive']['full9_correct']for r in rr]),vec(cum(n,g)['prefix_correct'])])
add(table(['集合','运行','N','真实读出','单步预测','初态递归','递归前缀全对'],rows))
add('旧AB的仅Z整链两个种子均12/12；新AB仍各11/12。C选点整链均4/6→5/6；C核验整链分别2/6→3/6、1/6→4/6。早期预测与单步保留条件通过。真实状态读出在两臂逐概率、逐logit完全一致，不能将预测变化解释成感知或query改动。')
rows=[]
for g in ['old_AB','new_AB','C_select','C_validate']:
 for n in NAMES:
  rr=metric(n,g);rows.append([g,n,rr[0]['roots'],*[vec([r['recursive']['groups'][q]['wrong']for r in rr])for q in ['Holding','In','drawer']]])
add(table(['集合','运行','N','Holding错误h1/2/3/4','In错误h1/2/3/4','drawer错误h1/2/3/4'],rows))
add('Holding表示持有哪个物体；In表示严格在抽屉内、受支撑且未持有；drawer为开/关/中间三分类。组错误是该组任一命题错的根数，各组可能重叠，不能相加成总错误。按颜色与各路径的全体计数见 `step_summary.csv`；直接跨目标对齐的7560条命题记录见 `objective_proposition_pairs.csv`。其signed logit margin相对于分类边界，正值为正确侧，负值为错误侧，不解释为校准置信度。')
add('## 第一个有影响的错误\n\n阻止晋级的是种子0的 `C_intermediate_v3_pink_3`：前两步两臂都对，放置后真实Holding(pink)=false；固定真实状态读出logit为−8.050，混合预测为−0.895（对），仅Z预测为+0.299（错）。这把该根的第一个错误从关屉后提前到放置后，使种子0的h3从5/6降至4/6。标签和读出头没变，不能把它归因为读出头被重新训练；但此处也不能仅凭全局Z下降断言局部语义会改善。')
add('收益也定位到具体边界：种子0 `pink_2` 关屉后的Holding误报从logit+1.191变为−0.070，刚跨到正确侧；种子1 `blue_3` 关屉后+0.440→−0.347、`pink_2`关屉后+2.153→−0.279，以及`red_3`放置后+0.329→−0.834均被修正。种子0一项改善距离边界较近，故只报告实际分类与margin，不称为高置信或稳定保证。`red_2` 的严格In与抽屉失败状态仍有错误，且部分真实状态读出本来就错。')
add('没有针对这些单例继续训练或改阈值。既定“用真实开屉后latent替换一次”的结果与原始预测一同保存于`metrics.json`的`window1`路径；它同时缩短递归深度和消除首步预测误差，只作诊断，不是新模型或晋级依据。')
add('## 持物关屉分支\n\n它未加入新的多步监督，仍只保留原单步每批4个样本。下表是预测全9正确n/N，不能解释为控制器拒绝率。')
rows=[]
for g in ['new_AB','C_select','C_validate']:
 for n in NAMES:
  rr=metric(n,g,branch='held_closed');rows.append([g,n,rr[0]['roots'],vec([r['actual']['full9_correct']for r in rr]),vec([r['one_step']['full9_correct']for r in rr]),vec([r['recursive']['full9_correct']for r in rr]),cum(n,g,branch='held_closed')['all4_correct']])
add(table(['集合','运行','N','真实读出h1/2/3/4','单步h1/2/3/4','递归h1/2/3/4','整链全对'],rows))
add('仅Z在此分支的若干早期和末步边界出现明显改善，但第三边界所有模型仍0/N，因此整链仍全部0。该改善保留为真实观察，不能替代普通链预定门槛，更不能据此宣称拒绝分支已可靠。')
add('## 结论与交接\n\n本次控制实验表明，训练目标确实会改变符号预测：仅Z把网络输出拉近真实后继latent，改善旧AB训练拟合、C末步/整链以及部分持物关屉边界。与此同时，种子0放置后出现新增Holding误报，显示更低原始Z仍不足以保证共享query语义逐步单调改善。当前只能作为有局部收益的候选证据，**不能按本轮门槛晋级或正式采用**。')
add('协议要求失败后停止；已执行。没有新增D独立测试，也没有用旧D选择、调整α或模型。正式模型未替换。父线程可据完整证据决定后续方向，本轮不自动扩展成追加训练或参数搜索。')
add('## 文件与复核\n\n实验目录：`'+str(E)+'`；报告目录：`'+str(R)+'`。')
add(table(['文件','内容'],[
['configs/protocol.json / protocol_frozen.json','校准前固定的协议、上限及输入源码哈希'],['results/calibration.json / configs/training_frozen.json','64批AB参数梯度原始记录、固定α及训练前第二次封存'],['configs/choices_frozen.json','评分前冻结选点：mixed750、latent1000'],['results/train_*.json / models/','1000更新批次原始F/Z/梯度与全部检查点'],['results/metrics.json / cumulative.json','784个聚合组及根级前缀全对计数'],['paired_propositions.csv / objective_proposition_pairs.csv','15120条单模型事实与7560条跨目标直接配对'],['step_summary.csv / cumulative_summary.csv / first_wrong_boundary.csv','全体和颜色分组n/N、累计及首错边界'],['results/gate.json','预定门槛：种子0失败、种子1通过、总体失败'],['results/verification.json / configs/delivery_manifest.json','独立复核与最终SHA256封存']]))
add('独立复核已通过：55项冻结哈希核验、135个既有封存文件未改变；固定α及64个AB校准批次匹配；两个种子的全量三步与单步调度一致；四组初始模型相同且冻结组件未变；混合目标10个候选参数与原三步基线完全相同；784组全9/分组错误与全部累计计数重新核对一致；真实读出逐概率/logit一致；4000更新完成且无梯度裁剪。两张图已检查坐标、颜色、分母、含义与可读性。')
add('只读复验：\n\n```bash\nsource /home/robot/research/projects/wm_symbolic/world_model_env.sh\npython /home/robot/research/projects/wm_symbolic/experiments/transition_objective_ablation_20261016/source/verify.py\n```')
(R/'REPORT.md').write_text('\n\n'.join(out)+'\n');print('REPORT_WRITTEN',len(pair),'direct proposition pairs',flush=True)
