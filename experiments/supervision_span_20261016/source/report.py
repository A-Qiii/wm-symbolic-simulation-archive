from config import *
import csv, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties

font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fontManager.addfont(font)
plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none'})
M=read(E/'results/metrics.json'); C=read(E/'results/cumulative.json')
PRED=read(E/'results/predictions.json'); LAB=read(DIAG/'data/_private/labels.json')
names=['span3_seed0','span4_seed0','span3_seed1','span4_seed1']
groups=['new_AB','C_select','C_validate']
gname={'new_AB':'A/B训练','C_select':'C选点','C_validate':'C核验'}
mn={'actual':'真实状态读出','one_step':'真实前态单步','recursive':'初态递归'}
def metric(name,g,method='recursive',branch='normal',path='full4'):
    return sorted([r for r in M if r['condition']==name and r['group']==g and r['branch']==branch and r['path']==path and r['color']=='all'],key=lambda r:r['global_step'])
def cum(name,g,method='recursive',branch='normal'):
    return next(r for r in C if r['condition']==name and r['group']==g and r['branch']==branch and r['method']==method and r['color']=='all')
def vec(v):return ' / '.join(map(str,v))
def csvout(path,rows):
    with path.open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |'for r in rows]])

flat=[]
for r in M:
    for method in mn:
        x=r[method]
        flat.append({k:r[k]for k in ['condition','group','branch','path','global_step','local_depth','color','roots']}|{'method':method,'full9_correct':x['full9_correct'],'full9_N':x['full9_N'],**{k+'_correct':x['groups'][k]['correct']for k in ['Holding','In','drawer']},'feature_mean':x['feature_mean'],'raw_latent_MSE':x['raw_MSE']})
csvout(R/'step_summary.csv',flat)
csvout(R/'cumulative_summary.csv',[{k:v for k,v in r.items()if k!='prefix_correct'}|{f'prefix_h{i+1}':v for i,v in enumerate(r['prefix_correct'])}for r in C])
logs={n:read(E/'results'/f'train_{n}.json')for n in names}
choices=read(E/'configs/choices_frozen.json')['selected']
csvout(R/'checkpoint_scores.csv',[{'condition':n,'update':x['update'],'common_C_multistep_feature':x['score'],'original_C_single_feature':x['original_single_C_score'],'selected':choices[n]['path'].endswith(f'update{x["update"]:04d}.pt'),**{f'C_recursive_h{i+1}_feature':v for i,v in enumerate(x['C_recursive_step_feature'])}}for n in names for x in logs[n]['log']])

# Describe the first incorrect boundary in every root, with readout controls.
first=[]
for name in names:
    for g in groups:
        for branch in ['normal','held_closed']:
            roots=sorted({r['root_id']for r in PRED if r['group']==g and r['branch']==branch})
            for root in roots:
                rr=sorted([r for r in PRED if r['condition']==name and r['group']==g and r['branch']==branch and r['root_id']==root and r['path']=='full4'],key=lambda r:r['global_step'])
                for r in rr:
                    lab=LAB[str(r['after'])];y=np.asarray(lab['y']);v=np.asarray(lab['v']);a=discrete(r['actual_probability']);p=discrete(r['recursive_probability']);bad=v&(p!=y)
                    if bad.any():
                        first.append({'condition':name,'group':g,'branch':branch,'root_id':root,'first_wrong_step':r['global_step'],'actual_full9_correct':bool(v.all()and np.array_equal(a,y)),'wrong_facts':';'.join(FACTS[i]for i in np.where(bad)[0]),'truth_on_wrong_facts':';'.join(str(int(y[i]))for i in np.where(bad)[0]),'recursive_probability_on_wrong_facts':';'.join(f'{r["recursive_probability"][i]:.8f}'for i in np.where(bad)[0]),'actual_probability_on_wrong_facts':';'.join(f'{r["actual_probability"][i]:.8f}'for i in np.where(bad)[0])})
                        break
csvout(R/'first_wrong_boundary.csv',first)

colors={'span3':'#2878B5','span4':'#E08028'}
fig,axs=plt.subplots(2,2,figsize=(12,8.4),sharex=True,sharey=True)
for seed in [0,1]:
    for col,mode in enumerate(['step','prefix']):
        ax=axs[seed,col]
        actual=metric(f'span3_seed{seed}','C_validate')
        av=[r['actual']['full9_correct']for r in actual]if mode=='step'else cum(f'span3_seed{seed}','C_validate','actual')['prefix_correct']
        ax.plot(range(1,5),av,color='#6D737A',marker='s',ls='--',lw=1.6,label='真实状态读出（两臂相同）',zorder=2)
        for arm in ['span3','span4']:
            n=f'{arm}_seed{seed}'
            vals=[r['recursive']['full9_correct']for r in metric(n,'C_validate')]if mode=='step'else cum(n,'C_validate')['prefix_correct']
            ax.plot(range(1,5),vals,color=colors[arm],marker='o'if arm=='span3'else'D',ls='-'if arm=='span3'else'--',lw=2,label='三步监督'if arm=='span3'else'四步监督',zorder=3)
        ax.set_title(f'种子{seed} · '+('当前边界9事实全对'if mode=='step'else'截至当前边界始终全对'))
        ax.set_ylim(-.2,6.4);ax.set_yticks(range(7));ax.set_xticks(range(1,5),['1 开屉','2 抓起','3 放置','4 关屉']);ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('正确根数 / 6');ax.set_xlabel('真实动作链中的技能边界')
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False)
fig.suptitle('监督跨度对照：C核验6条普通链，四组模型均按共同标准选中第250更新',y=.98,fontsize=14)
fig.text(.5,.02,'同一批反复用于开发的C根；左图逐步正确，右图要求之前每步也正确。\n这些计数是预测正确性，不是任务成功率或新测试集泛化成绩。',ha='center',fontsize=10)
fig.subplots_adjust(top=.84,bottom=.14,hspace=.42,wspace=.18)
for ext in ['png','svg']:fig.savefig(R/f'C_step_and_prefix.{ext}',dpi=160)
plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
for seed in [0,1]:
    for col,(key,title)in enumerate([('score','共同多步C分数（用于选点）'),('original_single_C_score','原单步C分数（只作诊断）')]):
        ax=axs[seed,col]
        for arm in ['span3','span4']:
            n=f'{arm}_seed{seed}';ls=logs[n]['log'];ax.plot([x['update']for x in ls],[x[key]for x in ls],color=colors[arm],marker='o'if arm=='span3'else'D',ls='-'if arm=='span3'else'--',label='三步监督'if arm=='span3'else'四步监督')
            chosen=next(x for x in ls if choices[n]['path'].endswith(f'update{x["update"]:04d}.pt'))
            ax.scatter([chosen['update']],[chosen[key]],s=130,facecolors='none',edgecolors=colors[arm],lw=2,zorder=4)
        ax.axvline(250,color='#aaa',ls=':',lw=1);ax.set_title(f'种子{seed} · {title}');ax.set_xlabel('参数更新次数');ax.set_ylabel('特征误差（越低越好）');ax.set_xticks([0,250,500,750,1000]);ax.grid(alpha=.2)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
fig.suptitle('两臂使用相同多步C选点规则；空心圈标出实际采用的检查点',y=.99,fontsize=14)
fig.text(.5,.02,'左：同6条C选点链从初态递归，h2/h3/h4权重0.25/0.50/0.25。\n右：原C单步分数保留比较，两列尺度与汇总对象不同；C核验与D均不参与选点。',ha='center',fontsize=10)
fig.subplots_adjust(top=.84,bottom=.15,hspace=.45,wspace=.25)
for ext in ['png','svg']:fig.savefig(R/f'C_selection_curves.{ext}',dpi=160)
plt.close(fig)

out=[]
def add(s):out.append(s)
add('# 同根监督跨度有限对照：结果封存\n\n2026-10-16，本机执行。预定双种子晋级门槛**未通过**，已停止本阶段；没有采集新D、没有追加更新、没有改阈值或重选模型。')
add('把三步窗口改成从真实初态连续监督四步，没有产生跨种子一致的末步和整链收益。种子0的C核验末步及整链各增加1/6，种子1末步与整链均不变；两个种子的共同C多步特征分数均略差。该结论限于固定数据、当前特征/潜变量目标与1000更新预算，不能推断四步监督一般无效。')
add('这里的执行轨迹来自既有CALVIN PhysicalOracle对照实验；本次只训练转移模块，未运行机器人或规划闭环。它不是原HULC/RSSM/TS-LTL全方案完成或验收。C已多次用于开发，下面的C结果不能称为新泛化证据。')
add('## 先看判定\n\n所有分母为根数，不是帧数。同一批C核验6根，每色2根。h3为放置后，h4为关屉后；“整链”要求四个边界的9个事实全部正确，不代表物理任务成功。')
gate_rows=[]
for seed in [0,1]:
    a=f'span3_seed{seed}';b=f'span4_seed{seed}';ra=metric(a,'C_validate');rb=metric(b,'C_validate')
    gate_rows.append([seed,f'{choices[a]["C_score"]:.6f} → {choices[b]["C_score"]:.6f}',f'{ra[2]["recursive"]["full9_correct"]}/6 → {rb[2]["recursive"]["full9_correct"]}/6',f'{ra[3]["recursive"]["full9_correct"]}/6 → {rb[3]["recursive"]["full9_correct"]}/6',f'{cum(a,"C_validate")["all4_correct"]}/6 → {cum(b,"C_validate")["all4_correct"]}/6',f'{cum(a,"new_AB")["all4_correct"]}/12 → {cum(b,"new_AB")["all4_correct"]}/12','未通过'])
add(table(['种子','C选点多步误差 ↓','C核验h3','C核验h4','C核验整链','AB训练整链','门槛'],gate_rows))
add('表中箭头均为三步监督→四步监督。预先要求两个种子都满足：共同C多步分数严格更低；C核验h3不降、h4至少多1根、h3+h4合计至少多2个正确边界、整链至少多1根；早期和单步退化不超过1/6，AB整链退化不超过1/12。种子0只多1个晚期正确边界且C分数未改善；种子1的C分数、h4、晚期合计和整链条件均未满足。其余保留条件通过。门槛是预先固定的推进规则，不是显著性检验。完整布尔判定见 `results/gate.json`。')
add('![C核验逐步与累计正确](C_step_and_prefix.png)\n\n图1：蓝色实线为三步监督，橙色虚线为四步监督，灰色方点虚线为真实状态经固定读出头得到的控制结果；同种子两臂的真实状态概率完全相同。横轴是四个技能边界，纵轴是6根中正确的根数；上、下行分别是种子0、1。左列允许后续恢复正确，右列要求前缀全部正确，故右列不会上升。其实际意义是：少数晚期边界改善没有转化为两个种子一致的完整预测链改善。')
add('## 对照具体改变了什么\n\n两臂共用12条已完成采集的A/B普通四技能链，A与B各6根、每色4根；每条有开屉、抓起、放置、关屉四次真实调用。完整记录包括 `A_intermediate_v3_blue_0` 的真实放置/关屉异常，不按成功或标签筛除。GT只在离线评分时读取，未用于损失、模型输入或选点。')
add(table(['元素','三步监督','四步监督','控制理由'],[
['初始模型/固定组件','同一原始转移；同种子修正读出头','完全相同','编码器、解码器、DINO、归一化、读出阈值均冻结'],
['单步样本','每次64：48旧+12新普通+4新持物关屉','相同样本及顺序','每种子复用既有1000×64调度'],
['多步频率与根','每4次更新一次；48起点','相同','每批12根各出现4次，根顺序相同'],
['递归跨度','[开/抓/放]和[抓/放/关]各2次/根','[开/抓/放/关]各4次/根','四步臂始终从真实初态出发'],
['直接多步损失','局部深度2、3各0.25','深度2、3、4为0.125/0.25/0.125','每个多步批次总系数均0.5'],
['全局技能位置有效权重','抓0.125、放0.25、关0.125','相同','避免将跨度效果与位置权重增加混在一起'],
['目标和优化器','特征误差+0.25原始潜变量MSE','相同','无查询/命题损失；AdamW 1e-4，权重衰减1e-4，梯度裁剪5'],
['预算与种子','种子0/1，每组1000更新','相同','只有转移参数可训练']]))
add('等效位置权重不意味着梯度路径相同：三步臂的“放置”混合了从初态递归3步与从真实开屉后状态递归2步，四步臂全部从初态递归3步；“关屉”新增了深度4的反传路径。这正是本次有限干预。没有在预测序列中插入后续真实潜变量。')
compute=[]
for n in names:
    t=logs[n];compute.append([n,t['transition_state_evaluations'],t['decoder_target_evaluations'],f'{t["elapsed_seconds"]:.3f} s'])
add(table(['运行','训练转移状态评估数','训练解码目标评估数','实测运行计时'],compute))
add('四步臂每多步更新的转移状态评估从144增至192（+33.3%）；计入单步后的总量从100000增至112000（+12%），解码目标从88000增至100000（+13.6%）。这些是训练样本计算计数，未含检查点评估。实测计时覆盖训练循环、检查点C评估与保存，不包含已有DINO/编码特征缓存生成。模型只处理缓存的4×64潜变量，故本轮计时较短；这不是等算力实验，也不是端到端耗时。')
add('## 共同选点与隔离\n\n两臂都在同6条C选点普通链上，从真实初态递归4步，以h2/h3/h4的特征误差按0.25/0.50/0.25加权选最低检查点，同分选较早。候选仅0/250/500/750/1000更新；四组均选择250。没有用C核验或D做选择。旧单步C分数仍按旧37根与新选点6根分别根平均后各占一半，仅保留诊断。')
selrows=[]
for n in names:
    best=next(r for r in logs[n]['log']if choices[n]['path'].endswith(f'update{r["update"]:04d}.pt'));selrows.append([n,best['update'],f'{best["score"]:.6f}',f'{best["original_single_C_score"]:.6f}'])
add(table(['运行','实际选点更新','共同多步C分数 ↓','该点原单步C分数 ↓'],selrows))
add('![C检查点分数](C_selection_curves.png)\n\n图2：蓝/橙分别为三步/四步监督，横轴为更新次数，纵轴为特征误差，越低越好。空心圈是实际采用的250更新点，灰竖虚线仅帮助定位；每条线上其余点是预先保存的候选，不是用C核验重新选出的模型。左列是实际选择标准；右列保留旧单步标准，汇总对象及数值尺度不同，不宜横向比较绝对高低。两种子的四步臂共同多步最优分数均没有超过三步臂。')
add('训练数据文件只包含A/B与C选点，训练器文件访问守卫阻止C核验/D读取；推理器禁止读取私有GT目录。先冻结四个模型及其哈希，随后生成预测，再由独立评分程序读取标签。本轮未读取D3/D4用于推理或评分，也没有D5采集。两臂的500个多步批次（250×2种子）逐根、逐技能位置有效权重均经独立脚本核对。')
add('## 训练集与C的逐步和累计结果\n\n以下均为普通分支，顺序为开屉/抓起/放置/关屉。每个数是该边界9事实全对的根数；真实读出直接编码真实后态，单步预测每次输入真实前态，初态递归只在开始输入真实状态。')
rows=[]
for g in groups:
    for n in names:
        rr=metric(n,g);rows.append([gname[g],n,rr[0]['roots'],vec([r['actual']['full9_correct']for r in rr]),vec([r['one_step']['full9_correct']for r in rr]),vec([r['recursive']['full9_correct']for r in rr]),vec(cum(n,g)['prefix_correct'])])
add(table(['数据','运行','N','真实读出','单步预测','初态递归','递归前缀全对'],rows))
add('末列最后一个数就是整链正确根数。A/B训练种子0的四步臂在抓起边界多错1根，使整链11/12降至10/12；两个种子的放置边界都由11/12升至12/12，但关屉仍11/12。按颜色、读出方式、路径拆分的全部数据见 `step_summary.csv` 与 `cumulative_summary.csv`，逐命题配对概率见 `paired_propositions.csv`。颜色仅作为数据分组，不是上述图中蓝/橙线的含义。')
add('## 错误属于什么，以及最先在哪里发生\n\n下表列出初态递归的组错误根数；Holding组三个事实表示持有哪个物体，In组三个事实表示物体是否严格位于抽屉内并受支撑，drawer组三分类为关/开/中间。只要组内任一事实错，该组该根计一次；不同组可在同一根重叠，不能相加当作总错误数。')
rows=[]
for g in groups:
    for n in names:
        rr=metric(n,g);rows.append([gname[g],n,rr[0]['roots'],*[vec([r['recursive']['groups'][k]['wrong']for r in rr])for k in ['Holding','In','drawer']]])
add(table(['数据','运行','N','Holding错误h1/2/3/4','In错误h1/2/3/4','抽屉错误h1/2/3/4'],rows))
add('逐根比较表明：种子0的 `C_intermediate_v3_pink_3` 在三步臂前三步均对，第四步误报仍Holding(pink)，四步臂修正此末步，所以整链增加1根。种子1同一根的第一个错误从放置后推迟到关屉后，但末步仍误报Holding(pink)，因此没有增加整链正确数。没有据此对该单例另行训练或调阈值。')
add('更早的失误仍存在：`C_intermediate_v3_pink_2` 在抓起后即漏报Holding，种子0的概率从0.279降到0.131，种子1从0.470降到0.272；两种子的真实状态读出约0.98且判对。这是预测潜变量进入固定读出后的错误，不能归因为这两个真实状态读出失败。另一方面，`C_intermediate_v3_red_2` 的放置后真实In=false，而真实状态读出也错报In=true；这一根同时含读出错误，不能把所有错误都归给转移。')
add('作为既定单项替换诊断，下面只将开屉后的预测潜变量替换成真实潜变量，然后继续抓/放/关；其余模型、动作和阈值不变。每串是h2/h3/h4的全9正确根数。此替换既消除了第一步预测误差，也缩短递归深度，不能单独证明是哪一种机制。')
rows=[]
for g in groups:
    for n in names:
        rows.append([gname[g],n,metric(n,g)[0]['roots'],vec([r['recursive']['full9_correct']for r in metric(n,g)][1:]),vec([r['recursive']['full9_correct']for r in metric(n,g,path='window1')])])
add(table(['数据','运行','N','从初态递归：h2/h3/h4','替换为真实开屉后：h2/h3/h4'],rows))
add('C核验种子0替换后两臂末步均4/6，种子1两臂末步均2/6；晚期Holding误报与部分读出/几何错误仍在。此次只改变监督跨度，不能进一步区分预测潜变量拟合不足与固定读出头对该潜变量的敏感性。')
add('## 持物关屉分支单列\n\n此分支不参与正常四步跨度的多步目标，但原单步调度仍含每批4个相应样本。沿用已有真实调用与拒绝结果，不把动作拒绝改成虚构成功。下列是全9递归正确根数，不是控制器拒绝正确率。')
rows=[]
for g in groups:
    for n in names:
        rr=metric(n,g,branch='held_closed');rows.append([gname[g],n,rr[0]['roots'],vec([r['actual']['full9_correct']for r in rr]),vec([r['one_step']['full9_correct']for r in rr]),vec([r['recursive']['full9_correct']for r in rr]),cum(n,g,branch='held_closed')['all4_correct']])
add(table(['数据','运行','N','真实读出h1/2/3/4','单步h1/2/3/4','递归h1/2/3/4','整链'],rows))
add('本次跨度改变没有解决持物关屉分支：A/B与C选点的递归各边界均0，C核验只有零星单边界正确，所有整链均0。应保留这个负结果，不能由普通链少量改善推断拒绝分支可用。')
add('## 证据与复核\n\n实验目录：`'+str(E)+'`；本报告目录：`'+str(R)+'`。模型、日志、冻结调度、协议与评分都留在独立目录，既有数据和模型只读复用。')
add(table(['文件','用途'],[
['configs/protocol.json / training_frozen.json','训练前协议、输入与源码哈希'],['configs/chains.json / schedule_seed*.npz','根、动作连续性与两臂调度'],['configs/choices_frozen.json','评分前冻结的四个250更新模型'],['results/train_*.json','全部候选C分数、冻结组件哈希与实际计时'],['results/predictions.json / metrics.json / cumulative.json','1344预测记录、672聚合组、根级前缀评分'],['first_wrong_boundary.csv / paired_propositions.csv','首错边界与12096条命题配对，保留真实读出控制'],['step_summary.csv / cumulative_summary.csv / checkpoint_scores.csv','含颜色分组的可复算表格'],['results/gate.json','原定门槛及失败后停止动作'],['results/verification.json / configs/delivery_manifest.json','独立核对记录与最终文件SHA256封存']]))
add('独立复核已通过：29个冻结哈希条目、前两阶段103个封存文件未改变；12个AB根及单步顺序一致；500个多步批次根/位置权重一致；初始转移相同，编码器/解码器/查询头未改变；4组共4000更新、256000次单步抽样、48000次多步起点；672组全9与三类错误计数及全部累计前缀重新计算一致。封存后可只读复核：\n\n```bash\nsource /home/robot/research/projects/wm_symbolic/world_model_env.sh\npython /home/robot/research/projects/wm_symbolic/experiments/supervision_span_20261016/source/verify.py\n```')
add('本阶段结束。阻止推进到新D的原因是预定实证门槛未通过，不是环境或传输故障；后续实验方向交由父线程决策。本报告不建议根据已见C单例继续调参。')
(R/'REPORT.md').write_text('\n\n'.join(out)+'\n')
print('REPORT_WRITTEN',len(flat),'table rows',len(first),'first-error rows',flush=True)
