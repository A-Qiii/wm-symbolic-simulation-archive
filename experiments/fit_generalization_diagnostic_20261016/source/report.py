from config import *
import csv,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(FONT);plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
NAMES={f'{a}_seed{s}':f'{n} · 种子{s}'for a,n in [('old_chains','旧链'),('new_chains','新增链')]for s in [0,1]};GROUPS={'old_AB':'旧A/B训练根','new_AB':'新增A/B训练根','C_select':'C选择根','C_validate':'C验证回测根'}
def table(h,rows):return ['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |'for r in rows]
def main():
    ms=read(E/'results/metrics_selected.json');cs=read(E/'results/metrics_checkpoints.json');loss=read(E/'results/frozen_objective_curves.json');protocol=read(E/'configs/protocol.json');names=['old_chains_seed0','old_chains_seed1','new_chains_seed0','new_chains_seed1'];colors={'own':'#2471a3','C_select':'#c58b17','C_validate':'#b03a2e'}
    def own(n):return 'old_AB'if n.startswith('old_')else'new_AB'
    def get(n,g,path='full4',color='all',branch='normal',update=None):return sorted([r for r in(ms if update is None else cs)if r['condition']==n and r['group']==g and r['path']==path and r['color']==color and r['branch']==branch and(update is None or r['checkpoint_update']==update)],key=lambda r:r['global_step'])
    def seq(rr,k):return ' / '.join(f'{r[k]["full9_correct"]}/{r[k]["full9_N"]}'for r in rr)
    fig,axs=plt.subplots(2,2,figsize=(13,9.5))
    for ax,n in zip(axs.flat,names):
        for g,label,col in [(own(n),'本臂实际训练根（12）',colors['own']),('C_select','C选择根（6）',colors['C_select']),('C_validate','C验证回测根（6）',colors['C_validate'])]:
            rr=get(n,g);vals=[100*r['recursive']['full9_correct']/r['recursive']['full9_N']for r in rr];ax.plot(range(1,5),vals,'o-',color=col,label=label,lw=2)
        ax.set(title=NAMES[n],xticks=range(1,5),xticklabels=['开屉','抓取','放置','关屉'],yticks=[0,25,50,75,100],ylim=(-4,116),ylabel='四步整链：该步九命题全对率（%）');ax.grid(alpha=.15);ax.legend(fontsize=9)
    fig.suptitle('训练根总体更好，但不是完全学会后才在 C 根失败',fontsize=16);fig.text(.035,.025,'每个面板是一个正式选中模型；蓝＝本臂参与多步训练的12根，橙＝用于单步特征选点的6个C根，红＝本轮未用于拟合/选点的6个C根。\n横轴为固定动作顺序；每条曲线从同一初态连续递归。C根历史上已见，不能作为新的独立测试。各点分母按根，不把帧数当独立样本。',fontsize=10);fig.tight_layout(rect=[0,.1,1,.95]);fig.savefig(R/'fit_vs_C.png',dpi=170);fig.savefig(R/'fit_vs_C.svg');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,9.5))
    for ax,n in zip(axs.flat,names):
        gg=[own(n),'C_select','C_validate']
        for j,(path,label,col)in enumerate([('full4','从初态连续4步','#64748b'),('window1','从真实开屉后状态连续3步','#218c7f')]):
            rr=[get(n,g,path)[-1]for g in gg];vals=[100*r['recursive']['full9_correct']/r['recursive']['full9_N']for r in rr];bars=ax.bar(np.arange(3)+(j-.5)*.3,vals,.29,color=col,label=label);ax.bar_label(bars,labels=[f'{r["recursive"]["full9_correct"]}/{r["recursive"]["full9_N"]}'for r in rr],padding=3)
        ax.set(title=NAMES[n],xticks=range(3),xticklabels=['本臂训练根','C选择根','C验证回测根'],ylabel='关屉后九命题全对率（%）',ylim=(0,125),yticks=[0,25,50,75,100]);ax.legend(fontsize=8.7);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('只替换开屉后的潜状态：C 验证关屉结果均从 2/6 到 4/6',fontsize=15);fig.text(.035,.022,'灰＝整链初态起算4步；绿＝将开屉后的潜状态替换为真实编码，继续相同抓取/放置/关屉3步。模型、查询头、动作和评分真值相同。\n绿色对应实际训练过的第二种窗口长度；仍未全对。替换同时移除首步潜误差并缩短递归深度，不能据此单独认定“只缺四步监督”。',fontsize=10);fig.tight_layout(rect=[0,.1,1,.95]);fig.savefig(R/'window_vs_full4.png',dpi=170);fig.savefig(R/'window_vs_full4.svg');plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(13,12))
    for seed in [0,1]:
        ax=axs[0,seed]
        for arm,label,col in [('old_chains','旧链','#426d9c'),('new_chains','新增链','#218c7f')]:
            rr=[r for r in loss if r['condition']==f'{arm}_seed{seed}'];ax.plot([r['update']for r in rr],[r['time_average_objective']for r in rr],'o-',color=col,label=label);sel=next(r for r in rr if r['formally_selected']);ax.scatter(sel['update'],sel['time_average_objective'],s=130,facecolors='none',edgecolors=col,lw=2)
        ax.set(title=f'固定训练日程重新计算目标 · 种子{seed}',ylabel='单步 + 1/4频率多步目标');ax.legend(fontsize=9)
        for row,g in [(1,'C_select'),(2,'C_validate')]:
            ax=axs[row,seed]
            for arm,label,col in [('old_chains','旧链','#426d9c'),('new_chains','新增链','#218c7f')]:
                for h,style in [(3,'-'),(4,'--')]:
                    yy=[get(f'{arm}_seed{seed}',g,update=u)[h-1]['recursive']['full9_correct']for u in [0,250,500,750,1000]];ax.plot([0,250,500,750,1000],yy,style,marker='o'if h==3 else's',color=col,label=f'{label}第{h}步')
            ax.set(title=f'{GROUPS[g]} · 四步整链中第3/4步',ylabel='九命题全对根数 / 6',ylim=(-.3,8.2),yticks=range(7));ax.legend(fontsize=8,ncol=2,loc='upper center')
    for ax in axs.flat:ax.set(xticks=[0,250,500,750,1000],xlabel='已有检查点更新次数');ax.grid(alpha=.15)
    fig.suptitle('目标误差持续下降，C 递归语义不随之持续改善',fontsize=16);fig.text(.035,.018,'上排是固定权重重评同一完整训练日程的目标，非训练日志中交替的当批loss；不同来源的绝对目标值不作优劣比较。空心圈标正式选点。\n下两排只作诊断：实线＝放置后，虚线＝关屉后；蓝/绿＝旧链/新增链。20个检查点均为已有文件，没有重新选择模型或新增训练。',fontsize=10);fig.tight_layout(rect=[0,.075,1,.96]);fig.savefig(R/'checkpoint_diagnostic_curves.png',dpi=170);fig.savefig(R/'checkpoint_diagnostic_curves.svg');plt.close(fig)
    # Loss coverage counts include the effective selected checkpoint, not only the completed budget.
    coverage=[]
    for n in names:
        seed=int(n[-1]);arm=n.rsplit('_seed',1)[0];selected=int(Path(protocol['models']['selected'][n]['path']).stem.replace('update',''));sched=np.load(STUDY/'configs'/f'schedule_seed{seed}.npz')[arm]
        for count,label in [(selected,'selected_checkpoint'),(1000,'completed_budget')]:
            active=sum((sched[u]>=0).all()for u in range(count));coverage.append({'condition':n,'scope':label,'updates':count,'multi_updates':active,'per_root_global2_direct_targets':active*2,'per_root_global3_direct_targets':active*4,'per_root_global4_direct_targets':active*2,'global4_depth4_targets':0})
    with(R/'loss_coverage.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
    curveflat=[]
    for r in cs:curveflat.append({k:r[k]for k in ['condition','checkpoint_update','group','path','global_step','local_depth']}|{k+'_'+f:r[k][f]for k in ['actual','one_step','recursive']for f in ['full9_correct','full9_N','feature_mean','raw_MSE']})
    with(R/'checkpoint_semantics.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(curveflat[0]));w.writeheader();w.writerows(curveflat)
    sup=['# 全分组数值','','所有分母为完整有效九命题的根数。H/I/屉错误数是组三项至少一项错，可以重叠。每根真实后继、一跳和递归配对；三技能窗口0等于整链前3步，窗口1从实际开屉后状态起算。完整颜色分组与概率边界见CSV。','']
    for n in names:
        sup+=['## '+NAMES[n],''];rows=[]
        for group in ['old_AB','new_AB','C_select','C_validate']:
            for branch in ['normal','held_closed']:
                for path in ['full4','window0','window1']:
                    for r in get(n,group,path,branch=branch):
                        rows.append([GROUPS[group],branch,path,f'{r["global_step"]}/{r["local_depth"]}',*[(f'{r[k]["full9_correct"]}/{r[k]["full9_N"]}')for k in ['actual','one_step','recursive']],'/'.join(str(r['recursive']['groups'][g]['wrong'])for g in ['Holding','In','drawer']),r['recursive_paired']['actual_right_prediction_wrong'],r['recursive_paired']['both_wrong'],f'{r["recursive"]["feature_mean"]:.4f}',f'{r["recursive"]["raw_MSE"]:.4f}'])
        sup+=table(['根组','分支','路径','全局步/递归深度','真读','单步','递归','递归H/I/屉错','真读对递归错','两者均错','递归特征误差','递归潜MSE'],rows)+['']
    (R/'SUPPLEMENT.md').write_text('\n'.join(sup)+'\n')
    lines=['# 冻结模型：训练拟合、泛化与监督长度诊断','','完成时间：'+now(),'','当前证据同时显示训练窗口内残余错误、C根泛化差距，以及潜误差向后传播；不能归为单一原因。最关键的排除项是：并非训练链完全正确、只在未训练根失败，也不能把所有关屉错误都归给没有四步监督。','','本轮仅推理4个正式模型与20个已有检查点，新增训练0、仿真0、数据采集0。正式检查点、查询头与阈值全部保持不变；C选择/验证根均已在历史实验中见过，检查点曲线只用于诊断。','','## 先排实现、帧对齐和动作误用','','| 核查 | 证据与实际结果 |','|---|---|','| 帧与链 | 每条链同根同分支，上一after等于下一before；旧记录public末帧机器人状态与private相同，float32副本按相同dtype精确匹配 |','| 目标颜色 | A/B/C的scene_obs对象顺序不同；离线真值按各原生配置解析并逐物体核对位置。模型使用固定红/蓝/粉视觉槽，不把scene_obs输入模型 |','| 动作 | 每步核验技能one-hot、仅lift含目标颜色、place不带对象参数、posthold=96；训练/诊断一致 |','| 潜缓存 | 冻结编码器重算与缓存最大绝对差2.87×10⁻⁶；未发现错帧或表示版本错配 |','| 推理 | 真实前态单步每次重置到真实z；整链递归不纠正；窗口1仅在开屉后用真实z重新起算 |','| 原结果复现 | 当前C验证逐命题与上一阶段正式结果一致；模型/源文件哈希未变 |','','旧链没有现成的当前9标签，本轮从已有完整XYZ/欧拉角、接触、Holding和drawer q恢复同一几何定义。离线URDF抽屉位姿与1100份保存的完整快照核对，最大位置差8.92微米，标签分类全部一致。旧接触记录过滤了≤0.01N的接触，因此对“非持有、非明确在外、无确定支撑路径”保留未知，并对10微米几何边界留未知。60个实际旧边界恰好全部可确定；没有用旧版In或补假0。','','当前编码器只有当前静态相机帧、腕相机帧和15维robot_obs；名为`robot_history`的字段仅为当前向量。没有观测历史编码，也没有递归后验；转移用自身预测潜变量继续推进。诊断不更改这一结构。','','## 同一训练链是否能学会','','下表只列各模型真正参与多步训练的本臂12根；每格为全九项正确/有效根数。四步顺序为开屉、抓取、放置、关屉。','']
    lines+=table(['模型','真实后继读出（四步）','真实前态单步（四步）','从初态递归（四步）','训练窗口1递归（抓取/放置/关屉）'],[[NAMES[n],seq(get(n,own(n)),'actual'),seq(get(n,own(n)),'one_step'),seq(get(n,own(n)),'recursive'),seq(get(n,own(n),'window1'),'recursive')]for n in names])
    lines+=['','旧链种子1在实际训练过的窗口1中仍有1条放置后的Holding误报；新增链种子1同一窗口有2条Holding误报和1条In错误。因此已监督的三技能窗口本身也未完全拟合。所有这些训练边界的真实后继读出都全对，不能归咎于查询头读不懂真实训练状态；但仍不能只凭此区分转移潜变量偏移与查询头对预测潜变量的适应性。','','旧12根第三步都已松手且入屉，第四步均关屉。新增12根第三步都已松手，其中11根严格入屉；`A_intermediate_v3_blue_0`未满足严格入屉，后续关屉实际失败，抽屉停在Intermediate。两种子都把该失败结果预测成入屉/关闭。这是训练数据内实际异常结果未拟合，原记录保留，不能把它当成功样本评分。','','## 泛化差距与逐命题证据','','![训练与C根](fit_vs_C.png)','','| 模型 | 本臂训练放置/关屉递归 | C选择放置/关屉递归 | C验证放置/关屉递归 |','|---|---|---|---|']
    for n in names:lines.append('| '+NAMES[n]+' | '+' | '.join(seq(get(n,g)[2:],'recursive')for g in [own(n),'C_select','C_validate'])+' |')
    lines+=['','训练根整体明显更好，C验证第四步所有模型都只有2/6。但这是不同根分布之间的描述性差距，C选择根又参与了选点；不能将其单独当成严格因果的过拟合结论。完整逐色、逐步九项与Holding/In/抽屉分组见[颜色与技能位置CSV](summary_by_color_and_position.csv)，每色训练4根、C各2根；数据量有限。','','下面给出训练窗口内的同根概率配对。二值阈值保持0.5，真实Holding均为0；“真实p”是编码实际后继后查询，“预测p”是窗口1递归后查询：','','| 模型与根 | 命题 | 真实p | 预测p | 预测特征误差 | 含义 |','|---|---|---:|---:|---:|---|','| 旧链1 · A蓝1 | Holding(蓝) | 0.0101 | 0.7595 | 1.1403 | 训练窗口内仍高概率误报持有 |','| 新增链1 · B红1 | Holding(红) | 0.0202 | 0.6465 | 0.6258 | 已松手，但预测超过边界0.1465 |','| 新增链1 · B蓝1 | Holding(蓝) | 0.0090 | 0.5361 | 0.4382 | 误差低于该组中位0.6069，仍跨语义边界 |','| 新增链0 · A蓝0 | In(蓝,屉) | 0.0062 | 0.9987 | 1.4724 | 真实未严格入屉，预测高度确信入屉 |','','所以既有靠近概率边界的错误，也有高度确信的错误；不能统一解释为阈值微小抖动。特征误差较低也不保证语义正确，0.4382仅为本组相对较低，未声明绝对“足够小”。[21600行逐命题概率表](paired_propositions_probabilities.csv)保留真值、mask、每条路径概率、带正确方向的边界距离、原空间误差。抽屉距离为真实类别概率减最大其他类别概率，不错误套用0.5二值阈值。','','C验证红2实际未严格入屉且关屉失败，真实In读出也会错；该项与“真实读对、预测错”分开列出。新增链1第三步另有红3与粉3的Holding误报，真实p分别0.0047/0.0008，递归p为0.5815/0.5931，说明C上还存在真实读出正确而动态路径出错的根。','','## 三技能监督不等于四技能监督','','窗口0：开屉→抓取→放置，从根初态开始；窗口1：抓取→放置→关屉，从实际开屉后状态开始。两窗口都只在局部第2、3步产生多步损失。因而关屉曾在局部深度3得到监督，却从未在根初态起算的递归深度4得到监督。','','![窗口与整链](window_vs_full4.png)','','将开屉后的潜变量替换为真实编码后，C验证关屉正确数四模型均从2/6升至4/6，仍未全对。新增链1自己的训练窗口1放置仅9/12，反而低于从初态递归的11/12；潜误差可能被后续抵消，不能假设越短总是越好。该替换支持前段潜状态影响后续的判断，但同时缩短了递归深度，未独立证明缺四步监督就是全部原因。','','## 实际损失覆盖与曲线','','设F为四组等权、变化加权的解码特征误差，Z为原潜变量MSE。每次更新都有单步`F+0.25Z`；每4次更新才加入局部深度2与3，每项均为`0.25×(F+0.25Z)`。没有符号查询损失。','','| 全局技能位置 | 哪个窗口直接计损失 | 多步更新内F系数（窗口均衡平均后） | 按每4次一次折算F系数 |','|---|---|---:|---:|','| 1 开屉 | 无直接多步项，但梯度经后续步骤传回 | 0 | 0 |','| 2 抓取 | 窗口0局部2 | 0.125 | 0.03125 |','| 3 放置 | 窗口0局部3、窗口1局部2 | 0.25 | 0.0625 |','| 4 关屉 | 窗口1局部3 | 0.125 | 0.03125 |','','上表Z系数再乘0.25；折算不含另行共同单步项，也不是梯度大小。完整1000更新中，每根抓取/放置/关屉直接多步目标次数为500/1000/500。正式选中500更新模型实际到达250/500/250；选中750更新模型到达374/748/374（187次多步更新），不能用完成的1000更新冒充选中权重覆盖。详情见[loss覆盖表](loss_coverage.csv)。','','![目标和C语义曲线](checkpoint_diagnostic_curves.png)','','原日志只在250倍数保存当批总loss：250/750没有多步项，500/1000有多步项，且batch不同；不能据交替高低宣称过拟合。这里对每个固定检查点重新计算相同日程的频率加权单步均值和均衡窗口多步均值，目标随更新下降。新增链种子0由0.6621降至0.5508，种子1由0.6623降至0.5493，但C验证第四步全程维持2/6。旧链种子0的目标0.5989→0.5302也没有提高C验证第四步。','','单步特征选点与语义曲线并非同一目标。例如旧链种子1正式选750更新，C验证第三步为2/6，而500更新诊断为3/6；新增链种子0的C选择窗口1在750更新达6/6，正式500更新为5/6。它们仅说明指标可能不一致，不授权重新挑选模型。未发现一个被漏选的检查点能解决C验证第四步；[全部20检查点数值](checkpoint_semantics.csv)保留三步及四步结果。','','## 本次能够支持的判断','','| 判断 | 支持证据 | 仍不能下的结论 |','|---|---|---|','| 已监督训练窗口也有残余语义错误 | 自身AB窗口1仍错，真实后继读出正确，存在明确概率翻转 | 尚未分离潜转移拟合不足与查询头对预测潜空间的敏感性 |','| 有训练到C的明显性能差距 | 训练关屉11–12/12，C验证2/6；逐根、颜色和真假分组可追溯 | 不能只用不同分布正确率差认定唯一泛化机制 |','| 初段潜误差会影响关屉 | 只换开屉后的真实潜状态，C验证由2/6到4/6 | 不能把剩余错误全部归为未监督第四步 |','| 当前损失降低未确保符号一致 | 固定目标曲线下降；低于组中位误差的样本仍跨Holding边界 | 不等于已证明应改哪个损失、加多少权重或训练更久 |','','下一项单变量实验应在这些未分开的因素中选择一个：训练窗口内的潜转移—符号接口一致性，或保持其他条件不变的监督深度差异。现有证据不足以把“再补相似根”作为优先解释；本轮没有开始任何新训练或修正。','','## 交付与限制','',f'- [冻结协议]({E}/configs/protocol.json)；[对齐与离线真值核验]({E}/results/source_alignment.json)；[推理审计]({E}/results/inference_audit.json)。','- [完整分组表](SUPPLEMENT.md)；[逐根诊断](per_root_diagnostics.csv)；[逐命题概率](paired_propositions_probabilities.csv)；[颜色与位置汇总](summary_by_color_and_position.csv)。','- 拒绝支路均单列在完整表和CSV中，未混入正常链或增加独立根数。',f'- [独立复核]({E}/results/verification.json)与最终`configs/delivery_manifest.json`覆盖源码、配置、推理结果、图表和报告；历史实验保持封存。','','这是既有PhysicalOracle条件下的冻结诊断，不是原HULC/TS-LTL总体方案的新验收。未启动训练或仿真，后续实验由父任务决定。']
    idx=next(i for i,x in enumerate(lines)if x.startswith('设F为'))
    lines[idx+1:idx+1]=['','F的四组为物体视觉256维、物体几何/可见性12维、全局视觉1024维、当前机器人状态15维；各组先求均值再等权平均，逐维乘`1+4×min(|真实后继−真实前态|,1)`。Z在冻结的4×64潜变量原坐标内计算，尚未施加查询头归一化；它不是像素误差或米制位姿误差。真实特征只参与评分，不输入递归转移。','','即使使用实际后继潜变量，冻结解码器也有重建误差。例如上表B蓝1实际潜变量的F为0.3235，递归潜变量F为0.4382、Z为0.1335。因此不能把整个特征误差都归给转移，也不能把总特征误差较低当作Holding必然正确。','']
    (R/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT_READY',str(R),flush=True)
if __name__=='__main__':main()
