from config import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fontManager.addfont(FONT)
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
NAMES={'original_head':'原冻结头'}|{f'{a}_seed{s}':f'{label}{s}' for a,label in [('starting','修正起点'),('control','等预算对照'),('empty_open','空手开屉修正')] for s in [0,1]}
STRATA=[('pre_grasp','空手开屉\n抓取前'),('held_open','抓取后\n持物开屉'),('hover_above_drawer','抽屉上方\n持物悬停'),('before_release','松手前'),('post_place','放置后\n稳定等待'),('held_closed','持物关屉')]
METHODS=[('actual','真实后继读出','#2471a3'),('one_step','真实前态单步','#d68910'),('recursive','仅初态递归','#b03a2e')]

def seq(m,name,method,branch='normal'):
    return ' / '.join(str(r[method]['full9_correct']) for r in m['trajectory_prediction'][name][branch])

def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]

def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    d=read(E/'results/metrics_D4.json');c=read(E/'results/metrics_C_backtest.json');cov=read(E/'results/coverage_D4.json')
    before=read(E/'configs/before_D4_frozen.json')
    fig,axs=plt.subplots(1,2,figsize=(15,5.9),sharey=True)
    for seed,ax in enumerate(axs):
        for j,(name,label,color) in enumerate([('original_head','原冻结头','#b8bec8'),(f'control_seed{seed}','起点＝等预算对照','#64748b'),(f'empty_open_seed{seed}','补入空手开屉','#218c7f')]):
            vals=[d['true_state_readout'][name]['strata'][s]['full9_correct'] for s,_ in STRATA]
            bars=ax.bar(np.arange(6)+(j-1)*.25,vals,.24,color=color,label=label);ax.bar_label(bars,fontsize=10,padding=2)
        ax.set(xticks=np.arange(6),xticklabels=[t for _,t in STRATA],yticks=[0,3,6,9,12],ylim=(0,14.8),title=f'查询头种子{seed}',ylabel='九命题全部正确的真实状态数 / 12')
        ax.legend(loc='upper center',ncol=3,fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('同一批 D4 根：目标空手开屉状态恢复，其他关键状态大体保留',fontsize=16)
    fig.text(.035,.025,'横轴＝真实执行中的六类关键时刻，每类每根1帧，12根各含红/蓝/粉目标4根。纵轴要求全部9命题同时正确。\n浅灰＝原冻结头；深灰＝修正起点和C选择后的等预算对照（两者相同）；绿色＝修正头。比较使用相同观测和标签；本图不评价未来预测。',fontsize=10)
    fig.tight_layout(rect=[0,.12,1,.94]);fig.savefig(R/'D4_readout.png',dpi=170);fig.savefig(R/'D4_readout.svg');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13.5,10.5));h=np.arange(1,5)
    for seed in [0,1]:
        ax=axs[0,seed];rr=d['trajectory_prediction'][f'empty_open_seed{seed}']['normal']
        for j,(method,label,color) in enumerate(METHODS):
            bars=ax.bar(h+(j-1)*.25,[r[method]['full9_correct'] for r in rr],.24,label=label,color=color);ax.bar_label(bars,fontsize=10,padding=2)
        ax.set(title=f'修正后的三个评测路径 · 种子{seed}',ylabel='九命题全对根数 / 同12根');ax.legend(loc='upper center',ncol=3,fontsize=8.7)
        ax=axs[1,seed]
        for j,(name,label,color) in enumerate([(f'control_seed{seed}','修正前＝等预算对照','#64748b'),(f'empty_open_seed{seed}','修正后','#218c7f')]):
            rr=d['trajectory_prediction'][name]['normal'];bars=ax.bar(h+(j-.5)*.32,[r['recursive']['full9_correct'] for r in rr],.31,label=label,color=color);ax.bar_label(bars,fontsize=11,padding=2)
        ax.set(title=f'只比较递归预测 · 种子{seed}',ylabel='九命题全对根数 / 同12根');ax.legend(loc='upper center',ncol=2,fontsize=10)
    for ax in axs.flat:
        ax.set(xticks=h,xticklabels=['1 开屉','2 抓取','3 放置','4 关屉'],yticks=[0,3,6,9,12],ylim=(0,15.2));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('正常四技能链：首步恢复，放置后的递归预测仍失败',fontsize=16)
    fig.text(.035,.025,'上排蓝＝编码实际后继再读出；橙＝每步从真实前态预测一次；红＝仅输入初态，连续预测、不纠正。\n下排灰/绿＝修正前后对同一12根的递归预测。所有条件共用同一编码器和转移，只有查询头不同。\n每柱只统计该步的9命题全对数；不是整条任务成功率。第三步两种子均有9根把已松手对象判为持有。',fontsize=10)
    fig.tight_layout(rect=[0,.12,1,.95]);fig.savefig(R/'D4_normal_prediction.png',dpi=170);fig.savefig(R/'D4_normal_prediction.svg');plt.close(fig)

    # Full numerical supplement: retain all references, groups, masks and paired errors.
    sup=['# 完整数值与指标定义','','九命题为三色Holding、三色In(drawer)、抽屉Closed/Open/Intermediate。Holding表示实际持有；In要求完整定向包围盒在抽屉内、有抽屉支撑且非持有。抽屉三类互斥。二值阈值0.5，抽屉argmax，均未更改。','',
         '真实读出＝编码实际观测后的查询结果；单步＝每次以真实前态潜变量输入转移；递归＝只编码链初态，后续不以真实帧纠正。全9正确要求9个真值均有效且全部相符。H/I/屉错组为该组三项至少一项错误的状态数；错组可重叠。原MSE为同一固定256维潜变量空间的逐维平方误差，不代表语义正确率。', '',
         'C_backtest是之前已看过的开发验证根回测；D4为模型封存后新采集的12根首次测试，评测后现已见。两条分支共用根，不能合并称24个独立样本。相邻状态也不是独立样本。','']
    csvrows=[];readoutrows=[]
    for phase,m in [('C_backtest',c),('D4',d)]:
        sup+=['## '+phase+' · 真实状态读出','']
        rows=[]
        for name,x in m['true_state_readout'].items():
            for scope,metric in [('all_states',x['all_states'])]+list(x['strata'].items()):
                r={'phase':phase,'condition':name,'scope':scope,'states':metric['N'],'roots':metric['roots'],'full9_N':metric['full9_N'],'full9_correct':metric['full9_correct'],'full9_root_macro':metric['full9_root_macro'],'valid_facts':metric['valid_facts'],'correct_valid_facts':metric['correct_valid_facts']}
                for g in ['Holding','In','drawer']:
                    for k in ['wrong_states','FP_facts','FN_facts']:r[g+'_'+k]=metric['groups'][g][k]
                readoutrows.append(r);rows.append([NAMES[name],scope,f'{metric["full9_correct"]}/{metric["full9_N"]}',f'{metric["full9_root_macro"]:.2%}','/'.join(str(metric['groups'][g]['wrong_states'])for g in ['Holding','In','drawer'])])
        sup+=table(['条件','状态组','全9正确/有效','根平均','H/I/屉错组'],rows)+['']
        for branch,title in [('normal','正常链'),('held_closed','持物关屉支路')]:
            sup+=['## '+phase+' · '+title,''];rows=[]
            for name,branches in m['trajectory_prediction'].items():
                for r in branches[branch]:
                    nums=[r[k]['full9_correct']for k,_,_ in METHODS];a=r['one_step_paired']['states'];b=r['recursive_paired']['states']
                    rows.append([NAMES[name],r['horizon'],f'{r["observed_roots"]}/{r["planned_roots"]}','/'.join(map(str,nums)),*['/'.join(str(r[k]['groups'][g]['wrong_states'])for g in ['Holding','In','drawer'])for k,_,_ in METHODS],a['actual_right_prediction_wrong'],a['both_wrong'],b['actual_right_prediction_wrong'],b['both_wrong'],f'{r["one_step_raw_MSE"]:.5f}/{r["recursive_raw_MSE"]:.5f}'])
                    csvrows.append({'phase':phase,'branch':branch,'condition':name,'horizon':r['horizon'],'observed_roots':r['observed_roots'],'planned_roots':r['planned_roots'],'actual_all9':nums[0],'one_step_all9':nums[1],'recursive_all9':nums[2],'one_step_actual_right_prediction_wrong':a['actual_right_prediction_wrong'],'one_step_both_wrong':a['both_wrong'],'recursive_actual_right_prediction_wrong':b['actual_right_prediction_wrong'],'recursive_both_wrong':b['both_wrong'],'one_step_raw_MSE':r['one_step_raw_MSE'],'recursive_raw_MSE':r['recursive_raw_MSE']})
            sup+=table(['条件','步','观测/计划根','真读/一步/递归全对','真读H/I/屉错','一步H/I/屉错','递归H/I/屉错','真读对一步错','二者均错','真读对递归错','二者均错','一步/递归原MSE'],rows)+['']
    sup+=['## 模型选择和旧状态保留','']+table(['条件','完成更新','C选中更新','C选择损失','旧C完整标签全对/110'],[[NAMES[name],600,int(Path(v['path']).stem.replace('update','')),f'{v["C_score"]:.8f}',c['old_C_retention'][name]['full9_correct']]for name,v in before['selected_heads'].items()])+['','旧C保留集共116状态、6根，其中110状态9标签完整；其余6个部分标签状态只按有效命题评分，不填补未知。选模型只用旧/新C选择根，不使用该保留集。全量掩码和分组值在实验results/metrics_C_backtest.json。','']
    (R/'SUPPLEMENT.md').write_text('\n'.join(sup)+'\n');write_csv(R/'prediction_summary.csv',csvrows);write_csv(R/'readout_summary.csv',readoutrows)
    lines=['# 空手开屉覆盖修正：独立预测测试','','完成时间：'+now(),'','本轮证明一个具体结论：把已有A/B数据中的12个空手开屉状态加入查询训练后，两个种子在新D4根上的首步预测都从0/12恢复到12/12。编码器和转移完全不变，因此支持此前首步错误确有查询覆盖成因。这里验证的是“覆盖加上相应抽样调整”的整体处理，不能独立归因于样本数量。','','这项修正没有解决多步预测：正常链第三步仍只有3/12、第四步4/12；持物关屉支路递归第2至4步仍全部0/12。存在小幅其他状态退化，故不扩大环境复杂度。本轮已停止训练和采集，后续方向交回独立审查。','','## 对照怎样保持可解释','','| 元素 | 实际配置 | 原因 |','|---|---|---|','| 唯一训练处理 | 对照144状态；修正156状态，新增已有12个A/B根各1个空手开屉帧 | 检验上一轮漏掉的明确状态条件 |','| 每批配对 | 32样本中29个完全相同；只把3个旧状态抽样改为新增状态 | 限制更新差异，不同时改变结构或损失 |','| 初始化与预算 | 每种子两臂起点相同；2种子×2臂，各600更新、19200抽样 | 两臂都有同等继续训练机会 |','| 保持不变 | DINO/编码器、原canonical_matched转移、查询结构和归一化、损失权重、阈值、物理控制器 | 隔离查询覆盖修正 |','| 选择规则 | 原规则：旧/新C选择根各占一半损失；候选0/150/300/450/600，早者破同分 | 模型选择不访问D4 |','| 实际选中 | 两个对照均0更新；两个修正均150更新 | 全部600更新都已完成，不能说选中模型训练了600步 |','| 独立测试 | 同环境D的12个新初态根，三色各4；先封存模型再采集 | D4是新初态批次，不是新增环境 |','','C回测的6个根此前已看过，只作开发回测；没有再称独立验证。D3未用于本轮训练或模型选择。D4已经完成首次测试，现在也是已见批次，不可再用于“未见测试”的声明。','','本轮转移始终使用原`canonical_matched_seed0/update1000.pt`，未使用上一轮新增转移训练模型。这样首步差异由查询头对照产生；不能把D3和D4不同根的百分比差直接当配对收益。','','## 新D4数据与真实状态读出','',f'本机真实物理仿真共12根、24条分支、96次调用：成功{cov["status_counts"].get("success",0)}、拒绝{cov["status_counts"].get("refused",0)}、失败0、未达0；没有补换根或外部重试。保存557份观测，9项真值全部可判定，包含完整四元数、支撑/接触和双相机。三色每色4根均达到严格入屉条件；这不抹去上一轮保留的物理失败。采集约176秒，原始记录141.64 MB，特征与模型另计。','','每根两次原生复位：正常分支为开屉→抓取目标→放置→关屉；持物关屉分支为抓取→放置→开屉→关屉，后三次为拒绝。它们共享初始配置，仅计1个独立根。每次技能最多900控制步、96步稳定等待、内部最多2次尝试。复位不算动作转移。','','| 读出条件 | D4全9正确/557 | D4根平均 | 旧C保留全9正确/110 | 已见C回测全9正确/273 |','|---|---:|---:|---:|---:|']
    for name in ['original_head','control_seed0','empty_open_seed0','control_seed1','empty_open_seed1']:
        dm=d['true_state_readout'][name]['all_states'];lines.append(f'| {NAMES[name]} | {dm["full9_correct"]} | {dm["full9_root_macro"]:.2%} | {c["old_C_retention"][name]["full9_correct"]} | {c["true_state_readout"][name]["all_states"]["full9_correct"]} |')
    lines+=['','557/273为相关观测帧，独立根数12/6；根平均先在根内求正确率，再对各根等权平均。对照的选中权重与修正起点相同，故省略重复行。D4种子0有19帧错→对、5帧对→错；种子1有31帧错→对、7帧对→错。旧C种子0仍从93降到90，不能只报净收益。','','![真实状态关键时刻](D4_readout.png)','','空手开屉关键帧分别从7/12、4/12恢复到12/12；悬停保持12/12、11/12；放置后保持12/12、11/12；持物关屉为11/12、10/12→11/12、11/12。所有差异均在同一D4观测上配对比较。','','## 正常链预测：首步恢复后，首个主要错误在放置','','每格依次为技能1/2/3/4的九命题全对根数，每步分母12。它不是整条任务成功率。真实读出作为单独诊断路径，不会在递归中纠正潜变量。','']
    lines+=table(['条件','真实后继读出','真实前态单步预测','仅初态递归预测'],[[NAMES[n],seq(d,n,'actual'),seq(d,n,'one_step'),seq(d,n,'recursive')]for n in ['original_head','control_seed0','empty_open_seed0','control_seed1','empty_open_seed1']])
    lines+=['','![正常四技能链预测](D4_normal_prediction.png)','','修正后两种子递归前两步均12/12；第三步只有3/12，9条错误都属于Holding假阳性：已放下仍预测持有。相同第三步的真实后继读出为12/12和11/12，单步为11/12和9/12；逐根配对均有9条“真实读出对、递归错”，而非从总体百分比相减得到。','','第四步两种子均4/12，8条仍为Holding错误；种子0还比对照的5/12少1条。修正后仅3/12根在四个递归边界都全9正确，这仅是固定动作序列的预测一致性，不是规划或控制任务成功率。','','固定转移的原潜空间MSE没有因查询头变化而变化。第二步语义全对仍不表示潜状态精确；第二步单步/递归MSE约0.2152/0.2575，第三步0.2265/0.3030。它们支持潜状态失配的观察，但不能仅凭MSE宣称已确定第三步的因果机制。','','## 拒绝支路单列，不用它稀释正常链错误','']
    lines+=table(['条件','真实后继读出','单步预测','递归预测'],[[NAMES[n],seq(d,n,'actual','held_closed'),seq(d,n,'one_step','held_closed'),seq(d,n,'recursive','held_closed')]for n in ['control_seed0','empty_open_seed0','control_seed1','empty_open_seed1']])
    lines+=['','每步分母同为12。第1步实际抓取，后3次调用被控制器拒绝，但仍推进并保存96步真实稳定过程。两种子第2至4步递归都0/12，拒绝后的状态保持问题仍未解决。','','## 与上一轮D3的关系和下一步边界','','[D3完整补数对照报告](../intermediate_data_prediction_20261016/REPORT.md)已封存：新查询头读出339/549、320/549，原头266/549；悬停由5/12到10/12，持物关屉由0/12到8/12，但空手开屉由12/12降至9/12、2/12。新增转移的正常链第三步递归7/12、4/12与等预算旧转移训练相同，不能归功于新增转移数据。','','本轮只完成预先批准的一次空手开屉覆盖修正，没有连续变更多个变量。独立D4支持目标状态和首步预测的改善，但不足以通过“多步稳定改善且无明显其他状态退化”的复杂度扩展条件。没有启动新环境、新模型或另一轮修补。后续审查可检验放置后的递归潜状态问题；上一轮新增转移只参与单步辅助损失是一个尚未验证的假设，不能当成原因结论。','','本轮沿用先前批准的PhysicalOracle物理控制器。它是当前有限预测研究的执行条件，不能据此声称原附件所要求的官方HULC完整实验、TS-LTL规划或总体项目验收已经完成。','','## 可复核交付','','- [完整统计与指标含义](SUPPLEMENT.md)；[逐步摘要](prediction_summary.csv)；[真实读出分组](readout_summary.csv)。','- [D4逐命题配对6048行](paired_propositions_D4.csv)；[已见C回测逐命题配对3024行](paired_propositions_C_backtest.csv)。',f'- 实验目录：`{E}`；冻结预算和条件在`configs/protocol.json`，测试前模型清单在`configs/before_D4_frozen.json`。',f'- [复核结果]({E}/results/verification.json)：700项冻结哈希、先前封存3969文件、557份完整标签、33858控制帧、24段视频和配对计数复核通过；编码器与转移未变。',f'- [最终交付清单]({E}/configs/delivery_manifest.json)记录本轮源码、配置、原始数据、模型、结果、图表、报告和日志的SHA256。','','本轮无运行中的训练或采集任务。剩余问题是预测质量未达扩展条件，后续工作范围由父任务审查决定。']
    (R/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT_READY',str(R),flush=True)

if __name__=='__main__':main()
