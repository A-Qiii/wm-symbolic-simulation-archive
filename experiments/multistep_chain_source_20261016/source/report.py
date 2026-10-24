from config import *
import csv,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(FONT)
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
NAMES={f'{a}_seed{s}':f'{n}{s}'for a,n in [('initial','固定起点'),('old_chains','旧链对照'),('new_chains','新增链')]for s in [0,1]}
METHODS=[('actual','真实后继读出','#2471a3'),('one_step','真实前态单步','#d68910'),('recursive','仅初态递归','#b03a2e')]
def table(headers,rows):return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |'for row in rows]
def seq(m,n,k,b='normal'):return ' / '.join(str(r[k]['full9_correct'])for r in m['trajectory_prediction'][n][b])
def write_csv(path,rows):
    with path.open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def errors(r,k):
    pred=discrete(r[k+'_probability']);return ', '.join(f+('误报'if pred[j]else'漏报')for j,f in enumerate(FACTS)if r['valid'][j]and pred[j]!=r['truth'][j])or'全对'

def main():
    m=read(E/'results/metrics_C_backtest.json');f=read(E/'configs/choices_frozen.json');gate=read(E/'results/gate.json');scores=read(E/'results/scored_C_backtest.json');roots=read(E/'results/root_trajectories_C_backtest.json');assert not gate['passed']
    fig,axs=plt.subplots(2,2,figsize=(13.5,10.5));h=np.arange(1,5)
    for seed in [0,1]:
        ax=axs[0,seed]
        for j,(arm,label,color)in enumerate([('initial','固定起点','#b8bec8'),('old_chains','旧链对照','#426d9c'),('new_chains','新增链','#218c7f')]):
            vals=[r['recursive']['full9_correct']for r in m['trajectory_prediction'][f'{arm}_seed{seed}']['normal']];bars=ax.bar(h+(j-1)*.25,vals,.24,color=color,label=label);ax.bar_label(bars,padding=2,fontsize=10)
        ax.set(title=f'只改变多步链来源 · 种子{seed}',ylabel='递归九命题全对根数 / 6');ax.legend(loc='upper center',ncol=3,fontsize=9)
        ax=axs[1,seed]
        for j,(k,label,color)in enumerate(METHODS):
            vals=[r[k]['full9_correct']for r in m['trajectory_prediction'][f'new_chains_seed{seed}']['normal']];bars=ax.bar(h+(j-1)*.25,vals,.24,color=color,label=label);ax.bar_label(bars,padding=2,fontsize=10)
        ax.set(title=f'新增链模型的三条诊断路径 · 种子{seed}',ylabel='九命题全对根数 / 同6根');ax.legend(loc='upper center',ncol=3,fontsize=8.7)
    for ax in axs.flat:
        ax.set(xticks=h,xticklabels=['1 开屉','2 抓取','3 放置','4 关屉'],yticks=range(7),ylim=(0,7.5));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('隔离 C 开发回测：新增链未显示跨种子的明确后段收益',fontsize=16)
    fig.text(.035,.025,'上排浅灰＝固定初始转移；蓝＝旧A/B连续链训练；绿＝新增A/B连续链训练，三者共用同种子的固定查询头。\n下排蓝＝读取实际后继；橙＝每步从真实前态预测；红＝只输入初态后连续预测。柱高为该步9命题全对根数。\n同6个C根（每色2根），两支路共用根；此前已见，本轮未用于训练或选模型。未通过预定门槛，因此没有采新D测试。',fontsize=10)
    fig.tight_layout(rect=[0,.12,1,.95]);fig.savefig(R/'C_multistep_comparison.png',dpi=170);fig.savefig(R/'C_multistep_comparison.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.9),sharey=True)
    for seed,ax in enumerate(axs):
        for arm,label,color in [('old_chains','旧链对照','#426d9c'),('new_chains','新增链','#218c7f')]:
            t=read(E/'results'/f'train_{arm}_seed{seed}.json');xx=[r['update']for r in t['log']];yy=[r['score']for r in t['log']];ax.plot(xx,yy,'o-',color=color,label=label);sel=min(t['log'],key=lambda x:x['score']);ax.scatter([sel['update']],[sel['score']],s=170,facecolors='none',edgecolors=color,lw=2)
        ax.set(title=f'种子{seed}',xlabel='已完成的更新次数',ylabel='C选择特征误差（越低越好）',xticks=[0,250,500,750,1000]);ax.legend(fontsize=10);ax.grid(alpha=.15)
    fig.suptitle('按预定 C 损失选检查点，未使用验证命题或 D 结果',fontsize=15)
    fig.text(.035,.025,'蓝/绿＝仅多步链来源不同的两臂；空心大圈＝该臂选中检查点。纵轴为旧C37根与新C选择6根各半权的单步特征误差，\n不是符号准确率。所有运行都完成1000更新；选中更新分别为旧750/750、新500/750。C验证不参与此选择。',fontsize=10)
    fig.tight_layout(rect=[0,.14,1,.93]);fig.savefig(R/'C_checkpoint_selection.png',dpi=170);fig.savefig(R/'C_checkpoint_selection.svg');plt.close(fig)

    sup=['# 完整逐步统计','','H/I/屉分别为三色Holding、三色In(drawer)、抽屉Closed/Open/Intermediate；错组是该组三项至少一项出错的根数，可重叠。所有格分母6。全9须全部真值有效并同时匹配。实际、一步、递归使用同根同动作；真实读出不纠正递归状态。原MSE是固定256维潜变量均方差，不等于命题准确率。','','本次仅C回测，6个根此前已在历史实验中见过，但本轮训练及模型选择未读取。D3/D4未重新推理，也未采D5。',''];csvrows=[]
    for branch,title in [('normal','正常链'),('held_closed','持物关屉/拒绝支路')]:
        sup+=['## '+title,''];rows=[]
        for name,branches in m['trajectory_prediction'].items():
            for r in branches[branch]:
                nums=[r[k]['full9_correct']for k,_,_ in METHODS];a=r['one_step_paired']['states'];b=r['recursive_paired']['states']
                rows.append([NAMES[name],r['horizon'],'/'.join(map(str,nums)),*['/'.join(str(r[k]['groups'][g]['wrong_states'])for g in ['Holding','In','drawer'])for k,_,_ in METHODS],a['actual_right_prediction_wrong'],a['both_wrong'],b['actual_right_prediction_wrong'],b['both_wrong'],f'{r["one_step_raw_MSE"]:.5f}/{r["recursive_raw_MSE"]:.5f}'])
                csvrows.append({'condition':name,'branch':branch,'horizon':r['horizon'],'observed_roots':r['observed_roots'],'planned_roots':r['planned_roots'],'actual_all9':nums[0],'one_step_all9':nums[1],'recursive_all9':nums[2],**{k+'_'+g+'_wrong':r[k]['groups'][g]['wrong_states']for k,_,_ in METHODS for g in ['Holding','In','drawer']},'one_actual_right_prediction_wrong':a['actual_right_prediction_wrong'],'one_both_wrong':a['both_wrong'],'rec_actual_right_prediction_wrong':b['actual_right_prediction_wrong'],'rec_both_wrong':b['both_wrong'],'one_raw_MSE':r['one_step_raw_MSE'],'recursive_raw_MSE':r['recursive_raw_MSE']})
        sup+=table(['条件','步','真读/一步/递归全对','真读H/I/屉错','一步H/I/屉错','递归H/I/屉错','真读对一步错','两者均错','真读对递归错','两者均错','一步/递归原MSE'],rows)+['']
    diag=[]
    for name,rr in scores.items():
        for root in sorted({r['root_id']for r in rr}):
            nr=sorted([r for r in rr if r['root_id']==root and r['branch']=='normal'],key=lambda r:r['slot']);place=nr[2];end=nr[3];first=next(x for x in roots if x['condition']==name and x['root']==root and x['branch']=='normal'and x['method']=='recursive')
            diag.append({'condition':name,'root':root,'place_native_status':place['status'],'place_Holding_any_truth':any(place['truth'][:3]),'place_In_any_truth':any(place['truth'][3:6]),'place_all9_valid':all(place['valid']),'place_actual_errors':errors(place,'actual'),'place_one_step_errors':errors(place,'one_step'),'place_recursive_errors':errors(place,'recursive'),'close_recursive_errors':errors(end,'recursive'),'first_recursive_wrong_step':first['first_wrong_step'],'first_recursive_wrong_facts':','.join(first['first_wrong_facts']),'all_four_recursive_correct':first['all_four_correct']})
    sup+=['## 每根第3步的真实放置与首次递归错误','']+table(['条件','根','真实持有/入屉','第三步真读错','第三步一步错','第三步递归错','第四步递归错','首次递归错误'],[[NAMES[r['condition']],r['root'].replace('C_intermediate_v3_',''),str(int(r['place_Holding_any_truth']))+'/'+str(int(r['place_In_any_truth'])),r['place_actual_errors'],r['place_one_step_errors'],r['place_recursive_errors'],r['close_recursive_errors'],r['first_recursive_wrong_step']or'全4步正确']for r in diag])+['']
    (R/'SUPPLEMENT.md').write_text('\n'.join(sup)+'\n');write_csv(R/'prediction_summary.csv',csvrows);write_csv(R/'per_root_place_diagnostics.csv',diag)
    gate_names={'C_selection':'C选择损失低于旧链','initial_step3_gain':'第三步比起点至少多1根','initial_step3plus4_gain':'三四步合计比起点至少多2','old_chains_step3_gain':'第三步比旧链至少多1根','old_chains_step3plus4_gain':'三四步合计比旧链至少多2'}
    lines=['# 只替换多步训练链来源的有限对照','','完成时间：'+now(),'','本轮没有证明新增连续链带来跨种子的明确后段收益。正常链第三步递归全9正确数：旧链对照为5/6、2/6，新增链为5/6、3/6；第四步全部2/6。种子0的第三步改善同样出现在旧链训练中，不能归功于新增链来源。种子1仅改善1根，未达到预先冻结的后段合计门槛。','','已完成全部4次训练与隔离C回测，并按协议停止：没有采集新D根、没有追加训练、没有调整损失或阈值。此处C是历史上已看过、但本轮拟合和选择均隔离的开发回测，不是新的独立测试证据。','','## 对照匹配了哪些因素','','| 元素 | 两臂共同条件 | 本轮为何固定 |','|---|---|---|','| 转移起点 | 原canonical_matched种子0/update1000 | 排除不同预训练起点 |','| 查询头 | D4覆盖修正后按C选定的头0/1；各自配对固定 | 排除查询变好冒充动态变好 |','| 编码与解码 | DINO、预处理、编码器、解码器、归一化全部冻结 | 同一表示空间评价转移 |','| 单步训练 | 每臂1000更新×64样本；48旧A/B+12新增正常+4新增拒绝；两臂样本及顺序完全相同 | 只允许多步来源发生变化 |','| 多步数据 | 每臂12个A/B根，三色各4；每根2个连续三技能窗口，共24窗口 | 匹配根数、颜色和链长度 |','| 多步顺序 | 开屉→抓取→放置；抓取→放置→关屉 | 直接覆盖递归放置及其后续 |','| 动作接口 | 同技能、目标颜色、8维动作和96步后等待 | 不混合旧数据中无后等待的链 |','| 多步预算 | 每4更新1次，48链/批；每根每窗口各2次，共12000链抽样/运行 | 每次严格等根权重和等预算 |','| 损失与优化 | 原两项损失及权重；AdamW 1e-4，wd1e-4，clip5 | 不添加新的训练目标 |','','旧链取12个`A/B_longpair_{color}_closed_{0,1}`根，新增链取对应的12个`A/B_intermediate_v3_{color}_{0,1}`正常分支。每条链逐项核验同一根、同一episode、上一转移after等于下一转移before；从未跨复位或分支拼接。两臂每个窗口的完整动作向量逐步一致。','','初始物体布局、相机画面和实际轨迹随数据来源不同，未声称这些分布完全相同；这正是本次来源处理的一部分。旧数据中动作后等待为0的core链未混入多步对照，但它们仍按两臂完全相同的单步日程出现。没有基于执行成功或真值筛选训练窗口。','','多步损失保持原形式：第2、3步各加0.25×（原分组特征误差＋0.25×潜变量MSE）。为保证单步抽样不随链来源改变，多步窗口另从自己的真实初态算第1步潜变量；该步不额外增加损失。之后连续递归，没有中间真实状态纠正。','','## 模型选择与实际执行预算','','两种子、两臂均完成1000更新，总计256000次单步抽样、48000次三技能链抽样。保持原C选择规则：旧C37根和新增C选择6根各半权的单步特征误差，候选0/250/500/750/1000，损失最低且同分取更早检查点。','']
    rows=[]
    for n,s in f['selected'].items():rows.append([NAMES[n],1000,int(Path(s['path']).stem.replace('update','')),f'{s["C_score"]:.8f}'])
    lines+=table(['条件','完成更新','C选中更新','C选择损失'],rows)+['','![检查点选择](C_checkpoint_selection.png)','','全部选择在本轮C验证推理前封存。开发回测的命题正确率、D3/D4结果均未参与模型选择。D3/D4在本轮也没有重新推理。','','## 正常四技能链：同6根配对','','每格按技能1/2/3/4列出九命题全对根数，每步分母6。它们是预测正确数，不是任务或规划成功率。','']
    lines+=table(['条件','真实后继读出','真实前态单步','仅初态递归'],[[NAMES[n],seq(m,n,'actual'),seq(m,n,'one_step'),seq(m,n,'recursive')]for n in m['trajectory_prediction']])
    lines+=['','![C回测多步对照](C_multistep_comparison.png)','','各条件的真实状态读出逐概率完全一致，确认没有读出变化混入转移对照。种子0新增链递归第三步5/6与旧链5/6相同；种子1从旧链2/6到3/6。新增链两种子的第二步均6/6，第四步仍2/6。四个边界都全9正确的根数仍为2/6、1/6，与各自固定起点相同。','','## 第三步实际有没有放下，错误先发生在哪里','','6条实际正常链在放置后均已松手，Holding(red/blue/pink)全为0；其中5条满足严格In，红块第2号根不满足。后者虽被固定控制器报告成功且有支撑接触，完整姿态显示角点高出屉沿，原严格In标签仍为0。该状态与上一轮相同，没有改标签或剔除。','','新增链种子0的第三步剩1条错误是上述红块In误报，真实后继读出与递归同时犯错；没有“真实读对、第三步递归错”的根。其余3条链首次到第四步才出错。种子1第三步还有2条Holding误报，真实后继能读对而递归仍判持有；另1条是同一红块In错误。各根详情见[放置与首次错误表](per_root_place_diagnostics.csv)。','','| 新增链模型 | 第3步真读对/递归错 | 第3步两者均错 | 第3步递归 H/I/屉错组 | 第4步递归 H/I/屉错组 |','|---|---:|---:|---|---|']
    for seed in [0,1]:
        rr=m['trajectory_prediction'][f'new_chains_seed{seed}']['normal'];p=rr[2]['recursive_paired']['states'];g=lambda r:'/'.join(str(r['recursive']['groups'][k]['wrong_states'])for k in ['Holding','In','drawer']);lines.append(f'| 种子{seed} | {p["actual_right_prediction_wrong"]} | {p["both_wrong"]} | {g(rr[2])} | {g(rr[3])} |')
    lines+=['','组错可以重叠。第四步真实后继两种子均6/6能读对，递归各有4根错误，因此后续传播问题仍明确存在。潜空间MSE虽有所下降，不能把它替代命题正确率：例如种子0第三步旧/新链递归MSE为0.3911/0.3862，但两者同为5/6。','','## 拒绝支路单独报告','']
    lines+=table(['条件','真实后继读出','单步预测','递归预测'],[[NAMES[n],seq(m,n,'actual','held_closed'),seq(m,n,'one_step','held_closed'),seq(m,n,'recursive','held_closed')]for n in m['trajectory_prediction']])
    lines+=['','同6根的另一次复位：第一步抓取，后3次调用被拒绝。递归第2至4步所有条件仍0/6。该支路未并入正常链指标，不能把12条分支当作12个独立初态。','','## 为什么不进入新D测试','','开训前已冻结进入新D的门槛：两种子的新增链C选择损失都更低；C回测第三步比起点及旧链各至少多1根；第三、四步合计各至少多2个正确步；前两步递归以及第三、四步单步的每项退化不能超过1/6。此门槛只用于有限研究的推进决策，不是显著性检验或原指南验收线。','']
    lines+=table(['预定检查','种子0','种子1'],[[v,*['通过'if gate['seeds'][s]['checks'][k]else'未通过'for s in [0,1]]]for k,v in gate_names.items()])
    lines+=['','保留项两种子均通过，但核心收益条件未通过。失败原因不只是数值门槛：种子0与旧链的第三步结果相同，种子1只有单根收益，第四步及整条预测一致性没有提升。因此没有生成或采集D5，没有事后换选检查点，也没有额外修补。','','## 交付与边界','',f'- [冻结协议]({E}/configs/protocol.json)；[完整链匹配及根对应]({E}/configs/chain_matching.json)；[预定门槛结果]({E}/results/gate.json)。','- [完整统计](SUPPLEMENT.md)；[逐步汇总CSV](prediction_summary.csv)；[逐根放置/首次错误CSV](per_root_place_diagnostics.csv)；[逐命题配对2592行](paired_propositions_C_backtest.csv)。',f'- [独立复核]({E}/results/verification.json)确认4次完整预算、单步日程相同、链连续/动作匹配、各根均衡、冻结组件不变、逐命题计数正确，并复核先前5346个封存文件。',f'- 本轮完整文件及哈希清单：`{E}/configs/delivery_manifest.json`。','- [已完成D4修正报告](../empty_open_readout_correction_20261016/REPORT.md)及[先前D3补数报告](../intermediate_data_prediction_20261016/REPORT.md)保持封存。','','本轮沿用既有批准的PhysicalOracle控制器和有限预测研究条件，没有运行规划或任务成功率评测，不声称原HULC/TS-LTL完整指南已验收。当前无训练、采集进程；后续范围交回父任务审查。']
    (R/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT_READY',str(R),flush=True)
if __name__=='__main__':main()
