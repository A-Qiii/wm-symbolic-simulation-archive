from common_diag import *
import csv
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |'for r in rows)
q=frozen();audit=read(E/'results/sample_audit.json');g=read(E/'results/gradient_audit.json');info=read(E/'results/input_information_audit.json');done=read(E/'results/training_complete.json');fm=read(E/'results/final_metrics.json')
arms=['five_only','joint_original','joint_equal_groups'];labels=['仅五条','联合原权重','联合组等权'];hist={a:read(E/'runs'/a/'evaluations.json')for a in arms}
samples=table(['样本根','原持有→请求目标','真实后继七事实','掩码','原每种子展示','当前/后续读出'],[[r['pair_id'].replace('__b1__diagnostic',''),r['held_other']+'→'+r['target'],','.join(str(int(v))for v in r['y_after']),'7/7',175,'两端均全对']for r in audit['rows']])
fitrows=[];csvrows=[]
for a,lab in zip(arms,labels):
 for r in hist[a]:
  f=r['groups']['five'];o=r['groups']['old132'];fitrows.append([lab,r['update'],str(f['all7'])+'/5',str(o['all7'])+'/132',f"{f['BCE']:.6f}",f"{o['BCE']:.6f}"]);csvrows.append({'arm':a,'update':r['update'],'five_all7':f['all7'],'old132_all7':o['all7'],'five_BCE':f['BCE'],'old132_BCE':o['BCE'],'five_weighted_latent':f['weighted_latent'],'old132_weighted_latent':o['weighted_latent']})
fit_table=table(['对照','诊断步数','五条全7正确','原132全7正确','五条BCE','原132 BCE'],fitrows)
gradtable=table(['样本','目标错误概率','原始目标BCE','加权目标项','对目标logit导数','对目标隐状态梯度L2','参数梯度L2'],[[r['pair_id'].replace('_coverage_','_').replace('__b1__diagnostic',''),f"{r['target_probability']:.6f}",f"{r['raw_target_BCE']:.6f}",f"{r['contribution_to_five_total']:.6f}",f"{r['d_loss_d_target_logit']:.6f}",f"{r['target_latent_gradient_L2']:.6f}",f"{r['parameter_gradient_L2']:.6f}"]for r in g['five_target_facts']])
comp=table(['组','持有BCE/3','入屉BCE/3','打开BCE/3','对象潜状态×.025','场景潜状态×.025','总损失'],[[name]+[f"{g['group_components'][name][k]['loss']:.6f}"for k in ['held','in','open','object_latent','scene_latent','total']]for name in ['old132','five','joint137']])
near_table=table(['拒绝例','最近同动作成功例','目标h差异RMS','全局s差异RMS','另一物体h差异RMS','完全相同的目标输入'],[[r['new_pair'].replace('_coverage_','_').replace('__b1__diagnostic',''),r['nearest_success_pair'],f"{r['target_h96_RMS']:.6f}",f"{r['global_s96_RMS']:.6f}",f"{r['held_other_h96_RMS']:.6f}",'否']for r in info['rows']])
end_table=table(['对照','评估组','全7正确','BCE','加权隐状态项','384维MAE','绝对值>1比例'],[[labels[arms.index(r['arm'])],r['group'],f"{r['all7']}/{r['calls']}",f"{r['BCE']:.6f}",f"{r['weighted_latent']:.6f}",f"{r['latent_MAE']:.6f}",f"{100*r['abs_gt1_fraction']:.3f}%"]for r in fm])
proposal={'status':'proposal_only_not_executed','single_change':'Change only per-example group weights in the existing137 training objective: old132 and five open-refusal examples each contribute50%.','unchanged':['architecture and input contract','frozen reader and DINO','seven targets and masks','loss class weights and latent coefficient0.05','original data and no collection','same initialization, seed schedule, minibatch indices and fixed update budget for a future controlled comparison'],
 'exact_weighting':{'old132_group_share':.5,'five_group_share':.5,'uniform137_sampling_old_example_multiplier':137/(2*132),'uniform137_sampling_new_example_multiplier':137/(2*5),'normalization':'Multiply each example loss by the fixed importance weight then average the original minibatch; do not renormalize by number of members present in each batch and do not change sampler.'},
 'evidence':['At fixed step100 equal-group achieves five5/5 and old132/132 while uniform joint remains five0/5 and old132/132','At final500 both joint arms fit137/137; equal weighting not necessary for eventual fit','Final five target-held probabilities lower in equal group for all5','Five-only fit forgets60/132 old training calls; retaining original group matters','Nonzero finite-difference-validated gradients and correct labels/masks; no implementation bug found in checked paths'],
 'limits':['Single fixed seed0 and failed warm start','Fresh diagnostic optimizer and full batches differ from historical minibatches','Each sample has500 additional diagnostic presentations; this is not proof historical budget was sufficient','Only training cases assessed; no C/D evaluation','Higher new-group weight changes gradient clipping frequency under the same clipping rule','No claim of complete world-model or semantic generalization repair'],
 'next_validation_only_if_separately_authorized':'Create a separate fixed-budget controlled development comparison using this one objective change; evaluate old-C/state-C/drawer-C separately after freezing model selection. Do not promote diagnostic models or reuse oldD as independent test.'}
save(E/'NEXT_STEP_PROPOSAL.json',proposal)
save(E/'results/decision.json',{'created_utc':now(),'bounded_diagnostic_completed':True,'checked_implementation_bug_found':False,'correct_five_labels_masks_actions_and_batch_use':True,'frozen_current_and_next_readout_five5of5':True,'loss_and_gradient_effective':True,'five_only_fits':True,'uniform_joint_also_fits':True,'equal_group_improves_recorded_convergence':True,'architecture_cannot_fit_training_cases':False,'unique_cause_identified':False,'training_interference_supported':True,'latent_weight_only_cause_established':False,'semantic_information_use_proven':False,'generalization_evaluated':False,'candidate_promoted':False,'optimizer_updates':1500,'C_D_forwards':0,'original_negative_results_unchanged':True})

report=f'''# 五条打开拒绝样本的有边界训练失败诊断

**结论：模型能拟合这些训练例，也能在保留原任务的联合训练中拟合。** 本轮未发现所检查路径中的标签、掩码、动作/物体身份或梯度断路错误。五条与原132条的目标存在明显梯度冲突；只训五条会遗忘旧任务，组等权使联合对照更早在预定记录点达到全对。不过原权重联合训练到固定第500步也全部拟合，因此不能把权重称为唯一原因或必要修复。

本轮只做训练内部排错：固定种子0、三个临时对照各500步，总计1500次更新。临时模型没有成为候选，没有新增数据、C/D模型推理或多步实验。原五对事后探索、六根准入失败和独立D失败报告全部保留，整份原指南尚未完成。

## 先读的首错单项替换证据

上轮预定代表`open137_seed0`的首个C错误是`C_drawer_factor_red_0__b1__diagnostic`：已持蓝，请求抓红，抽屉打开；真实拒绝不改变持物。预测后继给持红0.888390、持蓝0.999677而触发UNKNOWN；只用已保存的真实后续编码替换预测后继，经同一个冻结头即读对七个事实。该例当前读出也正确。这是定位后继预测通路的证据，不能唯一定位到结构或损失。

本轮仅阅读[既有单项替换结果](../state_predicates_coverage_exploratory5_20261015/results/single_replacement_diagnostic.json)，没有重跑C。随后对真正参加上轮训练的五条AB打开样本进行以下独立核对。

## 五条实际训练目标与来源

七事实列顺序固定为：持红、持蓝、持粉、红入屉、蓝入屉、粉入屉、抽屉开。五条都是真实执行器拒绝、原因`requires_empty_gripper`，后继与当前事实一致；标签不是“请求抓取就应成功”。

{samples}

动作向量固定为[open,lift,place,close,target_red,target_blue,target_pink,budget/900,posthold/96]。五条都编码为lift，目标one-hot与声明颜色一致，预算为900、保持96步。完整AB137的声明动作逐条复算相等。RGB对象槽红/蓝/粉与共享头held前三列、in中三列顺序一致。

五条的before/after标签、35个有效掩码、当前/后续隐状态和动作均逐元素对齐原封存采集记录；没有掩掉错误事实或把真实持有物当作请求目标。原三个种子的批索引中，这五个局部槽132–136各实际出现175次。证据在[sample_audit.json](results/sample_audit.json)。

冻结读出器直接读取真实当前及真实后续隐状态，五条两端均5/5全7正确；原持物概率约0.99995，请求目标持有概率约0.00004。这说明当前保存的隐状态中存在可被冻结头读出的持有信息。双精度诊断与旧单精度缓存概率有微小舍入差（AB137当前最大约4.99e−5，后续约6.06e−6），数组来源本身逐元素相等，五条判定不受影响。

## 损失与梯度不是断开的

沿用三组类别加权BCE均值 + 0.05×(对象SmoothL1+场景SmoothL1)/2。持有/入屉/打开类别权重分别为[0.5625,4.5]、[0.5963855421686747,3.09375]、[1.375,0.7857142857142857]，并未按五条重新估计。手工softplus形式与训练BCE实现双精度核对一致，分组梯度重组相对误差1.45e−15。

以下在原失败种子0权重处计算，梯度诊断没有优化器更新。目标持有真值全部为0，正的logit导数表示梯度下降会降低错误的目标持有概率；贡献已含持有组/3、三颜色均值及五样本均值。

{gradtable}

对logit的解析导数、自动微分和中心差分逐条一致；参数方向差分为2.987593，自动微分预期2.987595。共享读出器参数保持冻结且无参数梯度，但预测隐状态与转移网络参数得到非零梯度，未出现把冻结读出误用为no_grad而截断训练的情况。

各项实际带权损失如下（各组按自身样本平均）：

{comp}

五条符号梯度与原132符号梯度余弦为−0.658；五条总目标与原132总目标余弦为−0.727，存在反向推动。五条自身符号与隐状态梯度余弦却为+0.382，因此不能仅凭之前的聚合竞争，就认定这五条被隐状态项压制或优先删掉该项。

在失败起点，沿原132总目标负梯度，五条目标BCE的一阶变化为+0.233050（变差）；沿五条总目标负梯度为−8.974517（改善）。按132:5混合后的负梯度仍为−0.102993，并非完全失效；等组权重下为−4.370734。它们是局部梯度方向探针，不能当作带动量、预条件和裁剪的AdamW实际更新贡献。

![损失竞争证据](figures/gradient_evidence.png)

左图横轴是参数梯度余弦，红色负值表示方向冲突、绿色正值表示同向；右图蓝/橙柱分别为原132/五条加组权重后的符号梯度范数。右侧分量范数不能直接相加成总范数。组等权把五条总占比从5/137=3.65%改为50%，并非让它们出现更多次。

## 预先固定的小样本拟合设计

原残差网络的零末层初始化相当于保持当前隐状态，天然适合无变化拒绝例。因此不拿全新初始化对五条直接得分当“已经学会”，而从实际失败的`open137_seed0/update1500.pt`权重出发，三个对照同一起点SHA256 `fca50be9a28de158996c6402bdf7749bd6aea060c5f74df15e73b7dc40703432`。

| 对照 | 每步参与目标的样本 | 目标 | 排错问题 |
| --- | --- | --- | --- |
| 仅五条 | 固定全批5 | 原完整损失 | 能否从实际失败权重恢复五条 |
| 联合原权重 | 固定全批137 | 原完整损失；原132/五条占比132/137与5/137 | 保留旧目标后是否仍能拟合 |
| 联合组等权 | 固定全批137 | 0.5×原132平均损失+0.5×五条平均损失 | 仅改组权重是否改变联合拟合 |

三个对照均固定种子0、原76224参数结构、冻结DINO/读出/归一化与目标、AdamW lr0.001/weight_decay0.0001/clip1；诊断优化器各自重新置零，不继承旧优化器动量。固定500步，即使提前拟合也继续到500；只有非有限损失/梯度或资源故障才提前停止，没有按准确率续训。预定记录点0、25、100、250、500，最终仅保留第500步临时模型，不挑最好检查点。

方案在任何更新前写入[protocol.json](configs/protocol.json)，SHA256 `07676aea9bf669901a4a3f5b4a8f7cdad8eebc6ad42b9b63db9f93fe31bb4de4`。三个对照均实际500步，优化器step和1500条日志核验通过。

## 拟合结果与旧任务回归

{fit_table}

![固定预算拟合与回归](figures/fit_and_regression.png)

橙色仅五条，蓝色联合原权重，绿色联合组等权；横轴临时更新次数，纵轴全7正确率。左图分母5，右图分母132。点为预定记录时刻，连接线不是中间每步测量；所以“第100步已全对”指该记录点，不能称精确首次成功时间。

只训五条在第25步记录为5/5，但旧任务76/132，最终72/132，显示明显遗忘。两个联合对照在全部预定记录点保持旧132/132。组等权第100步记录为五条5/5，原权重同点仍0/5；原权重到第250步为4/5，第500步也为5/5。这推翻“模型连训练例都不能拟合”的解释，也说明不能把旧失败简单写成不可解决的输入信息缺失。

![最终目标持有概率](figures/target_probabilities.png)

横轴五个训练条件，纵轴请求目标的持有概率；真实目标均未持有，固定阈值0.5。灰为失败起点，橙/蓝/绿为三个第500步临时模型。联合原权重最终最大目标误报概率0.167529，组等权0.038338，五个目标概率均更低。此处比较的是训练例的概率和边界，不是泛化准确率或校准结论。

最终连续损失与隐状态指标也完整保留：

{end_table}

两联合组虽然符号全对，隐状态误差仍非零；没有认定世界模型整体修复。只训五条产生的旧任务退化也没有删除。

## 输入信息能否被利用

对每条拒绝例，在原AB132里寻找动作向量相同、真实目标抓取成功的最近目标输入。对象更新实际读自身h（96维）、全局s（96维）、动作类型4、目标匹配1和预算2，共199维；不直接读其他对象h。距离是在这个实际199维输入上的RMS。下表仅是现有缓存的可辨识性检查，不新增样本。

{near_table}

![输入信息对比](figures/input_information.png)

蓝色为目标h差异，橙色为全局s差异，绿色为原持有物h差异；纵轴为RMS对数轴，不同颜色的柱是不同输入分量。另一物体h差异大且可由冻结头读出；只替换另一物体h、保持目标h/global s/动作固定时，目标输出变化严格为0，符合代码的输入路径。

这揭示了信息利用限制：目标只能经全局上下文间接获得另一物体状态。但最近输入并非完全相同，全局s中可能有足够线索；两个联合对照也确实能在这些样本上同时拟合。因此不能据此断言结构必然无法表达拒绝，也没有证明模型学到了可泛化的“已持物应拒绝”语义，而非利用这些训练布局的细微差异。

## 最值得尝试的一个修改

**只改现有137条目标中的组权重，让原132与五条打开拒绝条件各贡献50%，先不改结构或隐状态损失系数。** 两个联合对照唯一设计差别就是此项，组等权在预定记录点更早拟合且同点旧任务无准确率损失，支持把它作为下一项开发对照。

若后续在原均匀137样本流、原batch16上检验，应固定原采样索引，仅将旧样本损失乘137/(2×132)=0.518939、新五条损失乘137/(2×5)=13.7，再按原批次平均。不要按批内出现数量重新归一化，也不要同时改采样器。这样期望目标对应本次组等权；其他损失内部比例仍保持不变。方案只提出于[NEXT_STEP_PROPOSAL.json](NEXT_STEP_PROPOSAL.json)，未执行额外训练。

限制必须保留：原权重联合对照最终也成功，所以不是“必须重加权才可拟合”；本次从失败权重热启动、重置优化器、用全批并额外展示每条500次，与历史1500次batch16更新不同，不能将其当作原训练预算内的正式修复。两个联合对照每条展示次数相同，可支持组权重影响本轮收敛过程，但不能唯一分解旧失败中批噪声、历史优化路径和相对目标的全部贡献。相同clip1规则下实际裁剪次数也不同（仅五条4次、联合原权重1次、组等权16次）。没有C/D结果，不做候选推广或泛化结论。

## 文件与执行完整性

本机继续使用独立world_model_env.sh（Python3.11.13、PyTorch2.11cu128、RTX5080 Laptop），GPU锁串行；绘图在既有env.sh进行。没有改变旧模型或环境。

训练脚本首次启动因两处条件表达式缺少空格而在Python解析时停止，优化更新为0。冻结原脚本及失败日志保留；修正版另存`source/train_diagnostic_v2.py`，运行前记录[launch_fix.json](configs/launch_fix.json)及新源码SHA；仅改两处语法并增加自校验，模型、目标、预算、停止条件未变。审计日志另有一次将需梯度张量转标量的PyTorch提示，用于记录数值；后续差分和实际拟合验证了梯度有效，没有因此改变损失实现。

| 证据 | 内容 |
| --- | --- |
| [sample_audit.json](results/sample_audit.json) | 五条标签/掩码/来源/身份/批次及真实隐状态读出 |
| [gradient_audit.json](results/gradient_audit.json) | 分项损失、梯度范数/余弦、逐事实导数和差分 |
| [input_information_audit.json](results/input_information_audit.json) | 最近成功输入及其他对象替换探针 |
| [fit_history.csv](results/fit_history.csv) | 三组全部预定记录点的训练正确率及连续损失 |
| [final_training_audit.json](results/final_training_audit.json) | 1500次实际更新、三个优化器step500与最终文件SHA |
| [decision.json](results/decision.json) | 能支持及不能支持的机器可读结论 |
| [immutable_after.json](results/immutable_after.json) | 旧九个封存目录的逐文件内容和只读状态复核 |
| [HANDOFF.md](HANDOFF.md) | 命令入口、路径、后续限制和父任务交接 |

本轮按固定预算结束，未因看到有利或不利结果增加对照、更新、数据或种子。原五对探索失败及所有旧结果仍有效；临时模型只用于本诊断。
'''
(E/'REPORT.md').write_text(report)
with (E/'results/fit_history.csv').open('w',newline='',encoding='utf-8-sig')as f:
 w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
handoff=f'''# 训练失败诊断完成

本机目录：{E}
REPORT.md含完整结论、审计、四张中文图；NEXT_STEP_PROPOSAL.json只建议一次组权重修改，尚未执行正式开发训练。

核心证据：五条标签/35掩码/目标身份/动作/历史175次展示正确；真实当前与真实后继的冻结读出均5/5。目标错误的梯度非零，logit与参数方向差分通过。原132与五条符号梯度余弦−0.658；五条自身符号与隐状态梯度同向+0.382。未发现检查路径中的实现错误，不能笼统归因于潜状态项压制。

固定种子0，三个临时模型都从上轮失败open137_seed0最终权重出发，优化器重新置零，原结构/损失/阈值不变，各500步，总1500。仅五条：第25步记录5/5，最终旧任务72/132。联合原权重：第100步0/5，第250步4/5，第500步5/5，旧任务始终132/132。联合组等权：第100步已5/5，后续记录仍5/5，旧任务始终132/132。最终两联合均拟合全部137，组等权最终五条目标错误概率更低，不能声称等权是必要条件。

最值得尝试的单项修改：原132/五条组权重各50%。若在原均匀137样本流batch16对照中检验，仅乘固定每样本权重旧0.518939、新13.7，不改采样/结构/.05隐状态系数/原预算。该建议未执行。当前只证明训练内可拟合及本次等权收敛更早；热启动、全批、重置优化器与历史小批不同，不能混作正式修复。全局s可能包含其他持物信息，未证明输入表达能力不足，也未证明语义泛化。

运行命令入口（已执行，不应对封存目录重跑）：
1. source world_model_env.sh; python source/prepare.py（在本实验路径下的完整入口），旧九目录逐文件核验；预冻结协议SHA 07676aea9bf669901a4a3f5b4a8f7cdad8eebc6ad42b9b63db9f93fe31bb4de4。
2. python source/audit.py：仅AB输入、梯度和最近输入诊断，0优化更新。
3. 原source/train_diagnostic.py解析时因else500/else137缺空格失败，0更新。原文件和logs/training.log保留，configs/launch_fix.json冻结修订。
4. python source/train_diagnostic_v2.py：实际三组各500步，日志logs/training_after_syntax_fix.log。只有final500检查点；模型文件名含diagnostic_only。
5. python source/verify_training.py：CPU检查3优化器实际step500、1500日志、保存输出得分和源文件哈希；无模型新推理。
6. source env.sh; python source/render.py：四张中文图，图例避开数据区域并逐张目视检查。
7. python source/write_report.py；python source/seal.py：生成报告、旧证据完整性复核和新目录只读清单。

无C/D优化或模型前向，无仿真新增，无多步，无候选替换，无新种子/预算扩展。读取上轮首个C错误的已保存替换证据，不算重测。原始指南整体未完成，原D失败、六根准入失败、五对探索失败结论均保持原样。

关键路径：configs/protocol.json、configs/launch_fix.json、results/sample_audit.json、gradient_audit.json、input_information_audit.json、fit_history.csv、final_training_audit.json、decision.json、immutable_before.json/immutable_after.json、delivery_manifest.json。封存回执在/home/robot/Documents/Codex/2026-10-13/task-4/fit_diagnostic_delivery_receipt.json。
'''
(E/'HANDOFF.md').write_text(handoff)
fig=read(E/'results/figure_manifest.json');fig['requires_visual_inspection']=False;fig['all4_visually_inspected']=True;fig['inspection']='Chinese titles, axes, colour definitions and interpretation limits readable; legends moved outside data for probabilities and input-distance plots.';save(E/'results/figure_manifest.json',fig)
print(json.dumps({'report_characters':len(report),'decision':'training_cases_fit; group_weighting_accelerates_in_this_diagnostic; no_generalization_claim'},ensure_ascii=False))
