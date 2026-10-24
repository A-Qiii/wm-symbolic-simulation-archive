from common_control import *
def table(h,rows):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |'for r in rows)
q=frozen();ev=read(E/'results/evaluation.json');decision=read(E/'results/decision.json');detail=read(E/'results/call_details.json');err=read(E/'results/error_partition.json');audit=read(E/'results/training_audit.json');families=['start','uniform','equal_groups'];models=[f'{a}_seed{s}'for a in families for s in [0,1,2]]
main=[]
for m in models:
 main.append([m]+[f"{ev['metrics'][m][g]['all']['all7']}/{ev['groups'][g]}"for g in ['C_old','C_state','C_drawer','C_combined']]+[f"{ev['paired_summary'][m]['C_state']['both_all7']}/6",f"{ev['paired_summary'][m]['C_drawer']['both_all7']}/6",f"{ev['metrics'][m]['AB_new_open']['all']['all7']}/5"])
main_table=table(['模型','旧C24','状态C18','抽屉C12','合并C54','状态双侧','抽屉双侧','五条训练拟合'],main)
diff_table=table(['种子','原权重相对起点：抽屉双侧增加','等权相对起点：抽屉双侧增加','等权相对原权重：双侧增加','等权相对原权重：抽屉调用增加','旧C/状态C差'],[[r['seed'],r['drawer_both_uniform_minus_start'],r['drawer_both_equal_minus_start'],r['drawer_both_equal_minus_uniform'],r['C_drawer_equal_minus_uniform'],'0 / 0']for r in decision['same_seed_differences']])
read_table=table(['集合','当前读出全7正确','真实后续读出全7正确','两端均正确'],[[g,f"{v['current_correct']}/{v['calls']}",f"{v['actual_after_correct']}/{v['calls']}",f"{v['both_correct']}/{v['calls']}"]for g,v in ev['readout'].items()if g in ['AB_original132','AB_new_open','C_old','C_state','C_drawer','C_combined']])
error_table=table(['最终模型','C预测错误数','其中当前读出错误','当前对而后续读出错','两端读出均对仍预测错','完整两端读出正确根双侧'],[[r['model'],r['wrong'],r['current_readout_wrong'],r['current_correct_after_wrong'],r['both_readouts_correct'],f"{ev['paired_summary'][r['model']]['C_drawer']['clean_both_all7']}/5"]for r in err if r['set']=='C_combined'and r['model'].startswith(('uniform','equal_groups'))])
lat_table=table(['模型','对象MAE','场景MAE','384维MAE','绝对值>1比例','绝对值>2比例','最大绝对值'],[[m]+[f"{ev['latent_metrics'][m]['C_combined'][k]:.6f}"for k in ['object_MAE','scene_MAE','all384_MAE']]+[f"{100*ev['latent_metrics'][m]['C_combined'][k]:.3f}%"for k in ['abs_gt1_fraction','abs_gt2_fraction']]+[f"{ev['latent_metrics'][m]['C_combined']['max_abs']:.6f}"]for m in models])
lat_group=table(['最终模型','C组','对象MAE','场景MAE','384维MAE','绝对值>1比例'],[[m,g]+[f"{ev['latent_metrics'][m][g][k]:.6f}"for k in ['object_MAE','scene_MAE','all384_MAE']]+[f"{100*ev['latent_metrics'][m][g]['abs_gt1_fraction']:.3f}%"]for m in models if not m.startswith('start')for g in ['C_old','C_state','C_drawer']])
root_rows=[]
for m in models:
 rows=[r for r in detail if r['model']==m and r['group']=='C_drawer'];root_rows.append([m]+[f"{sum(r['all7_correct']for r in rows if r['drawer_condition']==side)}/6"for side in ['closed','open']])
side_table=table(['模型','关闭侧全7正确','打开侧全7正确'],root_rows)
training_rows=[];csvrows=[]
for r in audit['runs']:
 obs=read(E/'runs'/r['run']/'training_observations.json');training_rows.append([r['run'],500,68500,500,f"{obs[-1]['old132_all7']}/132",f"{obs[-1]['new5_all7']}/5",min(o['old132_all7']for o in obs),r['gradient_clipped_steps']]);csvrows +=[{'run':r['run'],**o}for o in obs]
training_table=table(['运行','新更新','样本展示','每条展示','最终旧132','最终五条','记录点旧任务最低正确数','梯度裁剪次数'],training_rows)
csvsave(E/'results/training_observations_all.csv',csvrows)

report=f'''# 损失组权重单项对照：有额外开发收益，尚未完整恢复

**在三个固定种子上，组等权相对同预算原权重都提高了抽屉C的开闭双侧正确数，分别多1、1、6根；旧C24和状态C18最终均保持全对。** 原权重种子0、1本身也有明显改善，这部分共同收益可由额外优化解释，不能算成组等权独有的作用。组等权仍有两个种子各错一个打开调用，而且合并C的隐状态MAE在三个种子上均比同预算原权重更大。

本轮六个运行各固定500次更新，共新增3000次；全部完成后才统一评估既有C。没有按开发结果挑种子或检查点、追加训练、重测D、增加仿真样本或进入多步。当前是开发对照，不是独立泛化证明，原指南整体验收仍未完成。

## 与原1500步及上一轮诊断的区别

训练前已保存[HISTORY_COMPARISON.md](HISTORY_COMPARISON.md)及[机器记录](configs/history_comparison.json)。关键区别如下：

| 项目 | 原1500步训练 | 上轮联合诊断 | 本轮两组配对对照 |
| --- | --- | --- | --- |
| 参数起点 | 每个种子从头初始化 | 原open137_seed0第1500步权重 | 各自open137_seed0/1/2第1500步权重；同种子两组字节相同 |
| 新增更新次数 | 1500 | 每组500 | 每组500，共6组3000 |
| 参数经历累计更新 | 1500 | 1500+500=2000 | 1500+500=2000，不接上轮临时模型 |
| 批次 | batch16小批次 | 全批137 | 相同全批137、相同顺序，重复500次 |
| 新增五条每条曝光 | 175次 | 额外500次，累计675次 | 额外500次，累计675次 |
| 每运行样本展示 | 24000 | 额外68500 | 额外68500 |
| 优化器 | 训练开始新建，运行中不重置 | 加载参数后重置AdamW | 两组都重置AdamW，不加载旧动量 |

上轮原权重再优化500步已经能达到五条5/5。该改善不能直接归因于组等权。额外步数、全批、曝光增加和优化器重置都与原训练不同；本轮仅在两组之间控制它们相同，从而检验组权重的额外效果。若原权重也改善，额外优化这套过程已足以解释那部分改善；本轮没有把“更多步数”与“全批/重置”进一步单独分离。

## 唯一改变的因素与冻结方案

共同训练数据为原AB132条加已有五条打开拒绝样本，共137条。两组使用同一个500×137批索引文件；固定种子0/1/2，同种子从同一个原open137检查点出发。没有改五条选择，仍是此前事后选择的五对严格匹配准备条件，颜色循环2/2/1、A/B3/2；原六根准入失败和全部旧数据保留。

| 元素 | 原权重组 uniform | 组等权 equal_groups | 为什么这样设置 |
| --- | --- | --- | --- |
| 原132总损失系数 | 132/137 | 0.5 | 改变训练条件的目标份额 |
| 新五条总损失系数 | 5/137 | 0.5 | 同样曝光，不通过重复采样增加次数 |
| 组内损失 | 原完整符号+隐状态损失 | 完全相同 | 不同时改类别权重或隐状态系数 |
| 起点、批次、优化器、更新预算 | 同种子配对相同 | 同种子配对相同 | 仅这两个组系数不同 |

两组都用相同的分组损失代码，再乘各自组系数；原权重计算与直接整批137计算在第一次更新前核验误差<1e−6。[协议](configs/protocol.json)SHA256为`182859996c9db26e3c0d5aba9c9f27557694e6d882b0d4388bec48533d015fb4`，训练前冻结。

结构仍为76224参数残差LatentTransition；DINO、归一化、编码器/GRU和共享读出头冻结。组内损失为三组类别加权BCE均值 + 0.025×对象SmoothL1 + 0.025×场景SmoothL1。held/in/open类别权重分别为[0.5625,4.5]、[0.5963855421686747,3.09375]、[1.375,0.7857142857142857]。AdamW lr0.001、weight_decay0.0001、clip1全部一致；类别权重不重算，0.5阈值和多持物UNKNOWN规则不改。

终止标准为每运行恰好500次更新；只有非有限损失/梯度或资源故障才提前停止。最终只比较第500步。预定种子0为每组展示代表，三个种子全部报告。训练观察点为0/25/100/250/500，仅AB；C在全部3000步后评价。

## 三组C分别及合并结果

全7正确要求持红/蓝/粉、红/蓝/粉入屉、抽屉开全部正确；UNKNOWN计错。双侧正确要求同一根两次对比调用均全7正确。状态C沿用原预定义对比，抽屉C为关闭/打开两侧。三组C各6根，合并18根54调用，均为已有开发材料，抽屉C还曾用于定位问题。

{main_table}

起点结果来自原封存预测缓存，没有为了本轮重新推理旧起点模型。`uniform`与`equal_groups`才是本轮同预算配对对照。状态C的双侧目标事实亦始终6/6，抽屉C目标事实双侧数与表内全7双侧数一致。

![开发评价及训练拟合](figures/development_results.png)

灰=原1500步起点、蓝=额外500步原权重、橙=额外500步组等权；柱为三种子均值，圆/方/三角分别为种子0/1/2。左纵轴全7正确率，右纵轴分别是五条训练例正确数（分母5）和抽屉双侧正确根数（分母6）。这些重复种子的结果不是新增独立环境样本。

{diff_table}

原权重种子0/1已将双侧正确从0增至4根，这些收益无需组等权也能发生；种子2原权重双侧仍0，抽屉C总正确反而由6降至5。等权对照则分别达到5、5、6根，满足预冻结的“三个配对种子均有额外双侧收益且旧C/状态C不下降”条件。这里只有三个固定种子与六个复用C根，不能把方向一致称为普遍统计结论。

{side_table}

![逐根开闭结果](figures/root_pairs.png)

横轴是六个抽屉C根，纵轴是训练组和种子；灰=两侧错，蓝=仅关闭对，橙=仅打开对，绿=两侧都对。原权重种子1/2在blue_0关闭侧产生退化，组等权保持全部关闭侧正确。等权种子0/1仍有pink_1打开侧错误，没有将其因接近阈值而判作通过。

## 旧任务退化与五条拟合

最终六个新模型在原AB132均132/132，在五条配对关闭条件均5/5。打开五条拟合为原权重[5,5,0]/5、组等权[5,5,5]/5。旧C24和状态C18最终全部全对。抽屉C属于需诊断的新增开发条件，其关闭侧退化已如实列出，不能只说“没有任何回归”。

{training_table}

![训练内拟合过程](figures/training_fit.png)

蓝为原权重，橙为组等权；上排纵轴五条全7正确数，下排是旧132全7正确数。横轴追加优化步数。下排使用截取纵轴以显示暂时退化，图下注明范围。点只在预定时刻记录，线不代表中间每步得分；不存在按最有利时刻选模型。组等权种子1第25步旧任务125/132、种子2第100步129/132，都在最终500步恢复132/132，未删除中途退化。

## 真实读出错误与预测错误分离

{read_table}

已知读出错误仍是原AB的`A_longpair_blue_open_1__m0`当前读出，以及抽屉C的`C_drawer_factor_blue_0__b0__diagnostic`当前/真实后续读出。它们始终保留在训练或评价分母中，未以读出是否正确筛选模型结果。

{error_table}

五个完整根的两侧读出均正确时，双侧数原权重为[4,4,0]/5、等权为[4,4,5]/5；种子0/1在这个较窄完整根子集上没有增加。原因是它们新增恢复的是blue_0的打开侧，而该根关闭侧存在读出错误，整个根被排除出“完整两侧读出都正确”子集。按调用而非整根看，blue_0打开侧读出正确，其预测改善仍是转移收益。不能用某一个子集掩盖完整结果，也不能把全部新增恢复说成所有子集都同样改善。

## 首个剩余错误的单项替换

预先固定的`equal_groups_seed0`首个C错误为`C_drawer_factor_pink_1__b1__diagnostic`：已持红、请求抓粉、抽屉打开；真实拒绝且状态不变。当前及真实后续读出均正确。原权重种子0给持粉0.875926，组等权降至0.519486，但仍超过固定0.5，与持红0.999988同时阳性，触发两个UNKNOWN。种子1同例持粉0.570781，也错误。

![首个剩余错误](figures/first_error.png)

横轴七个事实，纵轴真概率；灰/蓝/橙分别是原起点/同预算原权重/组等权，绿为保存的实际后续编码经同一个冻结头，黑菱形为真实0/1，虚线为0.5阈值。仅在离线诊断里替换后继编码，这个案例可被读对（真实后续持粉约0.00003965）；未把真实未来输入模型、未在线修复，也没有据此续训或调阈值。

模型种子0合并C为53/54，真实后续读出参考也是53/54，但错误位置不同：参考在blue_0关闭侧错，模型在pink_1打开侧错。相同总分不能解释为模型已经达到全部可用信息的正确转移。

## 隐状态偏离单独报告

真实GRU编码处于[-1,1]；残差转移输出无界，没有裁剪。MAE衡量预测与实际后续编码的差；绝对值>1比例描述预测范围，不是调用错误率。合并C54共有20736个坐标：每调用3×96对象维与96场景维。

{lat_table}

与起点相比，两组的合并MAE都有下降；但组等权相对同预算原权重在三个种子上均更高：约+0.008062、+0.015456、+0.084850。种子0/1组等权范围外比例略低，种子2更高；最大绝对值仍约12.55–12.68。符号任务的改善并未建立隐状态整体修复。

三组C分项为：

{lat_group}

![隐状态误差与范围](figures/latent_quality.png)

颜色与种子点形同前；横轴三组C，左纵轴384维MAE，右纵轴范围外坐标百分比。完整对象/场景分项、SmoothL1、>1.1及>2范围和最大值均在[evaluation.json](results/evaluation.json)，没有只报告更好的一项。

## 可支持的结论与下一步边界

本轮支持：在相同已训练起点、重置优化器、全批137及固定500次更新下，组等权对三个固定种子的抽屉双侧预测有额外开发收益，最终旧C/状态C准确率保持。原权重在种子0/1也改善，证明额外优化过程本身能解释共同收益；权重作用应以配对差1/1/6根衡量，而不是以相对原1500步的全部改善衡量。

本轮不支持：组等权是唯一能拟合的方法、每个种子已完整恢复、隐状态已修复、未见条件独立泛化已通过。原权重与等权都是从已训练参数继续；这也不是从头训练的组权重因果对照。

训练前协议分别规定了“额外收益”和更严格的“全种子完整恢复”两个条件。前者通过，后者两组均未通过：等权种子0/1仍11/12抽屉调用、5/6双侧。按本轮预冻结的保守后续准入规则，不提出可直接执行的新独立验收方案，不将种子2改选为代表；代表仍是预定种子0。结果与未满足条件记录在[NEXT_STEP_DECISION.json](NEXT_STEP_DECISION.json)。没有扩大训练或采集来追求通过，后续若改变准入目的或开展新实验需另立方案；这不是将当前失败门槛事后放宽。

## 运行与保存证据

本机独立世界模型环境为Python3.11.13、PyTorch2.11cu128、RTX5080 Laptop；GPU锁串行。没有修改驱动、旧项目、ROS或其他应用。训练源代码在冻结前通过语法检查，本轮六运行均一次完成。原权重采用与等权相同的分组求和实现，和上轮诊断直接全批求和存在浮点求和顺序区别；在起点验证数学等价误差<1e−6，不承诺跨实现训练轨迹逐位一致。

[训练审计](results/training_audit.json)验证六个最终优化器step500、3000条逐步日志、同种子起点摘要、同一个批次文件、每条展示500次、损失仅组系数不同。C优化器/选择行数0。[评价审计](results/evaluation_audit.json)从保存概率独立复算三组C及双侧计数、MAE，确认全部训练后再评价，模型输入仅current_objects/current_scene/declared_actions，未来标签在预测保存后读取评分；D前向0。

全部技术配置、模型、日志、逐调用CSV及源代码保存在本目录，旧十个封存目录前后逐文件哈希和只读状态验证见[immutable_after.json](results/immutable_after.json)。[HANDOFF.md](HANDOFF.md)列出入口与证据路径。新目录通过清单封存，原五对事后探索失败、六根准入失败和原D失败报告均保持不变。
'''
(E/'REPORT.md').write_text(report)
next_decision={'status':'no_independent_validation_plan_under_frozen_gate','stable_extra_weight_benefit':True,'complete_restoration_all3seeds':decision['complete_restoration_all3seeds'],'reason':'Equal-group seeds0/1 each retain pink_1 open error; no arm passes preregistered all-seed complete restoration gate for proposing independent validation.','not_a_new_user_required_acceptance_standard':'This is the conservative next-step proposal gate frozen for this comparison, separate from the original guide and the stable benefit criterion.','representative_unchanged':'equal_groups_seed0','not_selected':'equal_groups_seed2 despite its perfect C result','remaining_errors':[{'model':r['model'],'pair_id':r['pair_id'],'probabilities':r['probabilities'],'current_readout_correct':r['current_readout_correct'],'actual_after_readout_correct':r['actual_after_readout_correct']}for r in detail if r['model'].startswith('equal_groups')and r['split']=='dev'and not r['all7_correct']],'no_automatic_followup':['training extension','threshold change','seed replacement','D retest','new simulation collection','multistep','candidate promotion'],'result_to_retain':'Stable additional symbolic development benefit alongside latent tradeoff and unresolved seed0/1 transfer errors.'};save(E/'NEXT_STEP_DECISION.json',next_decision)
handoff=f'''# 三种子组权重单项对照完成

目录：{E}
主要交付：REPORT.md、HISTORY_COMPARISON.md、五张中文图、results/evaluation.json、decision.json、训练与评价审计，以及NEXT_STEP_DECISION.json。

实际六运行各500步，共3000新更新；同种子均从原探索open137_seed0/1/2第1500步权重出发，不接上一轮诊断临时模型。两组均重置AdamW，同全批137顺序、同500步预算，只变原132/新五条损失份额132:5→1:1。相对原训练累计2000参数更新、五条累计675次展示。原训练1500次batch16、每新样本175次；本轮额外500次全批使每样本再展示500次，不能把额外优化共同收益归因组权重。

结果（种子0/1/2）：旧C均24/24，状态C均18/18，状态双侧均6/6。抽屉C起点6/12各；原权重10/12、9/12、5/12，等权11/12、11/12、12/12。抽屉双侧原权重4/6、4/6、0/6，等权5/6、5/6、6/6；配对额外收益+1,+1,+6，三个种子均正，额外收益门槛通过。合并C为原权重52/54、51/54、47/54，等权53/54、53/54、54/54。五条训练拟合原权重5/5,5/5,0/5，等权全部5/5；最终原AB132和新五条关闭侧全部正确。

共同收益：原权重种子0/1已恢复四根双侧，不能算成权重独有作用。旧C/状态C最终未退化，但原权重种子1/2在Cblue_0关闭侧退化。组等权中途训练退化也保留：seed1在25步旧125/132，seed2在100步旧129/132，最终恢复。

读出错误：Cblue_0关闭侧当前/实际后续读出错仍保留；五个两侧读出均正确根的配对数原权重4,4,0、等权4,4,5。原权重seed0/1新增恢复的blue_0打开侧本身读出正确，完整根清洁子集会排除整个blue_0。种子0/1等权剩余同一Cpink_1打开错误，两端读出都对。预定seed0首错持粉p=.51948565（seed1=.57078063），与持红阳性冲突成UNKNOWN；阈值不改，保存真实后续编码同头能读对该例。整体真实后续读出53/54与seed0模型53/54错位不同，不能称同样完备。

隐状态：等权相对原权重的合并C384维MAE三个种子均更高（.483432 vs .475371；.539397 vs .523941；.577510 vs .492660）。符号收益不能说成隐状态修复。

方案冻结SHA：182859996c9db26e3c0d5aba9c9f27557694e6d882b0d4388bec48533d015fb4。训练/评估源预冻结，本轮无训练启动失败；之后添加审计/绘图/报告脚本仅处理已保存结果。

已执行入口（均相对于本实验source，勿在封存目录重跑）：world_model_env.sh下prepare.py→train.py→audit_training.py→evaluate.py；env.sh下verify_evaluation.py→render.py→write_report.py→seal.py。C在六运行完成之后一次评价，没有模型选择。预测输出先保存，再载入未来目标评分。D模型前向0、新数据0、多步0、追加训练0。三种子结果均保留，代表固定seed0。

后续：额外收益条件通过，但训练前另外冻结的全种子完整恢复/新独立验收提案门槛未通过。因此本轮不自动提供可执行独立验收计划，不改选全对seed2，不采集。NEXT_STEP_DECISION.json明确区分这项保守提案门槛与原指南验收，不把它声称为用户另加要求。后续任务可基于这些完整证据另行决定。原指南整体尚未完成。

关键证据：configs/protocol.json、configs/shared_fullbatch_schedule.npz、configs/model_selection_frozen.json、results/training_audit.json、evaluation_audit.json、all_metrics.csv、all_call_predictions.csv、same_seed_differences.csv、error_partition.json、root_contrasts.json、single_replacement_diagnostic.json、immutable_before/after.json、delivery_manifest.json。外部回执：/home/robot/Documents/Codex/2026-10-13/task-4/group_weight_delivery_receipt.json。
'''
(E/'HANDOFF.md').write_text(handoff)
fig=read(E/'results/figure_manifest.json');fig['visually_inspected']=True;fig['inspection']='All five plots checked: Chinese axes/legends, seed markers, units and limitations; errors and transient regression shown.';save(E/'results/figure_manifest.json',fig)
print(json.dumps({'report_characters':len(report),'stable_weight_advantage':decision['stable_weight_advantage'],'complete_restoration_all3seeds':decision['complete_restoration_all3seeds']},ensure_ascii=False))
