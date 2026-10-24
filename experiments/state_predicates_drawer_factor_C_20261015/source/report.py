from experiment import *
from evaluate import score,known,csvsave
import numpy as np
CN={'red':'红','blue':'蓝','pink':'粉'}
def vec(x):return '['+','.join('?'if v=='UNKNOWN'else str(int(v))for v in x)+']'
def main():
 q=frozen();r=read(E/'results/evaluation.json');m=read(B/'data_manifest.json');root=read(B/'roots_manifest.json');cons=read(E/'results/root_contrasts.json');rows=read(E/'results/call_details.json');matches=read(E/'results/pose_matching.json');integrity=read(E/'results/additional_pair_integrity.json');v=read(E/'results/data_validation.json');rep=read(E/'results/single_replacement_diagnostic.json');pubs={x['episode_id']:x for x in m['records']};baselines=[]
 for model in ['candidate_seed0','parallel_seed1','parallel_seed2','tree','copy_effect','persistence']:
  ss={}
  for drawer in ['closed','open']:
   rr=[x for x in rows if x['drawer']==drawer];ps=[np.load(read(pubs[x['episode_id']]['preaction_prediction_record_path'])['predictions_path'])[model][0]for x in rr];ss[drawer]={'calls':len(rr),'all7':sum(score(p,x['truth_after'],[True]*7)for p,x in zip(ps,rr)),'UNKNOWN':sum(int((~known(p)).sum())for p in ps)}
  baselines.append({'model':model,**ss})
 save(E/'results/fixed_comparators.json',baselines);csvsave(E/'results/fixed_comparators.csv',baselines)
 p=['# 抽屉开闭单因素开发诊断：6个新C根\n',
 '本轮完成预先冻结的6个新C根、12次已持另一物体时的抓取调用。六个根均满足开/闭配对姿态要求；真实调用全部拒绝，96步保持后七个事实均未改变。冻结候选在关闭侧6/6正确，打开侧0/6正确。五个根的两侧当前及真实后续读出全部正确，且均呈“关闭正确、打开错误”；第六根有读出错误，单独保留。\n',
 '**结论：值得优先开展一次只补训练条件组合的受控训练，但本轮没有训练。** 这些C根支持抽屉条件会因果性改变当前冻结系统的预测，并在五个读出无误的配对上定位到后继预测路径。它们不能证明补数据一定能修复，也不能证明旧D失败全部由抽屉造成。旧D仍为68/84、配对20/36，未通过；本轮既不是新的独立验收，也不是原指南全部完成。\n',
 '## 执行边界与冻结内容\n',
 '|元素|本轮实际设置|原因与限制|\n|---|---|---|',
 '|独立单位|6个全新C布局根；每根开/闭两个相关分支|12次调用不当作12个独立根；C仅开发归因|',
 '|颜色组合|已持蓝→抓红、已持粉→抓蓝、已持红→抓粉；各2根|保持原循环组合，不把反向颜色覆盖同时作为新因素|',
 '|唯一设计差异|复位时抽屉关0或开0.22米|相同种子、机器人、全部物体位姿和其余场景状态；不在准备中途瞬移|',
 '|准备|每侧相同合法lift-other技能；前后各96步保持|12次准备全部成功；原固定控制器未改|',
 '|配对顺序|每种颜色根0先关后开、根1先开后关|总计3根关先、3根开先；避免顺序与抽屉条件重合|',
 '|预测输入|双相机公开RGB、机器人状态；四帧间隔21步覆盖2.1秒；声明目标色及预算900/保持96|抽屉条件名、真值、拒绝结果、私有姿态均不输入模型|',
 '|冻结模型|预选aug132_seed0@1500，DINO、RGB注册、标准化、读出、0.5阈值、UNKNOWN规则全部不变|并行seed1/2仅记录，不能替换候选|',
 '|动作与结果|固定PhysicalOracle；原始拒绝0控制步，仍执行96真实保持步|不把拒绝当成未来模型输入；也不把重复帧伪装为真实保持|',
 '|样本排重|与539个既有布局及335份根来源元数据排重；新种子15193000–15193005|无替补根、无失败重试、无成功配额|',
 '|本机环境|Ubuntu24.04.5、RTX5080Laptop16GB、驱动595.91.07；原3.10执行器/3.11模型环境|单个持锁预测服务，物理控制与DINO/模型推理同步串行；未上传|\n',
 f'采集前冻结方案SHA256：`{sha(E/"configs/protocol.json")}`。候选SHA256：`{q["candidate_hash"]}`。完整根清单见[roots_manifest.json](../../data/predicate_drawer_factor_C_20261015/roots_manifest.json)，方案与阈值见[protocol.json](configs/protocol.json)。\n',
 '## 姿态匹配与数据完整性\n',
 '初始机器人及非抽屉场景坐标最大绝对差限值1e-5。计分前四个历史帧均要求：三个物体和机器人TCP位置差≤1毫米、四元数测得的姿态夹角≤0.5度、机械臂各关节差≤0.5度、夹爪宽度差≤1毫米、夹爪指令相同，其余场景关节/开关差≤1e-5；除抽屉开闭外的六事实一致，七事实标签均有效。未达匹配者预先规定为严格因果未覆盖，不能换根。\n',
 '|根|初始非抽屉最大差|四帧最大物体位置/姿态差|四帧最大TCP位置/姿态差|严格匹配|\n|---|---:|---:|---:|---|']
 for x in matches:p.append(f'|{x["root_id"]}|{x["initial_non_drawer_max_abs"]:g}|{x["max_object_position_mm"]:g} mm / {x["max_object_orientation_deg"]:g}°|{x["max_robot_TCP_position_mm"]:g} mm / {x["max_robot_TCP_orientation_deg"]:g}°|通过|')
 p+=['\n保存数值精度下上述差异均为零。补充核验还确认六对的**整个记录区间**机器人观测、三物体位置/四元数、控制命令序列均完全相同；双相机RGB则随抽屉开闭确实改变。零差异不代表图像相同，而说明没有同时改机器人/物体轨迹。完整逐帧数值见[姿态匹配](results/pose_matching.json)与[整段轨迹和图像差异](results/additional_pair_integrity.json)。\n',
 f'12个提前保存的预测均通过“预测完成→动作开始→后续读出”时序核验；24个窗口均使用真实2.1秒历史。共保存并校验{v["public_frames_hashed"]}个公开帧文件、{v["public_frame_bytes"]:,}字节；84个事实在前后两端均有效；12次拒绝后的96步保持完整，无未覆盖调用。原AB六例预检的原始特征/标准化逐元素相同，离散输出一致，概率最大差3.50e-5。见[采集验证](results/data_validation.json)。\n',
 '## 逐根结果与原因\n',
 '下表p为“请求目标已经被持有”的预测概率，实际在全部12次调用后都为0。全七事实正确要求每个事实均有效、已知且正确；UNKNOWN算错。打开侧均同时把原持物和目标持物判为阳性，触发既定多持物冲突规则，两项成为UNKNOWN。\n',
 '|根（持有→目标）|关闭p / 全7|打开p / 全7|前后读出|开闭是否改变预测及解释|\n|---|---|---|---|---|']
 for c in cons:
  a,b=c['sides']['closed'],c['sides']['open'];extra=[['红','蓝','粉'][j-3]for j in range(3,6)if b['prediction'][j]==1];why='改变：目标由假转为阳性，与原持物冲突。'+(('另误报'+','.join(extra)+'入屉。')if extra else'没有额外入屉阳性。');clean=c['both_sides_both_observed_readouts_correct'];why+= '可比较后继预测路径。'if clean else'闭侧当前误报红入屉；后续误报持红并与持粉冲突，不能作为读出无误的后继归因根。'
  p.append(f'|{c["root_id"]}（{CN[c["held_color"]]}→{CN[c["target_color"]]}）|{a["target_held_probability"]:.6f} / 正确|{b["target_held_probability"]:.6f} / 错误|'+('两侧两端均正确'if clean else'闭侧两端有错，开侧两端正确')+'|'+why+'|')
 p+=['\n五个无读出错误的根覆盖三个颜色组合，全部关闭正确、打开错误；没有反方向或两侧都正确/都错误的干净配对。第六根也出现相同预测转变，但不能把它与前五根合并声称六个都纯属转移错误。\n',
 '当前读出11/12正确，真实后续读出11/12正确，唯一有读出问题的是`C_drawer_factor_blue_0__b0`。打开侧除6次持物冲突外，4次还误报原持物在抽屉内。全部12次真实结果为`refused_nochange`，七个真实事实不变。\n',
 '![开闭概率图](figures/paired_probabilities.png)\n',
 '图中横轴每组是同一布局根，灰线连接该根的闭/开条件；纵轴为冻结模型事实概率。蓝点=抽屉关闭，橙点=抽屉打开，与物体颜色无关。虚线0.5为既定阈值，绿点线是真实结果：上图目标未持有=0，下图原持物仍持有=1。图的含义是：打开抽屉使目标持有预测升到约0.999，而原持物预测仍大于0.5，造成冲突。\n',
 '完整七事实的顺序固定为`[持红,持蓝,持粉,红入屉,蓝入屉,粉入屉,屉开]`；`?`为UNKNOWN，0/1为假/真。下表保留每侧当前读出、预测、真实结果和后续读出；前后真值相同。\n',
 '|根/条件|当前读出|未来预测|真实当前=真实后续|后续读出|\n|---|---|---|---|---|']
 for x in rows:p.append(f'|{x["episode_id"]}（'+('闭'if x['drawer']=='closed'else'开')+f'）|`{vec(x["readout_before"])}`|`{vec(x["prediction"])}`|`{vec(x["truth_after"])}`|`{vec(x["readout_after"])}`|')
 p+=['\n各概率原值及图像见[七事实概率CSV](results/all7_probabilities.csv)、[逐调用JSON](results/call_details.json)。每张配对图左侧展示调用前/96步后静态与腕部相机，右侧展示真值、当前/后续读出及提前生成的预测概率；红色预测文字=错误或UNKNOWN，蓝/橙标题=闭/开。\n']
 for c in cons:p.append(f'- [{c["root_id"]} 双相机及七事实图](figures/{c["root_id"]}.png)')
 p+=['\n## 第一个有影响的错误与单项替换\n',
 f'实际冻结执行顺序的第一个预测错误是`{rep["first_error_in_actual_fixed_collection_order"]}`。当前和真实后续读出都正确；当前已持蓝、红块在桌面。预测却给持红p={rep["target_probability_predicted"]:.6f}、持蓝p={rep["other_probability_predicted"]:.6f}，两者冲突为UNKNOWN。关闭配对的目标持有p约0.001且全部事实正确。\n',
 f'离线只把“预测后继潜变量”换成已经保存的“真实后续观测编码”，使用同一冻结读出头，持红p回到{rep["target_probability_actual_next_reference"]:.8f}，七事实正确；没有新推理或修改模型。这一单项替换在12次中纠正全部6个原预测错误，但也在原本预测正确的闭侧blue_0引入其真实后续读出的错误，因此仅为11/12参考正确。它把首错定位到后继预测路径，不能进一步唯一分辨表征/全局上下文与转移映射机制，也不是可在线使用的修复。\n',
 f'[首错两侧真实视频](videos/{rep["first_error_in_actual_fixed_collection_order"].rsplit("__b",1)[0]}_closed_open.mp4)按相对调用时刻对齐，展示真实2.1秒前史和3.2秒拒绝后保持，末帧停留3秒。1280×800、30fps、250帧全部解码验证；无插值、无瞬移。详见[替换诊断](results/single_replacement_diagnostic.json)。\n',
 '## 固定参考与后续选择\n',
 '下面只汇总本来就保存的固定对照，不重新选择代表模型。Persistence复制当前读出；copy_effect是原冻结AB效果复制规则；tree是原AB132浅树，均未用新C数据更新。\n',
 '|固定预测器|关闭全7正确|打开全7正确|两侧UNKNOWN事实数|\n|---|---:|---:|---:|']
 for b in baselines:p.append(f'|{b["model"]}|{b["closed"]["all7"]}/6|{b["open"]["all7"]}/6|{b["closed"]["UNKNOWN"]+b["open"]["UNKNOWN"]}|')
 p+=['\n建议下一步只研究AB训练条件覆盖：原有“已持其他物体→抓目标”的循环颜色组合在训练中只有关闭条件，可以在全新AB根上补相应打开条件，并保留同根关闭对照。任何训练开始前应另行冻结配额、失败保留规则、训练预算、种子、选择规则和比较方案；原AB132及固定候选是对照。本轮6个C根保持开发诊断用途，不能转入训练或作为独立验收。\n',
 '此建议有当前五个干净配对的方向一致证据，优先级高于同时改潜变量范围、网络结构或损失权重。不能据此承诺补数据有效；若以后只补条件覆盖仍失败，再另立单因素诊断。闭侧blue_0的读出错误作为独立限制保留，不在这轮同时修读出。反向颜色组合仍是未覆盖问题，不与本轮抽屉归因混为一项。\n',
 '本轮没有启动上述训练、没有新D或多步闭环，也未上传文件。未来若模型改变，需要重新冻结后用全部旧根之外的新D测试；按此前提案覆盖持物→目标六个异色有序组合及抽屉开/闭，不能重用旧D或本轮C来宣称新的独立通过。\n',
 '## 文件与复现\n',
 '源码在`source/`，冻结配置在`configs/`，新公开观测/私有计分审计分置于`../../data/predicate_drawer_factor_C_20261015/episodes/*/public`和`_private`；特征、归一化输入、已保存预测分别在`data/features`、`data/public`、`data/predictions`。表格/验证在`results/`，图在`figures/`，视频在`videos/`，运行日志在`logs/`。依赖缓存复用，无新模型下载。\n',
 '执行命令记录于[HANDOFF.md](HANDOFF.md)。`collect.py`不能重复执行已封存采集；需要新的授权才能建新版本。旧D及训练全量文件在执行前后按原清单SHA256和只读权限验证，结果见`results/immutable_before.json`和`results/immutable_after.json`。本轮冻结候选/输入/控制器源码均通过原哈希核验；无采集修正，无参数更新。\n']
 (E/'REPORT.md').write_text('\n'.join(p))
 proposal={'status':'recommendation_only_not_executed','decision':'Worth prioritizing a bounded condition-combination-only training study; not simultaneous architecture/loss changes.','evidence':{'strictly_matched_roots':6,'matched_readout_correct_closed_right_open_wrong':5,'colours':3,'actual_all_refused_nochange':12,'closed_candidate_all7':6,'open_candidate_all7':0},'candidate_next_factor':'Add only the missing open-drawer held-other lift condition on new AB roots with matched closed controls, keeping current colour cycles and fixed input/controller/reader/architecture/loss.','before_execution':'Separately authorize and freeze collection quota, retention rules, training budget/seeds/model selection and comparison; preserve originalAB132 model baseline.','current_C_roots':'diagnostic only; do not train or claim independent acceptance on them','no_automatic_training':True,'reader_limitation':'blue_0closed observed-readout error retained; do not simultaneously change reader','reverse_colour_combinations':'separate coverage issue, not added to this attribution trial','future_D':'New, deduplicated roots only after new version frozen; all6offdiagonal held-to-target combinations crossed with draweropen/closed. OldD68/84 and20/36 remain failed unchanged.','this_round_updates':0,'this_round_new_D':0};save(E/'NEXT_STEP_PROPOSAL.json',proposal)
 handoff=f'''# 本轮交接\n\n本轮6个新C根/12次调用完成；无训练、新D、旧D重算、上传或多步闭环。六根严格匹配，关闭6/6、打开0/6；五根两端读出均正确且关闭对/打开错。blue_0闭侧当前/后续读出有错，不能作为干净转移归因。建议下一轮仅补AB条件组合，必须另行授权冻结，不在此轮执行。\n\n报告：{E}/REPORT.md\n数据：{B}\n冻结方案SHA256：{sha(E/'configs/protocol.json')}\n冻结候选SHA256：{q['candidate_hash']}\n\n执行顺序（已运行；不要在封存目录重跑写入）：\n```bash\nsource {P}/env.sh\npython {E}/source/prepare.py\nsource {P}/world_model_env.sh\npython {E}/source/worker.py --preflight\nsource {P}/env.sh\npython {E}/source/collect.py --phase pilot\npython {E}/source/validate_data.py --phase pilot\npython {E}/source/collect.py --phase remaining\npython {E}/source/validate_data.py --phase full\npython {E}/source/evaluate.py\npython {E}/source/render.py\npython {E}/source/posthoc_audit.py\npython {E}/source/video.py\npython {E}/source/report.py\npython {E}/source/seal.py\n```\n\n技术证据：results/data_validation.json，pose_matching.json，additional_pair_integrity.json，call_details.json，root_contrasts.json，single_replacement_diagnostic.json。新根排除539旧布局，335来源；六个新seed15193000..15193005。12准备成功、12refused_nochange、12提前预测、24因果窗口、6708公开帧450050313字节。所有开闭对的整段控制指令/机器人及物体姿态完全相同；真正变化为抽屉及RGB。阈值和匹配规则采集前冻结，未改过源码、参数或根以追补成功。\n\n首错red_0open的持目标p=.999003、持原物p=.972412导致两项UNKNOWN；单项换成已保存真实后继编码，同头正确。参考替换纠正六次打开错误，且保留blue_0closed读出错误：总参考11/12，不能称修复模型。\n\n保留原D报告68/84、paired20/36未通过。文件封存前后全量校验旧D实验/数据及TRAIN清单、只读权限与哈希；结果写results/immutable_after.json。本轮与下一步提案的边界见NEXT_STEP_PROPOSAL.json。\n\n没有剩余执行阻塞；停止在本轮有限诊断授权边界。新的条件训练与独立D尚未开始，原完整指南也未由本轮达成。\n'''
 (E/'HANDOFF.md').write_text(handoff);print('REPORT_AND_PROPOSAL_WRITTEN',flush=True)
if __name__=='__main__':main()
