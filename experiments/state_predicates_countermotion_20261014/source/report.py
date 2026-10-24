from pathlib import Path
import json,base64,html,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_countermotion_20261014';B=P/'data/predicate_countermotion_replay_20261014';T=P/'experiments/state_predicates_temporal_20261014'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(head,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table>'
def img(name,cap):return '<figure><img alt="'+html.escape(cap)+'" src="data:image/png;base64,'+base64.b64encode((E/'figures'/name).read_bytes()).decode()+'"><figcaption>'+html.escape(cap)+'</figcaption></figure>'
def main():
 D=read(E/'results/evaluation.json');A=read(E/'results/existing_held_motion_audit.json');M=read(B/'data_manifest.json');meta=read(E/'data/metadata.json');prior=read(T/'results/evaluation.json');font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(11.7,4.8));names=['蓝／关','蓝／开','粉／关','粉／开','红／关','红／开']
 for r,c in zip(D['models'],['#397fb5','#d77e29','#258b72']):ax.plot(range(6),[max(q['held_probabilities_RGB'])for q in r['root_details']],marker='o',lw=2,color=c,label=f'冻结模型：种子{r["seed"]}')
 ax.axhline(.5,color='#bc4b4b',ls='--',label='原始held=true判界0.5');ax.set(xticks=range(6),xticklabels=names,yscale='log',ylim=(1e-5,1),ylabel='三物体中的最大held概率（对数轴，越低越好）',xlabel='既有C根：目标颜色／抽屉状态',title='空爪上移后仍判断未持有：三个冻结模型均无误报或UNKNOWN');ax.grid(axis='y',alpha=.15);ax.legend(ncol=2,fontsize=10,loc='upper center',bbox_to_anchor=(.5,-.19));fig.tight_layout();fig.savefig(E/'figures/frozen_counterexample_results.png',dpi=160,bbox_inches='tight');plt.close(fig)
 selected=['C_predicate_gap_red_closed','C_predicate_gap_red_open'];fig,axs=plt.subplots(4,4,figsize=(12.4,11.2))
 for ri,rid in enumerate(selected):
  r=next(q for q in meta if q['root_id']==rid)
  for ci,pp in enumerate(r['public_frames']):
   with np.load(pp)as z:
    for si,cam in enumerate(['static','gripper']):
     ax=axs[2*ri+si,ci];ax.imshow(z['rgb_'+cam],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
     if si==0:ax.set_title(['t−2.10秒','t−1.40秒','t−0.70秒','当前t'][ci])
     if ci==0:ax.set_ylabel(('红块／抽屉关'if ri==0 else'红块／抽屉开')+'\n'+('静态相机'if cam=='static'else'腕部相机'),fontsize=12)
 fig.suptitle('实际恢复的因果输入：机器人抬高后空爪闭合，三个物体仍在支撑面',fontsize=15);fig.tight_layout();fig.savefig(E/'figures/countermotion_frames.png',dpi=150,bbox_inches='tight');plt.close(fig)
 shutil.copyfile(T/'figures/remaining_old_C.png',E/'figures/remaining_old_C.png')
 # A concrete next development challenge is specified without replaying it or scoring it.
 chosen=[];used=set()
 for colour in ['red','blue','pink']:
  candidates=sorted([q for q in A['candidates']if q['condition']=='held_down'and q['colour']==colour],key=lambda q:(q['root_id'],q['trajectory'],q['anchor']))
  for q in candidates:
   if q['root_id']in used:continue
   chosen.append(q);used.add(q['root_id'])
   if sum(x['colour']==colour for x in chosen)==2:break
 save(E/'results/next_step_candidates.json',{'status':'proposal_only_not_replayed_or_evaluated','selection':'FirsttwoexistingdistinctCrootspercolour,lexicographictrajectoryandearliestqualifyinghelddownanchor;6rootsmax, no model predictions used.','purpose':'TesttrueheldwhileTCPandobjectmovedown, breakingupward-plus-aperturealternative withouttrainingnewmodel.','candidates':chosen})
 modelrows=[]
 for r in D['models']:
  held=r['metrics']['held'];io=r['metrics']['in_drawer'];op=r['metrics']['open_drawer'];modelrows.append([r['seed'],r['thresholds'][0],f"{r['target_held_FP_roots']}/6",f"{r['any_object_held_FP_roots']}/6",f"{held['TN']}/18",held['unknown'],f"{r['max_held_probability']:.8f}",f"{io['FP']+io['FN']}/18",f"{op['FP']+op['FN']}/6"])
 width=[r['gripper_width_mm']for r in meta];anchors=sum(len(r['verification']['saved_frame_checks'])for r in M['trajectories']);frames=sum(r['frames']for r in M['trajectories']);sources=[[r['root_id'],r['anchor_frame'],str(r['history_frame_indices']),f"{r['TCP_z_change_mm']:.2f}",f"{r['gripper_width_mm']:.4f}"]for r in meta]
 parts=['<h1>冻结反例诊断：空爪上移没有触发held误报</h1><p class="muted">本机 · 2026-10-14 · 六条既有C轨迹原动作回放 · 三个既有长窗模型 · 单次推理，无训练、调阈值、新根或独立测试</p>',
 '<p class="lead"><b>本轮结论：</b>三个冻结模型都把六次空爪上移后的三个物体判断为已知“未持有”，每模型18/18个对象事实正确、0个UNKNOWN；最高held概率仅0.00005417。这推翻了“模型只要看到末端上移就报held”的简单解释。它没有证明模型已依靠物体跟随运动：空爪夹宽接近0、物体与手爪的位置不同，仍是可用线索。持物下移存在于既有C日志，但缺完整长窗；符合预定2.1秒定义的持物静止窗未找到。</p>',
 '<h2>1. 先冻结，再恢复与评估</h2>',table(['固定内容','结果','为何需要'],[['六条来源与目标时刻','直接使用上一轮已列出的六条near_closed_empty；五条锚点69，一条红／开锚点91。','在看预测前确定，不挑容易通过的时刻。'],['历史输入','[t−63,t−42,t−21,t]，30Hz，覆盖2.10秒；与长窗模型处理相同。','保持因果性和时间跨度，当前标签不变。'],['模型与阈值','上一轮long_history第600次更新的种子0/1/2；held阈值0.6/0.5/0.5，in/open均0.5。','无新训练、选模或校准；每模型只对冻结批次推理一次。'],['回放完整性',f'6条、{frames}帧、{frames-6}个原动作；已存{anchors}个双相机图像锚点RGB与掩码逐位一致。','从原根初始化回放所有原动作，未在中间注入GT复位。'],['数值与事件','可比较TCP/物体位置、机器人姿态/夹宽、第24帧完整场景差均为0；接触/支撑/held事件一致。','满足恢复前冻结的容差；缺失帧是本次实际重放保存的观测，不伪称原文件。'],['公共特征','6窗×3实体注册全部一致；当前锚点特征与原四帧缓存逐项一致。','没有通过改变当前画面、实体对应或标签制造对照效果。']]),
 '<p>恢复容差沿用此前冻结值：位置0.1mm、角度0.001rad、事件偏移≤1控制步、原RGB平均绝对差≤2、目标掩码IoU≥0.95。实际所有可比较检查精确匹配。旧日志未保存逐帧完整物体姿态角，未宣称验证该缺失字段。六条仍是原C开发根，不因回放成为新根；这些当前状态曾出现在旧C四帧开发材料中，因此本次也不是独立测试。</p>',
 table(['原根','固定锚点t','四个输入帧索引','TCP上移mm','当前空爪夹宽mm'],sources),
 '<h2>2. 误报、置信度与回归结果</h2>',img('frozen_counterexample_results.png','横轴是六个既有C根的目标颜色与抽屉状态；每个点取该根三个物体held概率的最大值，三条颜色线分别为种子0/1/2。纵轴为对数概率，红虚线是原始0.5判界。概率全部远低于判界；原冻结校准同样输出已知假，结果不是通过UNKNOWN掩盖误报。'),table(['种子','held阈值','目标held误报根','任一物体误报根','已知未持有','UNKNOWN','最高p(held)','in错误','open错误'],modelrows),
 '<p>六个场景中的三个物体都未被持有，18个held事实均有有效标签且四个输入时刻可见。三个模型重复评估同六个场景，不计为18个独立场景；18个对象事实也不等于18个独立根。新反例中的in全是假，open为3真/3假，本轮只能确认这组in负例与open回归无错，不能从它推断in正例召回。</p>',
 '<h2>3. 实际图像与解释边界</h2>',img('countermotion_frames.png','展示红色目标的抽屉关／开两条既有根，每例上行为静态相机、下行为腕部相机。四列为完整的长窗模型输入。机器人在2.1秒内上移43.57～48.28mm，最终空爪闭合，物体仍受支撑。图片保留原颜色；它们不包含模型的接触真值输入。'),
 '<p>原时序对照中，54个窗口的末端上移/下移与held真假完全同向。本轮加入“末端上移但所有物体held为假”的真实既有反例，模型仍给出极低held概率，因此<b>单一运动方向不能解释全部输出</b>。但这组空爪夹宽约为'+f'{min(width):.4f}～{max(width):.4f}'+'mm；这些约±0.003mm的正负值是零夹宽附近的数值偏差。真正持物时夹宽不同，可见物体相对夹爪位置也不同。模型可能综合这些当前线索，而未真正利用物体与夹爪共同运动。本轮没有额外改输入、调整阈值或拟合新的解释器，不能把通过本例升级为一般关系识别已经证实。</p>',
 '<h2>4. 既有持物下移与静止资料审计</h2>',table(['资料/状态','找到什么','为什么尚未评估'],[['审计范围','32个既有C根，258条轨迹：240个核心技能片段、12条补充轨迹、6条已恢复抓取轨迹；76,779个记录时刻。','只读C资料；未重放额外轨迹或访问D轨迹内容。'],['持物下移','20根、63条轨迹有6,611个重叠合格窗口，覆盖红/蓝/粉。','所需四个时刻完整公共RGBD/掩码/本体输入数量为0；不能用有损视频或私有GT合成。'],['持物静止','没有满足完整2.1秒、TCP与物体均保持在起点1mm内、全程真实held的窗口。','“0”只对应预先冻结的长时静止定义，不代表没有更短停留或没有物理静止状态。'],['其他窗口选择','额外可直接评估窗口为0，已冻结空清单。','不根据模型结果放宽运动门槛、删掉不利样本或新增采集。']]),
 '<p>持物下移要求64个连续时刻均满足原held定义，TCP与物体同时下降至少20mm，物体相对TCP的最大漂移≤15mm。6,611是大量重叠窗口，不能当成6,611个独立样本。完整公共输入不足是当前实际阻塞，不是没有找到对应物理状态。</p>',
 '<h2>5. 综合已完成范围与仍未解决的状态</h2>',table(['范围','累计证据','边界'],[['导出与时间采样','已恢复既有遗漏；原0.10秒窗缺少明显抬升过程，2.10秒稀疏四帧在同容量、同预算比较中新C表现更稳定。','干预只作用于恢复窗口，旧973个训练窗口仍保留原四帧。'],['常见held与支撑负例','上一轮长窗3个种子在新C18窗全部正确且6个真正held均为已知；本轮空爪上移6根也全部已知为假。','不构成独立验收；同根、同当前状态的重复使用已明确。'],['open','上一轮9个模型旧C337窗与新C18窗均无原始错误；本轮冻结3模型的6窗也无错。','目前是开发场景内稳定表现。'],['in','上一轮长窗组仅种子0在旧C漏判1个红块in；本轮in负例均正确。','in正例回归仍有一个场景未解决。'],['held剩余错误','旧C仍有粉块held漏判，以及抬粉块时红块held误报，涉及2个不同场景。','旧C使用原短窗输入；本轮没有新观测可证明这些旧错误已修复。'],['语义反例缺口','持物下移有原日志但缺长窗公共输入；2.1秒持物静止缺符合定义的既有窗口。','尚不能证明对动作方向/阶段变化的held判断稳定。']]),
 img('remaining_old_C.png','沿用上一轮同一批冻结模型的残余旧C图：红块in漏判、粉块held漏判、红块held误报。当前任务未重新训练或调整这些结果。它们与本轮通过的空爪上移反例同时保留。'),
 '<h2>6. 明确下一步与停止点</h2><p>下一项有明确区分力：<b>从已找到的既有C持物下移轨迹中恢复少量公共长窗，再对同三个冻结模型单次评估。</b>这会让held为真但TCP方向为下，进一步检验模型是否依赖“上移＋夹宽”等组合线索，无需新根、新网络或训练。本轮按颜色各两根、原根与轨迹排序及最早合格锚点，列出了六个候选，保存在results/next_step_candidates.json；没有回放或评分它们。依本次范围，缺少完整输入的补充项只记录缺口，不新增采集。</p><p>长期静止目前没有同窗资料，不宜用短停留冒充。至此本轮任务完成：三个冻结模型通过了既定空爪上移开发反例，简单方向捷径受到反证；关系语义的一般性、残余旧C错误及独立测试仍未解决。未进行D验收、模型晋升、世界模型训练或规划实验。</p>',
 '<p class="muted">协议与源/模型哈希：configs/protocol.json；推理前输入冻结：configs/evaluation_inputs_frozen.json；完整预测：results/evaluation.json；既有资料审计：results/existing_held_motion_audit.json；核验：results/delivery_validation.json；文件校验：delivery_manifest.json。原始公共观测与私有标签分开保存在data/predicate_countermotion_replay_20261014。完整旧实验保持不变。</p>']
 css='body{font:16px/1.65 "Noto Sans CJK SC",sans-serif;background:#f3f6f8;color:#253745;margin:0}main{max-width:1220px;margin:auto;padding:40px;background:white}h1{font-size:30px}h2{font-size:23px;color:#245775;margin-top:38px}.lead{background:#edf7f4;border-left:4px solid #258b72;padding:18px}.muted{font-size:14px;color:#607586}table{border-collapse:collapse;width:100%;font-size:13px;margin:24px 0}th,td{border:1px solid #dce4ea;padding:9px;text-align:left;vertical-align:top}th{background:#edf3f7}tr:nth-child(even){background:#fbfcfd}figure{margin:25px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#607586}'
 (E/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>冻结空爪上移反例诊断</title><style>'+css+'</style><main>'+''.join(parts)+'</main></html>')
 save(E/'SUMMARY.json',{'status':'six_existing_C_counterexamples_replayed_and_frozen_evaluated','replay_roots':6,'replay_frames':frames,'old_image_anchors_bitwise_equal':anchors,'all_replays_exact_on_available_checks':True,'model_seeds':[0,1,2],'updates':600,'new_training_updates':0,'thresholds_unchanged':True,'any_object_held_FP_roots':[r['any_object_held_FP_roots']for r in D['models']],'held_unknown_atoms':[r['metrics']['held']['unknown']for r in D['models']],'max_held_probabilities':[r['max_held_probability']for r in D['models']],'development_roots_audited':32,'development_trajectories_audited':258,'held_down':{k:v for k,v in A['counts']['held_down'].items()if k!='examples'},'held_stationary':{k:v for k,v in A['counts']['held_stationary'].items()if k!='examples'},'simple_upward_only_explanation_contradicted':True,'general_held_relation_semantics_proven':False,'new_roots':0,'D_inference_or_independent_acceptance':False,'next_direction':'Recover small preselectedexistingChelddownlongwindows; samefrozenmodels,once-onlyevaluation. Not executed in this task.'})
 print('COUNTERMOTION_REPORT_READY',E/'report.html')
if __name__=='__main__':main()
