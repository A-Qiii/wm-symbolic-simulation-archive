from pathlib import Path
import json, gzip, hashlib, shutil, base64, html, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

P=Path('/home/robot/research/projects/wm_symbolic')
E=P/'experiments/state_predicates_held_down_20261014'
B=P/'data/predicate_held_down_replay_20261014'
T=P/'experiments/state_predicates_temporal_20261014'
C=P/'experiments/state_predicates_countermotion_20261014'
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(head,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table>'
def img(name,cap):return '<figure><img alt="'+html.escape(cap)+'" src="data:image/png;base64,'+base64.b64encode((E/'figures'/name).read_bytes()).decode()+'"><figcaption>'+html.escape(cap)+'</figcaption></figure>'
def main():
 D=read(E/'results/evaluation.json');meta=read(E/'data/metadata.json');M=read(B/'data_manifest.json');protocol=read(E/'configs/protocol.json');rule=read(E/'configs/integration_rule.json');prior=read(T/'results/evaluation.json');counter=read(C/'results/evaluation.json')
 assert D['reference_meets_predeclared_limited_candidate_rule'] and rule['prospective_reference_seed']==1
 # Packaging only: byte-for-byte copies, no weight loading or inference.
 candidate=E/'candidate';candidate.mkdir(exist_ok=True)
 assets={'reader_seed1.pt':Path(rule['reference_model']['checkpoint']),'model.py':E/'source/model.py','metrics.py':E/'source/metrics.py','normalization.json':P/'experiments/state_predicates_20261014/data/normalization.json','vision_cache.py':P/'src/wm_symbolic/vision_cache.py'}
 assetrows=[]
 for name,src in assets.items():
  dst=candidate/name
  if dst.exists():assert sha(dst)==sha(src)
  else:shutil.copyfile(src,dst)
  dst.chmod(0o444);assetrows.append({'filename':name,'source':str(src),'sha256':sha(dst)})
 contract={'status':'research_candidate_pending_independent_state_interface_acceptance','architecture':'Frozen DINOv2-small visible patch/ROI features, shared object MLP, object and scene GRUs, held/in/open heads; 675395 trainable reader parameters. Not the full guide ObjectTransformer/RSSM architecture.','history':{'fps':30,'frames':4,'past_offsets':[63,42,21,0],'span_seconds':2.1,'current_label_at_offset':0},'public_inputs':['static RGB','gripper RGB','visible instance masks and ROI geometry','15D public robot proprioception'],'not_used_as_model_inputs':['private contact/pose/labels','future frames','action phase or skill name','depth (stored only)'],'tensor_shapes_without_batch':{'objects':[4,3,780],'context':[4,3072],'robot':[4,15]},'object_order':['red','blue','pink'],'output_order':['held(red)','held(blue)','held(pink)','in(red,drawer)','in(blue,drawer)','in(pink,drawer)','open(drawer)'],'thresholds_held_in_open':[0.5,0.5,0.5],'decoding':'Existing model.py predicate_api and metrics.py known: p>=tau is true, p<=1-tau is false, otherwise UNKNOWN; simultaneous known-true held entities become UNKNOWN. At tau=0.5 ordinary confidence rejection is absent.','unverified_runtime_cases':['less than 2.1 seconds of history; no padding rule has been accepted','occluded/missing entity and registry reacquisition','full closed-loop consumption and completion false-confirmation','uniform 2.1 second sampling for the historical in/open regression states'],'external_cached_backbone':{'path':'/home/robot/research/models/facebook--dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056','sha256':read(E/'results/cache_audit.json')['DINO_weights_sha256']},'preprocessing_source':str(E/'source/evaluate.py'),'normalization':'Unchanged original training normalization; copied with SHA-256','limitations':'Seed1 is selected on C development evidence. A successful development challenge is not independent acceptance or probability calibration proof. No runtime deployment is performed.'}
 cp=candidate/'INPUT_CONTRACT.json'
 if cp.exists():assert read(cp)==contract
 else:save(cp,contract)
 cp.chmod(0o444)
 manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':contract['status'],'prospective_selection_rule':str(E/'configs/integration_rule.json'),'selection_rule_sha256':sha(E/'configs/integration_rule.json'),'model':rule['reference_model'],'assets':assetrows,'input_contract_sha256':sha(cp),'evaluation_sha256':sha(E/'results/evaluation.json'),'new_training_updates':0,'threshold_changes':False,'deployment_performed':False,'independent_acceptance_performed':False}
 cm=candidate/'candidate_manifest.json'
 if cm.exists():assert read(cm)['assets']==assetrows
 else:save(cm,manifest)
 cm.chmod(0o444)
 font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
 plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 (E/'figures').mkdir(exist_ok=True)
 traces={r['root_id']:json.load(gzip.open(B/'episodes'/r['episode_id']/'target_trace_private.json.gz','rt'))for r in meta}
 motionrows=[];fig,axs=plt.subplots(2,3,figsize=(13,7.8),sharex=True,sharey=True)
 for ax,r in zip(axs.flat,meta):
  t=traces[r['root_id']];ids=r['history_frame_indices'];z=np.array([q['robot_obs'][2]for q in t[ids[0]:ids[-1]+1]])*1000;z-=z[0];xx=np.arange(64)/30-2.1
  ax.plot(xx,z,color='#397fb5',lw=2,label='公共TCP高度');ax.scatter([-2.1,-1.4,-.7,0],z[[0,21,42,63]],color='#d77e29',s=35,zorder=3,label='实际4个输入时刻');ax.axhline(0,color='#8b969e',lw=.8,ls='--');ax.set_title(r['root_id']+' / '+{'red':'红块','blue':'蓝块','pink':'粉块'}[r['target_color']]);ax.set(xticks=[-2.1,-1.4,-.7,0],ylim=(-26,12));ax.grid(alpha=.12)
  motionrows.append({'root_id':r['root_id'],'net_TCP_z_mm':float(z[-1]),'max_rise_above_window_start_mm':float(z.max()),'four_input_z_relative_mm':z[[0,21,42,63]].tolist(),'input_action_phases':[t[i]['action'].get('phase')for i in ids],'monotone_downward':bool(np.all(np.diff(z)<=0))})
 fig.supxlabel('相对当前时刻的时间（秒；0为当前标签时刻）');fig.supylabel('TCP相对窗口起点的高度变化（mm）');fig.suptitle('实际窗口是净下移的持物运输：先升高约6.5～9.5mm，再下降',fontsize=15);handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,-.04));fig.tight_layout(rect=(.02,.02,1,.96));fig.savefig(E/'figures/held_down_motion.png',dpi=160,bbox_inches='tight');plt.close(fig)
 save(E/'results/public_motion_audit.json',{'source':'public robot_obs; private action phases used only for report interpretation','windows':motionrows,'all_monotone_downward':False})
 fig,axs=plt.subplots(6,4,figsize=(11.6,15.4))
 for ri,r in enumerate([meta[0],meta[2],meta[4]]):
  for ci,pp in enumerate(r['public_frames']):
   with np.load(pp)as z:
    for si,cam in enumerate(['static','gripper']):
     ax=axs[2*ri+si,ci];ax.imshow(z['rgb_'+cam],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
     if si==0:ax.set_title(['t−2.10秒','t−1.40秒','t−0.70秒','当前t'][ci])
     if ci==0:ax.set_ylabel(r['root_id']+'\n'+{'red':'红块','blue':'蓝块','pink':'粉块'}[r['target_color']]+' / '+('静态相机'if si==0 else'腕部相机'))
 fig.suptitle('恢复后的实际四帧输入：三种颜色的持物运输各展示一根',fontsize=15);fig.tight_layout(rect=(0,0,1,.975));fig.savefig(E/'figures/held_down_frames.png',dpi=140,bbox_inches='tight');plt.close(fig)
 shutil.copyfile(T/'figures/remaining_old_C.png',E/'figures/remaining_old_C.png')
 comparison=[]
 for p in prior['paired_fixed600']:
  comparison.append([p['seed'],p['current']['FP']+p['current']['FN'],p['short']['FP']+p['short']['FN'],p['long']['FP']+p['long']['FN'],f"{p['current']['known_true_positive']}/6",f"{p['short']['known_true_positive']}/6",f"{p['long']['known_true_positive']}/6"])
 rows=[]
 for r in meta:
  probs=[m['root_details'][meta.index(r)]['target_probability']for m in D['models']]
  rows.append([r['root_id'],r['target_color'],r['anchor_frame'],str(r['history_frame_indices']),f"{r['motion']['TCP_z_change_mm']:.2f}",f"{r['motion']['object_z_change_mm']:.2f}",f"{r['motion']['relative_drift_mm']:.3f}",*[f'{p:.6f}'for p in probs]])
 results=[]
 for r in D['models']:
  results.append([r['seed'],str(r['thresholds']),f"{r['target_known_TP']}/6",r['target_UNKNOWN'],*[f"{r['metrics'][g]['FP']+r['metrics'][g]['FN']}/{r['metrics'][g]['n']}"for g in ['held','in_drawer','open_drawer']]])
 remaining=read(T/'results/remaining_historical_errors.json');oldrows=[[r['metadata']['root_id'],r['seed'],r['predicate']+'('+r['target_colour']+')',r['truth'],f"{r['p']:.6f}",'漏判'if r['truth']else'误报']for r in remaining]
 total_frames=sum(s['target_recorded_frames']for s in M['episodes']);imagechecks=sum(len(s['verification']['image_checks'])for s in M['episodes']);counts={k:sum(s[k]for s in M['episodes'])for k in ['prefix_actions','setup_actions','settle_actions','target_actions']}
 parts=['<h1>当前状态读出：时序与反向运动诊断汇总</h1><p class="muted">本机 · 2026-10-14 · 本轮只恢复先前固定的六条C开发轨迹，冻结模型单次推理 · 无新根、训练或阈值调整</p>',
 '<p class="lead"><b>可以冻结一个有限研究候选：</b>推理前指定的长窗种子1通过全部六个持物净下移窗口，held、in、open均无错且无UNKNOWN；另外两个种子同样通过。结合此前空爪上移反例，“只看上移方向”的解释已不成立。当前证据支持进入独立状态接口验收，尚未证明一般持有关系、可靠闭环或指南整体完成。</p>',
 '<h2>1. 本轮实际恢复与评价</h2>',table(['核验对象','实际结果','意义与边界'],[
 ['来源','C_core1200_00～05，共6个既有C开发根，红/蓝/粉各2根。','这些根曾用于开发；中间窗口首次恢复，仍不变成新根或独立测试。没有D或历史测试来源。'],
 ['恢复路径',f"24步原稳定动作/根、原setup共{counts['setup_actions']}步、原前缀{counts['prefix_actions']}步、原目标{counts['target_actions']}步，共{sum(counts.values())}步。",'从原种子/原初始布局开始连续执行原动作；没有中间GT状态复位。'],
 ['保存与数值比对',f'{total_frames}个目标公共帧；前缀+目标共5017个逐动作记录与原robot/scene/完整物体姿态精确一致。','setup和稳定动作没有原逐步记录，通过完成后的原根完整状态与四幅根观测核验；不夸称6709步全部有逐步原记录。'],
 ['视觉与物理事件',f'{imagechecks}项原双相机图像检查，RGB及掩码逐位一致；所有可比数值最大差0、held/双侧接触/外部支撑事件无不一致。','原先缺失的中间图像是此次真实回放记录；不能声称它们与不存在的原文件逐位比较过。'],
 ['固定时间窗','[t−63,t−42,t−21,t]，30Hz，跨度2.10秒，当前t取原当前物理标签。','每窗64个连续时刻均真实held；四个输入时刻目标及其余两物体均可见。模型仅使用过去与当前公共观测。'],
 ['固定特征','原DINOv2-small与归一化不变，历史公共特征探针逐项差0，6窗×3实体身份检查通过。','当前中间锚点原来没有完整四帧缓存；不宣称和不存在的同锚点旧特征做过比较。'],
 ['固定评估','种子0/1/2的第600次更新，held阈值0.6/0.5/0.5，in/open均0.5；每模型一个冻结批次。','选种子1的规则先于当前预测冻结；本轮新训练更新数为0，未按结果换种子。']]),
 table(['根','目标','t','实际输入索引','TCP净Δz mm','物体净Δz mm','相对漂移mm','种子0 p(held)','种子1 p(held)','种子2 p(held)'],rows),
 table(['种子','held/in/open阈值','目标已知held正确','目标UNKNOWN','全部held错误','全部in错误','open错误'],results),
 '<p>每模型18个held事实为6真/12假，18个in事实为1真/17假，6个open事实全部为真。三个模型复用同六根，不能当作18个独立场景；同根多个对象也不独立。0错误覆盖当前这组开发状态，不能据此给出总体错误率保证。接近1的模型概率也不是经验可靠率。</p>',
 '<h2>2. 窗口中的元素与真实运动</h2>',img('held_down_motion.png','每个面板是一条固定C根。蓝线为64帧公共本体观测中的TCP高度，橙点是模型实际使用的四个时刻；横轴为距当前标签时刻的秒数，纵轴为相对窗口起点的高度变化mm。六例先短暂升高，再净下降20.07～22.80mm，还包含横向运输。它们反驳“held必须伴随净上移”，但不能替代单调下移或全程静止测试。'),
 img('held_down_frames.png','三组分别展示红、蓝、粉目标，每组上行为静态相机、下行为腕部相机；四列为t−2.10、t−1.40、t−0.70秒及当前t。画面保留原颜色，没有叠加私有接触/坐标。物体在抓持运输中位置改变，两相机提供不同遮挡角度；公共可见掩码用于对象特征抽取。'),
 '<h2>3. 把时序对照与两组运动反例放在一起</h2>',table(['种子','仅当前：新C held错误/18','0.10秒：错误/18','2.10秒：错误/18','仅当前：已知held正例','0.10秒：已知held正例','2.10秒：已知held正例'],comparison),
 '<p>上表引用此前已完成的固定600次更新对照，没有重训或重算推理。三个条件容量、预算、标签与批次流相同；时间处理只改36个新训练窗口与18个新C窗口，历史973个训练窗口及历史C保持原采样。因此它证明长窗在这批恢复窗口上更稳定，不能声称整个历史数据已统一为2.1秒。</p>',
 table(['运动/关系条件','来源与根数','三个冻结长窗种子的结果','检验意义'],[
 ['持物上移，held真','此前新C，6根','每种子6/6已知真','较长窗包含实际物体跟随位移；旧短窗位移很小。'],
 ['向下接近且仍受支撑，held假','同一批新C的6根','硬负例与普通负例均正确','接近、闭爪或双侧接触不能直接等同已持有。'],
 ['空爪上移，held假','前一轮既有C补充根，6根','每种子6/6目标已知假，三物体均无误报','上移不是报held的充分条件。'],
 ['持物净下移，held真','本轮既有C核心根，6根','每种子6/6目标已知真，其他事实无错','净上移不是报held的必要条件。'],
 ['完整2.1秒持物静止','既有C审计未找到合格窗','没有评估','不能用短暂停顿或降低静止定义补齐。'],
 ['单调下移、重新遮挡后显现等','本轮未覆盖','没有评估','保留给冻结后的独立接口验收。']]),
 '<p>前两行复用同六根，上述运动条件合计18个不同C开发根，并非24个独立根。两组反例共同限制简单方向解释，<b>没有唯一确定模型采用了物体—夹爪共同运动机制</b>：夹宽、相对位置、外观与阶段上下文仍可能组合起作用。本轮不再增加解释器或逐例追补。</p>',
 '<h2>4. 已知错误是否集中到简单原因</h2>',table(['既有根','种子','事实','真值','p(true)','原始错误'],oldrows),
 '<p>历史C337窗中，长窗种子0有2个held错误及1个in错误，种子2有1个held错误；四个“模型×事实”错误集中在3个场景，种子1在同批历史C无原始错误。它们仍使用原历史短窗，不能拿本轮其他场景通过来宣称修复。错误既有漏判也有误报，涉及不同对象与状态，目前没有证据把它们都归为一个简单原因，也没有证据据此认定重放、实体对应或标签管道存在共同故障。上下文与种子敏感性是一种描述，尚不是已证实的因果解释。</p>',
 img('remaining_old_C.png','原样保留此前的三个残余旧C场景：红块in漏判、粉块held漏判、抬粉块时红块held误报。该图来自上一轮固定模型结果，本轮未重新推理。图中场景和事实标签用于解释错误，私有评分信息未进入模型输入。'),
 '<h2>5. 冻结候选、下一项验收与停止点</h2>',table(['项目','明确状态/要求','原因'],[
 ['有限候选','long_history，种子1，update0600；权重、model.py、metrics.py、归一化及视觉代码原样只读保存于candidate/。','它在当前预测前已经按历史C、新C和空爪反例表现指定；当前六例用于有限候选规则检查。'],
 ['当前能力边界','冻结DINO可见特征＋对象MLP/GRU的held/in/open读出，675395个读出参数。','这是诊断读出器，不是指南完整对象Transformer/RSSM及共享未来符号读出实现。'],
 ['接口空白','不足2.1秒历史的启动、遮挡/失踪实体及重新注册、统一长窗的in/open回归仍需明确并验收。','本轮没有默默填充帧、改变UNKNOWN或部署新包装器。种子1阈值0.5没有通常意义的置信拒答区间。'],
 ['独立样本边界','冻结候选、阈值、输入策略与评分后，使用未参与训练/选择/校准/既往D诊断的新根身份；原生D的新布局/种子属于可考虑的严格来源。','之前已查看的D资料属于历史诊断，不能再称从未见过的验收集；校准留在C。'],
 ['独立验收内容','持物上移/单调下移/长静止，支撑接触负例，空爪上移，以及in正负、open回归；按根报告已知误断、漏判、UNKNOWN及非目标对象错误。','检查真值与动作方向、运动阶段、可见性变化的分离；不能只报总准确率或只挑有利画面。'],
 ['工程指标解释','指南实际完成误确认≤5%是后续工程目标；当前开发阈值或0/6错误不是总体统计保证。','不以诊断用更严门槛任意宣布项目失败，也不以小样本全对宣布可靠闭环。'],
 ['下一阶段顺序','先独立核验当前状态接口，再回到指南共享当前/未来符号头及世界模型一步预测；之后才是多步与TS/LTLf闭环。','本轮未执行新的采集、世界模型训练或规划验收；没有额外开新一轮逐例诊断。']]),
 '<p class="lead">本次有限任务结束：六条预列来源全部恢复、全部评估，无遗漏或失败剔除；有一个明确冻结的研究候选可进入下一项独立状态接口验收。独立验收尚未执行，长期静止等缺口仍保留。完整指南项目仍未完成。</p>',
 '<p class="muted">可复查材料：configs/protocol.json、configs/integration_rule.json、configs/evaluation_inputs_frozen.json；results/evaluation.json、results/public_motion_audit.json、results/delivery_validation.json；candidate/candidate_manifest.json、candidate/INPUT_CONTRACT.json；delivery_manifest.json。公共观测与私有标签分开存放在data/predicate_held_down_replay_20261014。此前所有实验与来源文件保持不变。回放入口source/replay.py需env.sh；推理入口source/evaluate.py需world_model_env.sh且防重复执行。报告及核验脚本只消费已存结果。</p>']
 css='body{font:16px/1.65 "Noto Sans CJK SC",sans-serif;background:#f3f6f8;color:#253745;margin:0}main{max-width:1240px;margin:auto;padding:40px;background:white}h1{font-size:30px}h2{font-size:23px;color:#245775;margin-top:38px}.lead{background:#edf7f4;border-left:4px solid #258b72;padding:18px}.muted{font-size:14px;color:#607586}table{border-collapse:collapse;width:100%;font-size:13px;margin:24px 0}th,td{border:1px solid #dce4ea;padding:9px;text-align:left;vertical-align:top}th{background:#edf3f7}tr:nth-child(even){background:#fbfcfd}figure{margin:25px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#607586}'
 (E/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>状态读出综合诊断与候选冻结</title><style>'+css+'</style><main>'+''.join(parts)+'</main></html>')
 save(E/'SUMMARY.json',{'status':'finite_six_existing_C_windows_complete; research_candidate_frozen_pending_independent_acceptance','roots':6,'new_roots':0,'target_public_frames':total_frames,'actions_executed':counts,'all_replays_exact_on_available_checks':True,'original_image_checks':imagechecks,'model_seeds':[0,1,2],'target_known_TP_each':[r['target_known_TP']for r in D['models']],'target_UNKNOWN_each':[r['target_UNKNOWN']for r in D['models']],'other_predicate_errors_each':[0,0,0],'new_training_updates':0,'candidate_seed':1,'candidate_manifest':str(cm),'candidate_selected_before_current_predictions':True,'motion_interpretation':'net downward transport with brief initial rise; not monotone descent','simple_direction_only_explanation_contradicted':True,'general_relation_semantics_proven':False,'independent_acceptance_performed':False,'historical_failure_scenarios':3,'entire_guide_complete':False,'next_step':'Separately frozen independent current-state interface acceptance; no more case-by-case replay in this task.'})
 print(json.dumps({'report':str(E/'report.html'),'candidate':str(cm),'figures':3,'new_model_inference':False},ensure_ascii=False))
if __name__=='__main__':main()
