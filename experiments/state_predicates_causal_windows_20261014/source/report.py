from pathlib import Path
import json,base64,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_causal_windows_20261014';B=P/'data/predicate_causal_replay_20261014'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(head,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table>'
def img(name,cap):return '<figure><img alt="'+html.escape(cap)+'" src="data:image/png;base64,'+base64.b64encode((E/'figures'/name).read_bytes()).decode()+'"><figcaption>'+html.escape(cap)+'</figcaption></figure>'
def main():
    m=read(B/'data_manifest.json');D=read(E/'results/evaluation.json');T=read(E/'results/training.json');F=read(E/'results/fixed1500_diagnostic.json')['results'];cases=read(E/'results/fresh_failures.json');allcases=read(E/'results/fresh_cases.json');train=[r for r in m['trajectories']if r['split']=='train'];dev=[r for r in m['trajectories']if r['split']=='dev']
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#397fb5','#d77e29'];names=['普通近邻负例组','困难支撑负例组'];arms=['ordinary_negative','hard_negative'];conditions=['hard_negative','ordinary_negative','held_positive'];titles=['困难负例误报 / 6根','普通负例误报 / 6根','真正held漏判 / 6根']
    fig,axs=plt.subplots(1,3,figsize=(13,4.4))
    for ax,cond,title in zip(axs,conditions,titles):
        key='FN'if cond=='held_positive'else'FP'
        for off,arm,c,name in zip([-.18,.18],arms,colors,names):
            vals=[r['fresh_conditions'][cond]['held'][key]for r in D[arm]['runs']];bars=ax.bar(np.arange(3)+off,vals,.34,color=c,label=name)
            for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.04,str(v),ha='center')
        ax.set(xticks=range(3),xticklabels=['种子0','种子1','种子2'],yticks=[0,1,2],ylim=(0,2),title=title,ylabel='错误根数（越少越好）');ax.grid(axis='y',alpha=.15)
    axs[0].legend(fontsize=9);fig.suptitle('固定网络与等量监督：补入困难负例没有得到稳定新C收益',fontsize=16);fig.tight_layout();fig.savefig(E/'figures/fresh_C_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for ax,arm,name in zip(axs,arms,names):
        counts=[]
        for r in D[arm]['runs']:
            q=r['fresh_conditions']['held_positive']['held'];counts.append([q['known_true_positive'],q['known_false_negative'],q['positive_UNKNOWN']])
        bot=np.zeros(3)
        for k,(col,label)in enumerate([('#2a8a70','已知且正确为held'),('#cb5746','已知但错误为非held'),('#a6adb4','UNKNOWN')]):
            vals=np.array(counts)[:,k];ax.bar(range(3),vals,bottom=bot,color=col,label=label);bot+=vals
        ax.set(xticks=range(3),xticklabels=['种子0','种子1','种子2'],ylim=(0,6.8),yticks=range(7),title=name,ylabel='6个真实held根的输出数量');ax.grid(axis='y',alpha=.12)
    axs[0].legend(fontsize=9,loc='upper right');fig.suptitle('校准后的接口可用性：UNKNOWN不计为正确识别',fontsize=16);fig.tight_layout();fig.savefig(E/'figures/held_known_unknown.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # Show actual current-frame pixels for all fresh failure windows.
    fig,axs=plt.subplots(len(cases),3,figsize=(13,3.7*len(cases)),gridspec_kw={'width_ratios':[1,1,1.6]})
    for axrow,q in zip(axs,cases):
        meta=q['metadata'];j=meta['target_index'];frame=q['physical_audit']['frames'][-1]
        with np.load(meta['public_frames'][-1])as z:
            for ax,cam in zip(axrow[:2],['static','gripper']):ax.imshow(z['rgb_'+cam],interpolation='nearest');ax.axis('off');ax.set_title('静态相机'if cam=='static'else'腕部相机')
        ax=axrow[2];ax.axis('off');text=[meta['sample_id'].replace('C_causal_dev_','').replace('__',' / '),'当前held真值：'+str(frame['held']),'双侧接触：'+str(frame['bilateral'])+'；外部支撑：'+str(frame['external_support_contacts']),f"夹宽 {frame['gripper_width_m']*1000:.3f} mm；TCP距 {frame['tcp_distance_m']*1000:.3f} mm"]
        for name,arm in zip(names,arms):text.append(name+' p(held)：');text.append(' / '.join(f"{q['predictions'][arm+'_seed'+str(s)]['probabilities'][j]:.4f}"for s in range(3)))
        ax.text(0,.98,'\n'.join(text),va='top',fontsize=11)
    fig.suptitle('错误首先出现在当前状态读出：全部新C失败窗的真实末帧',fontsize=16);fig.tight_layout();fig.savefig(E/'figures/fresh_failure_frames.png',dpi=160,bbox_inches='tight');plt.close(fig)
    rows=[]
    for arm,name in zip(arms,names):
        for r in D[arm]['runs']:
            c=r['fresh_conditions'];h=r['historical_C_all'];rows.append([name,r['seed'],r['selected_update'],c['hard_negative']['held']['FP'],c['ordinary_negative']['held']['FP'],c['held_positive']['held']['FN'],c['held_positive']['held']['known_true_positive'],h['held']['FN'],h['in_drawer']['FP'],'通过'if r['qualified']else'未通过'])
    parts=['<h1>因果窗口恢复与困难负例覆盖对照</h1><p class="muted">本机 · 2026-10-14 · 固定网络/执行器/DINO · 12个既有训练根恢复 + 6个新C开发根</p>',
    '<p class="lead"><b>结论：</b>完整观测已成功恢复，四帧稳定筛选导致的导出遗漏得到验证；但把困难负例覆盖从3个已有根补到12个，并未稳定减少新C误报。两组困难负例误报均为三个种子的[0,1,0]/6根，困难组还出现新的held漏判和普通负例误报。没有据此晋升模型或追加独立验收。</p>',
    '<p>这是一次完成的受控数据覆盖检验，不是“观测缺失所以没训练”的上一轮状态。网络结构没有改，控制器没有改，held物理定义没有放宽。当前结论不否定数据覆盖的重要性，也不证明两种方法等价；它限定了这批新增短暂负例、当前读出器和固定训练预算下的实际收益。</p>',
    '<h2>1. 恢复是否真实可靠</h2>',
    table(['检查','结果','含义'],[['两条试点','A红闭与B红闭均通过训练前冻结的状态/事件/图像容差。','随后继续其余10条，未放宽容差或更换根。'],['12个训练根恢复','3554记录帧；全部可比较TCP/物体位置、机器人姿态/夹宽和第24帧完整场景状态差为0。','使用原初始化和原动作序列；没有中间GT强制复位。'],['事件与已存图像','held、双侧接触、外部支撑事件时序一致；156个原保存图像锚点的双相机RGB与掩码逐位一致。','中间帧是本次真实物理重放的保存观测；原来没有落盘的帧无法逐位对原文件验证，不伪称原文件恢复。'],['新C开发','6根，3颜色×2抽屉状态；每根困难负例、普通负例、真正held正例各1窗，共18窗。','3根用于联合选模，另3根用于校准；它们是开发集，不是独立测试。'],['完整公共输入','18条轨迹共5330记录帧，保存RGB、深度、可见掩码、标定、本体状态；私有标签另存。','只对54个选定因果窗口缓存冻结DINO；编码器不读接触、物体GT位置、未来或成功标志。'],['特征/实体检查','原输入重编码通过冻结特征一致性检查；54窗×3实体=162项注册核对无错配。','旧模型、源数据、归一化与控制器仍保留。']]),
    '<p>严格复现容差预先写入 configs/protocol.json：位置≤0.1毫米、角度≤0.001弧度、事件偏移≤1控制步、已存RGB平均绝对差≤2灰度级、目标掩码IoU≥0.95；实际以上可比较指标均达到精确匹配。旧完整日志未保存逐帧物体姿态角，不能宣称这一未保存字段逐帧比较过；第24帧完整scene_obs包含可比较的完整场景状态。两次执行均属于同一个根，绝不分到不同split或重复计独立根。</p>',
    '<h2>2. 导出语义单独版本化</h2><p><code>causal_anchor_v2</code>以当前t帧的真实物理事实作标签，输入严格为[t−3,t−2,t−1,t]。过去三帧的指令、接触或held状态允许不同；目标仍须在四帧至少一个相机可见。held(t)仍是双侧指部承重且无外部承重接触（力阈值0.01N）。完整后续运动只做离线标签核验，不能成为输入。</p>',
    '<p>新增困难窗选择闭爪过程中第一个双侧接触且仍受支撑的合格锚点；普通窗选同根同抓取轨迹中最后一个开爪、无机器人承重接触、仍在支撑面且距离&lt;75毫米的锚点；正窗选择最终真实held并通过物体/末端运动跟随核验。12个训练根与6个新C根均形成完整三元组，无缺失替换或结果驱动重采。旧四帧稳定导出的样本与全部历史结果保持原版本。</p>',
    '<h2>3. 同根同数量比较固定了什么</h2>',
    table(['项目','两组相同条件','唯一训练样本差异'],[['网络与视觉','原675,395参数读出器，DINO冻结、原A/B归一化、原损失权重。','没有新的对象对或几何分支。'],['训练数据','原900+73窗不变；每根追加同一个真正held正窗。','每根另追加一个普通近邻负窗或困难支撑负窗，各12个。'],['监督数量','每组997窗；有效标签、类别计数和掩码逐项相同，新窗仅监督目标held/in与open。','负例的公共观测不同；其他标签/样本不变。'],['预算与随机性','种子0/1/2，每个1500更新×60=90,000窗口呈现，AdamW lr=.001、wd=.001、clip=1。','每个种子两组复用完全相同的997窗索引流；总9000更新。'],['联合选模','旧C180 + 历史补充C37 + 新C选模3根；按原门槛失败数、超标量、宏BCE排序。','没有按新C校准成绩换检查点。'],['联合校准','旧C120 + 历史补充C37 + 另3个新C根；沿用原tau网格和覆盖/错误/有效召回规则。','UNKNOWN必须单列；无合格阈值的谓词明确失败。']]),
    '<p>两组有效标签计数相同，排除了负例数、训练步数或正例监督不同这一类解释；但困难与普通窗的闭爪指令、夹宽、接触外观和时间相位不同。因此比较检验的是困难情境的训练覆盖作用，并未单独隔离外部支撑这一物理变量。原训练总根数仍为72，没有因重放或新增窗口变成84根。</p>',
    '<h2>4. 新旧开发结果与接口可用性</h2>',
    img('fresh_C_comparison.png','横轴是配对训练种子，纵轴是6个新C根中各类原始0.5阈值错误数。蓝色普通负例组，橙色困难负例组。三个条件各每根1窗；同根三个窗口不当作独立场景，三个模型种子也不当作新增场景。'),
    table(['组别','种子','选定更新','新C困难FP/6','新C普通FP/6','新C held FN/6','新C已知held TP/6','历史C held FN/97','历史C in FP/813','联合开发'],rows),
    '<p>历史C合计包括旧选择180窗、旧校准120窗、历史补充37窗；held共97正/840负，in共124正/813负，open共243正/94负。所有模型历史C与新C的open均无原始错误；新C的in全部是负例且无原始错误，因此只能检验该新C的in误报，不能验证其正例召回。两组都只有1/3种子联合合格，但合格种子不同，不能把相同通过数解释为等价。</p>',
    img('held_known_unknown.png','每根是真实held正例。绿段表示已知且正确，红段表示已知但判错，灰段表示UNKNOWN；柱高均为6根。普通组种子1、困难组种子0/1没有合格held校准阈值，因此全部UNKNOWN，不能靠拒答宣称识别正确。'),
    '<p>困难组种子0的新C原始held召回为5/6，其他模型为6/6。历史C还出现新增held漏判或in误报。无合格校准阈值时保存的1.000001是失败哨兵，不是可部署阈值；即便原始召回为6/6，全UNKNOWN也没有提供有效已知正断言。</p>',
    '<h2>5. 固定末次检查点与真实失败证据</h2>',
    '<p>选择结果冻结后另查每组固定第1500次更新，仅作原始概率诊断，不重新选模、不重校准：两组新C困难误报仍为[0,1,0]；困难组种子0仍漏掉一个真正held，种子1仍把一个普通负例读成held。因此新C的这组差异不能仅归因于选择了不同更新点。历史C部分错误会随更新变化，具体数字完整保留。</p>',
    img('fresh_failure_frames.png','展示全部3个新C失败窗的原始末帧；每行左为静态相机，中为腕部相机，右侧概率按种子0/1/2排列。接触/支撑数仅为离线物理核验，不进入模型。概率大于0.5代表原始held判断为真，并不保证校准可靠。'),
    '<p>蓝块闭抽屉的真正held窗：四帧均双侧承重、外部支撑为0、双视角可见，运动跟随核验通过；困难组种子0却给held概率0.0226。红块闭抽屉的普通窗（帧180）与困难窗（帧181）相邻：前者开爪仍受支撑，后者刚闭爪双侧接触但仍有2个外部支撑，二者held都应为假。困难组种子1分别给0.8029与0.9750；普通组种子1也在困难窗给0.6570。最早错误在当前感知读出，尚无世界模型或规划步骤参与。</p>',
    '<p>这些证据排除了本次失败样本中“没有真实held”“未来图像进入输入”或“恢复时画面/实体错位”的解释。它们没有唯一确定视觉可辨识性、四帧时间融合或某种外观关联究竟占主导；本轮未对失败单例追加修补或改变网络。</p>',
    '<h2>6. 覆盖假设得到什么支持，下一步是什么</h2><p>已有轨迹确实包含此前未导出的真实短暂支撑状态，且可在不改控制器的情况下恢复其完整公共输入。这一采样机制判断已获支持。另一方面，在严格配对、标签数量和训练预算一致的比较中，补入这些状态没有稳定降低新C误报，并损害了部分种子的召回或校准，因此“补足这12根就能修好接口”的具体假设未获支持。</p>',
    '<p>三个种子均未出现困难负例误报下降，故未触发预定的少量独立验收条件；没有D采集或验收后调整，也没有晋升模型。下一项更有价值的工作是利用现已保存的公共深度、本体状态与完整因果时间序列，先检查相近支撑负例与真实离台正例是否存在可稳定区分的公共线索，并比较其在训练/开发根中的一致性。应先验证可观测性与时间信息利用，再决定是否需要表示改动；这些后续检查尚未执行。</p>',
    '<p class="muted">完整协议：configs/protocol.json；选定权重与阈值：configs/selected_models_frozen.json；训练日志/126检查点：logs/、models/；新旧开发全指标：results/evaluation.json；同更新诊断：results/fixed1500_diagnostic.json；物理失败证据：results/fresh_failures.json；原始公共观测与私有审计分目录保存在 data/predicate_causal_replay_20261014。文件校验见 delivery_manifest.json 与 results/delivery_validation.json。无上传，无世界模型或规划实验。</p>']
    css='body{font:16px/1.65 "Noto Sans CJK SC",sans-serif;background:#f4f7fa;color:#253745;margin:0}main{max-width:1200px;margin:auto;padding:42px;background:#fff}h1{font-size:30px}h2{font-size:23px;color:#245775;margin-top:36px}.lead{padding:18px;background:#edf5fa;border-left:4px solid #397fb5}.muted{color:#657b8c;font-size:14px}table{border-collapse:collapse;width:100%;font-size:13px;margin:22px 0}th,td{border:1px solid #dce4ea;padding:9px;text-align:left;vertical-align:top}th{background:#edf3f7}tr:nth-child(even){background:#fbfcfd}figure{margin:24px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#607586}code{background:#edf3f7;padding:3px}'
    (E/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>因果窗口覆盖对照</title><style>'+css+'</style><main>'+''.join(parts)+'</main></html>')
    save(E/'SUMMARY.json',{'status':'recovery_and_matched_comparison_completed_no_stable_new_C_benefit','export_version':'causal_anchor_v2','held_definition_unchanged':True,'training_replay_roots':12,'all_replays_match_available_original_checks':True,'training_replay_frames':3554,'saved_image_anchors_compared':156,'fresh_C_roots':6,'full_public_frames':5330,'feature_windows':54,'training_pool_per_arm':997,'model_parameters':675395,'matched_updates_per_seed':1500,'seeds':3,'arms':2,'fresh_hard_FP_by_seed':{'ordinary':[0,1,0],'hard':[0,1,0]},'fresh_ordinary_FP_by_seed':{'ordinary':[0,0,0],'hard':[0,1,0]},'fresh_held_FN_by_seed':{'ordinary':[0,0,0],'hard':[1,0,0]},'new_C_effect_persists_at_fixed1500':True,'coverage_omission_confirmed':True,'benefit_of_correction_not_demonstrated':True,'independent_acceptance_run':False,'model_promoted':False,'uploaded':False,'world_model_or_planner_run':False})
    print('CAUSAL_REPORT_READY',E/'report.html')
if __name__=='__main__':main()
