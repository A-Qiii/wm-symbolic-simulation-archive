from pathlib import Path
import json,gzip,html,base64,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_interface_acceptance_20261014';B=P/'data/predicate_interface_acceptance_20261014'
NAMES={'baseline':'初始基线','near_open':'近物/开爪','near_closed':'近物/闭爪','held_after_lift':'抓取后持物','held_stationary':'长静止持物','held_downward':'连续下移持物','in_open':'已放入/抽屉开','in_closed':'已放入/抽屉关','natural_one_camera_occlusion':'单相机自然遮挡','natural_held_occlusion':'持物自然遮挡','natural_both_camera_occlusion':'双相机自然遮挡'}
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(head,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table>'
def image(name,cap):return '<figure><img alt="'+html.escape(cap)+'" src="data:image/png;base64,'+base64.b64encode((E/'figures'/name).read_bytes()).decode()+'"><figcaption>'+html.escape(cap)+'</figcaption></figure>'
def main():
 D=read(E/'results/evaluation.json');meta=read(E/'data/metadata.json');M=read(B/'data_manifest.json');protocol=read(E/'configs/protocol.json');videos=read(E/'videos/index.json');pred=np.load(E/'results/predictions.npy');rootmap={r['root_id']:r for r in read(B/'roots_manifest.json')};details=D['details'];byid={r['unique_window_id']:r for r in meta};videoids={q['window']:Path(q['path']).name for q in videos}
 font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 conditionrows=[];stats={};colors=['#2b947d','#c34c4c','#df9b29','#a1acb7'];bars=[]
 for name,x in D['conditions'].items():
  rr=[q for q in details if name in q['conditions']];n=len(rr);cat=[sum(q['all7_known_correct']for q in rr),sum(bool(q['known_error_columns'])for q in rr),sum(not q['known_error_columns']and bool(q['UNKNOWN_columns'])for q in rr),sum(not q['known_error_columns']and not q['UNKNOWN_columns']and not all(q['GT_valid'])for q in rr)];assert sum(cat)==n;bars.append(cat)
  groups={}
  for g,cols in {'held':[0,1,2],'in_drawer':[3,4,5],'open_drawer':[6]}.items():
   a=x['available']['metrics'][g];roots=x['available']['root_metrics'][g];win={'known_FP':0,'known_FN':0,'UNKNOWN':0}
   for q in rr:
    y=np.array(q['truth']);v=np.array(q['GT_valid']);k=np.array(q['known']);h=np.array(q['probabilities'])>=.5;win['known_FP']+=bool((h[cols]&~y[cols]&v[cols]&k[cols]).any());win['known_FN']+=bool((~h[cols]&y[cols]&v[cols]&k[cols]).any());win['UNKNOWN']+=bool((~k[cols]&v[cols]).any())
   groups[g]={'valid_facts':a['n'],'positive_facts':a['positive'],'negative_facts':a['negative'],'known_FP_atoms':a['known_false_positive'],'known_FN_atoms':a['known_false_negative'],'UNKNOWN_atoms':a['unknown'],'raw_FP_atoms':a['FP'],'raw_FN_atoms':a['FN'],'window_counts':win,'available_windows':n,'root_counts':{k:roots[k]for k in ['roots','positive_roots','negative_roots','known_FP','known_FN','UNKNOWN','GT_invalid']}}
  stats[name]={'planned_roots':x['planned_roots'],'available_windows':n,'distinct_roots':x['available']['distinct_roots'],'achieved_windows':x['achieved']['windows'],'all7_known_correct':cat[0],'known_wrong_windows':cat[1],'UNKNOWN_only_windows':cat[2],'GT_invalid_only_windows':cat[3],'groups':groups}
  f=lambda g:' / '.join(str(groups[g][key])for key in ['known_FP_atoms','known_FN_atoms','UNKNOWN_atoms'])
  wr=lambda g:' / '.join(str(groups[g]['window_counts'][key])for key in ['known_FP','known_FN','UNKNOWN'])
  conditionrows.append([NAMES[name],f"{n}/{x['planned_roots']}",x['achieved']['windows'],f'{cat[0]}/{n}'if n else'未覆盖',f('held'),wr('held'),f('in_drawer'),wr('in_drawer'),f('open_drawer'),cat[3]])
 save(E/'results/condition_denominators.json',stats)
 fig,ax=plt.subplots(figsize=(13,7.1));left=np.zeros(len(bars));names=list(NAMES)
 for j,label in enumerate(['7事实均有效且已知正确','存在已知错误','冲突导致UNKNOWN','仅GT边界不可判']):
  vals=np.array(bars)[:,j];ax.barh(range(len(names)),vals,left=left,color=colors[j],label=label,height=.68)
  for i,v in enumerate(vals):
   if v:ax.text(left[i]+v/2,i,str(v),ha='center',va='center',color='white'if j<2 else'#253745',fontweight='bold')
  left+=vals
 for i,name in enumerate(names):
  if left[i]==0:ax.text(.12,i,'0/12：原动作中未出现',va='center',color='#697d8d')
 ax.set(yticks=range(len(names)),yticklabels=[NAMES[n]for n in names],xlim=(0,12.7),xticks=range(0,13,2),xlabel='窗口数（每条件每根最多1窗；不跨条件累加为独立根）',title='独立验收：78个必需状态窗中76窗全事实通过；遮挡边界另外保留');ax.invert_yaxis();ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.12));ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(E/'figures/acceptance_conditions.png',dpi=160,bbox_inches='tight');plt.close(fig)
 bad=[q for q in details if q['known_error_columns']or q['UNKNOWN_columns']];assert len(bad)==2;fig,axs=plt.subplots(4,4,figsize=(12.5,11.3))
 for ri,q in enumerate(bad):
  r=byid[q['window']]
  for ci,pp in enumerate(r['public_frames']):
   with np.load(pp)as z:
    for si,cam in enumerate(['static','gripper']):
     ax=axs[2*ri+si,ci];ax.imshow(z['rgb_'+cam],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
     if si==0:ax.set_title(['t−2.10秒','t−1.40秒','t−0.70秒','当前t'][ci])
     if ci==0:ax.set_ylabel(('蓝块连续下移'if ri==0 else'粉块长静止')+'\n'+('静态相机'if si==0 else'腕部相机'),fontsize=12)
 fig.suptitle('两处首次已评分错误：仍持蓝块却报已放入；持粉块时蓝块也报held',fontsize=15);fig.tight_layout(rect=(0,0,1,.965));fig.savefig(E/'figures/first_failures.png',dpi=150,bbox_inches='tight');plt.close(fig)
 stationary=[r for r in M['records']if r['condition']=='held_stationary'];down=[r for r in M['records']if r['condition']=='held_downward'];fig,axs=plt.subplots(1,2,figsize=(13.4,5.4));xx=np.arange(12)
 axs[0].bar(xx-.18,[r['motion']['TCP_displacement_from_start_max_mm']for r in stationary],.36,color='#397fb5',label='TCP最大偏移');axs[0].bar(xx+.18,[r['motion']['object_displacement_from_start_max_mm']for r in stationary],.36,color='#df9b29',label='物体最大偏移');axs[0].axhline(1,color='#c34c4c',ls='--',label='预定静止门槛1mm');axs[0].set(xticks=xx,xticklabels=[f'{r["target_color"][0]}{r["initial_drawer"][0]}{rootmap[r["root_id"]]["replicate"]}'for r in stationary],ylim=(0,1.08),ylabel='2.1秒内距首位置的最大偏移（mm）',xlabel='目标r/b/p；初始抽屉c/o；重复0/1',title='12根真实持物长静止：最大物体偏移0.114mm');axs[0].legend(fontsize=9)
 for i,r in enumerate(down):
  trace=json.load(gzip.open(B/'episodes'/r['root_id']/'trace_private.json.gz','rt'));seq=trace[r['anchor_frame']-63:r['anchor_frame']+1];z=np.array([q['robot_obs'][2]for q in seq])*1000;z-=z[0];axs[1].plot(np.arange(64)/30-2.1,z,color={'red':'#bd4a42','blue':'#397fb5','pink':'#aa55a4'}[r['target_color']],ls='-'if rootmap[r['root_id']]['replicate']==0 else'--',lw=1.8,label=r['target_color']+str(rootmap[r['root_id']]['replicate']))
 axs[1].set(xlabel='相对当前时刻（秒）',ylabel='TCP相对首帧高度（mm）',title='6根连续下移：实际无上升，下降155.3～155.7mm');axs[1].legend(ncol=3,fontsize=9);axs[1].grid(alpha=.15);fig.tight_layout();fig.savefig(E/'figures/physical_coverage.png',dpi=160,bbox_inches='tight');plt.close(fig)
 # Diagnosis uses existing recorded states/images only, with no new inference.
 diagnosis=[]
 for q in bad:
  r=byid[q['window']];lab=read(r['labels_private']);target=r['target_color'];past=[x for x in details if x['root_id']==r['root_id']and x['anchor_frame']<r['anchor_frame']];same=[x for x in details if x['root_id']==r['root_id']and x['anchor_frame']>r['anchor_frame']];o=lab['frames'][-1]['objects_audit'][target]
  diagnosis.append({'root_id':r['root_id'],'first_scored_consequential_model_failure':q['window'],'condition':r['conditions'][0],'anchor_frame':r['anchor_frame'],'prior_scored_windows_all7_correct':all(x['all7_known_correct']for x in past),'current_target_physical_audit':o,'all64_true_held':r['motion']['held_all64'],'target_visible_both_cameras_all4':all(all(v>=4 for v in t['objects_audit'][target]['visible_pixels'].values())for t in lab['frames']),'registry_identity_correct':all(v['match']for v in q['registry_audit'].values()),'observed_error':'Premature held-to-in classification while bilateral contact and zero external support persist.'if q['known_error_columns']else'False positive held(blue) co-occurs with correct held(pink); conflict decoder makes both UNKNOWN.','causal_mechanism_identified':False,'localization':'Current state reader before any world-model successor or planning step; not evidence of a WM dynamics error.','new_inference_or_correction':False,'details':q})
 save(E/'results/failure_localization.json',diagnosis)
 missing=[(q['window'],c)for q in details for c,a in q['registry_audit'].items()if a['registered_public_uid']is None];wrong=[(q['window'],c)for q in details for c,a in q['registry_audit'].items()if a['registered_public_uid']is not None and not a['match']];overall=D['unique_window_metrics'];groups=overall['metrics'];aggregate=[]
 for name,g in groups.items():
  rr=overall['root_metrics'][name];aggregate.append([name,f"{g['positive']}/{g['negative']}",f"{g['FP']}/{g['FN']}",f"{g['known_false_positive']}/{g['known_false_negative']}/{g['unknown']}",f"{rr['known_FP']}/{rr['known_FN']}/{rr['UNKNOWN']}（根分母{rr['roots']}）",f"正根{rr['positive_roots']} / 负根{rr['negative_roots']}"])
 rootrows=[]
 for r in D['root_outcomes']:
  rootrows.append([r['root_id'],f"{r['mandatory_all7_correct_available_windows']}/{r['mandatory_window_count']}",r['mandatory_physical_coverage_complete'],r['mandatory_all7_correct_root'],len(r['controller_failures'])])
 videolinks=[]
 for r,q in zip(meta,details):
  name=videoids[r['unique_window_id']];videolinks.append('<li><a href="videos/'+html.escape(name)+'">'+html.escape(r['root_id']+' · '+','.join(NAMES[c]for c in r['conditions']))+'</a> — '+('全事实通过'if q['all7_known_correct']else'有错误/UNKNOWN'if q['known_error_columns']or q['UNKNOWN_columns']else'抽屉GT边界不可判')+'</li>')
 parts=['<h1>冻结种子1的独立状态接口验收</h1><p class="muted">本机 · 2026-10-14 · 12个全新D根 · 一次冻结推理 · 没有训练、调阈值、替换失败根或启动世界模型</p>',
 '<p class="lead"><b>结论：</b>常见抓取、近物负例、放入与抽屉开关状态支持继续开展研究；但接口仍有两处影响后续决策的边界：连续下移时一次把仍被持有的蓝块误报为已在抽屉中，长静止时一次出现双held冲突。<b>不建议把它作为无约束完成确认接口接入闭环。</b>可以考虑下一阶段离线的世界模型接口接合与误差分离研究，必须保留概率、UNKNOWN、已知失效条件和GT仅评分的边界；本轮只提出建议，没有执行接合。</p>',
 '<h2>1. 独立性、固定条件与实际覆盖</h2>',table(['固定内容','实际证据','意义'],[
 ['新根','三色×抽屉初始开/关×两种布局，共12个原生D根；新ID/种子，排除467种既有布局。','根清单和协议先于采集冻结；历史训练、开发、D诊断和控制器根均在排重范围。'],
 ['采集','11277个公共帧、11265个物理控制动作；42次既有技能请求和保持动作合计见根日志；控制失败0、全部12根保留。','只调用冻结控制原语与96步保持动作，没有中间瞬移、改控制速度/参数或根据模型结果选动作。'],
 ['长静止','12/12根；2.1秒内TCP最大偏移0.041mm、物体最大偏移0.114mm，均全程真实held。','门槛预定为TCP和物体距首位置≤1mm，不把短暂停顿充数。'],
 ['连续下移','6/6个初始开抽屉根；物体下降155.5～155.8mm，TCP下降155.3～155.7mm；逐步正向高度增量实际为0。','固定取原lower_into_drawer阶段首帧+63，未挪动锚点；有短上升的窗口不算通过。'],
 ['自然遮挡','单相机消失6窗、双相机少于4像素6窗；持物时自然遮挡0/12根。','自然遮挡来自原有动作，没有加遮挡物。没有构造出的持物遮挡明确未覆盖。'],
 ['时间处理','全部90个窗口使用[t−63,t−42,t−21,t]，30Hz，2.1秒；每根每条件一个窗口。','与此前973个历史训练窗口的原短历史不同；旧训练资料没有被改写。没有历史不足补帧。'],
 ['候选与解码','种子1/update0600，held/in/open阈值均0.5；保留原多held冲突→UNKNOWN规则。','只有一次候选推理。高概率不等于可靠；阈值0.5没有常规置信拒答区间。'],
 ['遮挡处理',f'所有真值有效事实继续评分；{len(missing)}个全隐藏对象未注册，按原管道保留零对象特征；错误实体映射{len(wrong)}项。','不把遮挡样本过滤成“不可评分”，不通过GT补对象ID，也不新增UNKNOWN规则。']]),
 image('physical_coverage.png','左图蓝柱表示TCP、橙柱表示目标物体在2.1秒内距首位置的最大偏移；红虚线是预定1mm门槛，横轴r/b/p表示红/蓝/粉，c/o表示初始抽屉关/开，0/1为两种布局。右图六条线是公共TCP高度，颜色对应目标颜色，实/虚线对应两种布局；横轴为相对当前时刻的秒数，纵轴为相对首位置的mm。该图验证运动条件确实构造成功，并不代替模型识别结果。'),
 '<h2>2. 各条件的实际输出与分母</h2>',image('acceptance_conditions.png','横轴为窗口数量，每个条件每根最多一个窗口。绿色是全部7个必要事实均有有效GT、且实际解码为已知正确；红色是已知错误；橙色是冲突后的UNKNOWN；灰色是没有模型已评分错误、但原抽屉标签处于边界，不能判全事实通过。持物自然遮挡没有样本。图中不同条件复用同一根，不是90个独立根。'),
 '<p>FP表示把假事实说成已知真，FN表示把真事实说成已知假；UNKNOWN单独计数，不能算正确。下表的“事实FP/FN/U”以各条件全部有效对象事实为单位；“窗/根FP/FN/U”表示有至少一项相应问题的窗口/独立根数。本设计每条件每根恰有一窗，因此该列的窗口数和根数相等；跨条件的根统计另见下一表。完整真/假事实分母、正/负根分母及原始阈值结果保存在results/condition_denominators.json。</p>',
 table(['条件','可用窗/计划根','物理条件达成窗','7事实同时正确','held事实FP/FN/U','held窗/根FP/FN/U','in事实FP/FN/U','in窗/根FP/FN/U','open事实FP/FN/U','GT边界窗'],conditionrows),
 table(['事实组','有效真/假事实','原始阈值FP/FN','实际已知FP/FN/UNKNOWN','跨窗有问题的独立根','真值根分母'],aggregate),
 '<p><b>严格全事实通过为83/90窗（92.2%）。</b>其中85窗的7事实都有有效GT，83窗已知正确；另5窗在关抽屉过程中靠近原标签分界，open真值无效，保留在90窗总分母但不记作模型FP/FN。模型在85个有效open事实上无错，这不证明边界状态应被直接判真/假。78个必需状态窗中76窗通过，完整必需条件同时通过为10/12根；若连自然遮挡的GT边界也要求可判，所有已采窗口都可判且正确为7/12根。不能把83/90解释为独立试验成功率或1%错误率认证。</p>',
 '<h2>3. 首个有影响的读出错误</h2>',image('first_failures.png','上两行是蓝块连续下移失败窗，下两行是粉块长静止冲突窗；每例上行静态相机、下行腕部相机，四列为实际模型输入时刻。两例目标在两相机四帧均可见、对象注册正确，私有接触评分表明64帧全程held。因此错误不能用“目标完全看不见”或“真实已经松开”解释；图像仅支持定位现象，不唯一证明模型内部采用了哪种线索。'),
 table(['根与首个已评分问题','物理证据与模型输出','实际接口后果','定位边界'],[
 ['D_interface_accept_blue_open_1，t598，连续下移','蓝块双侧接触、无外部支撑；p(held)=0.0002459，p(in)=0.9999284。此前5个已评分状态窗均通过。','已知held漏判和已知in误报在同一时刻发生；若下游仅看in目标，会提前认定放置完成。','错误首先出现在当前状态读出，尚未涉及世界模型预测或搜索；深度/接触真值未输入模型。'],
 ['D_interface_accept_pink_open_1，t471，长静止','粉块真实held，蓝块受支撑未持有；p(held蓝)=0.9997085，p(held粉)=0.9999505。此前已评分状态通过。','原始蓝块held误报触发多held保护，蓝/粉均变UNKNOWN；保护避免已知误报，但真实粉块held也丢失。','对象注册正确且目标持续可见。不能把UNKNOWN计作识别成功，也未证明一次简单阈值调整即可解决。']]),
 '<p>这里的“首个”指每根按时间排序的首个<b>已评分</b>有影响错误，并非声称对每一视频帧都做了模型推理。两处均保留原始结果，没有继续重放、单项替换、重新训练或修改阈值。可观察的错误类型是“仍在抓持却过早判为抽屉归属”和“物体身份间held冲突”；它们是否有共同内部原因尚未确定。</p>',
 '<h2>4. 能继续研究什么，哪些边界仍需解决</h2>',table(['能力/边界','本次结果','下一阶段含义'],[
 ['近而未持有、抓取完成','近物开爪12/12、闭爪12/12、抓取后12/12均7事实正确。','支持作为已覆盖场景的当前状态输入研究基础；闭爪负例不能代表全部复杂接触。'],
 ['长静止与连续下移','11/12与5/6全事实通过，确实摆脱必须净上移的单一条件，但各有一例重要失败。','需要保留失败状态边界，不能直接宣称普遍held关系可靠。'],
 ['in/open回归','已放入且开/关抽屉各6/6通过；有效open事实85/85正确。','放置结束状态可读，但一例在仍持物时提前报in，阻止把它直接当可靠完成确认。'],
 ['隐藏物体','单相机自然遮挡6/6；双相机遮挡对象事实均正确。关抽屉完成后6个目标整窗不可见，零对象特征下in仍正确。','没有把不可见状态删掉；这些成功可能利用缺席/场景线索，不能宣称已证明持续对象跟踪。持物遮挡仍未覆盖。'],
 ['边界真值与接口','5个关抽屉中间窗的open GT按原分界规则无效，模型仍输出确定概率。','需明确后续连续状态到三值接口的边界策略；本轮没有因结果改变标签或解码。'],
 ['世界模型接合建议','有条件继续离线接口/共享读出研究；暂不用于无约束闭环完成确认。','世界模型实验应把当前读出错误与未来转移错误分开记账，传播UNKNOWN并保留学习转移；GT只能用于离线评分/明示诊断。没有启动WM或完整规划。']]),
 '<h2>5. 每根结果与可复查视频</h2>',table(['独立根','必需窗全事实正确','必需物理条件完整','整根必需验收通过','控制失败'],rootrows),
 '<p>每段短视频来自64个连续原始公共观测，帧率30Hz；仅橙框标记的4帧是模型输入，其他视频帧用于查看实际运动。视频底部显示的概率是整窗末帧的输出，并非逐视频帧重新推理。视频采用H264便于播放，原始无损NPZ均保留。</p><details><summary>展开全部90段验收视频（包含失败及GT边界窗口）</summary><ul>'+''.join(videolinks)+'</ul></details>',
 '<p class="muted">协议与排重：configs/protocol.json、configs/root_exclusion_audit.json；推理前冻结：configs/evaluation_inputs_frozen.json；完整预测与根/窗统计：results/evaluation.json、results/condition_denominators.json；错误定位：results/failure_localization.json；校验：results/delivery_validation.json及delivery_manifest.json。源公共观测、私有评分标签、所有尝试和缺失条件分别保存在data/predicate_interface_acceptance_20261014。未改候选权重、原控制器、旧项目或历史验收结果，未上传任何数据。</p>']
 # Exact attempt count, derived rather than described as an assumed count.
 total_attempts=sum(len(s['attempts'])for s in M['roots']);skills=sum(a['kind']=='existing_frozen_skill'for s in M['roots']for a in s['attempts']);holds=total_attempts-skills
 parts=[x.replace('42次既有技能请求和保持动作合计见根日志',f'{skills}次既有技能请求、{holds}次保持动作')for x in parts]
 css='body{font:16px/1.65 "Noto Sans CJK SC",sans-serif;background:#f3f6f8;color:#253745;margin:0}main{max-width:1260px;margin:auto;padding:38px;background:white}h1{font-size:30px}h2{font-size:23px;color:#245775;margin-top:38px}.lead{background:#fff4e5;border-left:4px solid #df9b29;padding:18px}.muted{font-size:14px;color:#607586}table{border-collapse:collapse;width:100%;font-size:13px;margin:24px 0}th,td{border:1px solid #dce4ea;padding:9px;text-align:left;vertical-align:top}th{background:#edf3f7}tr:nth-child(even){background:#fbfcfd}figure{margin:25px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#607586}a{color:#246d94}summary{cursor:pointer;font-weight:bold}'
 (E/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>冻结候选独立状态接口验收</title><style>'+css+'</style><main>'+''.join(parts)+'</main></html>')
 save(E/'SUMMARY.json',{'status':'single_independent_state_interface_acceptance_complete_with_limits','independent_new_D_roots':12,'frames':11277,'controls':11265,'controller_skill_requests':skills,'hold_requests':holds,'control_failures':0,'unique_windows':90,'all7_valid_known_correct_windows':83,'GT_boundary_only_windows':5,'model_error_or_UNKNOWN_windows':2,'mandatory_windows':78,'mandatory_all7_correct_windows':76,'mandatory_all7_correct_roots':10,'held_stationary_correct_roots':'11/12','held_downward_correct_roots':'5/6','natural_held_occlusion_coverage':'0/12','known_in_false_positive':1,'known_held_false_negative':1,'raw_held_false_positive':1,'held_conflict_windows':1,'held_UNKNOWN_atoms':2,'seed':1,'new_training_updates':0,'acceptance_batches':1,'decision':'Conditionally supports offline WM interface research; not an unrestricted closed-loop completion confirmation interface. WM not started.','candidate_or_controller_changed':False,'full_guide_complete':False,'no_upload':True})
 print(json.dumps({'report':str(E/'report.html'),'figures':3,'videos':len(videos),'model_failure_windows':2,'GT_boundary_only_windows':5},ensure_ascii=False))
if __name__=='__main__':main()
