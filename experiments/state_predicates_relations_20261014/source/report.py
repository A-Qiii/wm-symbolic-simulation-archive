"""Local, self-contained report from frozen train/C-dev evidence only."""
from pathlib import Path
import json, hashlib, base64, html, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

P=Path('/home/robot/research/projects/wm_symbolic')
E=P/'experiments/state_predicates_relations_20261014'
J=P/'experiments/state_predicates_joint_20261014'
O=P/'experiments/state_predicates_20261014'
A=P/'experiments/state_predicates_augmented_20261014'
def read(p): return json.loads(Path(p).read_text())
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def esc(s): return html.escape(str(s))
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#ffffff','savefig.facecolor':'#ffffff'})
BLUE='#357cb4';ORANGE='#d57a25';GREEN='#268570';GRAY='#65717b'
T=read(E/'results/pair_training.json');D=read(E/'results/pair_diagnostics.json');B=read(J/'results/balanced_loss.json')

def figsave(fig,name):
    fig.savefig(E/'figures'/name,dpi=160,bbox_inches='tight');plt.close(fig)

def architecture():
    fig,ax=plt.subplots(figsize=(15,7.5));ax.set(xlim=(-.15,15),ylim=(0,8));ax.axis('off')
    def box(x,y,w,h,text,c):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=c,edgecolor='#9baab7',linewidth=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
    def arrow(a,b,c=GRAY): ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':c,'lw':1.6})
    ax.text(0,7.7,'实际输入与最小对象对分支',fontsize=19,weight='bold')
    box(0,5.3,2.4,1.7,'4帧 × 双相机 RGB\n静态 200×200\n腕部 84×84\n居中补边至 224×224','#eaf3fa')
    box(3,5.3,2.7,1.7,'冻结 DINOv2-small\n每图 16×16 patch\n每个 patch 384维\n不是单一全局向量','#eaf3fa');arrow((2.4,6.15),(3,6.15))
    box(6.3,5.3,3.2,1.7,'每相机：12个实例 ROI\n各保留2×2单元\n另有4×4场景网格\n8图共512个缓存 token','#eaf3fa');arrow((5.7,6.15),(6.3,6.15))
    box(10.1,5.3,4.5,1.7,'原读出器：目标 ROI 的4单元取均值\n384维外观 + 6维可见几何\n两相机拼成780维 → MLP128\nROI内部单元顺序不再显式保留','#edf6f0');arrow((9.5,6.15),(10.1,6.15))
    box(0,2.7,4.4,1.6,'场景4×4降为有序2×2网格\n两相机共3072维 + 15维本体状态\nTCP位置/姿态、夹宽、关节、指令\nMLP3087→128；4帧均有输入','#edf6f0');arrow((7.9,5.3),(4.1,4.3))
    box(5.1,2.7,4.4,1.6,'新增：目标与12个可见实例配对\n共享外观编码32维 + 8维相对几何\n注意力汇聚，两视角合成64维\n64→128残差，加到目标编码','#fff1df');arrow((7.9,5.3),(7.3,4.3),ORANGE);arrow((9.5,3.5),(12.3,5.3),ORANGE)
    box(10.1,2.7,4.5,1.6,'每目标：128目标 + 128场景\n共享 GRU256→96，按时间融合4帧\n共享 MLP96→64→2\n输出 held 与 in 概率','#edf6f0');arrow((12.3,5.3),(12.3,4.3))
    ax.plot([4.4,4.7,4.7,9.8,9.8],[3.0,3.0,2.35,2.35,3.0],color=GRAY,lw=1.6);arrow((9.8,3.0),(10.1,3.0))
    box(0,0.6,4.4,1.35,'open：独立场景 GRU128→96\n与独立 MLP96→64→1\n没有目标实体 query 或自注意力','#edf6f0');arrow((2.2,2.7),(2.2,1.95))
    box(5.1,.6,9.5,1.35,'蓝色：冻结视觉与公共缓存；绿色：既有读出器；橙色：本轮唯一新增分支。\n箭头表示数据流，不代表因果解释。外观对照将新增8维几何置零，参数量相同。\n原模型仍保留框、场景顺序及 DINO 上下文；不能据此宣称其没有空间或关系能力。','#f4f5f7')
    figsave(fig,'architecture.png')

def comparison():
    fig,axs=plt.subplots(1,2,figsize=(13.5,5.2),gridspec_kw={'width_ratios':[2.2,1]})
    ax=axs[0];x=np.arange(3)
    for off,arm,c,label in [(-.18,'appearance_only',BLUE,'同容量外观对象对'),(.18,'relative_geometry',ORANGE,'外观 + 8维相对几何')]:
        runs=T[arm]['runs'];ys=[r['tune']['joint_root_class_macro_BCE'] for r in runs]
        bars=ax.bar(x+off,ys,.34,color=c,label=label)
        for b,r,y in zip(bars,runs,ys): ax.text(b.get_x()+b.get_width()/2,y+.00035,('通过' if r['qualified'] else '未通过')+f'\n{y:.5f}',ha='center',va='bottom',fontsize=9)
    ax.set(xticks=x,xticklabels=['种子0','种子1','种子2'],ylabel='联合开发宏平均 BCE（无单位，越低越好）',ylim=(0,.017),title='按预定联合规则选择的检查点');ax.legend(loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.15)
    ax=axs[1];v=[T[a]['mean_joint_tune_BCE'] for a in ['appearance_only','relative_geometry']];bars=ax.bar([0,1],v,color=[BLUE,ORANGE],width=.58)
    for b,y in zip(bars,v): ax.text(b.get_x()+b.get_width()/2,y+.00015,f'{y:.5f}\n联合合格 2/3',ha='center',va='bottom',fontsize=10)
    ax.set(xticks=[0,1],xticklabels=['外观对照','相对几何'],ylabel='三个种子的平均 BCE',ylim=(0,.009),title='未出现组间几何收益');ax.grid(axis='y',alpha=.15)
    fig.suptitle('同数据、同采样、同损失、同参数量；只比较新增8维几何',fontsize=16)
    fig.text(.5,-.02,'颜色表示两组模型；“通过”含原C与补充C的选择和校准门槛。条柱为开发结果，不是独立测试结果；没有误差条或显著性结论。',ha='center',fontsize=10)
    fig.tight_layout();figsave(fig,'development_comparison.png')

def cases():
    om=read(O/'data/metadata.json');am=read(A/'data/metadata_train_dev.json');old=om[1185];new=[r for r in am if r['split']=='dev'][32]
    info=read(P/'data/core1200_v1/episodes/C_core1200_19_b0/00_open_drawer/endpoint_info_private.json')['before']
    nl=read(new['labels_private'])['frames'][-1]
    configs=[('old_cal',105,4,old['public_inputs'],info['scene_info']['movable_objects']['block_blue']['uid'],'蓝块：in(blue, drawer)=假',old['episode_id']),('new_dev',32,0,new['public_frames'],nl['objects_audit']['red']['object_uid_evaluator_only'],'红块：held(red)=假',new['sample_id'])]
    records=[];fig,axs=plt.subplots(4,4,figsize=(13,12))
    for ci,(domain,idx,col,frames,uid,title,sid) in enumerate(configs):
        pixels=[]
        for fi,path in enumerate(frames):
            with np.load(path) as z:
                counts={}
                for vi,cam in enumerate(['static','gripper']):
                    ax=axs[ci*2+vi,fi];ax.imshow(z['rgb_'+cam],interpolation='nearest');mask=z['seg_'+cam]==uid;counts[cam]=int(mask.sum());yy,xx=np.where(mask)
                    if len(xx): ax.add_patch(Rectangle((xx.min()-.5,yy.min()-.5),xx.max()-xx.min()+1,yy.max()-yy.min()+1,fill=False,edgecolor='#ffe343',linewidth=1.4))
                    ax.set_title(f'{"静态" if vi==0 else "腕部"}相机 · 第{fi+1}帧',fontsize=11);ax.axis('off')
                pixels.append(counts)
        probs={}
        for arm,loc,prefix in [('既有平衡损失读出器',J,'balanced_loss'),('同容量外观对象对',E,'appearance_only'),('外观+相对几何',E,'relative_geometry')]:
            probs[arm]=[float(np.load(loc/'results'/f'{prefix}_seed{s}_{domain}.npy')[idx,col]) for s in range(3)]
        records.append({'sample':sid,'domain':domain,'array_index':idx,'predicate_column':col,'truth':False,'target':title,'public_frames':frames,'visible_pixels_per_frame':pixels,'probabilities_seed012':probs})
        axs[ci*2,0].text(-.05,1.25,title,transform=axs[ci*2,0].transAxes,fontsize=15,weight='bold',ha='left')
    records[0]['physical_evidence']={'tabletop_position_m':old['object_positions_for_evaluation_only'][1],'drawer_m':old['drawer_m_for_evaluation_only'],'category':1,'upward_drawer_support':False,'audit_note':'blue identity verified by pixel ROI repacking and private endpoint UID; UID is evaluator-only'}
    records[1]['physical_evidence']={**nl['objects_audit']['red'],'gripper_width_m':nl['gripper_width_m'],'gripper_command':nl['gripper_command']}
    fig.suptitle('两个开发案例的全部4帧公共RGB；黄色框标注被评估物体',fontsize=17,y=1.01)
    fig.text(.5,-.005,'图像保持原始比例，按最近邻显示；黄色框由保存的分割掩码生成，仅帮助阅读。身份与接触真值只用于审计/评分，不送入网络。',ha='center',fontsize=10)
    fig.tight_layout(h_pad=3);figsave(fig,'development_cases.png');write(E/'results/development_cases.json',records);return records

def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
def image(name,caption):
    data=base64.b64encode((E/'figures'/name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'

def main():
    architecture();comparison();C=cases();sections=[]
    def add(s): sections.append(s)
    add('<h1>当前状态读出器：输入审计与对象对表示对照</h1><p class="meta">本机实验 · 2026-10-14 · 冻结 DINOv2 与物理执行器 · 仅训练集和 C 开发集</p>')
    add('<div class="conclusion"><b>本轮结论：</b>未发现标签、实体身份或特征打包错误。原读出器确实没有显式实例配对，但保留了框、场景空间顺序、本体状态和 DINO 上下文。新增二维相对几何没有优于同容量外观对照：两组均 2/3 种子通过联合开发标准，平均 BCE 分别为 0.00502 与 0.00650。因此没有补采新验收集，也没有晋升新模型。</div>')
    add('<p>这项结论只否定本轮“8维二维框关系能提供足够增益”的证据，不等于神经网络没有关系能力，也不等于整个世界模型与规划项目完成。既往 D、38窗口测试和83窗口验收仅保留为历史，本轮不据其成绩选架构、检查点或阈值。</p>')
    add('<h2>1. 实际喂给网络的内容</h2>')
    add(image('architecture.png','蓝色框为冻结视觉缓存，绿色框为既有读出路径，橙色框为本轮新增对象对分支；箭头仅表示数据流。尺寸是每帧/每目标的向量维数，时间维另外由4帧GRU处理。'))
    add(table(['问题','代码所显示的事实','含义与限制'],[
        ['全局特征还是 patch？','冻结 DINOv2-small 的 16×16×384 patch 图；每张图缓存48个ROI token和16个场景token。','不能把原输入描述成只有一个全局特征。'],
        ['对象空间位置保留到哪？','目标ROI的2×2 token取均值，加框坐标4维、可见性1维、占据率1维；双相机合成780维。','ROI内部显式单元顺序被平均；框与上下文仍保留空间线索。'],
        ['四帧如何融合？','目标MLP与场景MLP结果拼接，逐目标共享GRU；open另有场景GRU。','不是把四帧简单平均，也不是对象Transformer。'],
        ['实体 query 与共享头？','颜色像素注册选择目标ROI，调用端实体名索引输出；held/in共享编码器、GRU与双输出MLP。','无可学习实体ID、query或目标之间的自注意力；非目标实例原来只隐含在场景与DINO上下文。'],
        ['本体状态是否输入？','每帧15维：TCP xyz、RPY、夹宽、7关节位置、夹爪指令；用旧A/B统计量归一化。','本轮没有移除它；输入字段存在不等于已证明模型稳健利用。'],
        ['哪些观测没有利用？','保存的深度与相机标定没有输入旧读出器。','物体GT坐标、接触、未来帧、成功标志均未作为前向输入；未增加私有真值。']]))
    add('<h2>2. 标签、身份与可见性核对</h2>')
    add(table(['核对范围','实际结果','可支持的判断'],[
        ['旧训练/C开发：1200窗口、3600对象','实体重组误差为0；身份不匹配为0。','本轮审计没有发现换错对象或打包错误。'],
        ['旧数据 in：502个正对象记录','全部可沿向上承重接触追溯到抽屉；没有非held反例被标为负。','当前 in 是抽屉支撑/归属标签，不是图像框相交或纯几何点包含。'],
        ['补充训练/C开发：110窗口、330对象','重组误差为0；目标110个 in 负标签均与实际类别一致。','补充数据也没有实体映射故障；此集合没有 in 正例，不能独自验证正例召回。'],
        ['12个实例槽的容量限制','旧1200窗口都记录截断，最多2个实例；抽屉在1200个窗口均保留。','限制是真实的，但未发现抽屉因截断丢失；原读出器是在下游不单列其他实例。'],
        ['实际 held 标签链','core1200_collect → calibrated_labels_at → PhysicalOracle.held 覆盖 → drawer_ownership。','实际规则为双侧指部承重且无外部支撑；不是辅助文件中较弱的旧规则。']]))
    add('<p>首次审计脚本把“所有缓存都无截断”作为假设断言，遇到真实 overflow 后停止。随后改为完整计数并核对抽屉是否保留；未改缓存、标签或旧模型。这是审计假设修订，不是模型实现修复。接触和模拟器UID仅用于离线核对和图注。</p>')
    add('<h2>3. 同容量的最小表示对照</h2><p>新增目标—可见实例注意力：目标与邻居DINO各投影为32维，共享配对MLP输出32维值与1个权重；两相机汇聚后64→128，残差加到原目标编码。几何组额外使用中心偏移2维、对数宽高比2维、IoU、交叠/目标面积、交叠/邻居面积和占据率差，共8维。外观组将这8维置零。两组均用可见性及公共ROI匹配排除不可见/自身配对；没有命名“抽屉”槽位或数值实例ID输入。</p>')
    add(table(['固定条件','本轮设置','这样比较的原因'],[
        ['容量','675,395 → 702,788参数（两新组相同，增加27,393）','避免把几何收益与额外参数或其他实例外观混在一起。'],
        ['数据与监督','900旧A/B窗口 + 73补充A/B窗口；补充目标held/in/open三项有效。','沿用上一轮平衡损失的同一数据、掩码与类别权重。'],
        ['优化预算','每组种子0/1/2；每种子1500更新×60样本；两组总9000更新。','逐个复用上一轮的1500×60批索引；每种子90,000次窗口呈现。'],
        ['优化器','AdamW，学习率0.001，权重衰减0.001，梯度裁剪1。','同样从头初始化；残差初值为0，两组初始输出等同旧结构。'],
        ['选择/校准','选择：旧C180窗口/12根 + 补充C37/6根；校准：旧C120/8根 + 同一补充C37/6根。','补充C在选择与校准复用，明确属于开发集；不是独立验证。'],
        ['联合标准','各域各谓词原始FPR≤1%、正例召回≥95%；校准覆盖≥95%、已知错误≤2%、已知FP/全部负例≤1%、已知TP/全部正例≥95%。','空正例域不计算召回，UNKNOWN不得掩盖漏检。'],
        ['排序','先失败域/谓词数量，再超标量，再按域/谓词/根/类别宏平均BCE；同分取更早。','所有规则先于训练冻结；126个计划检查点均保留。']]))
    add(image('development_comparison.png','横轴为训练种子或组别，纵轴为开发宏平均二元交叉熵BCE，越低越好；蓝色为外观对照，橙色为相对几何。通过标记同时要求选择与校准合格。'))
    rows=[]
    for arm,label in [('appearance_only','外观对象对'),('relative_geometry','外观+相对几何')]:
        for r in T[arm]['runs']:
            rows.append([label,r['seed'],r['selected_update'],'通过' if r['qualified'] else '未通过',f"{r['tune']['joint_root_class_macro_BCE']:.8f}",'无超标项' if r['qualified'] else '补充C：held误报1/31负例（3.23%）',str(r['calibration']['thresholds'])])
    add(table(['组别','种子','所选更新','联合开发','宏平均BCE','门槛失败原因','held/in/open阈值'],rows))
    add('<p>阈值表中的1.000001是“没有合格校准阈值”的哨兵值，该谓词返回UNKNOWN，不是可部署阈值；相应模型仍明确判为未通过。</p>')
    add('<p>预先决定的补采条件：相对几何组至少2/3种子联合合格，且比外观组总原始门槛失败数更少；若相同，平均开发BCE至少降低20%。实际两组都是2/3、失败计数都是2（同一个补充C错误在选择和校准各计一次），几何组BCE反而高约29.4%，条件为假。把全部模型固定看第1500次更新时，两个组仍各有一个种子在同一补充C接触支撑样本误报；没有通过换检查点得到额外成功结论。</p>')
    add('<h2>4. 哪一步先出错：两个开发画面</h2>')
    add(image('development_cases.png','上两行：旧C蓝块容器归属案例；下两行：补充C红块接触支撑案例。每组横向为连续4帧，纵向为静态与腕部相机；黄色框仅标识被评估对象，无坐标轴。'))
    for c in C:
        add('<h3>'+esc(c['target'])+'</h3><p><code>'+esc(c['sample'])+'</code></p>')
        add(table(['模型','种子0概率','种子1概率','种子2概率'],[[k]+[f'{x:.6f}' for x in v] for k,v in c['probabilities_seed012'].items()]))
        add('<p>表内为相应谓词为真的原始概率，真值均为假；0.5为原始二分类分界，较大概率代表更强的错误正类倾向，不是校准可靠性保证。四帧可见像素数（静态/腕部）依次为 '+esc('、'.join(f"{x['static']}/{x['gripper']}" for x in c['visible_pixels_per_frame']))+'。</p>')
    add('<p>旧C蓝块位于桌面，抽屉闭合，四帧身份一致且双视角可见，没有指向抽屉的承重路径。上一轮平衡损失种子0把同一状态的4个重复窗口读成 in；另外两个种子没有这一错误。两个新增组在此开发状态均读对，因此不能把这一个画面作为“几何组特有收益”。</p>')
    add('<p>补充C红块的最后一帧已双侧接触，夹宽40.376毫米，TCP距离2.506毫米，但仍有2个外部支撑接触，held真值为假。外观种子1给0.557812，几何种子2给0.826129。最早可定位的错误仍是“接触且仍受支撑”被读成“已持有”，尚未进入任何世界模型或规划环节。只靠这两个失败种子，不能进一步唯一归因于视觉遮挡、二维信息不足或训练覆盖。</p>')
    add('<h2>5. 有限替换诊断：使用了什么，还不能证明什么</h2>')
    pr=read(E/'results/development_input_probes.json')['rows'];rows=[]
    names={'zero_explicit_object_bbox_only':'只把目标显式框置零','swap_left_right_context_cells':'交换场景左右网格','replace_normalized_proprioception_by_training_mean':'本体状态替换为训练均值'}
    for k,label in names.items():
        q=[x for x in pr if x['perturbation']==k]
        rows.append([label]+[f"{np.mean([x['groups'][g]['mean_abs_probability_change']for x in q]):.7f}" for g in ['held','in_drawer','open_drawer']]+[sum(x['groups'][g]['raw_prediction_flips']for x in q for g in ['held','in_drawer','open_drawer'])])
    add(table(['既有模型的预定输入替换','held平均|Δp|','in平均|Δp|','open平均|Δp|','原始分类翻转数'],rows))
    abl=[]
    for k,label in [('zero_relative_geometry','仅移除8维相对几何'),('zero_entire_pair_residual','移除整个对象对残差')]:
        q=[x for x in D['selected_relative_path_ablations']if x['change']==k]
        abl.append([label,f"{np.mean([x['mean_abs_logit_change']for x in q]):.6f}",f"{np.mean([x['mean_abs_probability_change']for x in q]):.7f}",sum(x['prediction_flips']for x in q),f"{sum(x['original_errors']for x in q)} → {sum(x['changed_errors']for x in q)}"])
    add(table(['选定几何模型的单项替换','平均|Δlogit|','平均|Δp|','分类翻转','错误计数变化'],abl))
    add('<p>每行均在3种子×3个C开发域上汇总：概率与logit变化是9组均值的平均，翻转和错误为这9组计数之和，不能看成9个独立实验或独立物理事件。替换后未训练、未保存改动权重、未据此选替代模型。几何置零没有翻转，整个分支移除则错误从2增到23：支持新增实例分支影响了已训练函数，不支持8个二维关系量已解决任务。</p>')
    add('<p>这些替换会产生分布外输入；既有模型的饱和概率、DINO上下文和其余输入也可能掩盖影响。零翻转不等于未使用该输入，更不是对“网络无关系能力”的证明。移除分支是功能敏感性证据，不能当作真实世界唯一因果诊断。</p>')
    add('<h2>6. 交付、边界与下一步依据</h2><p>本轮已完成输入与标签审计、三项预定开发替换、唯一对象对表示对照及固定末次检查点诊断。没有实际实现缺陷需要修复，旧源代码/数据/模型保持哈希不变。新代码、缓存、126个检查点、选择阈值和完整结果保存在独立目录；没有上传文件，没有生成新验收数据，也没有运行完整世界模型或LTL规划。</p>')
    add('<p>本轮二维框关系路线缺乏进入独立验收的开发证据。若继续解决支撑与归属混淆，需要先为新增信息建立可检查依据，例如检查公共深度/相机标定是否能区分承重高度与相对运动，或查明当前训练根覆盖遗漏了哪些接触支撑变化；这些均尚未实施，不能算作修复。这里停止的是证据不足的本轮候选晋升，不是以更新次数用完为由宣告整个目标结束。</p>')
    add(table(['本机文件','用途'],[
        ['configs/protocol.json；configs/representation_decision.json','训练前冻结范围、门槛与补采条件'],
        ['configs/selection_frozen.json','六个选择结果、阈值、权重哈希；补采否决理由'],
        ['source/model.py；source/pair_model.py；source/train_pairs.py','实际旧结构副本、新分支和训练流程'],
        ['results/input_and_label_audit.json；supplement_audit.json','3930个实体核对与旧标签接触路径'],
        ['results/pair_training.json；pair_diagnostics.json','开发完整指标、末次检查点、替换诊断'],
        ['results/development_cases.json；development_input_probes.json','图中具体概率、帧路径、审计证据与输入替换结果'],
        ['data/neighbor_sources.json；delivery_manifest.json','复用缓存来源与全量交付文件的哈希索引'],
        ['results/delivery_validation.json','冻结资产、源文件、图表、训练日志与交付检查']]))
    add('<p class="meta">复现实验需使用本机 world_model_env.sh；审计/报告脚本不训练模型。train_pairs.py 遇到已存在 pair_training.json 会拒绝覆盖。单卡任务使用 runs/gpu.lock 串行；完整原数据与模型缓存仅引用，不重复下载。报告内图片全部嵌入，可离线浏览。</p>')
    css='body{font:16px/1.65 "Noto Sans CJK SC","Microsoft YaHei",sans-serif;color:#233342;background:#f4f7fa;margin:0}main{max-width:1180px;margin:auto;background:white;padding:44px 46px}h1{font-size:30px;line-height:1.3}h2{font-size:23px;margin-top:38px;color:#194b6b;border-bottom:1px solid #dce5ec;padding-bottom:8px}h3{font-size:18px}p{margin:15px 0}.meta{color:#64788b;font-size:14px}.conclusion{background:#edf5fa;border-left:5px solid #357cb4;padding:20px;margin:25px 0}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{border:1px solid #dce3e8;padding:10px 12px;text-align:left;vertical-align:top}th{background:#edf3f7}tbody tr:nth-child(even){background:#fafcfd}figure{margin:25px 0}img{display:block;width:100%;height:auto}figcaption{font-size:14px;color:#596c7d;margin-top:8px}code{font-size:12px;word-break:break-all;background:#f1f4f7;padding:3px} @media(max-width:800px){main{padding:20px}table{font-size:12px}th,td{padding:6px}}'
    (E/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>状态读出器：表示诊断</title><style>'+css+'</style><main>'+''.join(sections)+'</main></html>')
    frozen={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(E/'configs/protocol.json'),'decision_sha256':sha(E/'configs/representation_decision.json'),'role':'Post-development freeze only; no new independent acceptance was collected','collection_warranted':False,'reason':'Both arms qualify2/3 and total raw failures2; relative mean tune BCE0.006498 exceeds appearance0.005020; predeclared benefit criterion false.','selected':[{k:r[k]for k in ['arm','seed','selected_update','checkpoint_path','checkpoint_sha256','qualified']}|{'thresholds':r['calibration']['thresholds']}for arm in ['appearance_only','relative_geometry']for r in T[arm]['runs']]}
    fp=E/'configs/selection_frozen.json'
    if not fp.exists():write(fp,frozen);fp.chmod(0o444)
    else: assert read(fp)['selected']==frozen['selected']
    write(E/'SUMMARY.json',{'scope':'Current-state representation diagnosis, train/C development only','status':'completed_narrow_comparison_no_dev_evidence_for_geometry','old_windows_audited':1200,'supplement_windows_audited':110,'entity_checks':3930,'entity_errors':0,'implementation_bug_found':False,'parameter_counts':[675395,702788],'appearance_qualified':2,'relative_qualified':2,'seeds_per_arm':3,'appearance_mean_BCE':T['appearance_only']['mean_joint_tune_BCE'],'relative_mean_BCE':T['relative_geometry']['mean_joint_tune_BCE'],'new_acceptance_collected':False,'model_promoted':False,'full_world_model_or_planning_run':False,'uploaded':False,'report':str(E/'report.html'),'remaining_scientific_issue':'Support/container ownership is not solved; no unique causal mechanism identified.'})
    print('REPORT_CREATED',str(E/'report.html'))
if __name__=='__main__': main()
