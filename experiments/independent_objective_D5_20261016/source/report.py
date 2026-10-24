from config import *
import csv,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.patches import Patch,Rectangle
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fontManager.addfont(FONT);plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':11,'svg.fonttype':'path','figure.facecolor':'white'})
METHODS={'actual':'真实后继读出','one_step':'真实前态单步','recursive':'真实初态递归'}
STEPS=['1 开屉','2 抬起','3 放入','4 关屉']
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |'for r in rows])
def csvout(name,rows):
 with (R/name).open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 primary=read(E/'results/primary.json');paired=read(E/'results/paired_effects.json');rr=read(E/'results/root_results.json');stages=read(E/'results/stage_metrics.json');facts=read(E/'results/fact_metrics.json');roots=read(E/'results/root_coverage.json');protocol=read(E/'configs/protocol.json');collection=read(E/'results/collection_verification.json')
 lookup={(x['condition'],x['method'],x['root_id']):x for x in rr}
 # A root keeps its identity across both seeds. Missingness is drawn explicitly.
 ordered=sorted(roots,key=lambda x:(COLORS.index(x['color']),x['root_id']))
 fig,(ax,bx)=plt.subplots(2,1,figsize=(14,7.5),gridspec_kw={'height_ratios':[1.2,1]},layout='constrained')
 palette={0:'#d8dde4',1:'#258459',2:'#d98832',3:'#3e6f9e',4:'#ffffff'}
 for seed in [0,1]:
  for i,r in enumerate(ordered):
   a=lookup[(f'mixed_seed{seed}','recursive',r['root_id'])]['all4_correct'];b=lookup[(f'latent_seed{seed}','recursive',r['root_id'])]['all4_correct']
   kind=4 if a is None or b is None else 3 if a and b else 1 if b else 2 if a else 0
   ax.add_patch(Rectangle((i-.48,seed-.35),.96,.7,facecolor=palette[kind],edgecolor='#87909b',hatch='xx'if kind==4 else None))
 ax.set(xlim=(-.6,29.6),ylim=(-.6,1.8),yticks=[0,1],yticklabels=['种子 0','种子 1'],xticks=range(30),xticklabels=[r['root_id'].split('_')[-1]for r in ordered],xlabel='每格为同一个初始配置 root；编号按目标物体分组，两个种子复用同一组配置')
 for pos,c in [(4.5,'目标 red（10）'),(14.5,'目标 blue（10）'),(24.5,'目标 pink（10）')]:ax.text(pos,1.52,c,ha='center')
 for x in [9.5,19.5]:ax.axvline(x,color='#64748b',lw=1)
 ax.set_title('四个技能边界的全部9项事实均正确：每个配置的配对结果',loc='left',weight='bold')
 ax.legend(handles=[Patch(facecolor=palette[k],edgecolor='#87909b',hatch='xx'if k==4 else None,label=l)for k,l in [(3,'两者全链正确'),(1,'仅Z全链正确（新增）'),(2,'混合全链正确（退步）'),(0,'两者至少一处错'),(4,'缺后续观测')]],ncol=3,loc='upper center',bbox_to_anchor=(.5,-.3),frameon=False)
 for seed in [0,1]:
  row=next(r for r in paired if r['seed']==seed and r['method']=='recursive');d=row['difference']*100;lo,hi=np.array(row['paired_root_bootstrap95'])*100
  bx.errorbar(d,seed,xerr=[[d-lo],[hi-d]],fmt='o',capsize=6,color='#258459',markersize=9);bx.text(53,seed,f"+{row['gains']} / −{row['losses']}；差值 {d:.1f} pp\n95%区间 [{lo:.1f}, {hi:.1f}] pp",va='center')
 bx.axvline(0,color='#7b8794',ls='--');bx.set(xlim=(-3,87),ylim=(-.7,1.7),yticks=[0,1],yticklabels=['种子 0','种子 1'],xlabel='仅Z减混合：全链正确率的配对差值（百分点，pp）',xticks=[0,10,20,30,40,50]);bx.set_title('按目标颜色分层、以root为单位的配对bootstrap（10,000次）',loc='left');bx.grid(axis='x',alpha=.2)
 fig.suptitle('D5：30个尝试配置，29条完整可评轨迹；两个种子不等于60个独立配置',fontsize=15,weight='bold')
 for ext in ['png','svg']:fig.savefig(R/f'paired_roots.{ext}',dpi=170,bbox_inches='tight')
 plt.close(fig)
 # Independent per-boundary accuracy; not cumulative chain accuracy.
 fig,axs=plt.subplots(1,2,figsize=(14,5.5),sharey=True,layout='constrained')
 for seed,ax in enumerate(axs):
  for arm,method,color,style,label in [('mixed','actual','#697586','--','真实后继读出（两者共用）'),('mixed','recursive','#4169a1','-','混合损失：递归'),('latent','recursive','#258459','-','仅Z损失：递归')]:
   rows=[next(r for r in stages if r['condition']==f'{arm}_seed{seed}'and r['method']==method and r['color']=='all'and r['global_step']==j)for j in range(1,5)];vals=[r['full9_correct']/r['full9_evaluable_roots']*100 for r in rows];ax.plot(range(1,5),vals,color=color,ls=style,marker='o',label=label)
   for j,r in enumerate(rows):ax.annotate(f"{r['full9_correct']}/{r['full9_evaluable_roots']}",(j+1,vals[j]),xytext=(-24,18)if method=='actual'else(18,9)if arm=='mixed'else(0,-16),textcoords='offset points',ha='center',color=color,fontsize=10)
  ax.set(title=f'种子 {seed}',xticks=range(1,5),xticklabels=STEPS,ylim=(0,112),yticks=range(0,101,20),xlabel='技能执行后的边界（每步单独计分）');ax.grid(alpha=.2)
 axs[0].set_ylabel('该边界全部9项事实正确的配置比例（%）');axs[1].legend(loc='lower left',frameon=False);fig.suptitle('仅Z的主要收益在末段；前段及种子0放置阶段仍有退化',fontsize=15,weight='bold')
 for ext in ['png','svg']:fig.savefig(R/f'stage_accuracy.{ext}',dpi=170,bbox_inches='tight')
 plt.close(fig)
 # Summaries strictly aggregate existing counts; no new model run or parameter choice.
 pooled=[]
 for seed in [0,1]:
  for arm in ['mixed','latent']:
   for fact in FACTS:
    rows=[r for r in facts if r['condition']==f'{arm}_seed{seed}'and r['method']=='recursive'and r['color']=='all'and r['fact']==fact];d={'seed':seed,'arm':arm,'fact':fact}
    for k in ['state_TP','state_FN','state_FP','state_TN','change_positive_N','predicted_change_TP','no_positive_change_N','change_FP','after_state_hit_on_true_change']:d[k]=sum(r[k]for r in rows)
    pooled.append(d)
 csvout('recursive_fact_summary.csv',pooled)
 md=[]
 md.append('D5固定候选独立配对验证已完成。仅Z损失在同一批新配置上的递归全链正确数由7/29提高至14/29和16/29；两组种子分别新增7条、9条且无退步。主要收益来自末段Holding误报减少，但前段及种子0的放置阶段仍有退化。保留仅Z为有独立正证据的研究候选，不自动替换正式模型。')
 md.append('这是对四个事前冻结候选的有限验证。此前开发集门槛失败原样保留；本轮未设新门槛、未训练、未按D选择检查点或阈值、未扩大环境或追加采样。当前执行器为CALVIN中的PhysicalOracle，转移网络为残差模型；本结果不等同于原指南的HULC、RSSM或TS-LTL闭环项目完成。')
 md.append('**样本、物理失败与分母**')
 md.append(table(['计数','实际值','含义'],[['计划 / 尝试配置','30 / 30','red、blue、pink各10个新初始配置'],['至少一个边界可评 / 完整四边界可评','30 / 29','29是完整可评分母，不是全部物理成功分母'],['物理四技能全部成功','28 / 30','执行器结果，与模型预测正确率分开'],['调用上限 / 实际调用','120 / 117','115成功、2失败；3个后续槽位未执行'],['可评边界 / 保留关键点','117 / 989','失败后的真实观测保留；全部关键点的9事实掩码有效'],['连续控制帧 / 双相机视频','59,195 / 30段','标签重算及视频帧数对齐复核通过'],['外部重试 / 替换配置','0 / 0','没有用成功轨迹替换失败']]))
 md.append(table(['配置','失败位置与观测证据','计分处理'],[['D5_objective_pink_03','开屉下降到把手时不可达/受阻，末端误差0.05147 m，抽屉位置0 m','保留第1边界；后3边界缺失，全链主分母排除，全部30配置的界限另报'],['D5_objective_pink_08','关屉水平推进不可达/受阻，末端误差0.09808 m，抽屉位置0.06311 m','完整4边界全部纳入主分母，按实际中间屉状态评分']]))
 md.append('上述原因是执行器事件记录，不能据此断言具体碰撞物；没有修复控制器或重放这些配置。首50转移边界文件去重后3,827,631字节；全采集含视频和物理轨迹259,440,236字节，后者不冒充首50的存储量。')
 md.append('**主结果：每条轨迹的四个边界、每个边界全部9事实均正确**')
 md.append('9事实为3颜色的Holding、3颜色的In(drawer)，以及Closed/Open/Intermediate三种屉状态。阈值始终为二元0.5、屉状态argmax。真实后继读出直接读取实际后继潜变量；真实前态单步在每一步从实际前态出发；递归仅从实际初态出发，后续不注入实际状态。四模型使用完全相同的真实轨迹，前两种读法帮助区分读出误差与递归累积误差。')
 pr=[]
 for r in primary:
  if r['condition'].startswith('latent')and r['method']=='actual':continue
  lo,hi=r['wilson95'];pr.append([r['condition'].replace('mixed','混合').replace('latent','仅Z'),METHODS[r['method']],f"{r['correct_roots']}/29",f"{100*r['proportion']:.1f}% [{100*lo:.1f}, {100*hi:.1f}]",f"{r['correct_roots']}/30 — {r['correct_roots']+1}/30"])
 md.append(table(['固定候选','读法','全链正确','比例与Wilson 95%区间','全部30配置的缺失界限'],pr))
 md.append('全部30配置的界限分别把唯一缺失轨迹视为不能再贡献或可能贡献1条全链正确轨迹；不是置信区间，也不把缺失当作事实预测错误。真实后继读出在同一种子下两候选逐值一致，因此只列一次。')
 md.append(table(['种子','读法','新增 / 退步 / 持平','配对差值','root bootstrap 95%区间'],[[r['seed'],METHODS[r['method']],f"{r['gains']} / {r['losses']} / {r['ties']}",f"{100*r['difference']:+.2f} pp",f"[{100*r['paired_root_bootstrap95'][0]:+.2f}, {100*r['paired_root_bootstrap95'][1]:+.2f}] pp"]for r in paired if r['method']!='actual']))
 md.append('不确定性以完整可评root为单位，在red=10、blue=10、pink=9各层内有放回抽样10,000次，固定种子16972000；两种模型种子共用同一抽样矩阵。两种子是同29个配置的两次模型比较，不能合并成58或60个独立样本。区间条件于这29条完整轨迹及固定的D初始化分布；小样本和一次缺失仍限制推广。单步种子0少1条的现象保留，但不能独自推翻递归配对正证据。')
 md.append('![每配置配对结果和不确定性](paired_roots.png)')
 md.append('图中每格是一条配置：蓝色为两者全链正确，绿色为仅Z新增，橙色为退步，灰色为两者均有错误，白色交叉纹为缺失。横轴编号按目标颜色分组，色块颜色表示模型结果而非物体颜色；两行复用同样配置。下图横轴为仅Z减混合的正确率差（百分点），点为观测差值、横线为分层root bootstrap区间、虚线为零差值。实际意义是新增整链正确轨迹的数量及其有限样本不确定性。')
 md.append('**阶段与事实错误**')
 md.append('![逐边界全9事实正确率](stage_accuracy.png)')
 md.append('图的横轴依次为开屉、抬起、放入、关屉后的边界，纵轴为该单个边界全部9事实正确率；灰虚线为实际后继读出，蓝实线为混合损失递归，绿实线为仅Z递归，点旁标注正确数/可评数。第1步分母30，后3步分母29。连线帮助看阶段变化，不是连续时间插值，也不是累计全链正确率。实际意义是定位收益在后段，避免只看末步或只看总体均值。')
 sr=[]
 for seed in [0,1]:
  for arm in ['mixed','latent']:
   for step in range(1,5):
    r=next(r for r in stages if r['condition']==f'{arm}_seed{seed}'and r['method']=='recursive'and r['color']=='all'and r['global_step']==step);N=r['full9_evaluable_roots'];sr.append([seed,'混合'if arm=='mixed'else'仅Z',STEPS[step-1],f"{r['full9_correct']}/{N}",f"{N-r['Holding_correct']}/{N}",f"{N-r['In_correct']}/{N}",f"{N-r['drawer_correct']}/{N}"])
 md.append(table(['种子','候选','边界','全9正确','Holding组错误','In组错误','屉状态组错误'],sr))
 md.append('组错误表示该组至少一个事实错误，组之间可能重叠，不能简单相加。关屉后Holding组错误由22/29降到12/29、11/29；这时真实Holding为假，错误体现为误报。种子0放置阶段Holding组错误则由8/29升至11/29；第1步屉状态错误由1/30升到3/30（种子0）或2/30（种子1）。这解释了末段改善不意味着各阶段都改善。没有对失败单例做新修正。')
 md.append('下面汇总递归读法的逐事实正变化召回与误报，所有117个已观测转移均保留。单元为“预测出0→1的GT正变化数 / GT正变化数；在无GT正变化时误报0→1数 / 无GT正变化数”。GT与预测都使用各自前后状态；递归预测前态是上一步预测，而非替换成实际前态。分母为0时无可估召回，不能解释为零性能。跨阶段汇总仅供定位；逐阶段、逐目标颜色的状态TP/FN/FP/TN、正变化召回/FPR及GT正变化后的命中数见CSV。')
 fr=[]
 for fact in FACTS:
  row=[fact]
  for seed in [0,1]:
   for arm in ['mixed','latent']:
    d=next(x for x in pooled if x['seed']==seed and x['arm']==arm and x['fact']==fact);row.append(f"{d['predicted_change_TP']}/{d['change_positive_N']}; {d['change_FP']}/{d['no_positive_change_N']}")
  fr.append(row)
 md.append(table(['事实','种子0 混合','种子0 仅Z','种子1 混合','种子1 仅Z'],fr))
 md.append('**固定条件、复核与文件**')
 md.append('模型均在本轮采集前由原C规则选定：混合F+0.25Z为update0750；仅Z为αZ的update1000，α=3.341086352737259。F为四个特征组等权平均的变化加权重建误差，Z为原始潜变量均方误差；α在此前AB训练批次上按参数梯度范数中位数校准并固定，未用D校准。两者此前训练数据、总更新预算、采样日程一致，编码器、解码器、读出器冻结；共同C选择规则选出的检查点更新数不同，如表所示。本轮是比较冻结候选，不新增训练。原始潜变量为4×64，固定DINOv2编码当前双RGB和机器人15维观测；GT只用于独立标签/评分，不进入模型推理。实际后继读出不是GT输入。')
 md.append(table(['候选','检查点','SHA256'],[[name,Path(spec['path']).name,spec['sha256']]for name,spec in protocol['models'].items()]))
 md.append('采集前封存时间：'+protocol['frozen_utc']+'。30个root使用种子16970000–16970029，执行顺序种子16971000；沿用D4原D桌面初始化分布：三物体槽位随机排列，x在桌面范围14%/49%/84%位置加±12mm均匀扰动，y在桌面边界内缩15mm后均匀采样，z=0.46m，yaw均匀[-π,π]，抽屉关闭。先排除已声明root/种子及三物体都在12mm内的近重复布局，再封存；没有依据模型结果筛选初态。')
 md.append('固定控制上限为每技能900控制步、技能后保持96步、复位稳定127步、内部尝试至多2次、外部重试0次。四技能固定顺序；失败后只停止该root余下槽位。989关键点标签已按真实物理姿态重算，59,195控制帧和30视频帧数已对齐；最终数值复核重算360条root结果、192组阶段指标、1,728组事实指标及2,106行直接配对事实。原消融70个封存文件及失败gate的哈希保持不变。')
 md.append(table(['交付','用途'],[[f'[协议]({E}/configs/protocol.json)','根列表、预算、主分母、采样与不确定性定义'],[f'[模型/源码预封存]({E}/configs/before_collection_frozen.json)','检查数据采集前的固定条件'],[f'[数值复核]({E}/results/verification.json)','冻结哈希、计数、配对区间和无更新审计'],[f'[采集复核]({E}/results/collection_verification.json)','标签、视频、完整物理轨迹与失败保留'],[f'[最终封存清单]({E}/configs/delivery_manifest.json)','实验和报告全部文件的SHA256'],['[逐root配对CSV](paired_root_outcomes.csv)','每个配置的新增/退步/持平'],['[逐边界指标CSV](stage_metrics.csv)','三种读法、阶段及颜色的分母和准确数'],['[逐事实及变化CSV](fact_state_and_change_metrics.csv)','状态TP/FN/FP/TN、正变化召回和误报完整表'],['[直接配对事实CSV](paired_propositions.csv)','同root事实真值、各读法预测与符号logit余量（非校准置信度）'],['[递归事实汇总CSV](recursive_fact_summary.csv)','报告正变化表的可复核整数计数'],[f'[连续视频索引]({E}/results/media_verification.json)','30段双相机视频及帧数']]))
 md.append('只读复核命令：\n\n```bash\nsource /home/robot/research/projects/wm_symbolic/world_model_env.sh\npython /home/robot/research/projects/wm_symbolic/experiments/independent_objective_D5_20261016/source/verify.py\n```\n\n报告在本机生成；采集、缓存、推理串行使用GPU锁，共用既有模型缓存。旧robot_state、ROS和单词本应用未改动。本轮固定验证到此结束，没有新的训练、插值诊断、阈值调整或数据扩展；下一步方向留待本轮结果归纳后决定。')
 (R/'REPORT.md').write_text('\n\n'.join(md)+'\n');save(E/'results/report_generation.json',{'utc':now(),'source_sha256':sha(__file__),'figures':['paired_roots.png','paired_roots.svg','stage_accuracy.png','stage_accuracy.svg'],'new_model_runs':0,'new_parameter_selection':False,'finite_validation_complete':True});print('REPORT_WRITTEN',R/'REPORT.md',flush=True)
if __name__=='__main__':main()
