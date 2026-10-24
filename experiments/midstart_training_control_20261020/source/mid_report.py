from common import *
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
NAMES={'full':'完整轨迹','history_mix':'相同裁剪＋公开历史','cold_mix':'完整＋中途冷启动'};COL={'full':'#596273','history_mix':'#2877b8','cold_mix':'#dd7e20'}
def table(rows,cols):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>'for k,label in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>'for k,label in cols)+'</tr>'for r in rows)+'</tbody></table>'
def find(rows,**kw):return next(r for r in rows if all(r[k]==v for k,v in kw.items()))
def selected(rows,**kw):return[r for r in rows if all(r[k]==v for k,v in kw.items())]
def ratio(a,b):return f'{a}/{b} ({100*a/b:.1f}%)'if b else'0/0（无正例）'
def figures(initial,whole,horizon):
 fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
 for ax,key,den,title in zip(axs,['In_TP','Held_TP','drawer_correct'],['In_positive','Held_positive','starts'],['冷启动 In 正例召回','冷启动 Held 正例召回','冷启动 drawer 三分类准确率']):
  for ai,a in enumerate(ARMS):
   for seed,marker in [(0,'o'),(1,'s')]:
    r=find(initial,source='development',arm=a,seed=seed,startgroup='middle',template='ALL',branch='all');ax.scatter(ai+(seed-.5)*.12,100*r[key]/r[den],color=COL[a],marker=marker,s=65)
  ax.set(xticks=range(3),xticklabels=['完整','裁剪＋历史','裁剪＋冷启动'],ylim=(-3,103),ylabel='正确率 / %',title=title);ax.grid(axis='y',alpha=.2)
 fig.suptitle('根级独立开发：每个非终止中途起点分别初始化；圆=种子0，方=种子1')
 for ext in ['png','svg']:fig.savefig(S/f'reports/cold_initial.{ext}',dpi=170)
 plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
 for ax,mode,title in zip(axs,['current','free','free_after2'],['原完整轨迹：逐步观测','从原h0自由推完整轨迹','观测到原h2后自由推剩余轨迹']):
  for ai,a in enumerate(ARMS):
   for seed,marker in [(0,'o'),(1,'s')]:
    r=find(whole,source='development',arm=a,seed=seed,startgroup='full',template='ALL',branch='all',mode=mode);ax.scatter(ai+(seed-.5)*.12,100*r['all_correct']/r['trajectories'],color=COL[a],marker=marker,s=65)
  ax.set(xticks=range(3),xticklabels=['完整','裁剪＋历史','裁剪＋冷启动'],ylim=(-3,103),ylabel='整轨所有 Held / In / drawer 正确率 / %',title=title);ax.grid(axis='y',alpha=.2)
 fig.suptitle('原完整轨迹回归；圆=种子0，方=种子1。自由推演不读取预测起点之后的观测')
 for ext in ['png','svg']:fig.savefig(S/f'reports/full_regression.{ext}',dpi=170)
 plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
 for ax,seed in zip(axs,[0,1]):
  for a in ARMS:
   rr=selected(horizon,source='development',arm=a,seed=seed,startgroup='middle',template='ALL',branch='all');ax.plot([r['horizon']for r in rr],[100*r['all_correct']/r['boundaries']for r in rr],'-o',color=COL[a],label=NAMES[a])
  ax.set(xlabel='距冷启动起点的技能步数',ylabel='该步所有符号正确率 / %',ylim=(-3,103),title=f'种子 {seed}');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('开发集冷启动后自由预测；各步分母随剩余长度减少，不能把曲线形状单独解释为时间效应')
 for ext in ['png','svg']:fig.savefig(S/f'reports/cold_rollout.{ext}',dpi=170)
 plt.close(fig)
def main():
 guard();initial=read(S/'results/initial.json');whole=read(S/'results/whole.json');horizon=read(S/'results/horizon.json');ret=read(S/'results/retention.json');probes=read(S/'results/probes.json');first=read(S/'results/first_errors.json');exposure=read(S/'results/exposure.json');figures(initial,whole,horizon)
 ini=[];reg=[];retain=[];expl=[];fresh=[]
 for seed in [0,1]:
  for arm in ARMS:
   r=find(initial,source='development',arm=arm,seed=seed,startgroup='middle',template='ALL',branch='all');ini.append({'model':NAMES[arm],'seed':seed,'In':ratio(r['In_TP'],r['In_positive']),'In_FP':ratio(r['In_FP'],r['In_negative']),'Held':ratio(r['Held_TP'],r['Held_positive']),'Held_FP':ratio(r['Held_FP'],r['Held_negative']),'drawer':ratio(r['drawer_correct'],r['starts']),'joint':ratio(r['all_correct'],r['starts'])})
   rr={'model':NAMES[arm],'seed':seed}
   for mode in ['current','one_step','free','free_after2']:
    r=find(whole,source='development',arm=arm,seed=seed,startgroup='full',template='ALL',branch='all',mode=mode);rr[mode]=ratio(r['all_correct'],r['trajectories'])
   reg.append(rr)
   for group in ['full','middle']:
    r=[x for x in ret if x['source']=='development'and x['arm']==arm and x['seed']==seed and((x['origin']==0)if group=='full'else x['origin']>0)];retain.append({'model':NAMES[arm],'seed':seed,'group':group,'both':ratio(sum(x['both_correct']for x in r),len(r)),'forgot':ratio(sum(x['forgot']for x in r),sum(x['previous_correct']for x in r))})
   for template in ['recovery_seen6','order_known4','order_novel4']:
    for branch in [1,2]:
     r={'model':NAMES[arm],'seed':seed,'template':template,'branch':'正常'if branch==1 else'外部落空扰动'}
     for mode in ['current','free']:
      q=find(whole,source='exploratory',arm=arm,seed=seed,startgroup='full',template=template,branch=branch,mode=mode);r[mode]=ratio(q['all_correct'],q['trajectories'])
     expl.append(r)
   if any(x['source']=='independent'for x in initial):
    for template in ['order_known4','order_novel4']:
     r={'model':NAMES[arm],'seed':seed,'template':template}
     q=find(initial,source='independent',arm=arm,seed=seed,startgroup='full',template=template,branch='all');r['initial_In']=ratio(q['In_TP'],q['In_positive']);r['initial_all']=ratio(q['all_correct'],q['starts'])
     for mode in ['current','free']:
      q=find(whole,source='independent',arm=arm,seed=seed,startgroup='full',template=template,branch='all',mode=mode);r[mode]=ratio(q['all_correct'],q['trajectories'])
     fresh.append(r)
 # First errors for old exploratory six-step branches; each sequence counted once.
 errors=[]
 for seed in [0,1]:
  for arm in ARMS:
   for branch in [1,2]:
    rr=selected(first,source='exploratory',arm=arm,seed=seed,template='recovery_seen6',branch=branch,origin=0,mode='free');ct=collections.Counter((r['step'],r['wrong_relations'])for r in rr)
    errors.extend({'model':NAMES[arm],'seed':seed,'branch':branch,'first_step':k[0],'relations':k[1],'episodes':v}for k,v in sorted(ct.items()))
 findings={'utc':now(),'development_initial':ini,'full_regression':reg,'retention':retain,'exploratory':expl,'independent':fresh,'first_errors':errors,'probe_summary':probes,'exposure':exposure,'validation_trigger':read(S/'results/independent_validation_trigger.json'),'final_interpretation_pending_human_read':False}
 save(S/'results/report_tables.json',findings)
 body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>中途起点训练对照</title><style>body{max-width:1200px;margin:32px auto;font:16px/1.65 system-ui;color:#202b38;padding:0 20px}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{padding:7px;border:1px solid #ccd3dd;text-align:left}th{background:#eef2f6}img{max-width:100%;border:1px solid #ddd}h2{margin-top:36px}.note{background:#fff4df;padding:16px;border-left:4px solid #dd7e20}code{overflow-wrap:anywhere}</style><h1>从已有经历中途初始化：真正训练的三组对照</h1>'''
 body+='''<p>旧 posterior reset 诊断固定模型、清除历史，检验分布变化；本轮从原训练经历中均匀选择非终止起点，实际优化当前观测初始化与后续转移网络。三个条件架构、参数量（2,490,068）、两个种子的初始化、原训练根、每模型1500次更新、优化器和阈值相同。全部使用RGB-D粗网格候选；local_gap不采用。本轮不是原指南整体验收。</p><p class="note">世界模型输入无GT；固定低层控制器有GT辅助，此研究条件已获允许。没有改控制器，没有新数据进入训练，没有上传。图中灰色=完整轨迹，蓝色=相同裁剪内容但保留公开历史，橙色=完整轨迹混入中途冷启动；圆/方分别表示种子0/1。</p>'''
 body+='''<h2>为何需要第三组</h2><p>裁剪组每次微批次保留一半完整轨迹，另一半随机取原训练轨迹的非终止中途起点；不按标签挑起点。与完整组相比，转移监督曝光减少约25%，多步和终点损失曝光也变化。因此冷启动组首先与相同裁剪＋公开历史组比较，后两者起点、状态/动作监督和总更新数完全相同。保留历史组只用此前公开观测和动作预填充递归先验，不反传前缀；从起点观测更新开始正常反传。冷启动组则用零递归先验。</p><p>初始化输入为原来正常可得的双相机四控制帧窗口和机器人状态：包含此前三帧及当前帧，没有未来帧。冻结特征缓存的区域注册器可能保留更早的公开历史；本轮清零的是世界模型递归状态，不能称为完全无历史单帧输入。标签只进入训练损失或评分，不进入初始化。独立新根使用自然新建的注册器。</p>'''
 body+=table(exposure,[(k,k)for k in ['arm','seed','sampled_episodes','full_samples','middle_samples','supervised_transitions','supervised_boundaries','multi3_transition_terms','terminal_terms','initial_any_In_samples','history_warmup_transitions']])
 body+='''<p>每模型抽样24,000条经历（含重复）；原训练228条经历、94个根，开发78条经历、34个根、14个布局族，与训练根和布局族分离。开发集此前已使用过，只能称开发，不能称新的独立测试。所有原训练h0的In正例为零；中途正例来自这些旧训练经历的已有后续边界。</p><h2>独立开发：冷启动符号与后续转移</h2><p>对每个非终止中途边界分别清零递归状态。每行的In/Held为对象级正例召回，FP是负例中的误报；drawer是三分类；联合准确要求三个对象的Held、In及drawer同时正确。不同起点可能来自同一轨迹，不能当作独立物理试验。</p>'''
 body+=table(ini,[('model','训练条件'),('seed','种子'),('In','In召回'),('In_FP','In误报'),('Held','Held召回'),('Held_FP','Held误报'),('drawer','drawer'),('joint','联合')]);body+='<img src="cold_initial.png"><p>纵轴是对应正确率；横轴是训练条件，不是时间。此图检查初态支持缺口是否被补上，同时显示Held/drawer是否受损。</p><img src="cold_rollout.png"><p>横轴为冷启动后的动作步数，纵轴为该步全部符号正确率。线只连接同一训练条件；两个面板为两个种子。后续预测不读未来观测，长度较短轨迹不进入较远步数分母；详细分母在horizon.csv。</p>'
 body+='<h2>原完整轨迹与非目标In回归</h2>'+table(reg,[('model','条件'),('seed','种子'),('current','逐步观测整轨'),('one_step','逐步一步预测整轨'),('free','h0自由整轨'),('free_after2','h2后自由整轨')])+'<img src="full_regression.png">'
 body+='<p>每个整轨分数要求所有报告边界均正确。完整轨迹含原4步与6步、多种分支；观测、一步预测和自由推演分开。中途裁剪没有只训练静态分类头：首状态、之后每步后验、一步先验、至3步自由预测、相对h2后终点和结果预测均有原损失监督。</p>'+table(retain,[('model','条件'),('seed','种子'),('group','原完整/中途'),('both','非目标In稳定区间两端正确'),('forgot','此前正确后遗忘/此前正确机会')])
 body+='''<h2>先澄清旧48/50与六步0/24</h2><p>旧粗网格非目标In的48/50正确区间中，熟悉六步链均48/48；四步冷启动模板分别只有0/48、2/48。全部48/46个错误区间来自起始已有In的四步任务，且无此前正确后遗忘。因此它主要是同一初态失败，不应重复称作另一种遗忘退化。正常六步链主要首错在对象已入抽屉后再请求桌面抓取；带外部落空分支多在未被输入的落空注入处首错。两类错误须分开。</p><p>刚采集的四族已经用于诊断，以下仅为探索回看，不作为新独立证明。正常分支与外部扰动分支分别保留，不能强迫模型从相同公开输入提前预测未输入的扰动。</p>'''
 body+=table(expl,[('model','条件'),('seed','种子'),('template','序列'),('branch','执行分支'),('current','观测整轨'),('free','h0自由整轨')]);body+='<details><summary>六步自由推演的首个有影响错误</summary>'+table(errors,[('model','条件'),('seed','种子'),('branch','1正常/2扰动'),('first_step','首错步'),('relations','错误关系'),('episodes','轨迹数')])+'</details>'
 body+='<h2>新独立验证</h2>'
 if fresh:
  body+='<p>按训练前冻结的触发条件，两个种子的开发冷启动In召回均提升至少20个百分点且联合初态不下降，故进行24条小规模独立验证：4个新A/B布局族、每族3个H/T循环配对、熟悉与全新四步顺序各12条，只有正常分支。每对顺序从相同物理初态开始，同一动作集合；无训练回填、无阈值调整或结果替换。统计单位只有4个布局族，外推范围有限。</p>'+table(fresh,[('model','条件'),('seed','种子'),('template','顺序'),('initial_In','初态In召回'),('initial_all','初态联合'),('current','观测整轨'),('free','h0自由整轨')])
 else:body+='<p>尚未完成新独立验证，不能根据探索四族宣布新顺序泛化。触发条件结果见independent_validation_trigger.json。</p>'
 body+='<h2>连续关系探针与限制</h2><p>探针只使用原执行器抓取完成阶段第4/8/12控制时刻，无新增hold动作或控制。Held/In正区间须GT在两端之间所有控制时刻连续为真；两端读对率和此前正确后遗忘数均报告。它不证明独立hold动作转移或长期保持。</p>'+table(probes,[(k,k)for k in ['source','arm','seed','template','Held_positive_intervals','Held_both_correct','Held_forgot','In_positive_intervals','In_both_correct','In_previous_correct','In_forgot']])
 body+='''<h2>复现与证据</h2><p>训练：source world_model_env.sh；python source/mid_train.py（原类型错误日志保留），再由source/mid_resume.py保留已完成基线并恢复未完成拟合。评估：source/mid_assess.py；评分：source/mid_score.py。完整基线两个种子均须逐张量复现旧候选。冻结方案、输入哈希、固定训练日程、曝光表、预检、模型、每100步日志和逐预测数组均保留。只有预设step1500用于报告。</p><ul><li><a href="../configs/protocol.json">训练前冻结方案</a></li><li><a href="../results/preflight.json">损失/梯度/无未来初始化检查</a></li><li><a href="../results/exposure.json">完整曝光计数</a></li><li><a href="../results/initial.csv">初始化分母与混淆矩阵</a></li><li><a href="../results/horizon.csv">多步预测分母</a></li><li><a href="../results/first_errors.json">全部首错证据</a></li><li><a href="../results/previous_error_decomposition.json">旧成绩拆解</a></li><li><a href="../results/runtime_dtype_adjustment.json">实现类型修正记录</a></li><li><a href="../results/final_verification.json">最终完整性核验</a></li></ul>'''
 body+='</html>';(S/'reports/REPORT.html').write_text(body);save(S/'results/report_complete.json',{'utc':now(),'figures':['cold_initial','full_regression','cold_rollout'],'colors_and_axes_explained':True,'tables':len(ini)+len(reg)+len(expl)+len(fresh),'independent_completed':bool(fresh)});print('REPORT_WRITTEN',flush=True)
if __name__=='__main__':main()
