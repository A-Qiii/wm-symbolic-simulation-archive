from common import *
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(headers,rows):
 return '<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>'for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'for v in row)+'</tr>'for row in rows)+'</table>'
def pct(x):return f'{100*x:.1f}%'
def main():
 guard();verify();a=read(S/'results/analysis.json');scores=a['group_scores'];summary=a['summary'];splits=['train','development']+(['fresh_independent']if(S/'results/fresh_evaluation_complete.json').exists()else[])
 names={'train':'原训练','development':'原开发·探索','fresh_independent':'新独立布局'};modes={'one_step':'h5→h6一步','suffix':'h2→h6四步'};groups={'success':'真实成功','push_object_bridge':'推物卡住','clearance_joint_limit':'转姿态关节限位','unclassified_fresh':'新样本物理机制未分组'}
 def find(split,seed,arm,mode):return next(r for r in scores if r['split']==split and r['seed']==seed and r['arm']==arm and r['mode']==mode and r['kind']=='role'and r['group']=='ALL')
 def summ(split,seed,arm):return next(r for r in summary if r['split']==split and r['seed']==seed and r['arm']==arm and r['kind']=='role')
 rows=[]
 for split in splits:
  for seed in [0,1]:
   for arm in ARMS:
    for mode in modes:
     r=find(split,seed,arm,mode);rows.append([names[split],seed,arm,modes[mode],f"{r['correct']}/{r['n']}",pct(r['macro_recall']),pct(r['success_recall']),pct(r['jam_recall']),pct(r['limit_recall']),pct(r['majority_baseline_accuracy'])])
 # Paired family resampling describes this exploratory cohort; it does not turn it into independent evidence.
 cases=list(csv.DictReader((S/'results/close_case_metrics.csv').open()));ci=[];family_scores=[]
 for split in splits:
  for seed in [0,1]:
   for arm in ARMS:
    for mode in modes:
     xs=[r for r in cases if r['kind']=='role'and r['split']==split and int(r['seed'])==seed and r['arm']==arm and r['mode']==mode]
     for fid in sorted({r['family_id']for r in xs}):
      rr=[r for r in xs if r['family_id']==fid];conf=np.zeros((3,3),int)
      for r in rr:conf[int(r['true_class']),int(r['pred_class'])]+=1
      counts=conf.sum(1);rec=np.diag(conf)/np.maximum(counts,1)
      family_scores.append({'split':split,'seed':seed,'arm':arm,'mode':mode,'family_id':fid,'n':len(rr),'truth_success':int(counts[0]),'truth_jam':int(counts[1]),'truth_limit':int(counts[2]),'success_correct':int(conf[0,0]),'jam_correct':int(conf[1,1]),'limit_correct':int(conf[2,2]),'correct':int(np.trace(conf)),'success_recall':float(rec[0])if counts[0]else None,'jam_recall':float(rec[1])if counts[1]else None,'limit_recall':float(rec[2])if counts[2]else None,'macro_recall':float(rec[counts>0].mean()),'prediction_counts':conf.sum(0).tolist()})
 csvwrite(S/'results/family_close_metrics.csv',family_scores);save(S/'results/family_close_metrics.json',family_scores)
 def metric(xs,k):
  conf=np.zeros((3,3),int)
  for r in xs:conf[int(r['true_class']),int(r['pred_class'])]+=1
  rec=np.diag(conf)/np.maximum(conf.sum(1),1)
  return float(rec[0]if k=='success_recall'else rec[conf.sum(1)>0].mean())
 for split in splits[1:]:
  for seed in [0,1]:
   for mode in modes:
    xx={arm:[r for r in cases if r['kind']=='role'and r['split']==split and int(r['seed'])==seed and r['arm']==arm and r['mode']==mode]for arm in ARMS};fids=sorted({r['family_id']for r in xx['coarse']});rng=np.random.default_rng(261020900)
    samples=rng.integers(len(fids),size=(2000,len(fids)))
    grouped={arm:{fid:[r for r in xx[arm]if r['family_id']==fid]for fid in fids}for arm in ARMS}
    for k in ['macro_recall','success_recall']:
     ds=[]
     for ids in samples:
      vs={arm:metric([r for i in ids for r in grouped[arm][fids[i]]],k)for arm in ARMS};ds.append(vs['local']-vs['coarse'])
     ci.append({'split':split,'seed':seed,'mode':mode,'metric':k,'difference':metric(xx['local'],k)-metric(xx['coarse'],k),'lo':float(np.quantile(ds,.025)),'hi':float(np.quantile(ds,.975)),'families':len(fids),'resamples':2000})
 save(S/'results/paired_family_intervals.json',ci)
 fig,axs=plt.subplots(2,2,figsize=(12,8));colors={'coarse':'#4477AA','local':'#EE7733'}
 for row,mode in enumerate(modes):
  for col,k in enumerate(['macro_recall','success_recall']):
   ax=axs[row,col];x=np.arange(2)
   for arm,offset in [('coarse',-.18),('local',.18)]:
    vals=[100*find('development',seed,arm,mode)[k]for seed in [0,1]];bs=ax.bar(x+offset,vals,.34,color=colors[arm],label=arm)
    for b,v in zip(bs,vals):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}',ha='center',fontsize=9)
   ax.set_xticks(x,['Seed 0','Seed 1']);ax.set_ylim(0,108);ax.set_ylabel('Recall (%)');ax.set_title(('One-step h5 to h6'if mode=='one_step'else'Four-step h2 to h6')+' / '+k.replace('_',' '));ax.grid(axis='y',alpha=.2)
 axs[0,0].legend();fig.suptitle('Exploratory development: same 48 episodes from 4 layout families');fig.tight_layout();fig.savefig(S/'reports/comparison.png',dpi=160);fig.savefig(S/'reports/comparison.svg');plt.close(fig)
 fig,axs=plt.subplots(2,4,figsize=(15,7));labels=['Closed','Partial','Open']
 for seed in [0,1]:
  for j,(mode,arm)in enumerate([(m,a)for m in modes for a in ARMS]):
   r=find('development',seed,arm,mode);conf=np.array(r['confusion']);ax=axs[seed,j];im=ax.imshow(conf,cmap='Blues',vmin=0,vmax=28)
   for yy in range(3):
    for xx in range(3):ax.text(xx,yy,str(conf[yy,xx]),ha='center',va='center',color='white'if conf[yy,xx]>14 else'black')
   ax.set_xticks(range(3),labels,rotation=25);ax.set_yticks(range(3),labels);ax.set_xlabel('Predicted drawer class');ax.set_ylabel('True drawer class');ax.set_title(f'Seed {seed} / {arm}\n'+('1-step'if mode=='one_step'else'4-step'))
 fig.suptitle('Development confusion: counts, same color scale 0–28');fig.tight_layout();fig.savefig(S/'reports/confusion.png',dpi=160);fig.savefig(S/'reports/confusion.svg');plt.close(fig)
 # Locate the earliest relation error on current h0..h2 then free h3..h6; do not repair any case.
 first=read(S/'results/first_errors.json');first_counts=[]
 for seed in [0,1]:
  for arm in ARMS:
   xs=[r for r in first if r['kind']=='role'and r['split']=='development'and r['seed']==seed and r['arm']==arm];counter=collections.Counter()
   for r in xs:
    p=np.array(r['predicted_relation_probabilities']);y=np.array(r['truth_relations']);bad=[]
    if not((p[:3]>=.5)==y[:3]).all():bad.append('Held')
    if not((p[3:6]>=.5)==y[3:6]).all():bad.append('In')
    if p[6:].argmax()!=y[6:].argmax():bad.append('Drawer')
    counter[(r['mode'],r['step'],'+'.join(bad))]+=1
   for(k,t,bad),n in sorted(counter.items()):first_counts.append({'seed':seed,'arm':arm,'mode':k,'step':t,'relation':bad,'episodes':n})
 save(S/'results/first_error_summary.json',first_counts)
 # Positive Held opportunities are evaluated explicitly; no opportunity is not success.
 held=[]
 for f in sorted((S/'results').glob('predictions_*.npz')):
  z=dict(np.load(f));p=rel(z['prob']);y=truth(z['cat']);n=forgot=0
  for eid in set(z['episode_id'][z['kind']=='role']):
   ids=np.flatnonzero(z['episode_id']==eid);ix={(str(z['mode'][j]),int(z['step'][j])):j for j in ids}
   for t in range(1,7):
    prev=ix['current',t-1];cur=ix['current',t];prior=ix['one_step',t];stable=y[prev,:3].astype(bool)&y[cur,:3].astype(bool);n+=int(stable.sum());forgot+=int((stable&(p[prior,:3]>=.5)&(p[cur,:3]<.5)).sum())
  held.append({'split':str(z['split'][0]),'arm':str(z['arm'][0]),'seed':int(z['seed'][0]),'stable_true_Held_opportunities':n,'forget_count':forgot})
 save(S/'results/Held_retention.json',held)
 fits=[read(S/f'results/fit_{arm}_seed{seed}.json')for seed in [0,1]for arm in ARMS];repro=read(S/'results/coarse_reproduction.json');fresh=(S/'results/fresh_evaluation_complete.json').exists();passed=a['supports_independent_test'];conclusion=('两种子均通过冻结的开发筛选条件，已完成新独立布局复核。'if fresh else'两种子均通过冻结的开发筛选条件，但本轮没有完成新独立布局复核。')if passed else'未同时通过两种子的冻结筛选条件；不采集新布局，不再调整本轮描述或重复训练。'
 missing=[{'split':s,**read(S/f'results/missing_{s}.json')}for s in splits];group_rows=[[names[r['split']],r['seed'],r['arm'],modes[r['mode']],groups.get(r['group'],r['group']),r['n'],f"{r['correct']}/{r['n']}"]for r in scores if r['kind']=='role'and r['group']!='ALL'];retrows=[]
 for s in splits:
  for seed in [0,1]:
   for arm in ARMS:
    r=summ(s,seed,arm);h=next(v for v in held if v['split']==s and v['seed']==seed and v['arm']==arm);retrows.append([names[s],seed,arm,f"{round(r['h0_held']*r['roots'])}/{r['roots']}",f"{round(r['normal_whole']*r['roots'])}/{r['roots']}",f"{r['In_forget_count']}/{r['stable_In_opportunities']}",f"{h['forget_count']}/{h['stable_true_Held_opportunities']}",pct(r['primary_joint3']),pct(r['primary_joint4']),pct(r['terminal'])])
 failed=[r for r in a['checks']if not r['passed']];failedtext='；'.join(f"seed{r['seed']} {r.get('mode','')} {r['metric']}"for r in failed)or'无'
 tradeoffs=[]
 for seed in [0,1]:
  for mode in modes:
   c=find('development',seed,'coarse',mode);l=find('development',seed,'local',mode)
   tradeoffs.append(f"seed{seed} {modes[mode]}：成功召回{100*(l['success_recall']-c['success_recall']):+.1f}pp，卡住召回{100*(l['jam_recall']-c['jam_recall']):+.1f}pp，限位召回{100*(l['limit_recall']-c['limit_recall']):+.1f}pp；预测closed/partial/open从{[c['prediction_class_counts'].get(str(i),0)for i in range(3)]}变为{[l['prediction_class_counts'].get(str(i),0)for i in range(3)]}")
 familytable=table(['队列','族','种子','组','预测','成功正确/真成功','卡住正确/真卡住','限位正确/真限位','宏召回'],[[names[r['split']],r['family_id'],r['seed'],r['arm'],modes[r['mode']],f"{r['success_correct']}/{r['truth_success']}",f"{r['jam_correct']}/{r['truth_jam']}",f"{r['limit_correct']}/{r['truth_limit']}",pct(r['macro_recall'])]for r in family_scores if r['split']!='train'])
 text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>可见局部边界与粗网格对照</title><style>body{{font:16px/1.65 sans-serif;max-width:1300px;margin:28px auto;padding:0 22px;color:#17202b}}table{{border-collapse:collapse;font-size:14px;margin:18px 0}}td,th{{border:1px solid #cbd5df;padding:6px 8px}}th{{background:#edf2f7}}img{{max-width:100%}}.note{{background:#fff0cf;padding:15px}}code{{overflow-wrap:anywhere}}</style>
 <h1>可见局部边界与粗场景网格：固定预算对照</h1><p><b>{conclusion}</b>未通过项：{html.escape(failedtext)}。本轮只检验把32维粗场景输入替换成32维可见局部边界描述的作用。开发集此前已用于提出这个假设，所有开发成绩仅是探索证据。</p><p class="note">原指南及旧实验的失败门槛保持原样。本轮不是完整世界模型、TS/LTLf搜索或闭环规划验收；没有扩大数据规模、修改阈值来宣称通过。旧已诊断确认集不重新称为独立测试。可见边界也不等于完整物理间隙。</p>
 <h2>严格控制的比较</h2>{table(['元素','两组共同条件','本轮唯一差异及原因'],[['模型','相同Object递归结构；各2,490,068参数','共同32维输入投影容量相同'],['冻结输入','原对象/场景512维、availability、对象RGB-D 16维全部相同','只有新增场景32维不同'],['coarse','上一轮geometry组的完整输入','双相机各2×2区域：三维中位点相对TCP+有效比例'],['local','同一当前边界最后一帧双相机RGB-D','两个最近可见对象周围的通用边界距离与缺失标志'],['训练','原1128转移；每组每种子1500更新；种子0/1','初始化、数据、批次顺序、损失、优化器与精度相同'],['归一化','只用原训练边界，排除补齐帧','局部距离仅用有效值估计均值和标准差'],['输入审计','所有非32维场景输入逐字节相同','粗网格输入逐字节等于上一轮geometry输入'],['复现审计',str(all(r['identical']for r in repro['checks'])),'粗网格最终模型所有张量与上一轮对应模型比较']])}
 <p>四次训练共6000更新，训练总计{sum(r['seconds']for r in fits)/60:.2f}分钟，最大显存分配{max(r['gpu_peak_bytes']for r in fits)/1024**2:.1f}MiB。DINO和已有前端缓存复用，GPU串行使用。原训练包含40个core根480转移与54个角色根648转移；角色训练有12个布局族，开发只有4个布局族。</p>
 <h2>哪些物体、坐标和边界进入32维</h2>{table(['元素','精确定义','为何保留/限制'],[['可见物体','当前RGB颜色在匿名可见实例区域内路由；每相机至少6像素、90%可见跨度≤0.13m','实例数值ID不作为特征；仿真实例掩码辅助条件仍在'],['坐标系','相机深度和标定反投影到world/base XYZ','不读私有物体位姿或抽屉真关节'],['边界','4邻域实例边缘或相邻深度跳变>10mm','含遮挡轮廓，不能等同CAD实体边'],['对象选择','可见中心距离最近的一对；按原颜色槽排序','通用几何选择，不读取Held/In或动作结果'],['方向','每个对象−X,+X,−Y,+Y,−Z,+Z','可见点5/95%包围盒；其他两轴范围各扩15mm'],['距离','方向上350mm内至少3个场景边界点；取距离10分位','无点意味着缺失，绝不据此断言自由空间'],['24维','每对象6距离+6有效位，共两对象','缺失距离归一化后置零，有效位仍为0'],['其余8维','对象对onehot3、相对中心XYZ3、中心距离1、对象对有效位1','不构造人工“间隙则成功/失败”谓词'],['场景','所有非物体可见区域，可含机器人','不按私有drawer/body/link编号挑选环境面']])}
 <p>禁止特征：私有接触、真物体姿态、抽屉真实关节位置、执行结果、未来观测。GT只用于监督及评分。局部距离通过共享状态学习动作后果，没有直接到符号标签或搜索前提效果的规则旁路。</p>
 <h2>先检查可见性，再训练</h2><img src="visibility_projection.png"><p>图的每一行是一条真实原数据边界：固定相机、夹爪相机、同一公开点云在Y–Z平面的投影。RGB图保留原颜色；黄线与青线分别对应选中对象对的第一、第二槽，旁边文字标对象颜色，叉号为查询到的前向支持点。右图以同色矩形画可见点5/95%包围盒，线段连接该对象前向面与支持点；线段不是运动轨迹。空间图横轴Y、纵轴Z，单位米，表示同一时刻物体与被查询的可见边界，不表示未来运动。配对训练卡住与开发成功案例用于检验公开几何是否含可见差异，不做新测试。</p>
 <p>全部108条训练、48条开发角色h5边界均能找到对象对及前向支持点；但人工可视复核发现，下层对象的前向点常来自容器表面，不能算作外部间隙完整可见。进一步只利用公开匿名区域归属检查：上层对象外部前向边界在训练108/108、开发48/48可见；按两对象合计则只有113/216、49/96。保持最初描述不变继续此次对照，是因为上层关键外部边界可见；没有补齐遮挡的下侧几何。此结论在训练前保存在visibility_decision.json。</p>
 {table(['队列','真实边界数','对象对缺失','每对象方向有效次数（−X,+X,−Y,+Y,−Z,+Z）'],[[names[r['split']],r['boundaries'],r['pair_missing'],str(r['direction_valid_counts'][:6])+' / '+str(r['direction_valid_counts'][6:])]for r in missing])}<p>每个方向的分母是该行真实边界数，不计序列补齐帧。没有有效深度/边界时保留缺失，不用GT填补。方向有效只说明出现可见支持，不能证明这一方向完整无遮挡。</p>
 <h2>关闭后果：宏平均与成功类不能被多数类掩盖</h2><img src="comparison.png"><p>蓝柱=coarse，橙柱=local；横轴为两个训练种子，纵轴为召回率百分比。上排是真实h5到h6的一步预测；下排从真实h2开始连续想象4步到h6，期间不输入未来RGB-D。左列对三种真实抽屉类别等权宏平均；右列单独看真实成功关闭类。两组使用完全相同的48条开发轨迹，来自4个布局族。</p>
 {table(['队列','种子','组','预测','抽屉正确','宏召回','成功召回','卡住类召回','限位类召回','总预测partial基准'],rows)}<p>类别固定为closed/partial/open。本批角色物理结果与三类对应关系由原独立物理诊断确定：成功关闭、推物卡住、转姿态关节限位；物理机制日志仅用于评分分层，不作为模型输入。总猜partial基准由训练多数类别固定，开发正确率为28/48。宏召回对各出现类别等权，避免只提高多数失败类别。</p>
 <img src="confusion.png"><p>混淆矩阵：纵轴是真实类别，横轴是预测类别；每格数字为轨迹数，蓝色越深数量越多，全部面板统一0–28色阶。每一行对应一个训练种子，列依次为coarse一步、local一步、coarse四步、local四步。它直接显示收益是否只是更偏向partial。</p>
 {table(['队列','种子','组','预测','实际物理组','轨迹数','正确数'],group_rows)}
 <h2>成功与卡住之间的取舍、布局族分布</h2>{''.join('<p>'+html.escape(t)+'</p>'for t in tradeoffs)}<p>三类分别列出；成功提高伴随卡住下降时，不能说所有失败机制都改善。此处仅补充解释，不修改训练前冻结的筛选条件。</p>{familytable}<p>表内0/0表示该族没有该种真实结果，不能算作通过；宏召回只平均该族出现的类别。完整训练族结果也保存在family_close_metrics.csv/json，不挑选有利布局族。四步宏召回在种子0的四个开发族全部上升，在种子1的四个开发族全部下降，因此这里的种子分歧并非只由一个布局族造成。</p>
 <h2>Held、In和正常观测轨迹回归</h2>{table(['队列','种子','组','h0 Held','正常全轨迹','In遗忘/稳定机会','Held遗忘/稳定机会','三步联合','四步联合','四步终点'],retrows)}<p>h0 Held要求三个物体Held都正确，两分支共同通过；正常全轨迹要求名义分支h0–h6所有真实观测后的Held/In/Drawer正确。In遗忘按非操作物体真实前后稳定为真、先验为真但加入观测后错误清除计数。正的稳定Held机会若为0/0，表示未覆盖该能力，不能写成通过。三步联合要求两分支当前h2与想象h3–h5全对；四步联合再加h6，四步终点仅要求两分支h6正确。</p>
 <h2>第一个有影响的错误</h2>{table(['种子','组','状态来源','边界','错关系','轨迹数'],[[r['seed'],r['arm'],r['mode'],'h'+str(r['step']),r['relation'],r['episodes']]for r in first_counts])}<p>每条轨迹按真实观测h0–h2、随后递归h3–h6查找首次错误；同一条轨迹只计一次。完整概率、可见性、绑定以及当时真实观测读出保留在first_errors.json。本轮只有场景32维这一项替换，没有针对单例二次修补或再训练。局部组种子1的最早退步已发生在真实观测h1：A_role_development_00中Hpink的两个T角色、两分支共4条轨迹把仍打开的抽屉读成关闭，closed概率0.540、open概率0.459；这些共享同一前缀，不是4个独立重复实验。B_role_development_01_Hred_Tpink的名义分支在h2把真实持有的粉块Held读成0.0086。三个角色根的正常观测全轨迹因此不再通过。可见本轮还损害了早期状态辨识，不能把全部退步归咎于最后关闭动作。</p>
 <h2>布局族不确定性与停止条件</h2>{table(['队列','种子','预测','指标','local−coarse','布局族配对95%区间','族数'],[[names[r['split']],r['seed'],modes[r['mode']],r['metric'],f"{100*r['difference']:+.1f}pp",f"[{100*r['lo']:+.1f}, {100*r['hi']:+.1f}]pp",r['families']]for r in ci])}<p>区间对同一布局族配对重采样2000次，保留族内全部角色和分支；它只描述这些布局的变化，4个开发族及两个种子不足以支持广泛泛化结论，且开发已参与假设选择。</p><p>训练前冻结的继续条件：两个种子的一步与四步宏召回及成功召回都严格提高；local正确率不低于固定多数类基准；h0 Held与正常全轨迹下降不超过5个百分点；In遗忘率增加不超过2个百分点。{conclusion}不修改这些条件来开启新测试，也不更改旧门槛。</p>
 <p>完整数据：<a href="../results/analysis.json">评分和冻结条件</a>、<a href="../results/group_scores.json">分类别结果</a>、<a href="../results/root_metrics.csv">逐根结果（含core）</a>、<a href="../results/close_case_metrics.csv">逐关闭预测</a>、<a href="../results/first_errors.json">首次错误</a>、<a href="../results/input_preflight.json">输入相等审计</a>、<a href="../results/coarse_reproduction.json">粗网格复现</a>、<a href="../configs/protocol.json">预冻结设计</a>。源代码、配置、缓存、模型和日志分目录保留；最终校验及只读封存在实验根目录。</p></html>'''
 (S/'reports/REPORT.html').write_text(text)
 save(S/'results/report_complete.json',{'utc':now(),'supports_independent_test':passed,'fresh_test_completed':fresh,'no_second_training':True,'charts':['comparison.png','confusion.png','visibility_projection.png']})
 print('REPORT_COMPLETE',conclusion,flush=True)
if __name__=='__main__':main()
