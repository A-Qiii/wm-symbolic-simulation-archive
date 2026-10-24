from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ast,html,re,base64
COL={'prior':'#999999','continuous':'#3269b0','reset':'#d58b26'}
EN={'prior':'Before current observation','continuous':'Continuous posterior','reset':'Single-boundary reset'}
def rows(n):return list(csv.DictReader((S/'results'/n).open()))
def table(h,rr):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|'for r in rr])
def relation(p):
 z=hard(p);return '持有'+('/'.join(COLORS[i]for i in range(3)if z[i])or'空')+'；入屉'+('/'.join(COLORS[i]for i in range(3)if z[i+3])or'无')+'；'+['关','半开','开'][int(z[6:9].argmax())]
def main():
 sys.addaudithook(guard);metrics=rows('current_metrics.csv');pairs=rows('current_pairs.csv');forecasts=rows('forecast_metrics.csv');memory=rows('memory_retention.csv');extras=rows('extra_relations.csv');records=read(S/'data/records.json');cases=read(S/'results/cases.json');align=read(S/'results/alignment_summary.json');comp=read(S/'results/compute_complete.json');figs=[]
 def cm(seed,cond,field='all_relations',cohort='all'):return next(r for r in metrics if(r['seed'],r['condition'],r['field'],r['cohort'],r['scene'],r['split'])==(str(seed),cond,field,cohort,'all','all'))
 def fm(seed,cond,mode,h):return next(r for r in forecasts if(r['seed'],r['condition'],r['mode'],r['horizon'],r['field'],r['origin_previous_status'])==(str(seed),cond,mode,str(h),'all_relations','all'))
 def finish(f,n):f.tight_layout(pad=2);f.savefig(S/'figures'/n,dpi=160,bbox_inches='tight');plt.close(f);figs.append(n)
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for seed in [0,1]:
  for j,cond in enumerate(COL):
   rr=[cm(seed,cond,f)for f in FIELDS];axes[seed].bar(np.arange(4)+(j-1)*.25,[float(r['root_mean'])*100 for r in rr],width=.25,color=COL[cond],label=EN[cond])
  axes[seed].set(xticks=range(4),xticklabels=['Held identity','InDrawer set','Drawer','All relations'],ylim=(0,105),ylabel='Root-mean accuracy (%)',title=f'seed {seed}: 240 observed boundaries / 20 roots');axes[seed].legend(fontsize=8,loc='lower left')
 finish(fig,'current_accuracy.png')
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for seed in [0,1]:
  r=next(r for r in pairs if(r['seed'],r['scene'],r['split'],r['cohort'],r['field'])==(str(seed),'all','all','all','all_relations'));values=[int(r['corrected']),int(r['regressed'])];axes[seed].bar([0,1],values,color=['#008c78','#c85555']);axes[seed].set(xticks=[0,1],xticklabels=['Corrected','Regressed'],ylabel='Observed boundary count',ylim=(0,170),title=f'seed {seed}: paired changes')
  for i,v in enumerate(values):axes[seed].text(i,v+3,str(v),ha='center')
 finish(fig,'correction_regression.png')
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for seed in [0,1]:
  rr=[next(r for r in memory if(r['seed'],r['field'],r['visibility'],r['stable_only'])==(str(seed),'InDrawer',v,'True'))for v in ['visible','occluded']]
  for j,cond in enumerate(['continuous','reset']):
   vals=[100*int(r['continuous_TP'if cond=='continuous'else'reset_TP'])/int(r['n'])for r in rr];axes[seed].bar(np.arange(2)+(j-.5)*.3,vals,width=.3,color=COL[cond],label=EN[cond])
   for i,(v,r)in enumerate(zip(vals,rr)):axes[seed].text(i+(j-.5)*.3,v+2,f'{r["continuous_TP"if cond=="continuous"else"reset_TP"]}/{r["n"]}',ha='center',fontsize=9)
  axes[seed].set(xticks=[0,1],xticklabels=['Visible in window','Not visible in window'],ylim=(0,110),ylabel='Retained InDrawer positive recall (%)',title=f'seed {seed}: persistent relation');axes[seed].legend(fontsize=8,loc='upper right')
 finish(fig,'memory_retention.png')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 settings=[('one_step',1,'One next action'),('free_root',4,'Root open-loop h4'),('free_suffix',1,'Suffix h1'),('free_suffix',2,'Suffix h2'),('free_suffix',3,'Suffix h3')]
 for seed in [0,1]:
  for j,cond in enumerate(['continuous','reset']):
   rr=[fm(seed,cond,m,h)for m,h,_ in settings];v=[float(r['root_mean'])*100 for r in rr];axes[seed].bar(np.arange(5)+(j-.5)*.3,v,width=.3,color=COL[cond],label=EN[cond])
  axes[seed].set(xticks=range(5),xticklabels=[x[2]for x in settings],ylim=(0,105),ylabel='Task-relation exact (%)',title=f'seed {seed}: future forecasts');axes[seed].tick_params(axis='x',rotation=15);axes[seed].legend(fontsize=8,loc='lower left')
 finish(fig,'future_forecasts.png')
 for ci,c in enumerate(cases):
  d=dict(np.load(G/f'data/dataset_{c["split"]}.npz'));e=int(np.flatnonzero(d['episode_id']==c['episode_id'])[0]);ix={(r['condition'],r['mode'],r['target']):r for r in c['rows']if r['mode']in ['current','prior_current']};fig,axes=plt.subplots(2,5,figsize=(16,6.8),gridspec_kw={'height_ratios':[1.5,1]})
  for t in range(5):
   f=np.load(str(d['public_frame'][e,t]));axes[0,t].imshow(f['rgb_static']);axes[0,t].axis('off');axes[0,t].set_title('Initial window'if t==0 else SKILLS[int(d['action'][e,t-1,:6].argmax())]+'\n'+STATUS[int(d['status'][e,t-1])],fontsize=9);prior=ix['continuous','current',0]['task_prob']if t==0 else ix['prior','prior_current',t]['task_prob'];p=[d['y'][e,t],prior,ix['continuous','current',t]['task_prob'],ix['reset','current',t]['task_prob']];z=hard(np.asarray(p));a=np.c_[z[:,:6],z[:,6:9].argmax(-1)/2];axes[1,t].imshow(a,cmap='Blues',vmin=0,vmax=1,aspect='auto');axes[1,t].set(xticks=range(7),xticklabels=['H-R','H-B','H-P','I-R','I-B','I-P','Door'],yticks=range(4),yticklabels=['GT','Before observation','Continuous posterior','Single-boundary reset']if t==0 else[],xlabel=f'boundary {t}')
   for y in range(4):
    for x in range(7):axes[1,t].text(x,y,['C','P','O'][int(z[y,6:9].argmax())]if x==6 else int(a[y,x]),ha='center',va='center',color='white'if a[y,x]>.7 else'black',fontsize=9)
  fig.suptitle(c['episode_id']+f' | seed {c["seed"]} | '+c['case_type']);c['figure']=f'case_{ci:02d}.png';finish(fig,c['figure'])
 save(S/'results/cases_with_figures.json',cases)
 md=['# 真实观察后的RSSM后验：保留历史与单边界重置','结论：直接重置不是可用修复。它能消除目标失败链种子1的错误关系，却使全20根大量遗忘真实持有/抽屉归属并新增持有误报。当前观察已实际进入后验，未发现旧帧、标签错位或更新顺序错误。','本轮固定全部权重，零训练、零采集、零C/D样本、零上传。20个A/B根均未用于相应模型训练，但已被历史探索检查过；60分支、240转移，两个种子分开计数。','## 图像与标签是否对齐',table(['检查','实际结果'],[['当前输入路径','300个边界的四帧窗口与缓存索引逐项一致'],['动作后最后帧','RGB、深度、分割、机器人状态及相机参数与final_public逐数组相同'],['控制时序','198个执行动作的末四帧机器人状态与private trace最后4个控制步完全相同'],['拒绝动作','42个0控制步请求合法保留前一窗口，状态未变'],['标签','各边界物体坐标与endpoint_info一致；执行动作后与trace末样本一致'],['关键失败链原图重编码','5个窗口的冻结DINO特征与缓存全部相同；两种子原4分支批次共40个E边界逐值复现'],['原评估复现','连续条件3000行原预测的全部数组完全相同'],['时间戳限制','原文件没有逐帧墙钟时间戳；用控制序号与精确状态对齐，没有虚构时间']]),'失败抓取的窗口对应局部控制步413–416；最后RGBD/机器人状态属于实际失败后。静态与夹爪画面相对动作前发生变化。后续0步拒绝重复这一窗口是记录定义，不是旧帧传输错误。','## 两种条件究竟改变什么','连续条件：原先验先吸收动作，随后observe接收当前E，保持deterministic h，更新z。额外保留吸收当前观察之前的原生先验作为灰色参照。','重置条件：在每个真实边界独立调用同一个model.initial(E_current)，h置零，z从同一后验网络的新条件分布抽样。当前E仍包含同一四帧、双相机窗口内的信息，只丢弃更早递归记忆；不会把一次重置写入后面边界的连续历史。重置同时改变h和z条件分布，也把中间状态放进训练时用于根初态的初始化路径，因此不能定位唯一模块，也不能把所有退化都归为遮挡记忆丢失。','8个样本、原随机种子81831、批量4、BF16和matmul high固定。连续/重置后验使用对应的同一GPU随机数状态；分布不同可抽到不同类别，不强制类别相同。一步转移共享原转移随机数；后缀使用预先保存的配对种子规则。比较完成后恢复随机流，连续原预测完全复现。','## 当前关系纠正与新增错误']
 data=[]
 for seed in [0,1]:
  for cond in COL:data.append([seed,cond]+[f'{cm(seed,cond,f)["count"]}/{cm(seed,cond,f)["n"]}'for f in FIELDS])
 md +=[table(['种子','条件','持有身份','抽屉归属集合','抽屉开合','全部关系'],data),'![当前正确率](figures/current_accuracy.png)','横轴为关系类型，纵轴为每根平均正确率%；灰色是观察前先验、蓝色连续后验、橙色单边界重置。每种子240个动作后边界，只有20个独立根；不含h0。初态h0作为单独控制，两条件概率逐值相同。','![纠正与退化](figures/correction_regression.png)','横轴为连续条件错误被修正、连续条件正确却被破坏两类；纵轴为边界计数，绿色纠正、红色退化。种子0纠正0/16原错误、破坏150/224原正确；种子1纠正4/19、破坏142/221。4个修正全部来自同一个B_gap_04根，不能称作4个独立成功案例。']
 md.append(table(['种子','条件','额外持有误报数','额外入屉误报数','持有漏报数','入屉漏报数'],[[seed,cond]+[next(r['count']for r in extras if(r['seed'],r['condition'],r['cohort'],r['field'])==(str(seed),cond,'all',field))for field in ['extra_held','extra_in','missing_held','missing_in']]for seed in [0,1]for cond in ['continuous','reset']]))
 md+=['这些是240边界×3对象上的关系位计数，不是独立根数。重置把额外入屉误报清零，同时漏掉全部123个真实入屉关系；不能把误报减少单独当成功。','## 实际失败之后：不是事前预测失败',table(['种子','子集','连续正确/n','重置正确/n','修正原错','破坏原对','根平均变化'],[[r['seed'],r['cohort'],f'{r["continuous_correct"]}/{r["n"]}',f'{r["reset_correct"]}/{r["n"]}',f'{r["corrected"]}/{r["continuous_wrong"]}',f'{r["regressed"]}/{r["continuous_right"]}',f'{100*float(r["root_mean"]):+.1f}百分点']for r in pairs if r['scene']=='all'and r['split']=='all'and r['cohort']in ['after_failure','after_first_failure']and r['field']=='all_relations']), '立即失败后有6个边界、5个独立根；从首次失败开始的后续观察共8个边界、仍是5根。种子1后者按帧看4/8→4/8，但恢复的4帧全在一个根，另4个根各坏1帧；按根平均净下降60个百分点。','B_gap_04种子1在观察到失败后当前关系4/4恢复；种子0最初不再误认持红，但把真实开抽屉判关，之后还误认持粉。下面逐边界保留两种子的差异。']
 failure=[r for r in records if r['episode_id']=='B_coverage_gap_04_b0'and r['mode']in ['current','prior_current']and r['target']>0]
 md.append(table(['种子','边界','条件','持红概率','红入屉概率','持粉概率','开屉概率','关系全对'],[[r['seed'],r['target'],r['condition']]+[f'{r["task_prob"][j]:.6f}'for j in [0,3,2,8]]+[r['correct'][3]]for r in failure]))
 md+=['第一次抓取动作的事前预测来自h0，两条件完全相同，都没有预报这次真实失败。上述修正只能发生在真实失败后的图像到来之后；不能改写为失败预测能力提高。','## 丢弃记忆的代价',table(['种子','关系','可见性','持续关系正例','连续召回','重置召回','丢失原正确','根数'],[[r['seed'],r['field'],r['visibility'],r['n'],r['continuous_TP'],r['reset_TP'],r['lost_positive'],r['roots']]for r in memory if r['stable_only']=='True']), '可见性只用于评分：当前四帧的任一相机中对应对象有至少2个分割像素即为可见，不向模型提供私有对象身份。持续关系指该对象在上一边界和当前都真实持有/在屉内。两种子连续后验保留55/60、53/60个持续入屉关系，重置均为0/60；其中不可见35个、可见25个。可见对象也全部漏掉，说明退化不只是丢失遮挡记忆，还与初始化状态分布变化有关。','![持续入屉](figures/memory_retention.png)','横轴区分窗口内可见与不可见；纵轴是持续InDrawer正例召回，蓝色连续后验、橙色重置。柱上为对象—边界计数，不是独立实验次数；可见组9根，不可见组20根。持续持有正例均可见，也从22/24下降到13/24、12/24。',table(['种子','0步拒绝后的连续正确/n','重置正确/n','修正','退化'],[[r['seed'],f'{r["continuous_correct"]}/{r["n"]}',f'{r["reset_correct"]}/{r["n"]}',r['corrected'],r['regressed']]for r in pairs if r['scene']=='all'and r['split']=='all'and r['cohort']=='after_refusal'and r['field']=='all_relations']),'## 从当前状态预测下一步，与纯开环后缀分开']
 md.append(table(['种子','条件','下一动作','h0纯开环第4步','边界后缀1步','后缀2步','后缀3步'],[[seed,cond]+[f'{fm(seed,cond,m,h)["count"]}/{fm(seed,cond,m,h)["n"]}'for m,h in [('one_step',1),('free_root',4),('free_suffix',1),('free_suffix',2),('free_suffix',3)]]for seed in [0,1]for cond in ['continuous','reset']]))
 md+=['![未来预测](figures/future_forecasts.png)','横轴明确区分从真实当前状态的一步、只从h0出发的纯开环第4步、以及在边界1/2/3分别起算的开环后缀；纵轴为关系全对的根平均比例。蓝色连续、橙色重置。后缀1/2/3步分别有180/120/60条预测，但仍是20根；后缀只在起点吸收一次真实观察，之后绝不读取未来E。h0纯开环概率逐值相同，因此不会凭空获得纠错收益。','目标失败链种子1虽然当前观察被修正，从该重置状态预测下一次放置却错误预测抽屉关闭；纯后缀的前两步仍错，末步才正确。当前关系读对不等于转移已经正确，更不等于执行了新闭环。','## 代表性真实链与第一个改变的边界']
 explanations={'failure_observation_correction':'这是目标B_gap_04失败链的种子1。边界1真实抓取失败、空手且抽屉开；先验和连续后验仍认持红，重置读对。后面的重置当前状态是各边界独立诊断，不能连成“无新观察的成功未来预测”。','lost_drawer_memory':'原连续后验保留真实入屉归属；选中边界的重置丢掉该归属。选中的是入屉归属退化边界；整条链更早的其他错误在下方首错记录和表中保留。','lost_holding_memory':'真实持有对象在当前图像中可见，连续后验正确；重置后读错。它反驳只把重置代价解释为遮挡导致的记忆需求。','refusal_regression':'真实请求被拒绝，0控制步，物理状态和输入窗口合法保持不变；重置却破坏了原正确关系。不是动作后果真的发生变化。','target_failure_no_full_correction':'目标失败链没有完整关系纠正；保留局部变化及新错误，不以个别概率下降冒充修复。'}
 for c in cases:
  first={cond:next((r['target']for r in sorted(c['rows'],key=lambda r:r['target'])if r['mode']=='current'and r['condition']==cond and not r['correct'][3]),None)for cond in ['continuous','reset']}
  md.append(f'当前关系的首错边界：连续={first["continuous"]}，重置={first["reset"]}；None表示整条观察链无错。下面单列本例选中的字段变化边界，不将其冒充更早错误的起点。')
  md +=[f'{c["episode_id"]}，种子{c["seed"]}；选中变化边界{c["first_selected_boundary"]}。',explanations[c['case_type']],'!['+c['case_type']+'](figures/'+c['figure']+')','上排为真实静态相机的边界0–4图像，标题是实际动作与执行结果。下排H-R/H-B/H-P表示红/蓝/粉持有，I-R/I-B/I-P表示入屉，Door的C/P/O为关/半开/开。深蓝表示真或开、浅色表示假或关，与像素颜色无关。行依次是真实标签、吸收当前观察前先验、连续后验、该边界独立重置。h0没有动作先验，灰色参照行显示共同初态。']
  cr=[r for r in c['rows']if r['mode']=='current'];ix={(r['condition'],r['target']):r for r in cr};md.append(table(['边界','真实','连续后验','独立重置'],[[t,relation(ix['continuous',t]['truth']),relation(ix['continuous',t]['task_prob']),relation(ix['reset',t]['task_prob'])]for t in range(5)]))
 md+=['## 回到主线后的决定','直接重置应停止作为修复方向：它对一个根的部分种子有观察后纠错效果，但总体损害巨大。证据支持检查历史状态与观察读出如何共同作用；尚不能断言唯一责任在h、z、后验网络或共享符号头。独立对象头没有作为GT，也没有作为本轮模型输入。','下一轮最小建议是冻结权重做两个互补的读出分量对照：保留连续h仅替换为重置z；保留连续z仅清零h。只检查同20根当前边界的纠错与退化，先分辨本轮联合重置效应，再决定是否需要小规模训练修正。该交叉对照仍可能改变潜状态联合分布，只是定位证据，尚未运行。本轮不追加训练或架构替换。','首次抓取失败的事前预测仍未解决；没有新增TS/LTL闭环运行，也没有完成原指南全部验收。','## 交付与运行记录',f'连续原预测3000行逐数组复现；h0及纯开环根预测控制完全相同。恢复后的配对RSSM运行{comp["seconds"]:.2f}秒、峰值{comp["peak_cuda_GiB"]:.3f}GiB；这个计时/峰值不含首次原图检查，首轮资源峰值没有保存。原图检查共5个DINO窗口、两个原批次空间编码前向，无训练。','首次运行在数据类命名空间缺少collections时停止，发生于像素/E校验通过后、任何配对预测产生前。原源码和失败日志保留；resume.py仅补导入并复用通过的校验，未改协议、随机规则、权重或阈值。源码、协议、逐边界核验、预测、分层CSV、图表和最终清单均保存在独立目录。']
 txt='\n\n'.join(md)+'\n';(S/'REPORT.md').write_text(txt);src=(O/'source/report.py').read_text();node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)and n.name=='render_markdown');ns={'re':re,'html':html};exec(compile(ast.Module(body=[node],type_ignores=[]),'<markdown>','exec'),ns);body=ns['render_markdown'](txt)
 for f in figs:body=body.replace('figures/'+f,'data:image/png;base64,'+base64.b64encode((S/'figures'/f).read_bytes()).decode())
 (S/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>后验历史诊断</title><style>body{max-width:1280px;margin:36px auto;font:16px/1.7 system-ui;color:#203047;padding:0 22px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{border:1px solid #ced8df;padding:6px;overflow-wrap:anywhere}th{background:#e9f3f3}img{max-width:100%}p{overflow-wrap:anywhere}h1,h2{color:#08766b}</style>'+body+'</html>');save(S/'results/report_complete.json',{'utc':now(),'figures':len(figs),'cases':len(cases),'case_roots':len({c['root_id']for c in cases}),'new_forwards':0});save(S/'results/next_action_decision.json',{'utc':now(),'reject_as_fix':'Unconditional posterior reset','proposed_next_only':'Frozen current-boundary h/z cross-components: continuous h+reset z, zero h+continuous z; same20 roots and shared readout, no training, preserve OOD caveat','why':'Joint reset helps one failure root in one seed but causes severe held/InDrawer regressions; cannot isolate h vs z with joint change','implemented':False});print('REPORT_COMPLETE',len(figs),flush=True)
if __name__=='__main__':main()
