from common import *
import html,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def table(headers,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>'for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'for v in r)+'</tr>'for r in rows)+'</table>'
def pct(x):return f'{x*100:.1f}%'
def main():
 guard();verify();a=read(S/'results/analysis.json');summary=a['summaries'];probe=a['probe_summary'];steps=read(S/'results/per_step_metrics.json');family=read(S/'results/family_metrics.json');sem=read(S/'results/semantic_outcomes.json');ret=read(S/'results/retention_opportunities.json');close=read(S/'results/close_metrics.json');audits=read(S/'results/episode_audits.json');col=read(S/'results/collection_complete.json');first=read(S/'results/first_errors.json')
 names={'recovery_seen6':'已见恢复六步·新几何','order_known4':'已见四步·新几何','order_novel4':'新四步顺序·新几何'};modes={'current':'观测后读出','one_step':'逐步一步预测','free_h0':'h0起整轨推演','free_h2':'h2起四步推演'};colors={'baseline':'#4477aa','geometry':'#ee7733'}
 def get(template,mode,seed,arm):return next(r for r in summary if r['template']==template and r['mode']==mode and r['seed']==seed and r['arm']==arm and r['branch']=='all')
 fig,axs=plt.subplots(1,3,figsize=(15,5))
 for ax,(template,title)in zip(axs,names.items()):
  modekeys=['current','one_step','free_h0']+(['free_h2']if template=='recovery_seen6'else[]);x=np.arange(len(modekeys))
  for arm in ARMS:
   for seed in [0,1]:ax.plot(x,[100*get(template,m,seed,arm)['all_whole']for m in modekeys],'-o'if seed==0 else'--s',color=colors[arm],label=f'{arm} seed{seed}')
  ax.set_xticks(x,['Observed','1-step each','Free from h0']+(['Free from h2']if template=='recovery_seen6'else[]),rotation=25,ha='right');ax.set_title(template);ax.set_ylabel('Entire relation trajectory correct (%)');ax.set_ylim(-3,103);ax.grid(axis='y',alpha=.2)
 axs[0].legend(fontsize=8);fig.suptitle('Frozen models, independent A/B layouts: all Held / In / drawer relations');fig.tight_layout();fig.savefig(S/'reports/whole_trajectory.png',dpi=160);fig.savefig(S/'reports/whole_trajectory.svg');plt.close(fig)
 fig,axs=plt.subplots(2,3,figsize=(14,8))
 for row,mode in enumerate(['current','free_h0']):
  for colu,template in enumerate(names):
   ax=axs[row,colu];fids=sorted({r['family_id']for r in family});keys=[(a,s)for s in [0,1]for a in ARMS];matrix=np.array([[next(r['all_whole']for r in family if r['template']==template and r['mode']==mode and r['family_id']==f and r['arm']==arm and r['seed']==seed)*100 for f in fids]for arm,seed in keys]);ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues')
   for yy in range(4):
    for xx in range(4):ax.text(xx,yy,f'{matrix[yy,xx]:.0f}',ha='center',va='center',color='white'if matrix[yy,xx]>55 else'black')
   ax.set_xticks(range(4),[f.split('_')[0]+' '+f.rsplit('_',1)[1]for f in fids]);ax.set_yticks(range(4),[f'{arm} s{seed}'for arm,seed in keys]);ax.set_title(template+' / '+mode)
 fig.suptitle('By layout family: whole-trajectory accuracy, shared color scale 0–100%');fig.tight_layout();fig.savefig(S/'reports/families.png',dpi=160);fig.savefig(S/'reports/families.svg');plt.close(fig)
 physical=[r for r in sem if r['arm']=='baseline'and r['seed']==0];physrows=[];recoveryrows=[];prows=[];rrows=[]
 for template in names:
  for branch in [1,2]:
   rr=[r for r in physical if r['template']==template and r['branch']==branch];physrows.append([names[template],branch,len({r['episode_id']for r in rr}),len(rr),sum(r['status']=='success'for r in rr),sum(r['status']=='failed'for r in rr),sum(r['status']=='refused'for r in rr)])
 for seed in [0,1]:
  for arm in ARMS:
   for template in names:
    rr=[r for r in ret if r['arm']==arm and r['seed']==seed and r['template']==template]
    for relation in ['Held','In']:
     ss=[r for r in rr if r['relation']==relation and r['stable_true']];rrows.append([seed,arm,names[template],relation,len(ss),sum(r['prior_positive']for r in ss),sum(r['forgot']for r in ss)])
   for branch in [1,2]:
    rr=[r for r in sem if r['arm']==arm and r['seed']==seed and r['template']=='recovery_seen6'and r['branch']==branch]
    for step in [2,4,5]:
     xs=[r for r in rr if r['step']==step];recoveryrows.append([seed,arm,branch,'h'+str(step),len(xs),sum(r['true_target_Held']for r in xs),sum(r['true_target_In']for r in xs),sum(r['true_H_In']for r in xs),sum(r['current_target_Held_correct']for r in xs),sum(r['one_step_target_Held_correct']for r in xs)])
 for r in probe:
  prows.append([r['seed'],r['arm'],'全部'if r['template']=='ALL'else names[r['template']],f"{r['positive_Held_frame_correct']}/{r['positive_Held_frames']}",f"{r['continuous_Held_both_correct']}/{r['continuous_Held_intervals']}",f"{r['Held_forgot']}/{r['Held_previous_correct_intervals']}",f"{r['positive_complete_Held_runs_all_correct']}/{r['positive_complete_Held_runs']}",f"{r['continuous_H_In_both_correct']}/{r['continuous_H_In_intervals']}",r['all_capture_runs'],r['complete_runs']])
 contrasts=a['contrasts'];closerows=[]
 for seed in [0,1]:
  for arm in ARMS:
   for template in names:
    for mode in ['current','one_step','free_h0']+(['free_h2']if template=='recovery_seen6'else[]):
     rr=[r for r in close if r['seed']==seed and r['arm']==arm and r['template']==template and r['mode']==mode];cc=[]
     for k in range(3):
      xs=[r for r in rr if r['true_drawer']==k];cc.append(f"{sum(r['drawer_correct']for r in xs)}/{len(xs)}")
     closerows.append([seed,arm,names[template],modes[mode],f"{sum(r['drawer_correct']for r in rr)}/{len(rr)}",*cc])
 first_counts=collections.Counter((r['seed'],r['arm'],r['template'],r['mode'],r['step'])for r in first)
 orders=[]
 for seed in [0,1]:
  for arm in ARMS:
   for mode in ['current','one_step','free_h0']:
    old=get('order_known4',mode,seed,arm);new=get('order_novel4',mode,seed,arm);orders.append([seed,arm,modes[mode],pct(old['all_whole']),pct(new['all_whole']),f"{100*(new['all_whole']-old['all_whole']):+.1f}pp"])
 text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>冻结模型关系泛化检验</title><style>body{{font:16px/1.65 sans-serif;max-width:1400px;margin:25px auto;padding:0 22px;color:#172331}}table{{border-collapse:collapse;font-size:13px;margin:15px 0}}td,th{{border:1px solid #cbd5df;padding:5px 8px}}th{{background:#eef3f7}}img{{max-width:100%}}.note{{padding:16px;background:#fff1cf}}details{{margin:20px 0}}code{{overflow-wrap:anywhere}}</style>
 <h1>冻结模型的任务关系泛化检验</h1><p><b>四个模型权重、阈值及归一化全部固定，训练更新为0。</b>完成{col['episodes']}条轨迹、{col['task_calls']}次请求技能调用；{col['setup_valid']}/{col['episodes']}初态满足预定关系。对照为同容量无新增几何Object与RGB-D粗网格Object，各两个种子。这里报告验证证据，不把候选版预先宣称为获胜，也不重新采用local_gap。</p>
 <p class="note">新几何与新顺序分开报告。所有关闭失败、抓取落空、拒绝执行与无效初态均留在完整评分中。不存在hold动作词，因此持续Held只验证现有抓取完成阶段的连续观测读出，不验证独立保持技能或长时间持物转移。旧失败门槛保持原样，本轮不是完整原指南或TS/LTLf闭环验收。</p>
 <h2>范围、模型与成本</h2>{table(['元素','冻结设置','原因/限制'],[['布局','4个新族，A/B各2；每族red→blue、blue→pink、pink→red三组H/T','3组覆盖全部颜色的H/T角色，未覆盖反向三组；族是独立抽样单位'],['对照','baseline与geometry均2,490,068参数，各seed0/1、step1500已封存权重','baseline仍有相同投影容量，输入为旧冻结特征的固定随机压缩；无新增显式RGB-D几何'],['可见信息','原冻结DINO、机器人15维、颜色路由保持；geometry为原对象16维和粗网格32维','模型只读当前四帧公共RGB-D/掩码/机器人状态，无私有姿态接触或结果'],['隔离','排除670条旧几何记录及本轮已接纳族；20mm全对象/任意颜色置换近邻拒绝','只能按几何可执行条件筛选；采集后无成绩驱动替换'],['执行器','未修改PhysicalOracle；动作只有open、close、lift指定对象、place','不增加新技能或大控制器；不输入私有执行判据到模型'],['预算','45分钟硬截止，采集留7分钟给缓存、评估与报告','最多72轨迹/336任务调用；setup另行记录，不混入任务调用数']])}
 <h2>如何区分几何与顺序</h2>{table(['模板','初态/建立方式','请求顺序','解释'],[['recovery_seen6','真实执行open、lift H，初态H已持有','P,L(T),P,L(T),P,C','实际训练见过的六步顺序，新几何；落空分支在第一次L(T)覆盖闭爪命令，后续原有L(T)恢复'],['order_known4','真实执行open、lift H、P，H已在抽屉且空手','L(T),P,C,O','三种目标各有4条相同训练顺序；新几何与关系上下文'],['order_novel4','与known4完全同一物理初态','C,O,L(T),P','相同动作集合，完整/连续四步顺序均未见；基本动作和部分短片段已知']])}<p>O/C/P/L分别为打开、关闭、放入抽屉、抓取指定桌面对象。两个四步模板在同族同H/T下共享精确初态；新顺序会改变后续接触结果，这是序列执行的实际影响，完整保留。它不是所有短片段都未见的强分布外检验。六步与四步长度/初态不同，不直接混作顺序效应。</p>
 {table(['模板','分支','轨迹','调用','执行成功','执行失败','拒绝'],physrows)}<p>分支1为名义执行，分支2为首个目标抓取的闭爪命令被保持打开。拒绝表示执行器未进入动作，不等于真正完成了技能；不删掉它，也不以此充当稳定持物动作。初态建立调用不在上表任务调用分母内，所有setup日志保留。</p>
 <h2>观测读出与无未来观测推演</h2><img src="whole_trajectory.png"><p>蓝线=baseline，橙线=geometry；圆点实线=seed0，方点虚线=seed1。每个面板是一种模板；横轴是评估方式，不是时间趋势，纵轴为整条关系轨迹全部正确的比例。Observed使用每步真实观测；1-step each每次从实际前一步状态预测；Free from h0从初态推完整4或6步；Free from h2仅用于恢复六步的后4步，已观察抓取成功或落空前缀。所有自由推演均不读取未来观测。</p>
 {table(['模板','种子','组','方式','轨迹','Held整轨','In整轨','Drawer整轨','三关系整轨','逐边界三关系'],[[names[r['template']],r['seed'],r['arm'],modes[r['mode']],r['episodes'],*[pct(r[k])for k in ['Held_whole','In_whole','drawer_whole','all_whole','all_step_accuracy']]]for r in summary if r['branch']=='all'])}
 <p>Held/In分别要求三个物体关系全部正确，drawer为三个状态argmax。整轨要求所有对应边界通过；观测整轨含h0。真值无效的边界计入分母并标记，不靠删除提高成绩。各关系固定阈值0.5，不重新校准。</p>
 <h2>相同初态与动作集合的新顺序</h2>{table(['种子','组','方式','已见四步整轨','新四步整轨','新−已见'],orders)}<p>此表才是本轮的顺序对比；两边来自相同布局族、角色和分支。每步的动作位置不同，因此单步结果按各自步骤与技能解释，不能把两个模板同一个h编号当成同一物理任务。</p>
 <h2>补上稳定Held正样本分母</h2>{table(['种子','组','模板','正Held帧读对','连续Held区间两端读对','前端读对后遗忘','完整正Held段全读对','连续非目标H-In两端读对','抓取完成段数','三帧完整段数'],prows)}
 <p>只在原执行器已有verify_stable_completion的控制4、8、12保存公共四帧窗口；没有新增保持控制、没有新动作标签。每个窗口独立使用同一个抓取动作先验再加入该窗口观测；三个后验不相互递推，也不插入主轨迹。连续Held区间分母要求目标对象在4→8或8→12的每个控制时刻真值均Held；每个区间两端都读对才计保持正确。非目标H-In同理。完整段要求三帧都读对。它们是同一抓取中的相关样本，不能视为独立几何重复；更长时间或移动中的保持尚未覆盖。失败抓取没有完成阶段时记录为缺少该阶段，不伪造Held正样本。</p>
 <h2>跨技能关系保持与恢复</h2>{table(['种子','组','模板','关系','真实稳定正机会','先验也为正机会','观测后遗忘数'],rrows)}<p>技能边界遗忘要求真实关系前后稳定为真、先验仍为真、加入新观测后却变假；In只统计非操作对象。0正机会表示未覆盖，独立的完成阶段表给出本轮新增Held正机会。</p>
 {table(['种子','组','分支','边界','轨迹','真实T Held','真实T In','真实非目标H In','观测T Held读对','一步T Held预测对'],recoveryrows)}<p>恢复六步h2是首次目标抓取之后，h4是再次目标抓取之后，h5是正常放置之后。分支2的h2→h4检验失败后恢复；分支1后续重复请求可能因目标已入抽屉而被拒绝，也照实评分。语义真值与模型正确性分列，避免把模型准确预测失败说成任务完成。</p>
 <h2>布局族差异与不确定性</h2><img src="families.png"><p>每格为一个布局族的整轨正确率，数字与统一蓝色深浅表示0–100%。列为A/B环境的两个族，行为模型/种子；上排观测读出，下排h0起自由推演。全部族均展示。每族同模板含3角色×2分支，共6轨迹。</p>
 {table(['种子','模板','方式','geometry−baseline整轨','族配对95%区间','族数'],[[r['seed'],names[r['template']],modes[r['mode']],f"{100*r['difference']:+.1f}pp",f"[{100*r['lo']:+.1f}, {100*r['hi']:+.1f}]pp",r['families']]for r in contrasts])}<p>同族配对bootstrap2000次，只有4个族，区间只描述该小队列，不覆盖更广环境和更多训练种子的变化；不把统计区间改成新的通过门槛。</p>
 <h2>关闭失败单列，仍属于完整成绩</h2>{table(['种子','组','模板','方式','抽屉正确/全部close','真实closed正确','真实partial正确','真实open正确'],closerows)}<p>三种状态不在本轮强行等同某个物理失败机制；任务执行状态另见上表与close_metrics.csv。这里保留close的所有错误，不再展开局部关闭特征搜索。</p>
 <details><summary>查看全部逐步Held/In/Drawer与联合成绩</summary>{table(['模板','种子','组','方式','h','边界数','Held正确','In正确','drawer正确','联合正确','无效GT'],[[names[r['template']],r['seed'],r['arm'],modes[r['mode']],r['step'],r['n'],r['Held_correct'],r['In_correct'],r['drawer_correct'],r['all_correct'],r['invalid_GT']]for r in steps if r['branch']=='all'])}</details>
 <details><summary>查看首次错误分布</summary>{table(['种子','组','模板','方式','首错h','轨迹数'],[[seed,arm,names[temp],modes[mode],step,n]for(seed,arm,temp,mode,step),n in sorted(first_counts.items())])}<p>逐轨迹首次错误概率保存在first_errors.json；不针对这些测试案例追加训练或替换布局。</p></details>
 <p>机器可读结果：<a href="../results/analysis.json">总体与冻结模型检验</a>、<a href="../results/per_step_metrics.csv">每步含分支</a>、<a href="../results/episode_metrics.csv">每条整轨</a>、<a href="../results/family_metrics.csv">全部布局族</a>、<a href="../results/probe_metrics.csv">Held连续观测逐项</a>、<a href="../results/semantic_outcomes.csv">恢复/放置语义真值</a>、<a href="../results/retention_opportunities.csv">保持分母</a>、<a href="../results/close_metrics.csv">全部关闭结果</a>、<a href="../configs/protocol.json">冻结方案</a>、<a href="../results/action_sequence_audit.json">训练顺序审计</a>。源码、配置、数据、缓存、日志和图表分目录保存；旧检查点只读引用。</p></html>'''
 (S/'reports/REPORT.html').write_text(text)
 save(S/'results/report_complete.json',{'utc':now(),'figures':['whole_trajectory.png','families.png'],'all_outcomes_retained':True,'held_scope':'observed lift-completion continuity only','no_training_or_model_selection':True});print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
