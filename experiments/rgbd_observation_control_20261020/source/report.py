from common import *
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import html
def table(headers,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>'for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'for v in row)+'</tr>'for row in rows)+'</table>'
def main():
 guard();a=read(S/'results/analysis.json');summ=a['summaries'];role=[r for r in summ if r['kind']=='role'];splits=[s for s in ['train','development','historical_evaluation','diagnosed_confirmation','fresh_independent']if any(r['split']==s for r in role)];cn={'train':'原训练','development':'原开发','historical_evaluation':'旧历史·探索','diagnosed_confirmation':'已诊断·开发证据','fresh_independent':'新独立测试'};en={'train':'Train','development':'Development','historical_evaluation':'Old exploratory','diagnosed_confirmation':'Diagnosed development','fresh_independent':'Fresh independent'}
 fits=[read(S/f'results/fit_{arm}_seed{seed}.json')for seed in [0,1]for arm in ARMS];missing=[{'split':s,**read(S/f'results/geometry_missing_{s}.json')}for s in splits]
 fig,axs=plt.subplots(2,2,figsize=(14,8.5));metrics=[('h0_held','Initial Held: all 3 objects'),('close_one_step_drawer_correct','Close: one-step drawer prediction'),('close_suffix_drawer_correct','Close: recursive drawer prediction from h2'),('normal_whole','Normal trajectory: all observed relations')]
 for ax,(metric,title)in zip(axs.flat,metrics):
  for arm,color in [('baseline','#4477aa'),('geometry','#ee7733')]:
   for seed,style in [(0,'-'),(1,'--')]:
    vals=[100*next(r[metric]for r in role if r['arm']==arm and r['seed']==seed and r['split']==s)for s in splits];ax.plot(range(len(splits)),vals,style,marker='o'if seed==0 else's',color=color,label=arm+' seed '+str(seed))
  ax.set_title(title);ax.set_ylabel('Correct (%)');ax.set_xticks(range(len(splits)),[en[s]for s in splits],rotation=12,ha='right');ax.set_ylim(-3,103);ax.grid(axis='y',alpha=.2)
 axs[0,0].legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(S/'reports/comparison.png',dpi=160);fig.savefig(S/'reports/comparison.svg');plt.close(fig)
 metrics_table=table(['队列','角色数','组','种子','h0 Held','正常全轨迹','关抽屉一步','关抽屉递归','三步联合','四步终点'],[[cn[r['split']],r['roots'],r['arm'],r['seed'],*[f'{100*r[k]:.1f}%'for k in ['h0_held','normal_whole','close_one_step_drawer_correct','close_suffix_drawer_correct','primary_joint3','terminal']]]for s in splits for r in role if r['split']==s])
 retain_table=table(['队列','组','种子','In遗忘/稳定机会','Held遗忘/稳定机会'],[[cn[r['split']],r['arm'],r['seed'],f"{r['In_forget_count']}/{r['stable_In_opportunities']}",f"{r['Held_forget_count']}/{r['stable_Held_opportunities']}"]for r in role])
 miss_table=table(['队列','对象边界数','固定相机缺失','夹爪相机缺失','双相机都缺失','缺失定义'],[[cn[r['split']],r['object_boundary_opportunities'],f"{r['view_missing']['static']} ({100*r['view_missing']['static']/r['object_boundary_opportunities']:.1f}%)",f"{r['view_missing']['gripper']} ({100*r['view_missing']['gripper']/r['object_boundary_opportunities']:.1f}%)",f"{r['both_views_missing']} ({100*r['both_views_missing']/r['object_boundary_opportunities']:.1f}%)",'无合格可见RGB-D区域；非GT填补']for r in missing])
 depth_rows=[]
 for split in splits:
  path=S/f'results/depth_visibility_breakdown_{split}.json'
  if not path.exists():continue
  for view,c in read(path)['counts'].items():
   n=c['object_opportunities'];depth_rows.append([cn[split],view,n,*[f"{c.get(k,0)} ({100*c.get(k,0)/n:.1f}%)"for k in ['RGB_region_under_6_pixels','missing_camera_depth_or_calibration','visible_RGB_but_under_6_valid_depth_points','visible_depth_span_rejected']]])
 depth_table=table(['队列','相机','对象机会','RGB区域不足','相机/深度/标定缺失','可见区域有效深度不足','跨度质量过滤'],depth_rows)
 anchors=read(S/'results/old_Object_reference.json');anchor_table=table(['队列','种子','旧Object h0 Held','旧Object正常全轨迹','旧Object关闭一步','旧Object关闭递归'],[[cn[r['split']],r['seed'],*[f'{100*r[k]:.1f}%'for k in ['h0_held','normal_whole','close_one_step_drawer_correct','close_suffix_drawer_correct']]]for r in anchors if r['split']in ['development','diagnosed_confirmation']])
 readout=read(S/'results/close_readout_vs_prediction.json');readout_table=table(['种子','条件','抽屉正确 /96','全关系正确 /96'],[[r['seed'],{'current':'看到真实h6后读出','one_step':'真实h5后一步预测','suffix':'h2开始四步预测'}[r['mode']],r['drawer_correct'],r['all_relation_correct']]for r in readout if r['split']=='diagnosed_confirmation'and r['arm']=='geometry'])
 # Exact root changes, including all regressions, are retained for causal interpretation of what stage changed.
 import csv
 roots=list(csv.DictReader(open(S/'results/root_metrics.csv')));case_changes=[]
 for split in splits:
  for seed in [0,1]:
   r={arm:{v['root_id']:v for v in roots if v['split']==split and v['kind']=='role'and v['seed']==str(seed)and v['arm']==arm}for arm in ARMS}
   for rid,b in r['baseline'].items():
    g=r['geometry'][rid]
    for metric in ['h0_held','normal_whole','primary_joint3','terminal']:
     if b[metric]!=g[metric]:case_changes.append({'split':split,'seed':seed,'root_id':rid,'metric':metric,'baseline':int(b[metric]),'geometry':int(g[metric])})
 save(S/'results/changed_cases.json',case_changes)
 contrasts=[v for v in a['contrasts']if v['kind']=='role'and v['seed']=='mean'and v['metric']in ['h0_held','normal_whole','close_one_step_drawer_correct','close_suffix_drawer_correct']]
 ci_table=table(['队列','指标','两种子平均差值','布局族bootstrap 95%区间','族数'],[[cn[r['split']],r['metric'],f"{100*r['difference']:+.1f}pp",f"[{100*r['lo']:+.1f}, {100*r['hi']:+.1f}]pp",r['families']]for r in contrasts])
 fresh='fresh_independent'in splits;conclusion='开发集达到预先冻结的继续条件，已执行新的独立测试。'if fresh else '开发集'+('达到预先冻结的继续条件，但本轮未完成新的独立测试。'if a['supports_fresh_test_under_frozen_rule']else'未达到预先冻结的继续条件，因此停止，不采集新布局。')
 htmltext=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>RGB-D输入对照</title><style>body{{font:16px/1.65 sans-serif;max-width:1300px;margin:28px auto;padding:0 20px;color:#18212b}}table{{border-collapse:collapse;margin:20px 0;font-size:14px}}td,th{{border:1px solid #ccd4dd;padding:6px 9px;text-align:left}}th{{background:#edf2f7}}img{{max-width:100%}}.note{{background:#fff1d2;padding:14px}}</style><h1>RGB-D可观测几何输入对照</h1>
 <p><b>新增几何最明确的收益是状态辨识与观测后的关系恢复；关抽屉后果预测仍不稳定。</b>增强组在原开发、旧历史及已诊断确认集上的h0 Held与正常观测全轨迹均为两种子100%，但关抽屉的开发集预测种子0略降、种子1上升。{conclusion}本轮围绕初始持有身份与关抽屉动作后果；没有优化桌面分类边界，也没有将几何规则变成监督读出或搜索前提效果。</p>
 <p class="note">旧训练门槛和关系确认门槛保持失败原样。本轮是信息输入对照，不能视为完整指南或规划接口验收。历史测试与上轮已诊断确认集都已用于设计，只作探索；所有新增测试失败保留。仿真实例掩码辅助定位条件仍存在。</p>
 <h2>比较中只改变什么</h2>
 {table(['项目','容量对照 baseline','geometry增强','原因'],[['训练模型','Object递归状态 + 两个线性残差投影','完全相同','排除结构/参数量混淆'],['参数','2,490,068','2,490,068','较旧Object均多24,576，约1%'],['新增投影输入','已有冻结对象/场景特征的固定随机压缩','公开RGB-D几何','基线新增层也有有效输入，不是闲置参数'],['机器人自身状态','原15维已进入冻结前端','相同，不重复加入','本轮只增缺失三维信息'],['训练','原1128转移，1500更新×2种子','完全相同数据、顺序、损失、预算','新确认数据不回填训练'],['初始参数/输出','同种子逐字节匹配','同种子逐字节匹配','新增投影零初始化，未预先注入规则'],['预测通路','观测→共享递归状态→符号/动作后果','完全相同','没有观测到标签输出的旁路']])}
 <p>原训练由40个core根的480转移与54个角色根的648转移组成。共4次固定训练、6000更新；训练耗时合计{sum(r['seconds']for r in fits)/60:.2f}分钟，最大显存分配{max(r['gpu_peak_bytes']for r in fits)/1024**2:.1f}MiB。DINO及旧前端冻结并复用缓存；两种子前端不同，但同种子的两组完全共用。</p>
 <h2>信息来源与缺失</h2><p>每个对象每个相机输入：可见点中位三维坐标相对当前TCP、可见90%尺寸、像素数、可见性。场景使用每相机2×2区域的可见点中位三维坐标相对TCP与有效比例。只读取当前边界最后一帧；世界坐标由深度和相机标定反投影。RGB仅用于已有颜色路由，实例数值ID不进入网络。没有真物体位姿、接触、Held/In标签、执行结果、故障标志或未来观测进入特征。</p>
 <p>真实部署需要RGB-D相机、相机内外参、机器人TCP读数及可见实例区域估计。本实验实例区域来自渲染器，尚未验证真实分割器；因此不宣称端到端部署条件已满足。原机器人输入包括TCP位置/姿态、夹爪宽度、七关节角及夹爪命令；新增支路仅借TCP定义相对几何，不重新输入这些原始状态。</p>{miss_table}
 <p>分母是所有真实边界×3对象，不含补齐序列。每视角至少6个有效颜色/实例像素且可见90%跨度≤0.13米，否则记缺失；这是已有公开区域提取条件。只用训练边界计算归一化统计，缺失块归一化后仍置零，并保留可见性；不做GT补齐。</p>{depth_table}<p>“RGB区域不足”只说明公开颜色/实例区域不足6像素，不用私有真值区分遮挡与颜色提取失败。“可见区域有效深度不足”单独统计有RGB区域但不足6个有效深度点的情况；它与对象不可见不是同一个原因。</p>
 <h2>改善发生在哪一步</h2><img src="comparison.png"><p>蓝线=容量对照，橙线=几何增强；实线圆点=种子0，虚线方点=种子1。横轴按数据用途排列，并不是时间趋势；纵轴为正确率。h0 Held要求全部3个对象持有关系正确；关抽屉一步从真实h5推到h6，递归则从真实h2连续推演4步，期间没有未来RGB-D；正常全轨迹要求名义分支全部真实观测边界的Held/In/Drawer都正确。</p>
 {metrics_table}<p>h0、正常轨迹、三步联合及四步终点的分母是表中的角色数。关抽屉两项按该角色的两分支计，因此分母为角色数×2。三步联合要求两分支当前h2和想象h3–h5全部正确；四步终点只检查两分支h6，不是三步联合的子集。Held/In固定0.5阈值，抽屉固定argmax；无阈值选择。原core数据的完整结果保留在summary.csv。</p>
 <h2>状态辨识改善，不等于会预测关闭结果</h2>{readout_table}<p>此表来自已诊断确认集，已属开发证据。看到真实关闭结果后，增强组抽屉读出两种子均96/96；四步预测只有49/96、48/96，且全关系各有额外In错误。这将主要剩余困难定位到动作后果泛化；训练集关闭预测均100%也不能证明新布局可靠。原有35成功、45推闭失败、16转姿态失败的物理轨迹完全相同，没有重新执行或删掉失败。</p>
 <p>具体初始案例 A_relation_confirm_02_Hblue_Tpink（种子0），基线蓝块Held概率0.0045，增强组0.9997；原有视觉路由中蓝块始终可见。它支持公开三维关系信息有用，但同一布局不同T角色初态重复，不能视作独立重复证据。该确认集增强组首错集中到h6：种子0为46条纯抽屉错+1条In/抽屉同时错；种子1为47条纯抽屉错+1条纯In错+1条组合错。</p>
 <h2>跨对象关系保持与正常轨迹退步</h2>{retain_table}<p>遗忘指真实关系动作前后保持为真、动作先验也预测为真，但加入下一观测后错误清除。In按非操作对象计，Held按稳定持有对象计。分母显示所有稳定真关系机会；0次机会不代表该能力通过。本批角色链没有正的稳定Held机会，因此不能凭0/0宣称Held长期保持已验证。逐角色所有改善与退步均记录在changed_cases.json，不只展示成功案例。</p>
 <h2>旧Object参考：排查基线波动</h2>{anchor_table}<p>这些是上一轮已封存模型的既有结果，没有重训或修改。旧Object参数2,465,492，不能代替本轮等容量主对照。旧开发h0 Held原本两种子都是24/24；本轮容量对照种子1变为22/24、增强回到24/24，单看此项不能声称超越旧模型。已诊断确认集旧Object为36/48、48/48，本轮容量对照40/48、46/48，增强均48/48；完整历史和关闭指标一并保留，避免只选较弱基线。</p>
 <h2>不确定性与继续条件</h2>{ci_table}<p>区间以布局族配对重采样2000次，先求两种子差值平均，再重采样族；旧开发只有4个角色布局族，且此区间没有覆盖更多训练种子的波动。开发集平均关闭一步/递归提高7.3/9.4个百分点，但种子0分别下降1/48、3/48，种子1上升8/48、12/48；平均区间为正也不能消除这种种子分歧。</p><p>继续条件先于结果冻结：同一个目标指标在两种子均改善；各种子h0 Held及正常全轨迹退步不超过5个百分点，In遗忘不恶化超过2个百分点。它只是本轮预先设置的独立复核筛选条件，不是用户原指南验收门槛。本轮没有达到同一目标两种子均改善，故没有启动新布局采集；不改门槛、不再训练第二轮，也不把历史探索数据称为独立测试。</p>
 <p>模型仍需学习任务相关状态、动作影响及跨对象关系，并由共同递归状态提供可解释规划接口。本轮新增空间信息可能改善当前辨识，却不保证四步预测、接触不确定性或闭环规划同步可靠；应以本表分别判断。</p>
 <p>配置和证据：../configs/protocol.json、../configs/design_frozen.json、../configs/scoring_frozen.json、../results/input_preflight.json、../results/analysis.json、../results/summary.csv、../results/close_metrics.csv、../results/changed_cases.json。训练日志、固定检查点及输入来源哈希均保留。</p></html>'''
 (S/'reports/REPORT.html').write_text(htmltext)
 save(S/'results/report_complete.json',{'utc':now(),'fresh_test_included':fresh,'summary_rows':len(summ),'all_changed_cases':len(case_changes),'old_gates_unchanged':True})
 print('REPORT_COMPLETE',conclusion,flush=True)
if __name__=='__main__':main()
