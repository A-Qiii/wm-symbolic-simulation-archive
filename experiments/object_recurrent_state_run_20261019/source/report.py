from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import html
def table(rows,cols):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>'for c in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(c,'')))+'</td>'for c in cols)+'</tr>'for r in rows)+'</tbody></table>'
def main():
 a=read(S/'results/analysis.json');summary=a['summaries'];contrasts=a['contrasts'];fits=[read(p)for p in sorted((S/'results').glob('fit_*.json'))];pre=read(S/'results/preflight.json');colors={'global':'#2675b7','object':'#e37b25'};plt.rcParams.update({'font.size':10});fig,axes=plt.subplots(2,3,figsize=(15,8.7));ax=axes.flat[0]
 for r in fits:
  rr=list(csv.DictReader((S/f"logs/{r['arm']}_seed{r['seed']}_train.csv").open()));loss=np.array([float(x['total'])for x in rr]);smooth=np.convolve(loss,np.ones(50)/50,mode='valid');ax.plot(np.arange(50,len(loss)+1),smooth,color=colors[r['arm']],linestyle='-'if r['seed']==0 else'--',label=r['arm']+' seed'+str(r['seed']))
 ax.set(xlabel='Optimizer updates',ylabel='Common supervised loss (50-update mean)',title='1. Training fit: same objective');ax.legend(fontsize=8);ax.grid(alpha=.2)
 panels=[('train','joint3_actual','2. Training paired current + 3 future steps',False),('historical_evaluation','h0_full','3. Historical h0 complete state',False),('historical_evaluation','normal_whole','4. Historical normal whole trajectory',False),('historical_evaluation','primary_joint3','5. Historical paired current + 3 future steps',False),('historical_evaluation','forget_fixed_denominator_rate','6. Historical non-target InDrawer forgetting',True)]
 for ax,(split,k,title,forget)in zip(axes.flat[1:],panels):
  for j,arm in enumerate(ARMS):
   rr=[next(r for r in summary if r['split']==split and r['kind']=='role'and r['seed']==seed and r['arm']==arm)for seed in [0,1]];vals=[r[k]*100 for r in rr];bars=ax.bar(np.arange(2)+(j-.5)*.32,vals,width=.30,color=colors[arm],label=arm)
   for bar,r,v in zip(bars,rr,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.7,(str(r['forget_count'])+'/'+str(r['stable_In_opportunities']))if forget else f'{v:.1f}%',ha='center',va='bottom',fontsize=8)
  ax.set_xticks([0,1],['Seed 0','Seed 1']);ax.set(title=title,ylabel='Errors / fixed eligible opportunities (%)'if forget else'Family-macro accuracy (%)',ylim=(0,100 if not forget else max(5,ax.get_ylim()[1]*1.4)));ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(S/'reports/comparison.png',dpi=160);fig.savefig(S/'reports/comparison.svg');plt.close(fig)
 checks=[{'数据':r['kind'],'结构':r['arm'],'种子':r['seed'],'指标':r['metric'],'准确率':f"{100*r['value']:.2f}%",'90%门槛':'通过'if r['passed']else'未通过'}for r in a['fit_checks']]
 hist=[{'结构':r['arm'],'种子':r['seed'],'h0完整状态':f"{100*r['h0_full']:.2f}%",'正常全轨迹':f"{100*r['normal_whole']:.2f}%",'成败联合3步':f"{100*r['primary_joint3']:.2f}%",'H-In当前正确':f"{100*r['H_In_h2_post_correct_both']:.2f}%",'保持错误/固定机会':f"{r['forget_count']}/{r['stable_In_opportunities']}",'保持错误/先验正确机会':f"{r['forget_count']}/{r['prior_correct_stable_In']}"}for r in summary if r['split']=='historical_evaluation'and r['kind']=='role']
 ci=[{'指标':r['metric'],'种子':r['seed'],'对象减全局百分点':f"{100*r['difference']:+.2f}",'95%家族重采样区间':f"[{100*r['lo']:+.2f}, {100*r['hi']:+.2f}]"}for r in contrasts]
 runtime=[{'结构':r['arm'],'种子':r['seed'],'更新':r['updates'],'可训练参数':r['parameters'],'训练秒':round(r['seconds'],1),'GPU峰值GiB':round(r['gpu_peak_bytes']/1024**3,3)}for r in fits]
 stages=collections.Counter((r['arm'],r['seed'],r['mode'],r['step'])for r in read(S/'results/first_errors.json')if r['split']=='historical_evaluation');first=[{'结构':k[0],'种子':k[1],'首个错误来源':k[2],'边界':k[3],'分支数':v}for k,v in sorted(stages.items())]
 out=list(csv.DictReader((S/'results/outcome_metrics.csv').open()));outsum=[]
 for arm in ARMS:
  for seed in [0,1]:
   for status in ['success','executed_failure','refused']:
    rr=[r for r in out if r['split']=='historical_evaluation'and r['kind']=='role'and r['arm']==arm and int(r['seed'])==seed and r['status']==status and r['mode']=='one_step']
    outsum.append({'结构':arm,'种子':seed,'真实执行状态':status,'转移数':len(rr),**{k:round(float(np.mean([float(r[k])for r in rr])),5)if rr else ''for k in ['success_brier','success_nll','event_brier','event_nll']}})
 csvwrite(S/'results/outcome_status_summary.csv',outsum)
 verdict='训练拟合门槛未通过；这是固定预算内的优化/拟合不足，不能据此否定整个结构。'if not a['training_fit_gate']else('训练拟合门槛通过，历史探索性比较通过预定继续门槛。'if a['continuation_gate']else'训练拟合门槛通过，但历史探索性比较未通过继续门槛。')
 body='<h1>对象递归状态与全局递归状态：固定预算对照</h1><p>'+verdict+' 本轮结束，不追加训练、种子、快照选择或调参。</p>'
 body+='<p>相同冻结前端、样本索引、监督目标、512维持久状态与共享输出头；两结构 × 两种子 × 1500次更新。全局模型输入保留三个对象各自的512维特征及身份顺序。放置绑定仅从状态推断，不输入真实持有状态。图中蓝色为全局模型、橙色为对象模型；训练曲线实线为种子0、虚线为种子1。</p><img src="comparison.png" style="width:100%">'
 body+='<p>横轴为训练更新或随机种子；准确率纵轴为家族平均百分比，越高越好。保持错误纵轴为固定真实稳定InDrawer机会中的错误比例，越低越好，柱顶写错误数/固定分母。训练损失只用于描述拟合，50步滑动均值不参与选模型。两个种子均固定使用1500更新终点。</p>'
 body+='<h2>1. 先检查训练拟合</h2><p>角色数据以12个几何家族为单位，旧数据以40个根为单位。h0完整状态要求三个对象8分类位置和抽屉分类同时正确；正常全轨迹与联合指标要求完整Held3、InDrawer3和抽屉状态正确。角色联合要求正常与扰动两分支的h2当前状态及不看未来图像递推h3–h5全部正确。所有分配角色保留，包括真实执行失败及无效物理对。</p>'+table(checks,list(checks[0]))
 body+='<h2>2. h0、正常轨迹、成败联合与非目标保持</h2><p>以下8个历史家族此前已经查看，只能作探索性证据，不能称新盲测。每家族6种角色，共48角色、96分支。物理无效对在主指标中计失败；joint3_actual单独列于原始表。两臂使用相同48角色。</p>'+table(hist,list(hist[0]))
 body+='<p>保持机会：真实非目标对象在动作前后均在抽屉中；错误要求先验In概率≥0.5、观测更新后&lt;0.5。固定分母与“先验已经正确”条件分母均报告。place动作的评价分层允许读取真实动作前Held标签，但该标签从未传给模型或绑定模块。</p>'
 body+='<h2>3. 配对家族差异</h2><p>差异为对象模型减全局模型。2000次重采样以相同8个几何家族为单位；不把角色或分支当独立样本。区间不消除历史数据已经查看的限制。</p>'+table(ci,list(ci[0]))
 body+='<h2>4. 最早影响结果的错误</h2><p>按h0、h1、h2当前状态再到h3–h6递推查找每分支首个关系错误。完整概率、绑定权重、可见性及先验/后验保存在first_errors.json。绑定或消息值仅是诊断轨迹，不构成因果证明；本轮没有对个例追加修补。</p>'+table(first,list(first[0])if first else[])
 body+='<h2>5. 执行成败与移动事件</h2><p>以下为历史角色的一步预测，按真实执行成功、执行失败、拒绝分层。Brier为概率平方误差、NLL为负对数似然，越低越好。非目标移动召回率、误报率、类别变化支持和新增In机会见summary.csv与retention_opportunities.csv。训练/开发显式目标lift与开关抽屉中的非目标移动阳性为281/115，全部来自开关抽屉；非目标离散位置变化仅1/0。证据限于容器与对象移动耦合和稳定关系保持，不能扩大为普遍对象间符号变化能力。</p>'+table(outsum,list(outsum[0]))
 body+='<h2>6. 输入、公平性与资源</h2>'+table(runtime,list(runtime[0]))+'<p>对象模型2,465,492参数，全局模型2,454,260参数，差异0.46%。共享冻结前端按种子提取一次，二臂读取相同文件。前端含渲染器可见实例掩码提供的定位和当前RGB多数颜色路由，不是纯RGB检测器；对象特征已通过全局注意力，不是纯局部特征。所有当前、未来、事件和成功预测都来自递归状态；未来预测不输入未来观测。</p>'
 body+='<p>仅使用旧40训练根+54训练角色，共1128转移；开发408转移；历史696转移。新增采集0、C/D样本0、上传0。冻结配置记录完整输入字段、初始化种子、固定样本顺序和所有哈希。旧研究与原模型只读，单卡串行。本实验不构成原指南HULC执行器、TS-LTL闭环及独立评估验收完成。</p>'
 body+='<p>继续门槛：训练拟合通过，两个种子主指标都改善，种子平均配对区间下界大于0，h0/正常轨迹/H-In不退步且保持错误不增加。实际判定：'+str(a['continuation_gate'])+'。回归项：'+html.escape(str(a['regressions']))+'。</p>'
 css='body{font:16px/1.6 system-ui,sans-serif;max-width:1250px;margin:36px auto;padding:0 24px;color:#24313b}h1{font-size:28px}h2{margin-top:32px;font-size:21px}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}td,th{border:1px solid #d6dfe5;padding:7px;text-align:left}th{background:#edf3f7}tr:nth-child(even){background:#f8fafc}'
 (S/'reports/REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>对象递归状态固定对照</title><style>'+css+'</style>'+body)
 mainci=next(r for r in contrasts if r['metric']=='primary_joint3'and r['seed']=='mean');(S/'reports/RESULT.md').write_text('# 固定预算对照结果\n\n'+verdict+'\n\n'+f"历史8家族主指标对象减全局：{100*mainci['difference']:+.2f}个百分点，95%配对家族区间[{100*mainci['lo']:+.2f}, {100*mainci['hi']:+.2f}]。历史数据仅探索。\n\n"+'四次训练均固定1500更新；未选择中间快照、未调参。新采集0，C/D0，上传0。完整输入、统计规则、逐根指标、概率及首错轨迹见configs和results。\n\n查看REPORT.html与comparison.png获取图例、分母和解释。\n')
 save(S/'results/report_complete.json',{'utc':now(),'verdict':verdict,'main_contrast':mainci});print(verdict,flush=True)
if __name__=='__main__':main()
