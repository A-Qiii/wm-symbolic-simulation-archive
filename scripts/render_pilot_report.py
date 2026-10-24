from pathlib import Path
import json,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/robot/research/projects/wm_symbolic');A=P/'runs/pilot_development_analysis_v2';r=json.loads((A/'analysis.json').read_text());figures=P/'reports/figures';figures.mkdir(exist_ok=True)
metrics=[]
for run in ['wm_pilot_protocol_v2_seed0','wm_pilot_protocol_v2_resume3000_seed0']:
 for q in sorted((P/'runs'/run).glob('metrics_*.json')):metrics.append(json.loads(q.read_text()))
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
colors={'current_nll':'#2375b5','prior_nll':'#d87719','free_3_nll':'#228b67','free_5_nll':'#8e59af','event_nll':'#6e7780'}
for key,name in [('current_nll','Current posterior'),('prior_nll','One step from real history'),('free_3_nll','3 steps from root'),('free_5_nll','5 steps from root'),('event_nll','Outcome + events')]:ax.plot([m['update'] for m in metrics],[m[key] for m in metrics],marker='o',color=colors[key],label=name)
ax.axvline(3000,color='#bbb',ls=':',label='Saved / resumed boundary');ax.axvline(5000,color='#222',ls='--',label='3-step training begins after 5000')
ax.set(xlabel='Optimizer update',ylabel='Development NLL (lower is better)',title='C development: 10 roots, 20 branches, 120 real transitions');ax.legend(fontsize=8,ncols=2);fig.savefig(figures/'pilot_development_nll.png',dpi=150);plt.close(fig)
def esc(value):return html.escape(str(value))
locations=['台面左','台面中','台面右','抽屉','滑门左区','滑门右区','手持','其他']
def labels(values):return [locations[x] for x in values[:3]]+[['闭','中间','开'][values[3]],['右','中间','左'][values[4]],'亮' if values[5] else '灭','亮' if values[6] else '灭']
parts=[]
for i,case in enumerate(r['cases'][:2]):
 row=case['record'];fig,axes=plt.subplots(1,2,figsize=(8,3.6),layout='constrained')
 for ax,key,title in zip(axes,['before_public','after_public'],['Actual before','Actual after']):
  with np.load(row[key]) as z:ax.imshow(z['rgb_static'])
  ax.set(xlabel='Image pixel x',ylabel='Image pixel y',title=title)
 fig.suptitle(f"{row['episode_id']}, macro {row['step']}: {row['skill']}",fontsize=10);name=f'pilot_case_{i+1}.png';fig.savefig(figures/name,dpi=150);plt.close(fig)
 table=''
 for label,key in [('当前观测读出','current'),('真实历史重启一步','one_step_real_history'),('从根自由滚动','free_from_root')]:
  pred=labels(row[key]['predicted']);target=labels(row[key]['target']);cells=''.join(f'<td class="{ "bad" if correct is False else "masked" if correct is None else "good"}">{esc(p)} / {esc(t)}</td>' for p,t,correct in zip(pred,target,row[key]['correct']))
  table+=f'<tr><th>{label}</th>{cells}</tr>'
 parts.append(f'<h2>离线案例 {i+1}：{esc(row["skill"])}</h2><img src="figures/{name}"><p>两图来自同一真实宏动作，左为执行前、右为执行后；颜色为原始物体外观，坐标是相机像素。模型不读取评分标签。下表每格为“预测 / 真值”；红底表示错误，灰底表示标签无效，白底表示一致。这里只定位预测差异，尚未证明该差异改变了某次真实计划。</p><table><tr><th>条件</th><th>红块</th><th>蓝块</th><th>粉块</th><th>抽屉</th><th>滑门</th><th>灯泡</th><th>LED</th></tr>{table}</table>')
closed=P/'runs/closed_loop_interface_C000';score_path=closed/'independent_score.json'
if score_path.exists():
 score=json.loads(score_path.read_text());diagnosis=json.loads((closed/'single_interface_diagnostics.json').read_text());ref=json.loads((P/'runs/reference_interface_C000/independent_score.json').read_text());d=diagnosis['actual_history_one_step_restart_before_place']
 runtime=json.loads((closed/'runtime.json').read_text());beliefs=[x['belief'] for x in runtime['trace'] if 'belief' in x and ('observation' in x or 'executed_skill' in x)]
 current=[float(np.asarray(x['symbol_probabilities']['objects'])[:,0,3].mean()) for x in beliefs];last=next(x for x in reversed(runtime['trace']) if 'public_observation' in x)['public_observation'][-1]
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 with np.load(last) as z:axes[0].imshow(z['rgb_static'])
 axes[0].set(xlabel='Image pixel x',ylabel='Image pixel y',title='Actual final observation')
 axes[1].plot(range(len(current)),current,'o-',color='#2375b5',label='Observed-history posterior: red in drawer')
 axes[1].plot(range(len(current)),[0]*len(current),'s-',color='#222222',label='Offline physical label: false')
 axes[1].scatter([3],[d['prior_red_drawer_probability']],color='#d87719',s=70,label='One-step prediction before final skill',zorder=5)
 axes[1].axhline(.5,color='#999999',ls='--',label='Uncalibrated diagnostic threshold')
 axes[1].set(xlabel='Completed macro actions',ylabel='Probability / binary label',ylim=(-.05,1.05),xticks=range(4),title='False completion after actual observation');axes[1].legend(fontsize=7)
 fig.savefig(figures/'closed_loop_false_confirmation.png',dpi=150);plt.close(fig)

 parts.append(f'<h2>真实闭环案例3：错误完成确认</h2><img src="figures/closed_loop_false_confirmation.png"><p>左图为最终真实固定相机图像，坐标为像素、颜色为物体原貌。右图横轴为已执行宏动作数，纵轴为红块属于抽屉的预测概率或0/1评分事实；蓝线为实际观察更新后的模型读出，黑线为离线事实（始终为0），橙点为执行最后技能前的一步预测，灰虚线为未校准的0.5诊断阈值。实际观察后的读出也未纠正错误；这些概率不是安全保证。</p><p>同一C开发根，未校准阈值0.5。学习搜索实际执行“向右推蓝块→向右推红块→放入抽屉”，共3项技能、600控制步；模型宣布完成。独立全轨迹评分发现红块未入抽屉，且粉块被移动，实际任务失败。</p><table><tr><th>单项对照</th><th>结果</th><th>能说明什么</th></tr><tr><td>只替换根标签</td><td>三项计划不变</td><td>保持同一潜状态、模型、动作、预算和随机状态；根标签替换没有修复该错误，不代表潜状态正确。</td></tr><tr><td>真实历史重启一步</td><td>预测红块入抽屉 {d["prior_red_drawer_probability"]:.1%}；预测粉块移动 {d["prior_pink_moved_probability"]:.1%}</td><td>实际红块未入抽屉、粉块发生移动。错误在一步预测中已存在，不能仅归为长滚动累积。</td></tr><tr><td>同起点参考序列</td><td>4项技能执行完，目标未确认完成</td><td>参考链同样受冻结执行器限制；无效/歧义评分标签单列，不把“序列执行完”算成功。</td></tr></table><p>完整证据：<a href="../runs/closed_loop_interface_C000/independent_score.json">独立评分</a>、<a href="../runs/closed_loop_interface_C000/single_interface_diagnostics.json">单项替换诊断</a>。没有添加人工动态前提或效果，失败记录完整保留。</p>')
m=r['metrics'];comparison=''
for title,key in [('当前后验','current'),('实际历史重启一步','one_step_real_history'),('从根自由滚动1—6步汇总','free_from_root')]:
 for item in r['by_predicate_family'][key]:comparison+=f'<tr><td>{title}</td><td>{esc(item["predicate_family"])}</td><td>{item["facts"]}</td><td>{item["accuracy"]:.1%}</td><td>{item["macro_F1_present_classes"]:.3f}</td><td>{item["unknown_rate_at_diagnostic_0_5"]:.1%}</td></tr>'
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>720转移试训：独立开发结果</title><style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1140px;margin:40px auto;padding:0 24px;color:#17233b}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #d6dce6;padding:9px;text-align:left}}th{{background:#edf2f9}}.bad{{background:#ffe3df}}.masked{{background:#e7e7e7}}img{{width:100%;max-width:1000px}}.note{{padding:18px;background:#eef3fa}}h2{{margin-top:36px}}</style>
<h1>720条真实训练转移：修正版试训结果</h1><p class="note">训练按规定完成6000步，5000步之后进入3步损失。按C开发集“3/5步符号NLL均值＋事件NLL”选择第{r['selected_update']}步，因此当前选中模型尚未接受多步损失训练。后续更新的开发NLL反而上升，表现为小数据过拟合；不能把训练集拟合当作泛化或闭环成功。</p>
<table><tr><th>元素</th><th>实际结果</th><th>为什么／边界</th></tr>
<tr><td>数据划分</td><td>A/B 60根720转移训练；C 10根120转移开发</td><td>同一根两分支不是两个独立初态；未读取D测试结果</td></tr>
<tr><td>已选模型：当前事实</td><td>{m['current_correct']}/{m['current_facts']} = {m['current_accuracy']:.1%}</td><td>有效互斥分类事实；灯状态另计入NLL和下表</td></tr>
<tr><td>已选模型：一步后果</td><td>{m['prior_correct']}/{m['facts']} = {m['prior_accuracy']:.1%}</td><td>从真实历史后验出发，只用动作和先验预测</td></tr>
<tr><td>已选模型：变化事实</td><td>{m['changed_correct']}/{m['changed_facts']} = {m['changed_accuracy']:.1%}</td><td>仅统计确实改变的有效分类事实，避免静止多数掩盖错误</td></tr>
<tr><td>恢复训练</td><td>3000→6000，新目录保存；完整恢复入口已验证</td><td>旧检查点缺随机状态，未宣称逐位连续；新检查点保存全状态</td></tr>
<tr><td>一次修正</td><td>物理台面分区＋完整控制过程中的设备事件</td><td>原标签与原版训练保留；该次修正已用完，不追加单例规则</td></tr>
<tr><td>接下来</td><td>按原指南扩至正式6000条总转移</td><td>当前跨场景预测仍较弱；没有降低配额、阈值目标或验收标准</td></tr></table>
<h2>为什么没有选最后一步</h2><img src="figures/pilot_development_nll.png"><p>横轴为优化更新次数，纵轴为独立C开发集负对数似然，越低越好。蓝线：当前事实；橙线：真实历史重启的一步预测；绿线／紫线：从根连续滚动3／5步；灰线：技能结果与过程事件。灰色点线标记3000步暂停恢复，黑色虚线标记开始3步训练。所有点来自相同120条转移、8条配对随机潜轨迹的平均概率，没有挑选最有利轨迹。曲线上升表示错误或过度自信增加，不等同于动作成功率下降。</p>
<h2>谓词族对照</h2><p>F1按在真值或预测中出现的类别取宏平均；混淆矩阵原始计数存于analysis.json。未知率采用未校准的0.5诊断阈值；这些数值不能作为实际完成确认阈值。</p><table><tr><th>条件</th><th>谓词族</th><th>有效事实数</th><th>准确率</th><th>宏F1</th><th>诊断未知率</th></tr>{comparison}</table>
{''.join(parts)}<p>完整证据：<a href="../runs/pilot_development_analysis_v2/analysis.json">分析JSON</a>；<a href="../runs/pilot_development_analysis_v2/predictions.jsonl">逐步预测</a>；<a href="../runs/checkpoint_resume_validation.json">恢复验证</a>。原版标签训练仅保留为修正前诊断，不与修正版数字直接混作同一语义。</p></html>'''
(P/'reports/pilot.html').write_text(page);print(P/'reports/pilot.html')
