from pathlib import Path
import json, datetime, collections, html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/robot/research/projects/wm_symbolic')
paired=json.loads((P/'runs/executor_visual_stop_v2_paired/results.json').read_text())
diag=json.loads((P/'runs/executor_table_occupancy_diagnostic_v1/results.json').read_text())
assert paired['status']==diag['status']=='complete'
assert len(diag['trials'])==54
colors=['red','blue','pink'];skills=paired['skill_names']
summary={s:{'v1_success':sum(x['v1_final_success'] for x in paired['pairs'] if x['skill']==s),
 'v2_success':sum(x['v2_final_success'] for x in paired['pairs'] if x['skill']==s),
 'roots':6,'v1_mean_controls':sum(x['v1_controls'] for x in paired['pairs'] if x['skill']==s)/6,
 'v2_mean_controls':sum(x['v2_controls'] for x in paired['pairs'] if x['skill']==s)/6} for s in skills}
occupancy={}
for color in colors:
 skill=f'lift_{color}_block_table'; occupancy[color]={}
 for condition in ['three','two']:
  rows=[x for x in diag['trials'] if x['skill']==skill and (x['moved_non_target'] is None)==(condition=='three')]
  acquisition=collections.Counter((x['first_contacted_and_raised'] or {}).get('color','none') for x in rows)
  claims=[x for x in rows if x['first_public_visual_decision'] and x['first_public_visual_decision']['decision']=='visual_completed']
  occupancy[color][condition]={'trials':len(rows),'roots':6,'ever_success':sum(x['ever_success'] for x in rows),'final_success':sum(x['final_success'] for x in rows),'first_acquisition':dict(acquisition),'shadow_visual_completion_claims':len(claims),'shadow_completion_correct':sum(x['offline_success_at_shadow_stop'] for x in claims)}
false_completion=json.loads((P/'runs/visual_stop_false_completion_diagnostic.json').read_text())
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stop_candidate':summary,'occupancy':occupancy,
 'max_unaffected_object_position_delta_m':max(v for x in diag['trials'] for v in x['unchanged_object_position_difference_after_settle_m'].values()),
 'max_robot_state_difference':max(x['robot_max_difference_after_settle'] for x in diag['trials']),
 'all_18_original_trajectories_reproduced':all(x['exact_original_actions_reproduced'] for x in diag['trials'] if x['moved_non_target'] is None),
 'main_manifest_unchanged':diag['main_manifest_sha256_before']==diag['main_manifest_sha256_after'],
 'conclusion':'The frozen visual stop candidate reduces drawer overrun but does not repair color acquisition. Moving one non-target block does not recover red acquisition in these C roots.',
 'limitations':['Six C roots; two interventions per root are correlated, not twelve independent roots','Public RGB-D/mask/robot stopping audit passed on calibration/evaluation frames; this is not a formal independent calibration guarantee','Current official-code adapter parity does not establish numerical equivalence to the original release software stack','Moving one block is not a complete reproduction of the standard CALVIN challenge distribution','No candidate promotion, formal training, new main-distribution adoption, or controller retraining performed'],
 'visual_stop_new_counterexample':{'run':false_completion['case'],'step':183,'estimated_rise_m':.05140957933234047,'offline_true_center_rise_m':.04304506327443991,'intervention_lift_completion_claims':6,'confirmed':5,'height_only_offline_replacement_suppresses_claim':True,'config_not_changed':True},
 'awaiting':'User review of parent-assigned five unscreened paired videos before any expensive continuation or protocol change'}
(P/'runs/executor_diagnosis_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
x=np.arange(5);a=axes[0]
for offset,key,color,label in [(-.18,'v1_success','#697887','Original fixed budget'),(.18,'v2_success','#168575','Visual stop candidate')]:
 vals=[summary[s][key] for s in skills];a.bar(x+offset,vals,.34,label=label,color=color)
 for xpos,v in zip(x+offset,vals):a.text(xpos,v+.1,f'{v}/6',ha='center',fontsize=8)
a.set_xticks(x,['Open drawer','Close drawer','Red lift','Blue lift','Pink lift'],rotation=20);a.set_ylim(0,7);a.set_ylabel('Successful endpoints (six C roots)');a.set_title('Same root, same action prefix');a.legend(fontsize=8)
a=axes[1];x=np.arange(3)
for offset,key,color,label in [(-.18,'three','#4279b4','Three blocks on table (6 trials/color)'),(.18,'two','#d7973d','One non-target moved (12 trials/color)')]:
 vals=[occupancy[c][key]['final_success']/occupancy[c][key]['trials'] for c in colors];a.bar(x+offset,vals,.34,label=label,color=color)
 for xpos,v,c in zip(x+offset,vals,colors):a.text(xpos,v+.015,f"{occupancy[c][key]['final_success']}/{occupancy[c][key]['trials']}",ha='center',fontsize=8)
a.set_xticks(x,['Red target','Blue target','Pink target']);a.set_ylim(0,1);a.set_ylabel('Endpoint success fraction at 360 controls');a.set_title('Bounded scene intervention; unchanged HULC');a.legend(fontsize=7.5,loc='upper left')
fig.savefig(P/'reports/figures/executor_diagnosis.png',dpi=160);plt.close(fig)
t=false_completion['trace'];fig,a=plt.subplots(figsize=(9,3.5),layout='constrained');steps=[x['control_step'] for x in t]
a.plot(steps,[x['estimated_visible_rise_m']*100 for x in t],'-o',color='#d7973d',label='Visible-point estimate (runtime input)')
a.plot(steps,[x['offline_true_center_rise_m']*100 for x in t],'-o',color='#168575',label='Object-center rise (offline evaluator only)')
a.axhline(5,color='#687582',linestyle='--',label='5 cm height threshold');a.axvline(183,color='#ba4845',linestyle=':',label='False completion claim at control 183');a.set_xlabel('Control step (30 controls / second)');a.set_ylabel('Height rise (cm)');a.set_title('Blue lift counterexample: first consequential stopping error');a.legend(fontsize=8,loc='upper left');fig.savefig(P/'reports/figures/visual_stop_false_completion.png',dpi=160);plt.close(fig)
zh={'open_drawer':'打开抽屉','close_drawer':'关闭抽屉','lift_red_block_table':'抬起红块','lift_blue_block_table':'抬起蓝块','lift_pink_block_table':'抬起粉块'}
rows=''.join(f"<tr><td>{zh[s]}</td><td>{v['v1_success']}/6 → {v['v2_success']}/6</td><td>{v['v1_mean_controls']:.0f} → {v['v2_mean_controls']:.0f}</td></tr>" for s,v in summary.items())
orows=''.join(f"<tr><td>{c}</td><td>{occupancy[c]['three']['ever_success']}/6；{occupancy[c]['three']['final_success']}/6</td><td>{occupancy[c]['two']['ever_success']}/12；{occupancy[c]['two']['final_success']}/12</td></tr>" for c in colors)
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>冻结执行器诊断：部分改善，颜色仍失败</title><style>body{{max-width:1120px;margin:32px auto;padding:0 24px;background:#f7f9fc;color:#17263b;font:16px/1.7 system-ui,sans-serif}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:28px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border-bottom:1px solid #dce3ed;text-align:left}}img{{width:100%;background:white}}.notice{{border-left:5px solid #c28a2d;background:#fff7e4;padding:14px}}a{{color:#176496}}small{{color:#53677e}}</style>
<h1>抽屉重复操作减少，颜色抓取仍未修好</h1><p class="notice">正式采集已安全暂停在4215条，正式训练尚未启动。视觉停止候选版未替代原执行器，也未进入主数据。需要先审查随机完整视频和下面的失败证据。</p>
<p>统计时间：{out['utc']}。本页只报告本机实际运行结果；六个C开发环境初态不构成主D测试或可靠性保证。</p>
<img src="figures/executor_diagnosis.png" alt="视觉停止配对结果与桌面占用诊断">
<p>左图：横轴是技能，纵轴为六个初态中的最终成功数；灰色为原固定预算，绿色为视觉停止候选版。两版同根、同随机种子，停止前动作逐值一致。右图：横轴是目标颜色，纵轴是360步结束时成功比例；蓝色为三块在桌面，橙色为只移走一个非目标块。颜色表示对比条件，不表示物体颜色。每根有两个移块变体，因此12次不等于12个独立初态。图中标注给出分子和分母，没有显著性推断。</p>
<h2>视觉停止做了什么</h2><table><tr><th>指令</th><th>原版 → 候选版最终成功</th><th>平均控制步数</th></tr>{rows}</table>
<p>候选版只允许提前停止或拒绝继续，不改变HULC的动作，也没有重训权重。它依据可见RGB-D、匿名可见分割、相机标定和公开机器人状态判断抽屉行程或抬起物体。校准与评估初始帧没有哈希重合，120帧的分割编号置换检查通过，运行输入没有接触信息、隐藏物体位姿或官方成功标签。</p>
<p>这批配对运行中，14次“完成”声明均被离线官方评分确认，但样本很小。另有两次“抓到错误颜色”拒绝；其中粉块指令先抬起蓝块，候选版在141步停止，而原版继续到360步后最终抬起粉块。因此粉块最终成功从2/6降为1/6，不能把拒绝计为抓取成功。</p>
<p><strong>随后场景干预诊断又发现一次抬取误报：</strong>移走一个非目标块的36条轨迹里，影子停止器发出6次完成声明，其中仅5次被离线评分确认。它只记录决定，未提前结束这36条轨迹。候选版抬取停止因此不能正式采用。</p>
<h2>三块同时在桌面是否是原因</h2><p>官方多步评估初态只允许一或两块在桌面；这不证明训练集没有三块桌面状态。诊断保留六个原始C根，每个目标分别将两个非目标块中的一个移到C场景本地滑门区域中心；目标块、机器人、相机、指令、HULC和随机种子固定，全部执行360步。主数据没有修改。</p>
<table><tr><th>目标色</th><th>原三块：曾成功；最终成功</th><th>移一块：曾成功；最终成功</th></tr>{orows}</table>
<p>18条原始轨迹重新运行均逐值一致。干预后其余物体最大位置差为{out['max_unaffected_object_position_delta_m']*1000:.4f}毫米，公开机器人状态最大差为{out['max_robot_state_difference']:.3g}。红块在两个条件中均未成功：减少一个桌面干扰块并未解决红块抓取。该实验只检验这个单因素，不代表完整官方评测，也没有授权更改主任务分布。</p>
<h2>第一个有实际影响的停止误报</h2><img src="figures/visual_stop_false_completion.png" alt="可见点高度误差导致蓝块抬取完成误报"><p>本图对应C根5、蓝块指令、红块移到滑门区的固定360步轨迹。横轴为控制步，纵轴为相对初态上升厘米数；橙线是停止器实际使用的可见点高度，绿线是仅用于离线评分的物体中心高度，灰虚线为5厘米阈值，红竖虚线为第183步误报。177、180、183步可见估计连续超过阈值，但真实中心没有越过阈值；183步分别为5.14和4.30厘米。</p><p>初态采用静态相机可见点，随后采用手部相机可见点，可见表面代表点会随视角改变。离线单项替换只将上升量换成中心高度，就会阻止该次完成声明，支持“可见点高度偏差导致误报”的解释；它不能证明相机切换是全部误差来源。没有把中心真值反馈给在线停止器，也没有针对这个反例再调阈值。</p>
<h2>已排除的接口问题与仍存在的限制</h2><p>25条指令与官方映射一致，三种颜色的嵌入不同；六个记录样本的输入张量和首动作与官方包装器完全一致；官方debug的8个记录起点×3个种子共24对完整轨迹也相同。三色指令在相同观察和随机种子下产生不同动作。因此没有发现简单的颜色指令串线或适配器张量错误。</p>
<p>这些对照比较的是当前官方源码与本项目适配器，不能宣称已经复现历史发布软件栈的数值行为或官方1000序列成绩。冻结策略与场景交互仍会抓错或抓不到；完整内部原因尚未唯一定位。当前没有证据支持再加一个颜色规则就能恢复可靠抓取。</p>
<p>后续若需改变HULC、主任务分布或验收标准，应先提交具体对照方案供用户决策。当前保持暂停，由父线程负责交付固定种子随机抽取的五组完整双相机视频。<a href="progress.html">主数据进度</a>；<a href="pilot.html">世界模型试跑及失败分析</a>。</p>
<small>可复查证据：runs/executor_visual_stop_v2_paired/results.json；runs/visual_stop_input_independence_audit.json；runs/executor_table_occupancy_diagnostic_v1/results.json；runs/visual_stop_false_completion_diagnostic.json；runs/executor_preprocessing_parity.json；runs/executor_official_reproduction_v1/results.json；runs/executor_diagnosis_summary.json。</small></html>'''
(P/'reports/executor_diagnosis.html').write_text(page)
print(json.dumps(out,ensure_ascii=False,indent=2))
