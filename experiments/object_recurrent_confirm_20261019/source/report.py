from common import *
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def table(rows):
 if not rows:return '<p>无。</p>'
 cols=list(rows[0]);return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>'for k in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>'for k in cols)+'</tr>'for r in rows)+'</table>'
def main():
 a=read(S/'results/analysis.json');ss=a['summaries'];checks=a['checks'];colors={'global':'#2675b7','object':'#e37b25'};plt.rcParams.update({'font.size':10});fig,axs=plt.subplots(2,3,figsize=(15,8.5));panels=[('h0_held','1. Initial held identity'),('h0_relation','2. Initial Held / InDrawer / drawer'),('h2_pair','3. Current success / miss pair at h2'),('normal_whole','4. Normal branch whole trajectory'),('primary_joint3','5. Paired current + 3 future steps'),('forget_rate','6. Non-target InDrawer forgetting')]
 for ax,(k,title)in zip(axs.flat,panels):
  for j,arm in enumerate(ARMS):
   rr=[next(r for r in ss if r['arm']==arm and r['seed']==seed)for seed in [0,1]];v=[r[k]*100 for r in rr];bars=ax.bar(np.arange(2)+(j-.5)*.32,v,width=.30,color=colors[arm],label=arm)
   for bar,r,value in zip(bars,rr,v):ax.text(bar.get_x()+bar.get_width()/2,value+(.15 if k=='forget_rate'else 1.1),f"{r['forget_count']}/{r['stable_In_opportunities']}"if k=='forget_rate'else f'{value:.1f}%',ha='center',fontsize=8)
  ax.set_xticks([0,1],['Seed 0','Seed 1']);ax.set(title=title,ylabel='Errors / fixed eligible opportunities (%)'if k=='forget_rate'else'Family-macro accuracy (%)');ax.set_ylim(0,max(5,max(r[k]*100 for r in ss)*1.35)if k=='forget_rate'else 108);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
  if k in ['h0_held','h0_relation','h2_pair']:ax.axhline(95 if k.startswith('h0')else 90,color='#555',ls=':')
 fig.tight_layout();fig.savefig(S/'reports/confirmation.png',dpi=160);fig.savefig(S/'reports/confirmation.svg');plt.close(fig)
 rows=[{'结构':r['arm'],'种子':r['seed'],'h0身份':f"{100*r['h0_held']:.2f}%",'h0归属关系':f"{100*r['h0_relation']:.2f}%",'h2当前成败对':f"{100*r['h2_pair']:.2f}%",'正常整轨':f"{100*r['normal_whole']:.2f}%",'当前+未来3步联合':f"{100*r['primary_joint3']:.2f}%",'In遗忘/固定机会':f"{r['forget_count']}/{r['stable_In_opportunities']}",'先验正确机会':r['prior_correct_stable_In']}for r in ss]
 cis=[{'指标':r['metric'],'种子':r['seed'],'对象减全局百分点':f"{100*r['difference']:+.2f}",'95%家族区间':f"[{100*r['lo']:+.2f}, {100*r['hi']:+.2f}]"}for r in a['contrasts']if r['metric']in['primary_joint3','normal_whole','h0_relation','h2_pair','forget_rate']]
 cr=[{'种子':r['seed'],'预登记条件':r['criterion'],'值':round(float(r['value']),5),'规则':r['rule'],'通过':r['passed']}for r in checks]
 full=[{'结构':r['arm'],'种子':r['seed'],'h0完整8分类+抽屉':f"{100*r['h0_full']:.2f}%",'正常整轨完整8分类':f"{100*r['normal_whole_full']:.2f}%"}for r in ss]
 audits=read(S/'results/root_audits.json');invalid=[{'角色':r['root_id'],'正常分支满足预定成功':r['branches'][0]['valid_success'],'扰动分支满足预定落空':r['branches'][1]['valid_failure'],'初态两分支有效':all(b['initial_valid']for b in r['branches'])}for r in audits if not r['valid_pair']];first=collections.Counter((r['arm'],r['seed'],r['mode'],r['step'])for r in read(S/'results/first_errors.json'));firstrows=[{'结构':k[0],'种子':k[1],'来源':k[2],'边界':k[3],'分支数':v}for k,v in sorted(first.items())]
 verdict='通过本轮预登记的关系确认门槛'if a['relation_confirmation_gate']else'未通过本轮预登记的全部关系确认门槛';body='<h1>全新A/B几何族：固定模型关系预测确认</h1><p><strong>'+verdict+'。</strong>本轮训练0、调阈值0、模型选择0。上一轮完整状态training_fit_gate=false原判定不变。</p>'
 body+='<p>8个全新几何族（A/B各4），每族6种初始持有H与请求目标T组合，共48角色；正常/落空两条6技能轨迹，共576真实转移。四个模型均为上一轮1500更新终点。新标签只用于评分，不回流模型。已知对象和技能不变；新证据只针对A/B布局迁移与观察恢复。</p><img src="confirmation.png" style="width:100%"><p>蓝色为全局模型、橙色为对象模型；横轴为种子，纵轴为家族平均准确率，越高越好。第6图为固定真实稳定In机会中的遗忘比例，越低越好，柱顶显示错误数/分母。灰色点线为95%初态或90%当前配对门槛。正常分支指无注入扰动，不保证所有技能成功；上述数值是模型预测准确率，不是机器人任务成功率。</p>'
 body+='<h2>任务关系与新旧门槛的区别</h2><p>主任务使用三个对象Held、三个对象InDrawer以及抽屉分类。h0要求所有身份与归属同时正确；h2和联合指标要求正常、抓取落空两分支均正确；联合额外要求从h2只读请求动作递推h3–h5全部正确，不输入未来图像。桌面左中右标签仍保留；目前未证明其会改变这条关系任务的成败，因此另列诊断，不追溯修改旧全面状态门槛。</p>'+table(rows)
 body+='<h2>预登记验收与家族区间</h2><p>所有48角色固定保留。物理无效成功/落空对在主指标计失败；actual标签一致性单列。2000次配对bootstrap按4A+4B族分层，种子分别报告并在同一家族内取种子均值；48角色、96分支不是独立样本。确认要求所有下列条件同时满足；无事后放宽。</p>'+table(cr)+table(cis)
 body+='<h2>完整位置分类仍然分列</h2>'+table(full)+'<p>这组结果只记录完整位置读出的剩余缺口，不将旧失败改写为通过，也不以关系确认代替完整规划安全验收。</p>'
 body+='<h2>失败、保持与首个影响错误</h2><p>无效物理配对：'+str(len(invalid))+'/48；没有替换。全部setup、控制、失败、拒绝和视频均保存。</p>'+table(invalid)+table(firstrows)+'<p>首错按h0、h1、h2当前关系，再到h3–h6未来关系查找。完整概率、绑定权重与可见性保存在first_errors.json。稳定In遗忘要求真实动作前后均In、先验≥0.5而更新后&lt;0.5；固定机会与先验正确条件分母同时报告。动作的非目标评价分层可以用GT，但真实持有状态从未输入放置绑定。移动事件召回、误报、Brier与NLL的逐转移数据见results。</p>'
 body+='<h2>固定条件与范围</h2><p>两个模型共享同种子的同一冻结DINO/E特征；全局模型保留三个独立对象向量及顺序，不将身份平均掉。前端继承渲染器可见实例掩码，再用当前RGB颜色路由，不能称纯RGB定位。四模型只推断、无更新；当前与未来输出共同来自递归状态。全部旧封存只读；实际运行状态见configs/runtime_state.json。</p><p>新几何从662个历史A/B几何及已接纳新族中去重；任意颜色排列下三对象位置全在20mm以内即拒绝。物理预检只检完整足迹、间距、碰撞、IK与静置稳定，全部几何拒绝留痕，不看模型分数或技能结果换样本。执行器仍为此前冻结的物理技能执行器，不是本轮完成HULC验收。</p><p>这些已知对象/技能、A/B布局与观察恢复结果不能推广为未知物体交互、完整TS-LTL规划或安全验收。跨对象变化标签主要由开关容器产生，仍不扩大为一般对象间符号动态能力。C/D采集与评分0，上传0。</p>'
 css='body{font:16px/1.6 system-ui;max-width:1250px;margin:32px auto;padding:0 24px;color:#24313b}h1{font-size:28px}h2{font-size:21px;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:13px;margin:15px 0}td,th{border:1px solid #d6dfe5;padding:7px;text-align:left}th{background:#edf3f7}tr:nth-child(even){background:#f8fafc}'
 (S/'reports/REPORT.html').write_text('<!doctype html><meta charset="utf-8"><style>'+css+'</style>'+body)
 ci=next(r for r in a['contrasts']if r['metric']=='primary_joint3'and r['seed']=='mean');(S/'reports/RESULT.md').write_text('# 新几何族关系确认\n\n'+verdict+'。旧全面状态训练门槛仍false，不追溯改变。\n\n'+f"联合主指标对象减全局平均{100*ci['difference']:+.2f}个百分点，95%配对家族区间[{100*ci['lo']:+.2f}, {100*ci['hi']:+.2f}]。\n\n"+'8新族、48角色、96完整分支、576转移；全部失败保留。四权重固定，训练0、C/D0、上传0。详见REPORT.html。\n');save(S/'results/report_complete.json',{'utc':now(),'verdict':verdict,'main_contrast':ci});print(verdict,flush=True)
if __name__=='__main__':main()
