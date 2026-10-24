from common import *
from report import table
guard();verify();rows=read(S/'results/semantic_outcomes.json');out=[]
for seed in [0,1]:
 for arm in ARMS:
  for template in ['ALL','recovery_seen6','order_known4','order_novel4']:
   rr=[r for r in rows if r['arm']==arm and r['seed']==seed and r['skill'].startswith('lift_')and(template=='ALL'or r['template']==template)]
   for mode,key in [('current','current_target_Held_correct'),('one_step','one_step_target_Held_correct')]:
    pos=[r for r in rr if r['true_target_Held']];neg=[r for r in rr if not r['true_target_Held']];tp=sum(r[key]for r in pos);fp=sum(not r[key]for r in neg)
    out.append({'arm':arm,'seed':seed,'template':template,'mode':mode,'lift_calls':len(rr),'positive_Held':len(pos),'true_positive':tp,'negative_Held':len(neg),'false_positive':fp,'recall':tp/len(pos)if pos else None,'false_positive_rate':fp/len(neg)if neg else None,'refused_lift_calls_included':sum(r['status']=='refused'for r in rr)})
csvwrite(S/'results/target_Held_discrimination.csv',out);save(S/'results/target_Held_discrimination.json',out)
htmltable=table(['种子','组','模板','方式','实际持有/抓取请求','Held真阳性','实际未持有','Held假阳性','拒绝调用仍计入'],[[r['seed'],r['arm'],r['template'],r['mode'],f"{r['positive_Held']}/{r['lift_calls']}",f"{r['true_positive']}/{r['positive_Held']}",r['negative_Held'],f"{r['false_positive']}/{r['negative_Held']}",r['refused_lift_calls_included']]for r in out])
block='<h2>持续Held正样本必须结合失败抓取负样本解释</h2>'+htmltable+'<p>完成阶段的连续Held样本主要来自成功抓取，不能单独证明模型能区分成功和落空。上表使用全部实际请求抓取边界，包含落空和拒绝；GT指定目标是否真正Held，模型只按原固定0.5阈值评分。Held召回与假阳性分列，避免总猜“抓取后一定持有”也得到漂亮的正样本保持率。current读过当步观测，one_step没有读到当步结果；无观测先验不能凭空知道被遮蔽的故障，二者任务信息不同。</p>'
p=S/'reports/REPORT.html';text=p.read_text();text=text.replace('<h2>跨技能关系保持与恢复</h2>',block+'<h2>跨技能关系保持与恢复</h2>');p.write_text(text)
