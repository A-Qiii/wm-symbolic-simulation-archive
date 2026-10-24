from common import *
from report import table
guard();verify();roots=read(S/'data/roots_manifest.json');rows=[]
for seed in [0,1]:
 for arm in ARMS:
  z=dict(np.load(S/f'results/predictions_{arm}_seed{seed}.npz'));p=rel(z['prob']);y=truth(z['cat'])
  for root in roots:
   t=root['fault_step']+1;ob=COLORS.index(root['matched_color']);xx={}
   for mode in ['current','one_step']:
    ix=[np.flatnonzero((z['root_id']==root['root_id'])&(z['branch']==b)&(z['mode']==mode)&(z['step']==t))[0]for b in [1,2]];xx[mode]=ix
   j,k=xx['one_step'];u,v=xx['current'];diff=float(np.abs(z['prob'][j]-z['prob'][k]).max());assert diff<1e-6
   rows.append({'arm':arm,'seed':seed,'root_id':root['root_id'],'family_id':root['family_id'],'template':root['template'],'fault_boundary':t,'same_public_prefix':True,'prior_max_abs_difference':diff,'nominal_true_Held':bool(y[u,ob]),'disturbed_true_Held':bool(y[v,ob]),'true_Held_diverged':bool(y[u,ob]!=y[v,ob]),'prior_both_Held_correct':bool((p[j,ob]>=.5)==y[j,ob]and(p[k,ob]>=.5)==y[k,ob]),'current_both_Held_correct':bool((p[u,ob]>=.5)==y[u,ob]and(p[v,ob]>=.5)==y[v,ob])})
summary=[]
for seed in [0,1]:
 for arm in ARMS:
  for template in ['recovery_seen6','order_known4','order_novel4']:
   rr=[r for r in rows if r['arm']==arm and r['seed']==seed and r['template']==template];dd=[r for r in rr if r['true_Held_diverged']];summary.append({'arm':arm,'seed':seed,'template':template,'matched_pairs':len(rr),'true_Held_divergent_pairs':len(dd),'prior_both_correct_on_divergent':sum(r['prior_both_Held_correct']for r in dd),'observed_both_correct_on_divergent':sum(r['current_both_Held_correct']for r in dd),'maximum_prior_difference':max(r['prior_max_abs_difference']for r in rr)})
csvwrite(S/'results/paired_information.csv',rows);save(S/'results/paired_information_summary.json',summary)
tab=table(['种子','组','模板','匹配分支对','真实Held分歧对','一步先验两边都对','观测后两边都对','先验最大差'],[[r['seed'],r['arm'],r['template'],r['matched_pairs'],r['true_Held_divergent_pairs'],r['prior_both_correct_on_divergent'],r['observed_both_correct_on_divergent'],r['maximum_prior_difference']]for r in summary])
block='<h2>故障不可见时，先验能知道什么</h2>'+tab+'<p>每对分支在首次目标抓取前公共观测完全相同，请求动作也相同；覆盖闭爪命令的故障标志不进入模型。因此对应一步先验必须相同。若实际Held结果分歧，确定性单一阈值预测不能把一对两边都分对，该子集的逐分支准确率上限为50%。这一限制来自可用信息，不应当单独归咎于关系模型退步。加入本步真实观测后的读出能否区分两边、从已观察h2后能否正确恢复，才是相应的纠正与条件转移检验；完整原始先验错误仍计入所有主表，没有删除。</p>'
p=S/'reports/REPORT.html';text=p.read_text().replace('<h2>补上稳定Held正样本分母</h2>',block+'<h2>补上稳定Held正样本分母</h2>');p.write_text(text)
