from common import *
import html
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(x)for x in row)+' |'for row in rows)+'\n'
def main():
 guard('report');a=read(S/'results/analysis_complete.json');metrics=a['metrics'];seal=read(S/'configs/models_frozen.json');post=list(csv.DictReader((S/'results/posterior_updates.csv').open()));errors=list(csv.DictReader((S/'results/first_errors.csv').open()));receipts=[read(S/f'results/train_{arm}_seed{seed}.json')for seed in [0,1]for arm in ARMS]
 def get(m,v,s):return next(r for r in metrics if r['split']=='evaluation'and r['metric']==m and r['variant']==v and r['seed']==s)
 def count(m,v,s):
  r=get(m,v,s);return f"{r['numerator']:.0f}/{r['denominator']}"
 selected=[];final=[]
 for seed in [0,1]:
  for arm in ARMS:
   c=next(r for r in seal['selection']if r['seed']==seed and r['arm']==arm);v=arm+'_final1000'if c['update']!=1000 else arm;final.append([seed,arm,c['update'],count('failure_h2_held',v,seed),count('failure_h2_joint',v,seed),count('failure_suffix3_full_prefix',v,seed),count('nominal_current_joint',v,seed)]);selected.append({'seed':seed,'arm':arm,'selected':c['update'],'final_variant':v})
 changes={m:[get(m,'recovery',s)['value']-get(m,'nominal',s)['value']for s in [0,1]]for m in ['failure_h2_held','failure_h2_joint','failure_suffix3_full_prefix','nominal_current_joint']}
 if all(x>0 for x in changes['failure_h2_held']):conclusion='失败后的持有身份纠错在两个模型种子上均优于同量常规经验。'
 elif all(x>=0 for x in changes['failure_h2_held'])and any(x>0 for x in changes['failure_h2_held']):conclusion='失败后的持有身份纠错有局部改善，但两个模型种子未同时取得提升。'
 else:conclusion='失败后的持有身份纠错尚未出现两个模型种子一致的改善。'
 if all(x>0 for x in changes['failure_suffix3_full_prefix']):conclusion+=' 从失败观测出发的三步完整前缀在两个种子上也提高，但独立评估仍只有6根。'
 elif any(x>0 for x in changes['failure_suffix3_full_prefix']):conclusion+=' 三步完整前缀的收益不一致，不能把当前状态纠错等同于可靠多步预测。'
 else:conclusion+=' 三步完整前缀没有取得一致提升，当前状态识别与未来预测需要分开评价。'
 if any(x<0 for x in changes['nominal_current_joint']):conclusion+=' 常规分支中存在退步，需要把这种代价和故障分支收益同时考虑。'
 rows=[]
 for seed in [0,1]:
  for v in ['original','nominal','recovery']:
   rr=[r for r in post if r['split']=='evaluation'and r['seed']==str(seed)and r['variant']==v and r['branch']=='2'];rows.append([seed,v,len(rr),sum(r['prior_held_correct']=='True'for r in rr),sum(r['posterior_held_correct']=='True'for r in rr),sum(r['corrected_held']=='True'for r in rr),sum(r['regressed_held']=='True'for r in rr),sum(r['retained_A']=='True'for r in rr),f"{np.mean([float(r['h2_nll'])for r in rr]):.4f}"])
 first=[]
 for seed in [0,1]:
  for v in ['original','nominal','recovery']:
   rr=[r for r in errors if r['split']=='evaluation'and r['seed']==str(seed)and r['variant']==v];ct=collections.Counter(int(r['first_wrong_boundary'])for r in rr);first.append([seed,v,ct.get(-1,0),*[ct.get(h,0)for h in range(7)]])
 md='\n\n'.join(['## 结果判读与固定末步补充',conclusion,'本轮比较只估计“共同终点修正条件下，失败恢复经验相对于常规经验”的影响。网络各部分共同续训；实际h与q变化不能进一步隔离成“状态更新模块自身修复”的因果结论。',
 '### 观察真实失败前后',table(['种子','模型','失败根','h2先验Held对','观察后Held对','错→对','对→错','观察后保留A入屉','h2关系NLL'],rows),
 '先验是在得到h2实际图像之前，后验是在收到h2图像之后；“错→对”才是这次观测带来的持有身份纠正。“保留A入屉”只检查先前物体的正关系，完整InDrawer集合的假阳性检查见主表。关系NLL越小越好，由浮点64裁剪后的概率计算，避免极端概率下的非有限值。',
 '### 匹配的最终1000更新',table(['种子','组','开发选中步','最终1000：Held','最终1000：完整关系','最终1000：三步前缀','最终1000：常规当前'],final),
 '该表无论开发选择500还是1000，都比较相同1000次更新的终点模型。它用于暴露checkpoint选择依赖，不能据评估结果重新选择模型。',
 '### 首个有影响的状态错误',table(['种子','模型','7边界全对','首错h0','首错h1','首错h2','首错h3','首错h4','首错h5','首错h6'],first),
 '每行覆盖6根×3分支=18条episode。首错从h0开始追踪；每条只记一次，避免把早期身份错误传播后的所有边界都算作新的根因。未额外训练补丁或进行模型选择。',
 '### 单卡实际训练资源',table(['种子','组','秒','最大分配显存GiB','选中步','开发500目标','开发1000目标'],[[r['seed'],r['arm'],f"{r['seconds']:.1f}",f"{r['peak_GiB']:.3f}",r['selected']['update'],f"{r['candidates'][0]['development_objective']:.6f}",f"{r['candidates'][1]['development_objective']:.6f}"]for r in receipts]),
 '四次训练固定共4000更新，不进行追加训练。独立数据采集、视觉缓存、训练与评估串行取得同一个GPU锁；没有修改驱动、防火墙、旧项目环境或启动云端任务。'])
 (S/'RESULTS_SUPPLEMENT.md').write_text(md);report=S/'REPORT.md';assert '## 结果判读与固定末步补充'not in report.read_text();original=report.read_text();head,body=original.split('\n',1);report.write_text(head+'\n\n'+conclusion+'\n\n'+body+'\n\n'+md+'\n')
 # Insert a readable text/table supplement in the standalone HTML without regenerating any frozen script.
 out=[];lines=md.splitlines();i=0
 while i<len(lines):
  l=lines[i]
  if l.startswith('|'):
   group=[]
   while i<len(lines)and lines[i].startswith('|'):group.append(lines[i]);i+=1
   out.append('<table>')
   for j,r in enumerate(group):
    if j==1:continue
    tag='th'if j==0 else'td';out.append('<tr>'+''.join(f'<{tag}>'+html.escape(v.strip())+f'</{tag}>'for v in r.strip('|').split('|'))+'</tr>')
   out.append('</table>');continue
  if l.startswith('#'):n=len(l)-len(l.lstrip('#'));out.append(f'<h{n}>'+html.escape(l[n:].strip())+f'</h{n}>')
  elif l.strip():out.append('<p>'+html.escape(l)+'</p>')
  i+=1
 p=S/'REPORT.html';doc=p.read_text().replace('</h1>','</h1><p><strong>'+html.escape(conclusion)+'</strong></p>',1);p.write_text(doc.replace('</html>',''.join(out)+'</html>'));save(S/'results/conclusion.json',{'utc':now(),'conclusion':conclusion,'selected_changes':changes,'matched_final1000_table':final,'posterior_update_table':rows,'first_error_table':first,'no_post_evaluation_selection':True});print(conclusion,flush=True)
if __name__=='__main__':main()
