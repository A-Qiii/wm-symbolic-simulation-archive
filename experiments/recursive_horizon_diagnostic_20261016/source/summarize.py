from pathlib import Path
import json,csv,html
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/recursive_horizon_diagnostic_20261016';R=P/'reports/recursive_horizon_diagnostic_20261016'
FACTS=['持红','持蓝','持粉','红块入屉','蓝块入屉','粉块入屉','抽屉关闭','抽屉打开','抽屉中间']
GROUPS={'Holding':(0,3),'In':(3,6),'drawer':(6,9)}
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def classify(p):
 y=[v>=.5 for v in p[:6]]+[False]*3;y[6+max(range(3),key=lambda j:p[6+j])]=True;return y
def annotate(row):
 assert all(row['truth']['v'])and len(row['truth']['v'])==9
 y=row['truth']['y'];pr=classify(row['recursive_probability']);ar=classify(row['actual_readout_probability']);row['predicted_labels']=pr;row['actual_readout_labels']=ar;row['prediction_all9_correct']=pr==y;row['actual_readout_all9_correct']=ar==y
 row['prediction_wrong_facts']=[FACTS[j]for j in range(9)if pr[j]!=y[j]];row['actual_readout_wrong_facts']=[FACTS[j]for j in range(9)if ar[j]!=y[j]]
def table(roots,fixed=True):
 out=[]
 for h in range(1,5):
  eligible=[r for r in roots if len(r['steps'])>=h and all(s['observed']for s in r['steps'][:h])]
  if fixed:assert len(eligible)==len(roots)
  rows=[r['steps'][h-1]for r in eligible];rec={'horizon':h,'N':len(rows),'prediction_all9_n':sum(s['prediction_all9_correct']for s in rows),'prediction_cumulative_n':sum(all(s['prediction_all9_correct']for s in r['steps'][:h])for r in eligible),'actual_readout_all9_n':sum(s['actual_readout_all9_correct']for s in rows),'actual_readout_cumulative_n':sum(all(s['actual_readout_all9_correct']for s in r['steps'][:h])for r in eligible),'prediction_wrong_actual_readout_right':sum(not s['prediction_all9_correct']and s['actual_readout_all9_correct']for s in rows),'both_wrong':sum(not s['prediction_all9_correct']and not s['actual_readout_all9_correct']for s in rows),'prediction_right_actual_readout_wrong':sum(s['prediction_all9_correct']and not s['actual_readout_all9_correct']for s in rows)}
  for name,(lo,hi)in GROUPS.items():
   for key,prefix in [('predicted_labels','prediction'),('actual_readout_labels','actual_readout')]:rec[prefix+'_'+name+'_wrong_states']=sum(s[key][lo:hi]!=s['truth']['y'][lo:hi]for s in rows)
  rec['prediction_fact_errors']={f:{'FP':sum(s['predicted_labels'][j]and not s['truth']['y'][j]for s in rows),'FN':sum(not s['predicted_labels'][j]and s['truth']['y'][j]for s in rows)}for j,f in enumerate(FACTS)}
  out.append(rec)
 return out
def fraction(n,N):return f'{n}/{N}（{100*n/N:.0f}%）'if N else'无可用观测'
def metric_table(t):
 lines=['| 完整技能步 | 当步预测9项全对 | 截至该步每步全对 | 真实后继编码读出9项全对 |','|---|---:|---:|---:|']
 for x in t:lines.append('| '+str(x['horizon'])+' | '+' | '.join(fraction(x[k],x['N'])for k in['prediction_all9_n','prediction_cumulative_n','actual_readout_all9_n'])+' |')
 return lines
def group_table(t):
 lines=['| 步 | 持物项出错的状态数 | 入屉项出错的状态数 | 抽屉项出错的状态数 |','|---|---:|---:|---:|']
 for x in t:lines.append('| '+str(x['horizon'])+' | '+' | '.join(f'{x["prediction_"+g+"_wrong_states"]}/{x["N"]}'for g in GROUPS)+' |')
 return lines
def main():
 data=read(E/'results/trajectories.json')
 for rows in data.values():
  for root in rows:
   for s in root['steps']:
    if s['observed']:annotate(s)
 ref=data['reference'];logged=data['recorded_action_logs'];normal=[r for r in ref if r['four_step_eligible']];refused=[r for r in logged if r['four_step_eligible']and r['contains_refusal_in_first4']]
 assert len(ref)==12 and len(logged)==12 and len(normal)==5 and len(refused)==10
 assert all(all(s['status']=='success'for s in r['steps'])for r in normal)
 result={'model':'canonical_matched seed0/update1000','normal_four_step':table(normal),'refusal_containing_four_step':table(refused),'reference_all_available_variable_N':table(ref,False),'logged_all_available_variable_N':table(logged,False),'normal_roots':[r['root_id']for r in normal],'refusal_roots':[r['root_id']for r in refused],'reference_coverage':[{'root_id':r['root_id'],'planned':r['planned_steps'],'observed':r['observed_steps'],'four_step_eligible':r['four_step_eligible'],'reason':r['exclusion_reason']}for r in ref],'logged_coverage':[{'root_id':r['root_id'],'observed':r['observed_steps'],'four_step_eligible':r['four_step_eligible'],'refusal_in_first4':r['contains_refusal_in_first4'],'reason':r['exclusion_reason']}for r in logged]}
 save(E/'results/metrics.json',result);save(E/'results/scored_trajectories.json',data)
 with (R/'per_root_steps.csv').open('w')as f:
  cols=['source','root_id','horizon','skill','status','observed','prediction_all9_correct','actual_readout_all9_correct','prediction_wrong_facts','actual_readout_wrong_facts'];writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
  for roots in data.values():
   for r in roots:
    for s in r['steps']:writer.writerow({k:r[k]if k in['source','root_id']else';'.join(s.get(k,[]))if k.endswith('wrong_facts')else s.get(k,'')for k in cols})
 prov=read(E/'results/inference_provenance.json')
 lines=['# 一至四步递归状态预测盘点','', '本次只评价预测，不评价任务完成率或规划成功率。固定模型：修复匹配臂 canonical_matched，seed0，update1000。全部数据来自已经使用过的D2（物理环境D）。一个“步”是一次完整技能调用，不是控制帧。', '', '主预测从每条轨迹的初始真实观测编码一次，之后按该条记录的实际动作顺序递推潜状态，中间不加入真实观测纠正。每个步长都要求全部9项命题正确：持红/蓝/粉、红/蓝/粉入屉、抽屉关/开/中间。前6项沿用0.5阈值，抽屉三类取最大概率。所有纳入评分的真值9项均有效。', '', '## 同一批完整四步正常执行记录：5条', '', '动作依次为打开抽屉、抓取目标块、放置、关闭抽屉。每列步长都使用完全相同的5条真实连续记录，没有按预测对错筛选。']+metric_table(result['normal_four_step'])+['', '四步末态5/5不等于连续四步都预测正确：累计为0/5。第二、三步出错后，第四步可能重新符合真实末态。各步动作与状态类型不同，不能据此声称远期预测比短期更可靠。', '', '## 含拒绝调用的连续记录：10条', '', '使用原匹配臂日志中至少有4次连续技能调用、且前4次包含拒绝的全部10条记录。拒绝也占一个技能调用步，其真实后继使用实际记录的等待后观测。没有拼接不同复位或独立单步分支。']+metric_table(result['refusal_containing_four_step'])+['', '这些动作在原采集时由既有程序选择；本次仅把已经发生的动作序列当作给定输入做开环递归预测，不运行或评价规划器。两组采集动作不同，且部分root重合，不能把5+10当成15个独立根，也不将两组百分比混成一个分数。', '', '## 预测主要错在哪', '', '每格统计有该组任一命题错误的状态数；一个状态可能同时在多组出错。抽屉关/开互判只按一个抽屉状态错误计，不因为两个布尔位翻转重复计数。', '', '正常链第二步抓取后，4/5虚报物块入屉，3/5多报抓持对象；第三步放置后，5/5仍输出抓持为真。第二、三步真实后继读出也各错2/5，不能把全部误差只归因于转移。', '', '正常执行的同批5条：']+group_table(result['normal_four_step'])+['', '含拒绝链第四步10/10虚报物块入屉；抽屉状态在四个步长分别错9/10、9/10、10/10、7/10。链按前4步是否出现拒绝分类，不表示每一步均被拒绝。', '', '含拒绝调用的同批10条：']+group_table(result['refusal_containing_four_step'])+['', '“真实后继读出”把实际后继编码后送入同一个冻结查询头，单独与真值比较。它只作对照，不回灌递归预测。它也出错，说明状态表示/查询读出本身有限；预测和真实读出同时出错时，不能把全部错误归因于转移模型。逐命题假阳性/假阴性保存在metrics.json。', '', '## 数据覆盖与缺失，没有补造步骤', '', '参考记录原始12个root：6条原计划4步，其中5条完整；pink_2首步open_drawer失败，只有第1步真实后继，后3步未到达。另6条原设计只有3步，不存在第4步观测。因此主表5条具有完整轨迹可用性的选择偏差，不能作为所有12个root的四步总体准确率。失败根仍保留在下方变分母表和逐root记录中。', '', '原匹配臂动作日志同样有12个root：10条至少4次调用，前4次都包含拒绝；red_1和red_3的原采集在3次技能后停止，没有第4步。不存在可凭空追加的真实第4步，也未把短链拼接。', '', '以下仅用于展示所有已观测数据，分母随步长变化，不能视为同批准确率曲线。', '', '全部参考记录（包括首步执行失败根）：']+metric_table(result['reference_all_available_variable_N'])+['', '全部既有动作日志：']+metric_table(result['logged_all_available_variable_N'])+['', '## 来源和复核', '', f'模型：`{prov["model"]["path"]}`',f'SHA256：`{prov["model"]["sha256"]}`',f'查询头：`{prov["query_path"]}`',f'查询头SHA256：`{prov["query_sha256"]}`', '', '参考链递归概率直接复用封存 reference_diagnosis.json；已核对其源码确实只从初始状态编码并连续转移。动作与真实calls日志、前后帧、连续边界和真值逐项对齐。含拒绝日志只新增冻结模型推理：12条初始状态一次批量编码，沿各自实际动作最多递推4次，共46次转移，不加载后续真实观测纠正。初始RGB/机器人观测与缓存参考初始状态逐数组相同；初始查询概率与原日志数值一致。真实后继查询概率直接复用原日志。', '', '未训练、未新增仿真、未新采样、未改阈值或模型选择、未重跑规划或任务评测。旧文件只读，新文件位于独立 experiments/recursive_horizon_diagnostic_20261016。配置和完整来源哈希见configs，逐root逐步的预测/真值/错项见results/scored_trajectories.json与本报告旁per_root_steps.csv。', '', '样本数量很小且来自已用数据；本报告是描述性预测诊断，不给总体泛化保证。原初版单步52/62、59/62没有混入任何表格。']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n')
 # Render actual HTML tables; all resources remain local, with no scripts.
 parts=[];inside_table=False
 for line in lines:
  if line.startswith('|'):
   if set(line.replace('|','').replace(':','').replace(' ',''))=={'-'}:continue
   first=not inside_table
   if first:parts.append('<table>');inside_table=True
   tag='th'if first else'td';parts.append('<tr>'+''.join(f'<{tag}>'+html.escape(x.strip())+f'</{tag}>'for x in line.strip('|').split('|'))+'</tr>');continue
  if inside_table:parts.append('</table>');inside_table=False
  if line.startswith('## '):parts.append('<h2>'+html.escape(line[3:])+'</h2>')
  elif line.startswith('# '):parts.append('<h1>'+html.escape(line[2:])+'</h1>')
  elif line:parts.append('<p>'+html.escape(line)+'</p>')
 if inside_table:parts.append('</table>')
 body=''.join(parts);(R/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>递归状态预测盘点</title><style>body{max-width:1200px;margin:30px auto;padding:0 20px;color:#223647;background:#fafcfd;font:16px/1.75 system-ui,"Noto Sans CJK SC",sans-serif}p{overflow-wrap:anywhere}a{color:#176aa1}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #d8e1e7}th{background:#eaf1f6}h2{margin-top:32px}</style><a href="REPORT.md">Markdown报告</a> · <a href="per_root_steps.csv">逐root逐步明细CSV</a>'+body+'</html>')
 print(json.dumps({k:result[k]for k in['normal_four_step','refusal_containing_four_step']},ensure_ascii=False))
if __name__=='__main__':main()
