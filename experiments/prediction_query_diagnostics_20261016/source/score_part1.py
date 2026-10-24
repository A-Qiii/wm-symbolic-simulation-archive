from config import *
import csv
def summarize(rows):
 out=[]
 for h in range(1,5):
  rr=[r for r in rows if r['step']==h];N=len(rr);x={'horizon':h,'N':N}
  for method in ['actual','one_step','recursive']:
   x[method+'_all9']=sum(r[method+'_ok']for r in rr)
   for name,lo,hi in [('Holding',0,3),('In',3,6),('drawer',6,9)]:x[method+'_'+name+'_wrong']=sum(np.any(np.array(r[method+'_labels'])[lo:hi]!=np.array(r['truth'])[lo:hi])for r in rr)
  for method in ['one_step','recursive']:
   x[method+'_paired_state']={'actual_right_prediction_wrong':sum(r['actual_ok']and not r[method+'_ok']for r in rr),'both_wrong':sum(not r['actual_ok']and not r[method+'_ok']for r in rr),'actual_wrong_prediction_right':sum(not r['actual_ok']and r[method+'_ok']for r in rr),'both_right':sum(r['actual_ok']and r[method+'_ok']for r in rr)}
   x[method+'_paired_facts']={'actual_right_prediction_wrong':0,'both_wrong':0,'actual_wrong_prediction_right':0,'both_right':0}
   for r in rr:
    aa=np.array(r['actual_labels'])==r['truth'];bb=np.array(r[method+'_labels'])==r['truth']
    for k,v in [('actual_right_prediction_wrong',aa&~bb),('both_wrong',~aa&~bb),('actual_wrong_prediction_right',~aa&bb),('both_right',aa&bb)]:x[method+'_paired_facts'][k]+=int(v.sum())
   x[method+'_mean_raw_latent_MSE']=float(np.mean([r[method+'_latent_MSE']for r in rr]))
  x['one_step_right_recursive_wrong']=sum(r['one_step_ok']and not r['recursive_ok']for r in rr);x['one_step_wrong_recursive_right']=sum(not r['one_step_ok']and r['recursive_ok']for r in rr);out.append(x)
 return out
def main():
 predictions=read(E/'results/part1_predictions.json');truth={r['trajectory_id']:r for r in read(E/'data/evaluator/part1_truth.json')};records=[];maxdiff=0
 for p in predictions:
  t=next(s for s in truth[p['trajectory_id']]['steps']if s['horizon']==p['step']);assert all(t['truth']['v']);r=p|{'truth':t['truth']['y'],'valid':t['truth']['v'],'status':t['status']}
  for method in ['actual','one_step','recursive']:r[method+'_labels']=discrete(p[method+'_probability']).tolist();r[method+'_ok']=r[method+'_labels']==r['truth']
  for new,old in [('actual_probability','actual_readout_probability'),('recursive_probability','recursive_probability')]:maxdiff=max(maxdiff,float(np.max(np.abs(np.array(p[new])-t[old]))))
  records.append(r)
 # Re-encoding logged observations must reproduce original frozen probabilities.
 assert maxdiff<1e-4,maxdiff
 normal=[r for r in records if r['source']=='reference'and truth[r['trajectory_id']]['four_step_eligible']];refused=[r for r in records if r['source']=='recorded_action_logs'and truth[r['trajectory_id']]['four_step_eligible']and truth[r['trajectory_id']]['contains_refusal']];assert len(normal)==20 and len(refused)==40
 metrics={'normal':summarize(normal),'refusal':summarize(refused),'max_probability_difference_from_sealed_run':maxdiff,'normal_root_count':5,'refusal_root_count':10,'all_observed_records':85,'all_root_count_per_source':12,'coverage':[{k:r[k]for k in['trajectory_id','planned_steps','observed_steps','four_step_eligible','exclusion_reason']}for r in truth.values()]};save(E/'results/part1_metrics.json',metrics);save(E/'results/part1_scored_records.json',records)
 with (R/'part1_paired_propositions.csv').open('w')as f:
  writer=csv.writer(f);writer.writerow(['source','root','skill_step','skill','fact','truth','actual_readout','one_step','recursive','actual_ok','one_step_ok','recursive_ok'])
  for r in records:
   for j,name in enumerate(FACTS):writer.writerow([r['source'],r['root_id'],r['step'],r['skill'],name,r['truth'][j]]+[r[k+'_labels'][j]for k in ['actual','one_step','recursive']]+[r[k+'_labels'][j]==r['truth'][j]for k in ['actual','one_step','recursive']])
 lines=['# 第一步诊断：真实前态一步预测与递归预测的配对比较','','固定 repair canonical_matched seed0/update1000 与原查询头。比较同一条真实轨迹、同一实际动作和同一真实后继；没有训练、仿真、规划或任务评测。全部9项命题正确才算状态正确。','', '真实前态一步预测每步重新使用实际前态；递归预测只使用最初真实观测，不中途纠正；真实后继读出是单独对照。','']
 for cohort,title in [('normal','正常执行的同批5条完整四步链'),('refusal','含拒绝调用的同批10条完整四步记录')]:
  lines += ['## '+title,'','| 技能步 | 真实后继读对 | 真实前态一步预测 | 递归预测 | 一步对、递归错 | 一步错、递归对 |','|---|---:|---:|---:|---:|---:|']
  for x in metrics[cohort]:lines.append(f'| {x["horizon"]} | {x["actual_all9"]}/{x["N"]} | {x["one_step_all9"]}/{x["N"]} | {x["recursive_all9"]}/{x["N"]} | {x["one_step_right_recursive_wrong"]} | {x["one_step_wrong_recursive_right"]} |')
  lines+=['','逐状态配对，不用总体准确率相减作错误归因：','','| 步 | 真实读对/一步错 | 真实与一步均错 | 真实读对/递归错 | 真实与递归均错 | 一步原空间MSE | 递归原空间MSE |','|---|---:|---:|---:|---:|---:|---:|']
  for x in metrics[cohort]:lines.append(f'| {x["horizon"]} | {x["one_step_paired_state"]["actual_right_prediction_wrong"]} | {x["one_step_paired_state"]["both_wrong"]} | {x["recursive_paired_state"]["actual_right_prediction_wrong"]} | {x["recursive_paired_state"]["both_wrong"]} | {x["one_step_mean_raw_latent_MSE"]:.4f} | {x["recursive_mean_raw_latent_MSE"]:.4f} |')
  lines+=['','错误状态组（持物 / 入屉 / 抽屉；同一状态可在多组出错）：','','| 步 | 真实读出错组数 | 一步预测错组数 | 递归预测错组数 |','|---|---|---|---|']
  for x in metrics[cohort]:lines.append('| '+str(x['horizon'])+' | '+' | '.join('/'.join(str(x[m+'_'+g+'_wrong'])for g in ['Holding','In','drawer'])for m in ['actual','one_step','recursive'])+' |')
 lines+=['','## 解释与范围','','原空间误差为固定编码器256维潜状态逐维平方差的均值，单位是该潜表示自身的平方尺度；不是命题真值，不是概率，也不是t-SNE距离。误差较小仍可能读错命题，递归误差增加也不必然对应更低的当步离散正确率。','','原参考组12根：5条完整四步、6条原设计三步、1条首步失败后3步未到达。原动作日志12根：10条有至少四步且包含拒绝，2条只记录三步。保留全部85个已观测转移、765条逐命题配对；没有补造缺失步骤或拼接不同复位。两组root有重合，不能合并成独立样本总量。','','时间与输入：真实调用start/end帧和前后边界已经逐项核对；相邻技能共享同一个真实边界。输入仅当前固定相机RGB、当前腕部RGB与同帧15维本体观测，不使用未来帧或多帧窗口。动作使用日志中的8维canonical接口，place不补入额外颜色，96帧稳定等待保持原记录。','','模型工作进程禁止读取_private/evaluator路径；真值只在评分进程使用。全部重新读取的原查询概率与旧封存结果最大差异为 '+str(maxdiff)+'，小于1e-4。完整状态组、逐命题配对与来源分别见实验results/part1_metrics.json、part1_scored_records.json及本报告旁CSV。']
 (R/'part1_prediction_pairing.md').write_text('\n'.join(lines)+'\n');print('PART1_SCORED',len(records),'transitions',len(records)*9,'paired facts','max_probability_difference',maxdiff)
if __name__=='__main__':main()
