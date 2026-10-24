from common import *
import csv,html,re,base64

def main():
 guard('interpret')
 pairs=list(csv.DictReader((REPORT/'paired_contrasts.csv').open()));out=[];errors=[]
 for size in [20,40]:
  for seed in [0,1]:
   r=next(r for r in pairs if r['seed']==str(seed)and r['split']=='test'and r['mode']=='probe'and r['fact']=='relation9_exact'and r['stratum']=='all'and r['change']=='all'and r['contrast']==f'joint_minus_control_roots{size}')
   out.append(f"| {size} | {seed} | {100*float(r['value']):.2f} | [{100*float(r['lo95']):.2f}, {100*float(r['hi95']):.2f}] |")
   for arm in ARMS:
    for split in ['test','externalC']:
     a=dict(np.load(B/f'cells/roots{size}/data/predictions_{arm}_seed{seed}_{split}.npz'))
     for mode in ['current','free3']:
      mask=a['mode']==mode;y=a['cat'][mask,:3];yp=a['prob'][mask,:24].reshape(-1,3,8).argmax(-1);wrong=yp!=y;tb=wrong&(yp<3)&(y<3)
      errors.append({'size':size,'seed':seed,'arm':arm,'split':split,'mode':mode,'object_field_errors':int(wrong.sum()),'within_table_bin_errors':int(tb.sum()),'fraction':float(tb.sum()/wrong.sum())if wrong.any()else None})
 with (REPORT/'object_error_types.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(errors[0]));w.writeheader();w.writerows(errors)
 text='''# 结果解释与交接

本轮已完成全部8条主训练和8个查询头，主训练147.05分钟，48,000＋4,800更新；无恢复重跑、无额外试验。观察到的结论分为三个层次。

**增加根数没有稳定降低共同A/B测试集上的平均NLL。** 四个规模配对的查询NLL点估计全部上升；其中仅当前损失组的种子1上升0.222，其整根95%区间为[0.077,0.414]。未来梯度组三步NLL在两个种子上分别变化+0.283和−1.051，方向不一致；全部四个三步规模差的区间均包含0。不能据此宣称“扩到40根已经解决泛化”，也不能凭这一套固定子集的两点断言更多经验永远无效。

**未来梯度进入E，对同场景新根上的线性关系组合读出有一致但有限的正证据。** 相同规模和种子内，未来梯度组的probe九关系同时正确率均高于控制组：

| 训练根 | 种子 | 未来梯度组减控制组（百分点） | 整根95%区间（百分点） |
| --- | --- | --- | --- |
'''+ '\n'.join(out)+'''

这四个区间都高于0，但对象仍是历史已见数据重新隔离出的10个A/B根、固定已知关系与一个线性查询头；根重采样没有涵盖更多训练种子、更多训练子集，未作多重检验校正。查询NLL的四个路由差区间仍均包含0；完整RSSM三步NLL的四个路由差区间也均包含0。不能把上述局部读出证据扩大为整体动态模型、长链或闭环规划优势。规模变化对两路的作用差见paired_contrasts.csv，两种子的三步结果并不一致。

**完整位置与外部持物组合仍是缺口。** A/B三步九关系在所有8个模型的非持物初态均24/24正确；规模变化的九关系收益来自仅2个持物测试根，必须保留小分母限制。40根模型在C非持物初态均47/48正确，但持物初态只有0–2/12正确（4个根）。C同时改变场景和初始持物→后续对象组合，不能把差距唯一归为场景变化。

补充的保存预测错误分解是事后描述，不更改主指标：A/B三步物体分类错误中，75%–100%是台面左/中/右之间混淆；40根、种子0控制模型九关系30/30全对，但五字段仅3/30全对，其38个物体字段错误全部为台面区间混淆。它解释了两种口径为何可能相差很大，不证明错误的唯一因果来源；也不能据此删去位置字段。逐条件原始计数在object_error_types.csv。

这轮控制共同20根的归一化、类别权重和probe监督预算，且保持每组6000主更新。它不是普通全流程扩容、不是等暴露对照，也不是无符号标签学习。原指南完整规划、长链和独立D验收仍未完成。本次授权的训练、预测、统计与报告已完成，没有自动增加拟合、采集、D测试或局部Holding修补。

交付位置：实验configs/source/models/data/results/logs在`experiments/rssm_scaling_run_20261019`；报告、图表、根级CSV和字段说明在`reports/rssm_scaling_run_20261019`。`REPORT.md`是预先固定报告程序的完整输出；本文件补充结果解释，不替换预先指定的主指标。`DATA_DICTIONARY.md`说明字段与分母。最终核验见`results/final_verification.json`，封存清单见`configs/delivery_manifest.json`。
'''
 (REPORT/'INTERPRETATION.md').write_text(text)
 # Self-contained HTML exposes all tables without changing their numeric contents.
 parts=[];lines=(text+'\n\n'+(REPORT/'REPORT.md').read_text()+'\n\n'+(REPORT/'DATA_DICTIONARY.md').read_text()).splitlines();i=0
 def inline(s):
  s=html.escape(s);s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s);s=re.sub(r'`(.+?)`',r'<code>\1</code>',s);return s
 while i<len(lines):
  line=lines[i]
  if not line.strip():i+=1;continue
  if line.startswith('|'):
   chunk=[]
   while i<len(lines)and lines[i].startswith('|'):chunk.append(lines[i]);i+=1
   cells=[[inline(c.strip())for c in row.strip('|').split('|')]for row in chunk]
   parts.append('<div class="table"><table><thead><tr>'+''.join('<th>'+c+'</th>'for c in cells[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+c+'</td>'for c in row)+'</tr>'for row in cells[2:])+'</tbody></table></div>');continue
  if line.startswith('!['):
   match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
   if match:
    raw=(REPORT/match[2]).read_bytes();parts.append('<img alt="'+html.escape(match[1])+'" src="data:image/png;base64,'+base64.b64encode(raw).decode()+'">');i+=1;continue
  if line.startswith('#'):
   n=len(line)-len(line.lstrip('#'));parts.append(f'<h{min(n,6)}>'+inline(line[n:].strip())+f'</h{min(n,6)}>')
  else:parts.append('<p>'+inline(line)+'</p>')
  i+=1
 css='body{max-width:1200px;margin:42px auto;padding:0 24px;color:#182536;background:#fafbfc;font:17px/1.7 system-ui,"Noto Sans CJK JP",sans-serif}h1{font-size:28px}h2{margin-top:40px;border-bottom:1px solid #ccd4df;padding-bottom:8px}img{width:100%;background:white}table{border-collapse:collapse;background:white;min-width:100%}td,th{border:1px solid #d4dce5;padding:8px 10px;text-align:left}th{background:#e9f0f8}tr:nth-child(even){background:#f5f8fc}.table{overflow-x:auto;margin:20px 0}code{font-size:.9em;word-break:break-word}strong{color:#153e70}p{margin:16px 0}@media print{body{max-width:none;margin:0;font-size:10pt}.table{overflow:visible}tr{break-inside:avoid}}'
 (REPORT/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RSSM规模与泛化诊断</title><style>'+css+'</style><body>'+''.join(parts)+'</body></html>')
 save(B/'results/interpretation_receipt.json',{'utc':now(),'saved_prediction_only':True,'new_fits':0,'new_inference':0,'new_thresholds':False,'posthoc_object_error_decomposition':True,'main_metrics_unchanged':True,'HTML_self_contained':True})
 print('INTERPRETATION_AND_HTML_COMPLETE',flush=True)
if __name__=='__main__':main()
