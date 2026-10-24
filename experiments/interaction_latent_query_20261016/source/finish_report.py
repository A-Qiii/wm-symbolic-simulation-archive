from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import Counter
plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
def main():
 pairs=read(E/'data/public/pairs.json');statuses={r['pair_id']:r['status']for r in read(E/'data/_private/pair_outcomes.json')};coverage=Counter((bool(sum(r['action'][4:7])),statuses[r['id']])for r in pairs if r['split']=='train'and r['skill']=='place_in_drawer');save(E/'results/place_argument_coverage.json',[{'bound_argument':k[0],'status':k[1],'calls':v}for k,v in coverage.items()])
 examples=['D_interaction_chain_pink_1','D_interaction_chain_red_0','D_interaction_chain_pink_0'];fig,axs=plt.subplots(3,4,figsize=(15,14));captions=[]
 for row,rid in enumerate(examples):
  folder=E/'data/closed_loop'/rid/'interaction';dec=read(folder/'_private/decisions.json');actions=read(folder/'_private/actions.json');inds=list(range(min(4,len(dec))));target=COLORS.index(rid.split('_')[-2])
  for col,idx in enumerate(inds):
   ax=axs[row,col];d=dec[idx]
   with np.load(d['public_path'])as z:ax.imshow(z['rgb_static'])
   truth=d['actual_labels_evaluator_only'];actual=bool(truth['y'][3+target]and truth['y'][6]);pr=d['current_probability'];nextaction=actions[idx]['skill']if idx<len(actions)else'停止';status=actions[idx]['native']['status']if idx<len(actions)else d['search']['status'];short=nextaction.replace('_block_table','').replace('_drawer','').replace('place_in','place');ax.set_title(f"边界{idx} → {short}\n{status}",fontsize=11);ax.set_xlabel(f"查询 In={pr[3+target]:.2f} Closed={pr[6]:.2f}\n物理目标={'真'if actual else'假'}",fontsize=10);ax.set_xticks([]);ax.set_yticks([])
  axs[row,0].set_ylabel({'D_interaction_chain_pink_1':'初始已开：完成三步','D_interaction_chain_red_0':'初始关闭且空手：反复被拒绝','D_interaction_chain_pink_0':'初始关闭：抓取后放置被拒绝'}[rid],fontsize=11)
 fig.suptitle('交互组真实闭环：成功链与最早造成失败的动作（固定seed0）',fontsize=17);fig.text(.5,.015,'每格是真实静态相机画面，箭头文字表示随后请求的完整技能。In/Closed是当前视觉查询概率；物理目标由独立评估器事后计算。未展示的重复拒绝仍完整保留。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.05,1,.96]);fig.subplots_adjust(hspace=.5);fig.savefig(R/'figures/closed_loop_examples.png',dpi=150);plt.close(fig)
 diagnostics=read(E/'results/closed_loop_single_substitution.json');decomp=read(E/'results/readout_transition_decomposition.json');path=R/'REPORT.md';s=path.read_text();s=s.replace('## 元素、作用和边界','新D正常后继：交互组52–59/62，只重构组47–50/62。真实匹配闭环：交互组3/6，另外两组0/6；交互组成功均为初始已开抽屉的3根，初始关闭的3根失败。这是当前版本的观测结果，并非严格配对的交互损失因果效应。\n\n## 元素、作用和边界',1)
 s=s.replace('## 文献依据与本实验的差别','''## 最早有影响的错误与单项替换

交互组关闭抽屉的红/蓝根从第一步就请求place，实际因空手被拒绝；粉根第一步抓取成功，第二步place因抽屉未开被拒绝。后续反复请求错误技能，直到预算耗尽。它们不是低层控制器随机失败：控制器按固定契约拒绝不满足条件的请求。

对每条已记录计划，仅用实际观测后继编码替换第一步预测隐状态，保持后续动作和查询头不变。红根第一步原计划目标分数0.997降至0.718；蓝根0.838降至接近0；粉根第二步0.998降至0.021，均不再达到0.8接受阈值。该单项替换证实这些转移预测错误影响了计划接受；不能据此唯一拆分所有编码器和查询头问题。没有追加模型修补或D调参。

数据覆盖提供了一个需后续对照验证的原因：AB有绑定place为186成功、12失败、0拒绝；30次place拒绝全部来自无对象绑定调用（另有24次无绑定成功）。因此绑定place在空手/关闭条件下的拒绝规律没有得到同形式负例覆盖。这个缺口与闭环失败一致，但本版未用补采后的严格配对训练验证其独立因果贡献，也未用GT补齐旧参数。

真实后继与预测后继的错误也会抵消。交互seed0的62次正常技能中，48次两者都正确，6次真实后继正确而预测错误，4次真实后继错误但预测正确，4次都错误；seed1分别55、2、4、1。预测读出有时高于真实读出，不是模型比真实观察更“真实”，而是两个误差路径及终点分布不同。详细逐次替换见`closed_loop_single_substitution.json`。

## 数值一致性和防坍塌检查

第一版单状态在线DINO与缓存批量DINO的bfloat16数值形状不同，在AB索引5造成0.0534的查询概率差；几何/本体感觉及缓存重组相同。用相同batch8重算核心缓存时DINO及查询完全一致。新D离线/在线统一8个当前观测（不足用同帧填充、丢弃额外输出），32个AB状态×3组的96次实测概率差均为0，原失败日志保留。训练缓存与模型未更改；旧AB/C缓存仍存在batch4/8来源差异，这是已披露的数值来源限制。

新D连续隐状态没有坍塌成常数：交互组平均标准差0.707/0.753、有效秩13.7/13.3；只重构组有效秩27.3/26.5。有效秩较低并不自动代表“更好的抽象”，因为当前语义覆盖反而较弱。图中1/2/3步分别来自72/30/18个真实连续片段、18/12/12个根；集合不同，不能把曲线变化唯一归因为推演步长。

## 文献依据与本实验的差别''',1)
 s=s.replace('4. `examples/`：','4. `figures/closed_loop_examples.png`：三条真实闭环的成功与拒绝过程。\n5. `examples/`：');path.write_text(s)
 for name in ['closed_loop_single_substitution.json','readout_transition_decomposition.json','place_argument_coverage.json','delivery_audit.json','runtime_bridge_preflight.json']:save(R/name,read(E/'results'/name))
 gallery=R/'gallery.html';t=gallery.read_text();t=t.replace('</html>','<section><h2>真实闭环：成功与最早失败</h2><p>上行成功完成；中行空手却反复请求放置；下行未打开抽屉就请求放置。每幅图均为实际运行观测，查询概率与物理目标分别显示。</p><img src="figures/closed_loop_examples.png"/></section></html>');gallery.write_text(t)
 (R/'reproduce.sh').write_text('''#!/usr/bin/env bash
# Commands below reproduce individual stages only into a NEW experiment copy.
# Existing model directories intentionally fail on overwrite.
# The canonical frozen paths in common.py must be pointed to the new copy.
set -euo pipefail
P=/home/robot/research/projects/wm_symbolic
E="$P/experiments/interaction_latent_query_20261016"
# Read-only verify of this exact delivered experiment:
python3 "$E/source/verify.py"
# Historical execution sequence (not automatically rerun by this file):
# source "$P/world_model_env.sh"
# python "$E/source/prepare.py"
# python "$E/source/features.py"
# python "$E/source/representation.py"
# python "$E/source/freeze.py"
# python "$E/source/train_stage1.py" --kind generic|ae|interaction --seed 0|1
# python "$E/source/train_queries.py" --kind generic|ae|interaction --seed 0|1
# python "$E/source/evaluate.py" --split C
# source "$P/env.sh"
# python "$E/source/prepare_D.py"
# python "$E/source/run_D.py"
# python "$E/source/run_D_open.py"
# source "$P/world_model_env.sh"
# python "$E/source/worker_fixed_batch.py" --preflight
# python "$E/source/convert_D_fixed_batch.py"
# python "$E/source/evaluate.py" --split D
# python "$E/source/bound_action_control.py"
# source "$P/env.sh"
# python "$E/source/closed_loop_fixed_batch.py"
# source "$P/world_model_env.sh"
# python "$E/source/audit_delivery.py"
# python "$E/source/diagnose.py"
# python "$E/source/report.py"
# python "$E/source/finish_report.py"
''')
 (R/'HANDOFF.md').write_text(f'''# 本机交接

实验目录：{E}
报告：{R}

全部授权的本轮两阶段比较、18新D根和三组6根匹配闭环已实际完成。没有活跃训练/采集任务。无环境权限阻塞；研究局限是闭环仅3/6，关闭初始抽屉的3根失败。

训练：102AB根888调用；开发35C根300调用。Stage1泛化特征/AE/交互三组各2种子，stage2每组相同144AB状态+72C选头状态。D18根72轨迹312窗72调用（62正常/6拒绝/4失败），闭环18轨迹96次调用。交互seed0闭环3/6，其余0/6；不声称同计算或纯交互损失因果优势。

重点证据：results/D_metrics.json、D_records.json、bound_action_control.json、closed_loop.json、closed_loop_single_substitution.json、delivery_audit.json。冻结源：configs/training_frozen.json、stage2_frozen.json、final_before_D.json。兼容/采集/批量修正：bridge_frozen.json、open_branch_collection_amendment.json、inference_batch_frozen.json。原失败日志和入口均保留。

推理环境Py3.11与执行环境Py3.10以JSON进程桥连接，无依赖新增。DINO固定batch8保证新D离线在线一致；96AB复算概率差0。GT仅标签/独立评价，推理进程拒绝私有路径，旧语义参数/隐状态未进入第一阶段，旧D没有进入训练或新盲测。

旧place72无绑定（AB54/C18）未GT补齐，目标干预只统计完整绑定lift/place。AE额外2000重构更新、采样顺序不同，不能作严格同预算归因。正文列出单项替换的首个有影响错误及覆盖缺口；停止D调参。
''')
 print('REPORT_FINISHED',flush=True)
if __name__=='__main__':main()
