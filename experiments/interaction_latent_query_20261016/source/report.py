from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
import html,csv,gzip
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=FONT.get_name();plt.rcParams['axes.unicode_minus']=False
NAMES={'generic':'固定通用特征','ae':'仅重构表示','interaction':'交互预训练表示'};COL={'generic':'#6c7a89','ae':'#d58a29','interaction':'#2676b8'}
def ratio(r):return r['all_valid_facts_correct_states']/r['states']
def main():
 R.mkdir(exist_ok=True);(R/'figures').mkdir(exist_ok=True);(R/'examples').mkdir(exist_ok=True);metrics={s:read(E/'results'/f'{s}_metrics.json')for s in['C','D']};bound=read(E/'results/bound_action_control.json');loops=read(E/'results/closed_loop.json');da=read(E/'results/D_data_audit.json');audit=read(E/'results/data_audit.json');budget=read(E/'configs/semantic_budget.json');label=read(E/'data/_private/budget_labels.json');records=read(E/'results/D_records.json');rows=read(E/'data/D/public/states.json');pairs=read(E/'data/D/public/pairs.json')
 table=[]
 for split,mm in metrics.items():
  for key,d in mm.items():
   arm,seed=key.split('_seed');tr=d['semantic_transition']['normal'];r={'split':split,'arm':arm,'seed':int(seed),'current_all_correct':d['all_current_states']['all_valid_facts_correct_states'],'current_states':d['all_current_states']['states'],'current_roots':d['all_current_states']['roots']}
   for name in['actual_current','actual_successor','predicted_successor']:r[name+'_correct']=tr[name]['all_valid_facts_correct_states'];r[name+'_states']=tr[name]['states'];r[name+'_roots']=tr[name]['roots']
   r.update(observation_MSE=d['feature_prediction']['normal']['raw_balanced_MSE'],copy_MSE=d['feature_prediction']['normal']['copy_raw_balanced_MSE']);table.append(r)
 with(R/'metrics.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 # Matched visualization: same metric and endpoint sets for each arm; seed range is not CI.
 fig,axs=plt.subplots(1,2,figsize=(13,5.5),sharey=True)
 conditions=['当前状态\n全部标注窗','真实后继\n正常技能','预测后继\n正常技能'];x=np.arange(3)
 for ax,split in zip(axs,['C','D']):
  for j,kind in enumerate(NAMES):
   values=[]
   for seed in[0,1]:
    d=metrics[split][f'{kind}_seed{seed}'];values.append([ratio(d['all_current_states']),ratio(d['semantic_transition']['normal']['actual_successor']),ratio(d['semantic_transition']['normal']['predicted_successor'])])
   vv=np.array(values)*100;ax.bar(x+(j-1)*.24,vv.mean(0),width=.22,color=COL[kind],label=NAMES[kind]);ax.errorbar(x+(j-1)*.24,vv.mean(0),yerr=np.stack([vv.mean(0)-vv.min(0),vv.max(0)-vv.mean(0)]),fmt='none',ecolor='#222',capsize=3)
  d=metrics[split]['interaction_seed0'];n=d['semantic_transition']['normal']['actual_successor'];ax.set_title(f"{'C开发'if split=='C'else'新D盲测'}：当前{d['all_current_states']['states']}窗 / {d['all_current_states']['roots']}根\n正常技能后继{n['states']}次 / {n['roots']}根");ax.set_xticks(x,conditions);ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2)
 axs[0].set_ylabel('每个状态全部有效命题正确率（%）');axs[1].legend(fontsize=9,loc='upper left');fig.suptitle('固定同一语义标注预算：真实后继读出与预测后继读出分开计分',fontsize=14);fig.text(.5,.015,'颜色表示方案；柱为两个种子的均值，黑线为两种子的最小—最大值（不是置信区间）。当前窗与后继集合不同，不横向混算。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(R/'figures/query_accuracy.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(13,5))
 for j,kind in enumerate(NAMES):
  vals=[]
  for seed in[0,1]:
   d=metrics['D'][f'{kind}_seed{seed}']['feature_prediction']['normal'];vals.append([d['raw_balanced_MSE'],d['copy_raw_balanced_MSE']])
  vv=np.array(vals);axs[0].bar(np.arange(2)+(j-1)*.24,vv.mean(0),width=.22,color=COL[kind],label=NAMES[kind])
  vals=[]
  for seed in[0,1]:
   d=metrics['D'][f'{kind}_seed{seed}']['feature_prediction']['normal'];c=bound[f'D_{kind}_seed{seed}']['normal'];vals.append([d['controls']['shuffled_action']['mean_root_loss_increase'],c['mean_root_loss_increase']])
  axs[1].bar(np.arange(2)+(j-1)*.24,np.array(vals).mean(0),width=.22,color=COL[kind])
 axs[0].set_xticks([0,1],['学习预测','复制当前状态']);axs[0].set_ylabel('四类输入等权的归一化观测MSE（越低越好）');axs[0].set_title('新D正常技能：62次 / 18根\n预测是否优于静止复制');axs[0].legend(fontsize=9)
 axs[1].set_xticks([0,1],['打乱技能及参数\n62次 / 18根','仅换目标对象\n完整绑定24次 / 12根']);axs[1].set_ylabel('相对正确动作的根均衡误差增量');axs[1].set_title('新D动作干预：正值说明预测依赖正确动作');axs[1].axhline(0,color='black',lw=.7)
 for ax in axs:ax.grid(axis='y',alpha=.2)
 fig.text(.5,.015,'灰/橙/蓝仍对应三方案。右图使用按真实变化加权的观测误差；只在右图内比较，不与左图数值直接相减。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(R/'figures/action_controls.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(13,5))
 for kind in NAMES:
  vals=np.array([[ratio(metrics['D'][f'{kind}_seed{s}']['observed_sequence_rollout'][str(h)])for h in[1,2,3]]for s in[0,1]])*100;axs[0].plot([1,2,3],vals.mean(0),'o-',color=COL[kind],label=NAMES[kind]);axs[0].fill_between([1,2,3],vals.min(0),vals.max(0),color=COL[kind],alpha=.12)
 axs[0].set_xticks([1,2,3]);axs[0].set_ylim(0,105);axs[0].set_xlabel('连续执行的完整技能数');axs[0].set_ylabel('预测末端全部有效命题正确率（%）');axs[0].set_title('仅使用真实连续链；阴影为两种子范围');axs[0].legend(fontsize=9);axs[0].grid(alpha=.2)
 for j,k in enumerate(NAMES):
  rr=[r for r in loops if r['kind']==k];success=sum(r['physical_strict_goal_success']for r in rr);false=sum(r['false_completion']for r in rr);axs[1].bar(j-.16,success,width=.3,color=COL[k]);axs[1].bar(j+.16,false,width=.3,color=COL[k],hatch='//',alpha=.5);axs[1].text(j-.16,success+.1,str(success),ha='center');axs[1].text(j+.16,false+.1,str(false),ha='center')
 axs[1].set_xticks(range(3),[NAMES[k]for k in NAMES]);axs[1].set_ylim(0,6.7);axs[1].set_ylabel('匹配根数量（每组6根）');axs[1].set_title('实际闭环：左柱物理目标成功，斜纹柱错误完成声明')
 fig.text(.5,.015,'左：1/2/3步分别72/30/18段，18/12/12根，集合不同。右：固定seed0、相同6根，每次执行后重观测；GT事后评分。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(R/'figures/rollout_closed_loop.png',dpi=160);plt.close(fig)
 # Four concrete normal-action examples: first correct, first readout error, first transition error, hardest fixture.
 candidate=[r for r in records if r['arm']=='interaction_seed0'and r['status']=='success'];errs=read(E/'results/D_errors.json');chosen=[]
 for mode in ['correct','actual_successor_readout','latent_transition_or_off_manifold_query']:
  rr=next((r for r in(candidate if mode=='correct'else errs)if r['arm']=='interaction_seed0'and r['status']=='success'and(r.get('first_component')==mode if mode!='correct'else not any(e['pair_id']==r['pair_id']and e['arm']==r['arm']for e in errs))),None)
  if rr and rr['pair_id']not in[x['pair_id']for x in chosen]:chosen.append(rr)
 for rr in candidate:
  if len(chosen)>=4:break
  if rr['pair_id']not in[x['pair_id']for x in chosen]and rows[rr['before_index']]['condition']!='chain':chosen.append(rr)
 examplelinks=[]
 for i,r in enumerate(chosen):
  fig,axs=plt.subplots(2,3,figsize=(13,7),gridspec_kw={'height_ratios':[1.1,1]});b=rows[r['before_index']];a=rows[r['after_index']]
  for j,(row,title)in enumerate([(b,'动作前：真实静态相机'),(a,'动作后：真实静态相机')]):
   with np.load(row['path'])as z:axs[0,j].imshow(z['rgb_static'])
   axs[0,j].set_title(title);axs[0,j].axis('off')
  with np.load(a['path'])as z:axs[0,2].imshow(z['rgb_gripper'])
  axs[0,2].set_title('动作后：真实腕部相机');axs[0,2].axis('off');yy=np.array(r['true_after']['y']);vv=np.array(r['true_after']['v'])
  for j,(name,title)in enumerate([('current_query','当前状态查询'),('actual_successor_query','真实后继查询'),('predicted_successor_query','预测隐状态查询')]):
   ax=axs[1,j];vals=np.array(r[name]);ax.barh(range(9),vals,color=['#2a9d8f'if val else'#d3d6da'for val in yy]if j else'#869ba7');ax.axvline(.5,color='black',ls='--',lw=1);ax.set_yticks(range(9),FACTS,fontsize=8);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('命题概率');ax.set_title(title)
  fig.suptitle(f"{r['root_id']} · {r['skill']} · 控制器{r['status']}\n{r.get('first_component','预测末端命题全部正确')}",fontsize=12);fig.text(.5,.015,'后继两图：青色=真实后继该命题为真，灰色=假；条长=模型概率，虚线=0.5。抽屉三分类取最大概率。图像均为实拍模拟器输出，未生成预测图像。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,.9]);path=R/'examples'/f'example_{i+1}.png';fig.savefig(path,dpi=150);plt.close(fig);examplelinks.append((path.name,r))
 # Full machine-readable endpoint/arm tables remain local.
 save(R/'results.json',{'metrics':metrics,'complete_argument_controls':bound,'closed_loop':loops,'D_data_audit':da,'interpretation_limits':read(E/'configs/interpretation_limits.json')})
 def ctext(d):return f"{d['all_valid_facts_correct_states']}/{d['states']}"
 lines=['# 无命题标签的技能条件预训练：本机比较实验','',f'生成时间：{now()}。所有模型、数据和运行日志保存在 `{E}`。','',
 '本版完成连续表示/整技能转移预训练、冻结后少标注共享查询、全新D根评估及真实匹配闭环。它是现有小场景中的有界方案比较，不代表原始HULC约6000转移与TS-LTL全指南已完成。没有根据D结果改模型、阈值或选种子。','',
 '## 结论如何读','',
 '真实后继查询把已经实际执行后的RGB输入编码器，检查感知和语义头；预测后继查询只把动作前隐状态与完整技能参数交给转移器，再由同一语义头读取。两者不能混成一个预测分数。模型使用动作并优于静止复制，不足以单独证明交互预训练优于重构。三方案计算预算及预训练批次不严格相同，因此表内差异是方案比较，不能唯一归因于交互损失。','',
 '## 元素、作用和边界','',
 '|元素|实际实现|为什么这样做 / 边界|','|---|---|---|',
 '|输入|当前静态/腕部RGB，机器人15维本体感觉|不使用历史、GT位姿、深度、仿真分割或旧语义隐状态|',
 '|对象对应|RGB固定色相连通域，红/蓝/粉三对象全部保留|提供可审计的身份先验；不能宣称学会通用对象发现；没有任务相关对象筛除|',
 '|通用特征|冻结本机DINOv2-small；双相机对象2×2及场景4×4空间网格|顺序保留，AB独立归一化/PCA32；DINO通道方差保留82.68%，这是共同压缩限制|',
 '|连续表示|三个对象token和一个场景token，各64维|固定通用PCA、只重构编码器、重构加交互编码器三组|',
 '|技能转移|共享对象残差MLP与场景MLP；技能类型、声明目标对象、0/96步后保持契约|整技能为动作单位；没有手写动态前提或逻辑效果|',
 '|防坍塌|固定观测重构目标、当前重构、后继重构、停止梯度匹配、方差监控|固定目标不能跟随编码器一起变常数；对象/几何/场景/本体四组等权，并按实际变化增权|',
 '|共享语义查询|同一个查询MLP读取对象、场景、对象均值及关系类型|Holding三项、In三项、Closed/Open/Intermediate三项；冻结所有第一阶段参数|',
 '|规划/闭环|学习转移给出候选后继；共享查询评分；beam32、深度4、最多6次真实技能|每次执行后重观测；搜索/停止不读GT；目标In(target)且Closed，各概率≥0.8|','',
 '## 数据与标签预算','',
 f"A/B：102个独立根，888次真实调用，1092个去重状态。C：35个独立根，300次调用，423个去重状态。分支和派生窗跟随根，跨分组无相同公开观测及12mm规则下近似重复初始布局。语义数据来自其中18个AB根和9个C根。",
 f"第一阶段零命题标签。第二阶段每组完全相同的144个AB真实状态、最多1296个二值答案；C选头固定72状态、最多648答案。实际有效答案：AB {sum(sum(label[str(i)]['v'])for i in budget['train_state_indices'])}，C {sum(sum(label[str(i)]['v'])for i in budget['development_selection_state_indices'])}。未用预测状态额外校准。其余C用于开发报告，全部C都不是盲测。",'',
 f"新D：{da['roots']}个全新独立根，{da['episodes']}条实际复位轨迹，{da['states']}个标注窗，{da['pairs']}次技能调用；原生状态为{da['native_statuses']}。正常、拒绝和失败在JSON中分表。D种子、布局先排除旧声明；旧D观测未用于新实验。",'',
 '六个已开抽屉根的原始open请求被拒绝，旧采集器据此停止。原拒绝保留；另在相同根下采集真实lift/place/close复位分支。新分支不增加独立根数，也不与原拒绝拼成连续链。所有失败与拒绝保留；没有用GT给72次旧无绑定place补目标（AB54/C18）。对象参数干预主统计仅纳入有完整声明目标的lift/place。','',
 '## 训练预算及公平性','',
 '|方案|表示学习更新|转移学习更新|表示是否接受交互梯度|','|---|---:|---:|---|',
 '|固定通用特征|AB线性PCA，无神经编码器训练|2000|否|',
 '|只重构表示|2000重构，随后冻结|2000|否|',
 '|交互预训练表示|与转移共同2000|含在共同2000中|是|','',
 '每组两个种子0/1。第一阶段只按C观测损失选检查点；AE先按C重构损失选表示。查询头各1200更新，相同初始化种子、相同采样规则、同一标注集合与关系损失。预训练随机数消费不同，批次并未严格配对；没有把本版声称为严格同计算的因果消融。','',
 '## 完整命题结果','',
 '|集合/方案/种子|当前全部标注窗|正常技能前状态|正常技能真实后继|正常技能预测后继|','|---|---:|---:|---:|---:|']
 for split,mm in metrics.items():
  for key,d in mm.items():
   t=d['semantic_transition']['normal'];lines.append(f"|{split} / {NAMES[key.split('_seed')[0]]} / {key[-1]}|{ctext(d['all_current_states'])}|{ctext(t['actual_current'])}|{ctext(t['actual_successor'])}|{ctext(t['predicted_successor'])}|")
 lines+=['','以上为每个状态全部有效命题正确的计数；不是单命题微平均。全部命题准确率、每关系TP/FP/FN、根宏平均、拒绝/失败、1/2/3步的真实样本数见`results.json`，逐次事实见实验`results/D_records.json`。','',
 '## 真实闭环','', '|方案（seed0预声明）|独立匹配根|物理目标成功|模型声明完成|错误完成|实际技能调用|','|---|---:|---:|---:|---:|---:|']
 for k in NAMES:
  rr=[r for r in loops if r['kind']==k];lines.append(f"|{NAMES[k]}|{len(rr)}|{sum(r['physical_strict_goal_success']for r in rr)}|{sum(r['reported_completion']for r in rr)}|{sum(r['false_completion']for r in rr)}|{sum(r['actions']for r in rr)}|")
 lines+=['','每个根三方案独立复位后实际执行。物理成功要求目标对象完整容纳并有承载路径，且抽屉Closed；轻微附带接触本身不被自动记为失败。只有6根，不能据此声称普遍规划可靠。原始每步查询、搜索轨迹、控制动作及物理追踪均保留。','',
 '## 图表与实例','',
 '1. `figures/query_accuracy.png`：三组语义读出，柱/误差线含义在图内。','2. `figures/action_controls.png`：学习预测对比静止复制，以及正确/错误动作条件。','3. `figures/rollout_closed_loop.png`：真实连续1–3步推演和实际闭环。','4. `examples/`：真实前后双相机画面及三种查询结果；颜色、概率与阈值解释在图内。预测的是隐状态和语义，没有伪造预测RGB。','',
 '## 文献依据与本实验的差别','',
 '[DeepSym正文](https://arxiv.org/html/2012.02532v3)以动作效果预测学习表示，同时展示局部效果可能遗漏全局目标信息。本实验借鉴效果监督，保留连续表示并加观测重构；不采用其二值瓶颈或塔高辅助谓词。',
 '[Relational DeepSym相关正文](https://aahmetoglu.com/static/rule_learning.pdf)的实验输入包含对象位姿/类型，预测误差与规划表现也不等价。本实验仅借鉴对象共享转移结构，不使用其GT输入或接触相关对象筛除。',
 '[PARL正文](https://arxiv.org/pdf/2405.03864)使用语言对齐示范和分割RGB-D点云，并执行后重规划。本实验只借鉴高层技能转移及重观测闭环，不能把它作为完全无语义预训练的证据。','',
 '## 复现、核验和剩余问题','',
 f'冻结协议：`{E}/configs/protocol.json`；数据清单：`data/public/`；限定标签：`data/_private/budget_labels.json`；模型：`models/`；全部日志：`logs/`。使用`reproduce.sh`中的分阶段命令，训练输出不会覆盖已有模型文件。',
 '本机Py3.11负责DINO/训练，现有Py3.10负责模拟器。后者transformers缺少Dinov2Model入口，因此采用项目已使用的JSON进程桥；未改依赖、驱动或防火墙。原失败入口和修正证据保留。',
 '首128转移进行了8次丢弃的有限梯度检查，正式模型重新初始化；首50状态存储已记录。原数据清单与D前冻结文件校验、对象排列等变、完整物理步进、匹配复位以及推理进程拒绝私有标签读取均有实测审计。',
 '剩余限制：固定三色身份先验、已知技能集和采样覆盖；同场景模拟环境；只有两个训练种子和6个闭环匹配根；计算/批次不严格配对；部分语义和转移仍有错误。保留失败，不用持续调参或D调参掩盖局限。','']
 (R/'REPORT.md').write_text('\n'.join(lines));(R/'README.md').write_text('# 本机实验交付\n\n打开 [报告](REPORT.md) 或 [图表与真实实例](gallery.html)。表格下载 [metrics.csv](metrics.csv)，完整结果 [results.json](results.json)。\n')
 cards=''.join(f'<section><h2>{name}</h2><img src="figures/{fn}"/></section>'for name,fn in [('共享查询：真实与预测分开','query_accuracy.png'),('动作是否真正被使用','action_controls.png'),('连续推演和实际闭环','rollout_closed_loop.png')])
 cards+=''.join(f'<section><h2>实例 {i+1}：{html.escape(r["skill"])}</h2><img src="examples/{fn}"/></section>'for i,(fn,r)in enumerate(examplelinks));(R/'gallery.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>无命题标签技能预训练</title><style>body{font:16px system-ui;background:#f5f6f8;color:#17202a;margin:30px auto;max-width:1300px}section{background:white;padding:20px;margin:20px 0;border-radius:12px}img{width:100%}p{line-height:1.7}</style><h1>无命题标签的技能条件预训练</h1><p>相同少量语义标注下比较三种表示。D为新根；图内解释颜色、轴、线及样本条件。完整证据和限制见 REPORT.md。</p>'+cards+'</html>')
 print('REPORT_READY',R,flush=True)
if __name__=='__main__':main()
