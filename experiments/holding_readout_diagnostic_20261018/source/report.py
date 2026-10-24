import sys
sys.path.insert(0,'/home/robot/research/projects/wm_symbolic/experiments/holding_readout_diagnostic_20261018/source')
from config import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
import shutil
plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'font.size':10,'axes.unicode_minus':False,'figure.dpi':140})
def one(rows,**kw):
 rr=[r for r in rows if all(r[k]==v for k,v in kw.items())];assert len(rr)==1,(kw,len(rr));return rr[0]
def table(h,rs):return '| '+' | '.join(h)+' |\n| '+' | '.join('---'for _ in h)+' |\n'+'\n'.join('| '+' | '.join(str(v)for v in r)+' |'for r in rs)+'\n\n'
def rgb(path):
 d=np.load(path);return np.concatenate([np.asarray(Image.fromarray(d[k]).resize((180,180),Image.Resampling.NEAREST))for k in ['rgb_static','rgb_gripper']],axis=1)
def main():
 verify();metrics=read(E/'results/actual_metrics.json');locations=read(E/'results/C_error_locations.json');training=read(E/'results/training.json');comparison=read(E/'results/original_boundary_comparison.json');coverage=read(E/'results/exact_sequence_coverage.json');figs=[]
 formnames={'object':'对象token','context':'对象+上下文'};styles=[('object',0,'#3678b4'),('object',1,'#74b2d5'),('context',0,'#d37626'),('context',1,'#edb46c')]
 fig,axs=plt.subplots(1,2,figsize=(12,5))
 for j,(form,seed,color)in enumerate(styles):
  a=one(metrics,form=form,seed=seed,cohort='new_AB_validation');c=one(metrics,form=form,seed=seed,cohort='C_validation');b=one(locations,form=form,seed=seed,group='skill_endpoint');nn=[a['all3_correct'],c['all3_correct'],b['all3_correct']];NN=[a['states'],c['states'],b['states']];fp=[a['empty_any_FP'],c['empty_any_FP'],b['no_Holding_any_FP']];neg=[a['empty_states'],c['empty_states'],b['no_Holding_states']]
  for ax,n,N in [(axs[0],nn,NN),(axs[1],fp,neg)]:
   bars=ax.bar(np.arange(3)+(j-1.5)*.2,np.array(n)/N,.19,color=color,label=f'{formnames[form]} / 种子{seed}');ax.bar_label(bars,labels=[f'{x}/{y}'for x,y in zip(n,N)],fontsize=8,padding=3,rotation=60);ax.set_xticks(range(3),['AB留出\n6根','C验证全部状态\n6根','C技能结束边界\n同6根']);ax.set_ylabel('状态比例');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axs[0].set_title('三个Holding事实同时正确');axs[0].set_ylim(0,1.2);axs[0].axhline(.95,color='#555',ls='--',lw=1);axs[1].set_title('物理标签未持有时，任一Holding误报');axs[1].set_ylim(0,.22);axs[1].axhline(.05,color='#555',ls='--',lw=1)
 h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.91),ncol=4,fontsize=9);fig.suptitle('真实观测可靠性：边界表现较好，过程状态仍有误报');fig.tight_layout(rect=[0,.06,1,.79]);fig.text(.5,.025,'虚线：预先固定的95%准确率 / 5%未持有误报上限。结束边界是事后定位子集，不替换完整验证条件。',ha='center',fontsize=10);fig.savefig(R/'actual_reliability.png',bbox_inches='tight');plt.close(fig);figs.append('actual_reliability.png')
 # Original D cases are read from saved outputs. Independent predicted-z readout is deliberately not executed.
 pp=read(D/'results/predictions.json');ds=read(D/'data/combo/public/states.json');labs=read(D/'data/combo/_private/labels.json');fig,axs=plt.subplots(2,2,figsize=(13,8),gridspec_kw={'width_ratios':[1,1.5]})
 for row,(branch,slot,label)in enumerate([('reverse_then_lift',1,'空手关屉后'),('normal',2,'放下蓝物体后')]):
  r=one(pp,seed=0,world_model='latent',query_arm='original',root_id='D_identity_blue_red_00',branch=branch,slot=slot);truth=labs[str(r['after'])]['y'][:3];axs[row,0].imshow(rgb(ds[r['after']]['path']));axs[row,0].axis('off');axs[row,0].set_title(label+'：真实双相机\n左固定相机 / 右腕部相机');ax=axs[row,1];ax.axis('off');cells=[[f'Holding({c})','真'if truth[j]else'假',*[f"{r[p+'_probability'][j]:.4f}"for p in METHODS],'未评']for j,c in enumerate(COLORS)]
  tab=ax.table(cellText=cells,colLabels=['事实','物理标签','实际读取','单步读取','递归读取','独立预测读取'],cellLoc='center',loc='center',colWidths=[.25,.13,.14,.14,.14,.2]);tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,2)
  for j in range(3):
   tab[(j+1,5)].set_facecolor('#dddddd')
   for k,path in enumerate(METHODS):
    if (r[path+'_probability'][j]>=.5)!=truth[j]:tab[(j+1,k+2)].set_facecolor('#f5bdb8')
  ax.set_title('同一事实、同一实际终点；仅输入潜状态来源不同\n红底为错误，灰色为按可靠性停止条件未执行')
 fig.suptitle('原读取器的两类问题不同：空手关屉单步已错，放下在递推中新出错\n固定原例 D_identity_blue_red_00 / 种子0 / αZ；阈值0.5',fontsize=13);fig.tight_layout(rect=[0,.05,1,.89]);fig.text(.5,.02,'独立头完整C可靠性未通过，未把它应用到预测z；未生成替代数值或据此判定转移模块责任。',ha='center',fontsize=10);fig.savefig(R/'target_boundary_cases.png',bbox_inches='tight');plt.close(fig);figs.append('target_boundary_cases.png')
 # One visible actual-state failure explains why the broad reliability gate was not passed.
 actual=read(E/'results/actual_predictions.json');cd=np.load(E/'data/evaluation/C_validation.npz');candidates=sorted([r for r in actual if r['form']=='context'and r['seed']==0 and r['cohort']=='C_validation'and not cd['y'][r['index']].any()and any(p>=.5 for p in r['probability'])],key=lambda r:(r['root_id'],r['state_id']));chosen=candidates[0];states=read(PRIOR/'data/ABC/public/states.json');state=states[int(chosen['state_id'].split(':')[1])];original=one(read(WHOLE/'results/actual_predictions.json'),seed=0,query_arm='original',cohort='C_validation',state_index=chosen['index']);obj=one(actual,form='object',seed=0,cohort='C_validation',index=chosen['index'])
 fig,axs=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1,1.3]});axs[0].imshow(rgb(state['path']));axs[0].axis('off');axs[0].set_title('真实观测：放入抽屉的中间阶段\n左固定相机 / 右腕部相机');axs[1].axis('off');truth=cd['y'][chosen['index']].astype(bool);cells=[[f'Holding({c})','真'if truth[j]else'假',f"{original['probability'][j]:.4f}",f"{obj['probability'][j]:.4f}",f"{chosen['probability'][j]:.4f}"]for j,c in enumerate(COLORS)];t=axs[1].table(cellText=cells,colLabels=['事实','物理标签','原读取器','独立对象头','独立上下文头'],cellLoc='center',loc='center',colWidths=[.27,.15,.19,.19,.2]);t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,2)
 for j in range(3):
  for k,vals in enumerate([original['probability'],obj['probability'],chosen['probability']]):
   if (vals[j]>=.5)!=truth[j]:t[(j+1,k+2)].set_facecolor('#f5bdb8')
 fig.suptitle(f"真实输入也会误报：{chosen['root_id']} / {chosen['state_id']} / 种子0\nHolding定义要求无外部支撑；接触/放入过渡阶段不能只按夹爪外观判断",fontsize=12);fig.tight_layout(rect=[0,.08,1,.83]);fig.text(.5,.025,'所有列都读取同一个真实z。红底为错误；按上下文头错误根ID/状态ID排序取首例，仅作事后解释。',ha='center',fontsize=9);fig.savefig(R/'actual_C_failure.png',bbox_inches='tight');plt.close(fig);figs.append('actual_C_failure.png');save(E/'results/report_example.json',{'root_id':chosen['root_id'],'state_id':chosen['state_id'],'selection':'first lexicographic context seed0 actual C no-Holding false positive; descriptive only','state_path':state['path'],'phase':state['phase']})
 text='# Holding独立读出诊断与真实训练性能\n\n本轮完成了输入对齐、训练覆盖审计，以及两种输入×两个初始化的固定600更新诊断。四个头在AB留出根表现较好，但完整C真实状态可靠性均未通过；因此没有把它们应用到预测z，没有用探针给转移模块定责，也没有修改正式模型或追加训练。预定前缀替换以至少一种独立读出形式通过可靠性为前提，本轮未触发。\n\n'
 text+='## 训练与验证为什么这样设计\n\n'+table(['元素','做法','原因'],[
 ['真实训练数据','既有36个AB训练根、210个状态；156旧状态加6个既有组合训练根的54状态','利用已授权真实标签和冻结潜特征，不收新数据、不跨根泄漏'],['对象输入','对应的64维上下文化对象token，共享64→32→1头，2113参数','检验对象token能否支持Holding读出'],['上下文对照','对应对象+场景+三对象均值，共192维，共享192→32→1头，6209参数','预定增加潜状态上下文；不输入动作、坐标或原query输出'],['公平预算','每头600更新、每批32状态；同seed两形式用相同根均衡抽样序列','不改变batch、阈值或预算来追求结果/显存'],['优化','AdamW lr=.001、wd=.001、clip1；训练正负比权重8.54545；阈值0.5','训练前固定；只读最终600更新，不搜索超参数或选点'],['可靠性','AB留出、旧C验证、新C验证分别H3≥95%、未持有误报≤5%、每色FPR≤5%及FNR≤10%、阳性至少2根；同形式两个seed均满足','这是本次归因工具的预定使用条件，不是改写原项目验收标准'],['边界','C与D不参与归一化或训练；C只作诊断验证，D未用于选点','避免把测试表现倒灌到诊断头']])
 text+='## 对齐与覆盖\n\n135个转移的前后帧、135个动作编码、四份保存潜状态及对应query均对齐。对象顺序为红/蓝/粉，末步抓取d与查询c已分别核对；动作第8维是posthold/96，非Holding真值。实际z与保存后的编码逐位一致，递推第一步与单步逐位一致，中途无真实状态注入。\n\n旧训练并非缺少全部空手关屉。补读历史longpair物理快照后，与当前posthold96一致的最新训练流覆盖如下；计数是最近转移目标消融训练的抽样，不冒充模型整个训练生命期总次数。下表为seed0、完整1000更新的单步流，mixed最终选750更新、αZ选1000；两种子各前缀及多步流完整次数见CSV。\n\n'
 rr=[r for r in coverage if r['seed']==0 and r['updates_prefix']==1000 and r['stream']=='single'and r['posthold']==1];text+=table(['实际训练条件','转移','独立根','采样次数'],[[r['category'],r['unique_transitions'],r['roots'],r['draws']]for r in rr])
 text+='其中预置打开且未持有的close有48转移/12根，放后close有42转移/42根；连续执行“空手open→close”则为0根。已有12根连续open→close是仍持物的分支。最新多步监督仍集中正常链的12个新AB根，不能把单步覆盖等同于任意顺序鲁棒性。较早的初步表漏读了24个longpair根，已明确标为不完整口径；最终以恢复物理快照后的exact_sequence_coverage为准，未改旧实验。\n\n原正式query最近一次选中150更新的4800次状态抽样中，两种子的未持有状态分别为3115、3139次，持红/蓝/粉为561/587/537和563/556/542次，来自30根。query没有动作序列输入。覆盖与错误关联不足以证明模型“只记动作阶段”，本轮不作该因果断言。\n\n'
 text+='## 先看真实状态可靠性\n\n![真实状态可靠性](actual_reliability.png)\n\n左轴为三个Holding事实同时正确的状态比例；右轴为物理标签未持有时任一Holding误报比例。蓝色两柱是对象头的两个种子，橙色两柱是上下文头；柱上保留n/N。虚线是训练前固定条件。每个数据组只有6个独立根；C结束边界是同一C集的事后子集，没有替换预定完整验证口径。\n\n'
 text+=table(['形式','种子','AB留出H3','完整C验证H3','C未持有误报','C技能结束H3'],[[formnames[f],s,f"{one(metrics,form=f,seed=s,cohort='new_AB_validation')['all3_correct']}/54",f"{one(metrics,form=f,seed=s,cohort='C_validation')['all3_correct']}/273",f"{one(metrics,form=f,seed=s,cohort='C_validation')['empty_any_FP']}/193",f"{one(locations,form=f,seed=s,group='skill_endpoint')['all3_correct']}/48"]for f in FORMS for s in [0,1]])
 text+='完整C的误报集中在靠近物体抓取、降低物体进入抽屉等中间阶段；“未持有”按物理定义，包含物体已受外部支撑但夹爪仍接触的过渡段，不全等于夹爪张开且离物体很远。原Holding定义为双侧承力手指接触且无外部支撑，本轮未更改。C结束边界的对象头两种子48/48、上下文头48/48及47/48，是有价值的有限证据；但不能事后删去错误过程帧来宣称完整可靠性通过。旧C验证还缺少蓝色持物阳性根，属于覆盖不足，不能将缺失的FNR算为0；新C各色2个阳性根且仍有明确误报，停止并非仅由旧C覆盖不足造成。\n\n'
 text+=table(['形式/种子','C颜色','FP / 负事实','FN / 正事实','阳性独立根'],[[formnames[m['form']]+'/'+str(m['seed']),c['color'],f"{c['FP']}/{c['FP']+c['TN']}",f"{c['FN']}/{c['FN']+c['TP']}",c['positive_roots']]for m in metrics if m['cohort']=='C_validation'for c in m['colors']])
 text+='![C真实输入错误](actual_C_failure.png)\n\n图左是真实双相机；右侧所有列都读取同一个实际z，分别为物理事实、原读取器、独立对象头、独立上下文头。红底为阈值0.5下的错误。这个过渡状态的共同误报不能单独证明潜状态没有物理信息，也不能证明读出器结构必然有问题。\n\n'
 text+='## 原读取器的单步与递推需要分开看\n\n'+table(['世界模型','种子','边界（各11根）','实际FP','单步FP','递推FP','仅递推新错'],[[r['world_model'],r['seed'],r['boundary'],r['actual_Holding_FP_roots'],r['one_step_Holding_FP_roots'],r['recursive_Holding_FP_roots'],r['only_recursive_wrong']]for r in comparison])
 text+='αZ空手close的单步已有10/11、6/11错误，递推为11/11、7/11；而放下和放后close的单步均0/11，递推分别6/11、7/11及6/11、6/11。这是两类不同模式。放下后的真实前缀替换原本可定位递推链中的误差传播，但本轮独立读出未通过预定完整可靠性，故停止后续模块归因；这里使用的均是已保存原query概率，没有新运行转移模型。\n\n![具体边界案例](target_boundary_cases.png)\n\n每行对应同一个真实终点，左侧固定/腕部相机，右侧依次为事实、标签、原query实际读取、单步读取、递归读取和独立预测读取。红底表示错误；灰色“未评”表示未满足独立读出的使用条件，不是0或成功。没有用不可靠探针的输出制造模块归因。\n\n'
 text+='## 显存与速度：测得什么、缺少什么\n\n当前训练的是缓存潜特征上的小头，DINO、编码器、世界模型和正式query全部冻结，且本轮无需实例化它们。低显存与此规模相符，不能通过扩大模型或batch来“填满显存”证明训练有效。\n\n'
 text+=table(['形式/种子','训练参数','batch状态','准备/传输秒','600更新秒','PyTorch峰值分配MiB','峰值预留MiB'],[[formnames[r['form']]+'/'+str(r['seed']),r['trainable_parameters'],r['batch_states'],f"{r['data_prepare_transfer_seconds']:.4f}",f"{r['training_wall_seconds']:.4f}",f"{r['peak_allocated_bytes']/2**20:.2f}",f"{r['peak_reserved_bytes']/2**20:.2f}"]for r in training['runs']])
 text+='四个run均真实使用RTX5080 Laptop CUDA。数据预检和准备约0.98秒（其后补读历史快照不包含在这个数值），CUDA初始化0.339秒。训练窗口按250ms采样8次，GPU利用率13%–34%、样本均值25%；设备总显存占用340–378MiB，与PyTorch约18MiB张量/工作区峰值不同，前者还包括CUDA上下文等。8个短窗口样本包含初始化和间隙，不能作为持续利用率或单次kernel占用率。CSV时间为宿主UTC+8，训练记录另存UTC时间。后续run的base内存包含进程内持久工作区，因此不能把“峰值减base接近0”解释为训练不占显存。\n\n'
 text+='最近的转移目标消融只训练67328参数，batch64，1000更新的已记录耗时分别mixed种子0/1为2.204/2.279秒，αZ为1.829/1.865秒；完整小型潜模型752219参数，其他参数冻结。正式query及整头覆盖轮各25473参数、batch32、CUDA；前轮对象In头2113参数、batch32、CPU4线程。这些旧query/In轮未记录训练耗时、峰值显存与利用率，标为不可追溯，不拿今天的空闲值替代。历史转移耗时也不能与本轮不同目标/规模直接比质量。\n\n本轮600更新不足一秒，显存容量不是当前速度限制。实际代码包含许多很小的前向、反向和优化器调用，CUDA初始化及CPU调度相对更显著；没有kernel profiler数据，不能进一步给某个具体kernel定责。后续若需要速度优化，应先量化重复数据读取/初始化及小算子调度，保持相同batch与更新数作对照；本轮没有做额外速度试验，也没有安装软件。\n\n'
 text+='## 结论与证据\n\n已证明输入对齐、相关负例并非全缺失，以及原query的空手close单步错误与放下后递推新增错误有所区别。独立小头的完整真实C可靠性未通过，所以“预测潜状态错误”还是“原读取器对预测分布不稳”仍未被分开；技能结束边界的较好表现应单独保留。下一步的依据是明确区分结束边界与接触过渡状态的使用范围，再决定是否能进行有限的预测潜状态对照；本轮未事后改验证范围、补训练或替换正式模型。\n\n'
 text+=''.join(f'- [{n}.csv]({n}.csv)\n'for n in ['actual_metrics','C_error_locations','exact_sequence_coverage','historical_transition_coverage','historical_query_sampling','original_boundary_comparison','historical_performance'])+'\n'
 text+=f'[训练前冻结协议]({E}/configs/protocol.json) · [训练性能原始记录]({E}/results/training.json) · [GPU采样]({E}/logs/gpu_utilization.csv) · [停止归因记录]({E}/results/attribution_stop.json) · [复核]({E}/results/verification.json) · [封存清单]({E}/configs/delivery_manifest.json)。\n'
 (R/'REPORT.md').write_text(text);save(E/'results/report_assets.json',{'utc':now(),'figures':figs,'report':'REPORT.md'});shutil.copy2(__file__,E/'source/report.py');print('HOLDING_REPORT_READY',len(figs),'figures',flush=True)
if __name__=='__main__':main()
