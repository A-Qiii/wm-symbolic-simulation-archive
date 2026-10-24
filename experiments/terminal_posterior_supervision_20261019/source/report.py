from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ast,html,base64
COL={'native':'#8a9098','control':'#3974b2','terminal':'#d58b26'}
EN={'native':'Original checkpoint','control':'Continue original loss','terminal':'Add terminal posterior loss'}
def table(h,rr):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|'for r in rr])
def rows(name):return list(csv.DictReader((S/'results'/name).open()))
def main():
 guard('eval');metrics=rows('metrics.csv');pairs=rows('current_pairs.csv');assimilation=rows('assimilation.csv');memory=rows('memory.csv');false=rows('false_relations.csv');changes=rows('internal_changes.csv');cases=read(S/'results/cases.json');audit=read(S/'results/training_audit.json');selection=read(S/'configs/models_frozen.json');training=read(S/'results/training_complete.json');evaluation=read(S/'results/evaluation_complete.json');figs=[]
 def metric(seed,variant,split='exploration',mode='current',horizon=0,cohort='all',field='all'):
  return next(r for r in metrics if(r['seed'],r['variant'],r['split'],r['mode'],r['horizon'],r['cohort'],r['field'])==(str(seed),variant,split,mode,str(horizon),cohort,field))
 def finaldev(r):return read(S/("results/train_"+r["arm"]+"_seed"+str(r["seed"])+".json"))["candidates"][-1]["dev_score"]
 def diffci(r):return '仅1根，不估计区间'if int(r['roots'])<2 else f"{float(r['root_mean'])*100:+.2f} [{float(r['ci_low'])*100:+.2f}, {float(r['ci_high'])*100:+.2f}]百分点"
 def val(r):return f"{int(float(r['count']))}/{r['n']}"
 def finish(f,name):f.tight_layout(pad=2);f.savefig(S/'figures'/name,dpi=150,bbox_inches='tight');plt.close(f);figs.append(name)
 fig,ax=plt.subplots(figsize=(9,3.7));st=audit['data']['train']['stats']
 for j,(f,label)in enumerate([('held_bits','Held'),('InDrawer_bits','InDrawer')]):ax.bar(np.arange(5)+(j-.5)*.3,[st[f'h{i}_{f}']for i in range(5)],width=.3,label=label)
 ax.set(xticks=range(5),xticklabels=['h0 / initial','h1','h2','h3','h4 / final'],ylabel='True positive object-relation bits',title='Original train40: 120 branches, 120 states per boundary');ax.legend();finish(fig,'training_distribution.png')
 fig,axes=plt.subplots(1,2,figsize=(12,4));settings=[('validation','nonterminal','Dev h1-3'),('validation','terminal','Dev h4'),('exploration','nonterminal','Explore h1-3'),('exploration','terminal','Explore h4')]
 for seed in [0,1]:
  for j,v in enumerate(COL):axes[seed].bar(np.arange(4)+(j-1)*.24,[float(metric(seed,v,s,cohort=c)['root_mean'])*100 for s,c,_ in settings],width=.24,color=COL[v],label=EN[v])
  axes[seed].set(xticks=range(4),xticklabels=[x[2]for x in settings],ylim=(0,105),ylabel='Root-mean task exact (%)',title=f'seed {seed}: observed state');axes[seed].legend(fontsize=7,loc='lower left')
 finish(fig,'current_relations.png')
 fig,axes=plt.subplots(1,2,figsize=(12,4));settings=[('one_step',1,'Next action'),('free_root',4,'Root free h4'),('free_suffix',1,'Suffix h1'),('free_suffix',2,'Suffix h2'),('free_suffix',3,'Suffix h3')]
 for seed in [0,1]:
  for j,v in enumerate(COL):axes[seed].bar(np.arange(5)+(j-1)*.24,[float(metric(seed,v,mode=m,horizon=h)['root_mean'])*100 for m,h,_ in settings],width=.24,color=COL[v],label=EN[v])
  axes[seed].set(xticks=range(5),xticklabels=[x[2]for x in settings],ylim=(0,105),ylabel='Root-mean task exact (%)',title=f'seed {seed}: exploratory20 future');axes[seed].legend(fontsize=7,loc='lower left')
 finish(fig,'future_predictions.png')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for seed in [0,1]:
  for j,v in enumerate(COL):
   rr=[next(r for r in memory if(r['seed'],r['split'],r['variant'],r['field'],r['visibility'])==(str(seed),'exploration',v,f,vis))for f,vis in [('held','all'),('InDrawer','visible'),('InDrawer','occluded')]]
   axes[seed].bar(np.arange(3)+(j-1)*.24,[float(r['count'])/int(r['n'])*100 for r in rr],width=.24,color=COL[v],label=EN[v])
  axes[seed].set(xticks=range(3),xticklabels=['Persistent held','Visible InDrawer','Occluded InDrawer'],ylim=(0,105),ylabel='True relation retained (%)',title=f'seed {seed}: exploratory20');axes[seed].legend(fontsize=7,loc='lower left')
 finish(fig,'memory_retention.png')
 fig,axes=plt.subplots(1,2,figsize=(10,3.7))
 for seed in [0,1]:
  for v in ['control','terminal']:
   rr=read(S/f'results/train_{v}_seed{seed}.json');c=rr['candidates'];axes[seed].plot([r['update']for r in c],[r['dev_score']for r in c],'o-',color=COL[v],label=EN[v]);r=rr['selected'];axes[seed].scatter([r['update']],[r['dev_score']],s=130,facecolors='none',edgecolors=COL[v],linewidths=2)
  axes[seed].set(xlabel='Continuation updates',ylabel='Common development objective (lower is better)',title=f'seed {seed}: candidates selected before exploration');axes[seed].legend(fontsize=7)
 finish(fig,'development_selection.png')
 public_inputs={};em={}
 for split in ['train','validation']:
  for e in read(S/f'data/episodes_{split}.json'):em[e['episode_id']]=e
 for r in read(R/'data/data_manifest.json')['records']:
  if r['split']in ['old_band','gap']:
   em.setdefault(r['episode_id'],{'rows':[]})['rows'].append(r)
 for e in em.values():e['rows'].sort(key=lambda r:r['step'])
 for ci,c in enumerate(cases):
  ix={(r['variant'],r['mode'],r['target']):r for r in c['rows']if r['mode']in ['current','original_head_current']};fig,axes=plt.subplots(2,5,figsize=(16,6.4),gridspec_kw={'height_ratios':[1.4,1]})
  for t in range(5):
   rr=em[c['episode_id']]['rows'];p=rr[0]['obs_before']if t==0 else rr[t-1]['obs_after'];public_inputs[p]=sha(p);f=np.load(p);axes[0,t].imshow(f['rgb_static']);axes[0,t].axis('off');axes[0,t].set_title('h0 / real initial'if t==0 else f"h{t}: {c['actions'][t-1]}\n{c['statuses'][t-1]}",fontsize=8)
   data=[ix['native','current',t]['truth']]+[ix[v,'current',t]['task_prob']for v in COL]+[ix['terminal','original_head_current',t]['task_prob']]
   axes[1,t].imshow(data,vmin=0,vmax=1,cmap='Blues',aspect='auto');axes[1,t].set_xticks(range(9),['H-R','H-B','H-P','I-R','I-B','I-P','D-C','D-P','D-O'],rotation=90,fontsize=7);axes[1,t].set_yticks(range(5),['True','Original','Control','Terminal','Terminal state / old head']if t==0 else ['']*5,fontsize=8);axes[1,t].set_title(f'Boundary {t}',fontsize=9)
  fig.suptitle(f"{c['episode_id']} / seed {c['seed']}\n{c['reason']}",fontsize=12);name=f'case_{ci:02d}.png';finish(fig,name);c['figure']=name
 save(S/'configs/report_inputs.json',{'utc':now(),'use':'Posthoc case images only, no training/selection','sha256':public_inputs})
 save(S/'results/cases_with_figures.json',cases)
 md=['# 真实终点后验监督：单项配对续训','结论：补齐真实终点h4的直接符号监督，没有得到稳定的失败纠错收益。探索20根中，相比原损失续训，当前全部关系正确数为种子0的224→223/240、种子1的220→220/240；h4正确数均不变。目标真实抓取失败链两种子仍在h1虚构持红，后续4个观察边界全部未修正。没有修改模型结构，没有重置状态，也没有把历史失败根加入训练。', '## 已核实的代码与数据事实',
 '原 history 只在h0调用initial(E)。原损失以“是否存在下一动作”的step_valid给posts[t]加权，实际监督h0–h3，真实终点h4被padding掩码排除；h4 prior仍有符号监督，h4 posterior只收到KL等间接约束，不能视作等价监督。',
 table(['集合','根/分支/转移','实际执行失败','非终点失败','实际抓取失败','h0入屉正例'],[[sp,f"{audit['data'][sp]['roots']}/{audit['data'][sp]['stats']['h0_boundaries']}/{sum(audit['data'][sp]['stats'][k]for k in ['success','refused','executed_failure'])}",audit['data'][sp]['stats']['executed_failure'],audit['data'][sp]['failed_nonterminal_examples'],audit['data'][sp]['failed_lift_examples'],audit['data'][sp]['stats']['h0_InDrawer_bits']]for sp in ['train','validation']]),
 '训练8次、开发2次执行失败全部是第4动作place_in_drawer。它们的任务关系标签仍是物体成功入屉（任务判定失败与关系标签不是同一件事），没有“抓取失败后实际空手”的训练例，也没有后续纠错动作链。原模型在训练600个边界和480个一步目标的任务关系都已全对，不能用“存在8个失败标签”推断已有所需纠错经历。',
 '![训练边界分布](figures/training_distribution.png)','横轴h0–h4是实际观察边界；纵轴是120分支×3对象中的关系正例数，蓝色持有、橙色入屉。h0有24个持有正例、0个入屉正例。上一轮重置同时丢弃递归历史并改变初始化分布，因此退化不能只解释成“丢失记忆”。',
 '## 唯一改动与配对控制','实验损失 = 原完整损失 + 0.5×真实终点posterior的符号NLL总和÷有效转移数。每条4转移分支新增项等于0.125×平均终点NLL，与原单个当前边界的系数相同；h0–h3及所有先验、多步、重建、KL、结果损失权重保持不变。按每条序列有效长度选择终点，排除padding。对照继续使用原完整损失。',
 '两组加载同一个原joint模型第6000步及其AdamW状态；使用同样40训练根、原索引/潜变量随机种子序列、8×2微批、750次更新、固定3e-5学习率、原k3多步项和分组裁剪。两个种子共4次短续训、3000次更新。原DINO冻结，原编码器/RSSM/读出正常共同更新；因此这是监督缺口修复，不能声称只改了状态更新网络。',
 '所有组都按相同开发综合目标（原k3总损失加终点项）在250/500/750步选点，测试前封存。主表用选点结果，另报全部750步配对。20个历史探索根已被过去分析看过，只能用于探索，不是盲测。没有C/D、新采集或上传。',
 table(['种子','组','开发选择步','选择目标','最终750目标'],[[r['seed'],r['arm'],r['update'],f"{r['dev_score']:.6f}",f"{finaldev(r):.6f}"]for r in selection['selection']]),
 '![开发选择](figures/development_selection.png)','横轴续训步数，纵轴同一开发综合损失，越低越好。蓝线对照、橙线终点监督；空心圈是事先定义规则选中的检查点。选点不使用任何历史探索失败例。',
 '## 当前状态与终点纠正',
 table(['种子','集合','边界','原模型','原损失续训','新增终点监督'],[[seed,sp,cohort]+[val(metric(seed,v,sp,cohort=cohort))for v in COL]for seed in [0,1]for sp in ['validation','exploration']for cohort in ['nonterminal','terminal','all']]),
 '![当前关系](figures/current_relations.png)','横轴分别是开发10根和探索20根的h1–h3、h4，纵轴每根平均的全部任务关系正确率。灰色原模型、蓝色原损失续训、橙色补终点监督。两个种子分开，分支和边界不是独立根。',
 table(['种子','集合','子集','对照正确','实验正确','纠正','退化','根平均差及95%区间'],[[r['seed'],r['split'],r['cohort'],f"{r['reference_correct']}/{r['n']}",f"{r['intervention_correct']}/{r['n']}",r['corrected'],r['regressed'],diffci(r)]for r in pairs if r['reference']=='control'and r['intervention']=='terminal'and r['split']in ['validation','exploration']and r['cohort']in ['all','failure_terminal','failure_nonterminal']]),
 '区间按独立根成组自助抽样2000次；两个训练种子、单次划分不代表方法的稳定泛化。立即失败后的子集和后续多次观察不可混成独立失败次数。',

 table(['种子','集合','组','边界','观察前先验正确','观察后正确','纠正/退化'],[[r['seed'],r['split'],r['variant'],r['cohort'],f"{r['prior_correct']}/{r['n']}",f"{r['posterior_correct']}/{r['n']}",r['corrected']+'/'+r['regressed']]for r in assimilation if r['split']in ['validation','exploration']and r['cohort']in ['terminal','failure_nonterminal']]),
 '这张表比较同一模型吸收当前观察前后的状态读出，保留自然先验/后验采样差异；不是跨模型的性能差。不能把已有正确先验延续到后验算作纠错成功。',
 '## 一步与真正不再观察的后缀',
 table(['种子','预测条件','原模型','对照','终点监督'],[[seed,label]+[val(metric(seed,v,mode=m,horizon=h))for v in COL]for seed in [0,1]for m,h,label in settings]),
 '![未来预测](figures/future_predictions.png)','纵轴每根全部任务关系正确率，颜色含义与上图相同。Next action从当前真实边界出发预测一步；Root free h4仅从h0出发；Suffix h1/h2/h3从真实中间边界出发后只调用imagine，不再读取后续图像。后者样本数分别180/120/60；不能把逐步重新观察的答案称为自由预测。已有数据在h4终止，没有h4之后的真实动作标签，无法直接检验终点被纠正后的后续预测；一步/后缀指标反映共享参数改变对较早边界预测的影响。',
 '## 等750步控制',
 table(['种子','条件','对照750','终点750'],[[seed,label]+[val(metric(seed,v+'_final750',mode=m,horizon=h))for v in ['control','terminal']]for seed in [0,1]for m,h,label in [('current',0,'当前全部关系'),('one_step',1,'下一动作'),('free_root',4,'根纯开环h4'),('free_suffix',3,'中间边界后纯后缀3步')]]),
 '## 仍成立关系、误报与内部状态',
 table(['种子','关系/可见性','原模型','对照','终点监督'],[[seed,f+'/'+vis]+[val(next(r for r in memory if(r['seed'],r['split'],r['variant'],r['field'],r['visibility'])==(str(seed),'exploration',v,f,vis)))for v in COL]for seed in [0,1]for f,vis in [('held','all'),('InDrawer','visible'),('InDrawer','occluded')]]),
 '![持续关系](figures/memory_retention.png)','这些关系在动作前后都真实成立。纵轴正例召回率，横轴持续持有、窗口内可见入屉、窗口内不可见入屉；蓝色和橙色需同时保留已成立关系，不能以全判空来减少误报。可见性由既有独立评分记录给出，未输入模型。',
 table(['种子','组','额外持有','额外入屉','漏持有','漏入屉'],[[seed,v]+[int(float(next(r for r in false if(r['seed'],r['split'],r['variant'],r['field'])==(str(seed),'exploration',v,f))['count']))for f in ['extra_held','extra_in','missing_held','missing_in']]for seed in [0,1]for v in COL]),
 '误报/漏报是240边界×3对象上的关系位数量，不是独立根数。',
 table(['种子','组','当前关系（原共享头读新状态）','下一步（原共享头读新状态）'],[[seed,v,val(metric(seed,v,mode='original_head_current')),val(metric(seed,v,mode='original_head_one',horizon=1))]for seed in [0,1]for v in ['native','control','terminal']]),
 table(['种子','集合','组','h平均绝对变化','q平均TV','E平均绝对变化'],[[r['seed'],r['split'],r['variant'],f"{float(r['h_mean_absolute_change']):.6g}",f"{float(r['q_mean_total_variation']):.6g}",f"{float(r['E_mean_absolute_change']):.6g}"]for r in changes if r['split']in ['old_band','gap']]),
 '内部h、q和观察编码E的变化证明本次续训实际改变了预测状态，不是只另加一个答案头；是否变好仍以未来预测为准。将原共享头读新状态是一次预声明的单组件替换诊断，潜空间坐标也可能变化，不能用它唯一归因于读出或后验。没有进一步修补单例。',
 '## 实际链与第一个有影响的错误']
 for c in cases:
  if c['reason']=='known_actual_failed_grasp_transfer':md.append('本例第1动作实际抓取失败；第一个有影响的错误就是h1仍高置信度判定持红，随后沿虚构持有推出红入屉。新增终点监督没有纠正这一链，不能把终点监督覆盖的改进解释成非终点抓取失败泛化。')
  elif c['reason']=='first_current_regression':md.append('该根最早在h0把真实持红误认成持粉。此次配对新增差异发生在h2：真实红入屉概率从对照0.5266降到终点监督0.4744，越过固定0.5阈值；用原共享头读终点组同一内部状态为0.5314，恢复这一边界。这个差异对读出敏感，不能唯一归因于历史状态，也不能把h2说成整条链最早错误。')
  else:md.append('该训练例最后动作记录为执行失败，但红和粉的真实关系标签均已入屉；原模型、对照和实验原本都读对，因此它不是所需的“实际空手却预测持有”的纠错经历。')
  md +=[f"{c['episode_id']}，种子{c['seed']}，{c['reason']}。最早当前关系错误边界（含h0）：{c['first_influential_error']}。None表示该条观察链全部正确。",f"![实际链](figures/{c['figure']})",'上排为h0–h4的真实静态相机图像，标题给出实际动作与success/failed/refused记录。下排各列H-R/B/P为红/蓝/粉持有，I-R/B/P为入屉，D-C/P/O为抽屉关/半开/开；深蓝接近1、浅色接近0。行依次是真实标签、原模型、原损失续训、终点监督、终点监督内部状态经原共享头读出。两个目标失败图来自同一个根的两个种子，不能算两次独立泛化成功。']
 md+=['## 对任务主线的决定','保留终点监督缺口的代码修正证据，但不将本轮模型升级为稳定纠错方案。开发和探索的终点正确率没有提升，同一模型先验到终点后验没有新增整组关系纠正；目标失败链依旧错误。选点时种子1的一步多1个正确、种子0少1个，不构成两种子一致收益；固定750步后种子1这一点改善消失，种子0三步后缀还少1个正确。','不追加h/z支线、不继续调参或增训。现有数据最直接的缺口是非终点实际抓取失败及其后续真实观察/动作，同时必须保留已成立的入屉关系。已保存最小补充建议proposed_data_gap_plan.json，尚未执行：预登记8个训练根、4个开发根，每根2条5技能分支，上限120转移；先放入一个物体，再尝试另一个物体及固定后续动作，完整保留成功、失败和拒绝。不得复制历史失败根几何或按模型表现挑样本；若不足4个训练根和2个开发根出现带3步后续的实际抓取失败，报告缺口并停止，不伪造标签或无限重采。该建议需另行决策，不能代替原指南正式数据配额。','离线NLL首次汇总因float32把上限1−1e−9舍入成1出现NaN。已保留首次输出与修复记录，仅在同一公式前转float64；7480行正确率/区间和其余5张指标表逐项或逐字节不变，所有最终NLL有限。模型、开发选择、阈值、训练和推理均未重跑。']
 md+=['## 边界、成本与可复现交付',f"实际续训{training['seconds']:.1f}秒；评估{evaluation['seconds']:.1f}秒。4组每组750步，12个候选模型；原模型{evaluation['original_rows_exact']}行既有预测逐数组完全复现。全量日志、每边界预测、根级CSV、协议、输入与模型哈希均在本目录。没有完整6000步重训，没有新采集、C/D或上传。",'这次修正解决的是一个真实损失掩码缺口。它不能自动补齐训练中不存在的抓取失败—新观察—恢复动作经历。若结果没有稳定收益，应停止重复调参，优先设计前瞻的最小纠错数据对照；本轮不会执行新采集。原指南的完整TS/LTL闭环验收仍未完成。']
 txt='\n\n'.join(md)+'\n';(S/'REPORT.md').write_text(txt)
 source=(P/'experiments/object_identity_readout_20261019/source/report.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='render_markdown');ns={'re':re,'html':html};exec(compile(ast.Module(body=[node],type_ignores=[]),'<render>','exec'),ns);body=ns['render_markdown'](txt)
 for name in figs:body=body.replace('figures/'+name,'data:image/png;base64,'+base64.b64encode((S/'figures'/name).read_bytes()).decode())
 (S/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>终点后验监督</title><style>body{max-width:1300px;margin:36px auto;font:16px/1.7 system-ui;color:#203047;padding:0 22px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{border:1px solid #ced8df;padding:6px;overflow-wrap:anywhere}th{background:#e9f3f3}img{max-width:100%}p{overflow-wrap:anywhere}h1,h2{color:#08766b}</style>'+body+'</html>')
 save(S/'results/report_complete.json',{'utc':now(),'figures':figs,'cases':len(cases),'independent_case_roots':len({c['root_id']for c in cases})});print('REPORT',len(figs),flush=True)
if __name__=='__main__':main()
