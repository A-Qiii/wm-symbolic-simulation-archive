from study import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64,html
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'savefig.dpi':160})
PALETTE={'control':'#2878b5','updated':'#c8552d'};LABEL={'control':'残差配对对照','updated':'残差三分支均衡'};SPLITS=['train','development','evaluation'];SPLABEL={'train':'训练12根','development':'开发6根','evaluation':'历史评估6根（探索性）'}
def table(h,rows):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+'\n'.join('| '+' | '.join(str(x)for x in row)+' |'for row in rows)+'\n'
def main():
 guard('report');(U/'figures').mkdir(exist_ok=True);analysis=read(U/'results/analysis_complete.json');metrics=analysis['metrics'];refs=read(U/'results/prediction_references.json');gates=read(U/'results/gates.json');figures=[]
 def get(v,s,sp,m):return next(r for r in metrics if(r['variant'],r['seed'],r['split'],r['metric'])==(v,s,sp,m))
 def fmt(v,s,sp,m):
  r=get(v,s,sp,m);return f"{r['sum']:.0f}/{r['roots']}"
 def load(v,s,sp):
  path=next(r['path']for r in refs if(r['variant'],r['seed'],r['split'])==(v,s,sp))
  with np.load(path)as z:return {k:z[k]for k in z.files}
 def figsave(f,n,c):f.savefig(U/f'figures/{n}.png',bbox_inches='tight');plt.close(f);figures.append((n,c))
 names=['b0_full_trajectory','b0_h2_future3_joint','h2_pair_joint','h2_and_suffix3_pair_joint','h0_held'];labs=['b0整条观察轨迹','b0当前与未来联合','成败h2完整配对','成败当前与未来联合','h0持有身份']
 fig,axes=plt.subplots(2,3,figsize=(16,8),sharey=True)
 for seed in [0,1]:
  for sp,ax in zip(SPLITS,axes[seed]):
   for j,v in enumerate(PALETTE):
    rr=[get(v,seed,sp,n)for n in names];bars=ax.bar(np.arange(5)+(j-.5)*.32,[r['mean']for r in rr],.3,color=PALETTE[v],label=LABEL[v]);
    for b,r in zip(bars,rr):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.015,f"{r['sum']:.0f}/{r['roots']}",ha='center',fontsize=8,rotation=45)
   ax.set_xticks(range(5),labs,rotation=25,ha='right');ax.set_ylim(0,1.2);ax.set_title(f'{SPLABEL[sp]}，种子{seed}');ax.set_ylabel('独立根正确比例');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
 fig.tight_layout();figsave(fig,'01_main_gates','每列为训练、开发、历史评估，每行为一个模型种子。蓝柱为复用且逐项核验的残差配对对照，橙柱为相同残差模型的三分支均衡训练。第一项要求b0的h0-h6每个观测边界都对；第二项要求b0当前h2与纯想象h3-h5都对；第三、四项要求成功b1和落空b2同时正确。最后一项单列h0，避免后续分数遮掩初始身份问题。纵轴为独立根比例，柱上写分子/分母。历史评估已用于先前分析，只作探索性回归。')
 fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
 for seed in [0,1]:
  for sp,ax in zip(SPLITS,axes[seed]):
   ax.fill_between([.5,1],0,.5,color='#e2f0e4');ax.plot([0,1],[0,1],'--',c='gray');ax.axhline(.5,c='black',lw=.6);ax.axvline(.5,c='black',lw=.6)
   for v in PALETTE:
    d=load(v,seed,sp);p=rel(d['prob']);xx=[];yy=[]
    for rid in sorted(set(d['root_id'])):
     ids=[np.where((d['root_id']==rid)&(d['mode']=='current')&(d['step']==2)&(d['branch']==b))[0][0]for b in [1,2]];t=COLORS.index(str(d['target_color'][ids[0]]));xx.append(p[ids[0],t]);yy.append(p[ids[1],t])
    ax.scatter(xx,yy,c=PALETTE[v],marker='s'if v=='control'else'o',alpha=.7,s=45,label=LABEL[v])
   ax.set_xlim(-.04,1.04);ax.set_ylim(-.04,1.04);ax.set_title(f'{SPLABEL[sp]}，种子{seed}');ax.set_xlabel('真实成功分支目标Held概率');ax.set_ylabel('真实落空分支目标Held概率');ax.legend(fontsize=7);ax.grid(alpha=.2)
 fig.tight_layout();figsave(fig,'02_observation_conditioning','每个点是一根的真实成功/落空配对：横轴为成功分支目标Held概率，纵轴为落空分支概率。蓝方/橙圆分别是对照/均衡训练；右下浅绿区才是两种结果都辨对，灰虚线代表对两种观测给出相同判断，黑线是固定0.5阈值。点可能重叠，独立根数以面板标题为准。它检验观察条件区分，不能单凭该图定位网络内部哪个模块发生修复。')
 fig,axes=plt.subplots(2,3,figsize=(15,7),sharey=True)
 for seed in [0,1]:
  for sp,ax in zip(SPLITS,axes[seed]):
   for v in PALETTE:
    for branch,style in [(0,'-'),(1,'--'),(2,':')]:ax.plot([1,2,3],[get(v,seed,sp,f'suffix{k}_b{branch}_joint')['mean']for k in [1,2,3]],style,marker='o',color=PALETTE[v],label=LABEL[v]+({0:'b0正常另一目标',1:'b1匹配成功',2:'b2匹配落空'}[branch]))
   ax.set_xticks([1,2,3]);ax.set_ylim(-.05,1.05);ax.set_xlabel('h2真实观测后纯想象步数');ax.set_ylabel('完整关系终点正确比例');ax.set_title(f'{SPLABEL[sp]}，种子{seed}');ax.grid(alpha=.2);ax.legend(fontsize=6)
 fig.tight_layout();figsave(fig,'03_learned_successors','从实际h2后验出发，仅调用学习到的imagine转移与共享符号读出，未来图像不再输入。横轴为1/2/3步，纵轴为每个终点完整关系正确比例；蓝/橙是对照/均衡训练，实线为b0另一目标正常分支，虚线为b1匹配成功，点线为b2匹配落空。终点正确不等于整个前缀正确，后者单列在主图。')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for seed,ax in enumerate(axes):
  for v in PALETTE:
   rr=read((C if v=='control'else U)/f'results/train_{"state_update"if v=="control"else"balanced"}_seed{seed}.json');ax.plot([r['step']for r in rr['curve']],[r['objective']for r in rr['curve']],c=PALETTE[v],label=LABEL[v]);ax.scatter([c['update']for c in rr['candidates']],[c['development_objective']for c in rr['candidates']],marker='x',s=65,c=PALETTE[v]);ax.axvline(rr['selected']['update'],ls=':',c=PALETTE[v],alpha=.5)
  ax.set_xlabel('同样1000次续训更新');ax.set_ylabel('完整目标值');ax.set_title(f'种子{seed}');ax.legend();ax.grid(alpha=.2)
 figsave(fig,'04_training','横轴为更新次数，纵轴为完全相同的损失公式。蓝/橙实线是两组各自抽样批次的训练目标；叉号是相同开发集的目标，虚线是按冻结规则选择的checkpoint。两组残差模型和损失完全相同，仅新数据分支从b1/b2配对改成b0/b1/b2均衡；每次抽样根身份、旧样本和latent随机数相同。目标降低仍不能替代当前状态和未来完整前缀的检查。')
 exposure=list(csv.DictReader((U/'configs/exposure_counts.csv').open()));exrows=[]
 for seed in [0,1]:
  for branch in [0,1,2]:
   rr=[r for r in exposure if r['seed']==str(seed)and r['branch']==str(branch)];exrows.append([seed,branch,sum(int(r['control_samples'])for r in rr),sum(int(r['balanced_samples'])for r in rr)])
 primary=['b0_full_trajectory','b0_h2_future3_joint','h2_pair_joint','h2_and_suffix3_pair_joint','h2_pair_A_retained','h0_held'];labels=['b0整条h0-h6观察轨迹','b0当前与未来三步联合','成败当前h2完整配对','成败当前与未来三步联合','成败h2已有A入屉保持','h0持有身份'];rows=[];fixed=[]
 for sp in SPLITS:
  for seed in [0,1]:
   for name,label in zip(primary,labels):rows.append([SPLABEL[sp],seed,label,fmt('control',seed,sp,name),fmt('updated',seed,sp,name)])
   for name in ['b0_full_trajectory','b0_h2_future3_joint','h2_and_suffix3_pair_joint','h0_held']:fixed.append([SPLABEL[sp],seed,name,fmt('control_final1000',seed,sp,name),fmt('updated_final1000',seed,sp,name)])
 forgotten=[];events=list(csv.DictReader((U/'results/forgetting_events.csv').open()));eventrows=[]
 for g in gates['gates']:
  for sp,branches in g['retention_root_sets'].items():
   for branch,rr in branches.items():forgotten.append([SPLABEL[sp],g['seed'],branch,', '.join(rr['control'])or'无',', '.join(rr['updated'])or'无',', '.join(rr['new_forgetting_roots'])or'无'])
 for r in events:
  if r['variant']in ['control','updated']:eventrows.append([r['split'],r['variant'],r['seed'],r['root_id'],r['branch'],r['step'],r['color'],f"{float(r['prior']):.3f}",f"{float(r['posterior']):.3f}"])
 priorrows=[]
 for seed in [0,1]:
  for variant in ['control','updated']:
   d=load(variant,seed,'development');p=rel(d['prob']);pr=[];po=[]
   for rid in sorted(set(d['root_id'])):
    ix=lambda mode,b:np.where((d['root_id']==rid)&(d['branch']==b)&(d['mode']==mode)&(d['step']==2))[0][0]
    aa=COLORS.index(str(d['held_color'][ix('current',1)]));pr.append(bool(all(p[ix('one_step',b),3+aa]>=.5 for b in [1,2])));po.append(bool(all(p[ix('current',b),3+aa]>=.5 for b in [1,2])))
   priorrows.append({'seed':seed,'variant':variant,'roots':len(pr),'prior_A_InDrawer_correct':sum(pr),'posterior_A_InDrawer_both_correct':sum(po)})
 csvwrite(U/'results/h2_A_retention_before_after.csv',priorrows)
 finding=('两个种子均通过预登记训练联合门槛。'if gates['training_all_seeds']else'预登记训练联合门槛未在两个种子上全部通过。')+(' 开发的不退步与逐根保持条件也通过，但仍不等于原指南独立验收。'if gates['development_all_seeds']else' 开发的不退步与逐根保持条件未全部通过，不能宣称已获得可靠跨根后继。')
 headline=table(['指标（种子0；种子1）','残差配对对照','残差三分支均衡'],[[SPLABEL[sp]+' '+label,'；'.join(fmt('control',s,sp,name)for s in [0,1]),'；'.join(fmt('updated',s,sp,name)for s in [0,1])]for sp,name,label in [('train','b0_full_trajectory','b0整条观察轨迹'),('train','b0_h2_future3_joint','b0当前未来联合'),('train','h2_and_suffix3_pair_joint','成败当前未来联合'),('development','h2_and_suffix3_pair_joint','成败当前未来联合'),('development','h0_held','初始身份')]])
 lines=['# 同一残差模型的三分支均衡训练',finding,headline,
 '具体结果：两个种子的训练b0整轨迹及当前未来联合均从0/12到12/12，成败联合保持12/12，三分支的观测后入屉遗忘训练根均为零。开发没有同步改善：种子0初始身份3/6降到1/6；种子1成败当前未来联合2/6降到0/6，并新增B_recovery_development_02的入屉遗忘。原规则选模为种子0的500更新、种子1的1000更新，没有按这些结果换checkpoint。',
 '核心目标是让世界模型同时解释当前成败、保持其他对象关系，并预测未来状态供TS/LTL搜索使用。本轮只改变既有训练数据的分支组成，未增加结构、门控、正则、独立读出头或仿真。完整研究指南与规划闭环仍未完成。',
 '## 唯一因素与实际曝光',
 '原对照在新12根上训练b1成功+b2落空；处理组加入已有b0另一目标正常分支，三者各约1/3。每次新抽样仍使用同一根，旧40根回放样本与latent随机数逐次不变。每根的分支按出现序号循环均衡，各分支曝光最多差1。总样本槽与1000更新预算不变，b1和b2各自曝光减少约1/3，必须把这项取舍算入数据组成效应。',table(['种子','分支','对照episode抽样次数','均衡episode抽样次数'],exrows),
 '每fit仍是8000旧episode槽和8000新episode槽。唯一新episode数量从24到36，既有不同转移从144到216；这不是等独特样本数的纯标注变化。所有轨迹和特征均已存在，没有新增物理执行、C/D或上传。',
 table(['条件','匹配与核验'],[['模型与损失','残差源码字节一致，W及全部原有模块均训练；完整k3+终点损失、E/R裁剪与精度不变'],['初始化','同原joint6000共享权重/AdamW状态，同零W与无W历史动量；完整初始state hash等于旧残差receipt'],['预算与随机数','两种子各1000更新，4旧+4新微批、累积2；旧槽/新根/latent逐项相同'],['复用对照','旧残差四个开发候选目标重算完全一致'],['选择','同6开发根三分支，500/1000完整目标取最小、平局取早；另报固定1000'],['信息边界','GT只用于训练标签和评分；未来想象无未来图像，保护关系掩码不作模型输入']]),
 '## 为什么先检验覆盖取舍',
 '前一限时审计发现：旧40根已有b0的局部对象顺序，但没有新根完整前史；本轮配对续训又排除了新根b0。残差相对原配对模型新增b0完整错误边界10/9个，分别恢复1个，多在h2/h3；新增关系错误集中在A已入屉丢失与未操作C误判入屉。训练过的b1/b2无此类观测后遗忘。因此本轮检验数据覆盖是否足以恢复b0并保留已学成败，不把所有退化归为当次残差过大。审计报告在相邻observation_retention_audit_20261019目录。',
 '## 预登记按根门槛',
 '训练要求两个种子：h0身份12/12、b1/b2当前h2与未来h3-h5联合12/12保持；b0整条h0-h6观察轨迹正确根数及b0当前未来联合根数均严格改善。入屉保持按具体根、种子和分支要求新遗忘根集合包含于对照，不允许用别处恢复抵消新增遗忘。开发要求h0、成败当前未来联合、b0整轨迹不退步，并应用同样逐根保持条件。历史6评估根此前已看过，只作探索性回归。',
 table(['种子','训练门槛','训练分项','开发门槛','开发分项'],[[g['seed'],g['training_joint_gate'],json.dumps(g['training_components'],ensure_ascii=False),g['development_gate'],json.dumps(g['development_components'],ensure_ascii=False)]for g in gates['gates']]),
 table(['划分','种子','检查','残差配对','三分支均衡'],rows),
 '完整关系在本轮指3物体Held/InDrawer真假与抽屉类别同时正确，不等于所有原指南谓词。b0整轨迹要求七个真实观察边界全部正确；当前未来联合还检查从h2开始的三个纯想象后继。两者不能互相替代。分母按根，不按每个边界充当独立样本；整根bootstrap结果在metrics与contrasts中。',
 '## 已有入屉关系的逐根遗忘',table(['划分','种子','分支','对照遗忘根','均衡遗忘根','新增遗忘根'],forgotten),
 '遗忘事件严格指非动作对象真实InDrawer在前后均为真，且模型在同次观察前先验正确、观察后后验变错。先验本来错误的持续错误单独反映在完整状态指标中，不算当次观察遗忘。h0、其他对象及阈值没有用GT覆盖；阈值固定0.5。',table(['划分','模型','种子','根','分支','边界','对象','先验In概率','后验In概率'],eventrows),
 '## 先验错误与观察新遗忘分开',table(['种子','模型','开发根数','h2观察前A入屉正确','观察后两分支均保持'],[[r['seed'],r['variant'],r['roots'],r['prior_A_InDrawer_correct'],r['posterior_A_InDrawer_both_correct']]for r in priorrows]),
 '种子0的新模型在h2观察前只正确保留3/6个A入屉关系，观察后仍为3/6；对照为6/6→6/6。因此无新增观察遗忘并不代表已有关系正确，这部分退化已经在先验/更早历史中出现。种子1均衡模型在B_recovery_development_02_b1观察成功后，把非动作粉物体InDrawer从0.846降至0.381，属于新增当次观察遗忘；同时其他根也可能恢复，不能用净数抵消。',
 '## 固定1000更新',table(['划分','种子','检查','对照1000','均衡1000'],fixed),
 '这里固定更新数，不能根据历史评估或更好个别指标重新选模型。',
 '## 图与读法']
 for name,caption in figures:lines.extend([f'![{name}](figures/{name}.png)',caption])
 lines+=['## 结论边界与主线',
 '本轮在不改变结构或损失的情况下恢复了训练b0，同时保住训练成败当前未来联合与已有关系，说明同根分支覆盖取舍能够解释这部分训练内退化，暂不需要为这些已见根再叠加保护结构。效果包含独特样本覆盖增加与各分支曝光重新分配，不能进一步拆成单一内部机制。开发根仍存在初态身份、已有关系先验及观察新遗忘问题，且成败联合可能下降；这些跨根问题不能用训练全对掩盖。下一主线是跨根初态与关系先验的可靠泛化，而非继续在相同根上加结构或追训练分数。当前开发根不转训练，本轮不追加调参或新仿真；TS/LTL仍需可靠学习后继与原指南独立闭环验收。']
 md='\n\n'.join(lines);(U/'REPORT.md').write_text(md);out=[];ll=md.splitlines();i=0
 while i<len(ll):
  l=ll[i]
  if l.startswith('|'):
   rr=[]
   while i<len(ll)and ll[i].startswith('|'):rr.append(ll[i]);i+=1
   out.append('<table>')
   for j,row in enumerate(rr):
    if j==1:continue
    tag='th'if j==0 else'td';out.append('<tr>'+''.join(f'<{tag}>'+html.escape(x.strip())+f'</{tag}>'for x in row.strip('|').split('|'))+'</tr>')
   out.append('</table>');continue
  if l.startswith('!['):path=l.split('](')[1][:-1];out.append('<img src="data:image/png;base64,'+base64.b64encode((U/path).read_bytes()).decode()+'">')
  elif l.startswith('#'):n=len(l)-len(l.lstrip('#'));out.append(f'<h{n}>'+html.escape(l[n:].strip())+f'</h{n}>')
  elif l.strip():out.append('<p>'+html.escape(l)+'</p>')
  i+=1
 (U/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>三分支均衡训练</title><style>body{max-width:1300px;margin:40px auto;padding:0 24px;font-family:sans-serif;line-height:1.7}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd;padding:6px}th{background:#eef3f7}img{width:100%}p{overflow-wrap:anywhere}</style>'+''.join(out));save(U/'results/report_complete.json',{'utc':now(),'finding':finding,'figures':len(figures),'historical_evaluation_exploratory_only':True});print(finding,flush=True)

if __name__=='__main__':main()
