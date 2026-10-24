from study import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64,html
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'savefig.dpi':160})
PALETTE={'control':'#2878b5','updated':'#c8552d'};LABEL={'control':'原配对模型','updated':'观测写入h'};SPLITS=['train','development','evaluation'];SPLABEL={'train':'训练12根','development':'开发6根','evaluation':'历史评估6根（探索性）'}
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
 names=['h2_pair_joint','next_pair_joint','suffix3_pair_full','h2_and_suffix3_pair_joint','h0_held'];labs=['h2完整配对','下一步完整配对','三步未来完整配对','h2与三步联合','h0持有身份']
 fig,axes=plt.subplots(2,3,figsize=(16,8),sharey=True)
 for seed in [0,1]:
  for sp,ax in zip(SPLITS,axes[seed]):
   for j,v in enumerate(PALETTE):
    rr=[get(v,seed,sp,n)for n in names];bars=ax.bar(np.arange(5)+(j-.5)*.32,[r['mean']for r in rr],.3,color=PALETTE[v],label=LABEL[v]);
    for b,r in zip(bars,rr):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.015,f"{r['sum']:.0f}/{r['roots']}",ha='center',fontsize=8,rotation=45)
   ax.set_xticks(range(5),labs,rotation=25,ha='right');ax.set_ylim(0,1.2);ax.set_title(f'{SPLABEL[sp]}，种子{seed}');ax.set_ylabel('独立根正确比例');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
 fig.tight_layout();figsave(fig,'01_main_gates','每列为训练、开发、历史评估，每行为一个模型种子。蓝柱为复用且逐项核验的旧恢复对照，橙柱为将真实观测写入h。横轴前四项均要求成败两分支完整Held/InDrawer/抽屉关系正确；三步项统计h3-h5，联合项再要求h2也对。最后一项单列h0，避免后续分数遮掩初始身份问题。纵轴为独立根比例，柱上写分子/分母。历史评估已用于先前分析，只作探索性回归。')
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
 fig.tight_layout();figsave(fig,'02_observation_conditioning','每个点是一根的真实成功/落空配对：横轴为成功分支目标Held概率，纵轴为落空分支概率。蓝方/橙圆分别是对照/观测更新模型；右下浅绿区才是两种结果都辨对，灰虚线代表对两种观测给出相同判断，黑线是固定0.5阈值。点可能重叠，独立根数以面板标题为准。它检验观察条件区分，不能单凭该图定位网络内部哪个模块发生修复。')
 fig,axes=plt.subplots(2,3,figsize=(15,7),sharey=True)
 for seed in [0,1]:
  for sp,ax in zip(SPLITS,axes[seed]):
   for v in PALETTE:
    for branch,style in [(1,'-'),(2,'--')]:ax.plot([1,2,3],[get(v,seed,sp,f'suffix{k}_b{branch}_joint')['mean']for k in [1,2,3]],style,marker='o',color=PALETTE[v],label=LABEL[v]+('成功分支'if branch==1 else'落空分支'))
   ax.set_xticks([1,2,3]);ax.set_ylim(-.05,1.05);ax.set_xlabel('h2真实观测后纯想象步数');ax.set_ylabel('完整关系终点正确比例');ax.set_title(f'{SPLABEL[sp]}，种子{seed}');ax.grid(alpha=.2);ax.legend(fontsize=6)
 fig.tight_layout();figsave(fig,'03_learned_successors','从实际h2后验出发，仅调用学习到的imagine转移与共享符号读出，未来图像不再输入。横轴为1/2/3步，纵轴为每个终点完整关系正确比例；蓝/橙是对照/观测更新模型，实线为成功分支，虚线为落空分支。终点正确不等于整个前缀正确，后者单列在主图。')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for seed,ax in enumerate(axes):
  for v in PALETTE:
   rr=read((C if v=='control'else U)/f'results/train_{"paired"if v=="control"else"state_update"}_seed{seed}.json');ax.plot([r['step']for r in rr['curve']],[r['objective']for r in rr['curve']],c=PALETTE[v],label=LABEL[v]);ax.scatter([c['update']for c in rr['candidates']],[c['development_objective']for c in rr['candidates']],marker='x',s=65,c=PALETTE[v]);ax.axvline(rr['selected']['update'],ls=':',c=PALETTE[v],alpha=.5)
  ax.set_xlabel('同样1000次续训更新');ax.set_ylabel('完整目标值');ax.set_title(f'种子{seed}');ax.legend();ax.grid(alpha=.2)
 figsave(fig,'04_training','横轴为更新次数，纵轴为完全相同的损失公式。蓝/橙实线是两组各自抽样批次的训练目标；叉号是相同开发集的目标，虚线是按冻结规则选择的checkpoint。两组使用完全相同的成败配对数据与损失，模型仅新增观测到h的残差路径。目标降低仍不能替代当前状态和未来完整前缀的检查。')
 diag=read(U/'results/prechange_diagnostic.json');reuse=read(U/'results/control_reuse_audit.json');grad=read(U/'results/gradient_start_check.json');control=list(csv.DictReader((U/'results/same_observation_control.csv').open()))
 rows=[];normal=[];fixed=[];updates=[]
 names=['h0_held','h2_pair_held','h2_pair_joint','h2_pair_A_retained','next_pair_joint','suffix3_pair_full','h2_and_suffix3_pair_joint']
 labels=['h0持有身份','h2成败Held同时对','h2成败完整关系同时对','h2已有A入屉两分支保持','下一步两分支完整关系','三步未来两分支前缀完整','h2与未来三步联合完整']
 for sp in SPLITS:
  for seed in [0,1]:
   for name,label in zip(names,labels):rows.append([SPLABEL[sp],seed,label,fmt('control',seed,sp,name),fmt('updated',seed,sp,name)])
   for branch in [0,1]:
    rr=[get(v,seed,sp,f'b{branch}_current_joint')for v in ['control','updated']];normal.append([SPLABEL[sp],seed,branch,*[f"{r['sum']*7:.0f}/{r['roots']*7}"for r in rr]])
   for name in ['h0_held','h2_pair_joint','suffix3_pair_full','h2_and_suffix3_pair_joint']:fixed.append([SPLABEL[sp],seed,name,fmt('control_final1000',seed,sp,name),fmt('updated_final1000',seed,sp,name)])
 pp=list(csv.DictReader((U/'results/posterior_updates.csv').open()))
 for sp in SPLITS:
  for seed in [0,1]:
   for v in ['control','updated']:
    for branch in [1,2]:
     rr=[r for r in pp if r['split']==sp and r['seed']==str(seed)and r['variant']==v and r['branch']==str(branch)]
     updates.append([SPLABEL[sp],seed,v,branch,len(rr),*[sum(r[k]=='True'for r in rr)for k in ['prior_held_correct','posterior_held_correct','corrected_held','regressed_held']]])
 ctrlrows=[]
 for sp in SPLITS:
  for seed in [0,1]:
   for v in ['control','updated']:
    rr=[r for r in control if r['split']==sp and r['seed']==str(seed)and r['variant']==v and r['control']=='same_success_E'];ctrlrows.append([SPLABEL[sp],seed,v,len(rr),f"{np.mean([float(r['actual_E_pair_h_rms_difference'])for r in rr]):.5f}",f"{np.mean([float(r['actual_E_target_Held_gap'])for r in rr]):.4f}",sum(r['actual_h2_pair_joint']=='True'for r in rr),sum(r['same_E_h2_pair_joint']=='True'for r in rr)])
 # Descriptive first incorrect real-observation boundary; no new model or selection.
 first=[];cases=[]
 for sp in SPLITS:
  for seed in [0,1]:
   for v in ['control','updated']:
    d=load(v,seed,sp);p=rel(d['prob']);ok=correct(p,truth(d['cat']));bad=[]
    for rid in sorted(set(d['root_id'])):
     for branch in [1,2]:
      ii=np.where((d['root_id']==rid)&(d['branch']==branch)&(d['mode']=='current'))[0];ii=ii[np.argsort(d['step'][ii])];wrong=[int(d['step'][i])for i in ii if not ok[i,3]];bad.append(wrong[0]if wrong else-1);first.append({'variant':v,'seed':seed,'split':sp,'root_id':str(rid),'branch':branch,'first_wrong_complete_current_boundary':wrong[0]if wrong else-1})
    cases.append([SPLABEL[sp],seed,v,*[bad.count(t)for t in [-1,0,1,2]],sum(x>2 for x in bad)])
 csvwrite(U/'results/first_error_boundaries.csv',first)
 example_rows=[]
 for sp,rid in [('train','B_recovery_train_05'),('development','A_recovery_development_00')]:
  a,b=load('control',1,sp),load('updated',1,sp);pa,pb=rel(a['prob']),rel(b['prob']);yy=truth(a['cat']);oa,ob=correct(pa,yy),correct(pb,yy)
  for branch in [1,2]:
   for mode,t in [('one_step',2),('current',2),('suffix',5)]:
    i=np.where((a['root_id']==rid)&(a['branch']==branch)&(a['mode']==mode)&(a['step']==t))[0][0];target=COLORS.index(str(a['target_color'][i]));held=COLORS.index(str(a['held_color'][i]));example_rows.append([rid,branch,mode+' h'+str(t),str(a['target_color'][i]),bool(yy[i,target]),f'{pa[i,target]:.3f}',f'{pb[i,target]:.3f}',f'{pa[i,3+held]:.3f}',f'{pb[i,3+held]:.3f}',bool(oa[i,3]),bool(ob[i,3])])

 if gates['strengthened_training_joint_gate_all_seeds']:finding='两个种子都通过了训练内“h2与未来三步联合完整”的加强门槛。'
 else:finding='训练内“h2与未来三步联合完整”的加强门槛未在两个种子上全部通过。'
 if gates['strengthened_development_joint_gate_all_seeds']:finding+=' 开发支持门槛也通过，但仍没有新的盲测或TS/LTL闭环验收。'
 else:finding+=' 开发支持门槛未通过，因此不能把训练内改善视为可泛化、可规划的状态转移。'
 headline=table(['检查（种子0；种子1）','原配对模型','观测写入h'],[[SPLABEL[sp]+' '+label,'；'.join(fmt('control',s,sp,name)for s in [0,1]),'；'.join(fmt('updated',s,sp,name)for s in [0,1])]for sp,name,label in [('train','h2_pair_joint','h2完整配对'),('train','h2_and_suffix3_pair_joint','当前与未来联合'),('train','h2_pair_A_retained','已有入屉保持'),('development','h2_pair_joint','h2完整配对'),('development','h2_and_suffix3_pair_joint','当前与未来联合'),('development','h0_held','初始持有身份')]])
 lines=['# 真实观测写入持久状态：最小状态更新对照',finding,headline,
 '本轮的具体进展是：相同预算下，两个种子首次在12个训练根上同时正确解释成败当前状态与三步后果，并保留已有入屉关系；开发集也首次从零出现完整配对，h2为1/6、3/6，当前与未来联合为1/6、2/6。整体开发门槛失败的分项是种子1初态身份从2/6降至1/6。另一个必须保留的问题是训练根另一目标的正常b0分支完整边界由两个种子各36/84降至27/84、28/84，因此不能把配对能力改善表述为所有能力一起改善。',
 '研究主线仍是世界模型从真实执行结果更新状态，保留已成立关系，并向TS/LTL搜索提供学习到的可信后继。本轮只改变状态更新，不建立独立读出头，不覆盖输出答案，不使用人工动态效果，也不宣称完成原指南。',
 '## 改动、证据与可比性',
 '原observe只让实际观测更新z，h保持动作预测值。新增h_post = h_prior + W·LayerNorm(E_actual)，W为512×512、无偏置、零初始化；LayerNorm不带可训练参数。q/z依旧由原h_prior和E计算，initial()仍为h=0。更新后的h与原z一起进入现有共享读出、重构与结果头，并作为下一次GRU的持久状态。E包括RGB特征和机器人状态，不能把效果单独归因于视觉。',
 '新增262144参数，约占原11513563参数的2.28%。原参数及AdamW动量逐项继承，新增矩阵没有历史动量，并加入原R组的范数裁剪，因此共享R参数的裁剪尺度也可能随新增梯度变化；效应包含新增容量、观测路径及其训练影响，不能声称是完全等参数的因果分离。原有模块的训练范围、完整损失、数据、抽样与更新数一致，新模型另外训练W；没有冻结不同原有模块后复用旧对照。',
 table(['项目','实际设置/证据'],[['数据','同旧40根120episode/480转移与现有配对12根24episode/144转移；无新增样本'],['训练','种子0/1，均1000更新；每微批4旧+4新、累积2；同抽样和latent随机数'],['初始化','原joint6000及共享AdamW状态；零残差时真实序列全部状态和符号输出精确相同'],['选模','同开发6根全部三分支，完整目标评分500/1000取最小；平局取早'],['对照复用','原配对模型四个开发目标重算完全一致，历史权重/数据哈希检查通过'],['梯度启动','零初始化真实首批次残差梯度非零，且仅做反传验证、没有优化器更新'],['资源','单GPU串行；仅两次1000更新，无补跑、扩采、C/D或上传']]),
 '短检查只使用同根真实相同前史的成败观测，没有混拼h/z。真实E与q已经不同而h相同；当前符号损失与表示KL在编码E上的局部梯度多数相反。500至1000更新仍未完全区分训练成败且开发损失升高，故没有直接加训练步数。这里说明了可检验的结构假设，不能排除所有欠训练，也不意味着对不可预测扰动使用概率先验NLL本身错误。',
 table(['种子','旧配对更新','有效梯度比较','反向比较','平均余弦','实际配对Held概率差'],[[r['seed'],r['checkpoint'],r['nonzero_current_KL_grad_comparisons'],r['negative_cosine_count'],f"{r['cosine_mean']:.3f}",f"{r['pair_Held_gap_mean']:.3f}"]for r in diag['summaries']]),
 table(['种子','新增矩阵初始梯度范数','非零梯度元素','优化器步数'],[[r['seed'],f"{r['zero_initialized_weight_gradient_norm']:.6f}",r['nonzero_gradient_elements'],r['optimizer_steps']]for r in grad]),
 '## 门槛与当前/未来状态',
 '原门槛全部保留；评估前按最新要求追加h2与未来h3-h5的联合指标。训练加强门槛要求每个种子h0身份12/12且联合配对12/12。开发要求原h2配对、三步配对改善及h0/入屉不退步，再要求联合配对改善。没有按结果放宽，也没有改变选模规则。',
 table(['种子','原训练门槛','加强训练联合','原开发支持','开发联合改善'],[[r['seed'],r['training_necessary_gate'],r['training_joint_current_future_gate'],r['development_support_gate'],r['development_joint_current_future_improves']]for r in gates['gates']]),
 table(['划分','种子','指标','原配对模型','观测更新h'],rows),
 '完整关系指所有物体Held/InDrawer真假与抽屉状态同时正确。三步未来前缀仅指h3-h5；联合指标还要求h2，避免当前判断错误被后续恢复掩盖。已有A入屉单列为保持真关系检查，不代替完整关系。所有统计按独立根列分母；置信区间在metrics/contrasts中使用整根bootstrap，两个种子不会被当成额外独立根。',
 '## 真实观察纠正与相同观察控制',table(['划分','种子','模型','分支','根数','先验Held正确','后验正确','错→对','对→错'],updates),
 '分支1为真实成功，分支2为真实落空；故障前先验因相同历史与动作而必须完全一致。相同观察控制分别把真实成功E、真实失败E给两条相同前史，使用相同随机数，检查h、q、当前及未来输出都相同，并失去成败配对区分。控制从原模型的合法共同先验计算，没有任意交换h/z。',
 table(['划分','种子','模型','根数','不同真实E的h差RMS','不同E的Held差','实际E配对正确','相同成功E配对正确'],ctrlrows),
 '相同失败E控制同样全部丧失配对区分，完整逐根记录在same_observation_control.csv。此控制使用另一个预先固定随机数，因此实际E配对计数只是机制检查，不替代主评分。它说明区分依赖输入观察，不说明唯一内部因果模块或跨根泛化。',
 '## 正常分支退化与初始错误',table(['划分','种子','正常分支','原完整边界正确','新完整边界正确'],normal),
 '正常b0与b1分别统计h0-h6七个边界；分母是边界数，不是独立根数。b1也是同请求成功分支，b0保持另一目标的正常轨迹。',table(['划分','种子','模型','全程无错','首错h0','首错h1','首错h2','首错更晚'],cases),
 '首错只定位首次完整关系判断不正确，不能把其后的所有错误都归因为它；没有实施GT状态替换或另一轮诊断。h0是独立泛化问题，本轮没有给initial新增直接h更新。',
 '## 固定案例的连续概率',table(['根','分支','位置','目标','真实Held','原目标Held','新目标Held','原A入屉','新A入屉','原完整关系','新完整关系'],example_rows),
 '案例在新模型评估前固定为种子1的B_recovery_train_05与A_recovery_development_00；不按改善程度挑选。one_step h2为观察前，current h2为观察后，suffix h5为三步纯想象终点。A指最初放入抽屉的物体，目标指本次抓取对象。完整未来前缀是否正确仍以联合指标为准。训练案例中，新模型对蓝物体的共同先验0.489，在真实成功/落空观察后变为0.993/0.015。开发案例揭示另一种限制：落空时目标Held从先验0.731降到0.102，纠正了空手判断，但已有A入屉也从0.621降到0.449，跨过0.5阈值而被遗忘，所以完整关系仍错。开发保持In的总数未降，不代表每一根都没有这种替换性退化。',
 '## 固定1000更新与选择',table(['划分','种子','指标','原固定1000','新固定1000'],fixed),
 '固定1000表用于排除选模步数差异，历史评估6根此前已看过，只作探索性回归，绝不据此重选模型。',
 '## 图与读法']
 for n,c in figures:lines.extend([f'![{n}](figures/{n}.png)',c])
 lines+=['## 回到世界模型与规划主线',
 '本轮已在固定预算内把训练根的当前判断与未来后果同时学对，且相同观察控制失去成败区分；这支持真实观察进入了可供后续动力学使用的状态。保持同样1000更新比较时，旧训练联合正确仍为6/12、8/12，新模型为12/12、12/12，所以结果不是仅由选模步数不同造成。它仍不是对唯一内部原因的证明。开发完整配对有所改善但数量很少，初态身份不退步门槛失败，正常b0出现退化，因此当前模型仍不能直接作为可靠TS/LTL后继服务。下一主线应转为跨根泛化与正常技能保持的独立验证：先冻结场景/对象组合覆盖与验收，再决定新增A/B训练根，保持现有开发根不入训练、C/D封存。不要继续追加相近诊断、独立读出头或训练步数追分。本轮没有启动新增数据、另一结构变体、盲测或规划闭环。']
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
 (U/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>真实观测写入状态</title><style>body{max-width:1300px;margin:40px auto;padding:0 24px;font-family:"Noto Sans CJK SC",sans-serif;color:#203044;line-height:1.7}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd6df;padding:7px}th{background:#eaf0f5}img{width:100%}p{overflow-wrap:anywhere}</style>'+''.join(out)+'</html>');save(U/'results/report_complete.json',{'utc':now(),'finding':finding,'figures':len(figures),'historical_evaluation_exploratory_only':True});print(finding,flush=True)

if __name__=='__main__':main()
