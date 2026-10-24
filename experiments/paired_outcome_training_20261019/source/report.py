from study import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64,html
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'savefig.dpi':160})
PALETTE={'control':'#2878b5','paired':'#c8552d'};LABEL={'control':'旧恢复组 b0+b2','paired':'共同结果组 b1+b2'};SPLITS=['train','development','evaluation'];SPLABEL={'train':'训练12根','development':'开发6根','evaluation':'历史评估6根（探索性）'}
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
 names=['h2_pair_joint','next_pair_joint','suffix3_pair_full','h0_held'];labs=['h2两分支完整关系','下一步两分支全对','3步两分支前缀全对','h0持有身份']
 fig,axes=plt.subplots(2,3,figsize=(16,8),sharey=True)
 for seed in [0,1]:
  for sp,ax in zip(SPLITS,axes[seed]):
   for j,v in enumerate(PALETTE):
    rr=[get(v,seed,sp,n)for n in names];bars=ax.bar(np.arange(4)+(j-.5)*.32,[r['mean']for r in rr],.3,color=PALETTE[v],label=LABEL[v]);
    for b,r in zip(bars,rr):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.015,f"{r['sum']:.0f}/{r['roots']}",ha='center',fontsize=8,rotation=45)
   ax.set_xticks(range(4),labs,rotation=25,ha='right');ax.set_ylim(0,1.2);ax.set_title(f'{SPLABEL[sp]}，种子{seed}');ax.set_ylabel('独立根正确比例');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
 fig.tight_layout();figsave(fig,'01_main_gates','每列为训练、开发、历史评估，每行为一个模型种子。蓝柱为复用且逐项核验的旧恢复对照，橙柱为同请求两种结果共同训练。横轴前三项分别要求成功和落空两条分支同时满足完整Held/InDrawer/抽屉关系；三步项要求每步均对。最后一项单列h0，避免后续分数遮掩初始身份问题。纵轴为独立根比例，柱上写分子/分母。历史评估已用于先前分析，只作探索性回归。')
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
 fig.tight_layout();figsave(fig,'02_observation_conditioning','每个点是一根的真实成功/落空配对：横轴为成功分支目标Held概率，纵轴为落空分支概率。蓝方/橙圆分别是对照/共同结果组；右下浅绿区才是两种结果都辨对，灰虚线代表对两种观测给出相同判断，黑线是固定0.5阈值。点可能重叠，独立根数以面板标题为准。它检验观察条件区分，不能单凭该图定位网络内部哪个模块发生修复。')
 fig,axes=plt.subplots(2,3,figsize=(15,7),sharey=True)
 for seed in [0,1]:
  for sp,ax in zip(SPLITS,axes[seed]):
   for v in PALETTE:
    for branch,style in [(1,'-'),(2,'--')]:ax.plot([1,2,3],[get(v,seed,sp,f'suffix{k}_b{branch}_joint')['mean']for k in [1,2,3]],style,marker='o',color=PALETTE[v],label=LABEL[v]+('成功分支'if branch==1 else'落空分支'))
   ax.set_xticks([1,2,3]);ax.set_ylim(-.05,1.05);ax.set_xlabel('h2真实观测后纯想象步数');ax.set_ylabel('完整关系终点正确比例');ax.set_title(f'{SPLABEL[sp]}，种子{seed}');ax.grid(alpha=.2);ax.legend(fontsize=6)
 fig.tight_layout();figsave(fig,'03_learned_successors','从实际h2后验出发，仅调用学习到的imagine转移与共享符号读出，未来图像不再输入。横轴为1/2/3步，纵轴为每个终点完整关系正确比例；蓝/橙是对照/共同结果组，实线为成功分支，虚线为落空分支。终点正确不等于整个前缀正确，后者单列在主图。')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for seed,ax in enumerate(axes):
  for v in PALETTE:
   rr=read((R if v=='control'else U)/f'results/train_{"recovery"if v=="control"else"paired"}_seed{seed}.json');ax.plot([r['step']for r in rr['curve']],[r['objective']for r in rr['curve']],c=PALETTE[v],label=LABEL[v]);ax.scatter([c['update']for c in rr['candidates']],[c['development_objective']for c in rr['candidates']],marker='x',s=65,c=PALETTE[v]);ax.axvline(rr['selected']['update'],ls=':',c=PALETTE[v],alpha=.5)
  ax.set_xlabel('同样1000次续训更新');ax.set_ylabel('完整目标值');ax.set_title(f'种子{seed}');ax.legend();ax.grid(alpha=.2)
 figsave(fig,'04_training','横轴为更新次数，纵轴为完全相同的损失公式。蓝/橙实线是两组各自抽样批次的训练目标；叉号是相同开发集的目标，虚线是按冻结规则选择的checkpoint。新训练包含同一隐藏故障前缀的两种结果，先验存在不可预知部分，故训练loss不要求降到旧对照相同水平；开发目标与具体状态/后果指标需分开解读。')
 # One concrete root per split, selected only for explanation after evaluation.
 examples=[];matrix=[]
 for sp in SPLITS:
  a=load('control',1,sp);b=load('paired',1,sp);assert np.array_equal(a['episode_id'],b['episode_id']);pa,pb=rel(a['prob']),rel(b['prob']);ca,cb=correct(pa,truth(a['cat'])),correct(pb,truth(b['cat']));rank=[]
  for rid in sorted(set(a['root_id'])):
   ix=np.where((a['root_id']==rid)&(a['mode']=='current')&(a['step']==2)&np.isin(a['branch'],[1,2]))[0];rank.append((int(cb[ix,3].all())-int(ca[ix,3].all()),str(rid)))
  rid=sorted(rank,reverse=True)[0][1];examples.append({'split':sp,'root_id':rid,'selection':'largest h2 complete pair improvement, deterministic root-id tie; report-only'})
  for branch in [1,2]:
   for mode,t in [('one_step',2),('current',2),('one_step',3),('suffix',5)]:
    i=np.where((a['root_id']==rid)&(a['branch']==branch)&(a['mode']==mode)&(a['step']==t))[0][0];target=COLORS.index(str(a['target_color'][i]));held=COLORS.index(str(a['held_color'][i]));matrix.append([sp,rid,branch,mode+' h'+str(t),f'{pa[i,target]:.3f}',f'{pb[i,target]:.3f}',f'{pa[i,3+held]:.3f}',f'{pb[i,3+held]:.3f}',bool(ca[i,3]),bool(cb[i,3])])
 # Summary must reflect the frozen gates, not select favorable metrics.
 if gates['training_gate_all_seeds']:finding='同请求两种结果共同训练通过了预先冻结的训练内必要门槛。'
 else:finding='同请求两种结果共同训练未通过全部训练内必要门槛，不能据局部分数宣称已学会可靠观测纠正。'
 if gates['development_support_all_seeds']:finding+=' 两个种子也同时通过开发集的配对状态、纯开环后果与不退步检查；这支持有限A/B范围内的机制改善，但仍需新独立验收。'
 else:finding+=' 开发集的泛化支持门槛未在两个种子上全部通过；训练内进步与未见根上的可靠后果预测应分开。'
 hnames=['h0_held','h2_pair_held','h2_pair_joint','h2_pair_A_retained','next_pair_joint','suffix3_pair_full'];hlabs=['h0身份','h2成败Held同时对','h2完整关系同时对','h2已有入屉同时保持','下一步两分支全对','三步两分支前缀全对'];rows=[]
 for sp in SPLITS:
  for seed in [0,1]:
   for m,label in zip(hnames,hlabs):rows.append([SPLABEL[sp],seed,label,fmt('control',seed,sp,m),fmt('paired',seed,sp,m)])
 finalrows=[]
 for sp in SPLITS:
  for seed in [0,1]:finalrows.append([SPLABEL[sp],seed,*[fmt(v,seed,sp,m)for m in ['h0_held','h2_pair_joint','suffix3_pair_full']for v in ['control_final1000','paired_final1000']]])
 pp=list(csv.DictReader((U/'results/posterior_updates.csv').open()));up=[]
 for sp in SPLITS:
  for seed in [0,1]:
   for v in PALETTE:
    for branch in [1,2]:
     rr=[r for r in pp if r['split']==sp and r['seed']==str(seed)and r['variant']==v and r['branch']==str(branch)];up.append([SPLABEL[sp],seed,v,branch,len(rr),sum(r['prior_held_correct']=='True'for r in rr),sum(r['posterior_held_correct']=='True'for r in rr),sum(r['corrected_held']=='True'for r in rr),sum(r['regressed_held']=='True'for r in rr)])
 lines=['# 同请求成败共同训练：世界模型受控迭代',finding,
 '按相同开发目标，两个种子均选500更新。训练根的h2成败完整配对由0/12、0/12升至4/12、8/12，已有A入屉保持仍为12/12、12/12；对应三步未来前缀由0/12、0/12升至7/12、10/12。开发根的h2配对与三步前缀仍均为0/6，h0持有身份也仍为2/6、2/6。因此本轮支持训练内的局部改善，不支持可靠的跨根观测更新或可规划转移。',
 '核心研究目标是让世界模型在收到真实执行结果后更新符号信念，保留已成立的关系，并提供可信的学习后继供TS/LTL规划使用。当前试验只检查这条链条的观测更新与未来预测环节；识别分数提高、训练样本记住、或最后一个终点碰巧正确都不足以完成原指南验收。',
 '## 唯一操作与真实效应范围',
 '旧恢复组训练每根b0普通成功+b2落空；本轮改为b1同请求成功+b2落空。b0与b1在第2、4动作的目标颜色不同，技能族顺序均为放置—抓取—放置—抓取—放置—关屉。替换既补齐相同历史/请求的两种结果，也改变每根成功样本的目标物体和后续实际轨迹，不能把结果归因为一个纯粹的配对标签效应。全局24个新episode仍各有8个红/蓝/粉目标；固定抽样计划的加权颜色曝光会随目标替换略变，明细见sampling_scope_note.json。',
 table(['条件','两组一致的实际设置','核验'],[['起点','原joint6000权重及AdamW状态，种子0/1','initial_hash与旧receipt一致'],['数据量','旧40根120episode/480转移；新12根24episode/144转移','每组同量、没有新增采集'],['抽样','每微批4旧+4新，积累2；每fit各8000旧/新episode抽样','old/new-slot/latent计划字节相同；同根同槽，只把槽0从b0映射到b1'],['训练','1000更新；AdamW lr1e-4、同clip、同BF16与确定性设置','直接抽取并执行旧训练循环AST'],['损失','原完整k3损失+相同真实终点后验监督','原losses.py未变'],['选择','同开发6根三分支；500/1000候选，目标最小，平局取早','旧4个候选开发目标重算一致，严格复用旧恢复对照'],['推理','同8样本、同根随机数、连续真实观察；h2后纯想象','同旧评分器；故障前配对概率精确一致'],['信息边界','RGB缓存/几何/可见性/robot与静态动作请求','隐藏扰动标志、成败、分支ID、GT和控制轨迹不作输入']]),
 '## 预先门槛与结果',table(['种子','训练内必要门槛','开发支持门槛','开发分项'],[[r['seed'],r['training_necessary_gate'],r['development_support_gate'],json.dumps(r['development_components'],ensure_ascii=False)]for r in gates['gates']]),
 '训练内必要门槛：每个种子h0身份12/12，且h2同根成败两分支完整关系12/12。开发支持要求两个种子均改善h2完整配对和三步完整配对，且h0身份、已有入屉保持不退步。阈值与选择规则在训练前冻结，未按结果放宽。旧6评估根此前已看过，全部表图均标为探索性，不用于选模或门槛。',table(['划分','种子','检查','旧恢复对照','共同结果训练'],rows),
 '## 观察前后与后果链',table(['划分','种子','模型','分支','根数','先验Held对','后验Held对','错→对','对→错'],up),
 '同一历史和请求在故障前的先验必须一致；隐藏扰动不可被提前确定预测。正确利用观察的证据是后验能同时区分成功与落空，而不是只让落空分支先验更偏向空手。完整关系还要求所有物体InDrawer真假与抽屉状态正确。下一步和三步检查从h2后验出发；三步前缀指h3、h4、h5每步都对，不包含h2起点；它没有读未来图像，也没有人工动态效果。固定1000更新时训练未来前缀可达12/12，但h2配对仍只有6/12、8/12，因此未来分数不能替代当前状态门槛，也不能据此断言内部表示已完整正确。h3动作成功标签包含大量全负支持，成功头准确率不作为主要机制证据。',
 '## 匹配最终1000更新',table(['划分','种子','h0对照','h0共同','h2配对对照','h2配对共同','三步配对对照','三步配对共同'],finalrows),
 '本表固定1000更新，防止开发选择500/1000的差异遮掩结果；不得据历史评估再选checkpoint。',
 '## 具体根的连续证据',table(['划分','根','分支','读取位置','对照目标Held','共同目标Held','对照A入屉','共同A入屉','对照完整关系','共同完整关系'],matrix),
 '这些案例为种子1，各划分按h2完整配对改善最大选择一根，仅用于解释，不影响训练/选模/门槛。逐根两种子完整概率、纠正与回退记录在results中。因模型共同续训，内部编码/后验/读出改变不能进一步被分解为某一层的独立因果效果。',
 '## 图与读法']
 for n,c in figures:lines += [f'![{n}](figures/{n}.png)',c]
 lines += ['## 回到主线与下一步边界',
 '本轮检验的是：在相同预算内，共同训练同一请求的两种真实结果是否足以让当前世界模型可靠使用观察。答案仍是否；训练根有局部纠正，但完整必要门槛未通过，开发根也没有获得成败配对区分。历史评估只在种子1出现1/6配对正确，两个种子的三步成败前缀仍为0/6，不能作为泛化验收。核心目标仍是让世界模型提供可信学习后继给TS/LTL搜索；目前应停在这一前置能力，不能用人工效果或独立读出头的分数代替。下一步建议在已有固定训练配对上，限定一次观察通路诊断，查看观察与先验冲突时第一个影响符号状态的环节；h0跨根身份失败继续单列，不与h2纠正混算。诊断应直接针对原世界模型，而非新建独立头、坐标探针或继续扩采同类轨迹。该建议未执行，本轮已按两次1000更新预算结束。',
 '本轮文件与历史输入哈希、控制复用凭据、原训练循环AST、逐根抽样次数和完整日志均保存。原失败恢复实验与初态审计只读保留；无C/D、上传、驱动或安全配置改动。']
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
 (U/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>同请求成败共同训练</title><style>body{max-width:1300px;margin:40px auto;padding:0 24px;font-family:"Noto Sans CJK SC",sans-serif;color:#203044;line-height:1.7}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd6df;padding:7px}th{background:#eaf0f5}img{width:100%}p{overflow-wrap:anywhere}</style>'+''.join(out)+'</html>');save(U/'results/report_complete.json',{'utc':now(),'finding':finding,'figures':len(figures),'case_roots':examples,'historical_evaluation_exploratory_only':True});print(finding,flush=True)
if __name__=='__main__':main()
