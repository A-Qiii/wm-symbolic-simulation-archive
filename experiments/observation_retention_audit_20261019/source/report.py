from audit import *
os.environ['MPLCONFIGDIR']=str(U/'cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64,html
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'savefig.dpi':150})
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |'for r in rows)
def loadcsv(name):return list(csv.DictReader((U/f'results/{name}.csv').open()))
def main():
 d=read(U/'results/audit.json');b=loadcsv('boundary_comparisons');o=loadcsv('object_relation_comparisons');x=loadcsv('nonaction_observation_changes');(U/'figures').mkdir(exist_ok=True);figs=[]
 fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
 for seed,ax in enumerate(axes):
  rr=[r for r in b if r['split']=='train'and r['seed']==str(seed)and r['branch']=='0'];lost=[sum(r['step']==str(t)and r['new_complete_regression']=='True'for r in rr)for t in range(7)];gained=[sum(r['step']==str(t)and r['new_complete_recovery']=='True'for r in rr)for t in range(7)]
  ax.bar(np.arange(7)-.17,lost,.32,color='#c8552d',label='原对→新错');ax.bar(np.arange(7)+.17,gained,.32,color='#2878b5',label='原错→新对');ax.set_xticks(range(7),['h0初态','h1放A','h2抓B','h3放B','h4再抓B','h5再放B','h6关屉'],rotation=25,ha='right');ax.set_ylabel('完整关系边界数');ax.set_title(f'训练根b0，种子{seed}');ax.legend();ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(U/'figures/01_b0_regression.png',bbox_inches='tight');plt.close(fig);figs.append(('01_b0_regression','横轴是实际六动作链的观测边界；A是最初持有并入屉的物体，B是b0请求抓取物体。纵轴是12根中发生该类完整关系变化的边界数。橙色表示原配对模型对而残差模型错，蓝色相反；各边界不是独立样本。两图分别为种子0/1。'))
 roles=[('A_initially_held','InDrawer'),('B_requested','Held'),('B_requested','InDrawer'),('C_other','InDrawer')];labels=['A的InDrawer','B的Held','B的InDrawer','第三物体C的InDrawer'];fig,axes=plt.subplots(1,2,figsize=(11,4))
 for seed,ax in enumerate(axes):
  values=np.array([[sum(r['split']=='train'and r['seed']==str(seed)and r['branch']=='0'and r['role']==role and r['relation']==rel and r['step']==str(t)and r['regression']=='True'for r in o)for t in range(7)]for role,rel in roles]);im=ax.imshow(values,cmap='Oranges',vmin=0,vmax=5,aspect='auto')
  for i in range(4):
   for j in range(7):ax.text(j,i,str(values[i,j]),ha='center',va='center')
  ax.set_yticks(range(4),labels);ax.set_xticks(range(7),[f'h{t}'for t in range(7)]);ax.set_title(f'b0新增关系错误，种子{seed}');fig.colorbar(im,ax=ax,label='原对→新错的关系数')
 fig.tight_layout();fig.savefig(U/'figures/02_object_relations.png',bbox_inches='tight');plt.close(fig);figs.append(('02_object_relations','每格统计原模型该关系正确、残差模型错误的次数，深橙色表示更多错误，格中文字给出精确数。行区分已放入的A、当前动作目标B、b0从未操作的C；列为实际边界。一个边界可同时错多个关系，不能把格子相加当作新增失败根数。'))
 coverage={}
 for r in d['coverage']:coverage[r['environment'],r['initial_A'],r['target_B']]=r
 covrows=[]
 for a in COLORS:
  for bb in COLORS:
   if a==bb:continue
   vals=[coverage[e,a,bb]for e in ['A','B']];covrows.append([a,bb,'b0'if vals[0]['branch']==0 else'b1',*[v['old_same_env_context_transitions']for v in vals],*[v['old_same_env_placeA_liftB_transitions']for v in vals]])
 errors=[]
 for seed in [0,1]:
  for role,relation in roles:
   rr=[r for r in o if r['split']=='train'and r['seed']==str(seed)and r['branch']=='0'and r['role']==role and r['relation']==relation];errors.append([seed,role,relation,sum(r['regression']=='True'for r in rr),sum(r['recovery']=='True'for r in rr)])
 examples=[r for r in x if r['variant']=='updated'and r['relation']=='InDrawer'and r['truth']=='True'and r['observation_regression']=='True']
 # Existing b0 pure-future predictions: no new forward pass.
 future=[]
 for ref in read(O/'results/prediction_references.json'):
  if ref['variant']not in ['control','updated']or ref['split']not in ['train','development']:continue
  with np.load(ref['path'])as z:data={k:z[k]for k in z.files}
  ok=right(rel(data['prob']),truth(data['cat'])).all(-1);rr=[]
  for rid in sorted(set(data['root_id'])):
   indices=[np.where((data['root_id']==rid)&(data['branch']==0)&(data['mode']==mode)&(data['step']==t))[0][0]for mode,t in [('current',2),('suffix',3),('suffix',4),('suffix',5)]];rr.append(bool(ok[indices].all()))
  future.append({'variant':ref['variant'],'seed':ref['seed'],'split':ref['split'],'b0_h2_and_future3_joint':sum(rr),'roots':len(rr)})
 csvwrite(U/'results/b0_current_future_joint.csv',future)
 proposal='''# 唯一执行方案：现有三分支均衡混合（最新授权）

问题：当前残差在训练过的b1/b2上不遗忘，b0和开发仍会改错其他对象。旧数据已经有b0的局部对象关系，因此不能仅归结为“从未见过该技能/颜色组合”；要检验的是这些新根上的正例覆盖取舍。

只改变新数据分支组成：对照b0=0、b1=1/2、b2=1/2；按最新授权，处理组b0/b1/b2各约1/3。b1和b2实际曝光均降低，不将结果解释为只增加b0而其他曝光完全不变。

执行映射：沿用现残差实验每个seed的旧/新slot/latent抽样数组；所有旧样本、每次新样本的根身份和latent随机数不变。按每根新样本出现序号循环映射b0/b1/b2，各分支计数最多相差1。总微批、旧/新比例和1000更新不变。现有b0真实轨迹与特征已缓存，不采集新根，不读开发数据训练。

模型保持本轮残差结构和完整损失；从同原joint6000共享权重及AdamW状态、同零W开始。两种子。原残差控制仅在协议/代码/候选开发评分核验一致后复用，否则停止报告匹配问题，不默默换对照。沿用500/1000完整开发目标选模，并报告固定1000。

这是一个训练数据组成因素，唯一episode从24增至36（144→216个既有不同转移），但抽样和更新量不变；b1和b2曝光均从约4000降到约2667次/种子；实际逐根计数在执行时冻结。不能把结果解释为等独特样本数的纯标签效应。

冻结主检查：两种子b1/b2的h2+三步联合仍12/12、初态身份仍12/12；b0的当前完整边界和h2+三步联合均改善；真实不变的非动作对象InDrawer在观察前正确→观察后错误不增加。开发按同根分别报b0、成败联合、h0及关系保持，不能靠总分抵消退化。历史6评估继续仅探索，不用于选模。本轮建议不声称这些内部门槛等于指南验收。

能排除什么：若补入同根另一目标的正例后b0及未来恢复、b1/b2联合与保持仍成立，就削弱“必须立刻加关系保护结构”的解释，支持覆盖取舍。如果b0已受训仍出现同样非动作关系遗忘，或改善仅靠牺牲成败联合，则不能用覆盖解释全部问题，需要另行决定是否检验状态修正的关系保持约束；不同时启动第二方案。

预计成本：核验与冻结2–3分钟，两种子训练约10分钟，统一推理/汇总/封存5–7分钟，总计约20分钟。新增存储约200MB，复用全部轨迹/特征。本审计没有执行训练；最新授权的唯一均衡混合对照另建独立目录实施。
'''
 proposal=proposal.replace('对照b0/b1/b2抽样比例为0/1/2/1/2；处理组为1/4、1/4、1/2。这里第一项应读作对照b0=0、b1=1/2、b2=1/2。','对照b0=0、b1=1/2、b2=1/2；处理组b0=1/4、b1=1/4、b2=1/2。')
 (U/'NEXT_CONTROL.md').write_text(proposal)
 lines=['# b0退化与非动作关系遗忘：现有数据限时审计',
 '结论：两种问题同时存在。旧数据覆盖b0的局部对象序列，但本轮排除了新根上的b0正例；残差模型相对原配对模型又增加了非动作对象关系错误。训练过的b1/b2没有观测后破坏已有入屉关系，故优先检验分支覆盖取舍，不立刻叠加新结构。最新授权采用三分支等比例；本审计只给依据，训练另建目录实施。',
 '审计只读取训练与开发的保存预测及旧40根元数据；无模型前向、反传、训练、采集、C/D或上传。原配对模型与残差模型接收相同公开RGB/robot记录，但编码器、历史状态及参数不同，故这里是整个训练改动的比较，不能单独归因于W。另统计各模型自己的同次观察前后变化，直接定位真实观察更新改错的关系。',
 '## 旧训练究竟覆盖了什么',
 'b0与b1的技能族顺序相同，均为放A—抓B—放B—再抓B—放B—关屉；第2与第4个命名抓取动作的目标颜色不同。旧数据有40根、120个四动作episode；全部新六动作完整前史均未出现，所有新根对应的初始place(A)→lift(B)前缀也未出现。旧数据确实出现过同环境、A已入屉、空手、B仍在桌面的抓取，以及紧接真实place(A)后的lift(B)。',
 table(['A','下一抓取B','新分支','旧情境转移A环境','旧情境转移B环境','紧邻placeA→liftB，A环境','同左，B环境'],covrows),
 '三个b0对象顺序red→blue、blue→pink、pink→red在旧数据中均有支持；b1是反向的另三种配对，局部情境旧支持为零。不能说b0因为“完全没见过对象序列”退化。更准确的差别是同一批新根只训练了b1成功+b2落空，b0所对应的另一目标、视觉位置与历史没有被共同续训覆盖。',
 '## 额外退化数量与时刻',table(['划分','种子','分支','边界总数','原对','新对','原对→新错','原错→新对'],[[r[k]for k in ['split','seed','branch','boundaries','control_correct','updated_correct','extra_regressions','recoveries']]for r in d['summary']]),
 '训练b0种子0新增10个错误边界、恢复1个，净少9；种子1新增9个、恢复1个，净少8。种子0新增完整失败在h2有6个、h3有4个；种子1在h2有3个、h3有5个、h5有1个。边界可来自同一根，不能视为84个独立样本。',table(['种子','对象角色','关系','原对→新错','原错→新对'],errors),
 'A是已放入的物体；B是b0目标；C是从未在b0操作的第三物体，恰好是b1/b2的抓取目标。A的新增InDrawer错误为真→假，C为假→真。多个关系可能在同一边界错误，不能直接相加当作额外完整失败。',table(['划分','种子','模型','b0当前+未来三步联合','独立根'],[[r['split'],r['seed'],r['variant'],r['b0_h2_and_future3_joint'],r['roots']]for r in future]),
 '## 观察是否改坏了非动作对象',
 '保护机会仅用于评分：该对象不是本次请求目标，且真实关系在动作前后不变。对已有入屉关系还要求两端真值都为真。抓取目标由命名动作确定；放置按本链请求的A/B确定；关屉没有动作目标物体。没有把这些标签或掩码输入模型。',
 table(['划分','种子','分支','模型','已有In机会','观察使对→错','观察使错→对','后验正确'],[[r[k]for k in ['split','seed','branch','variant','positive_In_opportunities','positive_In_regressions','positive_In_recoveries','positive_In_posterior_correct']]for r in d['collateral']]),
 '训练b0已有非动作InDrawer在一次观察中由对变错，原配对模型为1次/0次，残差模型为3次/3次；其训练b1/b2均为0次。开发残差仍有这种遗忘：不是单纯遗漏b0就能解释所有跨根错误。总后验正确数与单次更新破坏数不同，既有错误先验也会持续错误。',
 table(['种子','划分','根','分支','边界','受影响对象','观察前In概率','观察后In概率'],[[r['seed'],r['split'],r['root_id'],r['branch'],r['step'],r['protected_object'],f"{float(r['prior']):.3f}",f"{float(r['posterior']):.3f}"]for r in examples]),
 '例如种子1的A_recovery_development_00_b2，在抓粉物体的落空观测后，红物体InDrawer从0.621降到0.449；真实红物体始终在抽屉中。这是可观察的非动作关系破坏，尚不能单独归因于残差矩阵而排除共享训练影响。',
 '## 图与读法']
 for name,caption in figs:lines.extend([f'![{name}](figures/{name}.png)',caption])
 lines.extend([proposal]);md='\n\n'.join(lines);(U/'REPORT.md').write_text(md)
 out=[];ll=md.splitlines();i=0
 while i<len(ll):
  line=ll[i]
  if line.startswith('|'):
   rr=[]
   while i<len(ll)and ll[i].startswith('|'):rr.append(ll[i]);i+=1
   out.append('<table>')
   for j,row in enumerate(rr):
    if j==1:continue
    tag='th'if j==0 else'td';out.append('<tr>'+''.join(f'<{tag}>'+html.escape(v.strip())+f'</{tag}>'for v in row.strip('|').split('|'))+'</tr>')
   out.append('</table>');continue
  if line.startswith('!['):path=line.split('](')[1][:-1];out.append('<img src="data:image/png;base64,'+base64.b64encode((U/path).read_bytes()).decode()+'">')
  elif line.startswith('#'):n=len(line)-len(line.lstrip('#'));out.append(f'<h{n}>'+html.escape(line[n:].strip())+f'</h{n}>')
  elif line:out.append('<p>'+html.escape(line)+'</p>')
  i+=1
 (U/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>b0与关系保持审计</title><style>body{max-width:1200px;margin:35px auto;font-family:sans-serif;line-height:1.7}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border:1px solid #ccd;padding:6px}th{background:#eef3f7}img{width:100%}</style>'+''.join(out));print('REPORT_COMPLETE',json.dumps(future),flush=True)

if __name__=='__main__':main()
