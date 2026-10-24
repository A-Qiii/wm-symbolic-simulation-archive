from audit import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import html,base64
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'savefig.dpi':160})
COL={'original':'#777777','nominal':'#2878b5','recovery':'#c8552d'};NAMES={'original':'原模型','nominal':'常规续训','recovery':'恢复续训'}
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(x)for x in r)+' |'for r in rows)+'\n'
def main():
 guard();stats=list(csv.DictReader((D/'results/h0_summary.csv').open()));detail=list(csv.DictReader((D/'results/h0_detail.csv').open()));roots=read(D/'results/root_inventory.json');inputs=read(D/'results/input_audit.json');bound=list(csv.DictReader((D/'results/nominal_boundary_detail.csv').open()));chains=list(csv.DictReader((D/'results/nominal_episode_decomposition.csv').open()));figs=[]
 def get(scope,seed,arm):return next(r for r in stats if r['scope']==scope and r['seed']==str(seed)and r['arm']==arm and r['subset']=='all')
 def figsave(fig,name,caption):fig.savefig(D/f'figures/{name}.png',bbox_inches='tight');plt.close(fig);figs.append((name,caption))
 scopes=['old_train','new_train','new_development','new_evaluation'];fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
 for seed,ax in enumerate(axes):
  for j,arm in enumerate(NAMES):
   rr=[get(sc,seed,arm)for sc in scopes];vals=[int(r['held_correct'])/int(r['roots'])for r in rr];bars=ax.bar(np.arange(4)+(j-1)*.24,vals,.23,color=COL[arm],label=NAMES[arm]);
   for b,r in zip(bars,rr):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.02,r['held_correct']+'/'+r['roots'],ha='center',fontsize=9,rotation=45)
  ax.set_xticks(range(4),['旧训练40根','新训练12根','新开发6根','新评估6根']);ax.set_ylim(0,1.23);ax.set_ylabel('h0持有身份正确比例');ax.set_title(f'种子{seed}；每个物理根只计一次');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
 figsave(fig,'01_h0_generalization','横轴是数据划分，纵轴是h0持有身份正确比例；灰/蓝/橙是原模型、常规续训、恢复续训。柱上是独立根分子/分母。旧40根含32空手、8持物；新三组全部持物，因此应同时看报告中的旧8个持物子集，不能只比较总体百分比。旧40与新12训练根均能拟合，新开发/评估明显下降，说明初态泛化问题。')
 pairrows=[];pairdetails=[]
 for split in ['train','development','evaluation']:
  for seed in [0,1]:
   for arm in NAMES:
    path=(D if split=='train'else S)/f'results/predictions_{arm}_seed{seed}_{split}.npz'
    with np.load(path)as z:a={k:z[k]for k in z.files}
    ok=correct(rel(a['prob']),truth(a['cat']));p=rel(a['prob']);h2=(a['mode']=='current')&(a['step']==2);prior=(a['mode']=='one_step')&(a['step']==2);n=len(set(a['root_id']));both=0;differences=[];priorcorrect=0;postcorrect=0;corrections=0;suffix=0
    for rid in sorted(set(a['root_id'])):
     ix=[np.where(h2&(a['root_id']==rid)&(a['branch']==b))[0][0]for b in [1,2]];pr=np.where(prior&(a['root_id']==rid)&(a['branch']==2))[0][0];both+=bool(ok[ix,0].all());delta=float(np.abs(p[ix[0],:3]-p[ix[1],:3]).max());differences.append(delta);priorcorrect+=bool(ok[pr,0]);postcorrect+=bool(ok[ix[1],0]);corrections+=bool(not ok[pr,0]and ok[ix[1],0]);suf=np.where((a['mode']=='suffix')&(a['root_id']==rid)&(a['branch']==2))[0];suffix+=bool(ok[suf,3].all());pairdetails.append({'split':split,'seed':seed,'arm':arm,'root_id':str(rid),'b1_held_correct':bool(ok[ix[0],0]),'b2_held_correct':bool(ok[ix[1],0]),'paired_correct':bool(ok[ix,0].all()),'Held_probability_max_abs_difference':delta,'failure_prior_correct':bool(ok[pr,0]),'failure_post_correct':bool(ok[ix[1],0])})
    pairrows.append({'split':split,'seed':seed,'arm':arm,'roots':n,'b0_success_held':int(ok[h2&(a['branch']==0),0].sum()),'b1_success_held':int(ok[h2&(a['branch']==1),0].sum()),'b2_failed_held':int(ok[h2&(a['branch']==2),0].sum()),'b1_b2_both':both,'b2_suffix3_all':suffix,'b2_prior_correct':priorcorrect,'b2_post_correct':postcorrect,'b2_observation_corrections':corrections,'pair_Held_prob_diff_max':max(differences),'pair_Held_prob_diff_mean':float(np.mean(differences))})
 csvwrite(D/'results/pair_summary.csv',pairrows);csvwrite(D/'results/pair_detail.csv',pairdetails)
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
 for seed,ax in enumerate(axes):
  for j,arm in enumerate(NAMES):
   r=next(r for r in pairrows if r['split']=='train'and r['seed']==seed and r['arm']==arm);vals=[r[k]/12 for k in ['b0_success_held','b1_success_held','b2_failed_held','b1_b2_both','b2_suffix3_all']];bars=ax.bar(np.arange(5)+(j-1)*.24,vals,.23,color=COL[arm],label=NAMES[arm]);
   for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.02,f'{round(v*12)}/12',ha='center',fontsize=8,rotation=45)
  ax.set_xticks(range(5),['b0普通成功','b1同动作成功','b2实际落空','b1和b2都对','b2三步前缀全对'],rotation=15,ha='right');ax.set_ylim(0,1.23);ax.set_ylabel('12个训练根上的正确比例');ax.set_title(f'种子{seed}：见过根≠见过该根全部结果');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
 figsave(fig,'02_seen_root_pair','横轴前3项是h2持有身份，第4项要求同根b1成功与b2落空同时正确，第5项是落空后三步完整关系前缀。灰/蓝/橙是原/常规/恢复模型。常规组只训练b0+b1；恢复组只训练b0+b2，因此每组的b1/b2配对都有一个未训练分支。恢复组既能拟合b0成功又能拟合b2落空，但把未训练的同动作成功b1全部判错；仅有训练分支12/12或未来12/12不能证明观察条件区分。')
 fig,axes=plt.subplots(1,2,figsize=(12,4));markers={'old_train':'x','new_train':'o','new_development':'s','new_evaluation':'*'};cc={'red':'#d83c34','blue':'#2867bf','pink':'#be42b5'}
 for env,ax in zip(['A','B'],axes):
  for scope,mark in markers.items():
   rr=[r for r in roots if r['environment']==env and r['scope']==scope and r['held']!='empty']
   for r in rr:ax.scatter(r['held_x'],r['held_y'],marker=mark,c=cc[r['held']],s=110 if scope=='new_evaluation'else 60,edgecolors='black'if mark!='x'else None,linewidths=.4)
  for c in cc:ax.scatter([],[],c=cc[c],label='持有'+c,s=50)
  for sc,m in markers.items():ax.scatter([],[],c='gray',marker=m,label=sc,s=60)
  ax.set_title('场景'+env+'：实际h0被持物体位置');ax.set_xlabel('世界坐标x / 米');ax.set_ylabel('世界坐标y / 米');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
 figsave(fig,'03_initial_coverage','横纵轴为实际被持有物体的x/y坐标，颜色只表示真实物体颜色；叉号/圆/方/星分别是旧训练、新训练、新开发、新评估。每点是一个物理根；旧空手根不画。该图是已有物理状态的描述统计，不是坐标预测器。A场景新训练红色在左，蓝/粉在右；评估红/蓝互换侧。它证明有覆盖混杂，尚不能单凭此图认定模型使用了坐标捷径。')
 # Exact decomposition of ordinary branches: group by whether h0 was correct in each compared model.
 decomp=[]
 for seed in [0,1]:
  nr={r['episode_id']:r for r in chains if r['seed']==str(seed)and r['arm']=='nominal'};rr={r['episode_id']:r for r in chains if r['seed']==str(seed)and r['arm']=='recovery'}
  for n_ok,r_ok in itertools.product(['True','False'],repeat=2):
   ids=[eid for eid in nr if nr[eid]['h0_joint_correct']==n_ok and rr[eid]['h0_joint_correct']==r_ok]
   if ids:decomp.append({'seed':seed,'nominal_h0':n_ok,'recovery_h0':r_ok,'episodes':len(ids),'boundaries':7*len(ids),'nominal_correct':sum(int(nr[e]['correct_boundaries'])for e in ids),'recovery_correct':sum(int(rr[e]['correct_boundaries'])for e in ids),'roots':','.join(sorted({nr[e]['root_id']for e in ids}))})
 csvwrite(D/'results/regression_decomposition.csv',decomp)
 caseids=[('A_recovery_evaluation_00_b1',0),('B_recovery_evaluation_00_b1',1),('A_recovery_evaluation_02_b1',1)];casetable=[]
 for eid,seed in caseids:
  for step in [0,1,2,3,6]:
   rr=[next(r for r in bound if r['episode_id']==eid and r['seed']==str(seed)and r['arm']==arm and r['step']==str(step))for arm in ['nominal','recovery']];casetable.append([eid,seed,step,rr[0]['held_truth']+'; '+rr[0]['InDrawer_truth'],rr[0]['held_predicted']+'; '+rr[0]['InDrawer_predicted'],rr[1]['held_predicted']+'; '+rr[1]['InDrawer_predicted']])
 counts=collections.Counter((r['scope'],r['held'],r['InDrawer'],r['drawer'])for r in roots);mapping=collections.defaultdict(set)
 for r in inputs:mapping[next(x['environment']for x in roots if x['root_id']==r['root_id'])].add(str(r['colour_uid']))
 mainfinding='两道门槛必须分开：h0持有身份在训练根上已经拟合、跨根泛化失败；同一历史与动作之后的成功/落空区分，则连训练根上的未见配对分支也完全失败。旧40根h0保持全对，不能把问题概括为旧知识整体遗忘。'
 recommendation='下一轮唯一值得先改的因素是“完整同前缀结果对是否共同进入世界模型训练”：复用现有b1成功+b2落空，替换恢复组当前b0+b2的分支安排，保持根数、episode数、模型、损失、归一化和更新预算不变。这样同一个历史和请求动作对应两种实际结果，后验必须利用实际观测才能同时判对；应同时要求保留既有InDrawer和改善未来后果预测。先用现有训练/开发根验证这一训练内机制，不马上采新颜色位置数据，不加独立读出或坐标探针。本轮不执行该修正。'
 lines=['# 初态与同历史观测瓶颈审计',mainfinding,'## 范围与来源','只读旧40根和新24根，复用冻结DINO缓存与已保存开发/评估预测。补了312条h0冻结前向（新训练12根+旧40根，三模型×两种子；旧原模型重算用于共同随机数比较，原保存结果120/120也逐项分类核对），并用上轮完全相同评分器补齐新训练根6组完整预测。模型参数前后哈希相同。无训练、无采集、无C/D、无上传、无新读出或坐标探针。首次脚本括号语法错误日志保留；修正后才执行。',
 '## h0组合：已有组合的频率变化',table(['划分','真实持有','InDrawer集合','抽屉类别','独立根数'],[[*k,v]for k,v in sorted(counts.items())]),
 '抽屉0=关闭、2=打开。旧40根中已有8个“空屉＋单持有＋开屉”根；新24根全部使用这个已有组合，而非引入新的符号组合。三个持物颜色在新训练各4根、开发和评估各2根。新的空间布局与条件覆盖发生变化，不能称为仅加入失败轨迹、其他状态分布完全不变。',
 table(['划分','种子','原模型','常规续训','恢复续训'],[[sc,seed,*[get(sc,seed,a)['held_correct']+'/'+get(sc,seed,a)['roots']for a in NAMES]]for sc in scopes for seed in [0,1]]),
 '旧持物子集8/8在所有模型和两种子均正确；旧空手32/32也均正确。两续训组新训练12/12均正确，但新开发/评估明显下降，低训练loss不能替代这次实际前向证据。h0的InDrawer空集合在新24根、所有模型均正确；抽屉判定有两处例外：原模型种子1新训练11/12，恢复组种子1新开发5/6，其余均全对。新评估六根抽屉均正确，主要错误仍为持有身份。',
 '## 同根成功与落空：训练内也没有正确配对',table(['划分','种子','模型','b0成功Held','b1同动作成功Held','b2落空Held','b1/b2都对','b2先验已对','观察后错→对','b2三步前缀全对'],[[r['split'],r['seed'],r['arm'],*[str(r[k])+'/'+str(r['roots'])for k in ['b0_success_held','b1_success_held','b2_failed_held','b1_b2_both','b2_prior_correct','b2_observation_corrections','b2_suffix3_all']]]for r in pairrows]),
 '两组从未共同训练同根同动作的b1/b2两种结果。恢复组训练见过b0普通成功和b2落空，但二者请求的目标物体不同；因而可以按根、目标身份或动作记住结果，而不用真实h2观测纠正。恢复组训练内h0已12/12全对，故其b1全部错误不能归咎于h0。训练内三步前缀12/12只说明其训练过的失败后果可被拟合，不是可泛化观测条件转移证据。',
 '## 为什么是8/84和47/84',
 '84 = 6个评估根 × 2个常规分支 × 7个真实观测边界，不是84个独立根。种子0常规组12条episode全部h0读错，后续仅偶尔恢复，合计8/84。种子1常规组h0正确的6条episode贡献40/42，h0错误的6条仅7/42，合计47/84；37个错误中35个位于h0错误的那一半轨迹。这说明首错具有强烈传播关联，未做状态替换干预，不能据此断言每个后续错误的唯一原因。',
 table(['种子','常规h0','恢复h0','episode数','边界数','常规正确','恢复正确','根'],[[r[k]for k in ['seed','nominal_h0','recovery_h0','episodes','boundaries','nominal_correct','recovery_correct','roots']]for r in decomp]),
 '种子1从47降到33的14个净损失可拆为：h0从正确变错误的两根（4条episode）27→12，净−15；h0从错误变正确的一根7→11，净+4；两组h0都正确的A02根13→10，净−3。最后这一项说明即使初态正确，恢复经验仍可能损害真实成功后的判断；必须保留第二道门槛。旧40根未见h0遗忘，但新未见根发生了条件性泛化回退。',
 table(['episode','种子','边界','实际Held; InDrawer','常规Held; InDrawer','恢复Held; InDrawer'],casetable),
 '最小案例：A_eval00_b1的真实红块在h0被种子0常规组读成蓝块；第一次真实放入红块后模型持续把蓝块当作入屉，虽然之后粉块抓取判断正确，该分支仍0/7。B_eval00_b1种子1常规组7/7、恢复组0/7，恢复组从h0把红读成蓝开始。A_eval02_b1两组h0都正确，但恢复组在真实成功抓住蓝块的h2判为空手，6/7降至3/7，不能用初态问题解释掉。',
 '## 输入与绑定排查',table(['候选问题','实际证据','结论范围'],[
 ['同根复位/观测错位','24根×2比较=48对；四帧RGB、robot、标定、物理复位状态与h0标签全部逐项一致','未发现分支采集改变初态'],
 ['新旧相机条件','64根固定相机外参、两相机内参一致；RGB仍200×200与84×84，四帧窗口','腕部相机随机器人姿态变化属实际观测，不是配置改变'],
 ['目标可见性','新24根被持物体固定相机至少98像素、腕部至少279像素；三颜色均可登记','不能以完全不可见解释普遍错误；不保证DINO特征身份充分'],
 ['UID/颜色登记',str(dict(mapping))+'；64根登记与缓存逐项一致','A/B本来具有不同UID顺序，旧新不变；标签和读出均显式red/blue/pink顺序'],
 ['槽位路由','模型没有ROI槽位编号embedding，Transformer逐token后池化；visible_registry_ids不进入模型','未发现硬编码颜色索引错位；学习到的颜色/实例绑定能力仍可能不足'],
 ['归一化','逐根重算tokens与保存归一化精确一致；所有值有限；新robot最大abs(z)约3.09，旧约4.05；视觉abs(z)的99分位新≤2.78、旧≤2.76','没有更换归一化或明显数值爆炸证据'],
 ['4步变6步','两种dataset都固定7个观察张量；h0仅由当前观察初始化，无动作或未来状态输入；12次单帧/七帧检查h0符号概率完全相同','BF16编码有≤0.015625数值差，决定未变；排除直接长度输入错位，不能排除训练目标改变对权重的影响'],
 ['设置过程','旧新run_episode中reset、24步settle、setup_skills、history导出代码一致；持物setup均实际开屉后抓取','实验换了初态占比和空间布局，没有发现另改相机/夹爪初态生成器']]),
 '## 下一轮只改一个主线因素',recommendation,
 '两道验收门槛分别保留：(1)跨根h0持有身份泛化，包含颜色×位置×环境覆盖及普通成功分支不退步；(2)相同历史/动作下，看到不同真实结果后，Held与既有InDrawer同时判对，并使后续1/3步后果预测改善。当前6个评估根已看过，后续只可作为探索性回归，不能再次称为全新独立验收；若机制成立，再另行冻结新的验收设计。颜色位置偏相关是独立的覆盖风险，当前证据不支持跳过训练内配对门槛马上扩大采集。',
 '## 新24根逐根初态',table(['根','划分','真实Held','InDrawer','实际x','实际y','实际z'],[[r['root_id'],r['scope'],r['held'],r['InDrawer'],*[f'{r[k]:.4f}'for k in ['held_x','held_y','held_z']]]for r in roots if r['scope']!='old_train']),
 '## 可视化']
 for name,cap in figs:lines += [f'![{name}](figures/{name}.png)',cap]
 md='\n\n'.join(lines);(D/'REPORT.md').write_text(md);out=[];ll=md.splitlines();i=0
 while i<len(ll):
  line=ll[i]
  if line.startswith('|'):
   rows=[]
   while i<len(ll)and ll[i].startswith('|'):rows.append(ll[i]);i+=1
   out.append('<table>')
   for j,line in enumerate(rows):
    if j==1:continue
    tag='th'if j==0 else'td';out.append('<tr>'+''.join(f'<{tag}>'+html.escape(x.strip())+f'</{tag}>'for x in line.strip('|').split('|'))+'</tr>')
   out.append('</table>');continue
  if line.startswith('!['):path=line.split('](')[1][:-1];out.append('<img src="data:image/png;base64,'+base64.b64encode((D/path).read_bytes()).decode()+'">')
  elif line.startswith('#'):level=len(line)-len(line.lstrip('#'));out.append(f'<h{level}>'+html.escape(line[level:].strip())+f'</h{level}>')
  elif line.strip():out.append('<p>'+html.escape(line)+'</p>')
  i+=1
 (D/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>初态与观测纠错审计</title><style>body{max-width:1260px;margin:40px auto;padding:0 24px;font-family:"Noto Sans CJK SC",sans-serif;line-height:1.7;color:#203044}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd5dd;padding:7px}th{background:#eaf0f5}img{width:100%}p{overflow-wrap:anywhere}</style>'+''.join(out)+'</html>');save(D/'results/report.json',{'mainfinding':mainfinding,'next_single_factor':recommendation,'figures':len(figs),'no_new_trial_started':True,'two_gates_separated':True});print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
