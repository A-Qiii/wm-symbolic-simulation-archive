from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64,html
from analyze import load,NAMES
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'savefig.dpi':160})
PAL={'original':'#777777','nominal':'#2878b5','recovery':'#c8552d'}
LABEL={'original':'原模型参考','nominal':'常规新经验','recovery':'失败恢复新经验'}
FIG=[]
def figure(fig,name,caption):
 fig.savefig(S/f'figures/{name}.png',bbox_inches='tight');plt.close(fig);FIG.append((name,caption))
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(x)for x in row)+' |'for row in rows)+'\n'
def main():
 guard('report');(S/'figures').mkdir(exist_ok=True);analysis=read(S/'results/analysis_complete.json');metrics=analysis['metrics'];diff=analysis['differences'];seal=read(S/'configs/models_frozen.json');roots=read(S/'results/root_audits.json');train=read(S/'results/training_complete.json');physical=[read(S/f'results/audit_{phase}.json')for phase in ['train_dev','evaluation']]
 def metric(name,variant,seed,split='evaluation'):return next(r for r in metrics if r['split']==split and r['variant']==variant and r['seed']==seed and r['metric']==name)
 def fmt(name,v,s):
  r=metric(name,v,s);return f"{r['numerator']:.0f}/{r['denominator']} ({r['value']:.1%})"
 mainnames=['failure_h2_held','failure_h2_in_drawer','failure_h2_joint','paired_h2_held_both','failure_suffix3_full_prefix','nominal_current_joint']
 mainlabels=['落空后持有身份正确','落空后入屉集合正确','落空后完整关系正确','同根成功/失败均辨对','落空后3步前缀全对','常规分支当前关系正确']
 fig,axes=plt.subplots(1,2,figsize=(15,5),sharey=True)
 for seed,ax in enumerate(axes):
  for j,v in enumerate(['original','nominal','recovery']):
   rr=[metric(n,v,seed)for n in mainnames];x=np.arange(len(rr))+(j-1)*.25;bars=ax.bar(x,[r['value']for r in rr],.23,color=PAL[v],label=LABEL[v]);
   for b,r in zip(bars,rr):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.012,f"{r['numerator']:.0f}/{r['denominator']}",ha='center',fontsize=7,rotation=60)
  ax.set_xticks(range(len(mainlabels)),mainlabels,rotation=25,ha='right');ax.set_ylim(0,1.2);ax.set_title(f'锁定评估6根，模型种子{seed}');ax.set_ylabel('正确比例');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
 figure(fig,'01_primary','横轴为六项不同检查，纵轴为正确比例；灰色是未续训原模型参考，蓝色是常规经验续训，橙色是失败恢复经验续训。每根柱上写实际分子/分母。两个面板分别是模型种子0、1；根只有6个，不能把两个种子当成12个独立根。完整关系包含持有身份、三物体入屉集合和抽屉状态；这些分母不同，不能横向相加。')
 fig,ax=plt.subplots(figsize=(11,5));selected=['failure_h2_held','failure_h2_joint','paired_h2_held_both','failure_suffix3_full_prefix','nominal_current_joint'];dr=[next(r for r in diff if r['split']=='evaluation'and r['seeds']=='0,1'and r['metric']==m)for m in selected];y=np.arange(len(dr));v=np.array([r['recovery_minus_nominal']for r in dr]);lo=np.array([r['ci_low']for r in dr]);hi=np.array([r['ci_high']for r in dr]);ax.errorbar(v*100,y,xerr=np.array([v-lo,hi-v])*100,fmt='o',color=PAL['recovery'],capsize=4);ax.axvline(0,c='black',lw=1);ax.set_yticks(y,['持有身份','完整关系','配对两分支持有身份','3步前缀全对','常规分支当前关系']);ax.set_xlabel('失败恢复组减常规组（百分点）');ax.set_title('按整根配对自举的差值与95%区间');ax.grid(axis='x',alpha=.2)
 figure(fig,'02_paired_uncertainty','橙点是两组的配对差值，横线是10,000次整根自举的95%百分位区间，黑线为零差异。先在每个根内平均两模型种子，再重采样6个完整根；同根分支和两个模型种子不增加独立样本量。区间仅刻画这6根的不确定性，不保证新场景泛化。')
 fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
 for seed,ax in enumerate(axes):
  for v in ['original','nominal','recovery']:
   rr=[metric(f'failure_suffix{k}_joint',v,seed)for k in [1,2,3]];ax.plot([1,2,3],[r['value']for r in rr],'o-',color=PAL[v],label=LABEL[v])
  ax.set_xticks([1,2,3],['+1：尝试放置','+2：恢复抓取','+3：再次放置']);ax.set_ylim(-.05,1.05);ax.set_title(f'模型种子{seed}');ax.set_xlabel('从实际h2观察开始的想象步数');ax.set_ylabel('完整关系预测正确比例');ax.grid(alpha=.2);ax.legend()
 figure(fig,'03_suffix','折线横轴是观察真实落空后的未来1/2/3步，纵轴为对应终点完整关系正确比例。灰/蓝/橙分别为原模型、常规经验组、失败恢复组。三步均只调用学习到的转移和符号读出，不再输入未来图像；曲线上某个终点正确不等于三步前缀全部正确，后者单独报告。')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for seed,ax in enumerate(axes):
  for arm in ARMS:
   r=read(S/f'results/train_{arm}_seed{seed}.json');ax.plot([x['step']for x in r['curve']],[x['objective']for x in r['curve']],color=PAL[arm],label=LABEL[arm]);ax.scatter([c['update']for c in r['candidates']],[c['development_objective']for c in r['candidates']],marker='x',s=70,color=PAL[arm]);ax.axvline(r['selected']['update'],color=PAL[arm],ls=':',alpha=.5)
  ax.set_title(f'相同1000次续训更新，种子{seed}');ax.set_xlabel('优化器更新数');ax.set_ylabel('完整训练目标值');ax.legend();ax.grid(alpha=.2)
 figure(fig,'04_training','横轴是续训更新数，纵轴是原完整k3训练损失加相同终点监督项。蓝/橙实线是常规/恢复组被抽到的小批次训练损失；叉号是固定开发集在500/1000步的完整目标值，竖虚线是仅按开发集选择的更新。开发集包含真实成功和失败；评估集在选择冻结以后才进行物理采集。训练曲线与开发点样本构成不同，不应直接比较高低。')
 # Select distinct illustrative roots from precomputed per-root effects. Selection is report-only.
 roots_eval=[r for r in roots if r['split']=='evaluation'];scored=[]
 for root in roots_eval:
  vals=[]
  for seed in [0,1]:
   for v in ARMS:
    d=load(v,seed,'evaluation');mask=(d['root_id']==root['root_id'])&(d['branch']==2)&(d['mode']=='current')&(d['step']==2);vals.append((v,float(d['ok'][mask,-1][0])))
  delta=sum(x for a,x in vals if a=='recovery')-sum(x for a,x in vals if a=='nominal');scored.append((delta,root['root_id']))
 ordered=sorted(scored);chosen=[]
 for pair in [ordered[-1],ordered[0]]+ordered:
  if pair[1]not in chosen:chosen.append(pair[1])
  if len(chosen)==3:break
 case_rows=[];manifest=read(S/'data/data_manifest.json')['records']
 for ci,rid in enumerate(chosen):
  root=next(r for r in roots_eval if r['root_id']==rid);eid=rid+'_b2';rr=sorted([r for r in manifest if r['episode_id']==eid],key=lambda x:x['step']);fig=plt.figure(figsize=(15,10));gs=fig.add_gridspec(3,4,height_ratios=[1,1,1.3]);bounds=[1,2,4,5]
  for j,t in enumerate(bounds):
   with np.load(rr[t-1]['obs_after'])as z:
    for row,key in enumerate(['rgb_static','rgb_gripper']):
     ax=fig.add_subplot(gs[row,j]);ax.imshow(z[key]);ax.axis('off');ax.set_title(f'h{t} '+('固定相机'if row==0 else'腕部相机'))
  ax=fig.add_subplot(gs[2,:2]);bx=fig.add_subplot(gs[2,2:]);target=COLORS.index(root['matched_color']);held=COLORS.index(root['held_color'])
  for v in ['original','nominal','recovery']:
   d=load(v,0,'evaluation');mask=(d['episode_id']==eid)&(d['mode']=='current');steps=d['step'][mask];pp=d['rel'][mask];ax.plot(steps,pp[:,target],'o-',c=PAL[v],label=LABEL[v]);bx.plot(steps,pp[:,3+held],'o-',c=PAL[v],label=LABEL[v])
  labels=[read(rr[0]['labels_path'])['before']]+[read(r['labels_path'])['after']for r in rr];targettruth=[x['categorical'][target]==6 for x in labels];atruth=[x['categorical'][held]==3 for x in labels];ax.step(range(7),targettruth,where='mid',c='black',ls='--',label='实际状态');bx.step(range(7),atruth,where='mid',c='black',ls='--',label='实际状态')
  for a,title in [(ax,'目标'+root['matched_color']+'被持有'),(bx,'先放入的'+root['held_color']+'仍在抽屉')]:a.axhline(.5,c='gray',lw=.7);a.axvline(2,c='#999999',ls=':');a.set_ylim(-.05,1.05);a.set_xticks(range(7));a.set_xlabel('真实观测边界 h0–h6');a.set_ylabel('关系概率 / 实际0或1');a.set_title(title);a.legend(fontsize=8);a.grid(alpha=.2)
  fig.suptitle(rid+'：真实扰动分支与种子0的连续后验',fontsize=13);fig.tight_layout();caption=f'根{rid}的同一次真实执行。上两行分别是固定/腕部相机在h1已放入物体、h2抓取落空、h4恢复抓取、h5再次放置后的原始图像。底部横轴为观测边界，左图是目标{root["matched_color"]}的Held概率，右图是此前物体{root["held_color"]}的InDrawer概率；灰/蓝/橙对应原模型/常规组/恢复组，黑虚线是物理标签，水平灰线为固定0.5判定阈值。这里只展示种子0，完整两种子数据见表；三个案例根互不相同。'
  figure(fig,f'0{5+ci}_case',caption);case_rows.append([rid,root['held_color'],root['matched_color'],root['branches'][2]['grasp_controls'],str(root['branches'][2]['later_control_counts']),', '.join(root['branches'][2]['later_statuses'])])
 # Report text and tables are generated from actual recorded counts.
 lines=['# 真实落空与恢复经验：固定终点监督下的配对实验',f'生成时间：{now()}。本机执行；新实验目录：`{S}`。','',
 '本轮回答的是：观察到真实抓取落空后，加入失败与后续恢复经验，是否比同量常规经验更能修正持有状态、保留已经成立的入屉关系，并预测接下来的动作后果。这里的主要对照是常规组与恢复组；原模型只作参考。所有结果均来自真实CALVIN物理执行和实际训练，未生成假失败标签。','',
 '## 已完成的数据与公平性',table(['项目','实际执行','原因 / 解读'],[
 ['独立根', '12训练 + 6开发 + 6锁定评估 = 24','三分支始终跟随同一根；近似几何按颜色置换也排除'],
 ['真实转移','24 × 3 × 6 = 432','包含所有成功、执行失败及零步拒绝；没有按结果换根'],
 ['各组新增训练','24条完整episode / 144转移','同12根、同6步请求；常规b0+b1对恢复b0+b2'],
 ['共同旧训练','原train40的120episode / 480转移','同归一化、同权重、旧数据只读；未使用旧C/D'],
 ['共同目标','完整k3损失 + 真实终点后验符号监督','两组修复完全相同，只有新增经历不同'],
 ['更新与初始化','两种子 × 两组 × 1000 = 4000次','相同原6000步权重与AdamW状态、lr=1e-4、固定抽样和潜变量随机计划'],
 ['模型选择','开发6根、500/1000两个候选','全部选择和权重冻结后才采集独立评估6根'],
 ['隐蔽执行扰动','b2第二个动作将请求闭夹-1实际执行为开夹+1','每条请求/实际命令都记录；目标位姿和原执行器不变，随后恢复普通控制'],
 ['可观测性','扰动标志、控制轨迹、分支ID、成败和GT均非模型输入','不要求模型预知隐藏随机故障；主预测从真实失败观测h2之后开始']]),
 '评估与开发均固定为A/B场景、三种初始持有颜色各一根；并不代表跨环境C/D的独立验收，也不等于原指南的HULC与TS-LTL闭环项目已经完成。此次用此前获授权且哈希固定的物理执行器做机制实验。',
 table(['划分','根数','有效成功/落空配对根','成功转移','执行失败转移','零步拒绝'],[[sp,len([r for r in roots if r['split']==sp]),sum(r['valid_matched_root']for r in roots if r['split']==sp),*[(sum(b['all_statuses'].count(st)for r in roots if r['split']==sp for b in r['branches']))for st in ['success','executed_failure','refused']]]for sp in ['train','development','evaluation']]),
 '## 锁定评估结果',table(['检查','种子','原模型参考','常规新经验','失败恢复新经验'],[[label,seed,*[fmt(name,v,seed)for v in ['original','nominal','recovery']]]for name,label in zip(mainnames,mainlabels)for seed in [0,1]]),
 '完整关系判对要求Held身份、三个InDrawer真假及抽屉状态同时正确；入屉检查包含假阳性，不是只看正例。配对检查要求同根常规成功与落空两分支都辨对；三步前缀要求从h2出发的三个学习转移终点全部正确。常规分支结果用于检查“遇到抓取一律猜空手”是否以牺牲成功样本为代价。',
 table(['恢复减常规（两种子在根内平均）','差值百分点','95%整根自举区间','改善/变差/持平根'],[[r['metric'],f"{r['recovery_minus_nominal']*100:+.1f}",f"[{r['ci_low']*100:+.1f}, {r['ci_high']*100:+.1f}]",f"{r['positive_roots']}/{r['negative_roots']}/{r['ties']}"]for r in dr]),
 '只有6个独立评估根，区间较粗。两个模型种子是重复训练，不是新增物理样本。任何改善都限定于本轮开夹执行扰动、固定六技能序列与A/B分布；不能外推为任意自然抓取失败的纠错能力。',
 '## 更新、后继与失败归因',
 '连续后验保持完整历史，每个边界读实际双相机四帧窗口；没有用重置历史来伪造纠错。配对分支在扰动前的图像、编码、h0/h1概率和h2先验均逐项相同。h2后验差异来自随后实际观测。latent_changes.csv记录编码E、确定性h和后验类别概率q的实际变化；变化本身不能证明收益归因于某个子模块。',
 'first_errors.csv从h0开始记录首个完整关系错误，避免把早先的物体身份错误误称为故障后的首错。posterior_updates.csv逐例记录先验→后验纠正与回退。此次唯一实验因素是失败恢复经验，不再追加阈值调整、换根、重训或单例补丁。',
 '## 原始轨迹案例',table(['根','先入屉物体','目标物体','落空真实步数','后4动作真实步数','后4动作状态'],case_rows),
 '案例按评估后的最大/最小h2组间差异和补足独立根选择，仅作解释，未参与选择模型、改训练数据或调阈值。零步拒绝保留同一真实观测，不能冒充新的执行。',
 '## 图表及读法']
 for name,caption in FIG:lines+=['',f'![{name}](figures/{name}.png)','',caption]
 lines+=['','## 复现与资源',f'实际续训耗时：{train["seconds"]:.1f}秒；4次fit均1000更新。最大显存由每次训练receipt记录。首50转移实测：{read(S/"results/storage_first50.json")["bytes_per_transition"]:.0f}字节/转移（不含根快照和DINO缓存）。','',
 '`source /home/robot/research/projects/wm_symbolic/env.sh` 后执行 source/collect.py；`source .../world_model_env.sh` 后依次运行 audit_collection.py、audit_families.py、cache.py、train.py、evaluate.py、analyze.py、report.py、verify.py。训练器拒绝已有模型，采集器仅恢复缺失episode；不得直接覆盖本轮封存结果。原始终点、连续控制轨迹、请求/实际夹爪指令、标签、双相机窗口和完整日志都在本目录。',
 '训练前输入哈希保存在 configs/training_frozen.json；两个后续追加评估的聚合索引另存逐字节训练快照。封存校验验证旧记录未变、仅追加评估条目；源码、旧模型、执行器与训练/开发原始文件均受哈希检查。模型选择与评估采集先后由独立时间戳验证。','',
 '## 限制与下一步边界','本轮不改架构、不重置潜状态、不使用人工动态效果做搜索。评估仅6根，固定开夹扰动覆盖的失败类型有限；真实常规分支存在附带放置失败。若失败纠错提高而三步预测没有改善，应把两者分开报告，不应据此宣称世界模型已经可供可靠规划。是否扩大独立根和自然失败类型、以及继续原指南执行器与规划闭环，应依据本轮证据另行决定。']
 md='\n\n'.join(lines);(S/'REPORT.md').write_text(md)
 # Small standalone renderer; all local figure pixels embedded, no browser network dependencies.
 out=[];inside=False;rows=md.splitlines();i=0
 while i<len(rows):
  line=rows[i]
  if line.startswith('|'):
   group=[]
   while i<len(rows)and rows[i].startswith('|'):group.append(rows[i]);i+=1
   out.append('<table>')
   for j,l in enumerate(group):
    if j==1:continue
    tag='th'if j==0 else'td';out.append('<tr>'+''.join(f'<{tag}>'+html.escape(x.strip())+f'</{tag}>'for x in l.strip('|').split('|'))+'</tr>')
   out.append('</table>');continue
  m=re.match(r'!\[(.*?)\]\((.*?)\)',line)
  if m:out.append('<img alt="'+html.escape(m[1])+'" src="data:image/png;base64,'+base64.b64encode((S/m[2]).read_bytes()).decode()+'">')
  elif line.startswith('#'):level=len(line)-len(line.lstrip('#'));out.append(f'<h{level}>'+html.escape(line[level:].strip())+f'</h{level}>')
  elif line.strip():out.append('<p>'+html.escape(line)+'</p>')
  i+=1
 doc='<!doctype html><html lang="zh"><meta charset="utf-8"><title>失败恢复经验配对实验</title><style>body{font-family:"Noto Sans CJK SC",sans-serif;max-width:1250px;margin:36px auto;padding:0 24px;color:#17222d;line-height:1.7}h1,h2{color:#203a54}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{border:1px solid #d4dce3;padding:8px;text-align:left}th{background:#eef3f7}img{width:100%;height:auto;margin:20px 0}p{overflow-wrap:anywhere}</style>'+''.join(out)+'</html>'
 (S/'REPORT.html').write_text(doc);save(S/'results/report_complete.json',{'utc':now(),'figures':len(FIG),'case_roots':chosen,'self_contained_html':True});print('REPORT_COMPLETE',len(FIG),flush=True)
if __name__=='__main__':main()
