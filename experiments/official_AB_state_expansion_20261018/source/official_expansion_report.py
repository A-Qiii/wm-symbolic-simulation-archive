"""Static scientific figures and report; consumes verified records only."""
from pathlib import Path
import json,gzip,collections,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/official_AB_state_expansion_20261018';R=P/'reports/official_AB_state_expansion_20261018'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafbfc','axes.facecolor':'#fafbfc','savefig.facecolor':'#fafbfc'})
COLORS=['red','blue','pink'];CN={'red':'红','blue':'蓝','pink':'粉'}
def read(p):return json.loads(Path(p).read_text())
def load(n):return read(E/'results'/f'{n}.json')
def short(s):return s.replace('official_ABC_','').replace('_episode_','/')
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(x)for x in r)+' |\n'for r in rows)
def main():
 v=load('verification');assert v['passed'];rows=load('results');facts=load('candidate_facts');eps=load('episode_summary');colors=load('color_split_coverage');states=load('state_split_coverage');basic=load('basic_Holding_coverage');retest=load('old_source_retest_comparison');metrics=load('checkpoint_metrics');budget=load('run_budget');retrieval=load('retrieval_complete');matched=load('matched_task_segments');proto=read(E/'configs/protocol.json');figdir=R/'figures';figdir.mkdir(exist_ok=True)
 roots=sorted({r['root_id']for r in rows});splits=read(E/'configs/episode_split.json')['splits'];oldroots={r['root_id']for r in rows if r['source_kind'].startswith('old')}
 totals={g:{cl:sum(f['group']==g and f['classification']==cl for f in facts)for cl in['positive','negative','unknown']}for g in['Holding','In']}
 hcv=[z for f in facts if f['group']=='Holding'and f['classification']=='positive'for z in f['force_stability']['family_total_force_CV']]
 icv=[min(max(p['edge_total_force_CV'])for p in f['force_stability']['common_certain_paths']if p['passes'])for f in facts if f['group']=='In'and f['classification']=='positive']
 cv_summary={'Holding_positive_fact_count':totals['Holding']['positive'],'Holding_positive_max_finger_total_force_CV':max(hcv),'In_positive_fact_count':totals['In']['positive'],'In_positive_max_of_minimum_passing_path_worst_edge_CV':max(icv),'CV_threshold':.25,'In_summary_rule':'Per positive fact, among common certain paths passing the frozen criterion take minimum worst-edge CV; then report maximum across positive facts.'}
 (E/'results/force_stability_summary.json').write_text(json.dumps(cv_summary,ensure_ascii=False,indent=2)+'\n')
 fig,axes=plt.subplots(2,1,figsize=(14.6,8.4),sharex=True)
 for ax,g in zip(axes,['Holding','In']):
  data=[next(r for r in eps if r['root_id']==rid and r['group']==g)for rid in roots];bottom=np.zeros(len(roots))
  for key,label,color in [('positive','可靠为真','#21956f'),('negative','可靠为假','#3879ad'),('unknown','未知','#adb3bb')]:
   vals=np.array([r[key]for r in data]);bars=ax.bar(np.arange(len(roots)),vals,bottom=bottom,color=color,label=label,width=.72)
   for i,(val,bar)in enumerate(zip(vals,bars)):
    if roots[i]in oldroots:bar.set_hatch('//')
    if val>=5:ax.text(i,bottom[i]+val/2,str(val),ha='center',va='center',fontsize=9,color='white'if key!='unknown'else'#283340')
   bottom+=vals
  ax.set(ylabel='事实格数（每初态 × 3色）',title=g+'：只统计第96步候选，检查全部第32–96步',ylim=(0,100));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
  for i,r in enumerate(data):ax.text(i,bottom[i]+2,f"n={r['initializations']}",ha='center',fontsize=9)
 axes[1].set_xticks(range(len(roots)),[short(r)+'\n'+('验证'if splits[r]=='validation_candidate'else'训练')for r in roots])
 fig.legend(handles=[Patch(color=c,label=l)for c,l in[('#21956f','可靠为真'),('#3879ad','可靠为假'),('#adb3bb','未知')]],frameon=False,ncol=3,loc='center',bbox_to_anchor=(.5,.927))
 fig.suptitle('12个来源episode的标签覆盖；斜纹为原来源的少量复测',fontsize=15,y=.98)
 fig.text(.5,.018,'绿色=谓词可靠为真，蓝色=可靠为假，灰色=未知；不是预测对错或物体颜色。n为初态数，原episode仅各4次复测，新episode各30次。',ha='center',fontsize=10)
 fig.tight_layout(rect=(0,.045,1,.90));fig.savefig(figdir/'episode_coverage.png',dpi=170);plt.close(fig)

 fig,axes=plt.subplots(1,2,figsize=(11.8,5.8))
 labels=['Holding真','Holding假','In真','In假','桌面负例']
 for ax,split,nroots in zip(axes,['train_candidate','validation_candidate'],[8,4]):
  matrix=[]
  for g,cl in [('Holding','positive'),('Holding','negative'),('In','positive'),('In','negative')]:matrix.append([next(r for r in colors if r['split']==split and r['group']==g and r['color']==c)[cl+'_episodes']for c in COLORS])
  matrix.append([next(r for r in states if r['split']==split and r['color']==c and r['state']=='On_table_negative')['episodes']for c in COLORS]);a=np.asarray(matrix)
  ax.imshow(a/nroots,cmap='Blues',vmin=0,vmax=1,aspect='auto')
  for i in range(5):
   for j in range(3):ax.text(j,i,f'{a[i,j]} / {nroots}',ha='center',va='center',color='white'if a[i,j]/nroots>.55 else'#17344b',fontsize=13)
  ax.set(xticks=range(3),xticklabels=[CN[c]+'物体'for c in COLORS],yticks=range(5),yticklabels=labels,title=('训练候选'if split=='train_candidate'else'验证候选')+'：不同来源episode数')
 fig.suptitle('看独立来源覆盖，而不是把相邻帧当独立样本',fontsize=15,y=.97)
 fig.text(.5,.025,'单元格为“有该状态的episode数 / 该划分episode总数”；颜色越深覆盖比例越高，不表示精度。\n桌面负例需窗口内持续由原生桌面底座承重且Holding、In均可靠为假。',ha='center',fontsize=10)
 fig.tight_layout(rect=(0,.10,1,.92));fig.savefig(figdir/'split_color_episode_coverage.png',dpi=170);plt.close(fig)

 # Deterministic explanatory examples, separate from any sampling/adoption decision.
 selected=[]
 unknown_group='Holding'if any(f['group']=='Holding'and f['classification']=='unknown'for f in facts)else'In'
 for group,classification,color in [('Holding','positive','pink'),('In','positive',None),(unknown_group,'unknown',None)]:
  pool=[f for f in facts if f['group']==group and f['classification']==classification and(color is None or f['color']==color)and f['source_kind']=='new_official_frame']
  if not pool:pool=[f for f in facts if f['group']==group and f['classification']==classification]
  if not pool:continue
  f=sorted(pool,key=lambda f:(f['root_id'],f['frame_id'],f['color']))[0];r=next(r for r in rows if r['root_id']==f['root_id']and r['frame_id']==f['frame_id']);selected.append((f,r))
 fig,axes=plt.subplots(len(selected),4,figsize=(12.8,3.3*len(selected)+1.2),squeeze=False)
 for i,(f,r)in enumerate(selected):
  source=np.load(r['source_path'],allow_pickle=False);new=np.load(r['candidate']['observation_path'],allow_pickle=False)
  for j,array in enumerate([source['rgb_static'],new['rgb_static'],source['rgb_gripper'],new['rgb_gripper']]):
   axes[i,j].imshow(array,interpolation='nearest');axes[i,j].set(xticks=[],yticks=[])
   for sp in axes[i,j].spines.values():sp.set_visible(False)
   if i==0:axes[i,j].set_title(['原静态图（参考）','新第96步静态图','原腕部图（参考）','新第96步腕部图'][j])
  cls={'positive':'可靠为真','negative':'可靠为假','unknown':'未知'}[f['classification']]
  axes[i,0].set_ylabel(f"{short(f['root_id'])}\n帧{f['frame_id']}\n{f['fact']}\n新状态：{cls}",fontsize=9)
 fig.suptitle('原图与400 ms后新图：新标签仅关联新图',fontsize=15,y=.98)
 fig.text(.5,.018,'逐类按来源episode/帧/颜色字典序首例；仅解释。静态200×200、腕部84×84，仅显示缩放，没有裁剪或把新标签贴到旧图。',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(figdir/'original_new_examples.png',dpi=170);plt.close(fig)
 (E/'results/illustration_selection.json').write_text(json.dumps([f for f,r in selected],ensure_ascii=False,indent=2)+'\n')

 fig,axes=plt.subplots(1,3,figsize=(14.4,4.8));m=[x for x in metrics if x['split']=='all'];x=[t['step']for t in m]
 for ax,key,title,color,unit in [(axes[0],'max_object_drift_mm','物体最大位移','#c56b35','mm'),(axes[1],'max_object_rotation_deg','物体最大转角','#8054a4','度')]:
  ax.plot(x,[t[key]for t in m],'o-',color=color,label=title,lw=2);ax.set(title=title+'：三物体/256初态最大值',ylabel='相对本次初始化（'+unit+'）',ylim=(0,max(t[key]for t in m)*1.14));ax.legend(frameon=False)
 for key,label,color in [('static_MAE','静态相机','#218b70'),('wrist_MAE','腕部相机','#bd5289')]:axes[2].plot(x,[t[key]for t in m],'o-',label=label,color=color,lw=2)
 axes[2].set(title='新图与原官方图像差异',ylabel='平均像素绝对误差（0–255尺度）');axes[2].legend(frameon=False)
 for ax in axes:ax.set(xticks=x,xticklabels=['32\n133 ms','64\n267 ms','96\n400 ms'],xlabel='固定物理步 / 仿真时间');ax.set_ylim(bottom=0);ax.grid(alpha=.16)
 fig.suptitle('保留实际漂移，没有按漂移或正例数量挑选时刻',fontsize=15,y=.98)
 fig.text(.5,.02,'每点为固定检查时刻；最大值可来自不同初态/物体。图像误差对背景和物体共同平均，不能代替物理接触或姿态证据。',ha='center',fontsize=10)
 fig.tight_layout(rect=(0,.07,1,.93));fig.savefig(figdir/'drift_checkpoints.png',dpi=170);plt.close(fig)

 force_examples=[(f,r)for f,r in selected if f['group']=='Holding']
 if force_examples:
  fig,axes=plt.subplots(1,len(force_examples),figsize=(max(10,7*len(force_examples)),5.2),squeeze=False)
  for ax,(f,r)in zip(axes[0],force_examples):
   with gzip.open(r['candidate']['physics_history_path'],'rt')as stream:history=[json.loads(line)for line in stream]
   for k,label,color in[(0,'左指族总法向力','#2879ae'),(1,'右指族总法向力','#d47b31')]:ax.plot([t['physics_step']for t in history],[t['confidence'][f['color']]['family_total_normal_force_N'][k]for t in history],label=label,color=color)
   ax.axvspan(32,96,color='#b1b8c0',alpha=.18,label='固定可靠性窗口');ax.set(xlabel='物理步（96步=400 ms）',ylabel='新接触记录的总法向力（N）',title=short(f['root_id'])+' / '+str(f['frame_id'])+'\n'+f['fact']+'：'+('可靠为真'if f['classification']=='positive'else'未知'));ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.15)
  fig.suptitle('正例须保持接触证据，并通过窗口内力波动检查',fontsize=15,y=.98)
  fig.text(.5,.02,'蓝/橙为左右指族的接触力，不是电机最大力设置；阴影为第32–96步。\nCV=标准差/均值，正例各指族要求≤0.25。',ha='center',fontsize=10)
  fig.tight_layout(rect=(0,.10,1,.91));fig.savefig(figdir/'Holding_force_examples.png',dpi=170);plt.close(fig)

 ep_table=[]
 for rid in roots:
  h=next(x for x in eps if x['root_id']==rid and x['group']=='Holding');n=next(x for x in eps if x['root_id']==rid and x['group']=='In');ep_table.append([short(rid),'验证'if splits[rid]=='validation_candidate'else'训练','旧源复测'if rid in oldroots else'新源',h['initializations'],*[h[k]for k in['positive','negative','unknown']],*[n[k]for k in['positive','negative','unknown']],h['empty_hand_states']])
 color_table=[[('训练'if c['split']=='train_candidate'else'验证'),c['group'],CN[c['color']],*[f"{c[k]} / {c[k+'_episodes']}"for k in['positive','negative','unknown']]]for c in colors if c['split']!='all']
 basic_table=[[('训练'if c['split']=='train_candidate'else'验证'),CN[c['color']],c['positive_episodes'],c['negative_episodes'],'满足'if c['passes']else'不足']for c in basic]
 state_table=[]
 for split in ['train_candidate','validation_candidate']:
  for c in COLORS:
   vals=[next(s for s in states if s['split']==split and s['color']==c and s['state']==st)for st in['Holding','In_drawer','On_table_negative','Other_known_negative','unknown']];state_table.append(['训练'if split=='train_candidate'else'验证',CN[c],*[f"{s['count']} / {s['episodes']}"for s in vals]])
 reasons=collections.Counter(f['reason']for f in facts if f['classification']=='unknown');retest_counts=collections.Counter((r['old_step8_class'],r['new_step96_class'])for r in retest)
 lift=[s for s in matched if s['target_color_hint']is not None];paired=sum(s['contains_target_Holding_positive_and_negative']for s in lift)
 changes=[r for r in retest if r['old_step8_class']=='positive'and r['new_step96_class']!='positive'];shortmatch=all(r['first8_truth_matches_prior']and r['first8_mask_matches_prior']for r in retest)
 early_late=load('step8_to96_facts');early_h=[r for r in early_late if r['group']=='Holding'and r['step8_raw_value']and r['step8_native_valid']];late_h=collections.Counter(r['step96_classification']for r in early_h)
 newunknown=[f for f in facts if f['classification']=='unknown'];first=sorted(newunknown,key=lambda f:(f['root_id'],f['frame_id'],f['fact']))[0]if newunknown else None
 unknown_example=('无未知事实。'if first is None else f"按来源/帧/谓词排序的首个未知为{short(first['root_id'])}、帧{first['frame_id']}、{first['fact']}；首个窗口问题记录为`{json.dumps(first['first_window_failure'],ensure_ascii=False)}`，归类`{first['reason']}`。完整力波动与路径证据留在逐事实表；未据此修改控制或补采。")
 headline=('所有预先定义的Holding基本覆盖格均满足'if all(x['passes']for x in basic)else'仍有预先定义的Holding基本覆盖格不足')
 text=f'''# 官方A/B状态数据有界扩充：12个episode，256次长窗口验证

完成240个新增官方源帧和16个旧源复测，共256次96步重仿真。来源为原4个episode加A/B各4个新episode；新增源帧不含C/D。**{headline}**：此处只检查每个划分、每种颜色的Holding正/负事实是否各来自至少2个episode，不能视作训练完成或泛化提升。

最终第96步候选：Holding真{totals['Holding']['positive']}、假{totals['Holding']['negative']}、未知{totals['Holding']['unknown']}；In真{totals['In']['positive']}、假{totals['In']['negative']}、未知{totals['In']['unknown']}。每组有768个事实格（256状态×3色）。所有未知和漂移保留，没有补采直到达标。模型训练/前向均为0。

## 元数据、冻结选样与下载预算

此前未下载语言文件。本轮只下载中央目录核对过的标准`training/lang_annotations/auto_lang_ann.npy`，压缩23,031,646字节。通过`pickletools`操作码解释器构造惰性数据节点，解析17870条任务文本和帧区间，**未调用输入指定的构造器，未使用`pickle.load`或`allow_pickle=True`**；语言嵌入被忽略。A/B共12204条，跨episode/scene区间为0；C的元数据仅被识别并排除，不选取C/D源帧。

A有15、B有13个未用episode同时含三色桌面/屉内抓取及至少两段放入抽屉任务。按种子170819的固定哈希排序各选4个episode。每个新episode取三色×桌面/屉内各1段，在段内0、1/3、2/3、1取帧，另取2段放入抽屉的0、1/2、1，共30帧。任务仅用于找候选；真值完全来自新仿真物理记录，不从语言或模型预测推断。

{table(['下载阶段','累计响应体MiB','说明'],[
['此前官方试验',f'{268218222/2**20:.2f}','保留原台账，包含中央目录等全部已有响应体'],
['语言元数据后',f'{proto["network_used_before_new_frames"]/2**20:.2f}','只读一个标准语言文件'],
['采集前保守上界',f'{proto["projected_cumulative_upper_bytes"]/2**20:.2f}','按压缩成员大小加每帧最保守ZIP头预留'],
['全部240源帧下载后',f'{retrieval["cumulative_body_bytes"]/2**20:.2f}','实际累计，含新元数据与所有Range响应体'],
['授权硬上限','512.00','未扩大预算，未整包下载']])}

每次Range均检查206、Content-Range、长度、ETag；成员检查CRC32与SHA256，源NPZ只读取数值数组。当前累计下载{retrieval['cumulative_body_bytes']:,}字节，剩余{(512*2**20-retrieval['cumulative_body_bytes'])/2**20:.2f}MiB。剩余预算未用于追加样本。

## episode级预划分与独立性

原4个已查看的episode归训练候选；新A/B各4个中，各2个预留验证候选、各2个归训练候选，总8训练候选/4验证候选。冻结后不因结果改变划分。任何源帧及其32/64/96步派生图均随来源episode，不跨分割。

本轮长窗口统计为训练136初态、验证120初态。原128源帧完整保留，其中仅16个做长窗口复测；新240帧使累计不同官方源帧达到368，但**本报告96步候选只有256个**。768份检查点观测不是768条独立轨迹，相关任务段、颜色和相邻帧也不增加episode数。

## 固定控制与可靠性规则

- 初始化按官方状态恢复一次；未记录的物体速度置零。随后不传送、不施加外力、不改变动力学、不加抓取约束。
- 每次固定96个1/240秒物理步（400ms）。每8步重发同一记录TCP位置/朝向与夹爪命令，共12个原生控制周期；没有重放后续人类动作。
- 第0–96步都保存物理状态与接触，第32/64/96步保存新双RGB、深度和proprio。唯一候选取第96步，不按效果选时刻。
- 稳定窗口为第32–96步，共65个检查点、跨度266.7ms；每步严格Holding/In真值必须与末步一致，且原生有效性、接触置信界都通过。原谓词定义不变，边界继续未知。
- Holding正例须持续双指承重、无外部支持，且各指族总法向力CV≤0.25；In正例须有贯穿窗口的同一条确定容器内承重路径，该路径每条边的总确定承重力CV≤0.25。CV为总体标准差/均值。负例须逐步满足明确负证据，不强加正例力波动条件。
- “桌面负例”仅作覆盖诊断：窗口内始终有原生桌面底座的确定向上支撑，且Holding/In均可靠为假；没有修改正式标签集合。“空手”要求三色Holding均可靠为假。

这些是新物理状态的标签候选，不能贴到旧官方RGB；分别初始化的任务段帧也不构成执行过的技能转移。原始模型、执行器、已封存试验不被替换。

## 按episode、颜色及状态统计

{table(['episode','划分候选','来源','初态','H真','H假','H未知','In真','In假','In未知','空手状态'],ep_table)}

![逐episode覆盖](figures/episode_coverage.png)

纵轴为事实格数，绿色真、蓝色假、灰色未知；斜纹是原episode的4初态复测。新episode每组谓词90格，旧episode复测每组12格，所以不把柱高解读为表现优劣。

{table(['划分候选','谓词','颜色','真：事实/episode','假：事实/episode','未知：事实/episode'],color_table)}

![划分与颜色的独立episode覆盖](figures/split_color_episode_coverage.png)

热图数字是有该状态的episode数/划分总episode数，颜色深浅表示覆盖比例，不表示准确率。两个划分分别有8/4个episode，不能把分母不同的颜色直接当样本量比较。

{table(['划分候选','颜色','Holding：帧/episode','In：帧/episode','桌面负例：帧/episode','其他已知负例：帧/episode','未知：帧/episode'],state_table)}

同一物体的新状态按Holding、In、桌面负例、其他已知负例、未知分类，三色分别统计；“其他”可能是其他表面或非目标容器，不称作桌面。空手为整手状态，见逐episode表。

{table(['划分候选','颜色','Holding正episode','Holding负episode','基本覆盖检查'],basic_table)}

48段三色抓取候选中，{paired}/48段同时保留了目标颜色的可靠Holding正和负状态，详见[任务段配对表](matched_task_segments.csv)。它们只是来自同一标注段的匹配状态，不能当作本次执行得到的技能前后转移。放入抽屉任务不含目标颜色时，不凭语言猜测对象身份。

## 未知、长短窗口复测与漂移

{table(['未知的首要原因','事实格数'],sorted(reasons.items()))}

未知原因按固定判定顺序记录：先找到窗口内首个真值变化、原生几何边界或接触置信不足，再检查正例力CV/固定路径。{unknown_example}

全256次运行中，第8步原始严格Holding为真的事实有{len(early_h)}个；到第96步长窗口候选，其中{late_h['positive']}个可靠为真、{late_h['negative']}个可靠为假、{late_h['unknown']}个未知。这些早期夹持未保留的情况也纳入记录，即使在稳定窗口前已掉落、后段成为明确负例，也不隐藏。[物理变化事件表](physical_change_events.csv)保存每个事实在第8步后第一次变值/原生失效时的接触、位姿与几何证据；这是状态变化记录，不把语言任务自动解释为执行成功或失败。

16个旧源复测的前8步真值及旧4–8步掩码与此前记录{'全部一致'if shortmatch else'存在差异，逐项保留于比较表，不能仅归因为窗口延长'}。下表是96个事实格（16×6）的旧短窗口判定→新长窗口判定：

{table(['旧第8步候选','新第96步候选','事实格数'],[[a,b,n]for(a,b),n in sorted(retest_counts.items())])}

旧可靠正事实中有{len(changes)}个未保留为长窗口可靠正例；[逐事实复测表](old_source_retest_comparison.csv)保存第8步后的首个原始真值变化/失效及当时接触证据。长运行继续按原生周期保持同一TCP/夹爪目标，不是无控制自由演化；不据此宣称仅一个独立变量造成变化。旧源复测不增加独立来源。

![原始与新状态图像](figures/original_new_examples.png)

四列依次为旧静态图、新96步静态图、旧腕部图、新96步腕部图。按固定字典序选择粉色Holding正例、In正例及未知事实作解释；本轮Holding无未知，第三行展示In未知。物体红/蓝/粉色表示外观，标签只写给新状态。

![Holding力证据示例](figures/Holding_force_examples.png)

蓝/橙线为左/右指族实际接触总法向力，纵轴N，横轴物理步；灰色窗口第32–96步。它们不是原生命令中的电机最大力200/2；正例还需无外部承重和低于固定CV上限。曲线不用于调整采样或阈值。

109个Holding正事实的各指族总力CV最大为{max(hcv):.6f}，低于固定0.25上限。220个In正事实均有贯穿窗口且通过CV门槛的确定支撑路径；按每个事实取合格路径中最小的“路径最差边CV”，再跨事实取最大为{max(icv):.6f}。这里只描述已冻结规则下实际证据，不把力稳定替代几何条件，也不推断超过400ms的保持能力。

{table(['步','时间ms','物体最大位移mm','物体最大转角°','TCP最大位移mm','静态MAE','腕部MAE'],[[t['step'],f"{t['step']/240*1000:.1f}",f"{t['max_object_drift_mm']:.3f}",f"{t['max_object_rotation_deg']:.3f}",f"{t['max_TCP_drift_mm']:.3f}",f"{t['static_MAE']:.3f}",f"{t['wrist_MAE']:.3f}"]for t in m])}

![固定检查时刻的位姿与像素变化](figures/drift_checkpoints.png)

横轴为32/64/96步及时间。位移/转角相对本次初始化，统计三物体与全256初态的最大值；不同检查点极值可能来自不同对象。静态/腕部MAE在0–255像素尺度，对全图均值再对初态平均；背景会稀释物体变化，不能代替物理证据。全97个时刻最大物体位移{max(r['max_object_drift_all_ticks_m']for r in rows)*1000:.3f}mm，最大转角{max(r['max_object_rotation_all_ticks_rad']for r in rows)*180/np.pi:.3f}°；未以漂移阈值事后删样本。

## 验证与交付

256次初始化全部完成，共24576物理步，0重试，仿真执行耗时{budget['elapsed_seconds']:.2f}秒（下载另计）。离线从保存的全局接触重新计算Holding，并重算冻结原生几何标签、逐步力证据及固定窗口规则；核查{v['physics_records_recomputed']}份物理记录、{v['new_observations_verified']}份新观测、{v['candidate_fact_cells']}事实格及{v['motor_calls_verified']}电机调用。源帧及派生图的episode划分无交叉，C/D源帧为0，原两轮封存哈希未变。复核没有新仿真或模型调用。

{table(['文件','用途'],[
['[采集前可行性](FEASIBILITY.md)','安全元数据、候选数量与预算'],
['[冻结方案](../../experiments/official_AB_state_expansion_20261018/configs/protocol.json)','采样、96步、力稳定规则与无补采上限'],
['[episode预划分](../../experiments/official_AB_state_expansion_20261018/configs/episode_split.json)','12个来源的固定归属'],
['[逐事实表](candidate_facts.csv)','真值、未知原因、力CV与同刻新图路径'],
['[逐episode表](episode_summary.csv)','来源、划分、正负未知与空手'],
['[颜色/划分覆盖](color_split_coverage.csv)','事实数及独立episode数'],
['[状态覆盖](state_split_coverage.csv)','Holding、屉内、桌面负例等'],
['[独立复核记录](../../experiments/official_AB_state_expansion_20261018/results/verification.json)','复核范围及真实通过情况'],
['[累计下载台账](../../experiments/official_AB_state_expansion_20261018/results/network_ledger.json)','所有新Range和历史基数'],
['[运行说明](RUN_COMMANDS.md)','脚本与环境路径'],
['[交付哈希清单](../../experiments/official_AB_state_expansion_20261018/configs/delivery_manifest.json)','数据、图表、源码、日志完整性']])}

新源帧在`data/frames/`；每个新仿真在`data/resim/<episode>/<frame>/`，保存`physics_history.jsonl.gz`、控制审计、积分台账、三个新观测及唯一末步候选。GT/接触与公共观测分文件保存。源代码在本轮`source/`，完整配置在`configs/`。

本轮仅完成有界状态数据扩充与质量检查，尚无训练、模型比较或泛化结论。即使基本覆盖满足，也只是进入后续受控读取器试验的候选条件；原指南的数据量、正式模型与TS-LTL闭环验收仍未完成。是否采用数据和进入训练由父线程后续决定，本轮不继续补采或调参。
'''
 (R/'REPORT.md').write_text(text)
 (R/'RUN_COMMANDS.md').write_text('''# 本轮已执行命令与复核入口

全部在本机独立项目环境执行；单个采集进程持有项目GPU锁。以下是已执行脚本顺序，不应重新采集已封存的样本。

```bash
source /home/robot/research/projects/wm_symbolic/env.sh
python /home/robot/Documents/Codex/2026-10-13/task-4/official_expansion_fetch.py
python /home/robot/Documents/Codex/2026-10-13/task-4/official_language_safe_parse.py
python /home/robot/Documents/Codex/2026-10-13/task-4/official_expansion_select.py
python /home/robot/Documents/Codex/2026-10-13/task-4/official_expansion_download.py
python /home/robot/Documents/Codex/2026-10-13/task-4/official_expansion_run.py
python /home/robot/Documents/Codex/2026-10-13/task-4/official_expansion_verify.py
source /home/robot/research/projects/wm_symbolic/world_model_env.sh
python /home/robot/Documents/Codex/2026-10-13/task-4/official_expansion_report.py
```

前六步使用独立Python3.10原生仿真环境；图表使用Python3.11的NumPy/Matplotlib，不调用世界模型。一次性运行拒绝已有attempts台账；技术失败停止并保留，不自动重试。离线复核不运行物理，但会写复核产物；再次审计需写入新目录以保持交付哈希。
''')
 shutil.copy2(__file__,E/'source'/Path(__file__).name);print('REPORT_CREATED',totals,'BASIC',all(x['passes']for x in basic),flush=True)
if __name__=='__main__':main()
