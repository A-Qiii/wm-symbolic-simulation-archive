"""Predeclared figures and tables; human interpretation is saved separately after scoring."""
from config import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import itertools
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':10,'figure.dpi':140})
ARMS=['original','control','coverage'];AN={'original':'原 Query','control':'旧覆盖续训','coverage':'加入组合覆盖'};AC={'original':'#808080','control':'#3174b5','coverage':'#209461'}
BN={'normal':'正常顺序','reverse_then_lift':'开→关→开→抓取','place_then_other_lift':'放入后抓另一物体'};PN={'actual':'真实观测','one_step':'单步预测','recursive':'递推预测'}
def one(rows,**kw):
 r=[x for x in rows if all(x[k]==v for k,v in kw.items())];assert len(r)==1,(kw,len(r));return r[0]
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v)for v in row)+' |'for row in rows)+'\n'
def frac(m,g='full9'):return f"{m[g+'_correct']}/{m[g+'_N']}"
def savefig(fig,name):fig.savefig(R/name,bbox_inches='tight');plt.close(fig)
def main():
 verify();actual=read(E/'results/actual_metrics.json');facts=read(E/'results/actual_fact_confusions.json');paired=read(E/'results/actual_paired.json');primary=read(E/'results/trajectory_primary.json');terminal=read(E/'results/trajectory_terminal.json');stage=read(E/'results/trajectory_stage_metrics.json');budget=read(E/'results/sampling_budget.json');traj=read(E/'results/trajectory_predictions.json')
 fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
 for seed,ax in enumerate(axs):
  for i,a in enumerate(ARMS):
   r=one(actual,cohort='new_AB_validation',seed=seed,query_arm=a,subset='dual');values=[r[g+'_correct']for g in ['full9','Holding','In','drawer']];bars=ax.bar(np.arange(4)+(i-1)*.24,values,.23,label=AN[a],color=AC[a]);ax.bar_label(bars,labels=[f'{v}/6'for v in values],fontsize=8,padding=2)
  ax.set_xticks(np.arange(4),['九事实全对','Holding 三事实','In 三事实','抽屉状态']);ax.set_title(f'种子 {seed}：AB 验证双关系状态');ax.set_ylim(0,7.2);ax.set_yticks(range(7));ax.set_ylabel('正确状态数（6 个独立根，各 1 个状态）');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axs[0].legend(loc='upper left',fontsize=9);fig.suptitle('保持原 Query 结构，仅改变 25% 训练槽位的组合覆盖');fig.tight_layout();savefig(fig,'actual_dual_all_facts.png')
 cohorts=['historical_AB_training','new_AB_validation','old_C_selection','old_C_validation','C_selection','C_validation'];labels=['历史 AB 训练','新 AB 非组合','旧 C 选择集','旧 C 验证集','中间态 C 选择集','中间态 C 验证集'];fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True)
 for seed,ax in enumerate(axs):
  for j,base in enumerate(['original','control']):
   vals=[100*one(paired,cohort=c,seed=seed,subset='nondual'if c=='new_AB_validation'else'all',baseline=base)['root_macro_difference']for c in cohorts];bars=ax.bar(np.arange(6)+(j-.5)*.32,vals,.31,color=AC[base],label='组合覆盖 − '+AN[base]);ax.bar_label(bars,labels=[f'{v:+.1f}'for v in vals],fontsize=9,padding=2)
  ax.axhline(0,color='black',lw=.7);ax.set_ylabel('九事实全对率差（百分点）\n先逐根平均，再对根平均');ax.set_title(f'种子 {seed}；负值表示退化');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='best')
 axs[-1].set_xticks(range(6),labels);fig.suptitle('原能力保留：历史 C 已被看过，只作本轮描述性评估');fig.tight_layout();savefig(fig,'retention_root_macro.png')
 fig,axs=plt.subplots(4,3,figsize=(14,12),sharey=True)
 for row,(seed,wm)in enumerate(itertools.product([0,1],['mixed','latent'])):
  for col,b in enumerate(BRANCHES):
   ax=axs[row,col]
   for j,a in enumerate(ARMS):
    vals=[one(primary,split='validation',seed=seed,world_model=wm,query_arm=a,branch=b,path=p)['correct']for p in PATHS];bars=ax.bar(np.arange(3)+(j-1)*.25,vals,.24,color=AC[a],label=AN[a]);ax.bar_label(bars,labels=[str(v)for v in vals],fontsize=8,padding=1)
   ax.set_title(f'种子 {seed} / {"F+.25Z"if wm=="mixed"else"αZ"} / {BN[b]}');ax.set_xticks(range(3),[PN[p]for p in PATHS]);ax.set_ylim(0,7);ax.set_yticks(range(7));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
   if col==0:ax.set_ylabel('四步全部九事实正确的根数 / 6')
 axs[0,0].legend(fontsize=8,loc='upper left');fig.suptitle('AB 验证：同一批冻结潜状态，换用原 Query / 对照续训 / 组合覆盖 Query');fig.tight_layout();savefig(fig,'fixed_world_model_chains.png')
 # Fixed in advance: one AB validation root, original target blue, second held object pink.
 root='A_combination_blue_01';branch='place_then_other_lift';fig,axs=plt.subplots(1,2,figsize=(14,7),sharey=True);labs=read(COMBO/'data/combo/_private/labels.json');example=[]
 for seed,ax in enumerate(axs):
  rr=[one(traj,root_id=root,branch=branch,slot=3,seed=seed,world_model='latent',query_arm=a)for a in ARMS];truth=labs[str(rr[0]['after'])]['y'];matrix=np.array([r[p+'_probability']for r in rr for p in PATHS]).T;im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
  for y in range(9):
   for x in range(9):ax.text(x,y,f'{matrix[y,x]:.2f}',ha='center',va='center',color='black'if matrix[y,x]>.6 else'white',fontsize=8)
  ax.set_xticks(range(9),[AN[a]+'\n'+PN[p]for a in ARMS for p in PATHS],rotation=45,ha='right',fontsize=8);ax.set_yticks(range(9),[f'{f}  真值={int(t)}'for f,t in zip(FACTS,truth)]);ax.set_title(f'种子 {seed} / 冻结 αZ 模型 / 第4步');ax.axvline(2.5,color='white',lw=2);ax.axvline(5.5,color='white',lw=2)
  for r in [r for r in traj if r['root_id']==root and r['branch']==branch and r['seed']==seed and r['world_model']=='latent']:
   example.append(r|{'truth':labs[str(r['after'])]['y'],'prediction':{p:discrete(r[p+'_probability'])for p in PATHS}})
 fig.colorbar(im,ax=list(axs),fraction=.025,pad=.03,label='符号为真的概率：紫色接近0，黄色接近1');fig.suptitle('预先固定的验证根：粉色物体被抓住，蓝色物体仍在抽屉内\nHolding / In 以0.5判定；抽屉三类取最大概率');fig.subplots_adjust(bottom=.28,top=.88,wspace=.08);savefig(fig,'fixed_joint_state_example.png');save(E/'results/fixed_example.json',example)
 # Detailed tables are complete CSVs; the report emphasizes actual readout before dynamics.
 txt='# 固定 Query 结构的组合覆盖对照实验\n\n'
 txt+='本轮只检验：在原有 Query 结构和固定600次更新下，加入已有的双关系训练状态，是否改善读出，同时保留原有能力。结果解释见 [解读](INTERPRETATION.md)。本轮不自动采用新头，也不采集新配置。\n\n'
 txt+='## 对照与数据边界\n\n'+table(['元素','固定条件 / 原因'],[
 ['结构','原 Query：对象 token、场景 token、对象均值与查询类型输入同一 MLP；未换成前轮探针。'],['初始权重','两种子各自从原正式 Query 出发；同一种子的两臂初始权重逐张量相同。'],['监督','Holding(R/B/P)、In(R/B/P,drawer)、Closed/Open/Intermediate 共9事实，保留原有效掩码及损失权重。'],['共同修正','两臂共同使用原生几何修正的历史标签副本：2个训练状态改变；原封存数据未改写。'],['训练臂','对照156历史状态、30个AB根；组合臂再加6个AB根各1个真实双关系观测。'],['采样','每批32：24个旧样本完全一致，另外8槽对照仍取旧样本，组合臂换为新样本；每新根800次。'],['预算','每臂600次AdamW更新、19200状态抽样，lr=.001、wd=.001、clip=1；不搜索、不早停。'],['划分','新增AB的6个_00根训练；6个_01根验证。D的6个根已被观察，仅诊断。'],['保留集','历史C按原生几何重标4个状态；12个缺物体姿态的旧C状态维持In未知，不记作负例。'],['被冻结部分','DINO、encoder、decoder、全部世界模型转移不更新；预测阶段直接读取已保存潜状态。'],['因果边界','两臂更新次数一致，正负与有效标签预算不同；结论针对本固定采样替换方案，不把所有差异归为多6个唯一状态。']])+'\n'
 txt+='### 实际抽样标签预算\n\n'+table(['种子','臂','旧/新状态抽样','有效事实','正例事实','负例事实'],[[b['seed'],AN[b['arm']],f"{b['old_draws']}/{b['new_draws']}",b['valid_fact_cells'],b['valid_positive_cells'],b['valid_negative_cells']]for b in budget])+'\n'
 txt+='## 先看真实观测读出\n\nAB验证每种子为6个根、54个去重边界观测；双关系状态各根1个，非双关系48个。同一根内状态不当成独立试验。表格为九事实同时正确的状态数；每事实混淆矩阵在CSV中。\n\n'+table(['种子','Query','全部54状态','双关系6状态','其余48状态','双关系 Holding / In / 抽屉'],[[s,AN[a],frac(one(actual,cohort='new_AB_validation',seed=s,query_arm=a,subset='all')),frac(one(actual,cohort='new_AB_validation',seed=s,query_arm=a,subset='dual')),frac(one(actual,cohort='new_AB_validation',seed=s,query_arm=a,subset='nondual')),' / '.join(frac(one(actual,cohort='new_AB_validation',seed=s,query_arm=a,subset='dual'),g)for g in ['Holding','In','drawer'])]for s,a in itertools.product([0,1],ARMS)])+'\n'
 txt+='![真实观测的九事实与三个分组](actual_dual_all_facts.png)\n\n横轴分组，纵轴正确状态数/6。灰色原Query、蓝色旧覆盖续训、绿色组合覆盖；每组必须全部相关事实正确，九事实全对才说明双关系与其他状态兼容。\n\n'
 dualrows=[]
 for s,a,g in itertools.product([0,1],ARMS,['Holding','In']):
  ff=[one(facts,cohort='new_AB_validation',seed=s,query_arm=a,subset='dual',fact=f)for f in FACTS[:3]if g=='Holding']if g=='Holding'else[one(facts,cohort='new_AB_validation',seed=s,query_arm=a,subset='dual',fact=f)for f in FACTS[3:6]];counts={k:sum(r[k]for r in ff)for k in ['TP','FN','FP','TN']};dualrows.append([s,AN[a],g,*[counts[k]for k in ['TP','FN','FP','TN']],f"{counts['TP']}/{counts['TP']+counts['FN']}",f"{counts['TN']}/{counts['TN']+counts['FP']}"])
 txt+='### 双关系正、负例均检查\n\n'+table(['种子','Query','事实组','TP','FN','FP','TN','正例召回','负例召回'],dualrows)+'\n每组每根一个真物体、两个假物体，共6正例与12负例。TP是真值为真且预测真，FN是漏报，FP是误报，TN是真值为假且预测假；抽屉按三类最大概率，逐类混淆矩阵也全部保存。\n\n'
 txt+='## 原有状态是否退化\n\n'+table(['数据','种子','原Query','旧覆盖续训','组合覆盖'],[[c,s,*[frac(one(actual,cohort=c,seed=s,query_arm=a,subset='all'))for a in ARMS]]for c,s in itertools.product(['historical_AB_training','old_C_selection','old_C_validation','C_selection','C_validation'],[0,1])])+'\n'
 txt+='![原能力保留](retention_root_macro.png)\n\n横轴为旧能力评估集；纵轴是组合覆盖相对基线的九事实全对率差，先按根内状态平均，再对根平均。灰柱相对原Query，蓝柱相对旧覆盖续训，负值是退化。历史AB是训练内，C是已看过的保留诊断，不能当新测试。所有有效标签不足9项的状态不进入九事实分母，但进入有效的单事实及分组指标。\n\n'
 txt+='## 再读取冻结世界模型的预测\n\nF+.25Z和αZ沿用各自既定模型。真实观测列读取当前真实z，单步列每步从真实前态预测，递推列从同一起点连续预测。每格要求4个技能边界的9事实全对，单位为6个AB验证根；三个分支与两个种子不扩增独立根数。\n\n'
 txt+=table(['种子','世界模型','分支','路径','原Query','旧覆盖续训','组合覆盖'],[[s,w,BN[b],PN[p],*[f"{one(primary,split='validation',seed=s,world_model=w,query_arm=a,branch=b,path=p)['correct']}/6"for a in ARMS]]for s,w,b,p in itertools.product([0,1],['mixed','latent'],BRANCHES,PATHS)])+'\n'
 txt+='![冻结模型四步评估](fixed_world_model_chains.png)\n\n每小图固定种子、世界模型和技能分支；横轴为真实/单步/递推，纵轴是四步全部九事实正确的根数/6。柱色与真实观测图一致。真实z上的改善若没有传递到预测z，不等于世界模型已经得到修复。\n\n'
 txt+='### 第4步关键关系\n\n'+table(['种子','模型','路径','Query','新目标Holding TP/正例','原物体In TP/正例','Holding误报根数'],[[s,w,PN[p],AN[a],f"{r['target_Holding_TP']}/{r['target_Holding_positive_N']}",f"{r['original_In_TP']}/{r['original_In_positive_N']}",r['other_Holding_false_positive_roots']]for s,w,p,a in itertools.product([0,1],['mixed','latent'],PATHS,ARMS)for r in[one(terminal,split='validation',seed=s,world_model=w,path=p,query_arm=a,branch='place_then_other_lift')]])+'\n'
 txt+='### 预先固定的错误定位示例\n\n固定根A_combination_blue_01、放入后抓另一物体分支、αZ模型，未按本轮结果挑选。第4步蓝色物体在抽屉内、粉色物体被夹持。\n\n![同一状态的九事实概率](fixed_joint_state_example.png)\n\n行是事实及原生真值，列是Query与输入来源。颜色表示事实为真概率，紫近0、黄近1，格内是概率。Holding/In用0.5，抽屉取最大值。所有根的首个九事实错误及错误事实在trajectory_root_outcomes.csv；此例四步原始概率在fixed_example.json。解释应先定位最早影响结果的错误，再比较已有真实z与预测z；本轮不进行第二次训练修正。\n\n'
 txt+='## 已看过的D仅作诊断\n\n'+table(['种子','Query','真实全部状态','真实双关系','αZ递推正常','αZ递推反向','αZ递推抓另一物体'],[[s,AN[a],frac(one(actual,cohort='D_seen_diagnostic',seed=s,query_arm=a,subset='all')),frac(one(actual,cohort='D_seen_diagnostic',seed=s,query_arm=a,subset='dual')),*[f"{one(primary,split='diagnostic_D_seen',seed=s,world_model='latent',query_arm=a,branch=b,path='recursive')['correct']}/6"for b in BRANCHES]]for s,a in itertools.product([0,1],ARMS)])+'\nD未参与本轮训练、选点或筛模型，过去已被观察，本表不作为新的测试验收。完整mixed/latent与全部路径均在CSV。\n\n'
 txt+='## 可复核交付\n\n'+''.join(f'- [{n}.csv]({n}.csv)\n'for n in ['actual_metrics','actual_fact_confusions','actual_paired','trajectory_primary','trajectory_paired','trajectory_stage_metrics','trajectory_fact_confusions','trajectory_terminal','trajectory_root_outcomes'])+'\n'
 txt+=f'- [固定协议]({E}/configs/protocol.json)、[训练前哈希冻结]({E}/configs/frozen.json)、[训练记录]({E}/results/training.json)、[采样标签预算]({E}/results/sampling_budget.json)。\n- [独立复核]({E}/results/verification.json)、[最终文件哈希清单]({E}/configs/delivery_manifest.json)。\n\n本轮仅更新4个实验Query副本。未重新采集、未修改世界模型或原有封存。反向颜色配对或新场景是否采集，留待后续决策；原项目验收门槛未变，本报告不宣称原指南整体完成。\n'
 (R/'REPORT.md').write_text(txt);print('QUERY_CONTROL_REPORT_WRITTEN',R,flush=True)
if __name__=='__main__':main()
