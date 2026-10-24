from analysis import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'axes.unicode_minus':False,'font.size':10,'figure.dpi':140})
AN={'original':'原正式Query','object_control':'对象旧覆盖头','object_coverage':'对象组合头'}
CONDS=[('observed','actual','真实观测'),('mixed','one_step','mixed单步'),('mixed','recursive','mixed递推'),('latent','one_step','αZ单步'),('latent','recursive','αZ递推')]
BN={'normal':'①正常','reverse_then_lift':'②开关开抓d','place_then_other_lift':'③放c后抓d'}
ACTIONS={'normal':['开','抓c','放c','关'],'reverse_then_lift':['开','关','开','抓d'],'place_then_other_lift':['开','抓c','放c','抓d']}
def one(rows,**kw):
 a=[r for r in rows if all(r[k]==v for k,v in kw.items())];assert len(a)==1,(kw,len(a));return a[0]
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join('---'for _ in headers)+' |\n'+'\n'.join('| '+' | '.join(str(x)for x in r)+' |'for r in rows)+'\n\n'
def main():
 assert read(E/'results/verification.json')['passed']
 steps=read(E/'results/step_errors.json');chains=read(E/'results/chain_success.json');joints=read(E/'results/joint_root_errors.json');paired=read(E/'results/actual_prediction_summary.json');first=read(E/'results/first_errors.json');examples=read(E/'results/probability_examples.json');figures=[]
 for arm in ARMS:
  fig,axs=plt.subplots(2,3,figsize=(14,7.6));fig.subplots_adjust(left=.1,right=.93,top=.82,bottom=.18,wspace=.3,hspace=.52)
  for seed,g,ax in [(s,g,axs[s,j])for s in [0,1]for j,g in enumerate(['Holding','In','drawer'])]:
   vals=np.array([[one(steps,seed=seed,world_model=wm,path=path,query_arm=arm,scope='all',cohort='eligible_roots',branch=branch,step=step)[g+'_wrong']for branch,step in itertools.product(BRANCHES,range(1,5))]for wm,path,_ in CONDS])
   im=ax.imshow(vals,vmin=0,vmax=11,cmap='Reds',aspect='auto');ax.set_yticks(range(5),[x[2]for x in CONDS]);ax.set_xticks(range(12),[f'{j+1}.{step}'for j in range(3)for step in range(1,5)],rotation=45);ax.set_title(f'种子{seed} / {g}');ax.set_xlabel('分支.技能步骤')
   for x in [3.5,7.5]:ax.axvline(x,color='#777',lw=1)
   for (i,j),v in np.ndenumerate(vals):ax.text(j,i,str(v),ha='center',va='center',fontsize=8.5,color='white'if v>=7 else'#222')
  cbax=fig.add_axes([.95,.25,.014,.48]);fig.colorbar(im,cax=cbax,label='错误根数 / 可评11根',ticks=[0,3,6,9,11])
  fig.suptitle(f'{AN[arm]}：各步事实错误出现在哪里\n每格是该事实组至少一项出错的根数；颜色越深错误越多',fontsize=14)
  fig.text(.5,.05,'① 开→抓c→放c→关　　② 开→关→开→抓d　　③ 开→抓c→放c→抓d\nHolding与In各含3个物体事实；drawer按3类取最大概率。只含11个完整根，失败根另列；分支与步骤不增加独立样本量。',ha='center',fontsize=10)
  name=f'step_errors_{arm}.png';fig.savefig(R/name,bbox_inches='tight');plt.close(fig);figures.append(name)
 text='# D12错误定位附录：只读取既有概率\n\n原实验已于2026-10-18完成[2787文件封存]('+str(D/'configs/delivery_manifest.json')+')和[进程检查]('+str(D/'results/completion_audit.json')+')。本附录另行保存，原封存内容未变；没有模型前向、训练、采集、阈值调整或新修正。\n\n'
 text+='**两分支八边界全对为0/11，表示没有一个根能让②③所有边界均正确，不表示每个边界都错，也不等于正常四步链为0。** αZ递推、对象组合头的正常链为2/11、3/11；②链为0/11、2/11；③链为4/11、2/11（依次为种子0、1）。两个种子的联合失败均包含Holding错误；种子0有10根只有Holding组出错、1根同时In出错，种子1为8根、3根。11个完整根的该递推条件下drawer组全程正确。\n\n'
 text+='## 计数与比较口径\n\n'+table(['元素','定义与原因'],[
 ['同一状态对照','真实z、从真实前态出发的单步预测z、从真实初态连续递推z，都对照同一实际后态标签；读出头保持不变'],['独立单位','12个根，11个完整身份对照根。132个成功技能边界来自11×3×4；另有失败根3个真实观测边界。边界、分支及种子不增加独立根数'],['错误组','Holding或In任一有效事实出错即记该组一次；drawer按类别是否错误记一次。不能把drawer的一个错分类计成两个独立错状态'],['范围','正文定位图和配对表使用11个完整根；CSV同时保留all_observed的135边界、eligible_roots的132边界，以及已见/反向分组'],['首次错误','每个根、分支、条件取最早有任一九事实错误的实际观测边界；完整链之外的未到达步骤不补假值'],['实际输入去重','真实观测概率在mixed与αZ下逐位相同，因此只保留一份observed条件；总计30个条件，非新增样本'],['配对类别','两边都错、仅实际错、仅预测错、两边都对。组级“两边都错”可能错在不同事实；事实级明细另列']])
 text+='## 各步Holding / In / drawer错误\n\n![对象组合头各步错误](step_errors_object_coverage.png)\n\n每列标记“分支.步骤”，每行固定输入来源，每个面板固定种子及事实组；白色0根错误，深红最多11根，数字给出精确计数。①为开抓c放c关；②为开关开抓d；③为开抓c放c抓d。竖线仅分隔分支。Holding/In阈值0.5，drawer取三类最大概率。其他头同样统计，见[原正式Query图](step_errors_original.png)、[对象旧覆盖图](step_errors_object_control.png)与[所有条件分步明细](step_errors.csv)。\n\n'
 text+='Holding输出在三个头之间逐位相同。空手关屉后的②第2步，αZ递推Holding误报为11/11、7/11；种子0涉及红/蓝/粉7/5/4个错误事实，种子1为4/3/0，个别状态同时误报多物体。③第3步放c后，6/11、7/11仍误报Holding(c)，三个颜色均出现。因此主导模式是“应空手时仍报持物”，并不是某一个颜色始终出错。正常链第4步关屉后，两种子各6/11仍误报Holding(c)。这些都是实际未持有时的假阳性。\n\n'
 text+='## 首个错误：动作和物体\n\n'
 rows=[]
 for seed,branch in itertools.product([0,1],BRANCHES):
  cc=[r for r in first if r['seed']==seed and r['world_model']=='latent'and r['query_arm']=='object_coverage'and r['path']=='recursive'and r['eligible_root']and r['branch']==branch];counts=collections.Counter()
  for r in cc:
   groups=sorted(set('Holding'if f.startswith('Holding')else'In'if f.startswith('In(')else'drawer'for f in r['first_error_facts']))
   label=f"第{r['first_error_step']}步{ACTIONS[branch][r['first_error_step']-1]}：{'+'.join(groups)}"if r['first_error_step']else'四步全对';counts[label]+=1
  rows.append([seed,BN[branch],'；'.join(f'{k}×{v}根'for k,v in counts.items())])
 text+=table(['种子','分支（αZ递推、组合头）','首错分布 / 全对'],rows)
 text+='第1步的In误报有时早于Holding错误，因此“所有联合失败根都含Holding错误”不能改写成“所有根的第一个错误都是Holding”。表中首错按时间取最早一次；同一步多个错误事实并列保留。每个物体的颜色、在根中属于查询c/目标d/其他物体、假阳性或假阴性，以及具体动作，均见[first_error_objects.csv](first_error_objects.csv)；每条分支的完整记录见[first_errors.csv](first_errors.csv)，所有输入与读出头的首错分布见[first_error_summary.csv](first_error_summary.csv)。\n\n'
 text+='## 单条四步链与两条链联合指标\n\n以下每格用“种子0；种子1”表示两个独立报告结果，每个计数的分母均为11个完整根；各条件计划根数仍为12，失败根不可评。两条分支联合必须同一根的②③全部八边界正确，不能把单链全对数相加。\n\n'
 rows=[]
 for arm,(wm,path,label)in itertools.product(ARMS,CONDS):
  vals=[]
  for b in BRANCHES:
   vals.append('；'.join(f"{one(chains,seed=s,world_model=wm,path=path,query_arm=arm,scope='all',branch=b)['all4_correct']}/11"for s in [0,1]))
  vals.append('；'.join(f"{sum(not r['all9_any_error']for r in joints if r['seed']==s and r['world_model']==wm and r['path']==path and r['query_arm']==arm)}/11"for s in [0,1]))
  rows.append([AN[arm],label,*vals])
 text+=table(['读出头','输入','①正常四步','②四步','③四步','②③联合八边界'],rows)
 text+='## 实际读取也错，还是只有预测读取错\n\n下面逐一配对同一个真实后态。每格依次列“种子0；种子1”的边界数量；每个种子、每个条件均有132个成功边界，来自11个根。四列互斥且和为132。这里“两边都错”指九事实组至少一项错误，不保证两边错的是同一事实。\n\n'
 rows=[]
 for arm,(wm,path,label)in itertools.product(ARMS,CONDS[1:]):
  rr=[one(paired,seed=s,world_model=wm,path=path,query_arm=arm,scope='all',cohort='eligible_roots',branch='all',step=0,group='all9')for s in [0,1]]
  rows.append([AN[arm],label,*['；'.join(str(r[k])for r in rr)for k in ['both_wrong','actual_only_wrong','predicted_only_wrong','both_correct']]])
 text+=table(['头','预测路径','两边都错','仅实际错','实际对/预测错','两边都对'],rows)
 text+='对αZ递推组合头，按事实组再分解：\n\n'
 rows=[]
 for g in GROUPS:
  rr=[one(paired,seed=s,world_model='latent',path='recursive',query_arm='object_coverage',scope='all',cohort='eligible_roots',branch='all',step=0,group=g)for s in [0,1]]
  rows.append([g,*['；'.join(str(r[k])for r in rr)for k in ['both_wrong','actual_only_wrong','predicted_only_wrong','both_correct']]])
 text+=table(['组','两边都错','仅实际错','实际对/预测错','两边都对'],rows)
 text+='实际读取错误说明在本数据上仍有表示/读出瓶颈，不能从该对照再区分二者。实际读取正确而预测读取错误，支持问题出现在预测潜状态及其与固定读出头的配合，但不能自动把全部误差唯一归因于转移模块；读出对预测分布的敏感性也未被排除。单步路径仍出错时，不能仅解释为递推累积。完整分步/分支/分组配对见[组级统计](actual_prediction_summary.csv)，精确同状态记录见[配对明细](actual_prediction_pairs.csv)，同一事实的两边错误见[事实级统计](fact_level_actual_prediction_pairs.csv)。\n\n'
 text+='## 最少量同状态输出对照\n\n均固定种子0、αZ、对象组合头；数字为同一实际后态事实的概率，Holding/In阈值0.5。前两个状态沿用原报告示例，第三个按“两边九事实组均错”候选的根ID、分支、步数排序取首个，属于事后解释示例，不参与模型选择。\n\n'
 chosen={'empty_close':['Holding(red)','Holding(blue)'],'actual_In_miss':['In(blue,drawer)','Holding(red)'],'both_wrong_posthoc_first_lex':['Holding(red)','Holding(blue)']}
 rows=[]
 for r in examples:
  if r['fact']in chosen[r['case']]:rows.append([r['root_id'].removeprefix('D_identity_')+f" / {BN[r['branch']]}第{r['step']}步",r['fact'],'真'if r['truth']else'假',*[f"{r[p+'_probability']:.4f}"for p in PATHS]])
 text+=table(['状态','事实','标签','真实z','单步z','递推z'],rows)
 text+='空手关屉一例实际读出正确，但单步已误报两种持物，递推仍错。放蓝抓红一例，真实z漏掉In(blue)，预测z却读对；同一步单步Holding(red)又漏报，说明“某个In读对”不等于状态正确。第三例实际错误是误报Holding(red)，递推错误却是漏报Holding(blue)：组级“两边都错”不能被当成同一错误原样传播。完整九事实概率保存在[示例明细](probability_examples.csv)。\n\n'
 text+='## 范围、复核和停止\n\n本附录只定位已有D12概率中的错误，不新增成功门槛，不修改原协议，不替换正式模型。原C退化、反向实际In读出薄弱和未验证的规划闭环能力仍然成立。135个观测边界全部保留；主解释中的132边界排除了那个实际开屉失败根，CSV也提供包含失败根的完整口径，缺失九步没有填值。\n\n'
 text+=f"[独立复核]({E}/results/verification.json)逐项重算4050条条件边界、1080条分支首错/全链、11520组配对统计及330项联合根结果，原2787个封存文件保持不变。本附录[输入哈希]({E}/configs/input_manifest.json)、[完成审计]({E}/results/completion_audit.json)、[独立封存]({E}/configs/delivery_manifest.json)供复查。所有运算仅使用既有JSON概率、标签和动作；没有模型前向、采集或训练。到此结束本轮，未自动开启后续修改。\n"
 (R/'APPENDIX.md').write_text(text);shutil.copy2(__file__,E/'source/report.py');save(E/'results/report_assets.json',{'utc':now(),'figures':figures,'report':'APPENDIX.md'});print('LOCALIZATION_REPORT_READY',R)
if __name__=='__main__':main()
