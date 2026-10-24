"""Report and static scientific figures for the frozen endpoint appendix."""
try:
 from holding_endpoint_verify import *
except ModuleNotFoundError:
 from verify import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'axes.unicode_minus':False,'font.size':10,'figure.dpi':150})
CN={'all':'全部已观测端点','open':'开屉','empty_close':'空手关屉','place':'放下c','close_after_place':'放下后关屉','lift_c':'抓起c','lift_d_after_place':'放c后换抓d','lift_d_after_openclose':'开关开后抓d'}
FN={'object':'对象头','context':'上下文头'};WN={'mixed':'mixed','latent':'αZ'}
HEADS=[('object',0),('object',1),('context',0),('context',1)]
def one(rows,**kw):
 out=[r for r in rows if all(r[k]==v for k,v in kw.items())];assert len(out)==1,(kw,len(out));return out[0]
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join('---'for _ in headers)+' |\n'+'\n'.join('| '+' | '.join(str(v)for v in row)+' |'for row in rows)+'\n\n'
def rgb(path):
 d=np.load(path)
 return np.concatenate([np.array(Image.fromarray(d[k]).resize((200,200),Image.Resampling.NEAREST))for k in ['rgb_static','rgb_gripper']],axis=1)
def main():
 assert read(E/'results/verification.json')['passed']
 actual=read(E/'results/actual_endpoints.json');ms=read(E/'results/actual_condition_metrics.json');ag=read(E/'results/actual_agreement_subset.json');rootms=read(E/'results/actual_root_metrics.json');elig=read(E/'results/condition_eligibility.json');pred=read(E/'results/predicted_readouts.json');cmp=read(E/'results/prediction_comparison.json');prefix=read(E/'results/prefix_predictions.json');pm=read(E/'results/prefix_metrics.json');figs=[]
 # Exact fractions, not pooled independent sample sizes.
 conds=list(CN);arr=np.array([[1-one(ms,condition=c,form=f,seed=s)['H3_accuracy']for f,s in HEADS]for c in conds])
 fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(arr,cmap='Reds',vmin=0,vmax=.3,aspect='auto')
 ax.set_xticks(range(4),[f'{FN[f]} / 种子{s}'for f,s in HEADS]);ax.set_yticks(range(8),[f"{CN[c]}  ({one(ms,condition=c,form='object',seed=0)['roots']}根)"for c in conds]);ax.set_xlabel('四个固定探针；两种形式 × 两个初始化')
 for (i,j),v in np.ndenumerate(arr):
  m=one(ms,condition=conds[i],form=HEADS[j][0],seed=HEADS[j][1]);ax.text(j,i,f"{m['H3_correct']}/{m['N']}",ha='center',va='center',color='white'if v>.18 else'#222',fontsize=12)
 ax.axhline(.5,color='#444',lw=1.4);ax.axhline(4.5,color='#444',lw=1.4)
 fig.colorbar(im,ax=ax,label='三个Holding中任一出错的端点比例',shrink=.8)
 ax.set_title('实际技能端点：单元数字为三个Holding同时正确数 / 全部端点数\n白色表示零错误，红色越深错误越多；横线仅分隔总计、无持物、抓取条件',fontsize=12)
 fig.text(.5,.025,'上表保留135个真实边界，包括失败开屉的3个端点。91个无持物端点全部正确，但不检验持物召回。\n抓取条件的错误与所有探针一致正确子集均另列；不同分支、边界及初始化不能增加独立根数。',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(R/'actual_endpoints.png',bbox_inches='tight');plt.close(fig);figs.append('actual_endpoints.png')
 # Prefix intervention, keeping original formal readout fixed.
 fig,axs=plt.subplots(2,2,figsize=(12,8));styles=[('actual','#9ba4ad','真实z读取'),('baseline_recursive','#df893c','原连续递推'),('replacement','#3b87bd','抓取后替换一次')]
 for s in [0,1]:
  for j,w in enumerate(['mixed','latent']):
   ax=axs[s,j]
   for k,(path,color,label)in enumerate(styles):
    vals=[one(pm,seed=s,world_model=w,step=t,path=path)[key]for t,key in [(3,'Holding_correct'),(3,'full9_correct'),(4,'Holding_correct'),(4,'full9_correct')]]
    bars=ax.bar(np.arange(4)+(k-1)*.24,vals,.23,color=color,label=label);ax.bar_label(bars,labels=[str(v)for v in vals],padding=2,fontsize=9)
   ax.set_xticks(range(4),['放下\nHolding','放下\n全部9事实','关屉\nHolding','关屉\n全部9事实']);ax.set_ylim(0,12.7);ax.set_yticks([0,3,6,9,11]);ax.set_ylabel('正确根数 / 同一11个完整根');ax.set_title(f'{WN[w]} / 种子{s}');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.92),ncol=3)
 fig.suptitle('一次真实前缀替换：抓取后注入真实z，再连续预测 放下→关屉\n四个固定世界模型与原正式Query；放下后不再注入观测',fontsize=13)
 fig.tight_layout(rect=[0,.07,1,.86]);fig.text(.5,.02,'灰色为实际观测对照，橙色为原四步递推，蓝色为唯一一次替换。Holding含3事实；全部9事实再含In与抽屉状态。\n本例In与抽屉均正确，故两指标计数相同；这只是后两步局部对照，不代表整条链或规划闭环成功。',ha='center',fontsize=10);fig.savefig(R/'prefix_replacement.png',bbox_inches='tight');plt.close(fig);figs.append('prefix_replacement.png')
 # Same fixed root and seed as protocol, no result-based example choice.
 ds=read(D/'data/combo/public/states.json');ar={r['id']:r for r in actual};example=[]
 fig,axs=plt.subplots(3,2,figsize=(17,10),gridspec_kw={'width_ratios':[1,2.35]})
 for row,(branch,slot,label)in enumerate([('reverse_then_lift',1,'② 空手关屉后'),('normal',2,'① 放下蓝物体后'),('normal',3,'① 放下后关屉')]):
  r=one(pred,root_id='D_identity_blue_red_00',seed=0,world_model='latent',path='recursive',branch=branch,slot=slot);o=one(pred,root_id=r['root_id'],seed=0,world_model='latent',path='one_step',branch=branch,slot=slot);a=ar[r['id']]
  axs[row,0].imshow(rgb(ds[a['after']]['path']));axs[row,0].axis('off');axs[row,0].set_title(label+'\n真实图像：左固定 / 右腕部相机')
  pr=one(prefix,root_id=r['root_id'],seed=0,world_model='latent',step=slot+1)if branch=='normal'else None
  vals=[r['original_actual_probability'],o['original_probability'],r['original_probability'],r['object_probability'],r['context_probability'],pr['replacement_probability'][:3]if pr else None]
  cells=[]
  for k,c in enumerate(['red','blue','pink']):
   cells.append([f'H({c})','假',*[f'{v[k]:.4f}'if v is not None else'未做'for v in vals]])
   example.append({'root_id':r['root_id'],'branch':branch,'step':slot+1,'fact':f'Holding({c})','truth':r['truth'][k],'original_actual':vals[0][k],'original_one_step':vals[1][k],'original_recursive':vals[2][k],'object_recursive':vals[3][k],'context_recursive':vals[4][k],'prefix_replacement':vals[5][k]if vals[5]is not None else None})
  ax=axs[row,1];ax.axis('off');tab=ax.table(cellText=cells,colLabels=['事实','标签','原头实际z','原头单步z','原头递推z','对象头递推z','上下文递推z','替换后原头'],cellLoc='center',loc='center',colWidths=[.14,.06,.13,.13,.13,.14,.14,.13]);tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,2.25)
  for k in range(3):
   for col,v in enumerate(vals):
    if v is None:tab[(k+1,col+2)].set_facecolor('#dddddd')
    elif(v[k]>=.5)!=r['truth'][k]:tab[(k+1,col+2)].set_facecolor('#f2b0a5')
  ax.set_title('同一实际后态标签；红底为概率≥0.5的误报，灰色未运行')
 fig.suptitle('固定示例：D_identity_blue_red_00 / αZ / 种子0\n真实端点均未持物；同一个递推z在原头与独立头之间存在明显分歧',fontsize=14);fig.tight_layout(rect=[0,.07,1,.91]);fig.text(.5,.025,'独立两头均读取同一保存的递推z；最后一列改为抓取后替换一次真实z，再用原Query连续读出。\n空手关屉单步已错，依协议未做前缀替换。图像用于解释真实状态，物理标签来自既有评分记录，不按图像外观改标签。',ha='center',fontsize=10);fig.savefig(R/'fixed_endpoint_cases.png',bbox_inches='tight');plt.close(fig);figs.append('fixed_endpoint_cases.png');save('fixed_example',example)
 # Derived presentation tables keep denominators explicit.
 pairs=[]
 for s in [0,1]:
  for w in ['mixed','latent']:
   for t in [3,4]:
    rr=[r for r in prefix if r['seed']==s and r['world_model']==w and r['step']==t];counts={'both_wrong':0,'recovered':0,'new_wrong':0,'both_correct':0}
    for r in rr:
     old=all((p>=.5)==y for p,y in zip(r['baseline_recursive_probability'][:3],r['truth'][:3]));new=all((p>=.5)==y for p,y in zip(r['replacement_probability'][:3],r['truth'][:3]));counts['both_correct'if old and new else'new_wrong'if old else'recovered'if new else'both_wrong']+=1
    pairs.append({'seed':s,'world_model':w,'step':t,'N':11,**counts})
 save('prefix_paired_changes',pairs)
 predict_table=[]
 for s in [0,1]:
  for w in ['mixed','latent']:
   for c in elig:
    if not elig[c]:continue
    item={'seed':s,'world_model':w,'condition':c}
    for path in ['one_step','recursive']:
     x=one(cmp,seed=s,world_model=w,path=path,condition=c,form='object');y=one(cmp,seed=s,world_model=w,path=path,condition=c,form='context');item['N']=x['N'];item['roots']=x['roots'];item[path+'_original_errors']=x['both_wrong']+x['original_only_wrong'];item[path+'_object_errors']=x['both_wrong']+x['probe_only_wrong'];item[path+'_context_errors']=y['both_wrong']+y['probe_only_wrong']
    predict_table.append(item)
 save('predicted_error_counts',predict_table)
 text='# 仅技能端点、探索性定位：Holding读出与一次前缀替换\n\n'
 text+='**本附录完成了四个冻结探针在D12全部135个实际技能端点的评估，并在正常链抓取后做一次真实z替换、连续预测放下→关屉。没有训练、采集或阈值变更。** 原完整C可靠性失败及`attribution_stop`保持封存；本附录是在新的有限授权下独立保存，不把端点子集视为通过旧门槛。\n\n'
 text+='四个探针在91个无持物端点上全部正确；44个抓取端点仍有错误。实际可靠的无持物条件下，原Query与两个独立探针读取同一预测z时明显分歧，因而不能把原Query错误直接等同于转移模块错误。一次真实前缀替换使四组关屉结果从3–5/11升到11/11；放下为10/11或11/11。这支持这段链对前缀输入的局部敏感性，不能单独证明转移模块是纯粹原因。\n\n'
 text+='## 评估元素、原因与边界\n\n'+table(['元素','固定做法','含义与原因'],[
 ['数据与独立单位','沿用D12：12根、135已观测技能边界，含失败根3次开屉结束；9个未到达步骤不补值','分支、帧、两个seed不是新增独立根；前缀对照只用11个完整根'],
 ['四个探针','对应对象token 64→32→1；对象+场景+对象均值192→32→1，各两seed，最终600更新检查点','本轮全部冻结，归一化仍来自AB训练；不以D表现选择头、seed或阈值。对象token已带编码器上下文'],
 ['标签与阈值','既有物理Holding标签：双侧承力手指接触且无外部支撑；阈值0.5','GT仅评分；未作为预测输入。H3指红、蓝、粉三个事实同时正确'],
 ['探索性条件','每个完整条件至少6根，四头分别H3≥95%、逐色FPR≤5%；有阳性时FNR≤10%、阳性至少2根','在新前向前冻结，但D此前已查看；这是有限描述性检查，不是独立验证或新项目验收'],
 ['仅预测可靠条件','按整个条件决定，不从条件中挑正确样本；保留两输入形式和匹配seed','此次通过的全为无持物条件，只能检验误报，不能证明持物召回或完整潜状态正确'],
 ['一致正确子集','额外标记四头均与实际标签一致的127/135边界','只作描述，主成绩仍135；不拿该子集替换抓取条件总体结果'],
 ['前缀替换','只在正常链抓取后注入一次真实z，再用同一冻结T连续预测place→close；仍用原正式Query','移除部分前缀输入偏差，测试后两步敏感性；放下后不注入，空手close不追加此对照']])
 text+='## 所有实际技能端点\n\n![实际端点](actual_endpoints.png)\n\n图中列为四个固定探针，行是所有端点或具体技能条件；数字为H3正确数/全体端点数。白色为零错误，红色越深错误比例越大。行名括号为独立根数，横线仅划分汇总和条件。开屉47个包含正常/两个身份分支的开屉及3次实际失败；放下22个来自两个分支的同11根。\n\n'
 rows=[]
 for c in CN:
  m=one(ms,condition=c,form='object',seed=0);q=one(ag,condition=c)
  rows.append([CN[c],f"{m['N']} / {m['roots']}",*[f"{one(ms,condition=c,form=f,seed=s)['H3_correct']}/{m['N']}"for f,s in HEADS],f"{q['agreed_N']}/{q['all_N']}（{q['agreed_roots']}根）",'仅汇总'if c=='all'else'限定无持物预测对照'if elig[c]else'不用于预测定位'])
 text+=table(['条件','端点 / 根','对象0','对象1','上下文0','上下文1','四头一致正确子集','使用范围'],rows)
 text+='四头在全部91个无持物端点上均无FP；全部135端点的差异来自持物条件，不能用较高总体准确率掩盖逐色召回问题。以下FP/FN是**单个颜色事实**计数，不等于H3错误端点数；同一端点可同时有FP和FN。\n\n'
 text+=table(['头/seed','颜色','FP / 负事实数','FN / 正事实数','阳性根数'],[[FN[m['form']]+'/'+str(m['seed']),c['color'],f"{c['FP']}/{c['FP']+c['TN']}",f"{c['FN']}/{c['FN']+c['TP']}",c['positive_roots']]for m in ms if m['condition']=='all'for c in m['colors']])
 text+='上下文在抓c条件改善了对象头的蓝色漏报，但在换抓d及开关开后抓d仍出现错误，不能从各条件挑不同头拼出“可靠探针”。逐条件/逐色的全部TP、FN、FP、TN、FPR、FNR和根数见[实际逐色表](actual_color_metrics.csv)；零阳性颜色的FNR为空，不当作0。\n\n'
 rows=[]
 for rid in sorted({r['root_id']for r in actual}):
  rr=[r for r in actual if r['root_id']==rid];rows.append([rid.removeprefix('D_identity_'),len(rr),*[f"{one(rootms,condition='all',form=f,seed=s,root_id=rid)['H3_correct']}/{len(rr)}"for f,s in HEADS],sum(r['all_probes_actual_correct']for r in rr)])
 text+=table(['根（D_identity_）','实际端点数','对象0','对象1','上下文0','上下文1','四头均正确端点'],rows)
 text+='`pink_blue_01`仅有3个失败开屉的实际终点，其3/3不表示该根完成任何链。完整逐根×条件表见[actual_root_metrics.csv](actual_root_metrics.csv)，每个错误的颜色、概率、动作和c/d身份见[actual_error_facts.csv](actual_error_facts.csv)。一致子集共有127边界，分布在12根；其中抓c仅18/22（9根）、放c后抓d为8/11、开关开后抓d为10/11。所有8个非一致边界保留，没有从主分母删除。\n\n'
 text+='## 同一预测z：原Query与独立探针的分歧\n\n只评估上述四个完整无持物条件，共91边界；匹配世界模型seed与探针seed，对两形式均报告。每格为“原Query / 对象头 / 上下文头”的H3错误端点数；这些条件物理标签全为未持有，因此所有错误都是FP。实际z下三类读出在这些端点均正确。\n\n'
 text+=table(['模型/seed','条件','端点 / 独立根','单步错误：原/对象/上下文','递推错误：原/对象/上下文'],[[WN[r['world_model']]+'/'+str(r['seed']),CN[r['condition']],f"{r['N']} / {r['roots']}",*[f"{r[path+'_original_errors']} / {r[path+'_object_errors']} / {r[path+'_context_errors']}"for path in ['one_step','recursive']]]for r in predict_table])
 text+='原头空手关屉单步错误仍很明显；独立两头在这11根的单步与递推均无误报。放下后递推的对象头seed0仍有2/22错误（mixed与αZ各自如此），上下文头为0/22；两条放下分支来自同11根，不算22个独立试验。除此之外，此次限定条件里独立两头均无误报。所有独立头错误端点也有原头错误；完整四格配对及同错事实数量见[prediction_comparison.csv](prediction_comparison.csv)，原始逐端点概率见[predicted_readouts.csv](predicted_readouts.csv)。\n\n'
 text+='**分歧保留，不投票判定哪个模块“是真因”。** 固定原Query在预测z上的读出可以错误，而其他固定读出仍得到正确的空手结论；这说明原Query与预测表示的配合值得关注。它不证明T预测了正确的完整状态：这里仅有无持物负例，探针也可能对预测分布偏向空手，且其完整C及部分抓取端点可靠性未通过。本附录不检查不可靠抓取条件的预测z、不据此宣布T正确或错误。\n\n![固定案例](fixed_endpoint_cases.png)\n\n每行对应同一个真实后态，左图依次是固定和腕部相机；右表分别为物理标签、原头对实际/单步/递推z的概率、两独立头对相同递推z的概率，以及唯一前缀替换后原头概率。红底是错误，灰底是未做；0.0000是四位小数舍入，不表示精确零。固定例的空手close，原递推H(red/blue)=0.9667/0.8883，而两独立头均低于0.033；正常放蓝后，原递推H(blue)=0.5674，对象/上下文分别0.0222/0.0011。所有颜色和原始精度见[fixed_example.csv](fixed_example.csv)。\n\n'
 text+='## 一次真实前缀替换，再连续预测两步\n\n![前缀替换](prefix_replacement.png)\n\n每面板固定模型和seed；横轴为放下、后续关屉的Holding或全部9事实，纵轴为同一11个完整根的正确数。灰色实际读取作为对照，橙色原连续递推，蓝色仅在抓取后替换一次真实z。Holding包括3个颜色事实，全部9事实还包括3个In及3类抽屉状态（后者取最大概率）；本对照In与抽屉全对，因此两指标恰好相同。\n\n'
 rows=[]
 for s in [0,1]:
  for w in ['mixed','latent']:
   for t in [3,4]:
    mm=[one(pm,seed=s,world_model=w,step=t,path=path)for path in ['actual','baseline_recursive','replacement']];pp=one(pairs,seed=s,world_model=w,step=t)
    rows.append([WN[w]+'/'+str(s),'放下'if t==3 else'后续关屉',*[f"{m['Holding_correct']}/11；{m['full9_correct']}/11"for m in mm],f"{pp['recovered']} / {pp['new_wrong']} / {pp['both_wrong']}"])
 text+=table(['模型/seed','步骤','实际：H3；9事实','原递推：H3；9事实','替换：H3；9事实','修复 / 新错 / 仍错根数'],rows)
 text+='所有条件均没有新增错误；mixed种子0在放下后仍有`D_identity_red_blue_00`一根误报H(red)=0.5717，后续关屉恢复正确。两步联合全对为mixed种子0的10/11，其余三组11/11。没有继续修补这一根。\n\n'
 text+='放下是替换后的第一步，与既有真实前态单步预测概率最大差5.96×10⁻⁷，低于预置1×10⁻⁵；原始z最大差见复核记录。第二步确实使用第一步预测z，没有读取放下后的真实z。88条状态预测来自11根×4固定模型×2步；每个根/模型仅1次注入。该对照说明前缀产生的潜输入偏差对后两步结果有影响，**不能区分前缀T偏差、表示分布漂移、Query敏感性等贡献，也不是整条四步链或TS-LTL闭环成功率。** 空手close单步已错，未追加替换或修正。\n\n'
 text+='## 保留的结论与交付证据\n\n原完整C gate仍失败，旧C缺蓝色持物阳性根的限制仍在；D的正例不能事后补成旧C覆盖。原正式模型、阈值、训练流、旧实验封存、失败根和未到达步骤均未改变。此次端点探索没有采用新正式头，也未完成原指南的端到端验收。\n\n'
 v=read(E/'results/verification.json');audit=read(E/'results/run_audit.json')
 text+=f"独立复核重新计算{v['actual_confusion_cells']}个实际混淆计数、728条预测记录的原值对应、64组配对结果和88条前缀记录；新运行前冻结引用以及原Holding 64文件、D12 2787文件均哈希未变。源代码也核查了唯一注入位置。预算为0训练更新、0新物理调用、0参数修改。没有重跑模型来挑结果。\n\n"
 text+=table(['交付','内容'],[
 ['[原完整C报告](../holding_readout_diagnostic_20261018/REPORT.md)','训练设计、完整C失败、覆盖审计、真实GPU性能；保持原样'],
 [f'[冻结协议]({E}/configs/protocol.json) / [输入哈希]({E}/configs/frozen.json)','端点限制、四头共同条件、固定示例及唯一前缀操作'],
 ['[全部实际边界](actual_endpoints.csv) / [一致子集](actual_agreement_subset.csv)','135个端点、完整四头概率与标签；一致子集仅额外标记'],
 ['[逐条件统计](actual_condition_metrics.csv) / [逐色](actual_color_metrics.csv) / [逐根](actual_root_metrics.csv)','总体与条件分母、TP/FN/FP/TN、颜色与根数'],
 ['[预测比较](prediction_comparison.csv) / [概率明细](predicted_readouts.csv)','所有限定条件、两模型×两seed×两路径×两形式，保留分歧'],
 ['[前缀逐根概率](prefix_predictions.csv) / [指标](prefix_metrics.csv) / [配对变化](prefix_paired_changes.csv)','原Query全部9事实，实际、原递推、替换后输出'],
 [f'[复核]({E}/results/verification.json) / [运行审计]({E}/results/run_audit.json) / [封存]({E}/configs/delivery_manifest.json)','哈希、计数、输入复现、预算及独立交付清单']])
 (R/'APPENDIX.md').write_text(text);save('report_assets',{'figures':figs,'report':'APPENDIX.md','no_new_inference':True});shutil.copy2(__file__,E/'source/report.py');print('ENDPOINT_REPORT_READY',len(figs),'figures',R)
if __name__=='__main__':main()
