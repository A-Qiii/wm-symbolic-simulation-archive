from common_diag import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.unicode_minus':False,'figure.dpi':160})
ARMS=['five_only','joint_original','joint_equal_groups'];LABELS=['仅五条','联合原权重','联合组等权'];C=['#dc8a28','#3076bd','#299372']
hist={a:read(E/'runs'/a/'evaluations.json')for a in ARMS};short=['A红0','B红1','A蓝0','B蓝1','A粉0']
fig,axs=plt.subplots(1,2,figsize=(13,4.8))
for ax,g,title in zip(axs,['five','old132'],['新增五条打开拒绝条件（训练样本）','原132条训练样本（回归检查）']):
 for a,lab,c in zip(ARMS,LABELS,C):ax.plot([r['update']for r in hist[a]],[100*r['groups'][g]['all7']/r['groups'][g]['calls']for r in hist[a]],'-o',color=c,label=lab,alpha=.85)
 ax.set(title=title,xlabel='临时诊断更新次数',ylabel='全7事实正确率（%）',ylim=(-4,106));ax.set_xticks([0,25,100,250,500]);ax.grid(alpha=.18);ax.legend()
fig.suptitle('能拟合五条；联合优化能保住原任务，组等权更早达到全对',fontsize=17)
fig.text(.5,.02,'三组均从同一失败种子0权重开始，固定500步；点是预定记录时刻，线仅连接观测，不表示区间内每步全对。仅AB训练内诊断。',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.08,1,.92]);fig.savefig(E/'figures/fit_and_regression.png');plt.close(fig)

fig,ax=plt.subplots(figsize=(12,5));x=np.arange(5);series=[('失败起点',hist['five_only'][0]['groups']['five']['probabilities'],'#8b9099')]+[(lab,hist[a][-1]['groups']['five']['probabilities'],c)for a,lab,c in zip(ARMS,LABELS,C)]
for j,(lab,p,c)in enumerate(series):ax.bar(x+(j-1.5)*.19,[p[i][t]for i,t in enumerate([0,0,1,1,2])],width=.18,color=c,label=lab)
ax.axhline(.5,ls='--',color='#34495e',label='固定阈值0.5');ax.set_xticks(x,short);ax.set(ylabel='预测目标已持有的概率',xlabel='五个固定AB准备条件（字色名指抓取目标）',ylim=(0,1.03),title='第500步：两个联合对照均消除错误，组等权的目标误报概率更低');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3);ax.grid(axis='y',alpha=.15)
fig.text(.5,.015,'真实状态均为目标未持有（0）、仍持另一物体。灰=失败起点，橙=仅五条，蓝=联合原权重，绿=联合组等权。无C/D预测。',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(E/'figures/target_probabilities.png');plt.close(fig)

g=read(E/'results/gradient_audit.json');fig,axs=plt.subplots(1,2,figsize=(13,5))
cs=[g['cosines'][i]['cosine']for i in [0,1,2,3]];ylab=['五条符号 / 原132符号','五条符号 / 五条隐状态','五条总目标 / 原132总目标','联合符号 / 联合隐状态']
axs[0].barh(range(4),cs,color=['#cf625e'if v<0 else'#299372'for v in cs]);axs[0].set_yticks(range(4),ylab);axs[0].invert_yaxis();axs[0].axvline(0,color='#888');axs[0].set(xlim=(-1,1),xlabel='参数梯度余弦（−1反向，+1同向）',title='失败起点存在组间目标冲突')
for i,v in enumerate(cs):axs[0].text(v+(0.02 if v>=0 else-.02),i,f'{v:.3f}',ha='left'if v>=0 else'right',va='center',fontsize=10)
norm=g['group_components'];vals=[[norm['old132']['BCE']['gradient_L2']*132/137,norm['five']['BCE']['gradient_L2']*5/137],[norm['old132']['BCE']['gradient_L2']*.5,norm['five']['BCE']['gradient_L2']*.5]]
for j,(lab,c)in enumerate([('原132贡献','#3076bd'),('五条贡献','#dc8a28')]):axs[1].bar(np.arange(2)+(j-.5)*.3,[v[j]for v in vals],width=.28,label=lab,color=c)
axs[1].set_xticks([0,1],['原组占比132:5','组等权1:1']);axs[1].set(ylabel='加上组权重后的符号梯度L2',title='五条的目标权重可改变优化方向');axs[1].legend();axs[1].grid(axis='y',alpha=.15)
fig.text(.5,.02,'左红=方向冲突，绿=同向；右柱是分量范数，不能直接相加为总范数。均在原失败权重计算；不是AdamW实际更新贡献。',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(E/'figures/gradient_evidence.png');plt.close(fig)

rows=read(E/'results/input_information_audit.json')['rows'];fig,ax=plt.subplots(figsize=(12,5))
for j,(key,lab,c)in enumerate([('target_h96_RMS','目标对象h','#3076bd'),('global_s96_RMS','全局s','#dc8a28'),('held_other_h96_RMS','另一物体h（目标更新不直接读取）','#299372')]):ax.bar(np.arange(5)+(j-1)*.24,[r[key]for r in rows],width=.22,label=lab,color=c)
ax.set_xticks(np.arange(5),short);ax.set_yscale('log');ax.set(ylabel='隐状态差异RMS（对数轴）',xlabel='各打开拒绝例，对比AB内最近的同动作成功例',title='另一物体的持有信息可读出，但目标更新仅间接接收全局上下文');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3);ax.grid(axis='y',alpha=.16)
fig.text(.5,.02,'每组蓝/橙/绿对应不同输入分量；RMS越大表示数值差异越大，不直接代表可分类性。目标输入没有完全相同冲突，不能据此认定结构无法表达。',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.10,1,1]);fig.savefig(E/'figures/input_information.png');plt.close(fig)
save(E/'results/figure_manifest.json',{'files':[str(p)for p in sorted((E/'figures').glob('*.png'))],'all_labels_chinese':True,'requires_visual_inspection':True})
print('FOUR_FIGURES_WRITTEN',flush=True)
