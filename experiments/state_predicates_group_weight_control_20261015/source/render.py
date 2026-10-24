from common_control import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.unicode_minus':False,'figure.dpi':160})
ev=read(E/'results/evaluation.json');families=['start','uniform','equal_groups'];labels=['1500步起点','追加500步原权重','追加500步组等权'];colors=['#9198a3','#347bc0','#d9872c'];markers=['o','s','^'];models=[f'{a}_seed{s}'for a in families for s in [0,1,2]]
fig,axs=plt.subplots(1,2,figsize=(13,5))
for ax,groups,titles,ylabel in [(axs[0],['C_old','C_state','C_drawer','C_combined'],['旧C24','状态C18','抽屉C12','合并C54'],'全7事实正确率（%）'),(axs[1],['AB_new_open','C_drawer'],['新增五条训练例','抽屉C开闭双侧'],'全对调用或双侧全对根数')]:
 for j,(a,lab,col)in enumerate(zip(families,labels,colors)):
  values=[]
  for g in groups:
   if ax is axs[1]and g=='C_drawer':values.append([ev['paired_summary'][f'{a}_seed{s}'][g]['both_all7']for s in [0,1,2]])
   else:values.append([ev['metrics'][f'{a}_seed{s}'][g]['all']['all7']*(100/ev['groups'][g]if ax is axs[0]else 1)for s in [0,1,2]])
  x=np.arange(len(groups))+(j-1)*.25;ax.bar(x,np.mean(values,axis=1),width=.23,color=col,alpha=.7,label=lab)
  for seed,m in enumerate(markers):ax.scatter(x+(seed-1)*.035,[v[seed]for v in values],color=col,marker=m,edgecolor='white',zorder=5,s=35)
 ax.set_xticks(np.arange(len(groups)),titles);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.15);ax.set_ylim(0 if ax is axs[0] else -.2,106 if ax is axs[0] else 6.6)
axs[0].legend(loc='lower left',fontsize=9);axs[1].text(0,5.65,'分母5',ha='center');axs[1].text(1,6.2,'分母6',ha='center');fig.suptitle('组等权有额外开发收益，但两个种子仍有未恢复调用',fontsize=17)
fig.text(.5,.02,'灰=原起点，蓝=追加原权重，橙=追加组等权；柱为三种子均值，圆/方/三角=种子0/1/2。两组同起点、重置优化器、全批137、各500步。',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.08,1,.93]);fig.savefig(E/'figures/development_results.png');plt.close(fig)

roots=[r for r in read(E/'results/root_contrasts.json')if r['group']=='C_drawer'];grid=[];text=[]
for m in models:
 row=[];txt=[]
 for r in roots:
  b=r['models'][m]['side_all7_correct'];code=int(b[0])+2*int(b[1]);row.append(code);txt.append(['两侧错','仅关闭对','仅打开对','两侧对'][code])
 grid.append(row);text.append(txt)
fig,ax=plt.subplots(figsize=(12,6.4));ax.imshow(grid,vmin=0,vmax=3,cmap=ListedColormap(['#e2e4e8','#78a9df','#efb865','#73c2a3']),aspect='auto');ax.set_xticks(np.arange(6),[r['root_id'].replace('C_drawer_factor_','')for r in roots]);ax.set_yticks(np.arange(9),[labels[families.index(m.rsplit('_seed',1)[0])]+' / 种子'+m[-1]for m in models])
for i in range(9):
 for j in range(6):ax.text(j,i,text[i][j],ha='center',va='center',fontsize=10)
ax.set(xlabel='既有抽屉C根（blue_0关闭侧真实读出错误仍保留）',ylabel='固定训练条件和种子',title='逐根核查：不以平均值掩盖失败或关闭侧退化');fig.text(.5,.02,'灰=两侧错，蓝=仅关闭对，橙=仅打开对，绿=两侧均全7正确；每格两次同根调用，UNKNOWN计错。C不是独立测试。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(E/'figures/root_pairs.png');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(13,5));gs=['C_old','C_state','C_drawer']
for ax,key,label in [(axs[0],'all384_MAE','相对真实后续编码的384维MAE'),(axs[1],'abs_gt1_fraction','预测坐标绝对值>1比例（%）')]:
 for j,(a,lab,col)in enumerate(zip(families,labels,colors)):
  vals=np.array([[ev['latent_metrics'][f'{a}_seed{s}'][g][key]for s in [0,1,2]]for g in gs]);vals*=100 if key.endswith('fraction') else 1;x=np.arange(3)+(j-1)*.25;ax.bar(x,vals.mean(1),width=.23,color=col,alpha=.7,label=lab)
  for seed,m in enumerate(markers):ax.scatter(x+(seed-1)*.035,vals[:,seed],color=col,marker=m,edgecolor='white',s=35,zorder=5)
 ax.set_xticks(np.arange(3),['旧C24','状态C18','抽屉C12']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.15)
axs[0].legend(fontsize=9);fig.suptitle('符号改善不等于隐状态修复：误差与越界单独报告',fontsize=17);fig.text(.5,.02,'灰起点、蓝原权重、橙组等权；柱三种子均值，圆/方/三角种子0/1/2。真实GRU编码有界，残差预测未裁剪；越界比例不是调用错误率。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.08,1,.93]);fig.savefig(E/'figures/latent_quality.png');plt.close(fig)

fig,axs=plt.subplots(2,3,figsize=(13,7))
for seed in [0,1,2]:
 for a,lab,col in zip(families[1:],labels[1:],colors[1:]):
  rows=read(E/'runs'/f'{a}_seed{seed}'/'training_observations.json');x=[r['update']for r in rows]
  axs[0,seed].plot(x,[r['new5_all7']for r in rows],'-o',label=lab,color=col);axs[1,seed].plot(x,[r['old132_all7']for r in rows],'-o',color=col)
 axs[0,seed].set(title=f'种子{seed}：新增五条',ylim=(-.2,5.5),ylabel='全7事实正确数 / 5');axs[1,seed].set(title=f'种子{seed}：原132条',ylim=(124,133),ylabel='全7事实正确数 / 132',xlabel='追加优化步数（固定到500）');axs[0,seed].legend(fontsize=8)
 for ax in axs[:,seed]:ax.set_xticks([0,100,250,500]);ax.grid(alpha=.15)
fig.text(.5,.02,'蓝=原权重，橙=组等权；点仅在预定0/25/100/250/500步记录，连线不表示逐步结果。下排纵轴从124开始以显示暂时退化；最终步始终固定。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(E/'figures/training_fit.png');plt.close(fig)

err=read(E/'results/single_replacement_diagnostic.json');first=err['first_error'];rows=read(E/'results/call_details.json')
if first:
 series=[next(r for r in rows if r['model']==name and r['pair_id']==first['pair_id'])for name in ['start_seed0','uniform_seed0','equal_groups_seed0','real_next_observation']];fig,ax=plt.subplots(figsize=(12,5));x=np.arange(7)
 for j,(r,lab,col)in enumerate(zip(series,labels+['真实后续编码＋同一读出头'],colors+['#30966e'])):ax.bar(x+(j-1.5)*.18,r['probabilities'],width=.17,label=lab,color=col)
 ax.scatter(x,first['truth_after'],marker='D',color='#222',s=24,label='真实事实0/1',zorder=6);ax.axhline(.5,color='#555',ls='--',label='固定阈值0.5');ax.set_xticks(x,['持红','持蓝','持粉','红入屉','蓝入屉','粉入屉','屉开']);ax.set(ylabel='事实为真的概率',ylim=(-.03,1.05),title='预定种子0首错：已持红、请求抓粉，仍误报目标持有');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,fontsize=9);fig.text(.5,.02,'C_drawer_factor_pink_1 打开侧：当前与真实后续读出均正确；组等权持粉0.519486仍超过0.5，与持红冲突。仅离线替换后继，不在线修复。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.15,1,1]);fig.savefig(E/'figures/first_error.png');plt.close(fig)
save(E/'results/figure_manifest.json',{'files':[str(p)for p in sorted((E/'figures').glob('*.png'))],'labels_chinese':True,'visually_inspected':False});print('FIGURES_WRITTEN',flush=True)
