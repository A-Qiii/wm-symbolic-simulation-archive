from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
guard();verify()
root=read(S/'data/roots_manifest.json')[0];assert root['template']=='recovery_seen6';h=COLORS.index(root['held_color']);t=COLORS.index(root['matched_color']);colors={'baseline':'#4477aa','geometry':'#ee7733'}
for mode in ['current','free_h2']:
 fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
 for branch in [1,2]:
  for j,(index,label)in enumerate([(t,'Held('+root['matched_color']+')'),(3+h,'InDrawer('+root['held_color']+')')]):
   ax=axs[branch-1,j];actual=None
   for seed in [0,1]:
    for arm in ARMS:
     z=dict(np.load(S/f'results/predictions_{arm}_seed{seed}.npz'));ix=np.flatnonzero((z['root_id']==root['root_id'])&(z['branch']==branch)&((z['mode']==mode)|((z['mode']=='current')&(z['step']==2)&(mode=='free_h2'))));ix=ix[np.argsort(z['step'][ix])];pp=rel(z['prob'][ix]);yy=truth(z['cat'][ix]);xx=z['step'][ix];actual=(xx,yy[:,index])
     ax.plot(xx,pp[:,index],'-o'if seed==0 else'--s',color=colors[arm],label=f'{arm} seed{seed}',alpha=.85)
   ax.step(actual[0],actual[1],where='mid',color='black',lw=2,label='Actual relation',alpha=.75);ax.axhline(.5,color='gray',lw=.8,ls=':');ax.set_ylim(-.06,1.06);ax.set_xticks(range(7));ax.set_title(('Nominal'if branch==1 else'First target grasp missed')+' / '+label);ax.set_xlabel('Task boundary h');ax.set_ylabel('Relation probability / actual 0 or 1');ax.grid(alpha=.15)
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=9);fig.suptitle(root['root_id']+' / '+mode,fontsize=11);fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(S/f'reports/example_{mode}.png',dpi=160);fig.savefig(S/f'reports/example_{mode}.svg');plt.close(fig)
p=S/'reports/REPORT.html';text=p.read_text();block=f'''<h2>固定案例：同一目标抓取成功或落空后的关系</h2><p>案例按冻结根清单的第一项选定：{root['root_id']}，不是按模型成绩挑选。H为{root['held_color']}，T为{root['matched_color']}；h1放置H，h2首次抓取T，h3尝试放置，h4再次抓取T，h5再次放置，h6关闭。该单例仅帮助解释读图，不代替全部布局族成绩。</p><img src="example_current.png"><p>此图为每步真实观测后的读出。两行是名义执行与首次目标抓取落空分支；左列为T的Held，右列为非目标H的InDrawer。黑色阶梯为实际0/1关系；蓝线baseline、橙线geometry，实线圆点seed0、虚线方点seed1。横轴任务边界h，纵轴关系概率；灰色0.5线为固定判别阈值。曲线应随黑线改变，不能只用总体正确率掩盖短暂遗忘。</p><img src="example_free_h2.png"><p>同一案例从真实h2出发的后四步推演。h2的点是实际观测后状态，h3–h6的彩线完全不读取未来观测；黑线仅用于事后评分。它与上一图的差异区分观测纠正能力和无未来观测的转移预测。</p>'''
text=text.replace('<h2>相同初态与动作集合的新顺序</h2>',block+'<h2>相同初态与动作集合的新顺序</h2>');p.write_text(text)
save(S/'results/example_selection.json',{'utc':now(),'rule':'First entry of pre-rollout frozen roots_manifest; selected in code before model evaluation','root_id':root['root_id'],'models_or_cases_not_selected_by_performance':True})
