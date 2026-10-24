from training_core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';FP=FontProperties(fname=FONT)
ARMS=[('original132','#64748b','原132（背景）'),('closed137','#2474b5','132+5关闭'),('open137','#d97706','132+5打开')];SETS=[('C_old','旧C：24调用 / 6根'),('C_state','状态依赖C：18调用 / 6根'),('C_drawer','抽屉因素C：12调用 / 6根')]
def figsave(fig,name,footer):
 fig.text(.5,.015,footer,ha='center',fontproperties=FP,fontsize=10);fig.tight_layout(rect=[0,.055,1,.94]);fig.savefig(E/'figures'/name,dpi=160);plt.close(fig)
def main():
 q,df=dataset_frozen();r=read(E/'results/evaluation.json');fig,axes=plt.subplots(1,2,figsize=(14,5.8));ax=axes[0];x=np.arange(3)
 for j,(arm,col,label)in enumerate(ARMS):
  yy=np.array([[r['metrics'][f'{arm}_seed{s}'][g]['all']['all7_rate']*100 for g,_ in SETS]for s in [0,1,2]]);ax.bar(x+(j-1)*.23,yy.mean(0),width=.22,color=col,alpha=.75,label=label)
  for s,marker in enumerate(['o','s','^']):ax.scatter(x+(j-1)*.23+(s-1)*.035,yy[s],c=col,marker=marker,s=35,edgecolors='white',zorder=4)
 ax.set_xticks(x,[s[1]for s in SETS],fontproperties=FP,fontsize=9);ax.set_ylim(0,108);ax.set_ylabel('全七事实已知且正确的调用比例（%）',fontproperties=FP);ax.set_title('三组开发材料分别报告',fontproperties=FP);ax.legend(prop=FP,loc='lower left',fontsize=8);ax.grid(axis='y',alpha=.2);ax=axes[1];x=np.arange(2)
 for j,(arm,col,label)in enumerate(ARMS):
  yy=np.array([[r['paired_summary'][f'{arm}_seed{s}'][g]['both_all7']for g in ['C_state','C_drawer']]for s in [0,1,2]]);ax.bar(x+(j-1)*.23,yy.mean(0),width=.22,color=col,alpha=.75)
  for s,marker in enumerate(['o','s','^']):ax.scatter(x+(j-1)*.23+(s-1)*.035,yy[s],c=col,marker=marker,s=35,edgecolors='white',zorder=4)
 ax.set_xticks(x,['状态依赖C：成功 / 拒绝配对','抽屉因素C：关闭 / 打开配对'],fontproperties=FP,fontsize=9);ax.set_ylim(0,6.6);ax.set_yticks(range(7));ax.set_ylabel('双侧全七事实均正确的根数（每组共6根）',fontproperties=FP);ax.set_title('不允许只修好配对的一侧',fontproperties=FP);ax.grid(axis='y',alpha=.2);fig.suptitle('事后探索：5个匹配配对，两组137对、等更新预算',fontproperties=FP,fontsize=15);figsave(fig,'development_and_pairs.png','柱=3种子均值；圆/方/三角=种子0/1/2。灰=原132背景，蓝=加关闭样本，橙=加打开样本；两组137均固定1500更新。训练子集事后选取，颜色2/2/1；全部C为开发评估，抽屉C已用于定位。')
 fig,axes=plt.subplots(1,2,figsize=(13,5.6));x=np.arange(3)
 for ax,key,title in [(axes[0],'all384_MAE','潜变量与真实后续编码的平均绝对误差'),(axes[1],'abs_gt1_fraction','潜变量坐标越出[-1,1]的比例')]:
  for j,(arm,col,label)in enumerate(ARMS):
   yy=np.array([[r['latent_metrics'][f'{arm}_seed{s}'][g][key]for g,_ in SETS]for s in [0,1,2]])*(100 if key.endswith('fraction')else 1);ax.bar(x+(j-1)*.23,yy.mean(0),width=.22,color=col,alpha=.75,label=label)
   for s,marker in enumerate(['o','s','^']):ax.scatter(x+(j-1)*.23+(s-1)*.035,yy[s],c=col,marker=marker,s=30,edgecolors='white',zorder=4)
  ax.set_xticks(x,['旧C','状态依赖C','抽屉因素C'],fontproperties=FP);ax.set_title(title,fontproperties=FP);ax.set_ylabel('越界坐标百分比（%）'if key.endswith('fraction')else'MAE（未裁剪的潜变量坐标）',fontproperties=FP);ax.grid(axis='y',alpha=.2)
 axes[0].legend(prop=FP);fig.suptitle('符号变化与潜变量质量分开判断',fontproperties=FP,fontsize=15);figsave(fig,'latent_quality.png','每次预测384坐标=3物体×96+场景96；与冻结编码器的真实后续编码比较。越界指|坐标|>1，不等同于该坐标必错；无裁剪。柱和标记含义同上图。')
 fig,axes=plt.subplots(1,3,figsize=(15,4.8));x=np.arange(1,1501)
 for arm,col,label in ARMS[1:]:
  curves=[[json.loads(line)for line in(E/'runs'/f'{arm}_seed{s}'/'curve.jsonl').read_text().splitlines()]for s in [0,1,2]]
  for ax,key,title in zip(axes,['BCE','weighted_latent','total_loss'],['三组谓词加权BCE','潜变量项（已乘0.05）','实际总损失']):
   yy=np.array([[z[key]for z in c]for c in curves]);kernel=np.ones(50)/50
   for s in [0,1,2]:ax.plot(x[49:],np.convolve(yy[s],kernel,mode='valid'),color=col,alpha=.18,lw=.8)
   ax.plot(x[49:],np.convolve(yy.mean(0),kernel,mode='valid'),color=col,label=label);ax.set_title(title,fontproperties=FP);ax.set_xlabel('优化器更新次数',fontproperties=FP);ax.set_ylabel('损失值',fontproperties=FP);ax.grid(alpha=.2)
 axes[0].legend(prop=FP);fig.suptitle('训练预算固定：每组3种子，各1500次更新',fontproperties=FP,fontsize=15);figsave(fig,'training_curves.png','显示50次更新的后向滑动均值；粗线=3种子均值，淡线=每个种子。蓝/橙仅表示新增样本关闭/打开；原始每步损失保存在runs/*/curve.jsonl。')
 # Matched-root full7 restoration: each cell encodes both branches, not a selected single call.
 cons=read(E/'results/root_contrasts.json');rr=[c for c in cons if c['group']=='C_drawer'];names=[f'{a}_seed{s}'for a,_,_ in ARMS for s in [0,1,2]];arr=np.array([[int(c['models'][name]['side_all7_correct'][0])+2*int(c['models'][name]['side_all7_correct'][1])for c in rr]for name in names]);from matplotlib.colors import ListedColormap,BoundaryNorm
 colors=['#e5e7eb','#60a5fa','#fb923c','#16a34a'];fig,ax=plt.subplots(figsize=(13,6));ax.imshow(arr,cmap=ListedColormap(colors),norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),aspect='auto');ax.set_yticks(range(9),names);ax.set_xticks(range(6),[c['root_id'].replace('C_drawer_factor_','')for c in rr]);labels=['两侧都错','仅闭侧对','仅开侧对','双侧都对']
 for i in range(9):
  for j in range(6):ax.text(j,i,labels[arr[i,j]],ha='center',va='center',fontproperties=FP,fontsize=10)
 ax.set_xlabel('既有抽屉因素C根（均为开发诊断）',fontproperties=FP);ax.set_ylabel('训练组与固定种子',fontproperties=FP);fig.suptitle('逐根检查是否真正恢复开闭两侧',fontproperties=FP,fontsize=15);figsave(fig,'drawer_root_restoration.png','灰=两侧都错，蓝=仅关闭侧正确，橙=仅打开侧正确，绿=双侧全7正确。每格代表同一根两次相关调用；UNKNOWN计错。blue_0闭侧读出错误始终保留。')
 save(E/'results/figure_manifest.json',{'files':[str(E/'figures'/s)for s in ['development_and_pairs.png','latent_quality.png','training_curves.png','drawer_root_restoration.png']],'all_group_and_series_definitions_in_figures':True});print('FIGURES_WRITTEN',flush=True)
if __name__=='__main__':main()
