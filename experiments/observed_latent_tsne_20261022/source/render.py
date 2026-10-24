from pathlib import Path
import os,json,csv,collections,datetime,warnings
T=Path(__file__).resolve().parent.parent;os.environ['MPLCONFIGDIR']=str(T/'cache/matplotlib');os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np,matplotlib
matplotlib.use('Agg')
from fontTools.ttLib import TTCollection
from matplotlib import font_manager as fm
fontsource=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');fontout=T/'cache/NotoSansCJKSC-Regular.otf'
if not fontout.exists():
 tc=TTCollection(fontsource,lazy=True);face=next(f for f in tc.fonts if f['name'].getDebugName(1)=='Noto Sans CJK SC');face.save(fontout);tc.close()
fm.fontManager.addfont(str(fontout));fontname=fm.FontProperties(fname=str(fontout)).get_name();matplotlib.rcParams.update({'font.family':[fontname,'DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def rows(p):
 with p.open()as f:return list(csv.DictReader(f))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
statecols={0:'#9AA5B1',1:'#2C7BB6',2:'#C88719'};drawercols={0:'#248A69',1:'#C88719',2:'#2C7BB6'};splitmarkers={'development':'o','combination_test':'^'};figurestats=[]
for seed in [0,1]:
 rr=rows(T/f'results/coordinates_seed{seed}.csv');ss=rows(T/f'results/scene_coordinates_seed{seed}.csv');xy=np.array([[float(r['tsne_x']),float(r['tsne_y'])]for r in rr]);sxy=np.array([[float(r['tsne_x']),float(r['tsne_y'])]for r in ss]);assert len(rr)==2880 and len(ss)==960;fig,axes=plt.subplots(2,2,figsize=(18,13));fig.suptitle(f'最新868训练模型｜四份观测记忆的 t-SNE（模型种子{seed}）',fontsize=22,fontweight='bold',y=.98);fig.text(.5,.938,'红、蓝、紫物体共享同一联合嵌入；环境单独嵌入，坐标不可与物体比较',ha='center',fontsize=13,color='#415466')
 mins=xy.min(0);maxs=xy.max(0);pad=(maxs-mins)*.06;global_limits=[mins-pad,maxs+pad];report={'model_seed':seed,'object_points':2880,'scene_points':960,'object_errors':[]}
 for ob,(ax,col,zh,headcolor)in enumerate(zip(axes.flatten()[:3],['red','blue','pink'],['红块','蓝块','紫块（pink）'],['#B34545','#2769A2','#9863A6'])):
  ix=np.array([r['object']==col for r in rr]);ids=np.flatnonzero(ix)
  for split,marker in splitmarkers.items():
   jj=np.array([i for i in ids if rr[i]['split']==split]);states=[2 if int(rr[i]['true_Held'])else 1 if int(rr[i]['true_In'])else 0 for i in jj];ax.scatter(xy[jj,0],xy[jj,1],s=14 if split=='development'else 18,c=[statecols[k]for k in states],marker=marker,alpha=.72,edgecolors='none',zorder=2)
  errors=np.array([i for i in ids if rr[i]['In_error_type']in ['FP','FN']],int);heldonly=np.array([i for i in ids if int(rr[i]['pred_Held'])!=int(rr[i]['true_Held'])and rr[i]['In_error_type']not in ['FP','FN']],int)
  if len(errors):ax.scatter(xy[errors,0],xy[errors,1],s=66,facecolors='none',edgecolors='#D33D59',linewidths=1.1,zorder=4)
  if len(heldonly):ax.scatter(xy[heldonly,0],xy[heldonly,1],s=35,c='#202830',marker='x',linewidths=1.2,zorder=5)
  fp=sum(rr[i]['In_error_type']=='FP'for i in ids);fn=sum(rr[i]['In_error_type']=='FN'for i in ids);ax.set_title(f'{"ABC"[ob]}  {zh}记忆 h（128维）\n开发480点 + 留出480点｜In误报 {fp}、漏报 {fn}',loc='left',fontsize=13,color=headcolor,fontweight='bold');ax.set_xlim(global_limits[0][0],global_limits[1][0]);ax.set_ylim(global_limits[0][1],global_limits[1][1]);ax.set_xlabel('物体联合 t-SNE 维度1（任意单位）');ax.set_ylabel('物体联合 t-SNE 维度2（任意单位）');report['object_errors'].append({'object':col,'points':960,'In_FP':fp,'In_FN':fn,'Held_only_error':len(heldonly)})
 ax=axes.flatten()[3]
 for split,marker in splitmarkers.items():
  jj=np.array([i for i,r in enumerate(ss)if r['split']==split]);ax.scatter(sxy[jj,0],sxy[jj,1],s=14 if split=='development'else 18,c=[drawercols[int(ss[i]['drawer_true'])]for i in jj],marker=marker,alpha=.72,edgecolors='none',zorder=2)
 err=np.array([i for i,r in enumerate(ss)if int(r['drawer_error'])],int)
 if len(err):ax.scatter(sxy[err,0],sxy[err,1],s=72,facecolors='none',edgecolors='#D33D59',linewidths=1.2,zorder=4)
 ax.set_title(f'D  环境记忆 c（128维；独立嵌入）\n开发480点 + 留出480点｜抽屉分类错误 {len(err)}',loc='left',fontsize=13,color='#415466',fontweight='bold');ax.set_xlabel('环境独立 t-SNE 维度1（任意单位）');ax.set_ylabel('环境独立 t-SNE 维度2（任意单位）');report['scene_errors']=len(err)
 for ax in axes.flatten():
  ax.grid(alpha=.14,linewidth=.6);ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=9)
 handles=[Patch(color=statecols[1],label='物体真值：In'),Patch(color=statecols[2],label='物体真值：Held'),Patch(color=statecols[0],label='物体真值：其他'),Patch(color=drawercols[0],label='环境真值：关闭'),Patch(color=drawercols[1],label='环境真值：半开'),Patch(color=drawercols[2],label='环境真值：打开'),Line2D([],[],marker='o',color='#415466',linestyle='None',label='开发集'),Line2D([],[],marker='^',color='#415466',linestyle='None',label='完整组合留出'),Line2D([],[],marker='o',markerfacecolor='none',markeredgecolor='#D33D59',linestyle='None',label='红圈：In错／环境抽屉错'),Line2D([],[],marker='x',color='#202830',linestyle='None',label='黑×：仅Held错')]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.068),ncol=5,frameon=False,fontsize=11,columnspacing=1.6,handletextpad=.6)
 fig.text(.5,.038,'每点＝某次真实执行后的一份128维循环记忆；颜色及错误标记只作事后注释，不参与降维。未抽样。',ha='center',fontsize=11,color='#415466')
 fig.text(.5,.017,'标准化 → PCA50 → t-SNE（perplexity=30）；坐标与簇间距离不代表物理距离、动作成功概率或泛化证明。',ha='center',fontsize=10.5,color='#607282')
 fig.subplots_adjust(left=.065,right=.98,bottom=.185,top=.87,hspace=.37,wspace=.22)
 prefix=T/f'figures/最新868模型_四份观测记忆_tSNE_seed{seed}'
 for ext in ['png','svg','pdf']:fig.savefig(str(prefix)+'.'+ext,dpi=260,bbox_inches='tight')
 plt.close(fig);figurestats.append(report);print('RENDERED',seed,report,flush=True)
 bysplit={}
 for split in splitmarkers:
  tt=[r for r in rr if r['split']==split];bysplit[split]={'object_points':len(tt),'In_positive':sum(int(r['true_In'])for r in tt),'Held_positive':sum(int(r['true_Held'])for r in tt),'In_FP':sum(r['In_error_type']=='FP'for r in tt),'In_FN':sum(r['In_error_type']=='FN'for r in tt),'Held_only_errors':sum(int(r['pred_Held'])!=int(r['true_Held'])and r['In_error_type']not in ['FP','FN']for r in tt)}
 report['by_split']=bysplit
save(T/'results/figure_counts.json',figurestats);save(T/'results/font.json',{'family':fontname,'original_collection':str(fontsource),'local_selected_SC_face':str(fontout),'SVG_text_as_paths':True,'PDF_embedded_font_requested':42})
