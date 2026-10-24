from pathlib import Path
import os,csv,json,hashlib,collections
T=Path(__file__).resolve().parent.parent;O=T.parent/'observed_latent_tsne_20261022'
os.environ['MPLCONFIGDIR']=str(T/'cache/matplotlib');os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np,matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager as fm
from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN
font=O/'cache/NotoSansCJKSC-Regular.otf';fm.fontManager.addfont(str(font));fmname=fm.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.family':[fmname,'DejaVu Sans'],'font.size':16,'axes.unicode_minus':False,'savefig.facecolor':'white'})
COL={'in':'#267EB8','held':'#D28A14','other':'#9BA6B2','closed':'#218467','mid':'#D28A14','open':'#267EB8','yes':'#267EB8','no':'#B3BDC7'}
def read(p):
 with p.open()as f:return list(csv.DictReader(f))
def xy(rows):return np.array([[float(a['tsne_x']),float(a['tsne_y'])]for a in rows])
def dump(name,records):
 (T/'results'/name).write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
def savecsv(name,rr):
 with(T/'results'/name).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def state(a):return 'held'if a['true_Held']=='1'else'in'if a['true_In']=='1'else'other'
def medoid(x):
 mid=np.median(x,axis=0);return x[np.argmin(((x-mid)**2).sum(1))]
def canvas(fig,rect,title):
 l,b,w,h=rect;ax=fig.add_axes([l,b+h*.09,w*.57,h*.79]);tx=fig.add_axes([l+w*.60,b+h*.04,w*.40,h*.89]);tx.axis('off')
 fig.text(l,b+h*.94,title,fontsize=24,fontweight='bold',color='#33485A')
 ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.10);ax.set_axisbelow(True);ax.tick_params(labelsize=12)
 ax.set_xlabel('t-SNE 维度1（无物理单位）',fontsize=14);ax.set_ylabel('t-SNE 维度2',fontsize=14)
 return ax,tx
def scatter(ax,rr,x,classes,palette,limits=None):
 for split,marker in [('development','o'),('combination_test','^')]:
  for cl in sorted(set(classes)):
   ix=np.array([a['split']==split and classes[i]==cl for i,a in enumerate(rr)])
   ax.scatter(*x[ix].T,c=palette[cl],marker=marker,s=26,alpha=.88,edgecolors='none')
 if limits is None:
  low=x.min(0);high=x.max(0);pad=(high-low)*.07;limits=(low-pad,high+pad)
 ax.set_xlim(limits[0][0],limits[1][0]);ax.set_ylim(limits[0][1],limits[1][1])
 return limits
def label(ax,tx,x,lab,group_ids,y,text,color,targets=None,tag=None):
 mask=np.isin(lab,group_ids);assert mask.any();p=medoid(x[mask]);ann=ax.annotate(text,xy=p,xycoords=ax.transData,xytext=(.04,y),textcoords=tx.transAxes,ha='left',va='center',fontsize=17,linespacing=1.38,color='#243747',bbox=dict(boxstyle='round,pad=.45',fc='white',ec=color,lw=1.7),arrowprops=dict(arrowstyle='-',color=color,lw=1.3,relpos=(0,.5)),annotation_clip=False,zorder=6)
 if targets:
  for c in targets:
   v=x[lab==c]
   if len(v):ax.annotate('',xy=medoid(v),xycoords=ax.transData,xytext=(.04,y),textcoords=tx.transAxes,arrowprops=dict(arrowstyle='-',color=color,lw=.9,alpha=.55),annotation_clip=False,zorder=3)
 if tag:
  ax.text(*p,tag,ha='center',va='center',fontsize=16,fontweight='bold',bbox=dict(boxstyle='round,pad=.13',fc='white',ec='none',alpha=.92),color='#203A50',zorder=7)
 return mask
def base_text(fig,title,subtitle):
 fig.suptitle(title,fontsize=28,fontweight='bold',y=.993)
 fig.text(.5,.94 if fig.get_figheight()>15 else .925,subtitle,ha='center',fontsize=18,color='#425A6B')
 fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#425A6B',label='开发集'),Line2D([],[],marker='^',ls='',color='#425A6B',label='未见组合测试集')],loc='upper center',bbox_to_anchor=(.5,.915 if fig.get_figheight()>15 else .90),frameon=False,ncol=2,fontsize=18,markerscale=1.5)

# Four original memories. Coordinates are read verbatim; only annotations change.
r=read(O/'results/coordinates_seed0.csv');s=read(O/'results/scene_coordinates_seed0.csv');xx=xy(r);sx=xy(s)
ol=DBSCAN(eps=5,min_samples=12).fit_predict(xx);sl=DBSCAN(eps=5,min_samples=12).fit_predict(sx)
groups={'upper_other':[0,7,9,11],'held':[2],'in':[3,4,5,6,8,10,12,13],'lower_other':[1],'left_other':[14]}
assert set(sum(groups.values(),[]))==set(ol)-{-1}
fig=plt.figure(figsize=(25,19));base_text(fig,'四份观测记忆：这些区域主要是什么状态？','最新868训练模型 · 种子0 · 每面板960点；保留全部样本和上一版坐标')
stats=[];limits=(xx.min(0)-(xx.max(0)-xx.min(0))*.07,xx.max(0)+(xx.max(0)-xx.min(0))*.07)
rects=[(.045,.50,.44,.38),(.535,.50,.44,.38),(.045,.065,.44,.38),(.535,.065,.44,.38)]
for col,zh,rect in zip(['red','blue','pink'],['红块记忆 h（128维）','蓝块记忆 h（128维）','紫块记忆 h（128维）'],rects):
 ids=np.flatnonzero([a['object']==col for a in r]);rr=[r[i]for i in ids];x=xx[ids];lab=ol[ids];classes=[state(a)for a in rr];ax,tx=canvas(fig,rect,zh);scatter(ax,rr,x,classes,COL,limits)
 for key,y in zip(['upper_other','held','in','lower_other','left_other'],[.90,.69,.47,.26,.055]):
  sel=np.isin(lab,groups[key]);aa=[a for i,a in enumerate(rr)if sel[i]];cnt=collections.Counter(state(a)for a in aa);n=len(aa);test=sum(a['split']=='combination_test'for a in aa)
  if not n:continue
  if key=='held':
   text=f'B 被拿着 {100*cnt["held"]/n:.1f}%\n在抽屉里 {100*cnt["in"]/n:.1f}%'+(f'\n未被拿着且不在抽屉里 {100*cnt["other"]/n:.1f}%'if cnt["other"]else'')+f'\n{n}点，混合区域';color=COL['held'];targets=None;tag='B'
  elif key=='in':
   text=f'A 在抽屉里 {100*cnt["in"]/n:.1f}%\n未被拿着且不在抽屉里 {100*cnt["other"]/n:.1f}%\n{n}点，多个区域合计';color=COL['in'];targets=[5,4,12];tag='A'
  else:
   text=f'未被拿着且不在抽屉里\n{100*cnt["other"]/n:.1f}%（{n}点）';color=COL['other'];targets=[11]if key=='upper_other'else None;tag=None
   if cnt['in']:text+=f'\n其余：在抽屉里 {100*cnt["in"]/n:.1f}%'
   if key=='left_other':text+=f'\n左侧小团：测试集占{100*test/n:.1f}%'
  label(ax,tx,x,lab,groups[key],y,text,color,targets,tag)
  stats.append({'panel':col,'region':key,'clusters':groups[key],'n':n,'true_states':dict(cnt),'test_points':test})
ax,tx=canvas(fig,rects[3],'环境记忆 c（128维；独立坐标）');sceneclasses=[['closed','mid','open'][int(a['drawer_true'])]for a in s];scatter(ax,s,sx,sceneclasses,COL)
sg={'closed':[1,6],'open':[0,4,5,7],'mid':[2,8],'mixed':[3]}
for key,y in zip(['closed','open','mid','mixed'],[.90,.64,.38,.10]):
 sel=np.isin(sl,sg[key]);aa=[s[i]for i in np.flatnonzero(sel)];cnt=collections.Counter(['关闭','半开','打开'][int(a['drawer_true'])]for a in aa);n=len(aa);major={'closed':'关闭','open':'打开','mid':'半开'}.get(key)
 if key=='mixed':text=f'C 混合区域（{n}点）\n关闭{100*cnt["关闭"]/n:.1f}% · 半开{100*cnt["半开"]/n:.1f}%\n打开{100*cnt["打开"]/n:.1f}%';color='#765A87';targets=None;tag='C'
 else:
  text=f'{major} {100*cnt[major]/n:.1f}%（{n}点）'
  others=[f'{name}{100*cnt[name]/n:.1f}%'for name in ['关闭','半开','打开']if name!=major and cnt[name]]
  text+='\n'+('其余：'+' · '.join(others)if others else'两个区域合计'if key!='open'else'多个区域合计');color=COL[key];targets=sg[key][1:];tag=None
 label(ax,tx,sx,sl,sg[key],y,text,color,targets,tag);stats.append({'panel':'environment','region':key,'clusters':sg[key],'n':n,'true_states':dict(cnt),'test_points':sum(a['split']=='combination_test'for a in aa)})
fig.text(.5,.023,'百分比是标注区域内的真实状态占比，不是预测准确率。物体共用坐标；环境坐标独立，不与物体比较。',ha='center',fontsize=17,color='#425A6B')
fig.savefig(T/'figures/四份记忆_直接语义标注.png',dpi=210);plt.close(fig)
assign=[]
for rows,labels,panel in [(r,ol,'objects'),(s,sl,'environment')]:
 for i,a in enumerate(rows):
  b={k:a[k]for k in ['point_id','split','episode_id','step','tsne_x','tsne_y']};b['panel']=a.get('object',panel);b['cluster']=int(labels[i]);b['region']=next((k for k,v in (groups if panel=='objects'else sg).items()if int(labels[i])in v),'unassigned');assign.append(b)
savecsv('four_memory_region_membership.csv',assign);dump('four_memory_region_counts.json',stats)

# Blue object + environment: one 256-dimensional input, two truth-colored views.
rr=read(T/'results/blue_context_coordinates.csv');x=xy(rr);lab=np.load(T/'results/blue_annotation_regions.npy');bg={'A':[1],'B':[2,7],'C':[4,8],'D':[3,9],'E':[5],'F':[0,6]};assert set(sum(bg.values(),[]))==set(lab)-{-1}
fig=plt.figure(figsize=(26,11.3));base_text(fig,'蓝块＋环境：关系状态能否聚在一起？','最新868模型·种子0｜蓝块128维＋环境128维联合降维；两图逐点对应，只换真实状态颜色')
bstats=[]
for field,title,rect in [('true_In','是否在抽屉里',(.045,.13,.44,.68)),('true_Held','是否被拿着',(.535,.13,.44,.68))]:
 ax,tx=canvas(fig,rect,title);yes='在抽屉里'if field=='true_In'else'被拿着';no='不在抽屉里'if field=='true_In'else'未被拿着';color=COL['in']if field=='true_In'else COL['held'];scatter(ax,rr,x,[a[field]for a in rr],{'1':color,'0':COL['no']})
 for (tag,cc),y in zip(bg.items(),[.91,.74,.57,.40,.23,.06]):
  mask=np.isin(lab,cc);aa=[a for i,a in enumerate(rr)if mask[i]];n=len(aa);n1=sum(a[field]=='1'for a in aa);n0=n-n1;test=sum(a['split']=='combination_test'for a in aa)
  if n0 and n1:text=f'{tag} {"混合：" if tag=="A" else ""}{yes} {100*n1/n:.1f}%\n{no} {100*n0/n:.1f}%（{n}点）'
  else:text=f'{tag} {yes if n1 else no} 100%\n{n}点'+('，多个区域合计'if len(cc)>1 else'')
  label(ax,tx,x,lab,cc,y,text,color if n1 else COL['no'],cc[1:],tag)
  bstats.append({'field':field,'region':tag,'clusters':cc,'n':n,'true_positive':n1,'true_negative':n0,'test_points':test})
fig.text(.5,.075,'圆点：开发集480点；三角：未见组合测试集480点。彩色表示“是”，灰色表示“否”；不显示预测或错误。',ha='center',fontsize=17,color='#425A6B')
fig.text(.5,.035,'A 是混合区域；B 等区域更偏向单一状态。同一状态可以分成多团；这是联合信息的二维投影，不是状态概率。',ha='center',fontsize=17,color='#425A6B')
fig.text(.5,.009,'区域按二维点的密集程度划分；8个离散点照常显示，未计入区域占比。',ha='center',fontsize=13,color='#64798A')
fig.savefig(T/'figures/蓝块与环境拼接_关系语义.png',dpi=210);plt.close(fig);dump('blue_region_counts.json',bstats)
out=[]
for i,a in enumerate(rr):b=dict(a);b['cluster']=int(lab[i]);b['region']=next((k for k,v in bg.items()if int(lab[i])in v),'unassigned');out.append(b)
savecsv('blue_region_membership.csv',out)
dump('annotation_protocol.json',{'object_environment_original_xy_unchanged':True,'all_original_points_retained':True,'old_figure_object_points':2880,'old_figure_scene_points':960,'blue_joint_points':960,'error_data_used_or_shown':False,'four_memory_geometric_regions':{'algorithm':'DBSCAN on existing2D coordinates','eps':5,'min_samples':12,'groups':groups,'scene_groups':sg,'noise_points_plotted_but_not_in_region_percentages':{'object':int((ol==-1).sum()),'scene':int((sl==-1).sum())}},'blue_geometric_regions':{'algorithm':'DBSCAN on fixed2D coordinates','eps':5,'min_samples':8,'groups':bg,'noise_points_plotted_but_not_in_region_percentages':int((lab==-1).sum())},'labels':'State labels only annotate already-fixed geometric regions. Some regions combined to reduce repeated text; membership CSV defines exact denominators. Ratios are true state composition, never classifier accuracy.','coordinate_warp_or_subsampling':False})
print('RENDERED both figures; all points retained, no error marks')
