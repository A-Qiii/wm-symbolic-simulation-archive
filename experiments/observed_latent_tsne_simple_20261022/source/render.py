from pathlib import Path
import os,csv,json,hashlib,collections
T=Path(__file__).resolve().parent.parent;L=T.parent/'observed_latent_tsne_labels_20261022';O=T.parent/'observed_latent_tsne_20261022'
os.environ['MPLCONFIGDIR']=str(T/'cache/matplotlib');os.environ['OMP_NUM_THREADS']='4'
import numpy as np,matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
font=O/'cache/NotoSansCJKSC-Regular.otf';fm.fontManager.addfont(str(font));fontname=fm.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.family':[fontname,'DejaVu Sans'],'font.size':15,'axes.unicode_minus':False,'savefig.facecolor':'white'})
BLUE='#277FB7';GOLD='#D58D18';GRAY='#A5AFBA';GREEN='#24866B'
def read(p):
 with p.open()as f:return list(csv.DictReader(f))
def coords(r):return np.array([[float(a['tsne_x']),float(a['tsne_y'])]for a in r])
def mid(x):
 m=np.median(x,0);return x[np.argmin(((x-m)**2).sum(1))]
def setup(ax,x,title):
 lo=x.min(0);hi=x.max(0);pad=(hi-lo)*.10;ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_title(title,fontsize=23,pad=16,color='#30495B');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.10);ax.set_axisbelow(True);ax.set_xlabel('t-SNE 维度1（无物理单位）',fontsize=13);ax.set_ylabel('t-SNE 维度2',fontsize=13);ax.tick_params(labelsize=11)
def scatter(ax,r,x,cls,palette):
 for split,marker in [('development','o'),('combination_test','^')]:
  for c in sorted(set(cls)):
   ix=np.array([a['split']==split and cls[i]==c for i,a in enumerate(r)]);ax.scatter(*x[ix].T,c=palette[c],marker=marker,s=27,alpha=.87,edgecolors='none')
def call(ax,x,mask,text,pos,color):
 assert mask.any();target=mid(x[mask]);ax.annotate(text,xy=target,xytext=pos,ha='center',va='center',fontsize=22,color='#263E50',linespacing=1.2,bbox=dict(boxstyle='round,pad=.4',facecolor='white',edgecolor=color,linewidth=1.5,alpha=.96),arrowprops=dict(arrowstyle='-',lw=1.6,color=color),zorder=6);return target.tolist()
def split_legend(fig,y):
 fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#4A6171',label='开发集'),Line2D([],[],marker='^',ls='',color='#4A6171',label='未见组合测试集')],loc='upper center',bbox_to_anchor=(.5,y),ncol=2,frameon=False,fontsize=16,markerscale=1.7)
stats=[]
# The same960 joint blue+environment coordinates already delivered; no fit.
r=read(L/'results/blue_region_membership.csv');x=coords(r);cl=np.array([int(a['cluster'])for a in r]);held=np.array([a['true_Held']=='1'for a in r]);inside=np.array([a['true_In']=='1'for a in r])
fig,axs=plt.subplots(1,2,figsize=(21,10.8));fig.subplots_adjust(left=.065,right=.975,bottom=.17,top=.79,wspace=.20);fig.suptitle('蓝块＋环境：先看主要区域的状态语义',fontsize=28,fontweight='bold',y=.985);fig.text(.5,.925,'最新868模型·种子0｜蓝块128维＋环境128维联合投影；两图逐点对应，只换真实状态颜色',ha='center',fontsize=17,color='#4C6374');split_legend(fig,.895)
for ax,f,title,palette in zip(axs,['true_In','true_Held'],['是否在抽屉里','是否被拿着'],[{'1':BLUE,'0':GRAY},{'1':GOLD,'0':GRAY}]):
 scatter(ax,r,x,[a[f]for a in r],palette);setup(ax,x,title);labs=['在抽屉里','不在抽屉里']if f=='true_In'else['被拿着','未被拿着'];ax.legend(handles=[Patch(color=palette[c],label=t)for c,t in zip(['1','0'],labs)],loc='upper right',frameon=False,fontsize=14)
for mask,text,pos,color in [(cl==2,'在抽屉里',(-28,-32),BLUE),(cl==5,'以在抽屉里为主',(40,-15),BLUE),(cl==0,'未被拿着且\n不在抽屉里',(27,31),GRAY)]:
 target=call(axs[0],x,mask,text,pos,color);stats.append({'figure':'blue_joint','panel':'In','text':text,'n':int(mask.sum()),'In':int(inside[mask].sum()),'Held':int(held[mask].sum()),'region_rule':'existing DBSCAN cluster '+str(list(set(cl[mask]))),'target':target})
# This fixed visible upper rectangle is94.7%Held; do not name the much larger
#358-point mixed region a Held cluster. Every point remains in the scatter.
upper=(x[:,0]>=-20)&(x[:,0]<=5)&(x[:,1]>=15)&(x[:,1]<=27)
for mask,text,pos,color,rule in [(upper,'以被拿着为主',(-20,36),GOLD,'fixed rectangle x[-20,5],y[15,27]'),(cl==2,'未被拿着',(12,-28),GRAY,'existing DBSCAN cluster2')]:
 target=call(axs[1],x,mask,text,pos,color);stats.append({'figure':'blue_joint','panel':'Held','text':text,'n':int(mask.sum()),'In':int(inside[mask].sum()),'Held':int(held[mask].sum()),'region_rule':rule,'target':target})
fig.text(.5,.075,'只标注少数较大、语义清楚的区域；小团和混合区域不逐一命名，全部960个点保留。',ha='center',fontsize=16,color='#4C6374');fig.text(.5,.035,'这是一份联合信息的二维投影，同一状态仍可能出现在多个区域。',ha='center',fontsize=16,color='#4C6374');fig.savefig(T/'figures/蓝块与环境_主要语义.png',dpi=210);plt.close(fig)
# Four original memories: retain all3840points and both independent maps.
r=read(L/'results/four_memory_region_membership.csv');orig=read(O/'results/coordinates_seed0.csv')+read(O/'results/scene_coordinates_seed0.csv');assert len(r)==len(orig)==3840
for a,b in zip(r,orig):assert a['tsne_x']==b['tsne_x']and a['tsne_y']==b['tsne_y']
fig,axs=plt.subplots(2,2,figsize=(22,17));fig.subplots_adjust(left=.065,right=.975,bottom=.10,top=.84,hspace=.36,wspace=.20);fig.suptitle('四份观测记忆：只标主要语义区域',fontsize=29,fontweight='bold',y=.99);fig.text(.5,.945,'最新868模型·种子0｜红、蓝、紫物体共用原坐标；环境使用原先的独立坐标',ha='center',fontsize=18,color='#4C6374');split_legend(fig,.925)
object_limits=coords(r[:2880])
for ax,col,title in zip(axs.flat[:3],['red','blue','pink'],['红块记忆（128维）','蓝块记忆（128维）','紫块记忆（128维）']):
 ids=[i for i,a in enumerate(r)if a['panel']==col];rr=[r[i]for i in ids];oo=[orig[i]for i in ids];xx=coords(rr);cc=np.array([int(a['cluster'])for a in rr]);st=['Held'if a['true_Held']=='1'else'In'if a['true_In']=='1'else'neither'for a in oo];scatter(ax,rr,xx,st,{'Held':GOLD,'In':BLUE,'neither':GRAY});setup(ax,object_limits,title)
 for mask,text,pos,color in [(cc==3,'以在抽屉里为主',(-12,-63),BLUE),(cc==0,'未被拿着且\n不在抽屉里',(42,79),GRAY),(cc==2,'以被拿着为主',(55,-48),GOLD)]:
  target=call(ax,xx,mask,text,pos,color);cnt=collections.Counter(np.array(st)[mask]);stats.append({'figure':'four_memories','panel':col,'text':text,'n':int(mask.sum()),'states':dict(cnt),'target':target,'region_rule':'existing DBSCAN cluster '+str(list(set(cc[mask])))})
ax=axs.flat[3];ids=[i for i,a in enumerate(r)if a['panel']=='environment'];rr=[r[i]for i in ids];oo=[orig[i]for i in ids];xx=coords(rr);cc=np.array([int(a['cluster'])for a in rr]);st=[int(a['drawer_true'])for a in oo];scatter(ax,rr,xx,st,{0:GREEN,1:GOLD,2:BLUE});setup(ax,xx,'环境记忆（128维；独立坐标）')
for mask,text,pos,color in [(cc==0,'打开',(-29,-36),BLUE),(np.isin(cc,[1,6]),'关闭',(43,37),GREEN),(cc==8,'半开',(36,-39),GOLD)]:
 target=call(ax,xx,mask,text,pos,color);stats.append({'figure':'four_memories','panel':'environment','text':text,'n':int(mask.sum()),'drawer_states':dict(collections.Counter(np.array(st)[mask].tolist())),'target':target,'region_rule':'existing DBSCAN cluster '+str(list(set(cc[mask])))})
fig.text(.5,.04,'颜色表示真实状态；所有数据点保留。图中没有错误标记，也不为每个小团添加说明。',ha='center',fontsize=17,color='#4C6374');fig.savefig(T/'figures/四份记忆_主要语义.png',dpi=210);plt.close(fig)
(T/'results/annotation_evidence.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n');inputs=[L/'results/blue_region_membership.csv',L/'results/four_memory_region_membership.csv',O/'results/coordinates_seed0.csv',O/'results/scene_coordinates_seed0.csv'];(T/'results/input_hashes.json').write_text(json.dumps({str(a):hashlib.sha256(a.read_bytes()).hexdigest()for a in inputs},indent=2)+'\n');print('Rendered both; same coordinates, all points, no new dimensional reduction/model inference/training/collection')
