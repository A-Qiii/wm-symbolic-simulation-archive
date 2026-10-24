from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from PIL import Image,ImageDraw,ImageFont
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
F=FontProperties(fname=FONT)
BLUE='#2474b7';ORANGE='#e18d23';GREEN='#21956b';DARK='#233747';GRAY='#c7ced4';RED='#b44746'
plt.rcParams['axes.unicode_minus']=False

def text(ax,x,y,s,size=14,color=DARK,**kwargs):return ax.text(x,y,s,transform=ax.transAxes,fontproperties=F,fontsize=size,color=color,va='top',**kwargs)
def montage(path,out,label,frame):
 with np.load(path)as z:
  im=Image.new('RGB',(480,284),'white');im.paste(Image.fromarray(z['rgb_static']).resize((240,240),Image.Resampling.NEAREST),(0,29));im.paste(Image.fromarray(z['rgb_gripper']).resize((240,240),Image.Resampling.NEAREST),(240,29));dr=ImageDraw.Draw(im);ft=ImageFont.truetype(FONT,17);dr.text((4,4),f'{label} · frame {frame}',font=ft,fill=DARK);dr.text((242,4),'腕部相机（放大）',font=ft,fill=DARK);ft2=ImageFont.truetype(FONT,12);dr.text((4,269),'左：固定相机；右：腕部相机。均为真实观测。',font=ft2,fill='#5e707d');im.save(out,quality=94)
 return im
def scatter(ax,co,index):
 bg=co['background_xy'];xy=co['case_xy'][index];ax.scatter(bg[:,0],bg[:,1],c=GRAY,s=13,alpha=.45,edgecolors='none',zorder=1)
 for j,color,style in[(1,ORANGE,'--'),(2,GREEN,'-')]:
  if np.linalg.norm(xy[j]-xy[0])>1e-9:ax.annotate('',xy[j],xy[0],arrowprops={'arrowstyle':'-|>','color':color,'lw':1.8,'linestyle':style,'shrinkA':7,'shrinkB':9},zorder=3)
 for j,marker,color,size in[(0,'o',BLUE,100),(1,'^',ORANGE,130),(2,'*',GREEN,210)]:ax.scatter(*xy[j],s=size,c=color,marker=marker,edgecolors='white',linewidths=.9,zorder=4+j)
 lim=co['axis_limits'];ax.set_xlim(lim[0,0],lim[1,0]);ax.set_ylim(lim[0,1],lim[1,1]);ax.grid(alpha=.13);ax.set_axisbelow(True);ax.tick_params(labelsize=10);ax.set_xlabel('t-SNE 1（无物理单位）',fontproperties=F,fontsize=11);ax.set_ylabel('t-SNE 2（无物理单位）',fontproperties=F,fontsize=11)
 mid=lim.mean(0)
 for j,name,color,dy in[(0,'当前',BLUE,-24),(1,'预测',ORANGE,17),(2,'真实',GREEN,37)]:
  dx=24 if xy[j,0]<mid[0]else -24;dy=dy if xy[j,1]<lim[1,1]-.12*np.ptp(lim,axis=0)[1]else -dy;ax.annotate(name,xy[j],xytext=(dx,dy),textcoords='offset points',ha='left'if dx>0 else'right',fontproperties=F,fontsize=12,color=color,arrowprops={'arrowstyle':'-','color':color,'lw':.6},bbox={'boxstyle':'round,pad=.12','facecolor':'white','edgecolor':'none','alpha':.85},annotation_clip=False,zorder=9)

def legend(fig,y=.925):
 hs=[Line2D([],[],color=BLUE,marker='o',linestyle='none',markersize=9,label='当前状态'),Line2D([],[],color=ORANGE,marker='^',linestyle='none',markersize=9,label='预测后继'),Line2D([],[],color=GREEN,marker='*',linestyle='none',markersize=13,label='真实后继'),Line2D([],[],color=ORANGE,linestyle='--',label='当前→预测'),Line2D([],[],color=GREEN,linestyle='-',label='当前→真实'),Line2D([],[],color=GRAY,marker='o',linestyle='none',markersize=5,label='338个真实背景状态')]
 fig.legend(handles=hs,loc='upper center',bbox_to_anchor=(.5,y),ncol=6,prop=FontProperties(fname=FONT,size=13),frameon=False)

def card(fig,spec,co,index,r,large=False):
 sub=spec.subgridspec(2,2,height_ratios=[3.8,2.3],width_ratios=[1.13,1],hspace=.26,wspace=.18)
 ax=fig.add_subplot(sub[0,0]);scatter(ax,co,index)
 pics=sub[0,1].subgridspec(2,1,hspace=.12)
 for j,role in enumerate(['before','after']):
  ia=fig.add_subplot(pics[j]);ia.imshow(Image.open(R/'images'/f'{r["case_id"]}_{role}.jpg'));ia.axis('off')
 box=spec.get_position(fig);fig.text(box.x0,box.y1+.023,f'案例 {r["case_id"]} · {r["title"]}',fontproperties=F,fontsize=19 if large else 17,color=DARK,weight='bold');fig.text(box.x0,box.y1+.003,f'{r["root_id"]} · {r["id"]} · frame {r["before_frame"]} → {r["after_frame"]}',fontsize=10.5,color='#5d6b75')
 ta=fig.add_subplot(sub[1,:]);ta.axis('off');fs=14.5 if not large else 15
 lines=[('当前  '+r['current_semantics'],BLUE),('真实  '+r['actual_semantics'],GREEN),('预测  '+r['predicted_semantics'],ORANGE),(r['mismatch'],RED if r['case_id']!='A'else DARK),(f'原256维误差：{r["latent_mse_AB_standardized"]:.4f}（AB标准化 MSE）',DARK)]
 for j,(s,color)in enumerate(lines):text(ta,0,1-j*.19,s,fs,color)
 if large:text(ta,0,.01,r['diagnostic'],12.5,'#596c78')

def main():
 cases=read(E/'results/cases.json');co=dict(np.load(E/'results/coordinates.npz'))
 for r in cases:
  for role,label in[('before','执行前'),('after','执行后')]:montage(r[role+'_path'],R/'images'/f'{r["case_id"]}_{role}.jpg',label,r[role+'_frame'])
 fig=plt.figure(figsize=(22,17.5),facecolor='white');grid=fig.add_gridspec(2,2,left=.055,right=.97,top=.86,bottom=.075,hspace=.27,wspace=.15)
 for i,r in enumerate(cases):card(fig,grid[i//2,i%2],co,i,r)
 fig.suptitle('给定动作后：模型预测与真实结果是否一致？',fontproperties=F,fontsize=27,y=.98);fig.text(.5,.946,'同一匹配臂模型 seed 0 / update1000 · 四例与背景联合拟合一次 · 四面板坐标范围完全相同',ha='center',fontproperties=F,fontsize=15,color='#526878');legend(fig,.932)
 fig.text(.055,.035,'一个点是一个256维状态向量的二维位置，不是概率密度。箭头长度不能代替原空间误差；预测没有生成RGB。',fontproperties=F,fontsize=14,color=DARK);fig.text(.055,.014,'D例为放置请求被拒绝（0控制步）；其真实后继也被同一查询头误判为“打开”，不能将错误全部归因于转移模型。',fontproperties=F,fontsize=13,color=RED);fig.savefig(R/'action_cases_overview.png',dpi=200);plt.close(fig)
 for i,r in enumerate(cases):
  fig=plt.figure(figsize=(14,11.3),facecolor='white');g=fig.add_gridspec(1,1,left=.08,right=.96,top=.82,bottom=.10);card(fig,g[0,0],co,i,r,True);fig.suptitle('当前状态 + 实际动作 → 预测后继 / 真实后继',fontproperties=F,fontsize=24,y=.987);fig.text(.5,.938,'canonical_matched · seed 0 · update1000 · 复用总览的同一坐标与范围',ha='center',fontproperties=F,fontsize=13,color='#526878');legend(fig,.925);fig.text(.08,.04,'橙虚线表示预测，绿实线表示真实；灰点是实际状态背景。预测点没有预测图像，右侧均为真实RGB。',fontproperties=F,fontsize=12,color=DARK);fig.text(.08,.019,'误差 = mean[ ((预测向量 − 真实向量) / AB训练标准差)² ]，共256维，无物理单位。',fontproperties=F,fontsize=12,color=DARK);name='case_D_refused.png'if r['case_id']=='D'else f'case_{r["case_id"]}.png';fig.savefig(R/name,dpi=220);plt.close(fig)
 print('FIVE_CASE_FIGURES_READY',flush=True)
if __name__=='__main__':main()
