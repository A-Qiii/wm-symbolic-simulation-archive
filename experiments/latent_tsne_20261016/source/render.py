from common import *
from project import labels
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from PIL import Image,ImageDraw,ImageFont
import collections
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';F=FontProperties(fname=FONT)
TITLES={'holding':'真实抓持状态','red_in':'红块是否在抽屉内','blue_in':'蓝块是否在抽屉内','pink_in':'粉块是否在抽屉内','drawer':'真实抽屉状态','environment':'实际物理环境（A/B/C/D）','sampling_group':'训练 / 开发 / 已用测试批次'}
PAL={'holding':{'空手':'#87929c','持红':'#d54b48','持蓝':'#2d79be','持粉':'#c6579c','多物体':'#b18b25','未确定':'#cfcfcf'},'drawer':{'关闭':'#416fae','打开':'#df983c','中间':'#159b82','未确定':'#cfcfcf'},'environment':{'A':'#3779b7','B':'#e18e2c','C':'#319267','D':'#9465ae'},'sampling_group':{'AB训练':'#356eb4','C开发':'#35a16b','D已用测试':'#8550a7','D2已用测试':'#bd8ad2'}}
for name,color in [('red_in','#d54b48'),('blue_in','#2d79be'),('pink_in','#c6579c')]:PAL[name]={'不在屉内':'#a3aab1','在屉内':color,'未确定':'#ddd2b9'}
MARK={'AB训练':'o','C开发':'s','D已用测试':'D','D2已用测试':'^'};SYMBOL={'AB训练':'circle','C开发':'square','D已用测试':'diamond','D2已用测试':'triangle-up'}
GROUPPAL={'持物且关闭：拒绝':'#d14c4c','正常执行成功':'#239875','其他拒绝':'#8693a0','执行失败':'#a263b7'}
def style(ax,title,xy):
 ax.set_title(title,fontproperties=F,fontsize=13);ax.set_xlabel('t-SNE 1（无物理单位）',fontproperties=F,fontsize=9);ax.set_ylabel('t-SNE 2（无物理单位）',fontproperties=F,fontsize=9);ax.grid(alpha=.13);ax.set_axisbelow(True);ax.tick_params(labelsize=8)
 pad=np.ptp(xy,axis=0)*.08;ax.set_xlim(xy[:,0].min()-pad[0],xy[:,0].max()+pad[0]);ax.set_ylim(xy[:,1].min()-pad[1],xy[:,1].max()+pad[1])
def scatter(ax,xy,rows,field,mask=None):
 ls=[labels(r)[field]for r in rows];mask=np.ones(len(rows),bool)if mask is None else np.asarray(mask)
 for val,color in PAL[field].items():
  ix=np.array([i for i,v in enumerate(ls)if v==val and mask[i]],int)
  if not len(ix):continue
  for split,m in MARK.items():
   ids=[i for i in ix if rows[i]['sampling_group']==split]
   if ids:ax.scatter(xy[ids,0],xy[ids,1],s=26,c=color,marker=m,alpha=.85,edgecolors='white',linewidths=.3)
 handles=[Line2D([],[],marker='o',color='none',markerfacecolor=co,markeredgecolor='none',label=f'{v} ({sum(val==v and mask[i]for i,val in enumerate(ls))})',markersize=6)for v,co in PAL[field].items()if any(val==v and mask[i]for i,val in enumerate(ls))];ax.legend(handles=handles,prop=FontProperties(fname=FONT,size=9),loc='best',framealpha=.9)
def savefig(fig,name):fig.savefig(R/name,dpi=220,facecolor='white');plt.close(fig)
def truth_text(tr):
 y=tr['y'];v=tr['v'];h=','.join(c for c,b in zip(['红','蓝','粉'],y[:3])if b)or'无';inside=','.join(c for i,c in enumerate(['红','蓝','粉'])if y[3+i]and v[3+i])or'无';drawer=['关','开','中间'][int(np.argmax(y[6:]))]if all(v[6:])else'不确定';return f'持物={h}；入屉={inside}；抽屉={drawer}；有效命题={sum(v)}/9'
def query_text(pr):
 h=','.join(c for c,b in zip(['红','蓝','粉'],pr[:3])if b>=.5)or'无';inside=','.join(c for c,b in zip(['红','蓝','粉'],pr[3:6])if b>=.5)or'无';return f"查询读出：持物={h}；入屉={inside}；抽屉={['关','开','中间'][int(np.argmax(pr[6:]))]}；P关={pr[6]:.3f}"
def thumbnail(path,name):
 dst=R/'images'/f'{name}.jpg'
 with np.load(path)as z:
  im=Image.new('RGB',(440,264),'#f2f5f8');im.paste(Image.fromarray(z['rgb_static']).resize((220,220),Image.Resampling.NEAREST),(0,22));im.paste(Image.fromarray(z['rgb_gripper']).resize((220,220),Image.Resampling.NEAREST),(220,22));dr=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,14);dr.text((4,2),'固定相机',font=font,fill='#18344b');dr.text((224,2),'腕部相机（放大显示）',font=font,fill='#18344b');dr.text((4,244),name+' · 原始真实RGB，非预测图像',font=font,fill='#18344b');im.save(dst,quality=93)
 return 'images/'+dst.name
def main():
 rows=read(E/'data/main_states.json');pairs=read(E/'data/joint_pairs.json');co=dict(np.load(E/'results/coordinates.npz'));mainxy=co['interaction_xy'];shapehandles=[Line2D([],[],color='#555',marker=m,linestyle='none',markersize=5,label=k)for k,m in MARK.items()]
 fig,axs=plt.subplots(2,3,figsize=(18,10));fig.subplots_adjust(left=.05,right=.98,bottom=.09,top=.90,hspace=.27,wspace=.20)
 for ax,field in zip(axs.flat,['holding','red_in','blue_in','pink_in','drawer','environment']):scatter(ax,mainxy,rows,field);style(ax,TITLES[field],mainxy)
 fig.suptitle(f'交互预训练连续隐状态 · 同一组{len(rows)}点、同一次t-SNE、六种着色',fontproperties=F,fontsize=20);fig.legend(handles=shapehandles,loc='lower center',ncol=4,prop=FontProperties(fname=FONT,size=10));savefig(fig,'main_states.png')
 fig,axs=plt.subplots(1,2,figsize=(13,5.8));fig.subplots_adjust(left=.07,right=.98,bottom=.15,top=.86,wspace=.22)
 for ax,field in zip(axs,['environment','sampling_group']):scatter(ax,mainxy,rows,field);style(ax,TITLES[field],mainxy)
 fig.suptitle('同一坐标：环境域与采样批次分开看 · D和D2都属于物理环境D',fontproperties=F,fontsize=17);fig.legend(handles=shapehandles,loc='lower center',ncol=4,prop=FontProperties(fname=FONT,size=10));savefig(fig,'environment_and_batch.png')
 subsets={}
 fig,axs=plt.subplots(1,2,figsize=(13,5.7));fig.subplots_adjust(left=.07,right=.98,bottom=.13,top=.84,wspace=.23)
 for ax,dc,txt in zip(axs,[0,1],['空手 + 三物块均不在屉内 + 抽屉关','空手 + 三物块均不在屉内 + 抽屉开']):
  mask=np.array([all(r['truth']['v'])and not any(r['truth']['y'][:6])and r['truth']['y'][6+dc]for r in rows]);ax.scatter(mainxy[:,0],mainxy[:,1],c='#d5dbe0',s=9,alpha=.15);scatter(ax,mainxy,rows,'environment',mask);style(ax,txt,mainxy);subsets[txt]={'points':int(mask.sum()),'by_environment':dict(collections.Counter(r['environment']for i,r in enumerate(rows)if mask[i]))}
 fig.suptitle('同一完整语义状态跨环境 · 复用主图坐标，未重新拟合',fontproperties=F,fontsize=17);savefig(fig,'same_state_cross_domain.png');save(E/'results/cross_domain_subsets.json',subsets)
 fig,axs=plt.subplots(1,3,figsize=(18,5.5));fig.subplots_adjust(left=.05,right=.98,bottom=.15,top=.84,wspace=.20)
 for ax,field in zip(axs,['holding','drawer','environment']):scatter(ax,co['ae_xy'],rows,field);style(ax,TITLES[field],co['ae_xy'])
 fig.suptitle('AE只重构表示对照 · 同338个状态，但独立t-SNE坐标，不能跨模型比轴距离',fontproperties=F,fontsize=17);fig.legend(handles=shapehandles,loc='lower center',ncol=4,prop=FontProperties(fname=FONT,size=10));savefig(fig,'ae_comparison.png')
 fig,axs=plt.subplots(1,2,figsize=(13,5.7));fig.subplots_adjust(left=.07,right=.98,bottom=.13,top=.86,wspace=.23)
 for ax,key,pp in zip(axs,['interaction_xy','stability_xy'],[30,15]):scatter(ax,co[key],rows,'drawer');style(ax,f'perplexity={pp}；seed=42',co[key])
 fig.suptitle('预先指定的稳定性检查 · 两张均保留，不挑最好看的投影',fontproperties=F,fontsize=17);savefig(fig,'stability.png')
 mainnodes=[]
 for i,r in enumerate(rows):mainnodes.append({'id':r['point_id'],'key':r['key'],'root':r['root_id'],'environment':r['environment'],'batch':r['batch'],'sampling_group':r['sampling_group'],'truth':truth_text(r['truth']),'labels':labels(r),'image':thumbnail(r['path'],r['point_id']),'path':r['path'],'roles':r['roles'],'symbol':SYMBOL[r['sampling_group']]})
 predrows=read(REPAIR/'results/D2_records.json');lookup={(r['arm'],r['pair_id']):r for r in predrows if r['seed']==0};jointnodes=[];kindnames={'actual':'真实后继','canonical_control':'对照预测','canonical_matched':'修复预测'};kinds=['actual','canonical_control','canonical_matched'];symbols={'actual':'circle','canonical_control':'triangle-up','canonical_matched':'diamond'}
 for kind in kinds:
  for i,r in enumerate(pairs):
   group='持物且关闭：拒绝'if r['condition']=='held_closed'and r['status']=='refused'else'正常执行成功'if r['status']=='success'else'其他拒绝'if r['status']=='refused'else'执行失败';record=lookup['canonical_matched'if kind=='actual'else kind,r['pair_id']];q=record['actual_successor'if kind=='actual'else'predicted_successor'];im=thumbnail(r['after_path'],f'J{i+1:02d}_after')if kind=='actual'else f'images/J{i+1:02d}_after.jpg';bef=thumbnail(r['before_path'],f'J{i+1:02d}_before')if kind=='actual'else f'images/J{i+1:02d}_before.jpg';jointnodes.append({'id':f'J{i+1:02d}-{kind}','pair_index':i,'pair_id':r['pair_id'],'root':r['root_id'],'kind':kind,'kind_name':kindnames[kind],'symbol':symbols[kind],'group':group,'truth':truth_text(r['true_after']),'drawer':['关闭','打开','中间'][int(np.argmax(r['true_after']['y'][6:]))],'query':query_text(q),'condition':r['condition'],'skill':r['skill'],'status':r['status'],'image':im,'before_image':bef,'image_note':'两张缩略图都是这次真实执行的前后RGB；预测点没有预测RGB。'})
 chosen=[]
 for color in COLORS:
  for cond,rep in [('held_closed',0),('held_open',1)]:
   i=next(j for j,r in enumerate(pairs)if r['root_id']==f'D2_place_repair_{color}_{rep}'and r['condition']==cond);chosen.append(i)
 lines=[];n=len(pairs)
 for i in chosen:
  for offset in[1,2]:lines.append({'a':i,'b':offset*n+i,'pair_index':i,'kind':kinds[offset]})
 jxy=co['joint_xy'];fig,ax=plt.subplots(figsize=(11,8.4));fig.subplots_adjust(left=.09,right=.80,bottom=.12,top=.88)
 for line in lines:ii=[line['a'],line['b']];ax.plot(jxy[ii,0],jxy[ii,1],color='#71808a',alpha=.42,lw=.9,ls='--'if line['kind']=='canonical_control'else':',zorder=1)
 for kind,m in zip(kinds,['o','^','D']):
  for group,color in GROUPPAL.items():
   ix=[i for i,r in enumerate(jointnodes)if r['kind']==kind and r['group']==group]
   if ix:ax.scatter(jxy[ix,0],jxy[ix,1],s=42,c=color,marker=m,edgecolors='white',linewidths=.5,alpha=.87)
 for i in chosen:ax.annotate(f'J{i+1:02d}',jxy[i],xytext=(4,4),textcoords='offset points',fontsize=8,color='#243b4b')
 style(ax,'真实后继 + 两臂一步预测：141点共享一次拟合',jxy)
 handles=[Line2D([],[],marker='o',color='none',markerfacecolor=c,markeredgecolor='none',label=k)for k,c in GROUPPAL.items()if any(r['group']==k for r in jointnodes)]+[Line2D([],[],marker=m,color='#35424c',linestyle='none',label=kindnames[k])for k,m in zip(kinds,['o','^','D'])];handles += [Line2D([],[],color='#71808a',ls='--',label='对照配对线'),Line2D([],[],color='#71808a',ls=':',label='修复配对线')];ax.legend(handles=handles,prop=FontProperties(fname=FONT,size=10),loc='upper left',bbox_to_anchor=(1.01,1));fig.suptitle('一步后继联合投影 · 仅D2已用测试诊断',fontproperties=F,fontsize=19);fig.text(.09,.025,'细线仅连接6组预先指定配对；线长不是原空间误差。预测点着色表示应对应的真实后果。',fontproperties=F,fontsize=10);savefig(fig,'successor_joint.png')
 payload={'titles':TITLES,'palette':PAL,'group_palette':GROUPPAL,'main_nodes':mainnodes,'main_xy':mainxy.tolist(),'ae_xy':co['ae_xy'].tolist(),'stability_xy':co['stability_xy'].tolist(),'joint_nodes':jointnodes,'joint_xy':jxy.tolist(),'lines':lines,'representative_pair_indices':chosen,'joint_pair_count':n};save(E/'results/interactive_payload.json',payload);print('FIGURES_AND_RGB_READY',len(mainnodes),len(jointnodes),flush=True)
if __name__=='__main__':main()
