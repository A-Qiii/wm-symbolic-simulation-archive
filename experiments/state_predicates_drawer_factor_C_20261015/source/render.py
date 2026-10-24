from experiment import *
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';FP=FontProperties(fname=FONT);CN={'red':'红','blue':'蓝','pink':'粉'};FN=['持红','持蓝','持粉','红入屉','蓝入屉','粉入屉','屉开']
def ft(n):return ImageFont.truetype(FONT,n)
def text(im,xy,s,n=22,fill='#172b4d'):ImageDraw.Draw(im).text(xy,str(s),font=ft(n),fill=fill)
def public(path):
 with np.load(path)as z:return Image.fromarray(z['rgb_static']),Image.fromarray(z['rgb_gripper'])
def drawrow(im,row,y,windows):
 d=ImageDraw.Draw(im);side='关闭'if row['drawer']=='closed'else'打开';accent='#2474b5'if row['drawer']=='closed'else'#d97706';d.rounded_rectangle((15,y,1685,y+365),12,fill='#f1f5f9');text(im,(30,y+8),f'抽屉{side} | 真实调用：拒绝，96步后保持原状'if row['actual_all7_unchanged']and row['primitive_status']=='refused'else f'抽屉{side} | {row["outcome_group"]}',23,accent)
 for base,wid,header in [(30,row['before_window'],'调用前实拍'),(420,row['after_window'],'96步后实拍')]:
  a,b=public(windows[wid]['public_frames'][-1]);text(im,(base,y+45),header+'（静态 / 腕部）',20);im.paste(a.resize((230,230)),(base,y+80));im.paste(b.resize((125,125)),(base+240,y+80));text(im,(base+240,y+218),'腕部相机',17)
 x=815;cols=[x,x+115,x+230,x+365,x+515,x+630];labels=['事实','当前真值','当前读出p','预测p / 值','后续真值','后续读出p']
 for xx,s in zip(cols,labels):text(im,(xx,y+46),s,19)
 for j,name in enumerate(FN):
  yy=y+82+j*35;pred=row['prediction'][j];wrong=pred=='UNKNOWN'or pred!=int(row['truth_after'][j]);vals=[name,int(row['truth_before'][j]),f'{row["p_before"][j]:.4f}',f'{row["p_prediction"][j]:.4f} / '+('?'if pred=='UNKNOWN'else str(pred)),int(row['truth_after'][j]),f'{row["p_after"][j]:.4f}']
  for kk,(xx,s)in enumerate(zip(cols,vals)):text(im,(xx,yy),s,19,'#c62828'if kk==3 and wrong else'#172b4d')
 text(im,(815,y+330),'全7事实：'+('预测正确'if row['candidate_all7_correct']else'预测错误')+'；当前/后续读出：'+('均正确'if row['before_readout_correct']and row['after_readout_correct']else'至少一端错误'),20)
def main():
 q=frozen();rows=read(E/'results/call_details.json');cons=read(E/'results/root_contrasts.json');mm={m['root_id']:m for m in read(E/'results/pose_matching.json')};windows={w['window_id']:w for w in read(B/'data_manifest.json')['windows']};files=[]
 for c in cons:
  if not c['legally_covered_both_sides']:continue
  m=mm[c['root_id']];im=Image.new('RGB',(1700,1060),'white');text(im,(30,15),f'{c["root_id"]}：已持{CN[c["held_color"]]} → 请求抓{CN[c["target_color"]]}',32);text(im,(30,67),'同根开闭对照；冻结模型、颜色组合、合法准备、4帧2.1秒历史及96步保持均一致。',24)
  drawrow(im,c['sides']['closed'],120,windows);drawrow(im,c['sides']['open'],500,windows)
  text(im,(30,880),'匹配：'+('通过'if m['strictly_matched']else'未通过，不能作严格单因素因果结论')+f'。四帧最大物体位置差 {m["max_object_position_mm"]:.6f}mm / 姿态差 {m["max_object_orientation_deg"]:.6f}°',23)
  text(im,(30,916),f'机器人TCP位置差 {m["max_robot_TCP_position_mm"]:.6f}mm / 姿态差 {m["max_robot_TCP_orientation_deg"]:.6f}°；限值：位置1mm、姿态0.5°。',23)
  text(im,(30,953),'p为事实为真的概率，阈值0.5；1=真，0=假，?=UNKNOWN；多物体同时持有的阳性项均记UNKNOWN。',21)
  text(im,(30,990),'蓝/橙标题只表示抽屉闭/开；红色预测数值表示错误或UNKNOWN。真值仅用于图表评分，未输入模型。',21)
  path=E/'figures'/(c['root_id']+'.png');im.save(path);files.append(str(path))
 fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True);x=np.arange(len(cons));labels=[f'持{CN[c["held_color"]]}→抓{CN[c["target_color"]]}\n根{c["root_id"].split("_")[-1]}'for c in cons]
 for ax,key,title,actual in [(axes[0],'target_held_probability','预测目标物已持有的概率（实际两侧均为0）',0),(axes[1],'other_held_probability','预测原持物仍持有的概率（实际两侧均为1）',1)]:
  for drawer,offset,color,label in [('closed',-.13,'#2474b5','抽屉关闭'),('open',.13,'#d97706','抽屉打开')]:
   yy=[c['sides'].get(drawer,{}).get(key,np.nan)for c in cons];ax.scatter(x+offset,yy,c=color,s=70,label=label,zorder=3)
   for xx,y in zip(x+offset,yy):
    if np.isfinite(y):ax.annotate(f'{y:.3f}',(xx,y),xytext=(0,10 if y<.8 else-18),textcoords='offset points',ha='center',fontsize=9,color=color)
  for i,c in enumerate(cons):
   if c['legally_covered_both_sides']:ax.plot([i-.13,i+.13],[c['sides'][d][key]for d in ['closed','open']],color='#94a3b8',lw=1)
  ax.axhline(.5,color='#64748b',ls='--',lw=1,label='判定阈值0.5');ax.axhline(actual,color='#15803d',ls=':',lw=1.4,label='真实结果');ax.set_ylim(-.08,1.08);ax.set_ylabel('事实为真的预测概率',fontproperties=FP);ax.set_title(title,fontproperties=FP);ax.grid(axis='y',alpha=.2)
 axes[0].legend(prop=FP,loc='center right');axes[1].set_xticks(x,labels,fontproperties=FP);fig.suptitle('6个全新C根的抽屉开闭配对：同一冻结模型、同一抓取声明',fontproperties=FP,fontsize=16);fig.text(.5,.02,'横轴每组为同一布局根；灰线连接开/闭条件。蓝/橙表示抽屉条件，不表示物体颜色。多持物阳性冲突另计UNKNOWN。',ha='center',fontproperties=FP,fontsize=11);fig.tight_layout(rect=[0,.055,1,.95]);fig.savefig(E/'figures/paired_probabilities.png',dpi=150);plt.close(fig)
 # Every root is inspectable at full resolution; compact sheets are navigation only.
 for start in [0,3]:
  sheet=Image.new('RGB',(1700,1590),'#dce4ed')
  for j,path in enumerate(files[start:start+3]):sheet.paste(Image.open(path).resize((1700,530)),(0,j*530))
  sheet.save(E/'figures'/f'overview_{start//3+1}.png')
 save(E/'results/figure_manifest.json',{'pair_figures':files,'probability_figure':str(E/'figures/paired_probabilities.png'),'legend':'blue=drawerclosed,orange=draweropen;redprediction=wrong/UNKNOWN;actualRGBcolours=objectappearance','facts':NAMES,'probability_threshold':.5,'true_states_scoring_only':True})
 print('FIGURES_WRITTEN',len(files),flush=True)
if __name__=='__main__':main()
