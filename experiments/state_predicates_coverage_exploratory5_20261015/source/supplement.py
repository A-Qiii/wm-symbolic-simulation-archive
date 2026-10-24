from training_core import *
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';FP=FontProperties(fname=FONT)
def text(im,xy,s,size=22,col='#172b4d'):ImageDraw.Draw(im).text(xy,str(s),font=ImageFont.truetype(FONT,size),fill=col)
def main():
 dataset_frozen();r=read(E/'results/evaluation.json');ds=read(E/'results/call_details.json');meta=read(E/'data/training/_private/metadata.json');facts=[];per=[]
 for group in ['AB_coverage','C_drawer']:
  for seed in [0,1,2]:
   for arm in ['closed137','open137']:
    name=f'{arm}_seed{seed}';rows=[d for d in ds if d['model']==name and d['group']==group and d['drawer_condition']=='open'];pp=[d['probabilities'][COLORS.index(d['target_color'])]for d in rows];inwrong=sum(d['prediction'][j]==1 and not d['truth_after'][j]for d in rows for j in [3,4,5]);facts.append({'group':group,'model':name,'open_condition_calls':len(rows),'all7_correct':sum(d['all7_correct']for d in rows),'mean_target_held_probability':float(np.mean(pp)),'min_target_held_probability':min(pp),'max_target_held_probability':max(pp),'known_false_in_predictions':int(inwrong),'UNKNOWN_facts':sum(len(d['UNKNOWN_columns'])for d in rows)});
    for row,p in zip(rows,pp):per.append({'group':group,'model':name,'seed':seed,'arm':arm,'root_id':row['root_id'],'target_held_probability':p,'actual_target_held':0,'all7_correct':row['all7_correct']})
 save(E/'results/open_condition_details.json',facts);csvsave(E/'results/open_condition_details.csv',facts);csvsave(E/'results/target_held_probabilities.csv',per)
 fig,axes=plt.subplots(1,2,figsize=(14,5.5))
 for ax,g,title in zip(axes,['AB_coverage','C_drawer'],['新增打开训练条件（5根）','既有抽屉C的打开条件（6根）']):
  roots=list(dict.fromkeys(x['root_id']for x in per if x['group']==g));xs=np.arange(len(roots))
  for j,(arm,col,lab)in enumerate([('closed137','#2474b5','132+5关闭'),('open137','#d97706','132+5打开')]):
   yy=np.array([[next(x['target_held_probability']for x in per if x['group']==g and x['root_id']==rid and x['arm']==arm and x['seed']==s)for rid in roots]for s in [0,1,2]]);ax.bar(xs+(j-.5)*.28,yy.mean(0),.27,color=col,alpha=.7,label=lab)
   for s,mark in enumerate(['o','s','^']):ax.scatter(xs+(j-.5)*.28+(s-1)*.035,yy[s],color=col,marker=mark,edgecolor='white',s=35,zorder=3)
  ax.axhline(.5,color='#475569',ls='--',label='固定阈值0.5');ax.axhline(0,color='#15803d',ls=':',label='真实目标未持有=0');ax.set_ylim(-.05,1.05);ax.set_xticks(xs,[rr.replace('C_drawer_factor_','').replace('_coverage_','\n')for rr in roots],fontsize=9);ax.set_title(title,fontproperties=FP);ax.set_ylabel('预测目标已持有的概率',fontproperties=FP);ax.grid(axis='y',alpha=.15)
 axes[0].legend(prop=FP,loc='lower left');fig.suptitle('补打开条件降低部分错误概率，但未降到正确判定区间',fontproperties=FP,fontsize=15);fig.text(.5,.015,'这里所有调用都处于抽屉打开、已持另一物体状态，真实调用拒绝。蓝/橙指训练组；柱=3种子均值，圆/方/三角=种子0/1/2。不是独立泛化。',ha='center',fontproperties=FP,fontsize=10);fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(E/'figures/remaining_target_error.png',dpi=150);plt.close(fig)
 first=read(E/'results/single_replacement_diagnostic.json')['first_error'];assert first is not None;pid=first['pair_id'];control=next(x for x in ds if x['model']=='closed137_seed0'and x['pair_id']==pid);ref=next(x for x in ds if x['model']=='real_next_observation'and x['pair_id']==pid);cur=next(x for x in ds if x['model']=='persistence'and x['pair_id']==pid);row=meta[first['index']];windows={w['window_id']:w for w in read(CB/'data_manifest.json')['windows']};im=Image.new('RGB',(1700,1080),'white');text(im,(25,15),f'预先指定open137_seed0的首个C错误：{pid}',29);text(im,(25,62),'已持蓝、请求抓红；抽屉打开。真实拒绝，原持物和目标状态均未改变；当前与实际后续读出均正确。',23)
 for x,wid,lab in [(30,row['before_window_id'],'调用前真实图像'),(850,row['after_window_id'],'96步后真实图像')]:
  with np.load(windows[wid]['public_frames'][-1])as a:
   text(im,(x,110),lab,24);im.paste(Image.fromarray(a['rgb_static']).resize((290,290)),(x,150));im.paste(Image.fromarray(a['rgb_gripper']).resize((200,200)),(x+310,150));text(im,(x+310,365),'腕部相机',20);text(im,(x,450),'静态相机',20)
 xs=[30,210,420,690,1020,1370];labels=['事实','真实结果','当前读出p','关闭组p / 判定','打开组p / 判定','实际后续读出p'];names=['持红','持蓝','持粉','红入屉','蓝入屉','粉入屉','屉开']
 for x,s in zip(xs,labels):text(im,(x,505),s,21)
 for j,name in enumerate(names):
  vals=[name,first['truth_after'][j],f'{cur["probabilities"][j]:.5f}',f'{control["probabilities"][j]:.5f} / '+str(control['prediction'][j]).replace('UNKNOWN','?'),f'{first["probabilities"][j]:.5f} / '+str(first['prediction'][j]).replace('UNKNOWN','?'),f'{ref["probabilities"][j]:.5f}']
  for k,(x,s)in enumerate(zip(xs,vals)):text(im,(x,550+j*48),s,22,'#c62828'if k in [3,4]and (control if k==3 else first)['prediction'][j]!=first['truth_after'][j]else'#172b4d')
 text(im,(30,910),'0.5阈值固定；?=UNKNOWN。目标持有概率虽下降，仍与“原物仍持有”同时为阳性，触发既定冲突规则。',23);text(im,(30,955),'单项替换诊断：仅用已保存的真实后继编码、同一冻结读出头，可读对该例；这不是在线修复。',23);text(im,(30,1000),'没有因为本例追加更新、调阈值或更换种子。开/闭训练组相同预算，结果限于事后选择的五个配对条件。',23);im.save(E/'figures/first_error.png');print('SUPPLEMENT_WRITTEN',flush=True)
if __name__=='__main__':main()
