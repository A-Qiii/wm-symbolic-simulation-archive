from common import *
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from PIL import Image
# Presentation-only repair after visual inspection found an xlabel/title overlap.
check_manifest(E/'configs/labels_frozen.json');check_manifest(E/'configs/sensitivity_frozen.json')
fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':10})
old={x['image_id']:x for x in read(E/'data/labels/original_selected.json')['annotations']};ind={x['image_id']:x for x in read(E/'data/review/independent_annotations.json')['records']}
cases=[(0,'gripper_finger',(25,40,100,75),'三处灰白候选中，哪两处是夹指？'),(8,'gripper_finger',(45,105,108,143),'水平白段是否整段属于夹指？'),(1,'drawer_front',(0,48,84,84),'把手上方，哪些棕色像素属于前板？'),(9,'drawer_front',(0,40,84,84),'斜边在右下裁切处的前板范围？')]
fig,axs=plt.subplots(2,2,figsize=(14,11));fig.subplots_adjust(left=.06,right=.98,bottom=.09,top=.87,hspace=.6,wspace=.14)
for ax,(i,c,crop,q)in zip(axs.flat,cases):
 ax.imshow(np.asarray(Image.open(E/f'data/observations/id{i:02d}.png')),interpolation='nearest');ob=next(x['boxes']for x in old[i]['classes']if x['category']==c);ib=[ind[i]['drawer']['bbox_xyxy']]if c=='drawer_front'else[x['bbox_xyxy']for x in ind[i]['fingers']]
 for boxes,col,ls in[(ob,'#e08013','--'),(ib,'#007caa',':')]:
  for x,y,x1,y1 in boxes:ax.add_patch(Rectangle((x,y),x1-x,y1-y,fill=False,color=col,linestyle=ls,linewidth=2))
 x,y,x1,y1=crop;ax.set_xlim(x-.5,x1-.5);ax.set_ylim(y1-.5,y-.5);ax.set_xlabel('x 原图像素');ax.set_ylabel('y 原图像素');ax.set_title(f'id{i:02d}：{q}',fontsize=11,pad=12)
fig.legend(handles=[Line2D([0],[0],color='#e08013',ls='--',label='原标注候选'),Line2D([0],[0],color='#007caa',ls=':',label='独立复核候选')],loc='lower center',bbox_to_anchor=(.5,.015),ncol=2);fig.suptitle('需要外部部件辨认的4图：两种线型只区分来源，不表示正确与错误\n当前均为unknown；原图裁切、最近邻显示，不补像素、不显示模型输出',fontsize=13);fig.savefig(R/'figures/arbitration_cases.png',dpi=160);plt.close(fig)
shutil.copy2(__file__,E/'source'/Path(__file__).name)
p=R/'RUN_COMMANDS.md';p.write_text(p.read_text().replace('python adjudicate8_src/seal.py','python adjudicate8_src/repair_arbitration_layout.py\npython adjudicate8_src/seal.py')+'\n出图后检查发现未决问题图的一处横轴文字与下排标题靠得过近；仅增大行间距重新排版，未修改任何坐标、标签或分数。\n')
print('ARBITRATION_LAYOUT_REPAIRED_LABELS_AND_METRICS_UNCHANGED')
