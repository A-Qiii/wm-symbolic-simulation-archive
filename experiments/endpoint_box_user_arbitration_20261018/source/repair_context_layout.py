from common import *
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from PIL import Image
check_manifest(E/'configs/user_labels_and_audit_frozen.json');check_manifest(E/'configs/evaluation_frozen.json')
fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':10})
rgb=np.asarray(Image.open(E/'data/observations/id08.png'));fig,axs=plt.subplots(1,3,figsize=(15,5.5))
for ax,crop,title in zip(axs,[(0,0,200,200),(55,119,83,134),(35,80,120,150)],['完整原图：部件与场景位置','紧致外观裁切：连接信息被裁掉','扩大局部：保留掌部、蓝块及邻近灰指']):
 ax.imshow(rgb,interpolation='nearest');x,y,x1,y1=crop;ax.set_xlim(x-.5,x1-.5);ax.set_ylim(y1-.5,y-.5);ax.add_patch(Rectangle((55,120),26,13,fill=False,edgecolor='#c2442c',linestyle='--',linewidth=1.8));ax.set_title(title,fontsize=10);ax.set_xlabel('x 原图像素');ax.set_ylabel('y 原图像素')
fig.suptitle('同一图像、同一白段：用户确认为把手；紧裁可减少背景，也会删除身份判断所需上下文\n红虚框仅定位用户所指白段；右端跨邻近部件，不能整框直接作训练硬负例',fontsize=12);fig.tight_layout(rect=(0,.03,1,.87));fig.savefig(R/'figures/context_vs_tight_crop.png',dpi=160,bbox_inches='tight',pad_inches=.18);plt.close(fig)
shutil.copy2(__file__,E/'source'/Path(__file__).name)
p=R/'RUN_COMMANDS.md';p.write_text(p.read_text().replace('python user_arbitration_src/seal.py','python user_arbitration_src/repair_context_layout.py\npython user_arbitration_src/seal.py')+'\n视觉检查后，仅扩大上下文比较图的下边距，避免横轴标题被截；标签和结果哈希均保持冻结值。\n')
print('CONTEXT_FIGURE_MARGIN_REPAIRED_NO_LABEL_OR_SCORE_CHANGE')
