from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':9})
def main():
 rows=read(E/'data/public/images.json')
 for page in range(5):
  fig,axes=plt.subplots(4,2,figsize=(10,16))
  for ax,r in zip(axes.flat,rows[page*8:(page+1)*8]):
   rgb=np.load(r['path'])[r['key']];ax.imshow(rgb,interpolation='nearest');step=10 if r['camera']==1 else 20;ax.set(title=f"id{r['image_id']:02d} {r['scene']} idx{r['index']} {'静态'if r['camera']==0 else'腕部'}",xticks=np.arange(0,r['width'],step),yticks=np.arange(0,r['height'],step));ax.grid(alpha=.3)
  fig.suptitle('原始RGB标注页 '+str(page+1)+'；网格单位为原图像素',fontsize=14);fig.tight_layout(rect=(0,0,1,.97));fig.savefig(R/f'figures/raw_page_{page+1}.png',dpi=140);plt.close(fig)
 print('RGB_ANNOTATION_SHEETS_READY',flush=True)
if __name__=='__main__':main()
