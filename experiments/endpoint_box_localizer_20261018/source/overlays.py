from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':9})
def main():
 rows=read(E/'data/public/images.json');aa=read(E/'data/annotations/round1.json')['annotations']
 for page in range(5):
  fig,axes=plt.subplots(4,2,figsize=(10,16))
  for ax,r in zip(axes.flat,rows[page*8:(page+1)*8]):
   ax.imshow(np.load(r['path'])[r['key']],interpolation='nearest');labels=[]
   for cl,col in zip(aa[r['image_id']]['classes'],['#00dd80','#ed35bb']):
    labels.append(('前板'if cl['category']=='drawer_front'else'夹指')+':'+cl['status'])
    for k,b in enumerate(cl['boxes']):
     x0,y0,x1,y1=b;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=col,lw=1.2));ax.text(x0,y0,str(k+1),color=col,fontsize=8)
   ax.set(title=f"id{r['image_id']:02d} {r['scene']} {'静态'if r['camera']==0 else'腕部'} | "+' / '.join(labels),xticks=[],yticks=[])
  fig.suptitle('首次RGB框标注复核 '+str(page+1)+'；绿色=前板与把手，粉色=逐根夹指',fontsize=13);fig.text(.5,.008,'框仅包可见部件；partial标记遮挡或出框，uncertain不画框且不作为负例。未用自动框、固定ROI或仿真状态生成标签。',ha='center',fontsize=9);fig.tight_layout(rect=(0,.025,1,.97));fig.savefig(R/f'figures/round1_page_{page+1}.png',dpi=140);plt.close(fig)
 print('FIRST_PASS_OVERLAYS_READY',flush=True)
if __name__=='__main__':main()
