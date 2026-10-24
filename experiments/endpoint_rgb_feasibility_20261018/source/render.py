from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':8})
def main():
 rows=read(E/'data/states.json');pred=read(E/'results/part_matches.json');templates=read(E/'configs/templates.json')['annotations'];cols={'drawer_handle':'#00c87e','gripper_finger':'#e62eaf'}
 for page in range(2):
  rr=rows[page*7:(page+1)*7];fig,axes=plt.subplots(7,2,figsize=(8,16))
  for i,r in enumerate(rr):
   z=np.load(r['path'])
   for cam,key in enumerate(['rgb_static','rgb_gripper']):
    ax=axes[i,cam];im=z[key];ax.imshow(im,interpolation='nearest')
    pp=[p for p in pred if p['index']==r['index']and p['camera']==cam]
    for p in pp:ax.scatter(p['x'],p['y'],marker='o'if p['target']=='drawer_handle'else'x',s=60,facecolors='none'if p['target']=='drawer_handle'else cols[p['target']],edgecolors=cols[p['target']],linewidths=1.6)
    for t in templates:
     if t['index']==r['index']and t['camera']==cam:
      x0,y0,x1,y1=t['box'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=cols[t['target']],ls='--',lw=1))
    label=' / '.join(('把手'if p['target']=='drawer_handle'else'夹指')+f" {p['top_cosine']:.2f}"for p in pp)
    ax.set(title=f"{r['scene']} idx{r['index']} {'静态'if cam==0 else'腕部'} | {label}",xticks=[],yticks=[])
  fig.suptitle(f'部件参考匹配 {page+1}/2：每张图强制取最高相似patch，尚无缺失判定',fontsize=12)
  fig.text(.5,.008,'绿色圆=抽屉把手参考的最佳patch；粉色叉=夹指参考。虚框仅为A/B人工参考框。数字是余弦相似度，不是检测置信度；非边界、非身份保证。',ha='center',fontsize=8)
  fig.tight_layout(rect=(0,.025,1,.975),h_pad=1.2);fig.savefig(R/f'figures/part_matches_{page+1}.png',dpi=140);plt.close(fig)
 context=read(E/'results/context_sensitivity.json');fig,axes=plt.subplots(1,2,figsize=(10,4))
 x=np.arange(4);axes[0].bar(x,[r['full_image_ROI_cosine']for r in context],color='#377eb8');axes[0].axhline(1,c='#555555',ls='--');axes[0].set(xticks=x,xticklabels=['把手A','把手B','夹指A','夹指B'],ylim=(0,1.05),ylabel='区域特征余弦相似度',title='同一局部patch像素，改变其他patch')
 for i,r in enumerate(context):axes[0].text(i,r['full_image_ROI_cosine']+.02,f"{r['full_image_ROI_cosine']:.3f}",ha='center')
 axes[1].bar(x,[r['crop_reencoding_max_abs_diff']for r in context],color='#e99d38');axes[1].set(xticks=x,xticklabels=['把手A','把手B','夹指A','夹指B'],ylim=(0,.01),ylabel='局部裁剪重新编码的最大绝对差',title='先裁剪再编码：外部像素不进入网络')
 for i,r in enumerate(context):axes[1].text(i,.0003,str(r['crop_reencoding_max_abs_diff']),ha='center')
 fig.text(.5,.015,'左：蓝柱比较完整图与其余patch置灰后的ROI，虚线1表示完全同向。右：橙柱比较字节相同的裁剪输入；零差仅验证隔离外部像素，不证明定位正确或跨视角一致。',ha='center',fontsize=8);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(R/'figures/local_crop_vs_context.png',dpi=150);plt.close(fig)
 print('ENDPOINT_FEASIBILITY_FIGURES_READY',flush=True)
if __name__=='__main__':main()
