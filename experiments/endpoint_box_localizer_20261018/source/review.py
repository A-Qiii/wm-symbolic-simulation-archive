from common import *
import copy,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
 first=read(E/'data/annotations/round1.json');final=copy.deepcopy(first);fixes={10:[[70,132,79,142],[83,133,93,148]],26:[[86,103,100,116],[109,104,120,118]]};changes=[]
 for i,boxes in fixes.items():
  cl=final['annotations'][i]['classes'][1];changes.append({'image_id':i,'category':'gripper_finger','before':cl['boxes'],'after':boxes,'reason':'overlay review found vertical coordinate transcription error; zoomed original RGB confirms visible finger bounds; corrected before training, no prediction involved'});cl['boxes']=boxes
 final['utc']=now();final['review']='All40image overlays visually reviewed; two RGB coordinate errors corrected once before training; uncertainty and source selection unchanged';save(E/'data/annotations/reviewed.json',final);save(E/'data/annotations/review_changes.json',{'utc':now(),'changes':changes,'reviewed_pages':[f'round1_page_{i}.png'for i in range(1,6)],'no_model_predictions_used':True})
 shutil.copy2('/tmp/endpoint_box_detail.png',R/'figures/annotation_detail_corrections.png');rows=read(E/'data/public/images.json');fig,axs=plt.subplots(1,2,figsize=(10,5))
 for ax,i in zip(axs,fixes):
  r=rows[i];ax.imshow(np.load(r['path'])[r['key']]);
  for b in fixes[i]:x0,y0,x1,y1=b;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='magenta',lw=2))
  ax.set(title=f'Corrected id{i}',xticks=[],yticks=[])
 fig.tight_layout();fig.savefig(R/'figures/corrected_finger_boxes.png',dpi=140);plt.close(fig)
 ids=read(E/'configs/protocol.json')['quality_gate']['blind_repeat_image_ids'];fig,axs=plt.subplots(4,2,figsize=(10,16))
 for ax,i in zip(axs.flat,ids):
  r=rows[i];ax.imshow(np.load(r['path'])[r['key']],interpolation='nearest');step=10 if r['camera']==1 else 20;ax.set(title=f"REPEAT id{i} {r['scene']} {'static'if r['camera']==0 else'wrist'}",xticks=np.arange(0,r['width'],step),yticks=np.arange(0,r['height'],step));ax.grid(alpha=.35)
 fig.tight_layout();fig.savefig(R/'figures/repeat_RGB_only.png',dpi=140);plt.close(fig)
 print('ANNOTATION_REVIEW_SAVED_AND_REPEAT_RGB_READY',flush=True)
if __name__=='__main__':main()
