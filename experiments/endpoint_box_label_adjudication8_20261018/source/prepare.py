from common import *
import shutil,sys
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

def main():
 assert not E.exists()and not R.exists()
 for sub in['configs','source','data/observations','data/labels','data/review','logs']:(E/sub).mkdir(parents=True,exist_ok=True)
 (R/'figures').mkdir(parents=True)
 sys.addaudithook(label_guard)
 selection=read(REVIEW/'selection.json');assert selection['selected_ids']==IDS
 old=read(BASE/'data/annotations/final.json');ann=[x for x in old['annotations']if x['image_id']in IDS]
 definitions=read(BASE/'configs/protocol.json')['annotation'];save(E/'data/labels/original_selected.json',{'annotations':ann,'source_final_sha256':sha(BASE/'data/annotations/final.json')})
 save(E/'configs/frozen_component_definitions.json',definitions)
 refs={str(BASE/'data/annotations/final.json'):sha(BASE/'data/annotations/final.json'),str(BASE/'configs/protocol.json'):sha(BASE/'configs/protocol.json')}
 for name in['REPORT.md','selection.json','independent_annotations.json','comparison.json','comparison_contact_sheet.png']:
  p=REVIEW/name;shutil.copy2(p,E/'data/review'/name);refs[str(p)]=sha(p)
 rows=[]
 for r in selection['metadata']:
  src=Path(r['path']);dst=E/'data/observations'/src.name
  if not dst.exists():shutil.copy2(src,dst)
  refs[str(src)]=sha(src);new={**r,'original_path':str(src),'path':str(dst)};rows.append(new)
  with np.load(dst)as z:assert set(z.files)=={'rgb_static','rgb_gripper'};rgb=z[r['key']]
  assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['RGB_sha256']
  Image.fromarray(rgb).save(E/f'data/observations/id{r["image_id"]:02d}.png')
 save(E/'data/images.json',rows)
 save(E/'configs/protocol.json',{'utc':now(),'selected_ids':IDS,'scope':'Only8RGB label adjudication and sensitivity of previously saved predictions; no new model forward, training, feature extraction or data',
  'adjudicator_exposure':'Same project assistant has historical model-result context, so this is not blind independent ground truth. Current decisions use only selected rawRGB, frozen component definitions and both label sets; no model-output files are read before label freeze.',
  'decision_rule':'Keep/correct only when visible pixels support identity and extent under original definitions. Record identity/extent disputes as class-leveluncertain/unknown. No voting or averaging coordinates. No hallucinated hidden completion.',
  'partial':'Do not require full part visibility; when visible part identity/extent is reliable, cropped/occluded remainspartial. When boundary or identity cannot be adjudicated, useuncertain, notnot_visible.',
  'unchanged_policy':'Only eight selected records get a separate evaluation-labelversion. Neither original nor independent labels are overwritten; unresolved proposed boxes excluded from scoring.',
  'sensitivity':'After label freeze, use both saved original and corrected decoder predictions at224/448,seeds0/1. Score only selected8 by split/category with fixed thresholds and greedy1:1IoU.5; all retained unmatched predictions areFP on knownclassimages.',
  'denominators':'Report original-selected versus adjudicated-selected and common-known class-image support separately. List added/dropped class-images/instances. Selected set has4train,2Cseen,2Dseen images,0ABdevelopment. Do not infer full40metrics.',
  'status_of_full_metrics':'Label audit of other32 images incomplete. Prior full-data scores remain provisional and unchanged; do not replace full-data figures with8image sensitivity.',
  'stop':'Freeze proposals, report unresolved images requiring external domain adjudication if they are to enter scoring, calculate bounded sensitivity, seal and stop. No additional rounds of assistant voting.'})
 save(E/'configs/source_references.json',{'sha256':refs})
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 save(E/'configs/pre_adjudication_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in E.rglob('*')if p.is_file()}})
 fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':10})
 ind={x['image_id']:x for x in read(E/'data/review/independent_annotations.json')['records']};oa={x['image_id']:x for x in ann}
 # Full raw RGB and two sets shown in separate panels, no model prediction.
 for i in IDS:
  r=next(r for r in rows if r['image_id']==i)
  with np.load(r['path'])as z:rgb=z[r['key']]
  fig,axes=plt.subplots(1,3,figsize=(15,5))
  for j,ax in enumerate(axes):
   ax.imshow(rgb,interpolation='nearest');ax.set_xlim(-.5,r['width']-.5);ax.set_ylim(r['height']-.5,-.5);ax.set_xticks(np.arange(0,r['width'],20 if r['width']==200 else 10));ax.set_yticks(np.arange(0,r['height'],20 if r['height']==200 else 10));ax.grid(alpha=.25)
   if j==1:sets=[c['boxes']for c in oa[i]['classes']]
   elif j==2:sets=[[ind[i]['drawer']['bbox_xyxy']],[x['bbox_xyxy']for x in ind[i]['fingers']]]
   else:sets=[]
   for ci,boxes in enumerate(sets):
    for b in boxes:
     x0,y0,x1,y1=b;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color=['#19a266','#bf35cc'][ci],linewidth=1.8))
   ax.set_title(['原始RGB','原冻结标注','独立复核标注'][j])
  fig.suptitle(f'id{i:02d}：绿=前板及把手；紫=夹指；坐标为原图像素，未显示模型输出',fontsize=14);fig.tight_layout();fig.savefig(R/f'figures/review_id{i:02d}.png',dpi=150);plt.close(fig)
 # Larger raw-only crops with original-coordinate ticks for boundary inspection.
 crops={0:[(25,40,100,75),(35,110,180,170)],1:[(0,0,84,34),(0,50,84,84)],8:[(45,105,108,143),(25,150,170,200)],9:[(0,0,84,32),(0,40,84,84)],32:[(0,75,76,130),(15,150,155,200)],33:[(0,0,84,32),(0,24,84,84)],38:[(38,75,95,132),(35,150,170,200)],39:[(0,0,84,32),(30,5,80,80)]}
 for i in IDS:
  r=next(r for r in rows if r['image_id']==i)
  with np.load(r['path'])as z:rgb=z[r['key']]
  fig,axes=plt.subplots(1,2,figsize=(12,6))
  for ax,box,title in zip(axes,crops[i],['夹指区域原始像素','前板区域原始像素']):
   x0,y0,x1,y1=box;ax.imshow(rgb,interpolation='nearest');ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5);ax.set_xticks(np.arange((x0//5)*5,x1+1,5));ax.set_yticks(np.arange((y0//5)*5,y1+1,5));ax.grid(alpha=.35);ax.set_title(title)
  fig.suptitle(f'id{i:02d} 原图裁切放大：仅RGB，网格每5像素；无插值增强/模型框');fig.tight_layout();fig.savefig(R/f'figures/pixels_id{i:02d}.png',dpi=160);plt.close(fig)
 print('EIGHT_IMAGE_LABEL_REVIEW_INPUTS_READY',flush=True)
if __name__=='__main__':main()
