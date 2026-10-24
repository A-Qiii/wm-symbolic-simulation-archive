from common import *
import sys,shutil,copy
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
def main():
 assert not E.exists()and not R.exists();sys.addaudithook(label_guard)
 for sub in ['configs','data/observations','data/labels','data/evidence','data/audit','source','logs']:(E/sub).mkdir(parents=True)
 (R/'figures').mkdir(parents=True)
 original=read(BASE/'data/annotations/final.json');previous=read(PREV/'data/labels/user_semantics_v2.json');meta=read(BASE/'data/public/images.json')
 latest={a['image_id']:copy.deepcopy(a)for a in original['annotations']}
 for a in previous['annotations']:latest[a['image_id']]=copy.deepcopy(a)
 save(E/'data/labels/original_v0.json',original);save(E/'data/labels/assembled_reference_v2.json',{'meaning':'Original40 with the previously adjudicated8 user_semantics_v2 rows overlaid; synthetic reference, never used for training.','annotations':list(latest.values())})
 for name in ['user_transcript.json','panel_identity_mapping.json']:shutil.copy2(PREV/'data/evidence'/name,E/'data/evidence'/name)
 shutil.copy2(PREV/'configs/frozen_component_definitions.json',E/'configs/frozen_component_definitions.json')
 refs={str(BASE/'data/annotations/final.json'):sha(BASE/'data/annotations/final.json'),str(PREV/'data/labels/user_semantics_v2.json'):sha(PREV/'data/labels/user_semantics_v2.json')}
 out=[]
 for m in meta:
  p=Path(m['path']);refs[str(p)]=sha(p)
  with np.load(p)as z:assert set(z.files)=={'rgb_static','rgb_gripper'};rgb=z[m['key']]
  assert hashlib.sha256(rgb.tobytes()).hexdigest()==m['RGB_sha256']
  dest=E/f'data/observations/id{m["image_id"]:02d}.png';Image.fromarray(rgb).save(dest);out.append({**m,'png':str(dest)})
 save(E/'data/images.json',out);save(E/'configs/source_references.json',{'sha256':refs})
 save(E/'configs/protocol.json',{'utc':now(),'scope':'Audit all40 existing images, both classes, 80 cells. No collection, training, model execution, feature extraction, threshold tuning or old file changes.','definition':'Original component definitions, visible extent only; unknown class-image ignored entirely, not negative. Inclusive lower/exclusive upper integer RGB pixel coordinates.','evidence':'User identity evidence for4 specified panels, assistant full RGB/pixel review elsewhere; assistant labels are not independent human gold. Historical model context is disclosed. No saved output file access until new labels are frozen.','review':'Keep or fix only visually supportable boundaries, retain unknown for identity/boundary ambiguity. Prior8 decisions remain consistent unless new direct pixel evidence warrants a documented change.','splits':'Preserve exact original source episode/root splits; C/D are previously seen diagnostics.','score':'One saved prediction sensitivity evaluation: original/corrected decoder,224/448,seeds0/1,score>=.5,NMS already applied .3,IoU>=.5 greedy one-to-one. Fixed common-known support plus explicit additions/removals and full-known denominators. No model selection.','stop':'After frozen labels, supervision coverage, saved-output sensitivity, verification and bounded usability report.'})
 for k in range(0,40,4):
  fig,axs=plt.subplots(4,2,figsize=(12,20))
  for row,i in enumerate(range(k,k+4)):
   m=out[i];rgb=np.asarray(Image.open(m['png']));a=latest[i]
   for j in range(2):
    ax=axs[row,j];ax.imshow(rgb,interpolation='nearest',extent=(0,m['width'],m['height'],0));ax.set_title(f'ID {i:02d} | {"static"if m["camera"]==0 else"wrist"} | '+('raw RGB'if j==0 else'reference labels'));ax.set_xticks(np.arange(0,m['width']+1,20 if m['camera']==0 else 10));ax.set_yticks(np.arange(0,m['height']+1,20 if m['camera']==0 else 10));ax.grid(alpha=.15)
   for c in a['classes']:
    for x0,y0,x1,y1 in c['boxes']:axs[row,1].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#13bfdd'if c['category']=='drawer_front'else'#e047dc',lw=1.3))
   axs[row,1].set_xlabel(' | '.join(c['category']+':'+c['status']for c in a['classes']))
  fig.suptitle('Raw pixels and historical reference only. Cyan=drawer front; magenta=finger. x/y=original RGB pixels.',fontsize=12);fig.tight_layout(rect=(0,0,1,.98));fig.savefig(R/f'figures/review_{k:02d}_{k+3:02d}.png',dpi=120);plt.close(fig)
 save(E/'configs/pre_review_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in E.rglob('*')if p.is_file()}})
 print('Prepared40 immutable RGB copies and10 review sheets; no model outputs read.')
if __name__=='__main__':main()
