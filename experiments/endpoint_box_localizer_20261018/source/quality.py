from common import *
import copy,collections,itertools,shutil
# Second visual pass from repeat_RGB_only.png; existing boxes were not displayed.
REPEAT={0:{'drawer_front':('visible',[[58,121,164,158]]),'gripper_finger':('partial',[[63,49,76,64],[78,48,91,66]])},5:{'drawer_front':('uncertain',[]),'gripper_finger':('partial',[[0,0,29,27],[55,0,83,26]])},10:{'drawer_front':('visible',[[53,170,146,199]]),'gripper_finger':('partial',[[69,132,79,142],[83,133,94,148]])},15:{'drawer_front':('uncertain',[]),'gripper_finger':('partial',[[8,0,35,25],[50,0,74,25]])},20:{'drawer_front':('visible',[[57,124,164,157]]),'gripper_finger':('partial',[[44,66,55,81],[59,65,72,82]])},25:{'drawer_front':('uncertain',[]),'gripper_finger':('partial',[[3,0,29,27],[55,0,83,26]])},30:{'drawer_front':('partial',[[54,176,151,200]]),'gripper_finger':('visible',[[111,89,124,101],[127,88,139,104]])},35:{'drawer_front':('uncertain',[]),'gripper_finger':('partial',[[1,0,31,26],[51,0,83,26]])}}
def iou(a,b):
 a=np.array(a);b=np.array(b);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();return float(inter/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter))
def main():
 assert not(E/'data/annotations/final.json').exists();cfg=read(E/'configs/protocol.json');rows=read(E/'data/public/images.json');rev=read(E/'data/annotations/reviewed.json');ann=rev['annotations'];record=[];ious=[];agreements=0;counts_equal=True
 for i in cfg['quality_gate']['blind_repeat_image_ids']:
  for j,c in enumerate(CATS):
   status,bb=REPEAT[i][c];a=ann[i]['classes'][j];agree=status==a['status'];agreements+=int(agree);counts_equal&=len(bb)==len(a['boxes']);pair=[]
   if bb and len(bb)==len(a['boxes']):
    perm=max(itertools.permutations(range(len(bb))),key=lambda p:sum(iou(x,bb[k])for x,k in zip(a['boxes'],p)));pair=[iou(x,bb[k])for x,k in zip(a['boxes'],perm)];ious.extend(pair)
   record.append({'image_id':i,'category':c,'reviewed_status':a['status'],'repeat_status':status,'status_equal':agree,'reviewed_boxes':a['boxes'],'repeat_boxes':bb,'matched_IoUs':pair})
 save(E/'data/annotations/repeat.json',{'utc':now(),'method':'same assistant repeat on RGB-only sheet without existing box display; not independent annotator agreement or human gold accuracy','records':record})
 final=copy.deepcopy(rev);cl=final['annotations'][30]['classes'][1];cl['status']='visible';cl['reason']='none';cl['note']='Repeat review: two visible fingers over black desk feature; no reliable evidence that a held object occludes their visible portions. No physical contact inference.'
 final['utc']=now();final['second_review_adjudication']={'image_id':30,'category':'gripper_finger','change':'partial/occluded -> visible/none; boxes unchanged','reason':'one visibility-status discrepancy resolved from RGB; agreement gate retains pre-adjudication15/16 result'};save(E/'data/annotations/final.json',final)
 trainids=[r['image_id']for r in rows if r['split']=='train'];summary=[];allvalid=True
 for j,c in enumerate(CATS):
  rr=[(r,final['annotations'][r['image_id']]['classes'][j])for r in rows];train=[(r,a)for r,a in rr if r['image_id']in trainids];boxes=sum(len(a['boxes'])for _,a in train);roots=len({r['root_id']for r,a in train if a['boxes']});unknown=sum(a['status']=='uncertain'for _,a in train)/len(train)
  summary.append({'category':c,'train_boxes':boxes,'train_episodes_with_boxes':roots,'train_uncertain_fraction':unknown,'all_status_counts':dict(collections.Counter(a['status']for _,a in rr))})
  allvalid&=boxes>=6 and roots>=3 and unknown<=.5
  for r,a in rr:
   assert(a['status']in['visible','partial'])==bool(a['boxes']);assert len(a['boxes'])<=(1 if j==0 else 2)
   for b in a['boxes']:assert len(b)==4 and 0<=b[0]<b[2]<=r['width']and 0<=b[1]<b[3]<=r['height']
 gate={'utc':now(),'same_assistant_repeat_only':True,'status_agreement':agreements/16,'matched_repeat_boxes':len(ious),'median_box_IoU':float(np.median(ious)),'p10_box_IoU':float(np.quantile(ious,.1)),'instance_counts_equal':counts_equal,'summary':summary,'all40overlays_reviewed':True,'two_coordinate_errors_corrected_before_repeat':True,'uncertain_classes_masked':True,'no_not_visible_examples':True}
 gate['passed']=bool(allvalid and counts_equal and gate['status_agreement']>=.9 and gate['median_box_IoU']>=.7 and gate['p10_box_IoU']>=.5);save(E/'results/annotation_quality_gate.json',gate)
 for p in Path(__file__).parent.glob('*.py'):
  if not(E/'source'/p.name).exists():shutil.copy2(p,E/'source'/p.name)
 save(E/'configs/annotations_frozen.json',{'utc':now(),'before_features_and_training':True,'sha256':{str(p):sha(p)for p in(E/'data/annotations').glob('*.json')},'gate_passed':gate['passed']})
 print('ANNOTATION_QUALITY_GATE',json.dumps(gate,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
