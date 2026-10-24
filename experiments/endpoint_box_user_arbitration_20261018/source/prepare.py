from common import *
import sys,shutil
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

def main():
 assert not E.exists()and not R.exists();sys.addaudithook(label_guard)
 for sub in['configs','source','data/labels','data/observations','data/evidence','data/audit','logs']:(E/sub).mkdir(parents=True)
 (R/'figures').mkdir(parents=True)
 ev={'source_thread_id':'01a14e2c-4647-734c-b1aa-a3a4547ec02a','delivery':'trusted conversation transcript relayed in delegation','user_message_id':'Sentinel_e8ac4202db30819186484f78c1871a42','user_message_utc':'2026-10-18T12:02:56+00:00','assistant_question_utc':'2026-10-18T11:44:18+00:00','library_file_id':'libfile_057f412211848191935b3d5681bc0795','file_id':'file_00000000a94881f6b522dda6d550d93f','figure':'arbitration_cases.png','verbatim_user_text':'1.橙色框的那两个是\n2.那个白色段是抽屉把手\n3.棕色木板都是抽屉啊，都会动\n4.那个斜木边就是抽屉前板\n感觉这个抽屉把手和夹爪手指比较像','meaning':'User supplied component identity/motion semantics, not exact pixel boxes; coordinate review remains assistant responsibility.'}
 save(E/'data/evidence/user_transcript.json',ev)
 old=read(BASE/'data/annotations/final.json');prev=read(V1/'data/labels/adjudicated_eval_v1.json');allmeta=read(BASE/'data/public/images.json');selection=[r for r in allmeta if r['image_id']in IDS]
 save(E/'data/labels/original_selected.json',{'annotations':[a for a in old['annotations']if a['image_id']in IDS]});save(E/'data/labels/assistant_v1_selected.json',prev)
 save(E/'data/audit/original_AB_annotations_readonly.json',{'annotations':[a for a in old['annotations']if a['image_id']<32]})
 save(E/'configs/frozen_component_definitions.json',read(V1/'configs/frozen_component_definitions.json'))
 srcfig=P/'reports/endpoint_box_label_adjudication8_20261018/figures/arbitration_cases.png';assert sha(srcfig)=='46940cfda9c94c50fc9053eb2f519d57ea4aa0f62ca608fc7d36212261f4524a';shutil.copyfile(srcfig,E/'data/evidence/arbitration_cases_as_user_saw.png')
 ind=read(V1/'data/review/independent_annotations.json');save(E/'data/evidence/independent_selected_annotations.json',ind)
 mapping=[{'panel':1,'position':'左上','image_id':0,'category':'gripper_finger','orange_original_boxes':[[65,49,77,65],[78,48,91,67]],'blue_independent_boxes':[[36,51,59,60],[64,51,77,64]],'user_fact':'橙色两框为夹指；不能把蓝色水平白条套入这个回答'}, {'panel':2,'position':'右上','image_id':8,'category':'gripper_finger','orange_original_boxes':[[68,116,81,132],[87,113,99,131]],'blue_independent_boxes':[[55,120,81,133],[90,115,97,129]],'user_fact':'水平白段为抽屉把手；需另找真实夹指，不推成全图无夹指'}, {'panel':3,'position':'左下','image_id':1,'category':'drawer_front','orange_original_boxes':[],'blue_independent_boxes':[[25,59,64,75]],'user_fact':'所问棕色木板会随抽屉移动；仍区分前板与腔底的可见范围'}, {'panel':4,'position':'右下','image_id':9,'category':'drawer_front','orange_original_boxes':[],'blue_independent_boxes':[[0,49,61,84]],'user_fact':'斜木边为抽屉前板；用户未给精确边界'}]
 oa={a['image_id']:a for a in old['annotations']};ia={a['image_id']:a for a in ind['records']}
 for x in mapping:
  a=next(c for c in oa[x['image_id']]['classes']if c['category']==x['category']);assert a['boxes']==x['orange_original_boxes']
  ib=[f['bbox_xyxy']for f in ia[x['image_id']]['fingers']]if x['category']=='gripper_finger'else[ia[x['image_id']]['drawer']['bbox_xyxy']];assert ib==x['blue_independent_boxes']
 save(E/'data/evidence/panel_identity_mapping.json',mapping)
 refs={str(srcfig):sha(srcfig),str(BASE/'data/annotations/final.json'):sha(BASE/'data/annotations/final.json'),str(V1/'data/labels/adjudicated_eval_v1.json'):sha(V1/'data/labels/adjudicated_eval_v1.json'),str(V1/'source/repair_arbitration_layout.py'):sha(V1/'source/repair_arbitration_layout.py')}
 out=[]
 for r in allmeta:
  i=r['image_id']
  if i>=32 and i not in IDS:continue
  path=Path(r['path']);refs[str(path)]=sha(path)
  with np.load(path)as z:assert set(z.files)=={'rgb_static','rgb_gripper'};rgb=z[r['key']]
  assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['RGB_sha256']
  dest=E/f'data/observations/id{i:02d}.png';Image.fromarray(rgb).save(dest);out.append({**r,'png':str(dest)})
 save(E/'data/images_selected8.json',[r for r in out if r['image_id']in IDS]);save(E/'data/audit/AB_images_readonly.json',[r for r in out if r['image_id']<32]);save(E/'configs/source_references.json',{'sha256':refs})
 save(E/'configs/protocol.json',{'utc':now(),'primary_ids':IDS,'only_changed_ids':[0,1,8,9],'user_semantics_not_pixel_gold':True,'no_model_output_file_reads_until_labels_and_readonly_audit_frozen':True,'historical_model_context_disclosed':True,'audit_scope':'Read-only32 A/B training/development images for repeated handle/finger identity confusion; only4user-disputed images may change in new8image eval version. The other28 ABimages receive suspicion ledger only, not new labels or metric recomputation. Remaining4 C/Dimages outside8 not reviewed. No claim remaining32fully audited.','labels':'Use user component evidence plus original RGB and original definitions. Unknown remains if a reliable visible boundary cannot be determined; do not invent invisible parts or infer not_visible from one negative candidate.','comparisons':'Saved original/corrected decoder outputs at224/448,seeds0/1; score>=.5,existingNMS.3,matchingIoU.5. No decoder reexecution. Compare user_v2 versus assistant_v1 and original on fixed common-known support, newly included support, and full known selected8 separately.','new_work':'No training,features,forward,threshold change or architecture change. No automatic batch edits, no old-file modification.','method_goal':'State support for handle/finger hard-negative and local parent-connection checking; tightly cropped appearance alone can lose context; this evidence does not establish a required architecture replacement.','stop':'Freeze decisions, score saved outputs, report concrete confirmed/suspected/unverified items and a minimal proposed next step; do not train.'})
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 save(E/'configs/pre_review_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in E.rglob('*')if p.is_file()}})
 fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':10})
 # Full image retains parent/scene context; crop includes palm and existing labels only.
 for r in out:
  i=r['image_id']
  if i>=32:continue
  rgb=np.asarray(Image.open(r['png']));c=next(c for c in oa[i]['classes']if c['category']=='gripper_finger');bs=c['boxes']
  if r['camera']==0:
   if bs:x0=max(0,min(b[0]for b in bs)-30);x1=min(200,max(b[2]for b in bs)+20);y0=max(0,min(b[1]for b in bs)-45);y1=min(200,max(b[3]for b in bs)+20)
   else:x0,x1,y0,y1=0,135,30,165
  else:x0,x1,y0,y1=0,84,0,40
  fig,ax=plt.subplots(1,2,figsize=(12,5))
  ax[0].imshow(rgb,interpolation='nearest');ax[0].set_title('完整原始RGB：保留连接上下文');ax[1].imshow(rgb,interpolation='nearest');ax[1].set_xlim(x0-.5,x1-.5);ax[1].set_ylim(y1-.5,y0-.5);ax[1].set_xticks(np.arange((x0//5)*5,x1,5));ax[1].set_yticks(np.arange((y0//5)*5,y1,5));ax[1].grid(alpha=.2)
  for j,b in enumerate(bs):
   x,y,x1b,y1b=b;ax[1].add_patch(Rectangle((x,y),x1b-x,y1b-y,fill=False,color='#b431b5',lw=1.5));ax[1].text(x,y,str(j),fontsize=10,color='#b431b5',bbox={'facecolor':'white','alpha':.7,'edgecolor':'none'})
  ax[1].set_title('原封存夹指标注（紫框），非模型输出；'+c['status'])
  for a in ax:a.set_xlabel('x 原图像素');a.set_ylabel('y 原图像素')
  fig.suptitle(f'id{i:02d} | {r["split"]} | {r["root_id"]}',fontsize=12);fig.tight_layout();fig.savefig(R/f'figures/audit_id{i:02d}.png',dpi=150);plt.close(fig)
 print('USER_ARBITRATION_INPUTS_READY',len(out),'existing RGB images;',len(mapping),'panel mappings checked')
if __name__=='__main__':main()
