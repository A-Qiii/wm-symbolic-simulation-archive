from common import *
import sys,copy,shutil,collections

def main():
 sys.addaudithook(label_guard);check_manifest(E/'configs/pre_review_frozen.json');assert not(E/'data/labels/user_semantics_v2.json').exists()
 prev=read(E/'data/labels/assistant_v1_selected.json');new=copy.deepcopy(prev);aa={a['image_id']:a for a in new['annotations']};old={a['image_id']:a for a in read(E/'data/labels/original_selected.json')['annotations']};dec=[]
 def change(i,cat,status,reason,boxes,note,semantic,remaining,candidates=None):
  c=next(c for c in aa[i]['classes']if c['category']==cat);before=copy.deepcopy(c);c.update(status=status,reason=reason,boxes=boxes,note=note)
  dec.append({'image_id':i,'category':cat,'before_assistant_v1':before,'after_user_semantics_v2':copy.deepcopy(c),'user_semantic_fact':semantic,'assistant_pixel_reasoning':note,'remaining_boundary_uncertainty':remaining,'unscored_visible_candidates':candidates or[],'source_user_message_id':'Sentinel_e8ac4202db30819186484f78c1871a42'})
 change(0,'gripper_finger','partial','occluded',[[65,49,77,65],[78,48,91,67]],'用户左上面板回答明确对应橙色两框，即原标注的两条斜灰指，不是蓝色水平白条。原图两条灰指分别接在掌部下缘并位于蓝块两侧；原橙框覆盖可见指体，维持原坐标与partial。','橙色框的那两个是夹指','用户确认部件身份，不是逐像素边界；掌部连接处和物体接触处仍有约1–2像素不确定性。')
 change(8,'gripper_finger','uncertain','not_identifiable',[],'用户确认水平白段为抽屉把手，故独立候选[55,120,81,133]不能作为夹指。原左框[68,116,81,132]也包含该把手的大块可见像素，不能照抄。另检查到蓝块右侧确有灰色指体；蓝块左侧较短灰段与把手投影相叠，其与把手/掌部的分界仍无法稳健界定。原评分模式按整类忽略unknown，因此保留本类unknown，不能改成not_visible或只标右指后惩罚其余未知实例。','那个白色段是抽屉把手','已解决白段身份，未解决另一根夹指的完整可见范围；不把一处硬负例推成整图负例。',[{'role':'assistant_visible_right_finger_candidate','bbox_xyxy':[88,114,98,130],'scored':False,'reason':'右侧指体可见，但整类因另一指边界不确定而忽略'},{'role':'user_identified_handle_not_finger','bbox_xyxy':[55,120,81,133],'scored':False,'reason':'该矩形只是用户所指水平白段的位置索引，右端也可能跨过相邻指体，不能用整矩形作干净的训练负样本'}])
 change(1,'drawer_front','uncertain','not_identifiable',[],'用户确认所问棕色木板随抽屉运动，排除了先前把它直接当固定桌沿的语义猜测。但原图棕色表面跨出裁切上缘，前板与腔底/其它抽屉木面的分界并无清楚像素接缝。原定义排除腔底；不能把“会动的抽屉木板”自动扩为整个前板，也不能把裁切上缘y48或任意窄带当真实部件上边界。黑U形把手可界定，但只给把手框会更换目标定义，故保留前板整体unknown。','棕色木板都是抽屉，会动','父对象与运动语义已确认；前板/腔底之间的可见边界尚未确认。用户未提供精确像素边界，不虚构新框。',[{'role':'visible_pull_handle_only','bbox_xyxy':[25,58,63,75],'scored':False,'reason':'仅把手的定位提示，不替代前板连同把手的完整可见外接框'}])
 change(9,'drawer_front','partial','out_of_frame',[[0,47,84,84]],'用户确认左下把手连接的斜木边是前板。可见斜边从左界约y47进入并向右下延伸至图像裁切处，附着黑把手也在左下裁切；独立候选右界x61截断了右下仍可见前板。按原定义取其可见前板加把手外接框[0,47,84,84]，不补画外部分。','那个斜木边就是抽屉前板','上缘转角和木纹交界约1–2像素不确定；用户只确认身份，坐标由助手按原RGB标定。')
 # Only the four disputed class-image cells may change; all other12cells are exact copies.
 b={(a['image_id'],c['category']):c for a in prev['annotations']for c in a['classes']};n={(a['image_id'],c['category']):c for a in new['annotations']for c in a['classes']};allowed={(0,'gripper_finger'),(8,'gripper_finger'),(1,'drawer_front'),(9,'drawer_front')}
 for k in n:
  if k not in allowed:assert n[k]==b[k]
 meta={r['image_id']:r for r in read(E/'data/images_selected8.json')}
 for (i,cat),c in n.items():
  if c['status']=='uncertain':assert not c['boxes']
  for x,y,x1,y1 in c['boxes']:assert 0<=x<x1<=meta[i]['width']and 0<=y<y1<=meta[i]['height']
 new.update(version='user_semantics_v2',created_utc=now(),status='User-confirmed semantics for4disputed cases plus assistant visible-boundary review; not user-drawn pixel gold;2boundaries remain unknown',inherits_unchanged_other12class_images_from='assistant_eval8_v1',user_message_id='Sentinel_e8ac4202db30819186484f78c1871a42')
 save(E/'data/labels/user_semantics_v2.json',new);save(E/'data/labels/user_decision_ledger.json',dec)
 # Read-only AB review ledger. These classifications are NOT new training/evaluation labels.
 rows=read(E/'data/audit/AB_images_readonly.json');annotations={a['image_id']:a for a in read(E/'data/audit/original_AB_annotations_readonly.json')['annotations']};audit=[]
 descriptions={
 0:('user_confirmed_correct_instances','用户确认原橙框两根夹指；独立复核把水平白候选列为夹指不成立。原训练标签未采用该独立候选。','语义解决；本次仅恢复两指评估标签'),
 1:('confirmed_extent_error_previous_v1_fixed','原左/右夹指框左右明显超出灰指可见范围；左框还包含另一暗色背景结构。是过宽边界问题，不能据此推断原标注者把完整把手当夹指。','v1已有坐标修正，本轮不再次修改夹指框'),
 4:('suspected_connection_boundary','左夹指近端与邻近水平白结构投影相接；原框左上角可能包含非指结构，需核定指体从掌部末端开始的位置。未确认其整根身份错误。','优先核验近端分界；不自动改框'),
 8:('confirmed_nonfinger_pixels_in_left_extent','用户确认水平白段为把手；原左框[68,116,81,132]内大量像素属于该段，而非仅夹指。不能将独立扩大的水平白框作正例。真实右指仍可见，另一指边界未定。','保留整类unknown，保存用户硬负例身份提示但不生成训练负框'),
 12:('needs_new_visibility_review','原静态夹指为unknown；掌部附近白段、下方灰条及大黑把手相互邻近，当前没有逐指可见范围标注。没有把它当全图负例。','需以完整连接上下文核验；不因同类白段外形自动归类'),
 13:('confirmed_loose_extent_includes_handle','腕部原左框[6,0,41,33]、右框[42,0,63,32]下端延到黑U把手，而灰指可见尖端约y26。左框还覆盖大片远离灰指的左侧背景。确认可见范围框过宽，不等价于确认两根手指的类别身份都错。','训练原文件保持封存；先确认实际灰指边界后另版修正'),
 21:('confirmed_extent_misalignment_near_handle','开发腕部右框[50,0,74,26]的右侧大块区域是暗色背景把手；浅灰指上缘右端约x69，故右界明显过宽。矩形内部含背景本身不必然是错误，这里依据的是指体在该端没有对应可见像素。','只登记，未更新开发标签或开发分数'),
 22:('needs_new_visibility_review','开发静态夹指原为unknown；红块两侧有灰条但掌部/物体接触处边界尚需核验。邻近水平白段不因外观相似就标为夹指。','保留原unknown，需后续逐指审查'),
 28:('suspected_connection_boundary','开发静态左指近端与掌部下方水平白段投影接近，原框从[48,63]起是否混入连接以外结构仍有疑问。下方细灰指体本身可见，未确认整实例错标把手。','先核验近端连接范围，不改现有框'),
 30:('needs_new_occlusion_extent_review','开发静态左框[113,90,125,104]大部分为棕色背景，细灰段主要在上右角并与另一指投影接近；visible状态及可见长度需核验。水平白段在框左侧，未见把它直接作为这个实例。','核验遮挡与范围，不宣称已确认同一身份错误'),
 31:('confirmed_loose_extent_nonfinger_background','开发腕部两框下界33/34，而浅灰指尖约y26；多出的下部包含黑色物体/棕色背景。确认过宽范围，不能把该背景都称为把手。','只读登记，未修改开发标签')}
 for r in rows:
  i=r['image_id'];c=next(c for c in annotations[i]['classes']if c['category']=='gripper_finger')
  if i in descriptions:kind,note,nextstep=descriptions[i]
  elif r['camera']==0:kind='no_repeated_identity_error_observed';note='本图原框指向掌部远端灰指，邻近水平白段未被单独作为完整夹指实例；本次没有确认与用户第2图相同的身份错误。这不是逐像素正确性认证。';nextstep='保留原标注；正式使用前仍需完成完整标签审计'
  else:kind='no_repeated_identity_error_observed';note='两个上缘裁切浅灰指体与原标注对应；本次没有确认把下方黑把手作为完整夹指实例。边界精度及其它类别不在本项结论内。';nextstep='保留原标注；正式使用前仍需完成完整标签审计'
  audit.append({k:r[k]for k in ['image_id','split','root_id','state_id','camera']}|{'review_scope':'read-only handle/finger identity and obvious extent contamination, not complete label audit','status':kind,'original_finger_status':c['status'],'original_boxes':c['boxes'],'evidence':note,'next_verification':nextstep,'original_label_modified':False,'figure':str(R/f'figures/audit_id{i:02d}.png')})
 save(E/'data/audit/AB_readonly_findings.json',audit)
 save(E/'data/audit/confirmed_external_candidate_errors.json',[{'image_id':0,'source':'independent8review, not used by original training','candidate':[36,51,59,60],'finding':'用户确认橙色两根为夹指，该蓝色水平候选不能替换其中一根','training_contamination_claim':False},{'image_id':8,'source':'independent8review, not used by original training','candidate':[55,120,81,133],'finding':'用户明确白水平段是把手；该候选作为夹指身份错误','training_contamination_claim':False}])
 shutil.copy2(__file__,E/'source'/Path(__file__).name)
 frozen=[E/'data/labels/user_semantics_v2.json',E/'data/labels/user_decision_ledger.json',E/'data/audit/AB_readonly_findings.json',E/'data/audit/confirmed_external_candidate_errors.json',E/'source/freeze_labels.py',E/'configs/pre_review_frozen.json']
 save(E/'configs/user_labels_and_audit_frozen.json',{'utc':now(),'model_output_files_not_read_this_round_before_freeze':True,'historical_context_disclosed':True,'sha256':{str(p):sha(p)for p in frozen}})
 print('USER_SEMANTICS_V2_AND_READONLY_AUDIT_FROZEN',sum(c['status']!='uncertain'for c in n.values()),'knownclassimages',sum(len(c['boxes'])for c in n.values()),'boxes')
 print('AUDIT_COUNTS',json.dumps(dict(collections.Counter(r['status']for r in audit)),ensure_ascii=False))
if __name__=='__main__':main()
