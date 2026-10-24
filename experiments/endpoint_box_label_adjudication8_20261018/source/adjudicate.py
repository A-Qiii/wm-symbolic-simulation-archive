from common import *
import sys,copy,shutil

def main():
 sys.addaudithook(label_guard)
 check_manifest(E/'configs/pre_adjudication_frozen.json')
 dst=E/'data/labels/adjudicated_eval_v1.json';assert not dst.exists()
 old=read(E/'data/labels/original_selected.json')['annotations'];by={x['image_id']:x for x in old}
 decisions=[]
 def put(i,cat,status,reason,boxes,action,evidence,unresolved=''):
  c=next(x for x in by[i]['classes']if x['category']==cat)
  decisions.append({'image_id':i,'category':cat,'action':action,'original':copy.deepcopy(c),'adjudicated':{'category':cat,'status':status,'reason':reason,'boxes':boxes,'note':evidence},'pixel_evidence':evidence,'remaining_uncertainty':unresolved,'evidence_files':[str(R/f'figures/pixels_id{i:02d}.png'),str(R/f'figures/review_id{i:02d}.png')]})
 # Coordinates are decisions from displayed original pixels, not predictions.
 put(0,'drawer_front','visible','none',[[61,126,165,159]],'correct_visible_extent','闭合抽屉的棕色竖向前面及黑把手可识别。旧框从(53,121)起包含上方桌沿；面板上缘约y126、左缘约x61，右端约x164，把手最低像素约y158。修正为可见面板与把手的外接框。','边缘抗锯齿仍约1–2像素不确定性；不是逐像素分割真值。')
 put(0,'gripper_finger','uncertain','not_identifiable',[],'exclude_unresolved_identity','原标注选蓝块两侧的斜灰条，独立标注选左侧水平白条与其中一条斜灰条。原图有三个候选区域，水平条是否为远侧夹指/背景结构、蓝块上白条是否为夹指不能仅凭这张RGB可靠裁决。不得按模型匹配程度选择。','需外部辨认(36,51)–(59,60)、(64,51)–(77,64)、(78,48)–(91,67)三处哪个属于两根夹指；本类整图unknown。')
 put(1,'drawer_front','uncertain','not_identifiable',[],'retain_unknown','黑色U形把手清楚，但独立框[25,59,64,75]主要只包把手。上方棕色薄边与固定桌沿/前板的分界不能从本图可靠界定。定义要求可见前板连同把手，不能默默改为把手检测。','需外部指认可见前板的边界；不是把手不存在，也不是必须完整可见才可标。')
 put(1,'gripper_finger','partial','out_of_frame',[[11,0,35,27],[49,0,70,26]],'correct_visible_extent','两条浅灰夹指均从图像上缘进入并夹住蓝块。左端可见灰条约从x11开始，旧x2空含背景；右夹指上端到x69，旧x78含背景。末端分别到y26、y25；只标可见条体，左侧另一个暗灰背景结构不作为夹指。')
 put(8,'drawer_front','partial','out_of_frame',[[43,164,154,200]],'correct_visible_extent','抽屉前板的左上角在约(43,164)，前缘横向延伸至右下约x153；黑把手在底部被画面裁切。旧框(55,173)起漏掉可见左上前板，修正包含该前缘，不补全画外把手。')
 put(8,'gripper_finger','uncertain','not_identifiable',[],'exclude_unresolved_extent','旧左框[68,116,81,132]与独立左框[55,120,81,133]选取的近端部分差异大：掌部下面的水平白段与靠蓝块斜段如何划分仍不明确。右侧细灰条只有数像素宽，独立复核也标低置信度。当前RGB不能可靠排除少标夹指或混入掌部。','需外部指认水平白段是否整段属于掌部远端夹指及其与掌部连接点；不因已有一个较清楚候选就把另一个当负例，整类unknown。')
 put(9,'drawer_front','uncertain','not_identifiable',[],'retain_unknown','左下黑把手及斜向棕色边条可见，但独立框右界x61截断仍延伸到图像底/右侧的棕色条。该条中前板、侧板与腔体边缘交界在裁切处无法稳定界定；既不接受原unknown为缺失，也不把独立框直接当真值。','需外部标出左下斜边中哪些属于前板，尤其x61至右下裁切边；保持unknown。')
 put(9,'gripper_finger','partial','out_of_frame',[[12,0,35,27],[50,0,70,27]],'correct_visible_extent','两条灰色夹指在蓝块两侧且从上缘裁切。左上可见条体从约x12开始，到x34、y26；右条体可见范围约x50至69、末端y26。旧框左侧x8与右侧x72含背景，并截短约1–2像素尖端。粉色/蓝色物体不属夹指。')
 put(32,'drawer_front','partial','out_of_frame',[[27,165,133,200]],'correct_visible_extent','打开抽屉的前板左上尖角约(27,165)，右下可见端约x132，黑把手触及底部裁切。旧x42漏掉棕色前板左上端；旧x139额外包含侧面/背景。只包含前板与把手的可见外接范围。')
 put(32,'gripper_finger','visible','none',[[4,86,16,104],[49,104,64,121]],'include_identifiable_visible_parts','两处狭窄灰白指条分别位于掌部远端左下(4–15,86–103)及右下(49–63,104–120)，与棕色背景有清楚轮廓，外形细长且与粗掌部相接；无须推断被蓝块遮挡的部分。旧整类unknown过于保守，允许进入本次评估。','细小连接处仍有约1–2像素边界误差；这一判断是可复核的助手裁决，不是独立人工金标准。')
 put(33,'drawer_front','visible','none',[[20,33,76,77]],'include_identifiable_visible_parts','下右侧棕色斜向前板连接黑U形把手，前板两端约(75,33)与(20,76)，把手也在此范围内。它与左侧板、中央腔底在此RGB可区分。可见部件外接矩形必然跨过部分腔底；这不表示把腔底定义为前板。')
 c=next(x for x in by[33]['classes']if x['category']=='gripper_finger')
 put(33,'gripper_finger',c['status'],c['reason'],c['boxes'],'retain_original','两条上缘裁切灰指身份一致，两份坐标只相差约1像素，原框覆盖可见末端。没有足够明确的像素证据要求用独立坐标替换，保留原框；不宣称1像素边界已被精确验证。')
 put(38,'drawer_front','partial','out_of_frame',[[46,170,151,200]],'correct_visible_extent','前板左上端约(46,170)，右边缘至x150；把手下部在图像外。旧(54,177)漏掉可见前缘左上区域，修正可见外接框并维持画外partial。')
 put(38,'gripper_finger','partial','occluded',[[49,106,65,125],[73,84,84,107]],'correct_visible_extent','左下细灰指从约(49,106)斜向(64,124)，旧x56漏近端；右指从约y84沿灰条延伸至y106，旧y94漏上半段且包含粉块。新框沿清楚灰条修正，不包含粉色物体。','右指尖与粉色物体相接，是否完全无自遮挡无法确定，保留partial/occluded，不采纳独立visible状态；该状态差异不改变计分覆盖。')
 put(39,'drawer_front','visible','none',[[43,12,71,72]],'include_identifiable_visible_parts','图像右侧的斜棕色前板与外侧黑U形把手相连，前板上端棕色像素约(43,12)、下端至(70,71)。腔底在其左侧；两者边界可辨。按可见前板和把手外接范围标注，不将中央蓝块/腔底当作部件。')
 c=next(x for x in by[39]['classes']if x['category']=='gripper_finger')
 put(39,'gripper_finger',c['status'],c['reason'],c['boxes'],'retain_original','上缘两条灰指身份清楚。独立右框向x52扩展处主要为其后黑把手/背景，不能据此扩张；原框覆盖灰指可见段。左右末端1–2像素差异不足以支持进一步改框，保留原坐标。')
 assert {(d['image_id'],d['category'])for d in decisions}=={(i,c)for i in IDS for c in CATS}
 rows=read(E/'data/images.json');meta={r['image_id']:r for r in rows}
 anns=[]
 for i in IDS:
  classes=[d['adjudicated']for d in decisions if d['image_id']==i]
  for c in classes:
   assert c['status']in ['visible','partial','not_visible','uncertain']
   if c['status']=='uncertain':assert not c['boxes']
   for x0,y0,x1,y1 in c['boxes']:assert 0<=x0<x1<=meta[i]['width']and 0<=y0<y1<=meta[i]['height']
  anns.append({'image_id':i,'classes':classes})
 save(dst,{'version':'eval8_v1','created_utc':now(),'status':'evidence-backed assistant evaluation labels, not independent human gold; outside8 unchanged and unaudited','annotations':anns})
 save(E/'data/labels/decision_ledger.json',{'scope':'exactly8RGB; no prediction files read before freeze','decisions':decisions})
 arbitration=[{'image_id':d['image_id'],'category':d['category'],'question':d['remaining_uncertainty'],'current_status':'uncertain; excluded from scoring'}for d in decisions if d['adjudicated']['status']=='uncertain']
 save(E/'data/labels/external_arbitration_needed.json',arbitration)
 shutil.copy2(__file__,E/'source'/Path(__file__).name)
 save(E/'configs/labels_frozen.json',{'frozen_utc':now(),'no_model_outputs_read_before_freeze':True,'prior_context_exposure_disclosed':True,'version':'eval8_v1','sha256':{str(p):sha(p)for p in [dst,E/'data/labels/decision_ledger.json',E/'data/labels/external_arbitration_needed.json',E/'source/adjudicate.py',E/'configs/pre_adjudication_frozen.json']}})
 print('EIGHT_IMAGE_LABELS_FROZEN',flush=True)
 print(json.dumps({'new_known_class_images':sum(c['status']!='uncertain'for a in anns for c in a['classes']),'new_boxes':sum(len(c['boxes'])for a in anns for c in a['classes']),'uncertain':arbitration},ensure_ascii=False))
if __name__=='__main__':main()
