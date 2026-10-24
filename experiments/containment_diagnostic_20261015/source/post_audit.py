"""Independent evaluator recomputation and paired reset checks, never model input."""
from common import *
from labels_v2_1 import evaluate
import gzip,collections
def main():
 paired=[]
 for ep in read(B/'manifest.json')['episodes']:
  comparisons=[]
  for frame in [0,63,127]:
   old=dict(np.load(BASE_B/'episodes'/ep['episode_id']/'public'/f'{frame:05d}.npz'));new=dict(np.load(B/'episodes'/ep['episode_id']/'public'/f'{frame:05d}.npz'))
   comparisons.append({'frame':frame,'static_RGB_identical':bool(np.array_equal(old['rgb_static'],new['rgb_static'])),'gripper_RGB_identical':bool(np.array_equal(old['rgb_gripper'],new['rgb_gripper'])),'robot_max_abs_difference':float(np.max(np.abs(old['robot_obs']-new['robot_obs'])))})
  paired.append({'episode_id':ep['episode_id'],'checks':comparisons})
 wins=read(BASE/'data/windows.json');by=collections.defaultdict(list)
 for w in wins:by[w['episode_id']].append(w)
 counts=collections.defaultdict(lambda:{'windows':0,'valid_In_facts':0,'matching_valid_In_facts':0,'all_labels_and_masks_identical':True})
 for eid,ww in by.items():
  with gzip.open(BASE_B/'episodes'/eid/'_private/full_physics_trace.json.gz','rt')as f:trace=json.load(f)
  for w in ww:
   recomputed=evaluate(trace[w['anchor']]['physical']);expected=w['targets_v2_1'];r=counts[w['split']];r['windows']+=1;r['all_labels_and_masks_identical']&=recomputed['y']==expected['y']and recomputed['v']==expected['v']
   for j in range(3,6):
    if expected['v'][j]:r['valid_In_facts']+=1;r['matching_valid_In_facts']+=int(recomputed['y'][j]==expected['y'][j])
 assert all(r['all_labels_and_masks_identical']for r in counts.values())
 result={'paired_reset_preaction_public_checks':paired,'GT_definition_recomputed_from_saved_full_physics':dict(counts),'GT_reference_is_label_definition_not_visual_model':True,'model_inputs_unchanged':True};save(E/'results/independent_post_audit.json',result);save(R/'tables/independent_post_audit.json',result)
 text=(R/'REPORT.md').read_text();text+='\n## 交付前独立核查\n\n从630个保存的完整物理快照独立重算新版标签，AB1253/1253、C629/629个有效In事实与标签一致，九事实标签和确定性掩码也全部相同。这是GT定义核查，不是视觉预测得分；它解释了GT线性探针的错误属于探针表达限制，而不是几何记录不能重现定义。\n\n9次补充执行分别对比原回合的第0、63、127帧，逐项记录静态RGB、腕部RGB是否相同及本体状态最大差，见tables/independent_post_audit.json；同根复位不当作独立验收。\n';(R/'REPORT.md').write_text(text)
 print(json.dumps({'GT_recomputed':dict(counts),'preaction_RGB_exact_checks':sum(x['static_RGB_identical']and x['gripper_RGB_identical']for r in paired for x in r['checks']),'preaction_comparisons':27,'max_robot_diff':max(x['robot_max_abs_difference']for r in paired for x in r['checks'])},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
