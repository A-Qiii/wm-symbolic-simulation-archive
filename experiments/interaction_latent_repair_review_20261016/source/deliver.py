"""Write a final handoff and hash manifest for this review only."""
from audit import *
import datetime
def main():
 media=read(E/'results/media_manifest.json');rows=[]
 for r in media:rows.append(f"- {r['root_id']}: {r['calls']} calls, {r['frames']} frames at {r['fps']:g} fps, {r['duration_seconds']:g}s, {Path(r['video']).stat().st_size} bytes; {r['video']}")
 text=f'''# 本轮只读复核交接

报告入口：{R}/index.html
分析正文：{R}/REVIEW.md
未执行修正协议：{R}/PROPOSED_PROTOCOL.md
实验记录：{E}

已完成：覆盖审计；18个既有AB/C起点×6个冻结模型推理；3成功+1失败端点对比动画；12个未来D2根声明及重叠检查；协议哈希。训练更新0，新物理调用0，旧D重执行0，新D2观测0。无活跃后台任务。依赖未新增，旧工程未修改。

关键结论：
- AB有30次place拒绝，均无意图对象位、空手且关闭；AB有意图place无拒绝。C有意图仅1次拒绝，仍为空手、打开；用于训练的转移缺持物关闭。
- 既有未用匹配数据AB4根12place/C2根6place包含持物关闭拒绝；原执行参数均空，无权GT补目标。
- 首版144AB/72C标注预算以及全部630标注窗都没有持物关闭。冻结C探针中PCA/AE抽屉2/2，交互两个种子均1/2，不足以排除所有视觉局限但说明有可用线索。
- 红/蓝失败首状态9命题全对；实际后继观测latent单项替换使计划分数0.997→0.718、0.838→0.000049。粉在step1有读出错误，替换0.998→0.021。没有GT后继或手写效果注入。
- 原搜索8候选/beam32/深度4/阈值0.8/max6调用。低层仍PhysicalOracle；place对象参数未传入执行。成功对象匹配，旧3/6有效，但下一版共同改为无对象place。此轮未改旧接口或分数。
- D真实当前203/312、226/312；正常前态40/62、44/62；正常真实后态54/62、57/62；预测后态52/62、59/62（interaction seed0/1）。D18根，正常62+拒绝6+失败4。闭环仅预指定seed0、6个匹配根，3/6。

媒体均为真实端点对比卡，每卡4秒，无插值/重新执行，逐次展示包括拒绝。蓝成功前三次place拒绝未删；粉失败抓取后五次拒绝未删。源端点和视频SHA见results/media_manifest.json；原图集保持原位。
{chr(10).join(rows)}

修正提案：统一place接口后的两臂，共同冻结表示、解码器、查询头，仅更新转移；共同原AB888，附加16/64采样对照为旧place，实验为额外AB12匹配place；同种子同初始化、48/64共同预存样本、1000步。C35根和新增C2根预测误差各半选点，不用语义分数选点。持物关当前读出错误仍存在，不能承诺这次修正解决它。

未来D2仅声明12根seed16961000–16961011，3颜色×4布局，6开6关。42预定参考调用+48匹配place主调用；准备调用另记。闭环两臂seed0各12根max6；测试前冻结后只跑一次，失败准备保留，不换根。原D已用不能再叫盲测。该方案待父线程独立验收；未实现训练代码，未执行训练/采集。原指南完成度不变。

复核命令（读现有结果，无GPU/仿真）：
```
source /home/robot/research/projects/wm_symbolic/env.sh
python {E}/source/verify.py
```

审计脚本与材料：source/audit.py、probe.py、media.py、proposal.py、report.py；results/score_denominators.json、coverage_summary.json、place_used.json、semantic_budget_composition.json、frozen_AB_C_probe.json、matched_input_distances.json、first_errors.json、media_manifest.json；configs/proposed_protocol.json、proposed_D2_roots.json、proposed_D2_exclusion.json、proposal_freeze.json、matched_source_freeze.json。

下一步阻塞是本轮要求的独立协议验收与执行授权；没有环境或自动审批拒绝。不要把proposal_freeze当作启动许可。若验收决定改变模型/标签/目标，先另版本冻结，再执行，不改本复核和旧封存。
'''
 (R/'HANDOFF.md').write_text(text)
 files=[]
 for base in[E,R]:
  files.extend(p for p in base.rglob('*')if p.is_file()and'__pycache__'not in p.parts and p.name not in['review_delivery_manifest.json','verification_result.json'])
 save(E/'configs/review_delivery_manifest.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'role':'read-only review and future protocol proposal; not a new trained result','sha256':{str(p):sha(p)for p in sorted(files)},'prior_experiment_manifest_sha256':sha(OLD/'configs/delivery_manifest.json')})
 print('REVIEW_DELIVERY_SEALED',len(files),flush=True)
if __name__=='__main__':main()
