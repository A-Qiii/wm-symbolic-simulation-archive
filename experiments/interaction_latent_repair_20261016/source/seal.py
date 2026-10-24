from common import *
def main():
 a=read(E/'results/execution_audit.json');m=read(E/'results/primary_comparison.json');diag=read(E/'results/single_substitution.json');n=sum(r['first_supported_error'].get('current_readout_correct',False)for r in diag if r['arm']=='canonical_matched'and not r['physical_success']and r['first_supported_error'])
 text=f'''# 本机交接：最小转移修复已完成并封存

新实验：{E}
完整报告：{R}/REPORT.md
连续视频图集：{R}/index.html

结论：持物关闭place的连续后果MSE下降seed0 {m[0]['relative_error_reduction']*100:.1f}%、seed1 {m[1]['relative_error_reduction']*100:.1f}%，两种子均12/12根下降。但该条件全9命题正确仍两臂两种子均0/12；目标In假阳性12/12降到6/12与8/12。实际闭环两臂均3/12（相同3根成功），初始关闭均0/6，初始打开3/6，无错误完成声明。正常成功动作语义预测31/50→26/50、39/50→34/50，正常连续后果MSE也略退化。不能宣称修复了规划或实现整体优势。

实际工作：2臂×2种子×1000更新；同种子初始化相同，同基础48/64 batch与观察链索引，辅助16/64样本来源不同；唯一训练模块transition67328参数。编码器/解码器/query/预处理哈希不变，梯度检查通过，训练私有标签访问被守卫禁止。选点control seed0u500、seed1u250；matched seed0u1000、seed1u750。所有C37根等权、各色与旧任务回归都已报告；C2根占50%的选择偏置没有隐藏。

D2仅使用预声明12根。60离线分支，90计划任务槽位，87实际调用：成功50、拒绝36、失败1、未到达3。失败是pink_2参考open在descend_to_handle阶段受阻0.0521m，后3步不执行、不换根。24次准备全部完成；24段闭环130次任务调用。总任务{a['all_task_calls']}/234，总准备24/24，总原生技能{a['all_primitive_calls']}/258，总控制步{a['total_control_steps']}/267636。没有无限重试或未记录的额外采集。

连续视频：每控制帧30fps，合计{a['continuous_video_frames']}帧，仿真时间不包括模型等待。4个代表视频均实际解码验证，含两臂同红_1成功及红_0失败；全部24段闭环链接均保留。原4段端点动画与原权重不改。

失败诊断：所有失败段都做预定单项替换，实际已存后继RGB/本体观测重新编码，只替换首预测latent、保留后续动作与query。匹配臂9个失败段中{n}个在定位到的首个有影响错误处当前全命题读对，说明剩余失败不能都归因于当前query；不据此唯一隔离所有误差。没有第二轮修正。

旧匹配C缺物体姿态，新增6次调用只评H与drawer6有效命题，In全部无效；不能冒用旧7命题In定义。D2新采集完整physical snapshots，使用冻结v2.1全部9命题。训练不受这些标签影响，C选点只用观测MSE。

核心证据：configs/protocol.json、schedule_seed*.npz、training_frozen.json、before_D2_frozen.json；results/train_*.json、C_metrics.json、D2_metrics.json、D2_slots.json、D2_denominators.json、closed_loop.json、single_substitution.json、execution_audit.json、media_manifest.json。逐步原生调用与原因在data/D2_raw及data/closed_loop下的_private，public只有RGB/proprio。

复核：
```
source {P}/env.sh
python {E}/source/verify.py
```
训练复现（必须新目录，不复制或执行D2；下面destination需不存在；--run才训练）：
```
source {P}/world_model_env.sh
python {E}/source/reproduce_training.py --destination {P}/experiments/interaction_repair_reproduction --run
```
完整原运行顺序：setup.py → cache.py → train.py --preflight四组合 → training_frozen → train.py四组合 → evaluate.py --split C → bridge_preflight.py → before_D2_frozen → collect.py --phase offline → convert.py → cache.py --D2 → evaluate.py --split D2 → collect.py --phase closed_loop → diagnose.py → report.py → seal.py → verify.py。已经执行完毕，**不要按这个顺序再覆盖本目录**。源代码与环境脚本都已保存/引用，可在新目录复现训练；D2已用，复测不再叫盲测。

无活跃任务，无环境/权限阻塞。当前研究结果是修复局部有效而主指标未改善；本轮已完成并止于封存，不请求追加调参。原HULC完整指南未完成，任务范围与旧项目保持不变。
'''
 (R/'HANDOFF.md').write_text(text)
 files=[p for b in[E,R]for p in b.rglob('*')if p.is_file()and'__pycache__'not in p.parts and p.name not in['delivery_manifest.json','verification_receipt.json']]
 save(E/'configs/delivery_manifest.json',{'created_utc':now(),'status':'completed one controlled repair; no D2 retuning','sha256':{str(p):sha(p)for p in sorted(files)},'previous_review_manifest_sha256':sha(REVIEW/'configs/review_delivery_manifest.json'),'original_manifest_sha256':sha(OLD/'configs/delivery_manifest.json')});print('DELIVERY_SEALED',len(files),sum(p.stat().st_size for p in files),flush=True)
if __name__=='__main__':main()
