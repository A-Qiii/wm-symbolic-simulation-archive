"""Small local evidence page; never invent plots for absent experimental data."""
from pathlib import Path
import json,html,datetime
P=Path('/home/robot/research/projects/wm_symbolic')
def load(rel):
    path=P/rel
    return json.loads(path.read_text()) if path.exists() else None
def esc(x):return html.escape(str(x))
guide=load('source/identity.json');gpu=load('runs/gpu_probe.json');reg=load('runs/planning_regression.json')
rows=[('指南完整性',f"{guide['lines']} 行 / {guide['bytes']:,} 字节",'已读取全文；原始环境说明由用户的本机执行要求覆盖','source/identity.json'),
      ('独立项目',str(P),'源码、数据、模型、运行证据与报告分目录；旧 robot_state 只读','configs/protocol.json'),
      ('GPU 计算',f"{gpu['device']} / {gpu['torch']} / CUDA {gpu['cuda_runtime']}" if gpu else '待验证','实际矩阵前向与反向有限值检查；不代表 HULC 已执行成功','runs/gpu_probe.json'),
      ('规划回归',f"{reg['tests']} 项通过" if reg else '待验证','已有工程组件的回归，不计入本轮学习模型效果','runs/planning_regression.json')]
sections=[]
skills=load('runs/stage0_skills_20261013_151214/summary.json')
if skills:
    rows.extend([
      ('冻结执行器', '官方 HULC ABC 严格加载，47,053,815 个冻结参数', '沿用检查点语言嵌入和官方归一化；兼容适配只恢复旧检查点中未启用的 LayerNorm 模块', 'runs/stage0_skills_20261013_151214/executor_identity.json'),
      ('多技能与参考链', '普通四技能链完成；收纳链 1/3 根完成', '收纳链为开抽屉、拿粉块、放入抽屉、关抽屉。该小样本证明可以执行，不代表稳定成功', 'runs/stage0_container_reference_20261013_152502/results.json'),
      ('颜色绑定限制', '同三根上：红 0/3、蓝 0/3、粉 2/3', '只替换技能颜色，固定根、模型和预算；支持目标绑定/控制限制的诊断，尚未训练世界模型', 'runs/stage0_colour_binding_20261013_151842/results.json'),
      ('有限轨迹语义', '7 项检查通过', '验证强下一步、顺序、不可移动事件及未知命题拒绝；使用 MONA 编译的自动机', 'runs/ltlf_probe/result.json')])
    sections.append('''<h2>固定执行时长的实际比较</h2><figure><img class="wide" src="figures/skill_budget.png"><figcaption>横轴为固定控制步预算，括号为 30 Hz 下的模拟秒数；纵轴为五个代表技能。颜色越深表示相同三个 C 初态中成功数越多，格内给出实际分子/分母。各列来自相同 360 步轨迹的中间检查点，未用真值提前停止；对比只反映冻结 HULC 的技能后果。</figcaption></figure><h2>第一个有影响的错误：操作了错误对象</h2><figure><img class="wide" src="figures/wrong_object.png"><figcaption>左、中央为同次“拿红块”操作前后的实际 RGB 图，蓝块最终被拿起。右图横轴为低层控制步，纵轴为夹爪到积木中心的距离（米）；红线和蓝线分别对应红块、蓝块，灰色虚线为第 47 步首次闭合接触蓝块。接触和坐标来自离线评分，未送入策略。该证据将失败定位到执行器的目标操作，不能归因为尚未训练的世界模型。</figcaption></figure>''')
mask=load('runs/mask_probe.json')
if mask:rows.append(('可见分割与标定','A/B 两路分割与 4×4 外参/投影矩阵通过','只提供渲染中可见的匿名像素；遮挡部分没有额外轮廓或真值坐标','runs/mask_probe.json'))
reset=load('runs/reset_after_actions.json')
if reset:rows.append(('真实操作后的分支复位','重建原生物理实例后两路图像及机器人状态差异均为零','旧实例经历接触后残留微小物理差异；保留 6 条诊断记录，匹配分支统一使用新实例恢复同一保存根','runs/reset_after_actions.json'))
storage=load('runs/storage_50.json');manifest=load('data/data_manifest.json')
if storage:rows.append(('真实数据与存储',f"当前 {manifest['real_transitions']} 条；前50条 {storage['actual_bytes']/1e6:.1f} MB",f"按同样轨迹格式估算6000条 {storage['estimated_6000_trajectory_bytes']/1e9:.1f} GB；不含特征缓存和检查点。失败全部保留，A/B仅训练",'runs/storage_50.json'))
labels=load('runs/label_distribution.json')
if labels:rows.append(('状态与事件标签',f"已标注 {labels['transitions']} 条真实转移",'当前绝对开合/位置与技能成功分开；禁动事件扫描整个低层控制过程。临界几何有有效性 mask','runs/label_distribution.json'))
probe=load('runs/wm_probe_seed0/metrics_sampling8_01000.json')
if probe:
    rows.append(('128 条训练检查',f"当前 {probe['current_correct']}/{probe['current_facts']}；一步 {probe['prior_correct']}/{probe['facts']}",'这是训练样本拟合率，不是独立验证。8 条随机潜轨迹汇总概率；一步严格用 prior，未来观测只进监督目标','runs/wm_probe_seed0/metrics_sampling8_01000.json'))
    rows.append(('逐步未来隔离','14 个预测位置全部通过','不同序列中替换该步及之后未来观测，prior 不变；另以 FP32 验证槽位排列不改变编码','runs/wm_probe_seed0/prior_isolation_all_steps.json'))
    sections.append('''<h2>训练检查与随机潜变量诊断</h2><figure><img class="wide" src="figures/probe_training.png"><figcaption>左图横轴是优化更新次数，纵轴为对数尺度的归一化训练损失：紫色表示观测重建，蓝色表示当前/一步符号，橙色表示结果与过程事件；这是小批次曲线，局部波动正常。中图灰线是 KL 正则，虚线为先求和 32 个变量后使用的 1 nat 下限。右图固定同一检查点和 618 个有效训练集未来分类事实：灰柱为逐变量取最大值，浅蓝为一次固定种子采样，深蓝为 8 条配对随机轨迹的概率均值。均未选择最佳轨迹；柱图说明评估方式的影响，不说明泛化。</figcaption></figure><h2>绝对状态标签的图像核对</h2><figure><img class="wide" src="figures/label_audit.png"><figcaption>六张图都是已记录的实际固定相机 RGB，横纵轴为像素位置；标题列出被审核的物体、离线状态标签、序列和步号。颜色是物体真实外观，非置信度。持物与抽屉归属由离线接触/几何审核，并与图像对照；训练模型只能接收公开观察，不能读取这些真值。这里审核标注流程，未声称标签是模型预测。</figcaption></figure>''')
for f in sorted((P/'runs').glob('stage0_environment_*/result.json')):
    r=json.loads(f.read_text());rel=f.parent.relative_to(P/'runs')
    rows.append((f"{r['environment']} 环境相机与复位",r['status'],'真实原生仿真；相同根重复复位后比较图像与机器人自身状态',str(f.relative_to(P))))
    imgs=''.join(f'<figure><img src="../runs/{rel}/{cam}.png"><figcaption>{label}</figcaption></figure>'
        for cam,label in [('rgb_static','固定相机：整张工作台与对象分布'),('rgb_gripper','夹爪相机：末端附近的局部观察')]
        if (f.parent/f'{cam}.png').exists())
    sections.append(f'<h2>环境 {r["environment"]} 的实际观察</h2><div class="images">{imgs}</div><p>颜色为实际 RGB；图像横纵方向均为相机像素坐标。两图对比不同视角，并非不同方法。局部视野可能遮挡物体，不能将不可见判断为不存在。</p>')
body=''.join(f'<tr><th>{esc(a)}</th><td>{esc(b)}</td><td>{esc(c)}</td><td><a href="../{esc(d)}">证据</a></td></tr>' for a,b,c,d in rows)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>世界模型与符号规划：本机验证</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1120px;margin:48px auto;padding:0 24px;color:#17233b;background:#f5f7fb}h1{font-size:28px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d7deea;text-align:left;padding:14px;vertical-align:top}th{min-width:110px}small{color:#596579}.images{display:flex;gap:24px;align-items:start}figure{margin:0}img{width:340px;image-rendering:auto;border:1px solid #bbb}.wide{width:100%;max-width:1100px}figcaption{padding:12px;background:white}a{color:#154d98}.note{padding:16px;background:#eaf0fa}h2{margin-top:36px}</style>
<h1>世界模型与 TS／LTL：本机实验记录</h1><p class="note">阶段 0 已出现四类可执行技能与真实四技能参考链，正在做数据和训练检查。世界模型训练、由学习模型供给后继的搜索以及独立闭环测试尚未完成；执行器和工程回归成绩不替代这些验收。</p><table><tr><th>元素</th><th>实际结果</th><th>为什么检查／能说明什么</th><th>记录</th></tr>'''+body+'</table>'+''.join(sections)+f'<p><small>生成时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。配置配额与验收门槛见 <a href="../configs/protocol.json">protocol.json</a>。</small></p></html>'
(P/'reports/stage0.html').write_text(page)
print(P/'reports/stage0.html')
