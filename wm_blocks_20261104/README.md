# 三物块世界模型核心代码 · 2026-11-04

来源为 `coverage_alignment_20261104` 的冻结正式实现，与11月4日方法稿核对：双相机四帧 RGB → 共享对象状态 → 动作条件后果递推 → 区域/Held读出 → 有限候选程序选择。三对象为 large、small、small2；逐轴影响与直接位移两组模型分别训练。

| 文件 | 内容 |
|---|---|
| `models.py` | RGBState、HeldHead、共享对象对动力学；非操作对象逐轴影响/直接位移对照 |
| `rgb_front_three.py` | RGB颜色ROI与冻结DINOv2，输出三对象1567维特征 |
| `train.py` | 状态学习、抓持校准、单步/自由多步损失、匹配采样和开发选模 |
| `runtime.py` | 冻结权重加载、观测、动作预测和任务属性读出 |
| `task_policy.py` | 最多六动作的584候选、技能前提、保护条件、前缀复用和选择 |
| `dataset.py`、`cache_data.py` | 原始轨迹/标签合同、分组及留出检查、特征缓存入口 |
| `common.py` | 文件读写、实验根目录与GPU锁 |

## 使用

网络与候选库可直接导入：

```python
from models import RGBState, HeldHead, ObjectDynamics
from task_policy import bank
net = ObjectDynamics("diagonal")  # 或 "direct"
assert len(bank()) == 584
```

训练、提取与历史权重推理需要 Linux、CUDA、PyTorch、NumPy、torchvision、transformers，以及另行提供的本地实验资产。配置 `WM_EXPERIMENT_ROOT` 指向独立工作目录，其中保留原实验合同的 `configs/plan.json`、轨迹及其引用、`cache/fit/rows.json` / `all_features.npz` 和所需 `models/` 等。`WM_DINO_WEIGHTS` 指向本地 DINOv2-small 权重；不自动下载。多进程共用同一GPU时，用 `WM_GPU_LOCK` 指定同一个锁文件。

```bash
export WM_EXPERIMENT_ROOT=/path/to/new-experiment-workspace
export WM_DINO_WEIGHTS=/path/to/local-dinov2-small
python cache_data.py fit   # 仅在工作目录已备齐原始轨迹和完整计划时
python train.py            # 写入工作目录；已有冻结目录不要直接重跑
```

这是核心代码存档，未附完整实验运行包。完整采集/物理控制器、并行调度、报告、数据、权重与3.56MB逐轨迹计划留在原工程；不放一个空白或删减计划冒充原配置。历史权重记录中的checkpoint路径及其哈希需与实际本地文件对应。

核心网络、损失和候选策略来自正式源码。仅改三处部署连接：实验根目录/锁改为环境变量，DINO权重路径去除对旧包的常量导入，训练冻结摘要改为哈希本目录源码。未加入旧RSSM/ObjectGRU、双物块实现或抽屉记忆支线。

目标与保护条件由Python函数检查，未接LTL解析器/监视器；预测检查的是宏动作边界，物理动作内部评分不在本核心目录中。无模型权重时的结构检查不等于完整实验复现。
