# 物块视觉世界模型：v3最小复现工程

本目录保留最新v3、必要的v1/v2对照、固定协议及从零生成经历的入口。
**不包含实验报告、图表产物、原始数据、训练权重、视频、日志或缓存。**
运行时会在本机生成这些文件；它们均由`.gitignore`排除。

## 方法与规划范围

双相机RGB＋机器人本体观测 → 冻结DINO patch/颜色ROI → 学习对象位置、
尺寸、Held → 动作条件转移 → 区域/Held读出。观测器使用四帧窗口，无GRU。
三版共用同种子的冻结观测器及训练经历：

| 版本 | 自身运动输入 | 非目标影响 |
|---|---|---|
| v1 | 对象状态＋外观/场景上下文 | 全3×3条件矩阵；同文件保留直接MLP基线 |
| v2 | 两对象学得状态＋动作 | 全3×3条件矩阵 |
| v3 | 同v2 | 三个逐轴学习系数 |

承托/on/contact不作为模型标签。位置、尺寸、区域及Held允许作训练监督和
独立评分；部署输入不读取仿真位置、分割或深度。ROI来自已知蓝/红颜色。
抓放由脚本控制，真实双指接触成立后允许TCP—目标抓持辅助；两物块独立。

**实际规划器为固定候选序列枚举，不包含LTL/LTLf搜索实现。**

- 反向任务：等待；拿起大块→放到A；拿起大块→放到B。
- 基本任务：等待；直接搬大块到B；先把小块放A再搬大块到B；
  先把小块放大块上再搬大块到B。
- 在每个真实观测边界，用模型预测剩余候选后果，检查目标及全过程留区
  约束，选择剩余动作最少的可行候选；同长度按固定索引。
  无可行预测或低层技能失败即停止。没有最高分冒险fallback。

## 固定资源与环境

已使用的环境是Linux x86_64、Ubuntu24.04、RTX5080 Laptop16GB、
NVIDIA595.91.07。仿真Python3.10.18，学习Python3.11.13；
学习Torch2.11.0+cu128/Torchvision0.26.0+cu128。
环境脚本只激活工程内环境，不安装或降级宿主驱动。

- `requirements/physics.txt`：该物块仿真实际依赖闭包，无HULC/Torch要求。
- `requirements/model.txt`：学习/特征/分析依赖闭包，固定CUDA12.8轮子。
- `configs/external_resources.json`：CALVIN提交、DINO revision和文件校验和。
- `assets/`：两份必要物块URDF。Panda、桌面、纹理从固定CALVIN源取得。
- 无需CALVIN debug数据集或HULC执行器权重；经历由当前仿真生成。

需要能运行PyBullet EGL的NVIDIA/OpenGL环境。中文图表默认使用系统Noto CJK，
也可用`WM_BLOCKS_CJK_FONT`指定字体文件。完整第三方声明见`THIRD_PARTY.md`。
本次本地准备只做轻量检查，没有重新安装干净环境或重跑完整实验。

### 1. 创建独立环境

先提供对应版本解释器。以下命令会创建本目录的`.envs/`，已存在则停止，
保留原环境；安装过程需从Python/PyTorch源下载包。

```bash
python3 scripts/setup_envs.py \
  --executor-python /path/to/python3.10 \
  --model-python /path/to/python3.11
```

测试过的精确包版本与Python补丁版本见`reproducibility/*-runtime.json`。
依赖锁来自实际安装元数据，而非较早的旧项目requirements文件。

### 2. 取得外部资源（显式下载）

```bash
python3 scripts/resources.py fetch-calvin
.envs/world_model/bin/python scripts/resources.py fetch-dino
.envs/world_model/bin/python scripts/resources.py verify
```

CALVIN固定为`1431a46bd36bde5903fb6345e68b5ccc30def666`，
DINO固定为`ed25f3a31f01632728cabb09d1542f84ab7b0056`。
下载脚本只取资源并校验；不会推送仓库。已有路径不符时停止，不覆盖它。
Tacto和IKFast不用于此配置，标准Panda配置`use_ik_fast=false`。

复用已有缓存时，可显式设置`WM_BLOCKS_CALVIN_ENV`为CALVIN环境源码目录、
`WM_BLOCKS_DINO_WEIGHTS`为固定revision的模型目录。无需旧项目Python包。

### 3. 轻量检查

```bash
python3 scripts/smoke.py syntax
source env.sh
python scripts/smoke.py physics
source world_model_env.sh
python scripts/smoke.py model
```

这些检查编译源码、检查配置/URDF、在临时目录重建相同划分、导入运行模块，
并用合成CPU张量验证模型接口。它们不启动物理场景、不加载DINO权重到GPU、
不采集、不训练，也不提供实验准确率。

## 从零复现命令

默认只打印阶段清单；完整运行必须显式提供`--execute`：

```bash
python3 scripts/reproduce.py
python3 scripts/reproduce.py --execute
```

默认`--baseline-scope transition_holdout`只拟合最新对照需要的transition
留出臂：三种子状态估计器、直接MLP和矩阵基线，再拟合v2/v3各三种子。
`--baseline-scope full`可展开原v1全部24个动态基线。这个开关仅筛选外层
实验/臂，训练循环、样本顺序和损失不变。默认流程不重做历史v1/v2确认批次。

| 阶段名 | 环境 | 执行入口与参数 |
|---|---|---|
| engineering | executor | `GATE_REV=gate_06 python src/engineering_gate.py` |
| engineering-audit | executor | `python src/audit_engineering.py` |
| collect-fit | executor | `COLLECTION_VERSION=v1 COLLECT_MODE=fit python src/collect.py` |
| cache-fit | world_model | `COLLECTION_VERSION=v1 COLLECT_MODE=fit python src/cache_features.py` |
| smoke128 | world_model | `python src/smoke128.py` |
| train-baselines | world_model | `WM_BLOCKS_BASELINE_SCOPE=transition_holdout python src/train.py` |
| collect-dev-cross | executor | `python src/v2_development_challenge.py collect` |
| cache-dev-cross | world_model | `python src/v2_development_challenge.py cache` |
| train-v2 / train-v3 | world_model | `python src/train_v2.py` / `python src/train_v3.py` |
| collect-confirm | executor | `python src/confirm_v3_data.py collect` |
| cache-confirm | world_model | `python src/confirm_v3_data.py cache` |
| evaluate | world_model | `python src/evaluate_v3.py` |
| execute | world_model | `python src/live_v3.py` |
| analyze / inspect-condition | world_model | `python src/analyze_v3.py` / `python src/inspect_v3_condition.py` |

可用`--through train-v3`等阶段名在安全阶段边界结束；再次运行会核对完成
标记并续跑，不重复启动已完成阶段。失败门槛不通过会停止，确认集不解锁。
不将原工程生成结果复制到这个源码目录冒充从零运行。

单卡按阶段串行；工程锁阻止同目录重复驱动，GPU锁阻止同目录并发GPU任务。
多工程共用GPU时可将`WM_BLOCKS_GPU_LOCK`设为相同锁文件；先运行
`python3 scripts/init_workspace.py`创建对应锁。不要并行启动多个采集/训练。

## 种子、划分与分母

`configs/experiment_plan.json`保留原固定100组/实验、60/20/20根级划分；
`src/protocol.py`可重新生成，smoke会在临时根目录逐字节比较。
物理验收先完成60条独立恢复程序。正式fit候选池1120经历／2880转移，
含两个原实验及覆盖候选；每实际训练臂只使用其360经历／960转移。
保留完整候选采集池是为了保持原输入审计与固定划分，不能把全部候选算作训练。

v2/v3均沿用原transition留出臂。额外20条开发交叉程序只评分、不拟合。
v3确认固定20新组（270001000…019），每组7参考程序，共140经历／400转移，
在冻结后生成；同数据离线评估v1/v2/v3三种子。
实际执行固定种子0的v1/v3，各20组×6任务＝120次。
同布局的所有经历/帧保持同集合，三个种子不当作更多独立布局。

载物B→A不进训练/开发，载物B可作为正向终态出现。模型冻结前不生成确认
预测；确认结果不回灌训练或选模。无效初始化/技能失败保留在记录和分母。

## 文件边界与维护

源模型计算保留，便携改动仅包括路径集中配置、最小观测适配器、阶段驱动、
必要基线范围筛选和字体配置。`reproducibility/source-map.json`记录来源哈希，
`reproducibility/SHA256SUMS`记录本快照文件校验和。
本目录不是压缩归档包，源码可以逐文件浏览。

运行后生成的`configs/*models_frozen.json`、`configs/*confirmation_manifest.json`
也属于本机运行产物，已排除。固定方案配置、源资产、依赖和必要运行说明保留。
