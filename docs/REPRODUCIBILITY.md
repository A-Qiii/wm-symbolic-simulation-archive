# 恢复与数据生成

请使用新的工作目录和独立环境；不要覆盖本机原项目或封存实验。归档代码包含历史固定绝对路径、已到期预算、源码验签和只读写入保护，不能当成可无条件直接重跑的安装包。

## 环境与官方资源

实际环境在`environment/`：executor为Python3.10.18，world_model/planner为Python3.11.13，宿主Ubuntu24.04.5和RTX5080Laptop16GB。三个环境的包清单由本次读取实际安装元数据得到；原`configs/*requirements*.txt`仍保留，可能早于当前环境。不要修改原环境、降级宿主驱动或改防火墙。

官方仓库的remote、精确HEAD与许可证见`provenance/vendor_repositories.json`和`vendor_licenses/`。CALVIN、HULC、calvin_env、tacto源码树本次读取时均干净；vendor目录本身未上传。恢复相应官方版本和子模块后，再核对历史适配是否需要。`source/hulc_layernorm_compatibility.diff`是原适配证据，不应盲目重复应用。

CALVIN debug数据、HULC公开权重的历史下载地址、字节数、SHA256见`source/assets.json`；本次没有重新下载或验证地址当前可用性。DINOv2缓存、冻结编码器和下游模型同样不在快照中，须另行恢复。关键本地模型/下载包位置与哈希见`provenance/local_only_key_artifacts.json`。

## 数据生成代码

`experiments/factorial_expansion_20261020/source/`包含完整生成实现：

1. `prepare.py`与`geometry.py`定义布局、初态、根/分支和集合划分，预先冻结成员及代码哈希。
2. `episode.py`通过固定执行器实际准备并执行状态；双相机RGB-D/分割/标定、机器人观测和私有评分标签分开保存。
3. `collect.py --phase pilot`执行原试采；试采门槛通过后才运行`--phase all`。失败、拒绝、中断均保留，不能补造成功样本。
4. 原配置将新增数据按布局族划为640条训练、80条开发、80条未见组合测试。测试条件为至少两个物体在抽屉内、且三个物体均在抽屉内或被拿着。旧训练宏边界和新增控制/边界的隔离检查由原生成管线完成。

这里保存生成方式及冻结配置，不包含原始轨迹、控制记录、标签和生成报告。重采属于新的实验，不是原数据的逐位替代。应复制配置到新的实验目录、设定新预算并重新冻结，不能在原800条数据上覆盖写入。

## 训练与评估入口

最新入口为`experiments/data_expansion_training_20261021/source/pipeline.py`，顺序为训练/开发特征→训练→开发评估→模型冻结后测试特征和评估。对应组件：

- `features.py`、`inputs.py`：冻结编码器、RGBD特征与训练集归一化；GT不进入观测输入。
- `models.py`：ObjectAug三对象与共享环境循环状态，物体头读入`[h_i,c]`。
- `training.py`、`losses.py`、`mid_data.py`：完整/中途起点与监督目标；原两种子均1500更新。
- `configs/schedule_seed{0,1}.npz`：两份小型精确采样配置，是本快照中唯一保留的NPZ类型；不是相机数据或权重。
- `evaluation.py`、`scoring.py`：真实观测前缀与自由展开指标；测试不能用于后续调参。

它依赖历史`object_recurrent_state_run`、`object_identity_readout`、`rgbd_observation_control`、`midstart_training_control`等缓存和模型；只有源码与配置被归档。先恢复本地缺失资产并校验哈希，再在新的副本中处理路径、预算和新冻结清单。不要删除audit hook或验签来原地重跑。

原历史测试有些会写回原`runs/`，本次没有执行，避免改变已存结果。此次验证只检查快照文件完整性、语法、JSON及凭据模式，不是新增的实验有效性结论。
