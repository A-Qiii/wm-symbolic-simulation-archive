# wm_symbolic 仿真代码与配置快照

截至2026-10-23的CALVIN世界模型与符号规划实现归档。仅保留代码、冻结配置、依赖记录、来源哈希与必要运行说明。**不包含生成的实验报告、汇报文档、结果摘要、图表、视频、原始数据、缓存或模型权重。** 原项目及其所有输出保留在本机，没有删除或覆盖。

原项目位于`/home/robot/research/projects/wm_symbolic`；与旧`robot_state`、Habitat/ROS项目独立。本次备份不代表原核心方案全部完成，也没有把后续托盘任务混入旧实验。

## 文件入口

|路径|内容|
|---|---|
|`src/wm_symbolic/`|世界模型、视觉缓存、符号接口及规划实现|
|`scripts/`|历史采集、训练、校准与单元/回归入口|
|`source/`|原始输入方案、官方资源出处、历史兼容差异|
|`configs/`|任务、标签及冻结协议|
|`experiments/*/source/`、`configs/`|各轮实验原样代码与小型配置；失败尝试也保留|
|`environment/`|三个独立环境的Python、实际包版本和宿主信息|
|`provenance/`|源码逐文件哈希、官方仓库版本、许可证和本地资产线索|
|`docs/`|数据生成、恢复步骤与不在Git中的文件说明|

最新扩采训练实现为`experiments/data_expansion_training_20261021/source/pipeline.py`。它使用固定视觉前端、ObjectAug对象循环模型与中途起点训练。训练、数据生成、原始路径和只读保护均保持历史版本，不应直接在原目录重跑。

## 恢复与运行

先阅读[复现说明](docs/REPRODUCIBILITY.md)与[本地保留项](docs/LOCAL_ONLY.md)。仅有代码快照不能重新计算历史结果：还需要原始数据、特征缓存、官方资源和冻结权重。某些配置中的绝对路径与截止时间属于原冻结身份，移植须在新副本中单独记录，不能无声改写。

离线完整性与源文件检查：

```bash
python3 tools/verify_snapshot.py
```

检查器不运行模型或仿真。已知历史文件`experiments/state_predicates_fit_diagnostic_20261015/source/train_diagnostic.py:32`有原存语法错误，按原样归档，检查器会如实返回非零；这不是最新训练入口。没有为归档而修改或删除原实验源码。

本快照未重新运行仿真、采集或训练，未声称空机器端到端复现通过。未新增开源许可证；第三方代码按其原许可证使用。
