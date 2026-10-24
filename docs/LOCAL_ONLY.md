# 不在本次Git备份中的文件

原项目`/home/robot/research/projects/wm_symbolic`保持原样。以下目录及其各实验对应目录均只保存在本机：

- `reports/`、`results/`、`figures/`、`deliverables/`：按用户要求，不上传任何生成报告、汇报文档、结果摘要及导出图表。
- `data/`：原始RGB-D、分割、轨迹、控制、标签、复位记录与视频。
- `cache/`：官方下载包、视觉与几何特征、中间数据。
- `models/`及所有`.pt/.pth/.ckpt/.safetensors`：官方权重、冻结编码器、训练检查点及优化器状态。
- `.envs/`、`.tools/`、`vendor/`：二进制环境、工具及官方代码目录；仅环境依赖、官方版本和许可证线索进入快照。
- `runs/`、`logs/`：原执行记录、锁、恢复文件和日志。
- 密钥、认证文件和`.env*`：明确禁止上传，未创建新的凭据。

最新模型`experiments/data_expansion_training_20261021/models/new868_seed0_step1500.pt`与`seed1`对应文件仍在本地，没有上传。关键本地资产的路径、大小和SHA256见`provenance/local_only_key_artifacts.json`。

唯一保留的两个小型NPZ是最新训练的`configs/schedule_seed0.npz`与`seed1`，每个384504字节，属于精确采样配置。它们不含原始观测或权重。

`provenance/copied_sources.json`逐文件说明本次代码/配置的来源和哈希，`SNAPSHOT_MANIFEST.json`列出快照自身文件。不应把Git备份说成全量仿真数据、报告或训练模型备份。
