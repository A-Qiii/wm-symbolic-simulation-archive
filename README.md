# wm_symbolic 仿真代码备份

本仓库是新建的独立私有备份，不覆盖任何既有仓库。

代码以可校验的归档保存：[wm_symbolic_code_20261023.zip](wm_symbolic_code_20261023.zip)。包内包含2313个文件：项目源码、各轮实验源码、冻结配置、三个环境的依赖记录、来源哈希及必要README/运行说明。展开后可直接浏览完整代码树；本仓库根目录不是展开后的源码树。

**按用户要求，生成的报告、汇报文档、结果摘要、图表及视频均未上传。原始数据、特征缓存、环境目录、官方vendor目录和模型权重也未上传，仍保存在原本机项目。** 未删除或修改这些本地文件，未创建新的GitHub凭据。

## 下载和检查

```bash
sha256sum -c SHA256SUMS
unzip wm_symbolic_code_20261023.zip
cd wm_symbolic
python3 tools/verify_snapshot.py
```

包内`README.md`与`docs/REPRODUCIBILITY.md`说明恢复官方依赖、数据生成与训练入口，`docs/LOCAL_ONLY.md`列出本地保留项。仅取得此包不能重新计算历史模型结果，仍需恢复未上传的数据及权重。

代码快照的本地独立提交为`3c8a9e92d6870097b8b0b8e1339ee3824d8be0d5`；ZIP不包含`.git`或任何含报告的旧本地提交历史。每个归档文件的SHA256记录在包内`SNAPSHOT_MANIFEST.json`，原始代码/配置来源在`provenance/copied_sources.json`。

只读检查保留了一个旧诊断脚本的既有语法错误：`experiments/state_predicates_fit_diagnostic_20261015/source/train_diagnostic.py:32`。因此语法检查整体会返回非零，不声称所有历史脚本都可直接运行。归档未改变原实验代码，没有重跑仿真、训练或采集，也不代表新托盘实验已经完成。
