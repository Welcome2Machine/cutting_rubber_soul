# 首版验收记录

日期：2026-10-26，JST。范围为工程迁移、接口整合及已有结果后处理；没有启动新的 FE 加载、训练或 GPU 跟踪。

## 复制与来源

| 项目 | 复制文件数 | 字节数 | 逐文件 SHA256 核对的源码/文本 | 不一致 |
|---|---:|---:|---:|---:|
| softmodeling | 2568 | 8,646,899,741 | 93 | 0 |
| rec_inc_soft_cut | 67461 | 47,276,764,749 | 152 | 0 |

总量 55,923,664,490 bytes，约 52.08 GiB。所有未排除文件验证存在性和字节数；表列源码/文本进一步逐字节哈希。没有宣称对全部大型视频、权重及结果做内容哈希。排除 `.git`、缓存、`.venv`、`.codex`；Git 历史另存 bundle，仿真 `.runtime` 已复制。详见 `provenance/import_manifest.json`。

源仿真 commit 为 `2879b4c355235c6b1379d74aaf22e337bd6128f1`，源视频 commit 为 `0d83e1b7c936b42672ca4b3420eda67469d3e7d4`。两源库已有的未提交修改状态保留，未修改原目录。旧 ForceProcessing 副本已移除且不纳入仓库。

## 自动检查

| 检查 | 结果 |
|---|---|
| 整合层 `python -m pytest -q` | 99 passed |
| 桌面 `python -m softcut_solution gui --smoke` | 正常构建、退出；未启动计算 |
| FE 工作台 config/results/server 回归 | 35 passed，192 subtests passed |
| 视频 CLI、batch、source/exchange 等相关回归 | 68 passed；一个 Windows 并行管道测试在解除工具沙盒限制后通过 |
| 两解释器及必要库检查 | Python 3.10/3.11 分离，所检库及权重可用 |
| 视频 0930l30a1 预检、三试次 batch 预检 | ok=true；路径为本副本；gpu_work_started=false |
| 新 FE 工作台启动 | 127.0.0.1:8876 健康检查通过；只启动界面服务 |
| Git 发布检查 | 导入提交和整合提交通过；数据、媒体、状态、运行库不进入 Git |

整合测试覆盖已知仿射场、接受态参考、域外缺失、首个不合格 FE 状态后的屏蔽、不外推、原帧序、截止后不重入、遮挡/推断掩膜、实际输入指纹、路径重定位、作业错误/超时以及桌面线程/关窗生命周期。未运行全部子项目数值回归，因为数值源码未改；所选回归覆盖本次调用的接口。

## 真实跨项目验证

使用共同材料 baseline/angle33 与 baseline/angle57，分别导出 21 个实际接受态（0–2 mm），按条件视频深度在这些状态之间采样。输入 SHA256 前后不变。不是逐帧重新求解。

| 视频试次 | 帧数 | 总点数 | FE 域内映射 | 连续共同点 | 非单调深度步 |
|---|---:|---:|---:|---:|---:|
| 1014n33a1 | 406 | 1692 | 1691 | 1666 | 0 |
| 1014n57a1 | 401 | 1626 | 1595 | 1538 | 7 |

正式报告：

- [33° baseline 比较](../runs/20261026T062002Z_comparison_d004b53a/index.html)
- [57° baseline 比较](../runs/20261026T062003Z_comparison_1404d419/index.html)

上述路径为本地资产，不随 Git 发布。报告的 `implementation_checks_passed=true` 仅说明接口和数值实现检查通过；`laboratory_calibration_established=false`、`physical_accuracy_established=false`、`time_synchronization_established=false`、`intact_status=unknown` 仍明确保留。

## 首版边界

已实现桌面编排和固定材料点位移比较；同标距应变、区域距离、共同接触地址和经过确认的力—视频同步尚未接入统一比较器。视频解释器仍使用明确登记的外部现有虚拟环境，搬到新机器需配置。新仓库只在本地，尚未配置远端或上传。
