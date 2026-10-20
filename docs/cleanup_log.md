# 仓库整理记录

2026-10-20。任务是整理、核查、提交和推送；没有启动新的加载算例。
当前研究结论见[研究状态](research_status.md)，源码入口见[代码地图](../prototype/README.md)。

## 输出清理

清理前`outputs`有28,082个文件，共32.20 GiB。删除76个过时输出目标及冗余备份展开目录，共22,848个文件、24.99 GiB；增加精简证据和本轮核查记录后，约保留7.28 GiB。临时测试日志大小会有少量变化。

保留当前共同材料九组试验、原生三角度2 mm路径、最新完整3 mm路径及其验证、最早两条3 mm基线、2023材料复现和直接支撑当前判断的审查结果。用户实验数据、视频、论文原文和运行环境未删除。

退役算例先提取状态、参数、结论、失败原因和文件哈希到`outputs/retired_evidence`；长数组明确标记截断。MINI、深部解析接触和AL/P2的选定失败另保留失败场、最后接受态及冻结源码。**精简摘要不能代替完整历史或用于续算；已删除的原图和完整路径不再声称可用。**

本地证据：

- [删除清单](../outputs/repository_cleanup_20261020/deletion_completed.json)
- [清理前清单](../outputs/repository_cleanup_20261020/inventory_before.json)与[清理后清单](../outputs/repository_cleanup_20261020/inventory_after.json)
- [保留证据说明](../outputs/retired_evidence/README.md)
- [HTML核查报告](../outputs/repository_cleanup_20261020/index.html)

## 备份与结果完整性

修改前255个代码/文档文件写入本地ZIP，逐项读回并核验SHA256，见[源码备份清单](../outputs/repository_cleanup_20261020/source_backup.json)。不是对已删除输出再做一份隐藏的大备份。

原完整3 mm备份ZIP经CRC读回及SHA256核对，与原验证记录一致。保留的12组当前/基线算例共948个接受态，其检查点哈希和冻结源再次核验；另核对302态3 mm路径的冻结源。这里是文件完整性核对，没有冒称再次求解全部路径。见[保留结果核查](../outputs/repository_cleanup_20261020/retained_verification.json)。

## 主体结构整理

- `prototype/material_case.py`统一材料算例构造、签名、准确双历史恢复、场审查与实验CSV读取。旧CLI继续兼容导入。
- `prototype/pardiso.py`独立提供线性后端，原生主线不再为了使用PARDISO而导入AL/P2求解模块。
- 搬移的七个函数/类主体经AST核对保持一致；本轮不改变本构、接触方程、物理验收或恢复指纹规则。见[结构核查](../outputs/repository_cleanup_20261020/structural_moves.json)。
- 删除两个无人引用的旧入口和七个已退役工具；删除重复的日期快照文档。保留仍被运行、测试或技术诊断使用的模块，不按文件年龄误删依赖。
- 修正MINI测试指向不存在归档位置的导入。将历史输出失效链接改为明确标记的退役摘要，修复部分历史文档编码损坏。

源码布局改变后，不放宽哈希限制续算旧检查点。旧有效结果的冻结源码保持原样。

## 测试与发布

结构调整后完整测试652项：649通过，3项因原本缺失的本地历史夹具跳过，0失败。清理输出与删除旧入口后再次运行同一测试集，核对清理没有额外减少测试覆盖。详细结果与日志见[本地HTML报告](../outputs/repository_cleanup_20261020/index.html)。

Git仅发布代码、Markdown和批准的小型配置；结果、CSV、图片、视频、PDF、二进制及运行环境不上传。发布检查覆盖当前提交的全部可达历史。测试通过不表示57°失效已解决，也不表示跨角度实验验证完成。
