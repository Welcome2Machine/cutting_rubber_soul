# 代码结构

根目录只有 `softcut.py` 一个 Python 入口，按子命令延迟导入功能。当前计算和展示不再从根目录实验脚本相互导入。

| 位置 | 职责 |
|---|---|
| `src/io.py` | 路径、哈希、原子 JSON、无 pickle 数组和变长数组 |
| `src/video_sources.py` | 视频来源、采样时钟和输入配置 |
| `src/segmentation.py` | 当前模型的网络、输入处理与分割推理 |
| `src/tracking/` | 固定网格去除、去阴影、独立材料块跟踪 |
| `src/analysis/` | 刀具参考、边界/材料坐标、相对运动、MMT/OBT、时序输运 |
| `src/workflows/` | 导入、统一批处理、跟踪、测量、滑移和展示 |
| `src/exchange/` | 有限元交换数据源、观测、导出预览与独立验证 |
| `reference/` | 机器可读书目、中文导读；公开全文仅保存在本地 |
| `tests/` | 当前有效的回归测试 |

主要合并：

- 多个跟踪入口合为 `workflows/tracking.py`，清单调度合为 `workflows/batch.py`。
- 仅保留现有分割模型需要的网络与推理；旧训练、标注、神经跟踪和精修链删除。
- 多个滑移 runner 合为 `workflows/sliding.py`，输入准备集中在 `contact_inputs.py`。
- 时间滤波共用 `contact_transport_filter.py`，通过明确模式保留数值规则；求解类集中在 `dual_contact.py`。
- 刀尖参考保护和深度截断合入 `loading_blade_reference.py`；材料局部拟合、报告插值并入对应主体。
- 无文字视频、静态曲线和 PPT 压缩集中在 `workflows/gallery.py`，只读取已有估计，不夹带单个实验的筛选逻辑。
- 有限元数据源、轨迹和可选证据、导出与校验收拢到四个模块。

合并保留了方法间有物理意义的差异。清理时旧轨迹通过复用校验，同一分割权重的新旧网络输出逐位一致；共享滤波已用随机和真实输入做等价检查。后续身份优化扩展了跟踪文件中的核验与诊断，源码有意改变。当前相对 `98ec115` 的默认数值回归使用六帧已知仿射合成纹理及两种既有核验模式，154 个数组完全一致；不宣称所有输入等价、实际视频精度已验证或 AST 未改变。[证据](../results/20261025_surface_identity/default_equivalence.json)

`data/` 和 `results/` 不移动、不上传。`configs/retained_results.json` 记录成果依赖，`python softcut.py check` 只读核查链接和数组归档。23 组完整结果、五组展示、有限元交付物路径保持不变。

分割模型保留 `models/segmentation/clean_roi_20260909/` 及训练摘要。研究候选另部署 `models/point_tracking/cotracker3/` 和 `models/point_tracking/tapnextpp/`；各自包含官方源码、权重、许可证和哈希记录，不进入 Git。清理和等价验证的明细在本地 `environment/local_maintenance/`，正式摘要在 `environment/cleanup_20261024.json`。

研究候选仅由 `workflows/optimize.py` 调度：`tracking/finite_deformation.py` 负责有拓扑排除的有限变形网格图像修正，`analysis/surface_observability.py` 负责初始表面正向运输和可辨识性。两者均不由默认跟踪或滑移流程隐式调用。相关来源、部署与验收规则见 [优化方案](material_tracking_optimization_plan.md)。

`optimize --stage reliability` 延迟调用 `workflows/reliability.py`，统一四分支的数据、固定 ID、截止和报告；`analysis/blade_geometry_review.py` 扩展接触审计；身份验证复用 `material_patch_tracker.py` 的仿射核；力学项复用 `finite_deformation.py`；两个预训练网络共用一个 `neural_candidates.py` 适配层。没有增加根目录实验脚本，也没有复制第三套跟踪器。

`optimize --stage identity` 继续复用 `workflows/reliability.py`：`tracking/identity_chain.py` 只负责初始、近期与关键帧的固定 ID 连接和状态；单侧纹理核验仍在 `material_patch_tracker.py`，原始查询候选仍由 `neural_candidates.py` 生成。图像准备快照复用已有灰度、梯度和几何计算，参考更新即失效，不缓存通过结论。范围、证据含义与验收见[近表面身份优化](surface_identity_optimization.md)，未自动接入默认跟踪和滑移流程。
