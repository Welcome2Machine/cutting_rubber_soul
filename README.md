# 橡胶斜切仿真

用同一套材料与接触模型，计算0°、33°、57°橡胶下压的Fz、表面运动和刀尖接触材料的原始位置。现阶段只算完整材料；实验近线性及跨角度响应是检验标准。

**首先阅读：[当前研究状态](docs/research_status.md)。** 该文说明完成结果、失败位置、证据边界和下一步重点。

当前工作模型是有限应变P1 + 原生improved-max IPC + 等容Maxwell。最近共同参数检验已结束，六组0°/33°算到0.5 mm，三组57°失败；没有参数组通过联合力值与斜率门槛。已有三角度2 mm结果和可恢复的3 mm弹性基线，但尚未完成跨角度实验验证。

## 文档与入口

| 内容 | 入口 |
|---|---|
| 当前结论、未解决问题、结果位置 | [研究状态](docs/research_status.md) |
| 研究约束与执行规则 | [任务要求](docs/task_requirements.md) |
| 几何、本构、接触、数值参数 | [模型规格](docs/current_model_spec.md) |
| 方程与代码对应、正确性边界 | [实现核查](docs/mainline_audit.md) |
| 最近队列与本地HTML | [运行状态](docs/solver_status.md) |
| 主体模块与诊断工具 | [代码地图](prototype/README.md)、[工具入口](tools/README.md) |
| 文献及阅读程度 | [文献目录](references/catalog.md) |
| 清理范围、保留与删除记录 | [整理记录](docs/cleanup_log.md) |

## 运行与测试

依赖见`requirements.txt`。本地验证使用Python 3.10、已验证的`.runtime/ipc-python`接触内核；可选PARDISO位于`.runtime/pardiso`。IPC需要improved-max、接触刷新导数和原生静态障碍过滤接口，不能仅凭相同版本号认定另一份二进制等价。运行环境、实验数据和历史检查点不随Git发布。

Windows PowerShell，从仓库根运行：

```powershell
$env:PYTHONPATH="$PWD;$PWD/.runtime/ipc-python"
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:SOFTMODELING_PARDISO_RUNTIME="$PWD/.runtime/pardiso"
python -m unittest discover -s tests -p "test_*.py"
python -m tools.run_common_material_study --help
```

回归包含独立小型算例，也包含明确标记的本地历史夹具；没有夹具的克隆会跳过相应历史检查。不要把跳过当成验证通过。

`run_common_material_study`是本轮有预算、有门槛的研究入口；`prepare`依赖状态文档列出的本地实验和基线，缺数据时不能直接启动。历史`run_material_candidate`支持其它诊断加载策略，不代表本轮固定步长方案。**清理重构后，不把新源码直接接到旧冻结路径上续算。**

远端只上传代码、文档和少量模型配置。`outputs/`、论文原文、视频、图片、CSV和`.runtime/`留在本地。退役摘要仅用于追溯，不是可续算的完整路径。
