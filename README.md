# 橡胶斜切仿真

用同一套材料与接触模型，计算0°、33°、57°橡胶下压的Fz、表面运动和刀尖接触材料的原始位置。现阶段只算完整材料；实验近线性及跨角度响应是检验标准。

**首先阅读：[当前研究状态](docs/research_status.md)。** 该文说明完成结果、失败位置、证据边界和下一步重点。

当前执行：[解析历史偏导与G0/q/τ联合检查](docs/joint_history_sensitivity.md)、
[HTML状态与分析](outputs/joint_history_sensitivity_20261021/index.html)。
修正诊断摩擦上一位置偏导，两个旧失败夹具重解后通过；60项相关回归通过。
10月22日00:00:28诊断结束：0/33/57°分别通过.33/.32/.31 mm，共99态，均因预算停止，非力学失败。
.5 mm计划未完成。共同窗口内q/τ强耦合；同时改善三角度幅值和斜率演化的局部方向存在，
但力值RMSE预计仅改善约.13–.24%，未得到可晋级参数。原物理和默认保持，没有新FE加载。

此前核查：[共同参数与模型结构的可判定性](docs/model_identifiability_audit.md)。
已有结果不能排除共同材料参数路线；固定旧历史偏导不等于完整加载敏感性，57°也未通过形状验证。
本轮核对理论/实现，并沿保存场传播双历史导数；不新增加载或修改默认。
[HTML总报告](outputs/model_identifiability_audit_20261021/review/index.html)、[实际诊断日志](outputs/model_identifiability_audit_20261021/history_tau/index.html)。
以下条目为此前检查结果。

最新：[有限宽相容性检查](docs/finite_width_diagnostic.md)已完成六个真实受载场、21个条件方向；
主方向全部通过双尺度检验，但没有出现明显的小角度专属释放，不把条件导数当新Fz曲线。
底面不滑动和1–3 mm下压时单侧面外外鼓不到约1 mm已作为实验约束记录。
[HTML分析与运行证据](outputs/finite_width_probe_20261021/review/index.html)。无新加载队列，主线默认不变。

最新：[小角度幅值偏差与大角度保持](docs/angle_amplitude_factors.md)。不再以扩大0/33°差距为目标。
三角度9态/45个条件方向已完成，35个通过双尺度验证；新平面应力对照保持共同材料，检验面外约束因素。
队列已结束：0/33°完成.5 mm，末力8.618/8.680 N（实验8.27/7.61 N）；57°接受.16、在.17 mm停滞。
小角度幅值明显改善，但57°共同前段偏低加重、局部厚度伸长明显，未通过三角度联合验证，不替换主线。
[终态分析与审查HTML](outputs/angle_factor_study_20261021/review/index.html)、[状态/日志](outputs/angle_factor_study_20261021/index.html)。

最新：[刀尖与局部网格对照](docs/tip_radius_study.md)。粗网格六组已完成.5 mm；刀尖20→10 μm仅降力约1.6–1.8%，空间收敛未证明。
四组0/33°×R10/20 μm细化表面5→2.5 μm已结束：三组完成.5 mm，R10/0°在.35 mm发生局部PTC停滞。
三个完整末力只降.44–1.94%；[HTML结果与实际方向诊断](outputs/tip_mesh_20261021/failure_review/index.html)。
材料、摩擦、dhat、步长和物理验收保持，不按角度调参，不替换默认网格或半径。

最新终态：[实验对比与33°失败定位HTML](outputs/transported_blade_20261021/review/index.html)。
解析刀具候选57°完成0.16 mm、0°完成0.5 mm；33°因未处理的步内短暂接触停在0.01 mm。
0°0.5 mm力仍偏高38.8%，与旧原生结果近乎相同。队列已结束，本轮没有新开计算，候选未晋级。

最新：[P1边界与解析刀具的独立加载](docs/transported_blade_validation.md)。完整刀具接触积分、
路径滑移与切线已接入独立候选；139项回归、三场积分/整体切线资格及三角度0.01 mm预检通过。
有界队列状态见[HTML](outputs/transported_blade_20261021/index.html)，不将预检通过当作实验复现完成。

最新：[接触历史与完整切线资格](docs/contact_transport_qualification.md)。局部传输核、激活历史和32个
实际有载切线通过，124项回归通过；材料几何平滑改变接触状态，整体积分及自接触资格未完成。
本轮无新FE加载，详细证据及运行日志见[HTML](outputs/contact_transport_20261021/qualified/index.html)。

最新：[模型结构与实际承载场核查](docs/model_structure_review.md)。全宽贴合已确认；
六态/18方向检验后不继续局部材料拟合。新增跨节点表面历史原型和显式体积律选项，101项回归通过，
完整接触尚未通过FE资格，没有新开长路径。

最新：[物理资格核查](docs/physical_qualification.md)。四条新共同参数路径已结束，均未通过联合筛查。
实际材料切线/转动检查通过；局部客观接触力修正通过测试，但完整跨边/端点处理尚未合格，未新开长验证。

最新：[材料参数不确定性判读与试验](docs/material_uncertainty.md)。
四条0/33°半步对照现均完成1 mm；33°仍比实验高约28%。新试验固定较低瞬时刚度，
减弱松弛份额，共同参数在0/33°做独立路径；主线默认不变，57°接触切换尚未修复。

当前工作模型是有限应变P1 + 原生improved-max IPC + 等容Maxwell。最近共同参数检验已结束，六组0°/33°算到0.5 mm，三组57°失败；没有参数组通过联合力值与斜率门槛。已有三角度2 mm结果和可恢复的3 mm弹性基线，但尚未完成跨角度实验验证。

2026-10-21：时间正则对照及IO补齐已结束，0/33°完成、两组57°在.135 mm停滞。
直接删相邻VV因跨特征力跳变被否决，局部客观滑移核尚未接入FE。
新队列实际状态与HTML入口见[运行状态](docs/solver_status.md)。

## 文档与入口

| 内容 | 入口 |
|---|---|
| 当前结论、未解决问题、结果位置 | [研究状态](docs/research_status.md) |
| 研究约束与执行规则 | [任务要求](docs/task_requirements.md) |
| 几何、本构、接触、数值参数 | [模型规格](docs/current_model_spec.md) |
| 方程与代码对应、正确性边界 | [实现核查](docs/mainline_audit.md) |
| 最新接触运动学、时间正则与实际松弛证据 | [接触/材料审查](docs/contact_material_audit.md) |
| 本轮接触资格与六条长路径对照 | [验证方案](docs/contact_time_validation.md) |
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
