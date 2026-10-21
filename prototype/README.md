# 主体代码地图

当前主线为原生P1/improved-max IPC + 有限等容Maxwell。先读[研究状态](../docs/research_status.md)和[规格](../docs/current_model_spec.md)。

| 层次 | 模块 | 职责 |
|---|---|---|
| 输入与几何 | `model_config.py`、`initiation.py` | 坐标、试样/刀具、完整体网格；initiation文件名不表示当前启用断裂 |
| 材料核 | `../mechanics/finite_strain.py`、`material.py` | 有限应变响应、局部历史及切线；2023流动律与当前Maxwell显式区分 |
| 物理主线 | `ipc_reference.py`、`ipc_viscoelastic.py` | P1、原生接触、摩擦PTC、CCD、材料历史提交和物理验收 |
| 算例生命周期 | `material_case.py` | 共用构造/签名、准确双历史恢复与审查、明确单位的实验输入 |
| 线性后端 | `pardiso.py`、`linear_solve.py` | 与AL无关的可选PARDISO及受真实残差保护的LU复用 |
| 观测 | `tip_contact.py`、`contact_trace.py` | 原材料接触身份、轨迹、缺失与邻近区间 |
| 运行保障 | `../run_support.py` | 原子输出、冻结源码、后台进程及资源策略 |

## 独立诊断，非默认

`history_sensitivity.py`：逆黏性度量的SPD/固定行列式retraction、共同G0/q/τ坐标、
摩擦上一位置解析作用−Hfriction·x'previous及独立材料应力装配。
`../tools/probe_history_sensitivity.py`沿保存的准确双历史传播三个共同参数方向；
不是固定上一状态的条件偏导，也不是独立重新加载。双链/相邻两档实际残差与反力资格未通过即停止。
[方法与资格](../docs/joint_history_sensitivity.md)。不参与生产材料更新；旧τ诊断源码保留在其输出归档。

`finite_width_probe.py`与`../mechanics/finite_strain_3d.py`：既有受载场的有限宽相容面外模式及三维Maxwell材料核。
保留准确上一接受历史、原面内接触切线，检查释放面外约束的条件方向；没有面外摩擦和跨宽面内翘曲，
不是三维切割求解器、不能恢复加载或替换主线。[方案与限制](../docs/finite_width_diagnostic.md)。

`transverse_control.py`：共同材料的总应力平面应力诊断，显式凝聚面外伸长；
体积/厚度/局部应力增加独立验收，约束标签进入签名。原生面内接触与自接触保持；
不是有限宽三维解、不替换默认。见[因素研究](../docs/angle_amplitude_factors.md)。

`transported_blade.py`：P1参考边界面积积分、解析刚性刀具、分段路径切向滑移和完整整体切线。
整体刀具项替换，原生材料自接触完整保留；解析刀具外包整边CCD及额外加载路径验收。
仅刀具平移；固定εu和原生材料自接触的已知限制保留。三角度预检通过，尚未实验验证或晋级默认。
实际执行状态及范围见[独立加载](../docs/transported_blade_validation.md)。

`surface_contact_transport.py`：局部正权重客观库仑传输、激活时刻及完整局部Jacobian、
试探/接受历史、脱离/再接触、单点/曲线间隙路径检查。32个真实有载局部方向通过，
但材料平滑几何不等同P1；VV、端部、全局积分和整边/自接触CCD未覆盖，禁止混入原生组合。
实现与下一整体设计见[接触资格](../docs/contact_transport_qualification.md)。没有接入生产FE。

`surface_history.py`：C²材料表面参数、跨样条节点的弧长滑移与解析功共轭力。
持续正权重接触通过八项夹具检查；未覆盖接触事件/自接触/完整FE切线，且与P1几何不等价，不能替代原生EV/VV。
`ipc_viscoelastic.py`/`material_case.py`新增显式`volumetric='log_j'`，默认不变，体积律进入历史签名。
核查与限制见[结构审查](../docs/model_structure_review.md)。

`contact_controls.py::objective_edge_contact`：持续同一条边的客观滑移、力、力矩及完整刷新切线。
只接受正权重与旧/新均在边内部的投影；不是原生EV/VV的可直接替换项，未接入FE。
实场覆盖不足和验证结果见[物理资格](../docs/physical_qualification.md)。

`../tools/run_material_uncertainty.py`：复用材料算例生命周期，以较低共同G0检验松弛份额；
仅0/33°独立路径，保留57°求解门槛。观察实际PTC局部变量而不修改方向/验收，失败保留准确历史。
参数、旧对照与识别限制见[材料审查](../docs/material_uncertainty.md)。

`residual_trust.py`：列缩放LSMR/真实残差下降候选。实际57° .135 mm重放仍失败，
零接受增量，**未晋级、不进入长线**。保留用于重现否定结果；见[审查](../docs/contact_solver_review.md)。

2026-10-21：`contact_controls.py`提供显式速度正则对照和局部客观滑移核。
直接删相邻VV的分支已被跨特征力连续性反例否决，仅为重现否定证据保留，不能用于长算例。
局部客观滑移尚未接入全局FE；实际六条长验证仅改变后零深度步长/正则尺度。
见[资格与计划](../docs/contact_time_validation.md)。

| 模块 | 用途和限制 |
|---|---|
| `unrelaxed_control.py` | 固定G₀/K去松弛极限，摩擦仍有历史；没有实验材料资格 |
| `inclined_loading.py` | 显式法向/切向控制，与实际竖直下压分开 |
| `curve_contact.py` | 解析圆鼻积分诊断，深路径失败记录保留摘要 |
| `ipc_mini.py`、`pressure_pairing.py`、`mixed_system.py`、`macro_element.py` | 压力/位移配对、MINI及体积响应 |
| `ipc_p2.py`、`ipc_viscoelastic_p2.py`、`compact_p2_assembly.py` | 局部高阶及整单元J；不是已验证混合不可压单元 |
| `augmented_blade.py`、`continuous_al.py` | AL刀具接触；弱约束与整边几何需单独核查 |
| `al_globalization.py`、`al_performance.py` | AL求解及装配加速；PARDISO旧导入仍兼容 |
| `geometric_self_contact.py`、`native_geometric_contact.py` | GCP本体候选、保留原生刀具的组合 |
| `separated_self_contact.py`、`contact_refresh.py`、`shear_limited_contact.py` | 明确适用范围的界面/接触对照 |

旧乘子/IPM、FEBio与起裂相关模块因实际代码/回归依赖仍保留，不是推荐运行入口。无依赖的旧启动脚本已退役。工具调用和本地数据前提见[工具表](../tools/README.md)。
