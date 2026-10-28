# 主体代码地图

完整体计算主线为 P1 三角形、原生 improved-max IPC 接触与有限应变 Maxwell 材料，行程上限为 2 mm。2026-10-28 新增独立的保存态起裂诊断和 0° 有限间隙内聚界面研究分支；原完整体算法与基线保持。

先读[当前研究状态](../docs/research_status.md)、[任务要求](../docs/task_requirements.md)和[模型规格](../docs/current_model_spec.md)。

[数学模型说明](../docs/mathematical_model.md)集中推导平衡、本构演化与一致切线、法向/摩擦残差和PTC算法；其中代码索引与配套数值核查用于解释当前实现，不能替代真实路径审查或实验标定。

| 模块 | 职责 |
|---|---|
| `model_config.py` | 几何、材料及 P1 网格配置；保留原路径的配置签名兼容性 |
| `geometry.py` | 刀具和试样几何、分级 P1 网格；可选直刃碰撞边细分保留原轮廓和刀尖编号 |
| `material.py`、`../mechanics/finite_strain.py` | 平衡 Ogden 材料、有限应变 Maxwell 更新及切线；另保留 2023 论文复现所需材料律 |
| `ipc_reference.py` | P1 力与切线装配、刀具及自接触、CCD、正 J 检查与 PTC 非线性求解 |
| `ipc_viscoelastic.py` | 黏弹性增量问题、材料历史及耗散管理 |
| `cohesive.py`、`cohesive_cut.py` | 未标定的径向双线性界面、两侧独立网格、完整裂面接触和共同历史提交；限预设 0° 路径 |
| `cohesive_mixed.py` | 共转四节点、两Gauss点的混合模式强度/共同Gc势；不可逆损伤、解析残量/切线，非比例加载能量闭合 |
| `cohesive_closure.py` | 可选双侧有限线段距离闭合势的解析几何导数；修正远距滑开配对仍承压的缺陷，仍需独立切割/间隙/能量验证 |
| `cohesive_network.py` | 所有内部网格边的候选界面拓扑；独立三角形、真实IPC面、参考间隙/滑移、底面身份和面积损失 |
| `material_case.py` | 完整体加载流程、检查点、独立物理审查与 CSV 输出 |
| `material_performance.py` | 可选材料分块及线性后端配置；单独记录性能策略签名 |
| `pardiso.py`、`linear_solve.py` | 稀疏线性求解与后端选择 |
| `tip_contact.py`、`contact_trace.py` | 刀尖接触的参考材料位置、整刃支撑区域及轨迹输出 |

Maxwell 加载入口为 `python -m tools.run_material_candidate`，参数与串行队列说明见[工具入口](../tools/README.md)。
起裂后处理入口为 `python -m tools.fracture_onset`，耦合代表算例为 `python -m tools.run_cohesive_cut`；均要求独立输出和预算。合并桌面/API 的显式入口及证据见[断裂实现说明](../../../docs/fracture_implementation.md)。`tools.fracture_labels`区分物理起裂、可见裂口、力变化代理及组截止；不自动配准试次或标定阈值。
本地HTML入口为 `python -m tools.workbench --background --open`，只将已核查配置转换为同一加载命令。`workbench/config.py`负责输入，`server.py`负责单任务与预算，`results.py`负责保存场导出；没有复制核心求解器。`material_case.make_problem()`及`IPCProblem`可接收刀具厚度/楔角、圆鼻和网格参数，未传参数时保留原路径默认值。
`python -m prototype.ipc_reference` 是弹性参考入口；其 `ipc_baseline.json` 不能替代 Maxwell 共同材料基线。

`intact_2mm_baseline.json` 定义共同材料基线，并索引本地 `outputs/baseline/angle0`、`angle33`、`angle57` 的结果及审查指纹。结果文件不随代码发布，使用前需检查其实际存在性与签名。

每个非线性增量固定上一接受态的位置和材料度量，仅在物理验收后提交新历史。原生接触保留完整 EV/VV、自接触及 CCD；性能实现由独立原生对象审查，不以近似残差替代物理验收。运行期间不得编辑参与计算的源码；记录源码指纹，不复制冻结源码树。

旧混合元、相位场、接触替代及一次性诊断脚本已从当前入口移除。历史结论见[研究历史](../docs/research_history.md)，文献导读见 [references](../references/README.md)，独立论文复现保留在 [reproductions](../reproductions/dammass2023/README.md)。

专项研究内核（默认切割不切换）：`force_globalization.py`为可选真实残差Newton/LM，仍需全物理审查；`extrinsic_topology.py`为零间隙连接/释放算子，未接入零距离裂面接触；`cohesive_dynamics.py`为标量惯性内聚能量验证，非二维切割动力学。验证与限制见合并仓库断裂说明第18–19节。
