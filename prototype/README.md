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
