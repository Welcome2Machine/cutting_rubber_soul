# 当前科研原型代码地图

更新：2026-11-02。这里只描述活跃调用关系；执行范围见[科研原型任务](../../../researchtopic/research_prototype_task.md)。

## 活跃路径

| 研究输出 | 入口 | 核心模块 |
|---|---|---|
| 完整体受力、变形和接触 | `tools.run_material_candidate` | `material_case`、`model_config`、`geometry`、`ipc_viscoelastic`、`ipc_reference`、`material`、`mechanics/finite_strain` |
| 起裂候选指标与危险位置 | `tools.fracture_onset` | `workbench.results.saved_state_fields`、已有应力/能量恢复和邻域平均 |
| 裂后分离、新面接触与载荷传递 | `tools.run_coupled_growth` | `coupled_cohesive`、`coupled_contact`、`face_cut`、`face_cohesive_quadrature`及其既有基类 |

面级P1内聚与真实历史已接入当前耦合机制路线；旧“尚未接入”的说明属于历史阶段。
当前裂后路线限0°居中、预设mode-I路径，不能直接替代s=7 mm多角度完整体。

## 当前构造依赖与整理边界

现行耦合构造仍通过`tools.run_precrack_entry`、`tools.run_endpoint_face_route`、`tools.run_fracture_route`、`tools.run_zero_gap_cut`复用工厂，含运行时换类和模块函数替换。
这些依赖是已知维护问题；本轮固定一条已使用路径，不再叠加新派生层，不把清理变成重写求解器。

`zero_gap_cut`、`energy_cut`等历史模块仍是现行基类和来源哈希的一部分。冻结FFM/旧径向/全网络等研究用途，不等于删除其源码。
若某个真实原型阻断必须整理构造，限于最小工厂/参数传递调整，物理方程与历史提交同时保持回归。

## 保留的基础能力

- `ipc_reference`：P1装配、原生EV/VV/自接触、CCD、正J和真实非线性残差。
- `ipc_viscoelastic`、`material`、`mechanics`：有限应变Maxwell、物理时间和接受后历史提交。
- `face_cut`、`face_cohesive_quadrature`：精确完整连接、固定面求积历史、参考面积、Gc和新裂面身份。
- `material_case`及现有独立审查：来源、状态链、物理量重组和能量分账。
- `tip_contact`、`contact_trace`：接触材料迁移；不能自动称为真实界面滑移。

基线索引仍为`intact_2mm_baseline.json`，结果不随Git发布；使用前检查存在性与签名。
数学与参数见[数学模型](../docs/mathematical_model.md)和[当前规格](../docs/current_model_spec.md)。

## 历史地图

[调整前代码地图](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/78405e242bc4246d269dda5ff5bdf11814b1c0fa/projects/softmodeling/prototype/README.md)保留各专项路线与阶段性说明。
详细事实仍在[研究状态](../docs/research_status.md)和[方法说明](../../../docs/fracture_implementation.md)；不以历史“下一步”覆盖当前用户目标。
