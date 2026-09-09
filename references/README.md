# 参考方法索引

论文原文件仅留在本地，不上传 GitHub。面向 AI 的结构化摘要、原文定位、来源链接、适用边界和代码映射集中在 [method_notes.md](method_notes.md)。这些笔记是自行归纳的研究说明，不是论文全文或作者认证的实现规范。

| 文献编号 | 用途 |
|---|---|
| R01 — Spagnoli 2019 | 斜切运动、摩擦及能量分解 |
| R02 — Dammaß 2021 | 小应变黏弹性与相场的双向耦合 |
| R03 — Dammaß 2023 | 有限变形材料、相场、二维应力假设及独立复现 |
| R04 — Moreno-Mateos / Steinmann | 软材料切割的体材料、黏附与界面机制 |
| R05 — Pinto Carvalho 2022 | 刚体—可变形体无摩擦 dual mortar |
| R06 — Khandelwal 2022 | 二次边界接触自由度及误差分析 |
| R07 — Maas 2011 | 节点积分、稳定化和锁死权衡 |
| R08 — Acary / Brémond / Huber | 离散库仑摩擦求解器及尺度比较 |
| R09 — Lopez-Pamies 2025 | 独立强度与无缺口起裂预测 |
| R10 — Kamarei 2025 | 黏弹性、广义强度面与断裂能耦合候选 |
| R11 — Lopez-Pamies 2026 | 受强度约束的能量最小化预印本 |

求解器实现另参考 [FEBio v4.13](https://github.com/febiosoftware/FEBio/tree/v4.13)、[SuperLU_MT](https://github.com/xiaoyeli/superlu_mt/tree/8324379)、[FEBio 节点—面接触](https://febiosoftware.github.io/febio-feature-manual/features/solid_surfaceinteraction_sliding-node-on-facet/) 及 [GetFEM 大滑移](https://getfem.org/userdoc/model_contact_friction_large_sliding.html)。这些是实现依据，不是本实验验证。

[Zimmerman–Ateshian 2018](https://doi.org/10.1115/1.4040497) 及 [作者模型库](https://repo.febio.org/permalink/project/68) 可作为后续独立大变形摩擦基准；本轮未取得该论文全文，不声称复现其算例。[检索记录](../docs/literature_search_log.md) 保留查询和证据范围；[模型规范](../docs/current_model_spec.md) 的 CB245NE 参数不被论文替换。
