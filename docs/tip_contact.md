# 刀尖接触材料坐标与滑移观测

给定竖直刀具行程d，输出刀尖附近所对应的**原始上表面材料位置**相对名义初始落刀点的偏移Δs₀。该量用于根据独立实验起裂深度查询理论切点；它不是起裂判据，也不是累计摩擦滑移。当前模型仍为完整体。

## 定义

初始表面局部坐标为s，初始落刀点s₀=7 mm。最低刀尖是固定身份的圆弧顶点，随刚体刀具运动，不每步重新选择材料最低点。

对原始上表面边的当前端点xᵢ、xⱼ，把当前刀尖投影到线段：

```text
α = clamp(((x_tip−xᵢ)·(xⱼ−xᵢ))/|xⱼ−xᵢ|², 0, 1)
x_contact = (1−α)xᵢ+αxⱼ
X_contact = (1−α)Xᵢ+αXⱼ
Δs₀ = s(X_contact)−7 mm
```

在原始上表面边中选择最近材料位置，只有间隙小于dhat才进入接触候选。无候选记`no_tip_contact`，不同材料坐标等距记`ambiguous`；共享同一材料端点不算多义。`unique`仅指最近投影唯一，不证明整个激活范围只有一个支路。不能把缺失补成0或沿用前步值。

IPC正间隙屏障不是严格零间隙接触。该观测回溯参考材料坐标，不把当前水平距离、当前弧长或累积相对位移当作Δs₀。它不改变平衡方程、接触力或历史。

## 圆鼻与整刃区间

最低点未接触时圆鼻侧面仍可能接触，故另输出整个圆鼻距离小于dhat的`nose_reference_intervals_mm`及相对s₀的`nose_offset_intervals_mm`。区间可不连续，不跨空隙合并，不将直侧面并入圆鼻。

`blade_contact_footprint`覆盖圆鼻、两侧楔面、平行刀身及闭合顶边。对变形材料边与刀具线段邻域作参数裁剪，允许捕获端点均不活跃而边中部接近的部分；各段区间先取并集再计长度，避免重叠相加。

| 字段 | 含义 |
|---|---|
| `reference_s_intervals_mm` / `reference_offset_intervals_mm` | 原始表面材料区间及相对落刀点偏移 |
| `supported_reference_length_mm` | 区间并集的参考长度 |
| `supported_current_length_mm` | 同一材料子段当前长度，不是刀具弧长 |
| `supporting_top_edge_count` / `parts` | 涉及边数与刀具分区统计 |
| `status` | `no_support`、`connected`或`disconnected` |

这些是有限dhat下的几何支撑区间，不是压力分布、黏着区或累计滑程。最低点坐标平台不意味着整刃接触区停止增长。

## 输出与查询

`result.json`保存每个接受态的`tip_contact`与可用的`blade_contact_footprint`；`contact_trace.csv`提供逐深度表，区间用JSON数组。单值包括`status`、`reference_offset_mm`、`absolute_reference_offset_mm`、`reference_s_mm`、`reference_point_mm`、`current_point_mm`和`gap_mm`。先读状态再使用数值。

已有完整接受态可离线导出，无需重算力学；输出目录须独立：

```powershell
python -m prototype.contact_trace export --source outputs/EXISTING_COMPLETE_RUN --output outputs/NEW_CONTACT_TRACE
python -m prototype.contact_trace query --source outputs/NEW_CONTACT_TRACE --depth 1.5
```

查询返回精确接受深度或上下相邻记录，不默认插值或超出有效范围外推。观测异常为`error`，总体跟踪标记`partial`；力学完成不代表后处理一定完整。

## 解释边界

深度零为理想圆鼻相对未变形平面的名义首触。屏障激活、圆鼻侧触、最低点相触和视频首触不是同一事件；不得为匹配力曲线任意平移零点。初次新接触增量可能包含激活前运动，因此不能直接称为严格接触后累计滑程。

理论偏移与实验切点的对应取决于材料、几何、空间离散、屏障尺度、实验标定与可见性。起裂后的完整体计算不描述新裂面或裂后恢复。观测定义由本项目给出，IPC近接触与离散依据见[文献导读](../references/README.md)。
