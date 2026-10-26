# 视频与有限元的共同观测及分析规范

核查日期：2026-10-25。视频源仓库：`D:/Doctoral Research/Softcutting/0910imaging/rec_inc_soft_cut`。本轮仅只读检查其文档、代码和保存结果，没有运行、导入或修改该仓库的程序。另一个 agent 正在更新视频仓库，因此结论对应读取文件及指纹，不代表整个工作区的固定版本。

本文件规定如何把图像运动与有限元预测转换为同一类观测量。**现有格式、已完成的比较和拟议接口分别列出；共同接口尚未集成到工作台。** 既有交付字段及兼容规则见[光流接口](optical_flow_handoff.md)，FE 输出见[工作台规范](workbench.md)，接触量见[接触材料位置](tip_contact.md)。[本轮核查 HTML](../outputs/video_fe_interface_20261025/index.html)及同目录 `source_audit.json` 保存读取范围、指纹和数值检查。

## 1. 两个项目的分工及可检验对象

视频项目提供图像支持的运动、轮廓和条件接触地址；仿真项目计算完整体有限应变 Maxwell、摩擦接触、反力和内部状态。比较层负责参考态、单位、坐标、采样位置、空间标距及质量规则。比较层不改变任一原始结果，也不利用 FE 最佳匹配反向修改视频零点、比例尺或点位。

当前 FE 的速度固定 0.2 mm/s、加载上限 2 mm；外部视频约 3 mm 的行程标定与 FE 行程授权是不同概念。无断裂 FE 不能覆盖已确认开裂阶段的力学解释；未知起裂区间标为 `intact_status=unknown`，不把“前 1 mm”自动认证为完整体。

| 要检验的问题 | 视频能提供的约束 | 不能由此直接认定的量 |
|---|---|---|
| 橡胶是否被拖向刀具、两侧运动是否不对称 | 固定材料点位移、相对刀具运动、分区增量 | 摩擦系数、接触压力、黏附强度 |
| 局部压缩、伸长、转动是否合理 | 同标距二维有限变形、轮廓变化 | 三维体积比、点应力、材料模量 |
| 刀尖接触的是原表面哪里 | 有身份依据的原表面地址及其区间 | 内部近刀点的接触身份、累计摩擦滑程 |
| 力曲线接近是否靠不同机制补偿 | 力与区域运动的独立误差、不同参数的响应变化 | 单凭相关性辨认储能/黏性耗散/摩擦耗散占比 |
| 求解结果是否可信 | 实验运动可对模型构成外部约束 | 正 J、残差、CCD、材料历史等数值验收仍须 FE 自身完成 |

## 2. 视频仓库的实际处理原理

生产流程为：视频导入及源帧映射 → 材料/刀具分割 → 材料内去阴影 → 固定 ID 多参考局部仿射跟踪 → 测量支持与几何审核 → 刀具相对运动及原表面接触地址。

- **跟踪**：`src/tracking/material_patch_tracker.py` 使用初始、近期及关键帧模板，拟合局部平移/仿射变形和亮度增益/偏置，检查独立反向重拟合。LK 提供初值。当前默认流程不等同于神经网络光流。
- **去阴影**：平滑低频亮度增益，纹理仍来自当前帧，不将多帧材料纹理平均成测量证据。
- **支持**：`supported` 表示图像判据通过，不证明跨大变形后的原材料身份正确。无支持但有限的位置可能是预测。`measurement_support.npz` 又排除画面外、非材料区及刀体内点；保留原跟踪支持与测量支持两个字段，不能合成一个不可追溯的 `valid`。
- **时间**：播放视频 12 fps 不是物理时间；导入抽帧可能为源帧间隔 2/3 交替。物理时间需原 AVI 帧编号和采集频率，不能用播放时长或均匀本地帧差求速度。部分 200 fps 依据仍标为历史假设或未独立验证。
- **尺度/深度**：当前毫米量依赖声明的接触至反转约 3 mm 行程。刀尖图像竖向位移是条件刀具行程，不是表面法向压入，也未自动消除相机运动。生产接近阈值零点与后来材料运动起始零点是不同估计。
- **接触**：刀尖附近点到语义刀体约 0–4 px 的接近筛选是几何候选，不是实测机械接触。`MMT-v2` 组合材料运动/边界映射，`OBT-v2` 处理边界输运和共同刚体转动；二者估计原表面地址迁移，共用图像、轨迹及零点，不是独立重复测量。
- **研究候选**：CoTracker/TAP 候选及身份链仍需图像/身份核查。有限应变修正分支另外引入平面应变、可压缩 Neo-Hookean 内部平衡约束；其结果含力学先验，不能独立证明本项目的平面应变 Maxwell 模型正确。

实现依据：`src/workflows/tracking.py:175`、`src/workflows/measurement.py:159`、`src/workflows/tracking_inputs.py:265`；跟踪、时间、接触和候选的进一步定位见第 10 节。代码现状不自动代表旧结果的生产版本，结果自身 provenance 优先。

## 3. 结果家族、完成范围与实际不兼容项

以下路径均相对视频仓库的 `results/`。

| 目录 | 保存内容 | 对照用途及限制 |
|---|---|---|
| `20261012_optical_flow_exchange` | 10 组 0930 试次，CSV 轨迹、帧表、元数据、质量报告 | 正式旧接口；初态位移须 `valid=1 AND reference_frame_id=0`，断点后的新段不能拼回原位移 |
| `20261014_1005_1014_batch` | 分割、去阴影、固定 ID 原轨迹和测量支持 | 后续派生结果的来源；原位置含预测，不能不筛选直接比较 |
| `20261015_declared_fracture_sliding` | MMT-v2 / OBT-v2 原始 NPZ、配置及完成记录 | 截止深度来自声明，不是算法检测到起裂 |
| `20261016_contact_gap_interpolation` | 原估计和展示层补点、锚点、补点类型 | 使用原始地址及质量状态；完成的展示曲线不能替代缺失观测 |
| `20261024_n_group_first_1mm` | 33°/57° a1 固定 ID 位移、帧表、标定依据、检查视频 | 当前最适合先接入的已完成位移交付；没有 0°，不保证无裂 |
| `20261025_reliable_pipeline/final` | 图像身份及力学修正候选的检查 | 已完成算法内部检查，不等于物理精度通过 |
| `20261025_surface_identity/evaluation` | 新初始身份/关键帧链候选 | 本轮读取时正式汇总尚缺，不提升为已完成验证 |

### 3.1 前 1 mm 固定 ID 包

每组读取 `metadata.json`、`frames.csv`、`material_displacements.npz`、`contact_zero_audit.json`；NPZ 使用 `allow_pickle=False`。

| 试次 | 帧数 × ID 数 | 实际终点/mm | 终点端点支持数 | 全段测量支持数 | 条件比例尺/μm·px⁻¹ |
|---|---:|---:|---:|---:|---:|
| 1014n33a1 | 406 × 1692 | 0.998178 | 1684 | 1667 | 6.529965 |
| 1014n57a1 | 401 × 1626 | 0.997207 | 1596 | 1567 | 6.731305 |

上表为包内布尔标记计数，尚未叠加所有身份异常及 FE 域筛选，不是独立认证的正确点数。`measurement_motion_step_valid[0]` 可能描述片段前一源帧到接触帧的增量；当前片段速度从第 1 行开始计算。

| 量 | 实际 NPZ 字段 | 处理要求 |
|---|---|---|
| 帧/时钟 | `source_frames,avi_source_frames,clip_frames,capture_time_from_contact_s` | MP4 与 AVI 索引不同，保留映射；200 fps 依据保留 |
| 材料身份 | `point_ids` | 不与旧 CSV 的分段 ID 或新网络查询 ID 混用 |
| 原位置/参考位置 | `positions_source_px,contact_positions_source_px,contact_image_xright_yup_mm` | `seed_points_source_px` 不是接触参考位置 |
| 位移/深度 | `displacement_from_contact_xright_yup_mm,indentation_estimate_mm` | 条件图像轴毫米；未知实验室注册不伪造 |
| 支持 | `measurement_supported,displacement_endpoint_supported,all_intermediate_measurement_supported` | 端点与连续问题采用不同掩膜 |
| 异常/质量 | `tracker_inferred,tracker_ambiguous,tracker_blade_occluded,tracker_cycle_error,tracker_reference_disagreement` | 预测、歧义、遮挡单列；质量分数不当概率 |
| 刀具 | `blade_tip_source_px,blade_reference_state` | 区分观测、刚体推算、预测及缺失 |

### 3.2 同一视频的零点也不能混用

直接按源 MP4 帧身份核对旧 `comparison.csv` 与新 `frames.csv` 得到：

| 试次 | 旧接触地址深度 0 的源帧 | 新位移包深度 0 的源帧 | 旧零点帧在新包中的深度/mm |
|---|---:|---:|---:|
| 1014n33a1 | 234 | 164 | 0.174304 |
| 1014n57a1 | 437 | 92 | 0.863387 |

新包还根据新零点到反转的图像距离重新计算了尺度，不能只对旧 `depth_mm` 加常数；更不能将旧接触地址与新位移在同名深度直接拼接。应通过同一视频源帧建立关系，保留各自原轴，再由明确的共同零点/尺度定义生成派生轴。**重新标轴不自动修复旧接触算法以旧参考帧建立的材料地址历史**；若需原始首触地址，应重新审核参考表面身份，必要时由视频侧生成相容版本。

### 3.3 不宜当作独立真值的输出

`20261025_reliable_pipeline/final/validation.json` 的选中 240 个边界点次中，旧支持为 232；双参考原身份一致仅 15–20，新增恢复为 0，且显式写明 `physical_accuracy_not_established=true`。这是该困难边界样本和检查器的结果，不等于整段点阵只有这些点正确，也不能由高支持率推断身份无误。

`20261016_five_group_sliding/summary.json` 声明 `selection_uses_requested_value_range=true`。部分尾段属于条件外推；该展示集合不能作为无偏材料拟合或留出验证集合。MMT/OBT 的 `sigma_um` 是模型敏感性，非校准统计置信区间。新身份链结果未完成时继续保留缺失。

## 4. 共同比较轴：身份、参考态、坐标和时钟

### 4.1 源身份与位移参考

交接主键包含 `case_id + source_id + source_frame_id + observation_version`；材料点再加 `point_id + reference_id`。`source_id` 必须区分原 AVI、导入 MP4 和裁剪片段，使用文件指纹及帧映射连接，不以相同文件名推断相同试次。

工作台 `mesh.reference_positions` 是未变形网格；`frames.positions` 是接受态。历史视频对照实际在 **FE 深度 0 的已变形接受态** 中映射点，其预变形不能假定为零。视频按接触帧计位移时，比较量应为同一 FE 点的 `x_FE(t)−x_FE(t_ref)`，不是工作台从未变形参考算的位移模长 `displacement_mm`。

保存 `reference_kind`（undeformed / accepted_fe_state / selected_video_frame）、视频参考帧和 `fe_reference_state_index`。参考帧有预载或已经变形时，共同变形是“相对此帧”的增量构型变化，不把它解释为相对无载材料的总应变。

### 4.2 坐标、方向与单位

共同二维坐标记作 `(y,z)`：y 向右、z 向上，mm；FE 的公共 y 对应历史求解器 x。力沿用 `fy_N/fz_N`，刀具行程 `depth_mm` 向下为正。视频原像素 `p=(u,v)` 为右/下；当前仅作条件图像轴比较时：

```text
x_obs = s Q diag(1,-1) (p - p0) + b        # s: mm/px
u_obs(t) = x_obs(t) - x_obs(t_ref)
u_rel = u_material - u_blade
```

Q、b、比例尺及其依据必须显式记录；未注册时 Q=I 只是图像轴约定，不认证为实验室坐标。需要时变相机校正时，应由独立静止夹具建立每帧变换，并分别变换当前/参考位置再相减。不能从变形橡胶、叠加网格或最佳 FE 贴合估计相机运动。透视校正作用于位置，不把同一个常矩阵直接乘所有位移代替单应变换。

跨角度区域另可输出初始表面坐标 `s=e_s·(X−X_contact), n=e_n·(X−X_contact)`；本模型 `e_s=(cosθ,−sinθ), e_n=(sinθ,cosθ)`。它与图像左右区不同，必须保留区域定义版本；不得为角度对比强行旋转实验表面至名义模型角度。

### 4.3 时间和加载窗口

保留 `time_capture_s`、`time_playback_s` 和各自依据。材料时间响应不能使用播放时钟；采集时钟、相机运动和速度未经核实时，只报告条件深度匹配，不宣称 Maxwell 时间历史已得到验证。校核可比较刀尖相对夹具速度与设定 0.2 mm/s，但不能用该设定重新拟合比例尺后再声称速度验证通过。

按采集顺序保留加载/保持/卸载。局部非单调图像深度不得排序成单调材料轨迹。FE 接受态间的位置插值仅用于观测采样，记录左右状态和权重；不是新增平衡解，不能据此检查新时刻能量闭合。不得跨 FE 不合格历史、缺失视频段或阶段转换插值，不外推到 1.000 mm 替代实际约 0.997 mm 终点。

## 5. 共同空间尺度和观测算子

### 5.1 固定材料点采样 FE

先将视频参考点注册到明确的 FE 参考接受态，在该态三角形中定位一次，保存三角形编号与重心权重 `b1,b2,b3`。后续所有时刻固定这些权重：

```text
x_FE,i(t) = sum_a b_ia x_node,a(t)
u_FE,i(t) = sum_a b_ia [x_node,a(t)-x_node,a(t_ref)]
```

域外点标为缺失，不取最近节点。映射到接受态后，仍可用同权重映回未变形网格记录真正材料坐标；两种参考字段不混写。多个材料候选使用共同视频 ID、同一注册规则及有效域交集，不能由各自预测误差选择点。表面接触地址须另外映射到初始表面曲线，不能把体内插值点当表面点。

### 5.2 支持与固定区域

20261024 包令 `bad=inferred OR ambiguous OR blade_occluded`：

```text
endpoint_valid[t] = displacement_endpoint_supported[t] AND NOT bad[t] AND NOT bad[0]
continuous_valid[t] = all_intermediate_measurement_supported[t]
                      AND NOT any(bad[0:t+1])
```

另需有限坐标、可追溯参考身份及 FE 有效映射。连续趋势固定所选窗口所有时刻合格 ID；端点结果作为另一分支，不能把连续筛选与端点筛选混在同一曲线上。连续筛选偏向容易跟踪的点，须报告原点数、留存率、位置覆盖和被排除区域；仍不能认证近刀最困难区。

既有区域按参考刀尖中心距离固定：中心 `r≤0.25 mm`、左右环带 `0.25<r≤1 mm`（参考横坐标分侧）、远区 `1<r≤2 mm`、外区 `r>2 mm`。点等权为主，固定 0.25 mm 空间格等权作密度敏感性；保留每个区域的成员和权重。区域不随当前刀尖/变形换点；“当前近刀窗口”可另作欧拉描述，不能冒充同一批材料的演化。

先比较分区位移分量、刀具相对位移、有限窗口增量及以下质心距离变化：

```text
DeltaL_AB(t) = norm(mean_w(x_A(t))-mean_w(x_B(t)))
               - norm(mean_w(X_A)-mean_w(X_B))
```

该距离对共同二维平移/旋转不变，可用于相机运动未明时的诊断；仍受比例尺和投影影响，不是裂缝张开、界面滑移或局部应变。

### 5.3 同标距有限变形

对视频与 FE 采样点执行完全相同的加权拟合 `min Σ w_i ||x_i−F2 X_i−c||²`。已有中心间距 0.4 mm、半径 0.25/0.4/0.6 mm 三档；这是**分析标距，不是已验证的图像空间分辨率**。点间距、跟踪模板、平滑和拟合半径共同影响实际分辨能力。

沿用几何支撑检查：点数≥12、参考协方差条件数≤25、小主轴 RMS/半径≥0.1、质心偏心/半径≤0.65、最大角缺口≤210°。这些是既有算子的可计算门槛，不是实验精度门槛；不按拟合误差筛选出“看起来匹配”的区域。

输出 F2 奇异值 `lambda_max/min`、对数主伸长、`E2=(F2^T F2−I)/2` 的分量/主值、极分解转角、`detF2` 和仿射残差 RMS。有效参考身份、正向可辨识拟合不足时保留原因及缺失，不截断异常值使其物理化。

视频 `detF2` 是可见侧面投影的有限标距面积比，不等同三维 J；FE 原生 `green_max/min` 含平面应变第三主值 0，也不是视频二维主值。应先共同采样/拟合，再比较二维量。仿射残差同时含真实非仿射运动和跟踪误差，不作为已校准噪声方差。

`polar_rotation_deg` 仍包含未去除的相机转角。区域距离、主伸长等相应标量的共同二维刚体运动不变性，不适用于全部位移或 F2 分量；平面外转动也不在该不变性范围内。

### 5.4 接触地址、相对位移与滑移分开

| 量 | 定义 | 比较条件 |
|---|---|---|
| 原表面地址偏移 `delta_s0_mm` | 接触点在初始表面 Γ0 上距初始接触地址的有符号弧长 | 同一 Γ0、参考帧、方向、原表面身份；圆鼻区间/双侧歧义保留 |
| 刀具相对位移 `u_rel_y/z_mm` | 同一材料点与刀具参考点的位移差 | 可比较近刀运动，不要求该点始终接触，也不等于滑移 |
| 接触切向相对速度 | 已确认接触处 `(v_material−v_blade)·t_contact` | 同步、局部切线、接触证据和足够时空分辨率 |
| 累计滑程 | 接触有效区间内对上述速度绝对值积分 | 不能跨遮挡/丢失接触积分；有符号积分另列 |

MMT/OBT `offset_um` 先映为“条件原表面地址偏移”，`sigma_um` 映为“模型敏感性”；保留 `method_identifier`、`transport_state`、`prediction_only`、`holding`、`observation_age` 和原始有效性。报告层 `completed_offset_um` 及插值/外推不得覆盖 `original_offset_um`。两方法分别展示，不求平均当真值。

FE 的唯一接触地址、候选区间、圆鼻支撑区间和整刃足迹遵循[接触规范](tip_contact.md)。图像几何接近阈值与 IPC 激活距离不能直接等同。若形成新裂面，原 Γ0 地址可能不可定义；完整体模型的后续地址仅是无断裂条件预测。

## 6. 拟议的共同数据包

提出轻量比较包 `softcut-comparison/1`，**这是下一步实现规范，当前没有宣称已导出该包**。保留视频的原 NPZ/CSV 和 FE `softcut-result/1`；适配器只读来源，在仿真项目 `outputs/<比较主题>_<日期>/` 写派生结果。无需视频 agent 重构已有格式。

CSV 为 UTF-8、英文列名、逗号分隔，缺失为空、JSON 为 null；布尔 0/1。非有限数不得以零、−999 或成功状态替代。原像素、条件毫米、实验室注册毫米分开，不能仅改列名。

| 文件 | 最低内容和主键 |
|---|---|
| `manifest.json` | schema、case/trial、来源绝对路径/相对路径/SHA256/读取时间、生产算法版本和当前核查版本、工况、单位/坐标/变换、参考态、尺度/时钟/首触资格、完整性阶段、FE 连续合格前缀、观测算子及质量规则 |
| `frames.csv` | `(source_id,frame_id)`；AVI/MP4/片段映射、采集/播放时间、阶段、原始条件深度/共同深度及来源、刀尖/姿态/证据状态；FE 左右采样态/权重另标 |
| `points.csv` | `(observation_version,reference_id,point_id)`；种子身份、参考源帧、参考 px/mm、点类型、身份链来源；不可把侧面内部点标成原上表面 |
| `tracks.csv` | `(observation_version,source_id,frame_id,reference_id,point_id)`；当前 px/mm、位移分量、raw/measurement 支持、endpoint/continuous 资格、遮挡/推断/歧义/异常、图像质量量 |
| `mapping.csv` | `(fe_run_id,registration_id,reference_id,point_id)`；参考 FE 状态、triangle_id、b1/b2/b3、域内标记、重建误差及缺失原因 |
| `cohorts.csv` | `(comparison_id,cohort_id,reference_id,point_id)`；region/patch、参考中心/半径、固定权重、窗口、几何资格 |
| `observables.csv` | `(comparison_id,fe_run_id,source_id,frame_id,cohort_id,operator_id,component)`；obs/prediction/error、单位、有效性/原因、支持点数/覆盖、左右插值态；窗口量增加 start/end_frame |
| `validation.json` | 主键、有限值、帧映射、参考态、坐标、重心重建、同成员/权重、FE 资格、缺失/插值规则及文件读取一致性检查 |
| `contacts.csv`（有合格地址才提供） | 源帧、方法、原表面版本、side、原地址/偏移/上下界、参考身份、观测/推断/保持/展示补点状态及敏感性 |

每行所属来源由 manifest 中 `source_id` 解析；一个 comparison 固定一个实验试次、观测版本及参考定义，避免不同试次 point_id 撞号。数值较大时 `tracks` 可使用等价、禁止 pickle 的 NPZ 长表；字段、主键及缺失语义不变，不同时保存两份全量重复表。

`observation_version` 必须唯一绑定生产算法、输入和编号体系；`points.reference_id` 可追溯到完整的源帧键。固定全窗口连续点集是回顾性筛选，报告 `cohort_uses_future_availability=true`；它不同于跟踪器是否使用未来图像，不据此宣称在线预测能力。

观测来源标签至少区分 `image_supported`、`image_candidate_verified`、`kinematic_regularized`、`mechanics_regularized`、`prediction_only`、`display_interpolated`。网络可见性或亮度重拟合通过不自动升级为原材料身份已认证。机制/材料拟合的主评分只用预先选定的图像支持分支，其他来源单列敏感性；不混入同一 RMSE。

manifest 还应包含：`future_frames_used`、实际 `cutoff_source_frame`、掩膜版本/腐蚀距离、模板/平滑尺度、接触估计器版本、`force_trial_id` 与同步资格。未知值为 null，不把“未记录”改写为“未使用”。`uncertainty_kind` 区分校准测量不确定度、模型敏感性和人为扰动；无测量误差预算时不构造统计置信区间。

## 7. 分析顺序、评分和可解释结论

1. **先审计源而非先画叠图。** 验证试次/版本、参考身份、原始帧映射、比例尺和零点。两次读取发生变化的文件暂不纳入固定证据；路径当前文件的哈希不能冒充旧结果生产时的源码哈希。
2. **先位移再梯度。** 在实际共享窗口比较分区向量 RMSE、分量偏差、中位/P90 向量误差、区域间 ΔL，并同时给支持率和覆盖。保留像素量及条件毫米量，避免标定不足被隐藏。
3. **再比较同标距变形及轮廓。** 使用三档半径检查趋势是否稳定；轮廓给可见段的有符号法向距离，与材料身份误差分开。边界位置正确不证明沿边界跟踪了同一块材料。
4. **接触地址另开评分。** 只有原表面身份和共同参考明确时才比较偏移/区间；预测、保持、补点、外推与观测状态分图显示。
5. **力—运动联合解释。** 同材料候选用固定共同点集分别报告 Fz 幅值/分段斜率与运动误差，不以任意总分掩盖某一区域恶化。33°视频 1014n33a1 与力目标 1008n33a1 是不同试次；57°同名试次亦未认证同步，暂不做逐帧能量闭合。

公共检查点可取 0/0.25/0.5/0.75 mm 和实际共同终点，记录实际深度而非把名义 1 mm 写成完成值。FE 默认展示包约 11 帧不足以替代完整接受态历史；定量比较读取原 NPZ 或显式导出所需状态。插值误差与图像误差分别列出，不通过增加展示帧率制造精度。

| 组合现象 | 值得检查的方向 | 不能直接推出的结论 |
|---|---|---|
| 力变化明显，区域运动偏差在多个共同材料候选间几乎不变 | 接触运动模式、参考/投影、二维约束及跨变形模式响应 | 已证明材料参数无效或二维模型不可能 |
| 绝对位移差大，ΔL/同标距伸长接近 | 公共刚体运动、刀路/相机注册及参考态 | 材料与摩擦已经正确 |
| 仅边界/近刀误差大且支持率下降 | 掩膜腐蚀、原始身份、遮挡、轮廓及接触条件 | 已观测裂纹或粘滑切换 |
| 力接近但两侧拖曳/变形模式不同 | 材料—接触的参数补偿，额外力学约束 | 单独归因于 μ、K 或松弛时间 |
| 力学正则后的轨迹明显更贴合 FE | 先验及权重敏感性 | 独立实验验证成功 |

视频不能直接测量应力、摩擦功、黏性耗散或储能。材料随刀运动可与黏着接触运动学相容，但不证明黏附；有限图像分辨率下近零相对速度也不能独立给出库仑 stick/slip 状态。只有经身份、接触和时钟核查的相对速度才可进一步检验 FE 的局部状态。

## 8. 提高精度所需的最小验证与实施顺序

**可立即复用**：33°/57°固定 ID 包、既有 FE 接受态及[位移对照](../outputs/video_fe_1mm_20261024/index.html)、[区域运动](../outputs/video_fe_1mm_20261024/kinematic_check/index.html)、[有限标距变形](../outputs/video_fe_1mm_20261024/deformation_review/index.html)。既有代码位于 outputs 的研究后处理，尚不是主线接口。已有结果只证明算子可以执行，不证明视频精度或模型匹配。

**优先实施**：在仿真侧建立只读适配器，先产出 frames/points/mapping/cohorts 和位移/ΔL；用现有局部仿射算子生成同标距量。两组新旧零点先作为显式不兼容项，不混合接触结果。暂不需要重跑 FE、改变本构或让视频端改写已有结果。

**图像算法独立检验**：由视频侧使用自身固定版本和相同参数，处理原始纹理构造的已知位移图像；输入至少包括刚体平移/旋转、均匀有限变形、非 FE 生成的光滑非均匀场、FE 场，以及亮度变化、遮挡和掩膜边界扰动。比较位移、同标距伸长、身份误认和丢点，分别记录偏差、噪声与空间分辨能力。只使用 FE 生成场会遗漏非 FE 运动模式；平面合成图像通过也不能排除真实面外运动。

上述已知场检验与同标距要求参考 DIC Challenge 2.0 对空间分辨率、位移/应变噪声权衡的分析；本轮核读其机构摘要，不宣称本项目已满足其全部验证条件。[NIST 原始记录](https://www.nist.gov/publications/dic-challenge-20-developing-images-and-guidelines-evaluating-accuracy-and-resolution-2d)

**物理解释升级条件**：补齐静止夹具/尺度依据、接触零点区间、同试次力—视频同步以及侧面投影与内部平面应变截面的可比性。外侧表面存在面外膨胀时，先把二维视频作为条件投影约束；三维或广义平面应变是待检验模型选项，不由单个投影面积比直接选定。

## 9. 双方后续交付与验收

视频侧无需改变当前内部格式；交付不可变的结果版本和来源清单，提供源帧映射、原像素轨迹、参考身份、独立支持/预测/歧义/遮挡、标定/时钟依据、方法/先验谱系与缺失原因。新身份链需记录根参考、实际模板来源与链断裂；共享祖先的检查不当独立概率相乘。

仿真侧提供 `softcut-result/1` 或合格原始接受态、参数签名、审查前缀、参考网格与接受态位置；负责固定映射及共同算子，不要求实验按 FE 答案更改轨迹。报告包含覆盖/缺失图、分区域位移和 ΔL、同标距变形、可选接触地址，并链接源指纹及 CSV/JSON。

接口验收至少检查：主键唯一、源帧/参考可追溯、缺失不补零、有效坐标有限、零位移恒等式、重心和为 1 及参考点重建、域外缺失、双方同成员/权重、FE 连续合格历史、无跨缺失/阶段外推、原始值/派生值分离。共同算子的刚体不变性及已知仿射恢复是实现检查，不替代光流精度检查。

当前没有已校准的统一实验误差阈值。未知噪声、空间偏差和跨试次差异明确留空，不以密集相关点数或重叠标距构造虚假置信区间。算例预算、完整体限制及共享材料约束仍由[任务要求](task_requirements.md)管理。

## 10. 可追溯来源

外部代码及文档均相对视频仓库；完整读取文件清单、时间与 SHA256 见本轮 `source_audit.json`。

| 来源 | 核读内容 |
|---|---|
| `README.md`、`docs/code_structure.md` | 生产入口、工作流和产物职责 |
| `src/tracking/material_patch_tracker.py:255`、`:648`、`:687` | 局部仿射、多参考、反向检查及质量语义 |
| `src/tracking/shadow_normalization.py:24` | 材料内低频亮度修正 |
| `src/workflows/tracking.py:175`、`:219`；`measurement.py:159`、`:225` | 固定轨迹导出、测量支持及标定限制 |
| `src/workflows/tracking_inputs.py:265`、`:283`、`:458`；`src/workflows/video_import.py:138`；`src/video_sources.py:31` | 零点/比例尺/帧映射 |
| `src/analysis/dual_contact.py:94`；`objective_boundary_velocity.py:13`；`contact_transport_filter.py:29`；`src/workflows/sliding.py:177` | 原表面输运、状态、模型敏感性和展示补点 |
| `docs/reliable_pipeline_methods.md`、`docs/material_tracking_optimization_plan.md`；`src/tracking/finite_deformation.py:372` | 候选分支的平面应变/Neo-Hookean 假设及完成范围 |
| `docs/surface_identity_optimization.md`；`src/tracking/identity_chain.py` | 新根参考身份链，读取时正式效果尚待汇总 |
| `results/20261024_n_group_first_1mm/*/metadata.json`、`frames.csv`、`material_displacements.npz` | 两试次实际字段、计数、坐标、零点和时钟 |
| `results/20261025_reliable_pipeline/final/validation.json` | 已完成候选的内部检查及准确性限制 |
| `results/20261016_five_group_sliding/summary.json` | 展示集的选择依据，不充当无偏验证样本 |

文献统一保存在[引用导读](../references/README.md)。iDICs 2025 第二版指南的官方页面说明其覆盖测量设置、图像相关及应变后处理，并限定常见实验条件；本轮未重新阅读全文，不把指南引用当成本项目尖刀遮挡条件的认证。[官方页面](https://idics.org/guide/)
