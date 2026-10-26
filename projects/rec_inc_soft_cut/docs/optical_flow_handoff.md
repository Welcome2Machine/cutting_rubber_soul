# 侧视光流数据交接规范：橡胶斜切的位移、接触迁移与力学对照

版本：1.0；日期：2026-10-12。接收方：本仓库仿真 agent；提供方：侧视视频/光流 agent。

**请先判断现有数据能提供哪些项目，再按最小包导出。无法测量的项目标记缺失，不需要为了满足本规范重新做整套光流，也不需要从视频猜出接触力、应力、摩擦系数或裂纹。** 本文是数据接口要求，不是已经完成的实验测量报告。

## 1. 目的与任务边界

我们需要将同一实验的材料运动、刀具运动和可用的 Fz 数据对齐，以检查：

1. 仿真是否正确描述近刀与较远区域的水平/竖直位移、表面沉降和隆起。
2. 实验与仿真的早期 Fz 曲线形状不同，是否伴随接触范围、局部拖拽或整体约束响应的不同。
3. 条件允许时，估计刀具最低点附近对应的原始上表面材料坐标，检查参考接触迁移；不把近刀散斑点自动视为接触点。

首轮优先比较**首触前至下压 1 mm**，但只采用有证据支持的未起裂段；不能预先认定所有试次在 1 mm 内均未断裂。已有更长加载段可一并交付，保留阶段标签。回撤不是本轮重点。

当前有限元是无断裂完整体模型。数据接入不自动授权新仿真、参数标定或裂纹预测；未指定用于标定的试次继续保留为独立验证。光流可以检验表面运动，但不能单独测得应力、接触压力或材料模量。

## 2. 仿真端背景：只用于定义接口

| 项目 | 当前设置或定义 |
|---|---|
| 试样 | 局部长度 30 mm、高度 5 mm、出平面宽度 20 mm；底面固定；二维平面应变 |
| 当前比较工况 | 60° 倾斜试样，刀具沿实验室竖直方向下移 |
| 刀具 | 99755，厚度 0.245 mm、双面全刃角 23°；圆鼻半径 0.02 mm 为用户指定值，非本次视频实测 |
| 材料/界面 | G=0.47 MPa、ν=0.49、μ=0.35；μ 未经独立实验标定 |
| 参考表面坐标 | 局部 s 沿初始上表面增加；名义初始接触材料坐标 s₀=7 mm |
| 深度 d | 刀具相对初始位置的实验室竖直下移，以理想圆鼻与未变形平面的名义首触为零 |
| 接触迁移 Δs₀ | 当前对应材料点在**原始上表面**上的坐标 s 减去 s₀；不是累计摩擦滑程 |

参数依据见 [模型规范](current_model_spec.md)，观测定义见 [接触坐标说明](tip_contact.md)，实际计算完成与限制见 [求解状态](solver_status.md)。**不要把这些模型参数、未变形形状或运动趋势当作光流跟踪的真值约束。** 实验倾角、相机方向、润滑状态、实际刀具姿态等应报告其自身来源；文件名中的“60”不能代替坐标标定。

仿真端会负责从已有保存态提取位移、注册初始坐标和进行对照。光流端不需要生成有限元网格，也不需要输出模拟预测值。

## 3. 分级交付与优先级

| 级别 | 内容 | 能做什么 |
|---|---|---|
| P0：最小交接包 | `metadata.json`、`frames.csv`、`tracks.csv`、`quality_report.md` | 核查材料身份、时间和像素运动；标定齐全后可做位移–深度对照 |
| P1：优先补充 | 像素到实验室坐标标定、刀具轨迹、原表面/当前轮廓、独立力数据及同步信息 | 定量比较整体变形、Fz 与变形的对应关系 |
| P2：条件具备才做 | 接触候选及材料坐标区间、界面相对切向运动、局部有限应变 | 检查接触迁移与局部变形；必须附可见性和不确定性 |

P0 中字段存在但暂不可得时可为空并说明原因；缺少比例尺、深度零点或可信材料对应时，包仍可供核查，但不能声称已经满足毫米级定量比较。**P2 无法提供不阻塞 P0/P1。**

建议一个试次一个目录，存于本地 `outputs/optical_flow_exchange/<case_id>/`，或提供方自己的本地目录。实际数据、视频、图像和观测报告不得提交到公共 Git；本规范本身可以作为文档管理。

```text
<case_id>/
  metadata.json
  frames.csv
  tracks.csv
  quality_report.md
  contours.csv             # 可选：可见轮廓，不能假连遮挡段
  force.csv                # 可选：独立力传感器数据，保持自身采样率
  events.csv               # 可选：首触/起裂迹象/加载结束等区间标注
  contact_candidates.csv   # 可选：有证据的候选与材料对应
  transforms.csv          # 若处理变换逐帧变化，则需要
  masks/                   # 已有遮挡/材料/刀具掩膜时提供
  preview.mp4             # 可选：核查叠加图，不代替数值表
```

CSV 采用 UTF-8、逗号分隔、第一行为英文列名；小数点为 `.`，单位写在列名中。缺失数值为空，JSON 用 `null`，不写 0、-999 或字符串 `NaN`。布尔列为 0/1。数值有效时必须有限。下面 CSV 代码块**只有表头，不是实验数据**。额外列允许保留，但须解释。

## 4. 坐标、时间、深度与材料身份的统一约定

### 4.1 图像坐标与实验室坐标

- 原始图像：`x_px` 向右、`y_px` 向下；原视频左上角像素中心为 `(0,0)`。这里的“原始”指选定源视频的解码帧，不能把已叠加箭头的展示视频当作无处理源。
- 必须记录裁剪、缩放、旋转、镜像、畸变校正、稳像及其顺序。导出的 `x_px/y_px` 统一回到源视频像素坐标；若目前做不到，先报告现有坐标约定，不能直接改列名冒充原坐标。
- 可提供的实验室坐标：`x_lab_mm` 向右、`y_lab_mm` 向上；在相机倾斜或画面镜像时，需通过注册建立真正的实验室方向，不能只翻转图像 y。
- 同一试次使用固定的物理参考系。相机漂移只能用固定夹具/背景等独立参考估计；不能把橡胶的真实整体运动作为相机漂移扣除。
- 给出像素到毫米变换的适用平面、标尺来源和误差估计。不能只凭刀片型号，把未知姿态下的投影宽度当作 0.245 mm 标尺。
- 如使用二维单应矩阵，声明输入是在畸变校正之前还是之后，并给出矩阵方向、齐次除法和逐帧稳像变换；二维映射不能自动校正材料的出平面运动。

模型的实验室坐标 y 向上。对于模型中顺时针倾斜 θ 的试样：

```text
e_s = (cos θ, -sin θ)     # 初始表面的正切方向
e_n = (sin θ,  cos θ)     # 初始表面外法向
X_FE(s,n) = s e_s + n e_n
初始名义接触点 X_FE,c0 = 7 e_s + 5 e_n
```

实验图像到模型的平移/旋转注册由仿真端在核对后完成。提供方保留实测方向，不强行把每帧橡胶“转回”初始形状。可见侧面是三维试样的自由表面，而模型是平面应变截面，两者等价性需要检验。

### 4.2 时间与深度

- `frame_id` 为源视频从 0 开始的解码帧编号，抽帧后仍保留源编号。`time_s` 优先使用原始帧时间戳减去源视频第一帧时间戳；固定帧率推算时说明依据。不得混用原始采集时间与慢放/剪辑后的播放时间。
- 有变帧率、掉帧、时间跳跃、倍速或剪接时，保留实际时间映射；不能一律用导出视频的 fps 推算物理速度。
- 位移参考帧 `reference_frame_id` 优先选清晰、首触前、材料静止的帧。它与“首次接触帧”通常不同。
- `depth_mm` 是**刀具相对固定基底的竖直下移量**，相对于估计首触位置；不是橡胶表面的沉降、刀具到当前表面的距离，也不是沿斜面法向的投影。
- 若以同一个刀具刚体标志的位置计算，实验室 y 向上时：`d(t)=y_blade(t_contact)-y_blade(t)`。若基底移动，先转为基底参考系。刀具存在旋转时须跟踪刚体姿态，不能把不同刀具点混作同一标志。
- 最低刀极点可能隐藏，可跟踪刀身上稳定可见的标志推算刚体平移；记录推算方式。几何推算的极点不能标成直接观察。
- 首触可只给帧区间 `[last_separated_frame, first_contact_evidence_frame]` 和判断依据，不要求伪精确单帧。无法确定时，保留相对于参考帧的刀具位移，`depth_mm` 留空。
- 首次圆鼻侧触与最低刀极点接触不同；数值屏障开始作用也不是实验首触。不得为贴合 Fz 曲线任意平移深度零点。

### 4.3 材料点身份与光流累积

- 同一个 `point_id` 必须代表同一材料点，而不是每帧重新选择的同一像素或“距刀尖最近的点”。分组只是标签，不能代替点身份。
- 密集帧间光流 `v_k(p)` 需沿材料轨迹复合：`p_(k+1)=p_k+v_k(p_k)`，记录插值方法；不能在固定图像像素上直接把各帧光流相加。
- 记录是参考帧到当前帧的直接匹配，还是逐帧累积、关键帧重定位。参考坐标始终保留，不能随重定位静默重置。
- 遮挡后的外推/插值与实际再观测必须分开。重新识别只有在有材料身份匹配证据时才延续原 ID；否则新建 ID，记录新的参考帧。
- 变形后才出生的新点可分析后续增量，但没有回溯证据时，不具有首触前参考坐标，不能直接用于 Δs₀ 或未变形构型对照。

## 5. 最小交接包的字段

### 5.1 `metadata.json`：试次、处理方法与缺失信息

请记录以下内容，未知就明确写未知，不用模型默认值补齐实验参数：

| 类别 | 必需声明的信息 |
|---|---|
| 来源 | 唯一 `case_id`、源视频文件名及 SHA-256、源尺寸、帧数、时间戳/fps 来源；视频已加工则写加工链 |
| 工况 | 材料/批次已知情况、试样方向、倾角及其定义、刀具编号/姿态、干态或润滑、温度、速度、固定方式；每项来源或未知 |
| 拍摄 | 看哪一侧、是否镜像、相机是否固定、是否可能出平面、已知曝光/镜头/畸变信息 |
| 坐标标定 | 像素原点/轴方向、处理变换、像素到 mm 的方法和适用平面、固定参考物、误差或未评估状态 |
| 参考与首触 | 全局参考帧、是否未变形、首触帧区间/依据、初始表面及初始接触位置；未知允许留空 |
| 算法 | 光流/跟踪算法、版本或代码指纹、主要参数、窗口大小/间距、光照处理、遮挡处理、重识别/平滑策略 |
| 质量语义 | `confidence` 的定义/范围/高低方向、有效性阈值及依据、误差评估方法；分数不是已校准概率 |
| 力同步 | 是否存在同一试次的力数据、传感器采样率、同步方式/误差、深度来源；没有就写没有 |
| 使用范围 | `validation_only` / `calibration_allowed` / `undesignated`；只有用户明确指定才填 `calibration_allowed` |

可复制的骨架如下。它是接口占位模板，`null` 不是测量结果；`PLACEHOLDER_CASE` 需替换：

```json
{
  "schema_version": "1.0",
  "case_id": "PLACEHOLDER_CASE",
  "data_role": "undesignated",
  "source": {
    "video_name": null, "sha256": null,
    "width_px": null, "height_px": null, "frame_count": null,
    "fps_nominal": null, "time_source": null,
    "capture_to_playback_time_mapping": null
  },
  "experiment": {
    "specimen_dimensions_mm": null, "inclination_deg": null,
    "angle_definition": null, "blade_id": null,
    "lubrication": null, "temperature_C": null,
    "loading_speed_mm_s": null, "fixation": null,
    "sources_and_unknowns": []
  },
  "coordinates": {
    "pixel_convention": "source_pixel_centers_x_right_y_down_origin_0_0",
    "lab_convention": "x_right_y_up_mm_fixed_to_base",
    "observed_face": null, "mirrored": null,
    "processing_transform_chain": [],
    "calibration_method": null, "calibration_source": null,
    "pixel_to_lab_mapping": null,
    "camera_motion_correction": null,
    "uncertainty_definition": null
  },
  "reference": {
    "frame_id": null, "state": "unknown",
    "last_separated_frame": null,
    "first_contact_evidence_frame": null,
    "contact_zero_method": null,
    "initial_contact_source_xy_px": null,
    "initial_surface_direction_lab": null
  },
  "tracking": {
    "method": null, "version_or_commit": null,
    "parameters": {}, "trajectory_composition": null,
    "confidence_definition": null, "validity_rules": null,
    "uncertainty_definition": null, "postprocessing": []
  },
  "force_sync": {
    "available": null, "mapping": null, "method": null,
    "time_uncertainty_s": null
  },
  "files": {
    "frames": "frames.csv", "tracks": "tracks.csv",
    "quality_report": "quality_report.md"
  },
  "unavailable_items": []
}
```

### 5.2 `frames.csv`：每帧时间、刀具位置与深度

```csv
case_id,frame_id,time_s,stage,depth_mm,depth_source,depth_uncertainty_mm,blade_tip_x_px,blade_tip_y_px,blade_tip_status,blade_angle_deg,valid_frame,invalid_reason
```

| 字段 | 约定 |
|---|---|
| `stage` | `precontact/loading/hold/unloading/unknown`；下压和回撤不能只凭相同深度混在一起 |
| `depth_source` | `machine/video/derived/unknown`；处理方法及零点在 metadata 中说明 |
| `depth_uncertainty_mm` | 深度不确定性估计，须声明是标准差、区间半宽或其它量；未评估为空 |
| `blade_tip_x_px/y_px` | 固定身份的刀具最低极点在源图像中的位置；未知为空 |
| `blade_tip_status` | `observed/rigid_inferred/predicted/occluded/unavailable`；`rigid_inferred` 需已知几何和可见刚体标志支撑 |
| `blade_angle_deg` | 刀具轴从实验室 +y 朝 +x 旋转为正；不能可靠估计则为空 |
| `valid_frame` | 此帧整体是否可用；为 1 不表示其中所有点、刀尖或深度都有效 |

只有刀身标志轨迹时，增加 `blade_marker_id,blade_marker_x_px,blade_marker_y_px`；基底可能移动则同样增加固定参考标志轨迹或提供变换表。没有可信极点位置仍可做刀具位移对齐，但不能声称有直接极点接触观测。

`frame_id` 唯一，`time_s` 严格递增。保留实际帧/时间采样，不为凑齐有限元的 0.01 mm 输出网格制造实验帧。

### 5.3 `tracks.csv`：材料点长表

每行对应一个试次、一个帧、一个材料点。基本表头：

```csv
case_id,frame_id,point_id,reference_frame_id,reference_kind,x_ref_px,y_ref_px,x_px,y_px,point_kind,track_status,valid,confidence,fb_error_px,position_uncertainty_px,invalid_reason
```

| 字段 | 约定 |
|---|---|
| `point_id` | 字符串，同一试次内永久唯一，不复用已丢失点的 ID |
| `reference_kind` | `precontact/deformed/unknown`；`precontact` 需参考状态有证据，不能只因它是第一帧 |
| `x_ref_px/y_ref_px` | 该点在其参考帧的源像素坐标；同一 ID 不随帧改写 |
| `x_px/y_px` | 当前图像有匹配支持的源像素坐标；无观测为空。外推/插值只放可选 `x_pred_px/y_pred_px` 列，状态为 `predicted` 且 `valid=0` |
| `point_kind` | `rubber_face/original_top_surface/blade/fixed_background/unknown`；侧面内部散斑不能自动标成原上表面点 |
| `track_status` | `tracked/reidentified/predicted/occluded/out_of_frame/rejected/missing`；`tracked` 指当前图像有匹配支持 |
| `valid` | 当前点通过已声明的图像跟踪质量筛选为 1，否则 0；它不代表接触已确认或平面应变已成立 |
| `confidence` | 可选算法质量值，必须解释含义；没有就留空，不能统一填 1 |
| `fb_error_px` | 可选前后向一致性误差，需说明使用哪两帧及所在像素尺度；不等同真实位置误差 |
| `position_uncertainty_px` | 可选位置不确定性估计及其定义；未评估为空，不把质量分数直接当毫米误差 |
| `invalid_reason` | 如 `blade_occlusion/shadow/low_texture/identity_uncertain/motion_blur`；可用分号列多个原因 |

若已有毫米标定，推荐增加：

```csv
x_ref_lab_mm,y_ref_lab_mm,x_lab_mm,y_lab_mm,ux_mm,uy_mm,roi_ref
```

必须满足 `ux_mm=x_lab_mm-x_ref_lab_mm`、`uy_mm=y_lab_mm-y_ref_lab_mm`，两者处于同一基底固定参考系。`roi_ref` 按**初始材料位置**固定分组；若还需按当前距刀位置分组，另列 `roi_current`，不能混用。

不要求巨大的全帧×全点矩形表，但丢失点应至少在状态变化/追踪结束时记录原因和帧号；稀疏表的缺行只代表未提供，不自动代表静止、接触或沿用上一值。只有重识别证据充分时，`reidentified` 才能 `valid=1`；跨遮挡期间不补观测。

若现有输出只有密集光流，可先交其数组文件和明确映射：形状 `[N,H,W,2]`、每项的起止源帧、分量方向/单位、网格坐标、裁剪/缩放、有效掩膜和算法质量。不要只导出彩色光流图。仿真端可据此讨论转换，不要求供方先重做稀疏跟踪。

### 5.4 `quality_report.md`：必须如实回答的问题

1. 哪些文件/字段已提供，哪些无法提供，分别是什么原因；现有数据能支持 P0/P1/P2 的哪部分。
2. 参考帧、首触区间、比例尺、相机与基底的稳定性是否可信；哪些量只是人工估计。
3. 每帧或每个深度区间的有效点数/比例、丢失与重识别数；近刀与远场分别说明。
4. 静止帧/固定背景的表观位移有多大；若有刚体运动检查，说明方法。未做误差评估写“未评估”，不能写零误差。
5. 阴影归一化、平滑、插值、稳像是否做过；是否改变坐标；保留原始匹配与处理结果的对应关系。
6. 刀尖、左右刀面和原始上表面哪些区间被遮挡；是否存在材料身份混淆或可能的出平面运动。
7. 数据是否只有展示视频、是否还保留逐点数值；哪些数据被抽帧或抽点以及规则。
8. 哪段有明确未起裂证据，哪段有可见开口或身份拓扑变化，哪段无法判断。没有看见裂纹不等于证明无裂纹。

可附几张核查帧/短视频，叠加固定 ID、刀具标志、初始与当前轮廓、无效/预测点。目的在于检查跟踪是否跟随材料，不要求美化。

## 6. 可选数据格式

### 6.1 `contours.csv`：可见轮廓与可追溯的原上表面

```csv
case_id,frame_id,contour_id,component_id,vertex_order,contour_kind,x_px,y_px,visibility,evidence,point_id,reference_s_rel_mm
```

- `contour_kind`：`original_top_surface/visible_free_surface/blade_outline/unknown`。初始上表面、变形后的可见自由边界和新出现的裂面不是同一概念。
- 同一帧、同一 `contour_id/component_id` 内按 `vertex_order` 连接；被刀遮住的部分断开为不同 component，不能跨空白连线。
- `visibility`：`visible/occluded/uncertain`；`evidence`：`segmented/manual/tracked/predicted`。遮挡段可以只有状态，不给伪坐标。
- `point_id` 只有材料对应明确时才填写；逐帧独立分割的轮廓点编号不表示同一材料点。
- `reference_s_rel_mm` 只有能回溯到首触前的原上表面、并确定初始接触原点时才给出；否则为空。不能由当前弧长倒算。
- 最低要求是可见轮廓；没有材料身份的轮廓也能比较形状，但不能直接求 Δs₀。

### 6.2 `force.csv`：有独立力数据时提供

```csv
case_id,sample_id,time_force_s,Fz_raw_N,Fz_N,depth_machine_mm,valid,invalid_reason
```

保留力传感器原采样率。`Fz_N` 定义为抵抗下压的反力幅值，压入为正；若原仪器符号不同，保留 `Fz_raw_N` 并记录换号、去皮/零点及滤波。未处理时两列可相同；不能从光流“估计”后填入传感器力列。

`time_force_s` 是力数据自己的时钟。同步映射必须明确，例如 `time_video_s = a*time_force_s+b`，给出 a、b 的来源、锚点与误差；只有一个锚点而假定 a=1 时明确注明。禁止以拟合仿真曲线为目的调整时间、深度或力零点。

同一试次的测力若没有时间戳但有真实机器深度，可交 `case_id,sample_id,depth_machine_mm,Fz_raw_N,Fz_N,valid` 的替代表，并声明无法逐帧严格同步。不能把另一试次的力曲线配成当前视频的同步力。

### 6.3 逐帧变换与掩膜

若处理链中的稳像/裁剪等变换逐帧变化，`transforms.csv` 至少包含 `frame_id,transform_id,input_space,output_space,valid`，并给矩阵系数。二维齐次矩阵使用按行排列的 `h00,h01,h02,h10,h11,h12,h20,h21,h22`，说明应用顺序及最后的齐次除法；畸变校正等非线性步骤另给参数/实现，不能伪装为单一矩阵。稳定的全局变换可直接放 metadata，不必重复成表。

已有掩膜需声明图像尺寸、所处坐标空间、对应源帧、类别值（材料/刀具/背景/遮挡等）及是否逐帧独立分割。矩阵和掩膜只交实际存在的版本，不要求额外生成。

### 6.4 `events.csv`：事件只给可支持的区间

```csv
case_id,event,frame_start,frame_end,evidence_kind,notes
```

`event` 可为 `first_contact/visible_opening/loading_end/unloading_start/tracking_failure`。帧区间表示事件时间的不确定范围，不自动是事件持续时间；`evidence_kind` 为 `visible/manual/inferred`。真实材料起裂可能早于可见开口，不能将 `visible_opening` 改称精确起裂时刻。

### 6.5 `contact_candidates.csv`：非强制，不能伪造唯一切点

```csv
case_id,frame_id,target,status,candidate_id,point_id_a,point_id_b,alpha,x_px,y_px,gap_px,gap_uncertainty_px,reference_s_rel_mm,reference_s_rel_low_mm,reference_s_rel_high_mm,evidence,reason
```

- `target` 为 `lowest_blade_point/nose/left_flank/right_flank`；必须区别最低刀极点、圆鼻侧部和刀面。不同 target 的结果不得拼成一条“刀尖滑移”曲线。
- `status` 为 `unique_candidate/ambiguous/visibly_separated/occluded/unavailable`。每个导出的 frame/target 至少有一行状态；无候选时候选字段为空，多义时每候选一行，不能挑最顺滑的一条。
- `gap_px` 是可见几何的图像间隙，附分辨率/分割误差。小于像素误差的间隙不能证明零间隙接触，更不能给出压力或黏着状态。
- `evidence` 需说明直接可见、由两条可追踪表面点之间的几何插值，还是其它推断。仅“离刀最近”不足以证明该点接触。
- 只有两端材料身份、原上表面相邻关系和当前连续段都可信时，才能用 `point_id_a/b` 与 `0≤alpha≤1` 做材料坐标插值。约定 `x=(1−alpha)x_a+alpha*x_b`，参考材料坐标使用相同权重；不能跨遮挡、不同表面支路、裂缝或不可靠长段。
- 初始表面平直且已注册时，`s_rel=(X_ref-X_c0)·e_s`；若初始表面有曲率，用其首触前有符号弧长。`X_c0` 为固定初始接触材料位置，不能随刀具移动。
- 仅能给范围时，只填 `reference_s_rel_low_mm/high_mm`；无材料对应或原点未知时三列均空。几何点位置与材料坐标的可信度分别说明。
- 实际界面接触分辨率由视频决定。**不得照搬仿真的 `dhat=0.0025 mm` 作为实验判据，也不得将接近该阈值解释为已分辨接触。**

累计摩擦滑程如另有输出，须用不同列名/文件。材料点 i 相对同处刀具刚体的速度可写为：

```text
v_rel,i = v_material,i - [V_blade + ω_blade × (x_i-x_blade_origin)]
v_t,i   = v_rel,i · t_blade
```

这里 ω 的 z 分量按实验室 xy 右手系、逆时针为正，单位 rad/s；因此 `ω_z=−(π/180)*d(blade_angle_deg)/dt`。若相对位置为 `(r_x,r_y)`，转动速度分量为 `(-ω_z*r_y, ω_z*r_x)`。

这里只在已支持的连续接触段解释为界面切向滑动；刀具切向 `t_blade` 的方向必须声明。全局水平相对速度不自动等于刀面滑速；`v_t` 接近测量噪声不自动证明黏着。Δs₀ 是当前材料身份相对初始落刀点的迁移，不能用这段速度积分替代。

### 6.6 应变与误差场：可后置

首轮不要求应力/应变彩图，位移及可见轮廓更优先。如已有应变，请同时给计算方法、拟合窗口的像素/mm 尺寸、参考构型、平滑参数、掩膜及有效性。有限应变可以报告二维 `F_2D` 或 `E_2D=(F_2DᵀF_2D-I)/2`，不能把它与小应变分量混用。

不得跨裂面、刀具遮挡或材料身份断点求梯度；单侧二维面积变化不直接等于三维体积比 J。模型中的局部极小 J 需要独立数值检查，不能仅由视频面积比证实或否定。

## 7. 空间覆盖、采样与不确定性要求

- **导出已算出的全部可信点，不只保留当前贴刀小窗口。** 至少区分刀具两侧近场、左右较远稀疏区域；底部/固定端若可见也保留。给出每类区域的初始覆盖范围，视场不足如实说明。
- 不强制生成某个点数或插值加密。每帧点间距、算法窗口大小和实际空间分辨率不是一回事；相邻点也可能共享图像支持。
- 优先保留原有时间采样。首触与遮挡/重识别切换不要仅抽取平滑阶段；数据量过大时报告抽样规则和丢弃数量。
- 未估计误差不等于误差为零。静止帧漂移、前后向误差、手工复核、独立刚体检查各自反映不同误差，分别报告。
- 仿真侧此前 0.02 mm 的接触迁移筛选值是数值工程指标，**不是要求光流必须达到的精度，也不是实验误差上限**。若当前视频只能给更宽区间，交付区间即可。
- 不用仿真的不可压缩条件、平衡方程或预期力曲线修补实验光流，再把修补结果称为独立验证。若算法本身有先验/正则化，应公开其形式与参数。

## 8. 交接前的一致性检查与后续分工

提供方做轻量核查即可，不要求本轮开发完整数据平台：

1. 所有 `case_id` 一致；frame 主键唯一；tracks 的 `(case_id,frame_id,point_id)` 唯一，引用帧均存在。
2. 相同 point ID 的参考帧和参考坐标不变；有效点有当前坐标，预测/遮挡/拒绝点不计入有效观测。
3. 图像坐标、处理链和毫米变换可复现；一段刀具下移经变换后应为实验室 y 减少、深度增加；材料运动不强制单调。
4. 时间轴、加载阶段和首触区间相容；掉帧/拼接保留说明。若帧被抽样，参考帧也需保留在 frames 表内。
5. 实验室累计位移能由当前位置减参考位置复算；没有跨遮挡假连、全局零填充或静默重设参考帧。
6. 数值表与叠加图中的点、刀具标志、坐标及有效性一致；质量报告列明缺失项，不以“完成导出”代替精度验证。

仿真端收到数据后，先核查注册、参考状态、尺度与零点，再在相同初始材料位置采样有限元位移。对照优先顺序为：材料轨迹 → 表面轮廓 → 同步 Fz → 证据充分的接触迁移。连续有效位移可以在明确方法与误差下对齐采样；接触缺失/多义区间不默认插值补全，也不外推。

若只提供像素轨迹，我们先做定性和坐标核查；若提供已标定位移但缺少同步力，可先做运动学对照；若接触不可见，则保留接触未验证的结论。这些都比输出一个没有观测支持的精确切点更有用。

## 9. 请光流 agent 首次回复的格式

请先返回下面这张能力表，随后给出现有文件位置或导出计划；不用先承诺所有项目都能做到。

| 项目 | 已有 / 可导出 / 需重算 / 无法可靠提供 | 现有字段或拟采用字段 | 缺失或误差说明 |
|---|---|---|---|
| 源视频编号、帧号与采集时间映射 | | | |
| 固定材料 ID 与参考帧坐标 | | | |
| 当前材料点位置与有效性 | | | |
| 刀具刚体轨迹及最低极点的可见性 | | | |
| 比例尺、实验室坐标与相机漂移处理 | | | |
| 首触区间及下压深度 | | | |
| 近刀和远处稀疏点的覆盖 | | | |
| 原始上表面与当前可见轮廓 | | | |
| 参考接触材料坐标或候选区间 | | | |
| 同一试次的力数据与同步 | | | |
| 跟踪质量、遮挡与重识别记录 | | | |

建议首批提交一个已有试次的短加载片段用于接口检查，保留首触前参考帧；格式确认后再导出完整可用加载段。不要仅交付带箭头视频或分组平均曲线，因为它们无法重建单个材料点的历史。

## 10. 来源与适用限制

- 本接口的坐标、Δs₀、缺失值与分级交付约定来自本项目任务需求；不是引用文献规定的统一标准。
- [iDICs, A Good Practices Guide for Digital Image Correlation, Edition 2 (2025)](https://idics.org/guide/)，[DOI](https://doi.org/10.32720/idics/gpg.ed2)：官方指南说明其范围包括 DIC 测量设置、图像相关和应变后处理，可供质量检查参考；本文不声称现有光流已按该指南验证。
- [Tayeb et al., Identification of Constitutive Parameters Governing the Hyperelastic Response of Rubber by Using Full-field Measurement and the Virtual Fields Method (2019)](https://arxiv.org/abs/1907.02687)：展示橡胶全场位移/应变用于超弹性参数识别的研究方向；其异质变形试验不是本项目刀具接触反演，不能据此保证单视角光流能唯一识别 G 或 μ。
- 上述外部来源于 2026-10-12 核对官方页面/论文摘要；没有将它们当成本实验的测量结果。本文未处理或上传任何实验视频，也未运行新的力学算例。
