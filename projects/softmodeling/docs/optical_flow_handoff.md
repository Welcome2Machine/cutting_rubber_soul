# 光流数据接口与力学对照约定

当前有两类外部交付：20261024的前1 mm固定ID位移包，以及20261012的完整轨迹接口包。各包元数据定义实际字段；本文件是仿真端接入规则，不把期望字段当已测量数据。

2026-10-25只读核查补充：[视频—FE共同观测及分析规范](video_fe_comparison.md)说明当前生产/研究分支、接触地址结果、统一参考态/标距和拟议比较包。本文件保留既有两类输入的兼容规则；新比较规范尚未集成到工作台。外部仓库正在更新，不把新代码自动当作旧结果的生产版本。

同一试次的不同交付存在零点不兼容：旧接触地址的零点帧234/437，在新33°/57°位移包中已分别对应0.174304/0.863387 mm，且比例尺也不同。不能直接拼接深度列，须按源帧及各自参考/尺度重建派生比较轴。力学正则修正轨迹、预测/保持及展示补点单列，不混为独立实验真值。

## 20261024前1 mm固定ID位移包

目录：`D:\Doctoral Research\Softcutting\0910imaging\rec_inc_soft_cut\results\20261024_n_group_first_1mm`。仅含1014n33a1与1014n57a1；每组有原始/去阴影/位移展示视频、`material_displacements.npz`、`frames.csv`、`metadata.json`及零点核查。NPZ使用`allow_pickle=False`读取，位移已按固定材料ID重新参考到所选接触帧，不套用旧CSV的`reference_frame_id`字段。

- 端点比较使用`displacement_endpoint_supported`；连续轨迹使用`all_intermediate_measurement_supported`，同时剔除参考/当前或全历史的`tracker_inferred`、`tracker_ambiguous`、`tracker_blade_occluded`标记。跨深度趋势优先用全片共同合格ID，避免换点引入假趋势。
- `contact_image_xright_yup_mm`为初始位置，`displacement_from_contact_xright_yup_mm`为位移；是图像轴下x向右/y向上的条件毫米坐标，不是已注册实验室坐标。约3 mm总行程提供条件比例尺；所选零点是材料运动起始，未独立确认机械首触。
- `capture_time_from_contact_s`采用源AVI时基200 fps；展示视频12 fps不是加载时间。`indentation_estimate_mm`来自刀尖图像竖直下移，实际末端约0.99818/0.99721 mm；57°存在很小的局部非单调深度，不排序后冒充时序。
- 本次FE比较按初始刀尖平移一次，在FE深度0的已变形接受态中固定三角形重心映射，比较相对此态的位移；接受态间深度插值，域外点保留为缺失，不用最近节点替代。该参考态不同于工作台`mesh.reference_positions`的未变形构型。实验侧面投影与平面应变内部场的等价性、相机运动、尺度和角度差异单列。
- 刀具相对位移`u_material-u_blade`与绝对位移分别报告。它不等于接触界面滑移；点阵模长也不是累计滑程。视频未保证前1 mm完整无裂，不能由光流或局部暗痕自行认证。

[已完成的视频—FE比较HTML及数值证据](../outputs/video_fe_1mm_20261024/index.html)。33°视频1014n33a1不同于力目标1008n33a1；57°同名1014n57a1也未独立完成力—视频同步。原文件只读，不以模型最佳拟合反向修改尺度/零点。

### 固定区域的候选评价约定

[刀路/侧面可比性及区域运动核查](../outputs/video_fe_1mm_20261024/kinematic_check/index.html)已建立可复用指标，定义与源指纹在`kinematic_check/regional/regional_analysis.json`，固定ID与权重在同目录`fixed_cohorts.csv`。两角度、各模型、所有检查点使用同一持续有效ID；区域由初始坐标固定：中心r≤.25 mm、左右环带.25<r≤1 mm按x符号分开、远场1<r≤2 mm、更远外场r>2 mm。点等权为主，固定.25 mm空间格等权作敏感性，不按候选重新调整。

同时输出绝对/刀具相对位移、相邻窗口增量和区域平均位移差。后者抵消共同平移；固定点位置均值之间的距离变化`ΔL=|mean(X+u)_A−mean(X+u)_B|−|mean(X)_A−mean(X)_B|`还对共同二维旋转不变。该量不是局部应变、体积、界面滑移或裂缝张开，仍受尺度、投影和材料映射影响。没有静止实物参照时，不以叠加网格/文字或变形橡胶估计相机校正。

后续共同材料候选沿用相同观测算子及掩膜，分别报告Fz/斜率与上述运动误差；未知实验不确定度时保留阈值缺失，不给任意加权总分或宣称标定通过。平面应变内部截面与自由侧面尚未等价，运动指标先作为条件诊断约束。

### 有限标距变形约定

[变形量化报告](../outputs/video_fe_1mm_20261024/deformation_review/index.html)在上述固定ID比较上增加二维仿射拟合`x_i=F2 X_i+t`。预设中心间距.4 mm、半径.25/.4/.6 mm；点等权，仅按初始点数、协方差条件数、角覆盖和中心外推程度判定资格，不按实验/模型误差挑区域。FE与视频使用同一成员、权重和拟合，不能拿FE原生单元极值直接对比经过平滑的测量量。完整定义及缺失理由在`deformation_review/patches/patch_analysis.json`、`patch_geometry.csv`；逐点区域身份在`patch_memberships.csv`。

`lambda_max/min`为F2奇异值，`green_E_*`为有限标距Green量，`log_lambda_*`为对数主伸长；它们不是已经认证的点应变。`projected_area_ratio_detF2`为投影仿射面积比，不是3D J，不用于直接标定K或认定裂纹。`polar_rotation_deg`包含未去除的相机转动；`affine_residual_rms_mm`混合非仿射真实变形与跟踪误差，不直接作为噪声方差。重叠区域和密集点不是独立试次，不按点数构造虚假置信区间。

下一步观测资格检查需使用原光流程序及同参数：输入已知位移变形后的初始纹理、对应真值和遮挡，输出同一字段/固定ID规则；另设刚体运动和非FE光滑变形控制。评价左右/近远距离、同标距主伸长及支持点丢失，避免仅在生成FE插值上自证。合成平面图像通过不能排除真实面外运动/透视影响。未完成此检查前，视频作为条件运动约束，不把剩余误差全部归入FE物理模型。

## 20261012完整轨迹接口包

目录：`D:\Doctoral Research\Softcutting\0910imaging\rec_inc_soft_cut\results\20261012_optical_flow_exchange`。包内README、DATA_DICTIONARY、metadata与quality_report定义实际数据。以下`tracks.csv`和`reference_frame_id`约定适用于此包；新NPZ包按上节接入。

## 用途和限制

光流辅助检查表面运动、刀具相对运动和参考接触迁移，不能单独测得应力、压力、材料模量或摩擦系数，也不阻塞力学主线。当前FE为完整体平面应变；侧视自由表面的三维运动与模型截面是否等价需检查。

交付包有10组试次，`case_index.csv`为索引；每组同名目录有`metadata.json`、`frames.csv`、`tracks.csv`和`quality_report.md`，并附已有轮廓/掩膜/核查图。包级`validation.json`说明数值一致性，`exporter_code`为生成代码。当前实验室坐标标定、机械首触和力—视频同步尚未确认。

**轨迹先筛选`valid=1`；比较初始构型时另需`reference_frame_id=0`。** 断点后新参考段不能直接当初始位移；模型接触估计不是直接观测真值；不得默认整个加载段均未裂。

力学目标使用本仓库[任务要求](task_requirements.md)指定的三条Fz CSV。旧V01等交付包原力通道不用于标定，也不自行乘传感器系数。不同试次或未同步的力/视频不能强行逐帧对应。

## 坐标与材料身份

- 图像源像素：x向右、y向下，左上角像素中心为(0,0)。导出声明裁剪、缩放、旋转、镜像、校正/稳像链；不能只改列名冒充源坐标。
- 实验室坐标：x向右、y向上，固定于基底。比例尺、相机斜视和可能单应变换应有独立标尺/夹具依据，不能用模型倾角或未知投影刀厚反求真值。
- `point_id`代表同一材料身份；遮挡外推与再观测分开，没有重识别证据则建立新ID/新参考帧。
- 帧间光流沿材料轨迹复合，不在固定像素上直接相加。参考坐标不得随重定位静默改变。
- 侧面内部点不是原上表面点，近刀点也不是已确认接触点。当前距离分组与初始材料位置分组分别保存。

模型表面eₛ=(cosθ,−sinθ)、外法向eₙ=(sinθ,cosθ)，参考落刀点7eₛ+5eₙ。该定义用于注册，不要求实验跟踪结果服从模型。

## 时间与深度

`frame_id`保留源解码编号；`time_s`优先由采集时间戳，不能混用慢放/剪辑播放时长。位移参考帧与首触帧不同。

`depth_mm`是刀具相对固定基底的实验室竖直下移，不是表面沉降或斜面法向投影。首触可给“最后可证分离—首次接触证据”的帧区间；无法确认则保存参考帧刀具位移，条件深度明确标记。最低点隐藏时可用刚体标记推算，但状态为推断，不标直接观测。

同一深度的下压、保持、卸载必须分开。帧/时间原始采样保留，不为了匹配FE输出制造实验帧，不调时间/深度平移使曲线吻合。

## 最小可复用格式

CSV为UTF-8、逗号分隔、英文列名，单位写在列名；缺失留空、JSON用null，布尔0/1。数值存在时须有限，不用0或−999冒充缺失。

| 文件 | 主要内容 |
|---|---|
| `metadata.json` | 来源/指纹、视频尺寸和时间、实验工况及依据、坐标/标定、全局参考与首触、算法/参数、质量语义、同步与缺失项 |
| `frames.csv` | `case_id,frame_id,time_s,stage,depth_mm,depth_source,depth_uncertainty_mm,blade_tip_x_px,blade_tip_y_px,blade_tip_status,blade_angle_deg,valid_frame,invalid_reason` |
| `tracks.csv` | `case_id,frame_id,point_id,reference_frame_id,reference_kind,x_ref_px,y_ref_px,x_px,y_px,point_kind,track_status,valid,confidence,fb_error_px,position_uncertainty_px,invalid_reason` |
| `quality_report.md` | 有效空间/时间范围、遮挡、阴影、相关失败、误差定义、重识别和不能提供的量 |

可选毫米坐标：`x_ref_lab_mm,y_ref_lab_mm,x_lab_mm,y_lab_mm,ux_mm,uy_mm`，必须在同一基底参考系且u=当前−参考。缺少标定时保留像素量，不能添加伪毫米值。

`stage`为precontact/loading/hold/unloading/unknown；`blade_tip_status`为observed/rigid_inferred/predicted/occluded/unavailable。`track_status`可为tracked/reidentified/predicted/occluded/out_of_frame/rejected/missing；predicted值不能冒充`valid=1`的观测。confidence与前后向误差不是已校准的位置概率/毫米误差。

有现成轮廓/掩膜时保留点序、闭合关系、参考身份和可见性。接触候选需给证据类型和不确定性，允许区间、多义、缺失；不要根据FE答案反向定义实验接触。只输出彩色光流展示视频不能替代数值轨迹。

## 核查顺序

先验证主键、有限值、帧/时间递增、身份/参考一致、valid语义和坐标转换，再比较清晰且可解释的加载窗口。初态位移、增量运动、刀具相对运动分别报告。材料完整性、标定或同步不足时限制结论，不能通过平滑或插值补出证据。完整观测定义见[刀尖接触材料坐标](tip_contact.md)，文献依据见[统一导读](../references/README.md)。
