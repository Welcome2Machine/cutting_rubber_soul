# 光流数据接口与力学对照约定

当前外部交付目录：`D:\Doctoral Research\Softcutting\0910imaging\rec_inc_soft_cut\results\20261012_optical_flow_exchange`。包内README、DATA_DICTIONARY、metadata与quality_report定义实际数据；本文件是仿真端接入规则，不把期望字段当已测量数据。

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
