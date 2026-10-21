# 当前工具入口

`probe_history_sensitivity.py prepare|run|worker --output <独立目录>`：原生0/33/57°保存路径从rest起传播
logG0/logit(q)/logτ总导数。包含材料度量、解析上一摩擦位置作用和真实准备时间，固定K。
每角度最多.5 mm/1800 s诊断预算；六条链共用一次物理切线分解，双链及相邻两档实际扰动资格。
`prepare`冻结源并检查两个旧失败夹具，`run`有界并行三角度；不产生新的FE加载状态。
`report_joint_history.py --study <目录> --final`：在诊断结束后核对哈希，分析共同窗口、列相关性、
总误差投影与每角度均改善的局部方向，输出HTML。线性化投影不等于FE预测或完成标定。
见[解析历史偏导与三参数联合检验](../docs/joint_history_sensitivity.md)。
旧τ专用驱动保留于`outputs/model_identifiability_audit_20261021/history_tau/source`；
旧报告绑定原冻结源，不用新实现重标旧失败前缀，也不放宽旧哈希保护。

`probe_finite_width.py --output <独立目录>`：三角度.16/.5 mm六个准确历史受载场，连续面外位移及
1/4/8半宽分层；三维材料/原面内接触耦合条件方向、双尺度残差资格和底面面外边界控制。
不产生新加载曲线，不将导数当真实有限宽反力。[方程与限制](../docs/finite_width_diagnostic.md)。

`probe_angle_factors.py --output <独立目录>`：三角度9个原生接受场、五个共同参数的固定历史隐式方向，
双尺度实际扰动验证；非全路径灵敏度，未合格方向不用于选参。
`run_transverse_study.py prepare|worker|case|report --output <独立目录>`：同材料总应力平面应力诊断，
0/33/57°各自.01 mm预检及凝聚切线合格后至.5 mm。原生接触、宽度、物理验收保持；
自动HTML实验对照及57°保持门槛，失败继续，不自动晋级。[范围与理论依据](../docs/angle_amplitude_factors.md)。

`run_tip_mesh_study.py prepare|worker|case|report --output <独立目录>`：原生0/33°×R10/20 μm，
表面5→2.5 μm和邻近体网格，四组独立静止加载至.5 mm。四组几何/八个旧场重审，
冻结粗网格对照，原预算/物理验收/双历史/实际PTC探针；自动HTML粗细曲线，失败转下组。
一次细化不认证收敛，禁止跨网格续算；[范围和记录](../docs/tip_radius_study.md)。

`run_tip_radius_study.py prepare|worker|case|report --output <独立目录>`：原生IPC主线的半径10/20/40 μm、
0°/33°共6组至.5 mm；独立.03 mm预检/准确历史，原预算验收，失败继续下一组。共同材料与楔角厚度不变。
自动HTML/曲线/成对评价，候选不自动晋级；解析刀具事件缺口未混入或修复。[方案](../docs/tip_radius_study.md)。

`run_transported_blade.py prepare|worker|case --output <独立目录>`：统一P1边界/解析刀具的全局候选。
prepare要求回归证据，完成三场空间积分/整体切线及三角度独立0.01 mm预检；worker仅接受prepared。
57°限0.16 mm，0/33°限0.5 mm，原预算/验收、不自动缩步。复用准确历史生命周期和实际方向探针，
HTML持续更新。旧解析路线不自动获准重启。[实现与资格](../docs/transported_blade_validation.md)。

`qualify_contact_transport.py --case <冻结目录> --kind common|controls --output <新目录>`：
七个接受场/一个失败场的局部接触核、方向切线和样条—刀具静态交叉诊断；每进程独立冻结源码。
限定当前q30的0/33°及fixed_half57，600 s预算，不推进FE。不能把部分接触力当作Fz。
`report_contact_transport.py --study <证据目录>`：核对原输入/候选哈希、汇总与HTML/图形渲染，
不修改物理结果。详见[资格报告](../docs/contact_transport_qualification.md)。

`probe_model_structure.py fixtures|fields|report --output <新目录>`：共同材料端点结构、
冻结q30六态的18个条件方向、真实持续EV运动学与HTML；不推进FE，不拟合参数。
五个微扰尺度要求相邻两档通过；材料条件导数不是新历史曲线。报告可独立渲染，单独保存源码，
必须核对原测量源码/输入/方向哈希。见[结构核查](../docs/model_structure_review.md)。

`audit_physical_balance.py --study outputs/material_uncertainty_20261021 --output <独立目录> --volume-sensitivity`：
核验冻结q30的0/33°保存场，逐项虚功、实际材料切线/转动/耗散、加载几何的共同转动和局部修正覆盖。
不推进加载；可选完整切线K条件灵敏度会做线性解和微扰装配，不产生新接受态。
部分局部项的条件力差不是新Fz。[解释与门槛](../docs/physical_qualification.md)。

最新：[接触失败与IO修复审查](../docs/contact_solver_review.md)。
`probe_contact_failure.py`读取冻结失败场，提供方向/分量/两侧切线/条件求解探针；
`probe_nose_resolution.py`是保持材料历史的条件圆鼻面片对照，不能冒充完整新路径；
`report_contact_time_review.py`生成哈希核查后的同角度实验叠图和HTML；
`recover_contact_validation.py`显式冻结IO覆盖层、接管原进程并补齐IO中断前缀，
不绕过源码匹配、不重开失败57°。这些入口需要明确路径和范围，不能用历史命令自动续跑。

2026-10-21：新增`run_contact_controls.py`，六条独立0–1 mm时间正则对照、准确历史审查、
有界后台队列与HTML。`prepare`要求已有资格证据目录；直接删相邻VV已经否决，不能进入长队列。
局部客观滑移核尚未用于FE，见[方案](../docs/contact_time_validation.md)。

| 工具 | 用途 | 限制 |
|---|---|---|
| run_common_material_study.py | 最新共同Maxwell三方向真实FE、至多一个组合提案复算、条件.8 mm迁移与HTML | 三角度共用参数；固定.01/原验收；响应近似非FE力；数值失败或有限范围负结果不证明材料家族无解 |
| review_common_material.py | 冻结快照下全部接受前缀、斜率窗口敏感性、终态/拒绝场重新装配与HTML | 读取指定study的冻结物理源；不加载求解、不拟合、不改队列；运行中前缀明确标记 |
| review_ptc_acceptance.py | 三组57°失败的原PTC日志、重试shift和预测残差界审查 | 仅适用本次model-error完整步证据；发现非完整步即拒绝此排除判定；不重建虚构方向或启动FE |
| diagnose_ptc_directions.py | 冻结失败增量的实际PTC跟踪、方向/度量探针、Newton隔离对照及接触分量/端点几何 | trace/probe/candidate/components显式模式；准确双历史、原物理验收与有界预算；固定场探针不是平衡态，候选失败则不续加载 |
| report_ptc_directions.py | 上述20261020诊断的哈希核验、原跟踪回调去重、HTML与图形 | 仅呈现已有记录；真实线性调用核对去重计数；不运行力学，不能把诊断通过写成增量通过 |
| audit_contact_material.py | fixtures/fields/baseline/report：共同运动、摩擦时间尺度、逐项接触力、实际Maxwell状态及HTML | 无新FE加载；准确历史；旧基线每进程一个冻结源；载荷权重/功率记账不是因果Fz占比；保留物理检查不满足的结果 |
| plot_experiment_angles.py | docs中0/33/57°完整实验CSV与已有共同Maxwell路径叠加；HTML、PNG/SVG及对齐CSV | 仅绘图；保留各自零点和试次，后段已裂，不用全程差值宣称完整体验证 |
| analyze_angle_emergence.py | 当前共同Maxwell既有三角度至1.5 mm复核、准确历史锚点装配、力/材料身份CSV和HTML | 不推进加载；--review-three仅核验与呈现现有分析；标明估计起裂，裂后不用于完整体拟合 |
| run_unrelaxed_study.py | 已结束的固定G₀/K去松弛机制诊断 | 非当前默认；不自动恢复/深度延伸 |
| check_contact_transport.py | 保存路径的刀具节点承载、接触材料身份与连续节点承载历史 | 不推进加载；阈值敏感性、力分区闭合；邻近区不能当压力/真实接触时间 |
| run_native_gcp.py | 已停止晋级的原生P1/刀具IPC＋本体GCP诊断 | 不是当前默认；需重新限定资格后才能启动 |
| audit_mainline.py | 准确历史的切线、线性求解及成本 | 固定失败场，不推进载荷 |
| diagnose_al_mode.py | 位移模式、自接触、子空间探针 | 代数探针不是新物理平衡 |
| qualify_al_direction.py | 有界完整增量重放与原材料核审查 | separated-self显式改变自接触适用域，非原方程等价修复 |
| run_separated_contact.py | 未自接触区间的独立加载、自动原核审查 | depth/increment仍为q；--lab-speed .2才选择实际竖直速度 |
| run_geometric_contact.py | GCP原失败增量资格、资源实测、新独立长线及HTML日志 | CLI depth是竖直d，内部转换q；固定.005 mm加载，无自动缩步／追加预算；不能直接接旧分离诊断历史 |
| run_material_candidate.py | 共同Maxwell历史计算 | 明确速度、角度及预算 |
| run_augmented_blade.py、run_al_integration.py | AL/P2资格与终态审查 | 默认法向.1 mm/s为机理控制 |
| render_solver_dashboard.py | HTML状态/日志 | 旧PID不是当前状态 |
| visualize_ipc.py | 网格与场可视化 | 不用图掩盖未收敛 |
| check_publish.py | 发布文件筛选 | 原数据和结果留本地 |

其他工具按各自明确作用域使用。无依赖的旧archive脚本已删除，原文留在本地清理前源码备份。
当前决定见[研究状态](../docs/research_status.md)。历史诊断的默认输出可能已退役，不应直接运行旧命令。
# 2026-10-21新增材料不确定性试验

`run_material_uncertainty.py`复用`run_common_material_study.py`的冻结、独立历史、监督和审查流程，
检验同一G0=.966下长期份额.30/.50；只跑0/33°至.5 mm，57°仍受接触资格门槛限制。
真实PTC观察不改变物理方向/验收，原始方向与双历史保存。方案见
[材料参数审查](../docs/material_uncertainty.md)，本地HTML在`outputs/material_uncertainty_20261021`。
不要对已有目录重复prepare/launch；旧控制组仅用于比较，不能拿来恢复新材料。
