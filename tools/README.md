# 当前工具入口

| 工具 | 用途 | 限制 |
|---|---|---|
| run_common_material_study.py | 最新共同Maxwell三方向真实FE、至多一个组合提案复算、条件.8 mm迁移与HTML | 三角度共用参数；固定.01/原验收；响应近似非FE力；数值失败或有限范围负结果不证明材料家族无解 |
| review_common_material.py | 冻结快照下全部接受前缀、斜率窗口敏感性、终态/拒绝场重新装配与HTML | 读取指定study的冻结物理源；不加载求解、不拟合、不改队列；运行中前缀明确标记 |
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
