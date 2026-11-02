# 当前科研工具入口

更新：2026-11-02。执行范围见[科研原型任务](../../../researchtopic/research_prototype_task.md)，模型见[当前规格](../docs/current_model_spec.md)，结果见[研究状态](../docs/research_status.md)。

## 三条活跃科研入口

| 入口 | 目的 | 范围 |
|---|---|---|
| `python -m tools.run_material_candidate` | 一个实际角度的完整体计算；本地agent负责拟合 | 同一材料/接触描述，明确实验未裂有效区间 |
| `python -m tools.fracture_onset` | 读取合格完整体保存态，输出共享起裂指标与位置 | 只读后处理；未知阈值不称已预测起裂 |
| `python -m tools.run_coupled_growth` | 单一面级CZM/BE/IPC/Maxwell裂后机制 | 0°居中、预设裂路；新C2从自身无预裂初态开始 |

优先复用已有数据。完整体队列仅在当前任务明确需要补算时使用；不由UI或旧脚本自动恢复队列。
本轮C2使用新配置和当前任务的更小预算，不能直接复制历史2 mm或数小时命令。

## 复用的支持能力

| 工具/模块 | 当前用途 |
|---|---|
| `qualify_intact_baseline`、`material_case` | 完整体来源和必要独立力学审查 |
| `plot_experiment_angles`、`visualize_ipc` | 已有力曲线、接受状态和场量图 |
| `workbench.results` | 已有标准包与字段导出；保持有效区间和缺失值 |
| `report_staged_c1_event`、已有事件报告 | 读取历史机制证据；不升级原资格 |
| `run_coupled_growth audit`及现有必要审查 | 当前机制路线的物理、历史、几何和能量核查 |
| `report_coupled_growth` | 已审查路径的裂面、载荷与关键帧；C2按实际输入修正标签，不使用要求非空预裂的C1报告 |
| `face_witness`、`window_journal`、`audit_coupled_chunks` | 已有溯源/恢复能力；本轮不扩建通用设施 |
| `manage_outputs` | 已有无损归档/恢复；整理不启动FE，本轮不批量删除 |
| `check_publish` | 代码、文档、审阅配置发布检查 |

独立审查必须重组物理量，不因收束而改成信任生产布尔值。旧数据只读图表可复用原审查及哈希；它不是新源码的生产续算。

## 冻结的研究入口

以下代码保留供历史复现与现行依赖，不是本轮推荐执行路线：旧径向CZM、全内部边自由网络、FFM虚拟裂增量/摩擦功、离散梯度对照、条件释放、纯预裂控制、闭合C1时间控制和进一步性能扫描。

`run_cohesive_cut`、`run_zero_gap_cut`、`run_fracture_route`、`run_endpoint_face_route`、`run_precrack_entry`以及条件窗口系列中部分工厂仍被当前主线调用；冻结研究用途不等于可以删除文件。

不再为每个局部失败新增run/audit/report三件套。只在当前科研输出有真实阻断时修改现有入口，必要的小诊断有明确问题和停止条件。

## 运行与发布

先核对当前任务预算、解释器、输入/源码身份、活跃作业与合法资源。活动数值源码冻结，输出独立，不跨模型/参数/几何/历史续算。
只读取帮助或已有结果不会授权新加载。原物理方程、残差、CCD/J、边界、历史、不可逆性和Gc/能量账保持。

完整工作台使用说明见[工作台文档](../docs/workbench.md)，接触量定义见[tip_contact](../docs/tip_contact.md)，接口见[整合契约](../../../docs/interface_contract.md)。
原始实验、媒体、NPZ、结果和机器设置留本地；Git仅管理源码、文字和审阅配置，遵守本项目与根发布检查。

## 历史命令索引

[调整前完整工具说明](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/78405e242bc4246d269dda5ff5bdf11814b1c0fa/projects/softmodeling/tools/README.md)保留旧队列、性能、网络/间隙矩阵和归档命令。
历史命令的预算与运行授权不自动继承。旧结论见研究状态和方法说明，不从配置存在推断已通过。
