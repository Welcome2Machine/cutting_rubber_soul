# 当前科研工具入口

更新：2026-11-16。执行范围见[科研原型任务](../../../researchtopic/research_prototype_task.md)，模型见[当前规格](../docs/current_model_spec.md)，结果见[研究状态](../docs/research_status.md)。本轮仅服务器执行基线修复与运动学分析。

## 本轮主要入口

| 入口 | 目的 | 范围 |
|---|---|---|
| `python -m tools.run_material_candidate` | 服务器在任务条件成立时执行一个完整体短程 | 真实配置、自身初态、有限预算、完整接受历史 |
| `python -m prototype.contact_trace export` | 已有接受态的刀尖材料坐标与接触几何后处理 | 本轮修复新版几何／起点兼容及显式合格前缀，保留源资格 |
| 根目录 `python -m softcut_solution compare` | 有实验数据时进行固定材料点的条件比较 | 配准、时间、身份与实验未裂窗口独立核对 |

优先复用已有数据。完整体队列仅在当前任务明确需要补算时使用；不由UI或旧脚本自动恢复队列。
本轮不继续C2/F2。新增完整体路径严格受当前任务的数量、目标与预算约束，不复制历史2 mm或数小时默认命令。

### 完整体运动学导出

在FE子目录、计算节点执行已有状态后处理（不加载FE）：

```bash
python -m prototype.contact_trace export --source "$case" --output "$new_output" --qualified-prefix
# 旧审查没有source_binding时，另外提供当时已绑定result/audit的清单：
python -m prototype.contact_trace export --source "$case" --output "$new_output" --qualified-prefix --manifest "$prior_manifest"
```

`--audit`可指定独立审查sidecar。前缀从初态连续覆盖，逐态核对记录、NPZ、父位置、材料metric与真实dt；
新建指纹不能代替旧审查的来源绑定。未显式选前缀时仍要求完整请求完成；缺可信审查的完整旧结果只能得到未认证观测。
新版审查绑定最终result字节、原数值输入和逐态记录。运行结束时在独立记录不变的条件下完成最终IO绑定；
已有未绑定最终字节的审查须使用原清单或既有审查器另存sidecar，不在导出时追补认证。
圆鼻片数和准备位置从实际元数据与保存拓扑核对，缺失/冲突拒绝，不默认64片或重建旧起点。
`contact_trace.json`（`softcut-contact-kinematics/2`）、`kinematics.csv`和`index.html`保存全接受频率结果。
固定顶面材料ROI为落点左右近场/远场，固定标距端点和实际ID写入定义；图示帧抽稀不参与求差分。
`raw_mm_s`是实验室系同材料身份速度差的当前切向投影；`transported_mm_s`先用刀具有限刚体运动输运父态材料点。
后者消除有限gap共同转动的伪相对运动，但仍只是几何邻近诊断。累计量按连续有效端点段及右端增量积分；
接触新生步排除、正常相邻边跨越保留，失接触/歧义/非邻接跳变切段，不跨缺口积分。
快速跨越多条相邻材料边时，用两接受态之间最多32个子区间的几何弦采样检查投影连续性，避免因网格细而错误切段；
失接触、歧义或仍未解析的对应保持切段。中间几何不当作新平衡态，也不证明整步始终有物理承载接触。
几何状态切换另给实际接受步区间；CSV显式保留运动学valid/status，固定标距为端点间欧氏距离。
图中保留全部有定义的速度端点，包括新生/未解析对应处的孤立点；仅连线和累计量按有效轨迹切段，不把有效端点隐去。
footprint是距刀具小于dhat的初始顶面范围与当前长度，不是压力承载区。
`workbench.results.export_case(..., trace_path=...)`只消费同源共同观测，保留元数据与缺失，不另推断刀形。

主运行CLI的`--mu 0`使用原有无摩擦分支；负数、NaN、Inf拒绝。固定εu历史不变，运动学表逐步列出εu/dt。

## 复用的支持能力

| 工具/模块 | 当前用途 |
|---|---|
| `material_case`／`native_audit` | 使用实际保存配置重组完整体历史与原方程；复用既有独立审查 |
| `qualify_intact_baseline` | 固定旧23°/603态的历史核对，不是任意新路径的通用 `--source` 工具 |
| `plot_experiment_angles`、`visualize_ipc` | 已有力曲线、接受状态和场量图 |
| `workbench.results` | 已有标准包与字段导出；保持有效区间和缺失值 |
| `report_staged_c1_event`、已有事件报告 | 读取历史机制证据；不升级原资格 |
| `run_coupled_growth audit`及既有断裂审查 | 历史机制路线的物理、历史、几何和能量核查，本轮不启动新断裂路径 |
| `report_coupled_growth` | 历史已审查路径的裂面、载荷与关键帧，保留原资格 |
| `face_witness`、`window_journal`、`audit_coupled_chunks` | 已有溯源/恢复能力；本轮不扩建通用设施 |
| `manage_outputs` | 已有无损归档/恢复；整理不启动FE，本轮不批量删除 |
| `check_publish` | 代码、文档、审阅配置发布检查 |

独立审查必须重组物理量，不因收束而改成信任生产布尔值。旧数据只读图表可复用原审查及哈希；它不是新源码的生产续算。

## 冻结的研究入口

以下代码保留供历史复现与现行依赖，不是本轮推荐执行路线：起裂诊断、新的CZM/F2加载、旧径向CZM、全内部边自由网络、FFM虚拟裂增量/摩擦功、离散梯度对照、条件释放、纯预裂控制、闭合C1时间控制和进一步性能扫描。

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
