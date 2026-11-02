# 项目审查：从求解器诊断收束到科研原型

更新：2026-11-02（日本时间）。审查源码：`78405e242bc4246d269dda5ff5bdf11814b1c0fa`。

本页是当前项目决策；逐轮运行事实仍以既有 `research_status.md`、`fracture_implementation.md` 和运行库为准。[2026-10-31审查](project_review_20261031.md)作为历史保留。

## 1. 结论与负责人调整

第十四轮完成了合规的局部方法检验和止损，没有交付新的完整断裂过程。代码存在历史路线、运行入口和审查包装的累积；尚无证据表明大量核心算法被整段重复复制。

研究方向确实需要收束：闭合 C1 的微秒速度、事件相位和时间控制，已经被当成全部科研工作的统一前置门。它与真实偏心几何、0°/33°/57°受力—变形—滑移—起裂的关系尚未建立。负责人前几轮持续把这种门升级为下一唯一任务，对偏移负有责任。

**即日起冻结闭合 C1 的进一步控制器开发；活跃交付改为完整体研究输出、现有起裂诊断、一条面级 CZM 裂后机制原型。** 旧数值失败不改写；未知动态峰值不被宣称正确。新任务见[科研原型任务](research_prototype_task.md)。

## 2. 第十四轮结果

| 项目 | 实际结果 | 审查判断 |
|---|---|---|
| 三个完整父态的整步/两半步 | 真实 h=0.4 μs；必要处补4个0.1 μs子步 | 真实历史分支成立 |
| 接触局部速度 D01/D12 | 约0.3367% / 0.3434% | 未过旧严格递减门；不能据此判定发散 |
| 同三路径绝对数值功 | 1.30564e-4 → 7.99518e-5 → 4.98216e-5 N mm | 能量改善不自动认证速度 |
| 同父态8 μs控制 | 两策略均完成；124.398 → 125.329 s | 未取得可证实的成本收益 |
| 8 μs绝对数值功 | 下降约6.22% | 局部改善 |
| 政策间速度/反力差 | 0.34994% / 2.32726% | 政策对照，不是三档时间验证 |
| 正式400 μs矩阵、新面接触C | 未启动、未出现 | 未新增完整断裂或持续切削资格 |
| 全轮计算运行并集 | 9分50秒 | 方法停止，非预算耗尽 |

本地补跑本轮相关测试17项全部通过；未重算服务器完整NPZ路径。上述长路径/局部分支数值来自服务器报告，资格与代码逻辑经本次独立审查。

### 阴性结论的范围

旧热点位于原窗口约357 μs；本轮对照只到350.4 μs。旧配额为 `B*h/T`，修正候选单步上限为 `B/16384`（另有余额保护）。以T=400 μs计，两者交叉在h=24.414 ns。原最困难步长约24 ps，明显属于另一尺度。

因此只能说该候选在选定8 μs窗口未取得联合收益，不能说累计额度已在原热点被否定。这个限制也不构成继续开一轮控制器工程的自动理由。

## 3. 代码规模与真实复杂度

口径：审查SHA中Git跟踪的Python文件，物理行含注释/空行，不计配置、Markdown和运行数据。

| FE范围 | 文件数 | 行数 |
|---|---:|---:|
| tools | 80 | 14,503 |
| tests | 92 | 11,964 |
| prototype、mechanics、顶层模块，含历史路线 | 41 | 9,151 |
| workbench | 3 | 857 |
| reproductions | 10 | 665 |
| 合计 | 226 | 37,140 |

第十轮结项923005a至当前增加26个Python文件、2350行；其中2141行（约91.1%）在tools/tests。第十四轮本身仅增加325行，不应把全部历史复杂度归咎于本轮agent。视频代码及整合/UI不计入上述FE规模。

对非测试FE函数作AST函数体精确比较（去位置与首段docstring，至少12行且100个AST节点），未找到跨文件完全相同的大函数体。这不是近似克隆的完整分析；问题证据主要是职责/入口叠加。

### 可定位的结构问题

1. **模型构造穿过历史试验层。** 当前继承链为 `CoupledCohesiveProblem → CoupledContactProblem → PurePrecrackProblem → EndpointFaceCutProblem → FaceCutProblem → EnergyCutProblem → ZeroGapCutProblem → ViscoelasticProblem → IPCProblem`。其中两个类定义在CLI工具中，多处以 `__class__` 换型。维护耦合明显，但这本身不证明物理计算错误。
2. **模块全局替换选择运行行为。** `run_coupled_growth.install`、`run_coupled_growth_window`、`branch_coupled_window` 等替换工厂、恢复、来源和finish函数；条件审查又有分块包装。下一轮不再叠加新包装。
3. **审查职责反复组织。** own-initial、conditional、continuation、chunk几条路线分别组织时间链、历史链、来源和状态恢复。独立力学重组有价值；可复用的组织层重复不应成为新增通用框架项目。
4. **科学接口未跟上。** `tools/run_zero_gap_cut.py`把G∞、Gv、K、τ、μ写死；`prototype/zero_gap_cut.py`限制0°并把刀位居中。当前并不能直接装入本地拟合包或真实s=7 mm斜切工况。
5. **历史源码是现行指纹的一部分。** `run_support.source_hashes`覆盖整个prototype/mechanics包，部分旧虚拟路径模块也在当前来源集合内。直接搬移或删除可能打断历史恢复，因此先逻辑冻结，不大批删除。
6. **当前文档混有旧路线状态。** 当前模型规格、工具入口与代码地图逐轮追加旧控制和命令，存在“独立面核尚未接入”与实际已采用面级路线等时态冲突。本次已将活跃地图缩短；旧全文由固定版本链接及Git历史保留。

## 4. 验收为什么需要按用途分开

### 保留的基础

原物理方程、残差、完整CCD、正J、边界、参考面积/Gc、不可逆历史、失败回滚、物理与数值能量分账，以及可重放输入仍然必须正确。实验比较还需要真实几何、共同参数和实验标签的可比性。

### 不再作为全项目共同前置门

- 每一个局部速度差均严格单调下降；
- 每一档离散轨迹的事件检测区间都必须重叠；
- 约10 N响应上的10 μN机制差标签；
- 闭合C1完整400 μs瞬态必须先通过，才允许做完整体/起裂诊断/C2可行性。

事件检测括号不是连续真解的置信区间。例如 `t_h=t*+3h`、检测括号 `(t*+2h,t*+3h]`，减半后的括号与它不重叠，但事件时刻依然收敛。重叠可作诊断，不能作为所有用途的必要收敛条件。

第十四轮局部约0.34%的轻微非单调不能认证渐近阶，也不证明发散。第十三轮全窗25.75%速度差和13.29%反力差仍真实存在，动态峰值和掉力时程仍不应作为定量研究输出。

科研原型使用独立命名的观测量敏感性标准：先明确要比较的力—行程、区域变形、滑移、起裂指标或新裂面接触，再做一次有针对性的时间/必要空间检查。两档筛查不叫渐近收敛证明，缺少检查就限制该量的用途。不得改旧报告标签或事后滤波、平移以取得通过。

## 5. 收束后的研究产物

### 主线A：实际几何的完整体与起裂指标

本地agent继续共同参数拟合。服务器复用真实保存态和现有 `tools.fracture_onset`，形成0/33/57°统一表：Fz/Fy、法向/摩擦贡献、区域变形、接触材料迁移、可得的真实滑移、q_l指标与候选位置。未知阈值只输出指标，不宣布已预测起裂；未推送本地成果标为待交接。

### 主线B：单一裂后机制原型

只维护现有面级CZM、BE、IPC、Maxwell组合。先整理第十轮已有宽缺口新面接触案例，输出关键帧、力—行程、新裂长和载荷重分配，明确规定缺口、示例参数、条件窗口的边界。

随后在限定预算内做一次完整材料C2可行性试算，必要时一次同工况时间细化；不以闭合C1方法标签为前置。它是更贴近完整材料起裂的问题，但并不保证比C1容易。不能用给定预裂演示代替这个最终目标。

### 暂停范围

闭合C1进一步控制器/配额/有限精度/性能优化，FFM虚拟路线、旧径向/全网络路线、新相场/弧长框架、三维、参数扫描及新的通用审查系统均退出当前默认任务。底层依赖保留，防止清理演变为另一轮大工程。

## 6. 本次修改及下一轮边界

本次提交是当前任务、代码地图和模型范围的收束，没有改变数值方程或把旧失败改为通过。新稳定入口为 `research_prototype_task.md` 和 `server_agent_prompt.md`，以后更新这些文件，避免每轮再增加一组同职能任务入口。

下一轮优先交付能用于科研讨论的图表和可复现代表案例。任何新代码必须对应一个具体研究输出或阻断它的真实缺陷；不以代码量减少、测试数增加或新状态字段数量作为成果。

## 依据

- [第十四轮报告](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/78405e242bc4246d269dda5ff5bdf11814b1c0fa/projects/softmodeling/docs/research_status.md)
- [实际步加倍](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/78405e242bc4246d269dda5ff5bdf11814b1c0fa/projects/softmodeling/tools/step_doubling.py)
- [额度与事件控制](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/78405e242bc4246d269dda5ff5bdf11814b1c0fa/projects/softmodeling/tools/event_resolution.py)
- [断裂构造工厂](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/78405e242bc4246d269dda5ff5bdf11814b1c0fa/projects/softmodeling/tools/run_zero_gap_cut.py)
- [耦合模型与依赖](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/78405e242bc4246d269dda5ff5bdf11814b1c0fa/projects/softmodeling/coupled_cohesive.py)
- [原始任务](fracture_agent_task_brief.md)与[研究主题](oblique_cutting_experiment_and_research.md)
- [NASA-STD-7009B §4.1.1、4.2.2](https://standards.nasa.gov/sites/default/files/standards/NASA/B/1/NASA-STD-7009B-Final-3-5-2024.pdf)：借用按用途确定模型范围和验收的原则，不引入NASA工程流程。
- [Fan与Wang，PRE 110, 065001 (2024)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.110.065001)：弹性体慢/快裂速跃迁提示不能由慢加载直接推出无惯性；不据此认证本项目高速值。
- [Why cutting is easier than tearing elastomers (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11968795/)：局部刀具接触影响切割能量和局部场，新裂面接触仍属于最小裂后机制。
