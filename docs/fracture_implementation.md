# 起裂诊断与二维裂后能力：实现选择、证据与限制

日期：2026-10-28。任务依据为 [用户任务书](../researchtopic/fracture_agent_task_brief.md)。原文保留；本文说明实际采用的路线和尚未完成的研究验证。

## 1. 对任务书的判断

任务书把“局部指标越阈”“耦合损伤”“形成裂口”“力曲线转折”分开，并要求裂后材料分离、刀具继续进入、接触与载荷重分配，这些要求合理。本次保留真实时间、接受态提交、失败回退、能量分账和独立数值审查的约束。

有三处必须收紧：

1. **已有实验起裂标签需要重新核实。** 本地找到的是历史近似、分析截止及未确认事件。当前派生标签目录共 7 条，全部 `unused`；可用于物理起裂标定的已核实标签为 **0 条**。不能把 0° a1 的约 1 mm 估计移给 1008n0a2，也不能把 1014n33a1 视频截止移给 1008n33a1 力试次。57° 的 2.2 mm 分析截止不是测得的物理起裂。
2. **主应力阈值只作为可检验的事件假设。** 它没有裂纹耗能和裂后承载律。平面应变仍有面外应力；本次把截面内候选方向与三维危险指标分别保存，不把面外危险方向包装成二维裂纹预测。
3. **首版耦合能力限定为 0°、预设参考接缝、径向张开。** 未直接实现一般混合模式 CZM 或相场。有限参考间距使两张接缝边界从开始就能进入现有完整 IPC/CCD；代价是参考体包含数值空隙，且径向律不具备独立的切向强度。这是明确受限的机制实现，不是完整无缺陷橡胶的定量起裂理论。

当前选择不同于直接移植 Moreno-Mateos 等的切割模型：该文使用增广拉格朗日障碍接触及界面伪黏性；本工程保留现有 IPC 与物理时间 Maxwell 历史，不复制其伪时间阻尼或其他材料的参数。[R1]

## 2. 原有基础与本次分层架构

原模型为 P1 三角形、平面应变、有限变形等容 Maxwell 本构，接触采用 IPC、Coulomb 摩擦与连续碰撞检测。加载速度仍为 0.2 mm/s，研究行程上限仍为 2 mm。完整体基线、旧结果和原始实验数据保留。

| 层次 | 实际入口与职责 | 结果含义 |
|---|---|---|
| A：完整体起裂候选诊断 | `tools.fracture_onset`；复用 `workbench.results.saved_state_fields` | 从已接受、已审查状态恢复应力和历史；不插裂、不继续加载 |
| B：裂后机制计算 | `prototype.cohesive`、`prototype.cohesive_cut`、`tools.run_cohesive_cut` | 0° 有限参考间距、预设路径、径向不可逆内聚键及实际边界接触 |
| C：研究工作台 | `softcut_solution.api`、统一 CLI、桌面“研究资产”页 | 用户显式运行；独立输出、解释器、预算、输入哈希及日志 |
| 实验事件目录 | `tools.fracture_labels` 与 `configs/fracture_labels.example.json` | 保存试次、事件定义、深度参考、证据及标定/验证划分；不会自动认证来源 |

A 与 B 是两个不同模型层次。A 的示例应力阈值 **1 MPa** 不等于 B 的示例内聚强度 **0.2 MPa**，二者也未建立经过验证的参数映射。不能将两者不同的事件深度解释为一次共同标定的预测。

## 3. A 层：恢复状态与寻找首次事件

每个状态使用自己的 `material_metric`，不重置黏弹历史。由节点坐标恢复三维平面应变变形梯度，满足 F33=1。令 A 为保存的等容正定内部度量，B=FFᵀ、Be=FAFᵀ，则本实现恢复的 Cauchy 应力为

$$
\boldsymbol\sigma=J^{-5/3}\left[G_e\operatorname{dev}(B)+G_v\operatorname{dev}(B_e)\right]+p(J)I.
$$

体积压力沿用该算例的体积能定义；不替换本构。输出包括 F、完整 Cauchy 张量、主应力/方向、主伸长、J、平衡与非平衡储能。历史中已有的黏性耗散和摩擦功另列。状态 SHA256 必须匹配原审查，恢复的积分储能与最小 J 必须重现接受记录。

在参考单元中心上采用体积加权紧支撑核：

$$
q_\ell(X_i,d)=\frac{\sum_j w_{ij} V_{0j}\langle\lambda_{\max}(\sigma_{j,\mathrm{inplane}})\rangle_+}{\sum_j w_{ij}V_{0j}},\qquad
w_{ij}=\left(1-\frac{\lVert X_i-X_j\rVert}{\ell}\right)^2\quad(\lVert X_i-X_j\rVert<\ell).
$$

统一面外宽度在归一化中抵消。自由边界处截断邻域后重新归一化；这一处理本身属于模型假设。ℓ 是固定物理邻域尺度，不能与网格、刀尖半径或 IPC 屏障尺度混用。

默认搜索域为参考刀尖周围半径 3 mm，排除固定底部上方 0.3 mm 范围；同时保留全体最大值及其是否落在该域外。该搜索域不是以拟合误差挑选的区域，也不证明真实裂纹必在域内。报告另列三维最大正主应力平均量和面外正应力平均量。保存的局部法向是获选单元的特征方向，不是“非局部平均裂纹法向”。

从整个连续合格前缀中找首次越阈，报告跨越的两步和两步原始 Fz，不把插值点当成求解平衡态。无阈值、初态已超阈、合格区间内未触发及资格中断分别保留。基线第 0 态可能有预载，不能把它自动当未变形零力状态。

## 4. A 层已完成的运行证据

三角度完整体 baseline 各 **201 个状态**、合计 **603 个状态**已处理。统一使用 ℓ=0.15 mm、示例阈值 1 MPa，所得结果如下。

| 倾角 | 首次采样越阈行程区间 / mm | 对应原始 Fz 区间 / N | 状态索引 |
|---|---:|---:|---:|
| 0° | 0.89–0.90 | 18.8746–19.0617 | 89–90 |
| 33° | 1.03–1.04 | 21.5467–21.7218 | 103–104 |
| 57° | 0.61–0.62 | 11.0547–11.2137 | 61–62 |

正式整合入口证据：[多角度报告](../runs/20261027T154005Z_fracture_onset_4290c2cf/result/index.html)、[机器摘要](../runs/20261027T154005Z_fracture_onset_4290c2cf/result/onset_summary.json)。该运行包含 603 个状态和标签证据记录；这些是忽略于 Git 的本地资产。目录时间戳由运行器生成，本文日期采用研究工作日 2026-10-28。

以上是**共同示例阈值下的候选事件**，没有实验标定、留出预测或起裂准确度结论。各子目录的 `onset.json`、`onset.csv` 保存完整历史与来源；`fields_*.npz` 保存诊断快照及首次越阈状态的局部场。查看起裂附近场时按事件 `state_indices` 定位；不能把最终状态场误称起裂场。

首次越阈态的候选位置局部最大主伸长约为 **3.26 / 5.37 / 6.02**（按 0° / 33° / 57° 顺序）；对应全体网格最小 J 约为 **0.284 / 0.125 / 0.547**。这些大伸长和强局部压缩没有通过独立材料实验验证；数学上 J>0 不等于材料响应可靠。邻域尺度敏感性本轮尚未完成，因而不得把上述排序或阈值解释为已成立的物理规律。

## 5. B 层：有限参考间距的径向不可逆内聚律

研究几何保留 30×5 mm 截面、面外宽度 20 mm、底部固定、0°、刀具竖直加载；接缝位于参考坐标 `drop_s=7 mm`。两侧独立剖分，潜在切割面从初始化即为实际碰撞边界。当前示例配置见 [fracture_demo.json](../projects/softmodeling/configs/fracture_demo.json)。

设成对界面节点当前距离 r、参考距离 r0，定义本构张开量 δ=max(r−r0,0)。r0 不计入初始损伤或黏聚预拉。参考间距当前为 **0.006 mm**，IPC 距离尺度为 **0.0025 mm**，要求间距大于两倍 IPC 距离尺度，避免初始相互屏障。完整 IPC 碰撞集合及 CCD 同时包含两张接缝面，不通过屏蔽这些面来让刀具穿过。压缩闭合由 IPC 负责，内聚律不重复增加压缩罚力。

$$
\delta_0=T_c/K,\qquad \delta_f=2G_c/T_c,\qquad
\kappa=\max(\kappa_{\mathrm{accepted}},\delta),
$$

$$
d(\kappa)=\begin{cases}
0,&\kappa\le\delta_0,\\
\dfrac{\delta_f(\kappa-\delta_0)}{\kappa(\delta_f-\delta_0)},&\delta_0<\kappa<\delta_f,\\
1,&\kappa\ge\delta_f.
\end{cases}
\qquad T=(1-d)K\delta,\quad \psi_s=\tfrac12(1-d)K\delta^2.
$$

历史最大张开决定损伤，卸载时保持既有损伤。单调拉开到完全分离的单位参考面积功为 Gc；当前存储能与累计耗散分开。面积为参考接缝节点的分担长度乘面外宽度，只乘一次；Gc 已是形成裂纹的总界面耗能，不再为两张裂面额外乘 2。节点力沿当前连线，切线包含距离变化与方向变化，因此具有刚体转动客观性。

示例 Tc=0.2 MPa、K=50 MPa/mm、Gc=0.006 N/mm，故 δ0=0.004 mm、δf=0.06 mm。所有这些参数都标为 `demonstration_only_not_experimentally_calibrated`。当前目标行程为 0.6 mm、名义增量 0.01 mm；目标只指定加载范围，不按该行程触发损伤。

**能力限制：**这不是零厚度完整试样；参考空隙和额外界面柔度会改变切前响应。径向键在参考态的线性剪切刚度为零，没有独立切向强度、一般混合模式耗能或自由路径选择；0° 也因 30 mm 试样的落点为 7 mm 而不是全局对称问题。平面内检查不能验证沿刀刃方向的三维起裂位置。损伤关闭仅保持同一接缝模型的弹性键；其切前响应与原完整体的实测数值对照见下节，不能只用“旧求解器仍能运行”代替该检查。

## 6. 事件、历史与能量记录

| 事件/状态 | 当前操作性定义 | 解释边界 |
|---|---|---|
| A 层候选事件 | 首次 qℓ≥指定阈值的接受步区间 | 完整体后处理，不是耦合损伤 |
| B 层首次损伤 | 最大键损伤大于 10⁻⁸ | 数值事件阈值，不等于实验可见裂纹 |
| 首个完全分离键 | 键损伤达到 1−10⁻¹⁰ | 单个离散键失效，不自动证明连续宏观裂口 |
| 裂后继续切入 | 分离后仍有新增接受行程，并检查接缝面—刀具接触和变形 | 需实际结果证据，不能从函数存在推断 |
| 实验力趋势转折 | 与实验同类、事先定义的识别方法 | 当前没有已核实同试次标签和统一识别结果，尚未完成 |

每个试探更新均从上一接受步的黏弹与内聚历史出发，非线性迭代不会重复推进物理时间；失败步不得提交损伤。保存态包括前后内聚最大张开与材料度量，审查通过新建问题重算残量、J、接触及历史一致性。

结果分别记录外功、平衡/非平衡储能变化、内聚储能变化、黏性耗散、摩擦功及断裂耗散。当前外功采用接受步梯形积分。实际代表算例在准静态跳跃处发现显著能量余项，且已核查 IPC 屏障贡献很小；**能量验证没有通过，不能把余项含糊归因于接触屏障或普通时间积分误差**。人工稳定化若引入必须单列；不能把上述耗散或累计外功直接再当成额外断裂驱动力。[R2–R3]

## 7. B 层代表运行结果与验收状态

实际分离、刀具继续进入和载荷重分配已经在示例中出现；**定量裂后力学验证未完成，当前不支持实验预测**。最终结果为 [representative 报告](../projects/softmodeling/outputs/cohesive_20261028/representative/index.html) 和 [result.json](../projects/softmodeling/outputs/cohesive_20261028/representative/result.json)，另有[本任务总览](../runs/fracture_20261028/index.html)和[对照总页](../projects/softmodeling/outputs/cohesive_20261028/index.html)。

| 项目 | 代表运行证据 | 可支持的结论 |
|---|---|---|
| 求解范围 | 0–0.6 mm，61 个接受态 | 运行完成；不是预设行程触发开裂 |
| 首次损伤 | 0.22–0.23 mm，发生态 Fz=6.88935 N | 示例内聚律的损伤事件 |
| 首个完全分离键 | 0.30–0.31 mm，发生态 Fz=8.23010 N | 一个离散键达到完全分离 |
| 后续切入 | 继续接受至 0.6 mm；5 个连续完全分离键（46–50），与上表面连通的几何裂口长度 0.40 mm；离散分担长度总和 0.45 mm | 两种长度定义不同，都不是独立测得的实验裂纹长度 |
| 刀具—失效裂面接触 | 0.34–0.35 mm 间首次接触表面连通裂口；末态 7 个 EV 接触均映射到 damage=1 且表面连通的裂面节点 | 接触位置与失效区域同时核对；不是把未损伤接缝接触也当裂后证据 |
| 载荷重分配 | 0.34→0.35 mm，Fz 从约 8.35485 降至 2.62502 N；0.6 mm 为约 3.14294 N | 准静态解发生跳跃；不能直接当作已验证的实验力转折 |
| 独立接受态审查 | 61 态重算通过；最大残差 9.44716×10⁻⁵ N，最小 J=0.853618，几何、Maxwell 与内聚历史通过 | 说明这些离散状态及历史满足所检条件，不证明过渡过程能量正确 |

**关键失败证据：**0.34→0.35 mm 的准静态跳跃产生约 **0.498716 N·mm** 的未扣屏障单步能量余项；全程未扣屏障余项约 **0.488284 N·mm**，占外功 **19.86%**。最终逐步账目单列 IPC 屏障变化后，单步未计能量为 **0.501071 N·mm**，全程为 **0.487769 N·mm**；屏障不能解释差额。[步长减半算例](../projects/softmodeling/outputs/cohesive_20261028/step_005/index.html) 以 0.005 mm 加载至 0.6 mm、共 121 个接受态，全程未扣屏障相对余项仍约 **19.83%**。这表明仅把名义步长减半并未解决跳跃能量缺口。

因此，程序贯通和逐态平衡已经获得证据，但现有准静态跳跃没有建立足够的能量审查。下一步必须检验惯性动力学、能够计量耗散的稳定化或适合不稳定分支的路径跟踪，并判断哪种路线对应实际实验过程；不能以任意增大阻尼掩盖缺口。缺实验标定不是当前唯一限制，**模型的跳跃过渡处理本身也是定量预测阻塞项**。

已完成针对性对照（每次从自身初态开始）：

| 变化 | 首次损伤区间 / mm | 首次完全分离区间 / mm | 0.6 mm Fz / N |
|---|---:|---:|---:|
| 代表：间隙 0.006、接缝间距 0.10、加载步 0.01 mm | 0.22–0.23 | 0.30–0.31 | 3.14294 |
| 参考间隙 0.008 mm | 0.22–0.23 | 0.27–0.28 | 3.13530 |
| 加载步 0.005 mm | 0.225–0.230 | 0.305–0.310 | 3.15076 |
| 接缝网格间距 0.05 mm | 0.16–0.17 | 0.30–0.31 | 3.14634 |
| 关闭损伤，保持有限间隙和弹性键 | 无损伤 | 无分离 | 15.22116 |

接缝加密使首次损伤上界从 0.23 降至 0.17 mm，**空间收敛未通过**。间隙变化也显著移动完全分离事件。原完整体控制采用相同体节点采样参数和刀具离散，加载至 0.2 mm：Fz=6.33693 N；同深度无损伤接缝 Fz=6.06417 N，低 **4.30%**。两者网格拓扑不同，浅层 0.05–0.20 mm 差约 4.3–7.5%；这揭示界面近似误差，不能称恢复原完整体。尚未完成界面刚度、IPC 距离尺度、刀尖和整体体网格的系统收敛研究。

可复用测试覆盖局部峰值/耗能/卸载、参考间距不计入损伤、刚体转动与切线、接缝边界接触/CCD、失败不提交历史，以及审查失败时不发布合格事件。FE 全套本轮实测为 **279 passed、364 subtests passed**（37.48 s；一条 NumPy `trapz` 弃用提示），包含原完整体、Maxwell、接触与工作台回归；这不是重算三条 2 mm 基线。之后补充了禁止裂后历史导出为完整体标准包的保护，导出/诊断相关 **22 项**回归通过。整合层 **119 项**通过，覆盖启动不求解、预算与路径限制、命令参数、结果发现及耦合裂后历史不进入完整体候选列表。这些软件检查不能抵消上述能量和数值尺度失败。

实际从统一入口完成 A 层三角度后处理，并用 60 s 总预算运行 0.02 mm、3 态裂后入口贯通检查，见[作业记录](../runs/20261027T154900Z_fracture_run_e439ed74/job.json)。后者仅验证启动—求解—审查—报告链路，不是裂后代表案例；真实分离证据来自上面的 0.6 mm 代表运行。

## 8. 实际运行入口

以下统一命令均在总仓库根目录执行。默认不会因启动界面、刷新资产或加载配置开始计算。

```powershell
# 不指定阈值：输出指标，事件状态为 missing_threshold。
python -m softcut_solution fracture-onset --case projects/softmodeling/outputs/baseline/angle0 --length-mm 0.15 --seconds 300

# 共享示例阈值诊断；不能称实验标定。
python -m softcut_solution fracture-onset --case projects/softmodeling/outputs/baseline/angle0 --case projects/softmodeling/outputs/baseline/angle33 --case projects/softmodeling/outputs/baseline/angle57 --length-mm 0.15 --threshold-mpa 1 --seconds 300

# 显式启动裂后计算。
python -m softcut_solution fracture-run --config projects/softmodeling/configs/fracture_demo.json --seconds 300

python -m softcut_solution gui
python -m pytest -q
```

统一层读取 `settings.local.json` 的 `simulation_python`，以复制 FE 工程为工作目录。每次创建新的 `runs/<唯一作业>/`，外层保存 `job.json`、`process.log`、`artifact.json`，子工具输出到 `result/`。预算允许 1–3600 秒；外层监督超时为预算加 60 秒。计算失败保留已产生的日志及证据，不将退出状态当作物理验证。

桌面“研究资产”页选择配置后点击“运行裂后算例”；选择保存算例、邻域尺度及可选阈值后点击“诊断起裂指标”。完成后可打开本次报告，或从资产列表打开已有报告。

需要直接调用子工程时，在 `projects/softmodeling/` 中使用已配置 FE Python：

```powershell
python -m tools.fracture_labels configs/fracture_labels.example.json
python -m tools.fracture_onset --case outputs/baseline/angle0 --output outputs/fracture/onset_new --length-mm 0.15 --seconds 300
python -m tools.run_cohesive_cut --config configs/fracture_demo.json --output outputs/fracture/demo_new --seconds 300
```

输出目录须独立于源算例；重复运行使用新目录。统一入口自动处理唯一目录，直接调用需自行指定。示例标签文件不是可直接标定的数据集。

## 9. 下一步科研优先级

1. **建立同试次、同深度参考的实验事件区间。** 保存原始力趋势识别算法/参数、起裂前后原始帧及证据；分别标注物理起裂、可见张开、力转折。不能只补一个没有来源的“起裂深度”。
2. **先检验完整体和接缝切前响应。** 刀尖半径、材料黏弹参数、接触、尺度与时钟仍影响局部指标。数值网格细化不能补偿这些参数未标定。
3. **先解决跳跃能量缺口，再做有限目标的数值细化。** 目前步长减半后余项仍约 20%，应检验动力学、可计量耗散的稳定化或路径跟踪。随后在首次损伤与早期分离窗口分别改变空间网格、参考间距、IPC 距离尺度和内聚刚度，检查力、事件、分离长度、接触和能量余项。
4. **完成强度与韧性的可识别性分析。** 有核实标签后先固定材料/接触，再识别少量共享断裂参数，按试次或角度留出；不得逐角度自由调阈值。已有数据被用来选择参数后不能再叫独立验证。
5. **依据证据升级混合模式、自由路径或三维。** 若切向滑移、路径偏转或沿刀刃起裂位置是研究核心，再扩展相应方法。相场需单独解决强度、长度尺度、裂面开闭承载与刀具接触；削弱体单元不自动产生 IPC 裂面。[R4–R6]

科研价值应落在经验证的机制和预测范围，而非“有裂纹动画”。程序贯通、局部力学验证、实验参数标定和跨工况预测必须持续分开报告。

## 10. 原始文献与本次阅读范围

- **R1 — Moreno-Mateos, M. A.; Steinmann, P. (2026).** *Cutting soft materials: how material differences shape the response*. npj Computational Materials 12, 15. [正式全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12789027/)；[DOI](https://doi.org/10.1038/s41524-025-01869-y)。阅读层级：正式全文与 2025 作者稿相关方法；用于架构、功共轭、接触及伪黏性限制，不移植材料参数。
- **R2 — Kamarei, F.; Breedlove, E.; Lopez-Pamies, O. (2025).** *Nucleation and propagation of fracture in viscoelastic elastomers: A complete phase-field theory*. CMAME 446, 118337. [原始 PDF](https://pamies.cee.illinois.edu/assets/pdf/CMAME2025b.pdf)；[DOI](https://doi.org/10.1016/j.cma.2025.118337)。阅读层级：理论及适用范围相关章节；用于区分黏弹、强度、断裂能，未复现其相场实现。
- **R3 — Dammaß, F.; Kalina, K. A.; Ambati, M.; Kästner, M. (2023).** *Phase-field modelling and analysis of rate-dependent fracture phenomena at finite deformation*. Computational Mechanics 72, 859–883. [正式全文及 DOI](https://link.springer.com/article/10.1007/s00466-023-02310-1)。阅读层级：有限变形、驱动力和不可逆性相关章节。
- **R4 — Hakimzadeh, M.; Agrawal, V.; Dayal, K.; Mora-Corral, C.** *Phase-Field Finite Deformation Fracture with an Effective Energy for Regularized Crack Face Contact*. [作者稿](https://arxiv.org/abs/2206.15386)；[正式 DOI](https://doi.org/10.1016/j.jmps.2022.104994)。阅读层级：作者稿的裂面开闭与方向相关能量构造；不将其误作 IPC 拓扑转换算法。
- **R5 — Kristensen, P. K.; Niordson, C. F.; Martínez-Pañeda, E. (2021).** *An assessment of phase field fracture: crack initiation and growth*. Philosophical Transactions of the Royal Society A 379, 20210021. [作者稿](https://arxiv.org/abs/2103.05443)；[正式 DOI](https://doi.org/10.1098/rsta.2021.0021)。阅读层级：2022 修订作者稿相关模型和起裂/长度尺度讨论；2021 同时为初稿和正式出版年份，不能因作者稿修订而改写正式年份。
- **R6 — Li, M. et al. (2020).** *Incremental Potential Contact: Intersection- and Inversion-free Large-Deformation Dynamics*. ACM TOG 39(4), 49. [作者项目页](https://ipc-sim.github.io/)；[原始论文](https://web.uvic.ca/~teseo/profile/publications/ipc/2020-IPC.pdf)。阅读层级：屏障、可行间距与连续碰撞相关章节；零距离重合界面不直接作为普通屏障接触初态。
- **R7 — Satake, U.; Senda, R.; Sambe, R.; Enomoto, T. (2025).** *Investigation of biological tissue cutting for minimal tissue damage using finite element simulation*. Procedia CIRP, 495–500. [DOI](https://doi.org/10.1016/j.procir.2025.02.085)。阅读层级：已核实出版记录及摘要，未核读完整正文；只作为切前变形与切入的背景依据，不采用其参数或具体判据。
