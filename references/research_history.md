# 历次参考资料与采用记录（历史档案）

此文件由原 references/README.md 原样迁入，保留过去的来源、阅读范围和采用/否决理由。
其中“当前”“本轮”按段落日期理解，不能作为今天的运行指令或实现状态。
统一入口见 [README](README.md)，按主题检索见 [catalog](catalog.md)。

2026-10-13暂停审计：[宏混合笔记](macro_mixed_discretization.md#2026-10-13局部体积核查后的适用限制)补充粗压力测试空间与局部体积约束的区别，以及Fu等2025四场有限弹性原文的候选设计与适用限制。本地实际积分/导数和保存态审计见[求解状态](../docs/solver_status.md)，未实现四场替代、未恢复长算，也未将流体inf-sup稳定性当作刀尖局部精度保证。

2026-10-13 下一轮离散验证：[宏单元混合离散的依据与限制](macro_mixed_discretization.md)。保持同 G/K 和完整体物理，先通过材料/混合块/有限变形弯曲与真实接触短程，再决定独立1 mm检查；不把流体混合元的稳定性结论直接套到橡胶刀具接触。


2026-10-13补充：[前裂压入增长与时间响应](prefracture_rate_response.md)。新增原CR有限应变率依赖研究、压入历史积分及前裂近线性/弱速度依赖的反证；区分机制可行、全文/摘要核查范围及材料移植限制。具体短检查见[求解状态](../docs/solver_status.md)，派生实验/计算数据留本地。

2026-10-13本轮执行：联合观测已落实为固定身份/P1传递、区域差分及固定端点弦长的代码与回归。继续依据下述Oddes/Solav的可辨识性、Rokoš的边界误差及Moreno-Mateos/Steinmann的平滑切割转换证据安排模型筛查；不移植论文参数或假定所有加载段未裂。Kojima等的速度试验仅支持时间依赖的竞争机制判据，未新增CR黏弹参数。本轮实际数据/图表只留本地 `outputs/prefracture_behavior_20261013/`，方法说明在[建模路线](../docs/modeling_roadmap.md)。

2026-10-13 实验特征引导模型选择的针对性核查（非系统综述；未下载或发布新PDF）：

- [Oddes、Solav，2023，Identifiability of soft tissue constitutive parameters from in-vivo macro-indentation](https://pubmed.ncbi.nlm.nih.gov/36801779/)，DOI `10.1016/j.jmbbm.2023.105708`。本轮复核原论文摘要：合成轴对称压入中，联合表面位移改善所研究双参数材料模型的可辨识性，但并非全部参数因此可唯一识别。支持力与运动联合筛选；不是本项目CR斜切的验证，不移植其几何或参数。
- [Moreno-Mateos、Steinmann，Cutting soft materials: how material differences shape the response](https://www.nature.com/articles/s41524-025-01869-y)。期刊为 **2026-01-06，npj Computational Materials 12, 15**，DOI含2025不代表期刊年份；2025年预印本另计。本轮读Results、DIC及参数说明：采用力和表面位移对照，实验弹性体转入切割可无明显掉力。可借鉴分阶段、联合观测和机制隔离；其切后低接触压力/黏着结论不能套到本项目完整体压入，也不因采用CZM就重启本地CZM路线。力曲线经过标定的事实须保留，不能把曲线拟合当作所有机制的独立验证。
- [Rokoš、Hoefnagels、Peerlings、Geers，2018，On Micromechanical Parameter Identification With Integrated DIC and the Role of Accuracy in Kinematic Boundary Conditions](https://arxiv.org/abs/1810.08245)，期刊DOI [10.1016/j.ijsolstr.2018.04.004](https://doi.org/10.1016/j.ijsolstr.2018.04.004)。核查作者摘要及正文§1–2：边界位移误差会污染材料识别。用于要求边界注册、误差包络和条件子模型的输入/验证分离；其平面应变微结构虚拟试验不是本项目侧视切割，不能移植误差阈值。
- [Efremov等，2020，Viscoelasticity in simple indentation-cycle experiments](https://www.nature.com/articles/s41598-020-70361-y)，本轮复核曲线指数、速率和滞回的讨论；[Kojima等，2020，刀片压入速度对硅橡胶切割的影响](https://www.jstage.jst.go.jp/article/sosei/61/708/61_1/_article/-char/en)，复核原摘要。支持把真实时间/速度/保持段作为时间依赖的区分证据；前者线性黏弹半空间，后者硅橡胶及不同刀形/支承，均不给当前CR参数，也不能凭单条近线性Fz就断言存在黏弹主导。

可执行建议、观测定义与停止规则见 [建模路线](../docs/modeling_roadmap.md)“实验特征引导模型选择”。检索日2026-10-13；主要检索词为 `indentation full-field displacements force constitutive parameter identifiability`、`rubber cutting digital image correlation blade`、`integrated digital image correlation hyperelastic boundary conditions`，经出版方/作者原始来源核查。未对文献覆盖率作系统性声明。视频派生观察和原始交付能力审查只留本地输出，不公开数据。

2026-10-12～13 曲率成因的针对性资料核查（不是系统综述）：

- [Lubarda 与 Lubarda，二维圆柱压入与位移约束，作者原文§1、§3、§4](https://arxiv.org/html/2309.02361v1)，[2024 年期刊版 DOI](https://doi.org/10.1016/j.euromechsol.2024.105362)：二维线载荷的位移具有对数远场依赖，压入量依赖选定有限位移基准，线弹性也能有非线性力–深度关系。本次读取作者全文，期刊版只核对元数据；不能把小变形圆柱公式移植为有限应变斜切的定量解。其意义是支承/域尺度值得受控检查，而非证明本项目边界已错或“二维必线性”。
- [solids4foam 官方 Neo-Hookean 材料说明](https://www.solids4foam.com/documentation/material-models/neoHookeanElastic.html)：给出 `U=K/4(J²−1−2 ln J)` 的体积能。本地保持同 G/K 的条件平衡使用这一候选，解析增量应力/切线与现有谱核及有限差分交叉核验；实际改善局部 J 而总 Fz 变化很小。资料支持候选形式，不证明该橡胶服从此律，更不将条件解视为完整历史。
- [Efremov、Kotova、Timashev，Scientific Reports 10, 13302 (2020)](https://www.nature.com/articles/s41598-020-70361-y)：压入循环中的黏弹松弛可影响力曲线指数。其线性黏弹、光滑接触假设不能直接用于当前有限应变摩擦刀片；未对现有 Fz 作卷积冒充重算，也未添加任意松弛参数。后续需可靠时间/速率和材料响应约束。
- [Kojima、Nagasawa、Igarashi，JSTP 61(708), 1–6 (2020)](https://www.jstage.jst.go.jp/article/sosei/61/708/61_1/_article/-char/en)：刀片切硅橡胶的速度依赖提供相关机制线索，材料、刀形、支承及裂纹阶段不同，不能移植参数到本批 CR 的完整体标定。

三维圆杆深压入文献仍仅作机制参考：它的二次力尺度不等于二维刀片结论。对二维自相似且应变幅度固定的理想场，本地量纲推导为 `U∝G b d²`、`F∝G b d`；当前接受场不满足该自相似假设。证据和适用限制在本地 `outputs/material_cause_20261012/method_review.md`，实际数值结果另见同目录 `review.md`。

2026-10-12 接触机制核查采用与运行版本一致的官方IPC Toolkit v1.6.0源码：[参考边界权重](https://raw.githubusercontent.com/ipc-sim/ipc-toolkit/v1.6.0/src/ipc/collision_mesh.cpp)、[带符号碰撞补偿](https://raw.githubusercontent.com/ipc-sim/ipc-toolkit/v1.6.0/src/ipc/collisions/normal/normal_collisions_builder.cpp)、[摩擦平滑函数](https://raw.githubusercontent.com/ipc-sim/ipc-toolkit/v1.6.0/src/ipc/friction/smooth_friction_mollifier.cpp)。本地运行态确认切向项保留负权重；当前宿主传入位置增量，平滑参数因此是每步位移尺度。上述实现依据不代表当前离散已收敛。新增无预应力参考态检查源于同几何密采样的本地反例，未擅自删除补偿、自接触或修改受载定律。具体试验与独立审查留在`outputs/contact_mechanism_20261012/`。

2026-10-12 出平面约束诊断：[Ahmadi、McBride、Steinmann与Saxena（2024），*Plane stress finite element modelling of arbitrary compressible hyperelastic materials*](https://arxiv.org/abs/2410.22562)，[作者原文§3.1](https://arxiv.org/html/2410.22562v1)。支持逐积分点求出平面变形变量和一致切线凝聚；有限应变近不可压缩材料不能直接以小应变泊松公式代替。它不证明本项目20 mm厚、固定底面的试样满足平面应力，也不把局部F33场变成相容三维解。本轮仅据此设计PE/PS体能方向的原解灵敏度。α2/二次J的专用根括区由本地应力方程推导并用独立标量公式检验，不是该论文原算法的照搬；原PE生产方程保持不变。

2026-10-12 完整案例后的体积约束离散核查：

- [MOOSE — Volumetric Locking Stabilization](https://mooseframework.inl.gov/modules/solid_mechanics/Stabilization.html)：单元内F-bar修正对线性三角形/四面体无效。当前P1每单元F与J为常数，单元内再平均不能改变体积约束；不能以少积分或同单元P0压力消元冒充稳定混合参考。该资料支持筛除无效改法，不证明当前曲率偏差已确诊为锁死。
- Bonet与Burton（1998），[A simple average nodal pressure tetrahedral element for incompressible and nearly incompressible dynamic explicit applications](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291099-0887%28199805%2914%3A5%3C437%3A%3AAID-CNM162%3E3.0.CO%3B2-W)。采用节点平均压力的原始方法依据；对象是三维四面体和显式动力学，不能直接保证本地二维有限应变准静态IPC稳定。本地有界对照使用体积能的节点投影、剪切尺度方差稳定化和完整跨单元切线，属于项目实现，不能称作论文原样复现或已验证的免锁死单元。
- Bonet、Marriott与Hassan（2001），[An averaged nodal deformation gradient linear tetrahedral element for large strain explicit dynamic applications](https://doi.org/10.1002/cnm.429)：指出早期平均节点压力方法在弯曲主导问题中的短板。仅改变体积能并通过导数检查，不足以排除全部弯曲、局部畸变及接触面离散误差。

本轮方法试验及适用边界记入 [建模路线](../docs/modeling_roadmap.md) 和 [求解状态](../docs/solver_status.md)。论文参数不移植到橡胶标定；短窗口失败也不证明混合有限元路线不可行。

2026-10-12 完整案例前的窄范围路线核查：

- [PETSc SNESNEWTONTR](https://petsc.org/release/manualpages/SNES/SNESNEWTONTR/)给出Newton信赖域、半径收缩/扩张及Cauchy/dogleg后备选项。可作为当前PTC反复拒步时的后备设计依据；本地`residual-merit`只有附加接受判据，并非该完整算法。非对称摩擦切线、CCD裁剪后的预测量和原平衡验收仍须专门处理，文档不证明本例一定收敛。
- [PETSc稳态伪时间](https://petsc.org/release/manual/ts/#solving-steady-state-problems-with-pseudo-timestepping)支持辅助时间推进求稳态的现有路线。保留原真实残差验收，辅助移位不解释成物理黏性；本轮先跑当前模型全程，不为引入新名称而更换框架。
- [IPC主文PDF第10页/印刷49:10页](https://ipc-sim.github.io/file/IPC-paper-350ppi.pdf#page=10)，小标题“Friction direction and magnitude”、式(17)后明确指出lagging没有收敛保证；第19页结论重申限制。[技术补充](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf)第9节只用于冻结摩擦势及其导数公式，不能代替该收敛限制的出处。外层未收敛时，内层最优不等于完全隐式摩擦平衡；因此不将旧lagged-friction外循环当作无条件可靠替代。本仓库此前出现过外循环二周期，需额外保护与全程验证。

以上为选择完整案例策略的针对性资料，不是系统综述或创新性检索。依据、失败认定范围见 [建模路线](../docs/modeling_roadmap.md#2026-10-12完整案例优先与路线退出条件)。

2026-10-11～12 新版方法依据（小型针对性核查，非系统综述）：

- **Microsoft进程调度API**：[SetProcessInformation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation)、[Quality of Service](https://learn.microsoft.com/en-us/windows/win32/procthread/quality-of-service)、[CPU Set Information](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information)。2026-10-12核查：显式控制EXECUTION_SPEED并关闭其节流状态请求HighQoS；不显式控制时由系统启发式选择。EfficiencyClass较高代表较快/较不节能核心。本地固定态A/B/A支持将可恢复的进程策略用于解决此次装配减速；不用理想CPU编号冒充实际驻留、不更改全局电源计划。这是运行工程依据，不是材料或接触理论。
- **PETSc官方非线性求解文档及实现**：[SNESLINESEARCHBT](https://petsc.org/release/manualpages/SNES/SNESLINESEARCHBT/)、[回溯源码](https://petsc.org/main/src/snes/linesearch/impls/bt/linesearchbt.c.html)。2026-10-12以“SNESLINESEARCHBT residual norm Armijo slope Jacobian”核查残差范数充分下降及方向导数。本地借用 `φ=||R||²/2` 的Armijo判据，作为原PTC误差模型拒步后的可选额外接受条件，另要求10%真实残差下降并保持移位；仍保留原终验收和保守后备。此为任务相关的局部改动，不是原文给出本项目全程收敛保证，也不能平滑OGC特征跳变。
- **Chen、Hsu、Liu、Macklin、Yang、Yuksel，Offset Geometric Contact，ACM TOG / SIGGRAPH 2025**，DOI [10.1145/3731205](https://doi.org/10.1145/3731205)，[作者项目页](https://graphics.cs.utah.edu/research/projects/ogc/)、[作者原文](https://graphics.cs.utah.edu/research/projects/ogc/Offset_Geometric_Contact-SIGGRAPH2025.pdf)。其接触几何可消除本地直线压缩合成测试中的相邻端点排斥，且保留折叠自接触；但原文第6节/Fig.22也明确指出凹侧开放几何块边界可能产生力不连续，并可能导致慢收敛。本地当前“OGC集合＋原IPC激活函数＋CCD/PTC”在实际0.01 mm加载中触发了对应的特征切换失效，**不再视为通过准入的生产替代**。这不是论文完整算法复现，也不推翻其全部理论或其它应用结果；不借用布料/GPU速度作为橡胶切割性能。
- **ipctk v1.6.0原生实现**：[邻接可行域](https://raw.githubusercontent.com/ipc-sim/ipc-toolkit/v1.6.0/src/ipc/ogc/feasible_region.cpp)、[碰撞集合构造](https://raw.githubusercontent.com/ipc-sim/ipc-toolkit/v1.6.0/src/ipc/collisions/normal/normal_collisions_builder.cpp)、[官方OGC教程](https://ipctk.xyz/tutorials/ogc.html)。二维内部点边投影和端点可行域采用不同判断。实际失败态跨端点时，OGC删去EV项却无连续VV补偿，自由DOF接触＋摩擦力跳约0.0153301 N，与PTC日志模型误差一致；将两侧扰动减至1e−12 mm仍保留该跳变，同态improved-max变化降至舍入量级。固定特征区内导数通过不能证明切换兼容。官方TrustRegion主要处理防穿透步长，不能据其加入就宣称硬切换已经光滑；本地证据在`outputs/solver_repair_20261012/contact/switch_audit.json`。后续准入必须包含真实特征切换回归，不能仅加迭代预算或任意删除全部自接触。
- **Abaqus官方超弹性说明**：[Hyperelastic Behavior of Rubberlike Materials](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-hyperelastic.htm)。用于核对有限应变超弹性、小应变G/K与试验模式的关系；材料形状选择应由应力–应变数据评估，单一硬度/静态G不决定大应变曲线。本地新增固定G的单项Ogden指数仅为诊断，不移植其它材料参数，也不因模型名成熟就声称已实验验证。

2026-10-11已按用户要求采用现行目录G=.47 MPa，旧冻结基线保留.66；资料版本及批次不确定性仍如本文件2026-10-09核查所述。实现与局限统一见 [建模路线](../docs/modeling_roadmap.md)，数值检查见 [求解状态](../docs/solver_status.md)。

2026-10-08 加载失败后的有界减步：[MOOSE IterationAdaptiveDT](https://mooseframework.inl.gov/source/timesteppers/IterationAdaptiveDT.html)（本轮核查）说明按非线性求解难度调整步长，失败后使用更小步长重试，并可强制经过指定函数点。本项目只借鉴这些控制思想：从上一接受态二分加载增量、重试使用仿射初值、保留原输出深度及所有接受子步，力学方程和验收门限不变；不是 MOOSE 算法复现，也不提供切割全程收敛保证。摩擦历史依赖接受的增量，减步后仍需检查观测的步长敏感性。

本地候选 B 在 0.54 mm 出现 PTC 残差反弹，几何审计未发现反转、穿透或全刀面接触逃出细化窗；仅增加迭代预算缺少依据。**其 0.32～0.53 mm 共同前缀已未通过原定网格精度筛查；减步即使收敛，也不能消除此精度失败。** 方法选择、审计范围与误差门槛见 [建模路线](../docs/modeling_roadmap.md)，实际运行结论见 [求解状态](../docs/solver_status.md)。

2026-10-07 快速 3 mm 候选的依据与限制：

- [MOOSE SimplePredictor](https://mooseframework.inl.gov/source/predictors/SimplePredictor.html)（本轮核查）：利用历史解构造下一次非线性求解的初值，预测幅度考虑新旧步长比。借鉴到本项目的 secant 初值时，只使用前两个已接受态；原仿射候选始终保留，以整段 CCD/正 J、规定边界和固定摩擦历史下的刷新力筛选。候选失败可退回原初值，所有候选不安全则拒步。文档支持预测思想，不证明本地切割更快、接触分支唯一或 3 mm 必然收敛；本地保护不是 MOOSE 算法的原样复现。
- [IPC Convergent Formulation](https://ipctk.xyz/tutorials/convergent.html)（本轮核查）：本地继续保留面积权重、改进最大值近似与 physical barrier，保持 `dhat=0.0025 mm`、原接触/摩擦和真实力验收。它不能替代当前体内网格的精度对照，也不能把较少节点本身当成同精度证明。

候选 B 的 `local/surface/fine_halfwidth = 0.10/0.005/1 mm` 来自本地旧完整细网格的**全刀面**接触审计，不是上述资料给出的通用网格尺度：刀尖偏移平台约 +0.227 mm，但刀面支撑的原表面范围为 −0.478～+0.522 mm，0.05 mm 几何近邻为 −0.747～+0.640 mm。只按刀尖缩窗会漏掉刀面接触；旧路径包络也不保证新网格包络。粗化体内后须以同物理、同 0.01 mm 步长和 1e−4 N 力门限对照；工程筛查为 `|ΔFz|≤max(0.2 N,5%|Fz_ref|)`、有效唯一接触处 `|Δs₀−Δs₀_ref|≤0.02 mm`，不能称为实验验证或网格收敛结论。P1 的 `diagnostics=False` 仅省去求解不用的派生输出，保持能量/力/切线，不支撑全程速度承诺。方案与本地证据位置见 [建模路线](../docs/modeling_roadmap.md)，短测与完整运行结论见 [求解状态](../docs/solver_status.md)。

2026-10-06 面向滑移与标定的开源路线核查：

- [IPC Convergent Formulation](https://ipctk.xyz/tutorials/convergent.html)：面积权重、改进最大值近似与 physical barrier 三项需同时设置；本地均已启用，不把它们当作新增修复。屏障刚度具有压力量纲，独立固定它时改变材料 G 会改变相对数值接触柔度。
- [PolyFEM IPC Quick Start](https://polyfem.github.io/tutorials/ipc_quick_start_guide/ipc_quick_start_guide/)：提供 IPC 摩擦/求解设置，未保证无限摩擦迭代收敛。同家族框架不自动成为当前失效的解药，暂不迁移。
- [GetFEM finite strain](https://getfem.org/userdoc/model_nonlinear_elasticity.html) 与 [large sliding contact](https://getfem.org/userdoc/model_contact_friction_large_sliding.html)：提供 Python 可调用的有限应变和大滑移库仑增广拉格朗日接触，作为唯一独立后备候选。其小滑移 contact bricks 不能替代本问题的大滑移接口；本地尚未适配或验证性能。
- [Oddes & Solav, 2023, *Identifiability of soft tissue constitutive parameters from in-vivo macro-indentation*](https://pubmed.ncbi.nlm.nih.gov/36801779/)，DOI 10.1016/j.jmbbm.2023.105708：合成轴对称压入研究中，单独力–深度不足以可靠识别所研究模型的两个参数，联合表面位移改善识别；并非所有模型都因此唯一。用于本项目设计敏感性/联合观测/留出验证，不移植材料参数，不把其软组织压入直接当斜切验证。

实现决策和未经验证的边界集中在 [建模路线](../docs/modeling_roadmap.md)。用户已允许从截图人工读图，当前只做临时力幅值诊断和候选误差比较；它们不等于材料参数标定或滑移验证，估读数据保留在本地。

2026-10-06 线性求解加速：[SciPy GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html) 明确左预条件只改变求解过程，收敛按原方程 `b−Ax` 验收；显式指定 `callback_type=pr_norm` 时 `maxiter` 为重启循环数。[SciPy splu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html) 提供 CSC 稀疏 LU、COLAMD 排序和重复三角求解。用于本项目同一加载步内以旧 LU 预条件当前切线；真实线性残差、迭代/时间预算及直接分解回退独立保护。资料支持线性代数方法，不证明切割接触路径收敛；不复用旧力或旧切线代替当前方程。实现与适用限制见 [离散对照](../docs/history/topics/element_comparison.md)。

2026-10-06 高阶慢收敛诊断：[PETSc SNESNEWTONLS](https://petsc.org/release/manualpages/SNES/SNESNEWTONLS/) 与 [SNESLINESEARCHBT](https://petsc.org/release/manualpages/SNES/SNESLINESEARCHBT/) 给出以非线性残差范数为目标的 Newton 回溯及充分下降判据。本轮仅将其作为同一保存态的短探测依据；未把 PTC 主求解器替换为 PETSc，也不因一个缩短试步残差下降就认定方案能算完整切割路径。性能修复保持原 PTC 策略、真实力门限及物理设置，独立记录源等价检验。

2026-10-06 高阶对照：[DefElement — Degree 2 Lagrange on a triangle](https://defelement.org/elements/examples/triangle-lagrange-equispaced-2.html) 给出六个节点泛函与二次基函数，核查了本轮所用 P2 插值。本项目为保留原 IPC 接触离散，将外边中点位移消元为端点平均，仅内部边增加自由度；这是项目的受限 P2 诊断空间，并非文献提供的免锁死切割单元。保持完整材料能量，以正权重高阶积分与导数、弯曲、有限压缩基准验证；细节和限制见 [离散对照](../docs/history/topics/element_comparison.md)。该资料与下述 COMSOL 说明不构成本算例已经锁死或已经消除锁死的证明。

2026-10-06 断裂前力曲线审查：[COMSOL Mixed Formulation](https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.101.html) 解释近不可压缩纯位移离散的体积应变误差可产生虚假压力，压低剪切/弯曲柔度，并指出混合空间需稳定性条件；[Using Reduced Integration](https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.029.html) 给出单元类型依赖，不支持给线性三角形简单少积分就宣称免锁死。适用于当前 P1/ν=0.49 的离散风险评估，但不能据此证明本算例曲率偏差已经确诊为锁死。需同物理条件的稳定混合离散/高阶对照。最小 J 和体积能占比单独均不构成锁死的判据。

2026-10-06 接触材料点追踪：[IPC Toolkit — Getting Started](https://ipctk.xyz/tutorials/getting_started.html)（本轮核查）说明二维碰撞网格以边表示，`NormalCollisions` 按激活距离 `dhat` 构造近接触对。本项目据此区分圆刀尖几何接近、最低点接触与无有效接触；对当前材料边投影后，用相同 P1 插值系数回到参考材料坐标。后者是本项目的观测量定义，不是文献给出的裂纹预测算法；`dhat` 不是实验空间分辨率，几何接触区间也不等于压力分布。定义、零点及适用限制见 [刀尖接触材料坐标](../docs/tip_contact.md)。保留原材料、接触力及摩擦方程，数据后处理不构成实验验证。

2026-09-12 伪瞬态延续：[PETSc：Solving Steady-State Problems with Pseudo-Timestepping](https://petsc.org/release/manual/ts/#solving-steady-state-problems-with-pseudo-timestepping)，以辅助演化求原方程零点，支持 B 度量和残差比伪时间步更新；[Gee 等，Pseudo-transient continuation for nonlinear transient elasticity，2009](https://doi.org/10.1002/nme.2527) 给出非线性弹性中的相关应用。用于处理本地含摩擦耦合矩阵近奇异、单调残差搜索停滞，不能据此声称橡胶切割全程稳定或实验动态响应正确。代码用投影 Hessian 作为辅助度量，保留未移位力验收；详情见 modeling_roadmap。

2026-09-12 摩擦外层：[Li 等 2020 IPC 技术补充第 9 节](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf) 与 [Physics-Based Simulation：Fixed-Point Iteration](https://phys-sim-book.github.io/boundary-treatments/frictional-contact/fixed_point_iter.html)。可参考冻结法向力/切向基底的摩擦势与外层刷新流程；内层收敛不保证外层固定点收敛。本地依据失败日志增加重新组装力残差的外层回溯，不改变摩擦系数，也不把冻结摩擦残差作为最终验收。详见建模路线及 solver_status。

2026-09-12 能量差精度：[ANSYS Newton-Raphson 理论手册 14.10.5](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/ans_thry/thy_tool10.html) 描述沿步方向的势能导数线搜索；[A Noise-Tolerant Quasi-Newton Algorithm for Unconstrained Optimization](https://arxiv.org/abs/2010.04352) 讨论数值噪声对线搜索的影响。用于理解近收敛处总能量相减不可靠的原因；本地采用带原力/微小位移/方向功多重门槛的精度保护，不是上述算法复现，不放宽力平衡门限。
2026-09-11 接触非凸方向补充：[Gast 等，Optimization Integrator for Large Time Steps，2015](https://math.ucdavis.edu/~jteran/papers/GSSJT15.pdf)，DOI 10.1109/TVCG.2015.2459687，第 3.1 节关于不定 Newton 方向、截断 CG 和下降角度；[Cornell CS4220，2026-04-20](https://www.cs.cornell.edu/courses/cs4220/2026sp/lec/2026-04-20.html) 关于非凸二次模型、移位 Hessian 与信赖域。用于理解 0.77 mm 接触主导的负曲率；本地 `regularized` 为 PN 度量下的全局曲率移位，不是上述动力学或完整信赖域方法复现，尚无完整路径稳定性保证。

2026-09-11 求解器核查：[Fernández-Fernández、Löschner、Bender，Progressively Projected Newton's Method](https://arxiv.org/abs/2505.21013v2)，初版 2025、修订 2026。可借鉴结论是逐单元正定投影可能过度改变全局切线；论文同时报告准静态下 PN 可能更优，因此必须做本问题的同前态对照。本地 `--hessian-mode hybrid` 只试验未投影 Newton 方向加 PSD 回退，不声称复现 PPN。切线接口依据 [IPC Toolkit 官方 API](https://ipctk.xyz/python-api/potentials.html)，沿用锁定 ipctk 1.6.0，并以有限差分核对。

2026-09-11 实现补充：本轮采用 [IPC Toolkit 官方 Python 接口](https://ipctk.xyz/build/python.html) 的 `ipctk==1.6.0` Windows 原生 wheel。依据 [模拟教程](https://ipctk.xyz/tutorials/simulation.html) 组织表面/体自由度映射、屏障能量和 CCD；本地静态宿主使用原有限应变材料，未加入惯性项，不是教程动力学案例的直接复现。原生 `make_static_obstacle_filter` 排除刀具内部碰撞并保留橡胶自接触，避免 Python 回调阻塞并行 CCD。API 以锁定版本的绑定签名为准，不能直接套用在线开发版。方法适用边界及参数敏感性要求见 [建模路线](../docs/modeling_roadmap.md)。

论文原文件仅留在本地，不上传 GitHub。面向 AI 的结构化摘要、原文定位、来源链接、适用边界和代码映射集中在 [method_notes.md](method_notes.md)。这些笔记是自行归纳的研究说明，不是论文全文或作者认证的实现规范。

2026-10-09 橡胶/刀具来源核查：检索 `CB245NE 静的せん断弾性率`、`CB245NE 0.66`、`99755 フェザー`。[Maxell Kureha 现行CR目录](https://www.kurehae.maxell.co.jp/products/pdf/2_gousei.pdf) 与 [厂家CB245NE试验报告](https://www.kurehae.maxell.co.jp/dl/test_report/CB245NE.pdf) 给出静态G=0.47 MPa、拉伸强度5.6 MPa、断裂伸长340%；试验报告核查全页，标注23°C，数据不是保证规格，不是本批橡胶报告。[经销商保存的厂家旧目录](https://www.taiyo-kasei.co.jp/img/work/parts/pdf-1.pdf) 第一页则为0.66 MPa / 4.4 MPa / 320%；版本差异不能自行解释成配方何时改变。[供货方软质CR页面](https://gomusuke.com/products/detail/1103) 允许CB245NE或国内同等品，故采购名称及硬度不能唯一识别实际牌号。上述资料不提供完整应力–应变曲线、泊松比或刀具界面摩擦，不能替代有限应变标定。原始PDF和检查图只保存在本地输出目录。

99755数据以用户本次提供的规格表为直接依据：碳素工具钢、0.245 mm厚、18.4×39.3 mm外形、23°双面磨刃及涂装。[Feather 官方工业产品目录](https://www.feather.co.jp/i_Products/list.html/1000) 本次未检出99755；不能用其它型号的刃角、涂层或图片冒充99755。表未给刃口半径及涂层化学组成，外形尺寸不自动等于参与受力的刃长。

材料曲线校准方法参考 [Anssari-Benam、Bucchi、Destrade、Saccomandi：The generalised Mooney space for modelling the response of rubber-like materials](https://arxiv.org/abs/2403.01223)，期刊论文2022、作者预印本2024，DOI [10.1007/s10659-022-09889-1](https://doi.org/10.1007/s10659-022-09889-1)。本次核查摘要与元数据：可支持本构选择需按张拉、剪切等试验响应验证及非线性拟合的可辨识性审查；不提供本项目CR材料参数，也不是刀具压入求解或本例曲率原因的直接证据。本次未声称复现全文方法。

同次检索补充 [Shojaeifard 等：Deep Indentation of Soft Solids: A Limit State，arXiv:2606.30971v4，2026-09-19](https://arxiv.org/abs/2606.30971v4)。本次仅核查摘要/版本：研究圆杆压头、不可压缩Ogden半空间的深压入，报告深区力的二次尺度及其材料依赖。它说明凸曲线本身不等于求解错误；但轴对称圆杆几何不同于本任务的平面应变薄刃斜切，不能移植二次尺度、半径阈值或摩擦结论来证明当前曲线正确。保留为后续机制参考，未复现也未据此改模型。

| 文献编号 | 用途 |
|---|---|
| R01 — Spagnoli 2019 | 斜切运动、摩擦及能量分解 |
| R02 — Dammaß 2021 | 小应变黏弹性与相场的双向耦合 |
| R03 — Dammaß 2023 | 有限变形材料、相场、二维应力假设及独立复现 |
| R04 — Moreno-Mateos / Steinmann | 软材料切割的体材料、黏附与界面机制 |
| R05 — Pinto Carvalho 2022 | 刚体—可变形体无摩擦 dual mortar |
| R06 — Khandelwal 2022 | 二次边界接触自由度及误差分析 |
| R07 — Maas 2011 | 节点积分、稳定化和锁死权衡 |
| R08 — Acary / Brémond / Huber | 离散库仑摩擦求解器及尺度比较 |
| R09 — Lopez-Pamies 2025 | 独立强度与无缺口起裂预测 |
| R10 — Kamarei 2025 | 黏弹性、广义强度面与断裂能耦合候选 |
| R11 — Lopez-Pamies 2026 | 受强度约束的能量最小化预印本 |
| R12 — 接触全局化资料 | 平滑延续、互补函数及 Siconos 分裂求解的适用边界 |
| R13 — Deuflhard / PETSc | 修正量线搜索及本地失败边界 |
| R14 — Hohlfeld / Mahadevan 2011 | 局部折皱与分支问题；不能替代本实验断裂判据 |
| R15 — Hassani / Hild 等；HiGHS | 摩擦多解与全节点有界分支可行性诊断 |
| R16 — Qi / Sun 1998 | 半光滑 Newton 的局部条件与非光滑残差全局化 |
| R17 — Li 等 2023 | 当前自写方案退出后，优先采用完整 PolyFEM/IPC 流程；二维线性碰撞几何、平滑摩擦和材料适配另验 |
| R18 — Giacoma 等 2016 | LATIN 局部—整体摩擦耦合设计 |
| R19 — Both 等 2022 | 借鉴软材料多孔介质的稳定化分裂 |
| R20 — Degroote / preCICE | 流固界面加速及内层收敛要求 |
| R21 — Acary 等 2026 | 已核查全文；二维内点核心已开始实现，有限应变外层与 ANM 边界另验 |
| R22 — Gfrerer 等 2022 | 直接处理多值接触图的 SCD Newton 候选 |
| R23 — PETSc | 保持接触代数约束的伪瞬态延续候选 |
| R24 — Dickson 等 2007 | 伪弧长预测—校正与简单折点；非光滑适用限制 |
| R25 — Jong 2012 | 同伦路径到达性的假设；数值近端同伦候选与适用限制 |

求解器实现另参考 [FEBio v4.13](https://github.com/febiosoftware/FEBio/tree/v4.13)、[SuperLU_MT](https://github.com/xiaoyeli/superlu_mt/tree/8324379)、[FEBio 节点—面接触](https://febiosoftware.github.io/febio-feature-manual/features/solid_surfaceinteraction_sliding-node-on-facet/) 及 [GetFEM 大滑移](https://getfem.org/userdoc/model_contact_friction_large_sliding.html)。这些是实现依据，不是本实验验证。

[Zimmerman–Ateshian 2018](https://doi.org/10.1115/1.4040497) 及 [作者模型库](https://repo.febio.org/permalink/project/68) 可作为后续独立大变形摩擦基准；本轮未取得该论文全文，不声称复现其算例。[检索记录](../docs/literature_search_log.md) 保留查询和证据范围；[模型规范](../docs/current_model_spec.md) 的 CB245NE 参数不被论文替换。
