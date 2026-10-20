# 橡胶斜切建模：论文方法笔记

2026-10-13整理说明：R01–R25保留稳定编号；以下代码关联与路线选择多为2026-09历史记录，不能当作当前实现清单。现在默认是完整体P1/IPC，P2与混合配对属于独立诊断，断裂关闭。完整主题目录及新增资料见 [catalog](catalog.md)，当前状态见 [solver_status](../docs/solver_status.md)。

类型：面向后续 AI/开发者的研究工作笔记；2026-09-10 由 AI 整理，未经过论文作者或人工逐式复核。不是论文全文转录，也不替代原文。仅上传自行组织的摘要、必要公式、书目链接和实现判断；不上传 PDF、图片、读图曲线或实验/计算数据。

阅读规则：`Rxx` 为稳定文献标识；“论文范围”来自本地对应版本，“项目判断”是本仓库推论，不能写成作者已验证的结论。页码为本地 PDF 页序，从 1 开始。查公式时回到对应版本原文；不以摘要中的省略式直接编程。工况与 CB245NE 参数以 [模型规范](../docs/current_model_spec.md) 为准。

## R01 — Spagnoli 等，2019：斜切、摩擦与能量分解

- 文献：A. Spagnoli, R. Brighenti, M. Terzano, F. Artoni, *Cutting resistance of soft materials: Effects of blade inclination and friction*，Theoretical and Applied Fracture Mechanics 101, 200–206。[DOI](https://doi.org/10.1016/j.tafmec.2019.02.017)。本地 `2.pdf`。
- 核查位置：第 1 页摘要；第 6 页 Conclusions。对象为软弹性薄板的压入/切割实验及能量模型，特别讨论完全贯穿后的稳态阶段。
- 论文范围：分析推切与切片运动的组合如何改变耗散及所需力；其试验/模型用恒定黏附剪应力描述摩擦，而不是认定所有橡胶均服从同一压力依赖库仑参数。
- 项目判断：可借鉴功的分解与运动方向定义；先统一角度、参考面积、力和位移投影，再讨论“斜切省力”。不能把论文刀具倾角直接当成本项目工件倾角，更不能由稳态力反推首次起裂必在刀尖正下方。
- 代码关联：`prototype/loading.py`、`contact.py`；目前未据该文把库仑模型改成恒定剪应力，黏附参数仍未标定。

## R02 — Dammaß、Ambati、Kästner，2021：黏弹性与相场双向耦合

- 文献：*A unified phase-field model of fracture in viscoelastic materials*，Continuum Mechanics and Thermodynamics 33, 1907–1929。[DOI](https://doi.org/10.1007/s00161-021-01013-3)。本地同名 PDF。
- 核查位置：第 1 页摘要，第 3 页第 2 节明确限定 small strain，第 21 页 Conclusion and outlook。
- 论文范围：由含断裂耗散的伪能量泛函构造相场与黏性内变量的双向耦合；不同选择可退化为已有模型。是否让与黏性耗散有关的贡献驱动断裂，会影响蠕变失效等预测。
- 项目判断：适合学习能量记账与耦合假设，不应直接把该文小应变应力/应变方程用于本项目大转动、深压入。有限变形材料迁移应优先参考 R03。
- 代码关联：`mechanics/finite_strain.py` 与论文复现的能量审计；试探迭代不能重复累计物理耗散或提交内变量。

## R03 — Dammaß 等，2023：有限变形、黏弹性及相场

- 文献：F. Dammaß, K. A. Kalina, M. Ambati, M. Kästner, *Phase-field modelling and analysis of rate-dependent fracture phenomena at finite deformation*，Computational Mechanics 72, 859–883。[DOI](https://doi.org/10.1007/s00466-023-02310-1)。本地 `updated phase.pdf`。
- 核查位置：第 1 页摘要；第 5–9 页能量与应力；第 11 页 Algorithm 1；第 12 页图 5/材料标定；第 14 页图 7–8/表 3。完整复现实现说明见 [论文基准](../reproductions/dammass2023/README.md)。
- 论文范围：Ogden 型有限黏弹性材料，平衡/非平衡储能及黏性机制，与裂纹相场耦合；比较不同裂纹驱动力以及速率相关韧度。示例材料是 EPDM，不能拿其参数覆盖 CB245NE。
- 方法框架：以 `F = F_el F_vi` 表述非平衡支路的乘法分解；退化函数控制刚度，长度尺度控制弥散裂纹宽度，`Gc` 控制表面能。论文裂纹密度形式为 `gamma = d²/(4 ell_c) + ell_c |Grad d|²`，符号及参考构形须与其完整泛函一致。
- 实现判断：先求收敛的有限应变应力与一致切线，再与有不可逆约束的相场交替求解。物理历史只从上一接受步出发；失败减步不积累额外黏性历史。残余刚度是数值参数，不能拿来拟合力曲线。
- 二维厚度：有限变形材料可嵌入平面应变 `F33=1` 并按厚度积分；这属于本项目改造。论文平面应力算法需要解出平面伸长，不能在平面应变中照搬。厚度放大反力不等于证明实验真的出平面均匀。
- 边界：论文预缺口裂纹传播及速率效应不等于无缺口萌生、受压裂面与刀具进入新裂面的验证。主线目前只复用材料框架，尚无自由断裂。当前实现的平面应力条件解释及裂后能量偏差见复现说明，不隐去差异。
- 代码关联：`mechanics/finite_strain.py`、`reproductions/dammass2023/{material,solver,energy_audit}.py`；主线候选筛查不能称为已经实现本文相场。

## R04 — Moreno-Mateos、Steinmann：材料与界面机制共同影响切割

- 文献：*Cutting soft materials: how material differences shape the response*。[2025 预印本](https://arxiv.org/abs/2507.13565)、[期刊 DOI](https://doi.org/10.1038/s41524-025-01869-y)。本地文件为 2025 预印本，不能因期刊链接存在就假定与正式版逐字一致。
- 核查位置：本地第 1 页摘要，第 13 页 Conclusion。论文结合不同软材料切割实验及体大变形、界面断裂、黏附和摩擦模型。
- 论文范围：所研究材料表现出不同的压入到切割过渡；其配置中黏附和黏性 cohesive 机制可主导切向作用。文中关于刀下中央起裂及库仑项大小的结论有具体工况范围。
- 项目判断：可借鉴体材料—界面—刀具耦合和机制消融对照；不能把其中央起裂结论设为 N60A 的先验裂面，也不能未经标定照搬黏附、摩擦或黏性参数。模型必须允许全域竞争起裂位置。
- 代码关联：旧预设 cohesive 求解器已移除；当前 `initiation.py` 只作完整试样候选筛查，最终新裂面接触仍待开发。

## R05 — Pinto Carvalho 等，2022：刚体障碍的 dual mortar

- 文献：R. Pinto Carvalho 等，*An efficient algorithm for rigid/deformable contact interaction based on the dual mortar method*。[DOI](https://doi.org/10.1007/s00466-022-02226-2)。
- 核查位置：本地第 1 页摘要，第 21 页 Conclusions。
- 论文范围：有限变形、无摩擦、刚体—可变形体接触。修改加权间隙以保留离散间隙符号，并通过节点运动标架降低几何耦合；对二次插值采用保留几何信息的分片线性处理。
- 项目判断：很适合参考刚性刀具约束和二次边界离散，但当前干摩擦的客观切向滑移、库仑锥、一致切线和历史映射必须另外推导。平均间隙合格也不能自动排除节点之间穿透。
- 代码关联：`surface_contact.py`、`quadratic.py`；本仓库边界积分不是已复现的 dual mortar，不继承论文的完整数值证明。

## R06 — Khandelwal、Porwal、Wadhawan，2022：二次接触离散

- 文献：*Adaptive quadratic finite element method for the unilateral contact problem*。[作者预印本](https://arxiv.org/abs/2209.04906)。
- 核查位置：本地第 1 页摘要；第 7 页第 3 节、式 (3.2) 的可行位移集合含接触边顶点和中点。
- 论文范围：线性弹性无摩擦单边接触的二次有限元和后验误差估计，讨论可靠性与效率。
- 项目判断：有助于检查二次边界位移自由度与接触力空间是否匹配，不能只约束端点而忽略中点。有限应变、库仑摩擦、曲面刀具和新裂面不在该分析的直接保证范围。
- 代码关联：`quadratic.py`、`surface_contact.py` 及曲边穿透、虚功、切线回归。当前细化指示并非直接实现了该文完整后验估计器。

## R07 — Maas 等，2011：节点积分、稳定化与锁死

- 文献：*Implementation and Verification of a Nodally-Integrated Tetrahedral Element in FEBio*，技术报告 UUSCI-2011-007。[机构原文](https://publications.sci.utah.edu/publications/SCITechReports/UUSCI-2011-007.pdf)。
- 核查位置：第 1 页摘要、第 9 页 Conclusions。
- 论文范围：三维 UT4 节点积分四面体，比较对全部虚功或仅等容部分作稳定化；收敛依赖问题、网格与参数，要求针对应用验证网格。
- 项目判断：支持把锁死和软模式放在同一组对照中检查。不能把 UT4 等容稳定化参数直接搬到二维节点体积平均，或仅因反力变软就认定应力更准确。
- 历史代码关联：`prototype/material.py` 的 NodalVolume 是本仓库独立能量形式；本条撰写时曾采用 P2 完整积分。现主线为 P1/IPC，节点体积与高阶配对仅作独立比较。

## R08 — Acary、Brémond、Huber：离散摩擦求解器比较

- 文献：作者稿题名 *On solving frictional contact problems: formulations and comparisons of numerical methods*。[作者稿](https://tripop.inrialpes.fr/people/acary/publications/Acary_Bremond_Huber.pdf)。2018 章节书目见 [HAL](https://hal.science/hal-01878539)，题名为 *On solving contact problems with Coulomb friction: formulations and numerical comparisons*。
- 核查位置：本地第 1 页摘要；第 72 页总结。
- 论文范围：离散单边接触与三维库仑摩擦，比较变分不等式、非光滑方程、锥互补及优化等形式和多类算法；没有普适最优求解器，投影尺度等参数会影响效率。
- 项目判断：失效比较必须固定物理历史、残差定义、尺度与测试集合。优化器成功标记不是接触平衡；仅换等价函数或放宽容差不能算修复。该离散问题比较不自动证明本项目有限变形刚度与接触切线正确。
- 代码关联：`contact.py`、`contact_branches.py`、`audit_contact.py`。分支探索为独立实现，不是作者整套算法的复现。

## R09 — Lopez-Pamies 等，2025：强度必须独立进入萌生预测

- 文献：O. Lopez-Pamies, J. E. Dolbow, G. A. Francfort, C. J. Larsen, *Classical variational phase-field models cannot predict fracture nucleation*。[DOI](https://doi.org/10.1016/j.cma.2024.117520)。
- 核查位置：本地第 1 页摘要及后续强度面比较。该文讨论准静态弹性脆性材料，结论针对经典变分相场作为一般萌生描述的局限。
- 论文范围：强调材料强度作为独立宏观性质的必要性；仅弹性能、断裂能和正则化长度不能普遍描述不同多轴应力下的起裂。
- 项目判断：不能通过调 ell 迎合 N60A 起裂位移后再宣称盲预测。单轴名义强度阈值只提供候选，缺少多轴数据时不能补造强度面；也不能把该文解读为所有扩展相场均不适用。
- 代码关联：`fracture_criteria.py`、`initiation_event.py`；真实裂纹演化仍未实现。

## R10 — Kamarei、Breedlove、Lopez-Pamies，2025：完整黏弹性相场候选

- 文献：*Nucleation and propagation of fracture in viscoelastic elastomers: A complete phase-field theory*。[本地对应 v1](https://arxiv.org/abs/2506.16479v1)。
- 核查位置：本地第 1–3 页摘要与路线说明；具体离散需进一步核查第 4 节。
- 论文范围：同时引入黏弹性、广义强度面及断裂能；未知量包含变形、黏性内变量和相场。其数值方案使用非相容 Crouzeix–Raviart 空间及高阶显式 Runge–Kutta 时间离散。
- 项目判断：比只给经典相场加刀具更贴近无缺口起裂目标，是后续重点候选。不能把“显式”理解为可以忽略准静态平衡，或未经稳定步长、能量与耗散检查就得到任意大步推进。文中方法尚未与本项目摩擦刀具及新裂面接触耦合验证。
- 代码关联：可沿 `mechanics/finite_strain.py`、`fracture_criteria.py` 扩展；当前未实现该文的完整 PDE/ODE 系统和时间算法。

## R11 — Lopez-Pamies 等，2026：受强度约束的能量最小化

- 文献：O. Lopez-Pamies, F. Kamarei, G. A. Francfort, A. Giacomini, *Nucleation and propagation of brittle fracture as a constrained energy minimization problem*。[预印本](https://arxiv.org/abs/2608.28326)。截至核查仍按预印本线索使用。
- 核查位置：本地第 1 页摘要；第 31 页 Final comments；书目与作者页面交叉核对。
- 论文范围：准静态单调载荷下，裂纹只能在强度面超限区域萌生/扩展，同时由势能与表面能的最小化控制演化；展示弹性脆性硬/软材料挑战问题。
- 项目判断：适合研究“强度筛选允许区域、能量决定演化”的路线，但不能把筛选本身当作裂纹解。黏弹性、非保守摩擦和裂面开闭仍需一致扩展；大内存也不能代替数学/物理约束。
- 代码关联：目前 `initiation_event.py` 只有候选事件定位；不是此受约束断裂理论的实现。

## R12 — 接触全局化资料，2026-09-10 核查

- Areias、Pinto da Costa、Rabczuk、César de Sá，*A simple and robust Coulomb frictional algorithm based on 3 additional degrees-of-freedom and smoothing*，Finite Elements in Analysis and Design 167 (2019), 103321，[出版页与 DOI](https://doi.org/10.1016/j.finel.2019.103321)。核查范围为出版方摘要/Highlights，全文请求返回 403，未声称完成全文复现。文献提出同一时间增量内逐次减少平滑参数，并包含惯性和二维/三维验证。可借鉴延续思想；其动态基准不是本仓库准静态橡胶斜切的稳定性证明。
- 本地自主试算用紧支撑 C1 正部函数平滑摩擦投影，保持物理增量历史固定、最终检查原始库仑方程。0.01 / 0.5 MPa 两个初始辅助宽度在有界预算内均未通过旧 0.5 mm；试验实现已撤除，不增加未标定黏性或更换摩擦系数。
- [Siconos 官方摩擦接触文档](https://nonsmooth.gricad-pages.univ-grenoble-alpes.fr/siconos/users_guide/problems_and_solvers/friction_contact.html) 列出二维 NSGS、Lemke 和枚举等算法，并将标准凝聚问题的矩阵表述为对称半正定。当前有限应变/随动摩擦凝聚矩阵未满足该假设；局部 Gauss–Seidel 试算发散不能归咎于工具本身，也不能直接将该框架的有效性移植过来。
- Fischer–Burmeister 的原理参考 [Jiang 1999](https://doi.org/10.1287/moor.24.3.529) 的出版方摘要，以及 [GAMS NLPEC 官方说明](https://www.gams.com/latest/docs/S_NLPEC.html) 对非负互补条件的解释。自主尝试法向 FB 与切向嵌套标量 FB 的等价重写，仍保留原始接触验收；有界试算未通过，未并入主线。未检验理论的全部假设，不引用“全局收敛”标题作为本问题已获保证的依据。

这些资料支持比较不同数值方法，**不能替代局部根因核查、连续加载验收或实验验证**。具体失败证据只留本地，状态集中维护在 docs/solver_status.md。

## 面向主线的实施顺序（项目判断）

1. **接触可信性先行**：实际力函数的切线差分、固定历史回退、逐步力平衡、法切向投影、整边穿透和正 J；先用解析 patch，再连续跨过已知失效区。
2. **体离散独立对照**：保持相同连续材料能量，比较 P2 完整积分与三场 Hex8；同一材料点检查压力和主应力，不能仅比较总反力。
3. **起裂位置竞争**：全域搜索强度和伸长，区分坐标构形、采样点与平均量；候选不限定于刀尖下方。
4. **断裂演化扩展**：优先核对 R03 的材料/相场耦合与 R09–R11 的强度要求，再补充受压裂面、新裂面刀具接触和能量账。缺失标定须显式保留，不能调参掩盖数值失效。

本文件不提供论文原始曲线或可用于拟合的实验数据，也不声称已经完成上述方法的综合验证。

## R13 — Deuflhard / PETSc：以修正量控制 Newton 步长

来源：[PETSc NLEQERR 说明](https://petsc.org/main/manualpages/SNES/SNESLINESEARCHNLEQERR/)、[实现源码](https://petsc.org/main/src/snes/linesearch/impls/nleqerr/linesearchnleqerr.c.html)，及其所引 Deuflhard (2011), *Newton Methods for Nonlinear Problems*, DOI [10.1007/978-3-642-23899-4](https://doi.org/10.1007/978-3-642-23899-4)。2026-09-10 阅读官方算法说明及源码；未声称读完专著。

核心思想：用当前 Jacobian 解试点残差，检查简化 Newton 修正是否收缩，避免仅凭混合残差的量纲和权重决定步长。库仑接触不光滑，不能直接继承光滑问题的收敛保证。本地只试验这一收缩准则，包含全部开闭接触行，位移与乘子按罚刚度换算成长度尺度；没有移植完整 PETSc 算法，也没有改变最终物理验收。在固定 0.49875→0.5 mm 增量上，步长缩到接近浮点分辨率而失败；该实现已撤除，不能称为有效修复。

## R14 — Hohlfeld / Mahadevan 2011：软体表面局部折皱

来源：*Unfolding the Sulcus*, Physical Review Letters 106, 105702，DOI [10.1103/PhysRevLett.106.105702](https://doi.org/10.1103/PhysRevLett.106.105702)；[作者托管的正文和补充资料](https://softmath.seas.harvard.edu/wp-content/uploads/2019/10/2011-06.pdf) 第 3–6 页为论文正文，前两页是另一作者的评论。正文研究不可压 Neo-Hookean 弯曲条带的局部表面失稳，使用表面弯曲正则化、分级网格、约束接触和分支延续研究尺度趋零的极限。

适用价值：区分折皱、断裂和离散软模态。其自由表面临界压缩不能移植到受刀具压力、摩擦和有限可压缩性影响的 CB245NE 工况；论文也指出法向应力和可压缩性会改变临界值。当前不添加未经标定的表面硬层，不预设折皱或裂纹位置。只有在收敛平衡状态和网格对照上确认分支问题后，才考虑有物理或极限依据的正则化、分支跟踪或瞬态处理。

## R15 — 摩擦多解与有界分支诊断

Hassani、Hild、Ionescu、Sakki，*A mixed finite element method and solution multiplicity for Coulomb frictional contact*，[作者全文](https://www.math.univ-toulouse.fr/~phild/Publications/CMAME3.pdf)。核查摘要、引言及问题定义：论文用两个接触乘子的混合离散与特征问题研究摩擦系数变化时的多解。它支持区分体刚度谱和接触约束下的分支；线弹性分析不证明本项目有限应变刀具问题有多解或不存在解，也不提供本项目可直接使用的临界摩擦系数。

本项目自主增加全节点线性化可行性检查：将 gap/slip 对接触力的凝聚关系与二维库仑四种分支写成混合整数线性约束。[SciPy milp 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) 说明其 HiGHS 后端、时间预算和返回状态。二维切向力为标量，库仑锥是精确线性不等式；三维圆锥不能直接照搬此编码。有限压力/运动范围只用于诊断，不更改实际材料或验收。超时不等于不可行；即使返回候选，也独立检查原分支条件，且不自动接受非线性时间步。小型耦合系统用独立全枚举核对，另测试固定错误分支、压力上界排除解和优化器误报成功。该诊断位于 contact_branches.py，由 audit_contact 调用，不是论文算法复现或新的生产求解器。

## R16 — 半光滑 Newton 与单点切换障碍

Liqun Qi、Defeng Sun，*A Survey of Some Nonsmooth Equations and Smoothing Newton Methods*，1998 年修订稿，[作者全文](https://defengwebsite.github.io/files/qsreview1.pdf)。读取第 2–4 页广义 Jacobian、定理 2.1/2.2 和全局化讨论：局部超线性结论有根点、半光滑性及相应广义导数非奇异等前提；残差平方一般仍可不光滑，不能直接继承光滑 Newton 的全局化结论。B-导数集合与其凸包的非奇异条件不同，不能混为一谈。

项目自主诊断：在单个接触切换面 h=0 上固定其他分支，两侧 Jacobian 只差一行。用行替换公式计算行列式比及可能的奇异凸组合，分别解两侧 Newton 方程，再按方向真正进入的一侧计算残差范数方向导数，并用原非线性方程单边差分复核。以 H=(|x|+1,y) 的非根尖点作解析回归，并用普通光滑方程作反例。此处发现的是失败试值附近的方向不相容，未证明根不存在，也不否定其他半光滑算法；不从一个奇异凸组合推断所有 B-导数都奇异。代码在 audit_contact.py，未将论文局部定理冒充切割任务的全路径保证。

## R17 — Li 等 2023：Convergent Incremental Potential Contact

2026-09-11 最终核查后的用途更新：停止自写接触方案的生产推进，优先评估采用已集成 IPC 的完整 PolyFEM 流程。官方 [Toolkit 概览](https://ipctk.xyz/) 明确它本身不含物理/求解器，推荐 PolyFEM 作为完整宿主；[摩擦势源码](https://github.com/polyfem/polyfem/blob/main/src/polyfem/solver/forms/FrictionForm.cpp) 包含滞后接触集合及摩擦势导数更新。不能将其等同于精确、无平滑的库仑锥内点法。

[官方几何教程](https://ipctk.xyz/tutorials/getting_started.html) 支持二维边，但所用碰撞基本元是线性的；现有 P2 曲边需兼容映射或充分细化，并单独检查碰撞近似。Convergent 形式使用面积权重及重复接触项修正，[实现说明](https://ipctk.xyz/tutorials/convergent.html) 提供启用方式。选择依据是改变局部非穿透的实施和完整求解组织；不是对当前橡胶斜切必能计算的证明。平面应变材料等价、20 mm 宽度、屏障/摩擦平滑参数趋零、准静态性及自由断裂仍是本项目的独立适配责任。

R17–R20 的本地应用：已在 `prototype.contact_partitioned` 实现冻结摩擦阈值/切向框架、非线性体与法向接触、固定点消失的数值稳定化和受保护 Anderson。推导边界见 [建模路线](../docs/modeling_roadmap.md)，有界反例见 [求解状态](../docs/solver_status.md)。它尚未跨过旧失败点，也没有移植这些论文的完整算法或收敛证明。

来源：*Convergent Incremental Potential Contact*，[作者预印本](https://arxiv.org/abs/2307.15908)、[全文](https://arxiv.org/html/2307.15908)。2026-09-11 核查第 3、6、7 节及第 9.3–9.4 节基准。

方法：接触屏障、连续碰撞检测限制步长、摩擦耗散势；内层固定法向力与切向框架求解，外层更新至一致。投影能量 Hessian 用于生成方向，不替换能量梯度。有限平滑参数允许微小滑移，须做参数趋零检查。

项目判断：优先借鉴真正收敛的内外迭代；完整迁移属于独立替代实现。当前 P2 曲边、积分乘子及滑移虚功必须重新匹配，不能直接嫁接线性边界碰撞检测。此前只滞后几何切线或平滑投影残差的试验，不等于此算法。论文未验证本项目切割工况。

## R18 — Giacoma / Dureisseix / Gravouil 2016：LATIN 局部—整体分解

来源：*An efficient quasi-optimal space-time PGD application to frictional contact mechanics*，[全文及 DOI](https://doi.org/10.1186/s40323-016-0067-7)。核查摩擦求解部分的局部、整体阶段及搜索方向。

方法：交替满足整体力学可容许条件和局部摩擦关系，通过带界面刚度的搜索方向连接。本文的 PGD 压缩不是迁移所必需；固定正定整体矩阵的优势依赖其线弹性设置。

项目判断：可借鉴 Robin 型界面条件，把局部接触更新与完整非线性体平衡分开。必须重推有限应变稳定化与功共轭关系，不能宣称仍可只分解一次矩阵。此前冻结凝聚矩阵上的 GS 失败，不排除这种非线性迭代，也不证明它会成功。

## R19 — Both 等 2022：软材料多孔介质的稳定化分裂

来源：J. W. Both、N. A. Barnafi、F. A. Radu、P. Zunino、A. Quarteroni，*Iterative splitting schemes for a soft material poromechanics model*，CMAME 388, 114183，[DOI](https://doi.org/10.1016/j.cma.2021.114183)、[作者全文](https://arxiv.org/pdf/2011.13296)。核查第 2–5 节、6.2 节及表 8–9。

方法：对线性化软材料固液模型设计交替最小化和块对角稳定化，再用 Anderson 加速。表 8 中高体积模量下普通交替法加速后仍失败，而合适的稳定化分裂可收敛；不能把加速器当作所有不稳定分裂的补救。

项目推演：借用“先分析耦合块，再设计收敛时消失的稳定化”的原则。橡胶接触没有该文的孔压、渗透率方程，不能照搬系数或收敛定理。接触近似 Schur 响应还可能非对称、不定，需在本项目重新验证。

## R20 — Degroote 等 2010 / preCICE：界面准牛顿加速

来源：*Performance of partitioned procedures in fluid–structure interaction*，[DOI](https://doi.org/10.1016/j.compstruc.2009.12.006)；[preCICE 官方加速说明](https://precice.org/configuration-acceleration)。论文核查出版摘要，算法配置及限制核查官方全文。

方法：IQN-ILS 用界面迭代历史近似逆 Jacobian，与 Anderson 加速相关；QR 过滤移除近线性相关历史。外层隐式耦合仍要求内层参与问题正确收敛。

项目推演：可在现有 Python 进程内实现小规模界面加速，不必接入整个 preCICE。力/位移须缩放并保持功共轭；加速候选须受接触可行性与原残差检查约束。接触集合剧变时丢弃不适用的数值加速历史，绝不清空物理摩擦历史。它不能修复未收敛的体平衡。

## R21 — Acary / Armand / Nguyen 2026：内点法与路径跟踪

来源：V. Acary、P. Armand、H. M. Nguyen，*Solving contact problems with Coulomb friction using an interior point algorithm and path following techniques*，CMAME 461 (2026), 119227，[DOI](https://doi.org/10.1016/j.cma.2026.119227)。2026-09-11 用户补充本地 `solvingcontactproblemswithCoulombFriction.pdf`（25 页），现已核查全文模型、算法、证明条件和结果，不再限于摘要。首页明确 Available online 29 July 2026；这解决了此前在线日期未核实的问题。原 PDF 仅留本地，SHA256：`94705fefe78f6f8c5df21e19bd81a48a0fb137c967d09fe5785264c8ae857f13`。

**原文定位与主要结果。** 第 4–6 页式 (16)–(24) 用 de Saxcé 修正保持库仑非关联性和无剪胀条件，再写成锥互补系统；第 8–10 页证明扰动解和趋向原解的路径存在。第 10–11 页 Algorithm 1 仍使用半光滑 Newton，保持变量在锥内部并逐步减小互补扰动，并没有彻底消除非光滑性。第 15–19 页、图 13–14 给出普通 IPM 停滞反例：某段中心路径必须允许屏障参数暂时增加；ANM 用辅助弧长和高阶级数跟踪该路径。它与实际加载位移无关。

第 20 页表 3 在论文自身的 `tol=1e-12` 下报告：

| 每个子问题的接触数 | 子问题数 | 普通 IPM 失败 | IPM-ANM 失败 |
|---|---:|---:|---:|
| 1–10 | 83,576 | 68 | 0 |
| 11–100 | 3,279 | 33 | 0 |
| 101–1000 | 339 | 82 | 0 |
| 1001–5840 | 403 | 382 | 132 |

这是 FCLIB 三维离散接触子问题的结果，主要来自刚体集合/颗粒，不是橡胶切割验证。第 12 页先拆出真正独立的连通分量；不能把同一橡胶体上的不同接触点任意分开。其残差和单位未与本项目 N、MPa 验收统一，不能直接照抄 `1e-12`。第 19 页使用 25 阶级数和 Python/SciPy SuperLU；每段一次分解、多个回代，但 ANM 通常比单次 IPM 昂贵。第 22 页把高效的 IPM/ANM 自适应切换列为后续工作，不能称为本文已完成的成熟混合策略。

**关键适用条件。** 第 4 页式 (20) 假设力学矩阵 `M` 对称正定，故第 6 页 `W=H M^{-1} H^T` 半正定；接触矩阵 `H` 可以秩亏。要求正定的是力学块 `M`，不是完整耦合 Jacobian，论文式 (35) 的整体 Jacobian 本来就可以非对称。第 6 页 Assumption 1 还要求 Slater 严格可行性。第 10 页定理只保证存在合适的路径，既不保证路径唯一，也不保证任意初值都能到达它；第 11 页明确实际算法没有收敛保证。论文允许来自非线性材料的线性化问题，不能因此认为任意不定有限应变切线都满足这些条件。当前代码的凝聚关系为 `-C K^{-1} B`，带符号、面积权重及移动框架导数，不能直接认作论文的 `W`。

**本项目的使用判断。** 提升为有条件的短基准候选。先验证二维锥、摩擦系数缩放、接触力/位移功共轭及原方程恢复，再审计当前子问题能否写成满足正定性与严格可行性的形式。二维库仑锥可写为 `r_n±r_t/μ≥0`（μ>0）；仍须保留 de Saxcé 修正，不能换成有剪胀的关联锥投影。保留有限应变外层，以一致的冻结框架和正定近端模型构造内层是自主改造设想，稳定化应在固定点消失，最后验原残差。不得通过对称化整块耦合矩阵或删去物理项冒充满足论文假设。原方程的单点切换障碍与论文中心路径折返是不同对象，尚未证明本项目也有这种中心路径。

**照抄公式的风险与独立代数核查。** 已渲染并目视核对下列位置，问题确实出现在 PDF 排版中，不只是文本抽取误差；以下修正是本项目推导，不是作者勘误：

- 第 10 页式 (34) 的载荷号与式 (20)/(22) 及式 (38) 不一致。实现从选定的力平衡定义推导残差和导数，不混用符号。
- 第 18 页式 (41) 把切向范数乘积写进声称双线性的 `Q(x,y)`。在 `x=y` 时能给出范数平方，但计算不同阶的交叉系数时应使用切向内积：取 `(1,0)` 和 `(-1,0)`，范数乘积为 1，内积为 -1。这会实际改变 ANM 递推，不能机械照抄。
- 第 24 页式 (48) 左侧写成减去常量初始残差，右侧却按路径参数缩放。为同时满足初始点和保持零参数处的原目标，应使用 `R_tilde(x,λ)=R(x,λ)-(λ/λ0)R(x0,λ0)`。初始残差必须是向量，不能使用 Algorithm 3 所记的标量范数去修改各方程。简单 `R=x²-λ` 代数检查确认常量平移会改变零参数目标。
- 第 19 页 Algorithm 2 的一阶号、范数平方与附录式 (49)–(52) 需统一；第 20 页的“导数小于阈值”不能直接把所有负导数当作平坦路径停止。采用一致的切向定向和绝对值退化检查，并由解析路径校核。
- 第 21 页承认平方范数方程收敛不代表未平方摩擦方程达到同等精度。必须另验 `s≥0` 和 `s-|u_t|`；黏着处 `s=u_t=0` 会让平方方程梯度退化，不能只凭平滑多项式残差宣布成功。

建议先做二维单接触开闭/黏滑解析基准、已知弧长转折基准和原残差校核，再决定是否实现完整 ANM。现已按此顺序实现独立残差同伦的伪弧长预测—校正短测，见 R24；它不是本论文的中心路径、内点法或 ANM 复现。实际运行结果集中于求解状态页，不把辅助路径点当作物理加载状态。

**2026-09-11 重构实现。** 新增二维内点核心，不复现全文 ANM。按本仓库压力 p、阻止正滑移的剪力 t、侵入为正的 gap 和滑移 v，令 `r=(p−t/μ,p+t/μ)`、`s=(−gap+μ(|v|+v),−gap+μ(|v|−v))`。严格内点中解 `r*s=η`，独立 slack 方程和体平衡一起校正；η 趋零时恢复 Signorini 与库仑关系。比如 v>0 滑动时 p=t/μ、gap=0，不能删掉 μ|v| 导致剪胀。采用不可行初始等式残差、锥内比例步、预测—校正二阶交叉项，并允许中心 Newton 回退；仍含 |v| 的半光滑导数，没有普遍全局收敛保证。

冻结框架后，现有积分接触满足 `B=J.T diag(weights)`，其中 J 的行是 gap/slip 位移导数；代码在每次构造时核对。单位仍是 N、mm、MPa 和参考面积。子问题使用正定体步模型，外层每次重新计算完整有限应变与移动接触，最终只验原方程。数值主元与 Gershgorin 检查处理的是子问题矩阵，不证明实际平衡稳定或 Slater 可行性。完整参考模型、旧失败增量和路径仍分别验收。

极简对照采用 μ=0 的单法向互补系统 `p≥0, −gap≥0, p·(−gap)=0`；不把含 1/μ 的二维锥公式直接代入零。其法向摩擦分离可对照 [GetFEM 接触条件说明](https://getfem.org/userdoc/model_contact_friction.html)，实现仍为本项目代码。另核查 [Abaqus/Explicit 准静态说明](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-m-Quasi-sb.htm)：显式时间推进是复杂接触的另一种算法组织，但转用它仍需检查惯性及能量，不能把不再进行静态 Newton 迭代当成真实准静态结果。当前未实现显式动力学。

## R22 — Gfrerer 等 2022：SCD semismooth* Newton

来源：H. Gfrerer、M. Mandlmayr、J. V. Outrata、J. Valdman，*On the SCD semismooth\* Newton method for generalized equations with application to a class of static contact problems with Coulomb friction*，[作者全文](https://arxiv.org/pdf/2205.15129)。核查引言、第 6 节模型与近似步骤、第 7 节算例。

方法：直接对多值接触图构造近似点和导数子空间，区别于对 Alart 投影残差选一侧导数。论文的接触应用具有正定弹性矩阵，局部收敛有正则性条件。

项目判断：适合作为保留精确库仑关系的第二种算法研究，但移动曲面、积分接触自由度及非线性体项需重新推导。R16 的非根切换障碍既不否定这种方法，也不证明它必能绕过障碍；当前代码不是本论文的实现。

## R23 — PETSc：微分—代数伪瞬态延续

来源：[TSPSEUDO 官方说明](https://petsc.org/main/manualpages/TS/TSPSEUDO/)，及其引用的 Kelley–Keyes 1998、Coffey–Kelley–Keyes 2002 论文；本轮只核查官方说明，不声称阅读全文。

方法与项目推演：自适应伪时间可在位移方程引入随迭代变化的稳定项，接触乘子仍保持代数约束。它不同于此前固定对角刚度扰动；伪时间不推进实验时间或摩擦历史，最终必须检查原静力残差及稳定项消失。它是内层候选工具，不提供平衡不存在时仍能得到静态解的保证。

## R24 — Dickson / Kelley / Ipsen / Kevrekidis 2007：伪弧长与简单折点

来源：*Condition Estimates for Pseudo-Arclength Continuation*, SIAM J. Numer. Anal. 45(1), 263–276，[DOI](https://doi.org/10.1137/060654384)、[作者全文](https://ipsen.math.ncsu.edu/ps/sinum65438.pdf)。核查第 1–4 页的假设、切向归一化、增广系统与预测—校正算法。

论文说明：以弧长参数化状态及原参数，可在简单折点处保持增广 Jacobian 非奇异；普通参数接续在原状态 Jacobian 奇异处可能失败。其理论讨论光滑路径和简单折点，不能直接为库仑切换处的非光滑路径提供收敛保证。

本项目实现 `H(x,a)=R(x)-(1-a)R(x0)`。这里 R 为按原验收单位缩放的完整体平衡/接触投影向量；初始 a=0 精确过 x0，a=1 恢复原方程。增广导数为 `[R_x, R(x0)]` 加上一行固定前点切向超平面约束。采用固定列缩放、稀疏直接解、预测—校正及步长回退；切向按连续方向定向，允许 a 变为负值或暂时下降。该路径不是实验载荷路径，所有物理前态、深度、时间步保持固定；所有接触乘子参与求解，避免初始开离点被永久移除。

验证先用双折点三次曲线、带两个转角的分段线性曲线及有限压缩解析解。到达辅助终点仍需独立核查原力平衡、摩擦锥、耗散和整边穿透；这些基准通过不保证斜切失败点可被跨越。

针对当前非光滑系统，另加自主的事件处理：同时解辅助方程与最早的法向开闭或黏滑边界方程，以单侧 Jacobian 和进入目标分支的符号重建切向，不要求跨尖锐转角后仍与旧切向同向。法向事件用 `φ=λn+kn·gap=0`；开闭改变法向返回导数，也改变滑动摩擦容量的导数，因此同时更新两条接触方程行。只抑制刚被接受的同一点同一事件，允许稍后重新接触。已用新旧切向点积为负的分段线性曲线检查方向，再核对实际曲面接触的开离、重接触和黏滑边界梯度及两侧差分。法向和摩擦同时到达锥顶的多分支切向尚未实现，明确拒绝而不静默选择。该改造不是上述光滑路径定理的推论，也不是 R21 的 ANM 算法。

## R25 — Jong 2012：同伦路径到达目标的条件

来源：*A Newton-Fixed Point Homotopy Algorithm For Nonlinear Complementarity Problems With Generalized Monotonicity*，[作者预印本](https://optimization-online.org/wp-content/uploads/2012/08/3562.pdf)。核查第 2 节定理 2.1–2.2、假设 1 和式 (2.1)，未复现其算例。

论文的到达性结论需要光滑性、秩条件、唯一初始解和有界零集等假设；Newton–固定点同伦还要求存在正定矩阵 A，使 `F'(x)+A` 处处非奇异。其形式可改写为 `F(x)−(1−a)F(x0)+(1−a)A(x−x0)`，附加项在目标处消失。

项目判断：伪弧长能通过局部转折，不自动保证当前残差射线连到目标。非光滑库仑关系、有限应变可行域和路径有界性均未满足上述已证明条件，不能把论文的概率一结论移植过来，也不能声称加一个对角项就获得保证。该式仅为后续改变数值同伦的候选，当前未实现。现有正常走步但未到目标的记录不能证明闭环、原问题无解或发生断裂。

## 2026-10-15补核：IPC几何分支与残差求解

2026-10-16深部适用性更正：0°/33°/57°失败末态的整边间隙远小于Gauss采样间隙；固定场加密产生远超验收限的自由残差。因此浅部资格不能外推，解析单侧候选暂停用于深部定量预测。将位移上限提前到CCD之前的有界变体也未取得新接受态，不推广。[官方IPC Toolkit](https://github.com/ipc-sim/ipc-toolkit)分别提供接触势与碰撞检测接口；正确调用接口不等于任意自定义积分都与之匹配。下一轮恢复已有完整案例的原生接触作为对照，具体因果解释依赖本地实算而非论文保证，详见[求解状态](../docs/solver_status.md)。

实际采用状态：新增独立解析圆鼻候选，直接从圆弧距离求法向、曲率和完整刷新摩擦Jacobian。它没有实现Gregory片、mortar或半光滑活跃集；材料边界的单侧参考积分也不等于IPC improved-max。外包刀具多边形仅作保守CCD，材料自接触仍为IPC。原64片局部卡点及两级积分/双角度控制通过，各自新0–.05 mm路径通过，但无深行程保证，30°力偏差未解释。详见[状态](../docs/solver_status.md)。

物理barrier标度依据本地ipctk1.6.0源码：[barrier函数](https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.0/src/ipc/barrier/barrier.cpp)及[BarrierPotential](https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.0/src/ipc/potentials/barrier_potential.cpp)。本候选对真实间隙g采用 `B=−κ/dhat³ (g²−dhat²)² log(g²/dhat²)`（0<g<dhat），κ含出平面宽度；按固定参考边界长度积分，导数含距离曲率。摩擦沿当前法向的切平面使用原Coulomb平滑，残差切线包含法向容量与切向基刷新；“刷新后的摩擦能量”不是该总残差的全局势。单位/导数与本地源码及差分核对，不据此声称与原接触权重相同。

Puso & Laursen (2002), *A 3D contact smoothing method using Gregory patches*, IJNME54,1161–1194，[出版社摘要](https://onlinelibrary.wiley.com/doi/10.1002/nme.466)。本次只核对摘要：一致的间隙/力计算和相邻片切平面连续性用于减轻节点跨面片时的接触抖动，并改善文中算例收敛。与本次圆鼻折线节点切换证据相符，支持下一步核查曲线一致接触；并未取得算法全文、复现Gregory片或证明本项目适用。不能只平滑显示法向而保留不一致的力、切线和防穿透检测。

[IPC技术补充](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf)第8–9节区分点—边/点—点切向映射，说明冻结法向力与切向基后的摩擦势及其平滑。此处的连续Hessian结论不能直接扩展成“每次刷新接触几何后的总残差处处光滑”；本项目使用完整几何耦合Jacobian，刀鼻多边形顶点仍需单侧核查。

Hüeber、Stadler、Wohlmuth (2008), *A primal-dual active set algorithm for three-dimensional contact problems with Coulomb friction*, SISC30,572–596，[作者全文](https://math.nyu.edu/~stadler/papers/friction3d.pdf)，DOI10.1137/060671061。本轮读摘要与第1–2节：通过非光滑互补条件联系活跃集与半光滑Newton，论文的具体问题为线弹性、mortar/乘子离散；未复现，也不能直接给当前有限应变IPC提供收敛保证。

[PETSc SNES手册](https://petsc.org/release/manual/snes/)的Newton线搜索/信赖域是可比较的数值框架。本地诊断实现了残差Armijo线搜索，但没有调用PETSc；普通Newton、邻侧切线及切线混合的有界回放均未通过，不能因引用成熟方法就认定修复成功。新的单点分支消元证据只定位这次离散问题的局部停滞，不是原物理问题无解、真正失稳或裂纹的证明。后续优先检查圆鼻几何分片一致性，而非单纯增加迭代预算。实际结果统一见[求解状态](../docs/solver_status.md)。
