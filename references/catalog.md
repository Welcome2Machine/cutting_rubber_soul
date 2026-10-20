# 橡胶斜切项目统一文献目录

2026-10-20增补（原R编号目录保留）：

| 来源 | 阅读范围与用途 | 笔记 |
|---|---|---|
| Huang等（2025），*Geometric Contact Potential*，[DOI](https://doi.org/10.1145/3731142)、[官方教程](https://ipctk.xyz/tutorials/gcp.html) | 官方公式、源码、局部接口实测；未声称本轮逐页阅读作者PDF。定向本体自接触的独立候选 | [实现、单位与资格限制](geometric_self_contact_assessment.md) |
| Abide等（2022），*Unified primal-dual active set method for dynamic frictional contact problems*，[全文](https://link.springer.com/article/10.1186/s13663-022-00729-4) | 开放全文相关框架；借用库仑投影／活动集思想，未复现其动力学算法 | [活动集及负面结果](geometric_self_contact_assessment.md) |
| Grippo、Lampariello、Lucidi（1986），*A Nonmonotone Line Search Technique for Newton’s Method*，[DOI](https://epubs.siam.org/doi/10.1137/0723046) | 出版社书目与后续官方方法说明，未阅读全文；采用有限窗口启发式，不声称收敛定理自动适用 | [非单调资格结果](geometric_self_contact_assessment.md) |

截至2026-10-13。**F**=既往或本轮查过原文相关章节，非逐页全读；**A**=摘要/元数据；**S**=经官方文档或另一来源间接引用；**U**=身份/版本未充分核实。具体阅读位置与日期以链接笔记为准，整理入库不提升证据等级。R编号表示资料条目，有的包含论文与软件文档，不等于论文篇数。

归档范围：references、docs、prototype、mechanics、tools、reproductions 的Markdown及本地outputs中非源码快照的历史Markdown。扫描到的来源链接全部保留在 [source_index](source_index.md)。没有凭空补齐未记录的论文，也不把搜索结果里顺带出现、未实际采用的论文算成已参考原文。

## 切割、断裂与材料时间响应

| ID | 文献与原始来源 | 范围；对任务的用途与限制 | 阅读笔记 |
|---|---|---|---|
| R01 | Spagnoli等（2019），*Cutting resistance of soft materials: Effects of blade inclination and friction*，[DOI](https://doi.org/10.1016/j.tafmec.2019.02.017) | F；斜切能量/摩擦。其刀具角度定义、黏附剪应力和贯穿后稳态不能直接替代当前压入 | [R01](method_notes.md) |
| R02 | Dammaß、Ambati、Kästner（2021），*A unified phase-field model of fracture in viscoelastic materials*，[DOI](https://doi.org/10.1007/s00161-021-01013-3) | F；小应变黏弹相场，不能用于当前有限应变材料核 | [R02](method_notes.md) |
| R03 | Dammaß等（2023），*Phase-field modelling and analysis of rate-dependent fracture phenomena at finite deformation*，[DOI](https://doi.org/10.1007/s00466-023-02310-1) | F；有限变形黏弹材料与相场，已有独立复现；EPDM参数不移植CR | [R03](method_notes.md)、[复现](../reproductions/dammass2023/README.md) |
| R04 | Moreno-Mateos、Steinmann，*Cutting soft materials: how material differences shape the response*，[2025预印本](https://arxiv.org/abs/2507.13565)、[2026期刊](https://doi.org/10.1038/s41524-025-01869-y) | F；力/表面位移、体材料与界面机制。2026-01-06正式发表；DOI中的2025不是期刊年份。低接触压力和CZM结论依赖工况 | [R04](method_notes.md)、[历史补充](research_history.md) |
| R09 | Lopez-Pamies等（2025），*Classical variational phase-field models cannot predict fracture nucleation*，[DOI](https://doi.org/10.1016/j.cma.2024.117520) | F；独立强度与起裂，不把正则化长度拟合当盲预测 | [R09](method_notes.md) |
| R10 | Kamarei、Breedlove、Lopez-Pamies（2025），*Nucleation and propagation of fracture in viscoelastic elastomers: A complete phase-field theory*，[v1](https://arxiv.org/abs/2506.16479v1) | F（部分章节）；黏弹性/强度/断裂能联合候选，未耦合本项目刀具摩擦 | [R10](method_notes.md) |
| R11 | Lopez-Pamies等（2026），*Nucleation and propagation of brittle fracture as a constrained energy minimization problem*，[预印本](https://arxiv.org/abs/2608.28326) | F；强度约束允许萌生区域；非保守摩擦仍需扩展 | [R11](method_notes.md) |
| R28 | Bergström、Boyce（1998），*Constitutive modeling of the large strain time-dependent behavior of elastomers*，[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0022509697000756) | A；填炭黑CR有限应变率效应，不能移植支路参数 | [时间响应](prefracture_rate_response.md) |
| R29 | Efremov、Kotova、Timashev（2020），*Viscoelasticity in simple indentation-cycle experiments: a computational study*，[DOI](https://www.nature.com/articles/s41598-020-70361-y) | F；加载时间/松弛竞争影响曲线；总力卷积不是新的有限元重算 | [时间响应](prefracture_rate_response.md) |
| R30 | Kojima、Nagasawa、Igarashi（2020），*Cutting Characteristics of Silicone Rubber Sheet with Respect to Indentation Velocity of Keen Wedge Blade*，[DOI 10.9773/sosei.61.1](https://www.jstage.jst.go.jp/article/sosei/61/708/61_1/_article/-char/en) | F；早段近线性且斜率弱速度依赖，反对自动把前裂差异解释成黏性主导；硅胶/支承/刀形不同 | [时间响应](prefracture_rate_response.md) |
| R47 | Anssari-Benam、Bucchi、Destrade、Saccomandi（2022期刊/2024预印本），*The generalised Mooney space for modelling the response of rubber-like materials*，[DOI](https://doi.org/10.1007/s10659-022-09889-1)、[作者稿](https://arxiv.org/abs/2403.01223) | A；多变形模式约束本构选择，不提供本批CR材料参数 | [历史](research_history.md) |

## 近不可压离散、边界及压入机理

| ID | 文献与原始来源 | 范围；对任务的用途与限制 | 阅读笔记 |
|---|---|---|---|
| R07 | Maas等（2011），*Implementation and Verification of a Nodally-Integrated Tetrahedral Element in FEBio*，UUSCI-2011-007，[原文](https://publications.sci.utah.edu/publications/SCITechReports/UUSCI-2011-007.pdf) | F；锁死/软模/稳定化权衡；三维UT4不保证二维节点体积方案 | [R07](method_notes.md) |
| R14 | Hohlfeld、Mahadevan（2011），*Unfolding the Sulcus*，[DOI](https://doi.org/10.1103/PhysRevLett.106.105702) | F；局部折皱与分支，不能把本地不收敛直接认定为物理分岔或断裂 | [R14](method_notes.md) |
| R31 | Lubarda、Lubarda（2024期刊/2023作者稿），二维圆柱压入与位移约束研究，[DOI](https://doi.org/10.1016/j.euromechsol.2024.105362)、[全文](https://arxiv.org/html/2309.02361v1) | F（作者稿）；压入量依赖远场位移基准。小变形圆柱不能量化薄刀有限应变 | [历史](research_history.md) |
| R32 | Ahmadi等（2024预印本），*Plane stress finite element modelling of arbitrary compressible hyperelastic materials*，[原文](https://arxiv.org/html/2410.22562v1) | F；出平面变量与一致凝聚；不能据此将20 mm宽试样直接改平面应力 | [历史](research_history.md) |
| R33 | Bonet、Burton（1998），*A simple average nodal pressure tetrahedral element for incompressible and nearly incompressible dynamic explicit applications*，[出版来源](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291099-0887%28199805%2914%3A5%3C437%3A%3AAID-CNM162%3E3.0.CO%3B2-W) | A/方法来源；节点平均压力，非二维准静态接触保证 | [历史](research_history.md) |
| R34 | Bonet、Marriott、Hassan（2001），*An averaged nodal deformation gradient linear tetrahedral element for large strain explicit dynamic applications*，[DOI](https://doi.org/10.1002/cnm.429) | A/方法来源；平均压力在弯曲问题的短板，需单独验收柔度 | [历史](research_history.md) |
| R35 | Bercovier、Pironneau（1979），P1-iso-P2定义来源，[DOI](https://doi.org/10.1007/BF01399555) | S；由Symfem定义追溯。未独立全文核对，不把Stokes结论移植有限应变 | [宏混合](macro_mixed_discretization.md) |
| R36 | Koko，*Uzawa conjugate gradient method for the Stokes problem: Matlab implementation with P1-iso-P2/P1 finite element*，[作者稿](https://www.ljll.fr/~frey/papers/Navier-Stokes/Koko%20J.%2C%20Uzawa%20conjugate%20gradient%20method%20for%20the%20Stokes%20problem.pdf) | F；细位移/粗压力及积分。年份/正式出版身份未独立确认，搜索抓取时间不当发表年 | [宏混合](macro_mixed_discretization.md) |
| R37 | Fu、Neunteufel、Schöberl、Zdunek（2025），*A four-field mixed formulation for incompressible finite elasticity*，[DOI](https://doi.org/10.1016/j.cma.2025.118082)、[原文](https://arxiv.org/html/2503.00989v2) | F；非线性体积约束、独立梯度/应力及局部消元；未实现刀具耦合 | [本轮配对](pairing_review.md) |
| R38 | Karabelas等（2020，在线2019），*Versatile stabilized finite element formulations for nearly and fully incompressible solid mechanics*，[出版全文](https://link.springer.com/article/10.1007/s00466-019-01760-w) | F，本轮新增；MINI和压力投影，有限变形假设及计算代价；不保证本地摩擦全程 | [本轮配对](pairing_review.md) |
| R39 | Lederer等（2017），*Divergence-free Reconstruction Operators for Pressure-Robust Stokes Discretizations with Continuous Pressure Finite Elements*，[DOI](https://epubs.siam.org/doi/10.1137/16M1089964) | A，本轮新增；稳定配对不自动等于压力鲁棒；Stokes重构未移植 | [本轮配对](pairing_review.md) |
| R40 | Tian等（2021预印本），*Mixed displacement-pressure-phase field framework for finite strain fracture of nearly incompressible hyperelastic materials*，[作者稿](https://arxiv.org/abs/2112.00294) | A（本轮复核）；受损相压力退化与Q1/P0、P2/P1比较不能替代当前完整体配对检查 | [检索历史](../docs/literature_search_log.md)、[配对](pairing_review.md) |
| R48 | Shojaeifard等（2026），*Deep Indentation of Soft Solids: A Limit State*，[v4](https://arxiv.org/abs/2606.30971v4) | A；圆杆/Ogden半空间的深压入尺度，不是薄刃早段曲率的验证 | [历史](research_history.md) |

## 接触、摩擦及非线性求解

| ID | 文献与原始来源 | 范围；对任务的用途与限制 | 阅读笔记 |
|---|---|---|---|
| R05 | Pinto Carvalho等（2022），*An efficient algorithm for rigid/deformable contact interaction based on the dual mortar method*，[DOI](https://doi.org/10.1007/s00466-022-02226-2) | F；有限变形无摩擦，客观摩擦和新裂面需另做 | [R05](method_notes.md) |
| R06 | Khandelwal、Porwal、Wadhawan（2022），*Adaptive quadratic finite element method for the unilateral contact problem*，[作者稿](https://arxiv.org/abs/2209.04906) | F；二次边顶点/中点约束及后验误差；线性无摩擦假设 | [R06](method_notes.md) |
| R08 | Acary、Brémond、Huber（2018），*On solving contact problems with Coulomb friction: formulations and numerical comparisons*，[HAL](https://hal.science/hal-01878539)、[作者稿](https://tripop.inrialpes.fr/people/acary/publications/Acary_Bremond_Huber.pdf) | F；多类离散摩擦算法比较；作者稿题名略异，没有普适最佳算法 | [R08](method_notes.md) |
| R12a | Areias等（2019），*A simple and robust Coulomb frictional algorithm based on 3 additional degrees-of-freedom and smoothing*，[DOI](https://doi.org/10.1016/j.finel.2019.103321) | A；平滑参数延续；全文访问受限，本地尝试不是复现 | [R12](method_notes.md) |
| R12b | Jiang（1999），Fischer–Burmeister相关互补算法论文，[DOI](https://doi.org/10.1287/moor.24.3.529) | A；互补重写的依据；未将全局收敛标题当当前问题保证 | [R12](method_notes.md) |
| R13 | Deuflhard（2011），*Newton Methods for Nonlinear Problems*，[DOI](https://doi.org/10.1007/978-3-642-23899-4)；PETSc NLEQERR | S（专著）/F（官方算法实现）；只核查修正量线搜索，未读完专著 | [R13](method_notes.md) |
| R15 | Hassani、Hild、Ionescu、Sakki，*A mixed finite element method and solution multiplicity for Coulomb frictional contact*，[作者全文](https://www.math.univ-toulouse.fr/~phild/Publications/CMAME3.pdf) | F；多解与摩擦谱。刊年本次不补填，线弹性不证明当前发生分支变化 | [R15](method_notes.md) |
| R16 | Qi、Sun（1998修订稿），*A Survey of Some Nonsmooth Equations and Smoothing Newton Methods*，[原文](https://defengwebsite.github.io/files/qsreview1.pdf) | F；根点/广义导数非奇异等局部前提 | [R16](method_notes.md) |
| R17 | Li等（2023），*Convergent Incremental Potential Contact*，[作者稿](https://arxiv.org/abs/2307.15908) | F；面积权重/接触修正/线性碰撞几何；不覆盖材料、网格或实验验收 | [R17](method_notes.md) |
| R18 | Giacoma、Dureisseix、Gravouil（2016），*An efficient quasi-optimal space-time PGD application to frictional contact mechanics*，[DOI](https://doi.org/10.1186/s40323-016-0067-7) | F；LATIN局部—整体分解，非现成切割宿主 | [R18](method_notes.md) |
| R19 | Both等（2022），*Iterative splitting schemes for a soft material poromechanics model*，[DOI](https://doi.org/10.1016/j.cma.2021.114183) | F；借鉴稳定化耦合分裂；孔压并非接触压力 | [R19](method_notes.md) |
| R20 | Degroote等（2010），*Performance of partitioned procedures in fluid–structure interaction*，[DOI](https://doi.org/10.1016/j.compstruc.2009.12.006) | A（论文）/F（preCICE文档）；界面加速依赖内层收敛 | [R20](method_notes.md) |
| R21 | Acary、Armand、Nguyen（2026），*Solving contact problems with Coulomb friction using an interior point algorithm and path following techniques*，[DOI](https://doi.org/10.1016/j.cma.2026.119227) | F；用户原PDF已核查；平方方程、ANM符号与原始摩擦终验收限制另有记录 | [R21](method_notes.md) |
| R22 | Gfrerer等（2022），*On the SCD semismooth* Newton method for generalized equations with application to a class of static contact problems with Coulomb friction*，[原文](https://arxiv.org/pdf/2205.15129) | F；多值图线性化；本地试验不能宣称论文全套复现 | [R22](method_notes.md) |
| R23 | PETSc TSPSEUDO及其引用的Kelley–Keyes（1998）、Coffey–Kelley–Keyes（2002） | F（[文档](https://petsc.org/main/manualpages/TS/TSPSEUDO/)）/S（论文）；辅助时间求稳态，未通读所引两篇论文 | [R23](method_notes.md) |
| R24 | Dickson、Kelley、Ipsen、Kevrekidis（2007），*Condition Estimates for Pseudo-Arclength Continuation*，[DOI](https://doi.org/10.1137/060654384) | F；简单折点条件，非光滑接触不能直接继承 | [R24](method_notes.md) |
| R25 | Jong（2012），*A Newton-Fixed Point Homotopy Algorithm For Nonlinear Complementarity Problems With Generalized Monotonicity*，[作者稿](https://optimization-online.org/wp-content/uploads/2012/08/3562.pdf) | F；路径可延续不等于能到达目标 | [R25](method_notes.md) |
| R41 | Chen等（2025），*Offset Geometric Contact*，[DOI](https://doi.org/10.1145/3731205)、[作者页](https://graphics.cs.utah.edu/research/projects/ogc/) | F；本地与原IPC势/PTC组合在特征切换处失败，候选未通过准入；非论文完整算法否定 | [历史](research_history.md) |
| R42 | Li等（2020），*Incremental Potential Contact*，[主文](https://ipc-sim.github.io/file/IPC-paper-350ppi.pdf)、[补充](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf) | F；屏障/摩擦lagging。原文明确lagging无收敛保证，不以冻结内层残差代替真平衡 | [历史](research_history.md) |
| R43 | Gee等（2009），*Pseudo-transient continuation for nonlinear transient elasticity*，[DOI](https://doi.org/10.1002/nme.2527) | A/方法来源；弹性PTC应用，非本项目全程稳定证明 | [历史](research_history.md) |
| R44 | *A Noise-Tolerant Quasi-Newton Algorithm for Unconstrained Optimization*（2020预印本），[作者稿](https://arxiv.org/abs/2010.04352) | A/方法来源；近收敛能量相减与线搜索噪声，不放松真残差 | [历史](research_history.md) |
| R45 | Gast等（2015），*Optimization Integrator for Large Time Steps*，[原文](https://math.ucdavis.edu/~jteran/papers/GSSJT15.pdf)，DOI 10.1109/TVCG.2015.2459687 | F；不定方向、截断CG、下降方向；未采用其完整动力学算法 | [历史](research_history.md) |
| R46 | Fernández-Fernández、Löschner、Bender，*Progressively Projected Newton's Method*（2025初版/2026修订），[v2](https://arxiv.org/abs/2505.21013v2) | F（相关结论）；投影改变全局切线；本地hybrid并非完整PPN实现 | [历史](research_history.md) |
| R49 | Zimmerman、Ateshian（2018），*A Surface-to-Surface Finite Element Algorithm for Large Deformation Frictional Contact in FEBio*，[DOI](https://doi.org/10.1115/1.4040497) | A；官方模型库提供候选基准；未取得完整论文，不声称复现其全部结果 | [检索历史](../docs/literature_search_log.md) |

## 实验辨识、位移与测量边界

| ID | 文献与来源 | 用途及限制 |
|---|---|---|
| R26 | Oddes、Solav（2023），*Identifiability of soft tissue constitutive parameters from in-vivo macro-indentation*，[摘要](https://pubmed.ncbi.nlm.nih.gov/36801779/)，DOI 10.1016/j.jmbbm.2023.105708 | A；力＋位移有助参数识别，不保证所有参数唯一。[历史笔记](research_history.md) |
| R27 | Rokoš等（2018），*On Micromechanical Parameter Identification With Integrated DIC and the Role of Accuracy in Kinematic Boundary Conditions*，[作者稿](https://arxiv.org/abs/1810.08245)，[DOI](https://doi.org/10.1016/j.ijsolstr.2018.04.004) | F；边界运动误差污染材料反演，不能混用输入与独立验证。[历史笔记](research_history.md) |
| R50 | Tayeb等（2019），*Identification of Constitutive Parameters Governing the Hyperelastic Response of Rubber by Using Full-field Measurement and the Virtual Fields Method*，[作者稿](https://arxiv.org/abs/1907.02687) | A；全场识别路线，不等于单视角切割光流能唯一识别G/μ。[交付接口](../docs/optical_flow_handoff.md) |
| R51 | iDICs（2025），*A Good Practices Guide for Digital Image Correlation*, Edition 2，[官方](https://idics.org/guide/)、[DOI](https://doi.org/10.32720/idics/gpg.ed2) | 官方指南；设置/相关/应变质量要求，当前光流未宣称满足全部指南。[交付接口](../docs/optical_flow_handoff.md) |

Watson同伦综述/报告属于历史检索线索，原记录未保留确切版本且当时无可读全文：**U**，不补造题名、DOI或理论结论。Simo–Taylor–Pister、Simo–Armero等仅在Fu文中作为追溯来源出现，未独立阅读或移植，不冒充本项目已经核查的原文。

## 官方实现与厂商资料（非论文）

完整地址集中在 [source_index](source_index.md)，方法使用/限制保留在 [历史记录](research_history.md)。

| 类别 | 收录来源 | 本项目允许支持的结论 |
|---|---|---|
| IPC/PolyFEM | ipctk v1.6.0 API、源码、convergent、OGC、几何与模拟教程；PolyFEM摩擦及quickstart | 固定版本接口和公式，不等于完整切割验证 |
| 其它接触宿主 | GetFEM有限应变/大滑移；FEBio v4.13、模型库、滑动接触；ASiMoV示例 | 候选能力边界。ASiMoV所查示例为小应变，不冒充现成有限应变 |
| 非线性/线性代数 | PETSc线搜索/PTC/信赖域/互补；SciPy splu/GMRES/milp/least_squares；SuperLU_MT；Siconos；preCICE；GAMS | 算法机制与返回状态，仍需独立真方程验收 |
| 离散/材料说明 | Symfem、DefElement、OpenGeoSys Cook、MOOSE、COMSOL、solids4foam、Abaqus、ANSYS | 元定义/基准/体能/边界假设。单元内F-bar不改变常F线性三角形 |
| 实现工程 | Microsoft进程QoS/CPU Set；NumPy quadrature；threadpoolctl；Slurm；Cornell非线性课程 | 调度、求积和实现依据，非橡胶实验依据 |
| 原材料/刀具 | Maxell Kureha现行目录与23°C CB245NE报告；经销商旧目录；gomusuke；Feather工业目录 | G=.47与旧.66版本区别，不代表本批标定；99755以用户截图为准，官网未检出型号不可替代 |

## R52：楔入全场验证的补充资料

Tan、Kang、Patterson（2016），*Experimental investigation on surface deformation of soft half plane indented by rigid wedge*，Applied Mathematics and Mechanics (English Edition) 37(10),1349–1360，DOI [10.1007/s10483-016-2139-9](https://pubs-en.cstam.org.cn/data/article/amm/preview/pdf/20161007.pdf)。阅读出版PDF文本摘要、§1及§4.1；图像渲染请求失败，未声称目视核查图8。作者用DIC位移场判定楔入近似理论的适用范围；橡胶大变形及刀具几何会使假设失效。采用的是总力以外的空间判据，不采用其材料、1.68 mm半径、角度和失效深度。适用讨论见[近线性验证链](../docs/history/topics/linearity_verification.md)。没有新增本地PDF或上传论文。

## R54–R55：前裂近线性的定向补查（2026-10-14）

| ID | 文献与原始来源 | 阅读范围与用途 |
|---|---|---|
| R54 | Mitsomwang、Nagasawa、Kuroiwa、Fukushima（2014），*Deformation Analysis of Silicone Rubber Sheet Subjected to Keen WC Blade Indentation*，IJAT8(5),761–772；[DOI10.20965/ijat.2014.p0761](https://www.fujipress.jp/ijat/au/ijate000800050761/) | F（第3节、图14/17/18目视）；无损伤大应变平面应变先例，材料/垫层与当前目标不同；[笔记](prefracture_rate_response.md) |
| R55 | Ciavarella、Hills、Monno（1998），*Contact problems for a wedge with rounded apex*，IJMS40(10),977–988；[机构库](https://eprints.soton.ac.uk/23252/)，DOI10.1016/S0020-7403(97)00141-0 | A及可检索首页；完整PDF获取失败。浅楔/线弹性极限线索，未用于给当前模型套系数；[笔记](prefracture_rate_response.md) |

## 本地PDF映射（原件不发布）

2026-10-18补充：Ceglie、Violano、Portaluri、Algieri、Afferrante、Scaraggi、Menga，*Contact area shrinkage and increase in wavy frictional sliding contacts*，JMPS206(2026),106389（2025-10-10在线），[DOI](https://doi.org/10.1016/j.jmps.2025.106389)、[作者机构库全文](https://iris.unisalento.it/retrieve/533bbb63-a320-4e7c-87fc-4720e8ce5a05/1-s2.0-S0022509625003631-main.contactarea.pdf)。本轮阅读第2节式(6)、第4.3节：有限弹性摩擦接触及剪切强度上限；不移植PDMS/玻璃的参数或面积结论。项目采用节点平均压力的光滑诊断律，区别和完整导数见[实现笔记](../docs/history/topics/interface_candidate.md)。仅公开原创笔记和链接，不复制全文/图像。

R53：Sichetti等，*High-Order Continuous Geometrical Validity*，TOG2025，DOI [10.1145/3745763](https://doi.org/10.1145/3745763)。2026-10-14核查作者arXiv v2 HTML §7–8.3；高阶映射局部失效、积分和浮点保守性。未移植作者算法，不作为橡胶本构/实验依据。阅读范围及限制见[配对笔记](pairing_review.md)。

| ID | 本地文件名 | PDF页数 |
|---|---|---:|
| R01 | `2.pdf` | 7 |
| R02 | `A unified phase-field model of fracture in viscoelastic.pdf` | 23 |
| R03 | `updated phase.pdf` | 25 |
| R04 | `MorenoMateos_Steinmann_2025_preprint.pdf` | 41 |
| R05 | `PintoCarvalho_et_al_2022_dual_mortar.pdf` | 25 |
| R06 | `Khandelwal_Porwal_Wadhawan_2022_quadratic_contact.pdf` | 29 |
| R07 | `Maas_et_al_2011_nodal_integration_FEBio.pdf` | 10 |
| R08 | `Acary_Bremond_Huber_2018_contact_solvers_author.pdf` | 87 |
| R09 | `LopezPamies_et_al_2025_nucleation_review.pdf` | 17 |
| R10 | `Kamarei_Breedlove_LopezPamies_2025_preprint.pdf` | 46 |
| R11 | `LopezPamies_et_al_2026_constrained_fracture_preprint.pdf` | 34 |
| R21 | `solvingcontactproblemswithCoulombFriction.pdf` | 25 |
| R30 | `Kojima_2020_silicone_wedge_velocity.pdf` | 6 |
| R54 | `Mitsomwang_2014_silicone_blade.pdf` | 12 |

页码定位须按具体PDF版本；文件页数不同于期刊印刷页码跨度。新增公开笔记不含论文全文转录或原图。
