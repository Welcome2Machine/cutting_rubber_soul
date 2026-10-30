# 参考文献原文与引用导读

整理：2026-10-22。本目录合并原文献目录、方法笔记和专题导读，保留原始论文PDF；不保存重复检索日志、下载元数据或逐次运行状态。文献说明与项目结果分开：后者见[研究历史](../docs/research_history.md)。本次只整理已有阅读记录，没有重新联网验证，也不把摘要升级为全文证据。

阅读标记：**F**＝曾核读原文相关章节，并非逐页全读；**A**＝摘要/元数据；**S**＝官方文档或间接来源；**U**＝版本或身份未充分确认。保留R编号便于追溯，编号不等于论文篇数。

## 当前任务的阅读顺序

1. **材料与近线性压入：** R03、R28–R30、R54及下文Reese–Govindjee。当前主线是有限应变等容Maxwell，并非完整Bergström–Boyce；静态G不能独立识别G0、G∞或松弛谱。近线性可来自接触几何与时间响应共同作用，不能直接用乘指数衰减的总力替代FE。
2. **接触与运动学：** R17/R42、Duong–Sauer、GCP。共同刚体运动、表面材料身份、投影/激活、完整切线和接触积分须一起核查；不能在原IPC模板中任意删除VV或重复积分。
3. **离散与约束：** R32、R37–R38、R53及广义平面应变教程。稳定配对、高阶几何、局部平面应力与有限宽三维是不同问题，方法名称不证明本项目精度或可解性。
4. **拟合：** R26–R27、Hartmann–Gilbert及dolfin-adjoint。含内部变量的参数变化必须传播整个历史；固定旧历史偏导、局部切线预测和有限参数加载各自独立验收。
5. **断裂研究分支：** R01–R04、R09–R11及下方2026-10-28补充。完整体基线不启用断裂；独立分支已实现混合模式内聚律与全内部边候选网络。切后稳态、低压CZM结论和起裂强度不能移植到未裂高压区。

## 对本项目有直接约束的解释

- *Cutting soft materials*的弹性体可平滑地从压入转为切割；无明显掉力不证明未裂，力曲线斜率下降也不能独立给出起裂时刻。
- Kojima2020在其薄硅胶/垫层/13.9°刀实验中观察到前转折近线性且斜率弱速度依赖，不能把本项目全部早段偏差预先解释为黏性主导。
- Mitsomwang2014给出无损伤、大应变、平面应变、Mooney–Rivlin和正则库仑刀压入先例；试样/支承不同，图中仍有上弯，不是严格线性公式。
- 严格不可压缩、二维平面应变且第三主伸长为1时，标准两参数Mooney–Rivlin的I1/I2形状退化，不能靠重新分配这两个系数产生任意新曲线形状。可压缩性、面外运动或更一般本构需另行分析。
- 沿刃拉切与本项目面内斜压的角度定义不同；能量分解与力的坐标投影不是改变摩擦参数的因果贡献。
- 文献的收敛理论、材料参数及现成软件能力不替代本项目真实残差、正J、CCD、双历史与连续加载检查。

## 切割、断裂与材料时间响应

| ID | 来源 | 已有阅读范围、用途与限制 |
|---|---|---|
| R01 | Spagnoli等（2019），*Cutting resistance of soft materials: Effects of blade inclination and friction*，[DOI](https://doi.org/10.1016/j.tafmec.2019.02.017) | F；斜切能量/摩擦。其刀具角度定义、黏附剪应力和贯穿后稳态不能直接替代当前压入 |
| R02 | Dammaß、Ambati、Kästner（2021），*A unified phase-field model of fracture in viscoelastic materials*，[DOI](https://doi.org/10.1007/s00161-021-01013-3) | F；小应变黏弹相场，不能用于当前有限应变材料核 |
| R03 | Dammaß等（2023），*Phase-field modelling and analysis of rate-dependent fracture phenomena at finite deformation*，[DOI](https://doi.org/10.1007/s00466-023-02310-1) | F；有限变形黏弹材料与相场，已有独立复现；EPDM参数不移植CR |
| R04 | Moreno-Mateos、Steinmann，*Cutting soft materials: how material differences shape the response*，[2025预印本](https://arxiv.org/abs/2507.13565)、[2026期刊](https://doi.org/10.1038/s41524-025-01869-y) | F；力/表面位移、体材料与界面机制。2026-01-06正式发表；DOI中的2025不是期刊年份。低接触压力和CZM结论依赖工况 |
| R09 | Lopez-Pamies等（2025），*Classical variational phase-field models cannot predict fracture nucleation*，[DOI](https://doi.org/10.1016/j.cma.2024.117520) | F；独立强度与起裂，不把正则化长度拟合当盲预测 |
| R10 | Kamarei、Breedlove、Lopez-Pamies（2025），*Nucleation and propagation of fracture in viscoelastic elastomers: A complete phase-field theory*，[v1](https://arxiv.org/abs/2506.16479v1) | F（部分章节）；黏弹性/强度/断裂能联合候选，未耦合本项目刀具摩擦 |
| R11 | Lopez-Pamies等（2026），*Nucleation and propagation of brittle fracture as a constrained energy minimization problem*，[预印本](https://arxiv.org/abs/2608.28326) | F；强度约束允许萌生区域；非保守摩擦仍需扩展 |
| R28 | Bergström、Boyce（1998），*Constitutive modeling of the large strain time-dependent behavior of elastomers*，[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0022509697000756) | A；填炭黑CR有限应变率效应，不能移植支路参数 |
| R29 | Efremov、Kotova、Timashev（2020），*Viscoelasticity in simple indentation-cycle experiments: a computational study*，[DOI](https://www.nature.com/articles/s41598-020-70361-y) | F；加载时间/松弛竞争影响曲线；总力卷积不是新的有限元重算 |
| R30 | Kojima、Nagasawa、Igarashi（2020），*Cutting Characteristics of Silicone Rubber Sheet with Respect to Indentation Velocity of Keen Wedge Blade*，[DOI 10.9773/sosei.61.1](https://www.jstage.jst.go.jp/article/sosei/61/708/61_1/_article/-char/en) | F；早段近线性且斜率弱速度依赖，反对自动把前裂差异解释成黏性主导；硅胶/支承/刀形不同 |
| R47 | Anssari-Benam、Bucchi、Destrade、Saccomandi（2022期刊/2024预印本），*The generalised Mooney space for modelling the response of rubber-like materials*，[DOI](https://doi.org/10.1007/s10659-022-09889-1)、[作者稿](https://arxiv.org/abs/2403.01223) | A；多变形模式约束本构选择，不提供本批CR材料参数 |

## 近不可压离散、边界及压入机理

| ID | 来源 | 已有阅读范围、用途与限制 |
|---|---|---|
| R07 | Maas等（2011），*Implementation and Verification of a Nodally-Integrated Tetrahedral Element in FEBio*，UUSCI-2011-007，[原文](https://publications.sci.utah.edu/publications/SCITechReports/UUSCI-2011-007.pdf) | F；锁死/软模/稳定化权衡；三维UT4不保证二维节点体积方案 |
| R14 | Hohlfeld、Mahadevan（2011），*Unfolding the Sulcus*，[DOI](https://doi.org/10.1103/PhysRevLett.106.105702) | F；局部折皱与分支，不能把本地不收敛直接认定为物理分岔或断裂 |
| R31 | Lubarda、Lubarda（2024期刊/2023作者稿），二维圆柱压入与位移约束研究，[DOI](https://doi.org/10.1016/j.euromechsol.2024.105362)、[全文](https://arxiv.org/html/2309.02361v1) | F（作者稿）；压入量依赖远场位移基准。小变形圆柱不能量化薄刀有限应变 |
| R32 | Ahmadi等（2024预印本），*Plane stress finite element modelling of arbitrary compressible hyperelastic materials*，[原文](https://arxiv.org/html/2410.22562v1) | F；出平面变量与一致凝聚；不能据此将20 mm宽试样直接改平面应力 |
| R33 | Bonet、Burton（1998），*A simple average nodal pressure tetrahedral element for incompressible and nearly incompressible dynamic explicit applications*，[出版来源](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291099-0887%28199805%2914%3A5%3C437%3A%3AAID-CNM162%3E3.0.CO%3B2-W) | A/方法来源；节点平均压力，非二维准静态接触保证 |
| R34 | Bonet、Marriott、Hassan（2001），*An averaged nodal deformation gradient linear tetrahedral element for large strain explicit dynamic applications*，[DOI](https://doi.org/10.1002/cnm.429) | A/方法来源；平均压力在弯曲问题的短板，需单独验收柔度 |
| R35 | Bercovier、Pironneau（1979），P1-iso-P2定义来源，[DOI](https://doi.org/10.1007/BF01399555) | S；由Symfem定义追溯。未独立全文核对，不把Stokes结论移植有限应变 |
| R36 | Koko，*Uzawa conjugate gradient method for the Stokes problem: Matlab implementation with P1-iso-P2/P1 finite element*，[作者稿](https://www.ljll.fr/~frey/papers/Navier-Stokes/Koko%20J.%2C%20Uzawa%20conjugate%20gradient%20method%20for%20the%20Stokes%20problem.pdf) | F；细位移/粗压力及积分。年份/正式出版身份未独立确认，搜索抓取时间不当发表年 |
| R37 | Fu、Neunteufel、Schöberl、Zdunek（2025），*A four-field mixed formulation for incompressible finite elasticity*，[DOI](https://doi.org/10.1016/j.cma.2025.118082)、[原文](https://arxiv.org/html/2503.00989v2) | F；非线性体积约束、独立梯度/应力及局部消元；未实现刀具耦合 |
| R38 | Karabelas等（2020，在线2019），*Versatile stabilized finite element formulations for nearly and fully incompressible solid mechanics*，[出版全文](https://link.springer.com/article/10.1007/s00466-019-01760-w) | F，本轮新增；MINI和压力投影，有限变形假设及计算代价；不保证本地摩擦全程 |
| R39 | Lederer等（2017），*Divergence-free Reconstruction Operators for Pressure-Robust Stokes Discretizations with Continuous Pressure Finite Elements*，[DOI](https://epubs.siam.org/doi/10.1137/16M1089964) | A，本轮新增；稳定配对不自动等于压力鲁棒；Stokes重构未移植 |
| R40 | Tian等（2021预印本），*Mixed displacement-pressure-phase field framework for finite strain fracture of nearly incompressible hyperelastic materials*，[作者稿](https://arxiv.org/abs/2112.00294) | A（本轮复核）；受损相压力退化与Q1/P0、P2/P1比较不能替代当前完整体配对检查 |
| R48 | Shojaeifard等（2026），*Deep Indentation of Soft Solids: A Limit State*，[v4](https://arxiv.org/abs/2606.30971v4) | A；圆杆/Ogden半空间的深压入尺度，不是薄刃早段曲率的验证 |

## 接触、摩擦及非线性求解

| ID | 来源 | 已有阅读范围、用途与限制 |
|---|---|---|
| R05 | Pinto Carvalho等（2022），*An efficient algorithm for rigid/deformable contact interaction based on the dual mortar method*，[DOI](https://doi.org/10.1007/s00466-022-02226-2) | F；有限变形无摩擦，客观摩擦和新裂面需另做 |
| R06 | Khandelwal、Porwal、Wadhawan（2022），*Adaptive quadratic finite element method for the unilateral contact problem*，[作者稿](https://arxiv.org/abs/2209.04906) | F；二次边顶点/中点约束及后验误差；线性无摩擦假设 |
| R08 | Acary、Brémond、Huber（2018），*On solving contact problems with Coulomb friction: formulations and numerical comparisons*，[HAL](https://hal.science/hal-01878539)、[作者稿](https://tripop.inrialpes.fr/people/acary/publications/Acary_Bremond_Huber.pdf) | F；多类离散摩擦算法比较；作者稿题名略异，没有普适最佳算法 |
| R12a | Areias等（2019），*A simple and robust Coulomb frictional algorithm based on 3 additional degrees-of-freedom and smoothing*，[DOI](https://doi.org/10.1016/j.finel.2019.103321) | A；平滑参数延续；全文访问受限，本地尝试不是复现 |
| R12b | Jiang（1999），Fischer–Burmeister相关互补算法论文，[DOI](https://doi.org/10.1287/moor.24.3.529) | A；互补重写的依据；未将全局收敛标题当当前问题保证 |
| R13 | Deuflhard（2011），*Newton Methods for Nonlinear Problems*，[DOI](https://doi.org/10.1007/978-3-642-23899-4)；PETSc NLEQERR | S（专著）/F（官方算法实现）；只核查修正量线搜索，未读完专著 |
| R15 | Hassani、Hild、Ionescu、Sakki，*A mixed finite element method and solution multiplicity for Coulomb frictional contact*，[作者全文](https://www.math.univ-toulouse.fr/~phild/Publications/CMAME3.pdf) | F；多解与摩擦谱。刊年本次不补填，线弹性不证明当前发生分支变化 |
| R16 | Qi、Sun（1998修订稿），*A Survey of Some Nonsmooth Equations and Smoothing Newton Methods*，[原文](https://defengwebsite.github.io/files/qsreview1.pdf) | F；根点/广义导数非奇异等局部前提 |
| R17 | Li等（2023），*Convergent Incremental Potential Contact*，[作者稿](https://arxiv.org/abs/2307.15908)、[v1全文](https://arxiv.org/html/2307.15908v1) | F；§4.1节点积分权重为相邻参考边长之和的一半，§4.5说明max近似的几何误差条件，§9.2.1二维无摩擦平面滑动仍可有离散误差并随边界细分减小。本项目2026-10-23刀具直刃细分据此检验积分集中；不是新本构、解析刀具或当前求解成功的证明 |
| R18 | Giacoma、Dureisseix、Gravouil（2016），*An efficient quasi-optimal space-time PGD application to frictional contact mechanics*，[DOI](https://doi.org/10.1186/s40323-016-0067-7) | F；LATIN局部—整体分解，非现成切割宿主 |
| R19 | Both等（2022），*Iterative splitting schemes for a soft material poromechanics model*，[DOI](https://doi.org/10.1016/j.cma.2021.114183) | F；借鉴稳定化耦合分裂；孔压并非接触压力 |
| R20 | Degroote等（2010），*Performance of partitioned procedures in fluid–structure interaction*，[DOI](https://doi.org/10.1016/j.compstruc.2009.12.006) | A（论文）/F（preCICE文档）；界面加速依赖内层收敛 |
| R21 | Acary、Armand、Nguyen（2026），*Solving contact problems with Coulomb friction using an interior point algorithm and path following techniques*，[DOI](https://doi.org/10.1016/j.cma.2026.119227) | F；用户原PDF已核查；平方方程、ANM符号与原始摩擦终验收限制另有记录 |
| R22 | Gfrerer等（2022），*On the SCD semismooth* Newton method for generalized equations with application to a class of static contact problems with Coulomb friction*，[原文](https://arxiv.org/pdf/2205.15129) | F；多值图线性化；本地试验不能宣称论文全套复现 |
| R23 | PETSc TSPSEUDO及其引用的Kelley–Keyes（1998）、Coffey–Kelley–Keyes（2002） | F（[文档](https://petsc.org/main/manualpages/TS/TSPSEUDO/)）/S（论文）；辅助时间求稳态，未通读所引两篇论文 |
| R24 | Dickson、Kelley、Ipsen、Kevrekidis（2007），*Condition Estimates for Pseudo-Arclength Continuation*，[DOI](https://doi.org/10.1137/060654384) | F；简单折点条件，非光滑接触不能直接继承 |
| R25 | Jong（2012），*A Newton-Fixed Point Homotopy Algorithm For Nonlinear Complementarity Problems With Generalized Monotonicity*，[作者稿](https://optimization-online.org/wp-content/uploads/2012/08/3562.pdf) | F；路径可延续不等于能到达目标 |
| R41 | Chen等（2025），*Offset Geometric Contact*，[DOI](https://doi.org/10.1145/3731205)、[作者页](https://graphics.cs.utah.edu/research/projects/ogc/) | F；本地与原IPC势/PTC组合在特征切换处失败，候选未通过准入；非论文完整算法否定 |
| R42 | Li等（2020），*Incremental Potential Contact*，[主文](https://ipc-sim.github.io/file/IPC-paper-350ppi.pdf)、[补充](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf) | F；屏障/摩擦lagging。原文明确lagging无收敛保证，不以冻结内层残差代替真平衡 |
| R43 | Gee等（2009），*Pseudo-transient continuation for nonlinear transient elasticity*，[DOI](https://doi.org/10.1002/nme.2527) | A/方法来源；弹性PTC应用，非本项目全程稳定证明 |
| R44 | *A Noise-Tolerant Quasi-Newton Algorithm for Unconstrained Optimization*（2020预印本），[作者稿](https://arxiv.org/abs/2010.04352) | A/方法来源；近收敛能量相减与线搜索噪声，不放松真残差 |
| R45 | Gast等（2015），*Optimization Integrator for Large Time Steps*，[原文](https://math.ucdavis.edu/~jteran/papers/GSSJT15.pdf)，DOI 10.1109/TVCG.2015.2459687 | F；不定方向、截断CG、下降方向；未采用其完整动力学算法 |
| R46 | Fernández-Fernández、Löschner、Bender，*Progressively Projected Newton's Method*（2025初版/2026修订），[v2](https://arxiv.org/abs/2505.21013v2) | F（相关结论）；投影改变全局切线；本地hybrid并非完整PPN实现 |
| R49 | Zimmerman、Ateshian（2018），*A Surface-to-Surface Finite Element Algorithm for Large Deformation Frictional Contact in FEBio*，[DOI](https://doi.org/10.1115/1.4040497) | A；官方模型库提供候选基准；未取得完整论文，不声称复现其全部结果 |

## 实验辨识、位移与测量边界

| ID | 来源 | 已有阅读范围、用途与限制 |
|---|---|---|
| R26 | Oddes、Solav（2023），*Identifiability of soft tissue constitutive parameters from in-vivo macro-indentation*，[摘要](https://pubmed.ncbi.nlm.nih.gov/36801779/)，DOI 10.1016/j.jmbbm.2023.105708 | A；力＋位移有助参数识别，不保证所有参数唯一 |
| R27 | Rokoš等（2018），*On Micromechanical Parameter Identification With Integrated DIC and the Role of Accuracy in Kinematic Boundary Conditions*，[作者稿](https://arxiv.org/abs/1810.08245)，[DOI](https://doi.org/10.1016/j.ijsolstr.2018.04.004) | F；边界运动误差污染材料反演，不能混用输入与独立验证 |
| R50 | Tayeb等（2019），*Identification of Constitutive Parameters Governing the Hyperelastic Response of Rubber by Using Full-field Measurement and the Virtual Fields Method*，[作者稿](https://arxiv.org/abs/1907.02687) | A；全场识别路线，不等于单视角切割光流能唯一识别G/μ |
| R51 | iDICs（2025），*A Good Practices Guide for Digital Image Correlation*, Edition 2，[官方](https://idics.org/guide/)、[DOI](https://doi.org/10.32720/idics/gpg.ed2) | 官方指南；设置/相关/应变质量要求，当前光流未宣称满足全部指南 |

## R54–R55：前裂近线性的定向补查（2026-10-14）

| ID | 来源 | 已有阅读范围、用途与限制 |
|---|---|---|
| R54 | Mitsomwang、Nagasawa、Kuroiwa、Fukushima（2014），*Deformation Analysis of Silicone Rubber Sheet Subjected to Keen WC Blade Indentation*，IJAT8(5),761–772；[DOI10.20965/ijat.2014.p0761](https://www.fujipress.jp/ijat/au/ijate000800050761/) | F（第3节、图14/17/18目视）；无损伤大应变平面应变先例，材料/垫层与当前目标不同 |
| R55 | Ciavarella、Hills、Monno（1998），*Contact problems for a wedge with rounded apex*，IJMS40(10),977–988；[机构库](https://eprints.soton.ac.uk/23252/)，DOI10.1016/S0020-7403(97)00141-0 | A及可检索首页；完整PDF获取失败。浅楔/线弹性极限线索，未用于给当前模型套系数 |

## 补充论文、已核读方法及官方资料

| 来源 | 已有阅读范围与任务边界 |
|---|---|
| R52 Tan、Kang、Patterson (2016), *Experimental investigation on surface deformation of soft half plane indented by rigid wedge*, DOI 10.1007/s10483-016-2139-9，[出版PDF](https://pubs-en.cstam.org.cn/data/article/amm/preview/pdf/20161007.pdf) | F：摘要、§1、§4.1文本；未目视核查图8。DIC给出总力以外的空间判据；不移植其1.68 mm半径及失效深度。 |
| R53 Sichetti等 (2025), *High-Order Continuous Geometrical Validity*，[DOI](https://doi.org/10.1145/3745763) | F：作者v2 HTML §7–8.3；高阶映射、积分和浮点保守性。没有移植全文算法。 |
| Reese & Govindjee (1998), *A theory of finite viscoelasticity and numerical aspects*，[DOI](https://doi.org/10.1016/S0020-7683(97)00217-5)、[大学托管全文](https://giref.ulaval.ca/~deteix/bois/documents_references/article4.pdf) | F：§2.1式20–25与§3。乘法分解、主轴返回与有限黏弹流动背景；本地非线性流动候选不是BB复现，未移植参数。 |
| Bergström & Boyce (2001), *Constitutive modeling of the time-dependent and cyclic loading of elastomers and application to soft biological tissues*，[DOI](https://doi.org/10.1016/S0167-6636(01)00070-9) | A/出版节选；共同材料需跨模式/循环验证。当前单支路Maxwell并非其完整网络。 |
| Jerrams (2005), *Friction and adhesion in rigid surface indentation of nitrile rubber*，[机构摘要](https://arrow.tudublin.ie/cerart/2/)，DOI10.1016/j.matdes.2004.02.012 | A；HNBR钢压头，力、松弛、黏滑和表面轮廓联合辨别。不证明当前CR界面应增加黏附或μ过大。 |
| Persson & Volokitin (2006), *Rubber friction on smooth surfaces*，[作者稿](https://arxiv.org/abs/cond-mat/0607084)，DOI10.1140/epje/i2006-10045-9 | A及引言；支持速度/温度相关界面可能性，不直接移植到强变形锋利刀具。 |
| Reyssat等 (2012), *Slicing Softly with Shear*，[DOI](https://doi.org/10.1103/PhysRevLett.109.244301)、[原文](https://softmath.seas.harvard.edu/wp-content/uploads/2019/10/2012-24.pdf) | F：图1、式1–4；剪切辅助切割。沿刃拉切与面内斜压不同，不能直接套法向降力系数。 |
| Atkins & Xu (2005), *Slicing of soft flexible solids with industrial applications*，[DOI](https://doi.org/10.1016/j.ijmecsci.2005.01.013) | A/引言；切割功率与拉切机理，非完整体压入载荷公式。 |
| Lengiewicz等 (2020), *Finite deformations govern the anisotropic shear-induced area reduction of soft elastic contacts*，[DOI](https://doi.org/10.1016/j.jmps.2020.104056)、[作者摘要](https://arxiv.org/abs/2005.02107) | A；剪切与接触面积几何耦合，不能推出本项目固定角度降力。 |
| Menga等 (2021), *Exploring the effect of geometric coupling on friction and energy dissipation in rough contacts of elastic and viscoelastic coatings*，[DOI](https://doi.org/10.1016/j.jmps.2020.104273)、[作者摘要](https://arxiv.org/abs/2006.16133) | A；区分几何耦合和耗散。粗糙涂层参数不移植到刀具。 |
| Duong & Sauer (2019), *A concise frictional contact formulation based on surface potentials and isogeometric discretization*，[DOI](https://doi.org/10.1007/s00466-019-01689-0)、[全文HTML](https://arxiv.org/html/1808.10420v2) | F：§3.1–3.4、§4.1–4.3及线性化结构。区分当前投影、旧相互作用点、滑移点；统一面积积分与完整切线。本地IPC传输候选未实现论文运动摩擦锥。 |
| Tur等 (2012), *2D contact smooth formulation based on the mortar method*，[DOI](https://doi.org/10.1016/j.cma.2012.08.002) | A/引言；C1几何、mortar积分与一致线性化整体处理。全文未获，且为无摩擦，不能认证本项目摩擦历史。 |
| Huang等 (2025), *Geometric Contact Potential*，[DOI](https://doi.org/10.1145/3731142)、[作者稿](https://cims.nyu.edu/gcl/papers/2025-Contact.pdf)、[官方教程](https://ipctk.xyz/tutorials/gcp.html) | 官方公式、v1.6源码和本地接口实测；未宣称逐页读完PDF。方向性/局部最近条件与CCD联合，不是固定k-ring删对。独立候选未完成共同长程资格。 |
| Abide等 (2022), *Unified primal-dual active set method for dynamic frictional contact problems*，[全文](https://link.springer.com/article/10.1186/s13663-022-00729-4) | F：活动集/投影框架；本地只借鉴分支处理，不等同动力学算法复现。 |
| Grippo、Lampariello、Lucidi (1986), *A Nonmonotone Line Search Technique for Newton’s Method*，[DOI](https://doi.org/10.1137/0723046) | A及后续官方说明；有限窗口启发式不继承当前非光滑问题的全局收敛保证。 |
| Ligurský & Renard (2014), *A continuation problem for computing solutions of discretised evolution problems with application to plane quasi-static contact problems with friction*，[DOI](https://doi.org/10.1016/j.cma.2014.07.011) | A及背景；分段光滑预测校正，不保证本项目存在可达平衡。 |
| Haslinger等 (2018), *Nonsmooth continuation of parameter dependent static contact problems with Coulomb friction*，[DOI](https://doi.org/10.1016/j.matcom.2017.08.001)、[作者稿](https://homel.vsb.cz/~kuc14/papers_in_progress/contModelling.pdf) | F：引言和§5；摩擦参数路径的转向、出支与分离分支。同一连通支路不覆盖所有解，作者稿页码与正式版不同。 |
| McCarthy、Ní Annaidh、Gilchrist (2010), *On the sharpness of straight edge blades in cutting soft solids: Part II – Analysis of blade geometry*，[DOI](https://doi.org/10.1016/j.engfracmech.2009.10.003)、[机构全文](https://researchrepository.ucd.ie/server/api/core/bitstreams/ca936b39-6f89-4acb-90e3-f203da41eae8/content) | F：摘要、问题定义、§4.2/4.3。区分同深度反力与起裂力；1–5 μm聚氨酯体系结果不是99755半径实测。 |
| Ceglie等 (2026，2025在线), *Contact area shrinkage and increase in wavy frictional sliding contacts*，[DOI](https://doi.org/10.1016/j.jmps.2025.106389)、[机构全文](https://iris.unisalento.it/retrieve/533bbb63-a320-4e7c-87fc-4720e8ce5a05/1-s2.0-S0022509625003631-main.contactarea.pdf) | F：§2式6、§4.3；有限弹性界面剪切强度上限。PDMS/玻璃参数不移植，本地节点压力候选非原文完整算法。 |
| Chi等 (2016), *A paradigm for higher-order polygonal elements in finite elasticity using a gradient correction scheme*，[DOI](https://doi.org/10.1016/j.cma.2015.12.025)、[作者稿](https://paulino.princeton.edu/journal_papers/2016/CMAME_16_AParadigmForHigherOrder.pdf) | F相关积分讨论；借鉴独立高阶/正权重积分检验，不能直接要求P2使用其多边形梯度修正。 |
| Chi等 (2015), *Polygonal finite elements for finite elasticity*，[DOI](https://doi.org/10.1002/nme.4802)、[原文](https://pamies.cee.illinois.edu/assets/pdf/IJNME_2015.pdf) | F相关章节；局部强变形可早于全局大位移导致离散困难，不允许据此接受翻转单元。 |
| Hartmann & Gilbert (2021), Computational Mechanics68,633–650，[出版全文](https://link.springer.com/article/10.1007/s00466-021-01998-3) | F：摘要、§2、§4.1–4.2、§5.1–5.2网页文字，未逐式核查图片公式。有限黏弹内部状态敏感性、反力/应变与可辨识性；未执行其完整反演。 |

### 官方实现与原材料资料

| 来源 | 已核读内容及限制 |
|---|---|
| [IPC Toolkit](https://ipctk.xyz/tutorials/getting_started.html)、[IPC技术补充](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf)、[项目说明](https://github.com/ipc-sim/IPC/wiki) | 二维近接触屏障、CCD、冻结摩擦帧/法向载荷的内势。内层能量下降不等于刷新摩擦后的平衡，lagging本身无普适收敛保证。 |
| [PETSc SNES](https://petsc.org/release/manual/snes/)、[SNESNEWTONTR](https://petsc.org/release/manualpages/SNES/SNESNEWTONTR/)、[SNESNGMRES](https://petsc.org/release/manualpages/SNES/SNESNGMRES/)、[SNESASPIN](https://petsc.org/release/manualpages/SNES/SNESASPIN/) | 残差全局化、历史组合与非线性预条件的组织依据；当前并未接入PETSc或实现ASPIN/RASPEN。 |
| [GetFEM大滑移接触](https://getfem.org/userdoc/model_contact_friction_large_sliding.html) | 参考面积分、刚性障碍和库仑投影；其节点brick标注不可用，不能误当现成可用入口。 |
| [MOOSE乘子Petrov–Galerkin](https://mooseframework.inl.gov/syntax/Constraints/Petrov-Galerkin_approach_for_Lagrange_multipliers.html) | 乘子插值/测试空间与加权间隙区别，不等于自动排除点值穿透。 |
| [FEBio sliding-elastic](https://febiosoftware.github.io/febio-docs/features/features/solid_surfaceinteraction_sliding-elastic/)、[官方模型库](https://repo.febio.org/permalink/project/68) | 增广接触、非对称刚度及补丁/Hertz等基准，未在本项目运行其模型。 |
| [SciPy splu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html)、[NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss/latest/index.html) | 稀疏线性子问题的后端能力；不迁移材料/IPC/CCD，也不自动解决非线性方向困难。本项目未启用GPU。 |
| [dolfin-adjoint](https://dolfin-adjoint.github.io/dolfin-adjoint/documentation/maths/2-problem.html) | 已读问题定义/求解方法；时间相关目标依赖全部状态层，区别全历史导数与条件偏导。 |
| [Bleyer广义平面应变教程](https://bleyerj.github.io/comet-fenicsx/tours/linear_problems/generalized_plane_strain/generalized_plane_strain.html) | 位移场、整体面外自由度与合力约束正文；常数GPS、局部PS与有限宽三维不同。 |
| [COMSOL二维约束说明](https://www.comsol.com/blogs/what-is-the-difference-between-plane-stress-and-plane-strain/)、[Solid Mechanics接口](https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_solid.07.002.html?type=ext) | 2026-10-24核读Plane Strain/Plane Stress及2D Approximation：F33约束、非零面外应力、自由侧面与内部截面的差别。支持视频侧面可比性审查，不证明本试验三维效应大小，也不支持直接把全域模型改成平面应力。 |
| [Abaqus时间域黏弹性](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-timevisco.htm) | 瞬时/长期参数定义与材料时间；不证明当前CR参数已识别。 |
| [COMSOL氯丁橡胶压缩基准](https://www.comsol.com/model/download/1389851/models.nsm.chloroprene_rubber_compression_test.pdf)、[BB模型技术说明](https://www.ansys.com/content/dam/resource-center/datasheet/bergstrom-boyce-model.pdf) | 分别核读2–9页、2–5页文字；平衡/非平衡网络、保持和卸载检验。未照搬材料或复现，COMSOL尺寸文字需复核。 |
| [Maxell Kureha CB245NE报告](https://www.kurehae.maxell.co.jp/dl/test_report/CB245NE.pdf) | 原图核对23°C静态剪切模量0.47 MPa及“测定值非规格值”。不是本批样品G0/G∞标定；99755尺寸仍以用户截图为准。 |

## 2026-10-24：完整体下压响应的定向证据核查

本节为针对当前力、位移、接触及能量解释的定向核查，不是穷尽检索或首创性检索。已重新打开下列原文、作者稿或机构摘要；其他条目的阅读等级保持原记录。检索日期为2026-10-24，检索入口为网页学术检索，交叉核对出版社、作者/机构库及PubMed记录；未以聚合网站内容替代原文结论。使用的英文题名查询包括：`Cutting soft materials: how material differences shape the response`、`Cutting Characteristics of Silicone Rubber Sheet`、`Slicing Softly with Shear`、`Viscoelasticity in simple indentation-cycle experiments`、`Friction and adhesion in rigid surface indentation of nitrile rubber`、`Rubber friction on smooth surfaces`、`Finite deformations govern the anisotropic shear-induced area reduction`、`Identifiability of soft tissue constitutive parameters from in-vivo macro-indentation`、`Phase-field modelling and analysis of rate-dependent fracture phenomena at finite deformation`。逐项证据见本地分析结果中的`mechanism_review/literature_evidence.json`。

| 文献与本次核读位置 | 可支持的解释 | 对当前项目的限制 |
|---|---|---|
| R30 Kojima等2020，[出版全文](https://www.jstage.jst.go.jp/article/sosei/61/708/61_1/_pdf)，第3页§3.1、第4–5页§3.3 | 在其硅胶薄片/硬硅胶垫层/13.9°刀工况中，首次转折前的无表面损伤压入近线性，斜率对所测速度近乎不敏感；转折后的损伤由图像辅助确认。 | 近线性不是黏弹主导的唯一指纹。其尺寸、支承、硅胶与当前CR不同；不能直接移植斜率或把当前力曲线转折等同起裂。 |
| R04 Moreno-Mateos与Steinmann2026，[正式版机构全文](https://d-nb.info/1386915076/34)，第3–8页、表1 | 弹性体可从压入平滑进入切割；表1将库仑摩擦列为压入和紧密接触的重要作用。裂后低压接触的切向阻力可主要来自黏附/磨损。 | 不得从其裂后结论推出当前完整体高压接触可忽略库仑摩擦。当前模型没有裂面、黏附或磨损，不能解释整个裂后下压过程。 |
| Reyssat等2012，[作者托管全文](https://softmath.seas.harvard.edu/wp-content/uploads/2019/10/2012-24.pdf)，PDF第4–7页为论文正文，前3页为评论 | 沿切割线施加剪切可改变全局压缩与局部拉伸分布，降低达到局部起裂条件的法向载荷；滑动会限制剪切传递。 | 该文沿金属丝轴向拉切；当前为截面内倾斜试样竖直下刀。起裂时载荷随角度变化不等于相同竖直行程下Fz必须单调。 |
| R29 Efremov等2020，[出版全文](https://www.nature.com/articles/s41598-020-70361-y.pdf)，第4–6、11–13页 | 压入曲线形状受几何、加载时间和松弛谱共同影响；加载/卸载及不同速率有助区分响应。 | 其线性黏弹、小压入和恒泊松比假设不等同当前有限应变、固定K的模型；不能用总力乘指数衰减替代FE历史。 |
| Jerrams2005，[作者机构摘要](https://arrow.tudublin.ie/cerart/2/)，本次仍为A | HNBR/钢压头实验联合测量载荷、松弛、黏附与表面轮廓；界面黏附可在短期载荷松弛时保持。 | 仅核实摘要，未取得完整原文；其材料、压头和粗糙度不同。当前固定库仑系数不等于已识别真实界面黏附。 |
| Persson与Volokitin2006，[作者稿](https://arxiv.org/pdf/cond-mat/0607084)，摘要、引言、结论 | 光滑硬基底上橡胶界面剪应力可随滑动速度和温度变化，微观脱钉与体黏弹性相联系。 | 不能由此指定当前刀具μ的速度/温度函数；需界面信息或独立实验。宏观正则平滑区不是这种微观机制的直接观测。 |
| Lengiewicz等2020，[作者稿](https://arxiv.org/pdf/2005.02107)，§3、§5 | 非黏附、非黏性有限应变模型及示踪实验支持剪切下局部脱离、进入接触和面内变形共同改变接触面积。 | 球/平面、Tresca界面与当前楔刀/库仑不同；表面形变或接触缩小本身不能证明黏附，更不能移植面积比例。 |
| R26 Oddes与Solav2023，[PubMed原始摘要记录](https://pubmed.ncbi.nlm.nih.gov/36801779/)，本次仍为A | 在其合成压入辨识研究中，仅力–深度不足以可靠识别两参数；加入表面位移改善辨识，但部分本构仍不唯一。 | 不是本项目不可辨识的证明；当前侧视光流的标定、遮挡、同步与初始参考资格仍需独立检查。 |
| R03 Dammaß等2023，[出版全文](https://link.springer.com/article/10.1007/s00466-023-02310-1)，§2.2、§3.2 | 乘法分解有限黏弹框架区分平衡储能、非平衡储能与黏性耗散；内变量必须随真实加载历史演化。 | 本地主线不是该论文完整Ogden/相场模型；耗散不是可恢复储能，不能直接把累计耗散作为当前未定义的起裂阈值。 |

上述内容支持有限变形、黏弹性与接触共同参与的解释框架，不确定本批实验各机制的占比。当前模型中的“黏着”应明确指正则库仑界面的近零相对滑动状态，不等于具有吸引牵引或界面分离功的“黏附”。力分解是同一解的账目，不等于去掉某物理机制后的因果贡献。

可研究的项目问题包括：同一材料在不同角度的表面材料点迁移与局部停留历史是否改变松弛/受力分配；近线性是否来自接触几何增刚与历史松弛的相互抵消；是否存在可由力和表面运动共同支持的接触状态转变。这些目前属于待证伪假说或项目条件下的数值观察，不是经独立实验确认的首次发现。首创性必须针对具体结论进行更完整的先行工作检索，并排除数值接触作用和参数补偿。

## 2026-10-24：视频变形与高精度比较的定向依据

2026-10-25复核了R51的iDICs官方页面及下表DIC Challenge 2.0的NIST摘要；没有新增全文阅读或重复文献副本。对应[视频—FE共同观测规范](../docs/video_fe_comparison.md)采用同材料点、同标距及已知场图像检验；这些是项目比较设计，不表示已完成光流准确度认证。

2026-10-26[数学模型说明](../docs/mathematical_model.md)以本地源码及安装的ipctk 1.6.0为公式依据，核查物理屏障归一化、摩擦平滑、材料返回和真实切线。另打开[官方v1.6.0版本页](https://github.com/ipc-sim/ipc-toolkit/tree/v1.6.0)与R17作者预印本记录确认来源身份；Reese–Govindjee DOI直连本轮未取得正文，不新增全文核读声明。此次不是文献穷尽检索或一般收敛性证明。

本次仅针对观测误差、侧面可比性及离散资格，非系统综述。详细核读范围与来源指纹见本地`outputs/video_fe_1mm_20261024/deformation_review/model_review/sources.json`。既有COMSOL二维约束说明和Lengiewicz软接触论文继续使用上方条目，不重复建文献副本。

| 来源与实际阅读范围 | 支持内容与适用限制 |
|---|---|
| Sutton等（2008），*The effect of out-of-plane motion on 2D and 3D digital image correlation measurements*，[DOI](https://doi.org/10.1016/j.optlaseng.2008.05.005)，出版社索引摘要/片段；直连全文403，未宣称读完 | 单相机面外平移/转动可能进入表观面内运动及梯度。支持保留投影不确定性，不给出本视频误差的大小、符号或可直接应用的校正。 |
| Reu等（2022），*DIC Challenge 2.0: Developing Images and Guidelines for Evaluating Accuracy and Resolution of 2D Analyses: Focus on the Metrological Efficiency Indicator*，[NIST原始记录](https://www.nist.gov/publications/dic-challenge-20-developing-images-and-guidelines-evaluating-accuracy-and-resolution-2d)，DOI 10.1007/s11340-021-00806-6；核读机构摘要，PDF未完整核读 | 已知合成图像场可检验位移/应变准确度、空间分辨率及噪声权衡。支持向原光流管线输入已知变形的检查，不认证本项目光流算法，也不排除真实三维投影。 |
| [SIMULIA Abaqus：Hyperelastic Behavior of Rubberlike Materials](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-hyperelastic.htm)，官方文档中的可压缩性、材料试验与混合单元相关章节 | 受约束橡胶对体积响应敏感，近不可压场需要合适离散。支持跨模式/K/混合离散资格检查，不证明当前P1已锁定；超弹混合算法不能未经验证直接等同当前有限应变Maxwell接触实现。 |

当前项目已观察到相同有限标距下近刀尖投影伸长偏大、压缩偏多；原因仍待在测量过滤、有限宽侧面、接触传力和材料/离散之间区分。这是条件数值—实验比较结果，不是已认证的新物理发现。现有全域平面应力对照的局限仍保留，不能因侧面自由就直接替换整个模型。

## 2026-10-28 混合模式与自由裂路补充

以下本轮核对出版方或作者机构的摘要/元数据（A），不冒充完整复现。实现推导、选型与限制见[断裂说明第12–13节](../../../docs/fracture_implementation.md)。新代码不使用切割实验力或57°未裂下限识别参数。

| 来源 | 对当前选择的作用 |
|---|---|
| Ortiz、Pandolfi（1999），*Finite-deformation irreversible cohesive elements for three-dimensional crack-propagation analysis*，IJNME 44(9),1267–1282，[作者机构记录](https://authors.library.caltech.edu/records/pryf8-c1t67) | 有限变形不可逆内聚元素能形成显式自由表面。支持方法家族选择；当前二维全边网络、有限间隙与IPC不是论文逐式复现。 |
| Serpieri、Sacco、Alfano（2015），*A thermodynamically consistent derivation of a frictional-damage cohesive-zone model with different mode I and mode II fracture energies*，EJM A/Solids 49,13–25，[机构记录](https://bura.brunel.ac.uk/handle/2438/10020)，DOI 10.1016/j.euromechsol.2014.06.006 | 所选单损伤、等效开口、线性软化闭合形式采用共同Gc；不同模式总耗能需额外机制。没有把随当前模式变化的阈值拼入不可逆历史。 |
| Wu（2017），*A unified phase-field theory for the mechanics of damage and quasi-brittle failure*，JMPS103,72–99，[出版方](https://www.sciencedirect.com/science/article/pii/S0022509616308341)，DOI 10.1016/j.jmps.2017.03.015 | 保留PF-CZM为替代路线。本次为复用真实裂面IPC选择显式内聚网络；未实现该相场模型，亦不声称相场不能处理接触。 |

### 本地原文映射

PDF仅留本地，不发布原文或原图。页码必须按实际版本使用；以下原文件名保持不变。

| 条目 | 本地文件 |
|---|---|
| R01 | `2.pdf` |
| R02 | `A unified phase-field model of fracture in viscoelastic.pdf` |
| R03 | `updated phase.pdf`；[独立复现](../reproductions/dammass2023/README.md) |
| R04 | `MorenoMateos_Steinmann_2025_preprint.pdf`，预印本与2026正式版分开引用 |
| R05 | `PintoCarvalho_et_al_2022_dual_mortar.pdf` |
| R06 | `Khandelwal_Porwal_Wadhawan_2022_quadratic_contact.pdf` |
| R07 | `Maas_et_al_2011_nodal_integration_FEBio.pdf` |
| R08 | `Acary_Bremond_Huber_2018_contact_solvers_author.pdf`，作者版题名与正式章略异 |
| R09 | `LopezPamies_et_al_2025_nucleation_review.pdf` |
| R10 | `Kamarei_Breedlove_LopezPamies_2025_preprint.pdf` |
| R11 | `LopezPamies_et_al_2026_constrained_fracture_preprint.pdf` |
| R21 | `solvingcontactproblemswithCoulombFriction.pdf` |
| R30 | `Kojima_2020_silicone_wedge_velocity.pdf` |
| R54 | `Mitsomwang_2014_silicone_blade.pdf` |
| 历史保存原文 | `nsf_contact_saved.pdf`，原索引未提供可靠身份映射，不凭文件名补造题录 |
| 厂家原始报告 | `CB245NE.pdf` |

Watson同伦综述仅保留为未确认检索线索（U），不补造题名/DOI。Simo–Taylor–Pister、Simo–Armero在他文中追溯引用，不冒充独立核读。Mathews等2024的线性深度–力公式用于金属压痕弹性恢复，已排除用于本项目橡胶加载段。引用保存不提升阅读或本地验证级别。

### 2026-10-29 零间隙、动力学与卸载假设补充

Fz专项追加核查：R04正式版元数据与检索正文、[2025作者预印本全文](https://arxiv.org/html/2507.13565v1)第2.3、4.2、5.3、5.7节分别使用，不能把预印本公式疏漏或正则假设当作实测物性。本次正式版PDF下载未成功，不新增虚假的本地原文映射。

- Gonzalez（2000），*Exact energy and momentum conserving algorithms for general models in nonlinear elasticity*，CMAME190,1763–1783，[DOI](https://doi.org/10.1016/S0045-7825(00)00189-4)、[作者说明](https://web.ma.utexas.edu/users/og/numerics.html)。核查出版摘要/数学框架与作者说明，支持C空间离散梯度路线；本项目Maxwell分裂、IPC时间求积和半空间约束为自行推导与验证，不冒称全文算法的完整复现或继承其所有守恒结论。
- Leguillon（2002），*Strength or toughness? A criterion for crack onset at a notch*，EJM A/Solids21,61–72，[DOI](https://doi.org/10.1016/S0997-7538(01)01184-6)；核查出版摘要及[Leguillon与Yosibash后续原文中的准则复述](https://tzin.bgu.ac.il/me/yosibash/papers/IJF_03_Leguillon.pdf)。同一有限裂增量需同时满足强度与能量条件；当前非局部牵引规则没有虚拟裂增量重平衡，不能称为该耦合准则。

- Vavasis、Papoulia、Hirmand，*Second-order cone interior-point method for quasistatic and moderate dynamic cohesive fracture*，CMAME，DOI [10.1016/j.cma.2019.112633](https://doi.org/10.1016/j.cma.2019.112633)，[预印本](https://arxiv.org/abs/1909.10641)。支持初始刚性内聚、非光滑势与隐式动态约束的路线；当前实现不是其SOCP算法复现。
- Collins-Craft、Bourrier、Acary（2022），*On the formulation and implementation of extrinsic cohesive zone models with contact*，CMAME400，115545，[DOI](https://doi.org/10.1016/j.cma.2022.115545)。本轮核读出版摘要与[作者的动态/冲击源代码](https://github.com/nickcollins-craft/On-the-formulation-and-implementation-of-extrinsic-cohesive-zone-models-with-contact/blob/main/Python/spring_czm-dynamic_with_impacts.py)，源文件Git blob为`7eeeda2a70d3f35aaec5b2cf799cf53eb6b11b21`；未执行作者程序。代码中内聚冲量按`−h σc β`，卸载时β保持，区别于割线回原点。HAL正文受访问限制，不能声称逐式核读全文；原代码为线性弹性LCP框架，不能把其能量定理直接移给本项目的有限应变Maxwell/IPC。
- Collins-Craft、Acary，*On the formulation and implementation of mixed mode I and mode II extrinsic cohesive zone models with contact and friction*，[作者代码](https://github.com/nickcollins-craft/mixed-mode-i-and-mode-ii-extrinsic-cohesive-zone-models-with-contact-and-friction)、[2025归档](https://doi.org/10.5281/zenodo.16911870)。核查了法向内聚/接触、摩擦及能量处理入口；代码使用Akantu/Siconos，尚未接入本项目。它是一般裂面接触与混合模式的后续候选，不是已验证的橡胶切割模型。
- Acary、Collins-Craft（2025），*On the Moreau–Jean scheme with the Frémond impact law: energy conservation and dissipation properties for elastodynamics with contact, impact and friction*，[机构原文入口](https://researchportal.murdoch.edu.au/esploro/outputs/journalArticle/On-the-MoreauJean-scheme-with-the/991005884848507891)，DOI 10.46298/jtcam.13480。本轮通过机构开放PDF核读第1–3节与结论：式(56–67)以加权速度离散冲量/摩擦，能量论证针对线性Lagrange体动力学；有限变形势、一般材料与大滑移仍需扩展。不能把平均速度更新直接套在当前位置互补/端点Maxwell上就宣称获得其能量保证。未执行该论文的Siconos程序。

### 2026-10-29 有限裂增量双条件实施补充

- Doitrand、Martin、Leguillon（2020），*Numerical implementation of the coupled criterion: Matched asymptotic and full finite element approaches*，[出版入口](https://www.sciencedirect.com/science/article/pii/S0168874X19303002)，DOI 10.1016/j.finel.2019.103344。核查出版摘要的完整FE前后问题路线；未取得全文，不虚构其网格门槛。当前CC/CZM组合与原文弹性示例不同。
- Herrera-Garrido、Mantič、Leguillon（2025），*Application of the coupled criterion to interface crack growth in the end-notched flexure test considering friction between the crack faces*，[出版全文](https://link.springer.com/article/10.1007/s10704-025-00866-y)，DOI 10.1007/s10704-025-00866-y。核读第2.2、3节：固定边界的释放储能须扣摩擦功，不同裂面摩擦转变假设须区分。本项目冻结Maxwell历史/直线虚拟摩擦路径/动态CZM是自行提出的扩展，尚无原文定理保证。

### 2026-10-30 第六轮：释放路线的适用性复核

以下为本轮实际核读范围，补充而不抹去上面的历史阅读状态；未安装作者环境、
未运行作者算例，也没有把论文参数作为本批橡胶的实测输入。

| 来源 | 核读范围与本工程边界 |
|---|---|
| Moreno-Mateos–Steinmann，R04 [正式版](https://www.nature.com/articles/s41524-025-01869-y) | 正式开放PDF方法约10–14页：Smith–Ferrante界面、损伤历史、接触与摩擦、FEniCSx P1、伪时间自适应和式31的界面黏性正则项。其伪时间/正则化与切割曲线标定不等于真实质量动态释放；本轮不移植η_reg或实验拟合。正式PDF留本轮证据目录。 |
| Herrera-Garrido等 [全文](https://link.springer.com/article/10.1007/s10704-025-00866-y) | 第2.2节式7–17及NM1/NM2：固定加载释放过程中的摩擦、历史和动能假设需分别定义。NM2预载后解开连接；其ENF线性情形的近似一致性不能证明本项目大变形Maxwell直线虚拟路径。 |
| Collins-Craft等2022 [作者冲击代码](https://github.com/nickcollins-craft/On-the-formulation-and-implementation-of-extrinsic-cohesive-zone-models-with-contact/blob/main/Python/spring_czm-dynamic_with_impacts.py) | 重新核读自由飞行冲量、LCP接触分支、内聚冲量及速度/位置更新；README为Python3.8、Siconos4.4环境。HAL正文仍未取得；不声称完整原文复现。 |
| 混合模式作者实现 [release分支](https://github.com/nickcollins-craft/mixed-mode-i-and-mode-ii-extrinsic-cohesive-zone-models-with-contact-and-friction/tree/release) | 核查README、LICENSE.txt及czmlaw_I_II.py的双侧面应力/全求积点插入条件，实际内聚通过外部LCP处理，假内聚切线为零。Apache2.0；Python3.12、Siconos4.5、Akantu5.0.7.post1，后续需独立环境。这里的面级P1核不是该混合模式算法的移植。 |
| Convergent IPC [作者稿](https://arxiv.org/html/2307.15908v1) | 第8.1–8.2节：减小激活距离会使势更尖锐，时间算法可引入压力振荡。当前已有P1/improved-max；固定现场参数扰动只是因果探针，不能证明改变参数后的整条轨迹有效。 |
| Larionov等 [作者稿v3](https://arxiv.org/html/2211.10618v3) | 全隐式摩擦式6、12及IV-E式21–23的完整力分裂对比。当前已耦合完整动量；软约束方法不自动给出严格CCD或本项目能量资格。 |
| Acary–Collins-Craft [开放正式PDF](https://jtcam.episciences.org/15960/pdf) | 第2.1节及3.2节式56–57：线性黏弹动力学M、K、C与接触冲量、加权速度、Frémond冲击律整体定义。不能仅替换时间积分器名称就继承其能量性质；本项目有限应变Maxwell势和大滑移不属于该线性证明。 |
| Kamarei等 [作者稿v1](https://arxiv.org/html/2506.16479v1) | 第2.2、3.2.3、3.4节：本构、强度、断裂能独立约束，黏弹总撕裂能与平衡储能驱动的区别。没有本批材料参数；该相场路线也未自动包含刀具摩擦。 |
