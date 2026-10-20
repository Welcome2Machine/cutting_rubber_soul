# 文献检索记录

2026-10-20实现阶段：核对本机GCP局部原语顺序、原生局部／全局导数散装和二维势尺度。阅读Abide等2022开放全文的接触固定点／半光滑Newton／PDAS框架（DOI10.1186/s13663-022-00729-4）；核对Grippo等1986非单调搜索的出版社书目与后续官方描述（DOI10.1137/0723046），后者未声称阅读全文。采用的是项目分层活动集／有限窗口搜索原型，不继承未验证的全局收敛保证。GCP纯接入及两种全局化在57°原增量均未通过，33°通过；文献存在不是可计算性证明。详见[实现与失败限制](../references/geometric_self_contact_assessment.md)。

2026-10-20接触几何：定向检索Geometric Contact Potential二维、friction、IPC1.5/1.6实现。阅读官方GCP教程的几何作用集合、二维指数、参考自适应dhat、最近特征平滑与接口说明；核对v1.6.0的二维builder、Point2/Edge2、势和官方C1/C2测试。作者全文PDF超出网页工具提取大小，未声称逐页读完；本轮未另存PDF。上游发布说明的完整能力不等于当前Python绑定，本机实测缺adaptive-dhat/smooth-friction接口。固定几何/导数证据与书目集中于[评估文档](../references/geometric_self_contact_assessment.md)，不将库方法存在等同于主线可计算。

2026-10-18角度机理：定向检索“slicing soft materials slice/push ratio normal velocity”“rubber indentation tangential loading normal force coupling”“finite deformation contact area Lengiewicz”“bonded layer geometric coupling friction”。核查Reyssat2012作者托管原文（PDF第4–6页）、Atkins–Xu2005出版社索引提供的摘要/引言、Lengiewicz2020及Menga2021作者接受稿摘要和DOI。2023 angled adhesive indentation页面未取得足够可核查正文，未用其片段支撑核心判断。非系统综述，未声称全文阅读全部来源；采用范围和受控对照见[角度机理](history/topics/angle_force_mechanism.md)。未上传本地实验数据或下载重复PDF。

2026-10-18共同材料/约束/体积/离散验证：重读[Abaqus 2025超弹性官方文档](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-hyperelastic.htm)的可压缩性、瞬时/长期参数及平面试验段落；重开[Ahmadi等平面应力有限应变作者摘要](https://arxiv.org/abs/2410.22562)。用途是区分受限体积响应、偏量松弛和面外自由度，不引入论文的材料参数，不声称复现其算法或采用了Abaqus求解。实测材料仍未经跨模式标定；详见[执行报告](history/topics/cross_mode_verification.md)。没有新增论文PDF或外传实验数据。

2026-10-13近线性验证链：定向检索“plane strain wedge indentation hyperelastic rubber load indentation depth linear finite element thickness effect”和“MINI element finite strain local volume error bubble pressure nearly incompressible”。复核Karabelas等2020出版全文§4.3的J分布与边界波动；核读Tan等2016出版PDF文本摘要、§1、§4.1，新增R52。两篇PDF网页截图请求缓存失败，不声称目视检查图像。复核Kojima等2020§3.3：其前裂推动段斜率几乎不随速度变化，故不能凭橡胶有黏性就断言本项目近线性由松弛导致。三维压入、球形压入及塑性楔入的搜索结果未用于本项目公式或参数。此次采用的是同接触迹空间细化、全场J/位移与总力曲率联合判据，不增加物理参数。

2026-10-13重构实施：直接复核已选Karabelas等的[出版全文](https://link.springer.com/article/10.1007/s00466-019-01760-w)和Fu等的[作者全文v2](https://arxiv.org/html/2503.00989v2)，针对MINI内部自由度、局部消元与有限应变稳定性限制，不开展新的泛化检索。采用标准二维气泡/P1空间；文献的三维性能和四场主方法未移植。实施映射及本地验证记录在[重构说明](history/topics/mini_redesign.md)。

## 2026-10-13：位移—压力配对之后的定向补查与历史整理

本轮先实际执行配对、弯曲与已接受刀尖场的局部平衡检查。由于这些检查尚不能解释整体Fz曲率，补查如下关键词：`Taylor Hood finite elasticity incompressible contact quadratic displacement linear pressure boundary constraint inf sup`、`P2 P1 nearly incompressible finite elasticity quadrature pressure mass lumping Taylor Hood`、`Versatile stabilized finite element formulations`、`Taylor-Hood finite elasticity pressure robust`。另按DOI、作者和题名核对既往Koko、Bercovier–Pironneau、Gee、Zimmerman等记录。

实际采用出版方原文、arXiv作者版本、作者/机构资料及官方软件文档；这次是定向复核，不是完成了跨库穷尽检索的系统综述。新读[Karabelas等出版全文](https://link.springer.com/article/10.1007/s00466-019-01760-w)的空间、凝聚与有限应变限制（卷年2020，在线2019）；复核[Fu等作者原文](https://arxiv.org/html/2503.00989v2)§3–5；新读[Lederer等出版摘要](https://epubs.siam.org/doi/10.1137/16M1089964)。[Tian既往预印本](https://arxiv.org/abs/2112.00294)本轮只核查摘要，HTML失败；Karabelas的PMC入口出现验证码，转用出版社可访问全文。未绕过访问限制，未把搜索摘要冒充全文。

历史资料按主题汇总到[统一目录](../references/catalog.md)，旧R01–R25不改号；原README的详细记录迁入research_history，所有可恢复链接与位置收录source_index。references内12份PDF逐份首页身份核查，页数/哈希记录只留本地。论文参数不覆盖当前材料，不因文献有速度或稳定性结果就自动启动新生产任务。针对性判断见[配对笔记](../references/pairing_review.md)。

2026-09-12：查询 finite precision / Armijo line search / cancellation near Newton convergence。核对 ANSYS 2024R2 理论手册 Newton-Raphson 的收敛与线搜索段，以及 arXiv:2010.04352 的函数误差/噪声容忍线搜索摘要。结合 0.91 mm 原失败试值的能量差与梯度路径功审计，采用自主近收敛保护；未把文献的拟 Newton 全局收敛结论转用为本模型保证。原始 PDF 不另存重复副本。
2026-09-11 核查 0.77 mm：查询 projected Newton tiny line-search steps / indefinite Hessian / modified Newton eigenvalue shift。读取作者托管 Gast 等 2015《Optimization Integrator for Large Time Steps》第 3.1 节（方向、截断 CG、下降角与 Wolfe），以及 Cornell CS4220 2026-04-20 的非凸二次/信赖域讲义。核对逐单元 PN 与全局修正的不同；当前自主实现以 PN 刚度为度量作全局曲率移位，仅修改步矩阵。资料不是橡胶斜切验证，不将动力学收益直接迁移；同前态对照及适用边界记入 solver_status/modeling_roadmap。

2026-09-11 最后全盘核查：定向检索 IPC Toolkit 2D / convergent contact / PolyFEM friction quasi-static，并读取 ipctk.xyz 概览、Getting Started、Convergent Formulation、PolyFEM 首页及其 FrictionForm.cpp 官方源码。确认 Toolkit 非完整求解器、二维碰撞线性边适用范围及完整宿主的滞后摩擦实现。选择完整 IPC/PolyFEM 路线作为当前方案退出后的优先适配方向；未安装或宣称已验证该工具，也未下载重复论文。适用限制补入 R17，实施门槛见 modeling_roadmap.md。

2026-09-11 辅助路径实现：定向检索 pseudo arclength continuation simple folds augmented Jacobian predictor corrector，核对 Dickson/Kelley/Ipsen/Kevrekidis 2007 的 SIAM 书目和作者全文第 1–4 页。使用其增广方程、切向归一化和简单折点条件指导预测—校正；不把光滑路径定理外推至当前库仑非光滑系统。摘要和项目自主残差同伦集中于 R24，原 PDF 未另建本地副本。

2026-09-11 用户补充 2026 IPM-ANM 原文：读取本地 `references/solvingcontactproblemswithCoulombFriction.pdf`，25 页正式排版；首页核实在线日期 2026-07-29。核查第 4–11 页模型、Slater 条件、路径定理与 Algorithm 1；第 12 页样本拆分；第 15–22 页失败分析、ANM、表 3 和成本限制；附录 A/C 的锥代数与递推。PDF 阅读采用 pypdf 抽取和临时 pypdfium2 渲染，目视核查第 4、6、10、17–20、24 页的公式、图与表。对式 (41) 的双线性交叉项和式 (48) 的残差平移做独立代数检查；区分原文记号不一致与作者已确认勘误。R21、参考索引及路线已更新，覆盖此前仅摘要级判断；未将本地全文上传，也未启动计算。临时抽取、渲染和隔离安装的阅读依赖在核查后删除。

2026-09-11 摩擦切换替代设计与跨问题耦合：定向检索 Coulomb friction interior point path following 2026、Convergent Incremental Potential Contact lagged friction、LATIN friction local global、SCD semismooth Newton Coulomb、fluid structure interface quasi Newton、soft material poromechanics stabilized splitting Anderson、PETSc pseudo transient DAE。阅读全文相关章节：Li 等 2023 第 3/6/7/9 节；Giacoma 等 2016 摩擦求解段；Both 等 2022 第 2–5/6.2 节及表 8–9；Gfrerer 等 2022 第 6–7 节。核查 preCICE 和 PETSc 官方文档。Degroote 2010、Acary 等 2026 只取得出版摘要级材料；后者全文未取得，不据此实现或主张有限应变适用性。Rey 2026（DOI 10.5802/crmeca.362）仅核查出版摘要，其特定高摩擦多解例不用于解释本项目 μ=0.35。

自主摘要与适用边界集中于 references/method_notes.md 的 R17–R23；跨问题移植是本项目设计推演，不是论文验证过的切割算法。优先评估真正收敛的摩擦子问题与稳定化外层耦合，完整 IPC 和多值图/内点方法列为不同成本的备选。本轮只更新研究及路线文档，没有更改求解器或启动新算例。

2026-09-11 单点切换收紧：检索 semismooth Newton generalized Jacobian nonsingular、coherent orientation frictional contact。读取 Qi/Defeng Sun 作者托管 1998 综述第 2–4 页，对照广义 Jacobian 凸包、B-导数集合的不同条件及非光滑残差全局化限制；适用边界与自主诊断归入 R16。未把非根点的奇异凸组合称为物理分岔或无解证明。

2026-09-11 定向检索 Coulomb contact mixed integer linearization、frictional contact solution multiplicity。核对 Hassani/Hild/Ionescu/Sakki 作者托管全文的摘要、引言及问题定义，以及 SciPy milp 官方约束、时间预算、状态语义；自主摘要及有限应变适用限制归入 R15。2026 年 interior-point/path-following 接触论文只取得出版摘要，未据此声称实现其算法。没有读取用户保留的实验验证数据。

2026-09-10 返回 P2 旧失效点：查询 smoothing Newton Coulomb friction contact、Siconos friction contact 2D Gauss Seidel、nested Fischer Burmeister mixed complementarity。核对 Areias 等 2019 的出版摘要、Siconos 官方二维摩擦求解器说明、Jiang 1999 出版摘要及 GAMS NLPEC 官方互补函数说明；Areias 全文访问为 403。只借鉴算法思想，不宣称全文复现或理论假设已满足。适用边界和本地反例归入 references/method_notes.md 的 R12；未下载额外第三方全文。

2026-09-09 追加针对性查询：FEBio large deformation frictional contact deep indentation、GetFEM large sliding nonlinear elasticity、FEniCSx ASiMoV Coulomb hyperelastic contact。核对 Zimmerman–Ateshian 2018（DOI 10.1115/1.4040497）的书目及 [官方模型库](https://repo.febio.org/permalink/project/68)，其中包含深压入、摩擦、单/双遍及增广/罚法 patch test。阅读 [sliding-elastic 官方说明](https://febiosoftware.github.io/febio-feature-manual/features/solid_surfaceinteraction_sliding-elastic/)，核实非对称切线和增广约束；论文 PMC 全文与部分下载页遇到验证限制，未保存或声称读过本轮无法访问的全文。

核对 [GetFEM 大滑移](https://getfem.org/userdoc/model_contact_friction_large_sliding.html)、非线性弹性及 Windows 安装文档。其旧节点大滑移 brick 标为不工作，后续候选应核对积分射线追踪接口，不能笼统推荐所有接口。阅读 [ASiMoV 摩擦圆柱源码](https://github.com/Wells-Group/asimov-contact/blob/main/python/demos/hertz_contact/demo_friction_cylinders.py) 和 helpers：体应力为小应变线弹性，不能作为现成有限应变摩擦实现的证据。随后已隔离构建 FEBio 并运行本项目的短基准；GetFEM、FEniCSx/PETSc 仍只是二次开发候选。

同日实现核查：下载并阅读 FEBio v4.13 的 FESlidingElasticInterface、FESlidingInterface、FEUncoupledMaterial、FE3FieldElasticSolidDomain、FERigidSolver、数据记录及时间步控制源码。确认材料压力模型默认值并不等于本仓库体积能，三场体积压力在单元内平均，刚体数据按材料编号索引，min_residual 比较平方力范数，增广到 maxaug 会强制结束；这些发现已反映到导出和状态审计。独立压缩/滑移 patch 支持当前节点—面接触配置，但 sliding-elastic 在同一基准中未通过摩擦反力核对，未把可能的历史初始化机制当作已经证明的唯一原因。官方来源及构建版本见 references/README.md，实际短前缀边界集中于 solver_status.md。

## 2026-09-09：完整积分二次单元与接触求解复查

核对 [Khandelwal 等二次接触分析](https://arxiv.org/abs/2209.04906) 的顶点/中点约束和半边一次接触力空间，下载作者预印本至 references；其线性无摩擦假设不覆盖本项目有限应变干摩擦。对比 [Tian 等有限应变混合框架](https://arxiv.org/abs/2112.00294) 与 GetFEM 官方大滑移文档。根据本仓库材料切线、梁柔度、有限压缩和曲边接触测试实现完整积分 P2，不直接套用论文材料或结论。

进一步查看 [PETSc 修正量回溯及实现](https://petsc.org/main/manualpages/SNES/SNESLINESEARCHNLEQERR/)、[半光滑互补求解说明](https://petsc.org/main/manualpages/SNES/SNESVINEWTONSSLS/) 和 [Fischer–Burmeister 函数](https://petsc.org/main/manualpages/Tao/VecFischer/)。自主冻结切线修正量回溯在真实 0.395 mm 增量未通过，已删除探索代码；没有调用 PETSc 或复现其完整算法。互补函数仍需本项目独立检查，不能据文档宣称当前库仑问题必然收敛。

局部体网格对照只支持“体变形分辨率参与失效”，不提供网格独立性或起裂预测验证；具体结果集中于 solver_status.md。

本文件记录方法选择的检索范围，不声称系统综述、穷尽筛选或复现全部论文。书目与本地全文见 [参考资料](../references/README.md) 和 citation_metadata.json。

2026-09-09 混合元与二次接触补查：检索 P1-iso-P2/P1、finite strain mixed pressure、quadratic contact、Gauss–Lobatto contact constraints。核对 DefElement、GetFEM 官方文档、Tian 等 arXiv:2112.00294 摘要与 Khandelwal、Porwal、Wadhawan arXiv:2209.04906 原文第 3 节。下载后者至 references，确认其接触约束设在二次顶点和中点，并承认离散可行集不必满足连续约束；据此保留曲边穿透的独立审计。小应变无摩擦理论不能直接保证本项目大应变干摩擦稳定；Tian 的算例偏向 Q1/P0，也不能引用成 P2/P1 更优。具体推导与选择边界写入 initiation_model.md，连续结果集中于 solver_status.md。

## 2026-09-07 至 09-08：切割与自由起裂

查询包括 rubber cutting finite strain phase field fracture nucleation strength、cutting easier tearing elastomers blade radius、constrained fracture strength energy；渠道为 arXiv、Nature/ScienceDirect 出版方及作者机构主页。曾尝试 Semantic Scholar 定域检索，但没有可独立归档的有效结果。

阅读切割/撕裂差异论文、2025 黏弹性起裂论文的强度和方程部分、经典相场局限综述、2026 约束断裂预印本摘要。预印本及未校准材料不能当成已验证切割模型。DOI 验证脚本曾请求失败，不能据此判断 DOI 无效；主要条目通过出版方、作者资料及 Crossref 核对。

## 2026-09-08 至 09：接触稳定性

查询包括 nonmonotone line search frictional contact semismooth Newton、TSPSEUDO pseudo transient continuation steady state。复读 Acary 等作者全文；查看 GetFEM、MOOSE、PETSc 官方资料，DTU 的 Madsen 等非线性最小二乘教材，以及 Gfrerer 等静态库仑接触公开论文的引言和算法结构。

用途是区分等价投影、局部 Newton、多值接触和离散问题多解性；不提供新材料参数。阻尼、非单调、锥投影、辅助刚度及方向冻结等探索没有证明持续稳健，未作为未经验证的生产后备方案。2026 interior-point/path-following 论文仅查到摘要，未据此声称实现。

2026-09-11 法向事件修复后的全局路径核查：查询 Newton homotopy continuation initial point path closed loop global convergence nonlinear equations Watson probability one。查到 Watson 全局同伦综述及作者报告，但 PDF 没有可读文本，本轮未声称阅读其定理；转而核查 [Jong 2012 作者预印本](https://optimization-online.org/wp-content/uploads/2012/08/3562.pdf) 第 2 节定理 2.1–2.2、假设 1 与 Newton–固定点同伦式。用途是区分局部可延续与目标可达性；光滑、秩和有界性要求不能直接套用当前非光滑有限应变接触。摘要及实现限制见 R25，未新增 PDF 或声称实现其算法。

## 适用边界

2026-09-11 极简诊断：定向查询 frictionless contact augmented Lagrangian penalty large sliding、Abaqus quasi static explicit kinetic energy internal energy friction contact。核查 GetFEM 无摩擦/摩擦接触接口说明与 Abaqus/Explicit 准静态官方说明；前者支持分离法向接触作对照，后者说明算法替代仍有惯性/能量验收。当前只实现法向互补简化，没有调用这些软件，也未实现显式动力学；不将简化算例当作实验模型通过。

2026-09-10 积分与拓扑对照实现：核对 [NumPy leggauss 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html) 的点数、区间和多项式精度，以及 [threadpoolctl 作者说明](https://github.com/joblib/threadpoolctl) 的 BLAS/OpenMP 线程池限制。刀具距离和法向不是一般多项式，故多项式精度阶不能直接当作接触积分误差界，仍须实际阶数对照。拓扑比较保持参考顶点、边界、单元数及角度质量，加载从零独立生成历史。

2026-09-10 实际接触约束审计：检索 PETSc error-oriented affine covariant Newton、rubber indentation surface instability crease，以及 Hohlfeld Mahadevan unfolding sulcus。读取 PETSc NLEQERR 文档与源码、SLEPc SVD 手册，以及作者托管的 PRL 2011 正文中模型、网格、分支图和法向载荷适用边界；方法笔记 R13–R14 记录自主摘要与限制。ArXiv 1808.10420 仅查看摘要，不据此声称实现其表面势摩擦算法。没有使用用户保留的实验验证数据。

2026-09-09 针对 FEBio 渐变网格前缀，核对官方 v4.13 的 FETimeStepController、FESlidingInterface、DataRecord、FEBioApp 和 FEBioModel 源码，以及 feature manual 的 timecontroller/default 和 sliding-node-on-facet。检索重点是成功增量持续缩小、增广停止条件、摩擦切线、记录精度及原生重启。以保存输入的同条件对照确认时间控制器原因；Broyden 真实斜切对照未通过，未将官方选项的存在当作有效性证明。源链接集中在 references/README.md；后续网页抓取部分源码出现缓存错误，源码核对以先前下载的 v4.13 原文件为准。

2026-09-09 的 0.321 mm 复查：定向检索 nodal integration stabilization nearly incompressible finite strain、frictional contact semismooth Newton line search convergence。核对 Andrade Pires 等平均节点体积方法、Puso–Solberg 稳定节点积分与 FEBio UT4 技术报告，并下载后者。稳定化形式不同，不能直接继承其参数或证明；本项目体积方差候选的细网格连续试验反而在约 0.276 mm 失败，已记录这一反例。

求解方向参考 [SciPy 非线性最小二乘文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) 中的 Jacobian 列缩放及阻尼/正则化原则。当前采用自主实现的稀疏阻尼方向与原残差回溯，并非调用该求解器或复现 SCD 算法；没有普遍收敛保证。PMC 全文页面本次访问受到机器人验证限制，未把搜索摘要当作读过新的全文。

论文复现是独立 EPDM 平面应力拉伸；主线是 CB245NE 平面应变斜切。当前只复用已验证的材料内核，强度筛查尚未与自由裂纹、新裂面接触耦合。后续优先核验接触和近不可压缩离散，再验证无预裂纹起裂。


## 2026-10-20 理论路线记录

本节原文字存在编码损坏，整理时不猜测补写丢失的阅读结论。可核对的具体讨论保留在[理论HTML](../outputs/angle_history_review_20261020/theory.html)，当前适用边界见[理论审查](theoretical_model_audit.md)。

原记录保留了以下来源链接：Kojima（2020）的[J-STAGE原文](https://www.jstage.jst.go.jp/article/sosei/61/708/61_1/_pdf)、Lubarda与Lubarda的[文章](https://arxiv.org/html/2309.02361v1)、Moreno-Mateos与Steinmann的[文章](https://arxiv.org/html/2507.13565v1)，以及[Lengiewicz等的摘要](https://arxiv.org/abs/2005.02107)和[Menga等的摘要](https://arxiv.org/abs/2006.16133)。这些是历史来源索引，不代表本次清理重新阅读全文或完成新验证。
