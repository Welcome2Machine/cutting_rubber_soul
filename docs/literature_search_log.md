# 文献检索记录

## 2026-09-28：Cutting soft materials 的模型与实施核查

定向核查而非系统综述。搜索原文：`"Cutting soft materials" "01869" model cohesive`、`site.nature.com/articles/s41524-025-01869-y "Code availability"`、`site.bleyerj.github.io comet fenicsx cohesive zone submesh interface`。沿结果阅读作者版 arXiv:2507.13565v1、正式版 DOI 10.1038/s41524-025-01869-y 的 PDF、Glasgow 机构书目与 Bleyer 界面专用 CZM 官方教程；核对 Zenodo 15873110 是 DIC 数据而非求解器。

正式版为 2026-01-06 发表，不再统一称 2025 论文。Nature HTML/DOI 和 PMC 页面直开分别出现重定向/验证码；出版 PDF 文本可读，网页截图入口失败。没有把搜索摘要当成逐式排版核验。方程权重/开口因子疑点来自作者 HTML 与势能独立求导；论文程序未公开核验，不能据此声称作者实现错误。

本地仅执行指数包络的标量求导、数值积分和参数兼容性核算：积分 2.15 N/mm、导数相对误差约 3.8e-11；这不是 FEM 验证或 CB245NE 校准。结论和链接集中在 references/method_notes.md 的 R04，实施方案在 docs/fracture_route.md。未下载实验数据包、未改材料与求解代码、未新建计算输出。

## 2026-09-26：正向成核、压闭增长与新面接触

针对性查询：`rubber cutting fracture initiation indentation blade crack growth compression phase field finite strain`；`phase field fracture rubber nucleation strength finite deformation contact crack closure Hakimzadeh Kamarei`；`hyperelastic cohesive phase field Mandal 2020 107196`；`Poluektov Figiel 114234 fracture contact`。另核查已有 Han 2023 和 Moreno-Mateos / Steinmann 作者全文。为方法选型与实现核查，非穷尽/系统综述。

读取 Hakimzadeh arXiv:2206.15386v1、Kamarei arXiv:2506.16479v1、Moreno-Mateos / Steinmann arXiv:2507.13565v1 作者 HTML 原文，以及 Han DOI 10.1002/nme.7169 正文（含 §5.2 接触适用边界）；Poluektov / Figiel 只核实机构摘要/书目。Zhao 2025 Nature Communications 摘要/正文通过出版方搜索索引可读，直开触发重定向；Mandal DOI 本次直开失败，保留已有核查，不冒称本轮重读全文。结论及原文链接见 references/README.md、docs/fracture_route.md。

本地方程核查与论文结论分开：现有 RationalAT1 在 d=0 的梯度项一阶导为零，触发比中的 ell/Gc 消去，是本地代码推导；它不否定有限演化相场的正则化作用。没有新增论文 PDF、数值算例或实验反演。

## 2026-09-23：硅油润滑与 15–35°C 的定向核查

检索目标为决定当前完整橡胶压入/旁路起裂诊断采用何种润滑近似，以及是否有足够证据增加温度物理。渠道为普通网页检索后的原始出版页面、arXiv 作者全文、作者机构库、Maxell/Shin-Etsu 官方资料和一项制造方原始专利；本轮不是多数据库系统综述，不声称穷尽筛选。

主要查询原文：`CB245NE Maxell Kureha rubber 0.66 chloroprene`；`silicone oil rubber steel friction boundary lubrication chloroprene cutting silicone oil`；`Shin Etsu silicone fluid viscosity temperature KF-96 20 25 40 cSt`；`chloroprene rubber temperature mechanical properties 20 40 glass transition tensile modulus original research`；`rubber cutting silicone oil lubrication friction fracture neoprene blade puncture`；`polychloroprene crystallization room temperature time hours days strain induced crystallization 20 35`；`rubber lubrication silicone oil steel Stribeck mixed lubrication Persson 2021 paper`。

已核查 Maxell 2025-06-30 CB245NE SDS 第 1–2 页、Shin-Etsu KF-96 温黏与材料相容性章节、Son 等 2022 CR 温度拉伸正文和表 2、Xu–Persson arXiv:2505.16943v1 §2–3、Moreno-Mateos/Steinmann 2026 接触与界面段。Liu 等 2024 PMC 直接页遇验证，核查其被索引的原文实验/结论；Le Gac 2018/2019 出版摘要和亮点可读，作者机构 PDF 直接访问为 403，不称阅读全文。Triki 2015 刀刺研究找到原始机构书目但出版社正文为 403，未据二手摘要提取系数。Bridgestone JP2024119615A 实施例仅用于分清静置起晶与应变结晶，非 CB245NE 标定。网页截图接口未返回可供本轮目视检查的图像，所以未从扫描目录/曲线目测取值或采用未完整辨读的公式。

结论和原始 URL 集中在 [润滑/温度证据笔记](../references/lubrication_temperature.md)：μ_eff=0.10 只表示本项目的低摩擦假设；不把硅油等同无摩擦，不把温黏比直接转成 μ(T)。CB245NE 属于 CR，不是硅橡胶。15–35°C 的变化不能由其他 CR 配方定量移植；本轮共同名义 25°C 下保持原材料参数，温度、黏弹、结晶和溶胀效应明确记为未建模，而非已证明很小。约 15 s 的加载只限定时间尺度，不能排除较快的应变诱导结晶。未下载新 PDF，未读取独立验证数据，也没有新增本轮三组之外的长算例。

2026-09-16 应用实现定向复核：直接打开 Hakimzadeh arXiv:2206.15386v1 §2.C–2.D 与 Mandal 2020 作者上传全文，核对方向释放能和有限变形能量标定范围；本轮没有重新开展全面文献综述。重点是核验当前方程是否能按声明强度首次触发，发现旧轴法向校准与全方向搜索不一致并修正。来源和适用边界见 references/README.md，数值证据见 solver_status.md。

2026-09-16 用户提出简化相场并研究创新：定向检索 `phase field fracture third medium contact finite deformation cutting rubber 2026`、`finite strain phase field cohesive fracture contact strong compression`、Lammen 2023 题名。阅读 Kim 2026 v1 §2.2/3.4、Han 2023 DOI 10.1002/nme.7169 开放正文、Mandal 2020 作者上传全文的阈值/软化段，复查 Hakimzadeh 与 PF-MPM 适用边界。部分 Elsevier 直开返回 403；Lammen 只据出版摘要评价，不冒称读完推导。此轮是三种表示候选的定向取舍，非全面新颖性检索；未新增求解器/长算例。结论集中于 fracture_route.md 与 references/README.md。

2026-09-16 成熟路线实现核查：直接复查 Kamarei arXiv:2506.16479v1 式 (30)–(35) 与 §4，核验作者代码目录及公开 README；查询 `P1 iso P2 P1 mixed finite element incompressible elasticity stable pressure finite strain`，阅读 Guermond Chapter 54 的空间定义和 Lemma 54.12；再次核对 Tian arXiv:2112.00294 摘要。仅作方程/离散依据的定向核查，不称全面综述或 Tian 全文复现。来源、实现映射和局限集中见 references/README.md；无新论文 PDF 或实验数据发布。

2026-09-16 相场选型定向复核：直接打开 Dammaß 2023 Springer 正式全文、Kamarei 2025 与 Hakimzadeh 2022 作者 HTML、Tian arXiv:2112.00294 作者摘要；核查相场与显式裂面的区别、独立强度、压闭裂纹和近不可压缩性。Hussein 2020 DOI 页面本次直接访问失败，未扩大其证据等级。该工作是对已有候选的范围审查，不是新增全面综述；结论和适用限制见 fracture_route.md 与 references/README.md。无新 PDF、无模型修改或计算。

2026-09-16 闭合组件实施复核：重新打开 Moreno-Mateos / Steinmann 作者 HTML §5.6，核查法向间隙符号、互补条件与增广拉格朗日；这是实现核查，非新增全面综述。出版方 `S0045782503005802` 直接访问未成功，未据未读正文声称本地两遍节点接触具有 mortar 性质。来源与自主实现限制见 references/README.md；未新增 PDF。

2026-09-16 真实开裂/强挤压定向核查（非全面综述）：查询 `Kamarei elastomers fracture nucleation compressive stress strength surface cutting 2025`、`rubber cutting fracture initiation compression crack tip indentation nucleation energy finite fracture mechanics`、`extrinsic cohesive crack nucleation compression large deformation hyperelastic contact augmented Lagrangian finite element cutting`、`discontinuous material point method fracture contact DFGMPM zero gap 2022`。打开 Kamarei 2025 作者 HTML 并核对强度/参数/驱动力章节，Hakimzadeh 2022 作者 HTML 的有限变形开闭及无摩擦限制；核对 Poluektov / Figiel 的 Warwick 作者书目和摘要，未将其当作已复现全文。补充独立核查 IPC 初始化非零距离、软切割预设分离面/增广拉格朗日、DFG-MPM 的多场接触边界；具体适用性放在 references/README.md 与 fracture_route.md，不重复生成论文清单/PDF或宣称穷尽检索。未启动算例或修改物理模型。

2026-09-16 停止后提速评估：定向核查 SciPy GMRES 真实残差与预条件接口、Intel oneMKL PARDISO 非对称矩阵/求解阶段和直接—迭代回退接口、PETSc SNES 的全局化候选及 IPC 摩擦更新限制。只作软件/算法适用性核查，不声称完成新的全面文献综述。局部性能实测、失败反例及未安装/未验证边界见 references/README.md、solver_status.md，原始探针留本地 outputs/run_review_20260916。

2026-09-15 定向核查：查询 IPC friction lagged normal force / dissipative potential，打开官方 IPC 2020 技术补充（§9 摩擦实现）及 IPC Toolkit v1.2.0 getting_started 摩擦势部分；同时核对本地 v1.6.0 运行及摩擦梯度定义。用于审计端点耗散与梯形外功的离散一致性，非新一轮全面综述。来源、适用性及本地推导边界见 references/README.md 和 docs/fracture_route.md。

2026-09-13 framework 实施复核：打开 arXiv:2206.15386v1 全文，复核 QR 参考裂面基底、剪切松弛、单边法向松弛和数值实现限制；重新查询 `phase field cohesive AT1 rational degradation (1-d) 1 a d strength hyperelastic`。Mandal 2020 正文直接访问 403，仅采用已核对出版摘要/可见检索片段作 PF-CZM 背景；有理函数结构另核对 PhaFiDyn 2025 DOI 10.1016/j.softx.2025.102227 的公开式 (12)。本地 AT1 密度＋方向相关松弛能＋单轴标定为自主推导；通过有限差分、客观性及开闭子基准核查实现，不能宣称得到原论文所有性质。没有加入或发布新 PDF。

## 2026-09-13：三阶段断裂可行性范围检索

按用户要求先研究能否做到，再决定实施；未修改求解器或运行模型。本轮共整理 37 项不同研究，证据矩阵和停止条件集中于 [fracture_feasibility_review.md](fracture_feasibility_review.md)。不是穷尽检索、双人筛选或 PRISMA 综述，不声称全部全文精读。

实际检索通道：Crossref、OpenAlex、arXiv；用出版商、作者机构网页和作者公开实现核查关键条件。首选 parallel-cli 在本环境不可用，改用网页工具及数据库 API。Crossref/OpenAlex 三组完整字符串均为 `finite strain phase field fracture rubber contact`、`soft material cutting crack nucleation friction`、`phase field cohesive fracture hyperelastic strength`，每组前 10 条。Crossref 第三组限流，其他五组共 50 条原始返回记录。Crossref 另查 Tian 的 Mixed displacement-pressure-phase field 题名、Fei/Choo 的 A phase-field method for modeling cracks with frictional contact 题名，各返回 3 条；CRESSim-MPM 题名查询限流。原始 API 地址、响应及 32 项 DOI 解析/书目核查结果仅本地保存在 `outputs/literature_feasibility_20260913/`。DOI 身份核验不代替方法正确性评审。

网页检索围绕 finite strain cohesive phase field、rubber cutting nucleation、crack contact friction、nearly incompressible phase field、third-medium fracture、adaptive MPM fracture 及文章题名展开；再沿作者/参考文献追踪。已保留的网页返回批次存于同一本地目录的 `web_searches.json`，不声称它覆盖本轮全部查询。已记录的完整定向字符串包括：

- `finite strain cutting soft materials adaptive phase field discrete crack contact Hussein Hudobivnik Wriggers 2020 113329`
- `phase field fracture finite deformation crack face contact friction incompressible 2024 2025`
- `incremental potential contact fracture cutting topology changes zero distance 2024 2025`
- `soft material cutting material point method fracture blade contact 2024 2025 2026`
- `"cutting" "fracture" "IPC" simulation`
- `"soft" "cutting" "material point" fracture`
- `"phase-field" "cohesive" "finite strain" "strength" elastomers 2024 2025`
- `"finite fracture mechanics" rubber indentation crack nucleation cutting`
- `"phase field" "cohesive" "hyperelastic" fracture strength`
- `"rubber" "phase field" "length scale" "strength" 2025`
- `"DFGMPM" fracture frictional contact`（未取得可纳入的相关来源，不将名称作为方法证据）
- `"Bridging crack nucleation and contact"`

- `Tanne Li Bourdin Marigo Maurini 2018 Crack nucleation variational phase-field models characteristic length strength`
- `Fei Choo phase-field method modeling cracks frictional contact 10.1002/nme.6242 2020`
- `Bourdin Marigo Maurini Zolesi 2025 phase field strength degradation arbitrary convex strength surface 2506.22558`
- `"Cohesive phase-field fracture with an explicit strength surface" eigenstrain 2026`
- `"2026" "106424" strength phase field`
- `"10.1016/j.jmps.2025.106424"`
- `"2506.22558" "small" strain`

重点正文核查：Kamarei 2025 强度驱动力；2026 constrained energy 的时间离散和试探裂纹族限制；Hakimzadeh 2022 裂面有效能量、循环剪切方向约束与无摩擦边界；Moreno-Mateos 2025 作者版 §5 中预设 S0 分离面；Kim 2026 §3.4 Remark 2 的小应变断裂基体及无摩擦假设；Zhao 2025 的切割/撕裂过程区证据；CRESSim-MPM §III-C、§IV-B 的几何切割与迭代相关刚度；Bourdin 2025 和 Bourdin/Maurini 2026 强度退化框架/反平面限制；Hageman 2026 §2 的小应变线弹性假设及局部返回映射。其他条目的摘要级范围逐项标注于证据矩阵。

关键修正：有限变形 cohesive 相场和新强度面方法不能被“经典相场不能起裂”的结论一并排除；第三介质新论文也不能仅凭标题列为有限应变橡胶完整解法。公开代码入口仅作迁移线索，本轮未安装或运行。文献原文、用户媒体及计算输出不发布；研究路径图使用 Markdown Mermaid 保持可读文本，不生成论文图片或额外 PDF。

2026-09-13 双向耦合实施前直接复核 Bleyer 约束 AT1 教程和 arXiv:2506.16479v1 全文中的 external driving force / staggered 段。核查式 (21)、(31)、强度依赖说明和交替求解段，区分本轮经典 AT1 耦合控制与尚未实现的独立强度理论；没有新增重复论文文件。
2026-09-13 后续查询 AT1 phase field fixed displacement bound constrained damage / 3Gc 8 ell critical energy。直接核查 Bleyer FEniCSx 教程式 (28) 及固定 u 的不可逆约束；采用 P1 一致积分与独立 KKT 验收做局部能量对照，不用其 1D 中点损伤边界预设切割裂口，不把经典 AT1 的 ell 依赖萌生当作已标定强度。未将检索到的其他方法摘要当作已实现算法。

2026-09-13：定向查询 rubber fracture nucleation strength phase field / finite strain phase field crack contact incompressibility / adaptive phase field discrete cutting。复核 Dammaß 2023 正式全文材料和求解段、Kamarei 等 arXiv:2506.16479v1 §3.4–3.5 的强度驱动力、arXiv:2608.28326v1 §4.2 的假设和未完备边界。arXiv:2112.00294 的 HTML 不可用，核查作者摘要；arXiv:2206.15386 核查作者摘要；Hussein 等 DOI 10.1016/j.cma.2020.113329 核查出版检索摘要，直接打开全文未成功。没有将摘要核查说成全文阅读或方法复现。技术路线和实现范围集中于 fracture_route.md。

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
