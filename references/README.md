# 参考方法索引

2026-09-28：完成 [R04：Cutting soft materials 模型复核](method_notes.md)，补充 **2026 正式版**与 2025 预印本的区别、指数 CZM 强度/耗能约束和本仓库参数条件演算。结论为借鉴预设界面＋有限应变＋刀具接触结构，不能照搬刀厚相关开口、切割曲线拟合或伪时间阻尼。用户新提议允许评估固定潜在路径；下列历史条目“不能预设”的范围不再作为排除该候选的依据。未宣称实现或实验验证，下一步见 [断裂路线](../docs/fracture_route.md)。

2026-09-26 同日实现核查：连续 QR 最低能方向的标量化和 SVD 重建是针对本仓库 NH/quadratic-J、F33=1 的本地推导，记录于 directional_fracture.py 的 `_continuous_qr_normal`，并用独立密角扫、刚体/参考坐标旋转及压剪测试核验；不归因于文献作者，也不将其视为新的多轴强度理论或唯一物理路径准则。文献构件仍采用下面列出的方向开闭有效能量和独立强度/断裂能思想。60° 短窗口只得到微损伤，未验证真实裂纹成核或新裂面接触，结果见 docs/solver_status.md。

2026-09-26 正向断裂/压闭定向核查（不是系统综述）：重读 [Hakimzadeh 等 2022 原文](https://arxiv.org/html/2206.15386v1) 的有限应变模式松弛、循环开闭和讨论；其裂面无摩擦，支持闭合不等于愈合，不能据此宣称刀具可进入新裂缝。[Kamarei 等 2025 作者原文](https://arxiv.org/html/2506.16479v1) 区分本构、强度与断裂能，支持无缺口成核不能只取最大应力或只取 Gc，但多轴强度仍须声明/标定。[Han 等 2023 正文](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7169) 提供弥散到离散转换，§5.2 明说接触尚非其范围；[Poluektov / Figiel 2022 机构入口](https://wrap.warwick.ac.uk/id/eprint/159434/) 支持有限变形 CutFEM 分离/再接触，本轮只核对机构摘要/书目，未复现推导。[Moreno-Mateos / Steinmann 2025 作者版本](https://arxiv.org/html/2507.13565v1) §5.4–5.6 处理原边界/新切面与刀具接触，但预设 cutting interface，不能据此设定本任务裂纹位置。实现决策和单边起损判据的新核查集中于 [fracture_route.md](../docs/fracture_route.md)。

补充切割机理证据：[Zhao 等 2025，Why cutting is easier than tearing elastomers](https://www.nature.com/articles/s41467-025-58483-1)，DOI 10.1038/s41467-025-58483-1。出版方索引的摘要/正文说明：预拉伸 PDMS 的刀切与撕裂涉及不同的局部变形、钝化和分子损伤范围，刀具局部接触参与裂纹扩展的供能。它支持同时建模刀具接触与断裂，而不是将切割简单等同远场拉裂；不是 CB245NE 的强度/Gc 标定，也不证明本实验下压时必有闭合裂纹。出版页面直接打开触发访问重定向，本轮未逐图读全文。未新增或发布 PDF。

2026-09-23 润滑/温度定向核查：[硅油润滑与 15–35°C 证据笔记](lubrication_temperature.md) 区分官方 CB245NE 氯丁橡胶身份、硅油边界润滑、体黏弹阻力、温黏及结晶历史。主要新证据为 [Maxell 官方 SDS](https://www.kurehae.maxell.co.jp/dl/SDS/SDS_CB245NE_iC025_2025.06.30.pdf)、[Xu–Persson 2025 三角钢滑块/涂油橡胶作者全文](https://arxiv.org/html/2505.16943v1)、[Son 等 2022 CR 温度拉伸](https://doi.org/10.3744/SNAK.2022.59.3.157)、[Shin-Etsu KF-96 技术资料](https://www.shinetsusilicone-global.com/catalog/pdf/kf96_e.pdf) 和 [Le Gac 等 2019 填充 CR 应变结晶出版摘要](https://doi.org/10.1016/j.polymer.2019.04.019)。本轮涂油 μ_eff=0.10 是未标定低摩擦情景，不是文献测得的 CB245NE 系数；统一名义 25°C、保留材料参数，未把温度效应已识别但未标定误写为影响很小。未下载/发布原文 PDF。

2026-09-16 起始事件校准复核：定向重读 [Hakimzadeh 等 §2.C–2.D](https://arxiv.org/html/2206.15386v1) 的裂面模式松弛及低能方向，以及 [Mandal 等 2020 作者上传正文](https://www.researchgate.net/publication/342491699_A_length_scale_insensitive_phase_field_model_for_brittle_fracture_of_hyperelastic_solids) 的有限变形临界能量校准讨论。前者支持区分方向相关释放，后者支持使用同一本构的临界能量；二者均未给出本仓库 QR＋RationalAT1 组合的现成刀切标定。旧固定法向阈值与运行时方向搜索不一致的发现、连续法向阈值和 P1 首次变分公式是本地方程核查结果，不归因于作者。自由单轴向平面应变/多轴外推、目录伸长与本构不一致、新裂面刀具接触仍是限制；没有新增或发布 PDF。

2026-09-16 相场简化/创新方向核查：新增阅读 [Han 等，2023，有限应变下弥散到离散裂纹转换](https://doi.org/10.1002/nme.7169) 的开放正文摘要、引言、§2 与方法说明。相场给出路径、有限覆盖法生成离散面，并使用转换稳定化；证明此总体思路已有先例，不证明已覆盖本橡胶无缺口强度、压闭新面和摩擦刀切。项目候选贡献只能在这些特定耦合缺口上经对照建立。重新核查 [Kim 等 2026 v1 §2.2/3.4](https://arxiv.org/html/2603.15635v1)：第三介质初始占外部间隙、相场仅在基体，且基体使用小应变；不能把摘要中的统一框架当作新裂纹内自动生成刀具接触域。未下载/发布原文副本。

[Mandal 等 2020，有限变形 PF-CZM](https://doi.org/10.1016/j.engfracmech.2020.107196) 的作者上传全文 §3.2/4.2 支持按同一本构的临界能量设置起损阈值，不能直接套线性 ft²/(2E)。其有限变形软化参数及长度不敏感仍需数值核验，不据此保证本工况。单轴能量外推＋[Hakimzadeh 的有限变形开闭有效能量](https://arxiv.org/html/2206.15386v1) 只构成待验组合；后者裂面无摩擦，组合也不自动生成新刀具接触面。详细原创边界与整体验收见 docs/fracture_route.md。

2026-09-16 实现依据：逐式核查 [Kamarei 2025 §3.5–4](https://arxiv.org/html/2506.16479v1) 的强度源项与长度/网格修正。`strength_phase_field.py` 实现纯弹性、常数第二 Piola 强度的固定变形特例；显式声明静水参数，使用总弹性能，不能混用旧 QR 释放能。作者[官方代码目录](https://pamies.cee.illinois.edu/repositories/)链接的 [FEniCSx 公开示例](https://github.com/farhadkama/FEniCSx_Kamarei_Kumar_Lopez-Pamies) README 标为线弹性，不能称直接取得完整有限黏弹性刀切实现。未复制作者代码。

混合体离散依据为 [Guermond，Chapter 54 §54.4 / Lemma 54.12](https://people.tamu.edu/~guermond/M661_FALL_2023/chap54.pdf) 的 P1-iso-P2/P1 稳定空间对。该结论针对满足网格条件的 Stokes 离散，**不自动保证严重有限变形、接触或断裂下全局稳定**。本项目在有限应变能量中使用对应位移/压力空间和集总压力质量，通过均匀响应、能量/力/切线及独立弯曲基准验证迁移。[Tian 作者摘要](https://arxiv.org/abs/2112.00294) 支持近不可压缩与裂口应共同处理，其 Q1/P0、P2/P1 和损伤体积松弛并未被本轮完整复现。新方程与旧损伤能量未经共同推导前不直接耦合。

2026-09-16 相场适用性再核查：阅读 [Dammaß 2023 正式全文](https://link.springer.com/article/10.1007/s00466-023-02310-1) 的引言、弥散表示与数值验证范围，不能把“未显式切网格”解释成“只画裂纹”。核对 [Kamarei 2025 §2.2/3.2/3.5](https://arxiv.org/html/2506.16479v1) 独立强度及参数需求，[Hakimzadeh 2022 §9/结论](https://arxiv.org/html/2206.15386v1) 的闭合响应和无摩擦限制，以及 [Tian 作者摘要](https://arxiv.org/abs/2112.00294) 的近不可压缩/弥散开口矛盾与混合离散。项目判断为相场优先承担自由起裂/路径，裂后刀具接触单独补齐；不据这些分别成立的构件声称统一切割模型已成立。显式面只是当前 IPC 接口候选，并非所有相场物理开裂的必要条件。

2026-09-16 闭合裂面组件实现：再次逐段核对 [Moreno-Mateos / Steinmann §5.6](https://arxiv.org/html/2507.13565v1#S5.SS6) 的法向互补与增广拉格朗日。本文采用“正间隙为分离”的符号，与论文式 (35) 相反；本地离散 λ 单位 N，罚因子单位 N/mm，不将节点力直接称为接触压力。只借鉴允许零间隙的约束结构，未移植其预设 cohesive 面、椭圆刀尖或黏附标定。切向采用本地明确的平滑 Coulomb 势，不声称复现论文 §5.7。两遍节点—线段裂面接触只是组件实现，不能由它推断 mortar 的无偏性或复杂自接触能力。局部几何梯度由独立有限差分、刚体客观性和力矩测试核查；完整实验强挤压及自由萌生仍未验证。

2026-09-16 真实开裂与强挤压定向核查：重新核对 [Kamarei 2025 §2.2.1、§3.2.2、§3.5](https://arxiv.org/html/2506.16479v1) 的独立强度—变形面、参数需求及多轴外推限制；[Hakimzadeh 2022 §9–10](https://arxiv.org/html/2206.15386v1) 给出有限变形裂面闭合响应，但明确无裂面摩擦；[Moreno-Mateos / Steinmann §5](https://arxiv.org/html/2507.13565v1) 的软切割包括增广拉格朗日闭合接触，但预设分离面，不能作本任务自由起裂的证明。新增表示候选：[Poluektov / Figiel 2022，CutFEM 有限变形断裂与接触](https://doi.org/10.1016/j.cma.2021.114234)，[作者摘要及开放全文入口](https://wrap.warwick.ac.uk/id/eprint/159434/)；本轮核对摘要和书目，支持界面穿越单元、双侧分离/再接触的表示能力，尚未逐式复现或核实自由起裂/摩擦刀切。采用允许零法向间隙的接触，是为避免将重合新生裂面直接送入当前严格正距离 IPC；具体路线与能力关口见 fracture_route.md。未据任一单篇论文保证完整任务可解。

同轮多场 MPM 备选核查：[Homel / Herbold，Field-gradient partitioning for fracture and frictional contact in the material point method](https://doi.org/10.1002/nme.5317)，作者上传全文 §1.1.1、§2.2–2.7、§3.1/3.3 与结论，确认损伤梯度分区可产生双速度场及闭合摩擦，但原损伤律偏 Mode I 动态脆断，分离阈值影响耗能，逐对接触在超过两场时不严格保证不穿透。刀具加两裂面属于必须另验的多体接触情形。后续 [Wolper 等 2024 自由表面/强梯度 MPM 基准](https://doi.org/10.1002/nme.7414) 本轮仅核对出版摘要及 [TwoFieldMPM 官方代码](https://github.com/penn-graphics-research/TwoFieldMPM)，不能据开源或已有裂纹基准宣称自由橡胶刀切已可用。

2026-09-16 提速核查与实施：[SciPy GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html) 区分预条件残差与最终真实线性残差；当前已实现仅同一 x 的 shift 重试复用 SuperLU 因子，保留跨位移复用失败反例。带 callback 的 `callback_type='legacy'` 才使 `maxiter=8` 表示实际内迭代数；额外独立检查 `||A du-b|| <= max(1e-12,1e-8||b||)`，超时/不合格回退原直解。0.5 倍分解时间预算来自本地性能保护设计，并非文献收敛定理。[Intel PARDISO](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-1/pardiso.html) 支持非对称矩阵和分析/分解/回代分阶段，是并行稀疏直接求解候选，尚未本机安装或测出收益。接触稀疏结构改变时不能无条件复用符号分析。官方能力与本地性能结论分开；精确裁剪重复方向认证来自本地 `(1-g)` 加权方程，不是外部文献保证新的裂纹算法。割线初值的选择门限来自同一接受态短探针，仍须通过原非线性/物理验收，不据较小初始残差宣称选到了正确物理分支。

2026-09-15 性能参考：[SciPy spsolve 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html) 描述 COLAMD 与 MMD_AT_PLUS_A 等列排序，以及可选 UMFPACK 条件。本地 profile 显示当前调用为 SuperLU；排序比较未完成于短测试预算内，因此未据文档推断替代排序必然更快，也未切换生产求解器。本轮采用的成对矩阵装配只是复用同一状态的重复计算，已通过原算法数值一致性对照；不新增物理模型假设。

2026-09-15 能量验收核查：再次阅读 [IPC 技术补充 §9](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf) 和 [IPC Toolkit 摩擦势教程](https://ipctk.xyz/v1.2.0/tutorial/getting_started.html)，并核对本地原生摩擦平滑函数及势梯度源码。摩擦势是冻结法向力/切向映射的求解势，其值不等于本步 `摩擦梯度·位移增量`，后者也不等于沿未知真实路径积分的摩擦热。由本地方程推导端点虚功恒等式，取消将混合积分的梯形能量等式作为硬验收；保留单边能量产生检查，显式记录未解析余量。教程为 v1.2.0，实际运行二进制为 v1.6.0，未据旧教程假设全部接口相同。此修复不是新材料定律，也不构成非凸准静态断裂能量稳定性的文献证明。未新增或发布 PDF。

2026-09-13 framework 实施：直接复核 [Hakimzadeh 等 2022 §2.C 的 QR 模式松弛及 §8 数值设置](https://arxiv.org/html/2206.15386v1)；本地对原 quadratic-J neo-Hookean 推导平面应变法向松弛及一致切线。参考 [Mandal 等 2020 的超弹性 PF-CZM](https://doi.org/10.1016/j.engfracmech.2020.107196) 和 [PhaFiDyn 2025 的有理退化式](https://doi.org/10.1016/j.softx.2025.102227) 核对函数结构；本地采用 AT1 裂纹密度并重新推导单轴可释放能阈值，属于自主组合，不是这些论文算法复现或已经验证的多轴材料定律。裂后新刀具接触尚未实现，停止边界和适用限制见 fracture_route。

2026-09-13 最新研究入口：[起裂、裂纹扩展与裂后刀具接触可行性评估](../docs/fracture_feasibility_review.md)。集中整理 37 篇研究（F01–F37），标注正文/摘要/既有核查范围和预印本状态；32 篇正式 DOI 已核对解析和 Crossref 元数据。新增重点包括有限变形 cohesive 相场、2025–2026 年可调多轴强度面及返回映射、近不可压缩混合离散、裂面接触、切割端部效应和 MPM。该清单是自主中文摘要，不替代原文公式审计；不复制论文全文或新增重复 PDF。此前以尾迹显式化接 IPC 为确定路线的安排调整为待关口验证候选，详见评估与建模路线。

2026-09-13 双向耦合核查：再次核对上述 Bleyer AT1 能量/不可逆约束以及 [Kamarei 2025 §3.5、§4.4](https://arxiv.org/html/2506.16479v1) 的独立强度驱动力和交替求解。当前只将已有 AT1 子问题与 IPC 力平衡耦合，未复制论文的非协调单元、梯度流或黏弹性积分器。式 (21)/(31)/(32) 所需的单轴/静水强度和有限长度修正尚未迁移；不能用目录名义强度直接取代这些量。源码接口和适用限制见 [断裂路线](../docs/fracture_route.md)。
2026-09-13 能量对照：[Bleyer，Localized 1D solutions in phase-field approach to brittle fracture](https://bleyerj.github.io/comet-fenicsx/tours/nonlinear_problems/bound_constrained/bound_constrained.html)，核查 AT1 断裂能式 (28)、`d_old≤d≤1` 和固定 u 子问题。借鉴其变分结构建立二维 P1 约束能量诊断，未采用示例中强制中心损伤的边界条件。实现不包含多轴强度或裂面接触，不能作为正式切割模型；公式、适用限制见 fracture_route.md，数值证据见 solver_status.md。

2026-09-13 断裂路线：复核 [Dammaß 2023](https://doi.org/10.1007/s00466-023-02310-1)、[Kamarei 等 2025 §3.4–3.5](https://arxiv.org/html/2506.16479v1) 及 [2026 强度约束预印本 §4.2](https://arxiv.org/html/2608.28326v1)。选用独立强度与断裂能分开的开发路线，不直接将预缺口相场用于无缺口起裂。新增定向参考：[Tian 等，近不可压缩混合相场](https://arxiv.org/abs/2112.00294)、[有限变形裂面接触有效能量](https://arxiv.org/abs/2206.15386)、[Hussein 等，相场与离散裂纹混合表示](https://doi.org/10.1016/j.cma.2020.113329)。后三项本轮仅核查作者/出版摘要，尚未复现，不能声称完成裂后接触适配。自主适用性分析与代码边界见 [断裂路线](../docs/fracture_route.md)，不另存重复 PDF。

2026-09-12 伪瞬态延续：[PETSc：Solving Steady-State Problems with Pseudo-Timestepping](https://petsc.org/release/manual/ts/#solving-steady-state-problems-with-pseudo-timestepping)，以辅助演化求原方程零点，支持 B 度量和残差比伪时间步更新；[Gee 等，Pseudo-transient continuation for nonlinear transient elasticity，2009](https://doi.org/10.1002/nme.2527) 给出非线性弹性中的相关应用。用于处理本地含摩擦耦合矩阵近奇异、单调残差搜索停滞，不能据此声称橡胶切割全程稳定或实验动态响应正确。代码用投影 Hessian 作为辅助度量，保留未移位力验收；详情见 modeling_roadmap。

2026-09-12 摩擦外层：[Li 等 2020 IPC 技术补充第 9 节](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf) 与 [Physics-Based Simulation：Fixed-Point Iteration](https://phys-sim-book.github.io/boundary-treatments/frictional-contact/fixed_point_iter.html)。可参考冻结法向力/切向基底的摩擦势与外层刷新流程；内层收敛不保证外层固定点收敛。本地依据失败日志增加重新组装力残差的外层回溯，不改变摩擦系数，也不把冻结摩擦残差作为最终验收。详见建模路线及 solver_status。

2026-09-12 能量差精度：[ANSYS Newton-Raphson 理论手册 14.10.5](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/ans_thry/thy_tool10.html) 描述沿步方向的势能导数线搜索；[A Noise-Tolerant Quasi-Newton Algorithm for Unconstrained Optimization](https://arxiv.org/abs/2010.04352) 讨论数值噪声对线搜索的影响。用于理解近收敛处总能量相减不可靠的原因；本地采用带原力/微小位移/方向功多重门槛的精度保护，不是上述算法复现，不放宽力平衡门限。
2026-09-11 接触非凸方向补充：[Gast 等，Optimization Integrator for Large Time Steps，2015](https://math.ucdavis.edu/~jteran/papers/GSSJT15.pdf)，DOI 10.1109/TVCG.2015.2459687，第 3.1 节关于不定 Newton 方向、截断 CG 和下降角度；[Cornell CS4220，2026-04-20](https://www.cs.cornell.edu/courses/cs4220/2026sp/lec/2026-04-20.html) 关于非凸二次模型、移位 Hessian 与信赖域。用于理解 0.77 mm 接触主导的负曲率；本地 `regularized` 为 PN 度量下的全局曲率移位，不是上述动力学或完整信赖域方法复现，尚无完整路径稳定性保证。

2026-09-11 求解器核查：[Fernández-Fernández、Löschner、Bender，Progressively Projected Newton's Method](https://arxiv.org/abs/2505.21013v2)，初版 2025、修订 2026。可借鉴结论是逐单元正定投影可能过度改变全局切线；论文同时报告准静态下 PN 可能更优，因此必须做本问题的同前态对照。本地 `--hessian-mode hybrid` 只试验未投影 Newton 方向加 PSD 回退，不声称复现 PPN。切线接口依据 [IPC Toolkit 官方 API](https://ipctk.xyz/python-api/potentials.html)，沿用锁定 ipctk 1.6.0，并以有限差分核对。

2026-09-11 实现补充：本轮采用 [IPC Toolkit 官方 Python 接口](https://ipctk.xyz/build/python.html) 的 `ipctk==1.6.0` Windows 原生 wheel。依据 [模拟教程](https://ipctk.xyz/tutorials/simulation.html) 组织表面/体自由度映射、屏障能量和 CCD；本地静态宿主使用原有限应变材料，未加入惯性项，不是教程动力学案例的直接复现。原生 `make_static_obstacle_filter` 排除刀具内部碰撞并保留橡胶自接触，避免 Python 回调阻塞并行 CCD。API 以锁定版本的绑定签名为准，不能直接套用在线开发版。方法适用边界及参数敏感性要求见 [建模路线](../docs/modeling_roadmap.md)。

论文原文件仅留在本地，不上传 GitHub。面向 AI 的结构化摘要、原文定位、来源链接、适用边界和代码映射集中在 [method_notes.md](method_notes.md)。这些笔记是自行归纳的研究说明，不是论文全文或作者认证的实现规范。

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
