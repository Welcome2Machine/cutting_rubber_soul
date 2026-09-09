# 橡胶斜切建模：论文方法笔记

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
- 代码关联：`prototype/material.py` 的 NodalVolume 是本仓库独立能量形式；当前主线采用 P2 完整积分，节点体积路线保留作比较。

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

## 面向主线的实施顺序（项目判断）

1. **接触可信性先行**：实际力函数的切线差分、固定历史回退、逐步力平衡、法切向投影、整边穿透和正 J；先用解析 patch，再连续跨过已知失效区。
2. **体离散独立对照**：保持相同连续材料能量，比较 P2 完整积分与三场 Hex8；同一材料点检查压力和主应力，不能仅比较总反力。
3. **起裂位置竞争**：全域搜索强度和伸长，区分坐标构形、采样点与平均量；候选不限定于刀尖下方。
4. **断裂演化扩展**：优先核对 R03 的材料/相场耦合与 R09–R11 的强度要求，再补充受压裂面、新裂面刀具接触和能量账。缺失标定须显式保留，不能调参掩盖数值失效。

本文件不提供论文原始曲线或可用于拟合的实验数据，也不声称已经完成上述方法的综合验证。
