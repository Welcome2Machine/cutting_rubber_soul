# 文献检索记录

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

## 适用边界

2026-09-09 针对 FEBio 渐变网格前缀，核对官方 v4.13 的 FETimeStepController、FESlidingInterface、DataRecord、FEBioApp 和 FEBioModel 源码，以及 feature manual 的 timecontroller/default 和 sliding-node-on-facet。检索重点是成功增量持续缩小、增广停止条件、摩擦切线、记录精度及原生重启。以保存输入的同条件对照确认时间控制器原因；Broyden 真实斜切对照未通过，未将官方选项的存在当作有效性证明。源链接集中在 references/README.md；后续网页抓取部分源码出现缓存错误，源码核对以先前下载的 v4.13 原文件为准。

2026-09-09 的 0.321 mm 复查：定向检索 nodal integration stabilization nearly incompressible finite strain、frictional contact semismooth Newton line search convergence。核对 Andrade Pires 等平均节点体积方法、Puso–Solberg 稳定节点积分与 FEBio UT4 技术报告，并下载后者。稳定化形式不同，不能直接继承其参数或证明；本项目体积方差候选的细网格连续试验反而在约 0.276 mm 失败，已记录这一反例。

求解方向参考 [SciPy 非线性最小二乘文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) 中的 Jacobian 列缩放及阻尼/正则化原则。当前采用自主实现的稀疏阻尼方向与原残差回溯，并非调用该求解器或复现 SCD 算法；没有普遍收敛保证。PMC 全文页面本次访问受到机器人验证限制，未把搜索摘要当作读过新的全文。

论文复现是独立 EPDM 平面应力拉伸；主线是 CB245NE 平面应变斜切。当前只复用已验证的材料内核，强度筛查尚未与自由裂纹、新裂面接触耦合。后续优先核验接触和近不可压缩离散，再验证无预裂纹起裂。
