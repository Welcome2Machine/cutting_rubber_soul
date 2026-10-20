# 接触局部非线性与内外层求解：本轮方法依据

> Historical technical record. Obsolete outputs were retired on 2026-10-20; summary links do not include the original figures or full replay histories. See [current research status](../docs/research_status.md).


## 2026-10-19：接触乘子空间与弱间隙盲区

核查[GetFEM大滑移／大变形摩擦接触官方说明](https://getfem.org/userdoc/model_contact_friction_large_sliding.html)以及[MOOSE乘子Petrov–Galerkin说明](https://mooseframework.inl.gov/syntax/Constraints/Petrov-Galerkin_approach_for_Lagrange_multipliers.html)。前者给出更新几何、投影、节点／积分点接触与刚体障碍的方程；其中节点brick被明确标记尚不可用，本项目未把它当作已经可接入的现成求解器。后者强调乘子插值、测试空间与加权间隙的差别，不能仅凭方法名称推断点值穿透消失。

本轮自身的证据是直刀面上一条P0约束边的两端间隙约±.201255 μm而边平均约为零；改为正连续P1乘子可消除平直、完全接触P1片段中的这一交替模式。条件重放仍在圆弧边内穿透.204832 μm处拒绝，所以还需要细化接触表面。这个方案并非上述软件特定mortar／Petrov–Galerkin算法的完整移植；不声称整体inf-sup稳定、接触空间收敛或深路径成功。两套独立整边审查、原材料和力学接受门槛保留。推导与实际负面结果见[接触文档](../docs/history/topics/augmented_contact.md)和[HTML证据 (retired; summary only)](../outputs/retired_evidence/al_contact_repair_20261019.json)。

## 2026-10-19：AL失效态的方向控制依据

再次核查[PETSc SNES官方手册](https://petsc.org/release/manual/snes/)及[SNESNEWTONTR](https://petsc.org/release/manualpages/SNES/SNESNEWTONTR/)：残差求根可通过线搜索或信赖域控制步长／局部模型，单纯把不可靠的Newton方向不断缩小不构成收敛保证。当前候选借鉴方向尺度与实际／预测残差下降的检查，采用有界对角正则化Newton；没有接入PETSc，也不称为其信赖域算法。

[SciPy splu官方接口](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html)支持SuperLU分解及相关求解选项。本轮实际失败混合系统在原SuperLU、PARDISO与行列平衡SuperLU下给出近乎相同的942 mm方向；后两者虽过原线性残差，仍不能使巨大方向合理。这里的局部根因证据来自实际计算，不是从文档推断。固定同一已接受历史的对照进一步显示：仅换库仍失败，保留前步代数乘子初猜则完成原失败增量，受控半步检验也通过。文献只支持改进思路，不证明本任务的深路径、接触空间或实验曲线已验证。

证据与适用范围见[HTML核查 (retired; summary only)](../outputs/retired_evidence/al_failure_20261019.json)及[方法文档](../docs/history/topics/augmented_contact.md)。共同材料、摩擦参数、刀具、边界和物理验收保持不变。

## 2026-10-19：连续失败后的替代接触路线评估（尚未实现）

[HTML路线评估 (retired; summary only)](../outputs/retired_evidence/native_paths_20261019.json)。建议暂停原样延长C57，保留共同有限应变材料与已完成基线，优先评估大滑动面接触＋增广拉格朗日＋库仑返回映射。当前代码已经含摩擦几何耦合切线，不能把简单增加耦合Newton当作未尝试的新方向。新接触离散须独立验收压力互补、穿透、积分和整边几何；旧固定Gauss解析圆鼻的深部漏积分问题不能原样带入。

- Zimmerman, B. K., & Ateshian, G. A. (2018). *A Surface-to-Surface Finite Element Algorithm for Large Deformation Frictional Contact in FEBio*. Journal of Biomechanical Engineering, 140(8). DOI: 10.1115/1.4040497。[作者官方模型库](https://repo.febio.org/permalink/project/68)核查了书目信息及补丁/Hertz/叠块等可复现模型目录；本轮没有下载或运行它们。
- [FEBio sliding-elastic官方说明](https://febiosoftware.github.io/febio-docs/features/features/solid_surfaceinteraction_sliding-elastic/)明确增广拉格朗日接触及非对称刚度；作为可借鉴实现，不承诺细刀尖问题的稳定性或速度。
- [GetFEM大滑动官方文档](https://getfem.org/userdoc/model_contact_friction_large_sliding.html)含接触面积分、刚性有符号距离障碍与库仑投影；其nodal projection brick另标注暂不能工作，不能推荐为可直接使用接口。小滑动模块不适用于本任务。

本轮只分析与记录，没有新求解、修改物理或安装后端。原有算法记录如下，执行状态以最新终态为准。

2026-10-19后续计算路线核查（定向官方文档核查，非系统综述）：

- [SciPy `splu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html)给出SuperLU的CSC输入和COLAMD/MMD等排序接口。本地同一真实非对称PTC矩阵实测COLAMD约.214 s、MMD_AT_PLUS_A约9.43 s、MMD_ATA约.240 s；虽某排序减少填充，却不更快，继续采用原COLAMD。每个方向都核对当前矩阵线性残差；这是同场成本测试，不是收敛证明。
- [NVIDIA cuDSS官方文档](https://docs.nvidia.com/cuda/cudss/latest/index.html)提供GPU稀疏直接求解，支持一般矩阵、双精度及Windows。它处理线性子问题，不自动把本地NumPy材料返回映射、IPC接触与CCD迁到GPU。当前固定场装配约.77 s，其中材料约.58 s；线性分解约.214 s，因此不能假定只换GPU线性库就解决主要开销。本轮未安装或启用cuDSS，不报告GPU提速。
- [PETSc SNES](https://petsc.org/release/manual/snes/)是可复用的非线性/线性求解组织框架。迁移仍需要本题残差、一致切线、接触可行步长及历史提交规则；不会自动认证力曲线、摩擦或近不可压离散。当前优先保留已核查的原生路径，用可测的材料分批和独立任务并行降成本，是否迁移以后由实测主瓶颈决定。

上述本地数值证据在`outputs/native_paths_20261019/benchmark.json`和`batch_benchmark.json`。材料分批保留同一每点方程及停止容差；默认关闭，三个准确历史场及独立小路径核查后才用于新P1诊断。并行上限两组不等于保证吞吐翻倍，内存和实测路径时间仍需报告。原接触刷新独立资格已失败，不从其两个成功单步推断总体算法有效。

核查日期：2026-10-19。用途是改善相同有限元平衡方程的计算效率，不改变橡胶本构、接触几何、摩擦系数或实验解释。

## 参考资料与适用范围

- [PETSc SNES 官方手册](https://petsc.org/release/manual/snes/)介绍非线性预条件与内外层求解：内层构造更新，外层仍求原始非线性方程。它支持将困难局部非线性更充分地处理，但没有保证本项目的具体候选会收敛。
- [PETSc SNESASPIN 官方文档](https://petsc.org/release/manualpages/SNES/SNESASPIN/)及其引用的 Cai–Keyes 2002、Brune 等 2015 提供非线性分区/预条件的研究背景。本轮没有实现 ASPIN、RASPEN 或 Schur 消元，不能借用这些名称宣称算法等价或继承其理论保证。

## 本项目的证据与候选

在两个C57失败试探场中，约98%的自由度残差平方位于表面，前10个节点承载约99%。这描述的是未平衡残差的位置，不是材料应力或能量有98%集中在表面。分量有限差分在合适扰动尺度下基本符合解析切线；实际有限步长的误差则以法向接触为主。固定摩擦几何没有消除该误差，不能继续把原因简单归为黏滑切换。

记原始自由度平衡方程为

`R(x) = R_bulk(x; history, dt) + R_contact(x; previous) = 0`。

在外层参考点 `x_k`，内层仅用材料切线构造

`R_inner(x) = R_bulk(x_k) + K_bulk(x_k)(x-x_k) + R_contact(x)`。

内层每次重新构建接触集合、法向压力和摩擦几何；材料/摩擦的上一已接受历史始终不变。内层收敛不是物理解：外层必须重算完整有限应变材料响应与接触残差，最后还走原始几何、CCD、耗散与`1e-4 N`力残差验收。材料切线冻结仅限代数求解中的局部近似，不是把橡胶改成线弹性。

内层仍使用非对称接触雅可比及PTC正则化，附带残差下降判据。该附加判据和材料装配复用是两个算法改动，性能收益需要分开核查；不能将全部迭代减少归因于装配复用。外层更新还单独核查起终点之间的CCD路径，避免错误地由一串安全折线路径推断直线回溯也安全。

本轮消融检查已实际执行：关闭附加残差下降判据后，.06→.0625的单步未通过；组合版本通过。因此现有正结果属于组合策略。另一.045→.05较大单步在60次和8次刷新版本中均失败，不能仅引用成功步宣称整体稳定。两个失败快照还核查了摩擦正则区间：全部非零绝对加权承载项都处于滑动区，主导相对增量约.003–.0046 mm，而平滑阈值是1e−5 mm；这不排除之前迭代发生过切换。

## 局限

材料线性化可能在内层多次移动后失准，因此必须限制内层工作量并以完整残差控制刷新。局部成功不能证明完整C57路径、深压入、起裂或实际跨角度力曲线已得到验证。运行证据与冻结实现见[本轮HTML报告 (retired; summary only)](../outputs/retired_evidence/path_coupling_20261019.json)，失败记录保留。
