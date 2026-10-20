# 相邻表面节点伪自接触：文献与本项目的证据边界

> Historical technical record. Obsolete outputs were retired on 2026-10-20; summary links do not include the original figures or full replay histories. See [current research status](../docs/research_status.md).


检索日期：2026-10-19。针对当前AL/P2困难增量的定向核查，不是系统综述。

## 直接相关资料

1. Huang 等，**Geometric Contact Potential**，ACM TOG / SIGGRAPH 2025。
   [作者论文](https://huangzizhou.github.io/assets/img/research/smooth-contact/paper.pdf)、[补充材料](https://huangzizhou.github.io/assets/img/research/smooth-contact/supp.pdf)、[作者工具库](https://github.com/JackZhouSz/geometric-contact-potential-toolkit)。补充材料对convergent IPC的讨论指出：接触作用距离小于边长是一个重要条件，否则可能出现相邻顶点之间的人工排斥。正文提出结合局部几何方向的连续接触势，并讨论相邻元素排除、最近点积分与可微性。**单纯删掉接触项并不等于实现了GCP。**

2. Huang 等，**Orientation-aware Incremental Potential Contact**，2024预印本。
   [arXiv:2402.00719](https://arxiv.org/abs/2402.00719)。以接触势的局部性、屏障性和静止状态无伪力等要求为出发点，强调几何相近不等于相对表面接触。与上述2025成果属于相关研究线，不能算两份独立实验支持。

3. Du 等，**No Free Slide: Spurious Contact Forces in Incremental Potential Contact**，作者预印本。
   [作者PDF](https://crl.ethz.ch/papers/DeformableContact.pdf)。研究分片表面离散造成的非物理切向力与求解困难，提出平滑表面表示。它支持审查离散接触，不能直接证明本项目本构错误，也不是本项目当前算法的验证。

## 本地检查与解释

准确失败场q=.065 mm，上一接受态q=.06、dt=.05 s，原材料与摩擦历史逐项匹配。检测出的1177个自接触项全部为同一边两个端点的VV；没有非相邻VV或EV，无表面相交。该场dhat=.0025 mm，与参考表面采样间距相同，受压后端点距离可低于dhat。

正常材料压缩并不需要同一条边两端通过额外接触势互相排斥。原Newton方向上的体材料曲率为+.00116558 Nmm，自接触为−.00051012 Nmm，刀具几何为−.00036827 Nmm；这是指定方向的二次型，不是整体稳定性特征值，也不是反力占比。

仅改变自接触适用域：若所有项均为相邻端点，令这部分势/摩擦为零；若出现任何非相邻项，立即拒绝并停止该候选的适用。原自碰撞CCD、整单元正J与整边刀具核查保留。这是在未自接触区间使用无自接触方程的显式诊断，**不是通用自接触算法**，也不在接触切换处静默启用过滤。

同一q=.06→.065增量、同一adaptive策略，旧模型240 s超时；该诊断约19.88 s完成，独立原材料核审查通过。其前态仍来自旧方程，因此只能作条件因果对照；独立路径、深度精度及跨角度实验解释需另行验证。

## 下一步边界

先从零检查路径、成本、积分/时间敏感性。若真正非相邻自接触出现，必须停下并另行实现完整且连续的几何接触势；不扩大邻域排除半径来掩盖折叠。原生IPC基线默认不变。本诊断不能证明历史实验曲率差异已解释、起裂预测已实现或2 mm已可计算。

本地证据：[HTML (retired; summary only)](../outputs/retired_evidence/mode_diagnosis_20261019.json)。
