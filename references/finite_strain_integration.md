# 大变形积分与局部单元退化：本轮依据

> Historical technical record. Obsolete outputs were retired on 2026-10-20; summary links do not include the original figures or full replay histories. See [current research status](../docs/research_status.md).


核查日期：2026-10-19。本页是针对当前失效的有限文献核查，不是系统综述。

Chi, Talischi, Lopez-Pamies, Paulino (2016), *A paradigm for higher-order polygonal elements in finite elasticity using a gradient correction scheme*, CMAME 306, 216–251，DOI [10.1016/j.cma.2015.12.025](https://doi.org/10.1016/j.cma.2015.12.025)。[作者页面](https://paulino.scholar.princeton.edu/publications/paradigm-higher-order-polygonal-elements-finite-elasticity-using-gradient-correction)与[作者稿](https://paulino.princeton.edu/journal_papers/2016/CMAME_16_AParadigmForHigherOrder.pdf)。论文讨论有限弹性中积分误差和单元一致性，采用子三角形积分并检查更高积分精度。其对象为多边形单元，不能直接把其梯度修正移植为本项目 P2 的必要修复；此处仅借鉴独立积分检查与正权重子域积分的验证思路。

同作者 (2015), *Polygonal finite elements for finite elasticity*, IJNME 101, 305–328，DOI [10.1002/nme.4802](https://doi.org/10.1002/nme.4802)，[原文](https://pamies.cee.illinois.edu/assets/pdf/IJNME_2015.pdf)。论文强调橡胶宏观变形尚可时，局部强变形已经可能阻碍离散求解；这说明应检查局部场，不能从全局下压量判断单元是否安全。该文的混合多边形方法不是我们当前的位移 P2 方法，也不构成允许本项目接受翻转单元的依据。

## 本地直接证据及决策

`tools/diagnose_al_geometry.py` 在最后接受态定位到单元 48307 的内边，参考坐标 s≈7.090596、n≈4.999404 mm，距离原表面约0.596 μm。整单元 J=0.0021612，而9个材料积分点的最小J=0.157849；失败迭代的同一边J约6e−13。直接梯度行列式与二次多项式最小值一致，支持真实的离散映射局部退化，而非仅由打印舍入造成。

固定接受位移历史，选取20个邻域单元，以新积分点独立从初始材料状态重放全部时间历史。3阶控制复现原历史，最大材料度量误差3.01e−11。末态5/8/10阶相对3阶的局部装配内力变化分别约0.468/1.253/1.523 N；能量变化很小并不能认证内力积分精度。这是局部条件场检查，**不是候选平衡场，也不是刀具反力变化**。

正权重子三角形复合规则也完成固定场检查。3阶、一次/二次细分分别需要36/144点，在该局部场没有显示比25/64点高阶规则更好的成本优势，因此完整路径选择5阶与8阶对照。保留可选复合规则及一致性测试，不把它设成默认。

所有新积分点必须有自己的材料历史，不从旧积分点简单插值继承。新路径从静止独立开始，只能续接自身同规则前缀。新的几何试探检查只拒绝浮点表示后的非正J端点；不改变材料能量、摩擦、整体残差阈值或正J要求。

当前只确定“局部映射接近折叠且邻域内力积分未收敛”。尚未证明增加积分点能解决完整路径，也没有排除局部网格空间、约束或接触离散的影响。若独立高阶路径仍退化，不继续无上限增加积分阶数，下一轮转向该区域的空间离散与变形映射。

本轮实时证据：[HTML (retired; summary only)](../outputs/retired_evidence/al_geometry_20261019.json)。
