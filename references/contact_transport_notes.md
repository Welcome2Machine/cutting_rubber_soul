# 接触材料坐标、客观滑移与几何离散

核读日期：2026-10-21。以下区分文献方法与项目候选，没有把局部测试称为论文复现。

后续实现：[刚性刀具的分段路径滑移](../docs/transported_blade_validation.md)保留P1材料边界，
对圆鼻/直面连续切向进行路径积分，避免用不连续曲率度量缩放有限滑移。
再次核对Duong–Sauer第3.3节关于瞬时切向相对速度和有限步近似的区分；本地路径积分仍不是运动摩擦锥实现。

## Duong–Sauer 2019

Thang X. Duong, Roger A. Sauer. *A concise frictional contact formulation based on surface potentials and isogeometric discretization*. Computational Mechanics 64, 951–970. DOI：[10.1007/s00466-019-01689-0](https://doi.org/10.1007/s00466-019-01689-0)。[作者全文HTML](https://arxiv.org/html/1808.10420v2)。

阅读范围：第3.1–3.4节、4.1–4.3节及线性化附录的结构。方法区分当前最近投影、旧相互作用点和滑移点；采用相互作用间隙与运动摩擦锥条件，并在表面离散和积分中形成接触力及切线。参考面积和名义接触牵引的配对不能被任意点权重替代。法向与切向运动学的一致性以及跨单元滑移的身份追踪是直接相关的实现要求。

项目采用的启发是固定材料参数、完整投影导数和统一表面积分。当前`surface_contact_transport.py`仍采用IPC式屏障和正则库仑残差，其后向欧拉传输增量与激活路径是本地候选；并未实现该文的运动摩擦锥局部求解，也不继承论文完整热力学论证或全局稳健性结论。

## Tur–Giner–Fuenmayor–Wriggers 2012

M. Tur, E. Giner, F. J. Fuenmayor, P. Wriggers. *2D contact smooth formulation based on the mortar method*. DOI：[10.1016/j.cma.2012.08.002](https://doi.org/10.1016/j.cma.2012.08.002)。[出版商页面](https://www.sciencedirect.com/science/article/abs/pii/S0045782512002381)。

本次获取范围是索引返回的出版商摘要及引言；直接页面请求403，机构PDF未成功获取，不记为全文阅读。论文在二维大变形无摩擦接触中，将C¹平滑几何与段到段mortar积分结合，并在积分点处线性化。适用启发是：平滑后的间隙、接触区域积分和切线要作为一个整体处理。无摩擦结果不能直接证明本项目带历史摩擦、自接触和IPC屏障的实现有效。

## 项目中得到的可核查结论

[八场资格核查](../docs/contact_transport_qualification.md)发现：自然样条虽未与刀具发生静态交叉，仍改变部分原生接触的激活状态。33°、0.5 mm的最大同参数几何差达到1.35046 μm；这不是文献给出的数据，也不是法向力误差估计。

据此，下一实现优先保留P1材料表面，使用解析刀具主面及明确的参考面积积分。该选择是项目的实施判断，尚待整体积分、CCD、历史和连续加载验证。不得从文献名称推断旧解析路线或新局部候选已经解决57°失败。
