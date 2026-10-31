# 远程Agent任务书：先验证接触进入，再恢复裂纹扩展

日期：2026-10-31，日本时间

角色：远程断裂实现负责人

审查起点：codex/server-fracture-20261029，9519895376fe6c57e2f30d1a0d8ca575b9499a39

执行性质：实际实现、局部验证、实际案例与可重放交付。

## 0. 当前任务与最终目标

最终目标仍是基本二维断裂仿真：模型产生分离，刀具进入裂口并继续加载，能看见载荷重分配，随后扩展到起裂和斜切实验对照。

本轮先解决限制这一目标的接触时间推进。第一交付不是又一条更长但未进入裂面的前缀，而是一个运动学、冲量、功和几何安全同时可信的接触参考及纯预裂进入案例。通过后恢复面级裂纹扩展和完整材料激活。

可以用明确标注的示例强度、Gc与预设裂路验证机制。无需等待本地基线完美拟合或完成全部独立试验。但不能用实验Fz、57°未裂约束倒调断裂参数来制造通过，不能把示例参数触发称为实验起裂预测。

## 1. 先接续最新状态，避免重复第六轮

读取根/FE AGENTS.md、接口契约、最新research_status、fracture_implementation第28节及本地更新。记录真实HEAD、dirty源码hash、活跃队列、运行时和资源规则。若已有更新覆盖本任务，给出证据直接跳到未完成项，不回退到本审查快照。

应保留的已完成工作：

- typed failures与全局预算/物理cutback区分；
- P1面级面积、非对角牵引积分、Gauss历史与接触暴露；
- 完整面精确连接、拓扑回滚；
- 当前步提交前的端点强度检查；
- 预裂已有Gc排除和完整分项功账；
- 原顶面、裂口角点、严格内部裂面的接触身份判别。

不要把这些再次列为从零开发任务。

最新运行依据：

- C1端点预裂：.468422401381 mm、2275态；
- C2完整材料：.517955765798 mm、1777态；
- 两条前缀审查通过，刀具进入均未验证，完整事件/时间/网格门未过；
- 旧FFM生产最远1.210936593388 mm，不能用C1/C2较平滑短段宣布解决旧FFM；
- C1现有0.5 mm预裂仍允许新增激活和损伤，不是纯接触隔离控制。[F1、F2]

## 2. 本轮诊断假设及其边界

生产代码以 h v_end=2Δu−h v_old 驱动摩擦，但位移运动学为中点。其D_f=g_f·h v_end与W_alg=g_f·Δu可异号，已产生大量真实能量拒步。[F3、F4]

新方法要同时规定：

- 位置与速度更新；
- 体/内聚/接触动量；
- 刀具规定运动和外功；
- 法向与切向冲量及其共轭相对速度；
- Maxwell物理时间与历史更新；
- 接触模型储能、物理耗散和数值功；
- 实际步内CCD/J与接触身份。

只更换一个速度公式、减小摩擦、加阻尼、提高质量、删除负算法功检查，均不是合格修复。

“能量一致的硬接触—摩擦冲量参考”是本轮优先候选，不是预定正确答案。若反例否定它，交付失败和清楚的替代方向；不要继续以相同末态无限减半。

## 3. F1：先补齐最小可重放证据和资格字段

### 3.1 逐次激活见证

face_cut.propose_insertion已保存内存activation_positions，但最终文件缺少逐次激活前完整状态。增加独立见证：

- 父接受态hash和当前物理dt；
- 激活前positions及原始increment；
- 材料、面损伤、速度与激活前拓扑；
- 候选合力超额、候选排序和所选面；
- 端点强度与算法交接判据分别的输入/结果；
- 零位移拓扑屏障能跳和接触身份。

独立工具从见证恢复并重算候选、交接及功条件，不只把原face_events装回对象。先在一个短触发案例验证，不重跑全部历史长路径。

### 3.2 代表拒绝态

当前可回退分支只记录分类/标量，最终退出才落盘一个failed_trial。按失败签名保存首个、最严重和停止前代表，去重并设总量上限：

- 真正failed positions/increment；
- 父态与预测器来源、physical dt、失败阶段；
- 末速度、摩擦梯度、D_f、W_alg、R_f、门限；
- 面/接触拓扑、回滚后hash；
- 源码/配置/运行时身份。

必须验证失败状态与实际attempt一致，回滚后已接受历史逐位不变。原历史缺失的数据保持未知；“同源码重算”不能改名为逐位重放。

### 3.3 输出语义

将initiation_predicted、surface_connected等宽泛字段拆清：

~~~text
preexisting_connected_length_mm
new_connected_failed_length_mm
model_activation_occurred
activation_witness_verified
blade_contacted_strict_new_face
physical_initiation_prediction_validated
~~~

具体字段名可以沿用现有schema风格，但必须区分初始预裂、模型事件、数值资格和实验验证。添加最少反例：只有预裂、只有释放但无新裂面进入、事件见证缺失时，不得升级物理预测资格。

另对遗留FFM检查finite_increment + backward_euler + constant_traction组合：现有冻结虚拟势为secant。最小处理是明确拒绝不一致组合；该问题不属于当前face/secant主线的失败根因，不为它启动长计算。

## 4. F2：构造可证伪的功/冲量参考

### 4.1 先写清一套离散方程，再实现一个候选

推导应足够对应代码：惯性、内力、接触冲量、速度与位置、刀具功、Maxwell耗散、内聚功和数值项分别是什么。符号与单位明确；力残量N和冲量残量N·s不能混用。

可参考Moreau–Jean/Frémond及Siconos子问题，但其线性体动力学/接触矩阵等前提不能自动覆盖本项目的有限变形Maxwell与软化。本轮不要求把整个工程迁入Siconos。复用现有材料、面级核、几何候选和CCD，建立一个最小独立参考即可。[T3、T4]

同一接触不叠加IPC法向屏障与硬法向冲量。若用硬接触，其含义是无穿透约束，橡胶仍为可变形体。恢复系数如e=0属于明确模型假设，不冒充实测。位置修正、接触活化、法向更新的功必须记录。

### 4.2 首个否证测试：纯摩擦停滑

复用仓库已有反例，至少验证一个无弹性回弹、无外切向驱动、恒定法向载荷的质量块：

- 正确的停止时间；
- 端点速度，不出现无驱动力伪反向；
- 摩擦冲量和动量；
- 物理耗散与数值项。

平均速度为零、端点速度反号而算法功为零的构型不能被当作正确停滑。非负耗散不是充分条件。

另加运动刚壁或刀具滑移/接触小例，验证相对速度、刚壁外功、法向冲量与切向冲量。接触法则应有解析或独立高精度对照。

### 4.3 有限验证与停止条件

固定相同物理参数、几何、初态和共同绝对时间窗，做三档分辨率。开始前定义端点速度/停止时间、冲量、功、非穿透及数值误差容限，来源为解析误差目标和已有力学门，不能事后改到通过。

允许一轮三档对照和一次针对性修正。若仍不能得到正确停滑或可改善的误差，停止向完整FE迁移，交付反例与失败关系；不加第四、第五档去追一个未解释的极小dt。

## 5. F3：实际C1现场的条件接触参考

取一个已有合格C1锚点和一个新保存的负算法功代表试探。使用同一质量、几何、接触身份与非接触力，先作接触子问题；再逐层恢复体材料、Maxwell和面内聚。

输出同一时间窗内：

- 端点速度、质量加权速度误差；
- 法向/切向冲量及动量残量；
- 刀具功、摩擦、冲击/接触数值耗散；
- D_f与W_alg的关系；
- 接触状态和几何安全；
- 原方法与新参考的差异及归因范围。

旧IPC锚点含屏障储能。条件子问题可以明确转换模型并重新定义基准账，但不能把它拼接为旧生产续算；完整新路线必须从自己的未加载初态生成历史。不得通过位置投影或删屏障储能暗中平账。

## 6. F4：先完成纯预裂进入，再恢复裂纹演化

### 6.1 纯预裂接触验证

新增显式模式：已有预裂面保持其初始失效状态，其余完整面禁止新激活/新损伤。保留真实材料、质量、加载与接触。

先选择几何和边界明确允许进入的简单验证样例，检验实现能力；它是规定初始裂缝的算法基准，不是完整材料起裂预测。预裂既有Gc不重复计入，新生裂面不存在时新增断裂耗散应为零。

验收必须有：

- 刀具确实接触严格内部裂面，非原顶面/裂口角点；
- 裂面可张开、闭合并防穿透；
- 轨迹CCD、正J、动量/功账；
- 可复现完整窗口和至少一组有区分力的时间细化；
- 原始位置、速度、反力和接触身份。

然后用实际C1的0.5 mm预裂、原几何做同样控制。若仍物理闭合，保留阴性结果并分析接触合力和进入路径，不能免费打开顶端或取消合法约束制造进入。第六轮已说明某C1顶点受真实压向合力闭合。

### 6.2 恢复C1扩展

接触参考有效后，恢复当前面级连续内聚、事件定位和历史更新。保持独立示例T、Gc和材料；不把FFM虚拟门再次叠加。

至少完成一个短而完整的裂后代表窗口：已有裂口进入、新面损伤/完全分离、刀具继续加载和载荷重分配。记录新增连通完全失效长度、初始预裂长度和新增Gc，且Gc按参考面积只支付一次。内部损伤不自动算表面通道。

### 6.3 再推进C2与2 mm

通过前述局部机制后，从完整材料初态验证C2模型自产生事件，随后推进2 mm。2 mm是刀具行程，不是切穿5 mm试样。

严格区分：模型激活、材料完全分离、表面连通、刀具进入、仿真力转折。未经参数和实验验证，前者不叫真实起裂预测。

当前0°居中镜像核通过后，再规划偏心s=7 mm、33°/57°与一般混合模式；不能用0°镜像结果证明斜切断裂。

## 7. 完整事件和资源管理

- 原始Fz、算法区间平均Fz、瞬时端点力与冲量分别保存。
- 不要求Fz人为单调，也不裁剪负摩擦Fz；负竖直分量不等于负摩擦耗散。
- 完整事件窗覆盖事件前状态及事件后可解释的响应；若末动能仍高且未稳定，标明不完整，不用全程净误差掩盖。
- 给出逐步绝对数值项累计、净能量余项和分项，区分抵消。
- 每项运行都有实际墙钟、资源、停止条件；沿用已授权资源与调度规则，不复活旧队列。
- 若重复失败由相同关系导致，保存现场后停止扩展前缀；只在改变了可验证假设后重跑。
- 当前时间门未过时，不展开大型网格/自由裂路/多角度矩阵。
- 缺原始保存态时先用小基准和自己的新初态运行；不假造历史续算。

## 8. 与本地agent的交接

本地正在校准完整体有效参数；你先使用冻结示例参数验证方法。接收新包时核对材料模型、指数、Maxwell支路数、内部变量、积分点、体积能、几何、刀位、接触律、单位与来源。

当前离散能量核使用参考单Maxwell/指数2形式；不支持的模型必须拒绝或正确扩展。用相同变形历史比较应力、储能和耗散，再做同几何无损伤回归。不得将偏心完整体与居中镜像控制直接比较后称“断裂关闭不一致”，也不得移植异构历史。

57°到2 mm未裂只在真实57°几何、相应试次条件与材料包均接入后作为验证约束；当前0°模型不能直接检验它。

## 9. 交付与验收

首批交付：F1的代码/反例、F2的方程与基准结果、F3/F4最有区分力的小范围运行。完成一个步骤后继续下一个可开展步骤，不以“需要更多实验/预算”泛泛结束。

交付沿用已有状态文档、HTML/JSON/CSV与研究运行目录：

1. 精确源码、配置、运行时与输入身份。
2. 可重放激活见证、代表拒绝态及回滚验证。
3. 解析/独立参考与三档对照，失败同样保留。
4. 纯预裂进入结果及严格裂面接触身份。
5. C1扩展/C2激活/2 mm各自达到的真实范围。
6. 物理功、数值功、完整事件和时间/网格资格。
7. 本轮否定了什么解释，仍剩什么具体问题。
8. 下一步仅一个最高优先级任务及可证伪条件。

本任务的合格中间成果可以是接触参考和基本裂后窗口；最终完整2 mm、斜切及独立实验预测仍是后续目标。中间成果必须如实命名，不以报告字段把未完成项提升为通过。

## 来源与版本

以下代码与文档链接固定到本次读取的提交；运行结果数字属于对应仓库报告，本轮没有取得原始实验 CSV 和全部保存态重新计算。

- [B1 基线最新交接](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/e64697ded8cf3e712bb1bb14fdf7e1c89a662e91/docs/intact_baseline_upgrade_task_20261031.md)
- [B2 基线输入和比较约定](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/e64697ded8cf3e712bb1bb14fdf7e1c89a662e91/configs/intact_baseline_upgrade_20261031.json)
- [B3 当前实验比较实现](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/e64697ded8cf3e712bb1bb14fdf7e1c89a662e91/projects/softmodeling/tools/run_material_candidate.py#L64-L89)
- [B4 完整体独立审查](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/e64697ded8cf3e712bb1bb14fdf7e1c89a662e91/projects/softmodeling/prototype/ipc_viscoelastic.py#L150-L184)
- [B5 完整体累计耗散审查](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/e64697ded8cf3e712bb1bb14fdf7e1c89a662e91/projects/softmodeling/prototype/material_case.py#L157-L183)
- [B6 场导出及其支持范围](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/e64697ded8cf3e712bb1bb14fdf7e1c89a662e91/projects/softmodeling/workbench/results.py)
- [F1 服务器最新研究状态](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/9519895376fe6c57e2f30d1a0d8ca575b9499a39/projects/softmodeling/docs/research_status.md)
- [F2 断裂方法和第六轮证据，第28节](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/9519895376fe6c57e2f30d1a0d8ca575b9499a39/docs/fracture_implementation.md)
- [F3 面级连续内聚实现](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/9519895376fe6c57e2f30d1a0d8ca575b9499a39/projects/softmodeling/face_cut.py)
- [F4 端点摩擦实现](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/9519895376fe6c57e2f30d1a0d8ca575b9499a39/projects/softmodeling/endpoint_friction.py)
- [F5 面级路线运行、快照和审查](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/9519895376fe6c57e2f30d1a0d8ca575b9499a39/projects/softmodeling/tools/run_fracture_route.py)
- [F6 当前步端点事件入口](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/9519895376fe6c57e2f30d1a0d8ca575b9499a39/projects/softmodeling/tools/run_endpoint_face_route.py)
- [F7 有限裂增量遗留路线](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/9519895376fe6c57e2f30d1a0d8ca575b9499a39/projects/softmodeling/prototype/finite_increment_cut.py)
- [R1 研究主题原文](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/e64697ded8cf3e712bb1bb14fdf7e1c89a662e91/researchtopic/oblique_cutting_experiment_and_research.md)
- [R2 原断裂任务书](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/e64697ded8cf3e712bb1bb14fdf7e1c89a662e91/researchtopic/fracture_agent_task_brief.md)
- [T1 Dammaß et al. 2023：有限变形黏弹断裂与驱动力选择](https://doi.org/10.1007/s00466-023-02310-1)
- [T2 Kamarei et al. 2025：黏弹性、强度、断裂能及准静态相场理论](https://pamies.cee.illinois.edu/assets/pdf/CMAME2025b.pdf)
- [T3 Acary & Collins-Craft 2025：非光滑接触、摩擦和冲击的能量时间离散](https://jtcam.episciences.org/15960/pdf)
- [T4 Siconos 官方摩擦接触问题及求解器说明](https://nonsmooth.gricad-pages.univ-grenoble-alpes.fr/siconos/users_guide/problems_and_solvers/friction_contact.html)
