# 本地Agent任务书：无断裂基线的可信校准与交接

日期：2026-10-31，日本时间

角色：本地完整体基线维护者

参照提交：e64697ded8cf3e712bb1bb14fdf7e1c89a662e91

与远程交接版本：9519895376fe6c57e2f30d1a0d8ca575b9499a39

执行性质：实际修改、运行、验证和交付，不止于建议。

## 0. 任务目标与授权

建立一套在明确的起裂前适用窗口内，用共同材料与接触描述解释0°、33°、57° Fz的基线；交付可复算的拟合目标、数值误差、校准结果和可供断裂线使用的材料包。

用户本轮明确说本地正在拟合实验。因此允许做受约束的共同参数校准；旧交接中no_experimental_force_parameter_fitting=true只描述旧研究任务，不阻止本轮基线校准。保留历史参考配置，创建本轮研究配置并记录此授权。仍不得逐角度修改材料、面外宽度、力倍率或深度零点；当前授权也不自动允许远程用切割力倒调断裂强度和Gc。

先完成不依赖缺失实验的代码、数值核验和证据整理。缺少独立物性时将参数称为有效校准候选，不要求用户先完成庞大实验矩阵才推进。

## 1. 开始时先接续现有工作

1. 读取根AGENTS.md、FE AGENTS.md、最新本地任务与状态、模型规格、数学模型、原型代码图、接口契约。
2. 记录HEAD、分支、dirty文件的路径与SHA256、解释器和依赖、活动任务及正在使用的源码。已运行作业的源码保持冻结。
3. 与本任务列出的发现逐项对照；若本地已修复，提供对应代码和测试证据直接关闭，不重复实现或重新跑完整路径。
4. 对当前拟合的目标CSV、参数、试次窗口、损失函数和已得到结果先作快照；保留原始文件和此前成功/失败证据。
5. 使用当前本地工作区，不因远程快照较旧而覆盖、reset或清除用户工作。

首个输出：简短状态表，注明哪些任务已完成、哪些需修改，以及第一批有界计算的配置/停止条件。完成后继续执行，不停在书面评估。

## 2. 不要丢失的已知事实

- 0/33/45/50/55/57六组已按交接完成2 mm、201态；其中只有0/33/57有授权实验。
- 当前参考：G∞=.14 MPa、Gv=1.24 MPa、τ=1.7 s、K=23.34333333333331 MPa、μ=.35，单支路指数2，P1、平面应变。
- 30×5 mm截面、20 mm面外宽度，偏心s=7 mm，底面双向固定，速度.2 mm/s，2 mm对应10 s。
- 新宏观刀形为低性能A的约13.32°、厚度约.246 mm；20 μm刀尖半径仍为旧假设，不能直接把图像拟合约8 μm认证为实际接触半径。
- 57°曾参与历史选参；当前三个既有曲线上的交叉留出只能称内部检验。
- a1约1 mm标签不能移给0°a2；33°起裂近似；57°到2 mm未裂为工况级删失约束。
- 现有六角度残量/J审查不是空间、时间、实验验证。
- 公共Fy映射solver reaction_x_n；Fz是实验室分量。初始表面N/T投影、刀面法向/摩擦分量与实际滑移应分别命名。[B1、B2、R1]

## 3. A1：首先统一实验比较与拟合口径

### 3.1 修复公开入口的不一致

检查：

- projects/softmodeling/tools/run_material_candidate.py，尤其compare第64–89行；
- projects/softmodeling/tools/plot_experiment_angles.py；
- 当前本地拟合脚本与上层报告适配器。

原始CSV读取保持无修改。统一派生层：

F_exp_target(d) = F_exp_raw(d) − F_exp_raw(first sample)

F_FE_target(d) = F_FE_raw(d)

57°原首值为0.51 N；另两条授权输入为0。深度仅μm→mm，不另拟合平移。原始目标、零偏、原始hash、单位、处理版本与派生目标hash一起记录。既有“双方扣首值的形状误差”可以保留为辅助指标，但不可叫同一个目标RMSE。

### 3.2 建立试次与适用窗口清单

每条记录包含trial_id、实际角度、刀具身份、润滑、速度、宽度、首触定义、起裂证据来源/不确定区间、用于校准的深度区间和排除理由。未知保持未知。

0–0.5 mm可作为优先诊断窗，但不能仅因短就声称已经独立认证无裂。若暂无可靠起裂标签，明确采用保守窗口假设，并检验改变窗口后的参数稳定性。包含晚段的全程误差仍可以报告，不能直接作为完整体材料拟合目标。

### 3.3 验收

- 合成数据raw_exp=[.51,1.51]、FE=[.2,1.2]，共同网格下目标RMSE应为.20 N；不得得到.31或0并称为当前目标。
- 同一输入在拟合、绘图、比较报告中生成同一目标和指标。
- 采用固定共同深度网格/积分规则，只在实际共同支持内插值，不外推。
- 优化失败、数值失败和窗口外数据不被赋予“误差很小”。

## 4. A2：补齐能量和耗散的独立审查

现有audit_state主要比较材料度量和Fz；累计审查基于保存increments。复用新鲜重算对象，增加各项已有字段的独立对照：

- 真实Δt和上一接受态身份；
- 平衡、非平衡、总体储能；
- 黏性耗散增量及累计量；
- 增量势与物理储能分别命名；
- 接触/摩擦功中实际可重算的部分及其审查范围。

累计量由fresh increments累计，不只核对旧记录的自洽性。沿用已有力残量、J、CCD及材料历史门，不通过改变物理量定义来降低误差。

增加最少的反例回归：同时篡改某步耗散与之后全部累计值，使保存账彼此自洽；新审查仍必须发现。储能分项篡改也须失败。选帧总储能审查与全路径审查资格独立标注。

先用一个现有短前缀和一个有意篡改记录验证逻辑。通过后按需审查要交给远程的代表路径；不因补一个字段而无条件重跑全部FE。

## 5. A3：用最小数值诊断矩阵决定拟合应改什么

### 5.1 优先利用保存态

从相同状态及相同摩擦滞后对象，分别直接组装法向、摩擦梯度。验证刀具/材料作用反作用与区域身份，区分圆鼻、楔面、刀身及可能的自接触。总力减法向力只是记账恒等式，不能充当独立分解验证。

输出小J位置、体积加权分位数和占比、局部能量占比及靠近刀尖/边界的分布。不能从最小J直接推断全域可压缩，也不能把J>0当作局部物理合格。

### 5.2 有目的的新算例

先选0°、57°的共同短窗，必要时在33°确认最有区分力的结论。已有等价证据可复用。第一批分别改变：

1. 体网格；
2. 刀面接触离散；
3. 时间步与摩擦正则的明确定义。

每组固定物性、几何、边界和其它数值参数。先做少数成对对照；只有影响结论时才增加第三档与更多角度，避免全因素组合。

当前Maxwell入口要求μ>0。若使用严格μ=0控制，应补显式支持，验证同材料历史、零摩擦装配/功和回滚；μ极小不是严格无摩擦。μ=0是改变物理的诊断，不能从有摩擦解减去一个分量替代重算。

固定摩擦位移平滑εu时，改变Δt会改变εu/Δt。若目标是同一速度正则下的时间收敛，需要显式固定εv并传递εu=εvΔt；旧固定εu基线另存。不能一次改变正则和时间步后只归因于时间离散。

### 5.3 何时升级单元或三维

先确认实际切割对照中的离散误差量级，再决定是否投入P2/P2-P1或有限宽3D。降低K会改变材料，不等于抗锁定。混合形式需验证真实有限变形、积分点历史、边界中点、切线、全单元J和接触兼容；P1/P0局部凝聚等价性可作必要控制。

这是决策条件，不是要求本轮必须同时完成高阶单元、三维和新本构。

## 6. A4：执行受限的共同参数校准

1. 冻结完成上述检查后的数值配置与观测处理。
2. 根据敏感性选择少量共同参数；不要同时放开G∞、Gv、τ、K、μ、半径及非线性指数。
3. 材料跨角度共享；同界面条件共享接触描述；刀间几何差异须由试次证据支持。
4. 给出参数范围和来源。没有独立试验时，不把先验假设写成测量值。
5. 采用按试次/角度平衡的损失，避免点数多或自适应步密集区域占主导。归一化尺度固定且非零，不用逐点近零力做分母。
6. 按整个试次或角度留出，不能随机拆同一条曲线的深度点。若数据已反复用于模型选择，明确是内部交叉检验；真正独立验证需后续新试次。
7. 报告每角度偏置、RMSE/relative L2、阶段斜率、不同初值解和参数相关性。至少检查是否有明显不同参数给出近似同一Fz。
8. 若共同参数不能同时解释三角度，保留失败，并用A3证据定位需改变的物理/离散假设；不能逐角度加倍率消除残差。

判断“升级成功”要同时看数值可信度和预先指定的实验指标。不要为了保住57°旧吻合而隐瞒0/33退化，也不要因更准确的离散使Fz暂时变远就否认数值改进。

## 7. A5：晋升及远程交接

交付一个版本化包，至少包含：

~~~text
manifest.json：commit、dirty source hashes、依赖、配置和输入hash
material.json：模型ID/版本、全部参数/单位/来源、校准状态
conditions.json：试次身份、几何、边界、刀位、接触时间律、有效窗口
comparison.csv + report.html：各角度原始/派生观测、误差与资格
audit.json：逐态资格及每类能量/耗散覆盖范围
representative_states/：必要保存态、父态和history
reproduce.md：命令、运行预算、实际耗时、失败与限制
~~~

生成证据遵循既有本地结果管理，不将原始实验、NPZ或媒体强塞进公开Git。

当前场导出只支持特定单Maxwell/指数2/P1形式；远程离散能量核也有限制。若更改本构或单元，扩展对应应力/储能/耗散重建，或显式拒绝不支持的输出。远程导入前共同执行同一变形历史下的应力、储能、耗散对照；不移植旧内部变量。

最终三实测角度尽量提供完整2 mm数值路径；起裂后区间仅作为完整体控制，不能当无断裂实验适用区间。空间/时间敏感性、校准效果、独立实验验证分别给状态。

## 8. 本轮完成标准与汇报格式

第一轮必须交付A1、A2及最小A3的真实结果；然后继续A4，不停留在“建议拟合”。已有本地工作若已达到这些要求，直接以证据进入下一阶段。

报告固定回答：

1. 哪个问题已确认，哪个只是候选解释？
2. 修改了哪段代码，哪一条已有行为被保留？
3. 哪些实际运行/复算已完成，配置与结果在哪里？
4. 数值变化对各角度Fz的量级是多少？
5. 校准与检验用了哪些试次/窗口，参数是否可辨识？
6. 哪些字段/区间尚未合格？
7. 给远程的模型是什么，哪些能力仍不兼容？

不得以“全套测试通过”“更接近实验”或“有2 mm曲线”单独代替以上证据。

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
