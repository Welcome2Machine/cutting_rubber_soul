# 建模路线

目标与参数以 [任务要求](task_requirements.md)、[模型规范](current_model_spec.md) 和原始图为准。实验力及光流/DIC 由用户保留作独立验证，不提前索取或用于反向调参。结果仅在 [求解状态](solver_status.md) 更新，公式见 [模型说明](initiation_model.md)。

| 阶段 | 当前选择 | 验收门槛 |
|---|---|---|
| 材料 | 共享有限应变 Ogden/Maxwell 内核；CB245NE 用既定弹性极限 | 给定 G、K；不套用 EPDM 高阶项/松弛谱；独立能量和切线检查 |
| 断裂前接触 | Python P2 完整积分、二次边界积分；修正版 FEBio 独立对照 | 连续越过已知失败区，严格平衡、接触、整边穿透及耗散；网格敏感性 |
| 起裂候选 | 全域应力、伸长及名义强度，位置和加载区间 | 不预设刀尖下方裂纹；候选与实际萌生区分，排除固定端/单元尖峰 |
| 自由断裂 | 后续耦合独立强度与断裂能控制的有限应变相场 | 无缺口萌生、受压裂面、损伤区体积约束、新裂面刀具接触及能量核算 |

## 方法依据与边界

近不可压缩橡胶的局部剪切/弯曲使低阶逐单元体积约束容易锁死。节点体积平均可改善柔度，也可能产生局部软模式或压力振荡；固定父压力空间在位移细化后也需稳定性检查。P2 完整积分被选为当前 ν=0.49 下的较简单主线，对 ν→0.5 或任意大变形不作普遍免锁死保证。低层比较离散保留作真实失败回归。

[Khandelwal 等](https://arxiv.org/abs/2209.04906) 的二次边界节点和半边一次接触力空间提供参考，但其线性无摩擦假设不覆盖本任务；[Tian 等](https://arxiv.org/abs/2112.00294) 的有限应变混合框架也不能证明一种单元对所有问题最好。本仓库按实际 F、虚功、历史和整边几何独立验证。

修正版 FEBio 优先作为断裂前接触对照：[Zimmerman–Ateshian 模型库](https://repo.febio.org/permalink/project/68)、[官方节点—面接触](https://febiosoftware.github.io/febio-feature-manual/features/solid_surfaceinteraction_sliding-node-on-facet/) 和固定 v4.13 源码支持核查。Hex8 三场体积压力平均与 P2 完整积分不同；解析 patch 和短前缀成功不能替代应力场网格验证。已证实的切线及增广问题见状态页，不能把更多线程或罚参数变化本身称为物理改进。

若该对照不能稳定且可信推进，再评价 [GetFEM 大滑移](https://getfem.org/userdoc/model_contact_friction_large_sliding.html) 或 FEniCSx/PETSc 分块求解。已查看的 ASiMoV 摩擦圆柱示例采用小应变线弹性，不能直接替换本任务。任何替代都需保持有限应变、真实摩擦、历史与严格验收。

[Dammaß 2023 复现](../reproductions/dammass2023/README.md) 的 EPDM 预缺口平面应力与 N60A 分开。经典相场的长度尺度不能随意兼作强度标定；参考 [方法笔记 R09–R11](../references/method_notes.md)，再处理强度和断裂能的独立作用。先解决其裂后未核算功，再迁移断裂演化；回撤后置。

完整运行由日志和状态记录，只检查一次启动。超算先做平台和物理短验证，见 [运行准备](hpc.md)。资料依据与局限见 [参考索引](../references/README.md)；视频与其量化说明仅留本地。
