# 当前研究状态

2026-11-27，京都大学 System B 接手 `codex/deformation-sliding-baseline`，本地 agent 暂停。服务器工作根为 `<USER_WORK_ROOT>`。[唯一任务书](../../../researchtopic/research_prototype_task.md)

## A/R20 的 0–0.25 mm 数值资格

### System B 的新生产与条件对照

2026-11-27 16:38:36 JST 的启动快照中，独立作业 `27288606` 从参考态复核了 33° H1 的 2 个新态，深度为 0 与 0.0025 mm。首个正深度的自由力残差为 4.54736×10⁻⁶ N，最小 J 为 0.992345，整弦安全比例为 1，Fz/Fx 为 0.04399636/0.00828311 N。主观测间隙 0.5 µm 下，首个正深度的刀鼻材料地址区间为 [−5.164, 5.528] µm。两帧均无合格严格极点，唯一 SA 留空。采样时 H1 运行，H0/H2 等待同角度放行。

33° 批次 `27288605` 从无应力参考和单位 Maxwell 度量开始。H1 的严格零点目标为 10⁻⁶ N，正深度维持 10⁻⁴ N。H1 完成至 0.25 mm 后须通过原生全链审查、同角固定观测间隙接触对照、全窗力/力矩和能量账，才放行同角 H0、H2。本批采用 4 物理核、6 GiB、14 小时上限。三档实际表面 h 依 H0/H1/H2 为 7.5/5/3.3333 µm，对应节点 9519/20885/47018。33° 与 57° 的 Fz、Fx、固定材料位移、ROI、SA 集合、唯一 SA 及接触事件的整窗空间资格均为探索。

57° 从归档 H1 父态 0.17984375 mm 比较了两种方案。每个预测和摩擦 PTC 更新都须保持与接受父态相连的整弦可行性。整步对照 `27288305` 保住了整条弦，末次自由力残差为 0.01901627 N。第二项 `27288458` 在已知穿越之前引入真实子步。0.179853515625 mm 通过独立审查，残差 9.69451×10⁻⁵ N，最小 J 为 0.714157，真实 Δt 为 4.8828125×10⁻⁵ s。剩余增量到 0.18 mm 的整弦安全比例为 1，残差仍为 0.01929498 N。两项用完各自 900 秒求解预算，均拒绝 0.18 mm，拒绝前的坐标、Maxwell 度量和累计耗散保留。原 78 态生产链仍止于原父态。57° 新 H1/H0/H2 均只准备了配置，在本轮停止。

迁移重验通过原拒绝路径与仿射回放各 20 个有理数穿越见证，以及两角度各自的严格零点与首个正深度。作业 `27288237` 经登录端请求合作暂停，`27288275` 在新 allocation 上恢复，原四份检查点 SHA 一致，六态均通过复核。33° H1 同输入 4/8/16 核两步耗时 33.521/33.639/33.276 秒。[环境与操作](../../../docs/environment.md)

运行源码为 `fac912e903779219f7069aa524bda1c52d0cff79`。证据包 `intact-R20-systemB-shallow-start-20261127.zip` 含运行控制、资源实测、修复对照的接受/拒绝检查点、原始迭代曲线、力和三带宽 SA、启动快照及真实 P1 场。SHA256 为 `693a3af89d9fb197ff041897605cc7c5e3528148d91294e69653bdcc0c9b9da3`。附件交付记录见 [Issue #1](https://github.com/Welcome2Machine/cutting_rubber_soul/issues/1)。

### 接手时已发布的 H1 与 SA 证据

33° 原 H1 有 101 个独立审查态至 0.25 mm。57° 原 H1 有 78 态至 0.17984375 mm。最后一个 0.00015625 mm 增量的真实时间为 0.00078125 s。以原父态、两态预测历史和原生求解器重算，割线预测在 0.00422362 N 停滞。仿射预测的末态残差为 8.90×10⁻⁵ N，但直线路径穿越原表面。重算与原拒绝末态最大坐标差为 4.79×10⁻¹⁰ mm。

刀鼻全局顶点 20901 与材料边 8069–8075 首先限制碰撞步长，其材料坐标为 7.085–7.090 mm。原生整步 CCD 界为 0.0472412109375，单元翻转界为 1。对全部 67 个候选的有理数计算证实，到路径比例 0.06932567 都分离。到 0.06952567 已有边内穿越。41,096 个三角形沿整条路径的最小 J 为 0.703398。对这一个事件的两个真实半步，审查通过 0/2，最后接受深度为 0.179843750 mm。 原割线分支使用完整迭代额度后残差为 5.511e-05 N，仍在同一碰撞对被拒绝。减半时间步和放开提前停滞均未产生新的接受态。

两角度从各自的无应力参考态和单位 Maxwell 度量建立了共同的严格准备规则，0.005 mm 准备段的求解目标为 10⁻⁶ N，正深度恢复 10⁻⁴ N。每个严格零点与首个正深度态的父构形、材料度量和增量均已独立复核。33° 原零点整体力 / 力矩偏差为 0.10364% / 0.16517%，严格零点为 0.000103% / 0.000162%。57° 原零点为 0.02299% / 0.03906%，严格零点为 0.00000505% / 0.00001138%。严格准备对照和两条原生产历史分别留档。

固定 S₀=7 mm，以独立几何间隙 0.5 µm 输出严格刀尖和完整刀鼻材料地址区间，另查 0.125、1.25 µm。33° / 57° 的严格刀尖分别在 90/101、28/78 帧唯一合格，刀鼻有多个分支的帧数分别为 66、15。两角度同一构形改变观测带宽产生的集合最大差为 10.949 / 9.873 µm。固定宽观测带比较新旧接触设置，集合最大差为 8.709 / 12.058 µm。57° 的两条接受时序从 0.155 mm 后分开。

当时 H0/H2 四项均已准备，Fz、Fx、固定材料位移、SA 集合及单值 SA 均为探索。按量按窗的结果、原末态和重算证据均已封装。[接手时的原始证据与源码补丁](https://github.com/Welcome2Machine/cutting_rubber_soul/releases/download/intact-refinement-20261126/intact-R20-57-increment-and-SA.zip) · [当时的分析](https://github.com/Welcome2Machine/cutting_rubber_soul/releases/download/intact-refinement-20261126/intact-R20-57-increment-and-SA.md) · [此前 H1 终态](https://github.com/Welcome2Machine/cutting_rubber_soul/releases/download/intact-refinement-20261126/intact-R20-quarter-mm-terminal.zip) · [独立基准](https://github.com/Welcome2Machine/cutting_rubber_soul/releases/download/intact-refinement-20261126/intact-R20-numerical-credibility.zip)


## 可复用证据与资格边界

| 证据 | 已完成范围 | 尚不能推出 |
|---|---|---|
| B1/B2与B5 | 独立材料、仿射装配、摩擦/地址小例子 | 真实刀具场已收敛 |
| B3 | P1在所测光滑近不可压缩制造解中随网格逼近 | 接触强梯度、深部畸变均已无锁死 |
| B4 | 生产法向接触算子＋解析半空间顺应性；小dhat总力误差改善 | 完整体网格/有限应变/黏弹接触已合格 |
| 历史B6 | 3008态力、力矩与能量账重审 | Fz/Fx、局部变形和SA精度优秀 |
| 原网格完整路径 | 两角度5µm步长各401态、2.5µm各801态至2mm | 全程空间或唯一SA资格 |
| 接手时的 H1 | 33°101态至.25mm，57°78态至.17984375mm | 新严格准备全窗、H0/H2空间资格 |

新旧dhat实算：33°0–.25mm的Fz/Fx归一RMS差0.82%/5.27%，SA集合最大差9.94µm；57°共同深度到.1775mm为2.50%/1.93%和9.73µm，.155mm后含自适应时序混杂。数值资格按当前任务的总预算判断，不能只选对结果有利的大窗口。

此前重布网格在33°0.5mm的q_min=.02516、最小角.875°，q<.1区承担约6.69%ROI储能；局部重网格接边残差未过门。深部网格/SA仍需验证，不能把已接受路径自动升级成机理证据。

原活动树误入的114字节 `BladeA_R20_33_run_summary.csv` 仅含“33°、2.000mm、completed、401接受/401独立审查”摘要，已在本表保留；其历史blob留在原提交，不删除完整原始状态链。

## 历史与可下载结果

- [整理前状态全文](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/ec1f32745af51ba45658cce9c17c97a30ae06ae8/projects/softmodeling/docs/research_status.md)：早期任务、材料候选与来源。
- [当前统一Release](https://github.com/Welcome2Machine/cutting_rubber_soul/releases/tag/intact-refinement-20261126)：原/半步长全程、回放、次表层与重布网格、数值基准、H1终态和最新失败增量/SA包。各包保留当时输入、审查和补丁，不回写历史结果。
- [重布网格分析](https://github.com/Welcome2Machine/cutting_rubber_soul/releases/download/intact-refinement-20261126/intact-R20-distortion-analysis.md)、[数值可信度分析](https://github.com/Welcome2Machine/cutting_rubber_soul/releases/download/intact-refinement-20261126/intact-R20-numerical-credibility.md)、[H1终态分析](https://github.com/Welcome2Machine/cutting_rubber_soul/releases/download/intact-refinement-20261126/intact-R20-quarter-mm-terminal.md)：完整中间数值及其限制。
- 旧3mm/601态、F2/断裂与旧服务器轮次是独立历史，原数据不动；方法见[断裂历史](../../../docs/fracture_implementation.md)。它们不覆盖当前无断裂任务。

本页原位维护当前事实；详细运行日志/逐帧机器数据在Release与服务器ignored输出中，不继续把每轮全文叠加到入口。
