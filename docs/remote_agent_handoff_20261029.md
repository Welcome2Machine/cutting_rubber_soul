# 远程 1029 工作区接管说明

交接日期：2026-10-29。本文用于从 Windows 整合工程迁移到远程服务器，
是接管入口，不替代各项目的 AGENTS.md、数值规格与当前研究状态。
尚未登录或验证远程服务器；以下 Linux 命令需由接管 agent 在目标机执行。

## 1. 正确仓库与位置

- 仓库：<https://github.com/Welcome2Machine/cutting_rubber_soul>
- 整合发布分支：`codex/integrated-solution-blade-profile`。
- **远端 `main` 是旧独立 FE 工程，不能作为本次接管起点。**
- FE 最新算法提交为 `b530832`；显微参数及发布入口为 `a57accd`；
  用户确认侧视方向和反复使用状态为 `037dda0`。本文位于其后续文档提交。
- 目标工作区为用户指定远程父目录下的 `1029`，仓库根直接放在 `1029` 中。
  若目录已有内容，先检查，不删除、不重置、不在其中再无意嵌套一个同名仓库。

在目标父目录中执行（下例仅用于 `1029` 尚不存在的情形）：

```bash
git clone --branch codex/integrated-solution-blade-profile --single-branch https://github.com/Welcome2Machine/cutting_rubber_soul.git 1029
cd 1029
git rev-parse HEAD
git status --short
git switch -c codex/server-fracture-20261029
git config core.hooksPath .githooks
```

若目标开发分支已存在，检查其提交与未提交内容后继续，不强制重建。
所有新开发在 `1029` 内进行。先核对已有 AGENTS.md 及服务器资源/调度规则。
旧 Windows 路径是历史来源，不能批量替换历史报告，也不能在 Linux 上作为输入路径。

## 2. 必须先读的上下文

按顺序阅读：

1. 根目录 `AGENTS.md`、`README.md`、`docs/interface_contract.md`、`docs/git_workflow.md`。
2. `projects/softmodeling/AGENTS.md`，以及其 `docs/research_status.md` 开头的最新章节、
   `docs/task_requirements.md`、`docs/current_model_spec.md`、`prototype/README.md`。
3. 根目录 `docs/fracture_implementation.md` 第 18–22 节，尤其第 22 节的 CC/CZM 假设与未通过项。
4. `docs/blade_profile_config.md`、`configs/blade_profiles.measured.json`。
5. 涉及视频时再读 `projects/rec_inc_soft_cut/README.md`、`environment/README.md`、
   `environment/AGENT_CONTRACT.md`，以及共同观测接口。

整合代码属于 `softcut_solution/`；FE 数值代码在 `projects/softmodeling/`；
视频在 `projects/rec_inc_soft_cut/`。旧 `0915ForceProcessing` 不属于本工程。

## 3. 已完成与未完成必须分开

### 完整体基线

原始完整体模型为有限应变等容 Maxwell、原生 IPC 接触和 CCD；保留原物性与历史。
原材料三角度 0°/33°/57° 的 2 mm 完整体路径已在本地完成并审查，数据不在 Git。
它们没有断裂，不能当作裂后模型已完成。旧 q50 路径的审查问题是另一条历史研究线，
不得与原材料基线混淆或自动恢复其旧队列。

### 当前断裂研究

已实现强度—能量双条件有限裂增量门控，接入实际刀具与动态 CZM：
同一候选裂段同时满足参考牵引强度和冻结历史储能差减虚拟摩擦功的能量条件；
实际开口只支付一次 Gc。虚拟场不覆盖真实材料、摩擦或断裂历史。
已接入零间隙未断连接、单边裂面接触、两个方向的真实惯性、离散梯度积分、
Lobatto 接触求积、独立增量 PTC 和求解用摩擦延续。

**这些实际算例仍是 30×5×20 mm、刀具居中 s=15 mm、0°镜像对称、预设直裂路控制。**
不是原 s=7 mm 偏心基线，不支持已经合格的 33°/57°裂后预测或自由分叉。
2 mm 是刀具行程，不是切穿 5 mm 试样。

最终同源码结果（本地 `runs/finite_increment_20261029/cont_*`）：

| 算例 | 末行程 mm | 接受态数 | 前缀 Σ\|能量缺口\|/Σ\|外功\| | 原始 Fz 最大回落 N |
|---|---:|---:|---:|---:|
| `cont_base` | 1.195938 | 255 | 0.0028274% | 17.1422 |
| `cont_time2` | 1.198105 | 509 | 0.0006817% | 0.3068 |
| `cont_time4` | 1.199140 | 993 | 0.0004695% | 3.3819 |
| `cont_mesh075` | 0.936250 | 195 | 0.0005627% | 0.8805 |
| `cont_mesh05` | 1.081136 | 269 | 0.0054707% | 7.7207 |
| `cont_intact`，关闭损伤 | 2.000000 | 405 | 0.0001445% | 0 |

五条含断裂路径均未完成 2 mm，表面连通完全失效长度均为 0，有内部孤立开口。
小的前缀能量缺口或曲线 L2 差不能掩盖尖峰、未覆盖的快速事件或失败后缀。
**裂后 2 mm、大掉力修复、时间/网格收敛、自由裂路和实验预测资格均未通过。**

已定位的关键问题：

- 特定候选裂段的虚拟开裂后平衡求解失败；基准及 .075 网格的真实残差已小于
  1.1e-9 N，不能一概归因于真实动态方程未收敛。未求完候选是未知，不能跳过后宣称未裂。
- 基准末段大掉力没有新释放事件，主要关联快速滑移时摩擦 Fz 反号。
  尚不能用“改起裂阈值”解释或解决所有掉力。
- 节点面积与实际 P1 裂面开口支撑、内部空腔和表面贯通的竞争仍需验证。
- 曾尝试的分段虚拟闭合载荷路径未通过功账与细化验证，未进入生产；不要直接照搬成定论。

当前数值示例采用 T=0.2 MPa、Gc=0.006 N/mm、ρ=1000 kg/m³，**不是独立实测参数**。
共同材料 G∞=.14 MPa、Gv=1.24 MPa、τ=1.7 s、K=23.34333333333331 MPa、μ=.35。
速度固定 .2 mm/s、最大行程 2 mm。独立材料、强度、断裂能与摩擦试验方案在
`projects/softmodeling/configs/independent_fracture_test_plan.json`，目前是计划，没有实测资格。

## 4. 用户最新补充：刀具和实验约束

- 显微图就是仿真的侧视截面，用户已明确确认；不要再次询问方向。
- A 高、低性能刀具是同型号；两组照片都是反复使用后的状态。
- 同一把刀反复使用的性能变化不明显，刀具之间差异巨大。优先考虑稳定的刀间特征；
  该观察降低“测试中持续磨损主导”的优先级，但不能证明差异一定来自出厂状态。
- A 两组截面宏观楔角均约 13.3°，厚度约 .246/.241 mm；图像约 10 μm/像素，
  厚度差约半个像素，不能据此认定已解释性能差距。高性能图尖端附近有局部凸起，
  其真实来源尚未确认。刃口半径、末端微刃和表面性质仍未分辨。
- 13.3° 与旧模型默认 23° 的差异需要作为几何依据核查；不要混淆宏观主刃与末端微刃。
  保留旧配置作为数值对照，新测量几何单独建候选，不借力曲线选择半径或楔角。
- `softcut_solution.blade_config` 已能把轮廓配置转换为工作台 `softcut-config/1`，
  仍需明确提供独立半径及其来源、接受对称近似；**尚未把刀形参数接入
  `softcut-zero-gap-dynamics/1` 的求解配置**。不要把两种 schema 混用。
- 多次 57°测试在 2 mm 仍未开裂，这是工况级右删失验证约束；不是起裂点恰好 2 mm。
- 用户观察起裂后 Fz 继续增长但增幅减缓，当前大掉力需解释；它是验证信息，
  不是强制单调 Fz 的算法约束。
- 禁止用实验切割力、57°未裂约束或数值能量误差反推强度、Gc、材料、摩擦、
  鼻部半径、有效厚度或角度专用参数。实验只能用于独立参考验证。

## 5. 远程环境与只读验收

先记录 OS、CPU、内存、可用磁盘、Python、调度器及已有进程。FE 是 CPU 路线，
不要因服务器有 GPU 就自动启动视频推理。Windows `.venv`、`.runtime` 二进制不能
直接作为 Linux 运行库。以下为 Linux 新环境示例；已有兼容环境可复用并记录路径。

在 `1029` 根目录：

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -r projects/softmodeling/requirements.txt
export MPLBACKEND=Agg
python -c "import sys,ipctk; print(sys.executable); print(ipctk.__version__)"
python -m pytest -q
python tools/check_publish.py --ref HEAD
cd projects/softmodeling
python -m pytest -q tests
cd ../..
```

IPC 必须为 1.6.0，不能遇到安装困难就静默升级或换接触模型。
若平台缺少对应 wheel，按该版本官方构建说明处理并记录编译环境。
测试含 Tk 导入：所选 Python 需有 Tk；无显示时 GUI 测试可能按既有规则跳过，
如需完整 GUI 检查使用虚拟显示。导入失败不能记为测试通过，跳过数量要报告。
本地最新整合检查为 128 项通过；FE 最近算法版本为 612 项、537 子检查通过。
这些是 Windows 本地记录，不是远程已经验证的结果。

视频保持独立 Python 3.11 环境及其 PyTorch/FFmpeg 依赖；只有需要视频任务时再准备。
工作台 `settings.local.json` 使用服务器实际解释器路径，不复制 Windows 路径。
无图形服务器优先命令行；页面启动不触发 FE、GPU 跟踪、旧队列或自动续算。

## 6. 数据边界与可选传输

Git 提供代码、配置、测试、方法文档，**不包含**原始视频、权重、显微 PPT、
实验 CSV、完整体基线或上述算例 NPZ。克隆后链接缺失是数据未传输，不是可以忽略的验证步骤。
不需要搬运约 52 GB 全部数据才能开始 FE：从当前配置初态重算不依赖实验力和视频。

若需要重放本地失败态，优先从原工作区 `D:\Doctoral Research\Softcutting\1026solution`
按相同相对结构另行传输：

| 相对路径 | 用途 |
|---|---|
| `runs/finite_increment_20261029/cont_base/`、`cont_time2/`、`cont_time4/`、`cont_mesh075/`、`cont_mesh05/`、`cont_intact/`（后五项同父目录） | 完整状态、虚拟裂纹证据、失败态和结果；不能只传最后一个 NPZ |
| 上述父目录的 `cont_*.config.json`、`cont_*.path_audit.json`、`cont_*.finite_audit.json`、`cont_*.finite_audit_strict.json`（存在者） | 输入与独立审查；保留原字节 |
| `runs/finite_increment_20261029/index.html`、`delivery.json`、`delivery_provenance.json`、`assessment/`、`final_validation/` | 本地综合报告、失败边界和统计 |
| `researchtopic/bladetype.pptx` | 重做显微提取时需要；仅使用已提交刀形配置时不需要 |
| `projects/softmodeling/outputs/baseline/angle0/`、`angle33/`、`angle57/` | 旧完整体曲线/场对照，需要时传输完整目录 |
| `projects/softmodeling/docs/1008n0a2__Fz__curve_1.csv`、`1008n33a1__Fz__curve_2.csv`、`1014n57a1__Fz__curve_0.csv`（均在该 docs 下） | 实验参考验证，不用于推参 |

传输前后核对 SHA256。PPT 哈希已在显微清单中；保存态及虚拟证据哈希在各 result.json 中。
从 Windows 转移时保留数据字节，不做全目录换行转换；算法源码哈希以 Git/现有审查为准。
缺少数据就标记“尚未传输”，可以从初态复现；不能声称已重放本地失败现场。
源码、配置、几何或物性不同不能移植旧内部变量。续算工具会检查同源码及父路径审查，
不能删改其门槛来迁就跨平台差异。

## 7. 首轮执行和已知控制矩阵

首轮先复现原模型，不立即把刀形、裂纹律、求解器和摩擦同时改掉。
标准配置 `projects/softmodeling/configs/zero_gap_finite_increment.json` 与本地
`cont_base.config.json` 数值内容一致。所有输出目录必须是新目录。

以下命令的工作目录是 `1029/projects/softmodeling`，使用上一步激活的 FE Python。
只有接管任务明确授权运行后执行；各 `--seconds` 是独立墙钟预算：

```bash
python -m tools.run_zero_gap_cut --config configs/zero_gap_finite_increment.json --out ../../runs/server_1029_round1/base --seconds 3600
python -m tools.audit_zero_gap_cut ../../runs/server_1029_round1/base --output ../../runs/server_1029_round1/base.path_audit.json --seconds 600
python -m tools.audit_finite_increment ../../runs/server_1029_round1/base --output ../../runs/server_1029_round1/base.finite_audit.json --seconds 600
python -m tools.report_finite_increment ../../runs/server_1029_round1/base --output ../../runs/server_1029_round1/report
```

先读每步结果和审查字段；审查失败或耗尽预算不代表合格。报告工具要求合格前缀，
拒绝绘制时先解释原因，不修改标志绕过。保存 stdout/stderr 日志。
同源码/配置的已审查父路径才可显式使用 `tools.continue_zero_gap_cut`，每次新目录和新预算；
算法改动后应从自身初态验证。不要把增加预算当作已解决已知虚拟平衡失败。

从基准 JSON 复制新研究配置，下列为已有控制的**精确差异**：

| 控制 | 相对基准修改 |
|---|---|
| time2 | `increment_mm=.0025`，`energy_tolerance=.000125`，`energy_absolute_tolerance_n_mm=5e-8` |
| time4 | `.00125`，`.0000625`，`2.5e-8`（与上一行同顺序） |
| mesh075 | `seam_spacing_mm=.075`，`surface_spacing_mm=.075`，`local_spacing_mm=.5625`，`coarse_spacing_mm=1.125` |
| mesh05 | `.05`，`.05`，`.375`，`.75`（与上一行同顺序） |
| intact control | 仅 `damage_enabled=false`；其余保持基准 |

这里时间控制同时收紧步长和误差门，应称联合时间控制细化；后续如需分离两者影响，
分别设计对照。零间隙模型的 `seam_spacing_mm` 是沿裂路离散间距，**不是初始裂缝间隙**。
旧有限间隙网络的 gap 敏感性是另一项研究，不能把两者互相代替。

## 8. 后续研究顺序与交付要求

1. **复现与定位。** 记录服务器环境和回归结果；从初态或合格失败现场核查虚拟
   开裂后平衡、动态摩擦反号、接触切换及源增量精度，区分数值错误与模型机制。
2. **修复可审查的虚拟裂纹过程。** 可以评估带完整能量账的开口/滑移延续或虚拟瞬态；
   不必固守现实现，但新方法须解释外载功、摩擦功、历史冻结和能量可用性。
   虚拟惯性或人工载荷不能无说明进入真实裂纹驱动力，强度/Gc不能重复支付。
3. **实际切割完整路径。** 在同一未测示例参数下完成含断裂 2 mm，验证裂口是否从
   表面连通、刀是否进入裂口，分别报告内部空腔和真正切开；只关闭损伤跑完不合格。
4. **验证。** 重跑完整失稳窗口的时间控制、网格、接触/摩擦正则化敏感性，报告
   Fz 原始峰值/回落、全局与事件窗口能量、动能、体耗散、断裂功、摩擦耗散、
   接触势和数值缺口。先看完整事件与最大点差，再看曲线整体误差。
5. **几何与裂路扩展。** 测量刀形接入动态 schema 应独立实现并验证几何/接触，
   鼻部半径必须保持未测或来自独立证据。再扩展偏心/斜切与自由裂路、网格方向、
   界面附加柔度及旧有限间隙网络的 gap 检查。当前 0°对称算例不能认证这些能力。
6. **物理资格。** 独立强度/Gc、率效应和接触试验方案继续保留；刀具性能差异
   优先研究微刃几何/稳定表面状态，并区分刀内重复性和刀间差异。没有新证据时不宣称
   高低性能差异已解释或刀具批次已标定。

保持真实物性、质量、速度和历程；力残差门槛 1e-4 N、正 J、完整 CCD、边界、
摩擦/材料/断裂历史一致性不放松。不能截负力、强制单调、事后平滑或对齐事件来通过验证。
不能用任意阻尼、质量放大、关闭摩擦或关闭损伤冒充最终修复。
可做这些机制对照，但要显式标注不同模型，不能替代原目标。
运行中不得改动被求解进程使用的数值源码；保存配置、commit、解释器、依赖版本、
源码/输入哈希、线程/预算和完整失败记录。不要创建冻结源码副本树。

任务范围包含实现、回归、实际有界算例和报告，不只写建议或只跑小算例。
每轮更新既有研究状态/规格/方法文档，生成本地 HTML+JSON 证据，Git 只提交代码、
审阅配置和文字文档。提交前运行根测试、相关 FE 回归与 `tools/check_publish.py --staged`。
推送新的 `codex/server-*` 分支，不强推、不直接改远端 main 或本次整合发布分支。
报告必须明确“做到了什么、停在哪里、为何失败、哪些资格仍为 false”，无需为得出漂亮曲线隐瞒问题。

建议第一轮接管预算：同时最多 1 个 FE 加载作业，单次加载不超过 3600 s，单次独立
审查不超过 600 s，首轮总计算墙钟不超过 4 h；当前入口为 IPC=2、BLAS=2 线程。
以用户转交的任务文本为实际授权；预算到期保存现场并交付，不自动延长或启动旧队列。
有调度器时服从已分配资源，不争用其他任务的核和内存。
