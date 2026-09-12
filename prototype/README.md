# 运行入口

## 固定 IPC 基线

`ipc-frictionless-v1` 已完整计算 0→3 mm。参数集中在 [ipc_baseline.json](ipc_baseline.json)，物理含义和未完成项见 [求解状态](../docs/solver_status.md)。它使用原生 IPC 接触与本地 P1 有限应变宿主，无摩擦、无断裂，不是完整 PolyFEM。

在仓库根目录、Python 3.10 或以上环境执行：

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m prototype.ipc_reference --baseline --output outputs/new_ipc_baseline --detach
```

本地验证环境：Python 3.10.0、NumPy 2.2.6、SciPy 1.15.3、threadpoolctl 3.6.0、Matplotlib 3.10.6、ipctk 1.6.0。requirements.txt 锁定这五项直接依赖，IPC 入口额外检查其版本；这不是所有传递依赖或跨平台二进制的完整锁文件。其他平台先跑短基准，不能直接沿用本机耗时。已有隔离安装 `.runtime/ipc-python` 时，入口在普通导入失败后尝试该目录。

`--baseline` 固定 3 mm 终点、0.025 mm 步长、μ=0、0.01 mm 屏障距离、200 次内迭代、4 线程和 3600 s 预算，并拒绝不同覆盖值。未来变体使用单独输出目录并省略该标志，不能冒充固定基线。固定配置更改应另立版本。

```sh
python -m prototype.ipc_reference --case patch --output outputs/new_patch --depth .1 --seconds 60
python -m prototype.ipc_reference --output outputs/new_diagnostic --depth .4 --increment .025 --friction 0 --seconds 180
python -m prototype.ipc_reference --help
```

不带 `--detach` 为前台模式。输出目录必须为空；完整运行默认后台记录，只检查一次启动。`result.json` 的 `completed/status` 决定完成情况，PID 或正常启动不代表成功。

| 输出 | 内容 |
|---|---|
| `result.json` | 基线 ID（变体为 null）、配置/源码/IPC 二进制指纹、逐步反力、残差、J、间隙和最终状态 |
| `baseline_config.json`、`source/` | 实际使用的固定配置副本与 Python 源码快照 |
| `states/` | 每个接受状态的几何、实际位置、真实前态和深度 |
| `failed_trial.npz` | 求解失败试值，不是新的接受态 |
| `run.log` | 深度、内外迭代、真实自由力残差和接受记录 |
| 目录旁的 `.launch.json`、`.stdout.log`、`.stderr.log` | 后台命令、PID 与标准输出/错误 |

Fz 是刀具下压所需的竖直反力，正方向约定为抵抗下压，单位 N；竖直下压位移单位 mm。体能和接触能均计入 20 mm 宽度。零位移保留几何定义，不扣除屏障有限作用距离带来的预载。所有绘图应保留原始计算点，不通过平滑隐藏前期波动。

已有完整 IPC 输出可生成变形网格与最大主拉应力动画（另需 FFmpeg 在 PATH 中）：

这是常驻后处理功能。斜切算例成功后默认自动执行，使用该次运行保存的源码，进度/错误写入 `visualization.log`，状态保存在 `result.json` 的 `visualization` 字段。后处理失败不会把已收敛计算改标为力学失败；可按下方命令另选空目录重做。短诊断可用 `--no-visualize` 跳过，patch 基准不自动绘图。

```sh
python tools/visualize_ipc.py --source outputs/new_ipc_baseline --output outputs/new_ipc_baseline/visualization --fps 8
```

输出 MP4、关键阶段 PNG 和来源清单。工具核对材料源码与原运行一致，从保存位移重建三维 Cauchy 应力，显示最大主值的正部，包含平面应变出平面应力。P1 单元值不平均、不平滑；位移不放大，所有帧统一线性色标。整体和压入区视窗固定，拉应力近景随刀尖移动且跨度固定；播放帧率不是物理加载速度。图片和视频仅留本地。

网格/刀尖细化变体（局部 0.1 mm、表面 0.05 mm、圆弧 64 段；背景仍 1 mm、半径仍 0.08 mm）：

```sh
python -m prototype.ipc_reference --output outputs/ipc_refined_3mm --depth 3 --increment .025 --local-spacing .1 --surface-spacing .05 --nose-facets 64 --seconds 3600 --threads 4 --detach
```

以上三项离散参数保存到结果配置。`--baseline` 拒绝覆盖；圆弧段数要求为不少于 4 的偶数，以保留最下方刀尖顶点。同轮细化橡胶与刀具只能检验联合离散敏感性，不能据此分离二者贡献或宣称网格收敛。

角度/尺寸对照使用 `--inclination`（工件相对水平面，度）、`--tip-radius`（mm）和 `--fine-halfwidth`（参考表面 s 坐标中细化区半宽，mm），同时显式设置局部与表面间距。刀具始终沿实验室竖直方向移动；0° 是水平工件刀切，不是 `--case patch` 的平板压缩基准。这些参数均被固定基线锁定。用户现已指定 0.02 mm 半径、先只跑 60°；该输入不包含额外的实测来源声明。

本轮 0.02 mm 刀尖的 60° 独立算例（4 线程，6 小时墙钟上限；上限不是预计耗时），从零重算使用：

```sh
python -m prototype.ipc_reference --output outputs/new_r002_60 --hessian-mode regularized --inclination 60 --tip-radius .02 --local-spacing .05 --surface-spacing .005 --fine-halfwidth 3 --nose-facets 64 --dhat .0025 --increment .01 --depth 3 --seconds 21600 --threads 4 --detach
```

`hybrid` 先求未投影切线方向，非下降/线搜索失败时退回 PSD 投影；所有更新仍经 CCD、行列式限步和原能量下降检查。`projected` 仍为默认与固定基线方法。`--resume` 当前只支持无摩擦变体，核对同一物理/网格/步长/源码，独立验收末态，再复制接受前缀与来源记录，失败试值不进入历史。算法切换可能改变平衡分支；续算不是全程使用同一算法的重新计算。`elapsed_s` 仅计本轮续算时间。

`regularized` 模式以 PN 刚度为度量，对全局最小曲率作数值移位；只修改步矩阵，原力门限保持。已修复 0.91 mm 附近的能量相减误拒绝：仅在极小更新、能量差位于舍入尺度、路径功为负且梯形/Simpson 积分一致、残差至少减半并达原门限时允许精度补充分支。该分支记录为 `ROUND_OFF`，不用于接受一般的能量上升。详细对照和限制见求解状态。当前从原 0.90 mm 接受态续算命令：

```sh
python -m prototype.ipc_reference --output outputs/ipc_r002_60_precision --resume outputs/ipc_r002_60_curvature --hessian-mode regularized --inclination 60 --tip-radius .02 --local-spacing .05 --surface-spacing .005 --fine-halfwidth 3 --nose-facets 64 --dhat .0025 --increment .01 --depth 3 --seconds 21600 --threads 4 --detach
```

`CURVATURE lambda_min=... metric_shift=...` 记录迭代点的无量纲曲率估计和步矩阵修正量，负数不代表已发生裂纹。谱估计失败、非下降方向或线搜索失败会尝试原 PN 方向；正定修正不会改变接受所需的原始自由力残差。

## 五组后台队列

```sh
python tools/run_ipc_queue.py --output outputs/ipc_five_cases_20260912 --threads 2 --seconds 21600 --detach
```

固定顺序：30°/μ=0、0°/μ=0、60°/μ=0.35、30°/μ=0.35、0°/μ=0.35。每组从零运行到 3 mm，采用上述 0.02 mm 刀尖/细网格及 regularized 模式；每次只有一组运行，既有 60° 作业不属于队列且不受干预。每组成功、求解失败、异常退出均继续下一组。冻结源码在队列 `source/`，后续工作区修改不影响该批次；仍使用当前 Python 环境及锁定的 IPC 版本。

`queue.json` 保存五组实际命令、子进程 PID、结果摘要和队列状态；`queue.log` 记录开始/结束。`status=finished` 表示队列全部尝试完，不表示全部计算成功。各组保留完整日志、状态与成功后的可视化。每组求解预算 6 小时，进程总上限另加 30 分钟供初始化/导出；超限时仅终止该组进程树并继续。输出目录须为空。

## 旧 P2/IPM 诊断

以下代码用于必要失败证据和回归，已退出主线。完整算例不再默认从旧入口启动。新克隆不包含本地检查点，缺少夹具的测试必须明确 skipped。

| 入口 | 用途 |
|---|---|
| `python -m prototype.audit_contact --help` | 实际增量的力、接触、几何与约束谱审计 |
| `python -m prototype.contact_study --help` | 同历史恢复、局部细化、积分及旧无摩擦对照 |
| `python -m prototype.contact_partitioned --help` | 分块、残差路径及旧库仑内点单增量试验 |
| `python -m prototype.verify_mainline --help` | 旧模型独立短验证 |
| `python -m prototype.run_initiation --help` | 旧全域强度候选筛查；不是自由断裂 |

旧 `contact_study --case normal_baseline` 属于 P2/IPM 历史实验，**不是新的 IPC 固定基线**。单增量诊断可能正常退出但报告 failed/budget_exhausted；始终核对保存状态。残差同伦的辅助参数不是实验加载量，辅助平衡点不是接受的物理历史。

## FEBio 与论文复现

FEBio 仅作独立参考。`tools/build_febio.py --contact-audit` 构建修正版；使用 `python -m prototype.febio_reference --help` 查看导出、执行及审计选项。其 Hex8 三场离散不同于 IPC 的 P1；不能混用检查点或将短前缀通过称为完整切割验收。Windows 需 Visual Studio C++/CMake，Linux 目标节点仍需实编译验证。

[Dammaß 2023 复现](../reproductions/dammass2023/README.md) 保留独立入口、材料和验收；[超算说明](../docs/hpc.md) 仍需按实际平台适配。只发布代码、文档和运行配置，数据、图像、PDF、夹具及输出均留本地。

含摩擦采用耦合伪瞬态延续，`FRICTION_PTC` 记录原始力残差和辅助矩阵移位，`FRICTION_PTC_UPDATE` 记录步幅及局部模型误差。中间残差可以暂时上升；只有 `ACCEPT` 表示已满足刷新后的 1e-4 N 力门限、几何及摩擦耗散检查。辅助伪时间不代表实验时间，移位不属于材料黏性。无摩擦继续使用原来的 `ITER` / 能量 Newton。

续算 60°，然后运行 30°→0°（μ=0.35），取消两层墙钟时间限制：

```powershell
python tools/run_ipc_queue.py --output outputs/ipc_friction_unlimited_20260912 --friction-only --resume-first outputs/ipc_friction_ptc_20260912/01_angle60_mu0.35 --threads 2 --seconds 0 --detach
```

60° 校验并复制已接受历史后继续到 3 mm；后两组从零加载。省略 `--resume-first` 可使全部工况从零运行。`--seconds 0` 同时禁用求解器及队列进程超时；数值迭代上限、原始力验收和几何检查仍保留，成功或真实数值失败后继续下一组。含摩擦续算使用末个检查点的 `previous_positions` 验证真实滑移力，再开始新加载步。`queue.json` 记录状态，`queue.log` 记录调度；冻结源代码，工作区修改不影响已排定运行。
