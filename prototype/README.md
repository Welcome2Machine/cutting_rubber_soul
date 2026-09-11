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
