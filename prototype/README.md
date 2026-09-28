# 运行入口

更新：2026-09-28。服务器 CZM 开发先读 [交接文档](../docs/server_handoff.md)：计划预设初始完整黏结的潜在路径，CZM 切割入口尚未实现。下列命令属于已有完整材料/相场/接触研究组件，本次发布不执行。完整材料 IPC 保留为基线，起裂后的完整材料结果是假想无断裂延续。生效要求、方法与实际完成证据分别见 [任务要求](../docs/task_requirements.md)、[断裂路线](../docs/fracture_route.md)、[求解状态](../docs/solver_status.md)。

现存数据统一见 [已完成结果](../docs/completed_results.md)。下面的示例不会自动启动；使用新的空输出目录。历史未完成结果已清理，不再提供依赖已删检查点的推荐命令。新的短程/组件运行不能替代原 3 mm 完整目标。

## 1. 完整材料 IPC 主线

在仓库根目录、Python 3.10 或以上执行：

~~~sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m prototype.ipc_reference --help
~~~

本地依赖为 NumPy 2.2.6、SciPy 1.15.3、threadpoolctl 3.6.0、Matplotlib 3.10.6、ipctk 1.6.0，见 requirements.txt。IPC 入口检查版本；这不是跨平台二进制完整锁文件。已有 `.runtime/ipc-python` 时，普通导入失败后会尝试该安装目录。缺少本地夹具的测试必须 skipped；新环境先跑有界验证，不直接沿用本机耗时。

固定基线 `ipc-frictionless-v1` 已完成 0→3 mm：

~~~sh
python -m prototype.ipc_reference --baseline --output outputs/new_ipc_baseline --detach
~~~

配置集中于 [ipc_baseline.json](ipc_baseline.json)：μ=0、刀尖半径 0.08 mm、表面间距 0.1 mm、增量 0.025 mm、屏障距离 0.01 mm、内迭代上限 200、4 线程及 3600 s 预算。`--baseline` 拒绝不同覆盖值，不是当前真实尺寸模型，也不是完整 PolyFEM。

当前 0.02 mm 刀尖、60° 完整材料的无摩擦数值变体：

~~~sh
python -m prototype.ipc_reference --output outputs/new_r002_60 --hessian-mode regularized --inclination 60 --friction 0 --tip-radius .02 --local-spacing .05 --surface-spacing .005 --fine-halfwidth 3 --nose-facets 64 --dhat .0025 --increment .01 --depth 3 --seconds 21600 --threads 2 --detach
~~~

该命令是运行方法，不保证当前版本全程稳定。`--inclination` 是工件相对水平面角度，刀具始终沿实验室竖直方向运动；0° 也不是 `--case patch` 平板压缩。材料与 20 mm 宽度不因角度改变。摩擦、网格及算法更改必须在独立目录说明，不冒充固定基线。

有界力学检查可用：

~~~sh
python -m prototype.ipc_reference --case patch --output outputs/new_patch --depth .1 --seconds 60
~~~

`projected` 为固定基线策略；`hybrid` 尝试原始切线后按需回退；`regularized` 用 PN 度量修正步矩阵曲率。修正不改变实际材料、自由力门限或几何验收。`ROUND_OFF` 是极小步的受控能量舍入补充，不接受一般能量上升。含摩擦采用耦合伪瞬态延续；`FRICTION_PTC` 的伪时间和移位不是实验时间或材料黏性。只有 `ACCEPT` 才是新接受态。

`--resume` 只接受配置与源码匹配的完整材料历史，先核验末态及摩擦真实前态；在线起裂模式当前拒绝续算。续算改变算法不能证明从零使用该算法可完成，`elapsed_s` 只计续算段。不要把 element 检查点当作 mixed 平衡历史。

## 2. 接受历史导出、可视化与起裂诊断

下面读取本地完整 60°、μ=0.35 历史，不重跑力学：

~~~powershell
python tools/export_cutting_history.py --source outputs/ipc_friction_unlimited_20260912/01_angle60_mu0.35 --output outputs/study60/history --recover-fx
python -m prototype.ipc_initiation --source outputs/ipc_friction_unlimited_20260912/01_angle60_mu0.35 --output outputs/study60/onset --directional-onset --stop-at-onset --radii --threads 2
python tools/visualize_ipc.py --observables --source outputs/study60/history --onset-result outputs/study60/onset/result.json --output outputs/study60/figures
~~~

导出只保存力、原顶面材料点运动及来源，全部体节点仍从原接受态读取。原生 Fx 缺失时保持空值；`--recover-fx` 另存恢复值，须通过本构与 Fz 平衡核验。新求解直接保存 Fx/Fz，实验室 +x 向右、+z 向上，力是橡胶对刀具的反力。

`--directional-onset` 是方向松弛 AT1 的 d=0 单边稳定性筛查，不推进损伤。报告首次候选前后深度、参考/当前坐标、刀尖相对位置及方向；`--orientation-samples 32` 可作方向敏感性检查。`--radii` 不跟数字时跳过旧空间平滑统计。方向校准向多轴应力迁移仍是假设；ell 不是这一完整态一阶判据的已验证非局部平均半径。

筛查仅接受核验过的 element 无损伤历史，不把旧损伤或 mixed 场套入局部本构。`--stop-at-onset` 只停止分析；没有超限也不能证明不会断裂。时间若按未确认的 0.2 mm/s 映射，不能称实验起裂秒数。`--strength-surface PATH` 只读取显式带来源的强度标定，不自动补造多轴强度。

网格与最大主拉应力可视化是常驻功能：

~~~sh
python tools/visualize_ipc.py --source outputs/ipc_friction_unlimited_20260912/01_angle60_mu0.35 --output outputs/study60/mesh --fps 8
~~~

斜切完整材料运行默认结束后生成图像，MP4 另需 FFmpeg。逐单元三维 Cauchy 应力包含平面应变的出平面分量；位移不放大、不平滑，统一色标。起裂候选后的力曲线明确标为无断裂延续。短诊断可用 `--no-visualize`。

失败接受前缀需显式 `--allow-failed-prefix`（导出器为 `--allow-incomplete-prefix`）；活动过程和失败试值不能当接受历史。后处理失败不改写力学完成状态。公开仓库不提供这些数值输入，缺失时不能宣称已执行。

## 3. 起裂与增长的研究入口

下一关口是从首次候选前接受态验证短程位移—不可逆损伤耦合，全域无种子，并分开记录起损和局部化。`prototype.ipc_cutting` 的检查点窗口入口复用完整材料前缀、原摩擦前态和同一网格/物理参数；检查点只省去重算完整前段，不把候选点强制成裂纹。前缀会重新筛查，若选定锚点已经起损，应拒绝并选择更早的接受态，而不能把已发生的损伤重置为零。参数和当前可用选项见：

~~~sh
python -m prototype.ipc_cutting --help
~~~

本机已完成的有界窗口可用以下命令重建（输出目录须不存在）：

~~~powershell
python -m prototype.ipc_cutting --source outputs/ipc_friction_unlimited_20260912/01_angle60_mu0.35 --start-depth 2.06 --depth 2.12 --increment .005 --min-increment .000625 --seconds 300 --threads 2 --max-iterations 500 --predictor affine --friction-linear-solver direct --refine-orientations --output outputs/new_nucleation_window60
~~~

源模式继承原物理/网格，前缀若已起损会拒绝导入；`--refine-orientations` 仅适用于现有本构的连续方向候选。`--orientation-lock-damage` 默认保留旧数值历史阈值 1e-8，不代表物理裂纹阈值。结果标记 `completed_window`、`completed_full_case=false`，累计功仅属窗口；本次完成 12 个增量但未形成有限断裂带，详见求解状态。未安装 IPC 或缺少本地完整源数据时，不能声称复现此窗口。长运行可另加 `--detach` 后仅检查启动，记录在父目录 job.json。

当前 QR＋RationalAT1 是文献构件的本地组合，不等同于某一论文的完整方法；刀具—新裂面接触尚未接通。以下模块保留供分问题研究，不是已验收的裂后生产入口。

| 模块 | 已实现范围与限制 |
|---|---|
| `ipc_cutting`、`directional_fracture` | 方向松弛与有理 AT1 的连续耦合试验；高损伤需新裂面时停止为 `model_boundary`，不能当完整切割成功 |
| `ipc_fracture`、`phase_field` | 无损伤检查点上的总能量 AT1 双向耦合对照，含压缩驱动；不是从零连续损伤历史，不能据不同检查点推断起裂时刻 |
| `strength_phase_field` | Kamarei 2025 纯弹性、常数第二 Piola 强度的固定变形核；尚未耦合刀切，当前目录强度不兼容此常数强度特例 |
| `closed_crack`、`crack_contact_geometry` | 预设已脱粘界面的压闭、滑移和刀具接触；没有自由成核和新界面出生 |
| `material.NodalVolume`、`macro_element` | IPC `macro_mixed` 体离散对照；未完成与当前损伤算法的一致耦合 |

已有组件的有界验证：

~~~powershell
python -m unittest tests.test_directional_fracture tests.test_phase_field tests.test_strength_phase_field tests.test_ipc_mixed_volume -v
python -m prototype.closed_crack --output outputs/closed_crack_trial --threads 2 --seconds 180
python -m prototype.ipc_reference --output outputs/mixed60_short --case oblique --inclination 60 --friction .35 --tip-radius .02 --local-spacing .1 --surface-spacing .02 --fine-halfwidth 3 --nose-facets 64 --dhat .0025 --depth .02 --increment .01 --volumetric-discretization macro_mixed --hessian-mode regularized --max-iterations 500 --threads 2 --seconds 90 --no-visualize
~~~

闭合裂面组件由两块 1×1 mm 橡胶及预设重合界面组成，使用原 G/ν、20 mm 宽度、0.02 mm 圆头楔形刀具，裂面 μ=0、刀具 μ=0.35。默认 50 步阶段全部完成仍只证明该组件关口；节点—线段接触不是 mortar，没有网格无偏性保证。新目录自动可视化，单独重做用 `python tools/visualize_closed_crack.py CASE`，拒绝覆盖既有图像。

`StrengthPhaseField` 必须显式提供带来源及 measured/assumed/synthetic 标签的单轴与静水强度、同一本构临界能量。当前 4.4 MPa 名义强度对应状态已在常数 S2 峰后，接口拒绝该标定；不能通过调 ell 隐藏提前触发。固定变形求解势不是完整物理储能，不能套用旧总能量验收。

`ipc_cutting` 的 secant 预测和 retry-gmres 属于数值研究策略；短性能提升不保证整条加载路径稳健。可用 `--predictor affine --friction-linear-solver direct` 做同历史对照。`WORK` / `work_totals` 统计 PTC、分解、GMRES 与耗时，旧 `iterations` 字段不是全部成本；时间项有嵌套不能相加。`ENERGY` 将端点虚功与能量余量分列，`energy_balance_resolved=false` 不允许宣称能量精度通过。

## 4. 完整材料队列与记录约定

队列保留供同条件比较；本次整理不自动重启历史队列。三工况润滑比较命令为：

~~~powershell
python tools/run_ipc_queue.py --output outputs/ipc_lubrication_new --study lubrication --oil-friction .10 --temperature-c 25 --threads 2 --seconds 0 --detach
~~~

顺序是无润滑 60°（μ=0.35）、硅油等效 60°（μ=0.10）、硅油等效 30°（μ=0.10），每组从零尝试到 3 mm，成功或失败均继续下一组。它显式启用在线起裂诊断、secant 与 retry-gmres；现有整段回退问题没有因保留命令而宣称解决。

其他队列只保留一个入口：默认五组为 30°/μ0、0°/μ0、60°/μ0.35、30°/μ0.35、0°/μ0.35；`--friction-only` 为 60°→30°→0°、μ0.35。参数细节见 `python tools/run_ipc_queue.py --help`。`--seconds 0` 禁用墙钟上限，数值迭代和验收仍有效。

涂油是未标定的等效低摩擦情景，不包含油膜压力、黏性剪切或溶胀。`--temperature-c`、油牌号/黏度/等待时间均为元数据，不自动改变本构。15–35°C 的影响不能由现有数据量化，见 [文献依据](../references/lubrication_temperature.md)。

完整运行必须后台执行、只检查一次启动，保留配置、冻结源码/指纹、PID、stdout/stderr、日志和逐步接受态。输出目录不可覆盖：

| 文件或状态 | 含义 |
|---|---|
| `result.json`、`states/`、`run.log` | 配置/来源、逐步反力与平衡/几何验收、接受历史及最终状态 |
| `source/`、配置快照 | 实际使用的代码与参数；工作区后续编辑不影响冻结代码 |
| `failed_trial.npz` | 失败试值，不属于接受历史 |
| 启动记录、stdout/stderr | 后台命令、PID 与异常信息；启动不等于成功 |
| `queue.json`、`queue.log` | 串行调度与各组结果；`finished` 只表示全部已尝试 |
| `initiation/result.json` | 主事件、诊断状态及参数；与力学完成独立 |
| `visualization` 状态/日志 | 后处理完成情况；与力学完成独立 |

Fz 单位 N，深度 mm，体能与接触能均计入 20 mm 宽度；接触零点保持几何定义，不人为扣除屏障预载。模型完成、数值完成、后处理完成和实验验证必须分别报告。

## 5. 历史诊断与独立论文基准

以下入口已退出刀切主线，但仍被回归或工具依赖，因此保留代码，不重复列长算例命令：

| 入口 | 用途 |
|---|---|
| `prototype.audit_contact` | 旧 P2 增量的接触、几何与约束谱审计 |
| `prototype.contact_study` | 旧历史恢复、细化、积分与无摩擦对照；`normal_baseline` 不是 IPC 固定基线 |
| `prototype.contact_partitioned` | 旧分块、残差路径与库仑内点单增量研究 |
| `prototype.verify_mainline` | 旧模型独立短验证，名称不表示当前主线 |
| `prototype.run_initiation`、`start_initiation` | 旧 P2 全域强度筛查与后台启动；不是自由断裂 |
| `prototype.febio_reference` | 独立 Hex8 三场参考，检查点与 IPC 不兼容 |
| `tools/build_febio.py`、`febio_contact_patch.py` | 本机参考求解器构建/补丁，Windows 需 C++/CMake |

旧单增量工具正常退出仍可能报告 `failed/budget_exhausted`；残差同伦参数不是实验加载量。缺失失败夹具不可用后来的结果替代，测试必须 skipped。

[Dammaß 2023](../reproductions/dammass2023/README.md) 保留独立材料、入口与验收，原参考数据仅留本地；[超算说明](../docs/hpc.md) 需按目标平台适配。公开仓库只包含代码和文档，不包含 PDF、媒体、夹具或计算/实验数据。
