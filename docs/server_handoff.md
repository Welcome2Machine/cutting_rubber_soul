# CZM 分支：服务器 agent 交接

更新：2026-09-28。分支：`feature/czm-cutting`；远程：<https://github.com/Welcome2Machine/cutting_rubber_soul>。

**这次发布的是现有代码及 CZM 实施方案的起点，不是已经完成的 CZM 切割程序。用户本轮要求建分支、推送和交接，明确不启动计算。本文的安装、测试和算例命令供服务器接手后使用，本轮没有执行。** 不因分支名称包含 `czm`，就宣称起裂、真实分离及裂后刀具接触已经接通。

## 1. 接手顺序与当前能力

先读 [任务要求](task_requirements.md)、[断裂路线](fracture_route.md) 的 2026-09-28 部分、[论文方法笔记 R04](../references/method_notes.md)、[求解状态](solver_status.md)，再读 [运行入口](../prototype/README.md)。参数来源以 [current_model_spec.md](current_model_spec.md) 的实验、材料和界面章节为准；图片原件只留本地，公开仓库缺图不是可以另造实验条件的理由。

| 现有入口 | 实际作用 | 不能据此宣称的能力 |
|---|---|---|
| `prototype.ipc_reference` | 有限应变平面应变、完整橡胶、刀具接触；可附加不反馈力学的起裂筛查 | 不断开材料；候选后的反力是无断裂延续 |
| `prototype.ipc_initiation` | 审核完整接受历史并筛查起裂候选 | 不是裂纹增长求解器 |
| `prototype.ipc_cutting` | 连续位移上的方向松弛与不可逆损伤研究；可从核验过的历史导入短窗口 | 没有独立新裂面；`completed_window` 不是完整 3 mm 切割 |
| `prototype.closed_crack`、`crack_contact_geometry` | 两块初始已脱黏材料的压闭、滑移及刀具接触组件，有向间隙导数 | 没有初始完整黏结、界面起裂和新裂面出生 |
| `prototype.ipc_fracture`、相场核、`reproductions/dammass2023` | 独立研究、对照和论文复现 | 不等于当前斜切 CZM 已实现 |
| 未来 CZM | 需新增完整黏聚界面、不可逆软化、独立裂面位移及刀具/裂面接触 | **当前不存在可执行的 CZM 完整算例命令** |

本地曾保留 6 组完整材料 0–3 mm 结果；它们不是所有参数组合均可成功的保证。最近 60°、μ=0.35 的 2.06→2.12 mm 研究窗口完成 12 个增量，最大损伤约 2.57e-5，没有形成明显断裂带；局部最小 J 约 0.064，严重压缩问题仍需独立检查。数值证据及限制见 [求解状态](solver_status.md)，不要把这些历史结果写成服务器本轮复现结果。

## 2. CZM 要实现什么

采用 *Cutting soft materials: how material differences shape the response* 的“有限应变体材料＋预设潜在黏聚面＋原/新表面刀具接触”架构，独立核查本构、参数和验收。作者版及正式版链接和推导见 [R04](../references/method_notes.md)。现阶段不复制整套论文软件，也不直接迁移其拟合参数。

允许预设的是**初始完整黏结、随材料变形的潜在路径**。结果应称“给定路径条件下的起裂与增长”。不能预先放置开放裂缝，不能让刀尖经过即强制断开，也不能不断将材料路径拖到当前刀尖下方。

实施关口按顺序推进：

1. **界面核。** 实现有完整弹性段、拉压分离及剪切作用的不可逆 CZM；分别声明界面强度、初始刚度、断裂能和应力/面积度量。测试纯拉、纯剪、压剪、卸载再加载、刚体转动客观性、切线有限差分及纯模态耗能。体材料目录名义强度 4.4 MPa 不能无说明地当作当前构形界面强度。论文指数律微小开口即有损伤，不能用任意 `d>0` 定义实验起裂；其“特征开口＝刀厚/10”在当前 Gc、刀厚下隐含峰值约 32.3 MPa，不能当无关紧要的数值设置。
2. **真正的双面组件。** 在共形界面上建立配对、可独立位移的自由度，初始黏聚完整；验证分离、刀具进入、压闭与滑动。完整段、过程区、脱黏段分别记录。初始零间隙不能直接交给正距离 IPC 屏障；已有零间隙增广拉格朗日组件可供借鉴，但不等于通用接触已验证。明确黏聚剪切与裂面摩擦的分工，避免重复承载，压闭不能恢复已失去的黏结。
3. **真实 60°、冻结损伤对照。** 同一新网格比较连续连接与冻结损伤的黏聚连接，隔离界面人工柔度；随后与旧完整基线比较，区分网格变化。可预先用反力和归一化原表面位移差 2% 作为数值工程目标，不能称实验误差界。普通 Delaunay 网格不保证预设路径共形，需要显式界面边、配对积分和拓扑检查。
4. **起裂/分离短窗口。** 从一致的完整历史推进，分开锁定首次达到强度、软化区增长、充分脱黏及裂口张开。检查实际力平衡、正 J、接触互补、无穿透、不可逆性及能量账；做步长、网格、界面刚度和路径小偏置/倾角敏感性。大阻尼不能用来掩盖未收敛，稳定化耗能独立记录。
5. **通过后才跑 60°、0→3 mm。** 保存冻结代码、配置、所有接受态和失败状态。然后才扩展角度、润滑或回撤；不能先抛出三个完整 CZM 作业等待“撞运气”。

固定目标：CB245NE；侧面 30×5 mm、出平面宽度 20 mm；底面固定；原顶面名义落刀位置 s=7 mm；刀尖半径 **0.02 mm**；工件先做 **60°**；刀具沿实验室竖直方向下压，目标 **3 mm**；G=0.66 MPa、ν=0.49。Gc=2.15 N/mm、干摩擦 μ=0.35 是未独立标定的先验，不能为收敛而暗改。刀厚 0.245 mm、楔角 23° 为现实现输入，不要说成已由用户独立测量。裂面摩擦和刀具摩擦不是同一个已知参数。

规范和旧配置还保留 `δc=0.18 mm`、`σc≈4.39 MPa`、`ηreg=0.05 s` 等历史 cohesive 初值；它们不在当前完整材料/方向相场入口中生效，也没有独立标定。保留其来源记录，不能在新 CZM 中悄悄启用后称为已验证材料参数，尤其不能用旧 η 自动决定服务器求解稳定化。新界面采用/偏离这些初值的理由、量纲及对耗能的影响必须明确记录。

路径需要写明参考坐标：`es=(cosθ,-sinθ)`、`en=(sinθ,cosθ)`，初始竖直方向对应 `(Δs,Δn)=(q sinθ,-q cosθ)`。圆刀尖与斜面相切使刀轴和原接触点存在偏置；按刀轴与初始顶面的交点起线，60° 时 `s0=7+r tanθ≈7.03464 mm`。详细几何假设见断裂路线；不把局部法线 s=7 当成实验室竖直刀轴。

## 3. 获取代码和准备环境

以下为 Linux shell 示例；操作系统、CPU、调度器和模块名仍未知，需按服务器实际情况调整。现有本地证据来自 Windows，**尚未完成目标 Linux 节点的安装、测试或数值验收**。

```sh
git clone --branch feature/czm-cutting --single-branch https://github.com/Welcome2Machine/cutting_rubber_soul.git
cd cutting_rubber_soul
git status --short
git rev-parse HEAD
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -c "import numpy, scipy, ipctk; print(numpy.__version__, scipy.__version__, ipctk.__version__, ipctk.__file__); assert ipctk.__version__ == '1.6.0'"
```

优先选择 Python 3.10/3.11，匹配当前已用依赖而不盲目升级解释器。直接依赖固定为 NumPy 2.2.6、SciPy 1.15.3、threadpoolctl 3.6.0、Matplotlib 3.10.6、ipctk 1.6.0；这不是跨平台完整二进制锁文件。`pip check` 与导入成功仅说明环境初步可用，不说明数值正确。

### IPC 原生库与无外网环境

`ipctk` 是 C++ 扩展，优先使用匹配 Python/操作系统/架构的官方 wheel。没有对应 wheel 时，pip 可能源码编译，需要编译器和构建依赖；**不能复制 Windows 的 `.pyd` 或 `.runtime/ipc-python` 到 Linux**。IPC Toolkit 是接触工具集，主机有限元和求解器来自本仓库，不是直接调用完整 PolyFEM。[官方说明](https://github.com/ipc-sim/ipc-toolkit)

v1.6.0 源码要求 CMake ≥3.24、C++17 编译器，支持 Python 构建选项；第三方依赖可能在 CMake 配置时下载。先使用服务器提供的编译器/CMake 模块；如需 pip 安装构建工具，只放进虚拟环境。没有预编译包时的显式构建尝试为：

```sh
export CMAKE_BUILD_PARALLEL_LEVEL=2
python -m pip install 'cmake>=3.24' ninja
python -m pip install --no-binary=ipctk 'ipctk==1.6.0'
```

这是构建途径，不是本轮已通过的安装配方；若失败，保存完整构建日志，按 v1.6.0 的实际报错核查编译器、系统库、网络与架构，不跳过版本检查。不要直接改装上游 `main` 或下载不明二进制。[v1.6.0 构建配置](https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.0/CMakeLists.txt)、[官方发行包](https://pypi.org/project/ipctk/1.6.0/)

计算节点无外网时，在兼容 Linux/Python/架构的获准联网环境先准备 wheelhouse，并包含全部依赖；然后使用 `python -m pip install --no-index --find-links /path/to/wheelhouse -r requirements.txt`。仅缓存 PyPI 源码压缩包通常不足以离线完成 CMake 依赖下载。构建作业和运行作业都应在许可节点、资源配额内进行。

FFmpeg 仅用于 MP4；没有 FFmpeg 不能声称视频后处理通过。数值求解不需要先构建 FEBio。`tools/build_febio.py` 和 `tools/run_hpc.sh` 是独立 FEBio 参考路线，后者**不是 IPC 或未来 CZM 的提交脚本**。

## 4. 服务器验证顺序（本轮未执行）

先限制线程；这里 2 线程是起步配置，应不超过实际获配 CPU：

```sh
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLBACKEND=Agg
python -m unittest discover -s tests -p test_shared_material.py -v
python -m unittest discover -s tests -p test_model_config.py -v
python -m unittest discover -s tests -p test_directional_orientation.py -v
python -m unittest discover -s tests -p test_crack_contact_geometry.py -v
```

以上是无需本地结果夹具的材料/几何/方向核检查。随后在 ipctk 安装正确后执行原生 IPC 和窗口链校验，再执行全套；将 stdout/stderr 和真实退出码保存到新的服务器验证目录：

```sh
python -m unittest discover -s tests -p test_ipc_reference.py -v
python -m unittest discover -s tests -p test_nucleation_window.py -v
python -m unittest discover -s tests -p test_closed_crack.py -v
python -m unittest discover -s tests -v
```

`test_nucleation_window` 自建小网格，不需要本机完整算例，但其原生部分需要 ipctk。`test_closed_crack` 验证已脱黏组件，不是新 CZM。全套包含有界求解，并非单纯语法检查。服务器应按实测分别报告 tests run、passed、skipped、failures、errors，不照搬本地历史数量。

冷克隆预期会跳过若干本地资料相关回归：旧接触/舍入失败态、IPC 的 0.91 mm、30° 1.56 mm 和摩擦续算历史、方向损伤被拒绝试值、2023 大拉伸夹具；没有可选 FEBio 二进制时相应集成测试也跳过。缺失 ipctk 的部分测试同样会跳过，因此**“全套无失败但 IPC 测试全跳过”不允许进入 IPC 完整算例**。缺失夹具不能用伪造数据或新的不等价算例补成通过。

有界平台力学检查和已有接触组件的实际入口如下，均需全新输出目录：

```sh
python -m prototype.ipc_reference --case patch --output outputs/server_patch --depth .1 --seconds 60 --threads 2 --no-visualize
python -m prototype.closed_crack --output outputs/server_closed_crack --threads 2 --seconds 180
```

成功也只通过对应平台/组件关口。不要把 patch 平板压缩当成 0° 实验，也不要把上述已脱黏组件当成完整材料起裂。

## 5. 可选择重建的无断裂 60° 完整算例

只有服务器准备与短测试通过、并决定需要重建对照历史时才用本节。它是**现有完整材料基线的当前刀尖尺寸变体，不是 CZM**。最先重建干态 60° 即可；无须自动启动历史三组/五组队列。

在调度器分配的单节点作业内，前台执行：

```sh
python -u -m prototype.ipc_reference \
  --output outputs/server_intact60_dry \
  --case oblique --inclination 60 --interface dry --friction .35 \
  --tip-radius .02 --local-spacing .05 --surface-spacing .005 \
  --fine-halfwidth 3 --nose-facets 64 --dhat .0025 \
  --increment .01 --depth 3 --seconds 21600 --threads 2 \
  --max-iterations 500 --hessian-mode regularized \
  --predictor affine --friction-linear-solver direct --initiation
```

起裂诊断仍是方向 AT1 完整态筛查，之后继续无损伤。不要加 `--baseline`：该开关锁定旧的无摩擦、0.08 mm 刀尖、4 线程基线。当前命令可被 CLI 接受不等于已在服务器跑完；摩擦、软组织严重压缩和平台差异仍可能导致失败。目录必须不存在或为空；用唯一目录区分重试，不覆盖接受态。`--seconds 0` 可取消程序墙钟预算，但不取消调度器时限和数值验收，不应作为默认“修复”。

若是普通服务器、没有调度器，可在同一命令末尾使用现有 `--detach`，由程序保存启动记录和分开的 stdout/stderr；启动后确认一次即交给日志。Slurm/PBS 作业必须以前台进程运行，**不在作业内使用 `--detach`**，避免调度器失去进程生命周期管理。下方为需服务器 agent 按实际分区/账户补充的 Slurm 脚本框架，不是本轮已提交的作业：

```bash
#!/usr/bin/env bash
#SBATCH --job-name=intact60
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=intact60-%j.stdout.log
#SBATCH --error=intact60-%j.stderr.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
TASK_META="outputs/job-${SLURM_JOB_ID}"
mkdir -p "$TASK_META"
git rev-parse HEAD > "$TASK_META/git_commit.txt"
git status --short > "$TASK_META/git_status.txt"
python -m pip freeze > "$TASK_META/packages.txt"
python -c "import platform; print(platform.platform()); print(platform.processor())" > "$TASK_META/platform.txt"
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_META/exit_code.txt"' EXIT
# 在这里粘贴上面的已核实命令，将 --output 改为 outputs/intact60-${SLURM_JOB_ID}。
```

该框架最后需要填入真实求解命令后才能提交，16 GB/8 小时只是资源示例，不是耗时或内存保证。程序预算与作业时限之间应留出异常记录和后处理时间。被调度器强杀或节点故障时，shell trap 也可能无法完成；此时结合调度器最终状态记录 `killed/timeout`，不能把遗留 `running` 当作还在正常计算。

## 6. 数据、记录与效率要求

公开仓库不含 `outputs/`、`tests/fixtures/`、原始论文 PDF、图片、视频、实验数据或 `.runtime/`。文档中的 `outputs/ipc_friction_unlimited_20260912/...` 是原机器本地路径，冷克隆不会拥有这些文件。不能直接执行依赖它们的 `--source`、导出或窗口命令。可在服务器从零重建完整接受历史；如果用户以后独立传输本地数据，还需核验完整状态链、材料、网格、源码与 IPC 二进制指纹。跨平台二进制必然可能不同，不能删掉一致性检查就声称严格续算等价。当前 `--resume` 也不支持在线起裂模式。

完整作业运行期间保持执行目录代码不变，开发用另一工作目录；每个算例保留本身生成的 `source/` 冻结源码、指纹、基线/实际配置、接受态 `states/`、`run.log`、`result.json`，另保留调度器作业号、命令、Python/包版本、原生库指纹、stdout/stderr 和退出状态。不同入口的记录格式有差别：完整材料实际参数主要在 `result.json`，组件/研究窗口还可有 `config.json`，不能靠一个文件名判断记录齐全。

力学通过须同时检查完整目标、接受态链、实际力残差、正 J 和接触/摩擦验收。只认 `ACCEPT` 状态，`failed_trial.npz`/`rejected.npz` 不能混入结果。`completed_window`、组件 `prefix_completed`、队列 `finished`、后处理完成、CZM 模型完成和实验验证分别报告；进程存在、退出码 0 或优化器 success 都不足以证明完整切割正确。网格变形和最大主拉应力可视化保留，数值失败只画明确标记的有效接受前缀。

当前主要是单进程、单节点共享内存；没有已实现的多节点 MPI 或主线 GPU 加速。先在同一个有界问题比较 2/4/8 线程和峰值内存，再选资源；不要将“服务器很多核”直接改成全部线程。SciPy 稀疏直接分解和 Python 组装不会随 OpenMP 核数自动线性加速，多个案例并行还会争夺内存。优先计时接触构造、材料/界面组装、稀疏分解、全局迭代及输出，优化前后保留同条件反力与残差对照。

实验力曲线和光流/DIC 由用户保留作独立验证，暂不索取、不反演拟合。保持物理目标和真实参数不变；必要的界面参数假设与稳定化影响要可追溯。及时删除确认过时的临时内容，保护原始资料、活动作业和必要失败证据。发布继续只包含代码和文档；本次特许推送 `feature/czm-cutting`，不推送 legacy `master`、标签、`--all` 或 `--mirror`。后续推送审计以仓库最新发布规则为准，不把历史全量数据带入新分支历史。
