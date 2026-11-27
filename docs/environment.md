# 京都大学 System B：当前运行与环境

2026-11-27 核对官方文档。本页只维护机器和运行方法；研究输入、数值预算与任务顺序见 [唯一任务书](../researchtopic/research_prototype_task.md)。旧服务器环境、作业账本和路径保留在 [整理前环境记录](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/ec1f32745af51ba45658cce9c17c97a30ae06ae8/docs/environment.md)，不是本轮资源限额。

## 1. 工作目录与环境复用

用户工作根由用户启动提示词指定，以 `<USER_WORK_ROOT>` 表示。若尚未有仓库，推荐其下 `cutting_rubber_soul/`；已有正确仓库直接复用，不强制复制。运行、证据、调度脚本位于仓库忽略的 `runs/server_intact/`，不覆盖旧 `SoftCutting/1029`。

本次在计算节点实测既有 FE 解释器为 Python 3.11.13，NumPy 2.2.6、SciPy 1.15.3、ipctk 1.6.0、threadpoolctl 3.6.0、psutil 7.1.0。忽略目录引用服务器已有的 pypardiso 0.4.7，实际加载 oneMKL 2023.2 的 `libmkl_rt.so.2`。原虚拟环境继续使用同一解释器。现场准确路径保存在 `runs/server_intact/env.sh`。

旧现场曾使用 SysB/2022、slurm/2022、intel/2023.2、intelmpi/2023.2 和 Python FE 环境；这些是寻找已有环境的线索，必须现场检查 `module list/avail/show`、解释器和实际导入，不能照搬 Windows DLL、盘符或 PYTHONPATH 分号。

先确认既有 Python 能导入 numpy/scipy/ipctk/threadpoolctl/psutil 和所需 PARDISO。复用兼容解释器与已装库；只有确实不兼容时才建一套服务器 FE 环境，并记录原因。保留虚拟环境解释器的调用路径，不解析符号链接后误调用系统 Python。FE `requirements.txt` 为已用版本清单，IPC 需通过 API 检查；不自动重装全部库或修改全局 shell 初始化。

记录 `sys.executable`、依赖版本、模块列表、BLAS/MKL 动态库、pypardiso 实际库、`ipctk.__file__`、源码/输入 SHA 和 CPU 亲和性。Linux PARDISO 使用可用的 `libmkl_rt.so` 与现有环境；`module load intel` 不等于 Python 已链接到 MKL。缺 PARDISO 时先验证 native 后端；后端改变形成新的性能/恢复身份，不能改旧 manifest 冒充同一续算。

## 2. System B 官方规则与本项目策略

System B（Laurel 3）为每节点双路 Xeon Platinum 8480+，112 物理核、512 GiB。[硬件说明](https://www.iimc.kyoto-u.ac.jp/en/services/comp/supercomputer/system/specification)

当前调度是 Slurm。`--rsc p=进程数:t=每进程线程数:c=每进程物理核:m=每进程内存`；`t=64` 不是 64 小时，时限另用 `-t HH:MM:SS`。B 的每进程核上限 112、可申请内存上限 500G；省略 m 时按 4571M×c 分配，过大的显式内存可能提高实际分配核数。[资源规则](https://web.kudpc.kyoto-u.ac.jp/manual/en/run/resource)

`tssrun` 也通过调度器，忙时可能等待。用户过去的 `tssrun --x11 --rsc t=64:c=64:m=128G mentat` 是 GUI 会话经验，不保证 Python 并行效果或免排队；无界面 FE 不需要 `--x11`。[交互执行](https://web.kudpc.kyoto-u.ac.jp/manual/en/run/interactive)

现场先查 `spartition`、`qgroup` 和自己的 `squeue`，确认候选队列 `<VERIFIED_B_PARTITION>`、DefTime/MaxTime、Rstd 与剩余额度。Rstd 是单作业上限，Rmax 是组队列并发总限；不能把整组额度视为个人可独占。不要套用 `pjsub` 或普通 Slurm 的 `#SBATCH --cpus-per-task/--mem/--nodes/--ntasks/--exclusive`，本站要求资源通过 `--rsc`。程序执行位置使用 `srun`。[批处理](https://web.kudpc.kyoto-u.ac.jp/manual/en/run/batch) · [站点限制](https://web.kudpc.kyoto-u.ac.jp/manual/en/run/tips)

本轮对同一 33° H1 从准备到首个 0.0025 mm 做了 4/8/16 物理核比较。两步耗时分别为 33.521、33.639、33.276 秒，峰值 RSS 为 340.0、359.8、376.8 MiB，材料更新约占 19 秒。每档均用 IPC 1、BLAS 1，PARDISO 线程数等于所申请物理核数。浅窗批次据此使用一个进程、4 物理核、6 GiB。当时短批次同时最多两项。本次启动前组内占用为 0/112 物理核。四条探索作业合计申请 16 物理核、24 GiB，每案一个前台进程。

线程池按实际嵌套关系核验，避免 IPC/BLAS/PARDISO/后处理worker 超订阅。站点 `SLURM_CPUS_PER_TASK` 是逻辑核数，不能直接拿它作物理核预算；核对 `SLURM_DPC_CPUS`、cpuset、lscpu及NUMA。比较墙钟耗时和物理核时，不把申请核数称为利用率。[环境变量](https://web.kudpc.kyoto-u.ac.jp/manual/en/run/tips)

## 3. 有限作业入口

先按任务书从可验证的 H_cases 输入生成本轮选定子集，在计算节点完成环境/短验证及 `--prepare-only`。显式传共同材料 CLI，不能依赖历史 `.14/1.24` 默认。准备 manifest 后才提交下例。脚本与日志保存到 ignored `runs/server_intact/`；提交前创建日志目录。

```bash
#!/bin/bash
#SBATCH -p <VERIFIED_B_PARTITION>
#SBATCH -t 02:00:00
#SBATCH --rsc p=1:t=4:c=4:m=6G
#SBATCH --no-requeue
#SBATCH -J rubber_intact
#SBATCH -o runs/server_intact/logs/%x.%j.out
#SBATCH -e runs/server_intact/logs/%x.%j.err
set -euo pipefail
: "${SOFTCUT_REPO:?set verified repository path}"
: "${SOFTCUT_PYTHON:?set existing FE interpreter}"
: "${SOFTCUT_BATCH:?set prepared manifest directory}"
# 这里加载现场已确认的模块和库路径；不写死旧Windows运行库。
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$SOFTCUT_REPO/projects/softmodeling"
srun "$SOFTCUT_PYTHON" -m tools.run_material_queue \
  --output "$SOFTCUT_BATCH" --launch-prepared --foreground
```

提交前将 `<VERIFIED_B_PARTITION>` 替换成 spartition 已确认的真实B队列；它不是可直接提交的字面值。从仓库根提交，三个 SOFTCUT 变量及库路径均用现场绝对路径设置；Slurm 时限须覆盖各case预算、审查额度与结束写盘。上例2小时是示例，不足时按已批准有限任务的实际总预算调整，且不超过spartition限制。

`--foreground` 保留原启动校验、唯一claim、状态和日志，等待worker终止并传播失败；不能用隐藏 `--worker` 跳过launch校验，也不能在Slurm中启动默认detached模式后退出。`--resume-queue --foreground` 同样须在新的有效allocation中运行。`--prepare-only` 不应自动启动求解。单个短诊断可在tssrun内前台执行现有单算例入口。

## 4. 查询、暂停与恢复

启动后只短时核实JobID、日志和一个新接受态（或有时限的诊断实际推进），然后汇报，不持续监控。

从仓库根执行 `runs/server_intact/control.sh CASE_ID status`、`log`、`pause` 或 `resume`。长批次的唯一 ID 为 `A33_H1_2mm`、`A33_H2_2mm`、`A57_H1_2mm`、`A57_H2_2mm`。每条有独立 manifest、源码摘要、预算、输出和 Slurm allocation，使用 `--no-requeue`。实际作业号存于 `runs/server_intact/long_validation/CASE_ID.jobid`。首次逐步日志在 `runs/server_intact/CASE_ID/CASE_ID/run.log`，Slurm stdout/stderr 在 `runs/server_intact/logs/rubber_CASE_ID.JOBID.out` 和同名 `.err`。控制入口跟随当前 attempt，重启前要有已收回的合作暂停回执。已消耗的求解、审查、导出和总墙钟均随 attempt 扣除。尚在排队的作业由该入口按提交回执执行 Slurm hold/release。原角度简写仅用于旧浅窗。

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,Elapsed,TotalCPU,AllocCPUS,MaxRSS,ExitCode
# 在 FE 项目目录读取共享状态并请求当前 attempt 合作暂停
"$SOFTCUT_PYTHON" -m tools.run_material_queue --output "$SOFTCUT_BATCH" --status
"$SOFTCUT_PYTHON" -m tools.run_material_queue --output "$SOFTCUT_BATCH" --pause
```

状态中的 `owner` 给出 JobID、节点、解释器、PID 和启动时间。跨节点显示 `remote_reported_running` 时用 `squeue` 核对调度状态。暂停标志只写入该批次当前 attempt，待 `status=paused` 和 `worker_reaped=true` 后，才能在新的 allocation 中以前台 `--resume-queue --foreground` 恢复。原检查点、求解及审查剩余预算均保留。强制取消或节点丢失须先独立核对现场，缺少正常退出回执时不能通过普通暂停恢复重开。作业依赖仅在相应数值验收通过后放行。

`--physical-core-affinity` 可从队列传给求解器。线程上限为 IPC×BLAS 与独立 PARDISO 调用线程数中的较大值，在 Slurm 内同时检查 cpuset 的实际 socket/core 对。用 `SOFTCUT_THREAD_BUDGET` 指定作业的物理核申请数。

## 5. 存储与节点边界

先查目标目录已有文件、写权限与 `quota -p`。/LARGE1 默认备份策略可被组管理员调整；不要假定已有异地备份，也不要自行改用户指定根。/FAST和作业私有/tmp可用于可再生临时数据，checkpoint、日志和最终证据写回指定LARGE工作区。[存储说明](https://web.kudpc.kyoto-u.ac.jp/manual/en/filesystem)

登录节点每用户有4核/16GB等限制；用于编辑、检查和提交。FE求解、性能比较和较重审查放到计算节点，不能靠nohup在登录节点长期运行。[登录节点说明](https://web.kudpc.kyoto-u.ac.jp/manual/en/config)

最终核账给申请/实际分配、线程、Elapsed、CPU、RSS、失败/排队时间和实际接受范围。作业与step统计避免重复累计；不高频机械轮询sacct，不把旧机器性能当作本机结果。
