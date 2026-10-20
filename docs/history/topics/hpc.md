# 超算运行准备

> Historical technical record. Obsolete outputs were retired on 2026-10-20; summary links do not include the original figures or full replay histories. See [current research status](../../research_status.md).

当前固定基线是 IPC，实际在本地 Windows 验证。若以后迁移到计算节点，先安装 requirements.txt 并运行 IPC 短测试；固定基线在作业内以前台运行：`python -m prototype.ipc_reference --baseline --output outputs/hpc_ipc_baseline`，至少分配 4 个 CPU。不要在 Slurm 作业内使用 `--detach`。改变线程或其他配置时需省略 `--baseline`，作为单独的平台诊断记录。下方 `tools/run_hpc.sh` 是保留的 **FEBio 参考模板**，不是 IPC 默认入口。

处理器、操作系统和调度器尚未知。当前按 Linux 单节点共享内存 CPU 准备，不指定学校账户、分区或厂商指令集。FEBio/OpenMP 和 SuperLU_MT 可使用多线程；大内存支持更细网格和直接分解，但不能自动解决非线性失收敛。Python 主线的 SciPy 稀疏 LU 不会因增加 OpenMP 线程而全面并行。

## 准备

按集群说明加载 Python 3.10+、GCC/G++、CMake、OpenMP 环境，在仓库根目录执行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/build_febio.py --contact-audit --jobs 8
python -m unittest discover -s tests -v
```

编译和数值测试应在获准的交互计算节点或作业内进行。构建固定 FEBio v4.13 / SuperLU_MT 8324379，使用内部参考 BLAS，未证明为最快配置。编译 `--jobs` 和求解 `--threads` 不得超过所获 CPU 配额。源码、许可证和二进制指纹保存在 .runtime。无外网时，在允许联网的兼容 Linux 环境准备依赖；不能复制 Windows exe 到 Linux。

Linux 分支已按官方 CMake 源码整理，当前仅验证 Windows 二进制和 Python 回归，仍需目标节点实编译。管理员提供的原版 FEBio 不能读取 corrected 专用参数，须先构建补丁版。

## 验证与完整诊断

```sh
python -m prototype.febio_reference --case patch --contact-variant corrected --threads 8 --output outputs/hpc_patch --execute --timeout 120
DEPTH_MM=.05 THREADS=8 bash tools/run_hpc.sh
```

确认 patch_verified、接触和短斜切审计后，完整 3 mm **无损伤参考诊断**命令为：

```sh
THREADS=8 bash tools/run_hpc.sh
```

它没有起裂停止或自由断裂，不能称为完整切割已验证。脚本默认 3 mm，失败仍记录状态，不为收敛缩减目标。线程从 SLURM_CPUS_PER_TASK 或 THREADS 读取，OUTPUT_DIR 可指定高速存储；默认目录唯一。调度器作业内前台执行，不使用脱离作业的后台入口。

若使用 Slurm，在仓库根目录提交：

```sh
sbatch --cpus-per-task=8 --mem=32G --time=24:00:00 tools/run_hpc.sh
```

这是示例资源请求，不是内存和耗时保证。按实际账户、分区调整；同一短算例比较 1/4/8/16 线程的时间和峰值内存后确定规模。目前没有加速实测结论或多节点 MPI 实现。

脚本保存 driver.log、Python 包版本、Git 提交/工作树状态、退出码；算例保存输入、线程/体系结构、源码/二进制哈希、日志和状态。调度器强杀可能留下 running 或不完整转储，应结合调度器最终状态判断。restart.dmp 读取后接续等价尚未验证。

Python 主线单独排队时使用 `python -u -m prototype.run_initiation --config prototype/initiation_config.json --output outputs/new_mainline --mode to_candidate`，由调度器记录日志。它仍存在连续失效，候选阈值及材料保持原定义。

依据：[FEBio CMake](https://github.com/febiosoftware/FEBio/blob/v4.13/CMakeLists.txt)、[SuperLU_MT](https://github.com/xiaoyeli/superlu_mt/tree/8324379)、[Slurm sbatch](https://slurm.schedmd.com/sbatch.html)。
