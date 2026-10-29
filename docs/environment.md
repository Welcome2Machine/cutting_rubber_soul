# 运行环境与迁移

## 1029 Linux 服务器（2026-10-29）

服务器从整合分支 `15b705f` 开始，在独立 `codex/server-fracture-20261029`
分支开发；仓库直接位于 `/LARGE1/gr10578/Zhao_Guorui/SoftCutting/1029`。
FE/整合解释器为该目录 `.venv/bin/python`，基于系统 Python 3.11.13 创建，
固定安装两份 requirements（IPC 1.6.0、NumPy 2.2.6、SciPy 1.15.3）。Tk 8.6
可导入，但无显示服务，GUI 回归按既有规则跳过，不能写作已通过 GUI 检查。
完整依赖冻结、数据清单及日志保存在本地 `runs/server_1029_round1/`。

POSIX 虚拟环境的 Python 通常是指向系统解释器的符号链接。整合层保留所选启动
路径，不对解释器执行 `Path.resolve()`；否则子进程会丢失虚拟环境的 `sys.prefix`
和安装包。数据目录的边界检查仍解析符号链接。`doctor` 同时检查可导入的 `ipctk`，
`local_ipc_runtime=false` 只表示不存在旧 Windows 本地运行库目录。

本机 `settings.local.json` 已登记 FE 解释器；视频独立环境、权重、原视频、显微 PPT、
实验 CSV、完整体及 Windows 断裂历史算例尚未传输。视频解释器保留未就绪状态，
没有用 FE 环境替代视频环境，也没有启动 GPU 推理。基准 FE 可以从初态复算，
这不属于重放 Windows 失败现场。

服务器有 Slurm，采用本站 `--rsc=p=1:c=4:t=4` 语法申请资源，IPC/BLAS 各2线程。
加载与审查串行执行，内部截止时间之外再设置子进程硬超时及调度器墙钟上限。
用户本轮授权最多一个 FE 加载作业，单加载≤3600 s、单审查≤600 s、首轮累计计算
墙钟≤4 h；到期保存证据，不自动续算或启动旧队列。每次作业的日志和预算账本留本地。

## 原 Windows 整合环境（历史记录）

整合层使用 Python 3.10 + NumPy + Matplotlib + Tkinter。当前机器的 Python 3.10 已满足版本，未新增安装。仿真固定依赖见 `projects/softmodeling/requirements.txt`，`.runtime/ipc-python` 已复制；不自动下载运行库。

视频解释器沿用源库已登记的 `D:/Doctoral Research/AcademicForge/softcutting0731/.venv/Scripts/python.exe`（Python 3.11）。这是明确记录的本机共享运行环境例外；代码、输入及新输出均指向当前复制工程，解释器本身尚未复制。PyTorch/FFmpeg/权重依赖按视频工程 `environment/` 管理，不能用仿真依赖表替代。

`settings.local.json` 保存本机两个解释器、仿真工作台端口和视频批处理预算，不提交 Git。新机器从 `configs/settings.example.json` 复制并修改，通过界面保存设置或直接编辑；执行 `python -m softcut_solution doctor` 检查。检查只说明包可找到，不代表数值或实验结果通过验收。

新克隆只包含代码、文档和配置，不能凭 Git 恢复约 52 GB 数据。另行复制两个项目的数据/结果/模型/运行库，或按各子项目说明建立环境。导入清单在 `provenance/import_manifest.json`，记录原路径、commit、未提交状态、所有复制文件数量/字节数和源码指纹。

任务使用参数数组调用 subprocess，工作目录固定为当前副本；历史报告及 plan 中原绝对路径作为来源证据保留，不批量替换。新视频 batch 清单会重建本地输出根目录，正常数据发现由副本 ROOT/data 执行。历史 HTML 的外部绝对链接可能仍指向原来源，不等于新任务从原目录读取。

仿真工作台选用独立本机端口，并记录会话 token，避免误连原库的 8765 服务。关闭桌面工作台不关闭独立 FE 服务。FE 内部的运行/停止/预算继续由原工作台管理，旧队列不会自动恢复。

启动工具写 UTC 机器时间戳以保持可排序；阅读本地报告时以 JST（UTC+9）换算。无需公网、上传或远程服务。
