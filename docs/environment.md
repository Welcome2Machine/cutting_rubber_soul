# 运行环境与迁移

整合层使用 Python 3.10 + NumPy + Matplotlib + Tkinter。当前机器的 Python 3.10 已满足版本，未新增安装。仿真固定依赖见 `projects/softmodeling/requirements.txt`，`.runtime/ipc-python` 已复制；不自动下载运行库。

视频解释器沿用源库已登记的 `D:/Doctoral Research/AcademicForge/softcutting0731/.venv/Scripts/python.exe`（Python 3.11）。这是明确记录的本机共享运行环境例外；代码、输入及新输出均指向当前复制工程，解释器本身尚未复制。PyTorch/FFmpeg/权重依赖按视频工程 `environment/` 管理，不能用仿真依赖表替代。

`settings.local.json` 保存本机两个解释器、仿真工作台端口和视频批处理预算，不提交 Git。新机器从 `configs/settings.example.json` 复制并修改，通过界面保存设置或直接编辑；执行 `python -m softcut_solution doctor` 检查。检查只说明包可找到，不代表数值或实验结果通过验收。

新克隆只包含代码、文档和配置，不能凭 Git 恢复约 52 GB 数据。另行复制两个项目的数据/结果/模型/运行库，或按各子项目说明建立环境。导入清单在 `provenance/import_manifest.json`，记录原路径、commit、未提交状态、所有复制文件数量/字节数和源码指纹。

任务使用参数数组调用 subprocess，工作目录固定为当前副本；历史报告及 plan 中原绝对路径作为来源证据保留，不批量替换。新视频 batch 清单会重建本地输出根目录，正常数据发现由副本 ROOT/data 执行。历史 HTML 的外部绝对链接可能仍指向原来源，不等于新任务从原目录读取。

仿真工作台选用独立本机端口，并记录会话 token，避免误连原库的 8765 服务。关闭桌面工作台不关闭独立 FE 服务。FE 内部的运行/停止/预算继续由原工作台管理，旧队列不会自动恢复。

启动工具写 UTC 机器时间戳以保持可排序；阅读本地报告时以 JST（UTC+9）换算。无需公网、上传或远程服务。
