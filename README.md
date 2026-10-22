# 橡胶斜切仿真

使用同一套材料与接触模型，计算 0°、33°、57°橡胶压入过程的竖直反力 Fz、表面运动与刀尖接触材料的参考位置。当前仅计算完整体，目标行程 **2 mm**；实验起裂后的曲线不作为完整体拟合依据。

- [当前研究状态](docs/research_status.md)：完成范围、待解决问题与当前运行入口。
- [任务要求](docs/task_requirements.md) · [模型规格](docs/current_model_spec.md) · [代码地图](prototype/README.md)。
- [共同材料 2 mm 基线](docs/intact_2mm_baseline.md) · [接触位置观测](docs/tip_contact.md)。
- [文献原文与统一导读](references/README.md) · [有效研究历史](docs/research_history.md)。
- [光流数据接口](docs/optical_flow_handoff.md) · [2023 论文复现](reproductions/dammass2023/README.md)。

当前基线为 P1 平面应变、有限应变等容 Maxwell 与原生 IPC 接触：G∞=0.14 MPa、Gv=1.24 MPa、τ=1.7 s、μ=0.35。已有三角度各 201 态的完整 2 mm 路径，603 态独立重组审查通过；它是可计算基线，尚未完成共同材料实验标定。

当前已按用户要求停止计算，无活动队列。可复用结果集中在 `outputs/baseline/angle0`、`angle33`、`angle57`；论文复现结果保留在 `outputs/paper2023`。基线保留参数、接受态和审查记录，不再保留整套冻结源码或旧运行器副本。

代码和文字文档可进入 Git；论文原文、实验数据、视频、运行环境和计算结果保留本地。`outputs/` 只保留可复用基线、必要研究证据及论文复现结果。项目测试应为可重复使用的回归测试，不保存一次性探针和临时运行副本。

Python 3.10 环境安装 `python -m pip install -r requirements.txt`；IPC版本固定为1.6.0。
执行 `python -m unittest discover -s tests` 运行可重复回归，不启动长计算。
本地 `.runtime/ipc-python` 可由程序自动发现；PARDISO为可选运行库，视频导出另需FFmpeg。
计算、队列、审查和后处理命令统一见[工具说明](tools/README.md)。
