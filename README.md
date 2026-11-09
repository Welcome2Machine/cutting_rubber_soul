# SoftCut Solution · 橡胶斜切研究工作台

> **当前研究任务（2026-11-09）**：[唯一任务书](researchtopic/research_prototype_task.md) · [本地 agent](researchtopic/baseline_agent_tasks_20261031.md) · [服务器 agent](researchtopic/server_agent_prompt.md) · [Git 协作](docs/git_workflow.md)。第一篇完成实验解释所需的共同前向模型；第二篇的学习/微分方向只保留基础。任务更新不代表新增计算结果。

把有限元仿真与视频材料点观测放在同一个本地工程内：启动各自流程、检查输入、导出已审查有限元状态、比较同参考态位移，以及显式运行起裂候选诊断和受限裂后研究算例。各流程保存可追溯的 HTML / CSV / JSON 报告。

## 开始使用

研究协作使用 `codex/server-fracture-20261029` 作为服务器集成分支，
本地参考与实验交接使用 `codex/intact-baseline-upgrade-20261031`。
先按[当前任务](researchtopic/research_prototype_task.md)和[Git 规则](docs/git_workflow.md)核对现场。
[早期远程接管说明](docs/remote_agent_handoff_20261029.md)仅保留环境/历史依据，其旧优先级不自动执行。
实际完成范围与失败证据见[研究状态](projects/softmodeling/docs/research_status.md)。

双击 `Launch.cmd`，或在当前目录执行：

```powershell
.\Start.ps1
# 等价入口
python -X utf8 -m softcut_solution gui
```

桌面工作台包含“项目与环境”“共同观测比较”“研究资产”。打开工作台不启动计算。

1. 在项目页检查两个解释器；点击“仿真工作台”打开复制工程的完整仿真界面，可配置材料、刀具、摩擦、角度和有界算例。
2. 视频可先按试次预检，再通过显式批处理按钮执行清单。新结果写入复制工程的 `results/integration/`，历史结果不覆盖。
3. 在比较页选择已审查 FE 算例并导出保存帧，选择视频固定 ID 包，核查注册、参考态、深度窗口与支持规则，再运行比较。
4. 结果页打开 HTML 报告和对应机器文件。33°、57°现成数据已接入；参考帧、比例尺与完整性资格仍独立记录。
5. “研究资产”页可对已有完整体算例执行“诊断起裂指标”，或选择 JSON 后点击“运行裂后算例”。默认配置为混合模式自由裂路内聚网络：所有内部网格边可起损，裂面进入真实IPC/CCD；也可选择预设接缝对照。自由裂路仍受网格边方向限制，参数为独立给定示例。打开页面和刷新资产不启动任务。

## 工程结构

新增显微刀形参数见 [测量配置](configs/blade_profiles.measured.json) 和
[提取及工作台转换说明](docs/blade_profile_config.md)。包括 C、E 两件、A 高/低性能两件；
用户已确认这些图为仿真所用侧视截面；A 两组均反复使用，同刀性能变化不明显而刀间
差异显著。截面尺寸保留像素测量范围，未分辨的刃口半径仍未知，不自动替换旧仿真基线。

| 路径 | 职责 |
|---|---|
| `projects/softmodeling/` | 真实 FE 工程完整工作区副本，包括原数据、保存状态、本地运行库 |
| `projects/rec_inc_soft_cut/` | 视频工程完整工作区副本，包括视频、权重和已有结果 |
| `softcut_solution/` | 统一桌面界面、任务编排、适配器与共同观测比较 |
| `configs/` | 可提交的环境模板和显式比较参数 |
| `researchtopic/` | 原始研究介绍、唯一当前任务和两个角色入口 |
| `runs/` | 新任务日志、运行记录、比较报告；本地保存 |
| `provenance/` | 导入清单；原 Git bundle 和未提交修改 patch 本地保存 |
| `docs/` | 接口、运行环境与 Git 约定 |

用户最初给出的 `0915ForceProcessing` 已排除。原始两个项目目录不修改；新工程没有嵌套 Git 仓库。源工作区的未提交修改和未跟踪源码也已复制。视频和仿真继续使用各自解释器，不将 Python 3.10/3.11、IPC 和 PyTorch 强行装进同一环境。

## 可复现命令

```powershell
python -m softcut_solution doctor
python -m softcut_solution inventory
python -m softcut_solution video-check 0930l30a1
python -m softcut_solution export-fe projects/softmodeling/outputs/baseline/angle33 --frames 21
python -m softcut_solution compare --fe <result.bundle.json> --video <固定ID包目录> --options configs/comparison.example.json
python -m softcut_solution fracture-onset --case projects/softmodeling/outputs/baseline/angle0 --length-mm 0.15 --seconds 300
python -m softcut_solution fracture-run --config projects/softmodeling/configs/fracture_demo.json --seconds 300
python -m softcut_solution fracture-run --config projects/softmodeling/configs/fracture_network.json --seconds 300
python -m pytest -q
python tools/check_publish.py --staged
```

视频批处理默认只预检：`python -m softcut_solution video-batch projects/rec_inc_soft_cut/configs/batch.example.json`。只有加 `--execute` 才运行跟踪，预算读取本地 `video_timeout_s`，默认 3600 秒。

`fracture-onset` 只读取已保存、已审查的完整体状态；`--case` 可重复，`--threshold-mpa` 为可选的有效阈值。没有已核实实验标签时，指定阈值也不代表完成标定。`fracture-run` 是显式求解命令，读取所选配置并写入新的 `runs/<作业>/result/`；预算允许 1–3600 秒。两者复用已配置的 FE 解释器、日志和进程监督。

## 比较结果的含义

断裂实现、历史路线及其证据统一见[方法记录](docs/fracture_implementation.md)。
局部本构/算例通过、机制演示、完整行程、时间/网格敏感性和实验预测分别判断；
不能用一个通过标签替代全部资格。本文不再逐轮复制进展，当前结果以研究状态为准。

首版实现固定材料点的位移分量、固定群组平均值、点误差和覆盖率。视频参考位置经显式变换映射到 FE **接受参考态**三角形，固定重心权重采样后续状态；不是直接比较位移模长。FE 仅采用独立审查的连续合格前缀，不外推，不给域外点补最近节点，不将遮挡/推断当有效观测。

默认刀尖平移注册和视频毫米尺度属于条件假设，并非实验室标定。播放时钟不用于物理同步。已有完整体 FE 基线仍为 0.2 mm/s、最大行程 2 mm、不含断裂；独立裂后研究入口不改变这些历史结果。分析截止不是起裂事件。仿真反力不自动归零。MMT/OBT 接触地址、材料位移和真实界面滑移保持不同语义。

同标距应变、区域距离、经确认的接触滑移、力—视频同步和更广的试次注册是后续扩展接口；当前报告不会宣称这些尚未实现或标定的能力。

详细说明：[接口契约](docs/interface_contract.md) · [起裂与裂后实现及证据](docs/fracture_implementation.md) · [环境与迁移](docs/environment.md) · [Git 管理](docs/git_workflow.md) · [首版验收记录](docs/verification.md)。
