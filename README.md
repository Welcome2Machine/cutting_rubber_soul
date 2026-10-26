# SoftCut Solution · 橡胶斜切研究工作台

把有限元仿真与视频材料点观测放在同一个本地工程内：启动各自流程、检查输入、导出已审查有限元状态、固定材料点映射、比较同参考态位移，并保存可追溯的 HTML / CSV / JSON 报告。

## 开始使用

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

## 工程结构

| 路径 | 职责 |
|---|---|
| `projects/softmodeling/` | 真实 FE 工程完整工作区副本，包括原数据、保存状态、本地运行库 |
| `projects/rec_inc_soft_cut/` | 视频工程完整工作区副本，包括视频、权重和已有结果 |
| `softcut_solution/` | 统一桌面界面、任务编排、适配器与共同观测比较 |
| `configs/` | 可提交的环境模板和显式比较参数 |
| `researchtopic/` | 用户提供的研究介绍，原文保留 |
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
python -m pytest -q
python tools/check_publish.py --staged
```

视频批处理默认只预检：`python -m softcut_solution video-batch projects/rec_inc_soft_cut/configs/batch.example.json`。只有加 `--execute` 才运行跟踪，预算读取本地 `video_timeout_s`，默认 3600 秒。

## 比较结果的含义

首版实现固定材料点的位移分量、固定群组平均值、点误差和覆盖率。视频参考位置经显式变换映射到 FE **接受参考态**三角形，固定重心权重采样后续状态；不是直接比较位移模长。FE 仅采用独立审查的连续合格前缀，不外推，不给域外点补最近节点，不将遮挡/推断当有效观测。

默认刀尖平移注册和视频毫米尺度属于条件假设，并非实验室标定。播放时钟不用于物理同步。完整体 FE 速度仍为 0.2 mm/s，最大行程 2 mm，不含断裂；分析截止不是起裂事件。仿真反力不自动归零。MMT/OBT 接触地址、材料位移和真实界面滑移保持不同语义。

同标距应变、区域距离、经确认的接触滑移、力—视频同步和更广的试次注册是后续扩展接口；当前报告不会宣称这些尚未实现或标定的能力。

详细说明：[接口契约](docs/interface_contract.md) · [环境与迁移](docs/environment.md) · [Git 管理](docs/git_workflow.md) · [验收记录](docs/verification.md)。
