# SoftCut Solution · 橡胶斜切研究工作台

把有限元仿真与视频材料点观测放在同一个本地工程内：启动各自流程、检查输入、导出已审查有限元状态、比较同参考态位移，以及显式运行起裂候选诊断和受限裂后研究算例。各流程保存可追溯的 HTML / CSV / JSON 报告。

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
5. “研究资产”页可对已有完整体算例执行“诊断起裂指标”，或选择 JSON 后点击“运行裂后算例”。默认配置为混合模式自由裂路内聚网络：所有内部网格边可起损，裂面进入真实IPC/CCD；也可选择预设接缝对照。自由裂路仍受网格边方向限制，参数为独立给定示例。打开页面和刷新资产不启动任务。

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
python -m softcut_solution fracture-onset --case projects/softmodeling/outputs/baseline/angle0 --length-mm 0.15 --seconds 300
python -m softcut_solution fracture-run --config projects/softmodeling/configs/fracture_demo.json --seconds 300
python -m softcut_solution fracture-run --config projects/softmodeling/configs/fracture_network.json --seconds 300
python -m pytest -q
python tools/check_publish.py --staged
```

视频批处理默认只预检：`python -m softcut_solution video-batch projects/rec_inc_soft_cut/configs/batch.example.json`。只有加 `--execute` 才运行跟踪，预算读取本地 `video_timeout_s`，默认 3600 秒。

`fracture-onset` 只读取已保存、已审查的完整体状态；`--case` 可重复，`--threshold-mpa` 为可选的有效阈值。没有已核实实验标签时，指定阈值也不代表完成标定。`fracture-run` 是显式求解命令，读取所选配置并写入新的 `runs/<作业>/result/`；预算允许 1–3600 秒。两者复用已配置的 FE 解释器、日志和进程监督。

## 比较结果的含义

新断裂分支的局部本构和最小自由选路基准已通过数值检查；完整切割、网格方向及间隙收敛尚未完成。全边网络的有限间隙会引入自由闭合和额外柔度，短前缀残差合格不代表材料预测有效。理论基准见[理论验证报告](runs/fracture_theory_20261028/index.html)；本轮新增可恢复压缩正则化、受控求解回溯与柔度/网格/间隙验证，实际完成范围见[收敛与敏感性报告](runs/fracture_convergence_20261028/index.html)。两份生成报告留在本机。

后续[前0.5 mm反力诊断](runs/front_diagnosis_20261028/index.html)定位了粗网格偏硬、内部压剪起损与损伤降力的叠加，并新增有限面距离闭合势，修正远距失效面仍承压的局部缺陷。修复版从初态只审查到0.2725 mm，完整路径仍未通过；研究配置单独保存，不能因Fz接近实验而提升物理资格。

首版实现固定材料点的位移分量、固定群组平均值、点误差和覆盖率。视频参考位置经显式变换映射到 FE **接受参考态**三角形，固定重心权重采样后续状态；不是直接比较位移模长。FE 仅采用独立审查的连续合格前缀，不外推，不给域外点补最近节点，不将遮挡/推断当有效观测。

默认刀尖平移注册和视频毫米尺度属于条件假设，并非实验室标定。播放时钟不用于物理同步。已有完整体 FE 基线仍为 0.2 mm/s、最大行程 2 mm、不含断裂；独立裂后研究入口不改变这些历史结果。分析截止不是起裂事件。仿真反力不自动归零。MMT/OBT 接触地址、材料位移和真实界面滑移保持不同语义。

同标距应变、区域距离、经确认的接触滑移、力—视频同步和更广的试次注册是后续扩展接口；当前报告不会宣称这些尚未实现或标定的能力。

详细说明：[接口契约](docs/interface_contract.md) · [起裂与裂后实现及证据](docs/fracture_implementation.md) · [环境与迁移](docs/environment.md) · [Git 管理](docs/git_workflow.md) · [首版验收记录](docs/verification.md)。

此前零间隙二维动力学结果见本地[实际刀具控制报告](runs/zero_gap_dynamic_20261028/index.html)与[方法/独立试验设计第18–20节](docs/fracture_implementation.md)。新增完整尺寸、居中刀具的镜像对称控制，接入精确未断连接、单边裂面接触和两个方向的惯性；原偏心/斜切及自由裂路尚未接入。三档含断裂算例均完成2 mm并通过逐态重算及时间路径CCD，但最细档全程能量缺口仍4.99%、首快速区间数值误差19.61%，失稳能量和时间/网格/裂路资格未通过。不使用切割力反推参数。旧[内核处理报告](runs/fracture_resolution_20261028/index.html)保留为历史证据。

上一轮[Fz异常修复与判据验证](runs/fracture_repair_20261029/index.html)接入C空间离散梯度、精确内聚路径功、Lobatto接触求积、刀身保形细分和独立邻域强度候选。已定位长刀边摩擦权重抵消产生负功的失败态；保守算法的已审查前缀显著降低能量缺口，但大降力仍存在，不能将其全部归于旧积分误差。完整含断裂2 mm、时间/网格/裂路和物理参数资格仍未通过；无释放的2 mm方案已否决。方法与独立试验增补见[第21节](docs/fracture_implementation.md)，生成图中的实验仅作趋势参考。

最新[强度—能量有限裂增量验证](runs/finite_increment_20261029/index.html)将冻结历史的成对虚拟裂纹判据接入实际刀具和动态CZM，并增加原始增量PTC、求解用摩擦延续及独立证据审查。三档时间设置到约1.196–1.199 mm，两档细化网格到.936/1.081 mm；已接受前缀能量误差减小，但仍有大掉力、内部孤立裂口和虚拟开裂后平衡困难，**裂后2 mm及时间/网格验证尚未通过**。入口为FE目录下`python -m tools.run_zero_gap_cut --config configs/zero_gap_finite_increment.json --out ../../runs/<新目录> --seconds <预算>`，仅显式执行才加载。未使用实验力反推参数；方法与独立试验映射见[第22节](docs/fracture_implementation.md)。
