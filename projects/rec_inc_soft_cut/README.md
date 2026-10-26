# SoftCutting

使用现成分割模型识别材料和刀具，在去阴影图像上跟踪固定材料 ID，再计算刀具相对运动与原始表面接触地址。默认使用独立材料跟踪，不训练模型。

## 使用

根目录唯一 Python 入口为 `softcut.py`。运行 `python softcut.py --help` 查看命令，子命令后加 `--help` 查看参数。

| 命令 | 功能 |
|---|---|
| `import` | AVI 压缩、抽帧、来源与物理时间映射 |
| `track` | 识别、去阴影、固定材料 ID 跟踪、点阵视频 |
| `measure` | 从已有轨迹生成刀具、接触和材料运动前置数据 |
| `motion` | 下压期间完整刀面邻域的材料相对运动与表面地址 |
| `slip` | 显式截止深度内同时计算 MMT、OBT；汇总与可选视频 |
| `batch` | 统一清单批处理，串行 GPU 准备、最多两组 CPU 任务 |
| `gallery` | 已有估计的无文字双画面视频、静态英文曲线、PPT 压缩 |
| `optimize` | 接触交叉复核、材料身份/边界匹配、网络候选、有限变形与力学对照，不改变默认跟踪 |
| `exchange-export` / `exchange-check` | 有限元交接导出与独立校验 |
| `check` | 只读检查保留结果及其依赖 |

```powershell
# 只读检查材料跟踪准备条件
python softcut.py track --names 0930l30a1 --check-only

# 从初始整块材料跟踪至回退；外围点较稀疏
python softcut.py track --names 0930l30a1 0930l60a1 0930n60a1 --target-mm 3 --out results/my_tracks

# 清单定义结果路径和截止深度，执行完整流程
python softcut.py batch --manifest configs/batch.example.json --out results/my_batch --check-only
python softcut.py batch --manifest configs/batch.example.json --out results/my_batch --tracking-videos

# 从保存轨迹重算滑移，不重跑光流
python softcut.py slip --names 0930l60a1 --source results/my_batch/tracks --measurements results/my_batch/measurements --stop-mm 1.4 --out results/my_sliding --render

# 只更新已计算目录的报告
python softcut.py slip --out results/my_sliding --report-only

# 检查已保留的完整计算结果
python softcut.py check
```

`track --full-video` 包含退刀；默认只覆盖约 3 mm 下压。`--stage prepare|track|render|compress` 可单独运行跟踪阶段。材料 ID 始终保留，无图像支持的预测不记为观测。当前权重为 `models/segmentation/clean_roi_20260909/checkpoint_best.pt`，预检会核查该文件。

展示清单格式见 `python softcut.py gallery --help`，每个 case 指定 `name`、`cutoff_um`、`estimate`、`prepared`、`tracks`、`measurements`；路径可为绝对路径或相对清单目录。`--stage ppt` 只从已有展示视频压缩，`--stage plot` 只生成静态图。

## 两种滑移方法

**Mixed-model transport（MMT，混合模型输运法）**组合材料局部映射和曲线边界映射。**Objective boundary transport（OBT，客观边界输运法）**使用边界材料坐标并处理共同刚体转动。两方法共享轨迹、刀尖参考、接触零点与截止范围，各自输出，不平均曲线。

`slip --estimator bounded` 是默认有界输运，保存 `MMT-v2`、`OBT-v2`。`recovery` 使用恢复观测窗口；`converged` 保留收敛审查策略。这些策略需显式选择，不因代码整理相互替换。`--audit-contact-zero` 只在候选策略中启用接触零点复查。模型敏感度不等于标定精度。

每组输出原始方法 NPZ/JSON、共同输入、`comparison.csv/png` 和 `complete.json`。报告层内部插值与末端条件估计分别标记，原始求解值不被补点覆盖。缺少观测不能因曲线平滑变成高置信度测量；刀尖下的新裂面也不一定具有原始自由表面地址。

截止深度是输入，不是自动起裂检测。清单的 `contact_analysis_stop_mm` 优先；0930 a 系列保留 l30=1.0、l60=1.4、n60=1.5 mm 的实验设定。其他批处理未指定时分析完整 3 mm；独立 `slip` 必须显式指定截止或使用含截止的清单。新计算不使用截止以后的观测更新。

## 现有本地成果

数据和结果不进入 Git，路径不随代码整理移动。保留范围见 [configs/retained_results.json](configs/retained_results.json)。

结果存储规则见 [无损精简说明](docs/results_storage.md)：有限元交接轨迹默认使用 `tracks.csv.gz`，读取时以 `metadata.files.tracks` 为准；历史数值和无损测量视频保留。

| 内容 | 本地入口 |
|---|---|
| 1005/1014 共 23 组识别、去阴影、材料轨迹与测量 | `results/20261014_1005_1014_batch/index.html` |
| 规定深度内两套滑移 | `results/20261015_declared_fracture_sliding/index.html` |
| 内部插值、最后有效值和规定深度值 | `results/20261016_contact_gap_interpolation/index.html` |
| 最新五组静态英文曲线与无文字视频 | `results/20261016_five_group_sliding/index.html` |
| PPT 视频：1024×384、3 fps、速度不变 | `results/20261016_five_group_sliding/ppt_videos/` |
| 0930 有限元交接 | `results/20261012_optical_flow_exchange/README.md` |
| 五工况掩膜特征、原始表面曲线与零深度诊断 | `results/20261024_boundary_followup/index.html` |

1005/1014 截止值仍为：30/33 组 1300 μm、60/57 a 组 2200 μm、up 550 μm、down 850 μm。最新五组依赖 `20261016_slip_bias_audit/recalculated/` 的准备输入。以上成果没有因精简重算。

## 环境与开发

默认解释器为项目 `.venv/Scripts/python.exe`。本机没有该环境，已登记沿用 `D:/Doctoral Research/AcademicForge/softcutting0731/.venv/Scripts/python.exe`。TAPNext++ 的 `einops` 隔离在模型目录内，共享 Python 环境不变。约定见 [environment/AGENT_CONTRACT.md](environment/AGENT_CONTRACT.md)。

```powershell
python -m pytest -q
python softcut.py check
```

[代码结构](docs/code_structure.md)说明职责；[有限元交接](docs/optical_flow_exchange_export.md)说明导出。旧实验入口、训练/精修链和无用模型已删除，历史实现由 Git 保存。保留测试覆盖当前数学、固定材料身份、截止因果性、输入保护与文件完整性。

[优化方案与部署状态](docs/material_tracking_optimization_plan.md)说明候选模块、验收规则和首轮结果；[文献导读](reference/READING_GUIDE.md)整理可借鉴的论文及其局限。当前基线备份为 `backup/pre-optimization-20261024`。新模块通过 `python softcut.py optimize --check-only` 检查，正式评测必须指定新的结果目录；默认跟踪及 MMT/OBT 不变。

五工况后续配置为 `configs/optimization.boundary.json`：`optimize --stage features` 对照掩膜隔离特征，`--stage surface` 对照三种原始表面模型并检查零深度几何。部署不等于验收；最新进度、负面结果和后续实施顺序见优化方案第 12 节。

四模块的统一入口为 `optimize --stage reliability --config configs/optimization.reliability.json --out results/new_reliability`；可先加 `--check-only` 检查。使用本地固定版本 CoTracker3、TAPNext++，不自动下载或训练。方法选择、论文支持范围、工程阈值和复现说明见 [方法依据链](docs/reliable_pipeline_methods.md)。最新结果和验收状态见优化方案第 13 节。

近表面身份优化通过 `optimize --stage identity --config configs/optimization.identity.json --out results/new_surface_identity` 运行：分开处理边界中心与单侧纹理，用经核验关键帧连接初始身份，并以初始查询运行 TAPNext++。五组初始至 1 mm 的连续前缀已完成，见[视频对照与报告](results/20261025_surface_identity/validated/index.html)和[方法及验收](docs/surface_identity_optimization.md)。本轮改善了可审计支持，网络的刀尖覆盖收益随工况变化，末帧近刀尖支持仍有累计风险，尚未证明物理精度提升；生产默认不替换。
