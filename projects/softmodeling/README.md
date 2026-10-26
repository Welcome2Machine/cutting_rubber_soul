# 橡胶斜切仿真

使用同一套材料模型，研究刀具楔角、圆鼻半径、摩擦/润滑与入侵角度对橡胶压入反力、变形及接触材料位置的影响。0°、33°、57°保留为已有实验对照；当前仅计算完整体，行程上限 **2 mm**，速度固定0.2 mm/s。实验起裂后的曲线不作为完整体拟合依据。

## 本地 HTML 工作台

环境安装后执行：

```powershell
python -m tools.workbench --background --open
```

访问 [http://127.0.0.1:8765](http://127.0.0.1:8765)。页面可选择或导入刀具/材料配置、调整圆鼻/摩擦/角度/网格和预算，核查后启动一个有界任务。结果页支持多工况Fz/Fy/Fyz、接触与能量诊断、保存场云图、PNG及压入动画导出。只启动工作台不会开始有限元计算。

[使用方法与数据格式](docs/workbench.md) · [研究思维导读](docs/research_guide.md)。刀具和材料目录为 `configs/blades.json`、`configs/materials.json`；润滑目前通过指定库仑摩擦系数近似，平倒角、任意刃形及油膜流动未实现。

- [当前研究状态](docs/research_status.md)：完成范围、待解决问题与当前运行入口。
- [任务要求](docs/task_requirements.md) · [模型规格](docs/current_model_spec.md) · [代码地图](prototype/README.md)。
- [数学模型说明](docs/mathematical_model.md)：控制方程、本构演化、接触残差、真实切线及求解/验收算法。
- [共同材料 2 mm 基线](docs/intact_2mm_baseline.md) · [接触位置观测](docs/tip_contact.md)。
- [文献原文与统一导读](references/README.md) · [有效研究历史](docs/research_history.md)。
- [光流数据接口](docs/optical_flow_handoff.md) · [视频—FE共同观测规范](docs/video_fe_comparison.md) · [2023 论文复现](reproductions/dammass2023/README.md)。

当前基线为 P1 平面应变、有限应变等容 Maxwell 与原生 IPC 接触：G∞=0.14 MPa、Gv=1.24 MPa、τ=1.7 s、μ=0.35。已有三角度各 201 态的完整 2 mm 路径，603 态独立重组审查通过；它是可计算基线，尚未完成共同材料实验标定。

可复用结果包括 `outputs/baseline/angle0`、`angle33`、`angle57` 与当前研究候选，论文复现结果保留在 `outputs/paper2023`。实际运行状态见工作台及[研究状态](docs/research_status.md)，不从静态说明推断进程存活。记录参数、接受态、源码指纹和审查，不复制冻结源码树。

输出存储采用“当前场原位保留、停用历史无损归档”：完整基线及balanced/q25保留可直接读取的NPZ，reference57/fast57和旧common_candidate场在本地`outputs/archives/`，原报告/曲线继续可读。需要旧场时按[存储工具说明](tools/README.md#输出存储与历史恢复)恢复指定工况；原数组精度、NPZ字节和审查SHA256保持。

## 环境与验证

使用 Python 3.10，在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests
```

IPC 版本固定为 1.6.0。本地 `.runtime/ipc-python` 可由程序自动发现；PARDISO 为可选运行库，MP4 导出另需 FFmpeg。前端是本地 HTML/CSS/JavaScript，无 Node 构建步骤或网络 CDN。回归测试不启动长研究计算；`python -m unittest discover -s tests` 保留旧核心回归入口，完整检查采用上面的 pytest 命令。计算、队列、审查和后处理命令统一见[工具说明](tools/README.md)。

## 文件分类

| 路径 | 职责 | 发布到 Git |
|---|---|---|
| `prototype/` | 主线有限元、几何、接触、求解器、历史与原生审查；含基线定义 | 代码、说明和基线 JSON |
| `mechanics/` | 主线与论文复现共用的有限应变材料运算 | 代码 |
| `workbench/` | 配置验证、本机服务、标准结果导出；`static/` 为 HTML/CSS/JS 前端 | 代码与静态前端 |
| `configs/` | 经过审阅的刀具和材料目录 | JSON |
| `tools/` | 可复用命令行入口、加载管理、审查与后处理 | 代码和入口说明 |
| `tests/` | 可重复项目回归；围绕方程、历史、输入输出和工作台行为 | 测试代码 |
| `docs/` | 当前要求、模型、状态、研究导读、接口文档；原始实验输入也暂存在此 | 文字文档；原始数据/媒体留本地 |
| `references/` | 原始论文及统一引用导读 | 仅 `README.md`；PDF 留本地 |
| `reproductions/` | 独立论文复现的代码、说明和配置 | 代码、文字、配置；参考原始数据留本地 |
| `outputs/` | 接受态、审查、日志、标准结果包、图和动画 | 全部留本地 |
| `.runtime/`、`.codex/`、缓存目录 | 本地库、个人工具配置与运行缓存 | 全部留本地 |

根目录的 `run_support.py` 是加载入口、审查与工作台共用的运行支持模块，负责子进程选项、可恢复的进程性能策略、原子保存及源码指纹。它有跨目录使用者，保留单一实现；主线方程不放入该文件。根目录其余文件仅保留项目入口说明、依赖和仓库规则。

发布内容不包含已完成曲线、实验 CSV、论文 PDF、视频、检查点或运行库，因此新克隆的仓库不会自带历史结果。文档中的 `outputs/` 链接指向本地证据。需要已有结果时，应另行提供标准结果包或经审查的原始算例目录。

## 命名与维护约定

- 新 Python 模块、工具及普通文档使用简短、表达职责的 `lower_snake_case`；测试为 `test_<功能>.py`。命令行入口放在 `tools/`，可复用方程放在 `prototype/` 或 `mechanics/`，网页代码放在 `workbench/static/`。`README.md`、`AGENTS.md` 等约定文件名保留。
- 配置 ID 使用稳定的型号或 `lower_snake_case`，例如 `99755`、`cr_baseline`；可读名称放在 `label`，来源说明放在 `source_note`。修改数值时保留完整配置和来源，不以显示名称代替物理签名。
- 单位使用 mm、N、s、MPa；新接口字段标明单位。现有 `softcut-config/1`、`softcut-result/1` 字段按[数据格式](docs/workbench.md)保持兼容，例如 `depth_mm`、`fz_N`，不为统一字母大小写随意更名。公开 Fy 对应求解器 x，Fyz 是面内合力大小。
- 手工研究输出采用 `outputs/<研究主题>_<YYYYMMDD>/<工况>/`；工作台使用自动生成的 UTC 时间加唯一编号。机器文件名固定为 `config.json`、`result.json`、`audit.json`、`result.bundle.json`、`curves.csv` 等；具体目录结构以[工作台说明](docs/workbench.md)为准。参数和预算写入配置，不靠冗长目录名编码全部工况。
- 实验原始文件名、试次编号和历史合格算例路径保持不变；不为命名整齐重命名输入或改写原始数据。派生数据另存并记录来源和转换。
- 一个主题保留一份当前规范；进展更新到 `docs/research_status.md`，有效历史收敛到 `docs/research_history.md`，文献导读集中在 `references/README.md`。不新增 `final_v2`、`latest_new` 或多份冻结源码树。
- 仅提交代码、文字文档和审阅后的配置；`.gitignore` 排除数据、媒体、输出、缓存及运行环境。清理前检查活跃进程和依赖；保留可复用基线、必要证据及论文复现结果。一次性探针、重复日志和无用失败输出不进入主线。

完成文件暂存后执行 `python -m tools.check_publish --staged` 检查实际待提交清单；提交后、推送前执行 `python -m tools.check_publish --ref HEAD` 检查可达提交中的发布内容。检查通过不替代回归测试，也不检查未提交的本地结果是否合格。
