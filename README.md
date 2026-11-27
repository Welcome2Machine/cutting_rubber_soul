# SoftCut Solution · 橡胶斜向压入研究

**当前入口：2026-11-27，京都大学 System B。** 服务器接手无断裂 A/R20 数值可信度与 sliding amount（SA）研究，本地 agent 暂停。活动分支为 `codex/deformation-sliding-baseline`，工作根由用户启动提示词指定，以 `<USER_WORK_ROOT>` 表示。旧分支名、历史轮次和 3 mm 任务不构成本轮指令。

## 新 agent 只需先读这些

1. [唯一任务书](researchtopic/research_prototype_task.md)：研究定义、最新证据、下一步与验收。
2. [当前状态](projects/softmodeling/docs/research_status.md)：已经完成和仍未通过的内容。
3. [System B 环境与运行](docs/environment.md)：现有环境复用、Slurm、资源和日志。
4. 根与 FE 的 `AGENTS.md`、[Git 协作](docs/git_workflow.md)。改公共接口时再读 [接口契约](docs/interface_contract.md)。

可直接复制的入口见 [服务器提示词](researchtopic/server_agent_prompt.md)。统一远程交流为 [Issue #1](https://github.com/Welcome2Machine/cutting_rubber_soul/issues/1)。

## 项目目标与现状

研究完整橡胶在斜面支撑上的竖直压入，先验证力、材料变形、摩擦与接触地址的数值可信度，再分析滑移成因及切割定位。SA 是当前接触位置对应的**原材料表面坐标减初始下刀坐标**，不同于材料位移或累计界面滑程。

目前新 H1 的 33° 到 0.25 mm、101 态；57° 到 0.17984375 mm、78 态。后者下一步的刀鼻—表面穿越已被独立几何核验，尚无安全接受态。系统 H0/H1/H2 空间资格未完成，不能把完成行程或平衡通过当作实验验证。详见当前状态，不在多个入口复制参数表。

## 代码地图

| 路径 | 职责 | 本轮处理 |
|---|---|---|
| `projects/softmodeling/mechanics/` | 材料能量、应力、切线与 Maxwell 更新 | 保持模型；按具体证据修数值实现 |
| `projects/softmodeling/prototype/` | P1、几何、IPC、平衡、历史及 SA | 本轮核心 |
| `projects/softmodeling/tools/` | 单算例、有限队列、验证与导出 | 复用现有入口 |
| `projects/softmodeling/workbench/` | 标准结果与图形显示 | 使用真实接受态 |
| `softcut_solution/` | FE／视频统一界面与接口 | 本轮不扩展平台 |
| `projects/rec_inc_soft_cut/` | 视频固定材料点跟踪 | 保留，当前不重跑 |
| `configs/` | 刀形测量与共享审阅配置 | 保留输入身份 |
| `researchtopic/` | 唯一任务与短入口 | 历史材料不自动执行 |
| `runs/`、各项目输出目录 | 本机日志、状态、证据与调度脚本 | Git 忽略，数据通过 Release 交接 |

Git clone 只包含源码和说明，不包含历史 NPZ、实验原始数据或运行库。当前模型输入由任务书和实际 manifest 对照；CLI 的历史默认值不自动代表本轮物性。

## 运行与验证入口

在仓库根目录，使用已确认的解释器执行：

```bash
python -m pytest -q
python tools/check_publish.py --staged
```

FE 在 `projects/softmodeling` 下运行。长期作业使用计算节点中的 `srun` 和现有队列 `--foreground`，详见 [环境说明](docs/environment.md)。启动、暂停、恢复均为显式动作，打开界面不会启动 FE。

已有桌面入口 `Launch.cmd` / `Start.ps1` / `python -m softcut_solution gui` 及视频比较继续保留。无断裂以外的历史内聚、动力学和断裂工具见 [方法历史](docs/fracture_implementation.md)，本轮不启动。研究原始目录、旧结果和其他用户数据不因维护而删除。
