# 本地基线 agent 启动提示词

更新：2026-11-09。沿用此文件名仅为保持已有链接；这是当前启动入口。
旧详细任务保存在[固定历史版本](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/bd99be77f312841514b9fddffb541c899cb0001d/researchtopic/baseline_agent_tasks_20261031.md)，不与当前任务叠加执行。

你负责第一篇“实验＋仿真解释”的参数与实验依据、共同参数校准及轻量交接。
第二篇计划研究学习、微分求解器或 PINN；当前保留可追溯的物理输入、历史和结果，不开发学习框架。

先检查真实本地 HEAD、未提交/暂存修改、活动作业和已经完成的拟合。保护这些工作，执行
`git fetch origin codex/server-fracture-20261029`，固定 `git rev-parse FETCH_HEAD` 的 SHA；
通过 `git show <该SHA>:researchtopic/research_prototype_task.md` 读取[完整任务书](research_prototype_task.md)，
并读同一 SHA 的 `docs/git_workflow.md`、根与 FE AGENTS。不要假设单分支 clone 已有服务器 tracking ref。
只为读取任务无需切换正在计算的工作区。你的提交分支保持
`codex/intact-baseline-upgrade-20261031`。

按主任务本地章节交付：核实 13.3168° 完整夹角和半径来源；冻结可用的共同参考输入；
从已有 0°/33°/57° 结果形成力、区域变形及适用窗口对照；整理现有校准和最有区分力的一项偏差诊断。
先用已完成参考参数，不等待新拟合，也不重跑全部六角度 2 mm。现有拟合按原授权继续，
参考参数与新候选分开；不得逐角度改材料、宽度或力倍率，也不得拿未知裂后段强行拟合完整体。

只维护本分支已有的 `configs/intact_baseline_upgrade_20261031.json` 和
`docs/intact_baseline_upgrade_task_20261031.md` 中的当前交接信息；不再创建一份轮次任务或报告。
先确认该 JSON 的交接用途，不把它冒充可直接执行的求解配置。原始数据/状态留在 Git 外，
交接短清单、相对路径、SHA 和必要代表态；没有可用传输通道就准确列缺口，不声称服务器已收到。

服务器负责共同数值核心。发现核心问题时提供最小复现或修改建议，由服务器集成；如需新核心，
在隔离工作区采用其已验证提交，不把整条服务器分支合进活动拟合目录。
按固定 SHA 交接自己的小提交；正常推送，不强推、不混入他人文件。完成后报告文件/提交、
当前可用参数、解释了什么、尚缺什么；不新增通用同步/审计/绘图框架。
