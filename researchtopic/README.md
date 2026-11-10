# 研究主题与执行入口

更新：2026-11-10。本目录只维护一个当前任务书；旧轮次的“下一优先级”不再作为启动指令。

| 入口 | 用途 |
|---|---|
| [科研原型任务](research_prototype_task.md) | **唯一现行任务**：两篇论文背景、共同模型、两端分工、具体方法、预算与验收 |
| [Git 协作规则](../docs/git_workflow.md) | 分支、文件责任、按提交交接、运行目录与仓库清洁 |
| [服务器启动提示词](server_agent_prompt.md) | 今天唯一执行端：局部修复接缝采样，复用F0、顺序完成F1及有条件F2 |
| [本地启动提示词](baseline_agent_tasks_20261031.md) | 今天暂停新增工作；保留a39ce7成果与既有证据 |
| [实际研究状态](../projects/softmodeling/docs/research_status.md) | 已完成结果、证据位置与尚未完成的阶段 |

服务器集成分支为 `codex/server-fracture-20261029`；本地数据/基线分支为
`codex/intact-baseline-upgrade-20261031`。服务器负责数值核心，本地负责参考输入和实验比较；
负责人维护任务与协作规则。今天仅服务器执行，本地暂停新增计算；任务更新不代表仿真已经完成。

## 原始目标与历史依据

- [研究主题与实验信息](oblique_cutting_experiment_and_research.md)
- [2026-11-02 历史审查](project_review.md)：保留数值事实和复杂度审查，后续决策以当前任务为准。
- [2026-10-31 历史审查](project_review_20261031.md)
- [原始断裂任务](fracture_agent_task_brief.md)

更早的日期任务和轮次文档只用于追溯，不逐个重新执行。新增进展原位更新现有任务/状态入口；
旧版本由 Git 保存，不再新增一份日期任务书、总审查或归档副本。
