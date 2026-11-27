# 服务器 agent 提示词

接手 Welcome2Machine/cutting_rubber_soul 的 codex/deformation-sliding-baseline，在用户启动提示词指定的工作根下工作；本地 agent 已暂停。先读 AGENTS.md、researchtopic/research_prototype_task.md、projects/softmodeling/docs/research_status.md、docs/environment.md 与 docs/git_workflow.md，再核对 Issue #1 最新交接。

复用现有服务器环境和队列入口。先验证迁移与资源配置，接着修复已证实的 57° 失败增量路径穿越，统一严格准备态与独立 SA 观测，再完成获准的短窗三档网格验收。物理改动须有依据；保留残差、CCD、正 J、真实时间及历史门槛。不要重做已完成基准，不重复搭框架或堆文件。

按照 System B 的 sbatch --rsc + srun 使用计算节点；用实测选择线程和内存，长期队列使用 --foreground。调试通过后启动有限批次，核验真实推进并留下 JobID、日志及查询/暂停/恢复命令即汇报，不持续跟踪。按现有规则正常提交、推送并回写 Issue #1，区分完成、运行、失败和未执行。
