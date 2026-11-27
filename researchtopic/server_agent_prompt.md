# 服务器 agent 提示词

接手 Welcome2Machine/cutting_rubber_soul 的 codex/deformation-sliding-baseline，在用户启动提示词指定的工作根下工作；本地 agent 已暂停。先读 AGENTS.md、researchtopic/research_prototype_task.md、projects/softmodeling/docs/research_status.md、docs/environment.md 与 docs/git_workflow.md，再核对 Issue #1 最新交接。

复用已有环境，按 Slurm、活动检查点和日志确认起点。迁移与合作暂停/恢复已经实测，直接从当前状态接续。独立诊断 57° 事件附近的路径与力平衡，同时推进 33° 浅窗网格验证。旧试验次数仅作为已发生的记录。保持原物理自由力、连续路径、正 J、真实时间和拒绝回滚要求，保留 SA 的原材料地址和缺测语义。

按照 System B 的 sbatch --rsc + srun 使用计算节点；用实测选择线程和内存，长期队列使用 --foreground。调试通过后启动有限批次，核验真实推进并留下 JobID、日志及查询/暂停/恢复命令即汇报，不持续跟踪。按现有规则正常提交、推送并回写 Issue #1，区分完成、运行、失败和未执行。
