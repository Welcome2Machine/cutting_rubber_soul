# 服务器 agent 直接执行提示词（2026-11-17）

你是唯一服务器执行者。本轮研究从已合格的57° A-low、μ=.35、3mm完整体601态出发，先安全清理盘点，再定位1.365mm最低刀尖极点追踪丢失，并重跑全部601态的观测导出/有限元可视化，实际交付可在聊天查看的证据。仅能在证明原物理数值有错误时再算最多一条3mm FE。本地agent不承担任务。

仓库Welcome2Machine/cutting_rubber_soul；服务器路径/LARGE1/gr10578/Zhao_Guorui/SoftCutting/1029；唯一集成分支codex/server-fracture-20261029。原生产SHA 8cd77e7d6db433de723e1d7478df76664a2f2039；公开结果d07c5324315927684fd3b2891aef20fca6e70d51。原601态独立sidecar已通过，原Slurm FAILED和原result audit_failed不可改写。

1. 先盘点origin/HEAD/worktree/index/未提交项/活动作业/源码路径。fetch并完整阅读当前researchtopic/research_prototype_task.md、根及FE AGENTS、docs/git_workflow.md。旧任务短程、B1–B7、F2、断裂探针不再执行；不要重写已完成工作。
2. 审查Git和服务器未跟踪目录占用：.git、runs、outputs、deliverables、worktree、日志与缓存。列出>100MB文件、SHA重复及manifest引用；产生cleanup_inventory.csv/cleanup_actions.json。仅删除可证明安全的无引用临时缓存或经校验的重复文件，保留601份NPZ/父史、result、audit、失败证据、唯一档案及用户数据。禁止rm -rf runs、git clean -fdx、历史重写和强推。报告实际回收字节。
3. 核查现有tip_contact.py中no_tip_contact的条件：固定knife pole到原始材料上表面的最近间隙g_pole必须小于dhat=0.0025mm。以真实1.360/1.365两态证明或否定阈值越界，并分析圆鼻/楔面/刀身几何支撑、可靠原生IPC接触力、局部网格/J。不要把无pole对应认定为全刀脱离或滑移停止。
4. 最小修复观测与报告：旧strict-pole迁移M和NULL缺失保留；新增分部位支撑材料S区间、身份/切换、必要的受力观测，并标明几何近邻不等于压力承载。禁止放大dhat、补零、改摩擦或将刀侧地址强拼到pole M。做解析及原真实态回归。
5. 必须从已有601态重新导出全频率运动学与有限元场，不进行重复FE加载。真实状态取1.340、1.355、1.360、1.365、1.380、1.470、2.000、3.000mm；同坐标范围对比1.360/1.365，出整体图/局部间隙微米放大，标P1网格、材料S、刀具极点、最近投影、dhat、接触部位、局部J。出g_pole(d)阈值图和支撑区间图。缺态不得伪造。
6. 仅若证明原FE物理/历史/网格有严重错误且必须修改生产数值，才允许一条3mm独立从初态重算（20h加载/24h作业，2物理核/8GiB，原1e−4N/CCD/整步弦/物理历史不变），原数据全部保留。只为“追踪缺失”不授权重复3mm计算。
7. 可实际上传的审阅包优先≤50MB，至少包含README_FIRST.md、离线index.html、5张指定真实FE PNG、kinematics_601.csv完整601行、8帧可复绘精简网格状态、review.json和逐文件manifest SHA、README_REPLAY.md、accounting.json及两个清理清单。关键是1.360/1.365真实FE几何与g_pole/dhat的定量证据。验证另一个目录打开/重绘成功。
8. 通过聊天附件或已获授权且实际可下载的交付渠道把审阅包和关键图片/CSV上传并核验。仅回报服务器绝对路径不算上传；权限缺失则明确UPLOAD_BLOCKED、路径、字节、SHA，等待用户提供渠道，不假装上传成功，不把大原始数据放入Git。

全部直接相关测试、根pytest、check_publish和git diff --check完成后，正常发布最小代码与既有research_status，不强推/不动他人暂存。最终回答：Git和服务器实际清理了多少、1.365mm缺失的证据和原因、是不是追踪定义而非FE数值故障、可访问图表及机器数据、为何做或未做FE重算、最终来源/审查/资源与作业终态。直接执行，不停在计划或等待旧agent。
