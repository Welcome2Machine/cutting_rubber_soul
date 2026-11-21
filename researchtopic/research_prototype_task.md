# 本地连续完整体七工况与服务器既有证据

2026-11-21。本地负责从各自几何首触连续下行 2.000 mm 的条件基线。由本地有限队列依次执行共同双刃 33°、0°、57°、33° 低摩擦、朝上/朝下两单刃、2 µm 双刃。每次运行使用独立材料、摩擦历史和固定代码身份。服务器此前的 3 mm 证据及其资格见下文。

本地实际工况输入见 projects/softmodeling/configs/intact_2mm_local.json；入口是 tools.run_material_queue 的 --case-config 和已有的 tools.run_material_candidate。共同 G∞=.14 MPa、Gv=1.24 MPa、τ=1.7 s、K=23.34333333333331 MPa、两指数均为 2；平面应变宽度 20 mm、落点 s=7 mm、下行速度 .2 mm/s。上表面向实验室右方下倾，x 指向右，z 向上。single_up 的开刃斜面在左侧（上坡侧），single_down 的斜面在右侧。首触与最低极点分别标识。13.31684633759535° 总夹角和 .24577088241513723 mm 厚度取自 configs/blade_profiles.measured.json。受控单刃形状、两种半径和两种摩擦系数均属假设。原始实验力只用于验证。

Fz 为竖直刀具反力，Fy 对应有符号 reaction_x_n。M 按固定刀具极点到原自由表面的合格材料地址定义，失联保留空值。跨网格比较使用相同参考 S 上插值的固定材料位移与标距。相对切向速度和分段累计量保留近界面运动学资格，局部正压力及真实滑移须有独立证据。真实步长决定 εu=εv Δt，准备段单列。保留 1e−4 N 残差、正 J、连续碰撞与整步弦检查。短程控制不构成全行程网格收敛或实验认证。

加密半宽按旧 A 刀完整体的真实接受态选择。核对旧态及独立审查 SHA 后，以 0.25 mm 间隔抽取至 2 mm，0°、33°、57° 全刀近邻区的参考 S 最大偏距分别为 0.327、0.395、0.302 mm；最小 J 单元中心最大偏距分别为 0.013、0.032、0.162 mm。本批取 ±0.8 mm，并在每个合格态复核近邻区是否越界。此检查不赋予新刀形或 2 µm 锐尖全程精度资格。原始逐态比较记录在本地批次父目录 historical_band.json。

在批次目录下运行以下只读命令可查看七个工况的实际状态。

```powershell
(Get-Content -Raw queue_status.json | ConvertFrom-Json).jobs | Select-Object case_id,status,solver_pid,qualified_depth_mm
```

如需手动停止，先在同一批次目录中读取启动回执，仅当进程命令行、创建时间和绝对目录全部符合时终止该有限进程树。手动终止可能留下尚未重审的前缀，后续恢复须由同版本单工况入口通过独立审查。

```powershell
$batchPath=(Get-Location).Path
$receipt=Get-Content -Raw launch.json | ConvertFrom-Json
$worker=Get-CimInstance Win32_Process -Filter ("ProcessId="+$receipt.worker_pid)
if ($worker -and $worker.CommandLine.Contains($batchPath) -and $worker.CommandLine.Contains('tools.run_material_queue') -and $worker.CommandLine.Contains('--worker') -and [Math]::Abs(($worker.CreationDate.ToUniversalTime()-([datetimeoffset]$receipt.started_utc).UtcDateTime).TotalSeconds) -lt 10) {
    taskkill /PID $worker.ProcessId /T /F
}
```

## 服务器历史任务：安全清理盘点、刀尖追踪缺失机理与可核验证据交付

更新：2026-11-17。当时的服务器工作范围如下。 本任务替代旧短程B1–B7和长程启动指示；不重启F2、断裂旧队列。上一完整任务的固定历史：[d07c532版任务](https://github.com/Welcome2Machine/cutting_rubber_soul/blob/d07c5324315927684fd3b2891aef20fca6e70d51/researchtopic/research_prototype_task.md)。当前已完成事实见[研究状态](../projects/softmodeling/docs/research_status.md)。

## 1. 研究目标与冻结证据

研究57°斜切下的刀尖材料坐标迁移、固定材料点变形、刀具接触部位转移及相对切向运动，为第一篇实验＋FE机理论文提供证据。“早期迁移趋稳”、“不同刀具起裂深度不同但最终入口偏移相近”是需要验证的假说，不是拟合目标。连续完整体不能预测实际断裂后的行为。

原数值生产提交 8cd77e7d6db433de723e1d7478df76664a2f2039；57° A-low、μ=.35、s=7mm、v=.2mm/s、dhat=.0025mm、固定εu=1e−5mm、假设刀鼻半径.020mm、平面应变P1 Maxwell+IPC；目标3.0mm。
原最终result SHA256 = 33a93700e4938f661f34d77b8800026df3fdcc6e3487f15a3da19c1aeac1f9c1。
已保存601态并通过另外运行的完整独立重审；保留原Slurm FAILED及原result audit_failed标签，新sidecar是数值资格来源。公开科学结果截至d07c532；原始证据位于服务器 /LARGE1/gr10578/Zhao_Guorui/SoftCutting/1029/runs/baseline_migration_long/。

固定极点追踪的no_tip_contact来自**刀具最低极点到原始材料上表面最近投影距离未满足g<dhat**，不等于刀具整体没有接触。1.360mm最后唯一极点对应，(1.360,1.365]变成缺失，1.47mm左刀身出现近邻支持，3mm整刃几何邻近长度约.915mm。不能将缺失填成平台或把刀侧材料地址接到原极点曲线。

## 2. 先做安全清理盘点

开工核对origin/HEAD/分支/index/未提交变更、活动作业与真实导入位置、已有证据和备份，保持用户工作不动；干净且无活动作业才快进，否则隔离worktree。唯一写入者与[Git规则](../docs/git_workflow.md)保持一致。

远程Git树已确认约494个已跟踪文件、5.62MB源文件、无NPZ/视频/压缩包；最大历史断裂文档约294KB/3219行，是有出处的科学历史，不因较大就删除。旧任务入口因叠加旧轮次而臃肿，本次已缩成现行任务，旧文由固定Git提交保留。

仍须**在服务器现场**审计.git、worktree、runs、outputs、历史deliverables、slurm日志与缓存：字节数、>100MB文件、SHA相同副本、源manifest/README引用、是否唯一副本、活动作业占用。交cleanup_inventory.csv及cleanup_actions.json，逐项列实际删除/归档/保留与回收字节。先列清单后操作；对不能确认安全的条目只给建议，不删除。

必须保留601态NPZ（约2.6GiB）及父史、原result/END、独立audit sidecar、原始失败现场、已有证据包唯一副本、实验原始数据、冻结生产树与其他用户文件。禁止git clean -fdx、rm -rf runs/outputs、历史重写/force push、清理他人暂存、删除活动作业文件。仅可清理明确可再生、无引用且无作业占用的临时缓存/重复导出或经双重SHA和副本可用性确认的冗余包；使用既有归档规则，不破坏重放。

## 3. 解释并最小修复刀尖极点丢失

先**只读**利用601个已认证真实态，逐深度计算并存档：
- 刀具极点坐标、最近原顶面边和同一材料地址S、投影点、最短g_pole、dhat、状态/缺失原因；
- 圆鼻、左右楔面、左右刀身和整刃几何邻近的参考材料S区间及当前长度，区间并集去重；
- 若已有原生IPC API可可靠独立重组，则另列真实承载法向/切向接触分量和受力支持位置；否则明确pressure_unknown。几何距离<dhat绝不自动等于有压力；
- 变形P1三角形、固定材料点u_s/u_n、局部J、最小J位置、刀具接触位置。

优先真实状态深度：1.340、1.355、**1.360、1.365**、1.380、1.470、2.000、3.000mm；若确无这些深度，列实际相邻态，禁止插值伪造已接受状态。重点证明1.360→1.365属于（a）极点间隙越阈、（b）承载转向其他鼻面/刀侧、（c）局部畸变或阈值敏感，或无法判定；支持多种同时发生。

修复对象主要是已有prototype/tip_contact.py、contact_trace.py和现有FE图表工具。原strict-pole M和no_tip_contact原样保留；**增设而非替代**分部位材料支撑区间/可验证的承载追踪。若要用刀侧材料地址作为新指标，必须有新字段、来源及切换规则，不准假装原刀尖极点M继续存在。
不为了画平滑曲线增大dhat、修改摩擦/接触、强制选最近点、跨缺口填值或删除接触约束。不是程序bug则保留no_tip_contact，修复解释与互补观测而非强行“有效化”。
增加轻量测试：最低点无支持而刀侧有支持、dhat跨越、网格相邻边迁移、多分离段支持、共同刚体输运；实时与离线同态结果一致。

## 4. 重跑观测与有限元图，不先重跑FE

**必须做**：在不改601份原始NPZ及源result的前提下，用新后处理在完整601态上重导出并通过原审查sidecar和哈希进行资格绑定。展示的是实际FE网格/变形/刀具，而不是示意图。全部曲线按真实深度与物理时间画；缺失处断线、NULL、不做外推。

生成可比较的整体截面图和局部/微米间隙放大图；在1.360/1.365使用同一坐标轴限和色标，保持equal aspect。叠加原顶面材料S着色、刀具极点、最近投影连线、g_pole/dhat、圆鼻/刀侧邻近区域、承载约束（若可信）、局部J场。另画g_pole随深度相对dhat的阈值线、各刀具部位支持材料S区间、固定材料变形与Fz。

**仅在有明确证据证明原物理/求解/网格状态错误、单纯后处理不能修复时**，才授权最多一条独立FE重算：同57° A-low μ=.35、总深度3.0mm、从新一致初态开始、独立目录、原601态不覆盖。先提交错误复现、最小物理修复与回归。加载≤20h、作业≤24h，2物理核/8GiB、IPC1/BLAS1、SysB sbatch→srun/no-requeue，保留1e−4N、正J、CCD/整步弦、真实dt及内部历史；不调物性、μ、网格、刀鼻、dhat或求解门“修复”观测。仅观察器/画图缺口**不授权重复算3mm**。失败保留原因并停止，不额外重投。

## 5. 必须上传可访问、可复查的证据

优先在既有忽略结果父目录新建唯一子目录contact_loss_review，不覆盖其他报告。交付最小内容：

- README_FIRST.md：五句话说明数值合格身份、1.365缺失实际原因、是否有刀侧承载、还不能判定什么。
- 离线可打开的index.html，及fig01_global.png、fig02_transition_1360_1365.png、fig03_pole_gap_dhat.png、fig04_nose_flank_support.png、fig05_J_hotspot.png。单位mm/μm、坐标和实际时刻完整，不能以示意图冒充FE。
- kinematics_601.csv：601个真实态每行有源index、深度、真实dt/时间、Fz、极点g/status/M（缺失为NULL）、分部位S范围、材料变形/J与观测资格；任何接触压力字段若不能独立验证则NULL并标明。
- keyframes：8个关键真态的精简可复绘网格/刀具/材料S数据，至少1.360及1.365，用小NPZ/机器数组；保留原始state的字节SHA和抽取方式，不打包原2.6GiB全链。
- review.json、manifest.json：逐文件SHA256、原生产和后处理提交、原result和sidecar审查SHA、坐标系/单位/阈值、实际重放命令、失败和缺失解释；README_REPLAY.md、accounting.json、cleanup_inventory.csv、cleanup_actions.json。

交付前验证601行/原状态哈希、图与真实state一致、HTML脱离服务器相对路径能打开、关键图在干净输出目录可重放、每文件hash吻合。审阅包尽量≤50MB，超出则说明原因并拆分关键图片/机器数据包；原601NPZ仍保存在服务器。

**上传是硬验收**：实际把审阅包和至少关键PNG/CSV交给用户可访问的聊天附件或现成获授权共享渠道，确认可下载、解压、校验。仅提供/LARGE1服务器路径或未托管HTML不算上传；若当前工具或权限无法上传，明确标记UPLOAD_BLOCKED并报告文件绝对路径、字节和SHA，等待用户提供可用渠道，绝不声称已上传。不得因上传受阻把大原始NPZ/凭据强行加到Git。

## 6. 仓库与完成标准

Git只接收现有模块的最小修复、直接回归和既有状态文档更新；全部生成数据留Git外。不加新求解框架、自动监控、重复runner或round/v2/final同职责文档。完整科研方法和实验信息见旧固定任务与research_status，不因清理而删除历史文献/失败证据。

运行直接相关测试、根pytest、check_publish、git diff --check；对server分支仅正常提交/推送，不强推，不收他人index。最终报告三层资格：代码和资料完整性、原3mm数值601态资格、承载滑移和实验解释资格；列明实际清理字节、是否新算FE和原因、图/CSV的实际上传状态、可复查的文件hash、资源和作业终态。完成后停止，不自动延长到其他刀具或新增实验。
