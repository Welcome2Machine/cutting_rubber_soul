# 工具入口

当前只有一套原生 P1 / IPC / 有限应变 Maxwell 计算主线。目标与参数见
[模型规格](../docs/current_model_spec.md)，研究结论见[研究状态](../docs/research_status.md)。
不运行历史诊断，不复制源码树。每个新结果记录当前源码、参数和检查点指纹；运行期间不得编辑参与计算的源码。

| 工具 | 用途 |
|---|---|
| `workbench.py` | 本机 HTML 调参、单任务运行、标准结果包/CSV/MP4 导出及多曲线/保存场比较 |
| `run_material_candidate.py` | 单角度完整体计算，准确材料/摩擦历史、刀尖材料身份与独立物理审查 |
| `run_material_queue.py` | 同一材料的独立初态串行队列；各组成功或失败后继续下一组，HTML 自动记录 |
| `qualify_intact_baseline.py` | 校验保留的三角度 2 mm 基线报告及检查点；可用当前原方程逐态重新审查 |
| `plot_experiment_angles.py` | 显式指定实验与 FE 路径，输出叠图、共同区间 CSV、分段斜率及 HTML |
| `visualize_ipc.py` | 接受状态的原始变形网格、完整三维 Cauchy 最大主拉应力和刀尖接触偏移视频 |
| `render_solver_dashboard.py` | 任意上述队列的只读 HTML 快照、日志与有界后台更新 |
| `check_publish.py` | 检查暂存区或提交历史的发布白名单；只允许代码、文字文档、已批准配置与工作台静态源码 |
| `manage_outputs.py` | 统计输出空间，将停用NPZ历史按内容去重、无损归档及按需恢复；保留原始字节和审查SHA256 |
| `run_cohesive_cut.py` | 显式有界断裂加载；旧径向律、混合模式预设接缝及全内部边自由选路网络分开配置，保留失败前缀与独立历史审查 |
| `verify_cohesive_mixed.py` | 纯模式、非比例历史、转动客观性、切线和功收敛；独立示例参数，不读取实验力 |
| `verify_cohesive_network.py` | 四三角形自由选路平衡验证；镜像加载与步长减半，含全IPC/CCD，但拉伸算例不验证实际闭合摩擦 |

## 断裂收敛与有限间隙验证

显式版本3配置 `configs/fracture_network_regularized.json` 是独立数值研究示例，采用有界可恢复压缩支路、固定准备净空和PTC方向回溯；它没有材料预测或完整切割资格，不会在工作台启动时运行。版本1/2与完整体默认行为保留。新配置仍需通过自身范围的残量、历史、接触和能量验证。

此配置保留已完成并审查的0.05 mm短程基准设置：g=0.00006 mm、dhat=0.000025 mm、名义步长0.005 mm、Kn=Kc=5000。它不代表网格/间隙独立或2 mm全程通过；更小间隙及高刚度的算例只按各自报告范围使用。UI原默认配置未被此研究配置替换。

| 工具 | 使用范围 |
|---|---|
| `verify_network_compliance.py` | 27个无损伤自由侧边拉/压小试样，分离网格、gap、界面刚度及联合极限；记录真实IPC接触数，不能替代刀切 |
| `verify_cutting_compliance.py` | 原圆鼻刀具下完整体/无损伤网络配对，刀尖网格三档、相同材料与真实时间，量化网络替代的切前反力偏差 |
| `verify_network_sensitivity.py` | 显式清单执行或只读比较；要求同物性、同准备与同目标，核对源/状态/审查，不外推未完成段 |
| `audit_network_pairing.py` | 只读检查完整失效原配对面的真实有限段距离及残余虚拟压缩力/力偶/能量；不求解FE、不把内部力范数当刀具载荷份额 |

```powershell
python -m tools.verify_network_compliance --output outputs/new_compliance --seconds 15 --steps 8 --strain .0002 --compression-regularization gap_bounded
python -m tools.verify_cutting_compliance --output outputs/new_blade_compliance --seconds 240 --gap-mm 6e-5 --dhat-mm 2.5e-5 --ccd-tolerance-mm 1e-9 --normal-stiffness 5000 --compression-stiffness 5000 --ptc-shift-policy backtrack_aware
python -m tools.verify_network_sensitivity runstudy --manifest <明确清单.json> --output <新目录> --python <解释器路径>
python -m tools.verify_network_sensitivity compare --manifest <已执行manifest.json> --output <新比较目录>
python -m tools.audit_network_pairing <已结束result目录> --output <新审计.json>
```

`runstudy` 才启动新加载，各案有软预算和额外60秒硬启动/收尾余量；`compare`与配对审计仅后处理。结果/源码在读取期间变化或指纹不符会降级，缺失不补猜。三档间隙相差小也不证明网格独立；完整的网格选路还需方向、接触、时间尺度及裂后共同区间检验。当前证据与限制见[研究状态](../docs/research_status.md)和总仓库[实现说明第16节](../../../docs/fracture_implementation.md)。

刀具柔度工具显式启动六个独立初态子进程，每案预算由`--seconds`给定，另有10秒硬收尾余量。默认目标仅0.01 mm，损伤关闭；表面网格0.02/0.01/0.005 mm、细化半宽0.2 mm。每对核对原完整网格、准备位置和零深度刀具的指纹，并要求实际接受深度/Maxwell时间完全一致。提高Kn/Kc是数值刚度对照，不拟合切割力，也不自动认证裂后接触。

两者共享内缩前的网格与外边界支撑线，实际材料域并不完全相同。反力差包含内缩面积损失、内聚柔度、内部IPC/摩擦及有界压缩的综合影响，不能直接当作单独界面模量误差；Kn/Kc对照只检验综合反力差对数值刚度的敏感性。

工具使用十倍力残量门（0.001 N）作为声明的反力比较筛查尺度，不把近零分母提升为可靠相对误差；它不是经条件数分析得到的严格反力误差上界。局部刀尖细化也不能替代远场、网格方向、物理时间或裂后区间的收敛验证。

保存态之间的额外几何审查允许将**同一直线路径**分片；每片仍须由原完整CCD/J以原容差和保守系数证明安全，连续覆盖[0,1]才通过。原整段CCD分数保留；不能将小于1的分数直接改成通过。最多64片、无进展或未覆盖均为未通过。分片只是几何后处理，不添加物理时间、不改变材料/断裂历史。

## HTML 工作台

```powershell
python -m tools.workbench --background --open
```

默认打开 `http://127.0.0.1:8765/`，仅本机访问。核查配置不启动加载，点击启动后才从独立初态计算；一次一个计算任务，固定速度 0.2 mm/s、行程不超过 2 mm。刀具/材料可选择或导入 JSON，油润滑目前只以给定正 μ 的库仑摩擦近似。默认材料为共同基线，标准网格远场为 1 mm；完整物理验收保持锁定。

结果页支持多个标准包叠图/隐藏、Fz/Fy/Fyz、接触/能量诊断、真实比例变形动画及原始单元应力/应变/J 云图。Fy 对应旧 `reaction_x_n`；标准包对未合格历史留空，不拼接后续孤立通过态。完整使用方式、配置/结果字段和已知边界见[工作台说明](../docs/workbench.md)，研究依据见[研究导读](../docs/research_guide.md)。

已有结果的独立导出只做后处理，不启动 FE：

```powershell
python -m tools.workbench --export outputs/baseline/angle57 --output outputs/workbench/baseline57_view --frames 11 --movie
```

## 发布检查

暂存明确的代码、文档和配置后，执行 `python -m tools.check_publish --staged`，核对本次将提交的精确文件。提交后执行 `python -m tools.check_publish --ref HEAD`，检查全部可达历史。白名单允许两个配置目录文件及三个工作台静态源文件，不因支持JSON/HTML而放行运行配置、结果或报告。检查同时拒绝二进制内容及未合并项；它不替代源码、机密信息和科学结论审阅。

原始实验CSV、论文PDF、图像/视频、结果、状态文件、日志和运行库留在本地；即使被`.gitignore`排除，也应审阅`git diff --cached --stat`后再推送。文件职责与命名规范见[仓库首页](../README.md)。

生成 `result.bundle.json`、`curves.csv` 及可选 `indentation.mp4`。导出复用已有审查并核对选中状态的指纹/J/储能，不替代新鲜全路径审查；视频编码失败单列于包内，JSON/CSV 仍保留。原结果目录不改写。

## 输出存储与历史恢复

计算中的接受态继续使用原压缩NPZ和float64历史，不在加载时做跨文件压缩。完整基线、论文复现及当前balanced/q25场保留原位；停用reference57、fast57和旧common_candidate的NPZ历史归档到本地`outputs/archives/`。原结果报告、审查JSON、曲线及分析HTML留在原目录。归档不是删除历史，也不改变路径资格；q50的57°仍只连续合格至1.25 mm。

```powershell
python -m tools.manage_outputs inventory
python -m tools.manage_outputs verify --manifest outputs/archives/reference57.tar.zst.manifest.json
python -m tools.manage_outputs restore --manifest outputs/archives/reference57.tar.zst.manifest.json --prefix outputs/angle57_priority_20261023/reference57/03_angle33
```

恢复按需选择一个工况；省略`--prefix`恢复该归档中的全部历史。恢复校验原始NPZ字节，并拒绝覆盖不同内容的文件。只有恢复实际需要的场后，才运行可视化、全路径审查或满足原一致性门槛的续算。重复字节可共享硬链接；保存入口通过原子替换写状态，不原位修改旧检查点。

归档格式为`softcut-output-archive/1`：清单记录每个原路径、长度和SHA256；压缩流按SHA256只存一次完整NPZ对象。Zstandard长窗口可继续压缩相邻状态的相同网格和双历史字节。`.tar.zst`需要可用的`zstd`命令；`.tar.xz`使用Python标准库，无额外依赖，但本机抽样压缩明显更慢。

其他已停用任务可显式归档：

```powershell
python -m tools.manage_outputs archive --source outputs/retired_study --output outputs/archives/retired_study.tar.zst --prune
```

只有指定目录中的NPZ进入归档；必须先停止依赖它们的计算和后处理。工具检查已知活动消费者及工作台任务，拒绝处理完整基线/论文复现。`--prune`仅在完整归档校验、所有源文件再次核对后移除原文件；省略它则仅创建归档副本，不释放原文件空间。目录中的`storage.archive.json`指出恢复清单。未经恢复的旧场不能宣称可直接重放；原报告中的源指纹也不等于当前源码可直接续算。

## 共同基线计算

以下是命令示例，不表示任务已经启动。三组均从独立初态开始，竖直刀具行程上限为 2 mm。

```powershell
python -m tools.run_material_queue --output outputs/new_angles --angles 0 33 57 --depth 2 --seconds 21600 --material-batch-size 2048 --linear-backend pardiso --linear-threads 4 --threads 2
```

默认共同参数为 G∞=0.14、Gv=1.24、K=23.34333333333331 MPa，τ=1.7 s、μ=0.35、速度 0.2 mm/s、名义步长 0.01 mm。
材料分批和 PARDISO 为显式选择，不改变物理验收；PARDISO 需要本地可用运行库。
没有该运行库时省略性能开关，采用原生求解后端。
`--prepare-only` 仅生成运行配置；`--launch-prepared --output <目录>` 启动一次已经准备的队列。
队列每组保留原求解预算与 1800 s 准备/审查余量，不自动重跑失败工况。
`index.html`、`queue_status.json` 与各组 `run.log` 记录实际进度；退出码或完成标签不能替代整条路径复核。

```powershell
python -m tools.run_material_candidate --output outputs/new57 --angle 57 --equilibrium-shear .14 --maxwell-shear 1.24 --bulk-modulus 23.34333333333331 --relaxation-s 1.7 --loading-rate .2 --depth 2 --seconds 21600 --material-batch-size 2048 --linear-backend pardiso --linear-threads 4 --threads 2
```

单工况 `--resume <自己的结果目录>` 要求完全匹配的物理签名、性能策略和物理源码指纹；
整理前的基线可用当前方程重新审查，但因源码集合变化，不能冒充同源续算。
队列默认建立独立初态。显式的`--resume-case 57 <自己的结果目录>`仅转接该角度已经通过审查的接受前缀；准备时核查物理签名、源码、性能策略、名义增量和全部检查点指纹，运行时再次独立重组整条历史。不能跨角度、材料或刀具离散续算，失败试探不作为恢复源。

`--diagnostics` 在原日志记录实际PTC方向、预测/实际残差、线性解误差、J/CCD/位移上限及阶段耗时；失败时另存准确尝试双历史和最后配对方向。`prototype.material_case.probe_ptc_state`可用保存方向重组条件探针，调用后恢复原问题的双历史，不推进加载。

`--blade-spacing .005`仅细分直刃和近端刀身碰撞边，不改变圆鼻、原轮廓或橡胶网格；它改变接触积分离散，须从独立初态开始，并检查分段收敛。默认不启用。刀尖原顶点编号保留，新增点不可当作新的刀尖。

队列可重复传入`--experiment-case 0 <CSV>`等各角度实验源。输入单位/速度与指纹先核查，每组只取得自己的源；队列结束后自动生成`experiment_comparison/index.html`，展示实际接受前缀、固定深度、割线和有符号角度差。未完成窗口不外推。

实验目标按当前约定先减去各自第一个Fz值，使ground truth从0开始。0°/33°原始文件已经为0；57°原文件初值为.51 N，另存减去.51 N的派生CSV并记录来源、偏置及指纹，再通过`--experiment-case 57 <派生CSV>`传入。读取器不自动猜测或重复扣除偏置，原始CSV和深度保留。

## 审查与后处理

```powershell
python -m tools.qualify_intact_baseline --output outputs/baseline_audit
python -m tools.qualify_intact_baseline --output outputs/baseline_fresh_audit --fresh
python -m tools.plot_experiment_angles --output outputs/comparison --case 0 outputs/baseline/angle0 docs/1008n0a2__Fz__curve_1.csv --case 33 outputs/baseline/angle33 docs/1008n33a1__Fz__curve_2.csv --case 57 outputs/baseline/angle57 docs/1014n57a1__Fz__curve_0.csv
python -m tools.visualize_ipc --source outputs/baseline/angle57 --output outputs/baseline_visualization
python -m tools.render_solver_dashboard --study outputs/new_angles
```

基线审查不推进 FE。已删除原冻结源码树，保留的来源指纹不等于原引擎可执行备份。
实验比较保留原接触零点，不缩放力，不平移深度；只对已有共同区间插值，不外推失败前缀。
未确认起裂深度时，后段差值不能直接归为完整体模型误差。
可视化需要 FFmpeg；Maxwell 应力从每个状态的准确上一材料度量和真实时间步重组，不能用弹性模型代算。
刀尖接触只读取明确接触身份；缺失或多解保留为空，不用最近点补造。

### 前段机制诊断与有限面闭合研究配置

`verify_cutting_compliance --front-diagnostic`允许显式无损伤诊断到0.5 mm（默认入口仍最多0.05 mm），使用同样的独立初态、时间历史、预算、残量及完整CCD/J审查。隐藏worker入口的case JSON可指定`configuration.diagnostic_depths_mm`，必须严格从0递增至目标；用于重放已有接受深度日程，**不读取或移植其材料历史**。报告核对实际深度/dt，新增自适应子步时不能声称日程相同。关闭损伤的对照不构成实验未裂标签。

`configs/fracture_network_finite_faces.json`显式启用双侧有限面距离的闭合势；用标准`run_cohesive_cut --config ... --output <新路径> --seconds <预算>`运行。新旧闭合模式检查点不互换，UI启动不运行或迁移模型。配对后处理`audit_network_pairing <result路径> --output <新JSON>`分别报告超出IPC激活距离与超出参考闭合支持距离的承压配对；有限正间隙补偿仍不等于物理接触收敛。

2026-10-28实际前段对照、数值修复和未完成事项以[研究状态](../docs/research_status.md)及[本地HTML](../../../runs/front_diagnosis_20261028/index.html)为准。修复版只审查到0.2725 mm，不能外推其后的反力或裂纹。

### 零间隙、方向与失稳能量专项验证

- `python -m tools.verify_fracture_resolution --output <新目录> --seconds 90`：零间隙连接/释放算子、三尺度/多种子候选边路径几何偏差、六时间步/两阻尼标量惯性断裂；不读取实验、不启动刀具FE，局部资格与生产切割资格分列。
- `python -m tools.audit_fracture_energy <已有结果目录> --output <新JSON>`：认证既有独立审查绑定的接受记录与检查点，重算逐步能量，额外检查Σ|R|以防抵消。不重算FE，不把能量差额改名为动能或耗散。
- `tools.run_cohesive_cut`的版本3可显式选`nonlinear_solver="newton_lm"`作研究诊断；默认仍`ptc`。真实残差最小化可能停滞于非零值，不是已验证的全程收敛修复。输入、预算、失败和审查照常记录。
- 独立材料试验的待填数据字段与工作量见`configs/independent_fracture_test_plan.json`。它是计划，不是运行配置或已标定物性。

2026-10-29离散功研究支路沿用`tools.run_zero_gap_cut`的显式新目录/预算入口，配置示例
`configs/zero_gap_discrete_work.json`。`time_integrator=discrete_gradient`并不表示整条路径
已通过能量或时间验证；算法Fz为区间平均，theta额外耗散和Maxwell物理耗散必须分开。
`barrier_quadrature_rule=lobatto`包含步末屏障；`gauss`保留用于重算已发现的内部求积反例。
`insertion_rule=nonlocal_intact`另需独立长度，邻域只覆盖未释放连接；旧`nonlocal_tensile`
将已释放点的零约束残量计入，保留为负对照。`connected_patch`在本轮2 mm未释放，
不可当作修复后裂后结果。实际证据见[本轮报告](../../../runs/fracture_repair_20261029/index.html)。
`blade_surface_spacing_mm`显式设置既有保形直边细分，示例配置采用.05 mm，theta=.5。
此文件仅是待验证配置；不表示这组空间/时间离散已经收敛。

### 实际刀具零间隙二维动力学控制

在FE项目目录显式执行：

```powershell
python -m tools.run_zero_gap_cut --config configs/zero_gap_dynamic_energy.json --out ../../runs/<唯一新目录> --seconds 1800
python -m tools.audit_zero_gap_cut ../../runs/<已有目录> --output ../../runs/<已有目录>/audit-new.json --seconds 600
```

第一条从初态启动有界加载，第二条只读取保存态并重新组装，不推进加载、不覆盖旧审查。
运行中不编辑数值源码；记录解释器、源码/配置/状态哈希。预算不足即停止，不自动续算。
`zero_gap_dynamic_energy.json`采用割线卸载、机械储能归一化的数值能量门，仍是未标定
研究配置，不能把小局部误差等同于失稳事件收敛。`zero_gap_dynamic_control.json`保留为
最初的基础控制配置。运行结果须分别检查末行程、独立审查、接受态间CCD、能量和时间
敏感性；30×5×20 mm居中0°镜像对称模型不代表原偏心/斜切或自由裂路已实现。

### 有限裂增量双条件与证据审查

`configs/zero_gap_finite_increment.json`为待验证研究输入，不是标定参数。运行入口仍是
`python -m tools.run_zero_gap_cut --config <配置> --out <新目录> --seconds <预算>`。
`fracture_criterion=finite_increment`要求同段强度与冻结历史能量减摩擦条件同时通过。

- `python -m tools.audit_finite_increment <结果目录> --output <新JSON> --seconds 180`：重组已接受有限裂段的虚拟前后场，核对强度、能量、CCD、独立分区复合Legendre积分和真实种子路径能量支持（旧包的17点历史算术单独标记）；不加载。
- `python -m tools.report_finite_increment <结果目录1> <结果目录2> ... --output <新报告目录>`：只读保存场，输出Fz/能量/时间步、裂口与表面连通性及统计。丢失后缀不插补，不自动给收敛资格。
- `python -m tools.continue_zero_gap_cut <已审查父目录> --out <新目录> --seconds <预算>`：显式同配置、同源码续算；先全路径及双条件重算，原样复制接受前缀，恢复精确物理历史，并记录父结果哈希。源码/物性不匹配则拒绝；不是UI自动重启或队列恢复。
- `python -m tools.recover_finite_increment <停止目录> --log <原日志> --seconds 240`：只恢复日志确认且完整写出的接受态；必须有停止原因、停止时源码哈希及原manifest。独立重算后状态仍为`interrupted_recovered_prefix`，丢失的拒绝候选/迭代次数保持未知，不冒充正常完成。

本轮[真实刀具结果](../../../runs/finite_increment_20261029/index.html)中，五项最终
断裂候选未完成2 mm，仍有末端摩擦瞬态和内部孤立开口。报告保留完整/不完整事件窗口，
以全部接受深度并集计算图示曲线差异，避免均匀采样漏掉窄脉冲；小L2不等于瞬态收敛。
只有关闭损伤对照完成2 mm，不能计为裂后修复。参数仍是未测示例。

第二轮数值求根、积分和性能诊断支持显式`--ipc-threads`、`--blas-threads`及可选`--physical-core-affinity`，
兼容默认2/2，服务器推荐按实测采用IPC 2/BLAS 1；先满足进程线程预算与亲和性。
加载、两类独立审查、失败诊断、恢复与同源码续算均记录实际线程/分配。线程数不改变物性。
精确有理数相交检查及只读绘图报告没有IPC求解循环，不作为多线程性能测量。

- `tools.benchmark_frozen_scene`：核查归档后做固定工作量装配、CCD、摩擦积分、稀疏解及I/O测试，记录实际SuperLU分派。
- `tools.diagnose_virtual_root`、`tools.audit_virtual_chord`：原力分量/方向/条件性及精确有理数相交见证；不把一次失败写作无根。
- `tools.diagnose_virtual_work`、`tools.validate_virtual_reliability`：原直线的功率尖峰、全局点数和分区独立方法，以及面积控制。
- `tools.diagnose_surface_competition`、`tools.report_surface_competition`：核查表面/内部候选的能量与开口；支撑并集面积不是替换Gc代价的规则。
- `tools.audit_virtual_root_accuracy`：将已接受事件的原方程力残差进一步收紧到1e-8 N，检查判据对端点精度的敏感性；不推进真实历史。
- `tools.trace_virtual_load_branch`：原种子上的辅助载荷求根跟踪，接受能量前必须撤去辅助载荷；保存失败阶段及原方程残差，不能作为物理续算。
- `tools.diagnose_constrained_root`：比较完整弦搜索目标对根、CCD及积分的影响；搜索目标属于数值方向控制，原1e-4 N门不变。

旧现场诊断明确标为`equation_compatible_diagnostic_only`，核对Git生产源、输入、
原物理方程、种子增量和冻结历史；它不能绕过物理续算的精确源码门。生成结果留新目录。

## 真实状态的条件动态释放

`dynamic_release.py`只用于已审查父现场的独立候选过程，不是生产续算或新的起裂规则。
父目录需要原`result.json`、失败试步、最后接受态，以及绑定相同result哈希的
`.path_audit.json`和`.finite_audit.json`侧车。`--seed accepted`精确继承最后已接受态；
`--seed trial`使用重新核查的未决真实试步；两组输出不得拼接或混作同一初值。
候选48条件激活后，原惯性/Maxwell/已有损伤/刀速均保持，不再选择其他新候选。
`--no-release-control`仅保持48的连接，**不是关闭损伤**。所有输出必须使用新目录。

```bash
# 在计算节点执行；服务器通过本站sbatch --rsc分配并由脚本内srun启动。
python -m tools.dynamic_release run <父现场> --out <新目录> --seed accepted --dt 6.25e-8 --duration 5e-5 --seconds 2400 --ipc-threads 2 --blas-threads 1 --physical-core-affinity
python -m tools.dynamic_release audit <条件结果目录> --out <新审查.json> --seconds 540 --ipc-threads 2 --blas-threads 1 --physical-core-affinity
```

加载≤3600 s、审查≤600 s；失败按既有计算/物理类别处理，未知保存停止，只有真实
非线性或能量分辨率失败回退dt。程序记录实际dt、完整分账、状态/源哈希和运行身份。
内置独立GL3/7/15给出接触时间积分的经验误差；未收敛不得用其数值认证。第四轮
另以已有`virtual_work_audit.audit_work`的几何分区GL8/16/32审查同一真实接受弦，
它与生产中点摩擦不同；积分收敛也不等于整个动态轨迹对dt收敛。


第五轮条件时间对照增加`--integrator midpoint|endpoint_be|endpoint_friction`、
`--accepted-index`与`--minimum-dt`。已接受索引从完整审查父路径读取，不篡改
原历史。条件续算只接受同源码、已完成窗口及绑定结果哈希的力/历史与独立接触功
双审查；条件后缀不成为生产路径。最小dt是停止界限，不放宽接受门。
`tools.audit_contact_time <目录> --out <新报告> --seconds 540`逐态核查完整CCD与
几何分区接触功、法向/摩擦冲量。GL4/8/16/32/64未解析时使用向量GK21；经验
误差不作严格上界。输出同时保留算法功与弦积分的差，不能用求积通过代替时间收敛。
