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
