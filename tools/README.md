# 工具入口

当前只有一套原生 P1 / IPC / 有限应变 Maxwell 计算主线。目标与参数见
[模型规格](../docs/current_model_spec.md)，研究结论见[研究状态](../docs/research_status.md)。
不运行历史诊断，不复制源码树。每个新结果记录当前源码、参数和检查点指纹；运行期间不得编辑参与计算的源码。

| 工具 | 用途 |
|---|---|
| `run_material_candidate.py` | 单角度完整体计算，准确材料/摩擦历史、刀尖材料身份与独立物理审查 |
| `run_material_queue.py` | 同一材料的独立初态串行队列；各组成功或失败后继续下一组，HTML 自动记录 |
| `qualify_intact_baseline.py` | 校验保留的三角度 2 mm 基线报告及检查点；可用当前原方程逐态重新审查 |
| `plot_experiment_angles.py` | 显式指定实验与 FE 路径，输出叠图、共同区间 CSV、分段斜率及 HTML |
| `visualize_ipc.py` | 接受状态的原始变形网格、完整三维 Cauchy 最大主拉应力和刀尖接触偏移视频 |
| `render_solver_dashboard.py` | 任意上述队列的只读 HTML 快照、日志与有界后台更新 |
| `check_publish.py` | 检查可发布的代码/文档文件类型；不上传原始数据、论文 PDF 或结果 |

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
