# 运行入口

在仓库根目录执行，算例输出使用新的空目录。能力边界见 [求解状态](../docs/solver_status.md)，方程见 [模型说明](../docs/initiation_model.md)。

## Python 主线

```sh
python -m prototype.start_initiation --mode to_candidate
python -m prototype.run_initiation --config prototype/initiation_config.json --output outputs/new_initiation --mode to_candidate
```

第一条后台运行；第二条前台运行，适合调度器作业。默认完整积分 P2 与二次边界积分接触。首次全域名义强度达到 4.4 MPa 时细分候选区间并停止，否则以 3 mm 为上限；候选不是实际裂纹。配置、源码指纹、接受状态、失败试值及状态随运行保存。历史检查点按保存配置恢复；低层 P1/混合元保留作回归对照。多轴判据须显式提供独立标定，参数见 `--help`。

```sh
python -m prototype.verify_mainline --output outputs/new_checks
python -m prototype.verify_mainline --case oblique_frictional --loading-end .13 --output outputs/new_prefix
python -m prototype.audit_contact --source outputs/current_failure --output outputs/new_audit.json
```

最后一条需要本机历史证据；新克隆不包含 outputs 或 tests/fixtures。数据回归缺少夹具时明确 skipped，解析/合成测试照常执行。

返回旧 P2 主线后的当前结果见求解状态页；已保留的严格检查点为 0.49875 mm，后续仍未修通。对完整积分 P2 的失败状态，可在 `audit_contact` 后添加 `--curvature` 检查缩放体刚度的最小特征值。该审计没有投影接触约束，负特征值不能单独解释为物理失稳或起裂。新增锥可行回溯每次求解最多尝试两次，原始接触/力验收不变。

审计额外检查至少 24 次原 Newton 回溯并记录切换节点，区别配置预算不足和极小下降步；生产步长预算不自动改变。对积分接触输出保留/去掉接触几何切线的凝聚响应，后者仅是诊断。`--branch-seconds 30 --threads 4` 可检查全接触点有界线性化四分支可行性，压力上界 100 MPa、gap/slip 相对试值 ±0.02 mm；这些是搜索界限，超时不等于无解，候选不作为物理接受状态。

`--switch-node 2413` 可沿该状态的原 Newton 方向定位指定点的精确黏着/滑动边界，核对两侧方向是否进入自身分支、原误差是否增加、行替换行列式比及单边有限差分。仅支持当前非黏附 Alart-Curnier 积分接触和无压力辅助块；找不到边界时明确报错，不伪造证书。它诊断非根点附近的局部线性化，不能证明物理失稳。备用搜索现在按真实投影切换函数筛选接触点，仍必须通过原方程验收。

已有真实历史的固定网格接续可后台诊断：

```sh
python -m prototype.contact_study --case recovery --source outputs/p2_tolerance_probe --output outputs/new_fixed_continuation --loading-end .52 --max-increment .00625 --fixed-mesh --case-seconds 1200 --threads 4 --detach
```

沿用来源的全部物理及容差设置；本例来源穿透容差已为 0.2 μm。`--fixed-mesh` 禁止途中换网格并记录每步前后历史；时间预算在增量之间检查，单增量可能超过预算。日志、接受状态及失败状态持续保存，启动记录/stdout/stderr 位于输出目录旁。该命令不改变完整实验的 3 mm 上限。

对实际接触约束和局部变形作进一步审计：

```sh
python -m prototype.audit_contact --source outputs/p2_repair_continuation --output outputs/new_branch_audit.json --modes 3 --mode-output outputs/new_modes.npz
```

输出非对称耦合切线的缩放奇异值、接触乘子行无谱质量的广义特征值、单独的负模态搜索，以及二次边伸长和局部压力变化。奇异值依赖缩放；失败试值的负特征值不能证明真实平衡失稳。默认导出奇异模态；`--mode-kind negative` 改为导出实负模态，若搜索结果不满足条件则拒绝导出。增加 `--perturb-mm .0005 --perturb-output outputs/new_mode_probe --iterations 40` 可沿每个导出模态作正负扰动，固定同一增量的物理历史，逐项记录原始方程验收结果。该命令是有界诊断，不是完整加载或起裂预测。

## 非线性分块接触短测

```sh
python -m prototype.contact_partitioned --source outputs/quadratic_unchanged_mesh_continuation --trial outputs/p2_repair_continuation/failed_trial.npz --output outputs/new_partitioned_probe --depth .5 --seconds 150 --initial-proximal .01
```

仅求解一个固定历史的物理增量，最多 30 次外层、每次最多 30 次内层迭代，4 线程。时间预算在内层残差计算前及外层之间检查，单次稀疏求解可能超过剩余时间。默认保留真实法向几何，冻结外层摩擦容量和切向框架；数值近端项在固定点消失。`--no-acceleration` 关闭外层 Anderson；`--affine-normal` 增加冻结法向几何对照；`--traction-compliance 0` 关闭界面近端项。物理参数及最终验收不变，当前还未通过旧 0.5 mm 失败点，未设为生产默认。

输出必须为空目录。`run.log` 记录内外残差，`result.json` 保存实际配置、源码哈希、时间、结果状态和原方程指标，`trial.npz` 保存试值及固定前态，`source` 保存源码快照。`budget_exhausted` 与 `failed` 都不是物理完成；命令正常退出仅表示诊断成功记录，须检查 `completed` 和 `status`。新克隆不含上述本地检查点。这是短测，不是完整 3 mm 算例。

辅助残差路径使用相同入口和记录格式：

```sh
python -m prototype.contact_partitioned --method path --source outputs/quadratic_unchanged_mesh_continuation --trial outputs/p2_repair_continuation/failed_trial.npz --output outputs/new_residual_path --depth .5 --seconds 180 --path-step-size .0001 --threads 4
```

保持完整原方程，以伪弧长同时求位移、接触力和辅助参数，默认定位法向开闭及压紧条件下的黏滑切换。`--no-path-switch-events` 关闭这两类事件，作普通伪弧长对照；`--path-metric-ratio .1` 固定初始状态/参数的相对度量，`--auxiliary-tolerance 1e-6` 是归一化辅助残差门限，不改变物理验收。`--path-steps 100` 限制预测尝试总数（包含回退），`--seconds` 限制残差调用之间的墙钟时间。单次分解或事件定位可能略超预算。辅助参数负值或下降均不代表实验卸载。

失败时 `trial.npz` 的常规字段是最佳原残差试值，`auxiliary_displacement`、`auxiliary_multipliers`、`auxiliary_parameter` 是最后通过辅助方程的路径点。它们都不自动成为已接受的物理状态。日志记录切换节点、辅助残差、单侧切向定向和原残差；`path_diagnostics` 记录路径点数及参数转向数。该入口仍是单增量诊断，未成为默认完整加载求解器。

增加 `--resume-path outputs/p2_residual_path_extended` 可从该目录的最后辅助点继续，仍须保留相同 `--source`、原始 `--trial`、深度和路径度量；不能把保存末点另当作 `--trial` 重新定义同伦。新检查点还保存 `auxiliary_tangent`，用于保持参数下降段的方向。恢复会核对配置、网格和物理历史，并检查起点满足原辅助方程；旧检查点缺少完整切向时只在日志参数切向非零的情况下恢复。省略 `--trial` 则从上一已接受物理状态另起辅助路径，作为不同初值对照。

## 固定网格与积分对照

```sh
python -m prototype.contact_study --case discretization_suite --source outputs/p2_repair_continuation --output outputs/new_discretization_paths --loading-end .52 --max-increment .00625 --case-seconds 1200 --threads 4 --detach
```

四条路径顺序运行：原拓扑每半边 4、8、16 个 Gauss 点，以及改连拓扑每半边 16 点。只读取来源的参考网格和参数，位移、接触历史及预测器从零开始；网格加载中固定，允许原门限下的增量二分。局部改连保持顶点、边界、单元数量及最小角约束，改动记录在 provenance.json。新选项 `contact_quadrature_order` 默认仍为 4；不能把改变积分规则后的旧加载检查点当成一致的接续初态。

这是 0.52 mm 有界诊断，实验目标仍为 3 mm。每条路径在增量之间检查 1200 s 预算，预算耗尽标为 budget_exhausted，不等同求解失败或路径通过。suite.json 汇总状态，各组 run.log/result.json/increments 记录过程和真实前后历史；在约 0.45、0.49 mm 及最后接受状态自动复核原始方程和约束谱。谱搜索有迭代上限，未收敛不等于不存在负模态。线程限制作用于支持的 BLAS/OpenMP 线程池，不代表所有稀疏步骤均并行。

任意已保存增量可独立审计：

```sh
python -m prototype.audit_contact --source outputs/new_discretization_paths/q16 --increment outputs/new_discretization_paths/q16/increments/00010.npz --output outputs/increment_audit.json --modes 3
```

加载真实前一状态和时间步，核对网格及配置指纹；不使用零增量回放替代加载分支。首次启动只检查进程与日志，后续由后台记录。

## FEBio 独立参考

```sh
python tools/build_febio.py --contact-audit --jobs 8
python -m prototype.febio_reference --case patch --contact-variant corrected --threads 8 --output outputs/new_patch --execute --timeout 60
python -m prototype.febio_reference --case oblique --depth .05 --spacing .5 --surface-spacing .0025 --contact-variant corrected --threads 8 --output outputs/new_febio_prefix --execute --timeout 900
```

构建要求 Windows Visual Studio C++/CMake 或 Linux GCC/OpenMP/CMake。Linux 分支仍待目标节点实编译。Windows 旧环境直接脚本曾遇到 MSBuild 无错误正文退出；已验证的替代调用为：

```powershell
python -c "import runpy,sys; sys.path.insert(0,'tools'); sys.argv=['tools/build_febio.py','--contact-audit']; runpy.run_path('tools/build_febio.py',run_name='__main__')"
```

`upstream` 保留原算法，`audit` 仅诊断，`corrected` 加切线修复和超预算拒收。后两者使用 `.runtime/febio_audit`，默认入口仍为 upstream。切向增广参数默认分别为 1、1、250 N/mm³，不是摩擦系数。`--mesh-depth .05 --depth .0275` 截短加载而保持同网格。`--executable` 指定其他兼容构建；不带 `--execute` 只导出。

FEBio 为严格平面应变 Hex8 三场离散，连续材料能量与主线一致，压力平均不同。输出 y 为实验竖直轴。没有候选停止或自由断裂；contact_audit.json 检查接受步接触，path_audit.csv 和 geometry_audit.json 检查每个同步输出状态的整边穿透、角点 J、平面应变约束、内部自由力及刀具合力。越限或输出不完整即停止并标记失败；这些必要检查尚不等于全体接触节点的独立平衡证明。restart.dmp 已验证生成，恢复等价尚未验证。

本地完整 3 mm 无损伤接触诊断可后台运行：

```sh
python -m prototype.febio_reference --case oblique --depth 3 --spacing .5 --surface-spacing .00125 --blade-arc-segments 256 --contact-variant corrected --threads 4 --output outputs/local_febio_refined --execute --detach
```

启动记录为 launch.json，后台检查日志为 driver.log；case/status.json 记录求解器 PID、审计深度和最终状态，case/model.log 为迭代日志。目录必须为空；已有同名运行时更换目录名。后台程序自行检查增量，agent 无须持续跟踪。4 线程是本机资源分配，不是全工况稳定性的保证。

细化窗口按实验室向下位移在材料坐标中的实际投影生成，60° 工况沿 s 增大方向延伸。默认另加 max(0.1, 0.25×网格目标深度) mm 的变形余量，`--contact-padding` 可显式指定；这是数值余量，不能保证任意变形后接触仍在细网格内。实际范围写入 config.json。`--blade-arc-segments` 控制圆弧分面数，默认 256，保留半径 0.02 mm 和全楔角 23°；没有把刀具钝化当成原实验。完整参考网格更大，运行可能更慢，是否改善连续推进仍需实际路径检验。停机后的接触审计选择接受步记录更完整的 model.log 或 stdout.log，来源写入状态，不拼接重复事件。

允许适度降低精度时，可用均衡配置：

```sh
python -m prototype.febio_reference --case oblique --speed-profile balanced --depth 3 --spacing .5 --threads 4 --contact-variant corrected --output outputs/local_febio_balanced --execute --detach
```

`balanced` 默认接触间距 0.0025 mm，保留全 Newton 和 0.0025 mm 目标增量；试用 Broyden 更新和加倍增量产生更多回退，未作为默认方案。内部 dtol 从 1e-5 调至 1e-4、etol 从 1e-8 调至 1e-6、残差平方下限从 1e-18 调至 1e-16，接触增广相对容差从 1e-6 调至 1e-5；rtol 仍为 1e-8。穿透门限为 0.0002 mm，真实必要力平衡 0.001 N 和接触投影 0.001 MPa 门限不变，失败仍回退。每个接受步仍保存节点、刀具反力和独立审计；只在正常结束时输出完整绘图场，省去重复的单元/接触面文本。重启文件仍保留，恢复等价未验证。`reference` 保留原迭代参数；局部峰值和起裂位置最终须用细网格复核。状态中记录含失败试步的矩阵重建总次数、接受步平均迭代次数及独立审计耗时。

完整算例只检查一次启动，由日志记录过程。超算作业内使用前台入口，见 [超算说明](../docs/hpc.md)。
