# Dammaß 等 2023 年论文独立复现

对象：*Phase-field modelling and analysis of rate-dependent fracture phenomena at finite deformation*，[DOI / 正式全文](https://link.springer.com/article/10.1007/s00466-023-02310-1)。原文件为用户提供的 `references/updated phase.pdf`。

本目录是用户单独要求的论文基准，采用论文 EPDM、预缺口和平面应力工况。它不修改 N60A 刀具切割参数，也不承担无预裂纹起裂预测的验收。

## 本次复现与预先声明的验收范围

- 图 5：三个伸长速率 `0.0056、0.044、0.3 /s` 的均匀单轴 Cauchy 应力—伸长曲线。
- 图 7 左图：approach A (`beta_vi=0`)，25、200 mm/min 的 SENT 力—完整夹头开口曲线；相场演化作为裂后降力的场证据。
- 两种网格尺度的对照，保持论文材料、断裂长度和断裂能不变。

判据在首轮正式运行前写入 [acceptance.json](acceptance.json)：材料曲线归一化 RMSE ≤15%；SENT 峰值力、峰值位移相对误差均 ≤25%；峰前曲线归一化 RMSE ≤20%；峰后降力至少10%；快速加载峰值力较高、峰值位移较低。每个被提交验收的算例还必须完成其声明终点。运行失败或部分曲线不能标记为通过。

这是一组明确范围的数量级和趋势复现，不是整篇所有算例复现；approach B、DENT、蠕变及速率相关韧度暂不在本次验收中。原论文完整传播问题包含预缺口，因此通过本复现也不等于无缺口刀具切割验证完成。

## 方程与参数

共享材料内核独立实现式 23–24 的等容/体积 Ogden 能量，式 58–61 的有限黏弹性乘法分解、谱空间指数映射隐式积分，并通过局部隐函数微分得到一致切线。重复主伸长采用解析极限，保留完整三维应力状态。

主线迁移后，算法集中维护在 `mechanics/finite_strain.py`，本目录的 `material.py` 保留论文参数和平面应力包装。纸面体积能、EPDM参数及复现约束保持不变；源码指纹包含共享内核。重新计算1803个材料状态，与迁移前应力最大差为0。

| 参数 | 取值 | 来源 |
|---|---|---|
| 平衡支路 `(mu_1, alpha_1), (mu_2, alpha_2)` | `(-1.103,-4.883), (0.0105,7.951)` | 表 2 |
| 非平衡支路 | `(-0.385,-4.29), (1e-6,8.4)` | 表 2 |
| 两支路 Poisson 比；松弛时间 | `0.48；2.3 s` | 表 2 |
| 相场长度 `ell_c`；`Gc` | `0.275 mm；10.7 N/mm` | 表 3 approach A |
| 相场黏性；残余刚度 | `1e-4 N s/mm²；1e-10` | 表 3 |
| 整体尺寸；几何裂口 | `75 × 30 × 2 mm；中心底部20 mm裂口` | 图 6 |

初始剪切模量按 `mu0=sum(mu_p*alpha_p)/2` 计算，体积模量由 Poisson 比换算；使用论文式 23 的 `K/4*(J²-2log(J)-1)`，不沿用切割代码的另一体积能。数值单位为 N、mm、s，mm/min 输入显式除以60。

**平面应力公式疑点：** 原 PDF 第10页式62及附近文字印为过应力的出平面分量为零。物理平面应力应满足平衡应力与过应力之和的出平面分量为零。本复现求解总应力条件，并报告残差；这是一处明确的公式解释差异，不声称获得作者源程序确认。图5对照用于独立验证这一实现。

`solver.py` 采用 Q1 四节点四边形、2×2 Gauss 积分和裂口附近分级网格；论文采用二次三角形，因此需要独立网格检查，不能称为逐行复制作者离散。模拟加载方向的半试样：宽37.5 mm、高30 mm；完整开口为 `U`，右夹头位移 `U/2`，右夹头纵向位移为0；对称边剩余韧带约束水平位移、裂口段自由。夹持条件是对图6刚性夹持的实现解释。

厚度2 mm参与全部积分。右夹头反力就是完整试样反力，不再乘2；半模型储能和耗散乘2得到完整试样能量。

相场采用式2的 `d²/(4ell_c)+ell_c*|grad(d)|²`、二次退化与式53–54的历史变量。固定相场时解力学，固定力学时解相场，交替至收敛；相场额外满足旧值至1的约束，避免离散过冲或愈合。每个试值的黏弹性和相场历史均从上一接受增量出发；相场子迭代和失败回退不累计物理时间或黏性演化。这里没有刀具或摩擦项。

## 参考曲线与原始来源

`reference/fig5.csv` 从原 PDF 的三条模型曲线矢量路径（含 Bezier 段）提取，轴坐标及原路径保留在 `reference/provenance.json`、`fig5_vector_paths.json`。

`reference/fig7_stroke_envelopes.csv` 根据图7左图的坐标轴和饱和蓝/橙模型线像素提取，每个横坐标保留上下线迹；竖直跳跃不能当成实验误差区间。自动验收对照峰前上边缘与峰值，图中额外标注论文末端降力。黑色符号覆盖及线宽会影响读图精度，定位不等于作者原始数据；未拟合或修改材料参数来匹配这些曲线。

参考数据生成器 `extract_reference.py` 不读取任何复现输出。PDF、读图 CSV/JSON 和历史结果均仅保留本地，不上传远程。新克隆需自行取得原文，将对应 PDF 放到 `references/updated phase.pdf`，安装可选依赖 `python -m pip install pymupdf`，再执行 `python -m reproductions.dammass2023.extract_reference`。坐标与路径定位针对原本地版本，换版须重新核对。正式计算只需 NumPy、SciPy 和 Matplotlib。作者实验范围与模型曲线分开；量化对照的是论文模型线。

## 运行与验收命令

在仓库根目录执行，每次选择新的输出目录：

```powershell
python -m reproductions.dammass2023.run --case material --output outputs/paper2023_new/material
python -m reproductions.dammass2023.start --output outputs/paper2023_new/sent --mesh-h 0.275
```

第二条命令启动25和200 mm/min两个后台进程，保存PID和独立stdout/stderr日志。`status.json`记录真实完成/失败；`results.csv`仅含接受状态；`last_state.npz`、`peak_state.npz`保存场。首次运行终点为完整开口22 mm，用于覆盖论文峰后区间；这不是N60A的3 mm工况。

一次启动材料、两种网格（0.275、0.18333 mm）与两种速度，并在全部进程结束后**自动生成验收报告**：

```powershell
python -m reproductions.dammass2023.benchmark --output outputs/paper2023_new/full_suite --detach
```

后台控制器写入 `suite.stdout.log`、`suite.stderr.log`、`suite_status.json`；各算例有独立日志。缺少本地读图参考时，控制器在启动长计算前报出准备说明。最终曲线和验收位于 `comparison/`，无需 agent 持续检查。算例数值完成与曲线验收分开：前者看各 `status.json`，后者看 `comparison/acceptance.json` 的 `all_accepted`。

结束后生成对照和验收：

```powershell
python -m reproductions.dammass2023.report --material outputs/paper2023_new/material --sent outputs/paper2023_new/sent/sent_v25 outputs/paper2023_new/sent/sent_v200 --output outputs/paper2023_new/comparison
python -m unittest discover -s tests -p test_paper2023.py -v
```

报告同时生成PNG、SVG和机器可读验收JSON。传入多个网格的算例时全部列出，不只保留有利结果。源码指纹随算例保存，不复制源码目录。网格/时间步独立性、严格能量平衡与所有论文算例复现不能由这组宽容的曲线判据自动推出。

## 当前验证边界

保留结果位于 outputs/paper2023/validated_suite：材料曲线和四组网格/速度的完整曲线、状态及 comparison/acceptance.json。图5三种速率的归一化 RMSE 约为 0.14%、0.66%、1.15%。图7峰前、峰值和速率次序较接近论文，但裂后降力偏晚，高速失稳跳跃仍有约 140–150 N mm 未核算功；energy_audit_mainline.json 保留这一问题。

早期材料浮点精度失效对应的大变形状态已内嵌到 `tests/test_paper2023.py` 的切线回归，不依赖本地数据夹具；旧中断算例及临时图片已删除。不能把宽容的 all_accepted 初筛单独解释为完整裂后响应、网格无关或切割主线验证通过。
