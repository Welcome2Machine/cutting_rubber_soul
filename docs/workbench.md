# 本地仿真工作台

工作台提供配置核查、本地计算、运行日志、结果叠图及保存场动画。数值计算继续使用现有 P1 / IPC / 有限应变 Maxwell 主线；前端不复制求解器。[研究导读](research_guide.md)说明已有证据和研究路线，[模型规格](current_model_spec.md)定义物理范围。

## 启动与使用

在仓库根目录、已安装项目依赖的 Python 环境中执行：

```powershell
python -m tools.workbench --background --open
```

默认地址为 `http://127.0.0.1:8765/`，仅监听本机。`--port 8766` 可指定其他端口；省略 `--background` 则前台运行，Ctrl+C 关闭服务。后台方式不依赖当前终端，关闭浏览器不会停止计算。打开页面、导入文件或核查配置均不自动启动加载。

Python依赖通过 `python -m pip install -r requirements.txt` 安装；前端使用本地HTML/CSS/JavaScript，不需要Node或网络CDN。MP4编码另需PATH中可用的FFmpeg；未安装时仍可导出JSON/CSV，并在支持MediaRecorder的浏览器导出WebM。SuperLU无需额外线性库，PARDISO为可选项。

1. 在“设置与运行”选择刀具、材料、圆鼻半径、角度和摩擦系数，设置行程、网格及预算。
2. 点击“核查配置与运行指令”；核查成功后点击“启动本地计算”。修改配置后必须重新核查。
3. “运行记录”显示实际接受行程、任务状态及审查状态，可查看日志或停止当前任务。停止保留已保存状态，不提供跨配置自动续算。
4. 在“结果比较”导入一个或多个 `result.bundle.json`，勾选/隐藏各条曲线。场量区域选择工况、保存帧、色标和观察范围，可保存 PNG 或浏览器 WebM。
5. 已有仓库结果可通过“从本仓库已保存状态生成标准结果包 / MP4”导出后加入比较；这是后处理，不推进 FE 加载。

本机服务一次只接受一个计算任务；另有单个导出任务限制。默认求解预算 7200 s、总求解器进程预算 9000 s，后者包含准备及独立审查，须大于前者；结束后的展示导出另计耗时。预算不是预计完成时间，57°可能在浅层出现昂贵增量。停止或预算耗尽不会自动重试、延长行程或增加预算。服务重启将未结束任务记录标为中断，不据旧 PID 自动恢复或终止进程；重新计算前需确认旧进程状态。

工作台只在点击启动时创建新工况；此前暂停的研究队列不会被恢复。新任务从自身初态建立材料/摩擦历史，不从旧曲线中拼接状态。

## 配置范围与默认值

单位为 **mm、N、s、MPa=N/mm²**。速度固定 **0.2 mm/s**，竖直行程不超过 **2 mm**，完整体模型不包含损伤、裂纹或裂后响应。

| 类别 | 默认值及约束 |
|---|---|
| 材料 | CR 基线：G∞=.14、Gv=1.24、K=23.34333333333331 MPa，τ=1.7 s；备选 balanced、q25 均为研究候选 |
| 刀具 | 99755：厚 .245 mm、总楔角 23°、刚性对称双刃；其他型号从配置选择 |
| 锋利度 | 圆鼻半径 .02 mm，是建模假设；不是实测半径，也不是平倒角长度 |
| 倾角 | 默认 57°；允许 0–75°，同时要求倾角加半楔角小于 90° |
| 接触 | μ=.35，必须 >0；dry/oil/custom 是条件标签，切换标签不自动修改 μ |
| 加载 | 目标 2 mm，名义物理步 .01 mm，对应 Δt=.05 s；步长不能大于目标行程 |
| 标准网格 | 局部体 .05 mm、表面 .005 mm、细区半宽 3 mm、**远场 1 mm** |
| 刀具离散 | 圆鼻 64 面片，直刃间距 .005 mm；原分段在 JSON 中写 `null` |
| 资源 | 原生 SuperLU、IPC 2 线程、线性线程设置 4、材料批大小 0；PARDISO 须另有可用运行库 |
| 展示输出 | 默认最多 11 个保存帧，不自动生成 MP4；界面允许 2–41 帧 |

`coarse/standard/fine` 三个网格预设的局部体/表面间距分别为 `.1/.01`、`.05/.005`、`.025/.0025 mm`，远场均为 1 mm。手动修改后选择 `custom`；必须满足表面间距 ≤ 局部体间距 ≤ 远场间距。网格预设是离散选择，不是精度认证。

当前几何只支持对称楔刀加圆鼻，半径必须小于刀厚的一半。平直倒角、单刃、任意轮廓、刀具弹性均未实现。硅油条件仅表示给定正 μ 下的正则库仑近似，不包含流体油膜、黏附、溶胀或温度依赖；μ=0 不会被静默替换成小正数。

自由力残差 ≤1e−4 N、正 J、完整 CCD、边界、耗散、材料和摩擦历史验收保持锁定。保留原割线初猜→同增量仿射恢复及 500 次预算；仿射没有 40 次停滞早退。界面不提供以放宽物理验收换取完成的选项。

### 导入额外刀具或材料

选择“导入刀具 JSON”或“导入材料 JSON”。文件可直接是单个对象，也可包在 `{"blade": {...}}` / `{"material": {...}}` 中；不要将整个配置目录数组作为单个对象导入。额外配置进入当前页面目录，使用“导出运行配置”保存完整任务，下次可“载入运行配置”。启动前服务器会重新校验，不信任界面显示名称。

刀具对象示例（未测量参数需保留来源说明）：

```json
{
  "id": "custom_wedge", "label": "研究用对称楔刀",
  "shape": "symmetric_wedge", "thickness_mm": 0.30,
  "included_angle_deg": 27.0, "source_note": "假设几何，待实测"
}
```

材料对象示例：

```json
{
  "id": "cr_custom", "label": "共同材料研究候选",
  "law": "isochoric_maxwell", "Ginf_MPa": 0.20,
  "Gv_MPa": 0.90, "tau_s": 1.7, "K_MPa": 23.34333333333331,
  "source_note": "同一候选用于全部角度，非独立标定"
}
```

长期保留的内置目录分别为 [`configs/blades.json`](../configs/blades.json) 和 [`configs/materials.json`](../configs/materials.json)。完整运行配置的 `schema_version` 为 `softcut-config/1`：

| 顶层字段 | 内容 |
|---|---|
| `name` | 工况名称，仅用于识别，不作为命令执行 |
| `material` / `blade` | 上述对象的完整副本，不能只有配置 ID |
| `geometry` | `angle_deg`、`tip_radius_mm` |
| `contact` | `condition`、`mu` |
| `simulation` | `depth_mm`、`increment_mm`、固定 `speed_mm_s`、`mesh`、四个网格尺度、`nose_facets`、`blade_spacing_mm` |
| `resources` | `solve_seconds`、`total_seconds`、`ipc_threads`、`linear_threads`、`linear_backend`、`material_batch_size` |
| `output` | `max_frames`、布尔 `movie` |

网格尺度字段为 `local_spacing_mm/surface_spacing_mm/fine_halfwidth_mm/coarse_spacing_mm`。未知字段、非有限数、以布尔值代替数字、错误单位假设或不支持的模型均不能通过配置核查。导出完整配置后修改比手写整份配置更不易遗漏字段。

## 结果包、资格与可视化

标准结果包是 `softcut-result/1` JSON，可独立携带曲线、网格、少量保存场和来源。浏览器导入保留在本地，不上传远端服务。多个结果可叠图、逐条隐藏，并导出当前可见曲线 CSV。

| 顶层字段 | 定义 |
|---|---|
| `metadata` | 物理签名、角度、目标、来源路径/哈希、审查范围、坐标、场重构方法、电影状态 |
| `series` | 每个原接受深度一行；包含合格标记、反力、接触、能量及数值诊断 |
| `mesh` | `reference_positions` 与零基索引 `triangles`；无合格帧时为 null |
| `frames` | 实际保存态的 `index/depth_mm/positions/blade/fields/tip_contact`，不在两态间制造新 FE 解 |
| `field_info` | 场量名称、单位、节点/单元关联、全部导出帧的数值范围 |

`curves.csv` 与 `series` 使用同样的列，JSON 的缺失值为 null，CSV 为空单元格。核心字段如下：

| 字段 | 含义 |
|---|---|
| `depth_mm`、`time_s` | 竖直向下刀具行程；材料记录中的物理时间 |
| `qualified`、`qualification` | 是否处于已有审查的连续合格前缀 |
| `fz_N` | 实验室竖直向上刀具反力，原 `reaction_fz_n` |
| `fy_N` | 实验室截面内横向有符号反力，**映射原 `reaction_x_n`** |
| `fyz_N` | `sqrt(fy_N²+fz_N²)`，面内合力大小，不是第三个方向分量 |
| `raw_fz_N/raw_fy_N/raw_fyz_N` | 原接受记录值，即使超出合格前缀仍保留供追溯；不代表认证 |
| `residual_N`、`min_J` | 接受态自由力残差和最小体积比 |
| `stored_energy_Nmm`、`viscous_dissipation_Nmm` | 材料储能、累计黏性耗散 |
| `equilibrium_energy_Nmm`、`overstress_energy_Nmm` | 平衡与非平衡支路储能；采用原材料账目定义 |
| `dt_s`、`viscous_dissipation_increment_Nmm` | 实际材料时间步及本步黏性耗散 |
| `friction_work_increment_Nmm` | 单步摩擦功，不能直接当累计值 |
| `normal_fz_N/friction_fz_N` | 局部 IPC 法向/摩擦的竖直贡献，不是初始表面 N/T 投影 |
| `tip_contact_status/contact_reference_offset_mm` | 刀尖接触身份及初始表面材料偏移；多义或缺失时偏移为空 |

标准包只公开绘制连续合格前缀：首个审查失败后，即使后面各态单独通过，也不拼接为合格路径；这些行的公开物理量留空，不跨空值连线。没有审查时不自动认证。导出复用已有审查，并重新核对选中检查点指纹、J 和储能；**不执行新的全路径平衡审查或材料更新**。旧源码哈希不同记录为来源差异，不单凭此否定当前显式支持方程的保存场。

也可直接导入旧 `result.json`，仅用于曲线，标为“旧格式 · 未核查资格”；这不等价于标准包合格前缀。需要有资格的图和场时，应从含 `result.json`、`audit.json`、`states/` 的算例目录生成标准包。当前浏览器不直接导入实验 CSV；标准 CSV 是导出接口。

仿真原始力值保留，包括零深度的屏障预载，不在图上自动归零。实验首值偏置属于独立实验比较约定，不能重复应用到仿真结果。

### 场量与动画含义

坐标显示为 `[y,z] mm`，公共 y 对应求解器 x，z 向上。变形不放大；“刀尖附近”和全截面使用固定观察范围。P1 单元应力/应变按原单元常值着色，不做节点平滑。

| 字段 | 精确定义 |
|---|---|
| `tensile_MPa` | 三维嵌入 Cauchy 应力最大主值与 0 的较大值；负值已裁零，只表示拉应力显示量 |
| `green_max/green_min` | `E=(FᵀF−I)/2` 的三维最大/最小主 Green–Lagrange 应变，无量纲；不是伸长比或小应变 |
| `J` | `det(F)`，平面应变三维嵌入的体积比 |
| `displacement_mm` | 节点相对未变形参考构型的位移模；界面以三角形三节点均值着色，原节点数值保留 |

平面应变有 F33=1、E33=0，主应变排序包含该零值；面外应力仍可能非零。场重构支持当前 α=2 的等容 Maxwell 及已实现体积势，使用保存的已提交材料度量，不用纯弹性应力代替，也不重新进行局部返回映射。不支持的本构明确拒绝导出。

每个场默认使用全部导出帧的统一色标；可指定显示上下限，范围外颜色被截断但数值不改。跨工况应使用共同色标。最大值、局部应变及接触图均不是起裂判据。

浏览器可保存当前 PNG，支持 MediaRecorder 时可录制 WebM。MP4 由本地 FFmpeg 编码，默认最大主拉应力、全局及接触区双视图。播放均为抽样接受帧，匀速播放不代表真实加载时间，也不补算中间状态。FFmpeg 缺失或编码失败时 JSON/CSV 仍可成功导出，`metadata.movie.status` 单独标为 `failed` 并给原因，不能把展示导出成功当成视频成功。

## 目录与命令行后处理

```text
outputs/workbench/
  server.log
  runs/<任务编号>/
    config.json       # 完整规范化输入
    job.json          # 命令、预算、版本、进度和状态
    worker.log
    case/             # 原主线 result/audit/states/run.log
    presentation/     # 标准 JSON、CSV、可选 MP4
  exports/<导出编号>/  # 已有算例的独立后处理，不改原结果
```

独立导出已有结果，无需启动页面或重新 FE 加载：

```powershell
python -m tools.workbench --export outputs/baseline/angle57 --output outputs/workbench/baseline57_view --frames 11 --movie
```

CLI 的帧数允许 2–201；界面限制为 2–41 以控制浏览器内存。大网格建议先导出少量帧。标准包仅含展示所需状态，不替代准确续算使用的原始 NPZ 双历史。

## 维护入口

`workbench/config.py` 负责规范化输入及构造参数向量；`server.py` 提供本机接口、任务和目录边界；`results.py` 负责标准包及保存场重构；`static/` 为无外部构建依赖的 HTML/CSS/JS；`tools/workbench.py` 是统一入口。求解仍调用 `tools.run_material_candidate`，核心方程位于 `prototype/` 和 `mechanics/`，不在界面实现第二套求解器。

配置变化、保存场支持范围或输出语义改变时同步相应回归与本文件。可复用检查见 `tests/test_workbench_config.py`、`tests/test_workbench_results.py`；参考材料和刀具配置进入 Git，原始数据、NPZ、视频及运行输出保留本地。
