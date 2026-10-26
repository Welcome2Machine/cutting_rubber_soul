# Optical-flow exchange exporter

接口依据：[optical_flow_handoff.md](optical_flow_handoff.md)。此入口只整理已有观测，不运行光流、训练、仿真或参数标定。

当前本地交付目录：`results/20261012_optical_flow_exchange/`。入口说明为该目录的 `README.md`；正式校验结果为 `validation.json`。实际数据和观测报告位于 Git 忽略目录，不进入源代码提交。

## 导出与校验

使用符合 `environment/AGENT_CONTRACT.md` 的项目 Python 环境：

```powershell
python softcut.py exchange-export --out results/optical_flow_exchange
python softcut.py exchange-check --root results/optical_flow_exchange
```

已经完成相机叠字审计的交付包保留其时间证据，可直接重新校验：

```powershell
python softcut.py exchange-check --root results/20261012_optical_flow_exchange --workers 3
```

包内的 `clock_audit/clock_audit.json` 和对应原图窗口记录已人工审阅的时间证据。
它提供相机叠字支持的时间估计，不把稀疏时间锚点核对升级为逐帧精确时钟验证。
当前交付直接使用本地目录。历史批次专用时间修订脚本与 ZIP 归档脚本已移除，已有交付数据及校验依据保留。

默认读取 `results/20261005_0930_all/manifest.json`。先做单组短段接口试验：

```powershell
python softcut.py exchange-export --out scratch/handoff_pilot --names 0930l60a1 --stop-mm 1
python softcut.py exchange-check --root scratch/handoff_pilot
```

`--stop-mm` 取保存的条件图像深度第一次跨阈值的时间前缀，不是独立测得的实验室深度，也不证明此前未起裂。默认拒绝覆盖非空输出；`--resume` 仅复用已完成且源路径/深度范围一致的试次，不接续半写入文件。`--workers 1/2/3` 控制独立 CSV 整理进程，不调用 GPU 或重新计算光流。

## 数据约束

- 源坐标统一到未叠加的本地 MP4，完整保留已保存的初始采样位置。
- `x_px/y_px` 仅放有图像和材料区域支持的观测；预测和被排除的原匹配另列。
- 无验证身份的跨缺口恢复使用新连续段 ID 和新参考；原追踪器 seed ID 仅作为来源保留。
- 图像比例尺、刀具几何推算和模型接触估计保留自身依据，不冒充实验室注册或直接接触观测。
- 传感器数据保持原通道、采样和独立时钟；轴向及视频同步未确认时不输出 Fz(depth)。
- 未知字段为空/null。初始化分数不是零误差证据，相关性与循环误差不等于已校准物理精度。

独立校验器逐行检查数值、状态、参考恒定性、连续段、帧引用及覆盖统计；源坐标采用固定抽样核对，帧映射和几何逐帧核对。校验不证明物理测量精度，详见每次交付的质量报告。

实现集中在 `src/exchange/`：`source.py` 核对来源与坐标，`observations.py` 流式输出轨迹和可选证据，`export.py` 组织交付与预览，`validate.py` 独立校验。公共哈希和 JSON 写入复用 `src/io.py`。
