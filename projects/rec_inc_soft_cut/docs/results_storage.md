# 结果存储与无损精简

2026-10-26。本轮仅维护存储，不重新识别、跟踪或求滑移。实验 `data/`、模型、数值精度和原有 NPZ/MKV/MP4/PNG 均保留。

## 存储规则

- 计算数组继续使用压缩 NPZ，固定材料 ID、帧映射、缺测状态及浮点类型不变。不通过降低浮点精度或抽掉时间帧节省测量数据。
- 有限元交接的大型文本表格使用 `*.csv.gz`。解压后与迁移前 CSV 的每一个字节一致，包括全部小数位、缺失字段、行顺序和换行符。`metadata.json` 的 `files` 字典提供当前文件名，交接检查器兼容两种格式。
- 新的 `exchange-export` 默认直接写 `tracks.csv.gz`，不先生成巨大的中间 CSV；需要旧格式可加 `--plain-csv`。底层 `export_tracks(..., compress=False)` 仍兼容既有调用。
- 新滑移求解的详细 JSON 使用紧凑排版，不改变键值、诊断内容或求解过程。小型配置和汇总继续便于阅读。
- 已有大型 JSON 使用 NTFS 透明压缩：仍是相同路径和相同文件字节，已有 SHA-256 不失效。该收益属于磁盘占用减少；文件属性中的逻辑长度不减少。移到不支持 NTFS 压缩的磁盘后可能展开，可用 Windows `compact /u <file>` 撤销。
- 浏览器自动审阅的用户配置、缓存属于临时产物，可删除。科学对照、失败反例和源码快照具有追溯价值，体积较小，保留。

无损去阴影视频用于继续核验纹理，不能为了省空间用有损 MP4 替换。现有展示 MP4 不再次有损编码。旧批次中含重复信息的数值文件仍可能被准备流程或交接引用，未按文件名或日期批量删除。

## 读取与核验

```python
import json
from pathlib import Path
import pandas as pd

case = Path('results/20261012_optical_flow_exchange/0930l60a1')
meta = json.loads((case / 'metadata.json').read_text(encoding='utf-8'))
for chunk in pd.read_csv(case / meta['files']['tracks'], chunksize=100_000):
    observed = chunk[chunk.valid == 1]
```

Pandas 根据 `.gz` 后缀自动解压；Python 标准库可用 `gzip.open(path, 'rt', encoding='utf-8', newline='')`。需要原始 CSV 时只需解压，无须重算光流。

迁移先写独立临时文件，完整读回并比较解压 SHA-256，再核对源文件未变化，最后替换；压缩失败、校验失败或压缩没有收益时保留原文件。迁移后更新交接元数据与文件校验清单，旧清单和旧元数据另存以维持历史可追溯性。文件系统压缩的每个 JSON 另作前后哈希比较。

本轮机器记录在 `environment/cleanup_20261026.json`，逐文件清单、旧交接校验表与旧元数据保存在忽略目录 `environment/local_maintenance/`。正式交接目录内的 `storage_migration.json` 连接旧、新文本表格的哈希。结果完整性可运行 `python softcut.py check`，交接内容可运行 `python softcut.py exchange-check --root results/20261012_optical_flow_exchange`。

这些验证只确认格式转换和保留数据完整，不能证明光流或滑移的物理精度。

## 本轮结果

Windows 报告的磁盘占用从 **39.60 GiB 降为 33.70 GiB**，释放 **5.90 GiB，约 14.9%**。逻辑文件总长度降为 36.46 GiB；其中已有 JSON 的透明压缩收益只体现在磁盘占用。

25 个大型 CSV 无损迁移，减少约 3.12 GiB；89 个大型 JSON 的透明压缩在操作时减少约 2.76 GiB，重建交接校验清单后的净收益已计入上面的总数。另清除约 20.34 MiB 浏览器审阅缓存及本轮中断压缩留下的临时文件。62,309 个既有数组、测量/展示视频与图像全部保留，长度不变；维护过程中还核对了其修改时间。

交接目录 14,258 个成员已全部读取并重建 SHA-256 清单。未修改成员匹配原清单，每个压缩表的解压哈希匹配原始交付 CSV。完整测试 724 passed、3 subtests passed；最后调整快速压缩级别后，相关 28 项测试再次通过。保留结果检查通过：633 条 HTML 链接、270 个数组包、23 组完整测量、5 组展示和 5 组身份优化结果。
