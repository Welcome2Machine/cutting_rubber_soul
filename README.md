# CB245NE 橡胶斜切建模

目标：N60A 无润滑、30×5 mm 侧面、20 mm 厚度、底面固定，竖直加载至首次起裂或 3 mm 上限。以 [模型规范](docs/current_model_spec.md) 为准，实验原图仅保存在本地。

**完整切割尚未实现并验证。** Python 主线提供无预裂纹的全域强度候选筛查；修正版 FEBio 独立参考验证至 0.05 mm，尚未跨过主线失效区。[当前状态](docs/solver_status.md) 集中记录结果与限制。

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

| 路径 | 用途 |
|---|---|
| mechanics / prototype | 共享有限应变材料、完整试样模型及诊断 |
| tools | FEBio 构建、接触补丁及单节点超算模板 |
| reproductions/dammass2023 | 独立论文复现与验收代码；参考数据自行在本地准备 |
| tests | 物理、数值、历史及失败状态回归 |
| docs / references | 要求、模型方法及带出处的论文方法摘要 |

远程仅包含代码、文档和运行配置。PDF、图片、视频、实验/计算数据、读图曲线及 tests/fixtures 均不上传；本地原件保留。实验力和光流/DIC 数据由用户保留作独立验证。缺少本地夹具的测试显式 skipped，不能称为完整回归通过；解析材料、合成接触和装配测试仍可运行。FEBio 测试另需本机构建二进制。

公开 main 使用独立历史，不包含旧的本地 master。推送前执行 `python tools/check_publish.py --ref main`，仅推送 `main:main`；不要使用 `--all`、`--mirror` 或推送本地旧分支/标签。

[运行命令](prototype/README.md) · [超算准备](docs/hpc.md) · [任务要求](docs/task_requirements.md) · [建模路线](docs/modeling_roadmap.md) · [论文复现](reproductions/dammass2023/README.md)
