# CB245NE 橡胶斜切建模

当前方向（2026-09-28）：`feature/czm-cutting` 是供服务器接手的 CZM 开发分支。**先读 [服务器交接](docs/server_handoff.md)**。计划采用初始完整黏结的预设潜在路径，预测该路径条件下的起裂、分离和刀具接触；CZM 切割求解器尚未实现，本次交接不启动计算。暂不使用实验力或视频/光流反向调参。实验目标为 30×5 mm 侧面、20 mm 出平面宽度、底面固定、0.02 mm 圆刀尖沿实验室竖直方向压入，完整加载目标保持 3 mm；输入以 [模型规范](docs/current_model_spec.md) 和 [任务要求](docs/task_requirements.md) 为准。

已能运行的是有限应变、完整橡胶的 IPC 接触模型。现存 **6 组完整 0–3 mm 算例**，另保留独立论文和接触组件基准；参数、路径与验收范围见 [已完成结果](docs/completed_results.md)。这些结果不等于已经实现真实开裂，也不保证当前每种配置均可稳定完成。

既有起裂诊断采用单轴校准的方向松弛 AT1 在完整状态的单边稳定性，报告条件性候选。另有位移—损伤短窗口，只得到微损伤，尚未形成真实分离；候选后的完整材料结果是假想延续。CZM 下一步先验证黏聚本构与零间隙接触，再接入真实 60° 网格和完整界面对照。方法与验收集中于 [断裂路线](docs/fracture_route.md)，已有实施证据集中于 [求解状态](docs/solver_status.md)。

## 安装与入口

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m prototype.ipc_reference --baseline --output outputs/new_ipc_baseline --detach
```

`--baseline` 是固定的 `ipc-frictionless-v1`：无摩擦、无断裂、0.08 mm 圆角，不能冒充当前 0.02 mm 刀尖模型。当前尺寸、起裂诊断、后处理及研究组件的准确命令见 [运行说明](prototype/README.md)。所有完整算例后台保存日志、配置、冻结源码和接受态，只检查一次启动；进程启动或队列结束不代表数值成功。

| 路径 | 职责 |
|---|---|
| `mechanics/`、`prototype/` | 共享有限应变材料、IPC 主线、起裂与接触研究组件；旧诊断按运行说明区分 |
| `tools/` | 队列、接受历史导出、常驻可视化、发布检查和 FEBio 构建 |
| `reproductions/dammass2023/` | 独立论文复现与验收；不是刀切主线 |
| `tests/` | 材料、数值、接触、状态及历史回归 |
| `docs/`、`references/` | 要求、方法、证据及原创文献摘要 |

润滑仍采用未标定的等效摩擦情景；环境温度只记元数据，没有油膜流体或温度本构。适用限制见 [润滑与温度证据](references/lubrication_temperature.md)。近不可压缩混合离散、旧 P2/IPM、FEBio、已脱粘裂面组件均属于独立研究或回归，不能据单项通过宣称完整切割已验证。

## 本地资料与发布

原始论文、图片/视频、实验与数值数据、读图曲线及测试夹具仅留本地。完整保留结果不重复复制；临时或已替代内容按 [清理记录](docs/cleanup_log.md) 清除。缺少本地夹具的测试必须显式 skipped，不能计为通过；FEBio 测试另需本机构建二进制。

公共 main 使用独立历史，只发布代码与文档。本次用户授权提交并推送 `feature/czm-cutting:feature/czm-cutting`，保留远端 main；发布前对 main 和该分支分别运行 `python tools/check_publish.py --ref REF`，检查全部可达历史。禁止推送或合并 legacy master、标签、`--all` 或 `--mirror`。

[运行说明](prototype/README.md) · [实验/模型参数](docs/intact_simulation_spec.md) · [建模路线](docs/modeling_roadmap.md) · [论文复现](reproductions/dammass2023/README.md) · [超算准备](docs/hpc.md)
