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

FEBio 为严格平面应变 Hex8 三场离散，连续材料能量与主线一致，压力平均不同。输出 y 为实验竖直轴。没有候选停止或自由断裂；contact_audit.json 检查接受步接触，geometry_audit.json 仅检查末态整边穿透、J 和必要力平衡。restart.dmp 已验证生成，恢复等价尚未验证。

完整算例只检查一次启动，由日志记录过程。超算作业内使用前台入口，见 [超算说明](../docs/hpc.md)。
