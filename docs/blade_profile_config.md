# 显微轮廓参数配置

来源为本机 `researchtopic/bladetype.pptx`，SHA256
`1b44cde22b7008c09ded49a107c02e8f4628f9527388f7902480064c32e189b7`。
五页分别为 C、E No.1、E No.2、A low performance、A high performance。
原 PPT、原图和逐像素提取结果留在本机；Git 保存提取程序、来源清单和参数配置。

`configs/blade_profiles.measured.json` 是 `softcut-blade-shape-catalog/1` 配置，
不是可直接运行的求解器配置。`apparent_profile` 保留原字段名。
2026-10-29 用户确认灰度图就是仿真所用的侧视截面，已记录
`view_plane=normal_to_cutting_edge`、`side_view_confirmed=true`；宽度/夹角按
截面厚度/宏观楔角解释，不再要求重复确认该批图片的方向。
A 两组均在反复使用后测量；用户报告同一把刀反复使用的性能变化不明显，
不同刀具之间差异巨大。该信息是定性实验观察，不是量化磨损试验，未提供循环次数
或新刀初态。仅 A 两组的使用状态更新为 `repeatedly_used`，C/E 使用状态仍未知。
A 与旧目录里的具体产品编号没有自动对应关系。

2026-10-29 用户补充两张 100 μm 标尺的刀头近景，最终更正的对应关系是：
**图 1 低性能 A，图 2 高性能 A**。来源和 SHA256 在
`configs/blade_tip_closeups.json`，原图留在 `researchtopic/blade_tip_closeups_20261029/`。
标尺为 50 像素，名义采样约 2 μm/像素，不等同于光学分辨率。
图 1 外包络收尖相对平顺，图 2 端部外包络较圆且上侧有局部不规则凸起；
这些是图像形态观察，不能直接把凸起鉴定为毛刺或认定其导致高性能。
不能仅按外包络“看起来尖/钝”或性能标签给实际接触点曲率排序。
原宏观图的亮度分割在这两张纹理近景上不稳定，因此没有提交圆鼻拟合值。
近景按性能类别关联宏观配置，尚无明确刀具序列号或沿刃测点位置。
旧宏观测量数值保持，刃口半径仍未知。

提取只使用内嵌灰度图的亮轮廓和红色 500 μm 标尺，约 10 μm/像素。
采用最大连通域、三个灰度阈值、亚像素交叉位置以及每侧连续两线段稳健拟合；
刀身中线倾斜需小于 5°。前五个像素不用于直线拟合，虚拟尖端为两侧斜面
外推交点，**不是实测圆鼻尖端**。表观斜面长度从该虚拟交点算起。
阈值、标尺两端各 ±1 像素及轮廓定位各 ±1 像素构成敏感性范围，
不是重复样品的置信区间，也不包含仪器标定及残余装夹方位误差。
亚像素插值不能提高原图的真实光学分辨率。

彩色图和三维渲染没有原始高度场及完整标定，不提取粗糙度、毛刺高度或刃口半径。
所有配置保留 `tip_radius_mm=null`、`physical_prediction_qualified=false`。
宏观楔角不能等同于可能存在的末端微刃角，不能据此自动替换旧 23° 基线。

## 重新提取

在新输出路径执行（不会启动 FE，不读取实验力）：

```powershell
python -X utf8 -m softcut_solution.blade_profiles --source researchtopic/bladetype.pptx --manifest configs/blade_microscopy_manifest.json --output runs/blade_measurement_new/extracted.json --catalog-output runs/blade_measurement_new/catalog.json
```

来源哈希改变时必须先审阅新来源和清单，不自动沿用旧图号及比例尺。

## 显式转换到工作台

`softcut_solution.blade_config` 接收用户选定的既有 `softcut-config/1` 作为基础，
保持其材料、接触、加载、网格、预算，只替换刀形。这批截面方位已确认，
转换直接读取配置中的确认状态；仍须接受对称楔形近似，并给出**独立提供的**
鼻部半径及来源。其他尚未确认方位的来源仍保留确认检查。
特别是 C 明显不对称，对称化会损失几何信息。五份测量配置均保留两侧角度。

```powershell
python -m softcut_solution.blade_config --catalog configs/blade_profiles.measured.json --profile a_low_profile --base-config <已有工作台配置.json> --tip-radius-mm <独立测量或显式假设值> --radius-source <来源说明> --accept-symmetric-approximation --output runs/blade_measurement_new/a_low_workbench.json
```

转换使用真实工作台验证器；只生成配置，不调用求解器。没有测得半径时只能标注为
假设，不能称为真实刀刃重建。当前动态断裂入口有各自 schema，此桥接不自动修改它们。
已知对称/不对称楔形、标尺、配置门控和真实工作台参数传递由自动测试验证。
