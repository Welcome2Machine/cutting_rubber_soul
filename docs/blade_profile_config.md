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
原宏观图的固定亮度分割在这两张纹理近景上不稳定；本次已增加局部背景扣除和
多设置局部圆拟合，数值与适用范围见下文及 `configs/blade_tip_radius_estimates.json`。
近景按性能类别关联宏观配置，尚无明确刀具序列号或沿刃测点位置。
旧宏观测量数值保持；已得到图像局部等效半径，实际参与接触的物理半径仍未认证。

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

## 刀头近景的半径计算（2026-10-29）

此前只做标尺与定性形态比较，没有完成半径计算；本节为实际新增测量。
程序为 `softcut_solution/blade_tip_radius.py`，输入及人工几何窗口保存在
`configs/blade_tip_closeups.json`，审阅后的参数摘要为 `configs/blade_tip_radius_estimates.json`。
计算没有读取实验力，也没有依据高低性能标签改变阈值或选取较有利的拟合结果。

两张图均为 265×108 像素；100 μm 红色标尺横跨 50 像素单元，得到 2 μm/像素。
逐列取 y=[39,44)、[78,83) 两条背景带中位数，在竖直方向线性插值后扣除。
背景带整体平移 −2/0/+2 像素；相对背景轮廓阈值用 15/25/35/45（灰度量程 255）；
同时比较原始等值线顶点和 0.5 像素等弧长重采样。
局部窗口从各特征最左端向后取 2/3/4 像素（4/6/8 μm），最小化点到圆的径向距离。
拟合使用多个初值，并记录残差、条件数、角覆盖和约束命中；小残差本身不代表半径可靠。
等弧长插值只改善权重，不增加独立像素或光学分辨率。

| 来源/图像特征 | 局部等效半径中位数 | 图像处理范围 | 加入标尺端点各 ±1 像素后的范围 |
|---|---:|---:|---:|
| 图 1，低性能 A，整体局部鼻端 | 7.6 μm | 6.8–8.8 μm | **6.5–9.2 μm** |
| 图 2，高性能 A，上侧局部凸头 | 5.9 μm | 4.4–6.8 μm | **4.2–7.1 μm** |
| 图 2，高性能 A，下侧局部凸头 | 4.3 μm | 1.7–5.7 μm | **1.7–5.9 μm** |

每一行有 72 个处理组合，全部保留，并非 72 次独立测量。范围不是统计置信区间，
也未覆盖光学模糊、三维投影、测点变化与仪器标定误差。图 1 鼻端存在小平段，
这里是有限窗口内的等效圆，不是每一点具有同样曲率。图 2 下侧部分结果接近一个像素，
只能称“数微米尺度的候选局部曲率”，不能以 1.7 μm 的小数精度报告真实刃口。

图 2 的可见边界有两个凸头及中间凹入，因此额外进行了整段圆拟合，使用相同设置并
把窗口扩展到 5/6/8 像素。144 个组合的半径从约 16 μm 漂到约 399 μm，11 个组合
命中拟合参数边界，说明单圆解释不适合；这些数字只是失败诊断，不是物理半径范围。
其整体外包络看起来更圆，不能据此认定局部接触端更钝。局部曲率较小是一种有图像依据的
候选解释，但两图范围有重叠，且尚不知道哪个局部特征承担真实接触，不能宣称已解释性能差距。
目前无法把局部凸起鉴定为毛刺、卷刃、涂层或附着物。

本地完整拟合与审图：`runs/blade_tip_radius_20261029/measured_final/estimates.json`、
`a_tip_image_1.png`、`a_tip_image_2.png`。图中圆为固定 level=35、背景不平移、
深度 3 像素、等弧长取点的示例，故不等于上表所有设置的中位数。
重新提取使用未存在的新输出目录：

```powershell
python -m softcut_solution.blade_tip_radius --manifest configs/blade_tip_closeups.json --output runs/blade_tip_radius_new
```

独立验证包括已知半径的平移圆弧、梯度背景上的已知圆鼻栅格图、共线点拒绝、
近直线圆弧条件数和来源哈希拒绝。它们验证实现，不认证真实显微图的光学质量。
所有求解器的刀形仍保持原值，不能把图像局部圆无条件当作完整圆鼻替换。

## 半径怎样影响切割：原始文献与本项目判断

1. **起裂阶段。** McCarthy、Ní Annaidh、Gilchrist，2010，
   [On the sharpness of straight edge blades in cutting soft solids: Part II – Analysis of blade geometry](https://doi.org/10.1016/j.engfracmech.2009.10.003)
   （[作者全文](https://researchrepository.ucd.ie/server/api/core/bitstreams/ca936b39-6f89-4acb-90e3-f203da41eae8/content)）
   对 PU 的 1–5 μm 圆鼻进行了实验/模型比较；半径增大时起裂所需压入和力增大，
   半径是其锋利度指标的重要几何因素。其结论支持“小半径提高局部应力集中、降低起裂负担”的
   机制，但研究的是特定材料的起裂，不能把力值或线性斜率直接移植到本项目裂后过程。
2. **橡胶与重复使用。** Ryzí 等，2025，
   [Test parameter sensitivity of the Lake-Yeoh cutting method for measurement of intrinsic strength of rubber](https://link.springer.com/article/10.1007/s10704-025-00859-x)
   在 EPDM 的 Lake–Yeoh 测试中发现刃尖半径对所测切割能影响显著，宏观厚度无法替代
   微刃几何。两种涂层刀在重复测试中都稳定，却可得到不同的切割能水平；作者讨论了
   摩擦与涂层厚度的可能贡献。这说明“同刀重复稳定、刀间差别大”仍可由稳定的表面或
   微几何差异造成。该论文不能证明用户刀具有涂层，也不能将其材料与试验数值直接套用。
3. **裂后并非无限受益于减小半径。** Goda、Ma、Fregonese、Bacca，2024，
   [Cutting soft matter: scaling relations controlled by toughness, friction, and wear](https://doi.org/10.1039/D4SM00279B)
   （[作者稿摘要](https://arxiv.org/abs/2412.02021)）研究圆丝稳态切割：小于转变尺度时
   力对半径可不敏感，较大半径区间出现近线性关系，并与界面磨耗剪切阻力联系。
   这是机制参照，圆丝不是 A 楔刀，不能把该关系当作本项目总 Fz 的通用公式。
4. **半径不能独自决定起裂后的力曲线形状。** Moreno-Mateos、Steinmann，2026，
   [Cutting soft materials: how material differences shape the response](https://www.nature.com/articles/s41524-025-01869-y)
   联合讨论体变形、内聚断裂和接触耗散。其弹性体呈平滑的增力过渡，文中讨论的
   压力无关黏附/磨耗剪切也能贡献裂后阻力；并非所有材料都无掉力。这里需要借鉴的是
   机制分解，不能以给定曲线反推本项目参数，或把未知黏附项补到刚好不掉力。

对本项目，半径首先影响局部应力/应变与起裂门控，同时改变裂尖附近接触及裂后耗散。
“半径小一半，所以全过程 Fz 小一半”没有依据；当前图像也不支持这样定量外推。
低性能 A 的图像等效半径约 8 μm，和旧模型默认 **20 μm** 并不相同；旧值仍是历史假设。
高性能图不宜强行替换为一个较小的完美圆鼻，应该同时考察局部特征与整体包络。

建议后续控制顺序：

- 先完成现有实际切割的残差、能量与完整路径修复；几何变更不能掩盖原数值失败。
- 保留旧 20 μm 对照；固定宏观角度、材料、强度、Gc 和接触参数，独立扫描
  R=4/6/8/12/20 μm。4–8 μm 覆盖图像候选尺度，12 μm 是中间控制，20 μm 是旧基线，
  均须显式标为几何研究设置，不以实验力挑选最佳值。
- 小半径必须同时验证刀面离散、材料局部网格与接触正则化尺度；当前 .05–.1 mm
  裂路间距远大于数微米特征，不能用“修改半径字段”冒充已分辨鼻端应力。
  具体需要的局部细化由收敛验证决定，不能预先承诺一个 h/R 就一定合格。
- 将圆鼻、带平段鼻端与图像支持的局部非圆轮廓分开比较，报告起裂行程、起裂力、
  裂纹连通性、裂后斜率及接触/断裂/体耗散，而非只比较最终 Fz。
- 用更高倍率、同一沿刃位置的多测点图或 SEM 独立测量局部半径、平段/微刃和沿刃变化；
  表面处理、清洁状态及摩擦/黏附另行核查，记录刀具编号与使用前后状态。
  两把刀的单个近景不能代替刀间统计或证明因果。
