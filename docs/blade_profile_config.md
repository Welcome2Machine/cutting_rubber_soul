# 显微轮廓参数配置

来源为本机 `researchtopic/bladetype.pptx`，SHA256
`1b44cde22b7008c09ded49a107c02e8f4628f9527388f7902480064c32e189b7`。
五页分别为 C、E No.1、E No.2、A low performance、A high performance。
原 PPT、原图和逐像素提取结果留在本机；Git 保存提取程序、来源清单和参数配置。

`configs/blade_profiles.measured.json` 是 `softcut-blade-shape-catalog/1` 配置，
不是可直接运行的求解器配置。`apparent_profile` 保存表观轮廓；只有确认观测平面
垂直于切削刃后，宽度/夹角才可作为截面厚度/楔角。当前 `view_plane=unconfirmed`，
拍摄前后使用状态未知，A 与旧目录里的具体产品编号没有自动对应关系。

提取只使用内嵌灰度图的亮轮廓和红色 500 μm 标尺，约 10 μm/像素。
采用最大连通域、三个灰度阈值、亚像素交叉位置以及每侧连续两线段稳健拟合；
刀身中线倾斜需小于 5°。前五个像素不用于直线拟合，虚拟尖端为两侧斜面
外推交点，**不是实测圆鼻尖端**。表观斜面长度从该虚拟交点算起。
阈值、标尺两端各 ±1 像素及轮廓定位各 ±1 像素构成敏感性范围，
不是重复样品的置信区间，也不包含未知投影角、仪器标定及切片方位误差。
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
保持其材料、接触、加载、网格、预算，只替换刀形。它要求确认截面方位、
接受对称楔形近似，并给出**独立提供的**鼻部半径及来源；缺项即拒绝。
特别是 C 明显不对称，对称化会损失几何信息。五份测量配置均保留两侧角度。

```powershell
python -m softcut_solution.blade_config --catalog configs/blade_profiles.measured.json --profile a_low_profile --base-config <已有工作台配置.json> --tip-radius-mm <独立测量或显式假设值> --radius-source <来源说明> --confirm-view-plane --accept-symmetric-approximation --output runs/blade_measurement_new/a_low_workbench.json
```

转换使用真实工作台验证器；只生成配置，不调用求解器。没有测得半径时只能标注为
假设，不能称为真实刀刃重建。当前动态断裂入口有各自 schema，此桥接不自动修改它们。
已知对称/不对称楔形、标尺、配置门控和真实工作台参数传递由自动测试验证。
