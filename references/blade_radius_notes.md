# 刀尖半径、压入反力与起裂力

核读2026-10-21。McCarthy, C. T., Ní Annaidh, A., & Gilchrist, M. D. (2010).
*On the sharpness of straight edge blades in cutting soft solids: Part II – Analysis of blade geometry.*
Engineering Fracture Mechanics, 77, 437–451.
[DOI](https://doi.org/10.1016/j.engfracmech.2009.10.003)；
[作者机构全文](https://researchrepository.ucd.ie/server/api/core/bitstreams/ca936b39-6f89-4acb-90e3-f203da41eae8/content)。
阅读范围：作者稿摘要、问题定义、4.2及4.3.1–4.3.2文字；未独立复现或数字化图像。

论文针对其聚氨酯/刀具体系，分开计算压入响应和达到失效条件的深度；1–5 μm半径变化明显影响起裂力和锋利度指标。
不能把“锋利刀起裂力较低”直接等同于任意未断裂深度下Fz必然较低，也不能将该体系半径当作99755实测值。

本项目据此把半径与固定楔角分离，并要求同一半径检验0°和33°。10/20/40 μm为围绕现有数值的探索范围；
较小半径必须考虑表面网格与接触激活层尺度。当前仅完整体计算，不套用论文失效阈值或材料常数。
具体边界见[对照方案](../docs/tip_radius_study.md)。
