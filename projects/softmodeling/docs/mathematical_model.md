# 完整体橡胶斜向压入：数学模型与代码对应

日期：2026-10-26。对象为当前原生P1、有限应变等容Maxwell、improved-max IPC及正则库仑摩擦主线。本文描述实际实现及其推导关系；不将参考论文的完整算法等同于本项目实现，不构成实验标定或一般收敛性证明。

物理输入见[模型规格](current_model_spec.md)，任务范围见[任务要求](task_requirements.md)，代码入口见[主体代码地图](../prototype/README.md)。文末记录实际数值核查及回归结果，详细数值和源码指纹保存在本地`outputs/mathematical_model_20261026/equation_checks.json`。源码有未提交工作区变更时，以文件指纹识别本文对应的实现，不以Git提交号单独替代工作区身份。

## 1. 范围、变量与假设

完整橡胶体按等温、准静态、二维平面应变处理。刀具为刚体，沿实验室竖直方向下移；无惯性、重力、热传导、黏附牵引、流体油膜、损伤、裂纹或裂后演算。润滑条件通过指定正的库仑系数近似。目标行程不超过2 mm，速度固定0.2 mm/s。

| 符号 | 定义及单位 |
|---|---|
| X、x、u | 材料参考位置、当前位置、位移，mm；x=X+u |
| F、J | 三维嵌入变形梯度、其行列式；F₂为面内2×2块 |
| B | 面外参考宽度20 mm；用于体积和接触积分，不产生自由侧面 |
| A | 逆黏性度量Cᵥ⁻¹，无量纲、对称正定、det A=1 |
| τ、σ、P | Kirchhoff、Cauchy和第一Piola应力，MPa；τᵣ另表示松弛时间 |
| W、Ψ | 参考体积储能密度、材料约化增量势密度，N/mm² |
| d、Δt | 向下为正的竖直刀具行程、真实材料时间步，mm、s |
| r、K_alg | 离散自由力残差N、真实耦合切线N/mm |
| M、s | PTC辅助度量N/mm、无量纲稳定化系数；s不是物理时间 |
| δ、κ、εᵤ | IPC激活距离mm、屏障刚度MPa、摩擦位移平滑尺度mm |

面内公开坐标为(y,z)，y向右、z向上；求解器数组中的历史名称x对应公开y。以下F的张量指标不表示公开横坐标名称。面外方向统一记3。

## 2. 几何、刀路与连续平衡

### 2.1 参考区域和位移边界

局部参考矩形为0≤s₀≤30 mm、0≤n₀≤5 mm。角度θ是试样初始表面相对水平的倾角；当前刀具自身保持竖直，没有随θ旋转刀具。

$$
e_s=(\cos\theta,-\sin\theta),\quad e_n=(\sin\theta,\cos\theta),\quad X=s_0e_s+n_0e_n.
$$

底面n₀=0的两个面内位移均为零；其余材料边界除接触作用外为自由边界。刀具99755采用厚0.245 mm、总楔角23°、圆鼻半径0.02 mm的对称刚性多边形，圆鼻用64段折线近似。半径为模型输入，未独立测量。

设名义接触点Xc=7eₛ+5eₙ、圆鼻半径a、竖直单位向量e_z=(0,1)，则理想圆心和刀具轨迹为：

$$
c(d)=X_c+a e_n-d e_z,\qquad x_b(d)=x_b(d_{\rm start})-(d-d_{\rm start})e_z.
$$

圆鼻多边形与理想圆弧的名义首触存在离散差别；几何最低点也不一定是相对倾斜表面的接触点。θ加半楔角必须小于90°，使初始接触落在所表示的圆鼻范围内。直刃可选细分只在原线段插点，不改变原圆鼻或多边形轮廓。

### 2.2 有限变形和平面应变

$$
F_2=I_2+\operatorname{Grad}_X u,\qquad F=\operatorname{diag}(F_2,1),\qquad J=\det F=\det F_2>0.
$$

F₃₃=1、F₁₃=F₂₃=F₃₁=F₃₂=0；面外应力σ₃₃通常不为零。平面应变不等于不可压缩；J由位移计算，未引入独立压力自由度或J=1乘子。

### 2.3 强形式与弱形式

体内无体力时满足Div P=0。接触牵引仅作用于当前边界，固定底面与刀具承担反力。以参考面元牵引T₀书写面内弱式：

$$
B\int_{\Omega_0}P:\operatorname{Grad}_X v\,dA-B\int_{\Gamma_{c0}}T_0\cdot v\,dS=0.
$$

v在固定底面为零。实际接触并不直接用该牵引积分求解，而是由第6节的离散屏障及摩擦残差代替。刀具位移是Dirichlet输入，不在自由材料平衡中施加指定的Fz。

## 3. 有限应变等容Maxwell材料

### 3.1 支路、内部变量与储能

平衡支路和一个非平衡支路并联。非平衡支路采用乘法分解；内部变量只记录逆黏性度量，不单独保存黏性转动：

$$
F=F_eF_v,\quad C_v=F_v^TF_v,\quad A=C_v^{-1},\quad b=FF^T,\quad b_e=FAF^T.
$$

等容黏性流动使det A=1，因此弹性支路体积比Jₑ=J。虽然F₃₃=1，A₃₃仍可演化；不能把内部变量强制为二维或令A₃₃=1。当前平面应变历史保持面内/面外块对角结构。

基线采用两支路指数α=2，即等容neo-Hookean能量。定义dev T=T−tr(T)I₃/3：

$$
W(F,A)=W_\infty(F)+W_v(F,A),\qquad W_\infty=\frac{G_\infty}{2}(J^{-2/3}\operatorname{tr}b-3)+\frac{K}{2}(J-1)^2.
$$

$$
W_v=\frac{G_v}{2}(J^{-2/3}\operatorname{tr}b_e-3).
$$

K只属于不松弛的体积支路，非平衡体积模量为零。二维面积压缩并不等于三维纯体积变形：F=diag(λ,λ,1)仍含偏变形，所以其总响应可以发生支路松弛。

二次J体积项本身在J→0时趋向K/2，不是防翻转的无限势垒。整个材料的等容项在某些平面应变压缩路径上仍可发散；不能由体积项的有限极限推断整个能量有界。几何有效性另由正J和CCD控制。

### 3.2 应力

对固定内部变量A求储能对F的偏导，得到瞬时应力：

$$
\tau_\infty=G_\infty J^{-2/3}\operatorname{dev}b+KJ(J-1)I_3,\qquad \tau_v=G_vJ^{-2/3}\operatorname{dev}b_e.
$$

$$
P=(\tau_\infty+\tau_v)F^{-T},\qquad \sigma=\frac{1}{J}(\tau_\infty+\tau_v).
$$

材料更新返回P的面内块和面外Kirchhoff应力。接口名称`plane_stress_residual`在平面应变下保存τ₃₃诊断量，不要求它为零；主线调用明确使用`kinematics='plane_strain'`。

### 3.3 连续黏性演化及耗散

采用偏应力驱动的正迁移率。令l=dot F F⁻¹，弹性左度量的Lie导数为dot bₑ−lbₑ−bₑlᵀ，则对应的等容Maxwell流动写为：

$$
\dot b_e-lb_e-b_el^T=-\frac{1}{G_v\tau_r}\tau_v b_e,\qquad \dot A=-\frac{1}{G_v\tau_r}F^{-1}\tau_v b_eF^{-T}.
$$

对各向同性支路τᵥ与bₑ共轴，右式对称；tr τᵥ=0使det A保持。材料黏性耗散率密度为：

$$
\mathcal{D}_v=\frac{\tau_v:\tau_v}{2G_v\tau_r}\geq0,\qquad P:\dot F=\dot W+\mathcal{D}_v.
$$

该连续式说明当前流动的物理含义；实际时间离散使用下一节的对数主应变返回，不是直接对A做后向Euler。`paper2023`复现采用另一套迁移率/体积流动约定，不能代替本主线。

### 3.4 小应变极限与参数含义

$$
G_0=G_\infty+G_v,\quad q=\frac{G_\infty}{G_0},\quad G(t)=G_\infty+G_v\exp(-t/\tau_r).
$$

此G(t)适用于小应变剪切松弛；不是大变形压入力的乘法衰减公式。固定K时，小应变瞬时/长期泊松响应分别由G₀/G∞与K给出：

$$
\nu(G)=\frac{3K-2G}{2(3K+G)}.
$$

基线G∞=0.14、Gᵥ=1.24、K=23.34333333333331 MPa，τᵣ=1.7 s；瞬时ν约0.471、长期约0.497。`Material.poisson`按平衡剪切模量编码K，不是整个松弛过程保持不变的泊松比。

## 4. 材料时间积分、增量势与一致切线

### 4.1 弹性试算和局部方程

物理步n→n+1固定已提交Aⁿ。由当前F构造bₑ,tr=FAⁿFᵀ，取其特征向量V和三个对数试算主伸长eᵗʳ；面外试算量为ln(Aⁿ₃₃)/2。记D=I₃−11ᵀ/3：

$$
L=\frac{\Delta t}{2G_v\tau_r}D,\qquad r_e=e-e^{\rm tr}+L\tau_v(e)=0.
$$

这里τᵥ(e)是主Kirchhoff偏应力向量。α=2时，令vᵢ=exp[2(eᵢ−mean e)]、v̄=mean v：

$$
\tau_{v,i}=G_v(v_i-\bar v),\qquad (H_v)_{ij}=2G_v[\delta_{ij}v_i-(v_i+v_j)/3+\bar v/3].
$$

局部Newton使用I+LHᵥ。代码保留一个厚度对数伸长变量η（源码变量名q）和三个e，平面应变方程是η=0；不是求τ₃₃=0。其4×4系统等价于固定η后的上述三变量方程。

局部最多35次检查/更新循环，单次对数变量修正的最大绝对值限制为0.4；主应变残差阈值为1e−11·max(1,max|eᵗʳ|)。保留的厚度/应力尺度判据为1e−9+64εmach·max(|H∞|,|Hᵥ|)。这些局部判据不替代全局1e−4 N力验收。

收敛后保持试算主方向，重构更新度量：

$$
b_e^*=V\operatorname{diag}(\exp(2e))V^T,\qquad A^{n+1}=F^{-1}b_e^*F^{-T}.
$$

面内按2×2谱分解实现，A₃₃另由exp(2e₃)恢复。该度量在非线性迭代中只是试算值；物理步接受前不提交。

### 4.2 为什么装配增量势而不是总储能导数

L在偏空间正定、在球空间为零。令L†为其偏空间逆，对满足sum e=sum eᵗʳ的变量取局部最小：

$$
\Psi(F;A^n,\Delta t)=W_\infty(F)+\min_e\{W_v(e)+\frac{1}{2}(e-e^{\rm tr})^TL^\dagger(e-e^{\rm tr})\}.
$$

其驻值方程就是rₑ=0；在已收敛返回映射上：

$$
\Psi=W+\frac{1}{2}\tau_v^TL\tau_v=W+\frac{\Delta t}{2}\mathcal{D}_v,\qquad \frac{\partial\Psi}{\partial F}=P.
$$

内部变量已消元的储能W(F,A*(F))之全导数还含W_A:dA*/dF，通常不等于P。装配中的`incremental_potential`是Ψ，输出的`stored_energy`是W，二者不可互换。

### 4.3 一致材料切线

固定Aⁿ和Δt，微分局部方程得到：

$$
de=(I+LH_v)^{-1}de^{\rm tr},\qquad C_{iJkL}^{\rm alg}=\frac{\partial P_{iJ}}{\partial F_{kL}}.
$$

完整C还包括平衡支路、F⁻ᵀ的微分及试算特征向量变化。源码使用谱差商并在重根处用解析极限，避免仅微分主应力而遗漏主方向转动。在材料光滑有效范围内，C对应约化势Ψ的Hessian；全局摩擦切线则另有非对称项。

具体地，对任意面内δF，先计算两套谱输入的变化：

$$
\delta b=\delta F F^T+F\delta F^T,\qquad\delta b_{e,\rm tr}=\delta F A^nF^T+FA^n\delta F^T.
$$

在对应特征基Q、V内，δe∞,i=(QᵀδbQ)ᵢᵢ/(2bᵢ)、δeᵗʳᵢ=(Vᵀδbₑ,trV)ᵢᵢ/(2bₑ,tr,i)。面外总伸长固定，δη=0。平衡主应力导数H∞由同一等容公式将Gᵥ换为G∞，再给每个分量加KJ(2J−1)；δτ∞=H∞δe∞，δτᵥ=Hᵥδe。

对i≠j，谱基内应力矩阵的非对角变化为：

$$
(Q^T\delta\tau_\infty Q)_{ij}=\frac{\tau_{\infty,i}-\tau_{\infty,j}}{b_i-b_j}(Q^T\delta bQ)_{ij},\qquad (V^T\delta\tau_vV)_{ij}=\frac{\tau_{v,i}-\tau_{v,j}}{b_{e,\rm tr,i}-b_{e,\rm tr,j}}(V^T\delta b_{e,\rm tr}V)_{ij}.
$$

重根处，平衡差商用(H∞,ii−H∞,ij)/(2bᵢ)，非平衡差商用(Hᵥ,alg,ii−Hᵥ,alg,ij)/(2bₑ,tr,i)，其中Hᵥ,alg=Hᵥ(I+LHᵥ)⁻¹。将两套应力变化各自转回空间基后：

$$
\delta P=(\delta\tau_\infty+\delta\tau_v)F^{-T}-P\delta F^TF^{-T}.
$$

对四个面内单位δF分别求值即可得到4×4算法切线；该推导也规定有限差分核查时必须保持Aⁿ、Δt不变，并重新完成每个扰动的材料返回。

物理步Δt=Δd/0.2。固定F的小剪切松弛经当前离散后的非平衡应力逐步比值为1/(1+Δt/τᵣ)，不是每一步精确乘exp(−Δt/τᵣ)。减小物理步会改变积分误差；PTC内部迭代不改变Δt。

## 5. P1空间离散与体残差

参考三角形中用线性形函数Nₐ。令Dₘ=[X₂−X₁,X₃−X₁]、Dₛ=[x₂−x₁,x₃−x₁]，则单元内F₂恒定：

$$
F_{2,e}=D_sD_m^{-1},\qquad V_{0e}=\frac{B}{2}\det D_m,\qquad \nabla_XN_a=\nabla_{\xi}N_aD_m^{-1}.
$$

每单元一个材料历史点；常F单元的参考积分直接用V₀ₑ乘密度。体残差和未投影切线为：

$$
r_{ai}^{\rm bulk}=\sum_e V_{0e}P_{iJ}^{e}N_{a,J}^{e},\qquad K_{ai,bk}^{\rm bulk}=\sum_e V_{0e}N_{a,J}^{e}C_{iJkL}^{e}N_{b,L}^{e}.
$$

只装配材料自由度，刀具没有体能。当前为原生位移P1，不是混合u–p元、高阶元或降阶积分修正。近不可压约束、网格形状与表面离散误差仍需专门验证，不能由正J推断空间精度充分。

## 6. 法向接触：实际屏障与积分

### 6.1 理想条件和实现的区别

理想无黏附接触要求间隙g≥0、压应力p≥0及pg=0；库仑切向牵引满足|tₜ|≤μp，滑动时反向于相对切向速度。当前用正间隙屏障和位移平滑摩擦近似这些条件，不精确求解零间隙互补系统。

边界由材料边界边与完整刀具多边形组成。过滤器阻止刀具—刀具碰撞，保留材料—刀具及材料自接触。二维使用EV及VV特征、最近点分类和improved-max组合，未用“只检测刀尖”代替整刃/本体碰撞。

### 6.2 平方距离屏障和宽度因子

令dₖ是特征的欧氏距离、aₖ=dₖ²，δ为距离激活阈值。当前最小间隙参数dmin=0，原始库函数作用在平方距离上：

$$
b(a;h)=-(a-h)^2\ln(a/h),\quad 0<a<h,\qquad h=\delta^2.
$$

a≥h时b=0；a≤0时势值发散。`BarrierPotential(..., use_physical_barrier=True)`还包含单位归一化，不能直接把上式乘κ当作实际接触势：

$$
\Phi_N(x)=\sum_{k\in\mathcal{C}(x)}w_k\,\kappa B\,\frac{\delta}{(\delta^2)^2}\,b(d_k^2;\delta^2).
$$

二维wₖ单位为mm，κB为N/mm，因此ΦN单位为N·mm。当前δ=0.0025 mm、κ=6.6 MPa，后者为固定数值屏障输入，不随材料剪切模量自动缩放。正间隙内已有排斥力，不能把激活距离当成穿透容许值。

权重来自参考边界长度：顶点基本权重为相邻参考边长之和的一半；EV/VV再按方向计数、重复特征合并及improved-max修正构建。组合中可有负的修正权重，不能把每个项都解释为独立非负物理接触力，也不能任意删除VV保留其余项。

### 6.3 法向残差、切线与法向载荷

在当前特征分支上，wₖ为参考权重，EV/VV的距离梯度与Hessian均参与：

$$
r_N=\nabla_x\Phi_N,\qquad H_N=\nabla_x^2\Phi_N,\qquad \lambda_k^N=-2d_k\kappa B\,\frac{\delta}{(\delta^2)^2}\,b'(d_k^2;\delta^2).
$$

λᴺ是未乘wₖ的正法向载荷密度，二维单位N/mm；单项乘权重后才有N量纲。接触特征和最近点会随当前构型重建，切线是该分支的局部线性化。防穿透和单分支导数正确不证明特征切换处具有任意步长下的准确线性模型。

令cₖ=wₖκBδ⁻³，在固定EV/VV分支上：

$$
r_N=\sum_kc_kb'(a_k)\nabla a_k,\qquad H_N=\sum_kc_k[b''(a_k)\nabla a_k\nabla a_k^T+b'(a_k)\nabla^2a_k].
$$

0<a<h区间内的标量导数为：

$$
b'(a;h)=(h-a)[2\ln(a/h)-h/a+1],\qquad b''(a;h)=(h/a+2)h/a-2\ln(a/h)-3.
$$

EV线段内最近点β=(xᵥ−xₑ₀)·(xₑ₁−xₑ₀)/|xₑ₁−xₑ₀|²；落在线段外时分类为相应VV距离。距离梯度和Hessian由此包含最近点几何变化，不是只对一个固定法向间隙弹簧求导。

## 7. 摩擦运动学、残差及真实耦合切线

### 7.1 上一接受位置与当前接触几何

每个物理增量固定全部上一接受位置xⁿ。对当前试算x重建接触数据ℓ(x)：特征、最近点坐标β、切向基pₜ、法向载荷和权重。EV特征中：

$$
\Delta x=x-x^n,\quad v_k^{\Delta}=\Delta x_v-(1-\beta)\Delta x_{e0}-\beta\Delta x_{e1},\quad \xi_k=p_t^Tv_k^{\Delta}.
$$

这里vΔ虽然沿用库的`velocities`形参名，实际传入的是mm单位的位移增量；没有除Δt。VV用两个顶点增量之差。该量不是独立表面材料地址，也不是跨步累计的界面滑程。

### 7.2 平滑库仑势

令ρ=|ξ|、ε=εᵤ=1e−5 mm。μ静、动系数相同，未启用各向异性、速度弱化或黏附。平滑函数为：

$$
\psi_\varepsilon(\rho)=\frac{\rho^2}{\varepsilon}-\frac{\rho^3}{3\varepsilon^2}+\frac{\varepsilon}{3},\quad 0\leq\rho<\varepsilon.
$$

ρ≥ε时ψε(ρ)=ρ；平滑区导数ψε′=2ρ/ε−ρ²/ε²，区外为1。ρ=0时势值ε/3但梯度为零，因此势值本身不是累计摩擦耗散。

冻结ℓ后定义局部摩擦增量势。令Γₖ把局部节点增量映为相对增量，Tₖ=Γₖᵀpₜ，因此ξₖ=TₖᵀΔxₖ：

$$
\Phi_F(x;x^n,\ell)=\sum_k w_k\mu\lambda_k^N\psi_\varepsilon(|\xi_k|),\qquad r_F=\left.\frac{\partial\Phi_F}{\partial x}\right|_\ell.
$$

$$
r_{F,k}=w_k\mu\lambda_k^N T_k\,\frac{\psi_\varepsilon'(\rho_k)}{\rho_k}\xi_k,\quad \frac{\psi_\varepsilon'(\rho)}{\rho}=\frac{2}{\varepsilon}-\frac{\rho}{\varepsilon^2}\ \ (\rho<\varepsilon).
$$

物理摩擦力是负残差。零点用上述有限极限处理。平滑区是连续阻力近似，没有独立切向弹簧、黏着参考点更新或黏附分离功；近零滑动状态不能直接等同真实黏附。

等效速度平滑尺度为εᵤ/Δt：名义Δd=0.01 mm时Δt=0.05 s，对应0.0002 mm/s。缩小真实加载步会改变该尺度；PTC的α缩小不改变它。`FEMConfig.friction_velocity_regularization_mm_s`不是本入口实际使用的εᵤ。

### 7.3 刷新残差与非对称切线

令S将自由材料坐标增量嵌入全部坐标，Sᵀ抽取自由残差。实际求解的是：

$$
R(x)=S^T[r_{\rm bulk}(x;A^n,\Delta t)+r_N(x)+r_F(x;x^n,\ell(x))]=0.
$$

先冻结ℓ取得Hfr，然后补接触数据变化的链式项：

$$
K_{\rm alg}=\frac{dR}{dx_f}=S^T[H_{\rm bulk}+H_N+H_F^{\rm frozen}+K_F^{\rm coupling}]S.
$$

耦合项包括切向基、最近点和法向载荷等对当前构型的变化；代码通过库的`force_jacobian(...LAGGED_DISPLACEMENTS)`取力导数并转换残差符号。该切线通常非对称，不能进行对称化或PSD投影后仍称为真实残差Jacobian。

该链式项可在单个固定特征分支上直接展开。令γ(ρ)=ψε′(ρ)/ρ、ξ=TᵀΔx；以下δ表示构型微分，不是接触激活距离：

$$
\delta\xi=(\delta T)^T\Delta x+T^T\delta x,\qquad Q_\xi=\gamma(\rho)I+\frac{\gamma'(\rho)}{\rho}\xi\xi^T.
$$

$$
\delta r_{F,k}=w_k\mu[(\delta\lambda_k^N)T\gamma(\rho)\xi+\lambda_k^N(\delta T)\gamma(\rho)\xi+\lambda_k^N TQ_\xi\delta\xi].
$$

ρ=0取连续极限。保持λᴺ、T固定，仅保留δξ=Tᵀδx，得到冻结摩擦Hessian；余项就是刷新几何耦合。在二维单切向、ρ≥ε的滑动区，ψε″=0，冻结摩擦Hessian为零，但λᴺ和T变化的耦合仍可非零。有限摩擦力不意味着冻结切向刚度非零。

刷新后的rF是冻结几何势梯度在ℓ=ℓ(x)处的取值，**不是**对标量ΦF(x;ℓ(x))做全导数。`evaluate()`返回的材料增量势+屏障+冻结摩擦势可作条件诊断，但一般不是刷新残差R的全局势。因此摩擦主线采用力残差PTC，不要求该标量每步单调下降。

## 8. 非线性求解与全局化

### 8.1 初猜及真实加载步

对Δd构造仿射初猜：材料节点竖直下移Δd·n₀/H，刀具下移Δd，底面保持。它是初猜，不是给上表面施加位移约束。有两个接受态时计算割线外推，再以权重1、1/2、1/4、1/8与仿射混合；加上纯仿射，共同经过边界、正J、CCD及刷新残差检查，选择有效且残差最小的初猜。

### 8.2 PTC方向与辅助度量

在同一个当前试算状态装配真实残差及未投影耦合切线。材料单元Hessian和接触冻结Hessian分别经负特征值截零，装配成辅助PSD度量，再加δM=1e−9 N/mm的对角项：

$$
M=S^T\widetilde H_{\rm PSD}S+\delta_M I,\qquad (K_{\rm alg}+sM)\Delta x_f=-R.
$$

s初值0.1。它只改变方向，不加入最终平衡方程、材料黏性或真实时间。原生后端调用SuperLU；可选PARDISO及受控LU预条件复用求解同一当前线性系统。原生直接分支不另设强制线性残差验收，诊断可记录该误差；性能后端有额外的原矩阵残差检查。

### 8.3 几何及位移限制

对每个单元，沿更新方向的行列式是二次式：

$$
\det(F_2+\alpha D_F)=c+b\alpha+a\alpha^2,\quad c=\det F_2,\quad a=\det D_F.
$$

取(0,1]内最早的零点并乘0.9安全系数，得到αJ；无零点时不限制。完整CCD给出αCCD。PTC还采用圆鼻半径a_tip作为特征长度：

$$
\alpha=\min(1,\alpha_J,\alpha_{\rm CCD},\frac{a_{\rm tip}}{2\|\Delta x_f\|_\infty}).
$$

此处限制的是所有自由材料坐标的最大迭代位移；PTC中刀具已固定在本物理步目标位置，不是再次缩小刀具行程。正J控制检查整条直线路径，不能只检查方向两端的行列式。

### 8.4 接受规则、稳定化及停滞

令Rₜ=R(x+αΔx)、Rpred=R+αKalgΔx。默认接受线性模型误差合格且位置确实变化、残差有限的内部步：

$$
\|R_t-R_{\rm pred}\|\leq0.25\max(\|R\|,\|R_t\|),\qquad\mathrm{or}\quad\|R_t\|\leq10^{-4}\ {\rm N}.
$$

接受后默认更新：

$$
s_{\rm next}=\operatorname{clip}\left(\frac{s\|R_t\|}{1.2\max(\|R\|,10^{-15})},10^{-10},10^6\right).
$$

拒绝时s←max(0.1,2s)，同一方向生成阶段最多18次稳定化重试。允许内部残差暂时上升；模型误差小或s下降不保证取得足够净进展。

可选`residual-merit`是额外接受通道：仅当默认模型误差未通过时，要求残差至少降低10%、RᵀKalgΔx<0且满足系数1e−4的平方残差Armijo条件。最终为两通道的逻辑OR，不是否决默认已接受步的过滤器；该通道接受后保持s≥max(0.1,原s)。

每次尝试最多500次方向更新，可检查第0至500次残差。割线尝试启用40次进展窗口：相对记录的进展基准未再降低至少10%则退出；同增量仿射恢复不启用该40次早退。两种尝试都受共同墙钟预算约束。

### 8.5 恢复与二分

割线尝试发生可恢复`LoadStepFailure`后，回到同一接受xⁿ、Aⁿ和相同Δt，以仿射初猜/默认模型误差重试。两次均失败才允许物理二分，最小增量0.000625 mm；每个接受子步都提交、保存自己的真实时间和双历史。超时、材料积分异常或未预期实现错误不会被自动吞掉后追加预算。

准备态不允许用负深度递归二分替代原准备求解。默认分离起点dstart=−2δ/max(cosθ,0.1)；从该点推进至名义d=0同样有正的材料Δt及可能的预变形。记录`time_s=max(0,d)/v`不表示准备阶段的真实积分时间为零。

## 9. 物理步验收、提交及独立审查

物理步只在以下条件通过后提交：刷新自由力残差≤1e−4 N；J>0；最小接触距离为正且完整几何无相交；材料度量有限、正定、det误差≤1e−8；材料耗散率不低于−1e−10，摩擦增量功不低于−1e−12。刀具及固定底面还由边界钩子和独立保存态审查核对。

提交顺序为：固定xⁿ/Aⁿ → 材料试算与接触平衡 → 几何/材料验收 → 提交Aⁿ⁺¹、累计耗散和预测器接受历史 → 保存位置及准确前一步历史。失败试探不提交内部变量和累计耗散。

独立审查逐态使用保存的`previous_positions`、`previous_material_metric`、`material_dt_s`重新计算；检查它们与上一接受状态精确衔接。材料度量重现采用rtol=1e−10、atol=1e−11，反力重现采用rtol=1e−9、atol=1e−8 N。只将连续通过前缀列为合格；后续孤立通过态不补接历史。

失败快照必须配对该次尝试实际起点`failed_previous`，不能使用控制器回滚后可能更早的`previous`。PTC保存方向还必须与其基点残差、稳定化和正则值配对，才能重放线性方向。

## 10. 反力、功、应力场及接触观测

### 10.1 反力与角度投影

全局残差g包含刀具自由度上的接触梯度；报告的刀具反力为作用于刀具的物理接触力：

$$
F_y=-\sum_{b\in\mathrm{blade}}g_{b,y},\qquad F_z=-\sum_{b\in\mathrm{blade}}g_{b,z},\qquad F_{yz}=\sqrt{F_y^2+F_z^2}.
$$

Fz向上为正，d向下为正。相对初始表面定义N=en·F、T=−es·F，则：

$$
N=F_y\sin\theta+F_z\cos\theta,\quad T=-F_y\cos\theta+F_z\sin\theta,\quad F_z=N\cos\theta+T\sin\theta.
$$

这与逐特征的法向/摩擦分解不同，不能对全局T设T=μN。竖直行程投影为d cosθ和d sinθ，但切向传力、变形几何及材料历史同时变化，因此该投影不证明每个深度的Fz都严格随θ递减。

### 10.2 能量和耗散账目

$$
U=\sum_eV_{0e}W_e,\qquad\Delta D_v=\Delta t\sum_eV_{0e}\mathcal{D}_{v,e},\qquad\Delta D_F=r_F^T(x^{n+1}-x^n).
$$

材料耗散按后向Euler端点近似积分；摩擦功使用刷新终点的冻结梯度与实际增量。由于improved-max组合权重、特征切换和平滑，单项不能一律解释为独立的非负真实耗散；代码检查的是装配后的增量功。

刀具无横移时，驱动器克服反力所做输入功按∫Fz dd计算，不能用∫Fyz dd。U、材料耗散、摩擦功和法向屏障能分别报告；势中ε/3常数、屏障及时间/外功积分误差均可能进入离散功账目，不宣称机械储能+两种耗散已严格闭合。

### 10.3 场与接触地址

Cauchy应力由已提交A恢复；Green应变E=(FᵀF−I₃)/2，主值包含面外零值。工作台主拉应力显示量为max(λmax(σ),0)，不是未经裁剪的完整应力。P1原始单元场与平滑显示值分开。

刀尖接触观测不影响求解方程。它通过当前接触附近参考边的插值追踪原表面地址；唯一偏移、歧义区间、圆鼻支撑和整刃足迹的定义见[接触材料位置](tip_contact.md)。该地址迁移不是材料位移，也不是累计接触滑程；找不到唯一接触时保持缺失。

## 11. 已知近似、配置层级及解释边界

| 内容 | 当前实际含义 |
|---|---|
| 体积与宽度 | F₃₃=1、B仅作参考积分宽度；不描述侧面外鼓或沿刃压力不均 |
| 有限屏障距离 | 共同刚体平移不会产生相对位移；有限间隙上的共同转动可产生表观切向增量 |
| 转动检验 | 对同转角φ的EV点/边，间隙g下当前切向增量大小为g·|sinφ|；这可进入当前摩擦功，并非真实零间隙接触的滑移证据 |
| 相邻VV | 受压连续边界可能产生额外排斥；直接删除会改变improved-max组合，原修补未通过准入 |
| 摩擦正则 | 真实步长改变εᵤ/Δt，近黏着不等于具有黏附力 |
| 材料批处理 | 改变局部停止所用的批内极值尺度，历史上曾触发严格重现差异；参数相同不自动保证数值历史可混用 |
| PTC | 稳定化不改变目标残差，但模型误差接受规则没有一般净进展保证；失败不证明根不存在 |
| 参数识别 | 同一材料跨角度共用参数；力值接近不等于位移、能量或接触机制已经验证 |
| 实验后段 | 可能已裂；完整体结果不表示能解释裂后的力和形态 |

当前输入优先级为工作台材料/刀具及任务配置 → `run_material_candidate` → `material_case.make_problem` → 主线对象。`ipc_baseline.json`是早期弹性入口和若干数值常量的来源，不能整表当当前Maxwell工况；共同基线由`intact_2mm_baseline.json`定义。

`FEMConfig`为旧签名兼容保留的3 mm行程、1e−3 N容差、黏附/损伤/CZM、速度正则及旧迭代字段，不代表这些方程在当前入口生效。实际行程上限2 mm、全局力阈值1e−4 N、500次PTC及εᵤ由当前入口/数值配置决定；断裂参数不进入本文残差。

## 12. 实现流程、代码索引及验证边界

```text
读取共同材料 / 刀具 / 网格 / 加载配置，记录来源与指纹
生成未变形P1材料和分离刀具；检查相交及初始屏障诱导力
初始化 A=I、累计耗散=0，从分离态建立准备历史
对每个明确目标行程：
    固定上一接受位置和A，计算真实Δt
    割线/仿射候选经完整几何保护后选初猜
    对每次PTC更新：
        从同一A_old做材料返回，装配残差/真实切线/PSD度量
        解当前稳定化系统；执行正J、CCD和位移限制
        刷新接触/材料残差，检查局部模型并更新稳定化
    通过原物理验收后提交并保存准确双历史
    可恢复失败：同增量仿射；仍失败才按原规则真实二分
独立逐态重组审查，标出连续合格前缀，再导出曲线/场/接触量
```

| 数学内容 | 当前实现位置 |
|---|---|
| 倾斜坐标、形函数梯度和体积 | `prototype/model_config.py`：`_rotation_vectors`、`FEMMesh.bulk_data` |
| 网格与圆鼻 | `prototype/geometry.py`：`intact_mesh`、`blade_boundary_vertices`、`subdivide_blade` |
| 储能、主应力和局部返回 | `mechanics/finite_strain.py`：`Material.moduli`、`principal`、`response` |
| 材料装配、提交及物理时间 | `prototype/ipc_viscoelastic.py`：`material_bulk`、`material_acceptance`、`step` |
| 屏障、摩擦、真实切线及PTC | `prototype/ipc_reference.py`：`collisions`、`lag`、`evaluate`、`coupled_friction_tangent`、`solve_friction` |
| 正J、CCD和初猜/控制器 | 同上：`determinant_step`、`safe_step`、`_loading_predictor`、`adaptive_load_steps` |
| 检查点、失败配对及独立审查 | `prototype/material_case.py`：`failure_checkpoint`、`audit`；`ipc_viscoelastic.audit_state` |
| 参数入口与性能策略 | `tools/run_material_candidate.py`；`prototype/material_performance.py` |
| 保存场恢复 | `workbench/results.py`：`saved_state_fields`；不重新推进材料历史 |

IPC细节按安装的ipctk 1.6.0源码核查：`potentials/barrier_potential.cpp`、`barrier/barrier.hpp`、`friction/smooth_friction_mollifier.cpp`、`collisions/normal/*`及`potentials/tangential_potential.cpp`。本地运行库文件不随Git发布，其指纹和实测API结果写入本轮JSON。

验证分为：材料势—应力—切线/客观性/耗散及小剪切时间常数；体装配能量导数；冻结/刷新接触切线；平方距离屏障归一化、摩擦平滑和有限间隙转动；提交/回滚与实际历史审查。已有小测试通过只核查相应局部关系，不证明大行程必收敛、空间/时间已收敛或实验标定完成。

### 本次数值核查（2026-10-26）

以下材料点检查采用共同基线G∞=0.14、Gᵥ=1.24、K=23.34333333333331 MPa、τᵣ=1.7 s，物理Δt=0.05 s；使用具有自身预加载内部变量的三个非均匀变形梯度样本。它们是局部公式检查，不是新的压入算例。

| 核查 | 实际结果 |
|---|---|
| 已提交度量恢复的应力与材料核输出 | 最大分量差3.38e−14 MPa |
| Ψ=W+Δt·Dᵥ/2恒等式 | 当前样本最大差0 |
| Ψ对F的有限差分与P | 最大分量差6.27e−10 MPa；扰动1e−6 |
| P对F的有限差分与算法切线 | 最大分量差2.77e−9 MPa；扰动1e−6 |
| det A及试算纯度 | 最大det误差8.88e−16；输入历史逐位不变 |
| 连续流动式与离散返回的小步极限 | Δt=1e−3/1e−4/1e−5 s，相对差7.55e−4/7.56e−5/7.56e−6 |
| 小剪切保持的后向Euler松弛比 | 理论0.9714285714，最大比值差3.52e−11 |
| 物理屏障和摩擦平滑公式 | 与安装的库API值在1e−12相对/绝对容差内一致 |
| 有限间隙共同转动的EV切向增量 | 与−g·sinφ在1e−15 mm绝对容差内一致；测得非零表观摩擦功 |
| 刷新耦合切线与残差方向差分 | 小型Maxwell接触夹具相对差5.20e−9 |
| 同一夹具的冻结Hessian与刷新方向差分 | 相对差约0.325；表明此夹具中遗漏耦合不可忽略，不代表真实加载路径的占比 |

执行的已有项目回归为：

```powershell
python -m pytest -q tests/test_shared_material.py tests/test_viscoelastic_candidate.py tests/test_material_structure.py tests/test_ipc_paired_assembly.py tests/test_ipc_friction_merit.py tests/test_ipc_adaptive_load.py tests/test_workbench_results.py
```

结果为68项通过、22个子检查通过，耗时8.17 s。覆盖材料、装配、接触导数、历史提交/回滚及保存场恢复；含已有的小型加载夹具，没有启动新的长行程研究任务。本轮未改动求解方程或物理验收，也未重新读取/修改外部视频仓库。

## 13. 理论来源与阅读范围

有限黏弹性乘法分解/返回映射背景见Reese–Govindjee及2023论文；IPC屏障/摩擦与边界积分背景见R42/R17。原始文献、版本和既有核读范围集中于[统一引用导读](../references/README.md)，不另建重复文献清单。

本轮以代码和安装库为公式依据。额外打开[IPC Toolkit v1.6.0官方版本](https://github.com/ipc-sim/ipc-toolkit/tree/v1.6.0)及[Convergent IPC作者预印本记录](https://arxiv.org/abs/2307.15908)核对来源身份；未以网页摘要证明本项目的离散精度或非线性收敛。Reese–Govindjee DOI本轮直连未取得正文，其全文阅读等级沿用原引用记录，不新增全文核读声明。
