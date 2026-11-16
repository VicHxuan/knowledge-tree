# 四旋翼几何控制

## 1. 概述

几何控制直接在旋转群上定义姿态误差，将位置跟踪所需的合力转化为目标姿态与推力，再用旋转动力学构造力矩。该表示保留姿态的几何结构，并适合大角度机动分析。

## 2. 基础与定义

### 2.1 刚体状态

位置、速度和姿态为 $p,v,R$，角速度 $\omega$ 在机体系表达。采用推力沿机体第三轴的模型

$$
m\dot v=fRe_3-mge_3,\qquad
J\dot\omega+\omega\times J\omega=\tau.
$$

### 2.2 跟踪目标

参考轨迹给出 $p_d,v_d,a_d$ 与偏航。位置误差为 $e_p=p-p_d$，速度误差为 $e_v=v-v_d$。

## 3. 核心原理

### 3.1 位置与推力

期望合力为

$$
F_d=-K_pe_p-K_ve_v+m(a_d+ge_3),\qquad b_{3d}=\frac{F_d}{\|F_d\|}.
$$

用 $b_{3d}$ 与参考偏航构造 $R_d$，总推力取 $f=F_d^\top Re_3$。该表达需结合推力可实现范围与 $F_d\ne0$ 的条件。

### 3.2 姿态误差

定义

$$
e_R=\frac12(R_d^\top R-R^\top R_d)^\vee,
\qquad e_\omega=\omega-R^\top R_d\omega_d.
$$

其中 $\vee$ 将反对称矩阵转换为三维向量，$\omega_d=(R_d^\top\dot R_d)^\vee$。

### 3.3 力矩控制

令 $\widehat\omega v=\omega\times v$，控制律为

$$
\tau=-K_Re_R-K_\omega e_\omega+\omega\times J\omega
-J\left(\widehat\omega R^\top R_d\omega_d-R^\top R_d\dot\omega_d\right).
$$

反馈项调节误差，后两项补偿刚体耦合与目标姿态变化。
<details class="kt-derivation" id="geometric-error">
<summary>角速度误差动力学</summary>

由 $\dot R=R\widehat\omega$ 与 $\dot R_d=R_d\widehat\omega_d$，可得

$$
\dot e_\omega=\dot\omega+\widehat\omega R^\top R_d\omega_d-R^\top R_d\dot\omega_d.
$$

代入力矩控制律后，抵消前馈项，得到 $J\dot e_\omega=-K_Re_R-K_\omega e_\omega$。姿态误差与角速度误差耦合，稳定性分析还需利用旋转误差函数及其适用域。

</details>

## 4. 实现与应用

从参考位置高阶导数生成 $R_d,\omega_d,\dot\omega_d$，计算位置和姿态误差，输出推力与力矩，再经执行器分配。检查坐标方向、惯量单位、旋转正交性、推力饱和和离散延迟。

## 5. 分析与总结

几何控制避免欧拉角坐标奇异，但旋转空间的拓扑仍限制连续全局渐近稳定化。标准误差在相对旋转为 $180^\circ$ 时存在额外临界点，稳定结论需采用对应吸引域条件。

## 6. 关联与参考

- [Lee、Leok 与 McClamroch：Control of Complex Maneuvers for a Quadrotor UAV using Geometric Methods on SE(3)](https://arxiv.org/abs/1003.2005)
- [微分平坦性与最小 Snap](../03_trajectory-optimization/02_differential-flatness-minimum-snap.md)
- [级联姿态与角速度控制](04_px4-controller-attitude-rate.md)
