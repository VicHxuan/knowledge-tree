# 四旋翼动力学建模

## 1. 概述

四旋翼通过总推力和机体力矩改变位置与姿态。平动加速度依赖推力方向，姿态运动依赖力矩，构成轨迹跟踪的动力学基础。

## 2. 基础与定义

### 2.1 坐标与输入

采用世界系 $z$ 轴向上，$R$ 为机体到世界旋转，机体 $e_3$ 为正推力方向。$f\ge0$ 为总推力，$\tau$ 为机体系力矩，$g>0$ 为重力加速度大小。

## 3. 核心原理

### 3.1 耦合动力学

$$
m\dot v=-mg e_3+fRe_3,\qquad
J\dot\omega=\tau-\omega\times J\omega,\qquad\dot R=R[\omega]_\times.
$$

该模型采用刚体、集中推力近似；阻力和执行器动态可作为额外模型项。

### 3.2 控制分配

各转子推力近似为 $f_i=k_f\Omega_i^2$，反扭矩符号由旋向决定。几何布局构成分配矩阵：

$$
\begin{bmatrix}f\\\tau\end{bmatrix}=B\begin{bmatrix}\Omega_1^2&\cdots&\Omega_4^2\end{bmatrix}^\top.
$$

速度平方的非负与上限限制可实现的合力及力矩。

### 3.3 平坦输出与悬停

理想模型采用位置与偏航作为平坦输出。给定期望加速度 $a_d$：

$$
F_d=m(a_d+ge_3),\qquad f_d=\|F_d\|,\qquad b_{3d}=F_d/\|F_d\|.
$$

结合偏航方向构造期望姿态，高阶位置导数用于角速度与力矩。悬停满足 $f=mg$、$Re_3=e_3$；在该配平点附近可线性化。

<details class="kt-derivation" id="quadrotor-hover">
<summary>悬停附近的水平加速度</summary>

取小滚转 $\varphi$、小俯仰 $\vartheta$，偏航为零，$Re_3\approx(\vartheta,-\varphi,1)^\top$。令 $f=mg+\delta f$，一阶近似给出 $\ddot x\approx g\vartheta$、$\ddot y\approx-g\varphi$、$\ddot z\approx\delta f/m$。

</details>

## 4. 实现与应用

### 4.1 建模验证

核对电机编号、旋向、单位及推力轴。以悬停配平和单轴小扰动验证符号，再比较推力饱和与执行器响应对轨迹的影响。

## 5. 分析与总结

动力学定义输入能产生什么运动，平坦映射定义轨迹导数如何转为前馈量。可执行性还取决于输入范围与模型误差。

## 6. 关联与参考

[Lynch 与 Park：Modern Robotics](https://hades.mech.northwestern.edu/index.php/Modern_Robotics)。

关联：[微分平坦与最小导数轨迹](../../02_classical-planning-control/03_trajectory-optimization/02_differential-flatness-minimum-snap.md)、[执行器动力学](../../02_classical-planning-control/04_tracking-control/06_tracking-latency-and-actuator-dynamics.md)。
