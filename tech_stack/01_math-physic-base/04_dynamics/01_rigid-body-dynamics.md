# 刚体动力学

## 1. 概述

刚体动力学将外力和力矩映射为质心平动与姿态变化。坐标系、惯量与角速度约定决定方程的表达形式。

## 2. 基础与定义

### 2.1 状态与参考系

$p,v$ 为惯性系位置和速度，$R$ 将机体系向量变换到惯性系，$\omega$ 为机体系角速度。质量 $m$ 恒定，$J$ 为质心处机体系惯量。

## 3. 核心原理

### 3.1 牛顿与欧拉方程

$$
\dot p=v,\quad m\dot v=F,\quad \dot R=R[\omega]_\times,\quad
J\dot\omega+\omega\times J\omega=\tau.
$$

$F$ 是惯性系合外力，$\tau$ 是机体系合外力矩。转动方程中的叉乘项来自旋转坐标系内角动量的变化。

### 3.2 惯性张量

$$
J=\int(\|r\|^2I-rr^\top)\,dm.
$$

坐标旋转后 $J'=RJR^\top$。参考点离开质心时，平行轴定理增加 $m(\|d\|^2I-dd^\top)$。

<details class="kt-derivation" id="rigid-euler">
<summary>角动量方程推导</summary>

惯性系角动量为 $h=RJ\omega$。求导得 $\dot h=R([\omega]_\times J\omega+J\dot\omega)$。由惯性系力矩 $R\tau=\dot h$，左乘 $R^\top$ 得到欧拉方程。

</details>

### 3.3 惯性测量

理想陀螺测量 $\omega$，加速度计测量比力 $R^\top(\dot v-g)$。实际测量还包含零偏与噪声；状态估计通过积分和外部观测校正这些误差。

## 4. 实现与应用

### 4.1 模型验证

无外力时检查动量守恒，无外力矩时检查惯性系角动量守恒。姿态积分通过指数映射保持旋转结构，比较不同时间步长下的能量与轨迹误差。

## 5. 分析与总结

平动与转动通过力的方向、作用点和姿态耦合。动力学模型与传感器测量模型共同决定状态预测。

## 6. 关联与参考

[Lynch 与 Park：Modern Robotics](https://hades.mech.northwestern.edu/index.php/Modern_Robotics)，第 8 章。

关联：[四旋翼动力学](./02_quadrotor-dynamics.md)、[坐标变换](../01_linear-algebra-geometry/02_coordinate-transforms.md)。
