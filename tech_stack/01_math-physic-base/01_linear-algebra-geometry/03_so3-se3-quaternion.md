# SO(3)、SE(3) 与四元数

## 1. 概述

旋转与刚体位姿位于非欧氏空间。李群表示描述完整变换，李代数描述局部增量，单位四元数提供紧凑的旋转存储形式。

## 2. 基础与定义

### 2.1 群与局部坐标

$SO(3)=\{R\mid R^\top R=I,\det R=1\}$，$SE(3)$ 由旋转和平移组成。对 $\phi\in\mathbb R^3$，$[\phi]_\times v=\phi\times v$。

### 2.2 四元数约定

使用标量在前的单位 Hamilton 四元数 $q=(w,v)$，表示机体系到世界系的主动旋转；$q$ 与 $-q$ 表示同一旋转。

## 3. 核心原理

### 3.1 旋转指数映射

令 $\theta=\|\phi\|$，Rodrigues 公式为

$$
\operatorname{Exp}(\phi)=I+\frac{\sin\theta}{\theta}[\phi]_\times+
\frac{1-\cos\theta}{\theta^2}[\phi]_\times^2.
$$

小角度下用级数计算系数。右扰动定义为 $R'=R\operatorname{Exp}(\delta\phi)$，左扰动定义为 $R'=\operatorname{Exp}(\delta\phi)R$。

<details class="kt-derivation" id="so3-point-jacobian">
<summary>旋转点的扰动雅可比</summary>

右扰动下 $R'p\approx R(I+[\delta\phi]_\times)p=Rp-R[p]_\times\delta\phi$，所以 $\partial(R'p)/\partial\delta\phi=-R[p]_\times$。左扰动给出 $-[Rp]_\times$。两种结果对应不同的增量坐标系。

</details>

### 3.2 四元数与插值

旋转向量对应

$$
q=\left(\cos\frac\theta2,\frac{\phi}{\theta}\sin\frac\theta2\right),\qquad
R(q)=(w^2-v^\top v)I+2vv^\top+2w[v]_\times.
$$

SLERP 沿单位球面插值。先按内积符号选择等价四元数，使插值采用较短弧；两个姿态很接近时用归一化线性插值处理数值极限。

### 3.3 刚体位姿增量

采用平移在前的增量 $\xi=(\rho,\phi)$，有

$$
\exp(\xi^\wedge)=\begin{bmatrix}\operatorname{Exp}(\phi)&J_l(\phi)\rho\\0&1\end{bmatrix},
$$

$$
J_l=I+\frac{1-\cos\theta}{\theta^2}[\phi]_\times+
\frac{\theta-\sin\theta}{\theta^3}[\phi]_\times^2.
$$

位姿伴随为 $\operatorname{Ad}_T=\begin{bmatrix}R&[t]_\times R\\0&R\end{bmatrix}$，用于在不同参考系之间转换局部增量。

## 4. 实现与应用

### 4.1 局部更新

状态估计中先在切空间求增量，再通过指数映射更新旋转。检查零增量、单位范数、旋转矩阵正交性，以及解析雅可比与有限差分的一致性。

## 5. 分析与总结

旋转存储、误差定义和增量方向共同决定公式。局部三维增量适合优化，矩阵和四元数适合表示完整姿态。

## 6. 关联与参考

[Lynch 与 Park：Modern Robotics](https://hades.mech.northwestern.edu/index.php/Modern_Robotics)，第 3 章。

关联：[坐标变换](./02_coordinate-transforms.md)、[FAST-LIO](../../02_classical-planning-control/01_perception-slam/02_fast-lio.md)。
