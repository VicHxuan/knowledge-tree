# 高斯分布与状态估计

## 1. 概述

高斯模型以均值和协方差描述不确定性，在线性变换与线性高斯观测下保持闭合。其条件分布为最小二乘与卡尔曼更新提供共同基础。

## 2. 基础与定义

### 2.1 多元高斯

对正定协方差 $P$：

$$
p(x)=\frac{\exp[-\tfrac12(x-\mu)^\top P^{-1}(x-\mu)]}{(2\pi)^{n/2}|P|^{1/2}}.
$$

独立噪声 $v\sim\mathcal N(0,R)$ 下，$z=Hx+v$ 的均值为 $H\mu$、协方差为 $HPH^\top+R$。

## 3. 核心原理

### 3.1 高斯条件分布

观测 $z$ 后：

$$
K=PH^\top(HPH^\top+R)^{-1},\quad
\mu^+=\mu+K(z-H\mu),\quad P^+=P-KHP.
$$

该静态更新在时序预测之后使用，即得到线性卡尔曼滤波的观测步骤。

### 3.2 信息形式

令 $\Lambda=P^{-1}$、$\eta=\Lambda\mu$，后验信息为

$$
\Lambda^+=\Lambda+H^\top R^{-1}H,\qquad \eta^+=\eta+H^\top R^{-1}z.
$$

<details class="kt-derivation" id="gaussian-completion">
<summary>平方配方与信息更新</summary>

负对数后验去掉常数后为

$$
\tfrac12(x-\mu)^\top P^{-1}(x-\mu)+\tfrac12(z-Hx)^\top R^{-1}(z-Hx).
$$

合并二次项得到 $\Lambda^+$，合并一次项得到 $\eta^+$。驻点满足 $\Lambda^+\mu^+=\eta^+$；协方差是后验二次曲率的逆。

</details>

## 4. 实现与应用

### 4.1 融合算例

标量先验均值 0、方差 4，直接观测为 3、噪声方差 1，则增益为 0.8，后验均值为 2.4、方差为 0.8。实现通过分解求解信息方程，并检查协方差对称性与正定性。

## 5. 分析与总结

线性高斯模型给出精确闭式后验。非线性线性化、非高斯噪声和相关观测会改变该结论的适用条件。

## 6. 关联与参考

[Goodfellow 等：Deep Learning](https://www.deeplearningbook.org/)，概率与线性代数章节。

关联：[矩阵分解](../01_linear-algebra-geometry/01_matrix-decomposition.md)、[卡尔曼滤波](../../02_classical-planning-control/01_perception-slam/01_kalman-filters.md)。
