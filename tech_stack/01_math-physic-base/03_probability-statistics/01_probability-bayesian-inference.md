# 概率论与贝叶斯推断

## 1. 概述

概率模型表达未知量与观测之间的关系。贝叶斯推断以先验和似然构造后验，再根据决策损失选取估计结果。

## 2. 基础与定义

### 2.1 随机变量与条件分布

连续变量使用密度 $p(x)$，离散变量使用概率质量。联合分布满足 $p(x,z)=p(z\mid x)p(x)$，边缘分布通过积分或求和消去其他变量。

### 2.2 矩与不确定性

$$
\mu=\mathbb E[x],\qquad P=\mathbb E[(x-\mu)(x-\mu)^\top].
$$

协方差描述二阶相关性；独立变量的联合分布可分解为边缘分布的乘积。

## 3. 核心原理

### 3.1 后验更新

$$
p(x\mid z)=\frac{p(z\mid x)p(x)}{\int p(z\mid x)p(x)\,dx}.
$$

$p(x)$ 为先验，$p(z\mid x)$ 为似然，分母是观测证据。条件独立观测的似然可相乘。

### 3.2 估计准则

平方误差损失的 Bayes 估计是后验均值。MAP 估计取后验密度最大点，最大似然估计只优化似然。

<details class="kt-derivation" id="bayes-squared-error">
<summary>后验均值的最优性</summary>

固定观测 $z$，令 $\mu_z=\mathbb E[x\mid z]$。展开得

$$
\mathbb E[\|x-a\|^2\mid z]=\mathbb E[\|x-\mu_z\|^2\mid z]+\|a-\mu_z\|^2.
$$

交叉项因条件均值为零而消失，第二项在 $a=\mu_z$ 时最小。

</details>

### 3.3 多源信息融合

给定状态后独立的观测满足 $p(z_1,z_2\mid x)=p(z_1\mid x)p(z_2\mid x)$。共享预处理、重复测量或共同噪声会引入相关性，融合模型需保留相应联合分布。

## 4. 实现与应用

### 4.1 推断流程

定义未知量与观测 → 建立先验及似然 → 计算后验 → 按损失选择估计 → 检查预测与观测的统计一致性。离散模型可检查归一化，连续模型可用采样和数值积分验证。

## 5. 分析与总结

概率推断的结论取决于条件独立假设和噪声模型。估计值与不确定性一起构成可用于后续决策的信息。

## 6. 关联与参考

[Goodfellow 等：Deep Learning](https://www.deeplearningbook.org/)，概率章节。

关联：[高斯估计](./02_gaussian-estimation.md)、[贝叶斯滤波](../../02_classical-planning-control/01_perception-slam/01_kalman-filters.md)。
