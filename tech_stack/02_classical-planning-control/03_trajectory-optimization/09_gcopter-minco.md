# GCOPTER

## 1. 概述

GCOPTER 将凸走廊、MINCO 参数化与非线性优化结合，用中间位置和时间生成受几何及动力学限制的平滑轨迹。MINCO 负责系数映射，外层代价和变量变换负责约束与任务目标。

## 2. 基础与定义

### 2.1 变量与映射

几何参数 $q$ 和正段时间 $T$ 决定系数 $c=\mathcal M(q,T)$。外层变量还可通过 $q=\chi(z)$、$T=\eta(\tau)>0$ 将位置与时间映射到指定域。

### 2.2 目标函数

典型目标为

$$
J=J_{\rm energy}+\rho\sum_iT_i+
\sum_j\mu_j\int\phi(g_j(p,\dot p,\ldots))dt,
$$

其中 $g_j\le0$ 表示空间或动力学约束，$\phi$ 惩罚违反量。

## 3. 核心原理

### 3.1 几何约束

先生成凸安全走廊，选择连接点的可行区域。中间位置映射可保证路径点属于相应凸集；整段曲线还通过走廊约束与最终验证处理。

### 3.2 平坦性与运动约束

由位置高阶导数与偏航恢复推力、姿态和角速度。将运动约束写成轨迹导数的函数，可将其梯度累加到系数与段时间上。

### 3.3 梯度计算

系数梯度经过 MINCO 伴随系统传至 $q,T$，再经过变量变换传至外层变量：

$$
\nabla_zJ=\left(\frac{\partial\chi}{\partial z}\right)^\top\nabla_qJ,
\qquad
\nabla_\tau J=\left(\frac{\partial\eta}{\partial\tau}\right)^\top\nabla_TJ.
$$

固定阶数的带状求解保留线性规模的系数与伴随计算；约束求积成本还取决于采样数量。
<details class="kt-derivation" id="gcopter-transform">
<summary>变量变换与驻点</summary>

若 $q=\chi(z)$ 在邻域内可逆且雅可比非奇异，则 $\nabla_zJ=0$ 等价于 $\nabla_qJ=0$，局部极小性质也随局部可逆映射保留。

一般满射可能具有退化雅可比，此时参数空间的驻点需要进一步检查原空间梯度与可行性。几何可行域的映射性质和目标函数的优化性质应分别验证。

</details>

## 4. 实现与应用

执行顺序为：地图与路径初始化 → 凸走廊 → 连接点与时间初值 → MINCO 系数求解 → 约束求积及梯度 → 外层更新 → 原始约束验收。

官方实现中 `minco.hpp` 提供参数化与梯度，`gcopter.hpp` 连接走廊和外层优化，`flatness.hpp` 提供平坦性映射。检查能量、时间、各类惩罚及最大实际违反量，避免总目标掩盖单项约束问题。

## 5. 分析与总结

GCOPTER 的优势来自低维轨迹变量、带状线性代数和解析梯度。有限罚函数的解可能保留约束残差；验收应覆盖连续轨迹、模型映射与执行边界。

## 6. 关联与参考

- [GCOPTER 论文](https://arxiv.org/abs/2103.00190)
- [GCOPTER 官方实现](https://github.com/ZJU-FAST-Lab/GCOPTER)
- [MINCO 参数化](06_minco-parameterization.md)
- [安全飞行走廊](04_safe-flight-corridor.md)
