# 动力学约束搜索

## 1. 概述

动力学约束搜索将速度等运动状态纳入搜索节点，以满足运动方程的基元连接节点，从前端生成可作为优化初值的时间轨迹。

## 2. 基础与定义

### 2.1 状态与运动基元

取双积分器 $x=(p,v)$、输入 $u=a$。常值加速度持续 $\tau$ 后：

$$
p'=p+v\tau+\tfrac12u\tau^2,\qquad v'=v+u\tau.
$$

基元需检查输入范围、状态范围和整段碰撞。

## 3. 核心原理

### 3.1 搜索目标

常用代价为

$$
J=\int_0^T\|u(t)\|^2dt+\rho T.
$$

基元代价由控制消耗和时长组成。节点键可由位置、速度及时间离散化得到；离散分辨率决定搜索分支和状态合并误差。

### 3.2 启发式与连接

放松障碍或部分约束后的最优控制代价可作为下界。解析连接尝试以连续轨迹连接目标，成功条件仍包含边界、控制和碰撞检查。

<details class="kt-derivation" id="kinodynamic-lower-bound">
<summary>固定时长的控制能量下界</summary>

对 $\dot x=Ax+Bu$，状态转移误差 $e=x_f-e^{AT}x_0$，可控 Gramian 为

$$
W(T)=\int_0^T e^{A(T-t)}BB^\top e^{A^\top(T-t)}dt.
$$

可控且 $W$ 可逆时，最小控制能量为 $e^\top W^{-1}e$。若进一步对 $T$ 求最小值并加入 $\rho T$，可构造放松问题的时间—能量下界。

</details>

### 3.3 方法关系

状态格预先定义连接基元，Hybrid A* 在离散索引上保留连续状态，动力学采样规划通过前向传播或边值连接扩展树。各自的完备性和最优性依赖对应的离散或连续问题假设。

## 4. 实现与应用

### 4.1 验证流程

积分基元 → 检查末端状态 → 检查全段约束 → 与启发式下界比较 → 回溯轨迹 → 作为后端优化初值。

## 5. 分析与总结

前端可行性只对采用的动力学模型和碰撞检查精度成立。提高状态维度增强表达能力，也增加搜索成本。

## 6. 关联与参考

[LaValle：Planning Algorithms](https://lavalle.pl/planning/)、[Fast-Planner](https://github.com/HKUST-Aerial-Robotics/Fast-Planner)。

关联：[最优控制](../../01_math-physic-base/05_optimal-control-foundations/01_optimal-control-formulation.md)。
