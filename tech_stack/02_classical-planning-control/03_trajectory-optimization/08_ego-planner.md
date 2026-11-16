# EGO-Planner

## 1. 概述

EGO-Planner 在轨迹碰撞区域提取局部障碍信息，用这些信息优化 B 样条控制点。核心过程是碰撞方向构造、带梯度的反弹优化，以及时间调整后的轨迹拟合。

## 2. 基础与定义

### 2.1 轨迹变量

均匀三次 B 样条由控制点 $Q_i$ 和间隔 $\Delta t$ 表示。其导数控制点为

$$
V_i=\frac{Q_{i+1}-Q_i}{\Delta t},\quad
A_i=\frac{Q_{i+2}-2Q_{i+1}+Q_i}{\Delta t^2},\quad
J_i=\frac{Q_{i+3}-3Q_{i+2}+3Q_{i+1}-Q_i}{\Delta t^3}.
$$

### 2.2 优化目标

反弹优化使用

$$
J=\lambda_sJ_s+\lambda_cJ_c+\lambda_dJ_d.
$$

平滑项约束高阶变化，碰撞项推动控制点离开障碍，动力学项惩罚导数超限。每个损失都通过控制点梯度参与更新。

## 3. 核心原理

### 3.1 碰撞信息

对碰撞区间搜索无碰撞引导路径，为相关控制点生成障碍基点 $p_{ij}$ 与指向自由空间的单位方向 $v_{ij}$。定义局部投影距离

$$
d_{ij}=(Q_i-p_{ij})^\top v_{ij}.
$$

这一距离描述当前碰撞方向上的分离程度。优化遇到新的障碍时，更新信息对并继续求解。

### 3.2 碰撞损失与梯度

令 $c_{ij}=d_s-d_{ij}$，$d_s>0$ 为裕量，采用平滑惩罚

$$
\phi(c)=\begin{cases}
0,&c\le0,\\
c^3,&0<c\le d_s,\\
3d_sc^2-3d_s^2c+d_s^3,&c>d_s.
\end{cases}
\qquad J_c=\sum_{i,j}\phi(c_{ij}).
$$

固定本轮信息对，梯度为

$$
\frac{\partial J_c}{\partial Q_i}=-\sum_j\phi'(c_{ij})v_{ij}.
$$

<details class="kt-derivation" id="ego-collision-gradient">
<summary>碰撞梯度推导</summary>

距离对控制点的导数为 $\partial d_{ij}/\partial Q_i=v_{ij}$，故 $\partial c_{ij}/\partial Q_i=-v_{ij}$。链式法则给出上式。其中

$$
\phi'(c)=\begin{cases}
0,&c\le0,\\3c^2,&0<c\le d_s,\\6d_sc-3d_s^2,&c>d_s.
\end{cases}
$$

在 $c=0$ 和 $c=d_s$ 处，函数、一阶和二阶导数均连续。负梯度方向沿 $+v_{ij}$，将控制点推向自由空间。

</details>

### 3.3 平滑与动力学梯度

将控制点按行堆叠，令 $D_k$ 表示包含时间尺度的 $k$ 阶差分算子。一个统一平滑表达为

$$
J_s=w_a\|D_2Q\|_F^2+w_j\|D_3Q\|_F^2,\qquad
\nabla_QJ_s=2w_aD_2^\top D_2Q+2w_jD_3^\top D_3Q.
$$

权重可选择加速度或 jerk 平滑；具体代码若省略固定的时间因子，需将其计入权重。动力学惩罚可写为 $J_d=\sum_k\Psi_k(D_kQ)$，梯度为

$$
\nabla_QJ_d=\sum_kD_k^\top\nabla\Psi_k(D_kQ).
$$

逐轴限制与向量范数限制对应不同的可行域，应与控制器能力一致。
<details class="kt-derivation" id="ego-difference-gradient">
<summary>差分梯度展开</summary>

单个 jerk 控制点 $J_i=(Q_{i+3}-3Q_{i+2}+3Q_{i+1}-Q_i)/\Delta t^3$ 对四个控制点的雅可比分别为 $[-I,3I,-3I,I]/\Delta t^3$。

令 $g_i=\partial\Psi/\partial J_i$，则控制点 $Q_k$ 收到

$$
\frac{\partial\Psi}{\partial Q_k}
=\frac{-g_k+3g_{k-1}-3g_{k-2}+g_{k-3}}{\Delta t^3},
$$

仅累加有效下标。平方平滑项有 $g_i=2J_i$；超限惩罚则使用对应违反量的导数。

</details>

### 3.4 时间调整与拟合

调整时间后，以原安全曲线为参考，优化

$$
J'=\lambda_sJ_s+\lambda_dJ_d+\lambda_fJ_f.
$$

新旧轨迹在归一化时间 $\alpha$ 处的误差为 $e=\Phi_f(\alpha T')-\Phi_s(\alpha T)$。设原曲线单位切向为 $u$，可用

$$
J_f=\int_0^1e^\top Me\,d\alpha,\qquad
M=\frac1{b^2}I+\left(\frac1{a^2}-\frac1{b^2}\right)uu^\top
$$

分别调节切向和法向误差。$a>b>0$ 时允许更多切向调整；零速处需使用邻近有效切向或各向同性度量。
<details class="kt-derivation" id="ego-fit-gradient">
<summary>各向异性拟合梯度</summary>

切向误差平方为 $(e^\top u)^2$，法向误差平方为 $\|e\|^2-(e^\top u)^2$。因此

$$
\frac{(e^\top u)^2}{a^2}+\frac{\|e\|^2-(e^\top u)^2}{b^2}=e^\top Me.
$$

固定参考曲线、切向和时间，$\Phi_f(t)=\sum_kB_k(t)Q_k$，得到

$$
\frac{\partial J_f}{\partial Q_k}=\int_0^12B_k(\alpha T')M(\alpha)e(\alpha)\,d\alpha.
$$

数值求积使用 $\sum_l2w_lB_k(\alpha_lT')M_le_l$。若时间也参与同轮优化，应继续计算基函数与采样时刻的时间导数。

</details>

## 4. 实现与应用

### 4.1 规划流程

状态估计与目标 → 轨迹初始化 → 碰撞区间及引导路径 → 信息对构造 → 反弹优化 → 导数检查 → 时间调整与拟合 → 碰撞复查 → 发布轨迹。

执行端依据轨迹起始时间计算局部时间，输出位置、速度和加速度。重规划从预计交接时刻的状态开始，检测失败时保留可用轨迹或进入预定停止流程。

### 4.2 实现索引

| 模块 | 主要入口 | 职责 |
| --- | --- | --- |
| 重规划状态机 | `EGOReplanFSM` | 目标、触发、碰撞监视与状态切换 |
| 规划管理 | `reboundReplan` | 初始化、反弹优化与精修调度 |
| B 样条拟合 | `parameterizeToBspline` | 采样位置与边界导数的最小二乘拟合 |
| 碰撞信息 | `initControlPoints` | 碰撞分段、A* 引导与信息对 |
| 反弹优化 | `costFunctionRebound` | 平滑、碰撞与动力学梯度 |
| 轨迹精修 | `costFunctionRefine` | 平滑、动力学与拟合梯度 |
| 轨迹执行 | `traj_server` | 按时间求值并发布参考状态 |

上述函数见官方仓库。具体参数、检查区间和话题名称按所使用版本核对。

## 5. 分析与总结

局部碰撞信息减少了全空间距离场维护，但优化效果依赖引导方向与初值。碰撞惩罚、导数代理和离散检查各自具有不同作用，最终评估应统计实际执行轨迹的碰撞裕量、约束残差与计算耗时。

## 6. 关联与参考

- [EGO-Planner 论文](https://arxiv.org/abs/2008.08835)
- [EGO-Planner 官方实现](https://github.com/ZJU-FAST-Lab/ego-planner)
- [Bézier 曲线与 B 样条](03_bezier-bspline.md)
- [重规划与轨迹连续性](10_replanning-and-trajectory-continuity.md)
