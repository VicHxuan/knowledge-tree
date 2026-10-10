# EGO-Planner 学习文档

## 1. 方法定位与整体思想

EGO-Planner 是一种无 ESDF 的梯度型四旋翼无人机局部轨迹规划器。传统梯度型轨迹优化方法通常需要预先维护 ESDF，并通过 ESDF 查询障碍物距离和梯度方向。EGO-Planner 的核心出发点是：局部轨迹优化实际只涉及轨迹附近的一小部分空间，维护完整 ESDF 会产生大量与当前轨迹无关的计算。因此，EGO-Planner 不再构建环境空间中的稠密距离场，而是在轨迹真正与障碍物发生碰撞时，只为发生碰撞的控制点提取必要的障碍物信息，并直接构造碰撞梯度。

系统将无人机轨迹表示为均匀 B 样条曲线。轨迹由控制点集合决定：

\[
Q=\{Q_1,Q_2,\cdots,Q_{N_c}\}, \quad Q_i \in \mathbb{R}^3
\]

整体优化目标为：

\[
\min_Q J=\lambda_sJ_s+\lambda_cJ_c+\lambda_dJ_d
\]

其中，\(J_s\) 表示平滑度代价，\(J_c\) 表示碰撞代价，\(J_d\) 表示动力学可行性代价。三个部分共同作用，使轨迹在保持平滑的同时远离障碍物，并满足速度、加速度和 jerk 等动力学约束。

EGO-Planner 的核心创新在于碰撞项的构造。传统方法依赖 ESDF 提供障碍物距离和梯度，而 EGO-Planner 在检测到轨迹与障碍物发生碰撞后，利用 A* 在局部生成一条无碰撞引导路径 \(\Gamma\)。随后，对于碰撞段中的每个控制点 \(Q_i\)，系统为其分配若干组障碍物信息：

\[
\{p_{ij},v_{ij}\}
\]

其中，\(p_{ij}\) 是障碍物表面附近的锚点，\(v_{ij}\) 是与该锚点相关的排斥方向向量。控制点到障碍物的等效有符号距离被定义为：

\[
d_{ij}=(Q_i-p_{ij})^\top v_{ij}
\]

若 \(d_{ij}\) 小于安全距离 \(s_f\)，则说明该控制点距离障碍物过近，需要受到碰撞惩罚。由于 \(p_{ij}\) 与 \(v_{ij}\) 在一次优化迭代中被视为固定量，碰撞代价对控制点的梯度可以直接由上式解析求得，无需查询 ESDF，也无需对体素地图进行三线性插值。


## 2. 完整时序流程

EGO-Planner 的运行流程可以理解为一个在线局部重规划循环。无人机在飞行过程中持续感知周围环境，并根据当前状态、局部地图和目标点不断生成新的局部 B 样条轨迹。

首先，系统接收当前无人机状态、局部环境信息和目标点。当前状态通常包括位置、速度和加速度，目标点可以是全局路径上的局部目标，也可以是任务指定的航点。规划器不会先构建 ESDF，而是直接根据当前状态和目标点初始化一条 B 样条轨迹。初始化轨迹通常只满足边界状态约束，并不保证完全避障。

随后，规划器进入反弹式轨迹优化阶段。系统对当前 B 样条控制点或由控制点生成的轨迹段进行碰撞检测。如果轨迹没有发生碰撞，则说明当前轨迹在几何上可行，可以进入动力学检查。如果轨迹存在碰撞，系统会提取连续碰撞段，并针对每个碰撞段执行局部 A* 搜索，生成一条绕过障碍物的无碰撞引导路径 \(\Gamma\)。

在得到引导路径后，系统会为碰撞段中的每个控制点分配障碍物锚点 \(p_{ij}\) 和排斥方向 \(v_{ij}\)。这些信息不会像 ESDF 那样覆盖整个局部地图，而是只绑定在对应的控制点上。一个控制点可以拥有多组 \(\{p,v\}\) 信息，这意味着它可能受到多个障碍物或多个局部碰撞约束的共同作用。

完成障碍物信息提取后，系统构造完整优化目标：

\[
J=\lambda_sJ_s+\lambda_cJ_c+\lambda_dJ_d
\]

其中，平滑项使轨迹的加速度和 jerk 尽量小，碰撞项将控制点从障碍物中推出，动力学项限制速度、加速度和 jerk 不超过设定上限。然后使用 L-BFGS 对控制点进行优化。L-BFGS 不显式计算 Hessian，而是利用历史梯度信息近似逆 Hessian，因此适合 EGO-Planner 中这种需要快速重启、目标函数随着新障碍物信息变化的优化问题。

优化后，系统再次进行碰撞检测。如果轨迹仍然碰撞，则说明当前障碍物信息还不足以将轨迹推出安全区域，规划器会继续提取新的碰撞信息并再次优化。这个过程会让轨迹在障碍物之间产生类似反弹的行为，直到轨迹进入无碰撞区域。

当轨迹通过几何碰撞检测后，系统继续检查动力学可行性。由于初始时间分配通常在优化前确定，而优化后的轨迹形状可能发生明显变化，因此速度、加速度或 jerk 可能超过限制。若轨迹满足动力学约束，则轨迹可以直接输出给控制器执行。若轨迹不满足动力学约束，则进入时间重分配和轨迹精修阶段。

时间重分配阶段首先计算动力学超限比例：

\[
r_e=\max\left\{
\left|\frac{V_{i,r}}{v_m}\right|,
\sqrt{\left|\frac{A_{j,r}}{a_m}\right|},
\sqrt[3]{\left|\frac{J_{k,r}}{j_m}\right|},
1
\right\}
\]

速度与时间间隔成反比，加速度与时间间隔平方成反比，jerk 与时间间隔三次方成反比。因此，系统根据最大超限比例放大 B 样条时间间隔：

\[
\Delta t'=r_e\Delta t
\]

时间被拉长后，原先超限的速度、加速度和 jerk 会下降。但单纯改变时间间隔可能破坏轨迹的平滑性和几何性质，因此 EGO-Planner 进一步使用各向异性曲线拟合方法生成新的 B 样条轨迹 \(\Phi_f\)，使其在保持原安全轨迹 \(\Phi_s\) 几何形状的同时，优化动力学可行性。

各向异性拟合的关键是将新轨迹点与原轨迹点之间的误差分解为切向误差和径向误差。沿轨迹切向方向的误差对应路径前后方向的偏移，对安全性影响较小，因此惩罚较弱。垂直于轨迹切向方向的径向误差可能导致轨迹偏离原来的安全通道，因此惩罚较强。

最终，经过碰撞检测、反弹优化、动力学检查、时间重分配和轨迹精修后，EGO-Planner 输出一条满足局部避障、安全距离、平滑性和动力学约束的 B 样条轨迹。


## 2.论文核心理论推导

### 2.1 B 样条导数控制点

EGO-Planner 使用均匀 B 样条。设相邻节点时间间隔为 \(\Delta t\)，位置控制点为 \(Q_i\)。根据均匀 B 样条的导数性质，速度、加速度和 jerk 的控制点可写为：

\[
V_i=\frac{Q_{i+1}-Q_i}{\Delta t}
\]

\[
A_i=\frac{V_{i+1}-V_i}{\Delta t}
=\frac{Q_{i+2}-2Q_{i+1}+Q_i}{\Delta t^2}
\]

\[
J_i=\frac{A_{i+1}-A_i}{\Delta t}
=\frac{Q_{i+3}-3Q_{i+2}+3Q_{i+1}-Q_i}{\Delta t^3}
\]

这些表达式说明，B 样条轨迹的高阶导数可以直接由控制点差分得到。由于 B 样条具有凸包性质，只要控制点导数满足约束，整条曲线也可以被有效约束在相应范围内。

### 2.2 碰撞避障力估计

这里为涉及到碰撞区域的控制点计算出p,v对，用于后续的优化环节计算梯度

设 B 样条轨迹的控制点为：

$$
Q={Q_1, Q_2, \dots, Q_{N_c}}, \quad Q_i \in \mathbb{R}^3
$$

每个控制点 $Q_i$ 都独立保存自己的障碍物信息。对于第 $i$ 个控制点，其对应的障碍物信息由若干组 ${p_{ij}, v_{ij}}$ 构成，其中：

$$
p_{ij}
$$

表示与控制点 $Q_i$ 相关的第 $j$ 个障碍物锚点，通常位于障碍物表面附近；

$$
v_{ij}
$$

表示该锚点对应的排斥方向向量，用于描述控制点应当沿哪个方向被推出障碍物。

因此，一个控制点可以对应多组障碍物信息：

$$
Q_i \longrightarrow {p_{i1}, v_{i1}}, {p_{i2}, v_{i2}}, \dots, {p_{iN_p}, v_{iN_p}}
$$

这意味着同一个控制点可能同时受到多个障碍物约束的影响。

#### 2.2.1 从碰撞轨迹到无碰撞引导路径

初始阶段，系统先生成一条满足起终点边界条件的朴素 B 样条轨迹：

$$
\Phi
$$

这条轨迹只保证满足边界状态，不保证无碰撞。因此在优化开始后，需要检查 B 样条控制点或控制点折线是否进入障碍物。

若检测到一段连续碰撞控制点，则记为一个碰撞段：

$$
S={Q_a, Q_{a+1}, \dots, Q_b}
$$

对于每一个碰撞段 $S$，系统执行一次局部路径搜索，得到一条绕开障碍物的无碰撞引导路径：

$$
\Gamma
$$

该路径不是最终轨迹，而是用来估计碰撞控制点应该向哪里移动的参考路径。

整体过程为：

```text
初始 B 样条轨迹 Φ
        ↓
检测连续碰撞段 S
        ↓
对碰撞段入口和出口执行 A* 搜索
        ↓
得到无碰撞引导路径 Γ
        ↓
利用 Γ 为碰撞控制点生成 {p, v} 信息
```

#### 2.2.2 避障信息对 `{p, v}` 的几何含义

对于碰撞段中的某个控制点 $Q_i$，EGO-Planner 会根据无碰撞引导路径 $\Gamma$ 构造一组障碍物信息：

$$
{p_{ij}, v_{ij}}
$$

其中 $p_{ij}$ 可以理解为障碍物表面附近的参考点，$v_{ij}$ 是排斥方向。

论文中的 Fig. 3(a) 展示了该思想：原始轨迹 $\Phi$ 穿过障碍物，而系统在碰撞区域内为多个控制点生成对应的 $p$ 和 $v$。这些 $v$ 类似于“虚拟避障力”的方向，用于在优化中推动控制点离开障碍物。

需要注意的是，${p, v}$ 不是全局地图信息，而是局部绑定到具体控制点上的信息。也就是说，EGO-Planner 不是提前为整个空间计算距离场，而是只在某个控制点碰撞时，为该控制点生成必要的局部障碍物约束。

#### 2.2.3 `{p, v}` 的生成思路

对于碰撞控制点 $Q_i$，首先构造其局部轨迹方向。论文中定义：

$$
R_i = \frac{Q_{i+1} - Q_{i-1}}{2\Delta t}
$$

该向量近似表示轨迹在控制点 $Q_i$ 附近的切向方向。

然后构造一个经过 $Q_i$、且垂直于 $R_i$ 的平面：

$$
\Psi_i = {X \mid (X - Q_i)^\top R_i = 0}
$$

这个平面可以理解为当前控制点处的局部截面。

接着，令该平面 $\Psi_i$ 与无碰撞引导路径 $\Gamma$ 相交。交点附近的引导路径位置代表：如果当前轨迹想绕开障碍物，那么控制点 $Q_i$ 在该局部截面上应该向哪个方向移动。

设引导路径与平面的交点为：

$$
G_i
$$

则从控制点 $Q_i$ 指向交点 $G_i$ 的方向可以作为避障方向：

$$
v_{ij} = \frac{G_i - Q_i}{\lVert G_i - Q_i \rVert}
$$

随后沿 $Q_i$ 到 $G_i$ 的方向搜索障碍物边界附近点，得到锚点：

$$
p_{ij}
$$

因此，${p_{ij}, v_{ij}}$ 的几何意义为：

```text
p_ij：障碍物边界附近的锚点
v_ij：从障碍物区域指向安全引导路径的方向
```

该方向随后被用于构造碰撞代价梯度，使控制点沿着安全方向被推出障碍物。

#### 2.2.4 基于 `{p, v}` 的等效距离定义

对控制点 $Q_i$ 及其第 $j$ 个障碍物信息对 ${p_{ij}, v_{ij}}$，论文定义其等效障碍物距离为：

$$
d_{ij} = (Q_i - p_{ij})^\top v_{ij}
$$

该式是整个 ESDF-free 碰撞建模的核心。

其中：

$$
Q_i - p_{ij}
$$

表示从锚点 $p_{ij}$ 指向控制点 $Q_i$ 的向量。

将其投影到方向 $v_{ij}$ 上，得到标量距离：

$$
d_{ij}
$$

如果 $d_{ij}$ 较大，说明控制点已经沿避障方向远离障碍物；如果 $d_{ij}$ 较小，说明控制点仍然靠近障碍物或处于障碍物内部。

该距离可以理解为一个局部线性化的有符号距离。不同于 ESDF 中每个空间点都有一个距离值，这里距离只对当前控制点和当前 ${p, v}$ 对有效。

#### 2.2.5 局部距离场的含义

由

$$
d_{ij} = (Q_i - p_{ij})^\top v_{ij}
$$

可以看出，在固定 $p_{ij}$ 和 $v_{ij}$ 的情况下，距离 $d_{ij}$ 是关于控制点 $Q_i$ 的线性函数。

令：

$$
d_{ij} = 0
$$

可得：

$$
(Q_i - p_{ij})^\top v_{ij} = 0
$$

这表示一个经过 $p_{ij}$、法向量为 $v_{ij}$ 的平面。该平面可以看作局部零距离面。

当控制点沿 $v_{ij}$ 方向移动时，距离线性增大；当控制点沿 $-v_{ij}$ 方向移动时，距离减小。因此，该局部距离场的梯度恒为：

$$
\frac{\partial d_{ij}}{\partial Q_i} = v_{ij}
$$

这就是 EGO-Planner 可以不依赖 ESDF 仍然获得碰撞梯度的根本原因。

传统 ESDF 方法需要查询：

$$
\nabla d(Q_i)
$$

而 EGO-Planner 直接构造：

$$
\nabla d_{ij} = v_{ij}
$$

因此避障方向不需要从体素距离场中插值获得，而是由碰撞轨迹和 A* 引导路径之间的几何关系直接给出。

#### 2.2.6 新障碍物判定准则

在优化过程中，轨迹可能需要多次迭代才能离开当前障碍物。如果每次迭代都为同一个控制点重复生成 ${p, v}$，会造成冗余约束，增加计算量。

因此论文提出一个新障碍物判定准则：只有当控制点 $Q_i$ 对所有已有有效 ${p_{ij}, v_{ij}}$ 都满足：

$$
d_{ij} > 0
$$

时，才认为它当前进入的是一个新障碍物，需要生成新的 ${p, v}$ 对。

也就是说，若控制点还没有从已有障碍物约束中逃离，就不重复添加新的障碍物信息。

该逻辑可以概括为：

```text
若 Q_i 仍受已有 {p, v} 约束影响
        ↓
不添加新的 {p, v}

若 Q_i 已经离开所有已有障碍物约束
但又检测到新的碰撞
        ↓
添加新的 {p, v}
```

这样做有两个作用：

1. 避免在同一个障碍物上重复生成冗余约束。
2. 只保留对最终轨迹优化有贡献的必要障碍物信息。

#### 2.2.7 与 ESDF 方法的区别

传统 ESDF 方法的碰撞梯度来自预先构建的距离场。对于任意轨迹点 $x$，ESDF 给出：

$$
d(x)
$$

以及梯度：

$$
\nabla d(x)
$$

优化器根据该梯度推动轨迹远离障碍物。

但是该方式有两个问题：

1. 需要维护大范围 ESDF，计算量较大。
2. 当轨迹深陷障碍物内部或传感器视野有限时，ESDF 梯度可能将轨迹推入局部极小值。

EGO-Planner 的方法不同。它不构建完整 ESDF，而是：

```text
检测轨迹是否碰撞
        ↓
只对碰撞段搜索一条无碰撞引导路径 Γ
        ↓
根据 Γ 为碰撞控制点构造 {p, v}
        ↓
用 {p, v} 直接形成局部距离和梯度
```

因此，它将避障梯度的来源从“全局距离场”转化为“局部无碰撞引导路径”。

#### 2.2.8 避障力估计的本质

从优化角度看，${p, v}$ 对实际上构造了一个局部虚拟力场。

对控制点 $Q_i$，其障碍物距离为：

$$
d_{ij} = (Q_i - p_{ij})^\top v_{ij}
$$

若要求控制点与障碍物保持安全距离 $s_f$，则希望满足：

$$
d_{ij} \geq s_f
$$

当：

$$
d_{ij} < s_f
$$

说明控制点距离障碍物过近，需要产生碰撞惩罚。此时优化器会沿 $v_{ij}$ 方向推动控制点，使 $d_{ij}$ 增大。

因此，EGO-Planner 中所谓的碰撞避障力，可以理解为：

```text
由 A* 引导路径决定方向
由 {p, v} 定义局部距离
由碰撞代价函数决定力的大小
由梯度下降推动控制点远离障碍物
```

这种避障力不是物理力，而是优化意义下的虚拟梯度。

#### 2.2.9 算法流程总结

论文中的碰撞信息提取过程可以整理为：

```text
输入：
    环境地图 E
    B 样条控制点 Q

输出：
    每个碰撞控制点对应的 {p, v} 信息

流程：
    遍历控制点 Q_i
        ↓
    检测连续碰撞控制点区间
        ↓
    得到碰撞段 S
        ↓
    对每个碰撞段 S_i 执行路径搜索
        ↓
    生成无碰撞引导路径 Γ
        ↓
    遍历碰撞段中的控制点 Q_j
        ↓
    根据 Γ 为 Q_j 生成 {p, v}
        ↓
    将 {p, v} 存入 Q_j 的障碍物信息列表
```

该流程对应论文中的 `CheckAndAddObstacleInfo`，其核心是只在轨迹真实碰撞时提取障碍物信息。

#### 2.2.10 与代码实现的对应关系

该理论在代码中的对应关系可以理解为：

```text
碰撞段检测
        ↓
initControlPoints 中检测 segment_ids

无碰撞引导路径 Γ
        ↓
对每个 segment 执行 A* 搜索

控制点局部切向方向 R_i
        ↓
由 Q_{i+1} - Q_{i-1} 近似得到

局部法平面 Ψ_i
        ↓
过控制点且垂直于 R_i 的平面

A* 路径与平面求交
        ↓
得到引导交点 G_i

构造避障方向 v_ij
        ↓
direction[j]

构造锚点 p_ij
        ↓
base_point[j]

基于 d_ij 的碰撞代价
        ↓
costFunctionRebound 中的碰撞项
```

因此，代码中的 `base_point` 和 `direction` 正是论文中 ${p_{ij}, v_{ij}}$ 的工程实现。




### 2.3 代价函数梯度推导

#### 2.3.1 平滑度代价 \(J_s\) 的梯度推导

论文中平滑度代价由加速度控制点和 jerk 控制点的平方和构成：

\[
J_s=\sum_i \|A_i\|^2+\sum_i \|J_i\|^2
\]

将其拆成两部分：

\[
J_s=J_A+J_J
\]

其中：

\[
J_A=\sum_i \|A_i\|^2
\]

\[
J_J=\sum_i \|J_i\|^2
\]

对于加速度项：

\[
A_i=\frac{Q_{i+2}-2Q_{i+1}+Q_i}{\Delta t^2}
\]

单个加速度代价为：

\[
\|A_i\|^2=A_i^\top A_i
\]

对控制点求导：

\[
\frac{\partial \|A_i\|^2}{\partial Q_k}
=
2A_i^\top \frac{\partial A_i}{\partial Q_k}
\]

由于 \(A_i\) 只与 \(Q_i,Q_{i+1},Q_{i+2}\) 有关，因此：

\[
\frac{\partial A_i}{\partial Q_i}=\frac{I}{\Delta t^2}
\]

\[
\frac{\partial A_i}{\partial Q_{i+1}}=-\frac{2I}{\Delta t^2}
\]

\[
\frac{\partial A_i}{\partial Q_{i+2}}=\frac{I}{\Delta t^2}
\]

于是有：

\[
\frac{\partial \|A_i\|^2}{\partial Q_i}
=
\frac{2A_i}{\Delta t^2}
\]

\[
\frac{\partial \|A_i\|^2}{\partial Q_{i+1}}
=
-\frac{4A_i}{\Delta t^2}
\]

\[
\frac{\partial \|A_i\|^2}{\partial Q_{i+2}}
=
\frac{2A_i}{\Delta t^2}
\]

因此，任意控制点 \(Q_k\) 接收到来自加速度平滑项的梯度为：

\[
\frac{\partial J_A}{\partial Q_k}
=
\frac{2}{\Delta t^2}A_k
-\frac{4}{\Delta t^2}A_{k-1}
+\frac{2}{\Delta t^2}A_{k-2}
\]

不存在的下标项直接忽略。

对于 jerk 项：

\[
J_i=\frac{Q_{i+3}-3Q_{i+2}+3Q_{i+1}-Q_i}{\Delta t^3}
\]

单个 jerk 代价为：

\[
\|J_i\|^2=J_i^\top J_i
\]

同理：

\[
\frac{\partial \|J_i\|^2}{\partial Q_k}
=
2J_i^\top \frac{\partial J_i}{\partial Q_k}
\]

由于 \(J_i\) 只与 \(Q_i,Q_{i+1},Q_{i+2},Q_{i+3}\) 有关，因此：

\[
\frac{\partial J_i}{\partial Q_i}=-\frac{I}{\Delta t^3}
\]

\[
\frac{\partial J_i}{\partial Q_{i+1}}=\frac{3I}{\Delta t^3}
\]

\[
\frac{\partial J_i}{\partial Q_{i+2}}=-\frac{3I}{\Delta t^3}
\]

\[
\frac{\partial J_i}{\partial Q_{i+3}}=\frac{I}{\Delta t^3}
\]

于是有：

\[
\frac{\partial \|J_i\|^2}{\partial Q_i}
=
-\frac{2J_i}{\Delta t^3}
\]

\[
\frac{\partial \|J_i\|^2}{\partial Q_{i+1}}
=
\frac{6J_i}{\Delta t^3}
\]

\[
\frac{\partial \|J_i\|^2}{\partial Q_{i+2}}
=
-\frac{6J_i}{\Delta t^3}
\]

\[
\frac{\partial \|J_i\|^2}{\partial Q_{i+3}}
=
\frac{2J_i}{\Delta t^3}
\]

因此，任意控制点 \(Q_k\) 接收到来自 jerk 平滑项的梯度为：

\[
\frac{\partial J_J}{\partial Q_k}
=
-\frac{2}{\Delta t^3}J_k
+\frac{6}{\Delta t^3}J_{k-1}
-\frac{6}{\Delta t^3}J_{k-2}
+\frac{2}{\Delta t^3}J_{k-3}
\]

最终，平滑度项的梯度为：

\[
\frac{\partial J_s}{\partial Q_k}
=
\frac{\partial J_A}{\partial Q_k}
+
\frac{\partial J_J}{\partial Q_k}
\]

如果考虑总目标函数中的权重，则该部分贡献为：

\[
\lambda_s\frac{\partial J_s}{\partial Q_k}
\]


#### 2.3.2 碰撞代价 \(J_c\) 的梯度推导

碰撞项是 EGO-Planner 区别于 ESDF 方法的核心。对于控制点 \(Q_i\) 及其对应的第 \(j\) 个障碍物信息对 \(\{p_{ij},v_{ij}\}\)，距离定义为：

\[
d_{ij}=(Q_i-p_{ij})^\top v_{ij}
\]

定义安全距离残差：

\[
c_{ij}=s_f-d_{ij}
\]

当 \(c_{ij}\leq 0\) 时，说明控制点到障碍物的距离已经大于等于安全距离，不产生碰撞惩罚。当 \(c_{ij}>0\) 时，说明控制点进入安全距离范围，需要产生惩罚。

论文采用二阶连续的分段碰撞惩罚函数：

\[
j_c(i,j)=
\begin{cases}
0, & c_{ij}\leq 0\\
c_{ij}^3, & 0<c_{ij}\leq s_f\\
3s_fc_{ij}^2-3s_f^2c_{ij}+s_f^3, & c_{ij}>s_f
\end{cases}
\]

总碰撞代价为：

\[
J_c=\sum_i\sum_j j_c(i,j)
\]

首先计算：

\[
\frac{\partial d_{ij}}{\partial Q_i}=v_{ij}
\]

又因为：

\[
c_{ij}=s_f-d_{ij}
\]

所以：

\[
\frac{\partial c_{ij}}{\partial Q_i}
=
-\frac{\partial d_{ij}}{\partial Q_i}
=
-v_{ij}
\]

分段求导。

当 \(c_{ij}\leq 0\) 时：

\[
j_c(i,j)=0
\]

因此：

\[
\frac{\partial j_c(i,j)}{\partial Q_i}=0
\]

当 \(0<c_{ij}\leq s_f\) 时：

\[
j_c(i,j)=c_{ij}^3
\]

先对 \(c_{ij}\) 求导：

\[
\frac{\partial j_c}{\partial c_{ij}}=3c_{ij}^2
\]

再乘以：

\[
\frac{\partial c_{ij}}{\partial Q_i}=-v_{ij}
\]

得到：

\[
\frac{\partial j_c(i,j)}{\partial Q_i}
=
-3c_{ij}^2v_{ij}
\]

当 \(c_{ij}>s_f\) 时：

\[
j_c(i,j)=3s_fc_{ij}^2-3s_f^2c_{ij}+s_f^3
\]

对 \(c_{ij}\) 求导：

\[
\frac{\partial j_c}{\partial c_{ij}}
=
6s_fc_{ij}-3s_f^2
\]

因此：

\[
\frac{\partial j_c(i,j)}{\partial Q_i}
=
-(6s_fc_{ij}-3s_f^2)v_{ij}
\]

最终，控制点 \(Q_i\) 的碰撞梯度为：

\[
\frac{\partial J_c}{\partial Q_i}
=
\sum_j
\begin{cases}
0, & c_{ij}\leq 0\\
-3c_{ij}^2v_{ij}, & 0<c_{ij}\leq s_f\\
-(6s_fc_{ij}-3s_f^2)v_{ij}, & c_{ij}>s_f
\end{cases}
\]

考虑总目标函数权重后，碰撞项对优化的贡献为：

\[
\lambda_c\frac{\partial J_c}{\partial Q_i}
\]

该梯度的方向与 \(v_{ij}\) 的定义有关。若 \(v_{ij}\) 被定义为距离增大的方向，则在梯度下降中，控制点会沿着 \(v_{ij}\) 所指示的方向移动，从而被推出障碍物区域。


#### 2.3.3 动力学可行性代价 \(J_d\) 的梯度推导

动力学可行性项用于限制速度、加速度和 jerk。论文中的形式为：

\[
J_d=
\sum_i w_vF(V_i)
+
\sum_i w_aF(A_i)
+
\sum_i w_jF(J_i)
\]

其中：

\[
F(C)=\sum_{r=x,y,z}f(c_r)
\]

这里 \(C\) 可以是速度控制点 \(V_i\)、加速度控制点 \(A_i\) 或 jerk 控制点 \(J_i\)，\(c_r\) 表示其在某一个坐标轴上的分量。

论文采用一个二阶连续的分段惩罚函数：

\[
f(c_r)=
\begin{cases}
a_1c_r^2+b_1c_r+c_1, & c_r\leq -c_j\\
(-\lambda c_m-c_r)^3, & -c_j<c_r<-\lambda c_m\\
0, & -\lambda c_m\leq c_r\leq \lambda c_m\\
(c_r-\lambda c_m)^3, & \lambda c_m<c_r<c_j\\
a_2c_r^2+b_2c_r+c_2, & c_r\geq c_j
\end{cases}
\]

其中，\(c_m\) 是对应导数的限制值，\(\lambda\) 是小于 1 的弹性系数，\(c_j\) 是三次段与二次段的分界点。这个函数在允许范围内为零，超过软阈值后开始产生惩罚，并在更大超限区域使用二次函数抑制斜率过大。

先给出标量函数的导数：

\[
f'(c_r)=
\begin{cases}
2a_1c_r+b_1, & c_r\leq -c_j\\
-3(-\lambda c_m-c_r)^2, & -c_j<c_r<-\lambda c_m\\
0, & -\lambda c_m\leq c_r\leq \lambda c_m\\
3(c_r-\lambda c_m)^2, & \lambda c_m<c_r<c_j\\
2a_2c_r+b_2, & c_r\geq c_j
\end{cases}
\]

因此，对三维向量 \(C=[c_x,c_y,c_z]^\top\)，有：

\[
\nabla_CF(C)=
\begin{bmatrix}
f'(c_x)\\
f'(c_y)\\
f'(c_z)
\end{bmatrix}
\]

记：

\[
g^V_i=\nabla_{V_i}F(V_i)
\]

\[
g^A_i=\nabla_{A_i}F(A_i)
\]

\[
g^J_i=\nabla_{J_i}F(J_i)
\]

下面分别将这些导数通过链式法则传递到位置控制点 \(Q_k\)。

速度项：

\[
V_i=\frac{Q_{i+1}-Q_i}{\Delta t}
\]

所以：

\[
\frac{\partial V_i}{\partial Q_i}=-\frac{I}{\Delta t}
\]

\[
\frac{\partial V_i}{\partial Q_{i+1}}=\frac{I}{\Delta t}
\]

单个速度惩罚 \(w_vF(V_i)\) 对控制点的梯度为：

\[
\frac{\partial w_vF(V_i)}{\partial Q_i}
=
-\frac{w_v}{\Delta t}g^V_i
\]

\[
\frac{\partial w_vF(V_i)}{\partial Q_{i+1}}
=
\frac{w_v}{\Delta t}g^V_i
\]

因此，任意控制点 \(Q_k\) 接收到的速度可行性梯度为：

\[
\frac{\partial J_{d,V}}{\partial Q_k}
=
\frac{w_v}{\Delta t}g^V_{k-1}
-
\frac{w_v}{\Delta t}g^V_k
\]

加速度项：

\[
A_i=\frac{Q_{i+2}-2Q_{i+1}+Q_i}{\Delta t^2}
\]

所以：

\[
\frac{\partial A_i}{\partial Q_i}=\frac{I}{\Delta t^2}
\]

\[
\frac{\partial A_i}{\partial Q_{i+1}}=-\frac{2I}{\Delta t^2}
\]

\[
\frac{\partial A_i}{\partial Q_{i+2}}=\frac{I}{\Delta t^2}
\]

单个加速度惩罚 \(w_aF(A_i)\) 对控制点的梯度为：

\[
\frac{\partial w_aF(A_i)}{\partial Q_i}
=
\frac{w_a}{\Delta t^2}g^A_i
\]

\[
\frac{\partial w_aF(A_i)}{\partial Q_{i+1}}
=
-\frac{2w_a}{\Delta t^2}g^A_i
\]

\[
\frac{\partial w_aF(A_i)}{\partial Q_{i+2}}
=
\frac{w_a}{\Delta t^2}g^A_i
\]

因此，任意控制点 \(Q_k\) 接收到的加速度可行性梯度为：

\[
\frac{\partial J_{d,A}}{\partial Q_k}
=
\frac{w_a}{\Delta t^2}g^A_k
-
\frac{2w_a}{\Delta t^2}g^A_{k-1}
+
\frac{w_a}{\Delta t^2}g^A_{k-2}
\]

jerk 项：

\[
J_i=\frac{Q_{i+3}-3Q_{i+2}+3Q_{i+1}-Q_i}{\Delta t^3}
\]

所以：

\[
\frac{\partial J_i}{\partial Q_i}=-\frac{I}{\Delta t^3}
\]

\[
\frac{\partial J_i}{\partial Q_{i+1}}=\frac{3I}{\Delta t^3}
\]

\[
\frac{\partial J_i}{\partial Q_{i+2}}=-\frac{3I}{\Delta t^3}
\]

\[
\frac{\partial J_i}{\partial Q_{i+3}}=\frac{I}{\Delta t^3}
\]

单个 jerk 惩罚 \(w_jF(J_i)\) 对控制点的梯度为：

\[
\frac{\partial w_jF(J_i)}{\partial Q_i}
=
-\frac{w_j}{\Delta t^3}g^J_i
\]

\[
\frac{\partial w_jF(J_i)}{\partial Q_{i+1}}
=
\frac{3w_j}{\Delta t^3}g^J_i
\]

\[
\frac{\partial w_jF(J_i)}{\partial Q_{i+2}}
=
-\frac{3w_j}{\Delta t^3}g^J_i
\]

\[
\frac{\partial w_jF(J_i)}{\partial Q_{i+3}}
=
\frac{w_j}{\Delta t^3}g^J_i
\]

因此，任意控制点 \(Q_k\) 接收到的 jerk 可行性梯度为：

\[
\frac{\partial J_{d,J}}{\partial Q_k}
=
-\frac{w_j}{\Delta t^3}g^J_k
+
\frac{3w_j}{\Delta t^3}g^J_{k-1}
-
\frac{3w_j}{\Delta t^3}g^J_{k-2}
+
\frac{w_j}{\Delta t^3}g^J_{k-3}
\]

最终，动力学可行性项的总梯度为：

\[
\frac{\partial J_d}{\partial Q_k}
=
\frac{\partial J_{d,V}}{\partial Q_k}
+
\frac{\partial J_{d,A}}{\partial Q_k}
+
\frac{\partial J_{d,J}}{\partial Q_k}
\]

考虑总目标函数权重后，其优化贡献为：

\[
\lambda_d\frac{\partial J_d}{\partial Q_k}
\]


#### 2.3.4  各向异性拟合项 \(J_f\) 的梯度推导

当反弹优化后的轨迹在几何上安全但动力学不可行时，EGO-Planner 会进行时间重分配，并通过各向异性曲线拟合生成新的轨迹。拟合目标为：

\[
J'=\lambda_sJ_s+\lambda_dJ_d+\lambda_fJ_f
\]

其中 \(J_f\) 用于让新轨迹 \(\Phi_f\) 贴近原安全轨迹 \(\Phi_s\)。

对归一化时间 \(\alpha\in[0,1]\)，定义新轨迹与安全轨迹之间的位移：

\[
e(\alpha)=\Phi_f(\alpha T')-\Phi_s(\alpha T)
\]

令原安全轨迹的单位切向量为：

\[
u(\alpha)=\frac{\dot{\Phi}_s(\alpha T)}{\|\dot{\Phi}_s(\alpha T)\|}
\]

切向误差为：

\[
d_a=e^\top u
\]

径向误差为：

\[
d_r=\|e\times u\|
\]

由于 \(u\) 为单位向量，有：

\[
d_r^2=\|e\|^2-(e^\top u)^2
\]

各向异性拟合代价为：

\[
J_f=\int_0^1
\left(
\frac{d_a^2}{a^2}
+
\frac{d_r^2}{b^2}
\right)d\alpha
\]

将 \(d_r^2\) 展开：

\[
J_f=\int_0^1
\left[
\frac{(e^\top u)^2}{a^2}
+
\frac{\|e\|^2-(e^\top u)^2}{b^2}
\right]d\alpha
\]

整理得：

\[
J_f=\int_0^1
e^\top
\left[
\frac{1}{b^2}I+
\left(\frac{1}{a^2}-\frac{1}{b^2}\right)uu^\top
\right]
e
\,d\alpha
\]

定义各向异性度量矩阵：

\[
M(\alpha)=
\frac{1}{b^2}I+
\left(\frac{1}{a^2}-\frac{1}{b^2}\right)u(\alpha)u(\alpha)^\top
\]

则：

\[
J_f=\int_0^1 e(\alpha)^\top M(\alpha)e(\alpha)d\alpha
\]

对新轨迹点 \(\Phi_f\) 求导：

\[
\frac{\partial J_f}{\partial \Phi_f(\alpha T')}
=
2M(\alpha)e(\alpha)
\]

由于 \(\Phi_f\) 是 B 样条曲线，可以写为：

\[
\Phi_f(t)=\sum_k B_k(t)Q_k
\]

其中 \(B_k(t)\) 是第 \(k\) 个控制点对应的 B 样条基函数。因此：

\[
\frac{\partial \Phi_f(t)}{\partial Q_k}=B_k(t)I
\]

由链式法则：

\[
\frac{\partial J_f}{\partial Q_k}
=
\int_0^1
2B_k(\alpha T')M(\alpha)e(\alpha)
\,d\alpha
\]

实际实现中，积分会离散化为有限个采样点。若采样点为 \(\alpha_l\)，权重为 \(\omega_l\)，则：

\[
\frac{\partial J_f}{\partial Q_k}
\approx
\sum_l
2\omega_lB_k(\alpha_lT')M(\alpha_l)e(\alpha_l)
\]

考虑总目标函数权重后，各向异性拟合项贡献为：

\[
\lambda_f\frac{\partial J_f}{\partial Q_k}
\]

该梯度的意义是：如果新轨迹偏离原安全轨迹，优化器会将其拉回；但由于 \(M(\alpha)\) 在切向和径向上的权重不同，轨迹更容易沿切向调整，而不容易沿径向偏离安全通道。


#### 2.3.5 完整优化梯度汇总

在反弹优化阶段，EGO-Planner 对每个控制点 \(Q_k\) 使用的总梯度为：

\[
\frac{\partial J}{\partial Q_k}
=
\lambda_s\frac{\partial J_s}{\partial Q_k}
+
\lambda_c\frac{\partial J_c}{\partial Q_k}
+
\lambda_d\frac{\partial J_d}{\partial Q_k}
\]

其中：

\[
\frac{\partial J_s}{\partial Q_k}
=
\frac{\partial J_A}{\partial Q_k}
+
\frac{\partial J_J}{\partial Q_k}
\]

\[
\frac{\partial J_d}{\partial Q_k}
=
\frac{\partial J_{d,V}}{\partial Q_k}
+
\frac{\partial J_{d,A}}{\partial Q_k}
+
\frac{\partial J_{d,J}}{\partial Q_k}
\]

在时间重分配后的轨迹精修阶段，优化目标变为：

\[
J'=\lambda_sJ_s+\lambda_dJ_d+\lambda_fJ_f
\]

对应梯度为：

\[
\frac{\partial J'}{\partial Q_k}
=
\lambda_s\frac{\partial J_s}{\partial Q_k}
+
\lambda_d\frac{\partial J_d}{\partial Q_k}
+
\lambda_f\frac{\partial J_f}{\partial Q_k}
\]

因此，EGO-Planner 的优化本质可以概括为：平滑项降低轨迹高阶导数，碰撞项沿局部障碍物排斥方向推出控制点，动力学项将速度、加速度和 jerk 拉回可行范围，拟合项在时间重分配后保持轨迹形状不偏离原安全通道。整个过程不依赖 ESDF，而是只在轨迹碰撞处提取必要障碍物信息，从而显著降低了局部规划计算量。

##  3. EGO-Planner框架讲解


### 3.1 EGO-Planner 状态机切换流程

```text
INIT
  │
  ├── 没有 odom
  │       ↓
  │     保持 INIT
  │
  ├── 没有 trigger
  │       ↓
  │     保持 INIT
  │
  └── 有 odom 且有 trigger
          ↓
        WAIT_TARGET

WAIT_TARGET
  │
  ├── 没有 target
  │       ↓
  │     保持 WAIT_TARGET
  │
  └── 有 target
          ↓
        GEN_NEW_TRAJ

GEN_NEW_TRAJ
  │
  ├── callReboundReplan 成功
  │       ↓
  │     EXEC_TRAJ
  │
  └── callReboundReplan 失败
          ↓
        GEN_NEW_TRAJ
        连续失败后启用随机多项式初始化

EXEC_TRAJ
  │
  ├── 当前局部轨迹执行完
  │       ↓
  │     WAIT_TARGET
  │
  ├── 接近最终目标
  │       ↓
  │     保持 EXEC_TRAJ，不再重规划
  │
  ├── 还靠近当前局部轨迹起点
  │       ↓
  │     保持 EXEC_TRAJ，暂不重规划
  │
  └── 已离开起点且未接近终点
          ↓
        REPLAN_TRAJ

REPLAN_TRAJ
  │
  ├── planFromCurrentTraj 成功
  │       ↓
  │     EXEC_TRAJ
  │
  └── planFromCurrentTraj 失败
          ↓
        REPLAN_TRAJ

安全检测 checkCollisionCallback
  │
  ├── 当前轨迹未来无碰撞
  │       ↓
  │     不改变状态
  │
  └── 当前轨迹未来有碰撞
          ↓
        先尝试 planFromCurrentTraj
          │
          ├── 成功
          │     ↓
          │   EXEC_TRAJ
          │
          └── 失败
                │
                ├── 碰撞时间小于 emergency_time
                │     ↓
                │   EMERGENCY_STOP
                │
                └── 碰撞时间大于 emergency_time
                      ↓
                    REPLAN_TRAJ

EMERGENCY_STOP
  │
  ├── 第一次进入
  │       ↓
  │     callEmergencyStop 发布急停轨迹
  │
  └── 速度降到小于 0.1
          ↓
        GEN_NEW_TRAJ
```
### 3.2 EGO-Planner 核心 replan 部分

这里对应状态机的GEN_NEW_TRAJ和REPLAN_TRAJ的两个状态

```text
生成或继承一条初始几何路径
        ↓
采样得到 point_set
        ↓
构造起点速度、终点速度、起点加速度、终点加速度
        ↓
调用 parameterizeToBspline
将 point_set 和边界导数参数化为 B 样条控制点 ctrl_pts
        ↓
调用 initControlPoints
检测 ctrl_pts 对应控制点折线中的碰撞区间
        ↓
对每个碰撞区间执行 A* 搜索
        ↓
根据 A* 路径为碰撞相关控制点生成 base_point 和 direction
        ↓
调用 BsplineOptimizeTrajRebound(ctrl_pts, ts)
        ↓
设置 B 样条时间间隔 bspline_interval_
        ↓
进入 rebound_optimize
        ↓
固定首尾各 order_ 个控制点
只将中间控制点展开为 L-BFGS 优化变量 q
        ↓
调用 L-BFGS
根据 costFunctionRebound 计算代价和梯度
优化中间控制点
        ↓
优化后用新的 cps_.points 构造三次均匀 B 样条轨迹
        ↓
对轨迹前 2/3 进行膨胀障碍物碰撞检测
        ↓
如果靠近起始段发生碰撞
直接返回失败
        ↓
如果非起始段仍然碰撞
重新 initControlPoints
重新生成 base_point 和 direction
增大碰撞惩罚权重 new_lambda2_
再次优化
        ↓
如果碰撞检测通过
认为 rebound 优化成功
        ↓
将 cps_.points 写回 ctrl_pts
输出 rebound 优化后的 B 样条控制点
        ↓
构造 UniformBspline pos = UniformBspline(ctrl_pts, 3, ts)
        ↓
设置物理限制
包括最大速度 max_vel、最大加速度 max_acc、可行性容忍度 feasibility_tolerance
        ↓
调用 checkFeasibility(ratio, false)
检查当前 B 样条轨迹是否满足速度和加速度约束
        ↓
如果速度和加速度均满足限制
跳过时间重分配，直接保留当前轨迹
        ↓
如果速度或加速度超限
根据最大超限程度计算时间放大比例 ratio
        ↓
调用 refineTrajAlgo(pos, start_end_derivatives, ratio, ts, optimal_control_points)
进入时间重分配与轨迹精修阶段
        ↓
调用 reparamBspline
根据 ratio 重新分配轨迹时间
生成新的控制点 ctrl_pts 和新的时间间隔 ts
        ↓
用新的 ctrl_pts 和 ts 重新构造三次均匀 B 样条轨迹
        ↓
沿重新参数化后的轨迹采样
生成参考点序列 ref_pts_
        ↓
调用 BsplineOptimizeTrajRefine(ctrl_pts, ts, optimal_control_points)
        ↓
设置 refine 阶段的控制点和 B 样条时间间隔
        ↓
进入 refine_optimize
        ↓
固定首尾各 order_ 个控制点
只将中间控制点展开为 L-BFGS 优化变量 q
        ↓
调用 L-BFGS
根据 costFunctionRefine 计算代价和梯度
对时间重分配后的控制点进行精修优化
        ↓
优化后用新的 cps_.points 构造三次均匀 B 样条轨迹
        ↓
对 refined 轨迹前 2/3 进行膨胀障碍物碰撞检测
        ↓
如果 refined 轨迹发生碰撞
认为时间重分配后的轨迹不安全
返回失败
        ↓
如果 refined 轨迹无碰撞
将 cps_.points 写入 optimal_control_points
        ↓
外层用 optimal_control_points 重新构造最终 B 样条轨迹 pos
        ↓
得到满足避障约束和动力学可行性的最终局部轨迹
```

这里最终会被callReboundReplan调用，成功则发布B样条
在下面的话题被发布、接收
发布端
```
    ros::Publisher bspline_pub_;
    bspline_pub_ = nh.advertise<ego_planner::Bspline>("/planning/bspline", 10);
```
接收端
```
  ros::Subscriber bspline_sub = node.subscribe("planning/bspline", 10, bsplineCallback);
```
接收到的B样条马上会在轨迹规划部分执行

### 3.3 EGO-Planner 轨迹执行 exec_traj 部分

这里对应EXEC_TRAJ的部分
核心是将接收到的B样条根据时间发布出合理的指令

```
  ros::Subscriber bspline_sub = node.subscribe("planning/bspline", 10, bsplineCallback);
  pos_cmd_pub = node.advertise<quadrotor_msgs::PositionCommand>("/position_cmd", 50);
```

### 3.4 EGO-Planner 核心函数目录


```text
simple_run.launch
  -> include advanced_param.xml
  -> include simulator.xml
  -> include rviz.launch
  -> 启动 ego_planner_node、traj_server、waypoint_generator、仿真器和 RViz

advanced_param.xml
  -> 启动 ego_planner_node
  -> remap /odom_world 到 /visual_slam/odom
  -> remap /grid_map/cloud 到 /pcl_render_node/cloud
  -> 加载 FSM、GridMap、manager、optimization 参数

simulator.xml
  -> 启动 mockamap_node
  -> 启动 quadrotor_simulator_so3
  -> 启动 so3_control
  -> 启动 pcl_render_node
  -> 启动 odom_visualization

mockamap_node
  -> 生成全局障碍点云 /map_generator/global_cloud
  -> mocka::Maps::setInfo()
  -> mocka::Maps::generate()
  -> mocka::Maps::perlin3D()
  -> mocka::Maps::pcl2ros()

pcl_render_node
  -> 订阅 /map_generator/global_cloud
  -> 订阅 /visual_slam/odom
  -> 根据 odom 从全局点云裁剪局部可见点云
  -> 发布 /pcl_render_node/cloud
  -> rcvGlobalPointCloudCallBack()
  -> rcvOdometryCallbck()
  -> renderSensedPoints()

GridMap
  -> 订阅 /grid_map/cloud，实际 remap 到 /pcl_render_node/cloud
  -> 订阅 /grid_map/odom，实际 remap 到 /visual_slam/odom
  -> 将局部点云直接写入膨胀占据栅格 occupancy_buffer_inflate_
  -> GridMap::initMap()
  -> GridMap::odomCallback()
  -> GridMap::cloudCallback()
  -> GridMap::resetBuffer()
  -> GridMap::getInflateOccupancy()

EGOReplanFSM
  -> 接收 odom
  -> 接收目标点或使用预设航点
  -> 生成全局参考轨迹
  -> 选择局部目标点
  -> 触发局部重规划
  -> EGOReplanFSM::init()
  -> EGOReplanFSM::odometryCallback()
  -> EGOReplanFSM::waypointCallback()
  -> EGOReplanFSM::planGlobalTrajbyGivenWps()
  -> EGOReplanFSM::execFSMCallback()
  -> EGOReplanFSM::getLocalTarget()
  -> EGOReplanFSM::callReboundReplan()
  -> EGOReplanFSM::planFromCurrentTraj()
  -> EGOReplanFSM::checkCollisionCallback()

EGOPlannerManager::reboundReplan()
  -> 检查起点和局部目标点是否过近
    -> (start_pt - local_target_pt).norm()

  -> 计算 B 样条时间间隔 ts
    -> pp_.ctrl_pt_dist
    -> pp_.max_vel_

  -> 生成初始离散点 point_set
    -> flag_first_call
    -> flag_polyInit
    -> flag_force_polynomial
    -> flag_randomPolyTraj
    -> PolynomialTraj::one_segment_traj_gen()
    -> PolynomialTraj::minSnapTraj()
    -> PolynomialTraj::evaluate()
    -> PolynomialTraj::evaluateVel()
    -> PolynomialTraj::evaluateAcc()
    -> UniformBspline::evaluateDeBoorT()

  -> 将 point_set 拟合成初始 B 样条控制点 ctrl_pts
    -> UniformBspline::parameterizeToBspline()

  -> 提取碰撞段和 A* 引导
    -> BsplineOptimizer::initControlPoints()
      -> ControlPoints::resize()
      -> GridMap::getResolution()
      -> GridMap::getInflateOccupancy()
      -> AStar::AstarSearch()
      -> AStar::getPath()
      -> BsplineOptimizer 内部计算 segment_ids
      -> BsplineOptimizer 内部计算 bounds
      -> BsplineOptimizer 内部计算 final_segment_ids
      -> BsplineOptimizer 内部写入 cps_.base_point
      -> BsplineOptimizer 内部写入 cps_.direction

  -> 可视化初始轨迹和 A* 路径
    -> PlanningVisualization::displayInitPathList()
    -> PlanningVisualization::displayAStarList()

  -> 反弹优化 B 样条控制点
    -> BsplineOptimizer::BsplineOptimizeTrajRebound()
      -> BsplineOptimizer::setBsplineInterval()
      -> BsplineOptimizer::rebound_optimize()
        -> lbfgs::lbfgs_load_default_parameters()
        -> lbfgs::lbfgs_optimize()
        -> BsplineOptimizer::costFunctionRebound()
          -> BsplineOptimizer::combineCostRebound()
            -> BsplineOptimizer::calcSmoothnessCost()
            -> BsplineOptimizer::calcDistanceCostRebound()
              -> BsplineOptimizer::check_collision_and_rebound()
                -> GridMap::getInflateOccupancy()
                -> AStar::AstarSearch()
                -> AStar::getPath()
            -> BsplineOptimizer::calcFeasibilityCost()
        -> BsplineOptimizer::earlyExit()
        -> UniformBspline::UniformBspline()
        -> UniformBspline::getTimeSpan()
        -> UniformBspline::evaluateDeBoorT()
        -> GridMap::getInflateOccupancy()
        -> BsplineOptimizer::initControlPoints()

  -> 检查速度/加速度可行性
    -> UniformBspline::UniformBspline()
    -> UniformBspline::setPhysicalLimits()
    -> UniformBspline::checkFeasibility()
      -> UniformBspline 内部检查控制点速度约束
      -> UniformBspline 内部检查控制点加速度约束
      -> UniformBspline 内部返回 ratio

  -> 必要时重新分配时间并 refine
    -> EGOPlannerManager::refineTrajAlgo()
      -> EGOPlannerManager::reparamBspline()
        -> UniformBspline::getTimeSum()
        -> UniformBspline::getControlPoint()
        -> UniformBspline::lengthenTime()
        -> UniformBspline::evaluateDeBoorT()
        -> UniformBspline::parameterizeToBspline()
      -> UniformBspline::UniformBspline()
      -> UniformBspline::getTimeSum()
      -> UniformBspline::evaluateDeBoorT()
      -> BsplineOptimizer::BsplineOptimizeTrajRefine()
        -> BsplineOptimizer::setControlPoints()
        -> BsplineOptimizer::setBsplineInterval()
        -> BsplineOptimizer::refine_optimize()
          -> lbfgs::lbfgs_load_default_parameters()
          -> lbfgs::lbfgs_optimize()
          -> BsplineOptimizer::costFunctionRefine()
            -> BsplineOptimizer::combineCostRefine()
              -> BsplineOptimizer::calcSmoothnessCost()
              -> BsplineOptimizer::calcFitnessCost()
              -> BsplineOptimizer::calcFeasibilityCost()
          -> UniformBspline::UniformBspline()
          -> UniformBspline::getTimeSpan()
          -> UniformBspline::evaluateDeBoorT()
          -> GridMap::getInflateOccupancy()

  -> 保存本轮局部轨迹
    -> EGOPlannerManager::updateTrajInfo()
      -> UniformBspline::getDerivative()
      -> UniformBspline::evaluateDeBoorT()
      -> UniformBspline::getTimeSum()

EGOReplanFSM::callReboundReplan()
  -> 选择局部目标点
  -> 调用 EGOPlannerManager::reboundReplan()
  -> 发布 /planning/bspline
  -> EGOReplanFSM::getLocalTarget()
  -> EGOPlannerManager::reboundReplan()
  -> UniformBspline::getControlPoint()
  -> UniformBspline::getKnot()
  -> PlanningVisualization::displayOptimalList()

traj_server
  -> 订阅 /planning/bspline
  -> 将 B 样条采样成 /planning/pos_cmd
  -> bsplineCallback()
  -> UniformBspline::UniformBspline()
  -> UniformBspline::setKnot()
  -> UniformBspline::getDerivative()
  -> UniformBspline::getTimeSum()
  -> cmdCallback()
  -> UniformBspline::evaluateDeBoorT()
  -> calculate_yaw()

so3_control
  -> 订阅 /planning/pos_cmd
  -> 输出 so3_cmd
  -> SO3ControlNodelet::onInit()
  -> SO3ControlNodelet::position_cmd_callback()
  -> SO3ControlNodelet::odom_callback()
  -> SO3ControlNodelet::publishSO3Command()
  -> SO3Control::calculateControl()
  -> SO3Control::getComputedForce()
  -> SO3Control::getComputedOrientation()

quadrotor_simulator_so3
  -> 订阅 so3_cmd
  -> 更新四旋翼动力学状态
  -> 发布 /visual_slam/odom
  -> cmd_callback()
  -> getControl()
  -> QuadrotorSimulator::Quadrotor::setInput()
  -> QuadrotorSimulator::Quadrotor::step()
  -> stateToOdomMsg()
  -> quadToImuMsg()

odom 反馈
  -> 更新规划起点状态
  -> 更新局部感知视点
  -> 更新控制器反馈
  -> 推动下一轮重规划
  -> EGOReplanFSM::odometryCallback()
  -> GridMap::odomCallback()
  -> rcvOdometryCallbck()
  -> SO3ControlNodelet::odom_callback()
```