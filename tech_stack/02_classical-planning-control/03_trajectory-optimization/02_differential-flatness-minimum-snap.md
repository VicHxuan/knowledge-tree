# Minimum Snap 轨迹生成方法与当前 C++ 代码解析

## 1. 概述（Overview）

> 待补充。

## 2. 基础与定义（Foundations）

### 0. 统一符号

设轨迹由 \(m\) 段组成：

\[
m=\text{seg\_num}
\]

给定路径点：

\[
\mathbf q_0,\mathbf q_1,\dots,\mathbf q_m\in\mathbb R^3
\]

其中：

\[
\mathbf q_i=
\begin{bmatrix}
x_i\\
y_i\\
z_i
\end{bmatrix}
\]

每段时间为：

\[
T_i,\quad i=0,1,\dots,m-1
\]

对应代码中的：

\[
T_i=\text{Time}(i)
\]

对单个维度 \(w\in\{x,y,z\}\)，第 \(i\) 段多项式为：

\[
w_i(t)=\sum_{r=0}^{n}p_{i,r}t^r,\quad t\in[0,T_i]
\]

系数向量为：

\[
\mathbf p_i=
\begin{bmatrix}
p_{i,0}&p_{i,1}&\cdots&p_{i,n}
\end{bmatrix}^T
\]

三维轨迹为：

\[
\mathbf r_i(t)=
\begin{bmatrix}
x_i(t)\\
y_i(t)\\
z_i(t)
\end{bmatrix}
\]

最终输出为每一段、每一个维度的多项式系数：

\[
\left\{
\mathbf p_{x,i},
\mathbf p_{y,i},
\mathbf p_{z,i}
\right\}_{i=0}^{m-1}
\]

---

## 3. 核心原理（Core Principles）

### 1. 原始 Minimum Snap 轨迹生成方法

#### 1.1 输入与输出

输入：

\[
\mathcal I=
\left\{
\mathbf q_0,\dots,\mathbf q_m,
T_0,\dots,T_{m-1},
\mathbf v_s,\mathbf v_e,
\mathbf a_s,\mathbf a_e
\right\}
\]

其中：

\[
\mathbf v_s=\dot{\mathbf r}(0)
\]

\[
\mathbf v_e=\dot{\mathbf r}(T)
\]

\[
\mathbf a_s=\ddot{\mathbf r}(0)
\]

\[
\mathbf a_e=\ddot{\mathbf r}(T)
\]

输出：

\[
\mathcal O=
\left\{
\mathbf p_{x,i},
\mathbf p_{y,i},
\mathbf p_{z,i}
\right\}_{i=0}^{m-1}
\]

即：

\[
x_i(t)=\sum_{r=0}^{n}p_{x,i,r}t^r
\]

\[
y_i(t)=\sum_{r=0}^{n}p_{y,i,r}t^r
\]

\[
z_i(t)=\sum_{r=0}^{n}p_{z,i,r}t^r
\]

---

#### 1.2 分段多项式表示

原始论文以 flat output 表示四旋翼轨迹：

\[
\sigma(t)=
\begin{bmatrix}
x(t)\\
y(t)\\
z(t)\\
\psi(t)
\end{bmatrix}
\]

位置轨迹为：

\[
\mathbf r(t)=
\begin{bmatrix}
x(t)\\
y(t)\\
z(t)
\end{bmatrix}
\]

分段多项式形式为：

\[
\sigma(t)=
\begin{cases}
\sum_{r=0}^{n}\sigma_{0,r}t^r,&t_0\le t<t_1\\
\sum_{r=0}^{n}\sigma_{1,r}t^r,&t_1\le t<t_2\\
\vdots\\
\sum_{r=0}^{n}\sigma_{m-1,r}t^r,&t_{m-1}\le t\le t_m
\end{cases}
\]

对单维度 \(w\)：

\[
w_i(t)=\sum_{r=0}^{n}p_{i,r}t^r
\]

其 \(k\) 阶导数为：

\[
w_i^{(k)}(t)=\sum_{r=k}^{n}\frac{r!}{(r-k)!}p_{i,r}t^{r-k}
\]

---

#### 1.3 Minimum Snap 目标函数

minimum snap 最小化位置四阶导数平方积分：

\[
J_s=\int_{0}^{T}\left|\mathbf r^{(4)}(t)\right|^2dt
\]

展开为：

\[
J_s=\int_{0}^{T}\left[x^{(4)}(t)^2+y^{(4)}(t)^2+z^{(4)}(t)^2\right]dt
\]

多段形式为：

\[
J_s=\sum_{i=0}^{m-1}\int_{0}^{T_i}\left|\mathbf r_i^{(4)}(t)\right|^2dt
\]

若同时优化 yaw：

\[
J_\psi=\int_{0}^{T}\left[\psi^{(2)}(t)\right]^2dt
\]

综合目标为：

\[
J=\mu_r\int_{0}^{T}\left|\mathbf r^{(4)}(t)\right|^2dt+\mu_\psi\int_{0}^{T}\left[\psi^{(2)}(t)\right]^2dt
\]

本文后续只讨论位置部分。

---

#### 1.4 单段代价矩阵

对单维度第 \(i\) 段：

\[
w_i(t)=\sum_{r=0}^{n}p_{i,r}t^r
\]

minimum snap 代价为：

\[
J_i=\int_0^{T_i}\left[w_i^{(4)}(t)\right]^2dt
\]

四阶导数为：

\[
w_i^{(4)}(t)=\sum_{r=4}^{n}\frac{r!}{(r-4)!}p_{i,r}t^{r-4}
\]

代入目标函数：

\[
J_i=\sum_{r=4}^{n}\sum_{s=4}^{n}p_{i,r}p_{i,s}\frac{r!}{(r-4)!}\frac{s!}{(s-4)!}\int_0^{T_i}t^{r+s-8}dt
\]

积分项为：

\[
\int_0^{T_i}t^{r+s-8}dt=\frac{T_i^{r+s-7}}{r+s-7}
\]

因此：

\[
J_i=\mathbf p_i^TQ_i\mathbf p_i
\]

其中：

\[
(Q_i)_{r,s}=\frac{r!}{(r-4)!}\frac{s!}{(s-4)!}\frac{T_i^{r+s-7}}{r+s-7},\quad r,s\ge4
\]

若 \(r<4\) 或 \(s<4\)，则：

\[
(Q_i)_{r,s}=0
\]

---

#### 1.5 多段整体代价矩阵

将所有段系数拼接：

\[
\mathbf p=
\begin{bmatrix}
\mathbf p_0\\
\mathbf p_1\\
\vdots\\
\mathbf p_{m-1}
\end{bmatrix}
\]

整体代价为：

\[
J=\mathbf p^TQ\mathbf p
\]

其中：

\[
Q=\operatorname{blkdiag}(Q_0,Q_1,\dots,Q_{m-1})
\]

---

#### 1.6 约束条件

路径点约束：

\[
\mathbf r_i(0)=\mathbf q_i
\]

\[
\mathbf r_i(T_i)=\mathbf q_{i+1}
\]

起点导数约束：

\[
\dot{\mathbf r}_0(0)=\mathbf v_s
\]

\[
\ddot{\mathbf r}_0(0)=\mathbf a_s
\]

终点导数约束：

\[
\dot{\mathbf r}_{m-1}(T_{m-1})=\mathbf v_e
\]

\[
\ddot{\mathbf r}_{m-1}(T_{m-1})=\mathbf a_e
\]

中间点连续性约束：

\[
\mathbf r_i^{(k)}(T_i)=\mathbf r_{i+1}^{(k)}(0),\quad k=0,1,2,3
\]

minimum snap 通常要求至少三阶导数连续。

---

#### 1.7 原始 QP 形式

将所有维度的多项式系数拼接为：

\[
\mathbf c
\]

原始论文将问题写成二次规划：

\[
\min_{\mathbf c}\quad\mathbf c^TH\mathbf c+\mathbf f^T\mathbf c
\]

满足：

\[
A\mathbf c\le \mathbf b
\]

若只考虑等式约束，可写为：

\[
\min_{\mathbf p}\quad\mathbf p^TQ\mathbf p
\]

\[
\text{s.t.}\quad A_{eq}\mathbf p=\mathbf b_{eq}
\]

对应 KKT 系统为：

\[
\begin{bmatrix}
2Q&A_{eq}^T\\
A_{eq}&0
\end{bmatrix}
\begin{bmatrix}
\mathbf p\\
\boldsymbol\lambda
\end{bmatrix}
=
\begin{bmatrix}
0\\
\mathbf b_{eq}
\end{bmatrix}
\]

求解后得到：

\[
\mathbf p^*
\]

即所有分段多项式系数。

---

#### 1.8 原始方法流程

```text
Algorithm Original_Minimum_Snap

Input:
    q_0 ... q_m          路径点
    T_0 ... T_{m-1}      每段时间
    v_s, v_e             起点和终点速度
    a_s, a_e             起点和终点加速度

Output:
    p_x, p_y, p_z        每段多项式系数

1:  选择多项式阶数 n
2:  对每一段构造 w_i(t)=Σ p_{i,r} t^r
3:  构造每段 snap cost 矩阵 Q_i
4:  拼接整体矩阵 Q=blkdiag(Q_0,...,Q_{m-1})
5:  构造路径点约束
6:  构造起点和终点导数约束
7:  构造中间点连续性约束
8:  求解 constrained QP
9:  输出每段多项式系数
```

原始方法的本质是：

\[
\boxed{
\min_{\mathbf p}\mathbf p^TQ\mathbf p\quad\text{s.t.}\quad A_{eq}\mathbf p=\mathbf b_{eq}
}
\]

#### 1.9 KKT 方法求解示例

为了更直观地理解上一节中的 KKT 系统，下面给出一个单维度、单段轨迹的求解示例。

考虑单维度轨迹：

$$
w(t)=\sum_{r=0}^{7}p_rt^r
$$

即采用 7 次多项式：

$$
w(t)=p_0+p_1t+p_2t^2+p_3t^3+p_4t^4+p_5t^5+p_6t^6+p_7t^7
$$

设轨迹时间为：

$$
T=1
$$

边界条件为：

$$
w(0)=0
$$

$$
\dot w(0)=0
$$

$$
\ddot w(0)=0
$$

$$
w^{(3)}(0)=0
$$

$$
w(1)=1
$$

$$
\dot w(1)=0
$$

$$
\ddot w(1)=0
$$

$$
w^{(3)}(1)=0
$$

也就是说，该轨迹需要从位置 (0) 平滑运动到位置 (1)，并且起点和终点的速度、加速度、jerk 均为 (0)。

---

##### 1.9.1 优化变量

将多项式系数写成向量：

$$
\mathbf p=
\begin{bmatrix}
p_0&p_1&p_2&p_3&p_4&p_5&p_6&p_7
\end{bmatrix}^T
$$

minimum snap 的优化目标为：

$$
\min_{\mathbf p}\quad \int_0^1 \left[w^{(4)}(t)\right]^2dt
$$

写成二次型：

$$
\min_{\mathbf p}\quad \mathbf p^TQ\mathbf p
$$

其中：

$$
w^{(4)}(t)=24p_4+120p_5t+360p_6t^2+840p_7t^3
$$

因此，只有 (p_4,p_5,p_6,p_7) 会影响 snap 代价，代价矩阵为：

$$
Q=
\begin{bmatrix}
0&0&0&0&0&0&0&0\
0&0&0&0&0&0&0&0\
0&0&0&0&0&0&0&0\
0&0&0&0&0&0&0&0\
0&0&0&0&576&1440&2880&5040\
0&0&0&0&1440&4800&10800&20160\
0&0&0&0&2880&10800&25920&50400\
0&0&0&0&5040&20160&50400&100800
\end{bmatrix}
$$

---

##### 1.9.2 构造等式约束

由起点约束可得：

$$
w(0)=p_0=0
$$

$$
\dot w(0)=p_1=0
$$

$$
\ddot w(0)=2p_2=0
$$

$$
w^{(3)}(0)=6p_3=0
$$

由终点约束可得：

$$
w(1)=p_0+p_1+p_2+p_3+p_4+p_5+p_6+p_7=1
$$

$$
\dot w(1)=p_1+2p_2+3p_3+4p_4+5p_5+6p_6+7p_7=0
$$

$$
\ddot w(1)=2p_2+6p_3+12p_4+20p_5+30p_6+42p_7=0
$$

$$
w^{(3)}(1)=6p_3+24p_4+60p_5+120p_6+210p_7=0
$$

因此，等式约束可以写成：

$$
A_{eq}\mathbf p=\mathbf b_{eq}
$$

其中：

$$
A_{eq}=
\begin{bmatrix}
1&0&0&0&0&0&0&0\
0&1&0&0&0&0&0&0\
0&0&2&0&0&0&0&0\
0&0&0&6&0&0&0&0\
1&1&1&1&1&1&1&1\
0&1&2&3&4&5&6&7\
0&0&2&6&12&20&30&42\
0&0&0&6&24&60&120&210
\end{bmatrix}
$$

$$
\mathbf b_{eq}=
\begin{bmatrix}
0\
0\
0\
0\
1\
0\
0\
0
\end{bmatrix}
$$

---

##### 1.9.3 构造 KKT 系统

原始优化问题为：

$$
\min_{\mathbf p}\quad \mathbf p^TQ\mathbf p
$$

$$
\text{s.t.}\quad A_{eq}\mathbf p=\mathbf b_{eq}
$$

构造拉格朗日函数：

$$
\mathcal L(\mathbf p,\boldsymbol\lambda)
========================================

\mathbf p^TQ\mathbf p
+
\boldsymbol\lambda^T(A_{eq}\mathbf p-\mathbf b_{eq})
$$

对 (\mathbf p) 求导：

$$
\frac{\partial \mathcal L}{\partial \mathbf p}
==============================================

2Q\mathbf p+A_{eq}^T\boldsymbol\lambda
$$

令梯度为零：

$$
2Q\mathbf p+A_{eq}^T\boldsymbol\lambda=0
$$

同时还需要满足原始等式约束：

$$
A_{eq}\mathbf p=\mathbf b_{eq}
$$

因此可得到 KKT 线性系统：

$$
\begin{bmatrix}
2Q&A_{eq}^T\
A_{eq}&0
\end{bmatrix}
\begin{bmatrix}
\mathbf p\
\boldsymbol\lambda
\end{bmatrix}
=============

\begin{bmatrix}
0\
\mathbf b_{eq}
\end{bmatrix}
$$

求解该线性系统后，即可得到最优多项式系数：

$$
\mathbf p^*
$$

---

##### 1.9.4 KKT 系统求解结果

对于上述示例，求解 KKT 系统可以得到：

$$
\mathbf p^*=
\begin{bmatrix}
0\
0\
0\
0\
35\
-84\
70\
-20
\end{bmatrix}
$$

因此，最终轨迹为：

$$
w(t)=35t^4-84t^5+70t^6-20t^7
$$

可以验证：

$$
w(0)=0
$$

$$
w(1)=1
$$

$$
\dot w(0)=0
$$

$$
\dot w(1)=0
$$

$$
\ddot w(0)=0
$$

$$
\ddot w(1)=0
$$

$$
w^{(3)}(0)=0
$$

$$
w^{(3)}(1)=0
$$

因此，该轨迹满足所有边界约束，并且在这些约束下最小化：

$$
\int_0^1 \left[w^{(4)}(t)\right]^2dt
$$

---

##### 1.9.5 对应的 Eigen 求解代码

上述 KKT 系统可以直接使用 Eigen 求解。

```cpp
#include <Eigen/Dense>
#include <iostream>

int main()
{
  constexpr int n_coef = 8;
  constexpr int n_cons = 8;

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n_coef, n_coef);

  Q(4, 4) = 576;
  Q(4, 5) = 1440;
  Q(4, 6) = 2880;
  Q(4, 7) = 5040;

  Q(5, 4) = 1440;
  Q(5, 5) = 4800;
  Q(5, 6) = 10800;
  Q(5, 7) = 20160;

  Q(6, 4) = 2880;
  Q(6, 5) = 10800;
  Q(6, 6) = 25920;
  Q(6, 7) = 50400;

  Q(7, 4) = 5040;
  Q(7, 5) = 20160;
  Q(7, 6) = 50400;
  Q(7, 7) = 100800;

  Eigen::MatrixXd Aeq(n_cons, n_coef);
  Aeq << 1, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 0, 6, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1,
         0, 1, 2, 3, 4, 5, 6, 7,
         0, 0, 2, 6, 12, 20, 30, 42,
         0, 0, 0, 6, 24, 60, 120, 210;

  Eigen::VectorXd beq(n_cons);
  beq << 0, 0, 0, 0, 1, 0, 0, 0;

  Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(n_coef + n_cons, n_coef + n_cons);

  KKT.block(0, 0, n_coef, n_coef) = 2.0 * Q;
  KKT.block(0, n_coef, n_coef, n_cons) = Aeq.transpose();
  KKT.block(n_coef, 0, n_cons, n_coef) = Aeq;

  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_coef + n_cons);
  rhs.segment(n_coef, n_cons) = beq;

  Eigen::VectorXd solution = KKT.fullPivLu().solve(rhs);

  Eigen::VectorXd p = solution.segment(0, n_coef);

  std::cout << "Polynomial coefficients:" << std::endl;
  std::cout << p.transpose() << std::endl;

  return 0;
}
```

输出结果为：

```text
0 0 0 0 35 -84 70 -20
```

即：

$$
w(t)=35t^4-84t^5+70t^6-20t^7
$$

---

##### 1.9.6 与多段轨迹的关系

上述示例只考虑了单维度、单段轨迹，因此所有约束都来自起点和终点边界条件。

对于多段轨迹，优化变量扩展为：

$$
\mathbf p=
\begin{bmatrix}
\mathbf p_0\
\mathbf p_1\
\vdots\
\mathbf p_{m-1}
\end{bmatrix}
$$

整体代价矩阵为：

$$
Q=\operatorname{blkdiag}(Q_0,Q_1,\dots,Q_{m-1})
$$

等式约束矩阵 (A_{eq}) 需要包含路径点约束：

$$
w_i(0)=w_i
$$

$$
w_i(T_i)=w_{i+1}
$$

起点边界导数约束：

$$
\dot w_0(0)=v_s
$$

$$
\ddot w_0(0)=a_s
$$

终点边界导数约束：

$$
\dot w_{m-1}(T_{m-1})=v_e
$$

$$
\ddot w_{m-1}(T_{m-1})=a_e
$$

以及中间点连续性约束：

$$
w_i^{(k)}(T_i)=w_{i+1}^{(k)}(0)
$$

其中通常取：

$$
k=0,1,2,3
$$

最终仍然通过同一个 KKT 系统求解：

$$
\begin{bmatrix}
2Q&A_{eq}^T\
A_{eq}&0
\end{bmatrix}
\begin{bmatrix}
\mathbf p\
\boldsymbol\lambda
\end{bmatrix}
=============

\begin{bmatrix}
0\
\mathbf b_{eq}
\end{bmatrix}
$$

该方法的核心思想是：直接以多项式系数 (\mathbf p) 为优化变量，将路径点、边界导数和连续性条件全部写入 (A_{eq}\mathbf p=\mathbf b_{eq})，然后通过 KKT 线性系统一次性求解所有轨迹系数。


---

### 2. Polynomial 论文中的改进方法

#### 2.1 核心思想

原始方法直接优化多项式系数：

\[
\mathbf p
\]

Polynomial 方法改为优化端点导数：

\[
\mathbf d
\]

多项式系数和端点导数之间满足：

\[
A\mathbf p=\mathbf d
\]

因此：

\[
\mathbf p=A^{-1}\mathbf d
\]

将其代入代价函数后，可将问题转为关于端点导数的无约束二次优化。

---
注意，这里是无约束二次优化
因此SFC引入的“凸约束”在这里不再起作用

#### 2.2 输入与输出

输入保持不变：

\[
\mathcal I=
\left\{
\mathbf q_0,\dots,\mathbf q_m,
T_0,\dots,T_{m-1},
\mathbf v_s,\mathbf v_e,
\mathbf a_s,\mathbf a_e
\right\}
\]

输出保持不变：

\[
\mathcal O=
\left\{
\mathbf p_{x,i},
\mathbf p_{y,i},
\mathbf p_{z,i}
\right\}_{i=0}^{m-1}
\]

---

#### 2.3 端点导数向量

对第 \(i\) 段、单维度 \(w\)，定义：

\[
\mathbf d_i=
\begin{bmatrix}
w_i(0)\\
w_i(T_i)\\
w_i^{(1)}(0)\\
w_i^{(1)}(T_i)\\
w_i^{(2)}(0)\\
w_i^{(2)}(T_i)\\
\vdots
\end{bmatrix}
\]

标准 minimum snap 通常使用到三阶导数：

\[
\mathbf d_i=
\begin{bmatrix}
w_i(0)\\
w_i(T_i)\\
w_i^{(1)}(0)\\
w_i^{(1)}(T_i)\\
w_i^{(2)}(0)\\
w_i^{(2)}(T_i)\\
w_i^{(3)}(0)\\
w_i^{(3)}(T_i)
\end{bmatrix}
\]

当前 C++ 代码使用到二阶导数：

\[
\mathbf d_i=
\begin{bmatrix}
w_i(0)\\
w_i(T_i)\\
w_i^{(1)}(0)\\
w_i^{(1)}(T_i)\\
w_i^{(2)}(0)\\
w_i^{(2)}(T_i)
\end{bmatrix}
\]

---

#### 2.4 Mapping Matrix

对当前代码中的五次多项式：

\[
w_i(t)=p_0+p_1t+p_2t^2+p_3t^3+p_4t^4+p_5t^5
\]

有：

\[
\mathbf p_i=
\begin{bmatrix}
p_0&p_1&p_2&p_3&p_4&p_5
\end{bmatrix}^T
\]

端点导数满足：

\[
A_i\mathbf p_i=\mathbf d_i
\]

其中：

\[
A_i=
\begin{bmatrix}
1&0&0&0&0&0\\
1&T_i&T_i^2&T_i^3&T_i^4&T_i^5\\
0&1&0&0&0&0\\
0&1&2T_i&3T_i^2&4T_i^3&5T_i^4\\
0&0&2&0&0&0\\
0&0&2&6T_i&12T_i^2&20T_i^3
\end{bmatrix}
\]

多段整体矩阵为：

\[
A=\operatorname{blkdiag}(A_0,A_1,\dots,A_{m-1})
\]

因此：

\[
A\mathbf p=\mathbf d
\]

\[
\mathbf p=A^{-1}\mathbf d
\]

---

#### 2.5 从系数空间转到导数空间

原代价函数为：

\[
J=\mathbf p^TQ\mathbf p
\]

代入：

\[
\mathbf p=A^{-1}\mathbf d
\]

得到：

\[
J=(A^{-1}\mathbf d)^TQ(A^{-1}\mathbf d)
\]

因此：

\[
J=\mathbf d^TA^{-T}QA^{-1}\mathbf d
\]

---

#### 2.6 固定导数与自由导数

将端点导数分为固定导数和自由导数：

\[
\tilde{\mathbf d}=
\begin{bmatrix}
\mathbf d_F\\
\mathbf d_P
\end{bmatrix}
\]

其中：

\[
\mathbf d_F
\]

表示固定导数，包括路径点位置、起点速度、起点加速度、终点速度、终点加速度。

\[
\mathbf d_P
\]

表示自由导数，包括中间路径点速度和加速度。

通过选择矩阵 \(C_t\) 映射回原始导数顺序：

\[
\mathbf d=C_t\tilde{\mathbf d}
\]

代入目标函数：

\[
J=\tilde{\mathbf d}^TC_t^TA^{-T}QA^{-1}C_t\tilde{\mathbf d}
\]

定义：

\[
R=C_t^TA^{-T}QA^{-1}C_t
\]

则：

\[
J=\tilde{\mathbf d}^TR\tilde{\mathbf d}
\]

---

#### 2.7 分块矩阵

按照固定导数和自由导数划分：

\[
R=
\begin{bmatrix}
R_{FF}&R_{FP}\\
R_{PF}&R_{PP}
\end{bmatrix}
\]

于是：

\[
J=
\begin{bmatrix}
\mathbf d_F\\
\mathbf d_P
\end{bmatrix}^T
\begin{bmatrix}
R_{FF}&R_{FP}\\
R_{PF}&R_{PP}
\end{bmatrix}
\begin{bmatrix}
\mathbf d_F\\
\mathbf d_P
\end{bmatrix}
\]

展开：

\[
J=\mathbf d_F^TR_{FF}\mathbf d_F+\mathbf d_F^TR_{FP}\mathbf d_P+\mathbf d_P^TR_{PF}\mathbf d_F+\mathbf d_P^TR_{PP}\mathbf d_P
\]

---

#### 2.8 自由导数闭式解

对 \(\mathbf d_P\) 求导：

\[
\frac{\partial J}{\partial \mathbf d_P}=2R_{PF}\mathbf d_F+2R_{PP}\mathbf d_P
\]

令梯度为零：

\[
2R_{PF}\mathbf d_F+2R_{PP}\mathbf d_P=0
\]

得到：

\[
R_{PP}\mathbf d_P=-R_{PF}\mathbf d_F
\]

因此：

\[
\mathbf d_P^*=-R_{PP}^{-1}R_{PF}\mathbf d_F
\]

由于：

\[
R_{PF}=R_{FP}^T
\]

所以：

\[
\boxed{
\mathbf d_P^*=-R_{PP}^{-1}R_{FP}^T\mathbf d_F
}
\]

---

#### 2.9 恢复多项式系数

求得：

\[
\tilde{\mathbf d}^*=
\begin{bmatrix}
\mathbf d_F\\
\mathbf d_P^*
\end{bmatrix}
\]

恢复原始导数顺序：

\[
\mathbf d^*=C_t\tilde{\mathbf d}^*
\]

恢复多项式系数：

\[
\boxed{
\mathbf p^*=A^{-1}C_t
\begin{bmatrix}
\mathbf d_F\\
\mathbf d_P^*
\end{bmatrix}
}
\]

---

#### 2.10 Polynomial 方法流程

```text
Algorithm Polynomial_Minimum_Snap

Input:
    q_0 ... q_m          路径点
    T_0 ... T_{m-1}      每段时间
    v_s, v_e             起点和终点速度
    a_s, a_e             起点和终点加速度

Output:
    p_x, p_y, p_z        每段多项式系数

1:  对每个维度 w ∈ {x,y,z} 独立求解
2:  构造端点导数向量 d
3:  构造映射矩阵 A，使 A p = d
4:  构造代价矩阵 Q
5:  构造选择矩阵 C_t，使 d = C_t [d_F; d_P]
6:  计算 R = C_t^T A^{-T} Q A^{-1} C_t
7:  将 R 分块为 R_FF, R_FP, R_PF, R_PP
8:  计算 d_P^* = -R_PP^{-1} R_FP^T d_F
9:  计算 p^* = A^{-1} C_t [d_F; d_P^*]
10: 输出每段多项式系数
```

Polynomial 方法本质为：

\[
\boxed{
\mathbf d_P^*=-R_{PP}^{-1}R_{FP}^T\mathbf d_F
}
\]

\[
\boxed{
\mathbf p^*=A^{-1}C_t
\begin{bmatrix}
\mathbf d_F\\
\mathbf d_P^*
\end{bmatrix}
}
\]

---

### 5. 当前代码的完整数学问题

对单维度 \(w\)，当前代码求解：

\[
\min\quad\sum_{i=0}^{m-1}\int_0^{T_i}\left[w_i^{(3)}(t)\right]^2dt
\]

约束为：

\[
w_i(0)=w_i
\]

\[
w_i(T_i)=w_{i+1}
\]

起点约束：

\[
\dot w_0(0)=v_s
\]

\[
\ddot w_0(0)=a_s
\]

终点约束：

\[
\dot w_{m-1}(T_{m-1})=v_e
\]

\[
\ddot w_{m-1}(T_{m-1})=a_e
\]

中间速度连续：

\[
\dot w_{i-1}(T_{i-1})=\dot w_i(0),\quad i=1,\dots,m-1
\]

中间加速度连续：

\[
\ddot w_{i-1}(T_{i-1})=\ddot w_i(0),\quad i=1,\dots,m-1
\]

因此当前代码生成的是：

\[
C^2
\]

连续的五次多项式轨迹。

---

## 4. 实现与应用（Implementation & Applications）

### 3. EGO-Planner实际 C++ 代码逐段解析

#### 3.1 当前代码的实际优化目标

当前代码使用五次多项式：

\[
w_i(t)=p_0+p_1t+p_2t^2+p_3t^3+p_4t^4+p_5t^5
\]

代码中的代价矩阵从三阶导数开始构造：

```cpp
for (int i = 3; i < 6; i++)
```

因此实际代价为：

\[
J=\sum_{i=0}^{m-1}\int_0^{T_i}\left[w_i^{(3)}(t)\right]^2dt
\]

这对应 minimum jerk，而不是标准 minimum snap。

标准 minimum snap 应为：

\[
J_s=\sum_{i=0}^{m-1}\int_0^{T_i}\left[w_i^{(4)}(t)\right]^2dt
\]

因此当前代码的结构属于 Polynomial 方法，但具体目标是五次多项式 minimum jerk。

---

#### 3.2 函数定义与变量初始化

```cpp
PolynomialTraj PolynomialTraj::minSnapTraj(const Eigen::MatrixXd &Pos, const Eigen::Vector3d &start_vel,
                                           const Eigen::Vector3d &end_vel, const Eigen::Vector3d &start_acc,
                                           const Eigen::Vector3d &end_acc, const Eigen::VectorXd &Time)
{
  int seg_num = Time.size();
  Eigen::MatrixXd poly_coeff(seg_num, 3 * 6);
  Eigen::VectorXd Px(6 * seg_num), Py(6 * seg_num), Pz(6 * seg_num);

  int num_f, num_p; // number of fixed and free variables
  int num_d;        // number of all segments' derivatives
```

路径点矩阵为：

\[
Pos=
\begin{bmatrix}
x_0&x_1&\cdots&x_m\\
y_0&y_1&\cdots&y_m\\
z_0&z_1&\cdots&z_m
\end{bmatrix}
\]

段数为：

\[
m=\text{seg\_num}
\]

每段五次多项式有 6 个系数，三维共有 18 个系数：

\[
\text{poly\_coeff}\in\mathbb R^{m\times18}
\]

所有 \(x\) 方向系数拼接为：

\[
P_x=
\begin{bmatrix}
\mathbf p_{x,0}\\
\mathbf p_{x,1}\\
\vdots\\
\mathbf p_{x,m-1}
\end{bmatrix}
\in\mathbb R^{6m}
\]

同理：

\[
P_y,P_z\in\mathbb R^{6m}
\]

---

#### 3.3 阶乘函数

```cpp
  const static auto Factorial = [](int x) {
    int fac = 1;
    for (int i = x; i > 0; i--)
      fac = fac * i;
    return fac;
  };
```

该函数计算：

\[
x!
\]

用于构造多项式导数系数：

\[
\frac{d^k}{dt^k}t^j=\frac{j!}{(j-k)!}t^{j-k}
\]

---

#### 3.4 构造端点导数向量

```cpp
  /* ---------- end point derivative ---------- */
  Eigen::VectorXd Dx = Eigen::VectorXd::Zero(seg_num * 6);
  Eigen::VectorXd Dy = Eigen::VectorXd::Zero(seg_num * 6);
  Eigen::VectorXd Dz = Eigen::VectorXd::Zero(seg_num * 6);
```

对第 \(k\) 段、单维度 \(w\)，端点导数顺序为：

\[
\mathbf d_k=
\begin{bmatrix}
w_k(0)\\
w_k(T_k)\\
\dot w_k(0)\\
\dot w_k(T_k)\\
\ddot w_k(0)\\
\ddot w_k(T_k)
\end{bmatrix}
\]

整体向量为：

\[
D_w=
\begin{bmatrix}
\mathbf d_0\\
\mathbf d_1\\
\vdots\\
\mathbf d_{m-1}
\end{bmatrix}
\in\mathbb R^{6m}
\]

代码分别构造：

\[
D_x,D_y,D_z
\]

---

#### 3.5 写入每段起点和终点位置

```cpp
  for (int k = 0; k < seg_num; k++)
  {
    /* position to derivative */
    Dx(k * 6) = Pos(0, k);
    Dx(k * 6 + 1) = Pos(0, k + 1);
    Dy(k * 6) = Pos(1, k);
    Dy(k * 6 + 1) = Pos(1, k + 1);
    Dz(k * 6) = Pos(2, k);
    Dz(k * 6 + 1) = Pos(2, k + 1);
```

对应：

\[
x_k(0)=x_k
\]

\[
x_k(T_k)=x_{k+1}
\]

\[
y_k(0)=y_k
\]

\[
y_k(T_k)=y_{k+1}
\]

\[
z_k(0)=z_k
\]

\[
z_k(T_k)=z_{k+1}
\]

即：

\[
\mathbf r_k(0)=\mathbf q_k
\]

\[
\mathbf r_k(T_k)=\mathbf q_{k+1}
\]

---

#### 3.6 写入起点速度和起点加速度

```cpp
    if (k == 0)
    {
      Dx(k * 6 + 2) = start_vel(0);
      Dy(k * 6 + 2) = start_vel(1);
      Dz(k * 6 + 2) = start_vel(2);

      Dx(k * 6 + 4) = start_acc(0);
      Dy(k * 6 + 4) = start_acc(1);
      Dz(k * 6 + 4) = start_acc(2);
    }
```

起点速度约束：

\[
\dot{\mathbf r}_0(0)=\mathbf v_s
\]

即：

\[
\dot x_0(0)=v_{s,x}
\]

\[
\dot y_0(0)=v_{s,y}
\]

\[
\dot z_0(0)=v_{s,z}
\]

起点加速度约束：

\[
\ddot{\mathbf r}_0(0)=\mathbf a_s
\]

即：

\[
\ddot x_0(0)=a_{s,x}
\]

\[
\ddot y_0(0)=a_{s,y}
\]

\[
\ddot z_0(0)=a_{s,z}
\]

---

#### 3.7 写入终点速度和终点加速度

```cpp
    else if (k == seg_num - 1)
    {
      Dx(k * 6 + 3) = end_vel(0);
      Dy(k * 6 + 3) = end_vel(1);
      Dz(k * 6 + 3) = end_vel(2);

      Dx(k * 6 + 5) = end_acc(0);
      Dy(k * 6 + 5) = end_acc(1);
      Dz(k * 6 + 5) = end_acc(2);
    }
  }
```

终点速度约束：

\[
\dot{\mathbf r}_{m-1}(T_{m-1})=\mathbf v_e
\]

终点加速度约束：

\[
\ddot{\mathbf r}_{m-1}(T_{m-1})=\mathbf a_e
\]

---

#### 3.8 构造 Mapping Matrix \(A\)

```cpp
  /* ---------- Mapping Matrix A ---------- */
  Eigen::MatrixXd Ab;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(seg_num * 6, seg_num * 6);

  for (int k = 0; k < seg_num; k++)
  {
    Ab = Eigen::MatrixXd::Zero(6, 6);
    for (int i = 0; i < 3; i++)
    {
      Ab(2 * i, i) = Factorial(i);
      for (int j = i; j < 6; j++)
        Ab(2 * i + 1, j) = Factorial(j) / Factorial(j - i) * pow(Time(k), j - i);
    }
    A.block(k * 6, k * 6, 6, 6) = Ab;
  }
```

第 \(k\) 段五次多项式为：

\[
w_k(t)=p_0+p_1t+p_2t^2+p_3t^3+p_4t^4+p_5t^5
\]

端点导数与系数满足：

\[
A_k\mathbf p_k=\mathbf d_k
\]

其中：

\[
A_k=
\begin{bmatrix}
1&0&0&0&0&0\\
1&T_k&T_k^2&T_k^3&T_k^4&T_k^5\\
0&1&0&0&0&0\\
0&1&2T_k&3T_k^2&4T_k^3&5T_k^4\\
0&0&2&0&0&0\\
0&0&2&6T_k&12T_k^2&20T_k^3
\end{bmatrix}
\]

整体矩阵为：

\[
A=\operatorname{blkdiag}(A_0,A_1,\dots,A_{m-1})
\]

满足：

\[
A\mathbf p=\mathbf d
\]

因此：

\[
\mathbf p=A^{-1}\mathbf d
\]

---

#### 3.9 构造选择矩阵 \(C_t\)

```cpp
  /* ---------- Produce Selection Matrix C' ---------- */
  Eigen::MatrixXd Ct, C;

  num_f = 2 * seg_num + 4; // 3 + 3 + (seg_num - 1) * 2 = 2m + 4
  num_p = 2 * seg_num - 2; //(seg_num - 1) * 2 = 2m - 2
  num_d = 6 * seg_num;
  Ct = Eigen::MatrixXd::Zero(num_d, num_f + num_p);
```

总导数数量为：

\[
n_d=6m
\]

固定变量数量为：

\[
n_f=2m+4
\]

自由变量数量为：

\[
n_p=2m-2
\]

选择矩阵：

\[
C_t\in\mathbb R^{6m\times(4m+2)}
\]

满足：

\[
\mathbf d=C_t
\begin{bmatrix}
\mathbf d_F\\
\mathbf d_P
\end{bmatrix}
\]

其中：

\[
\mathbf d_F\in\mathbb R^{2m+4}
\]

\[
\mathbf d_P\in\mathbb R^{2m-2}
\]

---

#### 3.10 第一段选择矩阵赋值

```cpp
  Ct(0, 0) = 1;
  Ct(2, 1) = 1;
  Ct(4, 2) = 1; // stack the start point
  Ct(1, 3) = 1;
  Ct(3, 2 * seg_num + 4) = 1;
  Ct(5, 2 * seg_num + 5) = 1;
```

第一段导数向量为：

\[
\mathbf d_0=
\begin{bmatrix}
w_0(0)\\
w_0(T_0)\\
\dot w_0(0)\\
\dot w_0(T_0)\\
\ddot w_0(0)\\
\ddot w_0(T_0)
\end{bmatrix}
\]

对应关系为：

\[
w_0(0)\in\mathbf d_F
\]

\[
\dot w_0(0)\in\mathbf d_F
\]

\[
\ddot w_0(0)\in\mathbf d_F
\]

\[
w_0(T_0)\in\mathbf d_F
\]

\[
\dot w_0(T_0)\in\mathbf d_P
\]

\[
\ddot w_0(T_0)\in\mathbf d_P
\]

第一段终点速度和加速度是第一个中间点的自由导数。

---

#### 3.11 最后一段选择矩阵赋值

```cpp
  Ct(6 * (seg_num - 1) + 0, 2 * seg_num + 0) = 1;
  Ct(6 * (seg_num - 1) + 1, 2 * seg_num + 1) = 1; // Stack the end point
  Ct(6 * (seg_num - 1) + 2, 4 * seg_num + 0) = 1;
  Ct(6 * (seg_num - 1) + 3, 2 * seg_num + 2) = 1; // Stack the end point
  Ct(6 * (seg_num - 1) + 4, 4 * seg_num + 1) = 1;
  Ct(6 * (seg_num - 1) + 5, 2 * seg_num + 3) = 1; // Stack the end point
```

最后一段编号为：

\[
k=m-1
\]

最后一段导数向量为：

\[
\mathbf d_{m-1}=
\begin{bmatrix}
w_{m-1}(0)\\
w_{m-1}(T_{m-1})\\
\dot w_{m-1}(0)\\
\dot w_{m-1}(T_{m-1})\\
\ddot w_{m-1}(0)\\
\ddot w_{m-1}(T_{m-1})
\end{bmatrix}
\]

对应关系为：

\[
w_{m-1}(0)\in\mathbf d_F
\]

\[
w_{m-1}(T_{m-1})\in\mathbf d_F
\]

\[
\dot w_{m-1}(0)\in\mathbf d_P
\]

\[
\dot w_{m-1}(T_{m-1})\in\mathbf d_F
\]

\[
\ddot w_{m-1}(0)\in\mathbf d_P
\]

\[
\ddot w_{m-1}(T_{m-1})\in\mathbf d_F
\]

最后一段起点速度和加速度是最后一个中间点的自由导数。

---

#### 3.12 中间段选择矩阵赋值

```cpp
  for (int j = 2; j < seg_num; j++)
  {
    Ct(6 * (j - 1) + 0, 2 + 2 * (j - 1) + 0) = 1;
    Ct(6 * (j - 1) + 1, 2 + 2 * (j - 1) + 1) = 1;
    Ct(6 * (j - 1) + 2, 2 * seg_num + 4 + 2 * (j - 2) + 0) = 1;
    Ct(6 * (j - 1) + 3, 2 * seg_num + 4 + 2 * (j - 1) + 0) = 1;
    Ct(6 * (j - 1) + 4, 2 * seg_num + 4 + 2 * (j - 2) + 1) = 1;
    Ct(6 * (j - 1) + 5, 2 * seg_num + 4 + 2 * (j - 1) + 1) = 1;
  }
```

令：

\[
k=j-1
\]

该循环处理第 \(1\) 段到第 \(m-2\) 段。

位置约束为：

\[
w_k(0)=w_k
\]

\[
w_k(T_k)=w_{k+1}
\]

中间速度共享同一自由变量：

\[
\dot w_{k-1}(T_{k-1})=\dot w_k(0)
\]

中间加速度共享同一自由变量：

\[
\ddot w_{k-1}(T_{k-1})=\ddot w_k(0)
\]

因此轨迹满足：

\[
C^2
\]

连续。

---

#### 3.13 重排端点导数

```cpp
  C = Ct.transpose();

  Eigen::VectorXd Dx1 = C * Dx;
  Eigen::VectorXd Dy1 = C * Dy;
  Eigen::VectorXd Dz1 = C * Dz;
```

因为：

\[
\mathbf d=C_t
\begin{bmatrix}
\mathbf d_F\\
\mathbf d_P
\end{bmatrix}
\]

所以：

\[
C=C_t^T
\]

得到：

\[
D_{x,1}=C_t^TD_x
\]

\[
D_{y,1}=C_t^TD_y
\]

\[
D_{z,1}=C_t^TD_z
\]

此时：

\[
D_{x,1}=
\begin{bmatrix}
D_{x,F}\\
D_{x,P}
\end{bmatrix}
\]

后续会重新计算自由部分。

---

#### 3.14 构造代价矩阵 \(Q\)

```cpp
  /* ---------- minimum snap matrix ---------- */
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(seg_num * 6, seg_num * 6);

  for (int k = 0; k < seg_num; k++)
  {
    for (int i = 3; i < 6; i++)
    {
      for (int j = 3; j < 6; j++)
      {
        Q(k * 6 + i, k * 6 + j) =
            i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * pow(Time(k), (i + j - 5));
      }
    }
  }
```

对五次多项式：

\[
w(t)=\sum_{r=0}^{5}p_rt^r
\]

三阶导数为：

\[
w^{(3)}(t)=\sum_{r=3}^{5}r(r-1)(r-2)p_rt^{r-3}
\]

单段代价为：

\[
J_k=\int_0^{T_k}\left[w_k^{(3)}(t)\right]^2dt
\]

展开：

\[
J_k=\sum_{i=3}^{5}\sum_{j=3}^{5}p_ip_j\,i(i-1)(i-2)\,j(j-1)(j-2)\int_0^{T_k}t^{i+j-6}dt
\]

积分项为：

\[
\int_0^{T_k}t^{i+j-6}dt=\frac{T_k^{i+j-5}}{i+j-5}
\]

因此：

\[
(Q_k)_{i,j}=i(i-1)(i-2)\,j(j-1)(j-2)\,\frac{T_k^{i+j-5}}{i+j-5}
\]

这正对应代码中的：

\[
i(i-1)(i-2)j(j-1)(j-2)\frac{T_k^{i+j-5}}{i+j-5}
\]

整体矩阵为：

\[
Q=\operatorname{blkdiag}(Q_0,Q_1,\dots,Q_{m-1})
\]

---

#### 3.15 构造 \(R\) 矩阵

```cpp
  /* ---------- R matrix ---------- */
  Eigen::MatrixXd R = C * A.transpose().inverse() * Q * A.inverse() * Ct;
```

由：

\[
A\mathbf p=\mathbf d
\]

得：

\[
\mathbf p=A^{-1}\mathbf d
\]

由：

\[
\mathbf d=C_t\tilde{\mathbf d}
\]

得：

\[
\mathbf p=A^{-1}C_t\tilde{\mathbf d}
\]

代入：

\[
J=\mathbf p^TQ\mathbf p
\]

得到：

\[
J=\tilde{\mathbf d}^TC_t^TA^{-T}QA^{-1}C_t\tilde{\mathbf d}
\]

因此：

\[
R=C_t^TA^{-T}QA^{-1}C_t
\]

代码中：

\[
C=C_t^T
\]

所以：

\[
R=CA^{-T}QA^{-1}C_t
\]

---

#### 3.16 提取固定导数

```cpp
  Eigen::VectorXd Dxf(2 * seg_num + 4), Dyf(2 * seg_num + 4), Dzf(2 * seg_num + 4);

  Dxf = Dx1.segment(0, 2 * seg_num + 4);
  Dyf = Dy1.segment(0, 2 * seg_num + 4);
  Dzf = Dz1.segment(0, 2 * seg_num + 4);
```

固定导数数量为：

\[
n_f=2m+4
\]

因此：

\[
D_{x,F}=D_{x,1}[0:n_f]
\]

\[
D_{y,F}=D_{y,1}[0:n_f]
\]

\[
D_{z,F}=D_{z,1}[0:n_f]
\]

---

#### 3.17 分块 \(R\) 矩阵

```cpp
  Eigen::MatrixXd Rff(2 * seg_num + 4, 2 * seg_num + 4);
  Eigen::MatrixXd Rfp(2 * seg_num + 4, 2 * seg_num - 2);
  Eigen::MatrixXd Rpf(2 * seg_num - 2, 2 * seg_num + 4);
  Eigen::MatrixXd Rpp(2 * seg_num - 2, 2 * seg_num - 2);

  Rff = R.block(0, 0, 2 * seg_num + 4, 2 * seg_num + 4);
  Rfp = R.block(0, 2 * seg_num + 4, 2 * seg_num + 4, 2 * seg_num - 2);
  Rpf = R.block(2 * seg_num + 4, 0, 2 * seg_num - 2, 2 * seg_num + 4);
  Rpp = R.block(2 * seg_num + 4, 2 * seg_num + 4, 2 * seg_num - 2, 2 * seg_num - 2);
```

按照：

\[
\tilde{\mathbf d}=
\begin{bmatrix}
\mathbf d_F\\
\mathbf d_P
\end{bmatrix}
\]

将 \(R\) 分块：

\[
R=
\begin{bmatrix}
R_{FF}&R_{FP}\\
R_{PF}&R_{PP}
\end{bmatrix}
\]

其中：

\[
R_{FF}\in\mathbb R^{n_f\times n_f}
\]

\[
R_{FP}\in\mathbb R^{n_f\times n_p}
\]

\[
R_{PF}\in\mathbb R^{n_p\times n_f}
\]

\[
R_{PP}\in\mathbb R^{n_p\times n_p}
\]

---

#### 3.18 闭式求解自由导数

```cpp
  /* ---------- close form solution ---------- */

  Eigen::VectorXd Dxp(2 * seg_num - 2), Dyp(2 * seg_num - 2), Dzp(2 * seg_num - 2);
  Dxp = -(Rpp.inverse() * Rfp.transpose()) * Dxf;
  Dyp = -(Rpp.inverse() * Rfp.transpose()) * Dyf;
  Dzp = -(Rpp.inverse() * Rfp.transpose()) * Dzf;
```

目标函数为：

\[
J=
\begin{bmatrix}
D_F\\
D_P
\end{bmatrix}^T
\begin{bmatrix}
R_{FF}&R_{FP}\\
R_{PF}&R_{PP}
\end{bmatrix}
\begin{bmatrix}
D_F\\
D_P
\end{bmatrix}
\]

展开：

\[
J=D_F^TR_{FF}D_F+D_F^TR_{FP}D_P+D_P^TR_{PF}D_F+D_P^TR_{PP}D_P
\]

对 \(D_P\) 求导：

\[
\frac{\partial J}{\partial D_P}=2R_{PF}D_F+2R_{PP}D_P
\]

令其为零：

\[
R_{PP}D_P=-R_{PF}D_F
\]

得到：

\[
D_P^*=-R_{PP}^{-1}R_{PF}D_F
\]

由于：

\[
R_{PF}=R_{FP}^T
\]

所以：

\[
D_P^*=-R_{PP}^{-1}R_{FP}^TD_F
\]

代码分别计算：

\[
D_{x,P}^*=-R_{PP}^{-1}R_{FP}^TD_{x,F}
\]

\[
D_{y,P}^*=-R_{PP}^{-1}R_{FP}^TD_{y,F}
\]

\[
D_{z,P}^*=-R_{PP}^{-1}R_{FP}^TD_{z,F}
\]

---

#### 3.19 写回自由导数

```cpp
  Dx1.segment(2 * seg_num + 4, 2 * seg_num - 2) = Dxp;
  Dy1.segment(2 * seg_num + 4, 2 * seg_num - 2) = Dyp;
  Dz1.segment(2 * seg_num + 4, 2 * seg_num - 2) = Dzp;
```

重排后的导数向量为：

\[
D_{x,1}=
\begin{bmatrix}
D_{x,F}\\
D_{x,P}
\end{bmatrix}
\]

将最优自由导数写回：

\[
D_{x,1}=
\begin{bmatrix}
D_{x,F}\\
D_{x,P}^*
\end{bmatrix}
\]

同理：

\[
D_{y,1}=
\begin{bmatrix}
D_{y,F}\\
D_{y,P}^*
\end{bmatrix}
\]

\[
D_{z,1}=
\begin{bmatrix}
D_{z,F}\\
D_{z,P}^*
\end{bmatrix}
\]

---

#### 3.20 恢复多项式系数

```cpp
  Px = (A.inverse() * Ct) * Dx1;
  Py = (A.inverse() * Ct) * Dy1;
  Pz = (A.inverse() * Ct) * Dz1;
```

由：

\[
\mathbf d=C_t\tilde{\mathbf d}
\]

\[
\mathbf p=A^{-1}\mathbf d
\]

得到：

\[
\mathbf p=A^{-1}C_t\tilde{\mathbf d}
\]

因此：

\[
P_x=A^{-1}C_t
\begin{bmatrix}
D_{x,F}\\
D_{x,P}^*
\end{bmatrix}
\]

\[
P_y=A^{-1}C_t
\begin{bmatrix}
D_{y,F}\\
D_{y,P}^*
\end{bmatrix}
\]

\[
P_z=A^{-1}C_t
\begin{bmatrix}
D_{z,F}\\
D_{z,P}^*
\end{bmatrix}
\]

---

#### 3.21 整理系数矩阵

```cpp
  for (int i = 0; i < seg_num; i++)
  {
    poly_coeff.block(i, 0, 1, 6) = Px.segment(i * 6, 6).transpose();
    poly_coeff.block(i, 6, 1, 6) = Py.segment(i * 6, 6).transpose();
    poly_coeff.block(i, 12, 1, 6) = Pz.segment(i * 6, 6).transpose();
  }
```

对第 \(i\) 段：

\[
\mathbf p_{x,i}=P_x[6i:6i+5]
\]

\[
\mathbf p_{y,i}=P_y[6i:6i+5]
\]

\[
\mathbf p_{z,i}=P_z[6i:6i+5]
\]

写入：

\[
\text{poly\_coeff}(i,0:5)=\mathbf p_{x,i}^T
\]

\[
\text{poly\_coeff}(i,6:11)=\mathbf p_{y,i}^T
\]

\[
\text{poly\_coeff}(i,12:17)=\mathbf p_{z,i}^T
\]

---

#### 3.22 构造轨迹对象

```cpp
  /* ---------- use polynomials ---------- */
  PolynomialTraj poly_traj;
  for (int i = 0; i < poly_coeff.rows(); ++i)
  {
    vector<double> cx(6), cy(6), cz(6);
    for (int j = 0; j < 6; ++j)
    {
      cx[j] = poly_coeff(i, j), cy[j] = poly_coeff(i, j + 6), cz[j] = poly_coeff(i, j + 12);
    }
    reverse(cx.begin(), cx.end());
    reverse(cy.begin(), cy.end());
    reverse(cz.begin(), cz.end());
    double ts = Time(i);
    poly_traj.addSegment(cx, cy, cz, ts);
  }
```

`poly_coeff` 中的系数顺序为：

\[
[p_0,p_1,p_2,p_3,p_4,p_5]
\]

对应：

\[
w(t)=p_0+p_1t+p_2t^2+p_3t^3+p_4t^4+p_5t^5
\]

反转后变为：

\[
[p_5,p_4,p_3,p_2,p_1,p_0]
\]

对应高次到低次排列：

\[
w(t)=p_5t^5+p_4t^4+p_3t^3+p_2t^2+p_1t+p_0
\]

每段时间为：

\[
t_s=T_i
\]

`addSegment` 将第 \(i\) 段加入轨迹：

\[
\mathbf r_i(t)=
\begin{bmatrix}
x_i(t)\\
y_i(t)\\
z_i(t)
\end{bmatrix},\quad t\in[0,T_i]
\]

---

#### 3.23 返回轨迹

```cpp
  return poly_traj;
}
```

最终返回：

\[
\mathcal T=
\left\{
\mathbf r_0(t),
\mathbf r_1(t),
\dots,
\mathbf r_{m-1}(t)
\right\}
\]

其中：

\[
x_i(t)=\sum_{r=0}^{5}p_{x,i,r}t^r
\]

\[
y_i(t)=\sum_{r=0}^{5}p_{y,i,r}t^r
\]

\[
z_i(t)=\sum_{r=0}^{5}p_{z,i,r}t^r
\]

---

## 5. 分析与总结（Analysis & Summary）

### 4. 当前代码与两篇论文的关系

#### 4.1 与原始 Minimum Snap 方法的关系

原始方法为：

\[
\min_{\mathbf p}\quad\mathbf p^TQ\mathbf p
\]

\[
\text{s.t.}\quad A_{eq}\mathbf p=\mathbf b_{eq}
\]

其特点是：

```text
1: 优化变量是多项式系数 p
2: 显式构造等式约束
3: 通过 constrained QP 求解
4: 标准目标是最小化 snap
```

当前代码没有显式构造：

\[
A_{eq}\mathbf p=\mathbf b_{eq}
\]

也没有求解 KKT 系统，因此不是原始方法的直接实现。

---

#### 4.2 与 Polynomial 方法的关系

Polynomial 方法为：

\[
\mathbf p=A^{-1}C_t
\begin{bmatrix}
\mathbf d_F\\
\mathbf d_P
\end{bmatrix}
\]

\[
R=C_t^TA^{-T}QA^{-1}C_t
\]

\[
\mathbf d_P^*=-R_{PP}^{-1}R_{FP}^T\mathbf d_F
\]

\[
\mathbf p^*=A^{-1}C_t
\begin{bmatrix}
\mathbf d_F\\
\mathbf d_P^*
\end{bmatrix}
\]

当前代码正是该结构。

对应关系如下：

| 数学符号 | 代码变量 |
| --- | --- |
| \(m\) | `seg_num` |
| \(T_i\) | `Time(i)` |
| \(D_x,D_y,D_z\) | `Dx`, `Dy`, `Dz` |
| \(A\) | `A` |
| \(A_i\) | `Ab` |
| \(C_t\) | `Ct` |
| \(C_t^T\) | `C` |
| \(Q\) | `Q` |
| \(R\) | `R` |
| \(R_{FF}\) | `Rff` |
| \(R_{FP}\) | `Rfp` |
| \(R_{PF}\) | `Rpf` |
| \(R_{PP}\) | `Rpp` |
| \(D_F\) | `Dxf`, `Dyf`, `Dzf` |
| \(D_P^*\) | `Dxp`, `Dyp`, `Dzp` |
| \(P_x,P_y,P_z\) | `Px`, `Py`, `Pz` |

---

### 6. 总结

原始 minimum snap 方法可以概括为：

\[
\boxed{
\min_{\mathbf p}\mathbf p^TQ\mathbf p\quad\text{s.t.}\quad A_{eq}\mathbf p=\mathbf b_{eq}
}
\]

它直接优化多项式系数，并通过约束 QP 保证路径点、边界导数和连续性。

Polynomial 方法可以概括为：

\[
\boxed{
\mathbf d_P^*=-R_{PP}^{-1}R_{FP}^T\mathbf d_F
}
\]

\[
\boxed{
\mathbf p^*=A^{-1}C_t
\begin{bmatrix}
\mathbf d_F\\
\mathbf d_P^*
\end{bmatrix}
}
\]

它将优化变量从多项式系数转为端点导数，并通过闭式解得到中间自由导数。

当前 C++ 代码实现了 Polynomial 方法的核心结构：

```text
1: 构造端点导数 d
2: 构造 A，使 A p = d
3: 构造选择矩阵 C_t
4: 构造 Q
5: 构造 R = C_t^T A^{-T} Q A^{-1} C_t
6: 分块 R
7: 闭式求解自由导数 d_P
8: 恢复多项式系数 p
9: 生成 PolynomialTraj
```

但当前代码实际最小化的是 jerk：

\[
\int\left[w^{(3)}(t)\right]^2dt
\]

而不是标准 minimum snap：

\[
\int\left[w^{(4)}(t)\right]^2dt
\]

因此，当前代码更准确地说是：

```text
基于 Polynomial 消元思想的五次多项式 minimum jerk 轨迹生成器。
```

## 6. 关联与参考（Related Topics & References）

> 待补充。
