# 坐标变换与旋转矩阵

## 1. 概述

坐标变换描述同一个几何对象在不同坐标系中的表示。统一变换方向后，可以连接传感器观测、机器人状态与世界坐标中的规划结果。

## 2. 基础与定义

### 2.1 坐标约定

$\,{}^Ap$ 表示点在坐标系 $A$ 中的坐标。$\,{}^AR_B$ 将 $B$ 系向量转换到 $A$ 系，$\,{}^At_B$ 是 $B$ 原点在 $A$ 系的位置。采用列向量与右手坐标系。

### 2.2 点与自由向量

点需要旋转和平移，自由向量只需旋转：

$$
{}^Ap={}^AR_B\,{}^Bp+{}^At_B,\qquad {}^Av={}^AR_B\,{}^Bv.
$$

## 3. 核心原理

### 3.1 旋转与刚体变换

旋转矩阵满足 $R^\top R=I$、$\det R=1$。齐次表示为

$$
T_{AB}=\begin{bmatrix}R_{AB}&t_{AB}\\0&1\end{bmatrix},\qquad
\begin{bmatrix}{}^Ap\\1\end{bmatrix}=T_{AB}\begin{bmatrix}{}^Bp\\1\end{bmatrix}.
$$

### 3.2 组合与逆变换

沿坐标链相乘：$T_{AC}=T_{AB}T_{BC}$。逆变换为

$$
T_{BA}=\begin{bmatrix}R_{AB}^\top&-R_{AB}^\top t_{AB}\\0&1\end{bmatrix}.
$$

<details class="kt-derivation" id="frames-composition">
<summary>坐标链推导</summary>

将 ${}^Bp=R_{BC}{}^Cp+t_{BC}$ 代入 ${}^Ap=R_{AB}{}^Bp+t_{AB}$，得到 $R_{AC}=R_{AB}R_{BC}$、$t_{AC}=R_{AB}t_{BC}+t_{AB}$。平移需先旋转到同一坐标系后相加。

</details>

### 3.3 欧拉角与坐标约定

采用 $R=R_z(\psi)R_y(\vartheta)R_x(\varphi)$ 时，三个角分别描述指定次序的旋转。该参数化在 $\vartheta=\pm\pi/2$ 处奇异。

ENU 与 NED 分别采用东—北—上和北—东—下轴序，向量转换为

$$
R_{\rm NED\leftarrow ENU}=\begin{bmatrix}0&1&0\\1&0&0\\0&0&-1\end{bmatrix}.
$$

机体系轴序需单独约定；传感器外参包含安装旋转与平移。

## 4. 实现与应用

### 4.1 变换验证

用三个单位轴检查旋转方向，用原点检查平移，用 $T^{-1}Tp=p$ 检查往返一致性。传感器融合还需将变换与观测的时间戳对齐。

## 5. 分析与总结

变换下标表达输入与输出坐标系，乘法次序表达坐标链。旋转的参数化与几何变换本身可分别处理。

## 6. 关联与参考

[Lynch 与 Park：Modern Robotics](https://hades.mech.northwestern.edu/index.php/Modern_Robotics)，第 3 章。

关联：[SO(3)、SE(3) 与四元数](./03_so3-se3-quaternion.md)。
