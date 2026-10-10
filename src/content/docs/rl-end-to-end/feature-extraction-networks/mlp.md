---
title: "多层感知机"
sidebar:
  order: 1
---

## 1.1 数学定义构造

多层感知机（MLP）是最基础的前馈人工神经网络。给定输入特征向量 $x \in \mathbb{R}^{d_{in}}$，MLP 通过多个全连接层与非线性激活函数的交替组合，构造出从输入空间到输出空间 $\mathbb{R}^{d_{out}}$ 的复杂非线性映射 $F(x)$。

对于一个包含 $L$ 层的 MLP，其数学定义由层级递推公式构造：

$$h^{(0)} = x$$
$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}, \quad l \in [1, L]$$
$$h^{(l)} = \sigma^{(l)}(z^{(l)})$$

其中：

* $h^{(l-1)} \in \mathbb{R}^{d_{l-1}}$ 为第 $l-1$ 层的输出（即第 $l$ 层的输入）。
* $W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$ 为第 $l$ 层的权重矩阵。
* $b^{(l)} \in \mathbb{R}^{d_l}$ 为第 $l$ 层的偏置向量。
* $\sigma^{(l)}(\cdot)$ 为逐元素作用的非线性激活函数。
* $h^{(L)}$ 即为网络最终的输出预测值 $y$。

## 1.2 层级架构性质

1. **全连接特性**
   第 $l$ 层的任意神经元均与第 $l-1$ 层的全体神经元相连。矩阵乘法 $W^{(l)} h^{(l-1)}$ 在几何上等价于对特征空间进行仿射变换。

2. **非线性激活**
   若无激活函数 $\sigma$，多层矩阵乘法 $W^{(L)} \dots W^{(1)} x$ 可坍缩为单层线性变换 $W_{eq} x$，网络将丧失非线性表达能力。常用的激活函数包括：

   * **ReLU**：$\sigma(z) = \max(0, z)$，计算高效，缓解梯度消失。
   * **Sigmoid**：$\sigma(z) = \frac{1}{1 + e^{-z}}$，常用于输出层二分类概率映射。

3. **万能逼近定理**
   只要隐藏层神经元数量足够多，包含至少一个带有非线性激活函数隐藏层的 MLP，能够以任意精度逼近闭区间上的任意连续函数。

## 1.3 代码实现

以下以标准三层架构（1个输入层、1个隐藏层、1个输出层）为例，实现完整的前向传播与参数初始化。隐藏层使用 ReLU 激活函数，输出层无激活函数。

### 1.3.1 PyTorch实现

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.relu(h)
        y = self.fc2(h)
        return y

# 使用示例
# model = MLP(input_dim=10, hidden_dim=64, output_dim=2)
# out = model(torch.randn(32, 10)) # 32为Batch Size
```

### 1.3.2 C++ 标准库实现

```cpp
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>

class MLP {
private:
    std::vector<std::vector<double>> W1, W2;
    std::vector<double> b1, b2;

    std::vector<double> linear(const std::vector<double>& x, const std::vector<std::vector<double>>& W, const std::vector<double>& b) const {
        std::vector<double> out(W.size(), 0.0);
        for (size_t i = 0; i < W.size(); ++i) {
            out[i] = b[i];
            for (size_t j = 0; j < x.size(); ++j) {
                out[i] += W[i][j] * x[j];
            }
        }
        return out;
    }

    std::vector<double> relu(std::vector<double> x) const {
        for (double& val : x) {
            val = std::max(0.0, val);
        }
        return x;
    }

public:
    MLP(int in_dim, int hidden_dim, int out_dim) {
        W1.assign(hidden_dim, std::vector<double>(in_dim));
        b1.assign(hidden_dim, 0.0);
        W2.assign(out_dim, std::vector<double>(hidden_dim));
        b2.assign(out_dim, 0.0);

        // 基础随机初始化权重 (-1.0 到 1.0)
        for (auto& row : W1) {
            for (auto& w : row) w = (std::rand() % 2000 - 1000) / 1000.0;
        }
        for (auto& row : W2) {
            for (auto& w : row) w = (std::rand() % 2000 - 1000) / 1000.0;
        }
    }

    std::vector<double> forward(const std::vector<double>& x) const {
        std::vector<double> h = linear(x, W1, b1);
        h = relu(h);
        return linear(h, W2, b2);
    }
};
```

### 1.3.3 C++ Eigen库实现

利用 Eigen 矩阵运算库，可大幅优化线性代数计算效率并精简代码。

```cpp
#include <Eigen/Dense>
#include <algorithm>

class MLPEigen {
private:
    Eigen::MatrixXf W1, W2;
    Eigen::VectorXf b1, b2;

public:
    MLPEigen(int in_dim, int hidden_dim, int out_dim) {
        // 使用均匀分布随机初始化矩阵，并置零偏置
        W1 = Eigen::MatrixXf::Random(hidden_dim, in_dim);
        b1 = Eigen::VectorXf::Zero(hidden_dim);
        
        W2 = Eigen::MatrixXf::Random(out_dim, hidden_dim);
        b2 = Eigen::VectorXf::Zero(out_dim);
    }

    Eigen::VectorXf forward(const Eigen::VectorXf& x) const {
        // 第一层：线性变换 + 逐元素 ReLU 激活
        Eigen::VectorXf h = W1 * x + b1;
        h = h.cwiseMax(0.0f); 
        
        // 第二层：线性变换输出
        return W2 * h + b2;
    }
};
```
