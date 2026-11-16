# 多层感知机

## 1. 概述

多层感知机通过仿射变换与非线性激活的复合学习向量之间的映射。网络输出由任务损失监督，损失梯度逐层回传，使隐藏表示和输出映射共同适应目标。

## 2. 基础与定义

### 2.1 网络结构

输入 $h^{(0)}=x$，第 $l$ 层为

$$
z^{(l)}=W^{(l)}h^{(l-1)}+b^{(l)},\qquad
h^{(l)}=\sigma_l(z^{(l)}).
$$

$W^{(l)}\in\mathbb R^{d_l\times d_{l-1}}$，$b^{(l)}\in\mathbb R^{d_l}$。输出层形式由任务决定。

### 2.2 学习目标

回归可取 $\ell=\tfrac12\|\hat y-y\|^2$；分类以输出 logits $z$ 构造交叉熵 $\ell=-\log\operatorname{softmax}(z)_y$。训练目标为样本损失的平均值与可选正则项。

## 3. 核心原理

### 3.1 非线性表示

隐藏层激活使网络能够表示非线性关系。ReLU 为 $\max(0,z)$，其梯度在正半轴为 1、负半轴为 0；Sigmoid 常用于二分类概率输出。层宽、深度和激活共同决定表示能力。

### 3.2 反向传播

令 $\delta^{(l)}=\partial\ell/\partial z^{(l)}$，则

$$
\nabla_{W^{(l)}}\ell=\delta^{(l)}(h^{(l-1)})^\top,
\quad \nabla_{b^{(l)}}\ell=\delta^{(l)},
$$

$$
\delta^{(l-1)}=((W^{(l)})^\top\delta^{(l)})\odot\sigma'_{l-1}(z^{(l-1)}).
$$

梯度由后续损失、层权重与激活导数共同决定。
<details class="kt-derivation" id="mlp-chain-rule">
<summary>仿射层梯度推导</summary>

对 $z=Wh+b$ 微分，$dz=(dW)h+Wdh+db$。由 $d\ell=\delta^\top dz$，比较各项得到 $\nabla_W\ell=\delta h^\top$、$\nabla_h\ell=W^\top\delta$ 和 $\nabla_b\ell=\delta$。批量损失对各样本的这些贡献求平均。

</details>

### 3.3 初始化与尺度

输入归一化和与层宽匹配的权重初始化有助于控制前向激活与反向梯度尺度。适当激活与足够宽度下，MLP 可逼近紧集上的连续函数；训练能否得到有效近似还取决于数据、优化和模型规模。

## 4. 实现与应用

训练依次执行批量输入、前向预测、损失计算、反向传播和参数更新。检查输入输出维数、损失下降与梯度有限性。

以下示例均为一个隐藏层。PyTorch 模块支持自动求导；C++ 示例展示初始化与前向计算，使用正维数和匹配长度的输入。

<details class="kt-derivation" id="implementation-0">
<summary>PyTorch 前向网络</summary>

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

</details>

<details class="kt-derivation" id="implementation-1">
<summary>C++ 前向计算</summary>

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

</details>

<details class="kt-derivation" id="implementation-2">
<summary>Eigen 前向计算</summary>

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

</details>

## 5. 分析与总结

MLP 适合结构化向量、策略头和价值头。空间与时间结构可分别由卷积、循环网络或注意力表达，再由 MLP 完成任务映射。

## 6. 关联与参考

- [Goodfellow 等：Deep Learning](https://www.deeplearningbook.org/)
- [梯度、雅可比与 Hessian](../../01_math-physic-base/02_calculus-optimization/01_gradient-jacobian-hessian.md)
- [编码器与解码器](04_encoder-decoder.md)
