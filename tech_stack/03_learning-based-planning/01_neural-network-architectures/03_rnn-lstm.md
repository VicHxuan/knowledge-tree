# 循环神经网络与 LSTM

## 1. 概述

循环网络将当前输入与历史隐状态共同映射到下一状态，从序列中学习有限维记忆。时序损失沿展开的计算图回传；LSTM 通过门控的细胞状态提供更可控的跨时刻信息路径。

## 2. 基础与定义

### 2.1 循环状态

标准 RNN 的更新为

$$
h_t=\tanh(W_xx_t+W_hh_{t-1}+b),\qquad \hat y_t=g(h_t).
$$

$h_t$ 是历史输入的学习表示，其容量由状态维数与训练目标决定。

### 2.2 时序目标

逐步监督使用 $L=\sum_tm_t\ell(\hat y_t,y_t)$，其中 $m_t$ 标记有效时间步。序列分类可仅监督最后有效状态。

## 3. 核心原理

### 3.1 时间反向传播

隐状态的局部雅可比为

$$
J_t=\frac{\partial h_t}{\partial h_{t-1}}
=\operatorname{diag}(1-h_t^2)W_h.
$$

远期梯度经过 $J_tJ_{t-1}\cdots$ 的乘积传播，激活饱和与矩阵方向共同影响梯度衰减或增长。
<details class="kt-derivation" id="rnn-bptt">
<summary>时序梯度递推</summary>

令 $g_t=\partial L/\partial h_t$ 表示包含未来时刻影响的总梯度，$r_t$ 为当前输出损失对 $h_t$ 的局部梯度，则

$$
g_t=r_t+J_{t+1}^\top g_{t+1}.
$$

参数在各时间步共享，故参数梯度为所有展开节点贡献之和。截断反向传播在片段边界停止这一梯度递推，但可继续传递隐状态的数值。

</details>

### 3.2 LSTM 门控

用当前输入与上一步隐状态计算输入门 $i_t$、遗忘门 $f_t$、输出门 $o_t$ 与候选状态 $g_t$：

$$
\begin{bmatrix}i_t\\f_t\\g_t\\o_t\end{bmatrix}
=\begin{bmatrix}\sigma\\\sigma\\\tanh\\\sigma\end{bmatrix}
\left(W\begin{bmatrix}x_t\\h_{t-1}\end{bmatrix}+b\right).
$$

细胞状态与隐状态更新为

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad
h_t=o_t\odot\tanh(c_t).
$$

### 3.3 记忆梯度

固定当前门值时，细胞状态直接路径的导数为 $\operatorname{diag}(f_t)$。遗忘门接近 1 时有利于保持梯度；完整导数还包含门值通过 $h_{t-1}$ 的依赖。
<details class="kt-derivation" id="lstm-memory">
<summary>细胞状态直接路径</summary>

沿直接细胞通路展开，$c_t$ 中历史 $c_{t-k}$ 的系数为

$$
\prod_{j=t-k+1}^t f_j
$$

的逐元素乘积。因此加法结构提供独立的信息通路，而长期保持仍依赖门值。梯度裁剪控制爆炸，训练窗口长度影响可学习的时序依赖。

</details>

## 4. 实现与应用

训练按序列组织批次，保留有效时间步掩码；在线推理逐步传递状态，回合边界复位状态。填充序列应取最后有效状态或使用打包序列。

以下示例使用等长序列并返回末步隐状态；每次调用从零状态开始。流式部署需将初始与最终状态显式加入接口。C++ 示例展示前向计算。

<details class="kt-derivation" id="implementation-0">
<summary>PyTorch RNN</summary>

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, h_n = self.rnn(x)
        # 返回最后一个时间步的隐状态
        return h_n.squeeze(0)
```

</details>

<details class="kt-derivation" id="implementation-1">
<summary>C++ RNN</summary>

```cpp
#include <vector>
#include <cmath>
#include <cstdlib>

class RNN {
private:
    int hidden_dim;
    std::vector<std::vector<double>> W_ih, W_hh;
    std::vector<double> b_h;

    std::vector<double> linear_add(const std::vector<double>& x, const std::vector<std::vector<double>>& W_i,
                                   const std::vector<double>& h, const std::vector<std::vector<double>>& W_h,
                                   const std::vector<double>& b) const {
        std::vector<double> out(hidden_dim, 0.0);
        for (int i = 0; i < hidden_dim; ++i) {
            out[i] = b[i];
            for (size_t j = 0; j < x.size(); ++j) out[i] += W_i[i][j] * x[j];
            for (size_t j = 0; j < h.size(); ++j) out[i] += W_h[i][j] * h[j];
            out[i] = std::tanh(out[i]);
        }
        return out;
    }

public:
    RNN(int in_dim, int h_dim) : hidden_dim(h_dim) {
        W_ih.assign(h_dim, std::vector<double>(in_dim, 0.1));
        W_hh.assign(h_dim, std::vector<double>(h_dim, 0.1));
        b_h.assign(h_dim, 0.0);
    }

    // x shape: [seq_len][input_dim]
    std::vector<double> forward(const std::vector<std::vector<double>>& x) const {
        std::vector<double> h(hidden_dim, 0.0);
        for (const auto& x_t : x) {
            h = linear_add(x_t, W_ih, h, W_hh, b_h);
        }
        return h;
    }
};
```

</details>

<details class="kt-derivation" id="implementation-2">
<summary>Eigen RNN</summary>

```cpp
#include <vector>
#include <Eigen/Dense>

class RNNEigen {
private:
    int hidden_dim;
    Eigen::MatrixXf W_ih, W_hh;
    Eigen::VectorXf b_h;

public:
    RNNEigen(int in_dim, int h_dim) : hidden_dim(h_dim) {
        W_ih = Eigen::MatrixXf::Random(h_dim, in_dim);
        W_hh = Eigen::MatrixXf::Random(h_dim, h_dim);
        b_h = Eigen::VectorXf::Zero(h_dim);
    }

    Eigen::VectorXf forward(const std::vector<Eigen::VectorXf>& x) const {
        Eigen::VectorXf h = Eigen::VectorXf::Zero(hidden_dim);
        for (const auto& x_t : x) {
            h = (W_ih * x_t + W_hh * h + b_h).array().tanh().matrix();
        }
        return h;
    }
};
```

</details>

<details class="kt-derivation" id="implementation-3">
<summary>PyTorch LSTM</summary>

```python
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 返回输出特征序列与最终状态 (h_n, c_n)
        out, (h_n, c_n) = self.lstm(x)
        return h_n.squeeze(0)
```

</details>

<details class="kt-derivation" id="implementation-4">
<summary>C++ LSTM</summary>

```cpp
#include <vector>
#include <cmath>
#include <cstdlib>

class LSTM {
private:
    int hidden_dim;
    // 将四个门的权重合并，提升计算效率 (i, f, g, o)
    std::vector<std::vector<double>> W_ih, W_hh;
    std::vector<double> b_h;

    double sigmoid(double x) const { return 1.0 / (1.0 + std::exp(-x)); }

public:
    LSTM(int in_dim, int h_dim) : hidden_dim(h_dim) {
        W_ih.assign(4 * h_dim, std::vector<double>(in_dim, 0.1));
        W_hh.assign(4 * h_dim, std::vector<double>(h_dim, 0.1));
        b_h.assign(4 * h_dim, 0.0);
    }

    std::vector<double> forward(const std::vector<std::vector<double>>& x) const {
        std::vector<double> h(hidden_dim, 0.0);
        std::vector<double> c(hidden_dim, 0.0);

        for (const auto& x_t : x) {
            std::vector<double> gates(4 * hidden_dim, 0.0);
            for (int i = 0; i < 4 * hidden_dim; ++i) {
                gates[i] = b_h[i];
                for (size_t j = 0; j < x_t.size(); ++j) gates[i] += W_ih[i][j] * x_t[j];
                for (size_t j = 0; j < h.size(); ++j) gates[i] += W_hh[i][j] * h[j];
            }

            for (int j = 0; j < hidden_dim; ++j) {
                double i_t = sigmoid(gates[j]);
                double f_t = sigmoid(gates[hidden_dim + j]);
                double g_t = std::tanh(gates[2 * hidden_dim + j]);
                double o_t = sigmoid(gates[3 * hidden_dim + j]);

                c[j] = f_t * c[j] + i_t * g_t;
                h[j] = o_t * std::tanh(c[j]);
            }
        }
        return h;
    }
};
```

</details>

<details class="kt-derivation" id="implementation-5">
<summary>Eigen LSTM</summary>

```cpp
#include <vector>
#include <Eigen/Dense>

class LSTMEigen {
private:
    int hidden_dim;
    Eigen::MatrixXf W_ih, W_hh;
    Eigen::VectorXf b_h;

    inline Eigen::VectorXf sigmoid(const Eigen::VectorXf& x) const {
        return 1.0f / (1.0f + (-x.array()).exp());
    }

public:
    LSTMEigen(int in_dim, int h_dim) : hidden_dim(h_dim) {
        W_ih = Eigen::MatrixXf::Random(4 * h_dim, in_dim);
        W_hh = Eigen::MatrixXf::Random(4 * h_dim, h_dim);
        b_h = Eigen::VectorXf::Zero(4 * h_dim);
    }

    Eigen::VectorXf forward(const std::vector<Eigen::VectorXf>& x) const {
        Eigen::VectorXf h = Eigen::VectorXf::Zero(hidden_dim);
        Eigen::VectorXf c = Eigen::VectorXf::Zero(hidden_dim);

        for (const auto& x_t : x) {
            Eigen::VectorXf gates = W_ih * x_t + W_hh * h + b_h;

            Eigen::VectorXf i_t = sigmoid(gates.segment(0, hidden_dim));
            Eigen::VectorXf f_t = sigmoid(gates.segment(hidden_dim, hidden_dim));
            Eigen::VectorXf g_t = gates.segment(2 * hidden_dim, hidden_dim).array().tanh().matrix();
            Eigen::VectorXf o_t = sigmoid(gates.segment(3 * hidden_dim, hidden_dim));

            c = f_t.cwiseProduct(c) + i_t.cwiseProduct(g_t);
            h = o_t.cwiseProduct(c.array().tanh().matrix());
        }
        return h;
    }
};
```

</details>

## 5. 分析与总结

RNN 与 LSTM 用固定大小状态处理增长的历史，适合在线估计与部分可观测控制。状态复位、截断边界和训练部署一致性与网络结构同样影响效果。

## 6. 关联与参考

- [Hochreiter 与 Schmidhuber：Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [PyTorch：LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [时序与多模态融合](07_temporal-multimodal-fusion.md)
