# 1. 循环神经网络

## 1.1 数学定义构造

给定时间步 $t$ 的输入向量 $x_t \in \mathbb{R}^{d_{in}}$ 以及上一时刻的隐藏状态 $h_{t-1} \in \mathbb{R}^{d_{h}}$，标准循环神经网络（RNN）通过内部的权重矩阵对序列数据进行单步状态更新，其数学解析式为：

$$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$

其中：

* $W_{ih} \in \mathbb{R}^{d_h \times d_{in}}$ 与 $W_{hh} \in \mathbb{R}^{d_h \times d_h}$ 为输入与隐状态的权重矩阵。
* $b_{ih}, b_{hh} \in \mathbb{R}^{d_h}$ 为对应的偏置向量。
* $\tanh$ 为双曲正切激活函数，将输出状态非线性映射至 $[-1, 1]$ 区间。

序列的最终输出可通过额外的全连接层由 $h_t$ 映射得到。

## 1.2 层级架构性质

1. **时序记忆性**
   当前时刻的隐状态 $h_t$ 包含了历史所有输入 $x_1, \dots, x_t$ 的累积信息，使网络具备处理不定长上下文序列的能力。

2. **参数共享**
   在整个序列的时间步迭代中，所有时刻 $t$ 共享同一套权重矩阵 $W_{ih}$ 与 $W_{hh}$，这极大地控制了模型规模并使得网络对输入序列的长度免疫。

3. **梯度消失与爆炸**
   基于时间的反向传播在展开计算连续连乘的雅可比矩阵时，若权重矩阵的最大奇异值偏离 1，随着序列长度增加，梯度会呈指数级衰减或增长，导致长距离依赖难以学习。

## 1.3 代码实现

以下实现接收整个时序序列，内部完成时间步循环，并输出最后一个时间步的隐藏状态。

### 1.3.1 PyTorch实现

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

### 1.3.2 C++ 标准库实现

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

### 1.3.3 C++ Eigen库实现

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

# 2. 长短期记忆网络

## 2.1 数学定义构造

为解决标准 RNN 的长程梯度衰减，长短期记忆网络（LSTM）引入了细胞状态 $c_t$ 和精密的门控结构。在时间步 $t$，根据输入 $x_t$ 与上一隐状态 $h_{t-1}$，计算过程分为以下阶段：

1. **输入门、遗忘门、输出门与候选细胞状态**
   $$i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})$$
   $$f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})$$
   $$o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})$$
   $$g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})$$

2. **状态更新**
   基于遗忘门舍弃旧信息，基于输入门添加新信息，完成细胞状态 $c_t$ 的线性更新，并通过输出门过滤得到当前隐状态 $h_t$：
   $$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$
   $$h_t = o_t \odot \tanh(c_t)$$

其中 $\sigma$ 为 Sigmoid 函数，$\odot$ 为矩阵的逐元素乘法（Hadamard积）。

## 2.2 层级架构性质

1. **门控机制**
   门控数值被 Sigmoid 限制在 $[0,1]$。遗忘门控制历史记忆的留存比例，输入门控制当前观测的融合权重。该机制使得网络能自适应地选择性遗忘无效背景并记住关键特征。

2. **双状态传递**
   模型将记忆拆分为慢变的全局细胞状态 $c_t$ 和快变的局部隐藏状态 $h_t$。细胞状态类似传送带，贯穿整个序列；隐藏状态则用于对外输出当前步的综合响应。

3. **长距离依赖**
   细胞状态的更新 $c_t = f_t \odot c_{t-1} + \dots$ 是一种纯加法操作。在反向传播时，即使序列极长，梯度也能通过加法分支无损流淌，从根源上缓解了梯度消失问题。

## 2.3 代码实现

### 2.3.1 PyTorch实现

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

### 2.3.2 C++ 标准库实现

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

### 2.3.3 C++ Eigen库实现

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
