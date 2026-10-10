---
title: "卷积神经网络"
sidebar:
  order: 2
---

## 1.1 数学定义构造

给定三维输入特征张量 $X \in \mathbb{R}^{H_{in} \times W_{in} \times C_{in}}$，卷积层通过滑动由 $C_{out}$ 个滤波器组成的卷积核组 $W \in \mathbb{R}^{K \times K \times C_{in} \times C_{out}}$，并加上偏置 $b \in \mathbb{R}^{C_{out}}$，生成输出特征张量 $Y \in \mathbb{R}^{H_{out} \times W_{out} \times C_{out}}$。

二维离散卷积的数学推导公式为：

$$Y_{i,j,c_{out}} = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} X_{i \cdot S + m, j \cdot S + n, c} \cdot W_{m, n, c, c_{out}} + b_{c_{out}}$$

其中 $S$ 为步长，特征图空间尺寸计算如下（无填充情况）：
$$H_{out} = \lfloor \frac{H_{in} - K}{S} \rfloor + 1, \quad W_{out} = \lfloor \frac{W_{in} - K}{S} \rfloor + 1$$

池化层通常紧随其后进行空间降采样。以最大池化为例，其操作定义为：
$$P_{i,j,c} = \max_{m,n} Y_{i \cdot S + m, j \cdot S + n, c}$$

## 1.2 层级架构性质

1. **局部感受野**
   图像空间存在强烈的局部相关性。卷积层节点仅与输入特征图的一个局部窗口相连，负责提取边缘、角点等底层视觉模式，降低了全连接带来的冗余参数。

2. **权值共享**
   同一个通道内的卷积核在整张特征图上滑动且参数固定不变。该性质使得模型参数量脱离了输入图像尺寸的限制，仅受卷积核大小和通道数决定。

3. **平移不变性**
   由于权值共享机制，图像中相同特征无论出现在何处，都会被相同的卷积核激活。结合池化层的降采样，使网络对目标的平移、微小缩放具有内在的鲁棒性。

4. **层次化表征**
   随着网络层级加深与感受野扩大，浅层提取的局部几何纹理会逐层组合，在高维通道映射为具有全局语义信息的抽象特征向量。

## 1.3 代码实现

为保持核心逻辑凝练与代码完整性，以下构建一个极简端到端架构：包含一层二维卷积、ReLU激活、全局平均池化以及一层线性分类器。

### 1.3.1 PyTorch实现

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int, kernel_size: int = 3):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

### 1.3.2 C++ 标准库实现

```cpp
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>

using Tensor3D = std::vector<std::vector<std::vector<double>>>;

class SimpleCNN {
private:
    int in_c, out_c, kernel_size, num_classes;
    // 卷积参数: [out_c][in_c][kernel_size][kernel_size]
    std::vector<std::vector<std::vector<std::vector<double>>>> conv_W;
    std::vector<double> conv_b;
    // 全连接参数: [num_classes][out_c]
    std::vector<std::vector<double>> fc_W;
    std::vector<double> fc_b;

    double random_init() { return (std::rand() % 2000 - 1000) / 1000.0; }

public:
    SimpleCNN(int ic, int oc, int ks, int nc) : in_c(ic), out_c(oc), kernel_size(ks), num_classes(nc) {
        conv_W.assign(out_c, std::vector<std::vector<std::vector<double>>>(in_c, 
               std::vector<std::vector<double>>(kernel_size, std::vector<double>(kernel_size))));
        conv_b.assign(out_c, 0.0);
        fc_W.assign(num_classes, std::vector<double>(out_c));
        fc_b.assign(num_classes, 0.0);

        for(auto& oc_w : conv_W) for(auto& ic_w : oc_w) for(auto& row : ic_w) for(auto& w : row) w = random_init();
        for(auto& row : fc_W) for(auto& w : row) w = random_init();
    }

    std::vector<double> forward(const Tensor3D& x) const {
        int h_in = x[0].size();
        int w_in = x[0][0].size();
        int h_out = h_in - kernel_size + 1;
        int w_out = w_in - kernel_size + 1;

        // 1. 卷积与ReLU
        Tensor3D conv_out(out_c, std::vector<std::vector<double>>(h_out, std::vector<double>(w_out, 0.0)));
        for (int oc = 0; oc < out_c; ++oc) {
            for (int i = 0; i < h_out; ++i) {
                for (int j = 0; j < w_out; ++j) {
                    double val = conv_b[oc];
                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int m = 0; m < kernel_size; ++m) {
                            for (int n = 0; n < kernel_size; ++n) {
                                val += x[ic][i + m][j + n] * conv_W[oc][ic][m][n];
                            }
                        }
                    }
                    conv_out[oc][i][j] = std::max(0.0, val);
                }
            }
        }

        // 2. 全局平均池化
        std::vector<double> gap_out(out_c, 0.0);
        int spatial_size = h_out * w_out;
        for (int oc = 0; oc < out_c; ++oc) {
            double sum = 0;
            for (int i = 0; i < h_out; ++i) {
                for (int j = 0; j < w_out; ++j) {
                    sum += conv_out[oc][i][j];
                }
            }
            gap_out[oc] = sum / spatial_size;
        }

        // 3. 全连接层
        std::vector<double> out(num_classes, 0.0);
        for (int nc = 0; nc < num_classes; ++nc) {
            out[nc] = fc_b[nc];
            for (int oc = 0; oc < out_c; ++oc) {
                out[nc] += gap_out[oc] * fc_W[nc][oc];
            }
        }
        return out;
    }
};
```

### 1.3.3 C++ Eigen库实现

使用 `std::vector<Eigen::MatrixXf>` 表达多通道张量，利用矩阵块提取计算卷积以精简循环。

```cpp
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

using Tensor3D_Eigen = std::vector<Eigen::MatrixXf>;

class SimpleCNNEigen {
private:
    int in_c, out_c, kernel_size, num_classes;
    // 卷积核以一维数组嵌套矩阵存放，方便与特征图做矩阵点乘
    std::vector<std::vector<Eigen::MatrixXf>> conv_W; 
    Eigen::VectorXf conv_b;
    Eigen::MatrixXf fc_W;
    Eigen::VectorXf fc_b;

public:
    SimpleCNNEigen(int ic, int oc, int ks, int nc) : in_c(ic), out_c(oc), kernel_size(ks), num_classes(nc) {
        conv_W.assign(out_c, std::vector<Eigen::MatrixXf>(in_c, Eigen::MatrixXf::Random(ks, ks)));
        conv_b = Eigen::VectorXf::Zero(out_c);
        fc_W = Eigen::MatrixXf::Random(nc, oc);
        fc_b = Eigen::VectorXf::Zero(nc);
    }

    Eigen::VectorXf forward(const Tensor3D_Eigen& x) const {
        int h_in = x[0].rows();
        int w_in = x[0].cols();
        int h_out = h_in - kernel_size + 1;
        int w_out = w_in - kernel_size + 1;

        // 1. 卷积与ReLU
        Tensor3D_Eigen conv_out(out_c, Eigen::MatrixXf::Zero(h_out, w_out));
        for (int oc = 0; oc < out_c; ++oc) {
            conv_out[oc].array() += conv_b(oc);
            for (int ic = 0; ic < in_c; ++ic) {
                for (int i = 0; i < h_out; ++i) {
                    for (int j = 0; j < w_out; ++j) {
                        // 提取局部窗口并通过点乘求和实现卷积步
                        conv_out[oc](i, j) += (x[ic].block(i, j, kernel_size, kernel_size).cwiseProduct(conv_W[oc][ic])).sum();
                    }
                }
            }
            conv_out[oc] = conv_out[oc].cwiseMax(0.0f); 
        }

        // 2. 全局平均池化
        Eigen::VectorXf gap_out(out_c);
        for (int oc = 0; oc < out_c; ++oc) {
            gap_out(oc) = conv_out[oc].mean();
        }

        // 3. 全连接层
        return fc_W * gap_out + fc_b;
    }
};
```
