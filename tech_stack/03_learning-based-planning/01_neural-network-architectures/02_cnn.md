# 卷积神经网络

## 1. 概述

卷积神经网络通过局部连接与空间共享权重提取视觉特征。任务损失将不同位置的梯度汇总到同一卷积核，使模型学习可在多个位置复用的局部模式。

## 2. 基础与定义

### 2.1 张量与卷积核

单样本输入为 $X\in\mathbb R^{C_{\rm in}\times H\times W}$，卷积核为 $W\in\mathbb R^{C_{\rm out}\times C_{\rm in}\times K\times K}$。常见深度学习卷积层计算互相关：

$$
Y_{o,i,j}=b_o+\sum_{c,m,n}W_{o,c,m,n}X_{c,is+m-P,js+n-P},
$$

其中 $s$ 为步长、$P$ 为填充，越界输入按填充规则处理。

### 2.2 输出目标

分类网络将特征池化为向量后输出类别 logits；分割与回归网络保留空间布局，用逐位置损失监督输出。

## 3. 核心原理

### 3.1 空间结构

步长为 1 的卷积在一致边界处理下具有平移等变性：输入平移对应特征平移。下采样与边界会改变这一关系；池化提供局部聚合，其效果取决于窗口与采样方式。

有膨胀率 $d$ 时，单轴输出尺寸为

$$
H_{\rm out}=\left\lfloor\frac{H+2P-d(K-1)-1}{s}\right\rfloor+1.
$$

### 3.2 权重梯度

令 $G_{o,i,j}=\partial L/\partial Y_{o,i,j}$，共享卷积核的梯度为

$$
\frac{\partial L}{\partial W_{o,c,m,n}}
=\sum_{i,j}G_{o,i,j}X_{c,is+m-P,js+n-P}.
$$

相同参数接收所有空间位置及批次样本的贡献。
<details class="kt-derivation" id="cnn-receptive-field">
<summary>感受野递推</summary>

令第 $l$ 层相对原图的采样间距为 $j_l$、感受野大小为 $r_l$，则

$$
j_l=j_{l-1}s_l,\qquad
r_l=r_{l-1}+(K_l-1)d_lj_{l-1},
$$

从 $j_0=r_0=1$ 开始。该式给出理论覆盖范围；实际梯度对不同像素的影响还由权重与激活决定。

</details>

### 3.3 池化与层次特征

全局平均池化为 $z_c=(HW)^{-1}\sum_{i,j}X_{c,i,j}$，输出梯度均匀分配到该通道空间位置。深层网络组合局部特征并扩大感受野，多尺度融合进一步连接不同分辨率。

## 4. 实现与应用

示例网络为卷积、ReLU、全局平均池化和分类头。输入采用通道优先布局，空间尺寸至少达到卷积核大小。训练时将 logits 直接送入交叉熵；C++ 示例展示前向计算。

<details class="kt-derivation" id="implementation-0">
<summary>PyTorch 卷积网络</summary>

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

</details>

<details class="kt-derivation" id="implementation-1">
<summary>C++ 卷积求值</summary>

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

</details>

<details class="kt-derivation" id="implementation-2">
<summary>Eigen 卷积求值</summary>

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
        conv_W.resize(out_c);
        for (auto& kernels : conv_W) {
            kernels.resize(in_c);
            for (auto& kernel : kernels) kernel = Eigen::MatrixXf::Random(ks, ks);
        }
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

</details>

## 5. 分析与总结

卷积的归纳偏置适合局部空间模式，参数量主要由通道与核尺寸决定。分辨率、步长和感受野共同影响精度、细节保留和计算成本。

## 6. 关联与参考

- [Goodfellow 等：Deep Learning](https://www.deeplearningbook.org/)
- [PyTorch：Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [特征金字塔与多尺度融合](06_feature-pyramid-multiscale.md)
